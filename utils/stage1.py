"""
Stage-I 训练辅助函数
====================

为 ``train/train_stage1.py`` 提供 **配置加载、设备选择、输出目录、
Stage-I 子模块冻结策略、损失组装、验证评估** 等逻辑，本身不包含 ``main`` 训练循环。

流水线位置
----------
- **阶段**：Stage-I（DINO + Slot Attention）
- **输入**：CLI 参数、YAML 配置、``DinoSlotStage1`` 模型与 DataLoader
- **输出**：解析后的 cfg dict、评估指标 dict、训练用 loss 张量等

依赖关系
--------
- ``utils.runtime``：种子与 requires_grad
- ``train.stage1_model``：Stage-I 模型与配置 dataclass
- ``modules.SlotAttentionConfig``：Slot Attention 超参
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from modules import SlotAttentionConfig  # noqa: E402
from utils.runtime import set_global_seed, set_module_requires_grad  # noqa: E402
from train.stage1_model import DinoSlotStage1, DinoSlotStage1Config  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.env_group_diagnostics import worst_group_id  # noqa: E402


def parse_args() -> tuple[str, list[str]]:
    """
    解析 Stage-I 训练脚本的命令行参数。

    Returns
    -------
    tuple[str, list[str]]
        - 配置文件路径或 config-name（可为空，后续有默认值）
        - Hydra 风格的 ``key=value`` 覆盖列表（未知参数）

    Notes
    -----
    ``--config`` 优先于 ``--config-name``；后者会映射到 ``config/<name>/stage1.yaml``。
    """
    parser = argparse.ArgumentParser(description="DINO-slot Stage-1 trainer.")
    parser.add_argument("--config-name", default="")
    parser.add_argument("--config", default="", help="YAML config path. Preferred over --config-name.")
    args, overrides = parser.parse_known_args()
    return str(args.config or args.config_name), overrides


def load_cfg(config_name_or_path: str, overrides: list[str]) -> dict[str, Any]:
    """
    加载 YAML 配置并应用 CLI 点号覆盖，同时展开 ``${paths.xxx}`` 占位符。

    Parameters
    ----------
    config_name_or_path : str
        绝对/相对 YAML 路径，或数据集名（如 ``waterbirds`` → ``config/waterbirds/stage1.yaml``）。
    overrides : list[str]
        形如 ``train.batch_size=32`` 的覆盖项。

    Returns
    -------
    dict
        完整配置 dict。

    Raises
    ------
    FileNotFoundError
        配置文件不存在。
    """
    raw = str(config_name_or_path or "").strip()
    if not raw:
        raw = "config/waterbirds/stage1.yaml"
    path = Path(raw)
    if not path.suffix:
        path = Path("config") / raw / "stage1.yaml"
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return load_config(path, overrides)


def set_seed(
    seed: int,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    deterministic_warn_only: bool = False,
) -> None:
    """
    Stage-I 训练入口使用的种子包装，转发到 ``set_global_seed``。

    默认 ``cudnn_benchmark=False`` 与 deterministic 一致，便于论文复现。
    """
    set_global_seed(
        int(seed),
        deterministic=bool(deterministic),
        cudnn_benchmark=bool(cudnn_benchmark),
        deterministic_warn_only=bool(deterministic_warn_only),
    )


def get_device(cfg: dict[str, Any]) -> torch.device:
    """
    根据配置与环境变量选择训练设备。

    环境变量 ``CAFIL_FORCE_CPU=1`` 可强制 CPU（调试或无 GPU 环境）。

    Parameters
    ----------
    cfg : dict
        读取 ``runtime.device``，默认 ``cuda``。

    Returns
    -------
    torch.device
        ``cuda`` 或 ``cpu``。
    """
    if os.environ.get("CAFIL_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.device("cpu")
    requested = str(cfg.get("runtime", {}).get("device", "cuda"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def build_output_dir(cfg: dict[str, Any], run_name: str) -> Path:
    """
    创建带时间戳的 Stage-I 运行输出目录。

    路径形如 ``{paths.output_root}/{run_name}/YYYYMMDD_HHMMSS/``。

    Parameters
    ----------
    cfg : dict
        含 ``paths.output_root``。
    run_name : str
        本次 run 的逻辑名称（通常来自配置或数据集名）。

    Returns
    -------
    Path
        已 ``mkdir(parents=True)`` 的目录。
    """
    root = Path(str(cfg.get("paths", {}).get("output_root", "results")))
    if not root.is_absolute():
        root = Path.cwd() / root
    out = root / str(run_name) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_float(x: torch.Tensor) -> float:
    """将单元素 GPU 张量安全转为 Python float（detach + cpu）。"""
    return float(x.detach().cpu())


def _freeze_module(module: torch.nn.Module, freeze: bool) -> None:
    """
    冻结或解冻整块子模块。

    freeze=True 时整块不参与梯度（requires_grad=False）。
    """
    set_module_requires_grad(module, not bool(freeze))


def _apply_stage1_train_masks(
    model: DinoSlotStage1,
    *,
    train_target: str,
    epoch: int,
    probe_train_epochs: int,
) -> tuple[str, bool]:
    """
    按 ``train_target`` 策略设置 Stage-I 各子模块的 requires_grad。

    Stage-I 模型大致分为：
    - **slot 路径**：``token_proj``、``slot_attention``、``recon_head``（无监督 slot 学习）
    - **probe 路径**：``classifier``（在冻结 DINO 特征上的短 refit 线性探针，论文 Stage-I 信号来源）

    Parameters
    ----------
    model : DinoSlotStage1
        Stage-I 模型。
    train_target : str
        ``slot_only`` | ``probe_only`` | ``both``。
    epoch : int
        当前 epoch（1-based），``both`` 模式下用于切换 probe 是否训练。
    probe_train_epochs : int
        ``both`` 模式下 probe 仅在前 N 个 epoch 参与训练。

    Returns
    -------
    tuple[str, bool]
        - 人类可读的模式描述字符串（写日志用）
        - ``include_cls_loss``：当前 backward 是否应包含分类损失项

    Raises
    ------
    ValueError
        非法 ``train_target``。
    """
    tgt = str(train_target).strip().lower()
    n_probe = max(1, int(probe_train_epochs))
    slot_parts = [model.token_proj, model.slot_attention, model.recon_head]
    probe_head = model.classifier

    if tgt == "slot_only":
        # 只训练 slot 重建相关模块，probe 冻结
        for m in slot_parts:
            _freeze_module(m, freeze=False)
        _freeze_module(probe_head, freeze=True)
        return "slot_only", False

    if tgt == "probe_only":
        # 只训练 probe（论文要求的短 refit），slot 冻结
        for m in slot_parts:
            _freeze_module(m, freeze=True)
        _freeze_module(probe_head, freeze=False)
        return "probe_only", True

    if tgt == "both":
        # 先联合训练若干 epoch，之后只训 slot（probe 固定）
        for m in slot_parts:
            _freeze_module(m, freeze=False)
        probe_on = int(epoch) <= n_probe
        _freeze_module(probe_head, freeze=not probe_on)
        if probe_on:
            return f"both_ep<={n_probe}_slot+probe", True
        return f"both_ep>{n_probe}_slot_only", False

    raise ValueError(f"method.stage1.train_target must be slot_only | probe_only | both, got {train_target!r}")


def _assemble_backward_loss(model: DinoSlotStage1, out, *, train_target: str, include_cls_loss: bool) -> torch.Tensor:
    """
    根据训练目标组装 **实际用于 backward 的标量 loss**。

    Parameters
    ----------
    model : DinoSlotStage1
        提供 ``cfg`` 中的 loss 权重 λ。
    out : model forward 输出
        含 ``out.losses`` 字典：``loss_recon``、``loss_overlap``、``loss_entropy``、``loss_cls`` 等。
    train_target : str
        当前训练模式。
    include_cls_loss : bool
        是否将 ``λ_cls * loss_cls`` 加入总 loss。

    Returns
    -------
    Tensor
        标量 loss，用于 ``loss.backward()``。

    Notes
    -----
    ``probe_only`` 时 **仅** 使用分类损失，符合「短 refit probe」设定。
    """
    cfg_m = model.cfg
    tgt = str(train_target).strip().lower()
    if tgt == "probe_only":
        return out.losses["loss_cls"]
    # slot 相关无监督项加权求和
    total = (
        float(cfg_m.lambda_recon) * out.losses["loss_recon"]
        + float(cfg_m.lambda_overlap) * out.losses["loss_overlap"]
        + float(cfg_m.lambda_entropy) * out.losses["loss_entropy"]
    )
    if include_cls_loss:
        total = total + float(cfg_m.lambda_cls) * out.losses["loss_cls"]
    return total


def _needs_new_optimizer(epoch: int, train_target: str, probe_train_epochs: int) -> bool:
    """
    判断是否应在当前 epoch 重建 optimizer（参数组变化时需要）。

    - epoch 1 总是新建
    - ``both`` 模式下 probe 冻结切换点（``probe_train_epochs + 1``）需新建，因可训练参数集合变了
    """
    if int(epoch) <= 1:
        return True
    return str(train_target).strip().lower() == "both" and int(epoch) == int(probe_train_epochs) + 1


def _build_model_cfg(cfg: Dict[str, Any]) -> DinoSlotStage1Config:
    """
    从 YAML 配置组装 ``DinoSlotStage1Config`` dataclass。

    合并 ``method.stage1``、``train.dino_teacher``、``model.slot_attention`` 等多处字段，
    并填充默认值（DINO 模型名、slot 数、各 λ 权重等）。
    """
    raw = cfg.get("method", {}).get("stage1", {})
    teacher = cfg.get("train", {}).get("dino_teacher", {})
    sa = cfg.get("model", {}).get("slot_attention", {})
    return DinoSlotStage1Config(
        dino_model_name=str(raw.get("dino_model_name", teacher.get("model_name", "dinov2_vitb14"))),
        dino_img_size=int(raw.get("dino_img_size", teacher.get("dino_img_size", 224))),
        dino_torch_home=str(raw.get("dino_torch_home", teacher.get("torch_home", Path.home() / ".cache" / "torch"))),
        dino_dim=int(raw.get("dino_dim", 768)),
        slot_dim=int(raw.get("slot_dim", cfg.get("model", {}).get("slot_dim", 256))),
        num_slots=int(raw.get("num_slots", cfg.get("model", {}).get("num_slots", 8))),
        num_classes=int(raw.get("num_classes", cfg.get("model", {}).get("num_classes", 2))),
        slot_attention=SlotAttentionConfig(
            num_iterations=int(sa.get("num_iterations", raw.get("slot_iterations", 3))),
            epsilon=float(sa.get("epsilon", 1.0e-8)),
            mlp_hidden_size=int(sa.get("mlp_hidden_size", 512)),
            qkv_bias=bool(sa.get("qkv_bias", False)),
            slot_init_scale=float(sa.get("slot_init_scale", 1.0)),
            slot_init_mode=str(sa.get("slot_init_mode", "random")),
        ),
        lambda_recon=float(raw.get("lambda_recon", 1.0)),
        lambda_cls=float(raw.get("lambda_cls", 0.2)),
        lambda_overlap=float(raw.get("lambda_overlap", 0.05)),
        lambda_entropy=float(raw.get("lambda_entropy", 0.02)),
        use_mock_dino=bool(raw.get("use_mock_dino", False)),
    )


@torch.no_grad()
def evaluate(
    model: DinoSlotStage1,
    loader,
    device: torch.device,
    *,
    train_target: str,
    include_cls_loss: bool,
) -> Dict[str, float]:
    """
    在验证/测试集上评估 Stage-I 模型，汇总 loss 与准确率指标。

    Parameters
    ----------
    model : DinoSlotStage1
        评估前会 ``model.eval()``。
    loader : DataLoader
        batch 需含 ``images``、``labels``；可选 ``bg_labels`` 用于 WGA。
    device : torch.device
        计算设备。
    train_target, include_cls_loss
        与训练时相同，用于 ``_assemble_backward_loss`` 选择报告哪类 total loss。

    Returns
    -------
    dict[str, float]
        键包括各 loss 分量、``acc``、``wga``（若有 group 标签）等；
        所有值为 **样本加权平均**。

    Notes
    -----
    WGA 通过 ``worst_group_id(label, bg)`` 构造 group id，再取各 group 准确率的最小值。
    """
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    all_pred = []
    all_label = []
    all_bg = []
    has_group_labels = False
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(images, labels)
        bsz = int(labels.shape[0])
        n += bsz
        # 与训练相同的 loss 选择逻辑，便于 val loss 曲线对齐
        selected = _assemble_backward_loss(model, out, train_target=train_target, include_cls_loss=include_cls_loss)
        totals["loss_total"] = totals.get("loss_total", 0.0) + _to_float(selected) * bsz
        for k, v in out.losses.items():
            if k == "loss_total":
                continue
            totals[k] = totals.get(k, 0.0) + _to_float(v) * bsz
        for k, v in out.aux.items():
            totals[k] = totals.get(k, 0.0) + _to_float(v) * bsz
        all_pred.append(out.logits.argmax(dim=-1).cpu())
        all_label.append(labels.cpu())
        if "bg_labels" in batch:
            all_bg.append(batch["bg_labels"].cpu())
            has_group_labels = True
    row = {k: v / max(n, 1) for k, v in totals.items()}
    if all_pred:
        pred = torch.cat(all_pred)
        label = torch.cat(all_label)
        row["acc"] = float((pred == label).float().mean().item())
        if has_group_labels and all_bg:
            bg = torch.cat(all_bg)
            gid = worst_group_id(label, bg)
            valid = gid >= 0
            if bool(valid.any()):
                group_acc = []
                for g in torch.unique(gid[valid], sorted=True).tolist():
                    m = gid == int(g)
                    if bool(m.any()):
                        group_acc.append(float((pred[m] == label[m]).float().mean().item()))
                row["wga"] = min(group_acc) if group_acc else float("nan")
            else:
                row["wga"] = float("nan")
        else:
            row["wga"] = float("nan")
    return row
