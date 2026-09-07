#!/usr/bin/env python3
"""CAFIL 流水线 Stage 3：Stage-II 图像分类器训练入口。

在 CAFIL 四阶段主流程中的位置
----------------------------
1. ``train_stage1.py`` → 2. ``concept_infer.py`` → **3. 本脚本（train_stage2）** → 4. ``inference.py``

本阶段在 ResNet 等 backbone 上训练 ``CAFILImageClassifier``，使用 Stage-1 概念推断产出的
``P.npy``（原型分布）与 ``consscore.npy``（一致性分数 ``s_i``）进行 **加权 CE + concept-bucket 特征对齐**。

读取的 artifact / 配置
--------------------
- **Stage-II YAML**：默认 ``config/waterbirds/stage2.yaml``（``--config`` 可覆盖）
- **Stage-1 概念产物目录**（``dataset.stage1_dir``）：
  - ``P.npy``：每样本对全局原型 U 的软分配（bucket id = argmax(P)）
  - ``consscore.npy``：原始一致性分数 ``s_i``（论文信号，未经 top-q 变换）
- **可选续训**：``output.run_dir`` 下 ``epoch_*.pth``、``best.pth`` 等

写入的 artifact
---------------
- **训练日志**：``{run_dir}/log.csv``、``{run_dir}/log_path 指定的 jsonl``
- **Checkpoint**：
  - ``best.pth``：按 ``eval.primary_metric``（默认 WGA）选优
  - ``best_val_mean.pth`` / ``best_wga.pth``：canonical 双轨选模
  - ``epoch_XXXX.pth``：滚动保留（``keep_last``）
  - ``last.pth``：训练结束快照
- **最终指标**：``metrics.json``、``test_metrics.json``（训练末在 test 上评估各 best track）
- **README.md**：损失监控说明（L_align、bucket 统计等）

默认损失：``cafil_align``
------------------------------------------------
- ``L_cls``：样本权重 ``w = 1 + η·relu(s)`` 的加权交叉熵；
- ``L_align``：同类不同 argmax(P) bucket 的 backbone 特征与动态 anchor 的 L2 对齐；
- 总损失：``L_cls + λ_align · L_align``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader.stage2 import build_eval_dataset, build_train_dataset, make_loader
from train.stage2_evaluation import evaluate, group_ids
from train.stage2_loss import bucket_alignment_loss, resolve_anchor_update_rate, weighted_cross_entropy
from models.cafil_classifier import CAFILImageClassifier
from utils.stage1 import load_cfg
from utils.runtime import set_global_seed
from train.checkpoint_resolver import (
    canonical_stage2_checkpoint_name,
    stage2_checkpoint_candidates,
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """解析 CLI 与未知 override 列表（传给 ``load_cfg``）。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/waterbirds/stage2.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="覆盖 yaml 中的 seed（用于多样子复现）")
    p.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="追加到 output.run_dir / log_path 文件名，避免覆盖已有 run（如 seed137）",
    )
    args, overrides = p.parse_known_args()
    return args, overrides


def _load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """加载 Stage-II YAML，支持 Hydra 风格 ``key=value`` override。"""
    return load_cfg(str(path), list(overrides or []))


def _existing_epoch_checkpoints(out_dir: Path) -> list[Path]:
    """扫描 ``run_dir`` 下 ``epoch_*.pth``，按 epoch 序号排序返回（用于续训）。"""
    ckpts: list[tuple[int, Path]] = []
    for path in out_dir.glob("epoch_*.pth"):
        try:
            epoch = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        ckpts.append((epoch, path))
    ckpts.sort(key=lambda item: item[0])
    return [path for _, path in ckpts]


def _checkpoint_payload(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    epoch: int | None = None,
    val: dict[str, Any] | None = None,
    opt: torch.optim.Optimizer | None = None,
    sched: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    bucket_means: torch.Tensor | None = None,
    bucket_seen: torch.Tensor | None = None,
) -> dict[str, Any]:
    """组装 checkpoint 字典：模型权重 + 可选优化器/调度器/scaler + bucket-anchor 状态。"""
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": cfg,
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if val is not None:
        payload["val"] = val
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    if sched is not None:
        payload["scheduler"] = sched.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if bucket_means is not None:
        payload["bucket_means"] = bucket_means.detach().cpu()
    if bucket_seen is not None:
        payload["bucket_seen"] = bucket_seen.detach().cpu()
    return payload


def _seed_everything(seed: int, runtime_cfg: dict[str, Any] | None = None) -> None:
    """设置全局随机种子（Python/NumPy/Torch/CUDA），支持 deterministic 模式。"""
    runtime_cfg = runtime_cfg or {}
    set_global_seed(
        int(seed),
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        cudnn_benchmark=bool(runtime_cfg.get("cudnn_benchmark", False)),
        deterministic_warn_only=bool(runtime_cfg.get("deterministic_warn_only", False)),
    )


def _artifact_root(cfg: dict[str, Any]) -> Path | None:
    """解析 artifact 根目录（``output.artifact_root`` 或 ``paths.artifact_root``）。"""
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    root = str(output_cfg.get("artifact_root", "") or paths_cfg.get("artifact_root", "")).strip()
    if not root:
        return None
    path = Path(root).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _resolve_output_path(path_str: str, cfg: dict[str, Any]) -> Path:
    """将相对路径解析为绝对路径；若配置了 artifact_root 则相对其拼接。"""
    path = Path(str(path_str)).expanduser()
    if path.is_absolute():
        return path
    artifact_root = _artifact_root(cfg)
    if artifact_root is not None:
        return artifact_root / path
    return Path.cwd() / path


def _make_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    """构建 SGD 优化器；若启用 warmup 则初始 lr 设为 ``warmup.start_lr``。"""
    optim_cfg = cfg["optim"]
    if str(optim_cfg["type"]).lower() != "sgd":
        raise ValueError("CAFIL Stage II currently supports only SGD")
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    warmup_cfg = train_cfg.get("warmup", {}) if isinstance(train_cfg.get("warmup", {}), dict) else {}
    base_lr = float(optim_cfg["lr"])
    init_lr = base_lr
    if bool(warmup_cfg.get("enabled", False)) and int(warmup_cfg.get("epochs", 0) or 0) > 0:
        init_lr = float(warmup_cfg.get("start_lr", base_lr))
    return torch.optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=float(optim_cfg.get("momentum", 0.9)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        nesterov=bool(optim_cfg.get("nesterov", False)),
    )


def _make_scheduler(cfg: dict[str, Any], opt: torch.optim.Optimizer, epochs: int):
    """构建 LR 调度器：可选 linear warmup + cosine，或纯 cosine / constant。"""
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    warmup_cfg = train_cfg.get("warmup", {}) if isinstance(train_cfg.get("warmup", {}), dict) else {}
    sched_cfg = cfg.get("sched", {}) if isinstance(cfg.get("sched", {}), dict) else {}
    sched_type = str(sched_cfg.get("type", "cosine")).strip().lower()
    warmup_enabled = bool(warmup_cfg.get("enabled", False))
    warmup_epochs = int(warmup_cfg.get("epochs", 0) or 0)
    if warmup_enabled and warmup_epochs > 0:
        base_lr = float(cfg["optim"]["lr"])
        start_lr = float(warmup_cfg.get("start_lr", base_lr))
        if start_lr <= 0.0 or base_lr <= 0.0:
            raise ValueError("warmup.start_lr and optim.lr must be positive")
        start_ratio = start_lr / base_lr
        warmup_epochs = min(max(warmup_epochs, 1), max(int(epochs), 1))
        cosine_total = max(int(sched_cfg.get("t_max", epochs)) - warmup_epochs, 1)

        def _lr_factor(step_idx: int) -> float:
            step_idx = int(step_idx)
            if step_idx < warmup_epochs:
                if warmup_epochs == 1:
                    current = base_lr
                else:
                    # LambdaLR applies the lambda at construction time, so step_idx=0
                    # corresponds to the learning rate used for epoch 1.
                    progress = float(step_idx) / float(max(warmup_epochs - 1, 1))
                    current = start_lr + (base_lr - start_lr) * progress
                return current / start_lr
            if sched_type == "cosine":
                progress = min(max(float(step_idx - (warmup_epochs - 1)) / float(cosine_total), 0.0), 1.0)
                current = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
                return current / start_lr
            if sched_type in {"constant", "none"}:
                return base_lr / start_lr
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_factor)
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(sched_cfg.get("t_max", epochs)))
    if sched_type in {"constant", "none"}:
        return None
    raise ValueError(f"Unsupported scheduler type: {sched_type}")


def _unpack_stage2_batch(batch):
    """统一解包 Stage-II batch：``(x, y, P, s[, sample_idx])``。"""
    if len(batch) == 5:
        x, y, P, s, sample_idx = batch
    elif len(batch) == 4:
        x, y, P, s = batch
        sample_idx = None
    else:
        raise ValueError(f"Unexpected Stage 2 batch length: {len(batch)}")
    return x, y, P, s, sample_idx


def main() -> None:
    """Stage-II 完整训练：数据加载 → cafil_align 损失 → 双轨选模 → 末段 test 评估。

    主流程
    ------
    1. 解析配置/种子/输出目录；
    2. 构建带 Stage-1 产物的 train set 与 val/test set；
    3. 初始化 ``CAFILImageClassifier``、优化器、动态 bucket-anchor 状态；
    4. 可选从 ``epoch_*.pth`` 续训；
    5. epoch 循环：加权 CE + bucket 对齐 → 验证 → 保存 best/mean/wga/rolling ckpt；
    6. 训练结束后在 test 上评估各 best track，写入 ``metrics.json``。
    """
    args, overrides = _parse_args()
    cfg = _load_config(args.config, overrides)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    base_seed = int(cfg.get("seed", runtime_cfg.get("seed", 42)))
    runtime_cfg.setdefault("seed", base_seed)
    cfg["runtime"] = runtime_cfg
    if args.run_suffix:
        suf = str(args.run_suffix).strip()
        rd = Path(str(cfg["output"]["run_dir"]))
        cfg["output"]["run_dir"] = str(rd.parent / f"{rd.name}_{suf}")
        lp = Path(str(cfg["output"]["log_path"]))
        cfg["output"]["log_path"] = str(lp.with_name(f"{lp.stem}_{suf}{lp.suffix}"))
    _seed_everything(base_seed, runtime_cfg)

    # --- 输出目录与 README（监控 L_align / bucket 统计的预期含义）---
    out_dir = _resolve_output_path(str(cfg["output"]["run_dir"]), cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _resolve_output_path(str(cfg["output"]["log_path"]), cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "log.csv"
    readme_path = out_dir / "README.md"
    loss_type_for_readme = str(cfg["loss"]["type"]).lower()
    if loss_type_for_readme == "cafil_align":
        readme_text = (
            "监控预期：\n"
            "- L_align：同类跨 argmax(P_i) bucket 的 backbone 特征均值对齐损失\n"
            "- bucket anchor：训练期 class–concept 动态特征参考，不进入 inference\n"
            "- anchor_update_rate：当前 batch mean 写入率；旧 anchor 保留率为 1-rate\n"
            "- n_buckets_active_c*：已初始化动态 anchor 的 bucket 数，最多为 num_buckets\n"
            "- pair_count_mean：每个 batch 中当前 bucket 与同类动态 anchor 的平均配对数\n"
        )
    else:
        readme_text = (
            f"Unsupported loss.type={loss_type_for_readme!r}; CAFIL mainline requires cafil_align.\n"
        )
    readme_path.write_text(readme_text, encoding="utf-8")

    # --- Dataset construction: train carries Stage-I P/s; evaluation remains image-only. ---
    data_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
    primary_metric = str(eval_cfg.get("primary_metric", "wga")).strip().lower()
    annotation_free = bool(eval_cfg.get("annotation_free", False))
    if annotation_free and primary_metric not in {"mean", "mean_acc"}:
        raise ValueError("eval.annotation_free requires eval.primary_metric=mean")
    train_set = build_train_dataset(data_cfg)
    val_set = build_eval_dataset(data_cfg, "val", annotation_free=annotation_free)
    test_set = build_eval_dataset(data_cfg, "test", annotation_free=annotation_free)
    loader = make_loader(train_set, train_cfg, shuffle=True, seed=base_seed, split="train")
    val_loader = make_loader(val_set, train_cfg, shuffle=False, seed=base_seed, split="val")

    # CAFIL has one published Stage-II objective.
    loss_cfg = cfg["loss"]
    if str(loss_cfg["type"]).lower() != "cafil_align":
        raise ValueError("loss.type must be cafil_align for the CAFIL mainline")
    anchor_update_rate = resolve_anchor_update_rate(loss_cfg)
    device = torch.device(args.device)
    pretrained = str(cfg["model"].get("pretrained", "")).lower() == "imagenet"
    _p = getattr(train_set, "P", None)
    # bucket 数 = loss.num_buckets 或 P 的 K 维（与 concept_infer 的 R 一致，通常 8）
    num_buckets = int(loss_cfg.get("num_buckets", _p.shape[1] if _p is not None else 8))
    if _p is not None and num_buckets != int(_p.shape[1]):
        raise ValueError(
            "loss.num_buckets must equal the Stage-I concept dimension: "
            f"configured={num_buckets}, P.npy={_p.shape[1]}"
        )
    model = CAFILImageClassifier(
        backbone_name=str(cfg["model"]["backbone"]),
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=pretrained,
    ).to(device)
    if device.type == "cuda" and bool(train_cfg.get("channels_last", True)):
        model = model.to(memory_format=torch.channels_last)
    opt = _make_optimizer(cfg, model)
    sched = _make_scheduler(cfg, opt, int(train_cfg["epochs"]))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("loss.label_smoothing must be in [0, 1)")

    use_amp = device.type == "cuda" and bool(train_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # 对每个 (类别 c, concept bucket m) 维护视觉特征 φ 的动态 anchor A_{c,m}。
    # anchor_update_rate 是当前 batch mean 的写入率，不是旧 anchor 的 decay。
    # bucket m 由 P_i.argmax() 决定：样本 i 被分配到哪个概念原型，即落入哪个 bucket。
    # bucket_seen[c,m]=False 时该 bucket 尚未有 anchor，L_align 中不参与跨 bucket 配对。
    bucket_means = torch.zeros(
        int(cfg["model"]["num_classes"]),
        num_buckets,
        int(model.feat_dim),
        device=device,
    )
    bucket_seen = torch.zeros(bucket_means.shape[:2], device=device, dtype=torch.bool)

    if args.dry_run:
        # 干跑：仅验证数据管道与模型 forward 形状
        x, y, P, s, _sample_idx = _unpack_stage2_batch(next(iter(loader)))
        if device.type == "cuda" and bool(train_cfg.get("channels_last", True)):
            x = x.to(memory_format=torch.channels_last)
        with torch.no_grad():
            logits, phi = model(x.to(device))
        print(json.dumps({"dry_run": True, "batch": int(x.shape[0]), "logits_shape": list(logits.shape)}, indent=2))
        print(json.dumps({"features_shape": list(phi.shape)}, indent=2))
        return

    # --- 选模追踪：primary_metric（默认 WGA）+ mean/WGA 双轨 best ---
    best_wga = -1.0
    best_mean = -1.0
    best_score = -1.0
    best_epoch = 0
    best_mean_epoch = 0
    best_wga_epoch = 0
    final = {}
    keep_last = int(train_cfg.get("keep_last", train_cfg.get("max_num_checkpoint", 0)) or 0)
    save_ckpt_every_epoch = bool(train_cfg.get("save_ckpt_every_epoch", False))
    primary_metric = str(eval_cfg.get("primary_metric", "wga")).lower()
    mean_checkpoint_name = canonical_stage2_checkpoint_name("mean")
    wga_checkpoint_name = canonical_stage2_checkpoint_name("wga")
    primary_checkpoint_name = canonical_stage2_checkpoint_name(primary_metric)
    metric_key = "worst_group_acc" if primary_checkpoint_name == wga_checkpoint_name else "mean_acc"
    existing_epoch_ckpts = _existing_epoch_checkpoints(out_dir)
    rolling_checkpoints: list[Path] = existing_epoch_ckpts[-keep_last:] if keep_last > 0 else []
    resume_epoch = 0
    # --- 可选续训：从最新 epoch_*.pth 恢复权重、优化器、bucket-anchor 状态 ---
    if existing_epoch_ckpts:
        resume_path = existing_epoch_ckpts[-1]
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt["model"])
        resume_epoch = int(resume_ckpt.get("epoch", 0) or 0)
        optimizer_state = resume_ckpt.get("optimizer")
        scheduler_state = resume_ckpt.get("scheduler")
        scaler_state = resume_ckpt.get("scaler")
        if optimizer_state is not None:
            opt.load_state_dict(optimizer_state)
        if sched is not None:
            if scheduler_state is not None:
                sched.load_state_dict(scheduler_state)
            else:
                for _ in range(resume_epoch):
                    sched.step()
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        if bucket_means is not None and "bucket_means" in resume_ckpt:
            bucket_means.copy_(resume_ckpt["bucket_means"].to(device))
        if bucket_seen is not None and "bucket_seen" in resume_ckpt:
            bucket_seen.copy_(resume_ckpt["bucket_seen"].to(device))
        for track_metric in ("mean", "wga"):
            path = next(
                (
                    out_dir / candidate
                    for candidate in stage2_checkpoint_candidates(track_metric)
                    if (out_dir / candidate).is_file()
                ),
                None,
            )
            if path is None:
                continue
            ckpt = torch.load(path, map_location=device)
            best_val = ckpt.get("val")
            if not isinstance(best_val, dict):
                continue
            if track_metric == "mean":
                best_mean = float(best_val["mean_acc"])
                best_mean_epoch = int(ckpt.get("epoch", 0) or 0)
            else:
                best_wga = float(best_val["worst_group_acc"])
                best_wga_epoch = int(ckpt.get("epoch", 0) or 0)
        best_path = out_dir / "best.pth"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device)
            best_val = best_ckpt.get("val")
            if isinstance(best_val, dict):
                best_score = float(best_val[metric_key])
                best_epoch = int(best_ckpt.get("epoch", 0) or 0)
        print(
            json.dumps(
                {
                    "resume": True,
                    "resume_from": str(resume_path),
                    "resume_epoch": resume_epoch,
                    "loaded_optimizer": optimizer_state is not None,
                    "loaded_scheduler": scheduler_state is not None,
                    "loaded_scaler": scaler_state is not None,
                    "loaded_bucket_state": ("bucket_means" in resume_ckpt and "bucket_seen" in resume_ckpt),
                },
                ensure_ascii=False,
            )
        )
    epochs = int(train_cfg["epochs"])
    if resume_epoch >= epochs:
        print(json.dumps({"resume": True, "status": "already_complete", "epoch": resume_epoch}, ensure_ascii=False))
        return
    csv_mode = "a" if resume_epoch > 0 and log_csv.exists() else "w"
    log_mode = "a" if resume_epoch > 0 and log_path.exists() else "w"
    write_header = not (csv_mode == "a" and log_csv.exists() and log_csv.stat().st_size > 0)
    with log_csv.open(csv_mode, newline="", encoding="utf-8") as csv_f, log_path.open(log_mode, encoding="utf-8") as log_f:
        writer = csv.writer(csv_f)
        val_group_ids = np.asarray([], dtype=np.int64) if annotation_free else group_ids(val_set)
        header = ["epoch", "train_loss", "lr", "eval_ran", "val_mean", "val_wga"]
        header.extend([f"val_g{int(g)}" for g in np.unique(val_group_ids).tolist()])
        header.extend(["w_mean", "w_max", "w_pos_frac"])
        header.extend(["L_cls", "L_align", "lam_x_L_align", "n_buckets_active_c0", "n_buckets_active_c1", "pair_count_mean"])
        if write_header:
            writer.writerow(header)
        eval_every = max(int(train_cfg.get("eval_every", 1)), 1)
        val_group_count = int(np.unique(val_group_ids).shape[0])
        for epoch in range(resume_epoch + 1, epochs + 1):
            model.train()
            total_loss = 0.0
            total = 0
            w_sum = 0.0
            w_max = 0.0
            w_pos = 0
            cls_loss_sum = 0.0
            align_loss_sum = 0.0
            pair_count_sum = 0.0
            align_batch_count = 0
            iterator = tqdm(
                loader,
                desc=f"stage2 train ep={epoch}/{epochs}",
                dynamic_ncols=True,
                leave=False,
            )
            # --- 单 epoch batch 训练：加权 CE + concept-bucket 特征对齐 ---
            #
            #
            #   L = L_cls + λ_align · L_align
            #
            # L_cls：逐样本 CE 再按 w_i 加权平均::
            #
            #   w_i = 1 + η · relu(s_i)     # s_i 来自 consscore.npy（高共识 → 更大训练权重）
            #   L_cls = Σ_i w_i · CE(logits_i, y_i) / Σ_i w_i
            #
            # L_align：同类内、不同 argmax(P) bucket 的 batch 均值特征，向其他 bucket 的动态 anchor 拉近::
            #
            #   bucket_id_i = argmax_k P_{i,k}
            #   对每个 batch 内出现的 (c, m)：mean_cm = mean(φ_i | y_i=c, bucket_id_i=m)
            #   对每个 m'≠m 且 bucket_seen[c,m']=True：L_align += ||mean_cm - μ_{c,m'}||²
            #   最后除以 pair 数与 feat_dim 做尺度归一化
            #
            # 动态 anchor 更新（无梯度）::
            #
            #   A_{c,m} ← (1-rho)·A_{c,m} + rho·mean_cm   （首次见到则直接赋值）
            for batch in iterator:
                x, y, P, s, _sample_idx = _unpack_stage2_batch(batch)
                x = x.to(device, non_blocking=True)
                if device.type == "cuda" and bool(train_cfg.get("channels_last", True)):
                    x = x.to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                P = P.to(device, non_blocking=True)
                s = s.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, phi = model(x)
                    if bucket_means is None or bucket_seen is None:
                        raise RuntimeError("CAFIL requires bucket-anchor state")
                    loss_cls, w = weighted_cross_entropy(
                        logits, y, s, eta=float(loss_cfg["eta"]), label_smoothing=label_smoothing
                    )
                    loss_align, pair_count_batch = bucket_alignment_loss(
                        phi, y, P, bucket_means, bucket_seen,
                        num_classes=int(cfg["model"]["num_classes"]), update_rate=anchor_update_rate,
                    )
                    loss = loss_cls + float(loss_cfg["lam_align"]) * loss_align
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                bsz = int(y.shape[0])
                total += bsz
                total_loss += float(loss.detach().cpu()) * bsz
                cls_loss_sum += float(loss_cls.detach().cpu()) * bsz
                align_loss_sum += float(loss_align.detach().cpu()) * bsz
                pair_count_sum += float(pair_count_batch)
                align_batch_count += 1
                w_detached = w.detach()
                w_sum += float(w_detached.sum().cpu())
                w_max = max(w_max, float(w_detached.max().cpu()))
                w_pos += int((w_detached > 1.0).sum().cpu())
                iterator.set_postfix(
                    loss=f"{total_loss / max(total, 1):.4f}",
                    cls=f"{cls_loss_sum / max(total, 1):.4f}",
                    wpos=f"{w_pos / max(total, 1):.3f}",
                )
            current_lr = float(opt.param_groups[0]["lr"])
            if sched is not None:
                sched.step()
            train_loss = total_loss / max(total, 1)
            w_mean = w_sum / max(total, 1)
            w_pos_frac = w_pos / max(total, 1)
            L_cls = cls_loss_sum / max(total, 1)
            L_align = align_loss_sum / max(total, 1)
            pair_count_mean = pair_count_sum / max(align_batch_count, 1)
            n_buckets_active = bucket_seen.sum(dim=1).detach().cpu().numpy().astype(np.int64).tolist()
            should_eval = (epoch % eval_every == 0) or (epoch == epochs)
            val = None
            # --- 验证与选模：双轨 best_val_mean / best_wga + primary_metric best ---
            if should_eval:
                val = evaluate(model, val_set, val_loader, device, annotation_free=annotation_free)
                current_score = float(val[metric_key])
                best_payload = _checkpoint_payload(
                    model,
                    cfg,
                    epoch=int(epoch),
                    val=val,
                    opt=opt,
                    sched=sched,
                    scaler=scaler,
                    bucket_means=bucket_means,
                    bucket_seen=bucket_seen,
                )
                if float(val["mean_acc"]) > best_mean:
                    best_mean = float(val["mean_acc"])
                    best_mean_epoch = int(epoch)
                    torch.save(best_payload, out_dir / mean_checkpoint_name)
                if not annotation_free and float(val["worst_group_acc"]) > best_wga:
                    best_wga = float(val["worst_group_acc"])
                    best_wga_epoch = int(epoch)
                    torch.save(best_payload, out_dir / wga_checkpoint_name)
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = int(epoch)
                    torch.save(best_payload, out_dir / "best.pth")
            if keep_last > 0 and (save_ckpt_every_epoch or should_eval):
                ckpt_path = out_dir / f"epoch_{int(epoch):04d}.pth"
                torch.save(
                    _checkpoint_payload(
                        model,
                        cfg,
                        epoch=int(epoch),
                        val=val,
                        opt=opt,
                        sched=sched,
                        scaler=scaler,
                        bucket_means=bucket_means,
                        bucket_seen=bucket_seen,
                    ),
                    ckpt_path,
                )
                rolling_checkpoints.append(ckpt_path)
                while len(rolling_checkpoints) > keep_last:
                    old = rolling_checkpoints.pop(0)
                    if old.exists() and old.name != "best.pth":
                        old.unlink()
            row = [
                epoch,
                f"{train_loss:.6f}",
                f"{current_lr:.8f}",
                int(should_eval),
                f"{val['mean_acc']:.6f}" if val is not None else "nan",
                f"{val['worst_group_acc']:.6f}" if val is not None else "nan",
                *([f"{x:.6f}" for x in val["group_acc"]] if val is not None else ["nan" for _ in range(val_group_count)]),
                f"{w_mean:.6f}",
                f"{w_max:.6f}",
                f"{w_pos_frac:.6f}",
            ]
            row.extend(
                [
                    f"{L_cls:.6f}",
                    f"{L_align:.6f}",
                    f"{float(loss_cfg['lam_align']) * L_align:.6f}",
                    int(n_buckets_active[0]),
                    int(n_buckets_active[1]) if len(n_buckets_active) > 1 else 0,
                    f"{pair_count_mean:.6f}",
                ]
            )
            writer.writerow(row)
            csv_f.flush()
            line = json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "lr": current_lr,
                    "eval_ran": bool(should_eval),
                    "val": val,
                    "w_mean": w_mean,
                    "w_max": w_max,
                    "w_pos_frac": w_pos_frac,
                    "L_cls": L_cls,
                    "L_align": L_align,
                    "lam_x_L_align": float(loss_cfg.get("lam_align", 0.0)) * L_align,
                    "n_buckets_active_c0": int(n_buckets_active[0]),
                    "n_buckets_active_c1": int(n_buckets_active[1]) if len(n_buckets_active) > 1 else 0,
                    "pair_count_mean": pair_count_mean,
                },
                ensure_ascii=False,
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            if val is not None:
                final = {"val": val}

    # Preserve the final-epoch state before loading selected checkpoints for test evaluation.
    final_payload = _checkpoint_payload(
        model, cfg, epoch=epochs, val=final.get("val") if isinstance(final, dict) else None,
        opt=opt, sched=sched, scaler=scaler, bucket_means=bucket_means, bucket_seen=bucket_seen,
    )

    # --- 训练结束：在 test 上评估各 best track，写入 metrics.json ---
    test_loader = make_loader(test_set, train_cfg, shuffle=False, seed=base_seed, split="test")
    test_at_end = bool(cfg.get("eval", {}).get("test_at_end", True))
    selection: dict[str, Any] = {}
    if test_at_end:

        def _eval_ckpt(ckpt_path: Path, *, track: str, ep: int) -> dict[str, Any]:
            """加载指定 best track checkpoint，在 test 上评估并返回 val/test 对照指标。"""
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            test_metrics = evaluate(model, test_set, test_loader, device, annotation_free=annotation_free)
            val_obj = ckpt.get("val") if isinstance(ckpt.get("val"), dict) else {}
            return {
                "track": track,
                "checkpoint": ckpt_path.name,
                "best_epoch": int(ep),
                "val_mean": float(val_obj.get("mean_acc", 0.0)),
                "val_wga": None if annotation_free else float(val_obj.get("worst_group_acc", 0.0)),
                "test_mean": float(test_metrics["mean_acc"]),
                "test_wga": None if annotation_free else float(test_metrics["worst_group_acc"]),
                "test": test_metrics,
            }

        for track, ckpt_name, ep in (
            ("mean", mean_checkpoint_name, best_mean_epoch),
            ("wga", wga_checkpoint_name, best_wga_epoch),
        ):
            ckpt_path = out_dir / ckpt_name
            if ckpt_path.exists():
                selection[track] = _eval_ckpt(ckpt_path, track=track, ep=ep)

    primary_key = "mean" if metric_key == "mean_acc" else "wga"
    test = selection.get(primary_key, {}).get("test", {})
    metrics = {
        "best_wga": best_wga,
        "best_mean": best_mean,
        "best_score": best_score,
        "primary_metric": str(cfg.get("eval", {}).get("primary_metric", "wga")),
        "final_wga": float(final["val"]["worst_group_acc"]) if isinstance(final, dict) and final.get("val") else 0.0,
        "final_mean": float(final["val"]["mean_acc"]) if isinstance(final, dict) and final.get("val") else 0.0,
        "best_epoch": best_epoch,
        "best_mean_epoch": best_mean_epoch,
        "best_wga_epoch": best_wga_epoch,
        "test_wga": float(test.get("worst_group_acc", 0.0)),
        "test_mean": float(test.get("mean_acc", 0.0)),
        "selection": selection,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if selection:
        (out_dir / "test_metrics.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(final_payload, out_dir / "last.pth")


if __name__ == "__main__":
    main()
