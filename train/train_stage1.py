#!/usr/bin/env python3
"""CAFIL 流水线 Stage 1：DINO + Slot Attention 训练入口。

在 CAFIL 四阶段主流程中的位置
----------------------------
1. **Stage 1（本脚本）** → 2. ``concept_infer.py`` → 3. ``train_stage2.py`` → 4. ``inference.py``

本阶段训练冻结 DINOv2 之上的 Slot Attention 模块，并附带轻量 ERM 分类探头（probe），
为后续概念推断提供 slot 表征与 probe NLL 信号。

读取的 artifact / 配置
--------------------
- **YAML 配置**：``config/<dataset>/stage1.yaml``（或通过 CLI override 指定）
- **可选续训 checkpoint**：``train.resume_checkpoint`` 指向的 ``.pth``（含 ``model`` / ``epoch`` / ``val``）
- **可选已有 best**：``{checkpoint_root}/{checkpoint_experiment_name}/best.pth``（用于恢复 ``best_val``）

写入的 artifact
---------------
- **训练指标 CSV**：``{output_dir}/dino_slot_stage1_metrics.csv``（train/val 逐 epoch 记录）
- **配置快照**：``{output_dir}/config.yaml`` 或 ``config_resume.yaml``
- **Checkpoint 目录**（默认 ``{checkpoint_root}/{checkpoint_experiment_name}/``）：
  - ``best.pth``：验证集 ``loss_total`` 最优
  - ``last.pth``：最后一轮权重
  - ``best_cafil_stage1.pth``：probe_refit 启用时的发布版（供 ``concept_infer.py`` 加载）

核心训练目标
------------
- 冻结 DINOv2 patch token 作为结构化视觉特征；
- Slot Attention 将 token 分解为软区域（slot）；
- **主损失**：token 重建（``loss_recon``）+ slot 正则（overlap / entropy）；
- **辅助探头**：接 **detach 的 DINO 池化特征**，用于后续 slot 敏感度评分。

训练范围由 ``method.stage1.train_target`` 控制
----------------------------------------------
- ``slot_only``：仅训练 token_proj / slot_attention / recon_head；cls 不参与 backward；
- ``probe_only``：仅训练 classifier；backward 只有 ``loss_cls``；
- ``both``：前 ``probe_train_epochs`` 轮 slot+probe 联合训练，之后仅 slot。

工程三模式说明
--------------
- 前向由 ``DinoSlotStage1`` 完成：probe 在图结构上本就不回传 slot；
- 本文件叠加 ``requires_grad`` 冻结与 **分项 loss 组装**（``_assemble_backward_loss``）：
  ``slot_only`` / ``both`` 后半段 **显式不把 cls 项加入 backward**，避免优化器动量污染；
- ``both`` 在 probe 阶段结束时 **重建 AdamW**（仅含当前可训参数）。
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dataloader.stage1 import build_dataloader  # noqa: E402
from utils.stage1 import build_output_dir, get_device, load_cfg, set_seed  # noqa: E402
from train.stage1_model import DinoSlotStage1  # noqa: E402
from utils.stage1 import (  # noqa: E402
    _apply_stage1_train_masks,
    _assemble_backward_loss,
    _build_model_cfg,
    _needs_new_optimizer,
    _to_float,
    evaluate,
    parse_args,
)
from utils import dump_yaml  # noqa: E402


def run_training() -> None:
    """Stage 1 完整训练流程：配置加载 → 数据/模型构建 → epoch 循环 → 可选 probe_refit。

    流程概要
    --------
    1. 解析 CLI 与 YAML，设置随机种子与设备；
    2. 创建输出目录与 checkpoint 目录，dump 配置快照；
    3. 构建 ``DinoSlotStage1``、train/val DataLoader；
    4. 可选从 ``resume_checkpoint`` 恢复 epoch 与 ``best_val``；
    5. 主循环：按 ``train_target`` 切换可训参数 → 前向 → 分项 loss → 反传 → 验证 → 存盘；
    6. 若启用 ``probe_refit``：加载 best slot 权重，冻结 slot、仅重训 classifier，发布 ``best_cafil_stage1.pth``。
    """
    # --- 配置与运行时环境 ---
    config_name, overrides = parse_args()
    cfg = load_cfg(config_name, overrides)
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    base_seed = int(runtime_cfg.get("seed", cfg.get("seed", 42)))
    set_seed(
        base_seed,
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        cudnn_benchmark=bool(runtime_cfg.get("cudnn_benchmark", False)),
        deterministic_warn_only=bool(runtime_cfg.get("deterministic_warn_only", False)),
    )
    device = get_device(cfg)
    # 输出目录：续训时可指定 resume_output_dir 以追加 metrics，否则按 experiment.name 新建
    resume_output_dir = str(paths_cfg.get("resume_output_dir", "") or "").strip()
    if resume_output_dir:
        out_dir = Path(resume_output_dir)
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
    else:
        out_dir = build_output_dir(cfg, str(cfg.get("experiment", {}).get("name", "dino_slot_stage1")))
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = str(train_cfg.get("resume_checkpoint", "") or "").strip()
    dump_yaml(out_dir / ("config_resume.yaml" if resume_checkpoint else "config.yaml"), cfg)

    # --- 本版三模式：method.stage1.train_target + probe_train_epochs（见 stage1.yaml 注释）---
    s1_raw = cfg.get("method", {}).get("stage1", {})
    train_target = str(s1_raw.get("train_target", "both"))
    probe_train_epochs = max(1, int(s1_raw.get("probe_train_epochs", 10)))

    model_cfg = _build_model_cfg(cfg)
    model = DinoSlotStage1(model_cfg).to(device)
    # train/val DataLoader：图像增广仅在 train split 启用
    train_loader = build_dataloader(cfg, split="train", is_train=True)
    val_loader = build_dataloader(cfg, split="val", is_train=False)
    optimizer = None
    lr = float(train_cfg.get("lr", 1.0e-4))
    wd = float(train_cfg.get("weight_decay", 1.0e-4))

    # checkpoint 根目录与实验子目录（best.pth / last.pth 存放处，供 concept_infer 读取）
    ckpt_root = Path(str(cfg.get("paths", {}).get("checkpoint_root", "results/checkpoints")))
    ckpt_name = str(cfg.get("paths", {}).get("checkpoint_experiment_name", "")) or str(
        cfg.get("experiment", {}).get("name", "dino_slot_stage1")
    )
    ckpt_dir = ckpt_root / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "split",
        "train_phase",
        "probe_cls_on",
        "loss_total",
        "loss_recon",
        "loss_cls",
        "loss_overlap",
        "loss_entropy",
        "slot_eff_area",
        "slot_peak",
        "slot_entropy",
        "acc",
        "wga",
        "elapsed",
    ]
    # CSV 指标文件：续训时 append，否则覆盖写 header
    metrics_path = out_dir / "dino_slot_stage1_metrics.csv"
    best_val = float("inf")
    epochs = int(train_cfg.get("epochs", 20))
    log_every = int(train_cfg.get("progress_log_every", 10))
    start_epoch = 1

    # --- 可选续训：恢复模型权重、起始 epoch、历史 best_val ---
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = Path.cwd() / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        if "val" in resume_state and isinstance(resume_state["val"], dict):
            best_val = float(resume_state["val"].get("loss_total", best_val))
        best_path = ckpt_dir / "best.pth"
        if best_path.exists():
            best_state = torch.load(best_path, map_location="cpu")
            if isinstance(best_state, dict) and isinstance(best_state.get("val"), dict):
                best_val = float(best_state["val"].get("loss_total", best_val))
        print(
            f"[stage1-dino-slot] resuming from checkpoint={resume_path} "
            f"next_epoch={start_epoch}/{epochs} best_val={best_val:.6f}"
        )
        if start_epoch > epochs:
            print(f"[stage1-dino-slot] resume checkpoint already reached target epochs={epochs}; nothing to do.")
            return

    print(f"[stage1-dino-slot] out_dir={out_dir}")
    print(f"[stage1-dino-slot] checkpoint_dir={ckpt_dir}")
    print(
        f"[stage1-dino-slot] device={device}, epochs={epochs}, train_target={train_target}, "
        f"probe_train_epochs={probe_train_epochs}, batches={len(train_loader)}"
    )

    # --- 主训练循环：逐 epoch 训练 + 验证 + 选模存盘 ---
    append_metrics = bool(resume_checkpoint and metrics_path.exists() and start_epoch > 1)
    with metrics_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append_metrics:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            # 按 train_target 与当前 epoch 设置 requires_grad，返回本 epoch 阶段名与 probe 是否参与 loss
            start = time.time()
            train_phase, probe_cls_on = _apply_stage1_train_masks(
                model,
                train_target=train_target,
                epoch=epoch,
                probe_train_epochs=probe_train_epochs,
            )
            # 参数集合变化时重建优化器（尤其 both：probe 训练结束→仅 slot），避免冻结参数上的 Adam 动量泄漏
            if optimizer is None or _needs_new_optimizer(epoch, train_target, probe_train_epochs):
                trainable = [p for p in model.parameters() if p.requires_grad]
                if not trainable:
                    raise RuntimeError(f"No trainable Stage-1 parameters for train_target={train_target!r}.")
                optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
                print(
                    f"[stage1-dino-slot] new optimizer at epoch={epoch} ({len(trainable)} tensors), "
                    f"phase={train_phase}, probe_cls_on={int(probe_cls_on)}",
                    flush=True,
                )
            model.train()
            totals: Dict[str, float] = {}
            n = 0
            iterator = tqdm(train_loader, desc=f"stage1 ep={epoch}/{epochs}", dynamic_ncols=True, leave=False)
            # --- 单 epoch 内 batch 训练 ---
            for step, batch in enumerate(iterator, start=1):
                images = batch["images"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(images, labels)
                # backward 只走分项组装后的 loss；probe 关闭时绝不把 loss_cls 加进图里
                loss = _assemble_backward_loss(
                    model,
                    out,
                    train_target=train_target,
                    include_cls_loss=bool(probe_cls_on),
                )
                loss.backward()
                # 只对当前可训参数裁剪梯度（与 optimizer 参数子集一致）
                params = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
                bsz = int(labels.shape[0])
                n += bsz
                totals["loss_total"] = totals.get("loss_total", 0.0) + _to_float(loss) * bsz
                for k, v in out.losses.items():
                    if k == "loss_total":
                        continue
                    totals[k] = totals.get(k, 0.0) + _to_float(v) * bsz
                for k, v in out.aux.items():
                    totals[k] = totals.get(k, 0.0) + _to_float(v) * bsz
                if step % max(log_every, 1) == 0:
                    print(
                        f"[stage1-dino-slot] ep={epoch}/{epochs} phase={train_phase} probe_cls={int(probe_cls_on)} "
                        f"step={step}/{len(train_loader)} L={_to_float(loss):.4f} recon={_to_float(out.losses['loss_recon']):.4f} "
                        f"cls={_to_float(out.losses['loss_cls']):.4f} area={_to_float(out.aux['slot_eff_area']):.3f} "
                        f"peak={_to_float(out.aux['slot_peak']):.3f}"
                    )

            # 聚合 train 指标（按样本数加权平均）并写入 CSV
            train_row = {
                k: totals.get(k, float("nan")) / max(n, 1)
                for k in ["loss_recon", "loss_cls", "loss_overlap", "loss_entropy", "slot_eff_area", "slot_peak", "slot_entropy"]
            }
            train_row["loss_total"] = totals.get("loss_total", float("nan")) / max(n, 1)
            train_row.update(
                {
                    "epoch": epoch,
                    "split": "train",
                    "train_phase": train_phase,
                    "probe_cls_on": int(bool(probe_cls_on)),
                    "wga": float("nan"),
                    "acc": float("nan"),
                    "elapsed": time.time() - start,
                }
            )
            writer.writerow({k: train_row.get(k, float("nan")) for k in fieldnames})

            # --- 验证集评估（与 train 使用相同的 loss 组装策略）---
            val = evaluate(
                model,
                val_loader,
                device,
                train_target=train_target,
                include_cls_loss=bool(probe_cls_on),
            )
            val_row = {
                "loss_recon": val.get("loss_recon", float("nan")),
                "loss_cls": val.get("loss_cls", float("nan")),
                "loss_overlap": val.get("loss_overlap", float("nan")),
                "loss_entropy": val.get("loss_entropy", float("nan")),
                "slot_eff_area": val.get("slot_eff_area", float("nan")),
                "slot_peak": val.get("slot_peak", float("nan")),
                "slot_entropy": val.get("slot_entropy", float("nan")),
                "loss_total": val.get("loss_total", float("nan")),
                "acc": val.get("acc", float("nan")),
                "wga": val.get("wga", float("nan")),
            }
            val_row.update(
                {
                    "epoch": epoch,
                    "split": "val",
                    "train_phase": train_phase,
                    "probe_cls_on": int(bool(probe_cls_on)),
                    "elapsed": time.time() - start,
                }
            )
            writer.writerow({k: val_row.get(k, float("nan")) for k in fieldnames})
            f.flush()

            # --- 选模逻辑：以 val loss_total 为准，更优则覆盖 best.pth；每轮另存 last.pth ---
            val_loss = float(val_row.get("loss_total", float("inf")))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "model_cfg": asdict(model_cfg),
                        "epoch": epoch,
                        "train_phase": train_phase,
                        "probe_cls_on": bool(probe_cls_on),
                        "val": val,
                    },
                    ckpt_dir / "best.pth",
                )
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "model_cfg": asdict(model_cfg),
                    "epoch": epoch,
                    "train_phase": train_phase,
                    "probe_cls_on": bool(probe_cls_on),
                },
                ckpt_dir / "last.pth",
            )
            print(
                f"[stage1-dino-slot] epoch={epoch} phase={train_phase} train_L={train_row['loss_total']:.4f} "
                f"val_L={val_loss:.4f} val_acc={val.get('acc', float('nan')):.4f} "
                f"val_wga={val.get('wga', float('nan')):.4f} val_area={val.get('slot_eff_area', float('nan')):.3f}"
            )

    # --- 可选 probe_refit：slot 冻结后单独重训 ERM 探头，发布供 concept_infer 使用的 checkpoint ---
    probe_refit_cfg = s1_raw.get("probe_refit", {}) if isinstance(s1_raw.get("probe_refit", {}), dict) else {}
    if bool(probe_refit_cfg.get("enabled", False)):
        refit_epochs = max(1, int(probe_refit_cfg.get("epochs", 3)))
        best_path = ckpt_dir / "best.pth"
        if best_path.exists():
            best_state = torch.load(best_path, map_location=device)
            model.load_state_dict(best_state["model"])
            print(f"[stage1-dino-slot] probe_refit loaded slot checkpoint={best_path}", flush=True)
        if bool(probe_refit_cfg.get("reset_classifier", True)):
            for module in model.classifier.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        _apply_stage1_train_masks(model, train_target="probe_only", epoch=1, probe_train_epochs=refit_epochs)
        refit_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=wd,
        )
        # probe_refit 内层循环：仅优化 loss_cls
        for refit_epoch in range(1, refit_epochs + 1):
            model.train()
            total_cls = 0.0
            total = 0
            iterator = tqdm(
                train_loader,
                desc=f"stage1 probe_refit ep={refit_epoch}/{refit_epochs}",
                dynamic_ncols=True,
                leave=False,
            )
            for batch in iterator:
                images = batch["images"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                refit_optimizer.zero_grad(set_to_none=True)
                out = model(images, labels)
                loss = out.losses["loss_cls"]
                loss.backward()
                refit_optimizer.step()
                bsz = int(labels.shape[0])
                total += bsz
                total_cls += _to_float(loss) * bsz
                iterator.set_postfix(cls=f"{total_cls / max(total, 1):.4f}")
            val = evaluate(model, val_loader, device, train_target="probe_only", include_cls_loss=True)
            print(
                f"[stage1-dino-slot] probe_refit epoch={refit_epoch}/{refit_epochs} "
                f"train_cls={total_cls / max(total, 1):.4f} val_cls={val.get('loss_cls', float('nan')):.4f} "
                f"val_acc={val.get('acc', float('nan')):.4f}",
                flush=True,
            )
        publish_name = str(paths_cfg.get("checkpoint_publish_name", "best_cafil_stage1.pth") or "best_cafil_stage1.pth")
        publish_path = ckpt_dir / publish_name
        # 发布 payload：含 probe_refit 后 val 指标，供 concept_infer --ckpt 使用
        publish_payload = {
            "model": model.state_dict(),
            "cfg": cfg,
            "model_cfg": asdict(model_cfg),
            "epoch": epochs,
            "train_phase": f"probe_refit_{refit_epochs}ep",
            "probe_cls_on": True,
            "val": evaluate(model, val_loader, device, train_target="probe_only", include_cls_loss=True),
        }
        torch.save(publish_payload, publish_path)
        torch.save(publish_payload, ckpt_dir / "last.pth")
        print(f"[stage1-dino-slot] probe_refit published checkpoint={publish_path}", flush=True)


def main() -> None:
    """CLI 入口：调用 ``run_training()`` 执行 Stage 1 训练。"""
    run_training()


if __name__ == "__main__":
    main()
