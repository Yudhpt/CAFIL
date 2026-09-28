#!/usr/bin/env python3
"""CAFIL 流水线 Stage 4：Stage-II 模型统一推理与评估入口。

在 CAFIL 四阶段主流程中的位置
----------------------------
1. ``train_stage1.py`` → 2. ``concept_infer.py`` → 3. ``train_stage2.py`` → **4. 本脚本（inference）**

本阶段不再训练，仅加载 Stage-II 已训好的 ``CAFILImageClassifier``，在 val/test 上计算
整体准确率（mean acc）与最差组准确率（worst-group acc, WGA）。

读取的 artifact / 配置
--------------------
- **Stage-II YAML**：``--config`` 指定（如 ``config/waterbirds/stage2.yaml``）
- **Stage-II checkpoint**（二选一）：
  - ``--checkpoint`` 显式路径；或
  - ``resolve_stage2_checkpoint(cfg)`` 按配置自动解析 canonical
    ``best_wga.pth`` / ``best_val_mean.pth``，并按顺序尝试 ``best_val_wga.pth`` / ``best.pth``
- checkpoint 内嵌 ``config`` / ``model`` / 可选 ``val`` / ``epoch``

写入的 artifact
---------------
- **可选 JSON 指标**：``--output_json`` 指定路径（含 group 级明细、primary_score 等）
- 标准输出：人类可读的 SUMMARY 行（mean / wga / primary）

与训练器的关系
----------------
- 直接使用 ``stage2_data`` 和 ``stage2_evaluation`` 的公共实现；
- 推理只使用图像，Stage-I artifact 仅参与训练目标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.cafil_classifier import build_cafil_classifier_from_config
from utils.stage1 import load_cfg
from train.checkpoint_resolver import resolve_stage2_checkpoint
from data.dataloader.stage2 import build_eval_dataset, make_loader
from train.stage2_evaluation import evaluate


def _parse_args() -> argparse.Namespace:
    """解析 CLI：config、可选 checkpoint 覆盖、设备、输出 split 与 JSON 路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Stage-II YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Optional Stage-II checkpoint override")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default=None, help="Optional JSON metrics output path")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    return parser.parse_args()


def _load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """加载 YAML 配置，支持 ``key=value`` 形式的 CLI override 列表。"""
    return load_cfg(str(path), list(overrides or []))


def _build_eval_dataset(cfg: dict[str, Any], split: str, *, annotation_free: bool):
    """Build validation data without group fields in annotation-free mode."""
    return build_eval_dataset(cfg["dataset"], split, annotation_free=annotation_free)

def _load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    """加载 checkpoint 并规范化为 ``{"model": state_dict, ...}`` 字典。

    接受两种磁盘格式：完整训练 payload（含 ``model`` 键）或裸 state_dict。
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt
    if isinstance(ckpt, dict):
        return {"model": ckpt}
    raise RuntimeError(f"Unsupported checkpoint type at {path}: {type(ckpt)}")


def _build_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """按配置实例化 ``CAFILImageClassifier``（eval 时不加载 ImageNet 预训练权重）。"""
    model = build_cafil_classifier_from_config(cfg, pretrained=False).to(device)
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    if device.type == "cuda" and bool(train_cfg.get("channels_last", True)):
        model = model.to(memory_format=torch.channels_last)
    return model


def run_inference(
    cfg: dict[str, Any],
    *,
    checkpoint: Path,
    device: torch.device,
    split: str = "test",
) -> dict[str, Any]:
    """执行一次完整推理评估，返回结构化指标字典。

    步骤
    ----
    1. 加载 checkpoint，优先使用 ckpt 内嵌 ``config``（保证与训练时结构一致）；
    2. 构建模型并 ``load_state_dict(strict=True)``；
    3. 构建 eval DataLoader（无 shuffle，**不挂载 P.npy**）；
    4. 计算整体准确率、逐组准确率和 WGA。

    与训练时评估的差异
    ------------------
    """
    payload = _load_checkpoint_payload(checkpoint, device)
    # 优先使用 checkpoint 内保存的训练配置，避免 YAML 与权重结构不一致
    model_cfg = payload.get("config") or payload.get("cfg")
    effective_cfg = model_cfg if isinstance(model_cfg, dict) and "model" in model_cfg else cfg

    model = _build_model(effective_cfg, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    eval_cfg = effective_cfg.get("eval", {}) if isinstance(effective_cfg.get("eval", {}), dict) else {}
    annotation_free = bool(eval_cfg.get("annotation_free", False))
    dataset = _build_eval_dataset(effective_cfg, split, annotation_free=annotation_free)
    train_cfg = effective_cfg["train"]
    seed = int(effective_cfg.get("seed", 42))
    loader = make_loader(dataset, train_cfg, shuffle=False, seed=seed, split=split)
    metrics = evaluate(model, dataset, loader, device, annotation_free=annotation_free)

    primary = str(eval_cfg.get("primary_metric", "wga")).strip().lower()
    primary_score = float(metrics["mean_acc"] if primary in {"mean", "mean_acc"} else metrics["worst_group_acc"])

    return {
        "checkpoint": str(checkpoint),
        "dataset": str(effective_cfg.get("dataset", {}).get("name", "")),
        "split": split,
        "primary_metric": primary,
        "primary_score": primary_score,
        "mean_acc": float(metrics["mean_acc"]),
        "worst_group_acc": float(metrics["worst_group_acc"]),
        "group_acc": metrics.get("group_acc", []),
        "group_ids": metrics.get("group_ids", []),
        "group_total": metrics.get("group_total", []),
        "stage2_epoch": payload.get("epoch"),
        "stage2_val": payload.get("val"),
    }


def main() -> None:
    """CLI 入口：解析参数 → 解析 checkpoint 路径 → 运行推理 → 打印/写入结果。"""
    args = _parse_args()
    cfg = _load_config(args.config)
    device = torch.device(args.device)

    # --- checkpoint 路径：显式指定或按 YAML 自动解析 ---
    if args.checkpoint:
        ckpt_path = Path(str(args.checkpoint))
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = resolve_stage2_checkpoint(cfg)

    result = run_inference(cfg, checkpoint=ckpt_path, device=device, split=str(args.split))

    # --- 标准输出：便于 shell 脚本 grep SUMMARY 行 ---
    print("=" * 70)
    print(f"checkpoint: {result['checkpoint']}")
    print(f"dataset:    {result['dataset']}")
    print(f"split:      {result['split']}")
    if result.get("stage2_epoch") is not None:
        print(f"stage2 epoch: {result['stage2_epoch']}")
    if isinstance(result.get("stage2_val"), dict):
        val = result["stage2_val"]
        print(
            "stage2 val: "
            f"mean={float(val.get('mean_acc', float('nan'))):.4f}  "
            f"wga={float(val.get('worst_group_acc', float('nan'))):.4f}"
        )
    print(f"mean acc:          {result['mean_acc']:.4f}")
    print(f"worst-group acc:   {result['worst_group_acc']:.4f}")
    print(f"primary ({result['primary_metric']}): {result['primary_score']:.4f}")
    print(
        f"SUMMARY: overall_acc={result['mean_acc']:.4f}  "
        f"wga={result['worst_group_acc']:.4f}  "
        f"primary={result['primary_score']:.4f}"
    )
    print("=" * 70)

    # --- 可选：将完整指标字典写入 JSON ---
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[CAFIL inference] wrote {out_path}")


if __name__ == "__main__":
    main()
