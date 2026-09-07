"""
Stage-II Checkpoint 路径解析
============================

本模块在 **不加载权重内容** 的前提下，根据 YAML 配置解析应使用的 Stage-II
checkpoint 文件路径。供 ``train/inference.py`` 及需要恢复模型的脚本调用。

流水线位置
----------
- **阶段**：Stage-II 训练完成之后 → 推理 / 评估之前
- **输入**：完整配置 dict（含 ``eval``、``output`` 段）、可选 ``run_dir`` 覆盖
- **输出**：``Path`` 指向单个 ``.pth`` 文件

解析优先级
----------
1. 显式路径：``eval.checkpoint`` / ``output.primary_checkpoint`` / ``output.ckpt_path``
2. 否则在 ``output.run_dir``（或传入的 ``run_dir``）下按 ``eval.primary_metric`` 搜索候选文件名
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# 以 WGA（worst-group accuracy）为主指标时的候选 checkpoint 文件名（按优先级排序）
_WGA_CANDIDATES = ("best_wga.pth", "best_val_wga.pth", "best.pth")
# 以平均准确率为主指标时的候选文件名
_MEAN_CANDIDATES = ("best_val_mean.pth", "best.pth")
_WGA_METRICS = frozenset({"wga", "worst_group_acc"})
_MEAN_METRICS = frozenset({"mean", "mean_acc"})


def canonical_stage2_checkpoint_name(metric: str) -> str:
    """Return the canonical Stage-II checkpoint filename for a selection metric."""
    normalized = str(metric).strip().lower()
    if normalized in _WGA_METRICS:
        return _WGA_CANDIDATES[0]
    if normalized in _MEAN_METRICS:
        return _MEAN_CANDIDATES[0]
    supported = ", ".join(sorted(_WGA_METRICS | _MEAN_METRICS))
    raise ValueError(f"Unsupported Stage-II selection metric {metric!r}; supported metrics: {supported}")


def stage2_checkpoint_candidates(metric: str) -> tuple[str, ...]:
    """Return canonical-first checkpoint candidates, including legacy fallbacks."""
    canonical = canonical_stage2_checkpoint_name(metric)
    return _WGA_CANDIDATES if canonical == _WGA_CANDIDATES[0] else _MEAN_CANDIDATES


def _explicit_checkpoint(cfg: dict[str, Any]) -> Path | None:
    """
    从配置中读取用户显式指定的 checkpoint 路径。

    会在 ``eval`` 与 ``output`` 两个配置块中查找以下键（任一非空即返回）：
    ``checkpoint``、``primary_checkpoint``、``ckpt_path``。

    Parameters
    ----------
    cfg : dict
        完整 YAML 配置。

    Returns
    -------
    Path | None
        显式路径；若未配置则返回 ``None``。
    """
    for section in ("eval", "output"):
        block = cfg.get(section, {})
        if not isinstance(block, dict):
            continue
        for key in ("checkpoint", "primary_checkpoint", "ckpt_path"):
            raw = block.get(key)
            if raw is None:
                continue
            token = str(raw).strip()
            if token:
                return Path(token)
    return None


def resolve_stage2_checkpoint(cfg: dict[str, Any], *, run_dir: Path | None = None) -> Path:
    """
    解析 Stage-II 应加载的 checkpoint 路径（不读取文件内容）。

    Parameters
    ----------
    cfg : dict
        完整配置，需含 ``output.run_dir``（无显式 checkpoint 时）及可选 ``eval.primary_metric``。
    run_dir : Path | None
        可选，覆盖 ``output.run_dir``，便于脚本传入自定义运行目录。

    Returns
    -------
    Path
        存在的单个 ``.pth`` 文件路径。

    Raises
    ------
    FileNotFoundError
        显式路径不存在，或在 ``run_dir`` 下找不到任何候选文件。
    ValueError
        ``output.run_dir`` 为空，或 ``primary_metric`` 不受支持。
    RuntimeError
        同一优先级 tier 下存在多个候选文件，无法唯一确定。

    Notes
    -----
    ``primary_metric`` 为 ``wga`` / ``worst_group_acc`` 时使用 WGA 候选列表；
    为 ``mean`` / ``mean_acc`` 时使用平均准确率候选列表。
    """
    # --- 第一优先级：配置里写死了路径 ---
    explicit = _explicit_checkpoint(cfg)
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Configured checkpoint not found: {explicit}")
        return explicit

    # --- 第二优先级：在 run_dir 下按主指标自动挑选 ---
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
    base_dir = run_dir or Path(str(output_cfg.get("run_dir", "")).strip())
    if not str(base_dir):
        raise ValueError("Cannot resolve Stage-II checkpoint: output.run_dir is empty")

    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
    primary = str(eval_cfg.get("primary_metric", "wga")).strip().lower()
    candidates = stage2_checkpoint_candidates(primary)

    # 只保留磁盘上真实存在的候选
    existing = [base_dir / name for name in candidates if (base_dir / name).is_file()]
    if not existing:
        searched = ", ".join(str(base_dir / name) for name in candidates)
        raise FileNotFoundError(f"No Stage-II checkpoint found under {base_dir}; searched: {searched}")

    # 按候选列表中的顺序排序，取优先级最高者
    priority = {name: idx for idx, name in enumerate(candidates)}
    existing.sort(key=lambda path: priority.get(path.name, len(candidates)))
    chosen = existing[0]
    # 若同一优先级 tier 有多个文件（例如同时存在两个 best_wga 变体），则报错避免误选
    same_tier = [
        path for path in existing
        if priority.get(path.name, len(candidates)) == priority.get(chosen.name, len(candidates))
    ]
    if len(same_tier) > 1:
        names = ", ".join(path.name for path in same_tier)
        raise RuntimeError(
            f"Ambiguous Stage-II checkpoint selection in {base_dir}: {names}"
        )
    return chosen
