"""Worst-group identifiers used only for evaluation metrics.

Training never consumes group labels. The released pipeline only needs this
small helper to calculate WGA when a dataset provides background/place labels.
"""

from __future__ import annotations

import torch


def worst_group_id(labels: torch.Tensor, bg_labels: torch.Tensor) -> torch.Tensor:
    """组合 (label, place) 为单一组 ID，与 WGA 子群一致。

    对有效样本 ``bg_labels >= 0``：``gid = label * (max_place+1) + place``，
    其中 ``max_place`` 取当前张量内 place 的最大值（通常为 0/1 → 4 组）。
    无效 place 置为 -1，后续指标会掩掉。
    """
    valid = bg_labels >= 0
    out = torch.full_like(labels, -1)
    if not bool(valid.any()):
        return out
    max_p = int(bg_labels[valid].max().item())
    n_p = max_p + 1
    return torch.where(valid, labels * n_p + bg_labels, out)
