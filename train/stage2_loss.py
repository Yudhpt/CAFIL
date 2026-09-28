"""CAFIL Stage-II objective and its training-time feature anchors.

The classifier only maps images to logits and visual features.  This module
consumes the Stage-I assignments ``P`` and consistency scores ``s`` to form
the CAFIL objective, keeping the model independent of training artifacts.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def resolve_anchor_update_rate(loss_cfg: dict) -> float:
    """Return the current-batch anchor update rate and validate its range."""
    if "ema_beta" in loss_cfg:
        raise ValueError("loss.ema_beta is obsolete; use loss.anchor_update_rate")
    raw_rate = loss_cfg.get("anchor_update_rate", 0.05)
    if isinstance(raw_rate, bool):
        raise TypeError("loss.anchor_update_rate must be numeric, not bool")
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("loss.anchor_update_rate must be a number in [0, 1]") from exc
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("loss.anchor_update_rate must be finite and in [0, 1]")
    return rate


def update_bucket_anchor_(
    bucket_means: torch.Tensor,
    bucket_seen: torch.Tensor,
    class_index: int,
    bucket_index: int,
    batch_mean: torch.Tensor,
    update_rate: float,
) -> None:
    """Update one detached class-concept feature anchor in place."""
    with torch.no_grad():
        if not bool(bucket_seen[class_index, bucket_index]):
            bucket_means[class_index, bucket_index] = batch_mean.detach()
            bucket_seen[class_index, bucket_index] = True
        else:
            bucket_means[class_index, bucket_index] = (
                (1.0 - update_rate) * bucket_means[class_index, bucket_index]
                + update_rate * batch_mean.detach()
            )


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    consensus: torch.Tensor,
    *,
    eta: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``L_cls`` and its per-sample weights ``1 + eta * relu(s)``."""
    weights = 1.0 + float(eta) * torch.relu(consensus)
    per_sample = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
    return (weights * per_sample).sum() / weights.sum(), weights


def bucket_alignment_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    assignments: torch.Tensor,
    bucket_means: torch.Tensor,
    bucket_seen: torch.Tensor,
    *,
    num_classes: int,
    update_rate: float,
) -> tuple[torch.Tensor, int]:
    """Align class-matched concept buckets to stored anchors.

    Each batch mean remains in the gradient graph while anchors are updated
    afterward without gradients.  The returned pair count is a diagnostic.
    """
    num_buckets = int(assignments.shape[1])
    bucket_ids = assignments.argmax(dim=-1)
    batch_means: list[tuple[int, int, torch.Tensor]] = []
    for class_id in range(num_classes):
        for bucket_id in range(num_buckets):
            mask = (labels == class_id) & (bucket_ids == bucket_id)
            if bool(mask.any()):
                batch_means.append((class_id, bucket_id, features[mask].mean(dim=0)))

    alignment = features.new_zeros(())
    pair_count = 0
    for class_id, bucket_id, mean in batch_means:
        for other_bucket in range(num_buckets):
            if other_bucket == bucket_id or not bool(bucket_seen[class_id, other_bucket]):
                continue
            alignment = alignment + (mean - bucket_means[class_id, other_bucket].detach()).pow(2).sum()
            pair_count += 1
    if pair_count:
        alignment = alignment / float(pair_count * features.shape[1])

    for class_id, bucket_id, mean in batch_means:
        update_bucket_anchor_(bucket_means, bucket_seen, class_id, bucket_id, mean, update_rate)
    return alignment, pair_count
