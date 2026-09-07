"""Dataset group construction and evaluation for CAFIL Stage II."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _labels(dataset: Any) -> np.ndarray:
    """Return the dataset's class labels using the common adapter attributes."""
    for name in ("_labels", "labels"):
        if hasattr(dataset, name):
            return np.asarray(getattr(dataset, name), dtype=np.int64)
    raise AttributeError(f"{type(dataset).__name__} does not expose labels")


def group_ids(dataset: Any) -> np.ndarray:
    """Return the ground-truth group id for every evaluation sample."""
    if hasattr(dataset, "groups_gt"):
        return np.asarray(dataset.groups_gt, dtype=np.int64)
    labels = _labels(dataset)
    for name in ("_places", "_place", "_group_labels", "places"):
        if hasattr(dataset, name):
            context = np.asarray(getattr(dataset, name), dtype=np.int64)
            break
    else:
        if not hasattr(dataset, "get_labels_places"):
            raise AttributeError(f"{type(dataset).__name__} does not expose context labels")
        _, context = dataset.get_labels_places()
        context = np.asarray(context, dtype=np.int64)
    if hasattr(dataset, "_co_occur_places"):
        return (labels * 2 + context) * 2 + np.asarray(dataset._co_occur_places, dtype=np.int64)
    return labels * (int(context.max()) + 1 if context.size else 1) + context


def evaluate(model: nn.Module, dataset: Any, loader: DataLoader, device: torch.device, *, annotation_free: bool = False) -> dict[str, Any]:
    """Compute mean accuracy; group metrics are disabled in annotation-free mode."""
    model.eval()
    sample_groups = None if annotation_free else group_ids(dataset)
    unique_groups = np.asarray([], dtype=np.int64) if annotation_free else np.unique(sample_groups).astype(np.int64)
    group_correct = np.zeros(len(unique_groups), dtype=np.int64)
    group_total = np.zeros(len(unique_groups), dtype=np.int64)
    total = correct = offset = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[:2]
            logits, _features = model(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            is_correct = (logits.argmax(dim=1) == labels).cpu().numpy().astype(bool)
            batch_groups = None if annotation_free else sample_groups[offset : offset + len(is_correct)]
            offset += len(is_correct)
            total += len(is_correct)
            correct += int(is_correct.sum())
            if not annotation_free:
                for position, group in enumerate(unique_groups):
                    mask = batch_groups == group
                    group_total[position] += int(mask.sum())
                    group_correct[position] += int(is_correct[mask].sum())
    if offset != len(dataset):
        raise RuntimeError(f"Evaluation loader yielded {offset} samples for dataset of length {len(dataset)}")
    group_acc = np.divide(group_correct, group_total, out=np.zeros(len(unique_groups)), where=group_total > 0)
    return {
        "mean_acc": float(correct / max(total, 1)),
        "worst_group_acc": None if annotation_free else float(group_acc.min()) if len(group_acc) else 0.0,
        "group_ids": [int(value) for value in unique_groups],
        "group_acc": [float(value) for value in group_acc],
        "group_total": [int(value) for value in group_total],
    }
