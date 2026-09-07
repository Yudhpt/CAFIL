from __future__ import annotations

import pytest
import torch

from data.dataloader.stage1 import collate_batch


def _sample(*, bg_label: int | None = None) -> dict[str, object]:
    sample: dict[str, object] = {
        "image": torch.zeros(3, 4, 4),
        "label": 1,
        "path": "sample.jpg",
    }
    if bg_label is not None:
        sample["bg_label"] = bg_label
    return sample


def test_collate_batch_keeps_group_labels_aligned_with_images():
    batch = collate_batch([_sample(bg_label=0), _sample(bg_label=1)])

    assert batch["images"].shape[0] == 2
    assert batch["bg_labels"].tolist() == [0, 1]


def test_collate_batch_rejects_mixed_group_label_schema():
    with pytest.raises(ValueError, match="either provide bg_label for every sample or for none"):
        collate_batch([_sample(bg_label=0), _sample()])
