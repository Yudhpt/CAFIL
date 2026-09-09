from __future__ import annotations

import pytest
import torch

from data.dataloader.stage1 import build_dataset, collate_batch


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



def test_nico_augmentation_follows_loader_mode(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """train_eval 与 concept inference 使用 train 样本但关闭随机 NICO 增强。"""
    import data.nico

    calls: list[tuple[str, bool, bool]] = []

    class FakeNICODataset:
        def __init__(self, split, args, *, augment, return_place):
            calls.append((split, bool(augment), bool(return_place)))
            self._labels = [0]
            self._places = [0]
            self._paths = ["sample.jpg"]

        def __len__(self):
            return 1

    monkeypatch.setattr(data.nico, "NICODataset", FakeNICODataset)
    cfg = {"data": {"format": "nico", "root": str(tmp_path)}}
    build_dataset(cfg, "train", is_train=True)
    build_dataset(cfg, "train", is_train=False)

    assert calls == [("train", True, True), ("train", False, True)]
