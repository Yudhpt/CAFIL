from __future__ import annotations

import pytest

from train.concept_infer import _require_train_only_splits


def test_concept_inference_accepts_train_only_splits():
    _require_train_only_splits("train", "train")


@pytest.mark.parametrize("prototype_split, eval_split", [("val", "train"), ("train", "test")])
def test_concept_inference_rejects_non_train_splits(prototype_split: str, eval_split: str):
    with pytest.raises(ValueError, match="只允许 train split"):
        _require_train_only_splits(prototype_split, eval_split)
