"""Direct Stage-II dataset construction and deterministic data loading."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.celeba import CelebADataset
from data.stage1_wrapper import DatasetWithStage1
from data.waterbirds import WaterbirdsDataset
from utils.runtime import make_torch_generator, seed_worker


def dataset_class(name: str):
    """Return the supported CAFIL dataset class for ``name``."""
    key = str(name).strip().lower().replace("-", "_")
    if key == "waterbirds":
        return WaterbirdsDataset
    if key in {"celeba", "celeb_a"}:
        return CelebADataset
    if key == "nico":
        from data.nico import NICODataset
        return NICODataset
    if key in {"metashift", "meta_shift"}:
        from data.metashift import MetaShiftDataset
        return MetaShiftDataset
    raise ValueError(f"Unsupported CAFIL dataset.name={name!r}")


def dataset_args(data_cfg: dict[str, Any]) -> SimpleNamespace:
    """Translate the small YAML dataset section to the dataset adapters' API."""
    data_dir = str(data_cfg.get("data_dir", "")).strip()
    if not data_dir:
        raise ValueError("CAFIL Stage II requires dataset.data_dir")
    return SimpleNamespace(
        data_dir=data_dir,
        backbone_class="Res50",
        image_size=int(data_cfg.get("image_size", 224)),
    )


def build_train_dataset(data_cfg: dict[str, Any]):
    """Build the CAFIL training dataset with required Stage-I artifacts."""
    stage1_dir = data_cfg.get("stage1_dir")
    if stage1_dir is None or not str(stage1_dir).strip():
        raise ValueError("CAFIL Stage II requires dataset.stage1_dir with P.npy and consscore.npy")
    return DatasetWithStage1(
        dataset_class(str(data_cfg["name"])),
        "train",
        dataset_args(data_cfg),
        stage1_dir=stage1_dir,
        augment=True,
    )


def build_eval_dataset(data_cfg: dict[str, Any], split: str, *, annotation_free: bool = False):
    """Build an evaluation dataset; annotation-free mode never loads group labels."""
    args = dataset_args(data_cfg)
    if str(data_cfg["name"]).strip().lower() in {"celeba", "celeb_a"}:
        return CelebADataset(
            split, args, augment=False, return_place=not annotation_free, eval_mode=not annotation_free,
            annotation_free_mode=annotation_free,
        )
    return dataset_class(str(data_cfg["name"]))(split, args, augment=False, return_place=False)


def labels(dataset: Any) -> np.ndarray:
    """Read integer labels from a supported dataset adapter."""
    for name in ("_labels", "labels"):
        if hasattr(dataset, name):
            return np.asarray(getattr(dataset, name), dtype=np.int64)
    raise AttributeError(f"{type(dataset).__name__} does not expose labels")


def make_loader(dataset: Any, train_cfg: dict[str, Any], *, shuffle: bool, seed: int, split: str) -> DataLoader:
    """Build a reproducible loader with optional class-balanced train sampling."""
    num_workers = int(train_cfg["num_workers"])
    split_offset = {"train": 0, "val": 1, "test": 2}.get(str(split).lower(), 7)
    sampler_cfg = train_cfg.get("sampler", {}) if isinstance(train_cfg.get("sampler"), dict) else {}
    sampler_type = str(sampler_cfg.get("type", "random")).strip().lower()
    sampler = None
    if shuffle and str(split).lower() == "train":
        if sampler_type == "class_balanced":
            values = labels(dataset)
            unique, counts = np.unique(values, return_counts=True)
            inverse = {int(y): 1.0 / float(count) for y, count in zip(unique, counts)}
            weights = np.asarray([inverse[int(y)] for y in values], dtype=np.float64)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=int(sampler_cfg.get("num_samples", len(weights))),
                replacement=bool(sampler_cfg.get("replacement", True)),
                generator=make_torch_generator(seed + split_offset),
            )
        elif sampler_type != "random":
            raise ValueError("train.sampler.type must be 'random' or 'class_balanced'")
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": make_torch_generator(seed + split_offset),
    }
    if num_workers:
        kwargs.update(
            worker_init_fn=seed_worker,
            persistent_workers=bool(train_cfg.get("persistent_workers", True)),
            prefetch_factor=int(train_cfg.get("prefetch_factor", 4)),
        )
    return DataLoader(dataset, **kwargs)
