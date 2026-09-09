"""
CAFIL 数据加载公共层
====================

为 **Stage-I 训练**（及概念推断中部分路径）提供统一的数据集构建、
collate 与 DataLoader 工厂。支持多种 ``data.format``：Waterbirds parquet、
CelebA、NICO、ImageFolder、CSV 等。

流水线位置
----------
- **阶段**：Stage-I 主训练循环、部分评估脚本
- **输入**：YAML 配置 dict、split 名称（``train``/``val``/``test``）
- **输出**：``torch.utils.data.Dataset`` 或 ``DataLoader``，batch 键为
  ``images``、``labels``、可选 ``bg_labels``、``paths``

与 Stage-II 数据层的区别
------------------------
Stage-II 使用 ``data/waterbirds.py`` 等数据集类 + ``data/stage1_wrapper.py``，
在样本上附加 ``P_i``、``s_i``；本模块的 ``build_dataset`` 面向 Stage-I 的 dict batch 格式。
"""
from __future__ import annotations

import csv
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from utils.runtime import make_torch_generator, seed_worker


def _env_flag(name: str, default: bool = False) -> bool:
    """
    读取环境变量是否为「真」（1/true/yes/on，大小写不敏感）。

    用于 ``ANNOTATION_FREE_MODE`` 等全局开关。
    """
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _annotation_free_mode_from_cfg(cfg: dict[str, Any]) -> bool:
    """
    判断是否启用 **无组标注（annotation-free）** 模式。

    优先读 ``data.annotation_free_mode``；未配置时回退环境变量 ``ANNOTATION_FREE_MODE``。
    该模式下 CelebA 等数据集不加载 spurious 属性（如 Male），训练不依赖 group 标签。
    """
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    if "annotation_free_mode" in data_cfg:
        return bool(data_cfg.get("annotation_free_mode"))
    return _env_flag("ANNOTATION_FREE_MODE", default=False)


def build_transform(image_size: int) -> transforms.Compose:
    """
    Stage-I 通用图像预处理：Resize 到正方形 + ToTensor（**无** ImageNet normalize）。

    Stage-I 使用 DINO，其内部会做归一化；此处保持简单 tensor 化即可。
    """
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
        ]
    )


class WaterbirdsParquetDataset(Dataset):
    """
    Waterbirds 数据集（HuggingFace parquet 格式），供 Stage-I ``data.format=waterbirds_parquet`` 使用。

    每样本返回 dict：``image``、``label``、``bg_label``（place）、``path``。
    """

    _split_to_file = {
        "train": "train-00000-of-00001.parquet",
        "val": "validation-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }

    def __init__(self, root: Path, split: str, transform: transforms.Compose) -> None:
        if split not in self._split_to_file:
            raise ValueError(f"Unsupported Waterbirds split: {split}")
        parquet_path = root / "data" / self._split_to_file[split]
        if not parquet_path.exists():
            raise FileNotFoundError(f"Waterbirds parquet not found: {parquet_path}")
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        self.transform = transform
        # parquet 中 image 列为 dict：{bytes, path}
        self.images = df["image"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.places = df["place"].astype(int).tolist() if "place" in df.columns else [-1] * len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.images[idx]
        if not isinstance(item, dict) or "bytes" not in item:
            raise ValueError(f"Unexpected Waterbirds image payload: {type(item)}")
        image = Image.open(BytesIO(item["bytes"])).convert("RGB")
        path = str(item.get("path", f"wb_{idx:06d}.jpg"))
        return {"image": self.transform(image), "label": int(self.labels[idx]), "bg_label": int(self.places[idx]), "path": path}

    def get_labels_places(self) -> tuple[list[int], list[int]]:
        """返回全量 label 与 place，供 balanced sampler 等使用。"""
        return list(self.labels), list(self.places)



class NICOStage1Dataset(Dataset):
    """NICO 的 Stage-I dict-batch 包装，按调用方指示是否启用训练增强。"""

    def __init__(self, root: Path, split: str, *, augment: bool) -> None:
        from data.nico import NICODataset

        class _Args:
            data_dir = str(root)
            backbone_class = "Res18"

        self.base = NICODataset(split, _Args(), augment=bool(augment), return_place=True)
        self._labels = list(self.base._labels)
        self._places = list(self.base._places)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, label, bg = self.base[idx]
        return {
            "image": image,
            "label": int(label),
            "bg_label": int(bg),
            "path": str(self.base._paths[idx]),
        }

    def get_labels_places(self) -> tuple[list[int], list[int]]:
        return self._labels, self._places


class CelebAAttrDataset(Dataset):
    """
    CelebA 属性 CSV 加载器（Stage-I ``celeba_attr`` 格式）。

    任务标签：Blond_Hair；背景/组标签：Male（annotation_free 时不加载 Male）。
    """

    _split_to_id = {"train": 0, "val": 1, "test": 2}

    def __init__(self, root: Path, split: str, transform: transforms.Compose, *, annotation_free: bool) -> None:
        if split not in self._split_to_id:
            raise ValueError(f"Unsupported CelebA split: {split}")
        img_dir = root / "Img" / "img_align_celeba"
        attr_csv = root / "Anno" / "list_attr_celeba.csv"
        split_txt = root / "Eval" / "list_eval_partition.txt"
        for path in (img_dir, attr_csv, split_txt):
            if not path.exists():
                raise FileNotFoundError(f"CelebA required path missing: {path}")

        # 官方 partition 文件：文件名 → split id (0/1/2)
        split_map: dict[str, int] = {}
        with split_txt.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    name, sid = line.split()
                    split_map[str(name)] = int(sid)

        import pandas as pd

        usecols = ["image_id", "Blond_Hair"] if bool(annotation_free) else ["image_id", "Blond_Hair", "Male"]
        df = pd.read_csv(attr_csv, usecols=usecols)
        if "image_id" not in df.columns:
            df = pd.read_csv(attr_csv, delim_whitespace=True, usecols=usecols)
        required_cols = ("image_id", "Blond_Hair") if bool(annotation_free) else ("image_id", "Blond_Hair", "Male")
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CelebA attr csv missing column: {col}")

        target_split = self._split_to_id[split]
        self.rows: list[tuple[Path, int]] = []
        self.annotation_free = bool(annotation_free)
        self._bg = [] if not self.annotation_free else None
        for _, row in df.iterrows():
            name = str(row["image_id"])
            if split_map.get(name, -1) != target_split:
                continue
            label = 1 if int(row["Blond_Hair"]) > 0 else 0
            self.rows.append((img_dir / name, label))
            if self._bg is not None:
                self._bg.append(1 if int(row["Male"]) > 0 else 0)
        if not self.rows:
            raise RuntimeError(f"No CelebA samples found for split={split}")
        self.transform = transform
        self._labels = [int(x[1]) for x in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, label = self.rows[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        item = {"image": image, "label": int(label), "index": int(idx), "path": str(path)}
        if self._bg is not None:
            item["bg_label"] = int(self._bg[idx])
        return item

    def get_labels_places(self) -> tuple[list[int], list[int]]:
        if self._bg is None:
            return self._labels, [-1] * len(self._labels)
        return self._labels, list(self._bg)


class ImageFolderDataset(Dataset):
    """标准 ``ImageFolder`` 目录结构（root/split/class_name/*.jpg）。"""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.base = datasets.ImageFolder(root=str(root), transform=transform)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, label = self.base[idx]
        path, _ = self.base.samples[idx]
        return {"image": image, "label": int(label), "bg_label": -1, "path": str(path)}

    def get_labels_places(self) -> tuple[list[int], list[int]]:
        return [int(label) for _, label in self.base.samples], [-1] * len(self.base.samples)


class CsvImageDataset(Dataset):
    """
    通用 CSV 索引数据集：列名由配置指定（image、label、bg_label、split）。
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        image_root: Path,
        image_col: str,
        label_col: str,
        bg_col: str,
        transform: transforms.Compose,
    ) -> None:
        self.rows = rows
        self.image_root = image_root
        self.image_col = image_col
        self.label_col = label_col
        self.bg_col = bg_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        img_path = self.image_root / row[self.image_col]
        image = self.transform(Image.open(img_path).convert("RGB"))
        bg = int(row[self.bg_col]) if self.bg_col and row.get(self.bg_col, "") != "" else -1
        return {"image": image, "label": int(row[self.label_col]), "bg_label": bg, "path": str(img_path)}

    def get_labels_places(self) -> tuple[list[int], list[int]]:
        labels = [int(row[self.label_col]) for row in self.rows]
        places = [int(row[self.bg_col]) if self.bg_col and row.get(self.bg_col, "") != "" else -1 for row in self.rows]
        return labels, places


def build_dataset(
    cfg: dict[str, Any],
    split: str,
    *,
    is_train: bool = False,
    include_group_labels: bool | None = None,
) -> Dataset:
    """
    根据 ``cfg["data"]["format"]`` 实例化对应数据集。

    Parameters
    ----------
    cfg : dict
        完整配置，需含 ``data.root``、``data.format`` 等。
    split : str
        ``train`` | ``val`` | ``test``。
    is_train : bool
        是否用于优化步骤。仅该模式可启用随机图像增强；因此 ``split="train"``
        的确定性评估与 concept inference 传入 False。
    include_group_labels : bool | None
        仅对 CelebA 有效：True 强制加载 group；False 强制 annotation-free；
        None 则从 cfg / 环境变量推断。

    Returns
    -------
    Dataset
        实现 ``__getitem__`` 返回 dict 的数据集。

    Raises
    ------
    ValueError
        不支持的 ``data.format``。
    """
    data = cfg["data"]
    transform = build_transform(int(data.get("image_size", 224)))
    root = Path(str(data["root"]))
    fmt = str(data.get("format", "waterbirds_parquet")).lower()
    if fmt == "waterbirds_parquet":
        return WaterbirdsParquetDataset(root=root, split=split, transform=transform)
    if fmt == "nico":
        return NICOStage1Dataset(root=root, split=split, augment=bool(is_train))
    if fmt == "celeba_attr":
        annotation_free = _annotation_free_mode_from_cfg(cfg) if include_group_labels is None else not bool(include_group_labels)
        return CelebAAttrDataset(root=root, split=split, transform=transform, annotation_free=bool(annotation_free))
    if fmt == "imagefolder":
        return ImageFolderDataset(root=root / split, transform=transform)
    if fmt == "csv":
        csv_path = Path(str(data["csv_path"]))
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        split_col = str(data.get("split_column", "split"))
        with csv_path.open("r", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row.get(split_col, split) == split]
        return CsvImageDataset(
            rows=rows,
            image_root=root,
            image_col=str(data.get("image_column", "image")),
            label_col=str(data.get("label_column", "label")),
            bg_col=str(data.get("bg_label_column", "bg_label")),
            transform=transform,
        )
    raise ValueError(f"Unsupported data.format: {fmt}")


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    将 list of sample dict 堆叠为 Stage-I 训练用的 batch dict。

    输出键：
    - ``images``：[B, C, H, W]
    - ``labels``：[B] long
    - ``paths``：长度 B 的路径字符串列表
    - 可选 ``indices``、``bg_labels``
    """
    out = {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "labels": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
        "paths": [str(x.get("path", "")) for x in batch],
    }
    if any("index" in x for x in batch):
        out["indices"] = torch.tensor([int(x.get("index", -1)) for x in batch], dtype=torch.long)
    has_bg_label = ["bg_label" in sample for sample in batch]
    if any(has_bg_label) and not all(has_bg_label):
        raise ValueError(
            "A Stage-I batch must either provide bg_label for every sample or for none; "
            "mixed sample schemas would misalign labels and images."
        )
    if all(has_bg_label):
        out["bg_labels"] = torch.tensor([int(sample["bg_label"]) for sample in batch], dtype=torch.long)
    return out


def build_dataloader(
    cfg: dict[str, Any],
    split: str,
    is_train: bool,
    *,
    include_group_labels: bool | None = None,
) -> DataLoader:
    """
    构建 Stage-I DataLoader（含可复现 shuffle 种子）。

    Parameters
    ----------
    cfg : dict
        读取 ``train.batch_size``、``eval.batch_size``、``runtime`` 等。
    split : str
        数据集划分。
    is_train : bool
        True 时 shuffle 并使用训练 batch size。

    Returns
    -------
    DataLoader
        ``collate_fn=collate_batch``，不同 split 使用不同 seed offset 避免混 shuffle。
    """
    ds = build_dataset(cfg, split, is_train=is_train, include_group_labels=include_group_labels)
    batch_size = int(cfg["train"]["batch_size"] if is_train else cfg.get("eval", {}).get("batch_size", cfg["train"]["batch_size"]))
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    num_workers = int(runtime_cfg.get("num_workers", 4))
    base_seed = int(runtime_cfg.get("seed", cfg.get("seed", 42)))
    # 不同 split 使用不同 loader 种子，避免 train/val 随机序列耦合
    split_offset = {"train": 0, "val": 1, "test": 2}.get(str(split).lower(), 7)
    loader_seed = base_seed + split_offset
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": bool(is_train),
        "num_workers": num_workers,
        "collate_fn": collate_batch,
        "pin_memory": torch.cuda.is_available(),
        "generator": make_torch_generator(loader_seed),
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = seed_worker
        kwargs["persistent_workers"] = bool(runtime_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(runtime_cfg.get("prefetch_factor", 4))
    return DataLoader(ds, **kwargs)
