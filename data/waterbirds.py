"""
Waterbirds 数据集加载器（Stage-II 主线）
=======================================

Spurious correlation 基准：预测 **鸟种**（label），背景 **place** 为混淆因素。
Stage-II 训练通过 ``data/stage1_wrapper.WaterbirdsWithStage1`` 附加 Stage-I 信号；
本模块提供标准 ``(image, label)`` 或 ``(image, label, place)`` 接口。

数据来源（典型路径）
--------------------
::

    {data_dir}/waterbirds/
        data/train-00000-of-00001.parquet
        data/validation-00000-of-00001.parquet
        data/test-00000-of-00001.parquet

Parquet schema
--------------
columns: ``image``, ``label``, ``place``, ``bird``；其中 ``image`` 为 dict：
``{'bytes': <jpeg bytes>, 'path': <filename>}``。

流水线位置
----------
- **阶段**：Stage-II 训练 / 评估（``WaterbirdsDataset``）；Stage-I 也可用 ``data/dataloader/stage1.py 中的 WaterbirdsParquetDataset``
- **输入**：``setname``、``args.data_dir``、是否 augment
- **输出**：tensor 图像 + 整数标签（+ 可选 place）
"""
from __future__ import annotations

import os.path as osp
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _build_default_transform(backbone_class: str, augment: bool) -> transforms.Compose:
    """
    Waterbirds 标准 224×224 预处理（对齐 ImageNet 预训练 ResNet）。

    Parameters
    ----------
    backbone_class : str
        参数用于统一数据集 API；Res18/Res50 使用相同的 normalize。
    augment : bool
        True：RandomResizedCrop + 水平翻转；False：Resize(256) + CenterCrop。

    Returns
    -------
    transforms.Compose
        含 ImageNet mean/std 归一化。
    """
    image_size = 224
    if augment:
        tfm_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        tfm_list = [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]

    # ResNet18/50 使用 ImageNet 统计量
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(tfm_list + [normalize])


def _build_strong_transform(backbone_class: str) -> transforms.Compose:
    """强增强入口：等价于 ``alpha=1`` 的渐进式强增强。"""
    return _build_progressive_strong_transform(backbone_class=backbone_class, alpha=1.0)


def _build_progressive_strong_transform(backbone_class: str, alpha: float) -> transforms.Compose:
    """
    渐进式强增强（``alpha`` ∈ [0, 1]）。

    - ``alpha=0``：近似默认 train augment（仅随机裁剪 + 翻转）
    - ``alpha=1``：完整强增强（ColorJitter、RandomGrayscale、RandomErasing）

    设计意图：对 **冲突组**（label 与 place 不一致的样本）加强像素多样性，
    不改变语义（不做背景合成），缓解 spurious 依赖。
    """
    a = float(max(0.0, min(1.0, alpha)))
    image_size = 224
    tfm_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3 * a, contrast=0.3 * a, saturation=0.3 * a, hue=0.05 * a),
        transforms.RandomGrayscale(p=0.1 * a),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25 * a, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(tfm_list + [normalize])


class WaterbirdsDataset(Dataset):
    """
    Waterbirds 稀疏相关数据集（parquet 版）。

    Parameters
    ----------
    setname : str
        ``train`` | ``val`` | ``test``。
    args : namespace-like
        需 ``data_dir``、``backbone_class``。
    augment : bool
        训练时是否增强（仅 ``setname=='train'`` 时生效）。
    return_place : bool
        True 时 ``__getitem__`` 返回 ``(img, label, place)``，用于 WGA / balanced sampler。

    Attributes
    ----------
    num_class : int
        类别数（通常为 2：水鸟/陆鸟）。
    """

    def __init__(self, setname: str, args, augment: bool = False, return_place: bool = False):
        if setname not in ["train", "val", "test"]:
            raise ValueError(f"setname must be 'train'/'val'/'test', got {setname}")

        self.setname = setname
        self.augment = augment and setname == "train"
        # 默认不返回 place；WGA 评估或 group-balanced 采样时再打开
        self.return_place = bool(return_place)
        # 强增强开关与强度，由 Trainer 在训练过程中动态设置
        self._strong_aug_enabled: bool = False
        self._strong_aug_alpha: float = 0.0

        data_dir = str(getattr(args, "data_dir", "") or "").strip()
        if not data_dir:
            raise ValueError("WaterbirdsDataset requires args.data_dir")
        # 按优先级搜索数据目录的两种大小写形式
        root_candidates = [
            osp.join(data_dir, "waterbirds"),
            osp.join(data_dir, "Waterbirds"),
        ]
        dataset_root = None
        for p in root_candidates:
            if osp.isdir(p):
                dataset_root = p
                break
        if dataset_root is None:
            raise FileNotFoundError(f"Waterbirds dataset root not found in: {root_candidates}")

        split_to_file = {
            "train": "train-00000-of-00001.parquet",
            "val": "validation-00000-of-00001.parquet",
            "test": "test-00000-of-00001.parquet",
        }
        parquet_path = osp.join(dataset_root, "data", split_to_file[setname])
        if not osp.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # 全量读入内存（约 200MB 量级；lazy IO 可作为后续优化）
        df = pd.read_parquet(parquet_path)
        if "image" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Unexpected parquet columns: {list(df.columns)}")

        self._images: List[Dict[str, Any]] = df["image"].tolist()
        self._labels: List[int] = df["label"].astype(int).tolist()

        # place=背景类型，bird=鸟种细分类（可选，group 分析用）
        self._place: Optional[List[int]] = df["place"].astype(int).tolist() if "place" in df.columns else None
        self._bird: Optional[List[int]] = df["bird"].astype(int).tolist() if "bird" in df.columns else None

        self.num_class = len(set(self._labels))

        backbone_class = getattr(args, "backbone_class", "Res18")
        self._backbone_class = backbone_class
        self.transform = _build_default_transform(backbone_class=backbone_class, augment=self.augment)
        self._strong_transform = None

    def enable_strong_augment(self, enabled: bool = True) -> None:
        """
        便捷开关：开启时 ``alpha=1``，关闭时 ``alpha=0``。

        Trainer 若使用渐进式 schedule，应改调 ``set_strong_augment_alpha``。
        """
        self.set_strong_augment_enabled(bool(enabled))
        self.set_strong_augment_alpha(1.0 if enabled else 0.0)

    def set_strong_augment_enabled(self, enabled: bool) -> None:
        """启用/关闭强增强；关闭后始终走 ``self.transform``。"""
        self._strong_aug_enabled = bool(enabled)
        if not self._strong_aug_enabled:
            self._strong_transform = None
        else:
            if self.augment:
                self._strong_transform = _build_progressive_strong_transform(
                    backbone_class=self._backbone_class,
                    alpha=self._strong_aug_alpha,
                )

    def set_strong_augment_alpha(self, alpha: float) -> None:
        """设置渐进式强增强强度；仅在 ``augment`` 且 enabled 时重建 transform。"""
        a = float(max(0.0, min(1.0, alpha)))
        self._strong_aug_alpha = a
        if self.augment and self._strong_aug_enabled:
            self._strong_transform = _build_progressive_strong_transform(backbone_class=self._backbone_class, alpha=a)

    def get_labels_places(self) -> Tuple[List[int], List[int]]:
        """
        返回全量 labels 与 places，供 ``ClassBalancedBatchSampler`` 等使用。

        不改变 ``__getitem__`` 的返回值格式。
        """
        if self._place is None:
            raise RuntimeError("WaterbirdsDataset has no 'place' column loaded.")
        return self._labels, self._place

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        item = self._images[idx]
        label = int(self._labels[idx])

        if isinstance(item, dict) and "bytes" in item:
            img_bytes = item["bytes"]
        else:
            raise ValueError(f"Unexpected image field type: {type(item)}")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Palette images with Transparency", category=UserWarning)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
            # 冲突组 (1,0) 与 (0,1)：label 与 place 不一致，对这类困难样本应用更强 augment
            if (
                self.augment
                and self._strong_aug_enabled
                and (self._place is not None)
                and (self._strong_transform is not None)
            ):
                place = int(self._place[idx])
                if (label, place) in {(1, 0), (0, 1)}:
                    img = self._strong_transform(img)
                else:
                    img = self.transform(img)
            else:
                img = self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode Waterbirds sample at index {idx}") from exc

        if self.return_place:
            if self._place is None:
                raise RuntimeError("return_place=True but parquet has no 'place' column.")
            place = int(self._place[idx])
            return img, label, place
        return img, label
