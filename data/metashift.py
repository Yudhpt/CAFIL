"""
MetaShift 数据集加载器（支持的补充 benchmark）
============================================================

SubpopBench 风格的 **Cat/Dog × indoor/outdoor** 划分，metadata 来自
``metadata_metashift.csv``。用于仓库内 supplementary experiments。

流水线位置
----------
- **阶段**：Stage-II 训练 / 评估（补充 benchmark）
- **输入**：``setname``、``args.data_dir``、CSV 中的 ``filename`` / ``y`` / ``a`` / ``split``
- **输出**：``(image, label)`` 或 ``(image, label, place)``

标签语义
--------
- ``y``：dog=0, cat=1
- ``a``（place）：outdoor=0, indoor=1
- ``group_labels = y * 2 + a``：四组联合编码
"""
from __future__ import annotations

import os.path as osp
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _build_transform(augment: bool) -> transforms.Compose:
    """
    MetaShift 标准 ResNet 预处理（224×224 + ImageNet normalize）。

    与 Waterbirds/NICO 策略一致：train 用 RandomResizedCrop，eval 用 CenterCrop。
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        tfm = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        tfm = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    return transforms.Compose(tfm)


_SPLIT_MAP = {"train": 0, "val": 1, "test": 2}


class MetaShiftDataset(Dataset):
    """
    SubpopBench MetaShift：二分类 × 二环境。

    Parameters
    ----------
    setname : str
        ``train`` | ``val`` | ``test``，对应 CSV ``split`` 列 0/1/2。
    args : namespace-like
        ``data_dir`` 指向 natural 根目录。
    augment : bool
        训练增强。
    return_place : bool
        是否返回环境标签 ``a``。
    """

    def __init__(
        self,
        setname: str,
        args,
        augment: bool = False,
        return_place: bool = False,
    ):
        if setname not in _SPLIT_MAP:
            raise ValueError(f"setname must be train/val/test, got {setname!r}")
        self.setname = setname
        self.return_place = bool(return_place)
        self.augment = bool(augment) and setname == "train"
        self.transform = _build_transform(self.augment)

        raw_data_dir = str(getattr(args, "data_dir", "") or "").strip()
        if not raw_data_dir:
            raise ValueError("MetaShiftDataset requires args.data_dir")
        data_dir = Path(raw_data_dir)
        # 优先用户 data_dir 下 metashift/，否则 fallback 到仓库 bundled metadata
        csv_candidates = [
            data_dir / "metashift" / "metadata_metashift.csv",
            Path(__file__).resolve().parent / "metadata" / "metadata_metashift.csv",
        ]
        csv_path: Optional[Path] = None
        for p in csv_candidates:
            if p.is_file():
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError(f"metadata_metashift.csv not found under {data_dir}")

        df = pd.read_csv(csv_path)
        split_id = _SPLIT_MAP[setname]
        df = df[df["split"].astype(int) == split_id].reset_index(drop=True)
        self._paths = df["filename"].astype(str).tolist()
        self._labels = df["y"].astype(int).to_numpy()
        self._places = df["a"].astype(int).to_numpy()
        # 四组 id：便于某些评估脚本一次性索引
        self._group_labels = self._labels * 2 + self._places

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        path = self._paths[idx]
        if not osp.isabs(path):
            path = osp.abspath(path)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = int(self._labels[idx])
        if self.return_place:
            return x, y, int(self._places[idx])
        return x, y
