"""
NICO 数据集加载器（Stage-II 主线）
==================================

NICO（Natural Images with Context）：动物 **类别** × 拍摄 **场景 (context)**，
用于研究跨场景泛化与 spurious correlation。文件名编码 label 与 context。

数据来源
--------
::

    {data_dir}/nico/
        NICO/multi_classification/{train,val,test}/*.jpg
        Animal_name2label.json
        Context_name2label.json

文件名格式
----------
``{label}_{context}_{index}.jpg``，例如 ``0_0_1.jpg`` 表示动物类 0、场景 0。

流水线位置
----------
- **阶段**：Stage-II 训练 / 评估
- **输入**：split、``args.data_dir``、是否 augment / return_place
- **输出**：``(image, label)`` 或 ``(image, label, place)``，place 即 context id

采样策略
--------
训练常配合 **类平衡采样**（``ClassBalancedBatchSampler``），与 CIFAR 类设定一致。
"""
from __future__ import annotations

import json
import os
import os.path as osp
import re
import warnings
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _build_transform(backbone_class: str, augment: bool) -> transforms.Compose:
    """
    NICO 标准 224×224 预处理（ImageNet ResNet 惯例）。

    训练：RandomResizedCrop + 水平翻转；评估：Resize(256) + CenterCrop(224)。
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(tfm_list + [normalize])


# 文件名正则：label_context_index.ext
_FNAME_PATTERN = re.compile(r"^(\d+)_(\d+)_\d+\.(jpg|jpeg|png)$", re.IGNORECASE)


def _parse_filename(name: str) -> Optional[Tuple[int, int]]:
    """
    从文件名解析 (label, context)。

    Returns
    -------
    tuple[int, int] | None
        无法解析时返回 None（跳过该文件）。
    """
    m = _FNAME_PATTERN.match(name.strip())
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


class NICODataset(Dataset):
    """
    NICO 多分类数据集（动物 × 场景）。

    Parameters
    ----------
    setname : str
        ``train`` | ``val`` | ``test``。
    args : namespace-like
        ``data_dir``、``backbone_class``。
    augment : bool
        是否训练增强（仅 train split）。
    return_place : bool
        True 时返回 context 作为 place，用于 WGA 与 FDL 环境监督。

    Attributes
    ----------
    num_class : int
        动物类别数（默认 10，可由 JSON 推断）。
    num_place : int
        场景/context 数（默认 33）。
    """

    def __init__(self, setname: str, args, augment: bool = False, return_place: bool = False):
        if setname not in ("train", "val", "test"):
            raise ValueError(f"setname must be 'train'/'val'/'test', got {setname}")

        self.setname = setname
        self.augment = augment and setname == "train"
        self.return_place = bool(return_place)

        data_dir = str(getattr(args, "data_dir", "") or "").strip()
        if not data_dir:
            raise ValueError("NICODataset requires args.data_dir")
        root = osp.join(data_dir, "nico")
        if not osp.isdir(root):
            raise FileNotFoundError(f"NICO root not found: {root}")

        data_folder = osp.join(root, "NICO", "multi_classification")
        split_dir = osp.join(data_folder, setname)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(f"NICO split dir not found: {split_dir}")

        # JSON 仅用于统计类别/场景数，id 仍来自文件名
        cxt_path = osp.join(root, "Context_name2label.json")
        class_path = osp.join(root, "Animal_name2label.json")
        self._n_places = 33
        self._n_classes = 10
        if osp.isfile(cxt_path):
            with open(cxt_path, "r") as f:
                cxt_dic = json.load(f)
            self._n_places = len(cxt_dic)
        if osp.isfile(class_path):
            with open(class_path, "r") as f:
                class_dic = json.load(f)
            self._n_classes = len(class_dic)

        self._paths: List[str] = []
        self._labels: List[int] = []
        self._places: List[int] = []
        for name in os.listdir(split_dir):
            parsed = _parse_filename(name)
            if parsed is None:
                continue
            label, context = parsed
            self._paths.append(osp.join(split_dir, name))
            self._labels.append(label)
            self._places.append(context)

        if len(self._paths) == 0:
            raise RuntimeError(f"No valid images found under {split_dir}")

        self.num_class = self._n_classes
        self.num_place = self._n_places
        backbone_class = getattr(args, "backbone_class", "Res18")
        self.transform = _build_transform(backbone_class=backbone_class, augment=self.augment)

    def get_labels_places(self) -> Tuple[List[int], List[int]]:
        """供 balanced sampler 使用。"""
        return self._labels, self._places

    @property
    def labels(self) -> List[int]:
        """类平衡采样器读取 ``dataset.labels`` 时的别名。"""
        return self._labels

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        path = self._paths[idx]
        label = int(self._labels[idx])
        place = int(self._places[idx])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Palette images with Transparency", category=UserWarning)
                img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode NICO sample at index {idx}: {path}") from exc
        if self.return_place:
            return img, label, place
        return img, label
