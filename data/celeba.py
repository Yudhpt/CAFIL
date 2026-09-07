"""
CelebA 数据集加载器（Spurious Correlation 标准设定）
===================================================

Stage-II 主线数据集之一。任务：预测 **Blond_Hair**；混淆因素 **Male**（性别）。
四组 (label × place) 中「金发男性」为典型 minority group，WGA 关注最差组。

数据路径（典型）
--------------
::

    {data_dir}/CelebA/
        Anno/list_attr_celeba.csv
        Eval/list_eval_partition.txt
        Img/img_align_celeba/*.jpg

流水线位置
----------
- **阶段**：Stage-II 训练 / 评估
- **输入**：``setname``、``args``、``annotation_free_mode``（是否隐藏 group 标签）
- **输出**：``(image, label)`` 或 ``(image, label, place)``

评估指标
--------
Worst-Group Accuracy (WGA)：四组准确率的最小值（标准 spurious correlation 协议）。
"""
from __future__ import annotations

import os
import os.path as osp
import warnings
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _build_default_transform(backbone_class: str, augment: bool) -> transforms.Compose:
    """
    CelebA 标准预处理（对齐主流 spurious correlation 论文）。

    流程：CenterCrop(178) 裁人脸 → Resize(224) → 可选 RandomHorizontalFlip → ImageNet normalize。
    CelebA 原图 178×218，先 center crop 再 resize 是社区通用做法。
    """
    center_crop = transforms.CenterCrop(178)
    resize      = transforms.Resize((224, 224))
    normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    if augment:
        tfm_list = [center_crop, resize,
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), normalize]
    else:
        tfm_list = [center_crop, resize,
                    transforms.ToTensor(), normalize]
    return transforms.Compose(tfm_list)


def _annotation_free_mode(explicit: bool | None = None) -> bool:
    """
    解析是否处于 **无组标注** 模式。

    显式参数优先；否则读环境变量 ``ANNOTATION_FREE_MODE``。
    annotation-free 时不加载 Male 属性，符合 CAFIL 无环境标签训练设定。
    """
    if explicit is not None:
        return bool(explicit)
    raw = str(os.environ.get("ANNOTATION_FREE_MODE", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class CelebADataset(Dataset):
    """
    CelebA Spurious Correlation 任务（Blond_Hair vs. Male）。

    Parameters
    ----------
    setname : str
        ``train`` | ``val`` | ``test``。
    args : Namespace-like
        ``data_dir``、``backbone_class``。
    augment : bool
        训练 split 是否随机水平翻转。
    return_place : bool
        是否返回 Male 作为 place/group 标签。
    eval_mode : bool
        评估模式下即使非 return_place 也可能加载 group（用于 WGA）。
    annotation_free_mode : bool | None
        覆盖全局 annotation-free 开关。

    Groups（标准四组）
    ------------------
    - (非金发, 女)、(非金发, 男)、(金发, 女) 为相对多数
    - (金发, 男) 为 spurious minority，常决定 WGA
    """

    _SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        setname: str,
        args,
        augment: bool = False,
        return_place: bool = True,
        eval_mode: bool = False,
        annotation_free_mode: bool | None = None,
    ):
        if setname not in self._SPLIT_MAP:
            raise ValueError(f"setname must be 'train'/'val'/'test', got {setname!r}")

        self.setname = setname
        self.return_place = bool(return_place)
        self.eval_mode = bool(eval_mode)
        self.annotation_free_mode = _annotation_free_mode(annotation_free_mode)
        self.augment = augment and (setname == "train")

        data_dir = str(getattr(args, "data_dir", "") or "").strip()
        if not data_dir:
            raise ValueError("CelebADataset requires args.data_dir")
        root_candidates = [
            osp.join(data_dir, "CelebA"),
            osp.join(data_dir, "celeba"),
            osp.join(data_dir, "celebA"),
        ]
        dataset_root: Optional[str] = None
        for p in root_candidates:
            if osp.isdir(p):
                dataset_root = p
                break
        if dataset_root is None:
            raise FileNotFoundError(
                f"CelebA root not found. Tried: {root_candidates}"
            )

        img_dir    = osp.join(dataset_root, "Img", "img_align_celeba")
        attr_file  = osp.join(dataset_root, "Anno", "list_attr_celeba.csv")
        split_file = osp.join(dataset_root, "Eval", "list_eval_partition.txt")

        for f in (img_dir, attr_file, split_file):
            if not osp.exists(f):
                raise FileNotFoundError(f"CelebA required file missing: {f}")

        import pandas as pd
        # 仅在需要 group 信息时读取 Male 列
        need_group = bool(self.eval_mode or (self.return_place and not self.annotation_free_mode))
        usecols = ["image_id", "Blond_Hair"] + (["Male"] if need_group else [])
        df_attr = pd.read_csv(attr_file, usecols=usecols)
        # 官方 CSV：-1/ +1 → 转为 0/1
        for col in df_attr.columns:
            if col != "image_id":
                df_attr[col] = ((df_attr[col] + 1) // 2).astype(int)

        split_dict: dict[str, int] = {}
        with open(split_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                split_dict[parts[0]] = int(parts[1])

        target_split = self._SPLIT_MAP[setname]

        self._img_paths: List[str] = []
        self._labels:    List[int] = []
        self._group_labels: List[int] | None = [] if need_group else None

        for _, row in df_attr.iterrows():
            img_name = row["image_id"]
            if split_dict.get(img_name, -1) != target_split:
                continue
            label = int(row["Blond_Hair"])
            self._img_paths.append(osp.join(img_dir, img_name))
            self._labels.append(label)
            if self._group_labels is not None:
                self._group_labels.append(int(row["Male"]))

        if len(self._img_paths) == 0:
            raise RuntimeError(
                f"CelebA: no samples found for split '{setname}'. "
                f"Check that {split_file} matches the images in {img_dir}."
            )

        self.num_class = 2

        backbone_class = getattr(args, "backbone_class", "Res50")
        self.transform = _build_default_transform(backbone_class, self.augment)

        print(f"[CelebA] {setname:5s} | n={len(self._img_paths):>6d} | blond={sum(self._labels):>5d}")

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, idx: int):
        path  = self._img_paths[idx]
        label = self._labels[idx]

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode CelebA sample at index {idx}: {path}") from exc

        if self._group_labels is not None and (self.return_place or self.eval_mode):
            return img, label, self._group_labels[idx]
        return img, label

    def get_labels_places(self) -> Tuple[List[int], List[int]]:
        """返回 (labels, places)；annotation-free 时 places 全为 -1。"""
        if self._group_labels is None:
            return self._labels, [-1] * len(self._labels)
        return self._labels, self._group_labels
