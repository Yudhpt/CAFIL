"""
Stage-II 数据集包装：注入 Stage-I  per-sample 信号
==================================================

在基础数据集（如 ``WaterbirdsDataset``）之上加载 Stage-I 产物：
``P.npy``（概念软分配）与 ``consscore.npy``（一致性分数 ``s_i``），使 Stage-II
每个 batch 样本附带 ``P_i, s_i``。

流水线位置
----------
- **阶段**：Stage-II 训练（``train/train_stage2.py``）
- **输入**：
  - 基础 ``Dataset`` 类与 ``stage1_dir`` 目录
  - Stage-I 输出的 numpy 文件（与训练集样本 **逐行对齐**）
- **输出**：``__getitem__`` 返回 ``(image, y, P_i, s_i, sample_idx)`` 五元组

重要约束
--------
- ``P`` 每行必须近似归一化为 1（概念分配概率）
- ``len(dataset)`` 必须与 ``P/s`` 第一维长度一致，否则立即 ``ValueError``
- ``consscore.npy`` 存 **原始** ``s_i``，供 Stage-II 损失或采样使用（见论文对齐规则）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import numpy as np
import torch
from torch.utils.data import Dataset

from data.stage1_artifacts import load_stage1_artifacts


class DatasetWithStage1(Dataset):
    """
    通用 Stage-I 信号包装器：任意返回 ``(img, y)`` tuple 的数据集均可使用。

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        基础数据集类（构造签名需兼容 ``setname, args, augment, return_place``）。
    setname : str
        ``train`` | ``val`` | ``test``。
    args : Any
        传给基础数据集的配置对象（通常含 ``data_dir``、``backbone_class``）。
    stage1_dir : str | Path
        Stage-I 产物目录，内含 ``P.npy``、``consscore.npy``。
    augment : bool
        是否对训练集做数据增强。

    Returns (via __getitem__)
    ------------------------
    tuple
        ``(img, y, p_i, s_i, idx)`` — ``p_i`` 为长度 K 的概念向量，``s_i`` 为标量 float。
    """

    def __init__(
        self,
        dataset_cls: Type[Dataset],
        setname: str,
        args: Any,
        stage1_dir: str | Path,
        augment: bool = False,
    ) -> None:
        # Stage-II 不需要 spurious place 标签，故 return_place=False
        self.base = dataset_cls(setname=setname, args=args, augment=augment, return_place=False)
        self.stage1_dir = Path(stage1_dir)
        # 加载 Stage-I 概念分配矩阵 [N, K]
        self.P, self.s = load_stage1_artifacts(self.stage1_dir, len(self))
        if len(self) != int(self.P.shape[0]) or len(self) != int(self.s.shape[0]):
            raise ValueError(
                f"Dataset/Stage1 length mismatch: len={len(self)} P={self.P.shape} s={self.s.shape}"
            )
        # P 的每一行应表示对 K 个概念的原型分配，行和为 1
        row_sum = self.P.sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=1.0e-4):
            raise ValueError(f"P rows are not normalized: min={row_sum.min():.6f}, max={row_sum.max():.6f}")

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str):
        """将 ``labels``、``num_class`` 等属性透传给底层数据集。"""
        return getattr(self.base, name)

    def __getitem__(self, idx: int):
        """返回 Stage-II 训练所需的五元组，idx 保证与 P/s 行对齐。

        字段语义
        --------
        - ``img``：经基础数据集增广后的图像张量
        - ``y``：类别标签（int）
        - ``p_i``：Stage-I 软分配 P[idx, :] ∈ R^K，行和≈1
        - ``s_i``：Stage-I 共识分数 consscore[idx]（raw，非 top-q 变换）
        - ``idx``：样本在训练集中的整数下标（用于 artifact 行对齐）
        """
        base_item = self.base[idx]
        if isinstance(base_item, tuple):
            img, y = base_item[:2]
        else:
            raise TypeError(f"Unexpected base dataset item type: {type(base_item)}")
        # 从 numpy 转为 tensor；不 copy 大矩阵，仅取第 idx 行
        p_i = torch.from_numpy(self.P[idx])
        s_i = torch.tensor(float(self.s[idx]), dtype=torch.float32)
        return img, int(y), p_i, s_i, int(idx)
