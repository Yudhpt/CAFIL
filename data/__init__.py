"""
数据加载器模块
============
该模块包含各种数据集的数据加载器实现。

作者: FDL-Natural项目组
"""

from .waterbirds import WaterbirdsDataset
from .celeba import CelebADataset
from .nico import NICODataset

__all__ = [
    'WaterbirdsDataset',
    'CelebADataset',
    'NICODataset',
]
