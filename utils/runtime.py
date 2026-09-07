"""
CAFIL 通用小工具（无训练循环）
==============================

提供 **随机种子**、**梯度开关** 等与具体阶段无关的基础函数，被 Stage-I/II
数据加载、训练脚本广泛 import。

流水线位置
----------
- **阶段**：全流水线（Stage-I、概念推断、Stage-II、推理）
- **输入**：模块实例、种子整数等
- **输出**：副作用（设置全局 RNG / cuDNN）或 ``torch.Generator`` 对象

本模块 deliberately **不包含** 训练循环、损失或模型定义，以保持依赖方向清晰：
``utils`` → 被 ``train/``、``data/`` 调用，而不反向依赖训练逻辑。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    统一设置某个 ``nn.Module`` 下所有参数的 ``requires_grad`` 标志。

    Stage-I 中用于冻结 DINO / Slot / Probe 等不同子模块（见 ``utils.stage1._apply_stage1_train_masks``）。

    Parameters
    ----------
    module : nn.Module
        目标子网络。
    requires_grad : bool
        True 表示参与反向传播；False 表示冻结。
    """
    for p in module.parameters():
        p.requires_grad_(bool(requires_grad))


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool | None = None,
    deterministic_warn_only: bool = False,
) -> int:
    """
    设置 Python / NumPy / PyTorch 全局随机种子，并配置确定性运行环境。

    Parameters
    ----------
    seed : int
        随机种子。
    deterministic : bool, default True
        是否启用 PyTorch 确定性算法（复现实验时建议 True）。
    cudnn_benchmark : bool | None
        若为 None：deterministic 时关闭 benchmark；非 deterministic 时保持原设置。
    deterministic_warn_only : bool
        传给 ``torch.use_deterministic_algorithms`` 的 ``warn_only`` 参数。

    Returns
    -------
    int
        规范化后的种子值（便于调用方记录）。

    Notes
    -----
    - 设置 ``PYTHONHASHSEED`` 保证 dict 等哈希顺序可复现。
    - deterministic 模式下设置 ``CUBLAS_WORKSPACE_CONFIG`` 避免部分 CUDA 算子非确定性。
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(deterministic_warn_only))
    except TypeError:
        torch.use_deterministic_algorithms(bool(deterministic))
    except AttributeError:
        pass

    torch.backends.cudnn.deterministic = bool(deterministic)
    if cudnn_benchmark is None:
        torch.backends.cudnn.benchmark = False if deterministic else torch.backends.cudnn.benchmark
    else:
        torch.backends.cudnn.benchmark = False if deterministic else bool(cudnn_benchmark)
    return seed


def make_torch_generator(seed: int) -> torch.Generator:
    """
    创建带固定种子的 ``torch.Generator``，供 ``DataLoader(..., generator=...)`` 使用。

    与 ``set_global_seed`` 配合：全局种子保证模型初始化；loader generator 保证
    shuffle 顺序可复现且不同 split 可用不同 offset（见 ``data/common.build_dataloader``）。
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id: int) -> None:
    """
    DataLoader ``worker_init_fn``：让每个 worker 进程的 NumPy/Python RNG 与 PyTorch 一致。

    Parameters
    ----------
    worker_id : int
        DataLoader 传入的 worker 编号（本实现未直接使用，仅满足签名）。

    Notes
    -----
    ``torch.initial_seed()`` 已含 base_seed 与 worker_id 信息，据此派生 worker 级种子。
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
