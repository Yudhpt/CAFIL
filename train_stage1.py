#!/usr/bin/env python3
"""CAFIL 四阶段流水线 — Stage-I 训练入口（仓库根 shim）。

本文件 **不包含训练逻辑**，仅做两件事：
1. 把仓库根目录加入 ``sys.path``，使 ``from train...`` 可导入；
2. 调用 ``train.train_stage1.main()``。

实际实现见 ``train/train_stage1.py``（DINO + Slot Attention + probe）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 仓库根 = 本文件所在目录；保证 ``train/`` ``data/`` 等包可被 Python 找到
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_stage1 import main  # noqa: E402


if __name__ == "__main__":
    # 等价于: python -m train.train_stage1 --config config/waterbirds/stage1.yaml
    main()
