#!/usr/bin/env python3
"""CAFIL 四阶段流水线 — Stage-II 训练入口（仓库根 shim）。

读取 Stage-I 的 ``P.npy`` / ``consscore.npy``，训练 ResNet + cafil_align 损失。
实现见 ``train/train_stage2.py``。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_stage2 import main  # noqa: E402


if __name__ == "__main__":
    main()
