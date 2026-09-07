#!/usr/bin/env python3
"""CAFIL 四阶段流水线 — Stage-II 概念推断入口（仓库根 shim）。

Stage-I 训练完成后运行本脚本，产出 ``P.npy`` 与 ``consscore.npy``，供 Stage-II 使用。
实现见 ``train/concept_infer.py``。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.concept_infer import main  # noqa: E402


if __name__ == "__main__":
    main()
