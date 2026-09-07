#!/usr/bin/env python3
"""CAFIL 四阶段流水线 — Stage-IV 推理/评估入口（仓库根 shim）。

加载 Stage-II checkpoint，在 val/test 上报告 mean acc 与 WGA。
实现见 ``train/inference.py``。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.inference import main  # noqa: E402


if __name__ == "__main__":
    main()
