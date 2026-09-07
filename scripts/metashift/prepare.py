#!/usr/bin/env python3
"""生成 SubpopBench 口径的 MetaShift Cat/Dog metadata。

划分与标签编码与 ``subpopbench/scripts/download.py::generate_metadata_metashift`` 一致：
  - 图像来源：``<data_root>/metashift/MetaShift-Cat-Dog-indoor-outdoor/{train,test}/...``
  - 随机划分：seed=42，test_pct=0.25，val_pct=0.10，split ∈ {0,1,2}
  - y/a 数值与 SubpopBench 生成脚本相同（见 data/metashift.py 文档）

输出：
  1) ``<data_root>/metashift/metadata_metashift.csv``  — SubpopBench 原生列
  3) ``<data_root>/metashift/metadata.csv``           — 本仓库统一列（split 为 train/val/test）

用法：
  python scripts/metashift/prepare.py --data-root "$CAFIL_DATA_ROOT"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_META_SHIFT_FOLDERS = {
    "train/cat/cat(indoor)": (1, 1),
    "train/dog/dog(outdoor)": (0, 0),
    "test/cat/cat(outdoor)": (1, 0),
    "test/dog/dog(indoor)": (0, 1),
}
_SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def group_id(label: int, place: int) -> int:
    """Encode the Cat/Dog and indoor/outdoor labels as one group id."""
    return int(label) * 2 + int(place)


def split_name(split: int) -> str:
    """Translate the SubpopBench numeric split to CAFIL's split name."""
    return _SPLIT_NAMES[int(split)]


_SUBPOP_SEED = 42
_TEST_PCT = 0.25
_VAL_PCT = 0.10


def build_subpopbench_rows(data_root: Path) -> pd.DataFrame:
    """复刻 SubpopBench ``generate_metadata_metashift``。"""
    ms_dir = data_root / "metashift"
    all_data: list[dict[str, object]] = []
    for rel_dir, (y, a) in _META_SHIFT_FOLDERS.items():
        folder = ms_dir / "MetaShift-Cat-Dog-indoor-outdoor" / rel_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"MetaShift folder missing: {folder}")
        for img_path in sorted(folder.glob("*.jpg")):
            all_data.append({"filename": str(img_path.resolve()), "y": int(y), "a": int(a)})
    if not all_data:
        raise RuntimeError("No MetaShift images found under MetaShift-Cat-Dog-indoor-outdoor")
    df = pd.DataFrame(all_data)
    rng = np.random.RandomState(_SUBPOP_SEED)
    n = len(df)
    test_idxs = rng.choice(np.arange(n), size=int(n * _TEST_PCT), replace=False)
    remain = np.setdiff1d(np.arange(n), test_idxs)
    val_idxs = rng.choice(remain, size=int(n * _VAL_PCT), replace=False)
    split_array = np.zeros(n, dtype=np.int64)
    split_array[val_idxs] = 1
    split_array[test_idxs] = 2
    df["split"] = split_array.astype(int)
    return df


def write_unified_csv(df: pd.DataFrame, out_path: Path, *, data_root: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "path", "y", "a", "group"])
        writer.writeheader()
        for row in df.itertuples(index=False):
            path = Path(str(row.filename))
            try:
                rel = path.relative_to(data_root)
                path_str = str(rel)
            except ValueError:
                path_str = str(path)
            y, a = int(row.y), int(row.a)
            writer.writerow(
                {
                    "split": split_name(int(row.split)),
                    "path": path_str,
                    "y": y,
                    "a": a,
                    "group": group_id(y, a),
                }
            )


def summarize(df: pd.DataFrame) -> str:
    lines = ["MetaShift (SubpopBench) split / group counts", f"total={len(df)}"]
    for split_id in (0, 1, 2):
        sub = df[df["split"] == split_id]
        lines.append(f"\n[{split_name(split_id)}] n={len(sub)}")
        for y in sorted(sub["y"].unique()):
            for a in sorted(sub["a"].unique()):
                n = int(((sub["y"] == y) & (sub["a"] == a)).sum())
                g = group_id(int(y), int(a))
                lines.append(f"  y={int(y)} a={int(a)} group={g}: {n}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Directory set as CAFIL_DATA_ROOT in the configs.")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    df = build_subpopbench_rows(data_root)

    subpop_out = data_root / "metashift" / "metadata_metashift.csv"
    subpop_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(subpop_out, index=False)

    unified_out = data_root / "metashift" / "metadata.csv"
    write_unified_csv(df, unified_out, data_root=data_root)

    print(summarize(df))
    print(f"\nWrote:\n  {subpop_out}\n  {unified_out}")


if __name__ == "__main__":
    main()
