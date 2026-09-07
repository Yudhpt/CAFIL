"""Validation for the Stage-I to Stage-II on-disk contract."""
from pathlib import Path

import numpy as np

def load_stage1_artifacts(stage1_dir: str | Path, dataset_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return validated P[N,K] and consensus scores s[N]."""
    root = Path(stage1_dir)
    P = np.load(root / "P.npy").astype(np.float32)
    s = np.load(root / "consscore.npy").astype(np.float32)
    if P.ndim != 2 or P.shape[1] < 1:
        raise ValueError(f"P.npy must have shape [N, K] with K >= 1, got {P.shape}")
    if s.ndim != 1:
        raise ValueError(f"consscore.npy must have shape [N], got {s.shape}")
    if P.shape[0] != dataset_length or s.shape[0] != dataset_length:
        raise ValueError(f"Dataset/Stage1 length mismatch: len={dataset_length} P={P.shape} s={s.shape}")
    if not np.isfinite(P).all() or not np.isfinite(s).all() or (P < 0).any():
        raise ValueError("Stage-I artifacts must be finite non-negative values")
    row_sum = P.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-4):
        raise ValueError(f"P rows are not normalized: min={row_sum.min():.6f}, max={row_sum.max():.6f}")
    return P, s
