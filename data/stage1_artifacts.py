"""Validation helpers for the Stage-I to Stage-II on-disk contract."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


LEGACY_IDENTITY_SCHEME = "sha256_utf8_path"
RELATIVE_IDENTITY_SCHEME = "sha256_utf8_dataset_relative_posix_v1"
IDENTITY_MANIFESTS = (
    "concept_intermediates_manifest.json",
    "stage1_artifacts_manifest.json",
)


def _canonical_relative_path(path: str | Path, dataset_root: str | Path) -> str:
    """Return a portable POSIX path below the dataset root."""
    raw = str(path).strip()
    if not raw:
        raise ValueError("Sample identity requires a non-empty path")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        root = Path(dataset_root).expanduser().resolve()
        try:
            candidate = candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Sample path is outside dataset root: path={path} root={root}"
            ) from exc
    normalized = PurePosixPath(str(candidate).replace("\\", "/")).as_posix()
    if not normalized or ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"Unsafe sample path: {path}")
    return normalized


def hash_sample_paths(
    paths: Iterable[str | Path],
    *,
    scheme: str = RELATIVE_IDENTITY_SCHEME,
    dataset_root: str | Path | None = None,
) -> np.ndarray:
    """Hash ordered sample paths under an explicit, versioned identity scheme."""
    raw_paths = [str(path).strip() for path in paths]
    if not raw_paths or any(not path for path in raw_paths):
        raise ValueError("Sample identity requires non-empty paths")
    if scheme == LEGACY_IDENTITY_SCHEME:
        keys = raw_paths
    elif scheme == RELATIVE_IDENTITY_SCHEME:
        if dataset_root is None:
            raise ValueError("Relative sample identity requires dataset_root")
        keys = [_canonical_relative_path(path, dataset_root) for path in raw_paths]
    else:
        raise ValueError(f"Unsupported sample identity scheme: {scheme!r}")
    identities = np.asarray(
        [hashlib.sha256(key.encode("utf-8")).hexdigest() for key in keys],
        dtype="<U64",
    )
    if len(set(identities.tolist())) != len(identities):
        raise ValueError("Duplicate sample identity detected")
    return identities


def dataset_sample_paths(dataset: Any) -> list[str]:
    """Extract ordered paths from every supported Stage-II dataset adapter."""
    for name in ("_paths", "_img_paths"):
        values = getattr(dataset, name, None)
        if values is not None:
            paths = [str(value) for value in values]
            if len(paths) == len(dataset):
                return paths
    images = getattr(dataset, "_images", None)
    if images is not None:
        paths = [
            str(item.get("path", f"wb_{idx:06d}.jpg"))
            if isinstance(item, dict)
            else ""
            for idx, item in enumerate(images)
        ]
        if len(paths) == len(dataset):
            return paths
    rows = getattr(dataset, "rows", None)
    if rows is not None:
        paths = [str(row[0]) for row in rows]
        if len(paths) == len(dataset):
            return paths
    raise TypeError(f"{type(dataset).__name__} does not expose ordered sample paths")


def dataset_labels(dataset: Any) -> np.ndarray:
    """Extract ordered target labels from a supported Stage-II dataset adapter."""
    for name in ("_labels", "labels"):
        values = getattr(dataset, name, None)
        if values is not None:
            labels = np.asarray(values, dtype=np.int64).reshape(-1)
            if labels.shape == (len(dataset),):
                return labels
    raise TypeError(f"{type(dataset).__name__} does not expose ordered target labels")


def _read_identity_manifest(stage1_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """Read the one complete manifest that defines the Stage-I artifact set."""
    root = Path(stage1_dir)
    manifests = [root / name for name in IDENTITY_MANIFESTS if (root / name).is_file()]
    if not manifests:
        raise FileNotFoundError(
            f"Stage-I identity manifest is required; expected one of "
            f"{IDENTITY_MANIFESTS} under {root}"
        )
    if len(manifests) != 1:
        raise ValueError(f"Multiple Stage-I identity manifests found: {manifests}")
    path = manifests[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Stage-I manifest is not complete: {path}")
    return payload, path


def load_identity_scheme(stage1_dir: str | Path) -> str:
    """Read the complete manifest and return its declared identity scheme."""
    payload, path = _read_identity_manifest(stage1_dir)
    scheme = str(payload.get("sample_identity_scheme", "")).strip()
    if scheme not in {LEGACY_IDENTITY_SCHEME, RELATIVE_IDENTITY_SCHEME}:
        raise ValueError(f"Unsupported sample identity scheme in {path}: {scheme!r}")
    return scheme


def _verify_manifest_artifacts(
    root: Path, payload: dict[str, Any], manifest_path: Path
) -> None:
    """Verify every required array against the manifest before loading it."""
    required = {"P.npy", "consscore.npy", "sample_ids.npy", "labels.npy"}
    declared = payload.get("artifacts")
    if isinstance(declared, dict):
        rows = declared
    else:
        arrays = payload.get("arrays")
        if not isinstance(arrays, list):
            raise ValueError(f"Manifest does not declare artifacts: {manifest_path}")
        rows = {
            str(row.get("name") or Path(str(row.get("path", ""))).name): row
            for row in arrays
            if isinstance(row, dict)
        }
    missing = required.difference(rows)
    if missing:
        raise ValueError(
            f"Manifest is missing required artifacts {sorted(missing)}: {manifest_path}"
        )
    for name in sorted(required):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-declared artifact is missing: {path}")
        row = rows[name]
        expected_bytes = row.get("bytes")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise ValueError(f"Artifact byte-size mismatch: {path}")
        expected_sha = str(row.get("sha256", "")).strip().lower()
        if not expected_sha:
            raise ValueError(f"Manifest is missing SHA-256 for {name}: {manifest_path}")
        if _sha256(path) != expected_sha:
            raise ValueError(f"Artifact SHA-256 mismatch: {path}")


def load_stage1_artifacts(
    stage1_dir: str | Path,
    dataset_length: int,
    *,
    expected_sample_ids: np.ndarray,
    expected_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P/s only after exact sample identity and target-label validation."""
    root = Path(stage1_dir)
    manifest, manifest_path = _read_identity_manifest(root)
    _verify_manifest_artifacts(root, manifest, manifest_path)
    P = np.load(root / "P.npy", allow_pickle=False).astype(np.float32)
    s = np.load(root / "consscore.npy", allow_pickle=False).astype(np.float32)
    sample_ids = np.asarray(
        np.load(root / "sample_ids.npy", allow_pickle=False), dtype=str
    )
    labels = np.asarray(
        np.load(root / "labels.npy", allow_pickle=False), dtype=np.int64
    )
    if P.ndim != 2 or P.shape[1] < 1:
        raise ValueError(f"P.npy must have shape [N, K] with K >= 1, got {P.shape}")
    if s.ndim != 1:
        raise ValueError(f"consscore.npy must have shape [N], got {s.shape}")
    expected_ids = np.asarray(expected_sample_ids, dtype=str)
    expected_y = np.asarray(expected_labels, dtype=np.int64)
    expected_shape = (int(dataset_length),)
    if sample_ids.shape != expected_shape or expected_ids.shape != expected_shape:
        raise ValueError(
            f"Sample identity length mismatch: len={dataset_length} "
            f"artifact={sample_ids.shape} dataset={expected_ids.shape}"
        )
    if labels.shape != expected_shape or expected_y.shape != expected_shape:
        raise ValueError(
            f"Target-label length mismatch: len={dataset_length} "
            f"artifact={labels.shape} dataset={expected_y.shape}"
        )
    if len(set(sample_ids.tolist())) != dataset_length:
        raise ValueError("sample_ids.npy contains duplicate identities")
    mismatch = np.flatnonzero(sample_ids != expected_ids)
    if mismatch.size:
        raise ValueError(
            f"Stage-I sample identity mismatch at row {int(mismatch[0])}"
        )
    label_mismatch = np.flatnonzero(labels != expected_y)
    if label_mismatch.size:
        raise ValueError(
            f"Stage-I target label mismatch at row {int(label_mismatch[0])}"
        )
    if P.shape[0] != dataset_length or s.shape[0] != dataset_length:
        raise ValueError(
            f"Dataset/Stage1 length mismatch: len={dataset_length} "
            f"P={P.shape} s={s.shape}"
        )
    if not np.isfinite(P).all() or (P < 0).any():
        raise ValueError("P.npy must contain finite non-negative probabilities")
    if not np.isfinite(s).all():
        raise ValueError("consscore.npy must contain finite scores")
    row_sum = P.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-4):
        raise ValueError(
            f"P rows are not normalized: min={row_sum.min():.6f}, "
            f"max={row_sum.max():.6f}"
        )
    return P, s


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_stage1_artifacts(
    stage1_dir: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    identity_root: str | Path,
) -> Path:
    """Publish identity-bound arrays atomically and write the complete marker last."""
    required = {"P.npy", "consscore.npy", "sample_ids.npy", "labels.npy"}
    if set(arrays) != required:
        raise ValueError(
            f"Stage-I publication requires exactly {sorted(required)}, got {sorted(arrays)}"
        )
    probability = np.asarray(arrays["P.npy"])
    score = np.asarray(arrays["consscore.npy"])
    identities = np.asarray(arrays["sample_ids.npy"], dtype=str)
    labels = np.asarray(arrays["labels.npy"])
    if probability.ndim != 2 or probability.shape[1] < 1:
        raise ValueError("P.npy must have shape [N, K] with K >= 1")
    n_samples = int(probability.shape[0])
    if score.shape != (n_samples,):
        raise ValueError("consscore.npy must have shape [N]")
    if identities.shape != (n_samples,):
        raise ValueError("sample_ids.npy must have shape [N]")
    if labels.shape != (n_samples,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels.npy must be a one-dimensional integer array")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("P.npy must contain finite non-negative probabilities")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-4):
        raise ValueError("P.npy rows must sum to one")
    if not np.isfinite(score).all():
        raise ValueError("consscore.npy must contain finite scores")
    if len(set(identities.tolist())) != n_samples:
        raise ValueError("Stage-I publication contains duplicate sample identities")
    if any(
        len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity.lower())
        for identity in identities
    ):
        raise ValueError("sample_ids.npy must contain lowercase or uppercase SHA-256 hex IDs")

    root = Path(stage1_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "stage1_artifacts_manifest.json"
    for name in IDENTITY_MANIFESTS:
        (root / name).unlink(missing_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    temporary: list[Path] = []
    try:
        for name in sorted(arrays):
            destination = root / name
            temp = root / f".{name}.{os.getpid()}.tmp"
            temporary.append(temp)
            with temp.open("wb") as handle:
                np.save(handle, np.asarray(arrays[name]), allow_pickle=False)
            os.replace(temp, destination)
            rows[name] = {
                "shape": list(np.asarray(arrays[name]).shape),
                "dtype": str(np.asarray(arrays[name]).dtype),
                "bytes": int(destination.stat().st_size),
                "sha256": _sha256(destination),
            }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "sample_identity_scheme": RELATIVE_IDENTITY_SCHEME,
            "sample_identity_root": "dataset_relative",
            "n_samples": n_samples,
            "artifacts": rows,
        }
        temp_manifest = root / f".{manifest_path.name}.{os.getpid()}.tmp"
        temporary.append(temp_manifest)
        temp_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_manifest, manifest_path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return manifest_path
