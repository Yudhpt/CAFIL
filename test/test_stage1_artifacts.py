from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.stage1_artifacts import (
    RELATIVE_IDENTITY_SCHEME,
    hash_sample_paths,
    load_identity_scheme,
    load_stage1_artifacts,
    publish_stage1_artifacts,
)


def _arrays(dataset_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sample_ids = hash_sample_paths(
        [dataset_root / "images/a.jpg", dataset_root / "images/b.jpg"],
        scheme=RELATIVE_IDENTITY_SCHEME,
        dataset_root=dataset_root,
    )
    arrays = {
        "P.npy": np.asarray([[0.75, 0.25], [0.1, 0.9]], dtype=np.float32),
        "consscore.npy": np.asarray([-1.5, 2.0], dtype=np.float32),
        "sample_ids.npy": sample_ids,
        "labels.npy": np.asarray([0, 1], dtype=np.int64),
    }
    return arrays, sample_ids


def _load(stage1_dir: Path, sample_ids: np.ndarray):
    return load_stage1_artifacts(
        stage1_dir,
        2,
        expected_sample_ids=sample_ids,
        expected_labels=np.asarray([0, 1], dtype=np.int64),
    )


def test_relative_sample_identity_is_portable_between_dataset_roots(tmp_path):
    left = tmp_path / "machine-a"
    right = tmp_path / "machine-b"
    left_ids = hash_sample_paths(
        [left / "images/a.jpg"],
        scheme=RELATIVE_IDENTITY_SCHEME,
        dataset_root=left,
    )
    right_ids = hash_sample_paths(
        [right / "images/a.jpg"],
        scheme=RELATIVE_IDENTITY_SCHEME,
        dataset_root=right,
    )
    np.testing.assert_array_equal(left_ids, right_ids)


def test_publish_and_load_accepts_signed_consensus_scores(tmp_path):
    arrays, sample_ids = _arrays(tmp_path / "dataset")
    publish_stage1_artifacts(tmp_path / "stage1", arrays, identity_root=tmp_path)

    probability, score = _load(tmp_path / "stage1", sample_ids)

    np.testing.assert_allclose(probability, arrays["P.npy"])
    np.testing.assert_allclose(score, arrays["consscore.npy"])


def test_load_rejects_reordered_sample_identity(tmp_path):
    arrays, sample_ids = _arrays(tmp_path / "dataset")
    arrays["sample_ids.npy"] = sample_ids[::-1]
    publish_stage1_artifacts(tmp_path / "stage1", arrays, identity_root=tmp_path)

    with pytest.raises(ValueError, match="sample identity mismatch"):
        _load(tmp_path / "stage1", sample_ids)


def test_load_rejects_target_label_mismatch(tmp_path):
    arrays, sample_ids = _arrays(tmp_path / "dataset")
    arrays["labels.npy"] = np.asarray([1, 0], dtype=np.int64)
    publish_stage1_artifacts(tmp_path / "stage1", arrays, identity_root=tmp_path)

    with pytest.raises(ValueError, match="target label mismatch"):
        _load(tmp_path / "stage1", sample_ids)


def test_load_rejects_artifact_changed_after_manifest(tmp_path):
    arrays, sample_ids = _arrays(tmp_path / "dataset")
    stage1_dir = tmp_path / "stage1"
    publish_stage1_artifacts(stage1_dir, arrays, identity_root=tmp_path)
    np.save(stage1_dir / "consscore.npy", np.asarray([9.0, 9.0], dtype=np.float32))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load(stage1_dir, sample_ids)


def test_name_based_extension_manifest_is_supported(tmp_path):
    arrays, sample_ids = _arrays(tmp_path / "dataset")
    stage1_dir = tmp_path / "stage1"
    manifest_path = publish_stage1_artifacts(stage1_dir, arrays, identity_root=tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["arrays"] = [
        {"name": name, "path": str(stage1_dir / name), **row}
        for name, row in payload.pop("artifacts").items()
    ]
    manifest_path.unlink()
    (stage1_dir / "concept_intermediates_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    _load(stage1_dir, sample_ids)


def test_multiple_identity_manifests_are_rejected(tmp_path):
    arrays, _ = _arrays(tmp_path / "dataset")
    stage1_dir = tmp_path / "stage1"
    manifest_path = publish_stage1_artifacts(stage1_dir, arrays, identity_root=tmp_path)
    (stage1_dir / "concept_intermediates_manifest.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Multiple Stage-I identity manifests"):
        load_identity_scheme(stage1_dir)
