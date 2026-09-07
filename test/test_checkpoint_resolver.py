from __future__ import annotations

from pathlib import Path

import pytest

from train.checkpoint_resolver import (
    canonical_stage2_checkpoint_name,
    resolve_stage2_checkpoint,
    stage2_checkpoint_candidates,
)


def _cfg(metric: str, run_dir: Path) -> dict:
    return {
        "eval": {"primary_metric": metric},
        "output": {"run_dir": str(run_dir)},
    }


@pytest.mark.parametrize("metric", ["wga", "worst_group_acc"])
def test_wga_canonical_name(metric: str):
    assert canonical_stage2_checkpoint_name(metric) == "best_wga.pth"


@pytest.mark.parametrize("metric", ["mean", "mean_acc"])
def test_mean_canonical_name(metric: str):
    assert canonical_stage2_checkpoint_name(metric) == "best_val_mean.pth"


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="supported metrics"):
        canonical_stage2_checkpoint_name("test_wga")


def test_wga_candidate_order():
    assert stage2_checkpoint_candidates("wga") == (
        "best_wga.pth",
        "best_val_wga.pth",
        "best.pth",
    )


def test_mean_candidate_order():
    assert stage2_checkpoint_candidates("mean") == (
        "best_val_mean.pth",
        "best.pth",
    )


def test_canonical_wga_precedes_legacy(tmp_path: Path):
    canonical = tmp_path / "best_wga.pth"
    legacy = tmp_path / "best_val_wga.pth"
    canonical.touch()
    legacy.touch()

    assert resolve_stage2_checkpoint(_cfg("wga", tmp_path)) == canonical


def test_legacy_wga_remains_readable(tmp_path: Path):
    legacy = tmp_path / "best_val_wga.pth"
    legacy.touch()

    assert resolve_stage2_checkpoint(_cfg("wga", tmp_path)) == legacy


@pytest.mark.parametrize("metric", ["wga", "mean"])
def test_best_checkpoint_remains_fallback(metric: str, tmp_path: Path):
    fallback = tmp_path / "best.pth"
    fallback.touch()

    assert resolve_stage2_checkpoint(_cfg(metric, tmp_path)) == fallback


def test_missing_candidates_raise(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No Stage-II checkpoint found"):
        resolve_stage2_checkpoint(_cfg("wga", tmp_path))
