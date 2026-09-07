from __future__ import annotations

from pathlib import Path

import pytest

from utils.config_io import load_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIGS = sorted((ROOT / "config").glob("*/*.yaml"))


def test_public_configs_require_explicit_environment_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CAFIL_DATA_ROOT", raising=False)
    monkeypatch.delenv("CAFIL_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("CAFIL_DINO_HOME", raising=False)

    with pytest.raises(ValueError, match="CAFIL_OUTPUT_ROOT"):
        load_config(ROOT / "config/waterbirds/stage1.yaml")


def test_all_formal_configs_resolve_public_environment_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("CAFIL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CAFIL_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("CAFIL_DINO_HOME", str(tmp_path / "dino"))

    assert len(FORMAL_CONFIGS) == 12
    for path in FORMAL_CONFIGS:
        cfg = load_config(path)
        assert "${" not in repr(cfg), path

    stage1 = load_config(ROOT / "config/waterbirds/stage1.yaml")
    stage2 = load_config(ROOT / "config/waterbirds/stage2.yaml")
    concept = load_config(ROOT / "config/waterbirds/concept_infer.yaml")

    assert stage1["paths"]["data_root"] == str(tmp_path / "data")
    assert stage1["method"]["stage1"]["dino_torch_home"] == str(tmp_path / "dino")
    expected = str(tmp_path / "outputs" / "cafil_best" / "waterbirds" / "concept")
    assert stage2["dataset"]["stage1_dir"] == expected
    assert concept["output_dir"] == expected
