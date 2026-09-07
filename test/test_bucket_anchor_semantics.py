from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from train.stage2_loss import resolve_anchor_update_rate, update_bucket_anchor_


ROOT = Path(__file__).resolve().parents[1]
FORMAL_RATES = {
    "waterbirds": 0.05,
    "celeba": 0.10,
    "nico": 0.90,
    "metashift": 0.95,
}


@pytest.mark.parametrize("rate", [0.05, 0.10, 0.90, 0.95])
def test_resolve_anchor_update_rate(rate: float):
    assert resolve_anchor_update_rate({"anchor_update_rate": rate}) == rate


def test_legacy_ema_beta_is_rejected():
    with pytest.raises(ValueError, match="obsolete"):
        resolve_anchor_update_rate({"ema_beta": 0.05})

def test_missing_rate_uses_historical_effective_default():
    assert resolve_anchor_update_rate({}) == 0.05


@pytest.mark.parametrize("rate", [0.0, 1.0, "0.1"])
def test_valid_boundary_and_numeric_string_rates(rate):
    assert resolve_anchor_update_rate({"anchor_update_rate": rate}) == float(rate)


@pytest.mark.parametrize("rate", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_invalid_range_or_nonfinite_rate_raises(rate: float):
    with pytest.raises(ValueError, match="anchor_update_rate"):
        resolve_anchor_update_rate({"anchor_update_rate": rate})


@pytest.mark.parametrize("rate", [True, False])
def test_bool_rate_raises(rate: bool):
    with pytest.raises(TypeError, match="not bool"):
        resolve_anchor_update_rate({"anchor_update_rate": rate})


def test_invalid_string_rate_raises():
    with pytest.raises(ValueError, match="anchor_update_rate"):
        resolve_anchor_update_rate({"anchor_update_rate": "not-a-rate"})


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("rate", [0.00, 0.05, 0.10, 0.50, 0.90, 0.95, 1.00])
def test_anchor_update_is_numerically_identical_to_legacy_formula(dtype, rate: float):
    generator = torch.Generator(device="cpu").manual_seed(20260728)
    old_anchor = torch.randn(3, 5, generator=generator, dtype=dtype)
    batch_mean = torch.randn(3, 5, generator=generator, dtype=dtype)
    legacy = (1.0 - rate) * old_anchor + rate * batch_mean

    bucket_means = old_anchor.unsqueeze(0).clone()
    bucket_seen = torch.ones((1, 3), dtype=torch.bool)
    for bucket_index in range(3):
        update_bucket_anchor_(
            bucket_means,
            bucket_seen,
            0,
            bucket_index,
            batch_mean[bucket_index],
            rate,
        )

    assert bucket_means.shape == (1, 3, 5)
    assert bucket_means.dtype == dtype
    assert bucket_means.device.type == "cpu"
    if dtype == torch.float64:
        torch.testing.assert_close(bucket_means[0], legacy, rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(bucket_means[0], legacy)


def test_rate_zero_preserves_anchor_and_rate_one_replaces_it():
    old_anchor = torch.tensor([1.0, 2.0])
    batch_mean = torch.tensor([8.0, 9.0])
    seen = torch.ones((1, 1), dtype=torch.bool)

    unchanged = old_anchor.reshape(1, 1, 2).clone()
    update_bucket_anchor_(unchanged, seen.clone(), 0, 0, batch_mean, 0.0)
    torch.testing.assert_close(unchanged[0, 0], old_anchor)

    replaced = old_anchor.reshape(1, 1, 2).clone()
    update_bucket_anchor_(replaced, seen.clone(), 0, 0, batch_mean, 1.0)
    torch.testing.assert_close(replaced[0, 0], batch_mean)


def test_unobserved_bucket_initializes_once_and_missing_bucket_stays_unchanged():
    bucket_means = torch.zeros((1, 2, 3), dtype=torch.float32)
    bucket_seen = torch.zeros((1, 2), dtype=torch.bool)
    first_mean = torch.tensor([1.0, 2.0, 3.0])

    update_bucket_anchor_(bucket_means, bucket_seen, 0, 0, first_mean, 0.05)

    torch.testing.assert_close(bucket_means[0, 0], first_mean)
    torch.testing.assert_close(bucket_means[0, 1], torch.zeros(3))
    assert bucket_seen.tolist() == [[True, False]]


def test_anchor_update_detaches_batch_mean_and_does_not_propagate_gradient():
    bucket_means = torch.zeros((1, 1, 2))
    bucket_seen = torch.ones((1, 1), dtype=torch.bool)
    batch_mean = torch.tensor([2.0, 4.0], requires_grad=True)

    update_bucket_anchor_(bucket_means, bucket_seen, 0, 0, batch_mean, 0.5)

    assert not bucket_means.requires_grad
    assert bucket_means.grad_fn is None
    assert batch_mean.grad is None


def test_all_active_stage2_yaml_files_use_migrated_key_and_preserve_rates():
    migrated = []
    legacy = []
    for path in sorted((ROOT / "config").glob("**/*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
        if "anchor_update_rate" in loss_cfg:
            migrated.append((path, float(loss_cfg["anchor_update_rate"])))
        if "ema_beta" in loss_cfg:
            legacy.append(path)

    assert legacy == []
    assert len(migrated) == 4
    for path, rate in migrated:
        dataset = path.relative_to(ROOT / "config").parts[0]
        assert rate == FORMAL_RATES[dataset]
        assert 1.0 - rate == pytest.approx(1.0 - FORMAL_RATES[dataset])


@pytest.mark.parametrize("dataset,rate", FORMAL_RATES.items())
def test_formal_stage2_effective_coefficients(dataset: str, rate: float):
    cfg = yaml.safe_load((ROOT / "config" / dataset / "stage2.yaml").read_text())
    resolved = resolve_anchor_update_rate(cfg["loss"])
    assert resolved == rate
    assert (1.0 - resolved, resolved) == pytest.approx((1.0 - rate, rate))
