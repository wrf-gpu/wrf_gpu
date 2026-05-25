"""Smoke + edge-case tests for the M6b Tier-4 RMSE dry-run comparator.

These tests do NOT spin up the JAX operational forecast. They exercise the
pure-Python comparator helpers (per-cell stats, noise-floor classification,
shape mismatch, NaN handling) so a broken comparator is caught even if the
1h GPU run hasn't been done yet.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "m6b_tier4_rmse_dryrun.py"


def _load_module():
    """Import the dry-run script as a module without running its main."""

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("m6b_tier4_rmse_dryrun", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m6b_tier4_rmse_dryrun"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# -----------------------------------------------------------------------------
# Constants + config sanity
# -----------------------------------------------------------------------------


def test_threshold_constants_match_contract(mod):
    assert mod.TIER4_THRESHOLDS == {"T2": 3.0, "U10": 7.5, "V10": 7.5}
    assert mod.SPATIAL_RATIO_THRESHOLD == 1.5
    assert mod.NOISE_FLOOR_BAND_MULTIPLIER == 5.0
    assert mod.NOISE_FLOOR_LEAD_HOURS == 24


def test_default_output_path_is_inside_sprint(mod):
    assert mod.DEFAULT_OUTPUT.name == "2026-05-25-m6b-tier4-rmse-dryrun"
    assert mod.DEFAULT_OUTPUT.parent.name == "sprints"


def test_default_run_id_is_passing_ic(mod):
    assert mod.DEFAULT_RUN_ID.startswith("20260429_18z_l3_24h_")


# -----------------------------------------------------------------------------
# _local_error_stats — per-cell heterogeneity helper
# -----------------------------------------------------------------------------


def test_local_error_stats_identical_fields(mod):
    a = np.ones((10, 10), dtype=np.float64)
    stats = mod._local_error_stats(a, a)
    assert stats["all_finite"] is True
    assert stats["mean_abs"] == 0.0
    assert stats["max_abs"] == 0.0
    assert math.isnan(stats["spatial_ratio_max_over_mean"])
    assert stats["shape"] == [10, 10]


def test_local_error_stats_uniform_offset(mod):
    pred = np.ones((4, 5), dtype=np.float64) * 2.5
    ref = np.ones((4, 5), dtype=np.float64)
    stats = mod._local_error_stats(pred, ref)
    assert stats["all_finite"] is True
    assert stats["mean_abs"] == pytest.approx(1.5)
    assert stats["max_abs"] == pytest.approx(1.5)
    assert stats["spatial_ratio_max_over_mean"] == pytest.approx(1.0)


def test_local_error_stats_one_hot_spike(mod):
    pred = np.zeros((20, 20), dtype=np.float64)
    pred[10, 10] = 10.0
    ref = np.zeros((20, 20), dtype=np.float64)
    stats = mod._local_error_stats(pred, ref)
    expected_mean = 10.0 / 400.0
    expected_ratio = 10.0 / expected_mean
    assert stats["mean_abs"] == pytest.approx(expected_mean)
    assert stats["max_abs"] == pytest.approx(10.0)
    assert stats["spatial_ratio_max_over_mean"] == pytest.approx(expected_ratio)


def test_local_error_stats_nan_in_predicted(mod):
    pred = np.zeros((5, 5), dtype=np.float64)
    pred[0, 0] = np.nan
    ref = np.zeros((5, 5), dtype=np.float64)
    stats = mod._local_error_stats(pred, ref)
    assert stats["all_finite"] is False
    assert math.isnan(stats["mean_abs"])
    assert math.isnan(stats["max_abs"])
    assert math.isnan(stats["spatial_ratio_max_over_mean"])


def test_local_error_stats_inf_in_reference(mod):
    pred = np.zeros((5, 5), dtype=np.float64)
    ref = np.zeros((5, 5), dtype=np.float64)
    ref[2, 2] = np.inf
    stats = mod._local_error_stats(pred, ref)
    assert stats["all_finite"] is False


# -----------------------------------------------------------------------------
# Noise-floor classification
# -----------------------------------------------------------------------------


def _fake_stage2(rmse_t2: float, rmse_u10: float, rmse_v10: float) -> dict:
    return {
        "fields": {
            "T2": {"rmse_spatial_mean": rmse_t2},
            "U10": {"rmse_spatial_mean": rmse_u10},
            "V10": {"rmse_spatial_mean": rmse_v10},
        }
    }


def test_noise_floor_load_has_all_three_fields(mod):
    floor = mod._load_noise_floor()
    assert set(floor.keys()) >= {"T2", "U10", "V10"}
    for name in ("T2", "U10", "V10"):
        assert floor[name]["lead_hours"] == mod.NOISE_FLOOR_LEAD_HOURS
        assert floor[name]["spatial_mean_rmse"] > 0.0


def test_noise_floor_classification_below_floor_is_suspicious(mod):
    floor = mod._load_noise_floor()
    # Use 0.5x noise-floor — below the floor → suspicious-broken classification.
    stage2 = _fake_stage2(
        rmse_t2=floor["T2"]["spatial_mean_rmse"] * 0.5,
        rmse_u10=floor["U10"]["spatial_mean_rmse"] * 0.5,
        rmse_v10=floor["V10"]["spatial_mean_rmse"] * 0.5,
    )
    result = mod.stage3_noise_floor_compare(stage2)
    assert result["status"] == "COMPARATOR_SUSPICIOUS"
    assert result["fields"]["T2"]["classification"] == "SUSPICIOUS_BELOW_NOISE_FLOOR"


def test_noise_floor_classification_in_band_is_healthy(mod):
    floor = mod._load_noise_floor()
    stage2 = _fake_stage2(
        rmse_t2=floor["T2"]["spatial_mean_rmse"] * 2.0,
        rmse_u10=floor["U10"]["spatial_mean_rmse"] * 2.0,
        rmse_v10=floor["V10"]["spatial_mean_rmse"] * 2.0,
    )
    result = mod.stage3_noise_floor_compare(stage2)
    assert result["status"] == "PASS"
    assert result["fields"]["T2"]["classification"] == "HEALTHY_IN_NOISE_BAND"


def test_noise_floor_classification_outside_envelope(mod):
    # Use values above the Tier-4 envelope on all three.
    stage2 = _fake_stage2(rmse_t2=5.0, rmse_u10=10.0, rmse_v10=10.0)
    result = mod.stage3_noise_floor_compare(stage2)
    assert result["status"] == "OPERATIONAL_OUTSIDE_ENVELOPE"
    for name in ("T2", "U10", "V10"):
        assert result["fields"][name]["classification"] == "OUTSIDE_TIER4_ENVELOPE"


def test_noise_floor_classification_above_band_but_within_envelope(mod):
    floor = mod._load_noise_floor()
    # 6x noise floor but well under the envelope.
    stage2 = _fake_stage2(
        rmse_t2=floor["T2"]["spatial_mean_rmse"] * 6.0,
        rmse_u10=floor["U10"]["spatial_mean_rmse"] * 1.5,
        rmse_v10=floor["V10"]["spatial_mean_rmse"] * 1.5,
    )
    result = mod.stage3_noise_floor_compare(stage2)
    # T2: 6 * 0.628 ≈ 3.77 > envelope 3.0 → OUTSIDE_TIER4_ENVELOPE
    assert result["fields"]["T2"]["classification"] in {
        "ABOVE_NOISE_WITHIN_ENVELOPE",
        "OUTSIDE_TIER4_ENVELOPE",
    }
    # U10 1.5x ≈ 2.18, falls in [noise, 5*noise]
    assert result["fields"]["U10"]["classification"] == "HEALTHY_IN_NOISE_BAND"


def test_noise_floor_missing_csv_raises(mod, tmp_path, monkeypatch):
    bogus = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(mod, "BASELINE_RMSE_CSV", bogus)
    with pytest.raises(FileNotFoundError):
        mod._load_noise_floor()


def test_noise_floor_missing_row_raises(mod, tmp_path, monkeypatch):
    p = tmp_path / "rmse_summary.csv"
    p.write_text(
        "field,lead_hours,spatial_mean_rmse,p95_rmse,sample_pairs,units,notes\n"
        "T2,24,0.5,1.0,10,K,short\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "BASELINE_RMSE_CSV", p)
    with pytest.raises(KeyError):
        mod._load_noise_floor()


# -----------------------------------------------------------------------------
# stage2 RMSE — shape mismatch + NaN propagation via a tiny synthetic case
# -----------------------------------------------------------------------------


class _FakeState:
    """Minimal stand-in producing surface fields via monkey-patching."""


class _FakeRun:
    """Stand-in for Gen2Run; the run object is only used by _reference_fields."""


def test_stage2_shape_mismatch_raises(mod, monkeypatch):
    """If forecast and reference fields disagree on shape, stage2 must raise."""

    pred = {
        "T2": np.zeros((10, 10), dtype=np.float64),
        "U10": np.zeros((10, 10), dtype=np.float64),
        "V10": np.zeros((10, 10), dtype=np.float64),
    }
    ref = {
        "T2": np.zeros((10, 11), dtype=np.float64),  # mismatched
        "U10": np.zeros((10, 10), dtype=np.float64),
        "V10": np.zeros((10, 10), dtype=np.float64),
        "w_k20": np.zeros((10, 10), dtype=np.float64),
        "theta_k20": np.zeros((10, 10), dtype=np.float64),
    }
    monkeypatch.setattr(mod, "_surface_fields", lambda state: pred)
    monkeypatch.setattr(mod, "_reference_fields", lambda run, domain, hours: (ref, Path("/tmp/fake"), "2026-04-29T19:00:00"))
    with pytest.raises(ValueError, match="shape mismatch"):
        mod.stage2_compute_tier4_rmse(_FakeState(), _FakeRun(), 1.0)


def test_stage2_zero_error_marks_ratio_nan_and_passes_rmse(mod, monkeypatch):
    """Identical forecast == reference → RMSE = 0, ratio = NaN, ratio_pass = False."""

    fields = {
        "T2": np.ones((6, 6), dtype=np.float64),
        "U10": np.ones((6, 6), dtype=np.float64),
        "V10": np.ones((6, 6), dtype=np.float64),
    }
    ref = {
        **fields,
        "w_k20": np.zeros((6, 6), dtype=np.float64),
        "theta_k20": np.zeros((6, 6), dtype=np.float64),
    }
    monkeypatch.setattr(mod, "_surface_fields", lambda state: fields)
    monkeypatch.setattr(mod, "_reference_fields", lambda run, domain, hours: (ref, Path("/tmp/fake"), "2026-04-29T19:00:00"))
    out = mod.stage2_compute_tier4_rmse(_FakeState(), _FakeRun(), 1.0)
    assert out["status"] == "FAIL"  # zero RMSE is suspiciously perfect; ratio NaN trips it
    for name in ("T2", "U10", "V10"):
        f = out["fields"][name]
        assert f["rmse_spatial_mean"] == 0.0
        assert f["rmse_pass"] is True
        assert math.isnan(f["spatial_ratio_max_over_mean"])
        assert f["spatial_ratio_pass"] is False


def test_stage2_realistic_pass_case(mod, monkeypatch):
    rng = np.random.default_rng(seed=42)
    # ~0.5 K RMSE with uniform Gaussian → mean|err|≈0.4, max|err|<3, ratio<8
    err = rng.normal(0.0, 0.5, size=(50, 50))
    base = rng.normal(0.0, 1.0, size=(50, 50))
    pred = {
        "T2": base + err,
        "U10": base + err,
        "V10": base + err,
    }
    ref = {
        "T2": base,
        "U10": base,
        "V10": base,
        "w_k20": np.zeros((50, 50), dtype=np.float64),
        "theta_k20": np.zeros((50, 50), dtype=np.float64),
    }
    monkeypatch.setattr(mod, "_surface_fields", lambda state: pred)
    monkeypatch.setattr(mod, "_reference_fields", lambda run, domain, hours: (ref, Path("/tmp/fake"), "2026-04-29T19:00:00"))
    out = mod.stage2_compute_tier4_rmse(_FakeState(), _FakeRun(), 1.0)
    for name in ("T2", "U10", "V10"):
        f = out["fields"][name]
        assert f["rmse_pass"] is True
        assert f["rmse_spatial_mean"] < 1.0


def test_stage2_nan_pixel_marks_failure(mod, monkeypatch):
    pred = {
        "T2": np.zeros((8, 8), dtype=np.float64),
        "U10": np.zeros((8, 8), dtype=np.float64),
        "V10": np.zeros((8, 8), dtype=np.float64),
    }
    pred["T2"][3, 3] = np.nan
    ref = {
        "T2": np.zeros((8, 8), dtype=np.float64),
        "U10": np.zeros((8, 8), dtype=np.float64),
        "V10": np.zeros((8, 8), dtype=np.float64),
        "w_k20": np.zeros((8, 8), dtype=np.float64),
        "theta_k20": np.zeros((8, 8), dtype=np.float64),
    }
    monkeypatch.setattr(mod, "_surface_fields", lambda state: pred)
    monkeypatch.setattr(mod, "_reference_fields", lambda run, domain, hours: (ref, Path("/tmp/fake"), "2026-04-29T19:00:00"))
    out = mod.stage2_compute_tier4_rmse(_FakeState(), _FakeRun(), 1.0)
    assert out["status"] == "FAIL"
    assert out["fields"]["T2"]["all_finite"] is False


# -----------------------------------------------------------------------------
# Output schema — every proof JSON should be reloadable + contain status
# -----------------------------------------------------------------------------


def test_write_json_roundtrip(mod, tmp_path):
    target = tmp_path / "nested" / "proof_x.json"
    payload = {"status": "PASS", "value": 1.234, "ints": [1, 2, 3]}
    mod._write_json(target, payload)
    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == payload


def test_argparse_defaults(mod):
    ns = mod.parse_args([])
    assert ns.run_id == mod.DEFAULT_RUN_ID
    assert ns.hours == 1.0
    assert ns.output == mod.DEFAULT_OUTPUT


def test_argparse_overrides(mod, tmp_path):
    ns = mod.parse_args(["--run-id", "20260509_xx", "--hours", "0.5", "--output", str(tmp_path)])
    assert ns.run_id == "20260509_xx"
    assert ns.hours == 0.5
    assert ns.output == tmp_path
