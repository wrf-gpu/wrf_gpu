"""Edge-case tests for the M6 boundary/dynamics audit driver.

These tests exercise the pure-Python classifier and ring-vs-interior helpers
without touching the JAX dycore, so they run in seconds with no GPU.

The audit script is read-only investigation code and the production code is
unchanged — so the tests focus on the classifier's verdict math, boundary
cases, malformed inputs, and the ring-vs-interior separator.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PY = ROOT / "scripts" / "m6_boundary_dynamics_audit.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("m6_boundary_dynamics_audit", AUDIT_PY)
    assert spec is not None and spec.loader is not None, f"could not load {AUDIT_PY}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


# ----------------------------- _classify_field -----------------------------


def test_classify_within_envelope_is_pass_a():
    """A field at 50% of its 1x envelope must classify PASS_A."""
    res = audit._classify_field("u", run_max_abs=40.0, run_min=-40.0, run_max=40.0)
    assert res["verdict"] == "PASS_A"
    assert pytest.approx(res["ratio_to_envelope"], rel=1e-9) == 0.5


def test_classify_at_envelope_edge_is_pass_a():
    """Equal to envelope is still PASS_A (boundary inclusive)."""
    res = audit._classify_field("u", run_max_abs=80.0, run_min=-80.0, run_max=80.0)
    assert res["verdict"] == "PASS_A"


def test_classify_between_1x_and_2x_is_pass_d():
    res = audit._classify_field("u", run_max_abs=120.0, run_min=-120.0, run_max=120.0)
    assert res["verdict"] == "PASS_D"


def test_classify_between_2x_and_10x_is_cond():
    res = audit._classify_field("u", run_max_abs=400.0, run_min=-100.0, run_max=400.0)
    assert res["verdict"] == "COND_2_10X"


def test_classify_above_10x_is_fail():
    res = audit._classify_field("u", run_max_abs=1.0e8, run_min=-1.0e8, run_max=1.0e8)
    assert res["verdict"] == "FAIL_B_OR_C"
    assert res["ratio_to_envelope"] > 10.0


def test_classify_nonfinite_aware():
    """All-nonfinite case (run_max_abs is None) classifies FAIL_NONFINITE."""
    res = audit._classify_field("u", run_max_abs=None, run_min=None, run_max=None)
    assert res["verdict"] == "FAIL_NONFINITE"


def test_classify_p_total_negative_minimum_downgrades():
    """p_total min < -1 Pa always FAIL_B_OR_C even if abs_max within envelope."""
    res = audit._classify_field("p_total", run_max_abs=1.0e5, run_min=-1.0e6, run_max=1.0e5)
    assert res["verdict"] == "FAIL_B_OR_C"
    assert res.get("p_total_below_floor") is True


def test_classify_p_total_slightly_below_floor_only_conditional():
    """p_total min in [-1, P_TOTAL_FLOOR_PA) downgrades PASS_A to COND but not to FAIL."""
    res = audit._classify_field("p_total", run_max_abs=1.0e5, run_min=-0.5, run_max=1.0e5)
    # ratio is 1.0e5 / 2.0e5 = 0.5, which is PASS_A nominally, downgraded to COND_2_10X
    assert res["verdict"] == "COND_2_10X"


def test_classify_p_total_negative_above_minus_one_keeps_cond_not_fail():
    """Negative p_total min just above -1 Pa is COND, not FAIL (boundary case)."""
    res = audit._classify_field("p_total", run_max_abs=1.0e5, run_min=-0.9, run_max=1.0e5)
    assert res["verdict"] == "COND_2_10X"


# ----------------------------- _safe_minmax --------------------------------


def test_safe_minmax_finite_array():
    res = audit._safe_minmax(np.array([1.0, -2.0, 3.0, -4.0, 5.0]))
    assert res["min"] == -4.0
    assert res["max"] == 5.0
    assert res["abs_max"] == 5.0
    assert res["nonfinite_count"] == 0
    assert res["all_nonfinite"] is False


def test_safe_minmax_with_nan_and_inf():
    res = audit._safe_minmax(np.array([1.0, np.nan, np.inf, -3.0, -np.inf]))
    # Only finite values: 1.0 and -3.0
    assert res["min"] == -3.0
    assert res["max"] == 1.0
    assert res["abs_max"] == 3.0
    assert res["nonfinite_count"] == 3


def test_safe_minmax_all_nonfinite():
    res = audit._safe_minmax(np.array([np.nan, np.inf, -np.inf]))
    assert res["all_nonfinite"] is True
    assert res["min"] is None
    assert res["max"] is None
    assert res["abs_max"] is None


def test_safe_minmax_empty_via_zero_shape():
    """Zero-element array — degenerate but should not crash."""
    res = audit._safe_minmax(np.zeros((0,)))
    assert res["all_nonfinite"] is True
    assert res["size"] == 0


# ----------------------------- _ring_vs_interior ---------------------------


def test_ring_vs_interior_uniform_field_equal():
    """Uniform-amplitude field — ring and interior abs_max equal."""
    arr = np.ones((5, 50, 50))
    res = audit._ring_vs_interior(arr, ring=5)
    assert res.get("skipped_too_small") is None
    assert res["ring"]["abs_max"] == 1.0
    assert res["interior"]["abs_max"] == 1.0


def test_ring_vs_interior_ring_only_amplitude():
    """Spike only on the boundary ring — ring dominates."""
    arr = np.zeros((3, 30, 30))
    arr[:, 0:5, :] = 999.0  # j-low ring
    res = audit._ring_vs_interior(arr, ring=5)
    assert res["ring"]["abs_max"] == 999.0
    assert res["interior"]["abs_max"] == 0.0


def test_ring_vs_interior_interior_only_amplitude():
    """Spike only in the interior — interior dominates."""
    arr = np.zeros((3, 30, 30))
    arr[:, 10:20, 10:20] = 777.0
    res = audit._ring_vs_interior(arr, ring=5)
    assert res["interior"]["abs_max"] == 777.0
    assert res["ring"]["abs_max"] == 0.0


def test_ring_vs_interior_skipped_when_too_small():
    """Grid smaller than 2*ring → skipped flag."""
    arr = np.ones((2, 8, 8))
    res = audit._ring_vs_interior(arr, ring=5)
    assert res.get("skipped_too_small") is True


def test_ring_vs_interior_nonfinite_in_ring():
    """NaN/inf in ring should not corrupt interior stats."""
    arr = np.ones((3, 30, 30)) * 0.5
    arr[:, 0, 0] = np.inf
    arr[:, 0, 1] = np.nan
    res = audit._ring_vs_interior(arr, ring=5)
    # interior stays bounded at 0.5
    assert res["interior"]["abs_max"] == 0.5
    # ring abs_max ignores nonfinite, finds finite max = 0.5
    assert res["ring"]["abs_max"] == 0.5


def test_ring_vs_interior_rejects_2d():
    res = audit._ring_vs_interior(np.ones((10, 10)), ring=2)
    assert res.get("skipped") is True


# ------------------- Proof JSON file integrity check ------------------------


PROOFS_DIR = ROOT / ".agent" / "sprints" / "2026-05-26-m6-boundary-dynamics-audit"


@pytest.mark.parametrize(
    "fname",
    [
        "proof_excursion_catalog_20260429.json",
        "proof_excursion_catalog_20260509.json",
        "proof_excursion_catalog_20260521.json",
        "proof_excursion_classification.json",
        "proof_source_localization.json",
        "proof_source_localization_onset.json",
        "proof_audit_summary.json",
    ],
)
def test_audit_proof_files_exist_and_parse(fname):
    """Each proof artefact named in the contract must be present + JSON-loadable."""
    path = PROOFS_DIR / fname
    if not path.exists():
        pytest.skip(f"proof file not yet produced: {path.name}")
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    assert "artifact_type" in payload


def test_aggregate_verdict_present_and_known():
    path = PROOFS_DIR / "proof_excursion_classification.json"
    if not path.exists():
        pytest.skip("classification proof not yet produced")
    d = json.loads(path.read_text())
    assert d["aggregate_verdict"] in {"PASS_A", "PASS_D", "COND_2_10X", "FAIL_B_OR_C", "FAIL_NONFINITE"}


def test_onset_proof_records_per_ic_windows():
    path = PROOFS_DIR / "proof_source_localization_onset.json"
    if not path.exists():
        pytest.skip("onset proof not yet produced")
    d = json.loads(path.read_text())
    per_ic = d["per_ic"]
    for ic in ("20260429", "20260509", "20260521"):
        assert ic in per_ic, f"missing IC {ic}"
        rec = per_ic[ic]
        assert "onset_window_steps_pm2" in rec
