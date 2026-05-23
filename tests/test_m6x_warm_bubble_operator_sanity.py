from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m6_warm_bubble_test", ROOT / "scripts" / "m6_warm_bubble_test.py")
assert SPEC is not None
warm_bubble = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(warm_bubble)


MANDATORY_SAMPLE_KEYS = {
    "w_max_m_s",
    "w_min_m_s",
    "w_abs_max_m_s",
    "theta_perturbation_max_K",
    "theta_perturbation_min_K",
    "p_perturbation_max_Pa",
    "p_perturbation_min_Pa",
    "mu_perturbation_max_Pa",
    "centroid_z_m",
    "mu_residual_Pa",
}


@pytest.fixture(scope="module")
def warm_bubble_payload() -> dict:
    return warm_bubble.run_warm_bubble_operator_sanity(run_preconditions=False)


def test_warm_bubble_runs_finite_on_unified_path(warm_bubble_payload: dict):
    assert warm_bubble_payload["first_nonfinite_step"] is None
    assert warm_bubble_payload["surviving_seconds"] == pytest.approx(600.0)
    assert warm_bubble_payload["verdict"] in {
        warm_bubble.PASS_OPERATOR_SANITY,
        warm_bubble.FAIL_PHYSICAL_BOUNDS,
        warm_bubble.FAIL_ANTI_CLAMP_DETECTION,
    }


def test_warm_bubble_extrema_reported_correctly(warm_bubble_payload: dict):
    assert set(warm_bubble_payload) >= {
        "verdict",
        "first_nonfinite_step",
        "samples",
        "bound_violations",
        "anti_clamp_warnings",
    }

    for checkpoint in ("300s", "600s"):
        sample = warm_bubble_payload["samples"][checkpoint]
        assert set(sample) >= MANDATORY_SAMPLE_KEYS
        for key in MANDATORY_SAMPLE_KEYS:
            assert isinstance(sample[key], float), key
            assert math.isfinite(sample[key]), key
        assert sample["w_abs_max_m_s"] >= abs(sample["w_max_m_s"])
        assert sample["w_abs_max_m_s"] >= abs(sample["w_min_m_s"])
        assert sample["theta_perturbation_max_K"] >= sample["theta_perturbation_min_K"]
        assert sample["p_perturbation_max_Pa"] >= sample["p_perturbation_min_Pa"]


def test_anti_clamp_scan_detects_known_patterns_in_test_fixtures():
    source = """
def clamp_to_target(w_next, theta):
    theta_target = 302.0
    theta = jnp.minimum(theta, theta_target)
    w_next = 9.0 * jnp.tanh(jnp.maximum(w_next, 0.0) / 9.0)
    updraft_drag = 0.0005
    return w_next - updraft_drag
"""
    warnings = warm_bubble.scan_anti_clamp_patterns(source_text_by_path={"fixture.py": source})
    hard_rules = {warning["rule"] for warning in warnings if warning["hard_fail"]}

    assert "target_band_tanh_clamp" in hard_rules
    assert "positive_only_w_velocity" in hard_rules
    assert "theta_target_clipping" in hard_rules
    assert "lift_bias_or_updraft_drag" in hard_rules


def test_r7_oracle_is_prerequisite_for_pass():
    payload = {
        "first_nonfinite_step": None,
        "bound_violations": [],
        "anti_clamp_warnings": [],
        "preconditions": {
            "r7_oracle": {"ok": False},
            "hydrostatic_rest": {"ok": True},
        },
    }

    assert warm_bubble._verdict(payload) == warm_bubble.FAIL_FINITENESS
