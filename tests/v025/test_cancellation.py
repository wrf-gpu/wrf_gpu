"""Regression tests for the M0 cancellation-map primitives (contract §10).

These pin the two things a critic must be able to check without rerunning the
whole map: that the classification thresholds are the frozen ones and behave as
written, and that the harness cannot silently produce a vacuous map.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v025 import cancellation as canc  # noqa: E402


# --------------------------------------------------------------------------- #
# frozen thresholds                                                            #
# --------------------------------------------------------------------------- #
def test_thresholds_are_the_frozen_ones():
    """A tuned threshold is a moved goalpost; pin the exact values."""

    assert canc.FROZEN["frozen_before_measurement"] is True
    assert canc.FROZEN["safe_fp32_max_amplification"] == 16.0
    assert canc.FROZEN["island_gain_threshold"] == 0.50
    assert canc.FROZEN["classes"] == [
        "SAFE_FP32",
        "COMPENSATED_OR_REFORMULATED",
        "FP64_ISLAND_CANDIDATE",
    ]
    assert canc.FROZEN["non_verdict_statuses"] == [
        "NOT_EXERCISED_ON_THIS_STATE",
        "INCONCLUSIVE_DTYPE_NORMALISED",
        "FP32_TRACE_FAILS",
    ]


def test_non_verdict_statuses_are_disjoint_from_verdicts():
    """A non-verdict must never be readable as a verdict."""

    assert not set(canc.FROZEN["classes"]) & set(canc.FROZEN["non_verdict_statuses"])


def test_eps32_is_the_unit_roundoff_not_the_spacing():
    """fp32 unit roundoff is 2**-24, half the spacing at 1.0."""

    assert canc.EPS32 == pytest.approx(2.0**-24)


# --------------------------------------------------------------------------- #
# classification                                                               #
# --------------------------------------------------------------------------- #
def test_inside_the_safe_band_is_safe_regardless_of_island_gain():
    verdict, reason = canc.classify(8.0 * canc.EPS32, 0.0, finite=True)
    assert verdict == "SAFE_FP32"
    assert "unit roundoff" in reason


def test_just_outside_the_safe_band_is_not_safe():
    verdict, _ = canc.classify(16.5 * canc.EPS32, 16.4 * canc.EPS32, finite=True)
    assert verdict != "SAFE_FP32"


def test_arithmetic_dominated_error_is_an_island_candidate():
    """Representation floor far below the total: an in-operator upcast helps."""

    verdict, _ = canc.classify(1e-3, 1e-9, finite=True)
    assert verdict == "FP64_ISLAND_CANDIDATE"


def test_representation_dominated_error_must_be_reformulated():
    """The damage is already in the fp32 inputs; an island cannot remove it."""

    verdict, reason = canc.classify(1e-3, 9.5e-4, finite=True)
    assert verdict == "COMPENSATED_OR_REFORMULATED"
    assert "algebraic" in reason


def test_non_finite_output_can_never_be_classified_safe():
    verdict, _ = canc.classify(1e-12, 1e-12, finite=False)
    assert verdict == "FP64_ISLAND_CANDIDATE"


def test_island_gain_is_bounded_and_monotone():
    assert canc.island_gain(1e-3, 0.0) == pytest.approx(1.0)
    assert canc.island_gain(1e-3, 1e-3) == pytest.approx(0.0)
    assert canc.island_gain(1e-3, 5e-4) == pytest.approx(0.5)
    # A representation error larger than the total cannot mean negative gain.
    assert canc.island_gain(1e-4, 1e-3) == 0.0
    # A zero or non-finite total is not evidence of anything.
    assert canc.island_gain(0.0, 0.0) == 0.0
    assert canc.island_gain(float("nan"), 0.0) == 0.0


# --------------------------------------------------------------------------- #
# error metrics                                                                #
# --------------------------------------------------------------------------- #
def test_identical_arrays_have_zero_error():
    value = np.linspace(-5.0, 5.0, 101)
    metrics = canc.error_metrics(value, value)
    assert metrics["max_rel"] == 0.0
    assert metrics["max_ulp_fp32"] == 0.0
    assert metrics["all_finite"] is True


def test_relative_denominator_floor_stops_near_zero_cells_dominating():
    """A perturbation field crossing zero must not report 1e12 relative error.

    Without the floor a single baseline cell at ~1e-300 turns an otherwise clean
    comparison into noise and every percentile becomes meaningless.
    """

    baseline = np.array([1.0, 1.0, 1e-300])
    candidate = np.array([1.0, 1.0, 2e-300])
    metrics = canc.error_metrics(candidate, baseline)
    assert metrics["relative_denominator_floor"] > 0.0
    assert metrics["max_rel"] < 1.0


def test_ulp_distance_is_measured_in_fp32_ulps():
    one = np.float64(1.0)
    one_ulp_up = np.float64(np.float32(1.0) + np.spacing(np.float32(1.0)))
    metrics = canc.error_metrics(np.array([one_ulp_up]), np.array([one]))
    assert metrics["max_ulp_fp32"] == pytest.approx(1.0, rel=1e-6)


def test_shape_mismatch_is_an_error_not_a_broadcast():
    with pytest.raises(ValueError):
        canc.error_metrics(np.zeros(4), np.zeros(5))


def test_combine_outputs_is_order_stable():
    """Multi-output operators must concatenate deterministically across arms."""

    first = canc.combine_outputs({"b": np.array([2.0]), "a": np.array([1.0])})
    second = canc.combine_outputs({"a": np.array([1.0]), "b": np.array([2.0])})
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# source algebra scan                                                          #
# --------------------------------------------------------------------------- #
def test_scan_finds_total_minus_perturbation():
    source = "x = state.p_total - state.p_perturbation\n"
    scan = canc.scan_algebra(source)
    assert scan.total_minus_perturbation == ("state.p_total - state.p_perturbation",)


def test_scan_counts_island_calls():
    source = "a, b = force_fp64_island(a, b)\nc = force_fp64_island(c)\n"
    scan = canc.scan_algebra(source)
    assert scan.force_fp64_island_calls == 2
    assert scan.has_explicit_island is True


def test_scan_reports_no_island_when_absent():
    scan = canc.scan_algebra("y = a - b\n")
    assert scan.has_explicit_island is False
    assert scan.force_fp64_island_calls == 0


# --------------------------------------------------------------------------- #
# condition probe                                                              #
# --------------------------------------------------------------------------- #
def test_condition_number_of_an_identity_is_about_one():
    baseline = np.linspace(1.0, 2.0, 64)

    def run(epsilon: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return baseline * (1.0 + epsilon * rng.uniform(-1.0, 1.0, baseline.shape))

    result = canc.condition_number(run, baseline, samples=4)
    assert 0.2 < result["kappa_median"] < 2.0
    assert result["samples"] == 4


def test_condition_number_detects_amplification():
    """A cancelling difference must report kappa far above one.

    ``a - b`` with ``a, b ~ 1e6`` and ``a - b ~ 1``: an independent relative
    perturbation of the two operands moves the difference by ``eps * 1e6``,
    a relative response of ``1e6 * eps`` on an output of order 1. That factor
    of ~1e6 IS the cancellation, and it is what the probe must surface. (Note
    this is conditioning of the algebra, not fp64 rounding -- fp64 represents
    1e6 + 1 exactly, so a rounding-based construction would report nothing.)
    """

    a = np.full(64, 1.0e6)
    b = a - 1.0
    baseline = a - b

    def run(epsilon: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        a_p = a * (1.0 + epsilon * rng.uniform(-1.0, 1.0, a.shape))
        b_p = b * (1.0 + epsilon * rng.uniform(-1.0, 1.0, b.shape))
        return a_p - b_p

    result = canc.condition_number(run, baseline, samples=4)
    assert result["kappa_median"] > 1.0e4


def test_condition_number_survives_an_all_zero_baseline():
    baseline = np.zeros(8)
    result = canc.condition_number(lambda e, s: np.zeros(8), baseline, samples=2)
    assert result["samples"] == 0
    assert np.isnan(result["kappa_median"])
