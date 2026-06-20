"""Unit tests for the ADR-031 S4 long-horizon divergence-growth metric.

Synthetic-series tests (pure numpy, no GPU) proving the metric correctly
classifies BOUNDED (non-escalating) vs ESCALATING divergence -- the reduced-
precision equivalence criterion gate. The real fp32/fp64 72 h series come in S4;
these tests pin the discriminator behaviour now.

Metric under test:
  proofs/perf/v015/fp32_oracles/divergence_growth_metric.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_METRIC = _ROOT / "proofs" / "perf" / "v015" / "fp32_oracles" / "divergence_growth_metric.py"
if not _METRIC.is_file():  # shared-checkout fallback
    _METRIC = Path("<USER_HOME>/src/wrf_gpu2") / "proofs" / "perf" / "v015" / "fp32_oracles" / "divergence_growth_metric.py"

_spec = importlib.util.spec_from_file_location("divergence_growth_metric", _METRIC)
dgm = importlib.util.module_from_spec(_spec)
# Register BEFORE exec so the module's @dataclass can resolve its own __module__.
sys.modules["divergence_growth_metric"] = dgm
_spec.loader.exec_module(dgm)


def _leads(n=37, dt_h=2.0):
    return np.arange(n, dtype=np.float64) * dt_h  # 0..72 h


def test_rmse_series_shapes_and_values():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((10, 4, 5))
    b = a.copy()
    d = dgm.rmse_series(a, b)
    assert d.shape == (10,)
    assert np.allclose(d, 0.0)
    b2 = a + 2.0
    d2 = dgm.rmse_series(a, b2)
    assert np.allclose(d2, 2.0)


def test_bounded_saturating_divergence_passes():
    """fp32 vs fp64: divergence saturates to ~envelope -> BOUNDED, PASS."""
    leads = _leads()
    env = 0.05  # oracle internal variability (e.g. fp64-GPU run-to-run)
    # saturating curve: d(t) = env * (1 - exp(-t/tau)), plateaus below 5*env.
    d = env * (1.0 - np.exp(-leads / 18.0))
    res = dgm.classify_divergence_growth(leads, d, envelope=env, field_name="T2")
    assert res.regime == "BOUNDED", res.regime
    assert res.passes
    # late slope must be far below early slope (saturating).
    assert res.late_slope <= 0.25 * abs(res.early_slope) + 1e-12


def test_escalating_linear_divergence_fails():
    """Sustained linear growth far past the envelope -> ESCALATING, FAIL."""
    leads = _leads()
    env = 0.05
    d = 0.02 * leads  # reaches ~1.44 at 72 h = ~29x envelope, constant slope
    res = dgm.classify_divergence_growth(leads, d, envelope=env, field_name="T2")
    assert res.regime == "ESCALATING", res.regime
    assert not res.passes
    assert res.max_over_envelope > 5.0


def test_escalating_superlinear_divergence_fails():
    """Super-linear (accelerating) growth -> ESCALATING, FAIL (late slope >= early)."""
    leads = _leads()
    env = 0.1
    d = 1e-3 * leads ** 2  # accelerating; late slope > early slope
    res = dgm.classify_divergence_growth(leads, d, envelope=env, field_name="W")
    assert res.regime == "ESCALATING", res.regime
    assert not res.passes
    assert res.late_slope_ratio > 1.0  # accelerating: late steeper than early


def test_bounded_noisy_divergence_passes():
    """Divergence wandering within the envelope (no trend) -> BOUNDED, PASS."""
    rng = np.random.default_rng(7)
    leads = _leads()
    env = 0.2
    d = np.abs(0.3 * env + 0.15 * env * rng.standard_normal(leads.size))  # ~within env
    res = dgm.classify_divergence_growth(leads, d, envelope=env, field_name="QVAPOR")
    assert res.passes
    assert res.regime in ("BOUNDED", "BOUNDED_GROWTH")
    assert res.max_over_envelope <= 5.0


def test_slow_growth_within_envelope_is_bounded_growth_not_fail():
    """Small sustained slope but still inside the envelope at the horizon.

    This is the realistic fp32 case: a tiny non-zero drift that stays within the
    oracle's variability -- it must NOT FAIL (regime BOUNDED_GROWTH, passes=True).
    """
    leads = _leads()
    env = 1.0
    d = 0.2 + 0.004 * leads  # reaches ~0.49 at 72 h, well under 5*env=5
    res = dgm.classify_divergence_growth(leads, d, envelope=env, field_name="U10")
    assert res.passes
    assert res.regime in ("BOUNDED", "BOUNDED_GROWTH")


def test_evaluate_paired_forecast_mixed_fields():
    """End-to-end multi-field gate: one escalating field fails the whole gate."""
    leads = _leads(n=31)
    rng = np.random.default_rng(11)
    shape = (leads.size, 6, 6)
    # bounded field: fp32 ~= fp64 with saturating perturbation
    base = rng.standard_normal(shape)
    sat = (0.03 * (1.0 - np.exp(-leads / 15.0)))[:, None, None]
    fp32_T = base + sat * rng.standard_normal(shape)
    # escalating field: linearly growing departure
    baseW = rng.standard_normal(shape)
    esc = (0.05 * leads)[:, None, None]
    fp32_W = baseW + esc * np.ones(shape)

    report = dgm.evaluate_paired_forecast(
        leads,
        fp32_series={"T": fp32_T, "W": fp32_W},
        oracle_series={"T": base, "W": baseW},
        envelopes={"T": 0.05, "W": 0.05},
    )
    assert report["per_field"]["T"]["passes"] is True
    assert report["per_field"]["W"]["passes"] is False
    assert report["per_field"]["W"]["regime"] == "ESCALATING"
    assert report["GATE_PASS"] is False  # one escalating field fails the gate


def test_perfect_match_passes():
    """Identical series -> zero divergence -> trivially BOUNDED, PASS."""
    leads = _leads()
    a = np.random.default_rng(3).standard_normal((leads.size, 5, 5))
    report = dgm.evaluate_paired_forecast(
        leads, {"P": a}, {"P": a.copy()}, {"P": 1.0}
    )
    assert report["GATE_PASS"] is True
    assert report["per_field"]["P"]["regime"] == "BOUNDED"
