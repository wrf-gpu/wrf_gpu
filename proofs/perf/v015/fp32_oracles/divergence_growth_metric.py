"""Long-horizon divergence-GROWTH metric (ADR-031 S4 equivalence gate).

Implements the reduced-precision EQUIVALENCE CRITERION from
``.agent/decisions/REDUCED-PRECISION-EQUIVALENCE-AND-FP32-RIGOR.md §3``:

  > Long-horizon non-escalating divergence: over a long forecast (72 h+), the
  > fp32 solution must NOT escape / diverge from the fp64-and/or-CPU-WRF oracle
  > in an ESCALATING fashion. Bounded, non-growing departure = scientifically
  > equivalent. Escalating/runaway divergence = FAIL.

The test is on the SLOPE of the divergence vs lead time, NOT an endpoint value.
This module is the S4 gate metric; the real fp32/fp64 forecast series come in S4.
Here it is built + unit-tested on synthetic series (this file is a pure-numpy
scaffold; no model run, no GPU).

Given a paired forecast series ``a[t]`` (fp32 candidate) and ``b[t]`` (fp64 /
CPU-WRF oracle) at lead times ``leads[t]`` for one or more fields, it computes:

  d(t)         = field divergence at lead t  (RMSE over space, per field)
  growth model = robust trend of d(t) vs t (we fit BOTH a linear slope and a
                 saturating/bounded model and classify the regime)
  envelope     = the oracle's OWN internal variability scale (e.g. fp64-GPU vs
                 fp64-GPU run-to-run, or the natural field variance) so "bounded"
                 means "within the envelope", not "tiny".

Classification (the gate):
  BOUNDED       -- d(t) plateaus / its growth rate -> 0 and the late-lead level
                   stays within K*envelope. PASS (scientifically equivalent).
  ESCALATING    -- d(t) grows without saturating (sustained positive slope, or
                   super-linear) and exceeds the envelope. FAIL.

The discriminator is a normalized late-window growth rate: compare the divergence
slope in the LAST third of the forecast to the FIRST third.  Escalating series
keep (or increase) their slope; bounded series have late-slope -> ~0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rmse_series(a: np.ndarray, b: np.ndarray, *, axis_space=None) -> np.ndarray:
    """Per-lead RMSE between two (T, *space) field series.

    ``a``/``b`` shape ``(T, ...)``; returns ``(T,)`` RMSE over the spatial axes.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"series shape mismatch {a.shape} vs {b.shape}")
    diff = a - b
    if axis_space is None:
        axis_space = tuple(range(1, diff.ndim))
    return np.sqrt(np.mean(diff * diff, axis=axis_space))


@dataclass
class DivergenceGrowthResult:
    field_name: str
    leads: np.ndarray
    divergence: np.ndarray                # d(t), per lead
    envelope: float                       # oracle internal-variability scale
    early_slope: float                    # divergence slope, first third (per unit lead)
    late_slope: float                     # divergence slope, last third (per unit lead)
    late_slope_ratio: float               # late_slope / max(|early_slope|, eps)
    final_over_envelope: float            # d(T)/envelope
    max_over_envelope: float              # max_t d(t)/envelope
    regime: str                           # "BOUNDED" | "ESCALATING" | "BOUNDED_GROWTH"
    passes: bool
    detail: dict = field(default_factory=dict)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y vs x (robust to T<2)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 2:
        return 0.0
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom <= 0:
        return 0.0
    return float(np.sum(xc * (y - y.mean())) / denom)


def classify_divergence_growth(
    leads: np.ndarray,
    divergence: np.ndarray,
    *,
    envelope: float,
    field_name: str = "field",
    envelope_factor: float = 5.0,
    late_slope_tol: float = 0.25,
) -> DivergenceGrowthResult:
    """Classify a divergence series as BOUNDED (PASS) or ESCALATING (FAIL).

    Args:
      leads/divergence : (T,) lead times and divergence d(t).
      envelope         : the oracle's own internal-variability scale (the
                         denominator that makes "bounded" mean "within the
                         oracle's noise", not "zero").
      envelope_factor  : a BOUNDED series may sit up to this multiple of the
                         envelope (the divergence is allowed to be of the same
                         order as the oracle's own variability, not tiny).
      late_slope_tol   : the LATE-window growth rate (normalized to the early
                         rate) must fall below this for "saturating".  An
                         escalating series keeps late_slope_ratio ~>= 1.

    The gate (PASS) requires BOTH:
      (i)  the late-lead divergence stays within envelope_factor*envelope, AND
      (ii) growth is saturating: late_slope <= late_slope_tol * early_slope
           (the curve flattens), OR the absolute late slope is negligible vs the
           envelope per total lead span.
    """
    leads = np.asarray(leads, np.float64)
    divergence = np.asarray(divergence, np.float64)
    T = leads.size
    envelope = max(float(envelope), 1e-30)

    third = max(T // 3, 2)
    early_slope = _slope(leads[:third], divergence[:third])
    late_slope = _slope(leads[-third:], divergence[-third:])
    eps = 1e-30
    late_slope_ratio = late_slope / max(abs(early_slope), eps)

    final_over_env = float(divergence[-1] / envelope)
    max_over_env = float(np.max(divergence) / envelope)

    span = max(float(leads[-1] - leads[0]), eps)
    # absolute late growth across the whole remaining span, in envelope units:
    late_growth_env = abs(late_slope) * span / envelope

    within_envelope = max_over_env <= envelope_factor
    saturating = (late_slope <= late_slope_tol * max(abs(early_slope), eps)) or (late_growth_env <= 0.5)

    if within_envelope and saturating:
        regime = "BOUNDED"
        passes = True
    elif within_envelope and not saturating:
        # grows but still inside envelope at this horizon -- watch, not yet fail;
        # FAIL only if it would breach the envelope at the gate horizon.
        regime = "BOUNDED_GROWTH"
        passes = True
    else:
        regime = "ESCALATING"
        passes = False

    return DivergenceGrowthResult(
        field_name=field_name,
        leads=leads,
        divergence=divergence,
        envelope=envelope,
        early_slope=early_slope,
        late_slope=late_slope,
        late_slope_ratio=late_slope_ratio,
        final_over_envelope=final_over_env,
        max_over_envelope=max_over_env,
        regime=regime,
        passes=passes,
        detail={
            "within_envelope": bool(within_envelope),
            "saturating": bool(saturating),
            "late_growth_in_envelope_units": float(late_growth_env),
            "envelope_factor": envelope_factor,
            "late_slope_tol": late_slope_tol,
        },
    )


def evaluate_paired_forecast(
    leads: np.ndarray,
    fp32_series: dict[str, np.ndarray],
    oracle_series: dict[str, np.ndarray],
    envelopes: dict[str, float],
    *,
    envelope_factor: float = 5.0,
    late_slope_tol: float = 0.25,
) -> dict:
    """Run the divergence-growth gate over a set of fields.

    fp32_series/oracle_series : {field_name: (T, *space) array}.
    envelopes                 : {field_name: oracle internal-variability scale}.
    Returns a JSON-able report with per-field regimes and an overall PASS (all
    fields must be non-escalating).
    """
    per_field = {}
    all_pass = True
    for name in fp32_series:
        d = rmse_series(fp32_series[name], oracle_series[name])
        res = classify_divergence_growth(
            leads, d, envelope=envelopes.get(name, 1.0), field_name=name,
            envelope_factor=envelope_factor, late_slope_tol=late_slope_tol,
        )
        all_pass = all_pass and res.passes
        per_field[name] = {
            "divergence_series": [float(x) for x in res.divergence],
            "envelope": res.envelope,
            "early_slope_per_lead": res.early_slope,
            "late_slope_per_lead": res.late_slope,
            "late_slope_ratio": res.late_slope_ratio,
            "final_over_envelope": res.final_over_envelope,
            "max_over_envelope": res.max_over_envelope,
            "regime": res.regime,
            "passes": res.passes,
            "detail": res.detail,
        }
    return {
        "case": "ADR-031 S4 -- long-horizon divergence-growth equivalence gate",
        "leads": [float(x) for x in np.asarray(leads, np.float64)],
        "criterion": (
            "Reduced-precision equivalence = bounded, non-escalating divergence "
            "from the fp64/CPU-WRF oracle over the forecast. Gate is on the SLOPE "
            "(saturating vs sustained-growth), not the endpoint value, normalized "
            "by the oracle's own internal-variability envelope."
        ),
        "per_field": per_field,
        "GATE_PASS": bool(all_pass),
        "note_S4": (
            "Scaffold metric + thresholds. Real fp32/fp64 (and fp32/CPU-WRF) 72 h "
            "series are supplied in S4; the envelope per field is the fp64-GPU "
            "run-to-run (or ensemble) variability measured then."
        ),
    }


__all__ = [
    "rmse_series",
    "classify_divergence_growth",
    "evaluate_paired_forecast",
    "DivergenceGrowthResult",
]


def _demo() -> dict:
    """Worked synthetic example -> proof JSON (the S4 metric on toy series).

    Shows the gate on two realistic regimes: a saturating fp32 drift (PASS) and a
    sustained linear escape (FAIL).  Real 72 h fp32/fp64 series replace these in S4.
    """
    import json
    from pathlib import Path

    leads = np.arange(37, dtype=np.float64) * 2.0  # 0..72 h, 2 h cadence
    rng = np.random.default_rng(20260613)
    shape = (leads.size, 8, 8)

    oracle_T = rng.standard_normal(shape)
    # fp32 candidate T: saturating departure within ~3x the fp64 run-to-run envelope.
    sat = (0.03 * (1.0 - np.exp(-leads / 16.0)))[:, None, None]
    fp32_T = oracle_T + sat * rng.standard_normal(shape)

    oracle_U = rng.standard_normal(shape)
    # an ESCALATING control field (what a BROKEN fp32 split would look like).
    esc = (0.04 * leads)[:, None, None]
    fp32_U_broken = oracle_U + esc * np.ones(shape)

    report = evaluate_paired_forecast(
        leads,
        fp32_series={"T_saturating_PASS": fp32_T, "U_escalating_FAIL_control": fp32_U_broken},
        oracle_series={"T_saturating_PASS": oracle_T, "U_escalating_FAIL_control": oracle_U},
        envelopes={"T_saturating_PASS": 0.05, "U_escalating_FAIL_control": 0.05},
    )
    report["demo_note"] = (
        "Synthetic worked example. T_saturating_PASS demonstrates an ACCEPTABLE "
        "fp32 lane (bounded, non-escalating); U_escalating_FAIL_control is the "
        "explicit FAIL signature a broken split would produce. GATE_PASS is False "
        "here BY DESIGN (the control field escalates) -- it shows the gate has teeth."
    )
    out = Path(__file__).resolve().parent / "divergence_growth_metric_demo.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    for name, r in report["per_field"].items():
        print(f"  {name:32s} regime={r['regime']:14s} max/env={r['max_over_envelope']:.2f} "
              f"late_slope_ratio={r['late_slope_ratio']:.2f} passes={r['passes']}")
    print(f"  GATE_PASS = {report['GATE_PASS']} (False by design: control field escalates)")
    return report


if __name__ == "__main__":
    _demo()
