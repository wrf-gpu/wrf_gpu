#!/usr/bin/env python3
"""Cancellation-sensitivity primitives for the M0 map (contract §10).

Everything numeric the map reports is computed here, so a critic can rerun the
arithmetic without re-reading the driver.

The central idea, and the reason the map can propose a *minimal* island set
rather than "keep everything for safety":

``force_fp64_island`` upcasts an operator's inputs at the operator boundary.
It therefore can only remove the error made *inside* the operator's arithmetic.
It cannot remove the error already present in the inputs because they were
stored in fp32. Splitting the total fp32 penalty into those two parts is what
makes the island question answerable by measurement:

    E_repr   fp64 arithmetic on fp32-ROUNDED inputs, vs the fp64 baseline.
             The irreducible storage cost. An island cannot touch it.
    E_total  fp32 arithmetic on fp32 inputs, vs the fp64 baseline.
             What an aggressive fp32 rewrite actually pays.
    E_arith  the part an island can remove:  E_total - E_repr  (in quadrature-free
             terms, measured as the ratio below, not by subtracting error fields).

    island_gain = 1 - E_repr / E_total   in [0, 1]

``island_gain -> 1`` means the damage is arithmetic and an fp64 island fixes it.
``island_gain -> 0`` means the damage is already in the stored inputs and an
island buys nothing; the fix must be algebraic (perturbation form, compensated
summation) -- which is exactly WRF's own strategy and the v0.25 M1 thesis.

Thresholds are frozen in ``FROZEN`` BEFORE any measurement, per contract §7's
standard that a gate is a pre-registered number rather than a judgment call.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# fp32 machine epsilon (unit roundoff): 2**-24.
EPS32 = float(np.spacing(np.float32(1.0)) / 2.0)

FROZEN = {
    "frozen_before_measurement": True,
    "eps32_unit_roundoff": EPS32,
    "safe_fp32_max_amplification": 16.0,
    "safe_fp32_note": (
        "p99 relative error of the aggressive-fp32 arm may not exceed 16x the fp32 unit "
        "roundoff (~9.5e-7), i.e. at most 4 significant bits lost relative to ideal fp32 "
        "rounding. An operator inside this band is SAFE_FP32."
    ),
    "island_gain_threshold": 0.50,
    "island_gain_note": (
        "an operator outside the safe band is an FP64_ISLAND_CANDIDATE only if upcasting "
        "its inputs at the operator boundary removes at least half of its measured error "
        "(island_gain >= 0.50). Otherwise the error is already in the fp32-stored inputs, "
        "an island cannot remove it, and the operator is COMPENSATED_OR_REFORMULATED."
    ),
    "conservation_relative_tolerance": 1e-9,
    "condition_probe_seed": 20260727,
    "condition_probe_samples": 4,
    "classes": ["SAFE_FP32", "COMPENSATED_OR_REFORMULATED", "FP64_ISLAND_CANDIDATE"],
    "non_verdict_statuses": [
        "NOT_EXERCISED_ON_THIS_STATE",
        "INCONCLUSIVE_DTYPE_NORMALISED",
        "FP32_TRACE_FAILS",
    ],
    "non_verdict_note": (
        "Three conditions make a zero measured error meaningless, and each is reported as its "
        "own status rather than as SAFE_FP32. (a) The fp64 baseline output is all zero -- the "
        "operator did not fire on this state (night-time shortwave, an untriggered limiter), so "
        "nothing was measured. (b) The operator re-normalises its inputs to fp64 internally, so "
        "the fp32 arm never ran in fp32 and a zero delta measures the adapter boundary, not the "
        "scheme. (c) The operator fails to TRACE at all under fp32 inputs, which is a harder "
        "result than a large error and must not be hidden behind one."
    ),
}


# --------------------------------------------------------------------------- #
# error metrics                                                                #
# --------------------------------------------------------------------------- #
def _flat64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def field_scale(value: Any) -> dict[str, float]:
    """Magnitude summary of one array, used to judge 'is this scale realistic'."""

    flat = _flat64(value)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "rms": math.nan, "median_abs": math.nan}
    absolute = np.abs(finite)
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "median_abs": float(np.median(absolute)),
    }


def error_metrics(got: Any, want: Any) -> dict[str, Any]:
    """Relative/absolute/ULP error of ``got`` against the fp64 baseline ``want``.

    The relative denominator is the fp64 baseline's own magnitude with a floor at
    the field's RMS times the fp32 unit roundoff. Without that floor a cell whose
    baseline value is ~0 (a perturbation field crossing zero, which is most of
    them) reports a relative error of 1e+12 and swamps every percentile with
    noise that has no physical meaning. The floor is reported so it can be
    audited rather than trusted.
    """

    a = _flat64(got)
    b = _flat64(want)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch in error_metrics: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a - b)
    rms = float(np.sqrt(np.mean(b[finite] ** 2))) if finite.any() else 0.0
    floor = max(rms * EPS32, np.finfo(np.float64).tiny)
    denom = np.maximum(np.abs(b), floor)
    rel = np.where(finite, diff / denom, np.nan)
    # ULP distance measured in fp32 ULPs at the baseline magnitude.
    spacing = np.spacing(np.abs(b).astype(np.float32)).astype(np.float64)
    spacing = np.maximum(spacing, np.finfo(np.float32).tiny)
    ulps = np.where(finite, diff / spacing, np.nan)
    valid = np.isfinite(rel)
    if not valid.any():
        return {
            "max_abs": float("nan"),
            "max_rel": float("nan"),
            "median_rel": float("nan"),
            "p99_rel": float("nan"),
            "rmse": float("nan"),
            "max_ulp_fp32": float("nan"),
            "p99_ulp_fp32": float("nan"),
            "all_finite": False,
            "relative_denominator_floor": floor,
            "baseline_rms": rms,
        }
    return {
        "max_abs": float(np.nanmax(diff[finite])) if finite.any() else float("nan"),
        "max_rel": float(np.nanmax(rel[valid])),
        "median_rel": float(np.nanmedian(rel[valid])),
        "p99_rel": float(np.nanpercentile(rel[valid], 99)),
        "rmse": float(np.sqrt(np.nanmean(diff[finite] ** 2))) if finite.any() else float("nan"),
        "max_ulp_fp32": float(np.nanmax(ulps[valid])),
        "p99_ulp_fp32": float(np.nanpercentile(ulps[valid], 99)),
        "all_finite": bool(np.isfinite(a).all()),
        "relative_denominator_floor": floor,
        "baseline_rms": rms,
    }


def combine_outputs(outputs: dict[str, Any]) -> np.ndarray:
    """Concatenate a multi-output operator's arrays into one comparable vector."""

    return np.concatenate([_flat64(value) for _, value in sorted(outputs.items())])


# --------------------------------------------------------------------------- #
# classification                                                               #
# --------------------------------------------------------------------------- #
def island_gain(p99_total: float, p99_repr: float) -> float:
    """Fraction of the fp32 penalty an in-operator fp64 upcast can remove."""

    if not math.isfinite(p99_total) or p99_total <= 0.0:
        return 0.0
    if not math.isfinite(p99_repr):
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (p99_repr / p99_total))))


def classify(p99_total: float, p99_repr: float, *, finite: bool) -> tuple[str, str]:
    """Frozen §10 classification. Returns (class, one-line reason)."""

    if not finite:
        return (
            "FP64_ISLAND_CANDIDATE",
            "aggressive fp32 produced non-finite output; fp32 is not admissible here",
        )
    if not math.isfinite(p99_total):
        return (
            "FP64_ISLAND_CANDIDATE",
            "aggressive-fp32 error is not finite; treated as failing the safe band",
        )
    amplification = p99_total / EPS32
    if amplification <= FROZEN["safe_fp32_max_amplification"]:
        return (
            "SAFE_FP32",
            f"p99 relative error {p99_total:.3e} = {amplification:.1f}x fp32 unit roundoff, "
            f"inside the frozen {FROZEN['safe_fp32_max_amplification']:.0f}x band",
        )
    gain = island_gain(p99_total, p99_repr)
    if gain >= FROZEN["island_gain_threshold"]:
        return (
            "FP64_ISLAND_CANDIDATE",
            f"amplification {amplification:.1f}x and island_gain {gain:.2f} >= "
            f"{FROZEN['island_gain_threshold']:.2f}: upcasting inputs in-operator removes "
            "most of the error",
        )
    return (
        "COMPENSATED_OR_REFORMULATED",
        f"amplification {amplification:.1f}x but island_gain {gain:.2f} < "
        f"{FROZEN['island_gain_threshold']:.2f}: the error is already in the fp32-stored "
        "inputs, so an fp64 island cannot remove it; the fix is algebraic",
    )


# --------------------------------------------------------------------------- #
# static source algebra scan                                                   #
# --------------------------------------------------------------------------- #
_TOTAL_TOKENS = ("total", "_tot", "mut", "muts", "phb", "_pb", "pb", "alt", "base", "mub")
_PERT_TOKENS = ("perturbation", "_pert", "pert", "_1", "_save", "_work", "prime")

_SUBTRACTION = re.compile(r"([A-Za-z_][A-Za-z0-9_.\[\]]*)\s*-\s*([A-Za-z_][A-Za-z0-9_.\[\]]*)")


@dataclass(frozen=True)
class AlgebraScan:
    total_minus_perturbation: tuple[str, ...]
    subtraction_count: int
    force_fp64_island_calls: int
    has_explicit_island: bool


def scan_algebra(source: str) -> AlgebraScan:
    """Find ``total - perturbation`` style algebra and explicit island calls.

    This is the static half of §10's "source algebra identity and any
    ``total - perturbation`` occurrence" requirement. It is deliberately a
    lexical scan: it over-reports rather than under-reports, and every hit is
    printed into the map so a reviewer can confirm or dismiss it.
    """

    hits: list[str] = []
    for match in _SUBTRACTION.finditer(source):
        left, right = match.group(1).lower(), match.group(2).lower()
        left_total = any(token in left for token in _TOTAL_TOKENS)
        right_pert = any(token in right for token in _PERT_TOKENS)
        if left_total and right_pert:
            hits.append(match.group(0).strip())
    island_calls = source.count("force_fp64_island(")
    return AlgebraScan(
        total_minus_perturbation=tuple(sorted(set(hits))),
        subtraction_count=len(_SUBTRACTION.findall(source)),
        force_fp64_island_calls=island_calls,
        has_explicit_island=island_calls > 0,
    )


# --------------------------------------------------------------------------- #
# condition probe                                                              #
# --------------------------------------------------------------------------- #
def condition_number(
    run: Callable[[float, int], np.ndarray],
    baseline: np.ndarray,
    *,
    relative_perturbation: float = EPS32,
    samples: int = 4,
) -> dict[str, float]:
    """Measured input->output amplification factor.

    ``run(relative_perturbation, seed)`` must apply an independent relative
    perturbation of that size to every float input and return the flattened
    output. The amplification reported is

        kappa = (||dout|| / ||out||) / (||din|| / ||in||)

    with ``||din||/||in||`` fixed at the requested relative perturbation by
    construction, so kappa reduces to the measured relative output response
    divided by the input perturbation. Values near 1 mean a well-conditioned
    operator; values >> 1 localise cancellation.
    """

    base_norm = float(np.sqrt(np.nanmean(baseline[np.isfinite(baseline)] ** 2)))
    responses: list[float] = []
    for index in range(samples):
        perturbed = run(relative_perturbation, FROZEN["condition_probe_seed"] + index)
        finite = np.isfinite(perturbed) & np.isfinite(baseline)
        if not finite.any() or base_norm == 0.0:
            continue
        delta = float(np.sqrt(np.mean((perturbed[finite] - baseline[finite]) ** 2)))
        responses.append(delta / base_norm / relative_perturbation)
    if not responses:
        return {"kappa_median": float("nan"), "kappa_max": float("nan"), "samples": 0}
    return {
        "kappa_median": float(np.median(responses)),
        "kappa_max": float(np.max(responses)),
        "samples": len(responses),
    }
