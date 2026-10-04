"""CPU-only fp32 perturbation-authoritative column prototype.

This module intentionally avoids JAX/CUDA. It implements the same idealized
RK/acoustic update in three forms:

1. fp64 reference;
2. naive fp32-total storage/differencing;
3. perturbation-authoritative fp32 storage with fp64 base gradients, local fp64
   cancellation brackets, fp64 Thomas solve, and compensated accumulation.

The columns are synthetic analytic stress cases, not WRF fixtures. They target
the v0.17/v0.20 cancellation mechanism directly: small pressure/geopotential/mass
differences riding on O(1e5) totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

try:
    from .acceptance_bands import check_metric_band
except ImportError:  # pragma: no cover - direct script/test execution fallback
    from acceptance_bands import check_metric_band


Mode = Literal["reference", "naive_totals", "perturbation"]


@dataclass(frozen=True)
class IdealizedColumnCase:
    name: str
    terrain_m: float
    dt: float
    diffusion: float
    damping: float
    p_base_left: np.ndarray
    p_base_right: np.ndarray
    p_base_center: np.ndarray
    p_delta_pert: np.ndarray
    ph_base_left: np.ndarray
    ph_base_right: np.ndarray
    ph_base_center: np.ndarray
    ph_delta_pert: np.ndarray
    mu_base_left: np.ndarray
    mu_base_right: np.ndarray
    mu_base_center: np.ndarray
    mu_delta_pert: np.ndarray
    p0: np.ndarray
    ph0: np.ndarray
    mu0: np.ndarray
    w0: np.ndarray
    accum_increment: float


@dataclass
class ProtoState:
    p_prime: np.ndarray
    ph_prime: np.ndarray
    mu_prime: np.ndarray
    w: np.ndarray
    accumulator: np.float64 | np.float32
    accumulator_comp: np.float64 | np.float32

    def copy(self) -> "ProtoState":
        return ProtoState(
            p_prime=self.p_prime.copy(),
            ph_prime=self.ph_prime.copy(),
            mu_prime=self.mu_prime.copy(),
            w=self.w.copy(),
            accumulator=self.accumulator,
            accumulator_comp=self.accumulator_comp,
        )


CASE_NAMES = ("sea", "lee_wave", "steep_ridge", "high_peak")


def build_case(name: str, nlev: int = 48) -> IdealizedColumnCase:
    """Build one deterministic cancellation-prone analytic column case."""

    params = {
        "sea": {
            "terrain_m": 0.0,
            "p_amp": 540.0,
            "ph_amp": 260.0,
            "mu_amp": 26.0,
            "delta": 0.026,
            "phase": 0.1,
        },
        "lee_wave": {
            "terrain_m": 720.0,
            "p_amp": 820.0,
            "ph_amp": 390.0,
            "mu_amp": 34.0,
            "delta": 0.034,
            "phase": 0.9,
        },
        "steep_ridge": {
            "terrain_m": 1650.0,
            "p_amp": 1180.0,
            "ph_amp": 560.0,
            "mu_amp": 48.0,
            "delta": 0.043,
            "phase": 1.7,
        },
        "high_peak": {
            "terrain_m": 3100.0,
            "p_amp": 1540.0,
            "ph_amp": 760.0,
            "mu_amp": 62.0,
            "delta": 0.055,
            "phase": 2.4,
        },
    }
    if name not in params:
        raise KeyError(f"unknown case {name!r}; expected one of {CASE_NAMES}")

    cfg = params[name]
    eta = np.linspace(0.0, 1.0, nlev, dtype=np.float64)
    phase = float(cfg["phase"])
    terrain_m = float(cfg["terrain_m"])
    surface_p = 101_325.0 * np.exp(-terrain_m / 8400.0)
    p_base_center = surface_p * np.exp(-1.15 * eta)

    height = terrain_m + 14_000.0 * eta**1.18
    ph_base_center = 9.80665 * height + 2.5e4

    mu_surface = 92_000.0 * np.exp(-terrain_m / 12_000.0)
    mu_base_center = mu_surface * (1.0 - 0.16 * eta + 0.015 * np.cos(np.pi * eta))

    wave1 = np.sin(2.0 * np.pi * eta + phase)
    wave2 = np.cos(5.0 * np.pi * eta - 0.5 * phase)
    p0 = float(cfg["p_amp"]) * (0.58 * wave1 + 0.23 * wave2)
    ph0 = float(cfg["ph_amp"]) * (0.45 * np.cos(1.7 * np.pi * eta + phase) + 0.18 * wave1)
    mu0 = float(cfg["mu_amp"]) * (0.36 * wave2 + 0.12 * wave1)
    w0 = 0.03 * np.sin(np.pi * eta + 0.7 * phase)

    delta = float(cfg["delta"])
    shape = 1.0 + 0.28 * np.sin(3.0 * np.pi * eta + phase)
    p_base_delta = 0.44 * delta * shape
    p_delta_pert = 0.56 * delta * (1.0 + 0.20 * np.cos(2.0 * np.pi * eta - phase))
    ph_base_delta = 0.50 * delta * (1.0 + 0.25 * np.cos(2.5 * np.pi * eta + phase))
    ph_delta_pert = 0.42 * delta * (1.0 + 0.18 * np.sin(4.0 * np.pi * eta))
    mu_base_delta = 0.62 * delta * (1.0 + 0.14 * np.sin(2.0 * np.pi * eta + 0.3))
    mu_delta_pert = 0.38 * delta * (1.0 + 0.11 * np.cos(3.0 * np.pi * eta - phase))

    def split_base(center: np.ndarray, base_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return center - 0.5 * base_delta, center + 0.5 * base_delta

    p_base_left, p_base_right = split_base(p_base_center, p_base_delta)
    ph_base_left, ph_base_right = split_base(ph_base_center, ph_base_delta)
    mu_base_left, mu_base_right = split_base(mu_base_center, mu_base_delta)

    return IdealizedColumnCase(
        name=name,
        terrain_m=terrain_m,
        dt=0.20,
        diffusion=0.022 + 0.002 * min(terrain_m / 1000.0, 4.0),
        damping=0.042,
        p_base_left=p_base_left,
        p_base_right=p_base_right,
        p_base_center=p_base_center,
        p_delta_pert=p_delta_pert,
        ph_base_left=ph_base_left,
        ph_base_right=ph_base_right,
        ph_base_center=ph_base_center,
        ph_delta_pert=ph_delta_pert,
        mu_base_left=mu_base_left,
        mu_base_right=mu_base_right,
        mu_base_center=mu_base_center,
        mu_delta_pert=mu_delta_pert,
        p0=p0,
        ph0=ph0,
        mu0=mu0,
        w0=w0,
        accum_increment=0.004 + 0.0002 * min(terrain_m / 1000.0, 4.0),
    )


def initial_state(case: IdealizedColumnCase, mode: Mode) -> ProtoState:
    dtype = np.float64 if mode == "reference" else np.float32
    return ProtoState(
        p_prime=case.p0.astype(dtype),
        ph_prime=case.ph0.astype(dtype),
        mu_prime=case.mu0.astype(dtype),
        w=case.w0.astype(dtype),
        accumulator=dtype.type(100_000.0) if hasattr(dtype, "type") else dtype(100_000.0),
        accumulator_comp=dtype.type(0.0) if hasattr(dtype, "type") else dtype(0.0),
    )


def _vertical_gradient(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    out = np.empty_like(arr)
    out[1:-1] = 0.5 * (arr[2:] - arr[:-2])
    out[0] = arr[1] - arr[0]
    out[-1] = arr[-1] - arr[-2]
    return out


def _thomas_solve(rhs: np.ndarray, diffusion: float, dtype: np.dtype[Any]) -> np.ndarray:
    """Solve a fixed tridiagonal vertical implicit smoothing system."""

    d = np.asarray(rhs, dtype=dtype).copy()
    n = d.size
    a = np.full(n - 1, -diffusion, dtype=dtype)
    b = np.full(n, 1.0 + 2.0 * diffusion, dtype=dtype)
    c = np.full(n - 1, -diffusion, dtype=dtype)
    b[0] = dtype.type(1.0 + diffusion)
    b[-1] = dtype.type(1.0 + diffusion)

    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    x = np.empty_like(d)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def _face_delta(
    left_base: np.ndarray,
    right_base: np.ndarray,
    center_pert: np.ndarray,
    perturb_delta: np.ndarray,
    mode: Mode,
) -> np.ndarray:
    """Return right-left face signal under a precision/storage form."""

    left_pert = np.asarray(center_pert, dtype=np.float64) - 0.5 * perturb_delta
    right_pert = np.asarray(center_pert, dtype=np.float64) + 0.5 * perturb_delta

    if mode == "reference":
        return (right_base + right_pert) - (left_base + left_pert)

    if mode == "naive_totals":
        left_total = np.asarray(left_base + left_pert, dtype=np.float32)
        right_total = np.asarray(right_base + right_pert, dtype=np.float32)
        return (right_total - left_total).astype(np.float64)

    if mode == "perturbation":
        base_delta = (right_base - left_base).astype(np.float64)
        pert_delta = (
            np.asarray(right_pert, dtype=np.float32)
            - np.asarray(left_pert, dtype=np.float32)
        ).astype(np.float64)
        # This is the local fp64 cancellation bracket: base64 + perturb32 delta.
        return base_delta + pert_delta

    raise ValueError(f"unknown mode {mode!r}")


def gradient_signals(
    case: IdealizedColumnCase,
    state: ProtoState,
    mode: Mode,
) -> dict[str, np.ndarray]:
    """Compute pressure/geopotential/mass face signals for a state and mode."""

    return {
        "p": _face_delta(
            case.p_base_left,
            case.p_base_right,
            state.p_prime,
            case.p_delta_pert,
            mode,
        ),
        "ph": _face_delta(
            case.ph_base_left,
            case.ph_base_right,
            state.ph_prime,
            case.ph_delta_pert,
            mode,
        ),
        "mu": _face_delta(
            case.mu_base_left,
            case.mu_base_right,
            state.mu_prime,
            case.mu_delta_pert,
            mode,
        ),
    }


def step(case: IdealizedColumnCase, state: ProtoState, mode: Mode) -> ProtoState:
    """Advance one idealized acoustic/RK substep."""

    signals = gradient_signals(case, state, mode)
    solve_dtype = np.float64 if mode in ("reference", "perturbation") else np.float32
    work_dtype = np.float64 if mode == "reference" else np.float32

    p = np.asarray(state.p_prime, dtype=solve_dtype)
    ph = np.asarray(state.ph_prime, dtype=solve_dtype)
    mu = np.asarray(state.mu_prime, dtype=solve_dtype)
    w = np.asarray(state.w, dtype=solve_dtype)

    rhs = (
        w
        + case.dt
        * (
            -3.20 * signals["p"]
            + 0.018 * signals["ph"]
            - 0.006 * signals["mu"]
            - case.damping * w
        )
    )
    rhs = rhs + case.dt * 0.0007 * _vertical_gradient(ph)
    w_new = _thomas_solve(rhs, case.diffusion, np.dtype(solve_dtype))
    div_w = _vertical_gradient(w_new)

    p_new = p + case.dt * (-0.12 * div_w - 0.018 * signals["p"] - 0.0008 * p)
    ph_new = ph + case.dt * (0.45 * w_new - 0.004 * signals["ph"] - 0.0006 * ph)
    mu_new = mu + case.dt * (
        -0.018 * signals["mu"] - 0.010 * float(np.mean(signals["p"])) - 0.0012 * mu
    )

    inc64 = case.accum_increment + 0.00005 * float(np.mean(np.abs(w_new)))
    if mode == "reference":
        acc = np.float64(state.accumulator) + np.float64(inc64)
        comp = np.float64(0.0)
    elif mode == "perturbation":
        acc, comp = _kahan_add_fp32(state.accumulator, state.accumulator_comp, inc64)
    else:
        acc = np.float32(state.accumulator) + np.float32(inc64)
        comp = np.float32(0.0)

    return ProtoState(
        p_prime=np.asarray(p_new, dtype=work_dtype),
        ph_prime=np.asarray(ph_new, dtype=work_dtype),
        mu_prime=np.asarray(mu_new, dtype=work_dtype),
        w=np.asarray(w_new, dtype=work_dtype),
        accumulator=acc,
        accumulator_comp=comp,
    )


def _kahan_add_fp32(
    current: np.float32 | np.float64,
    compensation: np.float32 | np.float64,
    increment: float,
) -> tuple[np.float32, np.float32]:
    s = np.float32(current)
    c = np.float32(compensation)
    y = np.float32(increment) - c
    t = np.float32(s + y)
    c_new = np.float32((t - s) - y)
    return t, c_new


def run_sequence(
    case: IdealizedColumnCase,
    nsteps: int = 20_000,
    sample_every: int = 200,
) -> dict[str, Any]:
    """Run reference, naive-total, and perturbation forms for many substeps."""

    states = {
        "reference": initial_state(case, "reference"),
        "naive_totals": initial_state(case, "naive_totals"),
        "perturbation": initial_state(case, "perturbation"),
    }
    max_abs: dict[str, dict[str, float]] = {
        mode: {"p": 0.0, "ph": 0.0, "mu": 0.0, "w": 0.0} for mode in states
    }
    samples: dict[str, list[float]] = {"naive_w_rmse": [], "perturbation_w_rmse": []}

    initial_gradient = cancellation_metrics(case)

    for step_idx in range(1, nsteps + 1):
        for mode in ("reference", "naive_totals", "perturbation"):
            states[mode] = step(case, states[mode], mode)  # type: ignore[arg-type]
            max_abs[mode]["p"] = max(max_abs[mode]["p"], float(np.max(np.abs(states[mode].p_prime))))
            max_abs[mode]["ph"] = max(max_abs[mode]["ph"], float(np.max(np.abs(states[mode].ph_prime))))
            max_abs[mode]["mu"] = max(max_abs[mode]["mu"], float(np.max(np.abs(states[mode].mu_prime))))
            max_abs[mode]["w"] = max(max_abs[mode]["w"], float(np.max(np.abs(states[mode].w))))

        if step_idx % sample_every == 0 or step_idx == nsteps:
            ref = states["reference"]
            samples["naive_w_rmse"].append(_rmse(states["naive_totals"].w, ref.w))
            samples["perturbation_w_rmse"].append(_rmse(states["perturbation"].w, ref.w))

    reference = states["reference"]
    mode_results = {
        "naive_totals": _mode_summary(case, states["naive_totals"], reference, lead_hours=24.0),
        "perturbation": _mode_summary(case, states["perturbation"], reference, lead_hours=24.0),
    }

    return {
        "case": case.name,
        "terrain_m": case.terrain_m,
        "nsteps": nsteps,
        "initial_cancellation": initial_gradient,
        "modes": mode_results,
        "stability": {
            "all_finite": {
                mode: _state_all_finite(state) for mode, state in states.items()
            },
            "max_abs": max_abs,
            "sampled_w_rmse": samples,
        },
    }


def cancellation_metrics(case: IdealizedColumnCase) -> dict[str, dict[str, float]]:
    """Measure initial total-field cancellation error against fp64 reference."""

    ref = initial_state(case, "reference")
    naive = initial_state(case, "naive_totals")
    perturb = initial_state(case, "perturbation")
    ref_sig = gradient_signals(case, ref, "reference")
    naive_sig = gradient_signals(case, naive, "naive_totals")
    pert_sig = gradient_signals(case, perturb, "perturbation")

    out: dict[str, dict[str, float]] = {}
    for field in ("p", "ph", "mu"):
        naive_rmse = _rmse(naive_sig[field], ref_sig[field])
        pert_rmse = _rmse(pert_sig[field], ref_sig[field])
        out[field] = {
            "naive_rmse_vs_fp64": naive_rmse,
            "perturbation_rmse_vs_fp64": pert_rmse,
            "naive_over_perturbation": naive_rmse / max(pert_rmse, 1.0e-30),
            "reference_signal_rmse": _rmse(ref_sig[field], np.zeros_like(ref_sig[field])),
        }
    return out


def run_all_cases(nsteps: int = 20_000) -> dict[str, Any]:
    """Run all four idealized columns and the accumulator drift experiment."""

    case_results = [run_sequence(build_case(name), nsteps=nsteps) for name in CASE_NAMES]
    return {
        "schema": "v020_fp32_cpu_proto_results_v1",
        "cpu_only": True,
        "cases": case_results,
        "accumulator_drift": accumulator_drift_experiment(),
        "summary": summarize_results(case_results),
    }


def summarize_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = []
    perturb_passes = []
    naive_passes = []
    naive_w_failures = 0
    naive_cancellation_case_failures = 0
    for result in case_results:
        case_ratios = []
        for field in ("p", "ph", "mu"):
            ratio = result["initial_cancellation"][field]["naive_over_perturbation"]
            ratios.append(ratio)
            case_ratios.append(ratio)
        perturb_passes.append(result["modes"]["perturbation"]["all_band_checks_pass"])
        naive_passes.append(result["modes"]["naive_totals"]["all_band_checks_pass"])
        if not result["modes"]["naive_totals"]["band_checks"]["W"]["passes"]:
            naive_w_failures += 1
        if max(case_ratios) > 27.0:
            naive_cancellation_case_failures += 1

    return {
        "local_cancellation_gate_ratio_threshold": 27.0,
        "min_naive_over_perturbation": float(min(ratios)),
        "median_naive_over_perturbation": float(np.median(np.asarray(ratios))),
        "max_naive_over_perturbation": float(max(ratios)),
        "naive_cases_failing_local_cancellation_gate": naive_cancellation_case_failures,
        "perturbation_cases_passing_bands": int(sum(bool(x) for x in perturb_passes)),
        "naive_cases_passing_bands": int(sum(bool(x) for x in naive_passes)),
        "naive_w_band_failures": naive_w_failures,
        "case_count": len(case_results),
    }


def accumulator_drift_experiment(nsteps: int = 120_000) -> dict[str, float]:
    """Compare naive fp32 accumulation with Kahan-compensated fp32."""

    truth = np.float64(100_000.0)
    naive = np.float32(100_000.0)
    kahan = np.float32(100_000.0)
    comp = np.float32(0.0)

    for i in range(nsteps):
        inc = np.float64(0.0042 + 0.0008 * np.sin(0.017 * i) + 0.0003 * np.cos(0.071 * i))
        truth += inc
        naive = np.float32(naive + np.float32(inc))
        kahan, comp = _kahan_add_fp32(kahan, comp, float(inc))

    kahan_value = np.float64(kahan) - np.float64(comp)
    naive_err = abs(float(np.float64(naive) - truth))
    kahan_err = abs(float(kahan_value - truth))
    return {
        "nsteps": float(nsteps),
        "truth_final": float(truth),
        "naive_fp32_final": float(naive),
        "kahan_fp32_final": float(kahan_value),
        "naive_fp32_abs_err": naive_err,
        "kahan_fp32_abs_err": kahan_err,
        "naive_over_kahan_err": naive_err / max(kahan_err, 1.0e-30),
    }


def _mode_summary(
    case: IdealizedColumnCase,
    candidate: ProtoState,
    reference: ProtoState,
    lead_hours: float,
) -> dict[str, Any]:
    metrics = {
        "W": _wind_metrics(candidate.w, reference.w),
        "P": {
            "psfc_rmse_increase": _rmse(candidate.p_prime, reference.p_prime),
            "psfc_bias_abs": abs(_bias(candidate.p_prime, reference.p_prime)),
        },
        "PH": {
            "psfc_rmse_increase": _rmse(candidate.ph_prime, reference.ph_prime),
            "psfc_bias_abs": abs(_bias(candidate.ph_prime, reference.ph_prime)),
            "geopotential_boundary_ring_growth": 0.0,
        },
        "MU": {
            "dry_mass_relative_drift": _relative_sum_drift(
                candidate.mu_prime + case.mu_base_center,
                reference.mu_prime + case.mu_base_center,
            )
        },
    }
    checks = {
        field: check_metric_band(field, lead_hours, field_metrics)
        for field, field_metrics in metrics.items()
    }
    serial_checks = {
        field: {
            "passes": check.passes,
            "group": check.group,
            "tier": check.tier,
            "checked": check.checked,
            "failures": list(check.failures),
        }
        for field, check in checks.items()
    }
    return {
        "metrics": metrics,
        "band_checks": serial_checks,
        "all_band_checks_pass": all(check.passes for check in checks.values()),
        "all_finite": _state_all_finite(candidate),
        "accumulator": float(np.float64(candidate.accumulator) - np.float64(candidate.accumulator_comp)),
        "accumulator_abs_err_vs_fp64": abs(
            float(
                (np.float64(candidate.accumulator) - np.float64(candidate.accumulator_comp))
                - np.float64(reference.accumulator)
            )
        ),
    }


def _wind_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "rmse_increase": _rmse(candidate, reference),
        "bias_abs": abs(float(np.mean(diff))),
        "p95_abs_diff": float(np.percentile(np.abs(diff), 95)),
        "domain_mean_speed_drift": abs(float(np.mean(np.abs(candidate)) - np.mean(np.abs(reference)))),
    }


def _rmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _bias(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.mean(diff))


def _relative_sum_drift(candidate: np.ndarray, reference: np.ndarray) -> float:
    numerator = abs(float(np.sum(np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64))))
    denominator = max(abs(float(np.sum(np.asarray(reference, dtype=np.float64)))), 1.0)
    return numerator / denominator


def _state_all_finite(state: ProtoState) -> bool:
    return bool(
        np.all(np.isfinite(state.p_prime))
        and np.all(np.isfinite(state.ph_prime))
        and np.all(np.isfinite(state.mu_prime))
        and np.all(np.isfinite(state.w))
        and np.isfinite(state.accumulator)
        and np.isfinite(state.accumulator_comp)
    )
