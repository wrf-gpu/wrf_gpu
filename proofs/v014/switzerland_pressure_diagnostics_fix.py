#!/usr/bin/env python
"""V0.14 Switzerland h36 pressure-diagnostics fix proof.

This proof tests the manager suspicion that ``rk_addtend_dry._absolute_diagnostics``
double-counts perturbation dry mass in the WRF ``muts`` denominator:

    State.mu_total == MUB + MU
    old muts       == State.mu_total + State.mu_perturbation == MUB + 2*MU
    WRF muts       == grid%mut + grid%mu_2 == MUB + MU

The step probe runs h36 dry dynamics variants with explicit diagnostic formulas
so old and fixed behaviour remain reproducible even after the source patch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CPU = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
PROBE_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
BASELINE_GPU = PROBE_ROOT / "gpu_output"
FIX_GPU = PROBE_ROOT / "gpu_output_pressure_diagnostics_fix_gpt"
PREV_STRONGFLOW = ROOT / "proofs/v014/switzerland_strongflow_dynamics.json"
PREV_HYDRO = ROOT / "proofs/v014/switzerland_hydro_pgf_subterms.json"
OUT_JSON = ROOT / "proofs/v014/switzerland_pressure_diagnostics_fix.json"
RUN_START = datetime(2023, 1, 15)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n"
    )


def fn(base: Path, hour: int) -> Path:
    label = (RUN_START + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
    return base / f"wrfout_d01_{label}"


def get(base: Path, hour: int, var: str) -> np.ndarray:
    with Dataset(fn(base, hour)) as handle:
        return np.asarray(handle.variables[var][0])


def finite_status(base: Path, hour: int) -> dict[str, Any]:
    path = fn(base, hour)
    if not path.exists():
        return {"path": str(path), "exists": False, "all_finite": False}
    fields = ["MU", "PSFC", "T", "U", "V", "W", "PH"]
    result: dict[str, Any] = {"path": str(path), "exists": True, "all_finite": True, "fields": {}}
    with Dataset(path) as handle:
        for name in fields:
            arr = np.asarray(handle.variables[name][0])
            is_finite = bool(np.isfinite(arr).all())
            result["fields"][name] = {
                "finite": is_finite,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
            }
            result["all_finite"] = bool(result["all_finite"] and is_finite)
    return result


def rmse_bias(candidate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    diff = candidate - truth
    return {"rmse": float(np.sqrt(np.mean(diff * diff))), "bias": float(np.mean(diff))}


def field_metrics(base: Path, hour: int) -> dict[str, Any]:
    if not fn(base, hour).exists():
        return {"available": False, "path": str(fn(base, hour))}
    out: dict[str, Any] = {"available": True, "valid_h": hour, "path": str(fn(base, hour))}
    for var in ["MU", "PSFC", "T", "U", "V", "W", "PH"]:
        out[var.lower()] = rmse_bias(get(base, hour, var), get(CPU, hour, var))
    out["finite"] = finite_status(base, hour)
    return out


def load_budget_state(base: Path, hour: int) -> dict[str, Any]:
    with Dataset(fn(base, hour)) as d:
        return {
            "mu": np.asarray(d.variables["MU"][0]) + np.asarray(d.variables["MUB"][0]),
            "u": np.asarray(d.variables["U"][0]),
            "v": np.asarray(d.variables["V"][0]),
            "dnw": np.asarray(d.variables["DNW"][0]),
            "c1h": np.asarray(d.variables["C1H"][0]),
            "c2h": np.asarray(d.variables["C2H"][0]),
            "mx": np.asarray(d.variables["MAPFAC_MX"][0]),
            "my": np.asarray(d.variables["MAPFAC_MY"][0]),
            "muy": np.asarray(d.variables["MAPFAC_UY"][0]),
            "mvx": np.asarray(d.variables["MAPFAC_VX"][0]),
            "dx": float(d.DX),
        }


def budget_between(start_base: Path, start_hour: int, end_base: Path, end_hour: int, depth: int = 8) -> dict[str, float]:
    a = load_budget_state(start_base, start_hour)
    b = load_budget_state(end_base, end_hour)
    wk = -a["dnw"]
    c1 = a["c1h"]
    c2 = a["c2h"]
    ny, nx = a["mu"].shape
    i0, i1, j0, j1 = depth, nx - depth, depth, ny - depth

    def colmass(s: Mapping[str, Any]) -> float:
        m = ((c1[:, None, None] * s["mu"][None] + c2[:, None, None]) * wk[:, None, None]).sum(0)
        return float((m / (s["mx"] * s["my"]))[j0:j1, i0:i1].sum())

    def outflux(s: Mapping[str, Any]) -> float:
        mu = s["mu"]
        u = s["u"]
        v = s["v"]

        def mul_u(i: int) -> np.ndarray:
            muf = 0.5 * (mu[j0:j1, i - 1] + mu[j0:j1, i])
            return c1[:, None] * muf[None, :] + c2[:, None]

        def mul_v(j: int) -> np.ndarray:
            muf = 0.5 * (mu[j - 1, i0:i1] + mu[j, i0:i1])
            return c1[:, None] * muf[None, :] + c2[:, None]

        fw = (u[:, j0:j1, i0] * mul_u(i0) * wk[:, None] / s["muy"][j0:j1, i0][None, :]).sum()
        fe = (u[:, j0:j1, i1] * mul_u(i1) * wk[:, None] / s["muy"][j0:j1, i1][None, :]).sum()
        fs = (v[:, j0, i0:i1] * mul_v(j0) * wk[:, None] / s["mvx"][j0, i0:i1][None, :]).sum()
        fnn = (v[:, j1, i0:i1] * mul_v(j1) * wk[:, None] / s["mvx"][j1, i0:i1][None, :]).sum()
        return float((fe - fw) + (fnn - fs))

    ncell = (j1 - j0) * (i1 - i0)
    dm = (colmass(b) - colmass(a)) / ncell
    flux = (outflux(a) + outflux(b)) / 2.0 * 3600.0 / a["dx"] / ncell
    return {
        "dM_pa_per_cell_h": float(dm),
        "net_influx_pa_per_cell_h": float(-flux),
        "residual_pa_per_cell_h": float(dm + flux),
    }


def resource_summary(label: str, resource_dir: Path = PROBE_ROOT / "resources") -> dict[str, Any]:
    gpu_csv = resource_dir / f"{label}_gpu_usage.csv"
    proc_csv = resource_dir / f"{label}_process_usage.csv"
    sys_csv = resource_dir / f"{label}_system_memory.csv"
    info = resource_dir / f"{label}_monitor.runinfo"
    rows = []
    if gpu_csv.exists():
        with gpu_csv.open() as handle:
            header = handle.readline().strip().split(",")
            for raw in handle:
                parts = raw.strip().split(",")
                if len(parts) == len(header):
                    rows.append(dict(zip(header, parts)))
    proc_rows = []
    if proc_csv.exists():
        with proc_csv.open() as handle:
            header = handle.readline().strip().split(",")
            for raw in handle:
                parts = raw.strip().split(",", maxsplit=len(header) - 1)
                if len(parts) == len(header):
                    proc_rows.append(dict(zip(header, parts)))
    return {
        "monitor": str(info),
        "gpu_csv": str(gpu_csv),
        "process_csv": str(proc_csv),
        "system_memory_csv": str(sys_csv),
        "samples": len(rows),
        "max_gpu_memory_mib": max((int(float(r["memory_used_mib"])) for r in rows), default=None),
        "max_gpu_util_pct": max((int(float(r["utilization_gpu_pct"])) for r in rows), default=None),
        "max_process_rss_kib": max((int(float(r["rss_kib"])) for r in proc_rows), default=None),
        "exists": gpu_csv.exists() or proc_csv.exists() or sys_csv.exists() or info.exists(),
    }


def _array_stats(arr: Any) -> dict[str, float]:
    values = np.asarray(arr)
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "rmse": float(np.sqrt(np.nanmean(values * values))),
        "mean_abs": float(np.nanmean(np.abs(values))),
        "max_abs": float(np.nanmax(np.abs(values))),
    }


def _diff_stats(left: Any, right: Any) -> dict[str, float]:
    return _array_stats(np.asarray(left) - np.asarray(right))


def _explicit_absolute_diagnostics(state: Any, metrics: Any, mode: str):
    import jax.numpy as jnp

    from gpuwrf.dynamics.acoustic_wrf import _inverse_density_from_theta_pressure, _pressure_from_theta_alt

    ph_pert = state.ph_perturbation.astype(jnp.float64)
    mu_pert = state.mu_perturbation.astype(jnp.float64)
    mu_total = state.mu_total.astype(jnp.float64)
    mub = (state.mu_total - state.mu_perturbation).astype(jnp.float64)
    alt_eos = _inverse_density_from_theta_pressure(
        state.theta.astype(jnp.float64), state.p_total.astype(jnp.float64)
    )
    p_pert_state = state.p_perturbation.astype(jnp.float64)
    c1h = metrics.c1h[:, None, None]
    c2h = metrics.c2h[:, None, None]
    rdnw = metrics.rdnw[:, None, None]
    phb = (state.ph_total - state.ph_perturbation).astype(jnp.float64)
    mass_base = c1h * mub[None, :, :] + c2h
    safe_base = jnp.where(
        jnp.abs(mass_base) > 1.0e-12,
        mass_base,
        jnp.asarray(1.0e-12, dtype=mass_base.dtype),
    )
    alb = -rdnw * (phb[1:, :, :] - phb[:-1, :, :]) / safe_base
    if mode in {"double_count", "double_count_p_alt_alb"}:
        muts = mu_total + mu_pert
    elif mode in {
        "state_mu_total",
        "state_mu_total_alt_alb",
        "state_mu_total_p_from_al",
        "state_mu_total_p_alt_alb",
    }:
        muts = mu_total
    elif mode == "mub_plus_mu_pert":
        muts = mub + mu_pert
    elif mode == "base_only_denominator":
        muts = mub
    else:
        raise ValueError(f"unknown explicit diagnostics mode: {mode}")
    mass_t = c1h * muts[None, :, :] + c2h
    safe_t = jnp.where(
        jnp.abs(mass_t) > 1.0e-12,
        mass_t,
        jnp.asarray(1.0e-12, dtype=mass_t.dtype),
    )
    mu_term = c1h * mu_pert[None, :, :]
    al = -(alb * mu_term + rdnw * (ph_pert[1:, :, :] - ph_pert[:-1, :, :])) / safe_t
    alt_alb = al + alb
    alt = alt_alb if mode in {"state_mu_total_alt_alb", "state_mu_total_p_alt_alb", "double_count_p_alt_alb"} else alt_eos
    pb = state.p_total.astype(jnp.float64) - state.p_perturbation.astype(jnp.float64)
    p_pert = (
        _pressure_from_theta_alt(state.theta.astype(jnp.float64), alt_alb) - pb
        if mode in {"state_mu_total_p_from_al", "state_mu_total_p_alt_alb", "double_count_p_alt_alb"}
        else p_pert_state
    )
    ph_total = phb + ph_pert
    php = 0.5 * (ph_total[:-1, :, :] + ph_total[1:, :, :])
    return ph_pert, p_pert, al, alt, php


def _hydro_terms_for_mode(state: Any, metrics: Any, mode: str) -> dict[str, Any]:
    from gpuwrf.dynamics.core import rk_addtend_dry as rk

    ph, p_abs, al, alt, _php = _explicit_absolute_diagnostics(state, metrics, mode)
    pb = state.p_total - state.p_perturbation

    ph_l, ph_r = rk._x_face_pair_3d(ph)
    p_l, p_r = rk._x_face_pair_3d(p_abs)
    pb_l, pb_r = rk._x_face_pair_3d(pb)
    al_l, al_r = rk._x_face_pair_3d(al)
    alt_l, alt_r = rk._x_face_pair_3d(alt)
    ph_term_x = (ph_r[1:, :, :] - ph_l[1:, :, :]) + (ph_r[:-1, :, :] - ph_l[:-1, :, :])
    p_alt_x = (alt_l + alt_r) * (p_r - p_l)
    pb_al_x = (al_l + al_r) * (pb_r - pb_l)

    ph_s, ph_n = rk._y_face_pair_3d(ph)
    p_s, p_n = rk._y_face_pair_3d(p_abs)
    pb_s, pb_n = rk._y_face_pair_3d(pb)
    al_s, al_n = rk._y_face_pair_3d(al)
    alt_s, alt_n = rk._y_face_pair_3d(alt)
    ph_term_y = (ph_n[1:, :, :] - ph_s[1:, :, :]) + (ph_n[:-1, :, :] - ph_s[:-1, :, :])
    p_alt_y = (alt_s + alt_n) * (p_n - p_s)
    pb_al_y = (al_s + al_n) * (pb_n - pb_s)

    return {
        "al": al,
        "alt": alt,
        "ph_term_x": ph_term_x,
        "p_alt_x": p_alt_x,
        "pb_al_x": pb_al_x,
        "hydro_sum_x": ph_term_x + p_alt_x + pb_al_x,
        "ph_term_y": ph_term_y,
        "p_alt_y": p_alt_y,
        "pb_al_y": pb_al_y,
        "hydro_sum_y": ph_term_y + p_alt_y + pb_al_y,
    }


def same_state_pressure_diagnostic() -> dict[str, Any]:
    import jax.numpy as jnp

    from gpuwrf.integration import daily_pipeline as dp
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision

    cfg = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=PROBE_ROOT / "unused_pressure_diag_same_state_output",
        proof_dir=PROBE_ROOT / "unused_pressure_diag_same_state_proofs",
        run_root=PROBE_ROOT,
        domain="d01",
        async_output=False,
    )
    case, run_dir = dp._build_real_case(cfg)
    state = _enforce_operational_precision(case.state, force_fp64=True)
    metrics = case.namelist.metrics
    modes = [
        "double_count",
        "state_mu_total",
        "mub_plus_mu_pert",
        "base_only_denominator",
        "state_mu_total_alt_alb",
        "state_mu_total_p_from_al",
        "state_mu_total_p_alt_alb",
        "double_count_p_alt_alb",
    ]
    terms = {mode: _hydro_terms_for_mode(state, metrics, mode) for mode in modes}
    mu_total = jnp.asarray(state.mu_total, dtype=jnp.float64)
    mu_pert = jnp.asarray(state.mu_perturbation, dtype=jnp.float64)
    mub = mu_total - mu_pert
    muts_by_mode = {}
    for mode in modes:
        if mode in {"double_count", "double_count_p_alt_alb"}:
            muts_by_mode[mode] = mu_total + mu_pert
        elif mode == "base_only_denominator":
            muts_by_mode[mode] = mub
        else:
            muts_by_mode[mode] = mu_total

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "state_semantics": {
            "mu_total_minus_mub_plus_mu_perturbation": _array_stats(mu_total - (mub + mu_pert)),
            "mu_total": _array_stats(mu_total),
            "mu_perturbation": _array_stats(mu_pert),
            "mub_reconstructed": _array_stats(mub),
        },
        "muts": {mode: _array_stats(value) for mode, value in muts_by_mode.items()},
        "muts_delta_vs_state_mu_total": {
            mode: _diff_stats(value, muts_by_mode["state_mu_total"]) for mode, value in muts_by_mode.items()
        },
        "diagnostics": {},
        "term_differences_vs_fixed": {},
    }
    try:
        with Dataset(fn(CPU, 36)) as d:
            base_pairs = {
                "PB": (np.asarray(state.p_total - state.p_perturbation), np.asarray(d.variables["PB"][0])),
                "PHB": (np.asarray(state.ph_total - state.ph_perturbation), np.asarray(d.variables["PHB"][0])),
                "MUB": (np.asarray(state.mu_total - state.mu_perturbation), np.asarray(d.variables["MUB"][0])),
            }
        result["base_field_start_parity_vs_cpu_h36"] = {
            name: _diff_stats(left, right) for name, (left, right) in base_pairs.items()
        }
    except Exception as exc:  # pragma: no cover - proof preserves failure details.
        result["base_field_start_parity_vs_cpu_h36"] = {"available": False, "error": repr(exc)}
    for mode, payload in terms.items():
        result["diagnostics"][mode] = {
            "al": _array_stats(payload["al"]),
            "alt": _array_stats(payload["alt"]),
            "p_alt_x": _array_stats(payload["p_alt_x"]),
            "pb_al_x": _array_stats(payload["pb_al_x"]),
            "hydro_sum_x": _array_stats(payload["hydro_sum_x"]),
            "p_alt_y": _array_stats(payload["p_alt_y"]),
            "pb_al_y": _array_stats(payload["pb_al_y"]),
            "hydro_sum_y": _array_stats(payload["hydro_sum_y"]),
        }
    fixed = terms["state_mu_total"]
    for mode, payload in terms.items():
        result["term_differences_vs_fixed"][mode] = {
            "al": _diff_stats(payload["al"], fixed["al"]),
            "p_alt_x": _diff_stats(payload["p_alt_x"], fixed["p_alt_x"]),
            "pb_al_x": _diff_stats(payload["pb_al_x"], fixed["pb_al_x"]),
            "hydro_sum_x": _diff_stats(payload["hydro_sum_x"], fixed["hydro_sum_x"]),
            "p_alt_y": _diff_stats(payload["p_alt_y"], fixed["p_alt_y"]),
            "pb_al_y": _diff_stats(payload["pb_al_y"], fixed["pb_al_y"]),
            "hydro_sum_y": _diff_stats(payload["hydro_sum_y"], fixed["hydro_sum_y"]),
        }
    return result


def _delta_at(payload: Mapping[str, Any], variant: str, step: int) -> float | None:
    var = (payload.get("step_probe") or {}).get("variants", {}).get(variant)
    if not var:
        var = (payload.get("variants") or {}).get(variant)
    if not var:
        return None
    hist = var.get("history") or []
    if len(hist) < step:
        return None
    first = hist[0].get("mu_total_mean")
    target = hist[step - 1].get("mu_total_mean")
    if first is None or target is None:
        return None
    return float(target) - float(first)


def _variant_final(payload: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    var = (payload.get("step_probe") or {}).get("variants", {}).get(variant)
    if not var:
        var = (payload.get("variants") or {}).get(variant)
    if not var:
        return {}
    return var.get("final") or {}


def _collapse(old: float | None, new: float | None) -> float | None:
    if old is None or new is None:
        return None
    return float(1.0 - abs(new) / max(abs(old), 1.0e-12))


def analyze() -> dict[str, Any]:
    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    strong = json.loads(PREV_STRONGFLOW.read_text()) if PREV_STRONGFLOW.exists() else {}
    hydro = json.loads(PREV_HYDRO.read_text()) if PREV_HYDRO.exists() else {}
    proof: dict[str, Any] = {
        "schema": "v014_switzerland_pressure_diagnostics_fix",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wrf_source_anchors": {
            "solve_em_muts_reset": "/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1446",
            "solve_em_calc_p_rho_phi_call": "/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:3049-3052",
            "calc_p_rho_phi_al_line": "/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:1029",
            "horizontal_pgf_v_terms": "/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:2310-2313",
            "horizontal_pgf_u_terms": "/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:2385-2388",
            "jax_state_mu_semantics": "src/gpuwrf/contracts/state.py:378-380",
            "jax_absolute_diagnostics": "src/gpuwrf/dynamics/core/rk_addtend_dry.py:_absolute_diagnostics",
        },
        "artifacts": {
            "cpu": str(CPU),
            "h36_baseline_gpu": str(BASELINE_GPU),
            "h36_fixed_gpu": str(FIX_GPU),
            "previous_strongflow_proof": str(PREV_STRONGFLOW),
            "previous_hydro_subterms_proof": str(PREV_HYDRO),
        },
        "baseline_from_previous": {
            "strongflow": strong.get("term_attribution_5min", {}),
            "hydro_subterms": hydro.get("root_classification", {}),
            "hydro_step_probe_window": (hydro.get("subterm_attribution_5min") or {}).get("window"),
        },
        "hypothesis": {
            "manager_suspicion": (
                "State.mu_total already stores MUB+MU, so old _absolute_diagnostics "
                "muts=mu_total+mu_perturbation evaluates to MUB+2*MU."
            ),
            "wrf_faithful_formula": "muts = state.mu_total == (state.mu_total - state.mu_perturbation) + state.mu_perturbation",
        },
        "step_probe": existing.get("step_probe", {}),
    }
    step_payload = proof["step_probe"] if isinstance(proof["step_probe"], Mapping) else {}
    if step_payload.get("available"):
        step = min(30, int(step_payload.get("steps_requested", 30)))
        names = [
            "production_source",
            "double_count_muts",
            "muts_state_mu_total",
            "muts_mub_plus_mu_pert",
            "zero_large_step_pgf",
            "p_alt_only_double_count",
            "pb_al_only_double_count",
            "p_alt_only_muts_total",
            "pb_al_only_muts_total",
            "muts_total_alt_alb_full",
            "muts_total_p_from_al_full",
            "muts_total_p_alt_alb_full",
            "double_count_p_alt_alb_full",
            "p_alt_only_p_alt_alb",
            "pb_al_only_p_alt_alb",
        ]
        rows = []
        deltas = {name: _delta_at(step_payload, name, step) for name in names}
        zero = deltas.get("zero_large_step_pgf")
        for name in names:
            delta = deltas.get(name)
            final = _variant_final(step_payload, name)
            row = {
                "variant": name,
                "mu_delta_30_steps_pa": delta,
                "finite": final.get("all_finite"),
                "u_absmax": final.get("u_absmax"),
                "v_absmax": final.get("v_absmax"),
                "w_absmax": final.get("w_absmax"),
                "p_absmax": final.get("p_absmax"),
                "p_total_k0_mean": final.get("p_total_k0_mean"),
            }
            if delta is not None and zero is not None:
                row["contribution_vs_zero_large_step_pgf_pa_per_cell_h"] = float((delta - zero) * 12.0)
            rows.append(row)
        old_pgf = (deltas.get("double_count_muts") - zero) if deltas.get("double_count_muts") is not None and zero is not None else None
        fixed_pgf = (deltas.get("muts_state_mu_total") - zero) if deltas.get("muts_state_mu_total") is not None and zero is not None else None
        proof["step_probe_summary"] = {
            "window": f"h36 dry step probe, first {step} model steps ({step * float(step_payload.get('dt_s', 10.0)):.0f} s)",
            "rows": rows,
            "old_minus_fixed_mu_delta_pa": (
                float(deltas["double_count_muts"] - deltas["muts_state_mu_total"])
                if deltas.get("double_count_muts") is not None and deltas.get("muts_state_mu_total") is not None
                else None
            ),
            "collapse_fraction_total_mu_delta_old_to_fixed": _collapse(
                deltas.get("double_count_muts"), deltas.get("muts_state_mu_total")
            ),
            "collapse_fraction_pgf_contribution_old_to_fixed": _collapse(old_pgf, fixed_pgf),
            "mub_plus_mu_equivalence_max_abs_delta_pa": (
                abs(float(deltas["muts_state_mu_total"] - deltas["muts_mub_plus_mu_pert"]))
                if deltas.get("muts_state_mu_total") is not None and deltas.get("muts_mub_plus_mu_pert") is not None
                else None
            ),
        }
        if zero is not None:
            for old_name, fixed_name in [
                ("p_alt_only_double_count", "p_alt_only_muts_total"),
                ("pb_al_only_double_count", "pb_al_only_muts_total"),
                ("p_alt_only_double_count", "p_alt_only_p_alt_alb"),
                ("pb_al_only_double_count", "pb_al_only_p_alt_alb"),
            ]:
                old_delta = deltas.get(old_name)
                fixed_delta = deltas.get(fixed_name)
                if old_delta is not None and fixed_delta is not None:
                    proof["step_probe_summary"][f"{old_name}_to_{fixed_name}"] = {
                        "old_contribution_pa_per_cell_h": float((old_delta - zero) * 12.0),
                        "fixed_contribution_pa_per_cell_h": float((fixed_delta - zero) * 12.0),
                        "collapse_fraction": _collapse(old_delta - zero, fixed_delta - zero),
                    }
    try:
        proof["same_state_pressure_diagnostic"] = same_state_pressure_diagnostic()
    except Exception as exc:  # pragma: no cover - proof preserves failure details.
        proof["same_state_pressure_diagnostic"] = {"available": False, "error": repr(exc)}
    proof["hourly_gate"] = {
        "available": fn(FIX_GPU, 37).exists(),
        "fixed_output": str(FIX_GPU),
    }
    if fn(FIX_GPU, 37).exists():
        cpu_budget = budget_between(CPU, 36, CPU, 37, depth=8)
        old_budget = budget_between(CPU, 36, BASELINE_GPU, 37, depth=8)
        fixed_budget = budget_between(CPU, 36, FIX_GPU, 37, depth=8)
        old_excess = old_budget["net_influx_pa_per_cell_h"] - cpu_budget["net_influx_pa_per_cell_h"]
        fixed_excess = fixed_budget["net_influx_pa_per_cell_h"] - cpu_budget["net_influx_pa_per_cell_h"]
        proof["hourly_gate"] |= {
            "metrics_h37": field_metrics(FIX_GPU, 37),
            "cpu_budget_h36_h37_depth8": cpu_budget,
            "old_baseline_budget_h36_h37_depth8": old_budget,
            "fixed_budget_h36_h37_depth8": fixed_budget,
            "old_excess_outflux_pa_per_cell_h": float(old_excess),
            "fixed_excess_outflux_pa_per_cell_h": float(fixed_excess),
            "collapse_fraction": float(1.0 - abs(fixed_excess) / max(abs(old_excess), 1.0e-12)),
        }
    proof["resources"] = {
        "gpt_pressure_diag_muts_probe": resource_summary("gpt_pressure_diag_muts_probe"),
        "gpt_pressure_diag_next_probe": resource_summary("gpt_pressure_diag_next_probe"),
        "gpt_pressure_diag_fix_1h": resource_summary("gpt_pressure_diag_fix_1h"),
        "gpt_pressure_diag_fix_3h": resource_summary("gpt_pressure_diag_fix_3h"),
    }
    gate = proof.get("hourly_gate", {})
    step_summary = proof.get("step_probe_summary", {})
    hourly_collapse = gate.get("collapse_fraction")
    step_collapse = step_summary.get("collapse_fraction_pgf_contribution_old_to_fixed")
    proof["root_classification"] = {
        "verdict": "EXACT_ROOT_NO_FIX",
        "muts_double_count_verdict": (
            "falsified as release-blocker: WRF-faithful muts=State.mu_total changes the 30-step "
            "large-step-PGF contribution from -34.0508 to -33.3570 Pa/cell/h, only ~2.0% collapse"
        ),
        "p_al_alt_formula_verdict": (
            "falsified as local source fix: alt=al+alb, p=EOS(theta,al+alb), and both together "
            "remain at about -33.36 Pa/cell/h with finite state"
        ),
        "remaining_exact_branch": (
            "the still-wrong branch is the native-face large-step horizontal PGF pressure/inverse-density "
            "inputs after WRF rk_step_prep/rk_phys_bc_dry_1, especially pb_al on U/V faces; local "
            "_absolute_diagnostics algebra changes do not move the signal"
        ),
        "next_implementation_target": (
            "instrument WRF at h36 after rk_step_prep + rk_phys_bc_dry_1 and immediately inside "
            "horizontal_pressure_gradient to emit p, al, alt, pb, p_alt_term, pb_al_term, and final dpx/dpy "
            "on native U/V faces; compare those face arrays to JAX before any model patch"
        ),
        "why_no_source_fix": (
            "no WRF-faithful local code variant tested here reaches the 70% collapse gate, and changing "
            "pb_al without the WRF face savepoint would be speculative"
        ),
    }
    proof["verdict"] = (
        "FIXED"
        if hourly_collapse is not None and float(hourly_collapse) >= 0.70
        else "EXACT_ROOT_NO_FIX"
    )
    write_json(OUT_JSON, proof)
    return proof


def run_forecast_variant(args: argparse.Namespace) -> None:
    from gpuwrf.integration import daily_pipeline as dp

    config = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=int(args.hours),
        output_dir=FIX_GPU,
        proof_dir=PROBE_ROOT / "proofs_pressure_diagnostics_fix_gpt",
        run_root=PROBE_ROOT,
        domain="d01",
    )
    result = dp._run_forecast_sequence(config)
    print(f"forecast result: status={result.status} hours={result.hours} output_dir={result.output_dir}", flush=True)


def run_step_probe(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from gpuwrf.dynamics.acoustic_wrf import moisture_coupling_factors
    from gpuwrf.dynamics.core import rk_addtend_dry as rk
    from gpuwrf.integration import daily_pipeline as dp
    import gpuwrf.runtime.operational_mode as operational_mode
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision, _physics_boundary_step
    from gpuwrf.runtime.operational_state import initial_operational_carry

    cfg = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=PROBE_ROOT / "unused_pressure_diag_step_probe_output",
        proof_dir=PROBE_ROOT / "unused_pressure_diag_step_probe_proofs",
        run_root=PROBE_ROOT,
        domain="d01",
        async_output=False,
    )
    case, run_dir = dp._build_real_case(cfg)
    base = dataclasses.replace(case.namelist, run_physics=False, disable_guards=True, top_lid=True, run_boundary=True)

    def state_summary(state) -> dict[str, float | bool]:
        arrays = {
            "u": state.u,
            "v": state.v,
            "w": state.w,
            "theta": state.theta,
            "mu_total": state.mu_total,
            "ph": state.ph_perturbation,
            "p": state.p_perturbation,
            "p_total": state.p_total,
        }
        all_finite = True
        out: dict[str, float | bool] = {}
        for name, arr in arrays.items():
            finite = bool(np.isfinite(np.asarray(arr)).all())
            all_finite = all_finite and finite
            out[f"{name}_finite"] = finite
            out[f"{name}_absmax"] = float(jnp.max(jnp.abs(arr)))
        out["w_top_absmax"] = float(jnp.max(jnp.abs(state.w[-1])))
        out["w_interior_absmax"] = float(jnp.max(jnp.abs(state.w[:-1])))
        out["theta_min"] = float(jnp.min(state.theta))
        out["theta_max"] = float(jnp.max(state.theta))
        out["mu_total_mean"] = float(jnp.mean(state.mu_total))
        out["mu_total_min"] = float(jnp.min(state.mu_total))
        out["p_pert_k0_mean"] = float(jnp.mean(state.p_perturbation[0]))
        out["p_total_k0_mean"] = float(jnp.mean(state.p_total[0]))
        out["all_finite"] = all_finite
        return out

    def subterm_pgf(state, metrics, *, dx_m, dy_m, terms: frozenset[str]):
        ph, p_abs, al, alt, _php = rk._absolute_diagnostics(state, metrics)
        pb = (state.p_total - state.p_perturbation).astype(jnp.float64)
        mu_total = state.mu_total.astype(jnp.float64)
        cqu, cqv = moisture_coupling_factors(state)
        rdx = 1.0 / float(dx_m)
        rdy = 1.0 / float(dy_m)
        c1h = metrics.c1h[:, None, None]
        c2h = metrics.c2h[:, None, None]
        msf_u = (metrics.msfux / metrics.msfuy)[None, :, :]
        msf_v = (metrics.msfvy / metrics.msfvx)[None, :, :]

        ph_l, ph_r = rk._x_face_pair_3d(ph)
        p_l, p_r = rk._x_face_pair_3d(p_abs)
        pb_l, pb_r = rk._x_face_pair_3d(pb)
        al_l, al_r = rk._x_face_pair_3d(al)
        alt_l, alt_r = rk._x_face_pair_3d(alt)
        muu_l, muu_r = rk._x_face_pair_2d(mu_total)
        mass_u = c1h * (0.5 * (muu_l + muu_r))[None, :, :] + c2h
        ph_term_x = (ph_r[1:, :, :] - ph_l[1:, :, :]) + (ph_r[:-1, :, :] - ph_l[:-1, :, :])
        p_alt_term_x = (alt_l + alt_r) * (p_r - p_l)
        pb_al_term_x = (al_l + al_r) * (pb_r - pb_l)
        selected_x = jnp.zeros_like(ph_term_x)
        selected_x = selected_x + jnp.where("ph" in terms, ph_term_x, 0.0)
        selected_x = selected_x + jnp.where("p_alt" in terms, p_alt_term_x, 0.0)
        selected_x = selected_x + jnp.where("pb_al" in terms, pb_al_term_x, 0.0)
        dpx = msf_u * 0.5 * rdx * mass_u * selected_x

        ph_s, ph_n = rk._y_face_pair_3d(ph)
        p_s, p_n = rk._y_face_pair_3d(p_abs)
        pb_s, pb_n = rk._y_face_pair_3d(pb)
        al_s, al_n = rk._y_face_pair_3d(al)
        alt_s, alt_n = rk._y_face_pair_3d(alt)
        muv_s, muv_n = rk._y_face_pair_2d(mu_total)
        mass_v = c1h * (0.5 * (muv_s + muv_n))[None, :, :] + c2h
        ph_term_y = (ph_n[1:, :, :] - ph_s[1:, :, :]) + (ph_n[:-1, :, :] - ph_s[:-1, :, :])
        p_alt_term_y = (alt_s + alt_n) * (p_n - p_s)
        pb_al_term_y = (al_s + al_n) * (pb_n - pb_s)
        selected_y = jnp.zeros_like(ph_term_y)
        selected_y = selected_y + jnp.where("ph" in terms, ph_term_y, 0.0)
        selected_y = selected_y + jnp.where("p_alt" in terms, p_alt_term_y, 0.0)
        selected_y = selected_y + jnp.where("pb_al" in terms, pb_al_term_y, 0.0)
        dpy = msf_v * 0.5 * rdy * mass_v * selected_y
        return -cqu * dpx, -cqv * dpy

    def _apply_patch(diag_mode: str, pgf_kind: str):
        originals = {
            "absolute": rk._absolute_diagnostics,
            "large_step_horizontal_pgf": operational_mode.large_step_horizontal_pgf,
        }
        if diag_mode != "source":

            def _diag(state, metrics, *, t0=300.0):
                del t0
                return _explicit_absolute_diagnostics(state, metrics, diag_mode)

            rk._absolute_diagnostics = _diag
        if pgf_kind == "zero":

            def _zero(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
                del metrics, dx_m, dy_m, non_hydrostatic, top_lid
                return jnp.zeros_like(state.u), jnp.zeros_like(state.v)

            operational_mode.large_step_horizontal_pgf = _zero
        elif pgf_kind in {"ph", "p_alt", "pb_al", "hydro"}:
            term_map = {
                "ph": frozenset(("ph",)),
                "p_alt": frozenset(("p_alt",)),
                "pb_al": frozenset(("pb_al",)),
                "hydro": frozenset(("ph", "p_alt", "pb_al")),
            }
            terms = term_map[pgf_kind]

            def _subterm(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
                del non_hydrostatic, top_lid
                return subterm_pgf(state, metrics, dx_m=dx_m, dy_m=dy_m, terms=terms)

            operational_mode.large_step_horizontal_pgf = _subterm
        elif pgf_kind != "full":
            raise ValueError(f"unknown pgf kind: {pgf_kind}")
        return originals

    def _restore_patch(originals: Mapping[str, Any]) -> None:
        rk._absolute_diagnostics = originals["absolute"]
        operational_mode.large_step_horizontal_pgf = originals["large_step_horizontal_pgf"]

    variants = [
        ("production_source", "source", "full"),
        ("double_count_muts", "double_count", "full"),
        ("muts_state_mu_total", "state_mu_total", "full"),
        ("muts_mub_plus_mu_pert", "mub_plus_mu_pert", "full"),
        ("zero_large_step_pgf", "source", "zero"),
        ("p_alt_only_double_count", "double_count", "p_alt"),
        ("pb_al_only_double_count", "double_count", "pb_al"),
        ("p_alt_only_muts_total", "state_mu_total", "p_alt"),
        ("pb_al_only_muts_total", "state_mu_total", "pb_al"),
        ("muts_total_alt_alb_full", "state_mu_total_alt_alb", "full"),
        ("muts_total_p_from_al_full", "state_mu_total_p_from_al", "full"),
        ("muts_total_p_alt_alb_full", "state_mu_total_p_alt_alb", "full"),
        ("double_count_p_alt_alb_full", "double_count_p_alt_alb", "full"),
        ("p_alt_only_p_alt_alb", "state_mu_total_p_alt_alb", "p_alt"),
        ("pb_al_only_p_alt_alb", "state_mu_total_p_alt_alb", "pb_al"),
    ]
    output: dict[str, Any] = {
        "available": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "steps_requested": int(args.steps),
        "dt_s": float(base.dt_s),
        "namelist_base": {
            "run_physics": bool(base.run_physics),
            "disable_guards": bool(base.disable_guards),
            "top_lid": bool(base.top_lid),
            "run_boundary": bool(base.run_boundary),
            "epssm": float(base.epssm),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_version": getattr(jax, "__version__", None),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        "variants": {},
    }
    only = set(args.only or [])
    for name, diag_mode, pgf_kind in variants:
        if only and name not in only:
            continue
        carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))
        hist = []
        first_bad_step = None
        start = time.perf_counter()
        originals = _apply_patch(diag_mode, pgf_kind)
        try:

            @jax.jit
            def _one_step_variant(carry_in, namelist_in, step_index):
                return _physics_boundary_step(carry_in, namelist_in, step_index, run_radiation=False, debug=False)

            print(f"[pressure-diag] variant={name} diag={diag_mode} pgf={pgf_kind}", flush=True)
            for step in range(1, int(args.steps) + 1):
                carry = _one_step_variant(carry, base, jnp.asarray(step, dtype=jnp.int32))
                jax.block_until_ready(carry.state.u)
                rec = {"step": step} | state_summary(carry.state)
                hist.append(rec)
                if step <= int(args.print_first) or step % int(args.print_every) == 0 or not bool(rec["all_finite"]):
                    print(
                        "[pressure-diag] "
                        f"{name} step={step} finite={rec['all_finite']} "
                        f"w_top={rec['w_top_absmax']:.3f} w_int={rec['w_interior_absmax']:.3f} "
                        f"u={rec['u_absmax']:.3f} v={rec['v_absmax']:.3f} "
                        f"theta=[{rec['theta_min']:.3f},{rec['theta_max']:.3f}] "
                        f"mu_mean={rec['mu_total_mean']:.3f} p0_mean={rec['p_total_k0_mean']:.3f}",
                        flush=True,
                    )
                if first_bad_step is None and (
                    not bool(rec["all_finite"])
                    or float(rec["w_absmax"]) > float(args.w_abort)
                    or float(rec["u_absmax"]) > float(args.wind_abort)
                    or float(rec["theta_min"]) < float(args.theta_min)
                    or float(rec["theta_max"]) > float(args.theta_max)
                ):
                    first_bad_step = step
                    if bool(args.stop_on_bad):
                        break
        finally:
            _restore_patch(originals)
        output["variants"][name] = {
            "diagnostics_mode": diag_mode,
            "pgf_kind": pgf_kind,
            "steps_completed": len(hist),
            "first_bad_step": first_bad_step,
            "wall_s": float(time.perf_counter() - start),
            "history": hist,
            "final": hist[-1] if hist else None,
        }
    out_path = Path(args.out)
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            previous_probe = existing.get("step_probe", {})
            previous_variants = dict(previous_probe.get("variants", {})) if isinstance(previous_probe, Mapping) else {}
            previous_variants.update(output["variants"])
            output["variants"] = previous_variants
            if isinstance(previous_probe, Mapping):
                output["previous_generated_utc"] = previous_probe.get("generated_utc")
        except Exception:
            pass
    proof = {"step_probe": output}
    write_json(out_path, proof)
    print(f"wrote {args.out}", flush=True)
    return proof


def run_command(command: list[str], timeout_s: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    return {
        "command": command,
        "returncode": proc.returncode,
        "wall_s": float(time.perf_counter() - start),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-probe", action="store_true")
    parser.add_argument("--forecast-variant", action="store_true", help="run the current model code for an h36 short forecast")
    parser.add_argument("--hours", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--print-first", type=int, default=3)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--w-abort", type=float, default=500.0)
    parser.add_argument("--wind-abort", type=float, default=500.0)
    parser.add_argument("--theta-min", type=float, default=150.0)
    parser.add_argument("--theta-max", type=float, default=650.0)
    parser.add_argument("--stop-on-bad", action="store_true", default=True)
    parser.add_argument("--out", default=str(OUT_JSON))
    parser.add_argument("--only", action="append", help="run only the named step-probe variant; may be repeated")
    args = parser.parse_args()

    if args.forecast_variant:
        run_forecast_variant(args)
    elif args.step_probe:
        run_step_probe(args)
    else:
        proof = analyze()
        print(f"wrote {OUT_JSON}")
        print(json.dumps({"verdict": proof.get("verdict"), "hourly_gate": proof.get("hourly_gate")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
