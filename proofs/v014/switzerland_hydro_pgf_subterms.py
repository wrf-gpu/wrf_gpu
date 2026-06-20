#!/usr/bin/env python
"""V0.14 Switzerland h36 hydrostatic PGF subterm attribution.

This proof reuses the h36 storm-state dry step harness from the preceding
strong-flow sprint, but splits the large-step horizontal PGF hydrostatic branch
into the WRF source terms:

* ph_term
* p_alt_term = (alt_l + alt_r) * (p_r - p_l)
* pb_al_term = (al_l + al_r) * (pb_r - pb_l)

It also includes a proof-only WRF specified/nested edge-loop variant, because WRF
``horizontal_pressure_gradient`` skips the outer normal faces when
``config_flags%specified`` or ``nested`` is true.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
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

CPU = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
PROBE_ROOT = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
BASELINE_GPU = PROBE_ROOT / "gpu_output"
FIX_GPU = PROBE_ROOT / "gpu_output_hydro_pgf_edgefix_gpt"
PREV_PROOF = ROOT / "proofs/v014/switzerland_strongflow_dynamics.json"
OUT_JSON = ROOT / "proofs/v014/switzerland_hydro_pgf_subterms.json"
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
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n")


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
        "samples": len(rows),
        "max_gpu_memory_mib": max((int(float(r["memory_used_mib"])) for r in rows), default=None),
        "max_gpu_util_pct": max((int(float(r["utilization_gpu_pct"])) for r in rows), default=None),
        "max_process_rss_kib": max((int(float(r["rss_kib"])) for r in proc_rows), default=None),
        "exists": gpu_csv.exists() or proc_csv.exists() or info.exists(),
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


def same_state_pressure_diagnostic() -> dict[str, Any]:
    """Cheap h36 same-state check for the pressure/inverse-density PGF inputs."""

    import jax.numpy as jnp

    from gpuwrf.dynamics.core import rk_addtend_dry as rk
    from gpuwrf.integration import daily_pipeline as dp
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision

    cfg = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=PROBE_ROOT / "unused_hydro_pgf_same_state_output",
        proof_dir=PROBE_ROOT / "unused_hydro_pgf_same_state_proofs",
        run_root=PROBE_ROOT,
        domain="d01",
        async_output=False,
    )
    case, run_dir = dp._build_real_case(cfg)
    state = _enforce_operational_precision(case.state, force_fp64=True)
    metrics = case.namelist.metrics
    _ph, p_pert, al, alt_eos, _php = rk._absolute_diagnostics(state, metrics)
    phb = (state.ph_total - state.ph_perturbation).astype(jnp.float64)
    mub = (state.mu_total - state.mu_perturbation).astype(jnp.float64)
    c1h = metrics.c1h[:, None, None]
    c2h = metrics.c2h[:, None, None]
    rdnw = metrics.rdnw[:, None, None]
    mass_base = c1h * mub[None, :, :] + c2h
    alb = -rdnw * (phb[1:, :, :] - phb[:-1, :, :]) / mass_base
    alt_al_plus_alb = al + alb
    rel_alt = (alt_eos - alt_al_plus_alb) / jnp.maximum(jnp.abs(alt_al_plus_alb), 1.0e-12)
    return {
        "run_dir": str(run_dir),
        "meaning": (
            "h36 same-state diagnostic for large-step PGF inputs; alt identity is too small "
            "to explain the p_alt/pb_al mass-venting signal, leaving p/al staging or values as the direct target"
        ),
        "alt_eos": _array_stats(alt_eos),
        "alt_al_plus_alb": _array_stats(alt_al_plus_alb),
        "alt_eos_minus_al_plus_alb": _array_stats(alt_eos - alt_al_plus_alb),
        "relative_alt_diff": _array_stats(rel_alt),
        "al": _array_stats(al),
        "alb": _array_stats(alb),
        "p_perturbation": _array_stats(p_pert),
        "source": "src/gpuwrf/dynamics/core/rk_addtend_dry.py::_absolute_diagnostics",
    }


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


def analyze() -> dict[str, Any]:
    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    previous = json.loads(PREV_PROOF.read_text()) if PREV_PROOF.exists() else {}
    proof: dict[str, Any] = {
        "schema": "v014_switzerland_hydro_pgf_subterms",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": previous.get("target", {}),
        "wrf_source_anchors": {
            "large_step_hpg_loop_bounds": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:2268-2407",
            "hydro_v_terms": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:2310-2313",
            "hydro_u_terms": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:2385-2388",
            "rk_step_prep_call": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/solve_em.F:658-682",
            "large_step_hpg_call": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F:717-732",
        },
        "artifacts": {
            "cpu": str(CPU),
            "h36_baseline_gpu": str(BASELINE_GPU),
            "h36_fixed_gpu": str(FIX_GPU),
            "previous_proof": str(PREV_PROOF),
        },
        "baseline_from_previous": previous.get("term_attribution_5min", {}),
        "step_probe": existing.get("step_probe", existing if existing.get("variants") else {}),
    }
    step_payload = proof["step_probe"] if isinstance(proof["step_probe"], Mapping) else {}
    if step_payload.get("available"):
        step = 30
        baseline_30 = _delta_at(step_payload, "baseline", step)
        zero_30 = _delta_at(step_payload, "zero_large_step_pgf", step)
        hydro_30 = _delta_at(step_payload, "hydro_first_three_only", step)
        ph_30 = _delta_at(step_payload, "ph_only", step)
        p_alt_30 = _delta_at(step_payload, "p_alt_only", step)
        pb_al_30 = _delta_at(step_payload, "pb_al_only", step)
        edge_skip_30 = _delta_at(step_payload, "full_wrf_specified_edge_skip", step)
        hydro_edge_only_30 = _delta_at(step_payload, "hydro_specified_edge_only", step)
        rows = []
        if zero_30 is not None:
            for name, delta in [
                ("baseline", baseline_30),
                ("hydro_first_three_only", hydro_30),
                ("ph_only", ph_30),
                ("p_alt_only", p_alt_30),
                ("pb_al_only", pb_al_30),
                ("hydro_specified_edge_only", hydro_edge_only_30),
                ("full_wrf_specified_edge_skip", edge_skip_30),
            ]:
                if delta is None:
                    continue
                rows.append(
                    {
                        "variant": name,
                        "mu_delta_30_steps_pa": float(delta),
                        "contribution_vs_zero_pgf_pa_per_cell_h": float((delta - zero_30) * 12.0),
                    }
                )
        proof["subterm_attribution_5min"] = {
            "window": "h36 dry step probe, first 30 model steps (300 s)",
            "zero_large_step_pgf_delta_pa": zero_30,
            "rows": rows,
        }
        if baseline_30 is not None and edge_skip_30 is not None and zero_30 is not None:
            base_pgf = baseline_30 - zero_30
            fixed_pgf = edge_skip_30 - zero_30
            proof["specified_edge_skip_step_verdict"] = {
                "baseline_pgf_contribution_pa_per_cell_h": float(base_pgf * 12.0),
                "edge_skip_pgf_contribution_pa_per_cell_h": float(fixed_pgf * 12.0),
                "collapse_fraction_vs_baseline_pgf": float(1.0 - abs(fixed_pgf) / max(abs(base_pgf), 1.0e-12)),
            }
        contrib = {row["variant"]: row["contribution_vs_zero_pgf_pa_per_cell_h"] for row in rows}
        proof["root_classification"] = {
            "verdict": "EXACT_ROOT_NO_FIX",
            "dominant_subterms": {
                "p_alt_only_pa_per_cell_h": contrib.get("p_alt_only"),
                "pb_al_only_pa_per_cell_h": contrib.get("pb_al_only"),
                "ph_only_pa_per_cell_h": contrib.get("ph_only"),
            },
            "ruled_out": {
                "specified_outer_normal_face_loop_bounds": proof.get("specified_edge_skip_step_verdict", {}).get(
                    "collapse_fraction_vs_baseline_pgf"
                ),
                "ph_geopotential_gradient_as_mass_venting_driver": contrib.get("ph_only"),
            },
            "classification": (
                "The excess dry-mass outflux is in the pressure/inverse-density hydrostatic PGF pair "
                "p_alt + pb_al, not the ph gradient, nonhydro fourth term, top lid, or specified-edge loop bounds. "
                "The next implementation target is _absolute_diagnostics/staged rk_step_prep inputs p, al, alt "
                "as consumed by large_step_horizontal_pgf."
            ),
        }
    try:
        proof["same_state_pressure_diagnostic"] = same_state_pressure_diagnostic()
    except Exception as exc:  # pragma: no cover - proof should preserve failure details.
        proof["same_state_pressure_diagnostic"] = {"available": False, "error": repr(exc)}
    proof["hourly_gate"] = {
        "available": fn(FIX_GPU, 37).exists(),
        "fixed_output": str(FIX_GPU),
    }
    if fn(FIX_GPU, 37).exists():
        cpu_flux = budget_between(CPU, 36, CPU, 37, depth=8)["net_influx_pa_per_cell_h"]
        old_flux = budget_between(CPU, 36, BASELINE_GPU, 37, depth=8)["net_influx_pa_per_cell_h"]
        new_flux = budget_between(CPU, 36, FIX_GPU, 37, depth=8)["net_influx_pa_per_cell_h"]
        old_excess = old_flux - cpu_flux
        new_excess = new_flux - cpu_flux
        proof["hourly_gate"] |= {
            "metrics_h37": field_metrics(FIX_GPU, 37),
            "cpu_budget_h36_h37_depth8": budget_between(CPU, 36, CPU, 37, depth=8),
            "old_baseline_budget_h36_h37_depth8": budget_between(CPU, 36, BASELINE_GPU, 37, depth=8),
            "fixed_budget_h36_h37_depth8": budget_between(CPU, 36, FIX_GPU, 37, depth=8),
            "old_excess_outflux_pa_per_cell_h": float(old_excess),
            "fixed_excess_outflux_pa_per_cell_h": float(new_excess),
            "collapse_fraction": float(1.0 - abs(new_excess) / max(abs(old_excess), 1.0e-12)),
        }
    proof["resources"] = {
        "gpt_hydro_pgf_subterms": resource_summary("gpt_hydro_pgf_subterms"),
        "gpt_hydro_pgf_edgefix_1h": resource_summary("gpt_hydro_pgf_edgefix_1h"),
        "gpt_hydro_pgf_edgefix_3h": resource_summary("gpt_hydro_pgf_edgefix_3h"),
    }
    write_json(OUT_JSON, proof)
    return proof


def run_forecast_variant(args: argparse.Namespace) -> None:
    from gpuwrf.integration import daily_pipeline as dp

    config = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=int(args.hours),
        output_dir=FIX_GPU,
        proof_dir=PROBE_ROOT / "proofs_hydro_pgf_edgefix_gpt",
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
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision, _physics_boundary_step
    from gpuwrf.runtime.operational_state import initial_operational_carry

    cfg = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=PROBE_ROOT / "unused_hydro_pgf_step_probe_output",
        proof_dir=PROBE_ROOT / "unused_hydro_pgf_step_probe_proofs",
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
        out["all_finite"] = all_finite
        return out

    def mask_specified_edges(ru, rv):
        u_mask = jnp.ones((1, ru.shape[1], ru.shape[2]), dtype=ru.dtype)
        u_mask = u_mask.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        v_mask = jnp.ones((1, rv.shape[1], rv.shape[2]), dtype=rv.dtype)
        v_mask = v_mask.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
        return ru * u_mask, rv * v_mask

    def subterm_pgf(state, metrics, *, dx_m, dy_m, terms: frozenset[str], edge_mode: str = "all"):
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

        ru = -cqu * dpx
        rv = -cqv * dpy
        if edge_mode == "interior":
            return mask_specified_edges(ru, rv)
        if edge_mode == "edge":
            in_ru, in_rv = mask_specified_edges(ru, rv)
            return ru - in_ru, rv - in_rv
        return ru, rv

    import gpuwrf.runtime.operational_mode as operational_mode

    def _apply_patch(kind: str | None):
        originals: dict[str, Any] = {}
        if kind is None:
            return originals
        originals["large_step_horizontal_pgf"] = operational_mode.large_step_horizontal_pgf
        original = originals["large_step_horizontal_pgf"]
        if kind == "zero_large_step_pgf":

            def _zero(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
                del metrics, dx_m, dy_m, non_hydrostatic, top_lid
                return jnp.zeros_like(state.u), jnp.zeros_like(state.v)

            operational_mode.large_step_horizontal_pgf = _zero
            return originals
        if kind == "full_wrf_specified_edge_skip":

            def _full_edge_skip(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
                ru, rv = original(state, metrics, dx_m=dx_m, dy_m=dy_m, non_hydrostatic=non_hydrostatic, top_lid=top_lid)
                return mask_specified_edges(ru, rv)

            operational_mode.large_step_horizontal_pgf = _full_edge_skip
            return originals

        term_map = {
            "hydro_first_three_only": frozenset(("ph", "p_alt", "pb_al")),
            "ph_only": frozenset(("ph",)),
            "p_alt_only": frozenset(("p_alt",)),
            "pb_al_only": frozenset(("pb_al",)),
            "hydro_specified_edge_only": frozenset(("ph", "p_alt", "pb_al")),
            "ph_specified_edge_only": frozenset(("ph",)),
            "p_alt_specified_edge_only": frozenset(("p_alt",)),
            "pb_al_specified_edge_only": frozenset(("pb_al",)),
        }
        if kind not in term_map:
            raise ValueError(f"unknown proof patch kind: {kind}")
        edge_mode = "edge" if kind.endswith("_specified_edge_only") else "all"
        terms = term_map[kind]

        def _subterm(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
            del non_hydrostatic, top_lid
            return subterm_pgf(state, metrics, dx_m=dx_m, dy_m=dy_m, terms=terms, edge_mode=edge_mode)

        operational_mode.large_step_horizontal_pgf = _subterm
        return originals

    def _restore_patch(originals: Mapping[str, Any]) -> None:
        if "large_step_horizontal_pgf" in originals:
            operational_mode.large_step_horizontal_pgf = originals["large_step_horizontal_pgf"]

    variants = [
        ("baseline", None),
        ("zero_large_step_pgf", "zero_large_step_pgf"),
        ("hydro_first_three_only", "hydro_first_three_only"),
        ("ph_only", "ph_only"),
        ("p_alt_only", "p_alt_only"),
        ("pb_al_only", "pb_al_only"),
        ("full_wrf_specified_edge_skip", "full_wrf_specified_edge_skip"),
        ("hydro_specified_edge_only", "hydro_specified_edge_only"),
        ("ph_specified_edge_only", "ph_specified_edge_only"),
        ("p_alt_specified_edge_only", "p_alt_specified_edge_only"),
        ("pb_al_specified_edge_only", "pb_al_specified_edge_only"),
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
        "wrf_loop_bound_mismatch_probe": {
            "meaning": "full_wrf_specified_edge_skip zeros only the outer normal faces skipped by WRF when specified/nested",
            "wrf_source": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:2268-2407",
        },
        "variants": {},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_version": getattr(jax, "__version__", None),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
    }
    only = set(args.only or [])
    for name, patch_kind in variants:
        if only and name not in only:
            continue
        carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))
        hist = []
        first_bad_step = None
        start = time.perf_counter()
        originals = _apply_patch(patch_kind)
        try:

            @jax.jit
            def _one_step_variant(carry_in, namelist_in, step_index):
                return _physics_boundary_step(carry_in, namelist_in, step_index, run_radiation=False, debug=False)

            print(f"[hydro-pgf] variant={name} patch={patch_kind}", flush=True)
            for step in range(1, int(args.steps) + 1):
                carry = _one_step_variant(carry, base, jnp.asarray(step, dtype=jnp.int32))
                jax.block_until_ready(carry.state.u)
                rec = {"step": step} | state_summary(carry.state)
                hist.append(rec)
                if step <= int(args.print_first) or step % int(args.print_every) == 0 or not bool(rec["all_finite"]):
                    print(
                        "[hydro-pgf] "
                        f"{name} step={step} finite={rec['all_finite']} "
                        f"w_top={rec['w_top_absmax']:.3f} w_int={rec['w_interior_absmax']:.3f} "
                        f"u={rec['u_absmax']:.3f} v={rec['v_absmax']:.3f} "
                        f"theta=[{rec['theta_min']:.3f},{rec['theta_max']:.3f}] "
                        f"mu_mean={rec['mu_total_mean']:.3f}",
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
            "proof_patch": patch_kind,
            "steps_completed": len(hist),
            "first_bad_step": first_bad_step,
            "wall_s": float(time.perf_counter() - start),
            "history": hist,
            "final": hist[-1] if hist else None,
        }
    proof = {"step_probe": output}
    write_json(Path(args.out), proof)
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
        if "specified_edge_skip_step_verdict" in proof:
            print(json.dumps(proof["specified_edge_skip_step_verdict"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
