#!/usr/bin/env python
"""Targeted M6b V3 20260521 wind-bound localizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np

from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    _physics_boundary_step,
    _rk_scan_step,
    run_forecast_operational,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry

from diagnostic_operator_term_budget_tracer import build_payload as build_term_budget_payload


config.update("jax_enable_x64", True)

RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")
RUN_ID = "20260521_18z_l3_24h_20260522T072630Z"
LOWER_LEVELS = 30
UPPER_LEVELS = 14
WIND_LIMITS = {"u_abs_max_m_s": 100.0, "v_abs_max_m_s": 100.0, "w_abs_max_m_s": 50.0}
BAD_WINDOW = range(40, 47)
DT_S = 10.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_scalar(value: Any) -> float | int | str | None:
    if value is None:
        return None
    try:
        item = np.asarray(value).item()
    except ValueError:
        return str(value)
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, (int, np.integer)):
        return int(item)
    if isinstance(item, (float, np.floating)):
        return float(item) if np.isfinite(float(item)) else str(float(item))
    return str(item)


def _clamp_index(index: int, size: int) -> int:
    return max(0, min(int(index), int(size) - 1))


def _load_case(run_id: str):
    run_dir = RUN_ROOT / run_id
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=DT_S,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
    carry = initial_operational_carry(_enforce_operational_precision(state))
    meta = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "grid": case.metadata["grid"],
        "namelist": {
            "dt_s": namelist.dt_s,
            "acoustic_substeps": namelist.acoustic_substeps,
            "run_physics": namelist.run_physics,
            "run_boundary": namelist.run_boundary,
            "use_vertical_solver": namelist.use_vertical_solver,
            "radiation_cadence_steps": namelist.radiation_cadence_steps,
        },
    }
    return case, namelist, carry, meta


def _all_leaves_finite(state: Any) -> bool:
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]
    return bool(np.asarray(jnp.all(jnp.asarray(checks))))


def _bounds_for_state(state: Any, *, step: int) -> dict[str, Any]:
    theta = state.theta
    lower_theta = theta[:LOWER_LEVELS, :, :]
    upper_theta = theta[LOWER_LEVELS:, :, :]
    values = {
        "step": int(step),
        "lead_seconds": float(step) * DT_S,
        "all_leaves_finite": _all_leaves_finite(state),
        "theta_full_min_k": float(np.asarray(jnp.min(theta))),
        "theta_full_max_k": float(np.asarray(jnp.max(theta))),
        "theta_lower_30_min_k": float(np.asarray(jnp.min(lower_theta))),
        "theta_lower_30_max_k": float(np.asarray(jnp.max(lower_theta))),
        "theta_upper_14_min_k": float(np.asarray(jnp.min(upper_theta))),
        "theta_upper_14_max_k": float(np.asarray(jnp.max(upper_theta))),
        "u_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.u)))),
        "v_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.v)))),
        "w_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.w)))),
    }
    lower_ok = 200.0 <= values["theta_lower_30_min_k"] and values["theta_lower_30_max_k"] <= 400.0
    upper_ok = 250.0 <= values["theta_upper_14_min_k"] and values["theta_upper_14_max_k"] <= 700.0
    wind_ok = (
        values["u_abs_max_m_s"] <= WIND_LIMITS["u_abs_max_m_s"]
        and values["v_abs_max_m_s"] <= WIND_LIMITS["v_abs_max_m_s"]
        and values["w_abs_max_m_s"] <= WIND_LIMITS["w_abs_max_m_s"]
    )
    values.update(
        {
            "theta_lower_30_bounded": bool(lower_ok),
            "theta_upper_14_bounded": bool(upper_ok),
            "wind_bounded": bool(wind_ok),
            "passed": bool(values["all_leaves_finite"] and lower_ok and upper_ok and wind_ok),
        }
    )
    return values


def _v_max_location(state: Any) -> tuple[tuple[int, int, int], float]:
    arr = np.asarray(jax.device_get(state.v))
    idx = tuple(int(item) for item in np.unravel_index(np.nanargmax(np.abs(arr)), arr.shape))
    return idx, float(arr[idx])


def _sample_array(array: Any, index: tuple[int, ...]) -> dict[str, Any]:
    arr = np.asarray(jax.device_get(array))
    idx = tuple(_clamp_index(index[pos], arr.shape[pos]) for pos in range(arr.ndim))
    return {"index": list(idx), "value": _json_scalar(arr[idx]), "shape": list(arr.shape)}


def _snapshot_at(carry: OperationalCarry, loc: tuple[int, int, int]) -> dict[str, Any]:
    k, j_v, i = loc
    state = carry.state
    j_mass = _clamp_index(j_v, state.theta.shape[1])
    i_mass = _clamp_index(i, state.theta.shape[2])
    return {
        "theta": _sample_array(state.theta, (k, j_mass, i_mass)),
        "mu": _sample_array(state.mu, (j_mass, i_mass)),
        "u": _sample_array(state.u, (k, j_mass, i)),
        "v": _sample_array(state.v, (k, j_v, i)),
        "w": _sample_array(state.w, (k, j_mass, i_mass)),
        "ww": _sample_array(carry.ww, (k, j_mass, i_mass)),
        "stagger_mapping": {
            "v_index_k_j_i": list(loc),
            "mass_j_i_used_for_scalar_fields": [j_mass, i_mass],
        },
    }


def _metric_tuple(carry: OperationalCarry, step: Any) -> tuple[Any, ...]:
    state = carry.state
    theta = state.theta
    lower_theta = theta[:LOWER_LEVELS, :, :]
    upper_theta = theta[LOWER_LEVELS:, :, :]
    return (
        step,
        jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)])),
        jnp.min(theta),
        jnp.max(theta),
        jnp.min(lower_theta),
        jnp.max(lower_theta),
        jnp.min(upper_theta),
        jnp.max(upper_theta),
        jnp.max(jnp.abs(state.u)),
        jnp.max(jnp.abs(state.v)),
        jnp.max(jnp.abs(state.w)),
    )


@partial(jax.jit, static_argnames=("steps",))
def _run_carry_scan(state: Any, namelist: OperationalNamelist, *, steps: int):
    current = initial_operational_carry(_enforce_operational_precision(state))
    captured45 = current
    indices = jnp.arange(1, int(steps) + 1, dtype=jnp.int32)

    def body(bundle, step_index):
        scan_current, scan_capture = bundle
        next_carry = _physics_boundary_step(scan_current, namelist, step_index, run_radiation=False)
        next_capture = jax.tree_util.tree_map(
            lambda old, new: jnp.where(step_index == 45, new, old),
            scan_capture,
            next_carry,
        )
        return (next_carry, next_capture), _metric_tuple(next_carry, step_index)

    (final, capture), metrics = jax.lax.scan(body, (current, captured45), indices)
    return final, capture, metrics


@partial(jax.jit, static_argnames=("steps", "loc"))
def _run_v_loc_scan(state: Any, namelist: OperationalNamelist, *, steps: int, loc: tuple[int, int, int]):
    k, j, i = loc
    current = initial_operational_carry(_enforce_operational_precision(state))
    indices = jnp.arange(1, int(steps) + 1, dtype=jnp.int32)

    def body(scan_current, step_index):
        next_carry = _physics_boundary_step(scan_current, namelist, step_index, run_radiation=False)
        v = next_carry.state.v
        row = (v[int(k), int(j), int(i)], jnp.max(jnp.abs(v)))
        return next_carry, row

    _final, rows = jax.lax.scan(body, current, indices)
    return rows


def _run_until_violation(carry: OperationalCarry, namelist: OperationalNamelist) -> tuple[OperationalCarry, dict[str, Any], dict[int, OperationalCarry]]:
    step_hours = DT_S / 3600.0
    start = time.perf_counter()
    current_state = carry.state
    per_step: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    step45_state = None
    for step in range(1, 47):
        current_state = run_forecast_operational(current_state, namelist, step_hours)
        block_until_ready(current_state)
        row = _bounds_for_state(current_state, step=step)
        loc, value = _v_max_location(current_state)
        row["v_abs_max_location_k_j_i"] = list(loc)
        row["v_value_at_abs_max_m_s"] = value
        per_step.append(row)
        if step == 45:
            step45_state = jax.tree_util.tree_map(lambda leaf: jnp.array(leaf), current_state)
            block_until_ready(step45_state)
        if not row["passed"]:
            first_bad = row
            break
    elapsed = time.perf_counter() - start
    if first_bad is None:
        first_bad = per_step[-1] if per_step else {"step": None}
    if step45_state is None:
        raise RuntimeError("step 45 state was not captured")
    final_carry = initial_operational_carry(current_state)
    step45_carry = initial_operational_carry(step45_state)
    carries: dict[int, OperationalCarry] = {45: step45_carry, int(first_bad["step"]): final_carry}
    return final_carry, {"per_step": per_step, "first_bad_step": first_bad, "wall_time_s": elapsed}, carries


def _read_wrf(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        return {name: np.asarray(np.ma.filled(ds.variables[name][0], np.nan)) for name in fields}


def _wrf_path(run_dir: Path, lead_hour: int) -> Path:
    files = sorted(run_dir.glob("wrfout_d02_*"))
    if lead_hour >= len(files):
        raise FileNotFoundError(f"{run_dir} has no d02 wrfout at lead hour {lead_hour}")
    return files[lead_hour]


def _wrf_reference_compare(run_dir: Path, loc: tuple[int, int, int], lead_seconds: float) -> dict[str, Any]:
    k, j_v, i = loc
    lower_hour = int(np.floor(float(lead_seconds) / 3600.0))
    upper_hour = min(lower_hour + 1, len(sorted(run_dir.glob("wrfout_d02_*"))) - 1)
    fraction = 0.0 if upper_hour == lower_hour else (float(lead_seconds) - 3600.0 * lower_hour) / 3600.0
    lower_path = _wrf_path(run_dir, lower_hour)
    upper_path = _wrf_path(run_dir, upper_hour)
    lower = _read_wrf(lower_path, ("V", "W", "XLAT_V", "XLONG_V"))
    upper = _read_wrf(upper_path, ("V", "W", "XLAT_V", "XLONG_V"))
    v_interp = (1.0 - fraction) * lower["V"] + fraction * upper["V"]
    w_interp = (1.0 - fraction) * lower["W"] + fraction * upper["W"]

    k_c = _clamp_index(k, v_interp.shape[0])
    j_c = _clamp_index(j_v, v_interp.shape[1])
    i_c = _clamp_index(i, v_interp.shape[2])
    patch = v_interp[k_c, max(0, j_c - 2) : j_c + 3, max(0, i_c - 2) : i_c + 3]
    v_any = float(np.nanmax(np.abs(v_interp)))
    w_any = float(np.nanmax(np.abs(w_interp)))
    near_max = float(np.nanmax(np.abs(patch)))
    value_at_cell = float(v_interp[k_c, j_c, i_c])
    physical = bool(max(v_any, near_max, abs(value_at_cell)) > 90.0)
    return {
        "status": "PHYSICAL" if physical else "NAMED_FIX_REQUIRED",
        "run_dir": str(run_dir),
        "lead_seconds": float(lead_seconds),
        "lead_hour_bracket": [lower_hour, upper_hour],
        "linear_interpolation_fraction": float(fraction),
        "reference_files": [str(lower_path), str(upper_path)],
        "bad_cell_v_stagger_index_k_j_i": [k_c, j_c, i_c],
        "bad_cell_lat_lon": {
            "lat": float(lower["XLAT_V"][j_c, i_c]),
            "lon": float(lower["XLONG_V"][j_c, i_c]),
            "source": str(lower_path),
        },
        "wrf_reference_v_at_cell_m_s": value_at_cell,
        "wrf_reference_abs_v_at_cell_m_s": abs(value_at_cell),
        "wrf_reference_abs_v_max_same_level_nearby_radius2_m_s": near_max,
        "wrf_reference_abs_v_max_same_level_domain_m_s": float(np.nanmax(np.abs(v_interp[k_c]))),
        "wrf_reference_vertical_max_abs_v_anywhere_domain_m_s": v_any,
        "wrf_reference_vertical_max_abs_w_anywhere_domain_m_s": w_any,
        "physical_threshold_m_s": 90.0,
        "recommendation": "BOUND-REVISION" if physical else "NAMED-FIX",
    }


def _manual_stage_decomposition(
    carry_before: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    step: int,
    loc: tuple[int, int, int],
    output_dir: Path,
) -> dict[str, Any]:
    k, j, i = loc
    dt = float(namelist.dt_s)
    start_state = carry_before.state
    after_rk = _rk_scan_step(carry_before, namelist)
    block_until_ready(after_rk)
    physical_origin = carry_before.state
    projected_state = after_rk.state.replace(
        theta=physical_origin.theta,
        mu=physical_origin.mu,
        mu_total=physical_origin.mu_total,
        mu_perturbation=physical_origin.mu_perturbation,
    )
    before_physics = projected_state
    after_thompson = thompson_adapter(before_physics, dt)
    after_mynn = mynn_adapter(after_thompson, dt, namelist.grid)
    after_surface = surface_adapter(after_mynn, dt)
    after_radiation = after_surface
    if False:
        after_radiation = rrtmg_adapter(after_surface, dt, namelist.grid)
    before_boundary = after_radiation
    lead_seconds = jnp.asarray(step, dtype=jnp.float64) * dt
    after_boundary = apply_lateral_boundaries(before_boundary, lead_seconds, dt, namelist.boundary_config)
    after_precision = _enforce_operational_precision(after_boundary)
    block_until_ready(after_precision)

    def val(state: Any) -> float:
        arr = np.asarray(jax.device_get(state.v))
        return float(arr[_clamp_index(k, arr.shape[0]), _clamp_index(j, arr.shape[1]), _clamp_index(i, arr.shape[2])])

    values = {
        "start": val(start_state),
        "after_rk_acoustic": val(after_rk.state),
        "after_projection": val(projected_state),
        "after_thompson": val(after_thompson),
        "after_mynn": val(after_mynn),
        "after_surface": val(after_surface),
        "after_boundary": val(after_boundary),
        "after_precision": val(after_precision),
    }
    terms = {
        "dycore_rk_acoustic": [(values["after_rk_acoustic"] - values["start"]) / dt],
        "carry_fix_projection": [(values["after_projection"] - values["after_rk_acoustic"]) / dt],
        "thompson": [(values["after_thompson"] - values["after_projection"]) / dt],
        "mynn": [(values["after_mynn"] - values["after_thompson"]) / dt],
        "surface": [(values["after_surface"] - values["after_mynn"]) / dt],
        "rk_acoustic": [(values["after_rk_acoustic"] - values["start"]) / dt],
        "boundary_application": [(values["after_boundary"] - values["after_surface"]) / dt],
        "precision_cast": [(values["after_precision"] - values["after_boundary"]) / dt],
    }
    input_path = output_dir / "proof_operator_decomposition_input.json"
    _write_json(
        input_path,
        {
            "terms": terms,
            "values_m_s": values,
            "cell": {"v_stagger_index_k_j_i": list(loc), "step": int(step), "dt_s": dt},
        },
    )
    budget = build_term_budget_payload(input_path)
    ranking = budget["measurements"]["ranking"]
    dominant = ranking[0]["term"] if ranking else "UNKNOWN"
    named_operator = "boundary_application" if dominant == "boundary_application" else dominant
    payload = {
        "artifact_type": "m6b_v3_521_operator_decomposition",
        "status": "OK",
        "diagnostic_tool": "diagnostic_operator_term_budget_tracer.py",
        "diagnostic_input": str(input_path),
        "bad_cell": {"v_stagger_index_k_j_i": list(loc), "step": int(step)},
        "values_m_s": values,
        "terms_per_second": terms,
        "term_budget": budget,
        "dominant_term": dominant,
        "named_fix_operator": named_operator,
        "cross_check": {
            "wrf_source_expected_boundary_order": "solve_em.F:2034-2285 applies specified lateral-boundary tendencies during the coupled step.",
            "wrf_source_expected_acoustic_loop": "solve_em.F:3065-4363 runs acoustic small steps inside RK stages.",
            "project_source": "src/gpuwrf/runtime/operational_mode.py:_physics_boundary_step",
            "note": "The V3 operational wrapper does not emit separate pressure-gradient/Coriolis/vertical-advection arrays; this proof localizes the observed dv/dt to the available WRF-ordered stage boundary.",
        },
    }
    return payload


def _load_v_interp(run_dir: Path, lead_seconds: float) -> np.ndarray:
    files = sorted(run_dir.glob("wrfout_d02_*"))
    lower_hour = int(np.floor(float(lead_seconds) / 3600.0))
    upper_hour = min(lower_hour + 1, len(files) - 1)
    fraction = 0.0 if upper_hour == lower_hour else (float(lead_seconds) - 3600.0 * lower_hour) / 3600.0
    lower = _read_wrf(files[lower_hour], ("V",))["V"]
    upper = _read_wrf(files[upper_hour], ("V",))["V"]
    return (1.0 - fraction) * lower + fraction * upper


def _first_divergent_step(
    run_dir: Path,
    state0: Any,
    namelist: OperationalNamelist,
    loc: tuple[int, int, int],
) -> dict[str, Any]:
    threshold = 10.0
    rows: list[dict[str, Any]] = []
    earliest = None
    k, j, i = loc
    state = state0
    step_hours = DT_S / 3600.0
    gpu_rows: dict[int, tuple[float, float]] = {}
    for step in range(1, 47):
        state = run_forecast_operational(state, namelist, step_hours)
        block_until_ready(state)
        if step in BAD_WINDOW:
            v = np.asarray(jax.device_get(state.v))
            gpu_rows[step] = (float(v[int(k), int(j), int(i)]), float(np.nanmax(np.abs(v))))
    for step in BAD_WINDOW:
        wrf_v = _load_v_interp(run_dir, float(step) * DT_S)
        kc = _clamp_index(k, wrf_v.shape[0])
        jc = _clamp_index(j, wrf_v.shape[1])
        ic = _clamp_index(i, wrf_v.shape[2])
        gpu_cell, gpu_max = gpu_rows[step]
        wrf_cell = float(wrf_v[kc, jc, ic])
        row = {
            "step": int(step),
            "lead_seconds": float(step) * DT_S,
            "gpu_v_at_bad_cell_m_s": gpu_cell,
            "wrf_interp_v_at_bad_cell_m_s": wrf_cell,
            "delta_at_bad_cell_m_s": gpu_cell - wrf_cell,
            "max_abs_v_delta_domain_m_s": None,
            "gpu_abs_v_max_domain_m_s": gpu_max,
            "wrf_abs_v_max_domain_m_s": float(np.nanmax(np.abs(wrf_v))),
        }
        row["detectable"] = bool(abs(row["delta_at_bad_cell_m_s"]) > threshold)
        rows.append(row)
        if earliest is None and row["detectable"]:
            earliest = int(step)
    return {
        "artifact_type": "m6b_v3_521_first_divergent_step",
        "status": "OK",
        "window_steps": [min(BAD_WINDOW), max(BAD_WINDOW)],
        "detectable_threshold_m_s": threshold,
        "earliest_detectable_divergence_step_in_window": earliest,
        "bad_cell": {"v_stagger_index_k_j_i": list(loc)},
        "method": "GPU V at the bad V-stagger cell compared against linear interpolation of hourly Gen2 wrfout V at each 10 s lead.",
        "domain_delta_note": "Domain-wide delta is not emitted to avoid materializing seven full V states; domain max |V| is reported as context.",
        "per_step": rows,
    }


def _write_memo(output_dir: Path, verdict: str, step_payload: dict[str, Any], wrf_payload: dict[str, Any], op_payload: dict[str, Any], div_payload: dict[str, Any]) -> None:
    if verdict.startswith("NAMED-FIX"):
        operator = verdict.split(":", 1)[1] if ":" in verdict else "operator"
        next_sprint = f"2026-05-25-m6b-v3-{operator.replace('_', '-')}-fix"
        next_scope = f"Fix and validate the `{operator}` V tendency that dominates the 20260521 step-46 acceleration."
    elif verdict == "BOUND-REVISION":
        next_sprint = "2026-05-25-m6b-v3-wind-bound-revision"
        next_scope = "Raise the operational V bound to 120 m/s with WRF-reference evidence."
    else:
        next_sprint = "2026-05-25-m6b-v3-localize-521-followup"
        next_scope = "Fill the missing evidence gap for the 20260521 step-46 wind-bound localization."
    lines = [
        "# Localization Memo",
        "",
        f"## Verdict: {verdict}",
        "",
        "## Evidence summary",
        f"- `proof_step46_violation.json`: first failed step {step_payload['first_bad_step']['step']} at lead {step_payload['first_bad_step']['lead_seconds']} s with |V|max {step_payload['bad_cell']['max_abs_v_m_s']:.6f} m/s.",
        f"- `proof_step46_violation.json`: bad V-stagger cell is {step_payload['bad_cell']['v_stagger_index_k_j_i']} with snapshots for theta, mu, u, v, w, and ww at steps 45 and 46.",
        f"- `proof_wrf_reference_compare.json`: Gen2 WRF interpolated V at that cell is {wrf_payload['wrf_reference_v_at_cell_m_s']:.6f} m/s.",
        f"- `proof_wrf_reference_compare.json`: Gen2 WRF same-level nearby max |V| is {wrf_payload['wrf_reference_abs_v_max_same_level_nearby_radius2_m_s']:.6f} m/s; domain vertical max |V| is {wrf_payload['wrf_reference_vertical_max_abs_v_anywhere_domain_m_s']:.6f} m/s.",
        f"- `proof_operator_decomposition.json`: dominant available WRF-ordered stage term is `{op_payload.get('dominant_term')}`.",
        f"- `proof_first_divergent_step.json`: earliest detectable divergence in the step-40..46 window is step {div_payload.get('earliest_detectable_divergence_step_in_window')}.",
        "",
        "## Recommended next sprint",
        f"- `{next_sprint}`: {next_scope}",
        "",
        "## Risks / caveats",
        "- WRF history is hourly; the proof uses linear interpolation for the 10 s step-46 valid time.",
        "- The V3 operational wrapper does not expose separate pressure-gradient, Coriolis, and vertical-advection term arrays, so Stage 3 localizes to the available WRF-ordered operational stages.",
        "- GPU-vs-CPU parity risk remains linked to the sister gpu-cpu-step2 sprint because this localizer runs the JAX operational path only.",
        "",
    ]
    (output_dir / "localization_memo.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"this sprint is pinned to {RUN_ID}, got {args.run_id}")
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    case, namelist, carry0, meta = _load_case(args.run_id)
    bad_carry, timeline, carries = _run_until_violation(carry0, namelist)
    first_bad = timeline["first_bad_step"]
    bad_loc, bad_v = _v_max_location(bad_carry.state)
    step45 = carries.get(45)
    if step45 is None:
        raise RuntimeError("step 45 carry was not captured")
    step_payload = {
        "artifact_type": "m6b_v3_521_step46_violation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": visible_gpu_name(),
        **meta,
        "bounds_policy": WIND_LIMITS,
        "first_bad_step": first_bad,
        "bad_cell": {
            "v_stagger_index_k_j_i": list(bad_loc),
            "max_abs_v_m_s": abs(bad_v),
            "signed_v_m_s": bad_v,
        },
        "field_snapshots": {
            "step45": _snapshot_at(step45, bad_loc),
            "step46": _snapshot_at(bad_carry, bad_loc),
        },
        "timeline": timeline,
    }
    _write_json(output_dir / "proof_step46_violation.json", step_payload)

    wrf_payload = _wrf_reference_compare(Path(meta["run_dir"]), bad_loc, float(first_bad["lead_seconds"]))
    wrf_payload["artifact_type"] = "m6b_v3_521_wrf_reference_compare"
    _write_json(output_dir / "proof_wrf_reference_compare.json", wrf_payload)

    if wrf_payload["recommendation"] == "BOUND-REVISION":
        op_payload = {
            "artifact_type": "m6b_v3_521_operator_decomposition",
            "status": "NOT_RUN",
            "reason": "Stage 2 classified the event as PHYSICAL.",
        }
        div_payload = {
            "artifact_type": "m6b_v3_521_first_divergent_step",
            "status": "NOT_RUN",
            "reason": "Stage 2 classified the event as PHYSICAL.",
        }
        verdict = "BOUND-REVISION"
    else:
        step46_start = carries.get(45)
        if step46_start is None:
            raise RuntimeError("step 46 start carry missing")
        op_payload = _manual_stage_decomposition(step46_start, namelist, step=46, loc=bad_loc, output_dir=output_dir)
        _fresh_case, fresh_namelist, fresh_carry0, _fresh_meta = _load_case(args.run_id)
        div_payload = _first_divergent_step(Path(meta["run_dir"]), fresh_carry0.state, fresh_namelist, bad_loc)
        verdict = f"NAMED-FIX:{op_payload['named_fix_operator']}"
    _write_json(output_dir / "proof_operator_decomposition.json", op_payload)
    _write_json(output_dir / "proof_first_divergent_step.json", div_payload)
    _write_memo(output_dir, verdict, step_payload, wrf_payload, op_payload, div_payload)

    summary = {
        "artifact_type": "m6b_v3_521_localization_summary",
        "status": "OK",
        "verdict": verdict,
        "proofs": [
            str(output_dir / "proof_step46_violation.json"),
            str(output_dir / "proof_wrf_reference_compare.json"),
            str(output_dir / "proof_operator_decomposition.json"),
            str(output_dir / "proof_first_divergent_step.json"),
            str(output_dir / "localization_memo.md"),
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
