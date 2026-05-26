#!/usr/bin/env python
"""Localize the 20260509 V3 theta-bound failure against Gen2 wrfout truth."""

from __future__ import annotations

import argparse
import importlib.util
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)

RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
PINNED_RUN_ID = "20260509_18z_l3_24h_20260511T190519Z"
LOWER_LEVELS = 30
UPPER_LEVELS = 14
LOWER_THETA_BOUNDS_K = (200.0, 400.0)
UPPER_THETA_BOUNDS_K = (250.0, 700.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray,)):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else str(scalar)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _case_state_and_namelist(run_id: str) -> tuple[Any, OperationalNamelist, Any, dict[str, Any]]:
    run_dir = RUN_ROOT / run_id
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
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
    return state, namelist, case, meta


def _state_after_steps(run_id: str, steps: int) -> tuple[Any, OperationalNamelist, Any]:
    state, namelist, case, _meta = _case_state_and_namelist(run_id)
    current = state
    step_hours = float(namelist.dt_s) / 3600.0
    for _ in range(int(steps)):
        current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
    return current, namelist, case


def _bounds_for_level(level: int) -> tuple[float, float]:
    return LOWER_THETA_BOUNDS_K if int(level) < LOWER_LEVELS else UPPER_THETA_BOUNDS_K


def _per_level_theta(theta: Any, *, step: int, lead_seconds: float) -> dict[str, Any]:
    arr = np.asarray(jax.device_get(theta), dtype=np.float64)
    levels = []
    for level in range(arr.shape[0]):
        lower, upper = _bounds_for_level(level)
        values = arr[level]
        level_min = float(np.nanmin(values))
        level_max = float(np.nanmax(values))
        levels.append(
            {
                "level": int(level),
                "min_k": level_min,
                "max_k": level_max,
                "lower_bound_k": lower,
                "upper_bound_k": upper,
                "bounded": bool(np.isfinite(level_min) and np.isfinite(level_max) and lower <= level_min and level_max <= upper),
            }
        )
    return {"step": int(step), "lead_seconds": float(lead_seconds), "levels": levels}


def _first_theta_violation(theta: Any, *, step: int, lead_seconds: float) -> dict[str, Any] | None:
    arr = np.asarray(jax.device_get(theta), dtype=np.float64)
    best: dict[str, Any] | None = None
    for level in range(arr.shape[0]):
        lower, upper = _bounds_for_level(level)
        values = arr[level]
        finite = np.isfinite(values)
        low_mask = finite & (values < lower)
        high_mask = finite & (values > upper)
        for kind, mask, bound in (("below_lower", low_mask, lower), ("above_upper", high_mask, upper)):
            if not np.any(mask):
                continue
            candidates = np.where(mask, np.abs(values - bound), -np.inf)
            j, i = np.unravel_index(int(np.nanargmax(candidates)), values.shape)
            exceedance = float(abs(values[j, i] - bound))
            row = {
                "step": int(step),
                "lead_seconds": float(lead_seconds),
                "level": int(level),
                "cell": {"i": int(i), "j": int(j), "k": int(level)},
                "kind": kind,
                "value_k": float(values[j, i]),
                "bound_k": float(bound),
                "exceedance_k": exceedance,
            }
            if best is None or exceedance > float(best["exceedance_k"]):
                best = row
    if best is not None:
        return best
    nonfinite = ~np.isfinite(arr)
    if np.any(nonfinite):
        k, j, i = np.argwhere(nonfinite)[0]
        return {
            "step": int(step),
            "lead_seconds": float(lead_seconds),
            "level": int(k),
            "cell": {"i": int(i), "j": int(j), "k": int(k)},
            "kind": "nonfinite",
            "value_k": str(arr[k, j, i]),
            "bound_k": None,
            "exceedance_k": None,
        }
    return None


def _array_value_at(array: Any, k: int, j: int, i: int) -> Any:
    arr = np.asarray(jax.device_get(array))
    if arr.ndim == 3:
        kk = min(max(int(k), 0), arr.shape[0] - 1)
        jj = min(max(int(j), 0), arr.shape[1] - 1)
        ii = min(max(int(i), 0), arr.shape[2] - 1)
        return _jsonable(arr[kk, jj, ii])
    if arr.ndim == 2:
        jj = min(max(int(j), 0), arr.shape[0] - 1)
        ii = min(max(int(i), 0), arr.shape[1] - 1)
        return _jsonable(arr[jj, ii])
    return None


def _state_snapshot(state: Any, cell: dict[str, int]) -> dict[str, Any]:
    k, j, i = int(cell["k"]), int(cell["j"]), int(cell["i"])
    fields = (
        "theta",
        "u",
        "v",
        "w",
        "qv",
        "p",
        "p_total",
        "p_perturbation",
        "ph",
        "ph_total",
        "mu",
        "mu_total",
        "mu_perturbation",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "t_skin",
    )
    return {field: _array_value_at(getattr(state, field), k, j, i) for field in fields if hasattr(state, field)}


def _column_profiles(state: Any, cell: dict[str, int]) -> dict[str, list[Any]]:
    j, i = int(cell["j"]), int(cell["i"])
    profiles = {}
    for field in ("theta", "w", "p", "p_total", "qv"):
        arr = np.asarray(jax.device_get(getattr(state, field)), dtype=np.float64)
        jj = min(max(j, 0), arr.shape[1] - 1)
        ii = min(max(i, 0), arr.shape[2] - 1)
        profiles[field] = [_jsonable(item) for item in arr[:, jj, ii].tolist()]
    mu = np.asarray(jax.device_get(state.mu), dtype=np.float64)
    profiles["mu_column_repeated"] = [float(mu[min(max(j, 0), mu.shape[0] - 1), min(max(i, 0), mu.shape[1] - 1)])]
    return profiles


def run_stage1(run_id: str, output: Path) -> tuple[dict[str, Any], Any, Any, OperationalNamelist, Any]:
    state, namelist, case, meta = _case_state_and_namelist(run_id)
    current = state
    timeline: list[dict[str, Any]] = []
    first_violation = None
    target_steps = int(round(3600.0 / float(namelist.dt_s)))
    step_hours = float(namelist.dt_s) / 3600.0
    start = time.perf_counter()
    for step in range(1, target_steps + 1):
        previous = current
        current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
        lead_seconds = step * float(namelist.dt_s)
        timeline.append(_per_level_theta(current.theta, step=step, lead_seconds=lead_seconds))
        first_violation = _first_theta_violation(current.theta, step=step, lead_seconds=lead_seconds)
        if first_violation is not None:
            break
    wall_s = time.perf_counter() - start
    cell = first_violation["cell"] if first_violation else {"i": 0, "j": 0, "k": 0}
    violation_step = int(first_violation["step"]) if first_violation else len(timeline)
    previous_for_snapshot, _previous_namelist, _previous_case = _state_after_steps(run_id, max(violation_step - 1, 0))
    current_for_snapshot, current_namelist, current_case = _state_after_steps(run_id, violation_step)
    proof = {
        "artifact_type": "m6b_v3_localize_20260509_theta_explosion",
        "status": "THETA_BOUND_VIOLATION" if first_violation else "NO_THETA_BOUND_VIOLATION_IN_1H",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": visible_gpu_name(),
        "input": meta,
        "target_steps": target_steps,
        "steps_completed": len(timeline),
        "wall_time_s": wall_s,
        "theta_bound_policy": {
            "lower_30_levels_k": list(LOWER_THETA_BOUNDS_K),
            "upper_14_levels_k": list(UPPER_THETA_BOUNDS_K),
        },
        "first_violation": first_violation,
        "per_step_theta_min_max_by_level": timeline,
        "field_snapshots": {
            "cell": cell,
            "step_n_minus_1": _state_snapshot(previous_for_snapshot, cell),
            "step_n": _state_snapshot(current_for_snapshot, cell),
        },
    }
    _write_json(output / "proof_theta_explosion.json", proof)
    return proof, previous_for_snapshot, current_for_snapshot, current_namelist, current_case


def _neighbor_slice(arr: np.ndarray, k: int, j: int, i: int, radius: int = 1) -> np.ndarray:
    if arr.ndim == 3:
        kk = min(max(int(k), 0), arr.shape[0] - 1)
        y0, y1 = max(int(j) - radius, 0), min(int(j) + radius + 1, arr.shape[1])
        x0, x1 = max(int(i) - radius, 0), min(int(i) + radius + 1, arr.shape[2])
        return arr[kk, y0:y1, x0:x1]
    y0, y1 = max(int(j) - radius, 0), min(int(j) + radius + 1, arr.shape[0])
    x0, x1 = max(int(i) - radius, 0), min(int(i) + radius + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def _stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
    }


def _sample_wrf_theta(run: Gen2Run, *, domain: str, time_index: int, k: int, j: int, i: int) -> float:
    theta = np.asarray(run.load(domain, "T", time=time_index, lazy=False), dtype=np.float64) + 300.0
    return float(theta[k, j, i])


def _interpolated_wrf_theta(run: Gen2Run, *, domain: str, lead_seconds: float, k: int, j: int, i: int) -> float:
    history = run.history_files(domain)
    lead_hours = max(float(lead_seconds) / 3600.0, 0.0)
    lo = min(int(np.floor(lead_hours)), len(history) - 1)
    hi = min(lo + 1, len(history) - 1)
    if hi == lo:
        return _sample_wrf_theta(run, domain=domain, time_index=lo, k=k, j=j, i=i)
    weight = lead_hours - float(lo)
    return (1.0 - weight) * _sample_wrf_theta(run, domain=domain, time_index=lo, k=k, j=j, i=i) + weight * _sample_wrf_theta(
        run, domain=domain, time_index=hi, k=k, j=j, i=i
    )


def run_stage2(run_id: str, stage1: dict[str, Any], output: Path) -> dict[str, Any]:
    run = Gen2Run(RUN_ROOT / run_id)
    violation = stage1.get("first_violation") or {}
    cell = violation.get("cell") or {"i": 0, "j": 0, "k": 0}
    i, j, k = int(cell["i"]), int(cell["j"]), int(cell["k"])
    lead_seconds = float(violation.get("lead_seconds") or 0.0)
    history = run.history_files("d02")
    time_axis = [item.isoformat() for item in run.time_axis("d02")]
    selected_index = min(max(int(round(lead_seconds / 3600.0)), 0), len(history) - 1)
    theta_series = []
    for index, path in enumerate(history):
        theta_value = _sample_wrf_theta(run, domain="d02", time_index=index, k=k, j=j, i=i)
        theta_series.append(
            {
                "time_index": int(index),
                "valid_time_utc": time_axis[index] if index < len(time_axis) else path.name[-19:],
                "theta_k": theta_value,
            }
        )

    theta_level = np.asarray(run.load("d02", "T", time=selected_index, lazy=False), dtype=np.float64)[k] + 300.0
    t = np.asarray(run.load("d02", "T", time=selected_index, lazy=False), dtype=np.float64) + 300.0
    qv = np.asarray(run.load("d02", "QVAPOR", time=selected_index, lazy=False), dtype=np.float64)
    w = np.asarray(run.load("d02", "W", time=selected_index, lazy=False), dtype=np.float64)
    lower, upper = _bounds_for_level(k)
    selected_theta = float(theta_level[j, i])
    horizontal_extremes = {
        "min_k": float(np.nanmin(theta_level)),
        "max_k": float(np.nanmax(theta_level)),
        "selected_cell_theta_k": selected_theta,
    }
    wrf_toward_bound = selected_theta < lower or selected_theta > upper or horizontal_extremes["min_k"] < lower or horizontal_extremes["max_k"] > upper
    proof = {
        "artifact_type": "m6b_v3_localize_wrf_reference_theta",
        "status": "PHYSICAL_SIGNAL_IN_REFERENCE" if wrf_toward_bound else "WRF_REFERENCE_BENIGN_AT_FAILURE_SITE",
        "run_id": run_id,
        "wrfout_count": len(history),
        "reference_time_resolution": "hourly_wrfout; sub-hour violation uses nearest hourly file plus hourly time series",
        "violation": violation,
        "selected_reference": {
            "time_index": int(selected_index),
            "path": str(history[selected_index]),
            "valid_time_utc": time_axis[selected_index] if selected_index < len(time_axis) else history[selected_index].name[-19:],
        },
        "wrf_reference_theta_at_cell_over_time": theta_series,
        "wrf_reference_horizontal_extremes_at_level": horizontal_extremes,
        "convective_signal_neighbor_3x3": {
            "theta_k_stats": _stats(_neighbor_slice(t, k, j, i)),
            "qv_kg_kg_stats": _stats(_neighbor_slice(qv, k, j, i)),
            "w_lower_face_m_s_stats": _stats(_neighbor_slice(w, k, j, i)),
            "w_upper_face_m_s_stats": _stats(_neighbor_slice(w, min(k + 1, w.shape[0] - 1), j, i)),
        },
        "physical_interpretation": (
            "WRF reference breaches the same theta envelope near the failure site."
            if wrf_toward_bound
            else "WRF reference stays inside the theta envelope at the failure site and selected level."
        ),
    }
    _write_json(output / "proof_wrf_reference_theta.json", proof)
    return proof


def _trace_violation_cell(
    run_id: str,
    cell: dict[str, int],
    *,
    steps: int,
    output: Path,
) -> tuple[dict[str, Any], Any]:
    state, namelist, _case, _meta = _case_state_and_namelist(run_id)
    run = Gen2Run(RUN_ROOT / run_id)
    current = state
    rows = []
    first_divergent = None
    k, j, i = int(cell["k"]), int(cell["j"]), int(cell["i"])
    step_hours = float(namelist.dt_s) / 3600.0
    for step in range(1, int(steps) + 1):
        current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
        lead_seconds = step * float(namelist.dt_s)
        forecast_theta = float(np.asarray(jax.device_get(current.theta[k, j, i])))
        wrf_theta = _interpolated_wrf_theta(run, domain="d02", lead_seconds=lead_seconds, k=k, j=j, i=i)
        delta = forecast_theta - wrf_theta
        row = {
            "step": int(step),
            "lead_seconds": float(lead_seconds),
            "forecast_theta_k": forecast_theta,
            "interpolated_wrf_theta_k": wrf_theta,
            "delta_k": float(delta),
            "abs_delta_k": float(abs(delta)),
        }
        rows.append(row)
        if first_divergent is None and abs(delta) > 1.0e-6:
            first_divergent = row
    payload = {
        "artifact_type": "m6b_v3_localize_cell_divergence_trace",
        "cell": cell,
        "threshold_k": 1.0e-6,
        "first_divergent_step": first_divergent,
        "per_step": rows,
        "reference_note": "WRF truth is hourly; sub-hour values use linear interpolation between hourly wrfout files.",
    }
    _write_json(output / "proof_cell_divergence_trace.json", payload)
    return payload, current


def _run_first_bad_helper(run_id: str, steps: int, output: Path) -> dict[str, Any]:
    tracer = _load_script_module("diagnostic_first_bad_step_tracer_local", "scripts/diagnostic_first_bad_step_tracer.py")
    run_dir = RUN_ROOT / run_id
    case = tracer.load_default_case(run_dir)
    replay_config = tracer.replay_config_for_steps(
        steps,
        dt_s=10.0,
        n_acoustic=10,
        final_radiation=False,
    )
    payload = tracer.run_sanitizer_off_replay(
        case,
        replay_config,
        steps=steps,
        toggle=tracer.DiagnosticToggle(name="v3-localize-20260509", description="V3 localization helper, sanitizer off."),
        abort_on_first_bad=False,
        localize_stage=False,
    )
    _write_json(output / "proof_first_bad_step_tracer.json", {"artifact_type": "diagnostic_first_bad_step_tracer", **payload})
    return payload


def _write_vertical_phase_space_input(
    output: Path,
    *,
    cell: dict[str, int],
    initial_state: Any,
    before_state: Any,
    after_state: Any,
    cell_trace: dict[str, Any],
) -> Path:
    k, j, i = int(cell["k"]), int(cell["j"]), int(cell["i"])
    series = {
        "theta": [row["forecast_theta_k"] for row in cell_trace["per_step"]],
        "w": [],
        "p": [],
        "mu": [],
    }
    for state in (initial_state, before_state, after_state):
        series["w"].append(_array_value_at(state.w, k, j, i))
        series["p"].append(_array_value_at(state.p, k, j, i))
        series["mu"].append(_array_value_at(state.mu, k, j, i))
    path = output / "diagnostic_vertical_column_phase_space_input.json"
    _write_json(
        path,
        {
            "columns": [
                {
                    "name": "theta_violation_column",
                    "i": i,
                    "j": j,
                    "profiles": _column_profiles(initial_state, cell),
                    "time_series": series,
                }
            ]
        },
    )
    return path


def _run_vertical_phase_space(input_path: Path, output: Path) -> dict[str, Any]:
    module = _load_script_module("diagnostic_vertical_column_phase_space_local", "scripts/diagnostic_vertical_column_phase_space.py")
    payload = module.build_payload(input_path)
    _write_json(output / "proof_vertical_column_phase_space.json", payload)
    return payload


def _run_boundary_ring(
    run_id: str,
    stage1: dict[str, Any],
    forecast_state: Any,
    output: Path,
) -> dict[str, Any]:
    module = _load_script_module("diagnostic_boundary_ring_error_profiler_local", "scripts/diagnostic_boundary_ring_error_profiler.py")
    run = Gen2Run(RUN_ROOT / run_id)
    violation = stage1.get("first_violation") or {}
    cell = violation.get("cell") or {"i": 0, "j": 0, "k": 0}
    k = int(cell["k"])
    selected_index = min(max(int(round(float(violation.get("lead_seconds") or 0.0) / 3600.0)), 0), len(run.history_files("d02")) - 1)
    forecast = np.asarray(jax.device_get(forecast_state.theta[k, :, :]), dtype=np.float64)
    reference = np.asarray(run.load("d02", "T", time=selected_index, lazy=False), dtype=np.float64)[k] + 300.0
    input_path = output / "diagnostic_boundary_ring_error_input.json"
    _write_json(
        input_path,
        {
            "field": "theta",
            "forecast": {"theta": forecast},
            "reference": {"theta": reference},
            "metadata": {
                "cell": cell,
                "selected_reference_time_index": selected_index,
                "lead_seconds": violation.get("lead_seconds"),
            },
        },
    )
    payload = module.build_payload(input_path)
    _write_json(output / "proof_boundary_ring_error.json", payload)
    return payload


def _run_operator_budget(before_state: Any, case: Any, cell: dict[str, int], output: Path) -> dict[str, Any]:
    tracer = _load_script_module("diagnostic_first_bad_step_terms_local", "scripts/diagnostic_first_bad_step_tracer.py")
    budget_module = _load_script_module("diagnostic_operator_term_budget_tracer_local", "scripts/diagnostic_operator_term_budget_tracer.py")
    dt = 10.0 / 10.0
    raw_terms = tracer._mpas_recurrence_terms(
        before_state,
        case.base_state,
        case.metrics,
        dt=dt,
        epssm=0.1,
        buoyancy_scale=tracer.acoustic_wrf.SOURCE_BACKED_COLUMN_BUOYANCY_TENDENCY_SCALE,
    )
    terms = {
        "buoyancy": _term_column(raw_terms["buoyancy_face"], cell),
        "pressure_restoring": _term_column(raw_terms["rhs_interior"], cell),
        "density_coupling": _term_column(raw_terms["rs"], cell),
        "theta_transport": _term_column(raw_terms["ts"], cell),
        "rayleigh": [],
        "smdiv": [],
        "boundary_forcing": [],
        "cofwr": _term_column(raw_terms["cofwr"], cell),
        "cofwz": _term_column(raw_terms["cofwz"], cell),
        "coftz": _term_column(raw_terms["coftz"], cell),
        "cofwt": _term_column(raw_terms["cofwt"], cell),
    }
    input_path = output / "diagnostic_operator_term_budget_input.json"
    _write_json(input_path, {"cell": cell, "terms": terms})
    payload = budget_module.build_payload(input_path)
    _write_json(output / "proof_operator_term_budget.json", payload)
    return payload


def _term_column(value: Any, cell: dict[str, int]) -> np.ndarray:
    arr = np.asarray(jax.device_get(value), dtype=np.float64)
    if arr.ndim == 3:
        j = min(max(int(cell["j"]), 0), arr.shape[1] - 1)
        i = min(max(int(cell["i"]), 0), arr.shape[2] - 1)
        return arr[:, j, i]
    if arr.ndim == 2:
        j = min(max(int(cell["j"]), 0), arr.shape[0] - 1)
        i = min(max(int(cell["i"]), 0), arr.shape[1] - 1)
        return np.asarray([arr[j, i]], dtype=np.float64)
    return arr


def _boundary_large(boundary_payload: dict[str, Any], cell: dict[str, int]) -> bool:
    rings = boundary_payload.get("measurements", {}).get("ring_rmse", [])
    if not rings:
        return False
    by_label = {item.get("band_grid_cells"): item.get("rmse") for item in rings}
    edge = by_label.get("0-5")
    interior = by_label.get("interior")
    if edge is None:
        return False
    distance = min(int(cell["i"]), int(cell["j"]))
    if interior is None:
        return bool(distance < 5 and float(edge) > 1.0)
    return bool(distance < 5 and float(edge) >= 0.75 * max(float(interior), 1.0e-12))


def _top_operator(operator_budget: dict[str, Any]) -> str:
    ranking = operator_budget.get("measurements", {}).get("ranking", [])
    return str(ranking[0]["term"]) if ranking else "unknown_operator"


def run_stage3(run_id: str, stage1: dict[str, Any], previous_state: Any, current_state: Any, case: Any, output: Path) -> dict[str, Any]:
    initial_state, _namelist, _case2, _meta = _case_state_and_namelist(run_id)
    violation = stage1.get("first_violation") or {}
    cell = violation.get("cell") or {"i": 0, "j": 0, "k": 0}
    steps = int(violation.get("step") or stage1.get("steps_completed") or 1)
    helper = _run_first_bad_helper(run_id, max(steps, 1), output)
    cell_trace, _ = _trace_violation_cell(run_id, cell, steps=max(steps, 1), output=output)
    phase_input = _write_vertical_phase_space_input(
        output,
        cell=cell,
        initial_state=initial_state,
        before_state=previous_state,
        after_state=current_state,
        cell_trace=cell_trace,
    )
    phase = _run_vertical_phase_space(phase_input, output)
    boundary = _run_boundary_ring(run_id, stage1, current_state, output)
    operator_budget = _run_operator_budget(previous_state, case, cell, output)

    first_divergent = cell_trace.get("first_divergent_step")
    first_divergent_step = int(first_divergent["step"]) if first_divergent else None
    boundary_is_large = _boundary_large(boundary, cell)
    if first_divergent_step is None:
        verdict = "INSUFFICIENT-EVIDENCE"
    elif first_divergent_step == 1 and boundary_is_large:
        verdict = "IC-SPECIFIC"
    elif first_divergent_step == 1:
        verdict = f"MATH:{_top_operator(operator_budget)}"
    elif first_divergent_step > 1:
        verdict = "NUMERICAL-DRIFT"
    else:
        verdict = "INSUFFICIENT-EVIDENCE"

    proof = {
        "artifact_type": "m6b_v3_localize_math_vs_ic",
        "status": verdict,
        "first_bad_step_tracer": {
            "first_bad_step": helper.get("first_bad_step"),
            "first_guard_limit_step": helper.get("first_guard_limit_step"),
            "first_nonfinite_step": helper.get("first_nonfinite_step"),
            "first_issue": helper.get("first_issue"),
        },
        "first_divergence_vs_wrf_reference": first_divergent,
        "vertical_column_phase_space": {
            "status": phase.get("status"),
            "path": str(output / "proof_vertical_column_phase_space.json"),
        },
        "boundary_ring_error": {
            "status": boundary.get("status"),
            "boundary_error_large": boundary_is_large,
            "path": str(output / "proof_boundary_ring_error.json"),
        },
        "operator_term_budget": {
            "top_operator": _top_operator(operator_budget),
            "path": str(output / "proof_operator_term_budget.json"),
        },
        "classification_rules": {
            "math": "first divergence step is 1 and boundary-ring error is not large",
            "ic_specific": "first divergence step is 1 and failure is boundary-ring dominated",
            "numerical_drift": "first divergence step is later than 1",
        },
    }
    _write_json(output / "proof_math_vs_ic.json", proof)
    return proof


def _memo(verdict: str, stage1: dict[str, Any], stage2: dict[str, Any], stage3: dict[str, Any]) -> str:
    violation = stage1.get("first_violation") or {}
    cell = violation.get("cell") or {}
    evidence = [
        f"V3 operational replay on `20260509_18z_l3_24h_20260511T190519Z` first breached theta bounds at step {violation.get('step')} / lead {violation.get('lead_seconds')} s.",
        f"Failure cell was k={cell.get('k')} j={cell.get('j')} i={cell.get('i')} with theta={violation.get('value_k')} K and bound={violation.get('bound_k')} K.",
        f"WRF hourly reference status at the failure site: `{stage2.get('status')}`.",
        f"WRF selected-cell theta at nearest hourly reference: {stage2.get('wrf_reference_horizontal_extremes_at_level', {}).get('selected_cell_theta_k')} K.",
        f"First divergence against interpolated hourly WRF reference: {stage3.get('first_divergence_vs_wrf_reference')}.",
        f"Boundary-ring profiler large-boundary classification: {stage3.get('boundary_ring_error', {}).get('boundary_error_large')}.",
        f"Operator budget top term: `{stage3.get('operator_term_budget', {}).get('top_operator')}`.",
    ]
    next_sprint = "m6b-v3-fix-20260509-theta-math" if verdict.startswith("MATH:") else "m6b-v3-ic-boundary-forcing-audit"
    return "\n".join(
        [
            "# M6b V3 20260509 Theta Localization Memo",
            "",
            f"**Verdict**: `{verdict}`",
            "",
            "**Evidence**:",
            *(f"- {item}" for item in evidence),
            "",
            f"**Recommended next sprint**: `{next_sprint}` — isolate the named cause without touching `dynamics/core/` or the operational-mode body.",
            "",
            "**Risks / caveats**:",
            "- WRF truth is hourly, so sub-hour divergence uses interpolation between hourly wrfout files rather than an acoustic-substep savepoint.",
            "- The diagnostic first-bad helper is the sanitizer-off replay helper, while Stage 1 uses the V3 operational entry point.",
            "- No GPU-vs-CPU step-2 NaN cross-link was found in this sprint's inputs; if a sister sprint reproduces that NaN, compare its first bad cell against this proof.",
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=PINNED_RUN_ID)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != PINNED_RUN_ID:
        raise SystemExit(f"this sprint is pinned to {PINNED_RUN_ID}, got {args.run_id}")
    args.output.mkdir(parents=True, exist_ok=True)
    stage1, previous_state, current_state, _namelist, case = run_stage1(args.run_id, args.output)
    stage2 = run_stage2(args.run_id, stage1, args.output)
    stage3 = run_stage3(args.run_id, stage1, previous_state, current_state, case, args.output)
    verdict = "PHYSICAL" if stage2["status"] == "PHYSICAL_SIGNAL_IN_REFERENCE" else str(stage3["status"])
    memo = _memo(verdict, stage1, stage2, stage3)
    (args.output / "localization_memo.md").write_text(memo, encoding="utf-8")
    summary = {
        "artifact_type": "m6b_v3_localize_20260509_summary",
        "status": verdict,
        "proofs": [
            str(args.output / "proof_theta_explosion.json"),
            str(args.output / "proof_wrf_reference_theta.json"),
            str(args.output / "proof_math_vs_ic.json"),
            str(args.output / "localization_memo.md"),
        ],
    }
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
