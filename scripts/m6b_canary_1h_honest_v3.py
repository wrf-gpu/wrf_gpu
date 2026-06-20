#!/usr/bin/env python
"""M6b V3: honest 1h Canary d02 operational-mode acceptance."""

from __future__ import annotations

import argparse
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
import numpy as np

from gpuwrf.integration.d02_replay import _reference_fields, _surface_fields, build_replay_case, forecast_comparison
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-honest-1h-canary-V3"
RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")
BASELINE_RMSE = ROOT / "data" / "fixtures" / "gen2_baseline" / "rmse_summary.csv"

PINNED_RUN_IDS = (
    "20260521_18z_l3_24h_20260522T072630Z",
    "20260521_18z_l3_24h_20260522T133443Z",
    "20260509_18z_l3_24h_20260511T190519Z",
)
TIER4_THRESHOLDS = {"T2": 3.0, "U10": 7.5, "V10": 7.5}
SPATIAL_RATIO_THRESHOLD = 1.5
LOWER_LEVELS = 30
UPPER_LEVELS = 14
WIND_LIMITS = {"u_abs_max_m_s": 100.0, "v_abs_max_m_s": 100.0, "w_abs_max_m_s": 50.0}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _operational_source_audit() -> dict[str, Any]:
    source_path = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"
    source = source_path.read_text(encoding="utf-8")
    forbidden = (
        "gpuwrf.dynamics.acoustic_loop",
        "gpuwrf.dynamics.dycore_step",
        "gpuwrf.dynamics.coupled_step",
        "device_get",
        "host_callback",
        "pure_callback",
        "io_callback",
        "sanitize_state",
        "snapshot(",
    )
    hits = [token for token in forbidden if token in source]
    return {
        "status": "PASS" if not hits else "FAIL",
        "source": str(source_path.relative_to(ROOT)),
        "entrypoint": "gpuwrf.runtime.operational_mode.run_forecast_operational",
        "forbidden_hits": hits,
        "sanitizer": "not_present_in_operational_path" if "sanitize_state" not in source else "present",
        "validation_mode_imports_absent": not any(
            "gpuwrf.dynamics." + name in source for name in ("acoustic_loop", "dycore_step", "coupled_step")
        ),
    }


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
    return state, namelist, case.run, meta


def _all_leaves_finite(state: Any) -> bool:
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]
    return bool(np.asarray(jnp.all(jnp.asarray(checks))))


def _bounds_for_state(state: Any, *, step: int, lead_seconds: float) -> dict[str, Any]:
    theta = state.theta
    if int(theta.shape[0]) != LOWER_LEVELS + UPPER_LEVELS:
        raise ValueError(f"expected {LOWER_LEVELS + UPPER_LEVELS} theta levels, got {theta.shape[0]}")

    lower_theta = theta[:LOWER_LEVELS, :, :]
    upper_theta = theta[LOWER_LEVELS:, :, :]
    values = {
        "step": int(step),
        "lead_seconds": float(lead_seconds),
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


def _first_bounds_blocker(step: dict[str, Any]) -> str | None:
    if step["passed"]:
        return None
    if not step["all_leaves_finite"]:
        return "NONFINITE"
    if not step["theta_lower_30_bounded"] or not step["theta_upper_14_bounded"]:
        return "THETA_BOUNDS"
    if not step["wind_bounded"]:
        return "WIND_BOUNDS"
    return "BOUNDS_UNKNOWN"


def _summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {"status": "NOT_RUN", "steps_checked": 0}
    first_bad = next((step for step in steps if not step["passed"]), None)
    keys = (
        "theta_full_min_k",
        "theta_full_max_k",
        "theta_lower_30_min_k",
        "theta_lower_30_max_k",
        "theta_upper_14_min_k",
        "theta_upper_14_max_k",
        "u_abs_max_m_s",
        "v_abs_max_m_s",
        "w_abs_max_m_s",
    )
    summary: dict[str, Any] = {
        "status": "PASS" if first_bad is None else "FAIL",
        "steps_checked": len(steps),
        "first_bad_step": first_bad,
    }
    for key in keys:
        values = [float(step[key]) for step in steps]
        if key.endswith("_min_k"):
            summary[f"{key}_over_run"] = float(np.min(values))
        else:
            summary[f"{key}_over_run"] = float(np.max(values))
    return summary


def _stepwise_bounds_audit(state: Any, namelist: OperationalNamelist, hours: float) -> tuple[Any, dict[str, Any]]:
    steps_total = int(round(float(hours) * 3600.0 / float(namelist.dt_s)))
    current = state
    steps: list[dict[str, Any]] = []
    step_hours = float(namelist.dt_s) / 3600.0
    wall_start = time.perf_counter()
    for step in range(1, steps_total + 1):
        current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
        row = _bounds_for_state(current, step=step, lead_seconds=step * float(namelist.dt_s))
        steps.append(row)
        if not row["passed"]:
            break
    wall_s = time.perf_counter() - wall_start
    summary = _summarize_steps(steps)
    summary.update(
        {
            "target_steps": steps_total,
            "target_hours": float(hours),
            "step_hours": step_hours,
            "wall_time_s_stepwise_audit": wall_s,
            "per_step": steps,
            "policy": {
                "lower_30_levels_k": [200.0, 400.0],
                "upper_14_levels_k": [250.0, 700.0],
                "full_column_finite_required": True,
                "wind_abs_limits_m_s": WIND_LIMITS,
            },
        }
    )
    return current, summary


def _rmse_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, threshold in TIER4_THRESHOLDS.items():
        value = float(comparison["rmse"][name]["value"])
        fields[name] = {
            "rmse": value,
            "threshold": threshold,
            "passed": bool(value <= threshold),
            "units": comparison["rmse"][name]["units"],
        }
    return {"status": "PASS" if all(field["passed"] for field in fields.values()) else "FAIL", "fields": fields}


def _spatial_audit(state: Any, run: Any) -> dict[str, Any]:
    surface = _surface_fields(state)
    reference, reference_path, valid_time = _reference_fields(run, "d02", 1.0)
    fields: dict[str, Any] = {}
    passed = True
    for name in TIER4_THRESHOLDS:
        error = np.asarray(surface[name], dtype=np.float64) - np.asarray(reference[name], dtype=np.float64)
        finite = np.isfinite(error)
        boundary = np.zeros(error.shape, dtype=bool)
        width = 5
        boundary[:width, :] = True
        boundary[-width:, :] = True
        boundary[:, :width] = True
        boundary[:, -width:] = True
        boundary_rmse = float(np.sqrt(np.nanmean(error[boundary] * error[boundary])))
        interior_rmse = float(np.sqrt(np.nanmean(error[~boundary] * error[~boundary])))
        ratio = interior_rmse / max(boundary_rmse, 1.0e-12)
        sq = error * error
        total_sq = float(np.nansum(sq))
        max_point_fraction = float(np.nanmax(sq) / total_sq) if total_sq > 0.0 else 0.0
        row_rmse = np.sqrt(np.nanmean(sq, axis=1))
        col_rmse = np.sqrt(np.nanmean(sq, axis=0))
        ok = bool(np.all(finite) and np.isfinite(ratio) and ratio <= SPATIAL_RATIO_THRESHOLD)
        fields[name] = {
            "boundary_ring_width": width,
            "boundary_ring_rmse": boundary_rmse,
            "interior_rmse": interior_rmse,
            "interior_to_boundary_ratio": float(ratio),
            "threshold": SPATIAL_RATIO_THRESHOLD,
            "max_point_error_fraction_of_total_sq": max_point_fraction,
            "max_row_rmse": float(np.nanmax(row_rmse)),
            "max_col_rmse": float(np.nanmax(col_rmse)),
            "all_error_finite": bool(np.all(finite)),
            "passed": ok,
        }
        passed = passed and ok
    return {
        "status": "PASS" if passed else "FAIL",
        "reference_path": str(reference_path),
        "valid_time_utc": valid_time,
        "fields": fields,
    }


def _full_hour_run(run_id: str, hours: float) -> tuple[Any, Any, dict[str, Any], float]:
    state, namelist, run, meta = _case_state_and_namelist(run_id)
    start = time.perf_counter()
    result = run_forecast_operational(state, namelist, hours)
    block_until_ready(result)
    wall_s = time.perf_counter() - start
    return result, run, meta, wall_s


def run_one(run_id: str, *, hours: float) -> dict[str, Any]:
    audit_state, namelist, _audit_run, meta = _case_state_and_namelist(run_id)
    _, bounds_audit = _stepwise_bounds_audit(audit_state, namelist, hours)
    record: dict[str, Any] = {
        **meta,
        "hours_requested": float(hours),
        "operational_entrypoint": "run_forecast_operational",
        "bounds_audit": bounds_audit,
        "status": "PASS" if bounds_audit["status"] == "PASS" else "FAIL",
    }
    if bounds_audit["status"] != "PASS":
        first_bad = bounds_audit.get("first_bad_step") or {}
        record["blocker"] = _first_bounds_blocker(first_bad) or "BOUNDS_UNKNOWN"
        return record

    forecast_state, run, _full_meta, full_wall_s = _full_hour_run(run_id, hours)
    final_bounds = _bounds_for_state(
        forecast_state,
        step=int(round(hours * 3600.0 / namelist.dt_s)),
        lead_seconds=hours * 3600.0,
    )
    record["full_hour_wall_time_s_including_compile"] = full_wall_s
    record["final_full_hour_bounds"] = final_bounds
    if not final_bounds["passed"]:
        record["status"] = "FAIL"
        record["blocker"] = _first_bounds_blocker(final_bounds) or "BOUNDS_UNKNOWN"
        return record

    comparison = forecast_comparison(forecast_state, run, lead_hours=hours)
    rmse = _rmse_gate(comparison)
    spatial = _spatial_audit(forecast_state, run)
    record["rmse"] = rmse
    record["spatial_divergence"] = spatial
    record["gen2_reference_path"] = comparison["gen2_reference_path"]
    record["valid_time_utc"] = comparison["valid_time_utc"]
    if rmse["status"] != "PASS":
        record["status"] = "FAIL"
        record["blocker"] = "RMSE_ENVELOPE"
    elif spatial["status"] != "PASS":
        record["status"] = "FAIL"
        record["blocker"] = "SPATIAL_DIVERGENCE"
    return record


def _aggregate_rmse(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rmse_runs = [run for run in runs if "rmse" in run]
    if not rmse_runs:
        blocker = next((run.get("blocker") for run in runs if run.get("status") != "PASS"), "UNKNOWN")
        return {"status": "NOT_RUN", "reason": f"blocked_before_valid_1h_rmse:{blocker}", "fields": {}}

    fields: dict[str, Any] = {}
    passed = True
    for name, threshold in TIER4_THRESHOLDS.items():
        values = [float(run["rmse"]["fields"][name]["rmse"]) for run in rmse_runs]
        mean_value = float(np.mean(values))
        ok = mean_value <= threshold
        fields[name] = {
            "mean_rmse": mean_value,
            "threshold": threshold,
            "passed": ok,
            "samples": len(values),
            "per_run_rmse": values,
            "units": rmse_runs[0]["rmse"]["fields"][name]["units"],
        }
        passed = passed and ok
    return {"status": "PASS" if passed else "FAIL", "fields": fields}


def _aggregate_spatial(runs: list[dict[str, Any]]) -> dict[str, Any]:
    spatial_runs = [run for run in runs if "spatial_divergence" in run]
    if not spatial_runs:
        blocker = next((run.get("blocker") for run in runs if run.get("status") != "PASS"), "UNKNOWN")
        return {"status": "NOT_RUN", "reason": f"blocked_before_spatial_audit:{blocker}", "runs": []}
    passed = all(run["spatial_divergence"]["status"] == "PASS" for run in spatial_runs)
    return {
        "status": "PASS" if passed else "FAIL",
        "runs": [{"run_id": run["run_id"], **run["spatial_divergence"]} for run in spatial_runs],
    }


def run_acceptance(run_ids: tuple[str, ...], *, hours: float) -> dict[str, Any]:
    source_audit = _operational_source_audit()
    runs = [run_one(run_id, hours=hours) for run_id in run_ids]
    aggregate_rmse = _aggregate_rmse(runs)
    aggregate_spatial = _aggregate_spatial(runs)
    blocker = next((run.get("blocker") for run in runs if run.get("status") != "PASS"), None)
    if source_audit["status"] != "PASS":
        blocker = "OPERATIONAL_SOURCE_AUDIT"
    elif blocker is None and aggregate_rmse["status"] == "FAIL":
        blocker = "RMSE_ENVELOPE"
    elif blocker is None and aggregate_spatial["status"] == "FAIL":
        blocker = "SPATIAL_DIVERGENCE"

    performance = {
        "status": "INFO",
        "cpu_reference_source": ".agent/sprints/2026-05-25-m6-perf-design-acceptance/proof_speedup.json",
        "cpu_reference_caveat": "28-rank CPU WRF denominator is recovered from Gen2 timestamps; not a clean successful CPU rerun.",
        "cpu_reference_wall_time_s": 687.3140738010406,
        "per_run_full_hour_wall_time_s": {
            run["run_id"]: run.get("full_hour_wall_time_s_including_compile") for run in runs
        },
    }
    for run in runs:
        wall = run.get("full_hour_wall_time_s_including_compile")
        if wall:
            run["wall_clock_vs_recovered_cpu"] = {
                "recovered_cpu_wall_time_s": performance["cpu_reference_wall_time_s"],
                "jax_wall_time_s": wall,
                "speedup": performance["cpu_reference_wall_time_s"] / wall,
                "informational_only": True,
            }

    payload = {
        "artifact_type": "m6b_honest_1h_canary_v3_acceptance",
        "status": "PASS" if blocker is None else "BLOCKER",
        "m6_close_recommendation": "CLOSE-M6" if blocker is None else "BLOCKER",
        "blocker": blocker,
        "device": visible_gpu_name(),
        "run_ids": list(run_ids),
        "cpu_cores": "0-3",
        "sanitizer": "OFF",
        "baseline_rmse_source": str(BASELINE_RMSE.relative_to(ROOT)),
        "operational_mode": {
            "confirmed": source_audit["status"] == "PASS",
            "not_validation_mode": source_audit["validation_mode_imports_absent"],
            "sanitizer": source_audit["sanitizer"],
            "source_audit": source_audit,
        },
        "bounds_policy": {
            "lower_30_levels_k": [200.0, 400.0],
            "upper_14_levels_k": [250.0, 700.0],
            "full_column_finite_required": True,
            "wind_abs_limits_m_s": WIND_LIMITS,
        },
        "tier4_thresholds": TIER4_THRESHOLDS,
        "spatial_ratio_threshold": SPATIAL_RATIO_THRESHOLD,
        "runs": runs,
        "aggregate_rmse": aggregate_rmse,
        "aggregate_spatial_divergence": aggregate_spatial,
        "performance": performance,
        "d2h_scope": "deferred_to_sister_sprint_per_V3_contract",
    }
    _write_json(SPRINT / "proof_1h_runs.json", payload)
    _write_json(SPRINT / "proof_bounds.json", {"artifact_type": "m6b_v3_bounds", "status": payload["status"], "runs": runs})
    _write_json(SPRINT / "proof_tier4_rmse.json", {"artifact_type": "m6b_v3_tier4_rmse", **aggregate_rmse})
    _write_json(SPRINT / "proof_spatial_divergence.json", {"artifact_type": "m6b_v3_spatial_divergence", **aggregate_spatial})
    _write_json(SPRINT / "proof_performance.json", {"artifact_type": "m6b_v3_performance", **performance})
    _write_json(SPRINT / "proof_operational_mode_audit.json", source_audit)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--hours", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_ids = tuple(args.run_ids or PINNED_RUN_IDS[: args.runs])
    payload = run_acceptance(run_ids, hours=args.hours)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
