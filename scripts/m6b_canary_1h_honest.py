#!/usr/bin/env python
"""M6b honest Canary d02 operational-mode acceptance probe.

This script deliberately uses the real Gen2 d02 replay state and the public
``run_forecast_operational`` entry point. If a kill/acceptance gate fails early,
it records the blocker and does not continue into downstream RMSE or speedup
claims that would be based on an invalid forecast.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.d02_replay import build_replay_case, forecast_comparison
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-honest-1h-canary"
ARTIFACTS = SPRINT / "artifacts"
RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DEFAULT_RUN_IDS = (
    "20260509_18z_l3_24h_20260511T190519Z",
    "20260521_18z_l3_24h_20260522T072630Z",
    "20260523_18z_l3_24h_20260524T004313Z",
)
THRESHOLDS = {"T2": 3.0, "U10": 7.5, "V10": 7.5}
SPATIAL_RATIO_THRESHOLD = 1.5


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cuda_profiler_call(name: str) -> None:
    library = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(library)
    result = getattr(cudart, name)()
    if int(result) != 0:
        raise RuntimeError(f"{name} failed with CUDA error {result}")


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
        "validation_mode_imports_absent": not any("gpuwrf.dynamics." + name in source for name in ("acoustic_loop", "dycore_step", "coupled_step")),
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
        acoustic_substeps=2,
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


def _leaf_finite(state: Any) -> bool:
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]
    return bool(np.asarray(jnp.all(jnp.asarray(checks))))


def _bounds(state: Any) -> dict[str, Any]:
    theta_min = float(np.asarray(jnp.min(state.theta)))
    theta_max = float(np.asarray(jnp.max(state.theta)))
    u_abs = float(np.asarray(jnp.max(jnp.abs(state.u))))
    v_abs = float(np.asarray(jnp.max(jnp.abs(state.v))))
    w_abs = float(np.asarray(jnp.max(jnp.abs(state.w))))
    return {
        "theta_min_k": theta_min,
        "theta_max_k": theta_max,
        "u_abs_max_m_s": u_abs,
        "v_abs_max_m_s": v_abs,
        "w_abs_max_m_s": w_abs,
        "theta_bounded": bool(200.0 < theta_min and theta_max < 400.0),
        "wind_bounded": bool(u_abs < 100.0 and v_abs < 100.0 and w_abs < 50.0),
    }


def _spatial_audit_from_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    passed = True
    for name in THRESHOLDS:
        error = np.asarray(comparison["spatial_mean_drift"][name]["error_map"], dtype=np.float64)
        boundary = np.zeros(error.shape, dtype=bool)
        width = 5
        boundary[:width, :] = True
        boundary[-width:, :] = True
        boundary[:, :width] = True
        boundary[:, -width:] = True
        boundary_rmse = float(np.sqrt(np.nanmean(error[boundary] * error[boundary])))
        interior_rmse = float(np.sqrt(np.nanmean(error[~boundary] * error[~boundary])))
        ratio = interior_rmse / max(boundary_rmse, 1.0e-12)
        ok = bool(np.isfinite(ratio) and ratio <= SPATIAL_RATIO_THRESHOLD)
        fields[name] = {
            "boundary_ring_rmse": boundary_rmse,
            "interior_rmse": interior_rmse,
            "interior_to_boundary_ratio": ratio,
            "threshold": SPATIAL_RATIO_THRESHOLD,
            "passed": ok,
        }
        passed = passed and ok
    return {"status": "PASS" if passed else "FAIL", "fields": fields}


def _rmse_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    passed = True
    for name, threshold in THRESHOLDS.items():
        value = float(comparison["rmse"][name]["value"])
        ok = value <= threshold
        fields[name] = {"rmse": value, "threshold": threshold, "passed": ok, "units": comparison["rmse"][name]["units"]}
        passed = passed and ok
    return {"status": "PASS" if passed else "FAIL", "fields": fields}


def run_one(run_id: str, *, probe_hours: float) -> dict[str, Any]:
    state, namelist, run, meta = _case_state_and_namelist(run_id)
    start = time.perf_counter()
    result = run_forecast_operational(state, namelist, probe_hours)
    block_until_ready(result)
    wall_s = time.perf_counter() - start
    finite = _leaf_finite(result)
    bounds = _bounds(result)
    record: dict[str, Any] = {
        **meta,
        "status": "PASS" if finite and bounds["theta_bounded"] and bounds["wind_bounded"] else "FAIL",
        "operational_entrypoint": "run_forecast_operational",
        "hours_requested_for_acceptance": 1.0,
        "hours_completed": float(probe_hours),
        "steps_completed": int(round(float(probe_hours) * 3600.0 / float(namelist.dt_s))),
        "wall_time_s_including_compile": wall_s,
        "all_leaves_finite": finite,
        "bounds": bounds,
    }
    if not finite:
        record["blocker"] = "NONFINITE"
        return record
    if not bounds["theta_bounded"]:
        record["blocker"] = "THETA_BOUNDS"
        return record
    if not bounds["wind_bounded"]:
        record["blocker"] = "WIND_BOUNDS"
        return record

    comparison = forecast_comparison(result, run, lead_hours=1.0)
    rmse = _rmse_gate(comparison)
    spatial = _spatial_audit_from_comparison(comparison)
    record["rmse"] = rmse
    record["spatial_divergence"] = spatial
    if rmse["status"] != "PASS":
        record["status"] = "FAIL"
        record["blocker"] = "RMSE_ENVELOPE"
    elif spatial["status"] != "PASS":
        record["status"] = "FAIL"
        record["blocker"] = "SPATIAL_DIVERGENCE"
    return record


def run_acceptance(run_ids: tuple[str, ...], *, probe_hours: float) -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    source_audit = _operational_source_audit()
    runs = [run_one(run_id, probe_hours=probe_hours) for run_id in run_ids]
    blocker = next((run.get("blocker") for run in runs if run["status"] != "PASS"), None)
    rmse_runs = [run for run in runs if "rmse" in run]
    aggregate_rmse: dict[str, Any] = {
        "status": "NOT_RUN" if blocker else "PASS",
        "reason": f"blocked_before_valid_1h_rmse:{blocker}" if blocker else None,
        "fields": {},
    }
    if rmse_runs:
        aggregate_pass = True
        for name, threshold in THRESHOLDS.items():
            values = [float(run["rmse"]["fields"][name]["rmse"]) for run in rmse_runs]
            mean_value = float(np.mean(values))
            ok = mean_value <= threshold
            aggregate_rmse["fields"][name] = {"mean_rmse": mean_value, "threshold": threshold, "passed": ok, "samples": len(values)}
            aggregate_pass = aggregate_pass and ok
        aggregate_rmse["status"] = "PASS" if aggregate_pass else "FAIL"
    payload = {
        "artifact_type": "m6b_honest_1h_canary_acceptance",
        "status": "BLOCKER" if blocker or source_audit["status"] != "PASS" else "PASS",
        "m6_close_recommendation": "BLOCKER" if blocker or source_audit["status"] != "PASS" else "CLOSE-M6",
        "blocker": blocker,
        "device": visible_gpu_name(),
        "run_ids": list(run_ids),
        "operational_mode": {
            "confirmed": source_audit["status"] == "PASS",
            "not_validation_mode": source_audit["validation_mode_imports_absent"],
            "sanitizer": source_audit["sanitizer"],
            "source_audit": source_audit,
        },
        "probe_policy": {
            "target_hours": 1.0,
            "probe_hours_before_downstream_gates": float(probe_hours),
            "reason": "fail fast on physical-bounds gate before claiming 1h RMSE/performance",
        },
        "runs": runs,
        "aggregate_rmse": aggregate_rmse,
    }
    _write_json(SPRINT / "proof_operational_runs.json", payload)
    _write_json(SPRINT / "proof_bounds.json", {"artifact_type": "m6b_bounds", "runs": runs, "status": payload["status"]})
    _write_json(SPRINT / "proof_tier4_rmse.json", {"artifact_type": "m6b_tier4_rmse", **aggregate_rmse})
    spatial_payload = {
        "artifact_type": "m6b_spatial_divergence",
        "status": "NOT_RUN" if blocker else "PASS",
        "reason": f"blocked_before_spatial_audit:{blocker}" if blocker else None,
        "runs": [{key: run[key] for key in ("run_id", "spatial_divergence") if key in run} for run in runs],
    }
    _write_json(SPRINT / "proof_spatial_divergence.json", spatial_payload)
    _write_json(SPRINT / "proof_operational_mode_audit.json", source_audit)
    return payload


def run_profile_only(run_id: str, *, probe_hours: float) -> dict[str, Any]:
    state, namelist, _, meta = _case_state_and_namelist(run_id)
    warmup = run_forecast_operational(state, namelist, probe_hours)
    block_until_ready(warmup)
    state2, namelist2, _, _ = _case_state_and_namelist(run_id)
    use_cuda_range = os.environ.get("GPUWRF_CUDA_PROFILER_RANGE") == "1"
    if use_cuda_range:
        _cuda_profiler_call("cudaProfilerStart")
    with jax.profiler.TraceAnnotation("m6b_honest_operational_first_step"):
        result = run_forecast_operational(state2, namelist2, probe_hours)
        block_until_ready(result)
    if use_cuda_range:
        _cuda_profiler_call("cudaProfilerStop")
    payload = {
        "artifact_type": "m6b_honest_profile_only",
        "status": "PASS",
        **meta,
        "profile_scope": "warmed run_forecast_operational first 10s step; full 1h blocked by physical-bounds failure",
        "hours_profiled": float(probe_hours),
    }
    _write_json(SPRINT / "proof_profile_only.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--probe-hours", type=float, default=10.0 / 3600.0)
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_ids = tuple(args.run_ids or DEFAULT_RUN_IDS)
    if args.profile_only:
        payload = run_profile_only(run_ids[0], probe_hours=args.probe_hours)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = run_acceptance(run_ids, probe_hours=args.probe_hours)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
