#!/usr/bin/env python
"""M6b carry-expansion 10 s operational probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    run_forecast_operational,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-standalone-vs-comparator-bisect"
RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DEFAULT_RUN_IDS = (
    "20260509_18z_l3_24h_20260511T190519Z",
    "20260521_18z_l3_24h_20260522T072630Z",
    "20260524_18z_l3_24h_20260525T074709Z",
)
DEFAULT_COMPARE_RUN_ID = "20260521_18z_l3_24h_20260522T072630Z"
DEFAULT_COMPARE_IC_TIME = "2026-05-21_18:00:00"
DT_S = 10.0
THETA_CHECK_LEVELS = 30


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _duration_steps(duration_s: float, dt_s: float = DT_S) -> int:
    raw = float(duration_s) / float(dt_s)
    rounded = int(round(raw))
    if rounded < 1 or abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"duration_s={duration_s} must be a positive integer multiple of dt_s={dt_s}")
    return rounded


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
        "forbidden_hits": hits,
        "validation_mode_imports_absent": not any(
            "gpuwrf.dynamics." + name in source for name in ("acoustic_loop", "dycore_step", "coupled_step")
        ),
    }


def _case_state_and_namelist(run_id: str, *, duration_s: float) -> tuple[Any, OperationalNamelist, dict[str, Any]]:
    run_dir = RUN_ROOT / run_id
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    steps = _duration_steps(duration_s)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=DT_S,
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
        "forecast_steps": steps,
    }
    return state, namelist, meta


def _all_leaves_finite(state: Any) -> bool:
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]
    return bool(np.asarray(jnp.all(jnp.asarray(checks))))


def _bounds(state: Any) -> dict[str, Any]:
    lower_theta = state.theta[:THETA_CHECK_LEVELS]
    upper_theta = state.theta[THETA_CHECK_LEVELS:]
    theta_min = float(np.asarray(jnp.min(lower_theta)))
    theta_max = float(np.asarray(jnp.max(lower_theta)))
    upper_theta_min = float(np.asarray(jnp.min(upper_theta)))
    upper_theta_max = float(np.asarray(jnp.max(upper_theta)))
    full_theta_min = float(np.asarray(jnp.min(state.theta)))
    full_theta_max = float(np.asarray(jnp.max(state.theta)))
    u_abs = float(np.asarray(jnp.max(jnp.abs(state.u))))
    v_abs = float(np.asarray(jnp.max(jnp.abs(state.v))))
    w_abs = float(np.asarray(jnp.max(jnp.abs(state.w))))
    return {
        "theta_levels_checked": [0, THETA_CHECK_LEVELS],
        "theta_min_k": theta_min,
        "theta_max_k": theta_max,
        "upper_theta_levels_checked": [THETA_CHECK_LEVELS, int(state.theta.shape[0])],
        "upper_theta_min_k": upper_theta_min,
        "upper_theta_max_k": upper_theta_max,
        "full_column_theta_min_k": full_theta_min,
        "full_column_theta_max_k": full_theta_max,
        "full_column_theta_caveat": "Lower 30 eta levels are checked against [200 K, 400 K]; upper levels are checked against [250 K, 700 K].",
        "u_abs_max_m_s": u_abs,
        "v_abs_max_m_s": v_abs,
        "w_abs_max_m_s": w_abs,
        "theta_bounded": bool(200.0 <= theta_min and theta_max <= 400.0 and 250.0 <= upper_theta_min and upper_theta_max <= 700.0),
        "wind_bounded": bool(u_abs <= 100.0 and v_abs <= 100.0 and w_abs <= 50.0),
    }


def _array_signature(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    bytes_view = contiguous.ravel().view(np.uint8)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        "first_3_bytes_hex": bytes_view[:3].tobytes().hex(),
    }


def _named_leaf_signatures(obj: Any, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {name: _array_signature(getattr(obj, name)) for name in names}


def _state_signature(state: Any) -> dict[str, dict[str, Any]]:
    return _named_leaf_signatures(state, tuple(state.__slots__))


def _carry_signature(state: Any) -> dict[str, dict[str, Any]]:
    carry = initial_operational_carry(_enforce_operational_precision(state))
    names = tuple(name for name in carry.__dataclass_fields__ if name != "state")
    return _named_leaf_signatures(carry, names)


def _namelist_signature(namelist: OperationalNamelist) -> dict[str, Any]:
    return {
        "config": {
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "epssm": float(namelist.epssm),
            "top_lid": bool(namelist.top_lid),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            "boundary_config": str(namelist.boundary_config),
            "grid_shape": [int(namelist.grid.nz), int(namelist.grid.ny), int(namelist.grid.nx)],
            "dx_m": float(namelist.grid.projection.dx_m),
            "dy_m": float(namelist.grid.projection.dy_m),
        },
        "tendencies": _named_leaf_signatures(namelist.tendencies, tuple(namelist.tendencies.__slots__)),
        "metrics": _named_leaf_signatures(namelist.metrics, tuple(namelist.metrics._array_names())),
        "metrics_provenance": str(namelist.metrics.provenance),
    }


def _input_signature(label: str, state: Any, namelist: OperationalNamelist, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "run_id": meta["run_id"],
        "run_dir": meta["run_dir"],
        "grid": meta["grid"],
        "state": _state_signature(state),
        "namelist": _namelist_signature(namelist),
        "carry_initialization": _carry_signature(state),
        "wrapping": {
            "state_replace": "p=p_total, ph=ph_total, mu=mu_total",
            "carry_init": "initial_operational_carry(_enforce_operational_precision(state))",
            "entrypoint": "run_forecast_operational",
        },
    }


def _max_abs_delta(left: Any, right: Any) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    diff = np.abs(a - b)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return 1.0e300
    max_finite = float(np.max(finite))
    return 1.0e300 if not np.all(np.isfinite(diff)) else max_finite


def _state_deltas(left: Any, right: Any) -> dict[str, dict[str, Any]]:
    deltas: dict[str, dict[str, Any]] = {}
    for name in left.__slots__:
        delta = _max_abs_delta(getattr(left, name), getattr(right, name))
        deltas[name] = {
            "max_abs_delta": delta,
            "left_shape": list(np.shape(getattr(left, name))),
            "right_shape": list(np.shape(getattr(right, name))),
            "exceeds_0": bool(delta != 0.0),
        }
    return deltas


def _largest_delta(deltas: dict[str, dict[str, Any]]) -> str | None:
    if not deltas:
        return None
    return max(deltas, key=lambda name: float(deltas[name]["max_abs_delta"]))


def run_one(run_id: str, *, duration_s: float) -> dict[str, Any]:
    state, namelist, meta = _case_state_and_namelist(run_id, duration_s=duration_s)
    start = time.perf_counter()
    result = run_forecast_operational(state, namelist, float(duration_s) / 3600.0)
    block_until_ready(result)
    wall_s = time.perf_counter() - start
    finite = _all_leaves_finite(result)
    bounds = _bounds(result)
    status = "PASS" if finite and bounds["theta_bounded"] and bounds["wind_bounded"] else "FAIL"
    record = {
        **meta,
        "status": status,
        "operational_entrypoint": "run_forecast_operational",
        "duration_s": float(duration_s),
        "steps_completed": int(meta["forecast_steps"]),
        "wall_time_s_including_compile": wall_s,
        "all_leaves_finite": finite,
        "bounds": bounds,
    }
    if not finite:
        record["blocker"] = "NONFINITE"
    elif not bounds["theta_bounded"]:
        record["blocker"] = "THETA_BOUNDS"
    elif not bounds["wind_bounded"]:
        record["blocker"] = "WIND_BOUNDS"
    return record


def run_probe(run_ids: tuple[str, ...], *, duration_s: float) -> dict[str, Any]:
    audit = _operational_source_audit()
    runs = [run_one(run_id, duration_s=duration_s) for run_id in run_ids]
    blocker = next((run.get("blocker") for run in runs if run["status"] != "PASS"), None)
    payload = {
        "artifact_type": "m6b_carry_expansion_10s_probe",
        "status": "PASS" if blocker is None and audit["status"] == "PASS" else "FAIL",
        "device": visible_gpu_name(),
        "duration_s": float(duration_s),
        "run_ids": list(run_ids),
        "source_audit": audit,
        "runs": runs,
        "blocker": blocker,
    }
    _write_json(SPRINT / "proof_10s_probe.json", payload)
    if float(duration_s) == 10.0:
        _write_text(SPRINT / "proof_10s_bounded_after_fix.txt", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_input_audit(run_id: str, ic_time: str, *, duration_s: float = DT_S) -> dict[str, Any]:
    from scripts.m6b_real_ic_operational_compare import _operational_state_for_run

    standalone_state, standalone_namelist, standalone_meta = _case_state_and_namelist(run_id, duration_s=duration_s)
    comparator_state, comparator_namelist, comparator_case, comparator_ic = _operational_state_for_run(run_id, ic_time)
    comparator_meta = {
        "run_id": run_id,
        "run_dir": str(RUN_ROOT / run_id),
        "grid": comparator_case.metadata["grid"],
    }
    standalone = _input_signature("standalone", standalone_state, standalone_namelist, standalone_meta)
    comparator = _input_signature("comparator", comparator_state, comparator_namelist, comparator_meta)
    state_match = standalone["state"] == comparator["state"]
    namelist_match = standalone["namelist"] == comparator["namelist"]
    carry_match = standalone["carry_initialization"] == comparator["carry_initialization"]
    payload = {
        "artifact_type": "m6b_standalone_vs_comparator_input_signatures",
        "status": "PASS" if state_match and namelist_match and carry_match else "FAIL",
        "run_id": run_id,
        "ic_time": ic_time,
        "ic_file": str(comparator_ic),
        "duration_s": float(duration_s),
        "standalone": standalone,
        "comparator": comparator,
        "matches": {
            "state": state_match,
            "namelist": namelist_match,
            "carry_initialization": carry_match,
        },
        "diagnosis": (
            "Harness inputs match for the contracted 10 s step."
            if state_match and namelist_match and carry_match
            else "Harness inputs differ; inspect state/namelist/carry signatures."
        ),
    }
    _write_json(SPRINT / "proof_input_signatures.json", payload)
    return payload


def run_standalone_vs_comparator(run_id: str, ic_time: str, *, steps: int) -> dict[str, Any]:
    from scripts.m6b_real_ic_operational_compare import _clone_state, _operational_state_for_run

    duration_s = float(steps) * DT_S
    standalone_state, standalone_namelist, standalone_meta = _case_state_and_namelist(run_id, duration_s=duration_s)
    comparator_state, comparator_namelist, comparator_case, comparator_ic = _operational_state_for_run(run_id, ic_time)
    comparator_namelist = replace(comparator_namelist, dt_s=DT_S)
    hours = duration_s / 3600.0
    standalone_result = run_forecast_operational(standalone_state, standalone_namelist, hours)
    block_until_ready(standalone_result)
    comparator_result = run_forecast_operational(_clone_state(comparator_state), comparator_namelist, hours)
    block_until_ready(comparator_result)
    deltas = _state_deltas(standalone_result, comparator_result)
    largest = _largest_delta(deltas)
    max_delta = float(deltas[largest]["max_abs_delta"]) if largest else 0.0
    finite = _all_leaves_finite(standalone_result) and _all_leaves_finite(comparator_result)
    payload = {
        "artifact_type": "m6b_standalone_step_matches_comparator" if int(steps) == 1 else "m6b_standalone_multistep_matches_comparator",
        "status": "PASS" if max_delta == 0.0 and finite else "FAIL",
        "run_id": run_id,
        "ic_time": ic_time,
        "ic_file": str(comparator_ic),
        "device": visible_gpu_name(),
        "steps": int(steps),
        "duration_s": duration_s,
        "grid": comparator_case.metadata["grid"],
        "standalone_meta": standalone_meta,
        "comparator_namelist": _namelist_signature(comparator_namelist)["config"],
        "largest_delta_field": largest,
        "final_max_abs_delta": max_delta,
        "all_leaves_finite": bool(finite),
        "field_deltas": deltas,
    }
    if int(steps) == 1:
        _write_json(SPRINT / "proof_standalone_step1_matches.json", payload)
    return payload


def run_multi_step_bisect(run_id: str, ic_time: str, *, steps_list: tuple[int, ...] = (2, 5, 10)) -> dict[str, Any]:
    records = [run_standalone_vs_comparator(run_id, ic_time, steps=steps) for steps in steps_list]
    first_bad = next((record for record in records if record["status"] != "PASS"), None)
    payload = {
        "artifact_type": "m6b_standalone_vs_comparator_multistep_bisect",
        "status": "PASS" if first_bad is None else "FAIL",
        "run_id": run_id,
        "ic_time": ic_time,
        "steps_checked": list(steps_list),
        "first_diverging_step_count": None if first_bad is None else first_bad["steps"],
        "records": records,
        "diagnosis": (
            "Standalone and comparator harnesses remain bitwise-identical for all checked multi-step durations."
            if first_bad is None
            else "Standalone and comparator outputs diverged; inspect first_bad record."
        ),
    }
    _write_json(SPRINT / "proof_multi_step_divergence.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--audit-inputs", action="store_true")
    parser.add_argument("--compare-comparator", action="store_true")
    parser.add_argument("--multi-step-bisect", action="store_true")
    parser.add_argument("--gen2-run-id", default=DEFAULT_COMPARE_RUN_ID)
    parser.add_argument("--gen2-ic-time", default=DEFAULT_COMPARE_IC_TIME)
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_inputs:
        payload = run_input_audit(str(args.gen2_run_id), str(args.gen2_ic_time), duration_s=float(args.duration_s))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 2
    if args.compare_comparator:
        payload = run_standalone_vs_comparator(str(args.gen2_run_id), str(args.gen2_ic_time), steps=int(args.steps))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 2
    if args.multi_step_bisect:
        payload = run_multi_step_bisect(str(args.gen2_run_id), str(args.gen2_ic_time))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 2
    selected = tuple(args.run_ids or DEFAULT_RUN_IDS[: int(args.runs)])
    payload = run_probe(selected, duration_s=float(args.duration_s))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
