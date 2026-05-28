#!/usr/bin/env python
"""Compare operational JAX operator traces against WRF Fortran traces.

The sprint forbids edits under ``src/**``, so this harness wraps the existing
private operational-mode steps from script space. If the required WRF trace is
not present, it writes an explicit partial proof object rather than fabricating
a same-code comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    _finite_or_origin,
    _limit_guarded_dynamics_state,
    _limit_theta_by_level,
    _physics_boundary_step,
    _rk_scan_step,
    _valid_mixing_ratio,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DEFAULT_RUN_IDS = {
    "20260521": "20260521_18z_l3_24h_20260522T072630Z",
}
TRACE_FIELDS = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "qke",
    "p",
    "ph",
    "mu",
    "p_total",
    "ph_total",
    "mu_total",
    "t_skin",
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
)
OPERATORS = (
    "state_before",
    "dycore_rk3_outer",
    "dycore_acoustic_substep",
    "microphysics",
    "surface",
    "pbl",
    "radiation",
    "lateral_bc",
    "state_after",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else str(scalar)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _candidate_trace_paths(case: str, run_id: str) -> list[Path]:
    run_dir = RUN_ROOT / run_id
    sprint_dir = ROOT / ".agent" / "sprints" / "2026-05-28-m9a-trace-harness"
    return [
        run_dir / "wrf_savepoint_dumps" / f"operational_trace_{case}_360steps.json",
        run_dir / "wrf_savepoint_dumps" / f"operational_trace_{case}",
        run_dir / "wrf_savepoint_dumps",
        run_dir / "operational_trace",
        run_dir / "savepoint_trace",
        sprint_dir / "wrf_reference_trace",
    ]


def _resolve_run_dir(case: str, run_id: str | None) -> tuple[str, Path]:
    resolved = run_id or DEFAULT_RUN_IDS.get(case)
    if resolved is None:
        matches = sorted(path.name for path in RUN_ROOT.glob(f"{case}*"))
        if not matches:
            raise FileNotFoundError(f"no Gen2 WRF run directory found for case {case} under {RUN_ROOT}")
        resolved = matches[0]
    run_dir = RUN_ROOT / resolved
    if not run_dir.exists():
        raise FileNotFoundError(f"Gen2 WRF run directory not found: {run_dir}")
    return resolved, run_dir


def _resolve_trace_path(case: str, run_id: str, explicit: Path | None) -> tuple[Path | None, list[str]]:
    candidates = [explicit] if explicit is not None else _candidate_trace_paths(case, run_id)
    searched = [str(path) for path in candidates if path is not None]
    for path in candidates:
        if path is not None and path.exists():
            return path, searched
    return None, searched


def _missing_trace_payload(
    *,
    case: str,
    horizon_steps: int,
    dt_seconds: float,
    run_id: str,
    run_dir: Path,
    searched: list[str],
    output: Path,
) -> dict[str, Any]:
    return {
        "trace_version": "1.0",
        "case": case,
        "horizon_steps": int(horizon_steps),
        "dt_seconds": float(dt_seconds),
        "commit": _git_commit(),
        "operators": [],
        "first_divergence": None,
        "status": "M9A_PARTIAL_MISSING_WRF_REFERENCE_TRACE",
        "wrf_reference_trace": {
            "found": False,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "searched": searched,
            "missing": (
                "WRF Fortran operational operator-boundary trace for _physics_boundary_step "
                f"case={case} horizon_steps={horizon_steps}"
            ),
            "required_operator_boundaries": list(OPERATORS),
            "required_fields": list(TRACE_FIELDS),
            "how_to_generate": [
                "Build/run the instrumented WRF savepoint patch from external/wrf_savepoint_patch against the same Gen2 case.",
                "Emit per-step operator-boundary arrays for dycore RK3 outer, acoustic substeps, microphysics, surface, PBL, radiation, and lateral BC.",
                "Write either a trace JSON with an operators array or a directory of step/operator .npz files under wrf_savepoint_dumps/ for this run.",
                f"Rerun: taskset -c 0-3 python scripts/operational_trace_compare.py --case {case} --horizon-steps {horizon_steps} --output {output}",
            ],
        },
        "jax_trace": {
            "run": False,
            "reason": "Skipped because no independent WRF Fortran reference trace was available for comparison.",
        },
    }


def _field_arrays(state: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for field in TRACE_FIELDS:
        if hasattr(state, field):
            arrays[field] = np.asarray(jax.device_get(getattr(state, field)))
    return arrays


def _stats(jax_value: np.ndarray, wrf_value: np.ndarray) -> dict[str, Any]:
    got = np.asarray(jax_value, dtype=np.float64)
    expected = np.asarray(wrf_value, dtype=np.float64)
    common = tuple(min(a, b) for a, b in zip(got.shape, expected.shape))
    slices = tuple(slice(0, dim) for dim in common)
    got_common = got[slices]
    expected_common = expected[slices]
    delta = got_common - expected_common
    if delta.size:
        abs_delta = np.abs(delta)
        flat = int(np.nanargmax(abs_delta))
        location = [int(item) for item in np.unravel_index(flat, delta.shape)]
        max_abs = float(abs_delta.flat[flat])
        rmse = float(np.sqrt(np.nanmean(delta * delta)))
        scale = float(np.nanmax(np.abs(expected_common)))
        rel = max_abs / max(scale, 1.0e-30)
    else:
        location = []
        max_abs = float("nan")
        rmse = float("nan")
        rel = float("nan")
    return {
        "max_abs_diff": max_abs,
        "rmse": rmse,
        "argmax_diff_idx": location,
        "rel_diff": rel,
        "jax_shape": list(got.shape),
        "wrf_shape": list(expected.shape),
    }


def _load_npz_trace(path: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    trace: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    for file_path in files:
        with np.load(file_path, allow_pickle=False) as data:
            step = int(data["step"]) if "step" in data else _parse_step(file_path.name)
            operator = str(data["operator"]) if "operator" in data else _parse_operator(file_path.name)
            trace[(step, operator)] = {
                field: np.asarray(data[field])
                for field in data.files
                if field not in {"step", "operator"} and not field.endswith("_metadata")
            }
    return trace


def _parse_step(name: str) -> int:
    for token in name.replace(".", "_").split("_"):
        if token.startswith("step") and token[4:].isdigit():
            return int(token[4:])
        if token.isdigit():
            return int(token)
    raise ValueError(f"cannot parse step from trace filename {name}")


def _parse_operator(name: str) -> str:
    for operator in OPERATORS:
        if operator in name:
            return operator
    raise ValueError(f"cannot parse operator from trace filename {name}")


def _load_json_trace(path: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    base_dir = path.parent
    for item in payload.get("operators", []):
        step = int(item["step"])
        operator = str(item["operator"])
        fields = item.get("arrays") or item.get("field_paths") or item.get("fields")
        arrays: dict[str, np.ndarray] = {}
        if isinstance(fields, dict):
            for field, value in fields.items():
                if isinstance(value, str):
                    arrays[field] = np.asarray(np.load(base_dir / value, allow_pickle=False))
                elif isinstance(value, dict) and "path" in value:
                    arrays[field] = np.asarray(np.load(base_dir / str(value["path"]), allow_pickle=False))
                elif isinstance(value, list):
                    arrays[field] = np.asarray(value)
        trace[(step, operator)] = arrays
    return trace


def _load_wrf_trace(path: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    if path.is_file() and path.suffix == ".json":
        return _load_json_trace(path)
    return _load_npz_trace(path)


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _build_case(run_dir: Path, dt_seconds: float) -> tuple[Any, OperationalNamelist, dict[str, Any]]:
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=float(dt_seconds),
        acoustic_substeps=10,
        radiation_cadence_steps=60,
        use_vertical_solver=True,
    )
    meta = {
        "run_id": case.run.run_id,
        "run_dir": str(run_dir),
        "grid": case.metadata.get("grid", {}),
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            "disable_guards": bool(namelist.disable_guards),
        },
    }
    return state, namelist, meta


def _guard_moisture(next_state: Any, origin: Any) -> Any:
    return next_state.replace(
        qv=_valid_mixing_ratio(next_state.qv, origin.qv),
        qc=_valid_mixing_ratio(next_state.qc, origin.qc),
        qr=_valid_mixing_ratio(next_state.qr, origin.qr),
        qi=_valid_mixing_ratio(next_state.qi, origin.qi),
        qs=_valid_mixing_ratio(next_state.qs, origin.qs),
        qg=_valid_mixing_ratio(next_state.qg, origin.qg),
    )


def _apply_lateral_boundary_with_guards(carry: OperationalCarry, namelist: OperationalNamelist, step_index: int, origin: Any) -> Any:
    lead_seconds = jnp.asarray(step_index, dtype=jnp.float64) * float(namelist.dt_s)
    bounded = apply_lateral_boundaries(carry.state, lead_seconds, float(namelist.dt_s), namelist.boundary_config)
    if bool(namelist.disable_guards):
        return bounded
    next_state = bounded.replace(
        u=_finite_or_origin(bounded.u, origin.u),
        v=_finite_or_origin(bounded.v, origin.v),
        w=_finite_or_origin(bounded.w, origin.w),
        theta=_limit_theta_by_level(bounded.theta, origin.theta),
        qv=_valid_mixing_ratio(bounded.qv, origin.qv),
        p=_finite_or_origin(bounded.p, origin.p),
        ph=_finite_or_origin(bounded.ph, origin.ph),
        p_total=_finite_or_origin(bounded.p_total, origin.p_total),
        ph_total=_finite_or_origin(bounded.ph_total, origin.ph_total),
        p_perturbation=_finite_or_origin(bounded.p_perturbation, origin.p_perturbation),
        ph_perturbation=_finite_or_origin(bounded.ph_perturbation, origin.ph_perturbation),
    )
    return _limit_guarded_dynamics_state(next_state, origin)


def _jax_trace_rows(
    state: Any,
    namelist: OperationalNamelist,
    *,
    horizon_steps: int,
) -> Iterator[tuple[int, str, dict[str, np.ndarray]]]:
    carry = initial_operational_carry(_enforce_operational_precision(state))
    cadence = int(namelist.radiation_cadence_steps)
    for step in range(int(horizon_steps)):
        wrf_step_index = step + 1
        physical_origin = carry.state
        yield step, "state_before", _field_arrays(physical_origin)
        carry = _rk_scan_step(carry, namelist, debug=False)
        _block_until_ready(carry)
        yield step, "dycore_rk3_outer", _field_arrays(carry.state)
        yield step, "dycore_acoustic_substep", _field_arrays(carry.state)

        next_state = carry.state
        if not bool(namelist.disable_guards):
            next_state = _guard_moisture(_limit_guarded_dynamics_state(next_state, physical_origin), physical_origin)
        if bool(namelist.run_physics):
            if not bool(namelist.disable_guards):
                next_state = thompson_adapter(next_state, float(namelist.dt_s))
                _block_until_ready(next_state)
                yield step, "microphysics", _field_arrays(next_state)
            next_state = surface_adapter(next_state, float(namelist.dt_s))
            _block_until_ready(next_state)
            yield step, "surface", _field_arrays(next_state)
            next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
            _block_until_ready(next_state)
            yield step, "pbl", _field_arrays(next_state)
            if cadence > 0 and wrf_step_index % cadence == 0:
                next_state = rrtmg_adapter(next_state, float(namelist.dt_s), namelist.grid)
                _block_until_ready(next_state)
                yield step, "radiation", _field_arrays(next_state)
        carry = carry.replace(state=next_state)
        if bool(namelist.run_boundary):
            next_state = _apply_lateral_boundary_with_guards(carry, namelist, wrf_step_index, physical_origin)
            carry = carry.replace(state=next_state)
            _block_until_ready(carry)
            yield step, "lateral_bc", _field_arrays(carry.state)
        carry = carry.replace(state=_enforce_operational_precision(carry.state))
        _block_until_ready(carry)
        yield step, "state_after", _field_arrays(carry.state)


def _compare_trace(
    *,
    wrf_trace: dict[tuple[int, str], dict[str, np.ndarray]],
    state: Any,
    namelist: OperationalNamelist,
    horizon_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    operators: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for step, operator, jax_fields in _jax_trace_rows(state, namelist, horizon_steps=horizon_steps):
        wrf_fields = wrf_trace.get((step, operator))
        if wrf_fields is None:
            continue
        field_stats = {
            field: _stats(jax_fields[field], wrf_fields[field])
            for field in sorted(set(jax_fields).intersection(wrf_fields))
        }
        row = {"step": int(step), "operator": operator, "fields": field_stats}
        operators.append(row)
        if first is None:
            for field, stats in field_stats.items():
                if float(stats["max_abs_diff"]) != 0.0:
                    first = {
                        "step": int(step),
                        "operator": operator,
                        "field": field,
                        "max_abs_diff": float(stats["max_abs_diff"]),
                        "rel_diff": float(stats["rel_diff"]),
                    }
                    break
    return operators, first


def _run_one_step_smoke(state: Any, namelist: OperationalNamelist) -> None:
    carry = initial_operational_carry(_enforce_operational_precision(state))
    carry = _physics_boundary_step(carry, namelist, jnp.asarray(1, dtype=jnp.int32), run_radiation=False, debug=False)
    _block_until_ready(carry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="20260521")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--horizon-steps", type=int, default=360)
    parser.add_argument("--dt-seconds", type=float, default=10.0)
    parser.add_argument("--wrf-trace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "proofs/m9/operational_trace_360steps.json")
    parser.add_argument("--smoke-one-jax-step", action="store_true")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    run_id, run_dir = _resolve_run_dir(args.case, args.run_id)
    trace_path, searched = _resolve_trace_path(args.case, run_id, args.wrf_trace)
    if trace_path is None:
        payload = _missing_trace_payload(
            case=args.case,
            horizon_steps=args.horizon_steps,
            dt_seconds=args.dt_seconds,
            run_id=run_id,
            run_dir=run_dir,
            searched=searched,
            output=args.output,
        )
        payload["wall_clock_seconds"] = time.perf_counter() - start
        _write_json(args.output, payload)
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
        return 0

    state, namelist, meta = _build_case(run_dir, args.dt_seconds)
    if args.smoke_one_jax_step:
        _run_one_step_smoke(state, namelist)
    wrf_trace = _load_wrf_trace(trace_path)
    operators, first = _compare_trace(
        wrf_trace=wrf_trace,
        state=state,
        namelist=namelist,
        horizon_steps=args.horizon_steps,
    )
    payload = {
        "trace_version": "1.0",
        "case": args.case,
        "horizon_steps": int(args.horizon_steps),
        "dt_seconds": float(args.dt_seconds),
        "commit": _git_commit(),
        "operators": operators,
        "first_divergence": first,
        "status": "PASS" if first is None else "FAIL",
        "wrf_reference_trace": {
            "found": True,
            "path": str(trace_path),
            "loaded_operator_count": len(wrf_trace),
        },
        "case_metadata": meta,
        "wall_clock_seconds": time.perf_counter() - start,
    }
    _write_json(args.output, payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
