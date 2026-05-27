#!/usr/bin/env python
"""M7 checkpoint/restart continuity probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
import os
from pathlib import Path
import subprocess
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

import jax  # noqa: E402
from jax import config  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.contracts.state import State  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name  # noqa: E402
from gpuwrf.runtime.checkpoint import read_checkpoint, read_checkpoint_with_runtime_state, write_checkpoint  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    OperationalNamelist,
    _enforce_operational_precision,
    _scan_forecast_segment,
)
from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: E402


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-27-m7-restart-continuity"
RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DT_S = 10.0
RUN_IDS = {
    "20260429": "20260429_18z_l3_24h_20260524T204451Z",
    "20260509": "20260509_18z_l3_24h_20260511T190519Z",
    "20260521": "20260521_18z_l3_24h_20260522T072630Z",
}


@partial(jax.jit, static_argnames=("start_step", "steps", "run_radiation"))
def _run_carry_segment_jit(
    carry: Any,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
    run_radiation: bool,
) -> Any:
    return _scan_forecast_segment(
        carry,
        namelist,
        start_step=int(start_step),
        steps=int(steps),
        run_radiation=bool(run_radiation),
        debug=False,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_id(run_key: str) -> str:
    return RUN_IDS.get(run_key, run_key)


def _build_case(
    run_key: str,
    *,
    run_physics: bool,
    run_boundary: bool,
    use_vertical_solver: bool,
) -> tuple[State, OperationalNamelist, Any, dict[str, Any]]:
    run_id = _run_id(run_key)
    run_dir = RUN_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"missing Gen2 run directory: {run_dir}")
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=DT_S,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=use_vertical_solver,
    )
    namelist = replace(namelist, run_physics=run_physics, run_boundary=run_boundary)
    meta = {
        "run_key": run_key,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "grid": case.metadata["grid"],
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
        },
    }
    return state, namelist, case.grid, meta


def _initial_carry(state: State) -> Any:
    return initial_operational_carry(_enforce_operational_precision(state))


def _run_steps_from_carry(
    carry: Any,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
) -> tuple[Any, float]:
    start = time.perf_counter()
    final_carry = _run_carry_segment_jit(
        carry,
        namelist,
        start_step=int(start_step),
        steps=int(steps),
        run_radiation=False,
    )
    block_until_ready(final_carry)
    return final_carry, time.perf_counter() - start


def _run_steps(
    state: State,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
) -> tuple[Any, float]:
    return _run_steps_from_carry(_initial_carry(state), namelist, start_step=start_step, steps=steps)


def _time_write_checkpoint(
    state: State,
    namelist: Any,
    grid: Any,
    step_index: int,
    path: Path,
    *,
    runtime_state: Any | None = None,
) -> float:
    start = time.perf_counter()
    write_checkpoint(state, namelist, grid, step_index, path, runtime_state=runtime_state)
    return time.perf_counter() - start


def _time_read_checkpoint(path: Path) -> tuple[State, Any, Any, int, float]:
    start = time.perf_counter()
    state, namelist, grid, step_index = read_checkpoint(path)
    block_until_ready(state)
    return state, namelist, grid, step_index, time.perf_counter() - start


def _time_read_checkpoint_with_runtime(path: Path) -> tuple[State, Any, Any, int, Any | None, float]:
    start = time.perf_counter()
    state, namelist, grid, step_index, runtime_state = read_checkpoint_with_runtime_state(path)
    block_until_ready((state, runtime_state))
    return state, namelist, grid, step_index, runtime_state, time.perf_counter() - start


def _threshold(dtype: np.dtype) -> float:
    return 1.0e-12 if np.dtype(dtype) == np.dtype(np.float64) else 1.0e-6


def _compare_states(reference: State, restarted: State) -> tuple[dict[str, Any], bool]:
    fields: dict[str, Any] = {}
    passed = True
    for field in State.__slots__:
        left = np.asarray(getattr(reference, field))
        right = np.asarray(getattr(restarted, field))
        threshold = _threshold(left.dtype)
        if left.shape != right.shape:
            max_delta = float("inf")
            finite = False
        else:
            delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
            finite = bool(np.all(np.isfinite(delta)))
            max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        field_passed = bool(finite and max_delta <= threshold)
        passed = passed and field_passed
        fields[field] = {
            "dtype": str(left.dtype),
            "shape": list(left.shape),
            "max_delta": max_delta,
            "threshold": threshold,
            "pass": field_passed,
        }
    return fields, passed


def _child_b1(args: argparse.Namespace) -> dict[str, Any]:
    state, namelist, grid, meta = _build_case(
        args.run_key,
        run_physics=not args.no_physics,
        run_boundary=not args.no_boundary,
        use_vertical_solver=not args.no_vertical_solver,
    )
    final_carry, forecast_wall_s = _run_steps(state, namelist, start_step=1, steps=args.n_steps)
    write_wall_s = _time_write_checkpoint(
        final_carry.state,
        namelist,
        grid,
        args.n_steps,
        args.checkpoint,
        runtime_state=final_carry,
    )
    payload = {
        "role": "B1",
        "status": "PASS",
        **meta,
        "n_steps": int(args.n_steps),
        "start_step": 1,
        "final_step_index": int(args.n_steps),
        "forecast_wall_s": forecast_wall_s,
        "checkpoint_write_s": write_wall_s,
        "checkpoint_path": str(args.checkpoint),
    }
    _write_json(args.output, payload)
    return payload


def _child_b2(args: argparse.Namespace) -> dict[str, Any]:
    state, namelist, grid, step_index, runtime_state, read_wall_s = _time_read_checkpoint_with_runtime(args.checkpoint)
    carry = runtime_state if runtime_state is not None else _initial_carry(state)
    final_carry, forecast_wall_s = _run_steps_from_carry(
        carry,
        namelist,
        start_step=int(step_index) + 1,
        steps=args.n_steps,
    )
    final_step = int(step_index) + int(args.n_steps)
    write_wall_s = _time_write_checkpoint(
        final_carry.state,
        namelist,
        grid,
        final_step,
        args.final_checkpoint,
        runtime_state=final_carry,
    )
    payload = {
        "role": "B2",
        "status": "PASS",
        "n_steps": int(args.n_steps),
        "start_step": int(step_index) + 1,
        "initial_step_index": int(step_index),
        "final_step_index": final_step,
        "forecast_wall_s": forecast_wall_s,
        "checkpoint_read_s": read_wall_s,
        "runtime_state_restored": runtime_state is not None,
        "final_checkpoint_write_s": write_wall_s,
        "checkpoint_path": str(args.checkpoint),
        "final_checkpoint_path": str(args.final_checkpoint),
    }
    _write_json(args.output, payload)
    return payload


def _run_child(cmd: list[str], stdout_path: Path, stderr_path: Path) -> tuple[int, dict[str, Any] | None]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    payload = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None
    return proc.returncode, payload


def _taskset_python_command(*items: str) -> list[str]:
    return ["taskset", "-c", "0-3", sys.executable, str(Path(__file__).resolve()), *items]


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    command_logs = SPRINT / "command_outputs"
    checkpoint_path = work_dir / f"{args.run_key}_step{args.n_steps}.pkl"
    final_checkpoint_path = work_dir / f"{args.run_key}_step{2 * args.n_steps}_restarted.pkl"
    reference_checkpoint_path = work_dir / f"{args.run_key}_step{2 * args.n_steps}_reference.pkl"
    b1_json = work_dir / "b1.json"
    b2_json = work_dir / "b2.json"

    state, namelist, grid, meta = _build_case(
        args.run_key,
        run_physics=not args.no_physics,
        run_boundary=not args.no_boundary,
        use_vertical_solver=not args.no_vertical_solver,
    )
    reference_carry, reference_wall_s = _run_steps(state, namelist, start_step=1, steps=2 * args.n_steps)
    _time_write_checkpoint(
        reference_carry.state,
        namelist,
        grid,
        2 * args.n_steps,
        reference_checkpoint_path,
        runtime_state=reference_carry,
    )

    common_flags = [
        "--run-key",
        args.run_key,
        "--n-steps",
        str(args.n_steps),
    ]
    if args.no_physics:
        common_flags.append("--no-physics")
    if args.no_boundary:
        common_flags.append("--no-boundary")
    if args.no_vertical_solver:
        common_flags.append("--no-vertical-solver")

    b1_cmd = _taskset_python_command(
        "child-b1",
        *common_flags,
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(b1_json),
    )
    b1_rc, b1_payload = _run_child(
        b1_cmd,
        command_logs / "restart_b1.stdout",
        command_logs / "restart_b1.stderr",
    )
    if b1_rc != 0 or b1_payload is None:
        payload = {
            "artifact_type": "m7_restart_continuity",
            "verdict": "BLOCKED",
            "reason": "B1 subprocess failed or did not emit JSON",
            "b1_returncode": b1_rc,
            "b1_command": b1_cmd,
        }
        _write_json(args.continuity_output, payload)
        return payload

    b2_cmd = _taskset_python_command(
        "child-b2",
        "--n-steps",
        str(args.n_steps),
        "--checkpoint",
        str(checkpoint_path),
        "--final-checkpoint",
        str(final_checkpoint_path),
        "--output",
        str(b2_json),
    )
    b2_rc, b2_payload = _run_child(
        b2_cmd,
        command_logs / "restart_b2.stdout",
        command_logs / "restart_b2.stderr",
    )
    if b2_rc != 0 or b2_payload is None:
        payload = {
            "artifact_type": "m7_restart_continuity",
            "verdict": "BLOCKED",
            "reason": "B2 subprocess failed or did not emit JSON",
            "b1": b1_payload,
            "b2_returncode": b2_rc,
            "b2_command": b2_cmd,
        }
        _write_json(args.continuity_output, payload)
        return payload

    restarted_state, _, _, restarted_step, _ = _time_read_checkpoint(final_checkpoint_path)
    field_deltas, field_passed = _compare_states(reference_carry.state, restarted_state)
    verdict = "PASS" if field_passed and restarted_step == 2 * args.n_steps else "FAIL"
    b1b2_forecast_s = float(b1_payload["forecast_wall_s"]) + float(b2_payload["forecast_wall_s"])
    write_s = float(b1_payload["checkpoint_write_s"])
    read_s = float(b2_payload["checkpoint_read_s"])
    overhead_payload = {
        "artifact_type": "m7_restart_overhead",
        "run_key": args.run_key,
        "n_steps": int(args.n_steps),
        "checkpoint_write_s": write_s,
        "checkpoint_read_s": read_s,
        "total_restart_overhead_s": write_s + read_s,
        "b1_b2_forecast_wall_s": b1b2_forecast_s,
        "overhead_percent_of_n_step_forecast": ((write_s + read_s) / max(b1b2_forecast_s / 2.0, 1.0e-12)) * 100.0,
    }
    _write_json(args.overhead_output, overhead_payload)

    payload = {
        "artifact_type": "m7_restart_continuity",
        "verdict": verdict,
        "device": visible_gpu_name(),
        **meta,
        "n_steps": int(args.n_steps),
        "reference": {
            "process": "A",
            "wall_s": reference_wall_s,
            "checkpoint_path": str(reference_checkpoint_path),
            "final_step_index": 2 * int(args.n_steps),
        },
        "restart": {
            "b1": b1_payload,
            "b2": b2_payload,
            "final_checkpoint_path": str(final_checkpoint_path),
            "final_step_index": restarted_step,
        },
        "threshold_policy": {
            "float64_abs": 1.0e-12,
            "float32_abs": 1.0e-6,
        },
        "fields": field_deltas,
        "overhead_path": str(args.overhead_output),
        "child_commands": {
            "b1": b1_cmd,
            "b2": b2_cmd,
        },
    }
    _write_json(args.continuity_output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"run", "child-b1", "child-b2"}:
        argv = ["run", *argv]
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--run-key", default="20260521")
        target.add_argument("--n-steps", type=int, default=10)
        target.add_argument("--no-physics", action="store_true")
        target.add_argument("--no-boundary", action="store_true")
        target.add_argument("--no-vertical-solver", action="store_true")

    run = sub.add_parser("run")
    add_common(run)
    run.add_argument("--work-dir", type=Path, default=SPRINT / "restart_work")
    run.add_argument("--continuity-output", type=Path, default=SPRINT / "restart_continuity.json")
    run.add_argument("--overhead-output", type=Path, default=SPRINT / "restart_overhead.json")

    b1 = sub.add_parser("child-b1")
    add_common(b1)
    b1.add_argument("--checkpoint", type=Path, required=True)
    b1.add_argument("--output", type=Path, required=True)

    b2 = sub.add_parser("child-b2")
    b2.add_argument("--n-steps", type=int, default=10)
    b2.add_argument("--checkpoint", type=Path, required=True)
    b2.add_argument("--final-checkpoint", type=Path, required=True)
    b2.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.n_steps) <= 0:
        raise ValueError("--n-steps must be positive")
    if args.command == "run":
        payload = run_orchestrator(args)
    elif args.command == "child-b1":
        payload = _child_b1(args)
    elif args.command == "child-b2":
        payload = _child_b2(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "PASS" or payload.get("verdict") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
