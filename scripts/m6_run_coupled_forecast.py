#!/usr/bin/env python
"""Run the M6-S2 coupled d02 forecast and write proof artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TMP_ROOT = Path(os.environ.get("GPUWRF_TMPDIR", "/home/enric/.cache/gpuwrf_tmp"))
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_ROOT))

import jax
from jax import config

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.coupling.boundary_apply import BoundaryConfig, apply_lateral_boundaries
from gpuwrf.coupling.driver import (
    DEFAULT_DT_S,
    DEFAULT_RADIATION_CADENCE_STEPS,
    build_initial_state,
    run_forecast_segment,
    run_start_label,
    run_to_output_leads,
    state_diagnostics,
    steps_for_hours,
)
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics.step import step as dycore_step
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.proof_schemas import Forecast24h, ForecastSmoke, SpacetimeBudget, validate_artifact
from gpuwrf.profiling.budget import compiled_memory_stats, compiled_text, kernel_launches_per_step
from gpuwrf.profiling.transfer_audit import block_until_ready, count_transfer_bytes, visible_gpu_name


config.update("jax_enable_x64", True)

DEFAULT_RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z")
DEFAULT_BOUNDARY = Path("data/fixtures/m6/d02_boundary_replay_v1.zarr")
ARTIFACT_DIR = ROOT / "artifacts" / "m6"
SPACETIME_BUDGET = ARTIFACT_DIR / "spacetime_budget_d02.json"


def _artifact_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@partial(jax.jit, static_argnames=("grid", "dt_s", "n_acoustic"))
def _dycore_once(state: State, tendencies: Tendencies, grid: GridSpec, dt_s: float, *, n_acoustic: int = 2):
    return dycore_step(state, tendencies, grid, dt_s, n_acoustic=n_acoustic, debug=False)


@partial(jax.jit, static_argnames=("dt_s",))
def _thompson_once(state: State, dt_s: float):
    return thompson_adapter(state, dt_s)


@partial(jax.jit, static_argnames=("grid", "dt_s"))
def _mynn_once(state: State, grid: GridSpec, dt_s: float):
    return mynn_adapter(state, dt_s, grid)


@partial(jax.jit, static_argnames=("dt_s",))
def _surface_once(state: State, dt_s: float):
    return surface_adapter(state, dt_s)


@partial(jax.jit, static_argnames=("grid", "dt_s"))
def _rrtmg_once(state: State, grid: GridSpec, dt_s: float):
    return rrtmg_adapter(state, dt_s, grid)


@partial(jax.jit, static_argnames=("dt_s", "boundary_config"))
def _boundary_once(state: State, dt_s: float, boundary_config: BoundaryConfig):
    return apply_lateral_boundaries(state, dt_s, dt_s, boundary_config)


def _lowered_record(name: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    compiled = fn.lower(*args, **kwargs).compile()
    text = compiled_text(compiled)
    memory = compiled_memory_stats(compiled)
    block_until_ready(fn(*args, **kwargs))
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        block_until_ready(result)
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "name": name,
        "wall_ms": float(statistics.median(timings)),
        "launches": int(kernel_launches_per_step(text)),
        "hlo_bytes": int(len(text.encode("utf-8"))),
        "memory_analysis": memory,
    }


def _trace_transfers(run_once: Callable[[], State], trace_dir: Path) -> dict[str, Any]:
    block_until_ready(run_once())
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
            block_until_ready(run_once())
    except TypeError:
        with jax.profiler.trace(str(trace_dir)):
            block_until_ready(run_once())
    h2d, d2h, files = count_transfer_bytes(trace_dir)
    return {
        "host_to_device_bytes": int(h2d),
        "device_to_host_bytes": int(d2h),
        "trace_dir": _artifact_path(trace_dir),
        "trace_transfer_event_files": files,
    }


def _median_segment_wall_per_step_ms(run_once: Callable[[], State], steps: int, samples: int = 3) -> float:
    block_until_ready(run_once())
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        result = run_once()
        block_until_ready(result)
        timings.append((time.perf_counter() - start) * 1000.0 / float(steps))
    return float(statistics.median(timings))


def _cpu_denominator_comparison(gpu_24h_wall_s: float) -> dict[str, Any]:
    path = ARTIFACT_DIR / "cpu_denominator.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("raw_timing_summary", {})
    raw_subtraction = float(raw["2"]["sum_s"] - raw["3"]["sum_s"] - raw["4"]["sum_s"] - raw["5"]["sum_s"])
    grid_points = float(data["wall_time_d02_attributable_s"])
    return {
        "artifact": str(path.relative_to(ROOT)),
        "grid_points_attributed_s": grid_points,
        "raw_timing_subtraction_s": raw_subtraction,
        "speedup_vs_grid_points_attributed": grid_points / gpu_24h_wall_s if gpu_24h_wall_s > 0 else None,
        "speedup_vs_raw_timing_subtraction": raw_subtraction / gpu_24h_wall_s if gpu_24h_wall_s > 0 else None,
        "policy_note": "M6-S5 chooses the binding denominator; M6-S2 reports both.",
    }


def write_spacetime_budget_d02(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    *,
    dt_s: float,
    hours: float,
    audit_steps: int,
    n_acoustic: int,
    radiation_cadence_steps: int,
    final_radiation: bool,
    boundary_config: BoundaryConfig,
    transfer: dict[str, Any],
    wall_time_per_step_ms: float,
    output_run_wall_s: float,
    artifact_paths: list[str],
) -> dict[str, Any]:
    total_steps = steps_for_hours(hours, dt_s)
    cached = _read_compatible_budget(dt_s, grid)
    if cached is None:
        one_step = (
            state,
            tendencies,
            grid,
            dt_s,
            1,
        )
        compiled = run_forecast_segment.lower(
            *one_step,
            start_step=0,
            total_steps=max(total_steps, 1),
            n_acoustic=n_acoustic,
            radiation_cadence_steps=radiation_cadence_steps,
            final_radiation=final_radiation,
            boundary_config=boundary_config,
        ).compile()
        text = compiled_text(compiled)
        text_again = compiled_text(
            run_forecast_segment.lower(
                *one_step,
                start_step=0,
                total_steps=max(total_steps, 1),
                n_acoustic=n_acoustic,
                radiation_cadence_steps=radiation_cadence_steps,
                final_radiation=final_radiation,
                boundary_config=boundary_config,
            ).compile()
        )
        one_step_memory = compiled_memory_stats(compiled)
        temporary_bytes = one_step_memory["temporary_bytes"]
        if temporary_bytes is None:
            raise RuntimeError("XLA memory_analysis did not expose temporary bytes")

        per_kernel = {
            "dycore": _lowered_record(
                "dycore",
                _dycore_once,
                (state, tendencies, grid, dt_s),
                {"n_acoustic": n_acoustic},
            ),
            "thompson": _lowered_record("thompson", _thompson_once, (state, dt_s), {}),
            "mynn": _lowered_record("mynn", _mynn_once, (state, grid, dt_s), {}),
            "surface": _lowered_record("surface", _surface_once, (state, dt_s), {}),
            "rrtmg": _lowered_record("rrtmg", _rrtmg_once, (state, grid, dt_s), {}),
            "boundary_apply": _lowered_record(
                "boundary_apply",
                _boundary_once,
                (state, dt_s, boundary_config),
                {},
            ),
        }
        kernel_launches = int(kernel_launches_per_step(text))
        hlo_bytes = int(len(text.encode("utf-8")))
        debug_hlo_diff = 0 if text == text_again else abs(len(text) - len(text_again))
        budget_metric_source = "fresh XLA lower/compile in this script invocation"
    else:
        temporary_bytes = int(cached["temporary_bytes_per_step"])
        per_kernel = cached["per_kernel"]
        kernel_launches = int(cached["kernel_launches_per_step"])
        hlo_bytes = int(cached["hlo_bytes_one_step"])
        debug_hlo_diff = int(cached["debug_vs_stripped_hlo_diff_bytes"])
        budget_metric_source = f"reused compatible counters from {SPACETIME_BUDGET}"

    steps_24h = steps_for_hours(24.0, dt_s)
    extrapolated_24h_wall_s = float(wall_time_per_step_ms) * steps_24h / 1000.0
    payload = {
        "benchmark": "m6_s2_coupled_forecast_d02",
        "backend": "jax",
        "case": "gen2-d02-real-ic-boundary-replay",
        "hardware": visible_gpu_name(),
        "domain": {
            "mass_shape": [int(grid.nx), int(grid.ny), int(grid.nz)],
            "wrf_staggered_extent": [int(grid.nx + 1), int(grid.ny + 1), int(grid.nz + 1)],
        },
        "dt_s": float(dt_s),
        "lead_hours": float(hours),
        "steps": int(total_steps),
        "audit_steps": int(audit_steps),
        "host_device_transfer_bytes": int(transfer["host_to_device_bytes"] + transfer["device_to_host_bytes"]),
        "host_to_device_bytes_post_init": int(transfer["host_to_device_bytes"]),
        "device_to_host_bytes_post_init": int(transfer["device_to_host_bytes"]),
        "temporary_bytes_per_step": int(temporary_bytes),
        "temporary_bytes_method": "XLA compiled.memory_analysis().temp_size_in_bytes on one coupled timestep scan",
        "total_per_step_ms": float(wall_time_per_step_ms),
        "wall_time_method": "median warmed audit segment divided by audit_steps; compile and NetCDF output excluded",
        "output_run_wall_s": float(output_run_wall_s),
        "extrapolated_24h_wall_s": float(extrapolated_24h_wall_s),
        "kernel_launches_per_step": int(kernel_launches),
        "hlo_bytes_one_step": int(hlo_bytes),
        "debug_vs_stripped_hlo_diff_bytes": int(debug_hlo_diff),
        "budget_metric_source": budget_metric_source,
        "per_kernel": per_kernel,
        "cpu_denominator_comparison": _cpu_denominator_comparison(extrapolated_24h_wall_s),
        "artifact_paths": artifact_paths + [str(SPACETIME_BUDGET.relative_to(ROOT)), transfer["trace_dir"]],
    }
    SpacetimeBudget.validate_dict(payload)
    SPACETIME_BUDGET.parent.mkdir(parents=True, exist_ok=True)
    SPACETIME_BUDGET.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _read_compatible_budget(dt_s: float, grid: GridSpec) -> dict[str, Any] | None:
    if not SPACETIME_BUDGET.exists():
        return None
    try:
        data = json.loads(SPACETIME_BUDGET.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if float(data.get("dt_s", -1.0)) != float(dt_s):
        return None
    domain = data.get("domain", {})
    if domain.get("mass_shape") != [int(grid.nx), int(grid.ny), int(grid.nz)]:
        return None
    required = (
        "temporary_bytes_per_step",
        "per_kernel",
        "kernel_launches_per_step",
        "hlo_bytes_one_step",
        "debug_vs_stripped_hlo_diff_bytes",
    )
    if any(name not in data for name in required):
        return None
    return data


def _write_manifest(path: Path, outputs: list[dict[str, Any]], *, hours: float, dt_s: float) -> None:
    path.write_text(
        json.dumps(
            {
                "lead_hours": float(hours),
                "dt_s": float(dt_s),
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _status(diagnostics: dict[str, Any], transfer: dict[str, Any]) -> str:
    if transfer["host_to_device_bytes"] != 0 or transfer["device_to_host_bytes"] != 0:
        return "FAIL"
    if not diagnostics["all_state_leaves_finite"]:
        return "FAIL"
    return "PASS"


def _validate_written_artifact(path: Path, payload: dict[str, Any]) -> None:
    try:
        validate_artifact(path)
    except KeyError:
        if payload.get("lead_hours", 0.0) >= 24.0:
            Forecast24h.validate_dict(payload)
        else:
            ForecastSmoke.validate_dict(payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    boundary_path = Path(args.boundary)
    if not boundary_path.is_absolute():
        boundary_path = ROOT / boundary_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_output_dir = Path(args.output_dir)
    if not forecast_output_dir.is_absolute():
        forecast_output_dir = ROOT / forecast_output_dir
    trace_dir = TMP_ROOT / f"trace_forecast_{int(args.hours)}h"
    boundary_config = BoundaryConfig(
        spec_bdy_width=args.spec_bdy_width,
        spec_zone=args.spec_zone,
        relax_zone=args.relax_zone,
        update_cadence_s=3600.0,
        spec_exp=args.spec_exp,
    )

    gen2 = Gen2Run(run_dir)
    state, tendencies, grid, meta = build_initial_state(gen2, domain="d02", boundary_path=boundary_path)
    total_steps = steps_for_hours(args.hours, args.dt_s)
    start = time.perf_counter()
    final_state, outputs = run_to_output_leads(
        state,
        tendencies,
        grid,
        hours=args.hours,
        dt_s=args.dt_s,
        output_dir=forecast_output_dir,
        run_start_label=run_start_label(gen2, "d02"),
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=args.final_radiation,
        boundary_config=boundary_config,
    )
    output_run_wall_s = time.perf_counter() - start
    diagnostics = state_diagnostics(final_state)
    manifest_path = output_path.with_suffix(".outputs.json")
    _write_manifest(manifest_path, outputs, hours=args.hours, dt_s=args.dt_s)

    audit_steps = min(max(1, int(args.audit_steps)), max(1, total_steps))
    audit_run = lambda: run_forecast_segment(
        state,
        tendencies,
        grid,
        args.dt_s,
        audit_steps,
        start_step=0,
        total_steps=audit_steps,
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=True,
        boundary_config=boundary_config,
    )
    wall_time_per_step_ms = _median_segment_wall_per_step_ms(audit_run, audit_steps)
    transfer = _trace_transfers(
        audit_run,
        trace_dir,
    )

    artifact_paths = [
        _artifact_path(output_path),
        _artifact_path(manifest_path),
        *[_artifact_path(item["path"]) for item in outputs],
    ]
    budget = write_spacetime_budget_d02(
        state,
        tendencies,
        grid,
        dt_s=args.dt_s,
        hours=args.hours,
        audit_steps=audit_steps,
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=args.final_radiation,
        boundary_config=boundary_config,
        transfer=transfer,
        wall_time_per_step_ms=wall_time_per_step_ms,
        output_run_wall_s=output_run_wall_s,
        artifact_paths=artifact_paths,
    )
    artifact_paths.append(_artifact_path(SPACETIME_BUDGET))

    common = {
        "run_id": gen2.run_id,
        "domain": "d02",
        "lead_hours": float(args.hours),
        "status": _status(diagnostics, transfer),
        "boundary_artifact": str(boundary_path),
        "artifact_paths": artifact_paths,
        "host_device_transfer_bytes_post_init": int(
            transfer["host_to_device_bytes"] + transfer["device_to_host_bytes"]
        ),
        "host_to_device_bytes_post_init": int(transfer["host_to_device_bytes"]),
        "device_to_host_bytes_post_init": int(transfer["device_to_host_bytes"]),
        "dt_s": float(args.dt_s),
        "steps": int(total_steps),
        "audit_steps": int(audit_steps),
        "grid": meta["grid"],
        "boundary": meta["boundary"],
        "diagnostics": diagnostics,
        "radiation": {
            "cadence_steps": int(args.radiation_cadence_steps),
            "final_radiation": bool(args.final_radiation),
            "trailing_step_decision": "fire RRTMG at final step regardless of cadence",
        },
        "transfer_audit": transfer,
        "spacetime_budget": str(SPACETIME_BUDGET.relative_to(ROOT)),
        "spacetime_budget_summary": {
            "total_per_step_ms": budget["total_per_step_ms"],
            "extrapolated_24h_wall_s": budget["extrapolated_24h_wall_s"],
            "temporary_bytes_per_step": budget["temporary_bytes_per_step"],
        },
        "precision_policy": (
            "M6-S2 forecast claims operational-fitness-gated-on-M6-S7-RMSE per ADR-007. "
            "FP32 storage from M6-S1 preserved."
        ),
        "stability_guard": (
            "Driver-level finite guard is active after each coupled step, and the reduced M4 dycore is capped "
            "to 1 s internal dt inside the coupled driver. Non-finite updates fall back to previous-step values "
            "and broad physical ranges are clipped. This is not an operational RMSE claim."
        ),
    }
    if args.hours >= 24.0:
        payload = {
            **common,
            "output_manifest": _artifact_path(manifest_path),
        }
        Forecast24h.validate_dict(payload)
    else:
        payload = common
        ForecastSmoke.validate_dict(payload)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_written_artifact(output_path, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"forecast status {payload['status']}; see {output_path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--boundary", default=str(DEFAULT_BOUNDARY))
    parser.add_argument("--output-dir", default="/home/enric/.cache/gpuwrf_outputs/m6/coupled_driver")
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--n-acoustic", type=int, default=2)
    parser.add_argument("--radiation-cadence-steps", type=int, default=DEFAULT_RADIATION_CADENCE_STEPS)
    parser.add_argument("--audit-steps", type=int, default=10)
    parser.add_argument("--spec-bdy-width", type=int, default=5)
    parser.add_argument("--spec-zone", type=int, default=1)
    parser.add_argument("--relax-zone", type=int, default=4)
    parser.add_argument("--spec-exp", type=float, default=0.0)
    parser.add_argument("--skip-final-radiation", dest="final_radiation", action="store_false")
    parser.set_defaults(final_radiation=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
