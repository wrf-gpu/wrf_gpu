#!/usr/bin/env python
"""Measure the binding M6-S5 ADR-007 full-domain 24h performance verdict."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Scratch root: env-overridable via GPUWRF_TMPDIR (config.paths.tmp_root); never a
# hardcoded <USER_HOME>/<name> so a clean checkout works out of the box.
from gpuwrf.config.paths import tmp_root  # noqa: E402

TMP_ROOT = tmp_root()
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.coupling.boundary_apply import BoundaryConfig, apply_lateral_boundaries
from gpuwrf.coupling.driver import (
    BisectionConfig,
    DEFAULT_DT_S,
    DEFAULT_RADIATION_CADENCE_STEPS,
    MAX_LIFTED_DYCORE_DT_S,
    SanitizeStats,
    build_initial_state,
    coupled_timestep_with_pre_sanitize,
    coupled_timestep_with_sanitize_stats,
    forecast_output_leads,
    run_forecast_segment,
    run_start_label,
    state_diagnostics,
    steps_for_hours,
    validate_lifted_coupled_dt,
    write_wrfout_gpu,
    sanitize_state_with_stats,
)
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics.step import step as dycore_step
from gpuwrf.io.gen2_accessor import DEFAULT_M6_BOUNDARY_REPLAY, DEFAULT_M6_GEN2_RUN_DIR, Gen2Run
from gpuwrf.io.proof_schemas import FullDomainBatchingVerdict, Tier2CoupledInvariants
from gpuwrf.profiling.budget import compiled_memory_stats, compiled_text, kernel_launches_per_step
from gpuwrf.profiling.transfer_audit import block_until_ready, count_transfer_bytes, visible_gpu_name
from gpuwrf.validation.tier2_coupled import (
    HYDROMETEOR_FIELDS,
    boundary_flux_closure,
    dry_mass_residual,
    hydrometeor_positivity,
    mu_continuity_residual,
    nan_inf_count,
    tke_positivity,
    water_budget_residual,
)


config.update("jax_enable_x64", True)

PERF_DIR = ROOT / "artifacts" / "m6" / "performance"
DEFAULT_OUTPUT = PERF_DIR / "full_domain_batching_verdict.json"
DEFAULT_FORECAST_OUTPUT_DIR = TMP_ROOT / "outputs" / "m6" / "full_domain_batching"
DEFAULT_BISECTION_OUTPUT_DIR = PERF_DIR / "empirical_bisection"
BINDING_CPU_DENOMINATOR = ROOT / "artifacts" / "m6" / "cpu_denominator.json"
CPU_DENOMINATOR_V2 = ROOT / "artifacts" / "m6" / "cpu_denominator_v2.json"
SPEEDUP_GATE = 4.0
TIER2_THRESHOLDS = {
    "dry_mass_max_abs_kg_m2": 1.0e-10,
    "total_water_domain_mean_abs_kg_kg": 1.0e-8,
    "hydrometeor_negative_count": 0,
    "tke_negative_count": 0,
    "nan_inf_count": 0,
}


def _artifact_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _binding_cpu_denominator() -> dict[str, Any]:
    data = json.loads(BINDING_CPU_DENOMINATOR.read_text(encoding="utf-8"))
    raw = data["raw_timing_summary"]
    exact = float(raw["2"]["sum_s"] - raw["3"]["sum_s"] - raw["4"]["sum_s"] - raw["5"]["sum_s"])
    rounded = round(exact, 2)
    payload: dict[str, Any] = {
        "cpu_wall_s": rounded,
        "raw_timing_subtraction_s_exact": exact,
        "basis": "raw_timing_subtraction",
        "artifact": _artifact_path(BINDING_CPU_DENOMINATOR),
        "rationale": (
            "M6-S5 binds the M6-S2a raw measured d02 self-time subtraction because it is actual "
            "timing work observed in the nested Gen2 run; max_dom=2 rebuild was not requested for this sprint."
        ),
        "fp_precision_evidence": data.get(
            "fp_precision",
            "FP32 default real observed (-r4); GPU FP32-gated per ADR-007 storage",
        ),
    }
    if CPU_DENOMINATOR_V2.exists():
        v2 = json.loads(CPU_DENOMINATOR_V2.read_text(encoding="utf-8"))
        payload["v2_grid_points_attributed_context_s"] = float(v2["wall_time_d02_attributable_s"])
        payload["v2_artifact_context"] = _artifact_path(CPU_DENOMINATOR_V2)
    return payload


def _forecast_segments(hours: float, dt_s: float) -> list[dict[str, int | float]]:
    total_steps = steps_for_hours(hours, dt_s)
    previous_steps = 0
    segments = []
    for lead in forecast_output_leads(hours):
        target_steps = steps_for_hours(lead, dt_s)
        segment_steps = target_steps - previous_steps
        if segment_steps > 0:
            segments.append(
                {
                    "lead_hours": float(lead),
                    "start_step": int(previous_steps),
                    "segment_steps": int(segment_steps),
                    "target_steps": int(target_steps),
                    "total_steps": int(total_steps),
                }
            )
        previous_steps = target_steps
    return segments


def _hlo_op_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if "=" in line and "(" in line and not line.lstrip().startswith("#"))


def _compile_forecast_segments(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    *,
    segments: list[dict[str, int | float]],
    dt_s: float,
    n_acoustic: int,
    radiation_cadence_steps: int,
    final_radiation: bool,
    boundary_config: BoundaryConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    compiled_segments: list[dict[str, Any]] = []
    compile_records: list[dict[str, Any]] = []
    compile_wall_s = 0.0
    for segment in segments:
        start = time.perf_counter()
        lowered = run_forecast_segment.lower(
            state,
            tendencies,
            grid,
            dt_s,
            int(segment["segment_steps"]),
            start_step=int(segment["start_step"]),
            total_steps=int(segment["total_steps"]),
            n_acoustic=n_acoustic,
            radiation_cadence_steps=radiation_cadence_steps,
            final_radiation=final_radiation,
            boundary_config=boundary_config,
        )
        compiled = lowered.compile()
        elapsed = time.perf_counter() - start
        compile_wall_s += elapsed
        text = compiled_text(compiled)
        memory = compiled_memory_stats(compiled)
        compile_records.append(
            {
                "lead_hours": float(segment["lead_hours"]),
                "start_step": int(segment["start_step"]),
                "segment_steps": int(segment["segment_steps"]),
                "compile_wall_s": float(elapsed),
                "op_count": int(_hlo_op_count(text)),
                "kernel_launches": int(kernel_launches_per_step(text)),
                "hlo_size_bytes": int(len(text.encode("utf-8"))),
                "memory_analysis": memory,
            }
        )
        compiled_segments.append({"segment": segment, "compiled": compiled})
    metric = {
        "compile_records": compile_records,
        "op_count": int(max((record["op_count"] for record in compile_records), default=0)),
        "kernel_launches": int(max((record["kernel_launches"] for record in compile_records), default=0)),
        "hlo_size_bytes": int(max((record["hlo_size_bytes"] for record in compile_records), default=0)),
        "temp_peak_bytes": int(max((record["memory_analysis"].get("temporary_bytes") or 0 for record in compile_records), default=0)),
    }
    return compiled_segments, metric, compile_wall_s


def _run_forecast_and_write_outputs(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    *,
    compiled_segments: list[dict[str, Any]],
    output_dir: Path,
    run_label: str,
) -> tuple[State, list[dict[str, Any]], float, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current = state
    outputs: list[dict[str, Any]] = []
    forecast_wall_s = 0.0
    output_write_wall_s = 0.0
    for record in compiled_segments:
        segment = record["segment"]
        start = time.perf_counter()
        current = record["compiled"](current, tendencies)
        block_until_ready(current)
        forecast_wall_s += time.perf_counter() - start

        lead = float(segment["lead_hours"])
        path = output_dir / f"wrfout_gpu_d02_p{int(round(lead)):03d}h.npz"
        write_start = time.perf_counter()
        write_wrfout_gpu(path, current, grid, lead_hours=lead, run_start_label=run_label)
        output_write_wall_s += time.perf_counter() - write_start
        outputs.append({"lead_hours": lead, "path": _artifact_path(path)})
    return current, outputs, forecast_wall_s, output_write_wall_s


def _write_output_manifest(path: Path, *, hours: float, dt_s: float, outputs: list[dict[str, Any]]) -> None:
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


def _trace_transfers(run_once, trace_dir: Path) -> dict[str, Any]:
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
    raw_files = _profiler_files(trace_dir)
    return {
        "host_to_device_bytes": int(h2d),
        "device_to_host_bytes": int(d2h),
        "host_device_transfer_bytes": int(h2d + d2h),
        "method": "jax.profiler.trace over warmed lifted-cap forecast audit segment",
        "trace_dir": _artifact_path(trace_dir),
        "trace_transfer_event_files": [_artifact_path(item) for item in files],
        "raw_profiler_files": raw_files,
    }


def _profiler_files(trace_dir: Path) -> list[str]:
    suffixes = (".pb", ".gz", ".json", ".trace")
    return [_artifact_path(path) for path in sorted(trace_dir.rglob("*")) if path.is_file() and path.suffix in suffixes]


def _cache_size_bytes() -> int:
    candidates = []
    for env_name in ("JAX_COMPILATION_CACHE_DIR", "XLA_CACHE_DIR"):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.append(Path.home() / ".cache" / "jax")
    for path in candidates:
        if path.exists():
            return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
    return 0


def _allocator_fragmentation() -> tuple[float, dict[str, Any]]:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        return 0.0, {"method": "no GPU device reported by JAX"}
    try:
        stats = devices[0].memory_stats()
    except Exception as exc:  # pragma: no cover - backend-specific.
        return 0.0, {"method": f"memory_stats unavailable: {type(exc).__name__}: {exc}"}
    pool = int(stats.get("pool_bytes") or stats.get("bytes_reserved") or 0)
    in_use = int(stats.get("bytes_in_use") or 0)
    largest_free = int(stats.get("largest_free_block_bytes") or 0)
    free = max(pool - in_use, 0)
    if free > 0 and largest_free > 0:
        fragmentation = max(0.0, min(1.0, 1.0 - (float(largest_free) / float(free))))
    else:
        fragmentation = 0.0
    return float(fragmentation), {"method": "1 - largest_free_block/free_pool from JAX memory_stats", "raw": stats}


def _sum_stats(left: SanitizeStats, right: SanitizeStats) -> SanitizeStats:
    return SanitizeStats(
        nonfinite_count=left.nonfinite_count + right.nonfinite_count,
        clip_count=left.clip_count + right.clip_count,
        changed_count=left.changed_count + right.changed_count,
        total_count=left.total_count + right.total_count,
    )


def _empty_stats() -> SanitizeStats:
    zero = jnp.asarray(0, dtype=jnp.int64)
    return SanitizeStats(zero, zero, zero, zero)


@partial(jax.jit, static_argnames=("grid", "dt_s", "n_acoustic", "run_radiation", "boundary_config"))
def _one_step_sanitize_stats(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
):
    return coupled_timestep_with_sanitize_stats(
        state,
        tendencies,
        grid,
        dt_s,
        global_step,
        n_acoustic=n_acoustic,
        run_radiation=run_radiation,
        boundary_config=boundary_config,
    )


@partial(jax.jit, static_argnames=("grid", "dt_s", "n_acoustic", "run_radiation", "boundary_config"))
def _one_step_legacy_capped_sanitize_stats(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
):
    next_state = dycore_step(state, tendencies, grid, 1.0, n_acoustic=n_acoustic, debug=False)
    next_state = thompson_adapter(next_state, dt_s)
    next_state = mynn_adapter(next_state, dt_s, grid)
    next_state = surface_adapter(next_state, dt_s)
    if run_radiation:
        next_state = rrtmg_adapter(next_state, dt_s, grid)
    lead_seconds = global_step.astype(jnp.float64) * float(dt_s)
    candidate = apply_lateral_boundaries(next_state, lead_seconds, dt_s, boundary_config)
    return sanitize_state_with_stats(candidate, state)


def _run_sanitize_audit(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    *,
    hours: float,
    dt_s: float,
    n_acoustic: int,
    radiation_cadence_steps: int,
    final_radiation: bool,
    boundary_config: BoundaryConfig,
    legacy_capped: bool,
    block_every_steps: int,
) -> dict[str, Any]:
    steps = steps_for_hours(hours, dt_s)
    current = state
    totals = _empty_stats()
    fired_steps = jnp.asarray(0, dtype=jnp.int64)
    runner = _one_step_legacy_capped_sanitize_stats if legacy_capped else _one_step_sanitize_stats
    start = time.perf_counter()
    for index in range(steps):
        step_number = index + 1
        run_radiation = step_number % int(radiation_cadence_steps) == 0
        if final_radiation and step_number == steps:
            run_radiation = True
        current, stats = runner(
            current,
            tendencies,
            grid,
            dt_s,
            jnp.asarray(step_number, dtype=jnp.int32),
            n_acoustic=n_acoustic,
            run_radiation=run_radiation,
            boundary_config=boundary_config,
        )
        totals = _sum_stats(totals, stats)
        fired_steps = fired_steps + (stats.changed_count > 0).astype(jnp.int64)
        if block_every_steps > 0 and step_number % block_every_steps == 0:
            block_until_ready(current)
            block_until_ready(totals)
    block_until_ready(current)
    block_until_ready(totals)
    block_until_ready(fired_steps)
    elapsed = time.perf_counter() - start
    changed = int(np.asarray(totals.changed_count))
    total = int(np.asarray(totals.total_count))
    fired = int(np.asarray(fired_steps))
    return {
        "mode": "legacy_capped_m6_s2_baseline" if legacy_capped else "lifted_cap_path_b",
        "hours": float(hours),
        "dt_s": float(dt_s),
        "steps": int(steps),
        "wall_s": float(elapsed),
        "nonfinite_count": int(np.asarray(totals.nonfinite_count)),
        "clip_count": int(np.asarray(totals.clip_count)),
        "changed_count": changed,
        "total_checked_values": total,
        "value_firing_rate": float(changed / total) if total else 0.0,
        "fired_steps": fired,
        "step_firing_rate": float(fired / steps) if steps else 0.0,
        "all_state_leaves_finite": bool(state_diagnostics(current)["all_state_leaves_finite"]),
    }


@partial(jax.jit, static_argnames=("grid", "dt_s", "n_acoustic", "run_radiation", "boundary_config"))
def _one_step_tap(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
):
    return coupled_timestep_with_pre_sanitize(
        state,
        tendencies,
        grid,
        dt_s,
        global_step,
        n_acoustic=n_acoustic,
        run_radiation=run_radiation,
        boundary_config=boundary_config,
    )


def _slice_tree(tree, index: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], tree)


def _total_water_column(state: State):
    return sum(getattr(state, field) for field in HYDROMETEOR_FIELDS).mean(axis=0)


def _total_water_outflow_closure(previous: State, candidate: State):
    return _total_water_column(previous) - _total_water_column(candidate)


def _tier2_step_record(previous, candidate, pre_boundary, boundary_tendency, *, step_index: int, dt_s: float) -> dict[str, Any]:
    dry = dry_mass_residual(previous, candidate, dt_s)
    mu = mu_continuity_residual(previous, candidate, dt_s, {"mu_tendency": boundary_tendency.mu})
    water = water_budget_residual(previous, candidate, dt_s, _total_water_outflow_closure(previous, candidate))
    tke = tke_positivity(candidate)
    hydro = hydrometeor_positivity(candidate)
    finite = nan_inf_count(candidate)
    boundary = boundary_flux_closure(
        previous,
        candidate,
        dt_s,
        {"pre_boundary": pre_boundary, "tendency": boundary_tendency},
    )
    water_scale = max(abs(water["domain_mean_abs"]), np.finfo(np.float64).eps)
    return {
        "step": int(step_index),
        "lead_seconds": float((step_index + 1) * dt_s),
        "dry_mass": dry,
        "mu_continuity": mu,
        "water_budget": water,
        "tke_positivity": tke,
        "hydrometeor_positivity": hydro,
        "nan_inf": finite,
        "boundary_flux_closure": boundary,
        "budget_closure_ratio": {
            "water_domain_mean_abs_over_scale": float(abs(water["domain_mean_abs"]) / water_scale),
            "boundary_max_abs": float(boundary["max_abs"]),
        },
    }


def _summarize_tier2(per_step: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    dry_max = max(step["dry_mass"]["max_abs"] for step in per_step) if per_step else 0.0
    water_max = max(step["water_budget"]["domain_mean_abs"] for step in per_step) if per_step else 0.0
    hydro_bad = sum(step["hydrometeor_positivity"]["violations"] for step in per_step)
    tke_bad = sum(step["tke_positivity"]["violations"] for step in per_step)
    nonfinite = sum(step["nan_inf"]["violations"] for step in per_step)
    boundary_max = max(step["boundary_flux_closure"]["max_abs"] for step in per_step) if per_step else 0.0
    budgets = {
        "dry_mass": {"max_abs": float(dry_max), "units": "kg m-2"},
        "total_water": {"max_domain_mean_abs": float(water_max), "units": "kg kg-1"},
        "hydrometeor_positivity": {"negative_count": int(hydro_bad)},
        "tke_positivity": {"negative_count": int(tke_bad)},
        "nan_inf": {"count": int(nonfinite)},
        "boundary_flux_closure": {"max_abs": float(boundary_max), "units": "native field units"},
    }
    threshold_results = {
        "dry_mass": {
            "threshold": TIER2_THRESHOLDS["dry_mass_max_abs_kg_m2"],
            "observed": float(dry_max),
            "pass": bool(dry_max < TIER2_THRESHOLDS["dry_mass_max_abs_kg_m2"]),
        },
        "total_water": {
            "threshold": TIER2_THRESHOLDS["total_water_domain_mean_abs_kg_kg"],
            "observed": float(water_max),
            "pass": bool(water_max < TIER2_THRESHOLDS["total_water_domain_mean_abs_kg_kg"]),
        },
        "hydrometeor_positivity": {
            "threshold": TIER2_THRESHOLDS["hydrometeor_negative_count"],
            "observed": int(hydro_bad),
            "pass": bool(hydro_bad == 0),
        },
        "tke_positivity": {
            "threshold": TIER2_THRESHOLDS["tke_negative_count"],
            "observed": int(tke_bad),
            "pass": bool(tke_bad == 0),
        },
        "nan_inf": {
            "threshold": TIER2_THRESHOLDS["nan_inf_count"],
            "observed": int(nonfinite),
            "pass": bool(nonfinite == 0),
        },
    }
    status = "PASS" if all(item["pass"] for item in threshold_results.values()) else "FAIL"
    return budgets, threshold_results, status


def _run_tier2_lifted_cap_audit(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    gen2: Gen2Run,
    meta: dict[str, Any],
    *,
    output: Path,
    hours: float,
    dt_s: float,
    n_acoustic: int,
    radiation_cadence_steps: int,
    final_radiation: bool,
    boundary_config: BoundaryConfig,
) -> dict[str, Any]:
    steps = steps_for_hours(hours, dt_s)
    previous = state
    current = state
    per_step: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index in range(steps):
        step_number = index + 1
        run_radiation = step_number % int(radiation_cadence_steps) == 0
        if final_radiation and step_number == steps:
            run_radiation = True
        current, tap = _one_step_tap(
            current,
            tendencies,
            grid,
            dt_s,
            jnp.asarray(step_number, dtype=jnp.int32),
            n_acoustic=n_acoustic,
            run_radiation=run_radiation,
            boundary_config=boundary_config,
        )
        block_until_ready(current)
        block_until_ready(tap.state)
        per_step.append(
            _tier2_step_record(
                previous,
                tap.state,
                tap.pre_boundary,
                tap.boundary_tendency,
                step_index=index,
                dt_s=dt_s,
            )
        )
        previous = current
    wall_s = time.perf_counter() - start
    budgets, thresholds, status = _summarize_tier2(per_step)
    payload = {
        "artifact_type": "tier2_coupled_invariants",
        "created_utc": _now_utc(),
        "run_id": gen2.run_id,
        "domain": "d02",
        "status": status,
        "dt_s": float(dt_s),
        "hours": float(hours),
        "steps": int(steps),
        "dycore_cap_status": "lifted_via_path_b",
        "audit_wall_s": float(wall_s),
        "grid": meta["grid"],
        "sanitize_policy": {
            "mode": "pre_sanitize_tap_one_step_streaming",
            "tap_steps": int(steps),
            "measured_state": "candidate State after boundary replay and before sanitize_state(candidate, previous)",
            "post_step_sanitize_used_only_to_reconstruct_next_step_carry": True,
        },
        "budgets": budgets,
        "thresholds": thresholds,
        "boundary_terms": {
            "source": "Streaming one-step PreSanitizeTap under M6-S5 Path-B dt",
            "config": {
                "spec_bdy_width": int(boundary_config.spec_bdy_width),
                "spec_zone": int(boundary_config.spec_zone),
                "relax_zone": int(boundary_config.relax_zone),
                "update_cadence_s": float(boundary_config.update_cadence_s),
                "spec_exp": float(boundary_config.spec_exp),
            },
        },
        "per_step": per_step,
        "artifact_paths": [_artifact_path(output)],
    }
    Tier2CoupledInvariants.validate_dict(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _run_nsys_profile(args: argparse.Namespace) -> dict[str, Any]:
    nsys = shutil.which("nsys")
    profile_dir = PERF_DIR / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    log_path = profile_dir / "m6_s5_nsys_audit.log"
    if nsys is None:
        log_path.write_text("nsys not found on PATH\n", encoding="utf-8")
        return {"available": False, "paths": [], "log": _artifact_path(log_path), "status": "missing"}
    output_base = profile_dir / "m6_s5_nsys_audit"
    command = [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--output",
        str(output_base),
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-child",
        "--profile-steps",
        str(args.profile_steps),
        "--dt-s",
        str(args.dt_s),
        "--n-acoustic",
        str(args.n_acoustic),
        "--radiation-cadence-steps",
        str(args.radiation_cadence_steps),
        "--run-dir",
        str(args.run_dir),
        "--boundary",
        str(args.boundary),
        "--spec-bdy-width",
        str(args.spec_bdy_width),
        "--spec-zone",
        str(args.spec_zone),
        "--relax-zone",
        str(args.relax_zone),
        "--spec-exp",
        str(args.spec_exp),
    ]
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    paths = [
        _artifact_path(path)
        for path in sorted(profile_dir.glob("m6_s5_nsys_audit*"))
        if path.is_file() and path != log_path
    ]
    return {
        "available": True,
        "status": "ok" if proc.returncode == 0 and paths else "failed",
        "returncode": int(proc.returncode),
        "paths": paths,
        "log": _artifact_path(log_path),
        "command": " ".join(command),
    }


def _profile_child(args: argparse.Namespace) -> int:
    validate_lifted_coupled_dt(args.dt_s)
    boundary = _resolve_path(args.boundary)
    gen2 = Gen2Run(Path(args.run_dir))
    state, tendencies, grid, _meta = build_initial_state(gen2, domain="d02", boundary_path=boundary)
    boundary_config = _boundary_config(args)
    result = run_forecast_segment(
        state,
        tendencies,
        grid,
        args.dt_s,
        int(args.profile_steps),
        start_step=0,
        total_steps=int(args.profile_steps),
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=True,
        boundary_config=boundary_config,
    )
    block_until_ready(result)
    print(json.dumps({"status": "ok", "profile_steps": int(args.profile_steps)}, sort_keys=True))
    return 0


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _boundary_config(args: argparse.Namespace) -> BoundaryConfig:
    return BoundaryConfig(
        spec_bdy_width=args.spec_bdy_width,
        spec_zone=args.spec_zone,
        relax_zone=args.relax_zone,
        update_cadence_s=3600.0,
        spec_exp=args.spec_exp,
    )


def _bisection_config(args: argparse.Namespace) -> BisectionConfig:
    disable_physics = bool(args.disable_physics)
    return BisectionConfig(
        disable_sanitize=bool(args.disable_sanitize),
        disable_thompson=disable_physics or bool(args.disable_thompson),
        disable_mynn=disable_physics or bool(args.disable_mynn),
        disable_surface=disable_physics or bool(args.disable_surface),
        disable_rrtmg=disable_physics or bool(args.disable_rrtmg),
        disable_boundary=bool(args.disable_boundary),
        disable_advection=bool(args.disable_advection),
        disable_acoustic=bool(args.disable_acoustic),
        disable_mu_continuity=bool(args.disable_mu_continuity),
    )


def _bisection_probe_requested(args: argparse.Namespace) -> bool:
    return any(
        bool(getattr(args, name))
        for name in (
            "bisection_probe",
            "disable_sanitize",
            "disable_physics",
            "disable_thompson",
            "disable_mynn",
            "disable_surface",
            "disable_rrtmg",
            "disable_boundary",
            "disable_advection",
            "disable_acoustic",
            "disable_mu_continuity",
        )
    )


def _bisection_label(args: argparse.Namespace, config: BisectionConfig) -> str:
    if args.probe_label:
        return str(args.probe_label)
    disabled = [name.removeprefix("disable_") for name, value in config._asdict().items() if value]
    return "baseline" if not disabled else "no_" + "_".join(disabled)


def _component_subset(config: BisectionConfig) -> dict[str, Any]:
    physics = []
    if not config.disable_thompson:
        physics.append("thompson")
    if not config.disable_mynn:
        physics.append("mynn")
    if not config.disable_surface:
        physics.append("surface")
    if not config.disable_rrtmg:
        physics.append("rrtmg")
    return {
        "dycore": {
            "advection": not bool(config.disable_advection),
            "acoustic": not bool(config.disable_acoustic),
            "mu_continuity": not bool(config.disable_mu_continuity),
        },
        "physics": physics,
        "boundary": not bool(config.disable_boundary),
        "sanitize": not bool(config.disable_sanitize),
    }


@jax.jit
def _state_nonfinite_count(state: State):
    leaves = jax.tree_util.tree_leaves(state)
    return sum((jnp.sum(~jnp.isfinite(leaf), dtype=jnp.int64) for leaf in leaves), jnp.asarray(0, dtype=jnp.int64))


def _state_nonfinite_summary(state: State) -> dict[str, Any]:
    by_field = {}
    total = 0
    for name in State.__slots__:
        value = getattr(state, name)
        count = int(np.asarray(jnp.sum(~jnp.isfinite(value), dtype=jnp.int64)))
        if count:
            by_field[name] = count
            total += count
    return {"nonfinite_count": int(total), "nonfinite_by_field": by_field}


@partial(
    jax.jit,
    static_argnames=("grid", "dt_s", "n_acoustic", "run_radiation", "boundary_config", "bisection_config"),
)
def _one_step_bisection(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig,
):
    return coupled_timestep_with_pre_sanitize(
        state,
        tendencies,
        grid,
        dt_s,
        global_step,
        n_acoustic=n_acoustic,
        run_radiation=run_radiation,
        boundary_config=boundary_config,
        bisection_config=bisection_config,
    )


def _run_bisection_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_lifted_coupled_dt(args.dt_s)
    boundary = _resolve_path(args.boundary)
    boundary_config = _boundary_config(args)
    config = _bisection_config(args)
    label = _bisection_label(args, config)
    output = _resolve_path(args.bisection_output) if args.bisection_output else DEFAULT_BISECTION_OUTPUT_DIR / f"{label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    gen2 = Gen2Run(Path(args.run_dir))
    state, tendencies, grid, meta = build_initial_state(gen2, domain="d02", boundary_path=boundary)
    steps = steps_for_hours(args.hours, args.dt_s)
    current = state
    first_nonfinite_step = None
    first_nonfinite_summary = None
    samples: list[dict[str, Any]] = []
    start = time.perf_counter()
    completed = 0
    for index in range(steps):
        step_number = index + 1
        run_radiation = step_number % int(args.radiation_cadence_steps) == 0
        if args.final_radiation and step_number == steps:
            run_radiation = True
        current, tap = _one_step_bisection(
            current,
            tendencies,
            grid,
            args.dt_s,
            jnp.asarray(step_number, dtype=jnp.int32),
            n_acoustic=args.n_acoustic,
            run_radiation=run_radiation,
            boundary_config=boundary_config,
            bisection_config=config,
        )
        block_until_ready(current)
        nonfinite_count = int(np.asarray(_state_nonfinite_count(tap.state)))
        completed = step_number
        if (
            step_number <= 3
            or nonfinite_count > 0
            or (args.probe_log_interval > 0 and step_number % int(args.probe_log_interval) == 0)
        ):
            samples.append(
                {
                    "step": int(step_number),
                    "run_radiation": bool(run_radiation),
                    "raw_candidate_nonfinite_count": int(nonfinite_count),
                }
            )
        if nonfinite_count > 0:
            first_nonfinite_step = step_number
            first_nonfinite_summary = _state_nonfinite_summary(tap.state)
            break
    block_until_ready(current)
    wall_s = time.perf_counter() - start
    payload = {
        "artifact_type": "m6x_empirical_bisection_probe",
        "created_utc": _now_utc(),
        "label": label,
        "run_id": gen2.run_id,
        "domain": "d02",
        "dt_s": float(args.dt_s),
        "hours_requested": float(args.hours),
        "steps_requested": int(steps),
        "steps_completed": int(completed),
        "n_acoustic": int(args.n_acoustic),
        "radiation_cadence_steps": int(args.radiation_cadence_steps),
        "component_subset": _component_subset(config),
        "bisection_config": dict(config._asdict()),
        "status": "nonfinite" if first_nonfinite_step is not None else "finite_through_limit",
        "first_nonfinite_step": None if first_nonfinite_step is None else int(first_nonfinite_step),
        "first_nonfinite_summary": first_nonfinite_summary,
        "samples": samples,
        "wall_s": float(wall_s),
        "grid": meta["grid"],
        "boundary_path": _artifact_path(boundary),
        "artifact_paths": [_artifact_path(output)],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _diagnostics_hit_sanitize_bounds(diagnostics: dict[str, Any]) -> bool:
    return bool(
        diagnostics.get("theta_min_k") == 150.0
        or diagnostics.get("theta_max_k") == 550.0
        or diagnostics.get("qv_max_kg_kg") >= 0.05
        or diagnostics.get("p_min_pa") == 1000.0
        or diagnostics.get("p_max_pa") == 120000.0
        or diagnostics.get("u_abs_max_m_s") == 150.0
        or diagnostics.get("v_abs_max_m_s") == 150.0
        or diagnostics.get("w_abs_max_m_s") == 50.0
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_lifted_coupled_dt(args.dt_s)
    output = _resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast_output_dir = _resolve_path(args.output_dir)
    boundary = _resolve_path(args.boundary)
    boundary_config = _boundary_config(args)
    nsys_profile = {"status": "skipped", "paths": []}
    if not args.skip_nsys:
        nsys_profile = _run_nsys_profile(args)
    gen2 = Gen2Run(Path(args.run_dir))
    total_steps = steps_for_hours(args.hours, args.dt_s)
    segments = _forecast_segments(args.hours, args.dt_s)

    cold_start_begin = time.perf_counter()
    state, tendencies, grid, meta = build_initial_state(gen2, domain="d02", boundary_path=boundary)
    block_until_ready(state)
    block_until_ready(tendencies)
    cold_start_wall_s = time.perf_counter() - cold_start_begin

    compiled_segments, compile_metric, compile_wall_s = _compile_forecast_segments(
        state,
        tendencies,
        grid,
        segments=segments,
        dt_s=args.dt_s,
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=args.final_radiation,
        boundary_config=boundary_config,
    )
    final_state, outputs, forecast_wall_s, output_write_wall_s = _run_forecast_and_write_outputs(
        state,
        tendencies,
        grid,
        compiled_segments=compiled_segments,
        output_dir=forecast_output_dir,
        run_label=run_start_label(gen2, "d02"),
    )
    total_end_to_end_wall_s = float(cold_start_wall_s + compile_wall_s + forecast_wall_s + output_write_wall_s)

    manifest_path = output.with_suffix(".outputs.json")
    _write_output_manifest(manifest_path, hours=args.hours, dt_s=args.dt_s, outputs=outputs)
    diagnostics = state_diagnostics(final_state)

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
    transfer_audit = _trace_transfers(audit_run, TMP_ROOT / "trace_m6_s5_full_domain_batching")
    profiler_raw_paths = list(transfer_audit["raw_profiler_files"])

    lifted_sanitize = _run_sanitize_audit(
        state,
        tendencies,
        grid,
        hours=args.hours,
        dt_s=args.dt_s,
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=args.final_radiation,
        boundary_config=boundary_config,
        legacy_capped=False,
        block_every_steps=args.audit_block_steps,
    )
    legacy_sanitize = None
    if args.legacy_baseline_sanitize_audit:
        legacy_sanitize = _run_sanitize_audit(
            state,
            tendencies,
            grid,
            hours=args.hours,
            dt_s=60.0,
            n_acoustic=args.n_acoustic,
            radiation_cadence_steps=10,
            final_radiation=args.final_radiation,
            boundary_config=boundary_config,
            legacy_capped=True,
            block_every_steps=args.audit_block_steps,
        )

    tier2_path = PERF_DIR / "tier2_lifted_cap_invariants.json"
    tier2 = _run_tier2_lifted_cap_audit(
        state,
        tendencies,
        grid,
        gen2,
        meta,
        output=tier2_path,
        hours=args.tier2_hours,
        dt_s=args.dt_s,
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=True,
        boundary_config=boundary_config,
    )
    tier2_status = str(tier2["status"]).lower()

    profiler_raw_paths.extend(nsys_profile.get("paths", []))

    cpu = _binding_cpu_denominator()
    cpu_wall_s = float(cpu["cpu_wall_s"])
    speedup_ratio = float(cpu_wall_s / total_end_to_end_wall_s) if total_end_to_end_wall_s > 0 else 0.0
    sanitize_le_baseline = (
        legacy_sanitize is not None
        and lifted_sanitize["step_firing_rate"] <= legacy_sanitize["step_firing_rate"]
        and lifted_sanitize["value_firing_rate"] <= legacy_sanitize["value_firing_rate"]
    )
    pass_gate = bool(speedup_ratio >= SPEEDUP_GATE and tier2_status == "pass")
    verdict_reason = []
    if speedup_ratio < SPEEDUP_GATE:
        verdict_reason.append(
            "speedup below 4x; Path-B lifted cap increases dycore/physics step count relative to the capped M6-S2 audit"
        )
    if tier2_status != "pass":
        verdict_reason.append("Tier-2 lifted-cap invariant audit failed")
    if not diagnostics["all_state_leaves_finite"]:
        verdict_reason.append("24h final state contains non-finite leaves")
    if _diagnostics_hit_sanitize_bounds(diagnostics):
        verdict_reason.append(
            "24h lifted-cap final state saturated finite-guard clip bounds; reduced M4 dycore is unstable at Path-B dt=10s"
        )
    if legacy_sanitize is not None and not sanitize_le_baseline:
        verdict_reason.append("lifted-cap sanitize firing rate exceeded legacy capped baseline")
    fragmentation, fragmentation_detail = _allocator_fragmentation()
    artifact_paths = [
        _artifact_path(output),
        _artifact_path(manifest_path),
        _artifact_path(tier2_path),
        *[item["path"] for item in outputs],
        *profiler_raw_paths,
    ]
    payload = {
        "artifact_type": "full_domain_batching_verdict",
        "created_utc": _now_utc(),
        "benchmark": "m6_s5_full_domain_batching_24h",
        "backend": "jax",
        "hardware": visible_gpu_name(),
        "case": "gen2-d02-real-ic-boundary-replay-lifted-cap-path-b",
        "run_id": gen2.run_id,
        "domain": "d02",
        "dt_s": float(args.dt_s),
        "steps": int(total_steps),
        "radiation_cadence_steps": int(args.radiation_cadence_steps),
        "speedup_ratio": speedup_ratio,
        "speedup_ratio_definition": "cpu_wall_s / gpu_wall_s",
        "speedup_gate": SPEEDUP_GATE,
        "pass": pass_gate,
        "verdict": "PASS" if pass_gate else "FAIL",
        "verdict_reason": verdict_reason or ["speedup and Tier-2 gates passed"],
        "gpu_wall_s": total_end_to_end_wall_s,
        "wall_time_s": total_end_to_end_wall_s,
        "cold_start_wall_s": float(cold_start_wall_s),
        "compile_wall_s": float(compile_wall_s),
        "coupled_forecast_24h_wall_s": float(forecast_wall_s),
        "output_write_wall_s": float(output_write_wall_s),
        "total_end_to_end_wall_s": total_end_to_end_wall_s,
        "cpu_wall_s": cpu_wall_s,
        "cpu_denominator_basis": cpu["basis"],
        "cpu_denominator_artifact": cpu["artifact"],
        "cpu_denominator_rationale": cpu["rationale"],
        "cpu_denominator": cpu,
        "fp_precision_caveat": "WRF -r4 default real; GPU FP32-gated per ADR-007 storage",
        "dycore_cap_status": "lifted_via_path_b",
        "dycore_cap_lift": {
            "path": "path_b",
            "coupled_dt_s": float(args.dt_s),
            "max_allowed_dt_s": float(MAX_LIFTED_DYCORE_DT_S),
            "policy": "dycore receives dt_s directly; 60s coupled calls raise instead of being capped",
        },
        "tier2_invariants_under_lifted_cap": tier2_status,
        "tier2_artifact": _artifact_path(tier2_path),
        "stability_evidence": {
            "final_state_diagnostics": diagnostics,
            "lifted_cap_sanitize_audit": lifted_sanitize,
            "legacy_capped_baseline_sanitize_audit": legacy_sanitize,
            "sanitize_firing_rate_lte_baseline": bool(sanitize_le_baseline) if legacy_sanitize is not None else None,
        },
        "profiler_raw_paths": profiler_raw_paths,
        "profiler": {
            "jax_trace": transfer_audit,
            "nsys": nsys_profile,
        },
        "transfer_audit": transfer_audit,
        "host_device_transfer_bytes": int(transfer_audit["host_device_transfer_bytes"]),
        "op_count": int(compile_metric["op_count"]),
        "kernel_launches": int(compile_metric["kernel_launches"]),
        "hlo_size_bytes": int(compile_metric["hlo_size_bytes"]),
        "temp_peak_bytes": int(compile_metric["temp_peak_bytes"]),
        "compile_retries": 0,
        "cache_size_bytes": _cache_size_bytes(),
        "allocator_fragmentation": fragmentation,
        "allocator_fragmentation_detail": fragmentation_detail,
        "compile_records": compile_metric["compile_records"],
        "output_manifest": _artifact_path(manifest_path),
        "artifact_paths": artifact_paths,
    }
    FullDomainBatchingVerdict.validate_dict(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, required=False, default=24.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-dir", default=str(DEFAULT_M6_GEN2_RUN_DIR))
    parser.add_argument("--boundary", default=str(DEFAULT_M6_BOUNDARY_REPLAY))
    parser.add_argument("--output-dir", default=str(DEFAULT_FORECAST_OUTPUT_DIR))
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--n-acoustic", type=int, default=2)
    parser.add_argument("--radiation-cadence-steps", type=int, default=DEFAULT_RADIATION_CADENCE_STEPS)
    parser.add_argument("--audit-steps", type=int, default=10)
    parser.add_argument("--audit-block-steps", type=int, default=120)
    parser.add_argument("--tier2-hours", type=float, default=1.0)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument("--skip-legacy-baseline-sanitize-audit", dest="legacy_baseline_sanitize_audit", action="store_false")
    parser.add_argument("--spec-bdy-width", type=int, default=5)
    parser.add_argument("--spec-zone", type=int, default=1)
    parser.add_argument("--relax-zone", type=int, default=4)
    parser.add_argument("--spec-exp", type=float, default=0.0)
    parser.add_argument("--skip-final-radiation", dest="final_radiation", action="store_false")
    parser.add_argument("--profile-child", action="store_true")
    parser.add_argument("--bisection-probe", action="store_true")
    parser.add_argument("--probe-label")
    parser.add_argument("--bisection-output")
    parser.add_argument("--probe-log-interval", type=int, default=30)
    parser.add_argument("--disable-sanitize", action="store_true")
    parser.add_argument("--disable-physics", action="store_true")
    parser.add_argument("--disable-thompson", action="store_true")
    parser.add_argument("--disable-mynn", action="store_true")
    parser.add_argument("--disable-surface", action="store_true")
    parser.add_argument("--disable-rrtmg", action="store_true")
    parser.add_argument("--disable-boundary", action="store_true")
    parser.add_argument("--disable-advection", action="store_true")
    parser.add_argument("--disable-acoustic", action="store_true")
    parser.add_argument("--disable-mu-continuity", action="store_true")
    parser.set_defaults(final_radiation=True, legacy_baseline_sanitize_audit=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.profile_child:
        return _profile_child(args)
    if _bisection_probe_requested(args):
        payload = _run_bisection_probe(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
