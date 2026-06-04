"""v0.10.0 Phase 0 warmed nsys driver.

This is measurement-only scaffolding. It builds the real d02 case, performs a
compile/warm call outside the NVTX range, then profiles one warmed forecast call
under a named NVTX range. Use via ``run_phase0_nsys.sh`` so CPU affinity and
JAX/CUDA allocator settings are recorded consistently.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any

import jax

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import run_forecast_operational


def _block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _visible_gpu_devices() -> list[str]:
    return [str(device) for device in jax.devices() if device.platform == "gpu"]


def _make_namelist(case, *, mode: str, disable_guards: bool):
    run_physics = mode == "coupled"
    run_boundary = mode == "coupled"
    return dataclasses.replace(
        case.namelist,
        run_physics=run_physics,
        run_boundary=run_boundary,
        disable_guards=bool(disable_guards),
        radiation_cadence_steps=180,
        time_utc=case.run_start,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("coupled", "dycore"), required=True)
    parser.add_argument("--disable-guards", action="store_true")
    parser.add_argument("--warm-steps", type=int, default=200)
    parser.add_argument("--profile-steps", type=int, default=240)
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    gpus = _visible_gpu_devices()
    if not gpus:
        raise RuntimeError("No JAX GPU backend visible; refusing to produce a fake nsys baseline")

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10, domain=args.domain)
    case, run_dir = _build_real_case(cfg)
    namelist = _make_namelist(case, mode=args.mode, disable_guards=bool(args.disable_guards))
    dt_s = float(namelist.dt_s)
    warm_hours = float(args.warm_steps) * dt_s / 3600.0
    profile_hours = float(args.profile_steps) * dt_s / 3600.0

    # Compile/warm the exact profile executable before entering the NVTX range.
    warm_state = _build_real_case(cfg)[0].state
    warm = run_forecast_operational(warm_state, namelist, profile_hours)
    _block(warm)

    # Additional warm steps at any requested length. This is useful when the
    # profile length differs from the warmup length, but avoids profiling first
    # autotune/allocator artifacts.
    if args.warm_steps != args.profile_steps:
        extra_state = _build_real_case(cfg)[0].state
        extra = run_forecast_operational(extra_state, namelist, warm_hours)
        _block(extra)

    profile_state = _build_real_case(cfg)[0].state
    t0 = time.perf_counter()
    with jax.profiler.TraceAnnotation("V0100_PHASE0_PROFILE"):
        out = run_forecast_operational(profile_state, namelist, profile_hours)
        _block(out)
    wall_s = time.perf_counter() - t0

    payload = {
        "schema": "V0100Phase0NsysDriver",
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "domain": args.domain,
        "disable_guards": bool(args.disable_guards),
        "warm_steps": int(args.warm_steps),
        "profile_steps": int(args.profile_steps),
        "profile_hours": profile_hours,
        "wall_s": wall_s,
        "per_step_ms": wall_s * 1000.0 / float(args.profile_steps),
        "run_dir": str(run_dir),
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "devices": gpus,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "namelist": {
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "disable_guards": bool(namelist.disable_guards),
            "force_fp64": bool(namelist.force_fp64),
            "dt_s": dt_s,
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
