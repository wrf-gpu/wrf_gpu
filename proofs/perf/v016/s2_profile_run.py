#!/usr/bin/env python
"""Warm-run profiler target for ADR-031 S2.

The script compiles and warms the operational forecast, prepares a fresh donated
state, then brackets one warm execution with cudaProfilerStart/Stop.  Use with
``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import jax

from gpuwrf.runtime.operational_mode import run_forecast_operational

from proofs.perf.v016 import fp32_s2_mixed_ladder as ladder


def _cuda_profiler_call(name: str) -> None:
    for libname in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            lib = ctypes.CDLL(libname)
            break
        except OSError:
            lib = None
    if lib is None:
        raise RuntimeError("could not load libcudart for CUDA profiler API")
    fn = getattr(lib, name)
    rc = int(fn())
    if rc != 0:
        raise RuntimeError(f"{name} failed with CUDA status {rc}")


def _parse_tile(raw: str) -> tuple[int, int]:
    return ladder._parse_tiles(raw)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=("fp64", "mixed_s2"), required=True)
    ap.add_argument("--tile", default="2x2")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("proofs/perf/v016/s2_profile_run.json"))
    args = ap.parse_args(argv)

    fy, fx = _parse_tile(args.tile)
    cfg = ladder.DailyPipelineConfig(
        run_id=ladder.ANCHOR_RUN_ID,
        run_root=ladder.ANCHOR_RUN_ROOT,
        domain=ladder.ANCHOR_DOMAIN,
        hours=1,
        dt_s=ladder.ANCHOR_DT_S,
        acoustic_substeps=10,
    )
    case, _ = ladder._build_real_case(cfg)
    ny0, nx0 = int(case.grid.ny), int(case.grid.nx)
    nl = ladder._namelist(case.namelist, ny0, nx0, fy, fx, args.precision)
    hours = float(args.steps) * float(case.namelist.dt_s) / 3600.0

    # Compile + warm outside the profiled range.
    st = ladder._fresh_state(case.state, ny0, nx0, fy, fx, args.precision)
    out = run_forecast_operational(st, nl, hours)
    ladder._block(out)
    st = ladder._fresh_state(case.state, ny0, nx0, fy, fx, args.precision)
    out = run_forecast_operational(st, nl, hours)
    ladder._block(out)

    st = ladder._fresh_state(case.state, ny0, nx0, fy, fx, args.precision)
    _cuda_profiler_call("cudaProfilerStart")
    t0 = time.perf_counter()
    out = run_forecast_operational(st, nl, hours)
    ladder._block(out)
    wall_s = time.perf_counter() - t0
    _cuda_profiler_call("cudaProfilerStop")

    payload = {
        "schema": "ADR031S2WarmProfileRun",
        "precision": args.precision,
        "tile_factor": [fy, fx],
        "ny": fy * ny0,
        "nx": fx * nx0,
        "ncol": fy * fx * ny0 * nx0,
        "steps": int(args.steps),
        "hours": hours,
        "wall_s": wall_s,
        "ms_per_step": wall_s / float(args.steps) * 1000.0,
        "device": str(jax.devices()[0]),
        "finite": bool(jax.numpy.all(jax.numpy.isfinite(out.theta))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
