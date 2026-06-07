#!/usr/bin/env python3
"""v0.12.0 standalone-path profiling driver (lightweight, short runs).

Reuses the EXACT production standalone path -- ``_build_real_case`` +
``run_forecast_operational_segmented`` (the same forecast_fn the CLI selects
for a standalone native-init case) -- so the measured numbers are the real
operational path, NOT a hand-rolled reimplementation.

Records, for a short (default 2-h) d01 standalone run:
  * per-forecast-hour wall (hour-1 = warm-compile/case-build; hour 2.. = warm),
  * peak GPU memory via ``device.memory_stats()['peak_bytes_in_use']``,
  * a ranked HLO op-count breakdown of the compiled forecast segment
    (lowered+compiled once, op histogram + fp64/fp32 convert counts), which is
    the cheap "where does the step spend its ops" profiling artifact.

Honesty: this is profiling, NOT validation. Keep ``--hours`` small. The cache
flag / precision are whatever the environment + production path set
(force_fp64=True on the standalone operational path, dt=10s).

Usage (under the GPU lock wrapper)::

    /tmp/wrf_gpu_run.sh taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true \
        XLA_PYTHON_CLIENT_PREALLOCATE=false GPUWRF_JAX_CACHE_DIR=/mnt/data/gpuwrf_jax_cache \
        python proofs/perf/v0120_profile_driver.py \
        --input-dir <case> --domain d01 --hours 2 --out proofs/perf/v0120_profile_run.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path


def _peak_mb() -> float:
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
        return float(stats.get("peak_bytes_in_use", 0)) / (1024.0**2)
    except Exception:
        return float("nan")


def _cur_mb() -> float:
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
        return float(stats.get("bytes_in_use", 0)) / (1024.0**2)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--domain", default="d01")
    ap.add_argument("--hours", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hlo", action="store_true", help="also dump compiled HLO op histogram")
    args = ap.parse_args()

    import jax

    from gpuwrf.integration.daily_pipeline import (
        DailyPipelineConfig,
        _build_real_case,
        _commit_to_operational_device,
        finite_summary,
    )
    from gpuwrf.runtime.compile_cache import CACHE_STATUS
    from gpuwrf.runtime.operational_mode import (
        run_forecast_operational_segmented,
    )

    config = DailyPipelineConfig(
        run_id=str(args.input_dir.resolve()),
        run_root=args.input_dir.parent,
        hours=int(args.hours),
        output_dir=Path("/tmp/_profile_unused_out"),
        proof_dir=Path("/tmp/_profile_unused_proof"),
        domain=args.domain,
    )

    t_build0 = time.perf_counter()
    case, run_dir = _build_real_case(config)
    build_s = time.perf_counter() - t_build0

    state = _commit_to_operational_device(case.state)
    namelist = case.namelist

    # Per-hour wall: hour 1 pays compile (cache read or cold) + first execute.
    per_hour: list[float] = []
    mem_after_hour: list[float] = []
    for hour in range(1, int(args.hours) + 1):
        t0 = time.perf_counter()
        state = run_forecast_operational_segmented(state, namelist, 1.0)
        jax.block_until_ready(state)
        per_hour.append(time.perf_counter() - t0)
        mem_after_hour.append(_peak_mb())

    summary = finite_summary(state)

    # NOTE: the --hlo lowering of run_forecast_operational_segmented is NOT supported
    # (it is a host-loop driver wrapping a jitted inner _advance_chunk; jit-lowering the
    # wrapper hangs). The authoritative warmed-step op breakdown is the Nsight Systems
    # profile already in proofs/perf/nsys_warmed_step_stats_*.csv; see v0120_profile.md.
    # The flag retained as a no-op for back-compat; it records that it was skipped.
    hlo_hist: dict[str, int] = {"skipped": "segmented host-loop entry is not jit-lowerable; see nsys CSVs"} if args.hlo else {}
    convert_counts: dict[str, int] = {}

    result = {
        "schema": "GpuwrfV0120ProfileRun",
        "domain": args.domain,
        "hours": int(args.hours),
        "device": str(jax.devices()[0]),
        "jax_cache_status": dict(CACHE_STATUS),
        "case_build_s": build_s,
        "wall_clock_per_hour_s": per_hour,
        "warm_s_per_forecast_hour": (per_hour[-1] if len(per_hour) >= 2 else None),
        "hour1_compile_plus_first_s": per_hour[0] if per_hour else None,
        "peak_gpu_mem_mb_after_each_hour": mem_after_hour,
        "peak_gpu_mem_mb": max(mem_after_hour) if mem_after_hour else None,
        "force_fp64": bool(namelist.force_fp64),
        "dt_s": float(namelist.dt_s),
        "acoustic_substeps": int(namelist.acoustic_substeps),
        "grid_mass_shape": list(case.state.theta.shape),
        "all_finite": summary["all_finite"],
        "hlo_op_histogram_top40": hlo_hist,
        "hlo_convert_counts": convert_counts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: result[k] for k in (
        "warm_s_per_forecast_hour", "hour1_compile_plus_first_s",
        "peak_gpu_mem_mb", "force_fp64", "all_finite", "grid_mass_shape",
    )}, indent=2))
    return 0 if summary["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
