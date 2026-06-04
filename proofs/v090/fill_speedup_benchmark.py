#!/usr/bin/env python
"""PART-3 piggyback fill: populate proofs/v090/speedup_benchmark.json GPU numbers.

NO separate/extra GPU runs. This reads the wall-clock that the post-qke-fix
VALIDATION runs already record into their pipeline_run JSON (via
gpuwrf.integration.daily_pipeline.execute_daily_pipeline, which times the whole
command-to-finish in wall_clock_total_s and the per-hour breakdown in
wall_clock_per_hour_s), and fills the GPU placeholders + computes the honest
real-user-time speedups against the CPU denominators recovered by
proofs/v090/cpu_wall_recover.py.

Real-user-time (HEADLINE) = compile-INCLUSIVE: the full wall_clock_total_s the
user waits, divided into the CPU full-forecast wall.
Steady-state (CONTEXT) = compile-EXCLUDED: median of per-hour walls AFTER hour 1
(hour 1 carries the one-time JAX compile), scaled to the forecast length.
compile_overhead_s = hour1_wall - steady_state_per_hour_wall.

Usage (manager, AFTER the qke-fix validation runs):
  taskset -c 0-3 python3 proofs/v090/fill_speedup_benchmark.py \
      --d02-pipeline proofs/v090/speedup_d02/pipeline_run_l2_d02.json \
      [--d03-pipeline proofs/v090/speedup_d03/pipeline_run_<id>.json]

Either case may be omitted; only the provided cases are filled.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "speedup_benchmark.json"


def _load(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _gpu_breakdown(pipeline: dict, forecast_h: int) -> dict:
    """Derive compile-inclusive total + steady-state + compile overhead.

    wall_clock_total_s   = full command-to-finish (compile-inclusive). HEADLINE.
    wall_clock_per_hour_s = list of per-forecast-hour walls; [0] holds JAX compile.
    """
    total = pipeline.get("wall_clock_total_s")
    per_hour = list(pipeline.get("wall_clock_per_hour_s") or [])
    n_hours = pipeline.get("hours") or forecast_h

    steady_per_hour = None
    compile_overhead = None
    if len(per_hour) >= 2:
        steady_pool = per_hour[1:]
        steady_per_hour = float(statistics.median(steady_pool))
        compile_overhead = float(per_hour[0] - steady_per_hour)
    elif len(per_hour) == 1:
        # single-hour run: cannot separate compile from steady-state
        steady_per_hour = None
        compile_overhead = None

    # scale measured run to the benchmark forecast length for a like-for-like wall
    total_per_fc_hr = (float(total) / float(n_hours)) if (total and n_hours) else None
    steady_per_fc_hr = steady_per_hour  # per-hour wall == per-forecast-hour wall
    return {
        "measured_hours": int(n_hours) if n_hours else None,
        "gpu_wallclock_total_s": float(total) if total is not None else None,
        "gpu_wallclock_per_fc_hr_s": round(total_per_fc_hr, 3) if total_per_fc_hr else None,
        "compile_overhead_s": round(compile_overhead, 3) if compile_overhead is not None else None,
        "steady_state_s_per_fc_hr": round(steady_per_fc_hr, 3) if steady_per_fc_hr else None,
        "device": pipeline.get("device"),
        "source_pipeline_json": None,  # set by caller
    }


def _fill_case(case: dict, pipeline_json: str, forecast_h: int, dt_floor_factor: float, denom_key: str) -> None:
    pipeline = _load(pipeline_json)
    gb = _gpu_breakdown(pipeline, forecast_h)
    gb["source_pipeline_json"] = str(pipeline_json)

    gpu = case["gpu"]
    gpu["PLACEHOLDER"] = False
    gpu["gpu_wallclock_total_s"] = gb["gpu_wallclock_total_s"]
    gpu["gpu_wallclock_per_fc_hr_s"] = gb["gpu_wallclock_per_fc_hr_s"]
    gpu["compile_overhead_s"] = gb["compile_overhead_s"]
    gpu["steady_state_s_per_fc_hr"] = gb["steady_state_s_per_fc_hr"]
    gpu["measured_hours"] = gb["measured_hours"]
    gpu["source_pipeline_json"] = gb["source_pipeline_json"]
    gpu["measured_device"] = gb["device"]

    denom = case["cpu"][denom_key]
    cpu_lo = denom["conservative_low_s_per_fc_hr"]
    cpu_mid = denom["midpoint_s_per_fc_hr"]
    cpu_hi = denom["realistic_high_s_per_fc_hr"]

    res = case["results"]
    res["PLACEHOLDER"] = False
    gpu_total_per_fc_hr = gb["gpu_wallclock_per_fc_hr_s"]
    gpu_steady_per_fc_hr = gb["steady_state_s_per_fc_hr"]

    def spd(cpu):
        return round(cpu / gpu_total_per_fc_hr, 2) if (gpu_total_per_fc_hr and cpu) else None

    def spd_steady(cpu):
        return round(cpu / gpu_steady_per_fc_hr, 2) if (gpu_steady_per_fc_hr and cpu) else None

    res["real_user_speedup_compile_inclusive"] = {
        "headline_conservative": spd(cpu_lo),
        "midpoint": spd(cpu_mid),
        "realistic_high": spd(cpu_hi),
        "definition": "cpu_s_per_fc_hr / gpu_total_s_per_fc_hr (compile-inclusive). HEADLINE = headline_conservative.",
    }
    res["steady_state_speedup_compile_excluded"] = {
        "conservative": spd_steady(cpu_lo),
        "midpoint": spd_steady(cpu_mid),
        "realistic_high": spd_steady(cpu_hi),
        "definition": "CONTEXT ONLY (NOT the headline): cpu_s_per_fc_hr / gpu_steady_state_s_per_fc_hr (compile excluded).",
    }
    if spd(cpu_lo) is not None:
        res["dt_matched_floor_speedup"] = {
            "conservative": round(spd(cpu_lo) / dt_floor_factor, 2),
            "factor": dt_floor_factor,
            "definition": "strict dt-matched floor: real_user headline / dt_floor_factor (GPU forced to CPU dt).",
        }


def main(argv: list[str] | None = None) -> int:
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0, 1, 2, 3})
        except OSError:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d02-pipeline", default=None, help="pipeline_run JSON from m7_l2_d02_replay.py")
    ap.add_argument("--d03-pipeline", default=None, help="pipeline_run JSON from d03_replay.py")
    ap.add_argument("--bench", default=str(BENCH))
    args = ap.parse_args(argv)

    bench = _load(args.bench)
    if args.d02_pipeline:
        _fill_case(bench["cases"]["nested_9_3km"], args.d02_pipeline, forecast_h=72, dt_floor_factor=1.67, denom_key="ADOPTED_CPU_DENOMINATOR_d02_standalone")
    if args.d03_pipeline:
        _fill_case(bench["cases"]["single_1km"], args.d03_pipeline, forecast_h=24, dt_floor_factor=1.5, denom_key="ADOPTED_CPU_DENOMINATOR_d03_standalone")

    bench["filled_on"] = "see git history"
    with open(args.bench, "w") as fh:
        fh.write(json.dumps(bench, indent=2) + "\n")
    print(json.dumps({
        "filled_d02": bool(args.d02_pipeline),
        "filled_d03": bool(args.d03_pipeline),
        "nested_9_3km_real_user_speedup": bench["cases"]["nested_9_3km"]["results"].get("real_user_speedup_compile_inclusive"),
        "single_1km_real_user_speedup": bench["cases"]["single_1km"]["results"].get("real_user_speedup_compile_inclusive"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
