"""Driver for nsys/kernel-level profiling of the WARMED per-step coupled forecast.

Builds the real d02 case, warms the compiled forecast, then runs N warmed steps
under an NVTX range so nsys can attribute GPU time to kernels and (critically)
measure the inter-kernel GAP time -- the signature of launch/latency-bound
execution (many tiny dependent kernels from the vertical lax.scan sweeps).

Usage (wrapped by nsys in the .sh runner):
  PYTHONPATH=src ... python proofs/perf/nsys_step_driver.py --steps 60 --warm-hours 0.05
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import jax

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warm-hours", type=float, default=0.05)  # 18 steps warm
    args = ap.parse_args()

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=180, time_utc=case.run_start,
    )
    dt_s = float(nl.dt_s)
    hours = args.steps * dt_s / 3600.0

    # Warm compile at the EXACT profiled hours so the profiled call is a cache hit.
    st = _build_real_case(cfg)[0].state
    out = run_forecast_operational(st, nl, hours)
    jax.block_until_ready(out.theta)
    print(f"warmed at hours={hours} ({args.steps} steps)", flush=True)

    # Profiled region: one warmed compiled forecast of N steps under NVTX.
    st = _build_real_case(cfg)[0].state
    rng = jax.profiler.TraceAnnotation("WARMED_STEP_REGION")
    t0 = time.perf_counter()
    with rng:
        out = run_forecast_operational(st, nl, hours)
        jax.block_until_ready(out.theta)
    wall = time.perf_counter() - t0
    print(f"profiled {args.steps} warmed steps wall={wall:.4f}s per_step_ms={wall/args.steps*1000:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
