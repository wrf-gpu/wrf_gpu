#!/usr/bin/env python
"""Win #1 wall-clock A/B: persistent compilation cache cold vs hot.

Builds the real d02 case and times a single short segmented forecast. Run twice
against a FRESH cache dir: the first run pays the cold XLA compile; the second
run (same process invocation, fresh process) reads the executable from the
on-disk cache. The delta is the compile time the persistent cache removes on
every repeat run -- with NO numerics change.

USAGE (cold then hot against a fresh cache):
  CACHE=/tmp/win1_cache_ab; rm -rf "$CACHE"
  for tag in cold hot; do
    PYTHONPATH=src JAX_ENABLE_X64=true GPUWRF_JAX_CACHE_DIR=$CACHE \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/perf/win1_cache_ab_timing.py --tag $tag \
      --run-id 20260521_18z_l3_24h_20260522T133443Z \
      --run-root /mnt/data/canairy_meteo/runs/wrf_l3 --hours 1 --segment-steps 60
  done
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax

import gpuwrf  # triggers the cache hook
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--segment-steps", type=int, default=60)
    ap.add_argument("--out", default="proofs/perf/win1_cache_ab_timing.jsonl")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(
        run_id=args.run_id, run_root=Path(args.run_root),
        output_dir=Path("/tmp/win1_ab_out"), proof_dir=Path("/tmp/win1_ab_proof"),
        hours=int(args.hours), domain="d02",
    )
    t_case0 = time.perf_counter()
    case, _ = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, time_utc=case.run_start)
    case_build_s = time.perf_counter() - t_case0

    t0 = time.perf_counter()
    out = run_forecast_operational_segmented(
        case.state, nl, float(args.hours), segment_steps=args.segment_steps
    )
    jax.block_until_ready(out.theta)
    forecast_s = time.perf_counter() - t0

    rec = {
        "win": "1-persistent-compilation-cache",
        "tag": args.tag,
        "cache_status": gpuwrf._JAX_CACHE_STATUS,
        "hours": args.hours,
        "segment_steps": args.segment_steps,
        "case_build_s": round(case_build_s, 2),
        "forecast_compile_plus_run_s": round(forecast_s, 2),
    }
    print(json.dumps(rec))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
