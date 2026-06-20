#!/usr/bin/env python
"""v0.15 kernel-final Task 3 — AFTER-tiling VRAM ladder (vs largest_probe.json).

Re-runs the lanes-1+2 largest-grid probe with the MP column tiling in place
(GPUWRF_MP_COLUMN_TILING default-on).  Same harness, same case build, same
cuda_async allocator; extends the ladder beyond the old OOM point to find the
NEW ceiling.  640x320 @1 km = 204,800 cols, bracketed by the (4,5) = 209,880
rung: if that fits, 640x320 fits.

Run (GPU lock):
  scripts/with_gpu_lock.sh --label kernel-final -- timeout 5400 \
    taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo \
    python proofs/perf/v015/km_bench/probe_largest_tiled.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_largest import run_one  # noqa: E402  (reuses _tile_state/_peak_gib chain)

# base 66x159 = 10,494 cols; ladder brackets the old 26.25 GiB point (4,4),
# the old OOM (4,5) = 209,880 >= the 640x320 target 204,800, and probes the
# new ceiling beyond it.
CANDIDATES = [(4, 4), (4, 5), (5, 5), (5, 6), (6, 6)]  # 168k, 210k, 262k, 315k, 378k


def main() -> int:
    assert os.environ.get("GPUWRF_MP_COLUMN_TILING", "1") != "0", "tiling must be ON"
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    bnl, bst = case.namelist, case.state
    dt = float(bnl.dt_s)
    ny0, nx0 = int(case.grid.ny), int(case.grid.nx)
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "(default bfc)")
    print(f"[probe-tiled] base {ny0}x{nx0} allocator={allocator}", flush=True)
    recs = []
    for fy, fx in CANDIDATES:
        ny, nx = fy * ny0, fx * nx0
        print(f"[probe-tiled] {ny}x{nx} ncol={ny*nx} ...", flush=True)
        t_start = time.perf_counter()
        try:
            r = run_one(bst, bnl, ny0, nx0, fy, fx, dt)
            r["wall_total_s"] = round(time.perf_counter() - t_start, 1)
            print(
                f"  OK ms/step={r['warmed_ms_per_step']:.1f} peak={r['peak_vram_gib']:.2f}G",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - probe must record, not crash
            is_oom = "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()
            r = {
                "ny": ny, "nx": nx, "ncol": ny * nx, "ran_ok": False, "oom": bool(is_oom),
                "error": f"{type(e).__name__}: {e}"[:300],
                "wall_total_s": round(time.perf_counter() - t_start, 1),
            }
            print(f"  FAIL oom={is_oom} :: {str(e)[:140]}", flush=True)
            recs.append(r)
            if is_oom:
                break
            continue
        recs.append(r)
    out = {
        "scope": "largest-grid probe AFTER MP column tiling (allocator=" + allocator + ")",
        "allocator": allocator,
        "mp_column_tiling": {
            "GPUWRF_MP_COLUMN_TILING": os.environ.get("GPUWRF_MP_COLUMN_TILING", "(default 1)"),
            "GPUWRF_MP_COLUMN_TILE_COLS": os.environ.get("GPUWRF_MP_COLUMN_TILE_COLS", "(default 16384)"),
        },
        "before_reference": "proofs/perf/v015/km_bench/largest_probe.json (26.25 GiB @167,904; OOM 28.44 GiB single alloc @209,880)",
        "records": recs,
    }
    (HERE / "largest_probe_tiled.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote proofs/perf/v015/km_bench/largest_probe_tiled.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
