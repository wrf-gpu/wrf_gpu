#!/usr/bin/env python
"""Capture XLA's full RESOURCE_EXHAUSTED breakdown at the 167,904-col grid that
binds the v0.15 1km targets, to name the binding allocation in XLA's own words.

Uses the PLATFORM (bfc) allocator (default) so the OOM message carries the
"Largest program allocation(s)" / peak-buffer breakdown. This is corroboration
for the buffer-assignment localization (the preallocated-temp arena).

Run (GPU lock):
  scripts/with_gpu_lock.sh --label v015-vram -- env \
    PYTHONPATH=src JAX_ENABLE_X64=true OMP_NUM_THREADS=4 \
    GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo taskset -c 0-3 \
    python proofs/perf/v015/km_bench/oom_breakdown_167904.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Default (platform/bfc) allocator gives the verbose OOM buffer breakdown.

import jax  # noqa: E402


def _load_bench():
    spec = importlib.util.spec_from_file_location("grid_scaling_bench", HERE / "grid_scaling_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    gsb = _load_bench()
    from gpuwrf.runtime.operational_mode import run_forecast_operational

    cfg = gsb.DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = gsb._build_real_case(cfg)
    bnl, bst = case.namelist, case.state
    ny0, nx0 = int(case.grid.ny), int(case.grid.nx)
    fy, fx = 2, 8  # 2*66 x 8*159 = 132 x 1272 = 167,904
    ny, nx = fy * ny0, fx * nx0
    print(f"[oom] base {ny0}x{nx0}; OOM grid {ny}x{nx} ncol={ny*nx}; allocator={os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR','platform/bfc')}", flush=True)
    nl = gsb._tile_namelist(bnl, ny0, nx0, fy, fx)
    st = gsb._tile_state(bst, ny0, nx0, fy, fx)
    st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
    rec = {"schema": "V015OOMBreakdown167904", "grid": {"ny": ny, "nx": nx, "ncol": ny * nx},
           "allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "platform/bfc")}
    try:
        o = run_forecast_operational(st, nl, 0.05)
        jax.block_until_ready(jax.tree_util.tree_leaves(o)[0])
        rec.update(ran_ok=True, oom=False)
        print("[oom] UNEXPECTED: ran OK", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        rec.update(ran_ok=False, oom=("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()))
        (HERE / "oom_breakdown_167904.txt").write_text(msg)
        lines = [l for l in msg.splitlines()
                 if re.search(r"GiB|MiB|[Aa]llocat|buffer|preallocat|temp|fusion|f64\[|peak|Total|Largest", l)]
        rec["breakdown_lines"] = lines[:200]
        print(f"[oom] OOM={rec['oom']}; {len(lines)} breakdown lines:", flush=True)
        print("\n".join(lines[:80]), flush=True)
    (HERE / "oom_breakdown_167904.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("wrote oom_breakdown_167904.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
