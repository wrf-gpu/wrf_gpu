#!/usr/bin/env python
"""v0.15 SHIP-GATE 1b (ceiling search) — find the ACTUAL post-MP-tiling VRAM
ceiling with the operational pipeline, descending from the Switzerland-scale
rung that OOM'd toward the targets, to locate where the grid actually fits +
runs finite.  Tiling is ON (GPUWRF_MP_COLUMN_TILING=1).

Reuses probe_largest.run_one (cuda_async, run_forecast_operational, peak VRAM)
and adds the per-rung finiteness check from probe_largest_tiled_finite.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import run_forecast_operational

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_largest import run_one  # noqa: E402
from grid_scaling_bench import _tile_state, _tile_namelist, _block  # noqa: E402

# base 66x159 = 10,494 cols.  Descend from the 264x636 = 167,904 OOM toward
# smaller multiples to find the real ceiling and bracket the named targets:
#   Canary 560x280 = 156,800 ; 640x320 @1km = 204,800.
# factors of base: pick (fy, fx) giving a descending ncol ladder below 167,904.
CANDIDATES = [(3, 4), (2, 6), (3, 3), (2, 5), (2, 4), (3, 2)]


def _finite_check(base_state, base_nl, ny0, nx0, fy, fx):
    nl = _tile_namelist(base_nl, ny0, nx0, fy, fx)
    st = _tile_state(base_state, ny0, nx0, fy, fx)
    st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
    _block(st)
    out = run_forecast_operational(st, nl, 0.15)
    _block(out)
    from gpuwrf.contracts.state import State
    all_finite = True
    flagged = {}
    for name in State.__slots__:
        v = getattr(out, name, None)
        if not hasattr(v, "shape"):
            continue
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if not np.issubdtype(a.dtype, np.floating):
            continue
        if not bool(np.isfinite(a).all()):
            all_finite = False
            flagged[name] = int((~np.isfinite(a)).sum())
    return {"all_finite": all_finite, "flagged": flagged}


def main() -> int:
    assert os.environ.get("GPUWRF_MP_COLUMN_TILING", "1") != "0", "tiling must be ON"
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    bnl, bst = case.namelist, case.state
    dt = float(bnl.dt_s)
    ny0, nx0 = int(case.grid.ny), int(case.grid.nx)
    print(f"[ceiling] base {ny0}x{nx0} cols={ny0*nx0}", flush=True)
    recs = []
    for fy, fx in CANDIDATES:
        ny, nx = fy * ny0, fx * nx0
        ncol = ny * nx
        print(f"[ceiling] {ny}x{nx} ncol={ncol} ...", flush=True)
        t0 = time.perf_counter()
        try:
            r = run_one(bst, bnl, ny0, nx0, fy, fx, dt)
            fin = _finite_check(bst, bnl, ny0, nx0, fy, fx)
            r.update({"finiteness": fin, "stable": bool(r["ran_ok"] and fin["all_finite"])})
            r["wall_total_s"] = round(time.perf_counter() - t0, 1)
            print(f"  OK peak={r['peak_vram_gib']:.2f}G ms/step={r['warmed_ms_per_step']:.1f} finite={fin['all_finite']}", flush=True)
        except Exception as e:  # noqa: BLE001
            is_oom = "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()
            r = {"ny": ny, "nx": nx, "ncol": ncol, "ran_ok": False, "oom": bool(is_oom),
                 "stable": False, "error": f"{type(e).__name__}: {e}"[:200],
                 "wall_total_s": round(time.perf_counter() - t0, 1)}
            print(f"  FAIL oom={is_oom} :: {str(e)[:120]}", flush=True)
        recs.append(r)
    out = {
        "schema": "V015VramCeilingTiled",
        "scope": "post-MP-tiling operational-pipeline VRAM ceiling search (tiling ON)",
        "allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "(default)"),
        "note": "264x636=167,904 OOM'd on a single 18.72 GiB NON-Thompson allocation; this search finds where the operational pipeline actually fits + runs finite under 32 GiB with MP tiling ON.",
        "targets": {"canary_560x280": 156800, "p1km_640x320": 204800, "switzerland_264x636": 167904},
        "records": recs,
        "largest_stable_ncol": max([r["ncol"] for r in recs if r.get("stable")], default=0),
    }
    (HERE / "vram_ceiling_tiled.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote proofs/perf/v015/km_bench/vram_ceiling_tiled.json", flush=True)
    print("largest_stable_ncol =", out["largest_stable_ncol"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
