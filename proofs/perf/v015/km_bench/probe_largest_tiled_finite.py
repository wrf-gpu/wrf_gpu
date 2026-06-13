#!/usr/bin/env python
"""v0.15 SHIP-GATE 1b — AFTER-tiling VRAM ladder WITH per-rung finiteness.

Wraps the committed `probe_largest.run_one` (same tiler, same cuda_async
allocator, same `run_forecast_operational` integration) and adds the two things
the v0.15 ship contract (deliverable 1b) requires beyond the committed
`probe_largest_tiled.py`:

  (1) per-rung STABILITY: after the timed h2 forecast, run one more (already
      compiled, cheap) forecast and assert every State leaf is finite — no
      NaN/Inf, no blow-up — reporting all_finite + the worst |field| max.
  (2) the explicit ship targets, by bracketing: the (4,5) = 209,880-col rung
      brackets the 640x320 @1km target (204,800 cols, was 32.3 GiB OOM
      pre-tiling); (4,4) = 168k brackets Canary 560x280 (156,800) and
      Switzerland-scale 264x636 (167,904). If a bracketing rung fits AND is
      finite, the smaller target it brackets fits AND is finite.

Run (GPU lock):
  scripts/with_gpu_lock.sh --label v015-shipgates -- timeout 5400 \
    taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo \
    python proofs/perf/v015/km_bench/probe_largest_tiled_finite.py
"""
from __future__ import annotations

import json
import math
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

# Same ladder as probe_largest_tiled.py; rungs bracket the ship targets.
#   (4,4)=168k brackets Canary 560x280 (156,800) and Switzerland 264x636 (167,904)
#   (4,5)=210k brackets 640x320 @1km (204,800, the pre-tiling 32.3 GiB OOM)
CANDIDATES = [(4, 4), (4, 5), (5, 5), (5, 6), (6, 6)]
TARGETS = {
    "640x320_1km": 204800,
    "canary_560x280": 156800,
    "switzerland_264x636": 167904,
}


def _finiteness(base_state, base_nl, ny0, nx0, fy, fx, dt):
    """Re-run the (already compiled) h2 forecast once and check the output
    state is entirely finite; report worst absolute magnitude per leaf class."""
    h2 = 0.15
    nl = _tile_namelist(base_nl, ny0, nx0, fy, fx)
    st = _tile_state(base_state, ny0, nx0, fy, fx)
    st = jax.tree_util.tree_map(
        lambda x: (x + 0) if hasattr(x, "shape") else x, st
    )
    _block(st)
    out = run_forecast_operational(st, nl, h2)
    _block(out)
    all_finite = True
    worst = {}
    n_checked = 0
    from gpuwrf.contracts.state import State

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
        n_checked += 1
        fin = bool(np.isfinite(a).all())
        if not fin:
            all_finite = False
            worst[name] = {"finite": False, "n_nonfinite": int((~np.isfinite(a)).sum())}
        else:
            mx = float(np.max(np.abs(a))) if a.size else 0.0
            if mx > 1e6:  # only surface suspiciously large finite magnitudes
                worst[name] = {"finite": True, "max_abs": mx}
    return {"all_finite": all_finite, "n_leaves_checked": n_checked, "flagged": worst}


def main() -> int:
    assert os.environ.get("GPUWRF_MP_COLUMN_TILING", "1") != "0", "tiling must be ON"
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    bnl, bst = case.namelist, case.state
    dt = float(bnl.dt_s)
    ny0, nx0 = int(case.grid.ny), int(case.grid.nx)
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "(default bfc)")
    print(f"[probe-finite] base {ny0}x{nx0} allocator={allocator}", flush=True)
    recs = []
    for fy, fx in CANDIDATES:
        ny, nx = fy * ny0, fx * nx0
        print(f"[probe-finite] {ny}x{nx} ncol={ny*nx} ...", flush=True)
        t_start = time.perf_counter()
        try:
            r = run_one(bst, bnl, ny0, nx0, fy, fx, dt)
            # stability: explicit finiteness on the integrated output state.
            fin = _finiteness(bst, bnl, ny0, nx0, fy, fx, dt)
            r.update({"finiteness": fin, "stable": bool(r["ran_ok"] and fin["all_finite"])})
            r["wall_total_s"] = round(time.perf_counter() - t_start, 1)
            print(
                f"  OK ms/step={r['warmed_ms_per_step']:.1f} "
                f"peak={r['peak_vram_gib']:.2f}G finite={fin['all_finite']}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            is_oom = "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()
            r = {
                "ny": ny, "nx": nx, "ncol": ny * nx, "ran_ok": False, "oom": bool(is_oom),
                "stable": False,
                "error": f"{type(e).__name__}: {e}"[:300],
                "wall_total_s": round(time.perf_counter() - t_start, 1),
            }
            print(f"  FAIL oom={is_oom} :: {str(e)[:140]}", flush=True)
            recs.append(r)
            if is_oom:
                break
            continue
        recs.append(r)

    # Resolve each ship target against the smallest bracketing rung that fits+stable.
    target_verdicts = {}
    for tname, tcols in TARGETS.items():
        bracket = None
        for r in recs:
            if r.get("ncol", 0) >= tcols and r.get("stable"):
                if bracket is None or r["ncol"] < bracket["ncol"]:
                    bracket = r
        target_verdicts[tname] = {
            "target_cols": tcols,
            "fits": bracket is not None,
            "bracketed_by": (
                {"ny": bracket["ny"], "nx": bracket["nx"], "ncol": bracket["ncol"],
                 "peak_vram_gib": bracket.get("peak_vram_gib"),
                 "ms_per_step": bracket.get("warmed_ms_per_step")}
                if bracket else None
            ),
        }

    out = {
        "schema": "V015VramLadderFinite",
        "scope": "largest-grid VRAM ladder AFTER MP column tiling, with per-rung finiteness",
        "allocator": allocator,
        "vram_ceiling_gib": 32607 / 1024.0,
        "mp_column_tiling": {
            "GPUWRF_MP_COLUMN_TILING": os.environ.get("GPUWRF_MP_COLUMN_TILING", "(default 1)"),
            "GPUWRF_MP_COLUMN_TILE_COLS": os.environ.get("GPUWRF_MP_COLUMN_TILE_COLS", "(default 16384)"),
        },
        "before_reference": "proofs/perf/v015/km_bench/largest_probe.json (26.25 GiB @167,904; OOM 28.44 GiB single alloc @209,880 pre-tiling)",
        "records": recs,
        "ship_target_verdicts": target_verdicts,
        "all_targets_fit_and_stable": all(v["fits"] for v in target_verdicts.values()),
    }
    (HERE / "largest_probe_tiled_finite.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote proofs/perf/v015/km_bench/largest_probe_tiled_finite.json", flush=True)
    print(json.dumps(target_verdicts, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
