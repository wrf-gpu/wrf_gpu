#!/usr/bin/env python
"""v0.15 SHIP-GATE 1a (fit variant) — GPU EXACT-output: tiled vs untiled Thompson.

Why this variant: the committed `mp_tiling_identity.py` widens the real
16,384-col Switzerland d01 reinit-h36 column state to 40,960 cols and runs the
UNTILED body at that full width as the reference.  The untiled Thompson body at
41k cols x 44 levels in fp64 needs >32 GiB of transient working set (the very
reason tiling exists) and OOMs during XLA autotuning on the 32 GiB RTX 5090 —
so the untiled REFERENCE cannot be produced at 41k on this card.

This variant keeps the identity contract IDENTICAL — per-leaf BYTE equality,
tiled vs untiled, on 100% REAL columns, exercising the multi-tile lax.scan AND
the repeat-pad path — but at a width where BOTH paths physically fit:

  * width = the FULL real 16,384-col state (the production v0.14 grid size; the
    untiled body provably fits here — it ran every v0.14 forecast).
  * tile_cols forced to 6144  ->  ceil(16384/6144) = 3 tiles, padded to 18,432
    (a 2,048-col repeat-pad of the last real column).  Multi-tile scan AND the
    repeat-pad path are BOTH genuinely exercised on real, non-synthetic data.

Tile size is a pure execution-shape knob (column_tiling module contract: no
math, no clamps, no cross-column coupling), so forcing it is identity-safe and
does not weaken the test of whether tiling changes per-column arithmetic.

Run (GPU lock):
  scripts/with_gpu_lock.sh --label v015-shipgates -- timeout 900 \
    taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    GPUWRF_MP_COLUMN_TILE_COLS=6144 \
    python proofs/perf/v015/km_bench/mp_tiling_identity_fit.py
"""
from __future__ import annotations

import json
import os
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration import daily_pipeline as dp
from gpuwrf.coupling.physics_couplers import _thompson_column_from_state
from gpuwrf.physics import thompson_column as tc

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
OUT = Path("proofs/perf/v015/km_bench/mp_tiling_identity_fit.json")


def main() -> int:
    assert tc._MP_COLUMN_TILING, "tiling must be enabled for the gate"
    # Force a small tile so the FULL real 16,384-col state is multi-tile + padded
    # while the untiled reference (same width = the production grid) still fits.
    forced_tile = int(os.environ.get("GPUWRF_MP_COLUMN_TILE_COLS", "6144"))
    assert tc._MP_COLUMN_TILE_COLS == forced_tile, (
        f"tile_cols must be the forced {forced_tile}; got {tc._MP_COLUMN_TILE_COLS} "
        "(set GPUWRF_MP_COLUMN_TILE_COLS in the env)"
    )
    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=1,
        output_dir=Path("/tmp/v015_perf/mp_tiling_identity_fit"),
        proof_dir=Path("/tmp/v015_perf/mp_tiling_identity_fit/proofs"),
        run_root=PROBE, domain="d01",
    )
    case, _ = dp._build_real_case(config)
    # Use the REAL production column view at its NATIVE (ny, nx, nz) = (128, 128,
    # 44) 3-D layout — exactly what `_thompson_column_from_state` emits on every
    # operational forecast.  This is 16,384 columns (128*128), the v0.14
    # production grid size, so the UNTILED reference body provably fits (it ran
    # every v0.14 forecast).  Forcing a small tile (4096) makes 16,384 cols span
    # 4 tiles, so the multi-tile lax.scan over the flattened (ny*nx) axis is
    # genuinely exercised on 100% real, non-synthetic data.
    column = _thompson_column_from_state(case.state)
    qv = jnp.asarray(column.qv)
    lead_shape = tuple(qv.shape[:-1])  # (ny, nx)
    ncol = 1
    for d in lead_shape:
        ncol *= int(d)
    n_tiles = -(-ncol // forced_tile)
    padded = n_tiles * forced_tile
    assert qv.ndim >= 3, f"expected production (ny,nx,nz) 3-D layout; got {qv.shape}"
    assert ncol > forced_tile, f"need ncol>{forced_tile} for multi-tile; got {ncol}"
    dt = float(case.namelist.dt_s)

    untiled_fn = jax.jit(
        partial(tc._step_thompson_column_full_impl, debug=False), static_argnames=("dt",)
    )

    t0 = time.perf_counter()
    out_t, pr_t = jax.block_until_ready(tc.step_thompson_column_with_precip(column, dt))
    wall_tiled = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_u, pr_u = jax.block_until_ready(untiled_fn(column, dt=dt))
    wall_untiled = time.perf_counter() - t0

    leaves = {}
    n_fail = 0
    for name in tc.ThompsonColumnState.__slots__:
        a = np.asarray(getattr(out_t, name))
        b = np.asarray(getattr(out_u, name))
        bitwise = bool(a.tobytes() == b.tobytes())
        rec = {"bitwise": bitwise, "dtype": str(a.dtype), "shape": list(a.shape)}
        if not bitwise:
            n_fail += 1
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            rec["max_abs_diff"] = float(np.nanmax(d))
            rec["n_diff"] = int(np.sum(a != b))
        leaves[name] = rec
    for key in sorted(pr_u):
        a = np.asarray(pr_t[key])
        b = np.asarray(pr_u[key])
        bitwise = bool(a.tobytes() == b.tobytes())
        rec = {"bitwise": bitwise, "dtype": str(a.dtype), "shape": list(a.shape)}
        if not bitwise:
            n_fail += 1
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            rec["max_abs_diff"] = float(np.nanmax(d))
            rec["n_diff"] = int(np.sum(a != b))
        leaves[f"precip.{key}"] = rec

    payload = {
        "schema": "V015MPTilingIdentity",
        "variant": "fit (REAL production (ny,nx,nz)=(128,128,44) 3-D layout, forced tile so it spans multiple tiles AND the untiled reference fits on 32GiB)",
        "case": "Switzerland d01 reinit-h36 REAL production column view at native (ny,nx,nz)",
        "backend": jax.devices()[0].platform,
        "device": str(jax.devices()[0]),
        "lead_shape_ny_nx": list(lead_shape),
        "ncol": ncol,
        "tile_cols": int(tc._MP_COLUMN_TILE_COLS),
        "n_tiles": n_tiles,
        "padded_ncol": padded,
        "repeat_pad_cols": padded - ncol,
        "dt_s": dt,
        "wall_s": {"tiled": round(wall_tiled, 3), "untiled": round(wall_untiled, 3)},
        "leaves": leaves,
        "n_leaves": len(leaves),
        "n_bitwise_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "leaves"}, indent=2))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
