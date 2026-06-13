#!/usr/bin/env python
"""v0.15 kernel-final Task 3 — GPU EXACT-output gate: tiled vs untiled Thompson.

Builds the REAL Switzerland d01 reinit-h36 column state (16384 cols), widens it
to 40,960 columns (2.5 tiles of 16384: exercises multi-tile + the repeat-pad
path), then runs the production entry `step_thompson_column_with_precip`
(tiling ACTIVE at this width) against the untiled body jitted directly.  The
gate is per-leaf BYTE equality on every output (11 state leaves + 5 precip
channels).  If any leaf differs by even one bit on GPU, the tiling does NOT
ship (sprint contract Task 3 stop rule).

Run (GPU lock):
  scripts/with_gpu_lock.sh --label kernel-final -- timeout 900 \
    taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python proofs/perf/v015/km_bench/mp_tiling_identity.py
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration import daily_pipeline as dp
from gpuwrf.coupling.physics_couplers import _thompson_column_from_state
from gpuwrf.physics import thompson_column as tc

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
OUT = Path("proofs/perf/v015/km_bench/mp_tiling_identity.json")


def main() -> int:
    assert tc._MP_COLUMN_TILING, "tiling must be enabled for the gate"
    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=1,
        output_dir=Path("/tmp/v015_perf/mp_tiling_identity"),
        proof_dir=Path("/tmp/v015_perf/mp_tiling_identity/proofs"),
        run_root=PROBE, domain="d01",
    )
    case, _ = dp._build_real_case(config)
    column = _thompson_column_from_state(case.state)
    base_ncol = int(jnp.asarray(column.qv).shape[0])

    # Replicate the REAL columns to a width comfortably exceeding the production
    # 16384-col tile AND a small forced tile width, so the SAME multi-tile scan
    # the production grid uses (which only engages at ncol > 16384, i.e. larger
    # grids than the 320-col reinit case build) is genuinely exercised on real
    # (non-synthetic) data.  Target >= 2.5 production tiles -> 40,960 cols.
    target = 40960
    reps = -(-target // max(base_ncol, 1))  # ceil
    half = max(1, base_ncol // 2)
    wide = jax.tree_util.tree_map(
        lambda a: jnp.concatenate(
            [jnp.asarray(a)] * reps + [jnp.asarray(a)[:half]], axis=0
        ),
        column,
    )
    ncol = int(jnp.asarray(wide.qv).shape[0])
    assert tc._MP_COLUMN_TILE_COLS > 0 and ncol > tc._MP_COLUMN_TILE_COLS, (
        f"gate misconfigured: ncol={ncol} must exceed tile_cols="
        f"{tc._MP_COLUMN_TILE_COLS} so tiling engages"
    )
    dt = float(case.namelist.dt_s)

    untiled_fn = jax.jit(partial(tc._step_thompson_column_full_impl, debug=False), static_argnames=("dt",))

    t0 = time.perf_counter()
    out_t, pr_t = jax.block_until_ready(tc.step_thompson_column_with_precip(wide, dt))
    wall_tiled = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_u, pr_u = jax.block_until_ready(untiled_fn(wide, dt=dt))
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
        "case": "Switzerland d01 reinit-h36 real columns, widened to 2.5 tiles",
        "backend": jax.devices()[0].platform,
        "device": str(jax.devices()[0]),
        "ncol": ncol,
        "tile_cols": int(tc._MP_COLUMN_TILE_COLS),
        "n_tiles": -(-ncol // int(tc._MP_COLUMN_TILE_COLS)),
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
