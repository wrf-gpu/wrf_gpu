#!/usr/bin/env python
"""v0.22 compile-pathology kill-gate (b): fp32-BouLac mixed fp32/fp64 fusion.

Drives the FULL operational forecast jit (dp._default_forecast_fn, the exact
production pipeline, same path as proofs/perf/v015/probe_ab_identity.py) and
measures:
  - cold compile+first-exec wall (hour1)  -> the "Very slow compile" axis
  - steady warm runtime per step (hour2+)  -> whether the executable runs (no stall)
  - whether it STALLS (a hung GPU exec): detected by the parent watchdog timeout,
    not by this process (a stall freezes here).

The env knob under test is set by the CALLER:
  GPUWRF_MYNN_BOULAC_FP32=1  -> the kill-gate config
  unset / =0                 -> the clean baseline

PASS (for the fp32-boulac config) = compile bounded (completes) AND runs
(steady step produced, finite state). The prior NO-GO (proofs/perf/v015/
fp32_definitive_verdict.json) claims this STALLS; this is the fresh empirical
re-measurement the verdict said it did not do.

Artifact: proofs/v022/compile_pathology/gate_b_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
HERE = Path(__file__).resolve().parent


def _finite_frac(state) -> tuple[bool, float, int]:
    """Return (all_finite, min_finite_frac_over_leaves, n_leaves)."""
    all_fin = True
    min_frac = 1.0
    n = 0
    for _name, value in dp._field_items(state):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        n += 1
        fin = np.isfinite(arr)
        frac = float(fin.mean()) if arr.size else 1.0
        if frac < min_frac:
            min_frac = frac
        if not fin.all():
            all_fin = False
    return all_fin, min_frac, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hours", type=int, default=3)
    args = ap.parse_args()

    t_import = time.perf_counter()

    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=args.hours,
        output_dir=Path(f"/tmp/v022_gateb/{args.tag}"),
        proof_dir=Path(f"/tmp/v022_gateb/{args.tag}/proofs"),
        run_root=PROBE, domain="d01",
    )
    t_build0 = time.perf_counter()
    case, _run_dir = dp._build_real_case(config)
    build_s = time.perf_counter() - t_build0

    state = case.state
    boundary_leaves = dp._capture_boundary_leaves(state, case.namelist)
    window_s = dp._boundary_window_cadence_s(case.namelist)
    record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)

    walls = []
    finite_log = []
    for hour in range(1, args.hours + 1):
        st_in = (
            dp._rewindow_boundary_leaves(
                state, boundary_leaves, segment_start_s=(hour - 1) * 3600.0,
                record_cadence_s=record_s, window_s=window_s,
            )
            if boundary_leaves
            else state
        )
        print(f"[{args.tag}] hour{hour} START t={time.perf_counter()-t_import:.1f}s", flush=True)
        t0 = time.perf_counter()
        state = dp._default_forecast_fn(st_in, case.namelist, 1.0)
        # block_until_ready via numpy materialization of one leaf done in finite check
        all_fin, min_frac, n_leaves = _finite_frac(state)
        wall = time.perf_counter() - t0
        walls.append(round(wall, 3))
        finite_log.append({"hour": hour, "all_finite": all_fin,
                           "min_finite_frac": round(min_frac, 6), "n_leaves": n_leaves})
        print(f"[{args.tag}] hour{hour} DONE wall={wall:.3f}s all_finite={all_fin} "
              f"min_frac={min_frac:.4f}", flush=True)

    cold_wall = walls[0]
    steady_wall = walls[-1] if len(walls) > 1 else None
    # cold compile ~= cold_wall - steady_wall (first call = compile + 1 forecast-hour exec)
    cold_compile_est = round(cold_wall - steady_wall, 3) if steady_wall is not None else None

    payload = {
        "schema": "V022GateBCompilePathology",
        "tag": args.tag,
        "gate": "compile-pathology kill-gate (b): fp32-BouLac mixed fp32/fp64 fusion",
        "case": "Switzerland d01 reinit-h36, 128x128x44, dt=18s (200 steps/fc-hr), RTX 5090, fp64",
        "env": {
            k: os.environ.get(k)
            for k in (
                "GPUWRF_MYNN_BOULAC_FP32", "GPUWRF_MYNN_COND_NITER",
                "GPUWRF_MYNN_COND_UNROLL", "GPUWRF_MYNN_BOULAC_ONZ",
                "GPUWRF_THOMAS_UNROLL", "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            )
        },
        "build_real_case_s": round(build_s, 3),
        "per_hour_wall_s": walls,
        "cold_wall_s": cold_wall,
        "steady_wall_s": steady_wall,
        "cold_compile_est_s": cold_compile_est,
        "steady_ms_per_step": round(steady_wall / 200.0 * 1000.0, 2) if steady_wall else None,
        "finite_log": finite_log,
        "completed": True,
        "stalled": False,
    }
    out = HERE / f"gate_b_{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "finite_log"}, indent=2), flush=True)
    print(f"WROTE {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
