#!/usr/bin/env python
"""Run the real operational forecast for a FEW steps and dump the final state.

Used to measure the PER-STEP delta seed between base and S0+S1 (before chaos
amplifies it over a full hour). Deterministic XLA is assumed (set by caller).

Usage: fewstep_probe.py --tag NAME --hours 0.01   (0.01h ~ 2 steps at dt~18s)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=2,
                    help="number of dt steps; hours = steps*dt/3600 (exact)")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=1,
        output_dir=Path(f"/tmp/v016_fewstep/{args.tag}"),
        proof_dir=Path(f"/tmp/v016_fewstep/{args.tag}/proofs"),
        run_root=PROBE, domain="d01",
    )
    case, _ = dp._build_real_case(config)
    state = case.state
    boundary_leaves = dp._capture_boundary_leaves(state, case.namelist)
    record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or
                     dp._boundary_window_cadence_s(case.namelist))
    st_in = (dp._rewindow_boundary_leaves(state, boundary_leaves, segment_start_s=0.0,
                                          record_cadence_s=record_s,
                                          window_s=dp._boundary_window_cadence_s(case.namelist))
             if boundary_leaves else state)
    # hours = steps*dt/3600, snapped so _steps_for_hours rounds back to exactly
    # args.steps (avoids the 1e-9 integer-step rejection from float hours).
    dt_s = float(case.namelist.dt_s)
    hours = (int(args.steps) * dt_s) / 3600.0
    out = dp._default_forecast_fn(st_in, case.namelist, float(hours))
    print(f"[fewstep {args.tag}] dt={dt_s}s steps={args.steps} hours={hours!r}", flush=True)
    leaves = {}
    for name, value in dp._field_items(out):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            leaves[name] = arr
    np.savez_compressed(here / f"fewstep_{args.tag}.npz", **leaves)
    print(f"[fewstep {args.tag}] steps={args.steps} dumped {len(leaves)} leaves", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
