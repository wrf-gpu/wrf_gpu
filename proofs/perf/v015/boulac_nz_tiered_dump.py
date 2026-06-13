"""v0.15 BouLac O(nz) tiered-gate state dump producer.

Runs the standard Switzerland d01 reinit-h36 case (the v0.15 tiered-gate case)
for N hours and dumps every numeric state leaf to ab_boulac_<algo>_state.npz,
where ``algo`` is the BouLac length-scale algorithm:
  - ``scan``  : production O(nz) lax.scan _boulac_length (current branch);
  - ``dense`` : the frozen pre-O(nz) (B,nz,nz) _boulac_length (commit 0b2a7066),
    monkeypatched in BEFORE the pipeline is built so the WHOLE forecast uses it.

Both runs use the v0.15 production defaults (niter=16, fp64, BOULAC_FP32=0); the
ONLY difference is the length-scale algorithm.  The two dumps feed
compare_tiered_identity.py -> the Tier-P field gate vs the v0.14 frozen manifest.
Also reports steady_ms_per_step (the full-pipeline wall) for the perf record.

GPU lock required.  Usage:
  python proofs/perf/v015/boulac_nz_tiered_dump.py --algo {scan,dense} [--hours 3]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
HERE = Path(__file__).resolve().parent


def _patch_dense_if_needed(algo: str):
    """Swap in the frozen dense _boulac_length for the 'dense' baseline.

    Must run BEFORE daily_pipeline (and thus mynn_pbl) is used to build/jit the
    forecast, so the patched function is the one that gets traced.
    """
    import gpuwrf.physics.mynn_pbl as m
    if algo == "dense":
        import _boulac_dense_frozen as df  # noqa: PLC0415
        m._boulac_length = df._boulac_length
        return "dense_nznz_frozen_0b2a7066"
    return "scan_onz_production"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["scan", "dense"], required=True)
    ap.add_argument("--hours", type=int, default=3)
    args = ap.parse_args()

    os.environ["GPUWRF_MYNN_BOULAC_FP32"] = "0"  # production default

    algo_desc = _patch_dense_if_needed(args.algo)
    from gpuwrf.integration import daily_pipeline as dp

    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=args.hours,
        output_dir=Path(f"/tmp/v015_boulac/{args.algo}"),
        proof_dir=Path(f"/tmp/v015_boulac/{args.algo}/proofs"),
        run_root=PROBE, domain="d01",
    )
    case, run_dir = dp._build_real_case(config)
    state = case.state
    boundary_leaves = dp._capture_boundary_leaves(state, case.namelist)
    window_s = dp._boundary_window_cadence_s(case.namelist)
    record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)

    walls = []
    for hour in range(1, args.hours + 1):
        st_in = (
            dp._rewindow_boundary_leaves(
                state, boundary_leaves, segment_start_s=(hour - 1) * 3600.0,
                record_cadence_s=record_s, window_s=window_s,
            )
            if boundary_leaves
            else state
        )
        t0 = time.perf_counter()
        state = dp._default_forecast_fn(st_in, case.namelist, 1.0)
        walls.append(round(time.perf_counter() - t0, 3))
        print(f"[{args.algo}] hour{hour}: {walls[-1]}s", flush=True)

    leaves = {}
    for name, value in dp._field_items(state):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            leaves[name] = arr
    out_npz = HERE / f"ab_boulac_{args.algo}_state.npz"
    np.savez_compressed(out_npz, **leaves)

    steady_ms = round(walls[-1] / 200.0 * 1000.0, 2)  # 1h = 200 steps @ dt=18s
    summary = {
        "algo": args.algo, "algo_desc": algo_desc, "hours": args.hours,
        "per_hour_wall_s": walls, "steady_ms_per_step": steady_ms,
        "leaves": len(leaves), "state_npz": str(out_npz),
    }
    import json
    (HERE / f"ab_boulac_{args.algo}_perf.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
