#!/usr/bin/env python
"""v0.15 kernel probe — A/B wall-clock + fp64 bit-identity harness.

Runs N production 1h forecast calls (rewindow between hours, exactly the
pipeline cadence), records per-hour walls, and sha256-hashes EVERY state leaf
after each hour. Two runs are bit-identical iff their hash maps are equal.

Usage:
  python probe_ab_identity.py --tag baseline [--hours 3]
Env knobs under test are set by the caller (GPUWRF_ACOUSTIC_UNROLL, XLA_FLAGS,
GPUWRF_THOMPSON_SED_UNROLL, ...).

Artifact: proofs/perf/v015/ab_<tag>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
HERE = Path(__file__).resolve().parent


def state_hashes(state) -> dict[str, str]:
    out = {}
    for name, value in dp._field_items(state):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        out[name] = hashlib.sha256(arr.tobytes()).hexdigest()[:16] + f":{arr.dtype}{list(arr.shape)}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hours", type=int, default=3)
    ap.add_argument("--dump-state", action="store_true",
                    help="also save every numeric leaf of the final state to "
                         "ab_<tag>_state.npz (v0.15 S1 tiered field gate input)")
    args = ap.parse_args()

    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=args.hours,
        output_dir=Path(f"/tmp/v015_perf/ab_{args.tag}"),
        proof_dir=Path(f"/tmp/v015_perf/ab_{args.tag}/proofs"),
        run_root=PROBE, domain="d01",
    )
    case, run_dir = dp._build_real_case(config)
    state = case.state
    boundary_leaves = dp._capture_boundary_leaves(state, case.namelist)
    window_s = dp._boundary_window_cadence_s(case.namelist)
    record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)

    walls = []
    hashes = {}
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
        hashes[f"hour{hour}"] = state_hashes(state)
        print(f"{args.tag} hour{hour}: {walls[-1]}s", flush=True)

    host_overhead = {}
    t0 = time.perf_counter()
    s_full = dp.finite_summary(state)
    host_overhead["finite_summary_s"] = round(time.perf_counter() - t0, 4)
    host_overhead["all_finite"] = bool(s_full["all_finite"])
    if hasattr(dp, "finite_guard_summary"):
        dp.finite_guard_summary(state)  # warm the jitted reduce
        t0 = time.perf_counter()
        s_fast = dp.finite_guard_summary(state)
        host_overhead["finite_guard_summary_s"] = round(time.perf_counter() - t0, 4)
        host_overhead["guard_agrees"] = bool(s_fast["all_finite"]) == bool(s_full["all_finite"])

    payload = {
        "schema": "V015ABIdentity",
        "tag": args.tag,
        "host_overhead": host_overhead,
        "env": {
            k: os.environ.get(k)
            for k in (
                "GPUWRF_ACOUSTIC_UNROLL", "GPUWRF_THOMPSON_SED_UNROLL",
                "GPUWRF_THOMAS_UNROLL", "GPUWRF_MYNN_COND_NITER",
                "GPUWRF_MYNN_COND_UNROLL", "GPUWRF_MYNN_EDMF_LEVEL_UNROLL",
                "GPUWRF_MYNN_BOULAC_FP32", "XLA_FLAGS",
            )
        },
        "per_hour_wall_s": walls,
        "steady_ms_per_step": round(walls[-1] / 200.0 * 1000.0, 2),
        "hashes": hashes,
    }
    out = HERE / f"ab_{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.dump_state:
        leaves = {}
        for name, value in dp._field_items(state):
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if np.issubdtype(arr.dtype, np.number):
                leaves[name] = arr
        np.savez_compressed(HERE / f"ab_{args.tag}_state.npz", **leaves)
        print(f"dumped {len(leaves)} leaves -> ab_{args.tag}_state.npz", flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != "hashes"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
