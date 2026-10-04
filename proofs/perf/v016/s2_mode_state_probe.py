#!/usr/bin/env python
"""Dump a real operational final state for an exact step count and precision mode.

Used by ADR-031 S2 to compare current fp64 against MIXED_PERTURB_FP32 with the
frozen v0.14 tiered-identity manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from gpuwrf.contracts.precision import AcousticPrecisionMode
from gpuwrf.integration import daily_pipeline as dp


PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
OUT_DIR = Path("proofs/perf/v016")


def _state_hashes(state) -> dict[str, str]:
    out = {}
    for name, value in dp._field_items(state):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            out[name] = hashlib.sha256(arr.tobytes()).hexdigest()[:16] + f":{arr.dtype}{list(arr.shape)}"
    return out


def _state_leaves(state) -> dict[str, np.ndarray]:
    leaves = {}
    for name, value in dp._field_items(state):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            leaves[name] = arr
    return leaves


def _namelist_for_mode(namelist, mode: str):
    if mode == "fp64":
        return dataclasses.replace(
            namelist,
            force_fp64=True,
            acoustic_precision_mode=AcousticPrecisionMode.FP64_DEFAULT,
        )
    if mode == "mixed_s2":
        return dataclasses.replace(
            namelist,
            force_fp64=False,
            acoustic_precision_mode=AcousticPrecisionMode.MIXED_PERTURB_FP32,
        )
    raise ValueError(f"unsupported mode {mode!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=("fp64", "mixed_s2"), required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=Path(f"/tmp/v016_s2_mode/{args.tag}"),
        proof_dir=Path(f"/tmp/v016_s2_mode/{args.tag}/proofs"),
        run_root=PROBE,
        domain="d01",
    )
    case, _ = dp._build_real_case(config)
    state = case.state
    namelist = _namelist_for_mode(case.namelist, args.mode)
    boundary_leaves = dp._capture_boundary_leaves(state, namelist)
    record_s = float(
        (case.metadata.get("boundary") or {}).get("interval_seconds")
        or dp._boundary_window_cadence_s(namelist)
    )
    st_in = (
        dp._rewindow_boundary_leaves(
            state,
            boundary_leaves,
            segment_start_s=0.0,
            record_cadence_s=record_s,
            window_s=dp._boundary_window_cadence_s(namelist),
        )
        if boundary_leaves
        else state
    )
    dt_s = float(namelist.dt_s)
    hours = int(args.steps) * dt_s / 3600.0
    t0 = time.perf_counter()
    out = dp._default_forecast_fn(st_in, namelist, float(hours))
    wall_s = time.perf_counter() - t0
    leaves = _state_leaves(out)
    state_path = args.out_dir / f"s2_{args.tag}_state.npz"
    np.savez_compressed(state_path, **leaves)
    payload = {
        "schema": "ADR031S2ModeStateProbe",
        "tag": args.tag,
        "mode": args.mode,
        "steps": int(args.steps),
        "dt_s": dt_s,
        "hours": hours,
        "wall_s": wall_s,
        "state_npz": str(state_path),
        "leaf_count": len(leaves),
        "namelist": {
            "force_fp64": bool(namelist.force_fp64),
            "acoustic_precision_mode": str(namelist.acoustic_precision_mode),
        },
        "env": {
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        },
        "dtypes": {k: str(v.dtype) for k, v in leaves.items()},
        "hashes": _state_hashes(out),
    }
    json_path = args.out_dir / f"s2_{args.tag}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("hashes", "dtypes")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
