"""Boundary-band vs interior decomposition of the retained d03 step-1 SP4 divergence.

Sprint: 2026-07-18-v0234-dycore-suboperator-kimi. CPU-only, retained evidence.

For the retained GPU-vs-WRF trajectory and the WRF-vs-WRF conditioning members,
decompose the SP4 (end-of-step) U/V difference by minimum distance to the d03
domain edge (spec_bdy_width = 5 → rows/cols 0..4 are the specified+relaxation
band). If the systematic GPU-vs-WRF step-1 difference lived in the nest
boundary update, its squared-error share would concentrate in the outer band;
if it lives in the dry-dycore interior, it concentrates inside.

Inputs (all retained, hash-audited):
  GPU:  .../nested_stage_omega_transport_470e6111_first_interval_momentum2/savepoints
  WRF:  <DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/compare/wrf_global_cache
  members: raw spot-check extractions under SCRATCH (mask-a-minus, mask-c-plus)
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

GPU_SAVEPOINTS = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_first_interval_momentum2/savepoints"
)
BASE_CACHE = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/compare/wrf_global_cache")
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/raw_spotcheck")
OUT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-dycore-suboperator-kimi/.agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/band-decomposition.json")

import sys
sys.path.insert(0, "<USER_HOME>/src/wrf_gpu2_wt/v0234-dycore-suboperator-kimi")
from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble  # noqa: E402

SPEC_BDY_WIDTH = 5
BANDS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 9), (10, 19), (20, 10**9))
STEPS = (1, 2, 5, 10, 20, 50, 100, 200)


def distance_to_edge(shape: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = shape
    jj = np.minimum(np.arange(ny), np.arange(ny)[::-1])[:, None]
    ii = np.minimum(np.arange(nx), np.arange(nx)[::-1])[None, :]
    return np.broadcast_to(np.minimum(jj, ii), shape)


def band_profile(diff: np.ndarray, dist: np.ndarray) -> list[dict]:
    sq = np.square(diff, dtype=np.float64)
    total = float(np.sum(sq, dtype=np.float64))
    rows = []
    for lo, hi in BANDS:
        mask = (dist >= lo) & (dist <= hi)
        sse = float(np.sum(sq[mask], dtype=np.float64))
        n = int(np.count_nonzero(mask))
        rows.append({
            "band": f"{lo}-{hi if hi < 10**9 else 'max'}",
            "cells": n,
            "sse": sse,
            "sse_share": sse / total if total > 0 else 0.0,
            "rmse": math.sqrt(sse / n) if n else 0.0,
        })
    return rows


def load_gpu(tag: str, field: str, step: int, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.load(GPU_SAVEPOINTS / f"step{step:06d}_{tag}__{field}.npy")
    if arr.shape == shape:
        return arr
    if arr.ndim == 3 and arr.shape[0] == shape[0] + 1 and arr.shape[1:] == shape[1:]:
        return arr[: shape[0]]
    raise RuntimeError(f"gpu shape {arr.shape} != {shape}")


def combined(rows_u: list[dict], rows_v: list[dict]) -> list[dict]:
    out = []
    for ru, rv in zip(rows_u, rows_v):
        sse = ru["sse"] + rv["sse"]
        n = ru["cells"] + rv["cells"]
        out.append({
            "band": ru["band"],
            "cells": n,
            "sse": sse,
            "sse_share": None,  # filled by caller
            "rmse": math.sqrt(sse / n),
        })
    total = sum(r["sse"] for r in out)
    for r in out:
        r["sse_share"] = r["sse"] / total if total > 0 else 0.0
    return out


def main() -> None:
    result = {"spec_bdy_width": SPEC_BDY_WIDTH, "steps": {}}
    member_ranks = {
        m: reassemble.load_ranks(SCRATCH / m / "momsp_dumps")
        for m in ("mask-a-minus", "mask-c-plus")
    }
    for step in STEPS:
        entry = {"gpu_vs_wrf": {}, "members_vs_wrf": {}}
        for field in ("u", "v"):
            cache = np.load(BASE_CACHE / f"step{step:06d}_sp4_exit__{field}.npy")
            gpu = load_gpu("sp4_exit", field, step, cache.shape)
            dist = distance_to_edge(cache.shape)
            entry["gpu_vs_wrf"][field] = band_profile(gpu - cache, dist)
            for member, ranks in member_ranks.items():
                if step in (1, 200):
                    arr = reassemble.reassemble3d(f"sp4_exit__{field}", step, ranks)
                    entry["members_vs_wrf"].setdefault(member, {})[field] = band_profile(arr - cache, dist)
        entry["gpu_vs_wrf_combined"] = combined(entry["gpu_vs_wrf"]["u"], entry["gpu_vs_wrf"]["v"])
        for member in entry["members_vs_wrf"]:
            entry["members_vs_wrf"][member] = {
                "combined": combined(entry["members_vs_wrf"][member]["u"], entry["members_vs_wrf"][member]["v"])
            }
        result["steps"][str(step)] = entry
        g = entry["gpu_vs_wrf_combined"]
        band5 = sum(r["sse_share"] for r in g[:5])
        print(f"step {step:3d}: GPU SP4 sse_share in spec band (0-4) = {band5:.4f}, "
              f"interior(>=10) = {sum(r['sse_share'] for r in g[6:]):.4f}")
    payload = {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.band-decomposition.v1",
        **result,
    }
    payload["proof_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
