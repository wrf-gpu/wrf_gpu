"""Retained-evidence savepoint ladder at d03 step 1: SP1 -> SP2 -> SP3 -> SP4.

Sprint: 2026-07-18-v0234-dycore-suboperator-kimi. CPU-only, retained evidence.

The seven conditioning-ensemble archives contain the full momsp chain
(sp1/sp2/sp3/sp4) for every member at every step. The GPU first-interval arm
captured the same chain. This script extracts step-1 sp2_pbl and sp3_tendf
rank dumps from all seven member archives, reassembles them, and runs the
preregistered-style envelope test at EVERY chain rung, so the first
envelope-exceeding boundary is known before any new instrumentation is
designed.

Baselines: unperturbed WRF reassembly from the Kimi baseline dumps
(<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/momsp_dumps).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-dycore-suboperator-kimi")
sys.path.insert(0, str(REPO))
from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble  # noqa: E402

MEMBERS_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_conditioning_ensemble_gpt/members")
BASE_DUMPS = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/momsp_dumps")
BASE_CACHE = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/compare/wrf_global_cache")
GPU_SAVEPOINTS = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_first_interval_momentum2/savepoints"
)
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/ladder_step1")
OUT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/step1-savepoint-ladder.json"

MEMBERS = ("control-repeat", "mask-a-minus", "mask-a-plus",
           "mask-b-minus", "mask-b-plus", "mask-c-minus", "mask-c-plus")
CHAIN = {
    "sp1_entry": ("u", "v"),
    "sp2_pbl": ("rublten", "rvblten"),
    "sp3_tendf": ("ru_tendf", "rv_tendf"),
    "sp4_exit": ("u", "v"),
}


def load_gpu(tag: str, field: str, step: int, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.load(GPU_SAVEPOINTS / f"step{step:06d}_{tag}__{field}.npy")
    if arr.shape == shape:
        return arr
    if arr.ndim == 3 and arr.shape[0] == shape[0] + 1 and arr.shape[1:] == shape[1:]:
        return arr[: shape[0]]
    raise RuntimeError(f"gpu {tag}/{field}/{step} {arr.shape} != {shape}")


def combined_rmse(diffs: list[np.ndarray]) -> float:
    sse = sum(float(np.sum(np.square(d, dtype=np.float64), dtype=np.float64)) for d in diffs)
    n = sum(d.size for d in diffs)
    return float(math.sqrt(sse / n))


def main() -> None:
    # 1. extract step-1 sp2/sp3 dumps from each member archive
    for member in MEMBERS:
        dest = SCRATCH / member
        marker = dest / ".extract_complete"
        if marker.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        archive = MEMBERS_ROOT / member / "evidence" / "momsp-dumps.tar.zst"
        patterns = ["momsp_dumps/rank*/meta.txt"]
        for tag, fields in CHAIN.items():
            for field in fields:
                patterns.append(f"momsp_dumps/rank*/step000001_{tag}__{field}.f64")
        # sp2 also carries mut (2D) - keep for provenance completeness
        patterns.append("momsp_dumps/rank*/step000001_sp2_pbl__mut.f64")
        subprocess.run(["tar", "-I", "zstd", "-xf", str(archive), "-C", str(dest),
                        "--wildcards", *patterns], check=True)
        marker.write_text("ok\n")
        print(f"extracted {member}", flush=True)

    # 2. baseline reassembly (unperturbed WRF)
    base_ranks = reassemble.load_ranks(BASE_DUMPS)
    baseline: dict[tuple[str, str], np.ndarray] = {}
    for tag, fields in CHAIN.items():
        for field in fields:
            key = f"{tag}__{field}"
            if tag in ("sp1_entry", "sp4_exit"):
                baseline[(tag, field)] = np.load(BASE_CACHE / f"step000001_{key}.npy")
            else:
                baseline[(tag, field)] = reassemble.reassemble3d(key, 1, base_ranks)
    print("baseline reassembled", flush=True)

    # 3. member diffs + envelope at each rung
    ladder: dict[str, dict] = {}
    member_rmse: dict[str, dict[str, float]] = {tag: {} for tag in CHAIN}
    member_rank_cache: dict[str, list] = {}
    for member in MEMBERS:
        member_rank_cache[member] = reassemble.load_ranks(SCRATCH / member / "momsp_dumps")
    for tag, fields in CHAIN.items():
        for member in MEMBERS:
            if member == "control-repeat":
                continue
            diffs = []
            for field in fields:
                arr = reassemble.reassemble3d(f"{tag}__{field}", 1, member_rank_cache[member])
                diffs.append(arr - baseline[(tag, field)])
            member_rmse[tag][member] = combined_rmse(diffs)
        print(f"rung {tag}: members done", flush=True)

    # 4. GPU diffs
    gpu_rmse: dict[str, float] = {}
    for tag, fields in CHAIN.items():
        diffs = [load_gpu(tag, field, 1, baseline[(tag, field)].shape) - baseline[(tag, field)]
                 for field in fields]
        gpu_rmse[tag] = combined_rmse(diffs)

    # 4b. control bitwise identity at every rung
    control_bitwise = {}
    for tag, fields in CHAIN.items():
        control_bitwise[tag] = all(
            np.array_equal(
                reassemble.reassemble3d(f"{tag}__{field}", 1, member_rank_cache["control-repeat"]),
                baseline[(tag, field)])
            for field in fields
        )

    # 5. ladder assembly
    for tag in CHAIN:
        members = member_rmse[tag]
        lo = min(members.values()) if members else None
        hi = max(members.values()) if members else None
        gpu = gpu_rmse[tag]
        ladder[tag] = {
            "gpu_vs_wrf_rmse": gpu,
            "member_rmse": members,
            "wrf_envelope_min": lo,
            "wrf_envelope_max": hi,
            "gpu_inside_literal_envelope": (lo is not None and lo <= gpu <= hi),
            "gpu_to_member_max_ratio": (gpu / hi if hi else None),
        }

    payload = {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.step1-savepoint-ladder.v1",
        "step": 1,
        "control_bitwise_equal_to_baseline": control_bitwise,
        "ladder": ladder,
    }
    payload["proof_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(ladder, indent=1, sort_keys=True))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
