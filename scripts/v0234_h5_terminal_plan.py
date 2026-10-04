#!/usr/bin/env python3
"""Seal the CPU-only plan and measured ETA for the H5 step-9000 terminal arm.

This script imports neither JAX nor gpuwrf.  It authenticates the accepted
step-3800 result, the original pinned step-9000 reference proof and the two
terminal frames, derives the wall-clock estimate from recorded timings, and
pre-registers the unchanged release gates plus the NW/QKE watch diagnostics.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
BASE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
LINEAGE = BASE / "corrected_ni_rca_max_22c2bd7a"
REFERENCE_NAMESPACE = LINEAGE / "v0234_gpt_v10_replay_c17fca1e201ae106_reference"

STEP3800_RESULT = SPRINT / "GPU_STEP3800_RESULT_28185746394c972f.json"
STEP3800_RESULT_FILE_SHA256 = (
    "a209d4146eed15cdf54a4138ec1e762a6cb5de9bbfbdc1ffd7a7aadd82c37b38"
)
STEP3800_RESULT_PROOF_SHA256 = (
    "22ce7afa4a0262f43eacc7c0cc3f07c203685cf127dda4673d42462b5138bfff"
)
REFERENCE_PROOF = REFERENCE_NAMESPACE / "v10-scoped-proof.json"
REFERENCE_PROOF_FILE_SHA256 = (
    "a75adea510164bd8411d846b546ab9cd971e54055363da22ebe3304bd69ae957"
)
REFERENCE_PROOF_CANONICAL_SHA256 = (
    "ee9843b7b6f068c675bab28bfbdc37367767287b1042fe7f50e998da135adaa3"
)

FRAME = "wrfout_d03_2025-03-01_15:00:00"
CPU_FRAME = BASE / "run/wrf" / FRAME
CPU_FRAME_SHA256 = "1ea00bd68bfa4098abd14d49e031389bae53bd0c921c80b2af264cd9dcbe2f0c"
REFERENCE_FRAME = REFERENCE_NAMESPACE / "gpu-output" / FRAME
REFERENCE_FRAME_SHA256 = (
    "2adcb7b029ed5a45b53a1084c0f10d02490a266a57a1a45c01e4bc7378b8ce5b"
)

AUTOTUNE_PIN = REFERENCE_NAMESPACE / "autotune-results.pb"
AUTOTUNE_PIN_SHA256 = (
    "6a0f30bc8e2ab1ca646ab346f85221565149f4a63b04193a9d76dc334e57ce18"
)

FROZEN_TERMINAL_RMSE = {
    "PSFC": 20.22747532736003,
    "T": 0.5421680888949643,
    "T2": 1.352612988238872,
    "U": 1.1323567330094144,
    "U10": 1.9737860008971808,
    "V": 1.1318203205639872,
    "V10": 2.1128268857679338,
    "W": 0.18633111790197587,
}
REFERENCE_TERMINAL_RMSE = {
    "PSFC": 16.683785071461703,
    "T": 0.5044488341709127,
    "T2": 1.1948902307997455,
    "U": 1.0112925412962208,
    "U10": 1.7860453538824674,
    "V": 1.1644191535446722,
    "V10": 2.3086370774842995,
    "W": 0.1783222793367627,
}

FORBIDDEN_PRIOR_NONCES = (
    "ad0e60072a2a0df1ceacf0f5593e3834f2d0da10086f46316375f8f17aa",
    "8520924a5f97acc1ea8337bb16ea860903b8436e2d30b10b6efca9d54f55552c",
    "ca1df30181823579e35b5c3dbadfa09782cfde6104efe3793937ed9d38e02f2a",
    "fbb2716e855e4fe572d1bbef538f8b26ce5f4a106a4ed8f7b44b41400e852687",
    "28185746394c972f906031a9f8b28510faef6b94b903010fb25bdba2c26ee5ad",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash drift for {path}: {actual} != {expected}")


def terminal_reference_diagnostics() -> tuple[dict[str, float], float, float]:
    strict: dict[str, float] = {}
    with Dataset(str(CPU_FRAME)) as cpu, Dataset(str(REFERENCE_FRAME)) as reference:
        for field in FROZEN_TERMINAL_RMSE:
            truth = np.asarray(cpu.variables[field][:], dtype=np.float64)
            arm = np.asarray(reference.variables[field][:], dtype=np.float64)
            strict[field] = rms(arm - truth)

        cpu_v10 = np.asarray(cpu.variables["V10"][:], dtype=np.float64)[0]
        ref_v10 = np.asarray(reference.variables["V10"][:], dtype=np.float64)[0]
        ny, nx = cpu_v10.shape
        nw = (slice(ny // 2, ny), slice(0, nx // 2))
        nw_v10_rmse = rms(ref_v10[nw] - cpu_v10[nw])

        cpu_qke = np.asarray(cpu.variables["QKE"][:], dtype=np.float64)
        ref_qke = np.asarray(reference.variables["QKE"][:], dtype=np.float64)
        qke_rmse = rms(ref_qke - cpu_qke)
    return strict, nw_v10_rmse, qke_rmse


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} OUTPUT_JSON")
    destination = Path(sys.argv[1])

    for path, expected in (
        (STEP3800_RESULT, STEP3800_RESULT_FILE_SHA256),
        (REFERENCE_PROOF, REFERENCE_PROOF_FILE_SHA256),
        (CPU_FRAME, CPU_FRAME_SHA256),
        (REFERENCE_FRAME, REFERENCE_FRAME_SHA256),
        (AUTOTUNE_PIN, AUTOTUNE_PIN_SHA256),
    ):
        require_hash(path, expected)

    step3800 = read_json(STEP3800_RESULT)
    reference = read_json(REFERENCE_PROOF)
    if canonical(step3800) != STEP3800_RESULT_PROOF_SHA256:
        raise RuntimeError("step-3800 canonical proof drift")
    if canonical(reference) != REFERENCE_PROOF_CANONICAL_SHA256:
        raise RuntimeError("reference canonical proof drift")
    if step3800.get("verdict") != "V10_H5_STEP3800_STRICT_GREEN":
        raise RuntimeError("step-3800 prerequisite is not GREEN")
    if reference.get("verdict") != "V10_V_STRICT_RED_RETAINED":
        raise RuntimeError(f"unexpected reference verdict: {reference.get('verdict')}")

    measured_strict, reference_nw_v10, reference_qke = terminal_reference_diagnostics()
    for field, expected in REFERENCE_TERMINAL_RMSE.items():
        if measured_strict[field] != expected:
            raise RuntimeError(
                f"terminal reference score drift for {field}: "
                f"{measured_strict[field]} != {expected}"
            )

    candidate_timing = step3800["timing"]
    candidate_overhead = (
        float(candidate_timing["wall_seconds"])
        - float(candidate_timing["domain_load_seconds"])
        - float(candidate_timing["model_through_step3800_seconds"])
    )
    direct_scaled = (
        float(candidate_timing["domain_load_seconds"])
        + float(candidate_timing["model_through_step3800_seconds"]) * 9000.0 / 3800.0
        + candidate_overhead
    )

    started = dt.datetime.fromisoformat(reference["timing"]["started_utc"])
    finished = dt.datetime.fromisoformat(reference["timing"]["finished_utc"])
    reference_end_to_end = (finished - started).total_seconds()
    segments = reference["prefix"]["segments"]
    if segments[-1]["own_steps"] != {"d01": 1000, "d02": 3000, "d03": 9000}:
        raise RuntimeError("reference segment inventory does not end at d03 step 9000")
    segment_sum = sum(float(segment["wall_seconds"]) for segment in segments)
    through_3618 = sum(float(segment["wall_seconds"]) for segment in segments[:6])
    fraction_of_segment_6 = (3800.0 - 3618.0) / (4221.0 - 3618.0)
    reference_segment_to_3800 = (
        through_3618 + fraction_of_segment_6 * float(segments[6]["wall_seconds"])
    )
    reference_segment_after_3800 = segment_sum - reference_segment_to_3800
    conservative = float(candidate_timing["wall_seconds"]) + reference_segment_after_3800

    if not (direct_scaled < conservative < reference_end_to_end):
        raise RuntimeError("measured ETA bracket lost its expected ordering")

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.h5-terminal-step9000-plan.v1",
        "verdict": "H5_STEP9000_CPU_PLAN_READY__FRESH_NONCE_REQUIRED",
        "accepted_candidate": {
            "commit": "3b81fb5b093639e70c12cce87d602c45b326b18b",
            "src_gpuwrf_tree": "e627605f6a8bc0dc23f5c474be4bb532b99297c1",
            "step3800_result": {
                "path": str(STEP3800_RESULT.resolve()),
                "file_sha256": STEP3800_RESULT_FILE_SHA256,
                "canonical_sha256": STEP3800_RESULT_PROOF_SHA256,
            },
        },
        "terminal_authority": {
            "reference_proof": {
                "path": str(REFERENCE_PROOF.resolve()),
                "file_sha256": REFERENCE_PROOF_FILE_SHA256,
                "canonical_sha256": REFERENCE_PROOF_CANONICAL_SHA256,
            },
            "cpu_frame": {
                "path": str(CPU_FRAME.resolve()),
                "sha256": CPU_FRAME_SHA256,
            },
            "pre_fix_reference_frame": {
                "path": str(REFERENCE_FRAME.resolve()),
                "sha256": REFERENCE_FRAME_SHA256,
            },
            "autotune_pin": {
                "path": str(AUTOTUNE_PIN.resolve()),
                "sha256": AUTOTUNE_PIN_SHA256,
                "read_only": True,
            },
        },
        "measured_wallclock": {
            "step3800_candidate": {
                "wall_seconds": float(candidate_timing["wall_seconds"]),
                "domain_load_seconds": float(candidate_timing["domain_load_seconds"]),
                "model_seconds": float(candidate_timing["model_through_step3800_seconds"]),
                "non_model_overhead_seconds": candidate_overhead,
            },
            "same_arm_linear_step9000_seconds": direct_scaled,
            "same_arm_linear_step9000_minutes": direct_scaled / 60.0,
            "original_reference_end_to_end_seconds": reference_end_to_end,
            "original_reference_end_to_end_minutes": reference_end_to_end / 60.0,
            "original_reference_segment_sum_seconds": segment_sum,
            "original_reference_segment_after_step3800_seconds": (
                reference_segment_after_3800
            ),
            "conservative_candidate_plus_reference_tail_seconds": conservative,
            "conservative_candidate_plus_reference_tail_minutes": conservative / 60.0,
            "expected_range_seconds": [direct_scaled, conservative],
            "expected_range_minutes": [direct_scaled / 60.0, conservative / 60.0],
            "recommended_gpu_reservation_seconds": 7200.0,
            "hard_fail_fast_seconds": 9000.0,
            "hard_bound_extended": False,
            "method": (
                "lower endpoint scales the measured H5 step3800 model time to 9000; "
                "upper endpoint adds the original reference run's measured post-step3800 "
                "segment tail to the measured H5 elapsed time; the 120.31-minute original "
                "end-to-end run is retained as the two-hour scheduling reservation"
            ),
        },
        "fresh_authorization_plan": {
            "nonce_regex": r"[0-9a-f]{32,128}",
            "forbidden_prior_nonces": list(FORBIDDEN_PRIOR_NONCES),
            "namespace_template": "v0234_gpt_v10_rootcause_{nonce[:16]}_step9000",
            "namespace_must_be_absent_and_not_symlink": True,
            "authorization_file_template": "GPU_STEP9000_AUTHORIZATION_{nonce[:16]}.json",
            "lock_label": "v0234-gpt-v10-rootcause",
            "candidate_processes": 1,
            "baseline_processes": 0,
            "root_step_upper_bound": 1000,
            "d03_terminal_step": 9000,
            "horizon_extension": False,
            "baseline_rerun": False,
        },
        "unchanged_terminal_gate": {
            "comparison": "candidate full-array float64 RMSE against pinned CPU-WRF",
            "operator": "less_than_or_equal",
            "ceilings": dict(FROZEN_TERMINAL_RMSE),
            "all_strict_fields_required": sorted(FROZEN_TERMINAL_RMSE),
            "all_arrays_finite": True,
            "static_identity_exact": True,
            "tolerance_change": False,
            "waiver_or_reclassification": False,
        },
        "watch_only_non_gating": {
            "NW_V10": {
                "definition": "north half, west half of full d03 V10 array",
                "pre_fix_reference_rmse": reference_nw_v10,
                "terminal_candidate_value_required": True,
                "report_delta_and_ratio": True,
            },
            "QKE": {
                "definition": "unmasked full-array float64 RMSE against terminal CPU-WRF QKE",
                "pre_fix_reference_rmse": reference_qke,
                "terminal_candidate_value_required": True,
                "report_delta_and_ratio": True,
            },
            "policy": "watch items are always reported but do not alter the frozen release gate",
        },
        "reference_terminal_rmse_recomputed": measured_strict,
        "no_gpu_commands_or_queries": True,
    }
    if re.fullmatch(payload["fresh_authorization_plan"]["nonce_regex"], "0" * 64) is None:
        raise RuntimeError("nonce plan regex is invalid")
    for value in payload["measured_wallclock"]["expected_range_seconds"]:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise RuntimeError("non-finite wall-clock estimate")

    payload["proof_sha256"] = canonical(payload)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": payload["verdict"],
        "expected_minutes": payload["measured_wallclock"]["expected_range_minutes"],
        "proof_sha256": payload["proof_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
