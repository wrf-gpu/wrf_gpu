#!/usr/bin/env python3
"""Validate H5, register its retained partial paired case, and seal closeout."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
RUN = (
    CASE_ROOT
    / "corrected_ni_rca_max_22c2bd7a"
    / "v0234_gpt_v10_rootcause_30b5cf89ff6027c4_full18h"
)
CPU_MANIFEST = CASE_ROOT / "run/cpu_oracle_manifest.json"
PAIRED_RUN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/gpu_oracles_paired_v1/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run"
)
PAIRED_MANIFEST = PAIRED_RUN / "gpu_paired_case_manifest.json"
PAIRED_RECEIPT = PAIRED_RUN / "gpu_paired_case_manifest.receipt.json"
OUTPUT = SPRINT / "TERMINAL_CLOSEOUT.json"
GATE = RUN / "step9000-result.json"
BLOCKER = RUN / "step9000-blocker.json"
LAUNCHER_LOG = RUN / "launcher.log"
METADATA = SPRINT / "TERMINAL_CASE_METADATA.json"

CASE_ID = "20250228_18z"
GRID_ID = "tenerife_operational_v2_fullbuffer_111x93"
GRID_SHA256 = "8ec38f8e1a90d70d390ad533a9c3cf3546aca242f37a93847c7441b925e7c688"
FORCING_SHA256 = "5256342931d4db2853426b05a3e10e67c3fac37e969387041ada6ec83c551fe4"
CPU_MANIFEST_SHA256 = "c8f087d7d6ca2e0b8d2e959429eb81c4584d3d1f94ef944723774b347ba831ad"
CPU_REGISTRY_ARTIFACT_ID = "dsr_6843276c"
CANDIDATE_COMMIT = "3b81fb5b093639e70c12cce87d602c45b326b18b"
CANDIDATE_TREE = "e627605f6a8bc0dc23f5c474be4bb532b99297c1"
GPUWRF_VERSION = "0.23.3"
RUNNER_COMMIT = "77f2d0ef"
EXPECTED_COUNTS = {"d01": 19, "d02": 19, "d03": 55}
OBSERVED_COUNTS = {"d01": 18, "d02": 18, "d03": 52}
GATES = {
    "PSFC": 20.22747532736003,
    "T": 0.5421680888949643,
    "T2": 1.352612988238872,
    "U": 1.1323567330094144,
    "U10": 1.9737860008971808,
    "V": 1.1318203205639872,
    "V10": 2.1128268857679338,
    "W": 0.18633111790197587,
}
EXPECTED_STEP3800_SHA256 = (
    "e43f35012bfef4a2b48d776a79258fdfb4e99b00067a540920487fe95ac33516"
)
COMPLETION_PENDING = (
    "late-ice/Ni non-finite at step10400 (17:20 UTC), separate known issue "
    "tracked independently, may be extended later with the same split_group/case_id "
    "once fixed -- not a blocker for using the 18/18/52 partial now"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical(payload: Mapping[str, Any], *, exclude: str | None = None) -> str:
    clean = dict(payload)
    if exclude is not None:
        clean.pop(exclude, None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_self_hashed(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or symlinked proof: {path}")
    payload = json.loads(path.read_text())
    observed = canonical(payload, exclude="proof_sha256")
    if payload.get("proof_sha256") != observed:
        raise RuntimeError(f"canonical proof mismatch: {path}")
    return payload, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256(path),
        "canonical_sha256": observed,
    }


def git(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def write_new_or_verify(path: Path, payload: Mapping[str, Any]) -> str:
    """Create an immutable JSON artifact, or verify a byte-exact prior creation."""
    data = json_bytes(payload)
    expected = hashlib.sha256(data).hexdigest()
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return expected


def valid_time(path: Path) -> str:
    stamp = path.name.split("wrfout_", 1)[1].split("_", 1)[1]
    return stamp.replace("_", "T", 1) + "Z"


def load_frame_manifest(expected_variables: list[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    from netCDF4 import Dataset

    output_dir = RUN / "gpu-output"
    manifests: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for domain in ("d01", "d02", "d03"):
        rows: list[dict[str, Any]] = []
        for path in sorted(output_dir.glob(f"wrfout_{domain}_*")):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"invalid output artifact: {path}")
            with Dataset(str(path)) as frame:
                variables = list(frame.variables)
            if variables != expected_variables:
                raise RuntimeError(f"375-variable inventory drifted: {path}")
            rows.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "valid_time": valid_time(path),
                    "variable_count": len(variables),
                }
            )
        manifests[domain] = rows
        counts[domain] = len(rows)
    return manifests, counts


def validate_authority() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    gate, gate_row = read_self_hashed(GATE)
    blocker, blocker_row = read_self_hashed(BLOCKER)
    if gate.get("verdict") != "V10_H5_STEP9000_STRICT_GREEN":
        raise RuntimeError("terminal gate is not strict green")
    if gate.get("own_steps_at_score") != {"d01": 1000, "d02": 3000, "d03": 9000}:
        raise RuntimeError("terminal gate clock changed")
    if gate.get("d03_steps_beyond_9000_dispatched_before_score") != 0:
        raise RuntimeError("post-gate steps were dispatched before scoring")
    decision = gate["score"]["decision"]
    scores = decision["strict_rmse"]
    if (
        decision.get("passed") is not True
        or decision.get("finite") is not True
        or decision.get("static_identity_pass") is not True
        or decision.get("frozen_field_regressions") != {}
        or set(scores) != set(GATES)
        or any(float(scores[name]) > limit for name, limit in GATES.items())
    ):
        raise RuntimeError("terminal gate semantics changed")
    correlations = gate["score"]["d03_spatial_pearson_correlation"]
    if set(correlations) != {"U10", "V10"}:
        raise RuntimeError("terminal correlation inventory changed")

    expected_exception = (
        "non-finite prognostic state detected: domain=d03 field=Ni level=0 "
        "step=10400 sim_time_s=62400 first_index=(0, 1, 1)"
    )
    if (
        blocker.get("exception_type") != "NonFiniteStateError"
        or blocker.get("exception") != expected_exception
        or blocker.get("scientific_falsification") is not False
    ):
        raise RuntimeError("late-Ni blocker semantics changed")

    if sha256(CPU_MANIFEST) != CPU_MANIFEST_SHA256:
        raise RuntimeError("CPU oracle manifest hash changed")
    cpu = json.loads(CPU_MANIFEST.read_text())
    if (
        cpu.get("schema") != "tenerife_fullbuffer_cpu_oracle_v1"
        or cpu.get("case_id") != CASE_ID
        or cpu.get("grid_id") != GRID_ID
        or cpu.get("grid_sha256") != GRID_SHA256
        or cpu.get("forcing_sha256") != FORCING_SHA256
        or cpu.get("status") != "complete"
        or cpu.get("qa_status") != "pass"
    ):
        raise RuntimeError("CPU oracle manifest authority changed")

    metadata = json.loads(METADATA.read_text())
    if (
        metadata["grid"]["id"] != GRID_ID
        or metadata["grid"]["grid_sha256"] != GRID_SHA256
        or metadata["forcing"]["sha256"] != FORCING_SHA256
        or metadata["candidate"]["commit"] != CANDIDATE_COMMIT
        or metadata["candidate"]["src_gpuwrf_tree"] != CANDIDATE_TREE
    ):
        raise RuntimeError("sealed case metadata changed")
    expected_variables = metadata["planned_full_output"]["variables"]
    if len(expected_variables) != 375 or len(set(expected_variables)) != 375:
        raise RuntimeError("sealed variable inventory changed")
    return gate, gate_row, blocker, blocker_row, expected_variables


def validate_source_and_frames(expected_variables: list[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    manifests, counts = load_frame_manifest(expected_variables)
    if counts != OBSERVED_COUNTS or sum(counts.values()) != 88:
        raise RuntimeError(f"observed partial output counts changed: {counts}")
    expected_last = {
        "d01": "2025-03-01T17:00:36Z",
        "d02": "2025-03-01T17:00:00Z",
        "d03": "2025-03-01T17:00:00Z",
    }
    if any(manifests[domain][-1]["valid_time"] != stamp for domain, stamp in expected_last.items()):
        raise RuntimeError("partial endpoint changed")
    step3800 = RUN / "gpu-output/wrfout_d03_2025-03-01_06:20:00"
    if sha256(step3800) != EXPECTED_STEP3800_SHA256:
        raise RuntimeError("full-cadence trajectory differs at accepted step3800")

    runner_path = ROOT / "scripts/v0234_h5_terminal_step9000.py"
    if git("rev-parse", f"{RUNNER_COMMIT}^{{commit}}")[:8] != RUNNER_COMMIT:
        raise RuntimeError("runner commit authority changed")
    if git("rev-parse", f"{RUNNER_COMMIT}:src/gpuwrf") != CANDIDATE_TREE:
        raise RuntimeError("runner model tree changed")
    runner_git = subprocess.check_output(
        ("git", "-C", str(ROOT), "show", f"{RUNNER_COMMIT}:scripts/v0234_h5_terminal_step9000.py")
    )
    if hashlib.sha256(runner_git).hexdigest() != sha256(runner_path):
        raise RuntimeError("executed runner differs from committed runner")
    candidate_init = subprocess.check_output(
        ("git", "-C", str(ROOT), "show", f"{CANDIDATE_COMMIT}:src/gpuwrf/__init__.py"),
        text=True,
    )
    if f'__version__ = "{GPUWRF_VERSION}"' not in candidate_init:
        raise RuntimeError("candidate gpuwrf version changed")
    return manifests, counts


def build_paired_manifest(
    gate: dict[str, Any],
    blocker: dict[str, Any],
    manifests: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    expected_variables: list[str],
) -> dict[str, Any]:
    scores = gate["score"]["decision"]["strict_rmse"]
    correlations = gate["score"]["d03_spatial_pearson_correlation"]
    raw_receipt = canonical(manifests)
    payload: dict[str, Any] = {
        "schema": "wrf_downscale.gpu_paired_case_manifest.v1",
        "status": "validated_partial",
        "case_id": CASE_ID,
        "split_group": CASE_ID,
        "grid_id": GRID_ID,
        "grid_sha256": GRID_SHA256,
        "forcing_sha256": FORCING_SHA256,
        "gpuwrf_version": GPUWRF_VERSION,
        "gpuwrf_commit": CANDIDATE_COMMIT,
        "gpuwrf_src_tree": CANDIDATE_TREE,
        "provenance": "gpu_output_paired_cpu_validated_partial",
        "valid_start": "2025-03-01T00:00:00Z",
        "valid_end": "2025-03-01T17:00:00Z",
        "target_valid_end": "2025-03-01T18:00:00Z",
        "frame_counts": counts,
        "target_frame_counts": EXPECTED_COUNTS,
        "completion_pending": COMPLETION_PENDING,
        "gate_verdict": "PASS",
        "gate_evaluated_at": {
            "domain": "d03",
            "step": 9000,
            "valid_time": "2025-03-01T15:00:00Z",
            "metric": "full-array RMSE against pinned CPU-WRF oracle",
        },
        "gate_thresholds": GATES,
        "variable_gates": {
            name: {
                "metric": "RMSE",
                "value": float(scores[name]),
                "operator": "<=",
                "threshold": limit,
                "pass": float(scores[name]) <= limit,
            }
            for name, limit in GATES.items()
        },
        "d03_pearson_correlation": {
            "U10": float(correlations["U10"]),
            "V10": float(correlations["V10"]),
        },
        "gate_invariants": {
            "all_strict_fields_finite": True,
            "static_identity": gate["score"]["decision"]["static_identity"],
            "frozen_field_regressions": {},
            "tolerance_changed": False,
            "baseline_rerun": False,
            "post_gate_horizon_changes_gate": False,
        },
        "cpu_oracle_reference": {
            "path": str(CPU_MANIFEST),
            "manifest_sha256": CPU_MANIFEST_SHA256,
            "artifact_id": CPU_REGISTRY_ARTIFACT_ID,
        },
        "raw_wrfout_paths": {
            domain: [row["path"] for row in rows] for domain, rows in manifests.items()
        },
        "raw_wrfout_artifacts": manifests,
        "raw_wrfout_receipt_sha256": raw_receipt,
        "raw_wrfout_total_bytes": sum(
            row["bytes"] for rows in manifests.values() for row in rows
        ),
        "output_variable_count": len(expected_variables),
        "output_variables": expected_variables,
        "output_variables_sha256": canonical({"variables": expected_variables}),
        "validation_evidence": {
            "step9000_gate": {
                "path": str(GATE),
                "file_sha256": sha256(GATE),
                "proof_sha256": gate["proof_sha256"],
            },
            "post_acceptance_completion_blocker": {
                "path": str(BLOCKER),
                "file_sha256": sha256(BLOCKER),
                "proof_sha256": blocker["proof_sha256"],
                "classification": "separate_known_late_ice_issue",
            },
        },
    }
    payload["receipt_sha256"] = canonical(payload, exclude="receipt_sha256")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and construct canonical payloads without creating artifacts",
    )
    args = parser.parse_args()

    gate, gate_row, blocker, blocker_row, expected_variables = validate_authority()
    manifests, counts = validate_source_and_frames(expected_variables)
    paired = build_paired_manifest(gate, blocker, manifests, counts, expected_variables)
    paired_bytes = json_bytes(paired)
    paired_file_sha = hashlib.sha256(paired_bytes).hexdigest()
    receipt: dict[str, Any] = {
        "schema": "wrf_downscale.gpu_paired_case_manifest_receipt.v1",
        "case_id": CASE_ID,
        "split_group": CASE_ID,
        "manifest_path": str(PAIRED_MANIFEST),
        "manifest_file_sha256": paired_file_sha,
        "manifest_receipt_sha256": paired["receipt_sha256"],
        "raw_wrfout_receipt_sha256": paired["raw_wrfout_receipt_sha256"],
        "frame_counts": counts,
        "target_frame_counts": EXPECTED_COUNTS,
        "status": "VALIDATED_PARTIAL_REGISTERED",
    }
    receipt["receipt_sha256"] = canonical(receipt, exclude="receipt_sha256")
    receipt_file_sha = hashlib.sha256(json_bytes(receipt)).hexdigest()

    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "CHECK_ONLY_PASS",
                    "manifest_file_sha256": paired_file_sha,
                    "manifest_receipt_sha256": paired["receipt_sha256"],
                    "receipt_file_sha256": receipt_file_sha,
                    "receipt_sha256": receipt["receipt_sha256"],
                    "raw_wrfout_receipt_sha256": paired["raw_wrfout_receipt_sha256"],
                    "counts": counts,
                },
                sort_keys=True,
            )
        )
        return 0

    if PAIRED_RUN.exists() and not PAIRED_RUN.is_dir():
        raise RuntimeError(f"paired target is not a directory: {PAIRED_RUN}")
    observed_paired_file_sha = write_new_or_verify(PAIRED_MANIFEST, paired)
    observed_receipt_file_sha = write_new_or_verify(PAIRED_RECEIPT, receipt)
    if observed_paired_file_sha != paired_file_sha or observed_receipt_file_sha != receipt_file_sha:
        raise RuntimeError("external paired artifact write verification failed")

    scores = gate["score"]["decision"]["strict_rmse"]
    correlations = gate["score"]["d03_spatial_pearson_correlation"]
    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.h5-terminal-closeout.v2",
        "verdict": "GPT_V10_ROOT_CAUSE_FIX_GREEN",
        "candidate": {
            "gpuwrf_version": GPUWRF_VERSION,
            "model_commit": CANDIDATE_COMMIT,
            "src_gpuwrf_tree": CANDIDATE_TREE,
            "runner_commit": git("rev-parse", f"{RUNNER_COMMIT}^{{commit}}"),
            "runner_sha256": sha256(ROOT / "scripts/v0234_h5_terminal_step9000.py"),
        },
        "terminal_acceptance": {
            "complete": True,
            "d03_step": 9000,
            "valid_time": "2025-03-01T15:00:00Z",
            "score": {name: float(scores[name]) for name in sorted(scores)},
            "ceilings": GATES,
            "correlations": {
                name: float(correlations[name]) for name in sorted(correlations)
            },
            "watch_items": gate["score"]["terminal_watch_items"],
            "all_fields_finite": True,
            "static_identity": True,
            "no_post_gate_dispatch_before_score": True,
            "gate_proof": gate_row,
        },
        "trajectory_continuity": {
            "step3800_frame_sha256": EXPECTED_STEP3800_SHA256,
            "byte_exact_to_independent_accepted_H5_step3800_arm": True,
        },
        "paired_case_handoff": {
            "registered": True,
            "registered_as": "validated_partial",
            "provenance": "gpu_output_paired_cpu_validated_partial",
            "frame_counts": counts,
            "target_frame_counts": EXPECTED_COUNTS,
            "valid_end": "2025-03-01T17:00:00Z",
            "completion_pending": COMPLETION_PENDING,
            "manifest": {
                "path": str(PAIRED_MANIFEST),
                "file_sha256": paired_file_sha,
                "receipt_sha256": paired["receipt_sha256"],
                "raw_wrfout_receipt_sha256": paired["raw_wrfout_receipt_sha256"],
            },
            "receipt": {
                "path": str(PAIRED_RECEIPT),
                "file_sha256": receipt_file_sha,
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "cpu_oracle_reference": paired["cpu_oracle_reference"],
        },
        "full18h_completion": {
            "complete": False,
            "register_as_complete_paired_case": False,
            "register_as_validated_partial": True,
            "late_failure": {
                "domain": "d03",
                "field": "Ni",
                "level": 0,
                "step": 10400,
                "simulated_time": "2025-03-01T17:20:00Z",
                "first_index": [0, 1, 1],
                "classification": "SEPARATE_POST_ACCEPTANCE_LATE_NI_STABILITY_BLOCKER",
                "causal_relationship_to_H5_proven": False,
                "blocker_proof": blocker_row,
            },
        },
        "execution": {
            "authorization": gate["authorization"],
            "one_candidate_process": True,
            "baseline_processes": 0,
            "gpu_queries_inside_runner": 0,
            "locked_execution": True,
            "lock_released_rc": 4,
            "launcher_log": {
                "path": str(LAUNCHER_LOG.resolve()),
                "bytes": LAUNCHER_LOG.stat().st_size,
                "sha256": sha256(LAUNCHER_LOG),
            },
            "wall_through_gate_seconds": gate["timing"]["wall_through_gate_seconds"],
            "wall_until_late_failure_seconds": blocker["wall_seconds"],
        },
        "case_metadata": {"path": str(METADATA.resolve()), "sha256": sha256(METADATA)},
        "scope": {
            "v10_acceptance_closed": True,
            "validated_partial_case_registered": True,
            "full18h_completion_deferred_to_separate_late_Ni_sprint": True,
            "no_tolerance_change": True,
            "no_gate_move": True,
            "no_step10800_rescore": True,
            "no_new_gpu_action_after_failure": True,
        },
    }
    proof["proof_sha256"] = canonical(proof, exclude="proof_sha256")
    closeout_file_sha = write_new_or_verify(OUTPUT, proof)
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "output": str(OUTPUT.resolve()),
                "file_sha256": closeout_file_sha,
                "proof_sha256": proof["proof_sha256"],
                "paired_manifest": str(PAIRED_MANIFEST),
                "paired_manifest_file_sha256": paired_file_sha,
                "paired_receipt_file_sha256": receipt_file_sha,
                "counts": counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
