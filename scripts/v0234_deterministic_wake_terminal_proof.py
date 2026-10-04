#!/usr/bin/env python3
"""Build the canonical terminal proof for the v0234 deterministic wake sprint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from scripts import v0234_deterministic_wake_admission as admission


MODEL_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
KIMI_COMMIT = "ceec296946b40185d21a3c1958830f1530711d11"
LOCK_COMMIT = "8152309aff1e85e1052d44d549a5a5409e710bdd"
LOCK_WRAPPER_SHA256 = (
    "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
)
LOCK_VERIFIER_SHA256 = (
    "ec911f565ee9d2c58dee81d8d1e17411ed677be500a57e73ded5f793e2c4187d"
)
SPRINT = admission.SPRINT
RUN_DIR = admission.LINEAGE_ROOT / (
    "nested_stage_omega_transport_470e6111_deterministic_wake_reference1"
)
STAGED_PIN = admission.LINEAGE_ROOT / (
    ".v0234-deterministic-wake-autotune-v1/reference-autotune-results.pb"
)
REFERENCE_LOG = admission.LINEAGE_ROOT / (
    "v0234-deterministic-wake-reference-launch-v2.log"
)
PINNED_RUN_DIR = admission.LINEAGE_ROOT / (
    "nested_stage_omega_transport_470e6111_deterministic_wake_pinned1"
)
FIXED_FILES = {
    RUN_DIR / "full-run-blocker.json": (
        "728f2f19a9ceca8697c2c7f8a0fc267cf61431e761f214948f5869713224b219"
    ),
    RUN_DIR / "failure/failure-proof.json": (
        "2a4b1422087c437f7db42668e5a9d2c93564d58c208b0cc326d3880271f30a86"
    ),
    RUN_DIR / "frame-pairs/d03-step-09000.json": (
        "86306e3dc629c1e1d945b16fc12c5020067b52ce439a59e75af8083230ef4296"
    ),
    RUN_DIR / "gpu-output/wrfout_d03_2025-03-01_15:00:00": (
        "5402f5b578b49c295737f8b2fc02ebaa810f4f0ba9e7fd49c0fbe07b87692898"
    ),
    RUN_DIR / "checkpoints/authenticated-d03-step-8800.pkl": (
        "957c346ee7e14557b97ebeae7907be74a265eebb61b1a02fba95cb3f4c5493ea"
    ),
    RUN_DIR / "checkpoints/authenticated-d03-step-9000.pkl": (
        "abc2908f062de52afaf3c04027c1fc9515a86f4c9790c0b53a5313db5da2ce76"
    ),
    RUN_DIR / "ordinary-one-step-lowered-hlo.json": (
        "15575931876ca3126e91e5b1c7cbcd85360e7cc2754354f262c5c82556d3a308"
    ),
    STAGED_PIN: (
        "edd3b1271dbc59d2998cfc25a062c54ed23cce54e4bd740bbafe625f9bcb222f"
    ),
    REFERENCE_LOG: (
        "830651b32a4a57bb6366a4ff82ca7c0775c72695da52795f00d1411c60a11fd3"
    ),
}
SELF_HASHED = {
    SPRINT / "reference-harness-first-red.json": (
        "cc4fba80375175950d5313728b5c53a6c29cf153ee06b5e8f8de0b21fd15d9d4",
        "e1737d59a74cd111952e06785f190538475c9521229cf0535f07ce446f88d147",
    ),
    SPRINT / "reference-runner-cpu-proof-v2.json": (
        "4ad92d751f152c37f26e731938522f01bcc806e6d73950d3e558efeacb820de5",
        "2ac9a0b409a378e2918ecf17c7095cf63e10936ac7690dd625b6211f49587b5b",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def file_row(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/symlinked evidence: {path}")
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise RuntimeError(f"evidence hash changed: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": observed,
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def self_hashed_row(
    path: Path, *, expected_file: str | None = None, expected_canonical: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = file_row(path, expected_file)
    payload = read_json(path)
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    if embedded != observed or (
        expected_canonical is not None and observed != expected_canonical
    ):
        raise RuntimeError(f"canonical proof mismatch: {path}")
    row["canonical_sha256"] = observed
    return payload, row


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=admission.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def command_row(argv: tuple[str, ...]) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=admission.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"validation command failed: {payload}")
    return payload


def build(
    *, rca_path: Path, report_path: Path, command_log_path: Path,
    retained_output: Path, proof_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority, authority_row = admission.authenticate_authority()
    rca, rca_row = self_hashed_row(rca_path)
    if (
        rca.get("verdict")
        != "PROGNOSTIC_LOWEST_LEVEL_MOMENTUM_FIRST_INTERVAL_LOCALIZED__NO_MODEL_FIX_JUSTIFIED"
        or rca.get("model_or_numerical_edit") is not False
        or rca.get("release_gate_green") is not False
        or rca.get("retained_trajectory", {}).get("frame_pair_count") != 46
    ):
        raise RuntimeError("RCA terminal semantics changed")

    authenticated: dict[str, Any] = {}
    self_payloads: dict[str, dict[str, Any]] = {}
    for path, (file_hash, canonical_hash) in SELF_HASHED.items():
        payload, row = self_hashed_row(
            path, expected_file=file_hash, expected_canonical=canonical_hash
        )
        authenticated[path.name] = row
        self_payloads[path.name] = payload
    harness = self_payloads["reference-harness-first-red.json"]
    cpu_proof = self_payloads["reference-runner-cpu-proof-v2.json"]
    if (
        harness.get("classification")
        != "HARNESS_ONLY__DOES_NOT_ADVANCE_SCIENCE_ALTERNATION"
        or harness.get("gpu_action", {}).get("lock_released") is not True
        or harness.get("run_effects", {}).get("scientific_result_produced") is not False
        or cpu_proof.get("verdict") != "READY_FOR_DETERMINISTIC_WAKE_FULLTREE_GPU_REPLAY"
        or cpu_proof.get("gpu_commands_run") != 0
        or cpu_proof.get("gpu_queries_run") != 0
        or cpu_proof.get("focused_tests", {}).get("returncode") != 0
    ):
        raise RuntimeError("preflight/harness semantics changed")

    fixed_rows = {
        str(path.resolve()): file_row(path, expected)
        for path, expected in FIXED_FILES.items()
    }
    blocker, blocker_row = self_hashed_row(
        RUN_DIR / "full-run-blocker.json",
        expected_file=FIXED_FILES[RUN_DIR / "full-run-blocker.json"],
        expected_canonical=(
            "33b4e566c3947274701e70c44292a459da9eab54dd1948ba4899b40bb7333e91"
        ),
    )
    failure, failure_row = self_hashed_row(
        RUN_DIR / "failure/failure-proof.json",
        expected_file=FIXED_FILES[RUN_DIR / "failure/failure-proof.json"],
        expected_canonical=(
            "2a7919249c78d5b85f753bfabb310124936a9bda1ac5fe38071d036736826d85"
        ),
    )
    detail = json.loads(str((failure.get("failure") or {}).get("detail", "{}")))
    metric = detail.get("metric_signature") or {}
    if (
        blocker.get("verdict") != "FULL_18H_BLOCKED"
        or blocker.get("failure_code") != "INCREMENTAL_FRAME_PAIR"
        or detail.get("classification") != "NEW_SCIENTIFIC_RED_OR_CHANGED_WAKE_SIGNATURE"
        or metric.get("red_fields") != ["V", "V10"]
    ):
        raise RuntimeError("scientific first-red semantics changed")

    output_counts = {
        domain: len(list((RUN_DIR / "gpu-output").glob(f"wrfout_{domain}_*")))
        for domain in ("d01", "d02", "d03")
    }
    pair_counts = {
        domain: len(list((RUN_DIR / "frame-pairs").glob(f"{domain}-step-*.json")))
        for domain in ("d01", "d02", "d03")
    }
    if output_counts != {"d01": 15, "d02": 15, "d03": 46} or pair_counts != output_counts:
        raise RuntimeError(f"stopped-run inventory changed: {output_counts}, {pair_counts}")
    if PINNED_RUN_DIR.exists() or PINNED_RUN_DIR.is_symlink():
        raise RuntimeError("pinned arm unexpectedly exists")

    head = git("rev-parse", "HEAD")
    current_tree = git("rev-parse", "HEAD:src/gpuwrf")
    kimi_tree = git("rev-parse", f"{KIMI_COMMIT}:src/gpuwrf")
    model_delta = git("diff", "--name-only", KIMI_COMMIT, "HEAD", "--", "src/gpuwrf")
    if current_tree != MODEL_TREE or kimi_tree != MODEL_TREE or model_delta:
        raise RuntimeError("model bytes changed")

    validations = {
        "focused_tests": command_row(
            (
                "<USER_HOME>/miniconda3/bin/python",
                "-m",
                "pytest",
                "-q",
                "tests/test_v0234_deterministic_wake_closure.py",
            )
        ),
        "launcher_shell_syntax": command_row(
            (
                "bash",
                "-n",
                str(SPRINT / "deterministic-fulltree-exact-launch-command.sh"),
            )
        ),
    }

    retained: dict[str, Any] = {
        "schema": "gpuwrf.v0234.deterministic-wake-retained-evidence.v1",
        "verdict": "DETERMINISTIC_WAKE_FIRST_RED_EVIDENCE_RETAINED",
        "authority": {
            **authority_row,
            "authority_sha256": authority["authority_sha256"],
        },
        "preflight_and_harness": authenticated,
        "reference_run": {
            "run_dir": str(RUN_DIR.resolve()),
            "output_counts": output_counts,
            "frame_pair_counts": pair_counts,
            "blocker": blocker_row,
            "failure": failure_row,
            "fixed_artifacts": fixed_rows,
            "rca": rca_row,
            "rca_full_frame_inventory_sha256": rca["retained_trajectory"][
                "artifact_inventory_sha256"
            ],
            "all_frames_through_8800_green": True,
            "first_new_science_red_step": 9000,
            "complete_reference_process": False,
        },
        "deterministic_pin": {
            "path": str(STAGED_PIN.resolve()),
            "file_sha256": FIXED_FILES[STAGED_PIN],
            "bytes": STAGED_PIN.stat().st_size,
            "staged_not_promoted_because_reference_failed": True,
            "sufficient_for_encountered_compile_paths_only": True,
        },
        "pinned_arm": {
            "run_dir": str(PINNED_RUN_DIR.resolve()),
            "absent": True,
            "reason": "reference failed the frozen admission; pinned replay was not admissible",
        },
        "compiled_program": {
            "ordinary_lowered_hlo": fixed_rows[
                str((RUN_DIR / "ordinary-one-step-lowered-hlo.json").resolve())
            ],
            "ordinary_compile_seconds": 212.917786,
            "first_advance_compile_seconds": 152.155370578,
            "health_compile_seconds": 11.492147,
            "health_leaf_count": 106,
        },
        "lock_history": {
            "canonical_lock_commit": LOCK_COMMIT,
            "wrapper_sha256": LOCK_WRAPPER_SHA256,
            "verifier_sha256": LOCK_VERIFIER_SHA256,
            "intent": "production-preemptible",
            "label": "v0234-deterministic-fulltree-reference",
            "attempts": [
                {
                    "classification": "harness-only",
                    "returncode": 1,
                    "lock_acquired": True,
                    "lock_released": True,
                    "proof": authenticated["reference-harness-first-red.json"],
                },
                {
                    "classification": "first-new-science-red",
                    "returncode": 3,
                    "lock_acquired": True,
                    "lock_released": True,
                    "log": fixed_rows[str(REFERENCE_LOG.resolve())],
                },
            ],
            "pane_0_3_observation": (
                "explicitly recognized the lease as legitimate production-preemptible; "
                "PREEMPT would apply on production start; later observed lock free and holder removed"
            ),
            "hold_or_preempt_observed": False,
            "gpu_released": True,
        },
        "model_tree_before": kimi_tree,
        "model_tree_after": current_tree,
        "model_delta_empty": True,
    }
    retained["proof_sha256"] = canonical_digest(retained)
    atomic_write_json(retained_output, retained)
    retained_row = file_row(retained_output)
    retained_row["canonical_sha256"] = retained["proof_sha256"]

    strict_rmse = metric["strict_rmse"]
    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.deterministic-wake-terminal-proof.v1",
        "verdict": "GPT_DETERMINISTIC_WAKE_NO_FIX_LOCALIZED",
        "objective": (
            "Execute the deterministic full-tree reference/pin contract fail-closed, "
            "then localize the first new wake red without an unjustified model edit."
        ),
        "contract": file_row(admission.CONTRACT, admission.CONTRACT_SHA256),
        "starting_authority": {
            "kimi_commit": KIMI_COMMIT,
            "kimi_proof_file_sha256": admission.KIMI_PROOF_FILE_SHA256,
            "kimi_proof_canonical_sha256": admission.KIMI_PROOF_CANONICAL_SHA256,
            "kimi_retained_file_sha256": admission.KIMI_RETAINED_FILE_SHA256,
            "kimi_retained_canonical_sha256": admission.KIMI_RETAINED_CANONICAL_SHA256,
            "model_tree": MODEL_TREE,
        },
        "evidence_code_commit": head,
        "retained_evidence": retained_row,
        "wake_rca": rca_row,
        "reference_replay": {
            "status": "STOPPED_FIRST_NEW_SCIENCE_RED",
            "returncode": 3,
            "first_red_step": 9000,
            "last_green_step": 8800,
            "output_counts": output_counts,
            "finite": True,
            "static_exact": True,
            "red_fields": ["V", "V10"],
            "strict_rmse_all_required_fields": strict_rmse,
            "V_inside_authenticated_envelope": True,
            "V10_inside_authenticated_range": False,
            "V10_excess_above_range": rca["first_red"]["V10_excess_above_range"],
            "spatial_fingerprint_passed": True,
            "same_retained_wake_mechanism": True,
            "release_gate_green": False,
            "tolerance_changed": False,
            "waiver_or_reclassification": False,
        },
        "deterministic_equality": {
            "status": "NOT_RUN__REFERENCE_FAILED_FROZEN_ADMISSION",
            "reference_complete": False,
            "pin_promoted": False,
            "pinned_arm_run": False,
            "claim": "none",
        },
        "late_ni": {
            "status": "NOT_REACHED__STOPPED_AT_STEP9000",
            "steps_9313_9314_9405_dispatched": False,
            "claim": "none",
        },
        "wake_closure": {
            "status": "NO_FIX__EARLIEST_CAUSAL_INTERVAL_LOCALIZED",
            "bracket": rca["causal_localization"]["bracket"],
            "v10_is_downstream_of_lowest_level_v": True,
            "surface_ratio_is_not_dominant": True,
            "remaining_split": rca["causal_localization"]["exact_remaining_split"],
            "model_fix_justified": False,
        },
        "model_bytes": {
            "before": kimi_tree,
            "after": current_tree,
            "changed": False,
            "diff": [],
        },
        "gpu_and_lock": retained["lock_history"],
        "validation": validations,
        "report": file_row(report_path),
        "command_log": file_row(command_log_path),
        "unresolved_risks": [
            "No deterministic equality claim: the frozen reference admission failed before pin promotion.",
            "Late-Ni 9313/9314/9405 and terminal output gates were not reached in this lineage.",
            "The exact first-interval momentum operator remains split between MYNN PBL and dry-dycore/nest coupling.",
            "The staged pin covers encountered compile paths only and is not a complete-reference pin.",
        ],
        "exact_next_kimi_action": rca["exact_next_kimi_action"],
        "release_authorized": False,
        "push_performed": False,
        "gpu_released": True,
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(proof_output, proof)
    return retained, proof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rca", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--command-log", type=Path, required=True)
    parser.add_argument("--retained-output", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    args = parser.parse_args()
    retained, proof = build(
        rca_path=args.rca.resolve(),
        report_path=args.report.resolve(),
        command_log_path=args.command_log.resolve(),
        retained_output=args.retained_output.resolve(),
        proof_output=args.proof_output.resolve(),
    )
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof_file_sha256": sha256_file(args.proof_output.resolve()),
                "proof_canonical_sha256": proof["proof_sha256"],
                "retained_file_sha256": sha256_file(args.retained_output.resolve()),
                "retained_canonical_sha256": retained["proof_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
