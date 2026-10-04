"""Seal the terminal proof for the one released v0234 dycore GPU arm."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-gpu-arm"
PROOF = SPRINT / "proof.json"
CONTRACT = SPRINT / "CONTRACT.md"
ANALYSIS = SPRINT / "gpu-cpu-ladder-analysis.json"
REPORT = SPRINT / "worker-report.md"
COMMAND_LOG = SPRINT / "command-log.txt"
MERGER = REPO / "scripts/v0234_dycore_suboperator_gpt_gpu_merge.py"
MERGER_TEST = REPO / "tests/test_v0234_dycore_suboperator_gpt_gpu_merge.py"
AUDIT = (
    REPO
    / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/"
    "dycore-suboperator-runner-cpu-proof.json"
)
LAUNCHER = (
    REPO
    / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/"
    "dycore-suboperator-ladder-exact-launch-command.sh"
)
LOCK_WRAPPER = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2/scripts/with_gpu_lock.sh"
)
LINEAGE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
RUN_DIR = LINEAGE / "nested_stage_omega_transport_470e6111_dycore_suboperator_ladder1"
RUNTIME = RUN_DIR / "dycore-suboperator-ladder-terminal-proof.json"
RUN_PIN = RUN_DIR / "autotune-results.pb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload: dict[str, Any], *, omit: str = "proof_sha256") -> str:
    value = {key: item for key, item in payload.items() if key != omit}
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_self_hashed(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    actual = canonical_digest(payload)
    if payload.get("proof_sha256") != actual:
        raise RuntimeError(f"invalid self hash: {path}")
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "canonical_self_hash": actual,
        "bytes": path.stat().st_size,
    }


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True,
    ).strip()


def tracked_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    analysis, analysis_row = read_self_hashed(ANALYSIS)
    audit, audit_row = read_self_hashed(AUDIT)
    runtime, runtime_row = read_self_hashed(RUNTIME)
    if analysis["verdict"] != "DECISIVE_SP3_TO_L1_TENDENCY_BUILD_SYSTEMATIC":
        raise RuntimeError("analysis is not terminal decisive")
    if runtime["verdict"] != "DYCORE_SUBOPERATOR_LADDER_SHORT_ARM_COMPLETE":
        raise RuntimeError("runtime arm is not complete")
    if audit["verdict"] != "READY_FOR_DYCORE_SUBOPERATOR_LADDER_SHORT_GPU_ARM":
        raise RuntimeError("CPU runner audit is not ready")
    if Path("/tmp/wrf_gpu2_gpu.lock.holder").exists():
        raise RuntimeError("GPU holder sidecar remains after released arm")
    if git("rev-parse", "HEAD:src/gpuwrf") != git(
        "rev-parse", "718ff45498dab8b5d2f56bc6ededa753d4d19229:src/gpuwrf"
    ):
        raise RuntimeError("src/gpuwrf changed after the frozen instrumentation root")

    first_span = analysis["spans_evaluated_in_order"][
        "sp3_to_l1_tendency_build"
    ]
    sp3 = analysis["rungs_evaluated_in_order"]["sp3_tendf"]
    l1 = analysis["rungs_evaluated_in_order"]["l1_rk1_tend"]
    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.dycore-suboperator-gpt-gpu-arm.terminal.v1",
        "verdict": (
            "TERMINAL_SINGLE_GPU_ARM_PROOF_COMPLETE_"
            "DECISIVE_SP3_TO_L1_TENDENCY_BUILD"
        ),
        "objective_complete": True,
        "release_authority": {
            "authority": "chief 0:3 explicit RELEASE_GPU/ADMIT relayed by principal",
            "scope": "exactly one preregistered proof-only dycore ladder arm",
            "intent": "production-preemptible",
            "small_grid_at_admission": "CPU-only",
            "preempt_on_production_gpu_need": True,
            "no_standing_followup_authority": True,
        },
        "root_authority": {
            "terminal_cpu_commit": "453c9fa5d9461dece7ba50de722c1f35c7976931",
            "gpu_arm_contract_commit": "850385270921ffc5be2bf403a35c2ded0185871e",
            "contract": tracked_file(CONTRACT),
            "contract_committed_before_gpu_use": True,
            "original_kimi_instrumentation_commit": (
                "718ff45498dab8b5d2f56bc6ededa753d4d19229"
            ),
            "runtime_runner_head": runtime["authority"]["candidate"]["runner_head"],
            "runtime_model_tree": runtime["model_tree"],
            "src_gpuwrf_unchanged_after_718ff454": True,
            "candidate_head_before_terminal_commit": git("rev-parse", "HEAD"),
        },
        "execution": {
            "exact_launcher": tracked_file(LAUNCHER),
            "actual_launcher_entry": (
                "bash .agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/"
                "dycore-suboperator-ladder-exact-launch-command.sh"
            ),
            "launcher_exit_code": 0,
            "direct_exec_permission_failure": {
                "exit_code": 126,
                "script_mode": "0644",
                "script_entered": False,
                "lock_or_gpu_activity": False,
            },
            "canonical_lock_wrapper": tracked_file(LOCK_WRAPPER),
            "lock_label": "v0234-dycore-suboperator-ladder",
            "lock_timeout_seconds": 0,
            "lock_intent": "production-preemptible",
            "successful_lock_acquisitions": 1,
            "actual_model_processes": 1,
            "gpu_queries": 0,
            "gpu_retries": 0,
            "diagnostic_model_dispatches": runtime["window"]["model_dispatches"],
            "sampled_d03_steps": runtime["window"]["sampled_d03_first_last"],
            "root_steps": runtime["window"]["own_steps"]["d01"],
            "runtime_cpu_affinity": runtime["authority"]["cpu_affinity"],
            "runtime_live_lock": runtime["authority"]["live_lock"],
            "wrapper_release_observed": True,
            "holder_sidecar_absent_after_exit": True,
            "runtime_gpu_released_at_exit": runtime["gpu_released_at_exit"],
            "production_preemption_observed": False,
        },
        "runtime_gates": {
            "runtime_proof": runtime_row,
            "cpu_runner_audit": audit_row,
            "production_hlo_identity": runtime["production_hlo_identity"],
            "health_row_count": len(runtime["health_rows"]),
            "all_health_rows_passed": all(row["passed"] for row in runtime["health_rows"]),
            "complete_carry_nonfinite_count_sum": sum(
                int(row["complete_carry_nonfinite_count"])
                for row in runtime["health_rows"]
            ),
            "savepoint_count": runtime["savepoint_manifest"]["count"],
            "savepoint_rows_sha256": runtime["savepoint_manifest"]["rows_sha256"],
            "all_180_savepoint_file_hashes_revalidated": (
                analysis["authority"]["savepoint_manifest"]["all_file_hashes_valid"]
            ),
            "autotune_run_pin": tracked_file(RUN_PIN),
            "staged_pin_moved_into_namespace": (
                analysis["authority"]["autotune_run_pin"][
                    "staged_pin_absent_after_move"
                ]
            ),
        },
        "scientific_result": {
            "analysis": analysis_row,
            "step": 1,
            "incoming_sp3": {
                "rmse": sp3["classification"]["gpu_rmse"],
                "wrf_member_max_rmse": sp3["classification"][
                    "wrf_literal_envelope_max_rmse"
                ],
                "ratio": sp3["classification"]["gpu_to_wrf_member_max_ratio"],
                "systematic": sp3["classification"]["systematic"],
            },
            "outgoing_l1": {
                "rmse": l1["classification"]["gpu_rmse"],
                "wrf_member_max_rmse": l1["classification"][
                    "wrf_literal_envelope_max_rmse"
                ],
                "ratio": l1["classification"]["gpu_to_wrf_member_max_ratio"],
                "systematic": l1["classification"]["systematic"],
            },
            "earliest_decisive_source_bound_span": {
                "name": "sp3_to_l1_tendency_build",
                "rmse": first_span["classification"]["gpu_rmse"],
                "wrf_member_max_rmse": first_span["classification"][
                    "wrf_literal_envelope_max_rmse"
                ],
                "ratio": first_span["classification"][
                    "gpu_to_wrf_member_max_ratio"
                ],
                "systematic": first_span["classification"]["systematic"],
                "max_abs": first_span["gpu_vs_control_update_difference"]["max_abs"],
                "argmax": first_span["gpu_vs_control_update_difference"]["argmax"],
                "band_sse_shares": {
                    name: row["sse_share"]
                    for name, row in first_span[
                        "gpu_vs_control_update_difference"
                    ]["bands"].items()
                },
            },
            "analysis_stopped_at_first_decisive_span": True,
            "later_spans_analyzed": False,
            "internal_operator_cause_proven": False,
            "interpretation": analysis["terminal_decision"],
        },
        "scope_closure": {
            "model_correction": False,
            "v10_expansion": False,
            "ni_expansion": False,
            "eighteen_hour_expansion": False,
            "profiler_or_performance_claim": False,
            "second_gpu_arm": False,
            "additional_agent": False,
            "gpu_followup_authorized": False,
            "scientific_plan_preserved": True,
        },
        "committed_evidence": {
            "worker_report": tracked_file(REPORT),
            "command_log": tracked_file(COMMAND_LOG),
            "merger": tracked_file(MERGER),
            "merger_test": tracked_file(MERGER_TEST),
            "final_relevant_cpu_tests": {
                "command": (
                    "python -m pytest -q "
                    "tests/test_v0234_dycore_suboperator_gpt_gpu_merge.py "
                    "tests/test_v0234_dycore_suboperator_kimi.py "
                    "tests/test_v0234_first_interval_momentum.py"
                ),
                "passed": 25,
                "failed": 0,
            },
        },
        "handoff": {
            "unresolved_risks": [
                "SP3 is already systematic before the decisive span",
                "the decisive span does not split relax_bdy_dry, rk_addtend_dry, and spec_bdy_dry",
                "the short proof is not a correction, release, performance, Ni, V10, or 18-hour result",
            ],
            "next_decision": (
                "manager may contract a new internal SP3-to-L1 split; this proof "
                "authorizes neither a follow-up arm nor a correction"
            ),
        },
    }
    payload["proof_sha256"] = canonical_digest(payload)
    PROOF.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "wrote": str(PROOF),
        "verdict": payload["verdict"],
        "proof_sha256": payload["proof_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
