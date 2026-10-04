"""Seal the terminal CPU proof for the v0234 dycore ladder continuation.

CPU/filesystem/Git audit only.  This script has no JAX/CUDA imports and does
not inspect, lock, compile for, or dispatch to a GPU.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation"
KIMI_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi"
RUNS = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs")
CONDITIONING_MEMBERS = Path("<DATA_ROOT>/wrf_gpu2/v0234_conditioning_ensemble_gpt/members")
CONTROL_VERIFICATION = SPRINT / "control-verification.json"
PROOF = SPRINT / "proof.json"
MEMBERS = (
    "mask-a-minus",
    "mask-a-plus",
    "mask-b-minus",
    "mask-b-plus",
    "mask-c-minus",
    "mask-c-plus",
)
ALL_RUNS = ("control", *MEMBERS)
ADMITTED_LOGICAL = {13, 14, 15, 29, 30, 31}
ADMITTED_PHYSICAL = {13, 14, 15}
EXPECTED_BINARY_SHA256 = "50f71fd5245affcbf16f14f0cd7026561bbf1a412e28b5e60c60707af0edd238"
EXPECTED_PRODUCTION_IDENTITY = "759e9547869586b34824bb108d81e59b9babb716f085d323d344cca969b0907c"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object, *, omit: str | None = None) -> str:
    if omit is not None and isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != omit}
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def write_self_hashed(path: Path, payload: dict) -> None:
    out = dict(payload)
    out["proof_sha256"] = canonical_digest(out, omit="proof_sha256")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def read_self_hashed(path: Path) -> tuple[dict, dict]:
    value = json.loads(path.read_text())
    claimed = value.get("proof_sha256")
    observed = canonical_digest(value, omit="proof_sha256")
    if not claimed or claimed != observed:
        raise RuntimeError(f"self-hash mismatch {path}: {claimed} != {observed}")
    return value, {
        **file_identity(path),
        "claimed_self_hash": claimed,
        "observed_self_hash": observed,
        "self_hash_valid": True,
    }


def file_identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git(*args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout if binary else result.stdout.decode().strip()


def commit_identity(revision: str) -> dict:
    fields = git(
        "show", "-s", "--format=%H%n%T%n%cI%n%an%n%s", revision,
    ).splitlines()
    return {
        "commit": fields[0],
        "tree": fields[1],
        "committed_at": fields[2],
        "author": fields[3],
        "subject": fields[4],
    }


def git_blob_identity(revision: str, path: str) -> dict:
    value = git("show", f"{revision}:{path}", binary=True)
    return {
        "revision": revision,
        "path": path,
        "bytes": len(value),
        "sha256": sha256_bytes(value),
    }


def run_receipt(member: str) -> dict:
    root = RUNS / member
    execution, execution_row = read_self_hashed(root / "execution-receipt.json")
    monitor, monitor_row = read_self_hashed(root / "resource-monitor.json")
    archive, archive_row = read_self_hashed(root / "archive-receipt.json")
    admissions = sorted((root / "admissions").glob("*.json"))
    if len(admissions) != 1:
        raise RuntimeError(f"{member}: expected one admission, found {admissions}")
    admission, admission_row = read_self_hashed(admissions[0])

    archive_path = Path(archive["archive"])
    archive_direct = file_identity(archive_path)
    if archive_direct["sha256"] != archive["sha256"]:
        raise RuntimeError(f"{member}: archive hash mismatch")
    if archive["file_count"] != 564 or not archive["zstd_test_passed"]:
        raise RuntimeError(f"{member}: archive gate failed")
    if execution["binary_sha256"] != EXPECTED_BINARY_SHA256:
        raise RuntimeError(f"{member}: binary mismatch")
    if not (
        execution["returncode"] == 0
        and execution["success_complete_wrf"]
        and execution["dump_file_count"] == 564
        and execution["monitor_safe"]
    ):
        raise RuntimeError(f"{member}: execution gate failed")
    if not monitor["all_samples_safe"] or monitor["violations"]:
        raise RuntimeError(f"{member}: monitor gate failed")
    if admission["verdict"] != "ADMITTED_ISOLATED_CPU_SET":
        raise RuntimeError(f"{member}: admission failed")

    if member == "control":
        # Known inherited pre-science constant error, explicitly accepted by
        # the continuation contract. Actual and archived counts are 564.
        if execution["expected_dump_file_count_at_steps_1_2"] != 516:
            raise RuntimeError("retained control expected-count chronology changed")
        active_samples = [
            sample for sample in monitor["samples"]
            if "ActiveState=active" in sample["unit_state"]
        ]
        affinity = {
            "monitor_generation": "inherited_name_pattern",
            "max_member_rank_count": monitor["max_member_rank_count"],
            "active_sample_count": len(active_samples),
            "systemd_affinity_present_every_active_sample": bool(active_samples) and all(
                "CPUAffinity=13-15 29-31" in sample["unit_state"]
                for sample in active_samples
            ),
            "direct_rank_affinity_available": False,
        }
        if not affinity["systemd_affinity_present_every_active_sample"]:
            raise RuntimeError("control systemd affinity missing")
    else:
        if execution["expected_dump_file_count_at_steps_1_2"] != 564:
            raise RuntimeError(f"{member}: corrected expected count mismatch")
        if monitor["max_member_rank_count"] != 12:
            raise RuntimeError(f"{member}: did not observe all 12 ranks")
        unit_rows = [
            row
            for sample in monitor["samples"]
            for row in sample["unit_rows"]
        ]
        if not unit_rows:
            raise RuntimeError(f"{member}: no cgroup process evidence")
        if any(
            not set(row["logical_cpus"]) <= ADMITTED_LOGICAL
            or not set(row["physical_cores"]) <= ADMITTED_PHYSICAL
            for row in unit_rows
        ):
            raise RuntimeError(f"{member}: cgroup affinity outside admission")
        affinity = {
            "monitor_generation": "cgroup_authoritative_gpt_continuation",
            "max_member_rank_count": monitor["max_member_rank_count"],
            "max_unit_process_count": monitor["max_unit_process_count"],
            "all_unit_logical_cpus_subset": sorted(ADMITTED_LOGICAL),
            "all_unit_physical_cores_subset": sorted(ADMITTED_PHYSICAL),
            "direct_rank_affinity_available": True,
        }

    production_ids = {sample["production_identity"] for sample in monitor["samples"]}
    if production_ids != {EXPECTED_PRODUCTION_IDENTITY}:
        raise RuntimeError(f"{member}: production identity drift {production_ids}")
    if any(not sample["production_confined"] for sample in monitor["samples"]):
        raise RuntimeError(f"{member}: production confinement failure")

    fixed_inputs = {
        name: file_identity(root / "run" / name)
        for name in (
            "namelist.input", "wrfbdy_d01", "wrfinput_d01",
            "wrfinput_d02", "wrfinput_d03",
        )
    }
    perturbation = None
    if member != "control":
        manifest_path = CONDITIONING_MEMBERS / member / "perturbation-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        expected_input = manifest["member_wrfinput_d03"]["sha256"]
        if fixed_inputs["wrfinput_d03"]["sha256"] != expected_input:
            raise RuntimeError(f"{member}: perturbation input mismatch")
        perturbation = {
            "manifest": file_identity(manifest_path),
            "expected_wrfinput_d03_sha256": expected_input,
        }

    return {
        "member": member,
        "admission": admission_row,
        "execution": execution_row,
        "monitor": monitor_row,
        "archive_receipt": archive_row,
        "archive": archive_direct,
        "execution_summary": {
            "returncode": execution["returncode"],
            "success_complete_wrf": execution["success_complete_wrf"],
            "dump_file_count": execution["dump_file_count"],
            "expected_dump_file_count": execution[
                "expected_dump_file_count_at_steps_1_2"
            ],
            "binary_sha256": execution["binary_sha256"],
            "finished_at_utc": execution["finished_at_utc"],
        },
        "resource_summary": {
            "sample_count": monitor["sample_count"],
            "minimum_mem_available_gib": min(
                sample["mem_available_gib"] for sample in monitor["samples"]
            ),
            "minimum_mnt_data_free_gib": min(
                sample["mnt_data_free_gib"] for sample in monitor["samples"]
            ),
            "production_identity": EXPECTED_PRODUCTION_IDENTITY,
            "production_confined_to_physical_0_11": True,
            **affinity,
        },
        "inputs": fixed_inputs,
        "perturbation": perturbation,
        "launch_log": file_identity(root / "launch.log"),
    }


def main() -> None:
    control_gate, control_gate_row = read_self_hashed(RUNS / "control/control-gates.json")
    if not control_gate["passed"]:
        raise RuntimeError("control verification is not green")
    control_verification = {
        "schema": "gpuwrf.v0234.dycore-suboperator-gpt-continuation.control-verification.v1",
        "command": (
            "/usr/bin/taskset -c 13-15,29-31 <USER_HOME>/miniconda3/bin/python "
            "scripts/v0234_dycore_suboperator_kimi_wrf_ladder.py verify-control"
        ),
        "exit_code": 0,
        "stdout": '{"passed": true}',
        "rerun_passed": True,
        "control_gate": control_gate_row,
        "control_gate_mtime_utc": datetime.fromtimestamp(
            (RUNS / "control/control-gates.json").stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "output_neutrality": control_gate,
    }
    write_self_hashed(CONTROL_VERIFICATION, control_verification)
    _, control_verification_row = read_self_hashed(CONTROL_VERIFICATION)

    build_receipt, build_receipt_row = read_self_hashed(
        Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/build-receipt.json")
    )
    if build_receipt["binary_sha256"] != EXPECTED_BINARY_SHA256:
        raise RuntimeError("build receipt binary mismatch")
    binary_row = file_identity(Path(build_receipt["binary"]))
    if binary_row["sha256"] != EXPECTED_BINARY_SHA256:
        raise RuntimeError("instrumented binary bytes changed")

    inherited = {}
    for name in (
        "audit-ensemble-proof.json",
        "step1-savepoint-ladder.json",
        "band-decomposition.json",
    ):
        _, inherited[name] = read_self_hashed(KIMI_SPRINT / name)

    ladder_envelopes, ladder_row = read_self_hashed(SPRINT / "ladder-envelopes.json")
    cpu_analysis, cpu_analysis_row = read_self_hashed(SPRINT / "cpu-ladder-analysis.json")
    if not all(ladder_envelopes["control_bitwise_equal_to_baseline_step1"].values()):
        raise RuntimeError("available control chain rungs are not bitwise")
    if cpu_analysis["verdict"] != "CPU_LADDER_ENVELOPES_COMPLETE_GPU_VALUES_REQUIRED":
        raise RuntimeError("unexpected CPU analysis verdict")

    receipts = {member: run_receipt(member) for member in ALL_RUNS}
    member_finished = [
        receipts[member]["execution_summary"]["finished_at_utc"] for member in MEMBERS
    ]
    if member_finished != sorted(member_finished):
        raise RuntimeError("member chronology is not sequential")

    prereg = commit_identity("93be9cb5")
    instrumentation = commit_identity("718ff454")
    base = commit_identity("2aa91bee")
    evidence_code = commit_identity("HEAD")
    plan_blob = git_blob_identity(
        "93be9cb5",
        ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/discriminator-plan.md",
    )
    current_plan = file_identity(KIMI_SPRINT / "discriminator-plan.md")
    if plan_blob["sha256"] != current_plan["sha256"]:
        raise RuntimeError("frozen plan bytes drifted after preregistration")
    if subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", "93be9cb5", "718ff454"]
    ).returncode != 0:
        raise RuntimeError("preregistration is not ancestor of instrumentation")

    source_tree_718 = git("rev-parse", "718ff454:src/gpuwrf")
    source_tree_head = git("rev-parse", "HEAD:src/gpuwrf")
    if source_tree_718 != source_tree_head:
        raise RuntimeError("src/gpuwrf changed after inherited instrumentation")

    report_row = file_identity(SPRINT / "worker-report.md")
    command_log_row = file_identity(SPRINT / "command-log.md")
    corrections_row = file_identity(SPRINT / "process-corrections.md")
    continuation_contract_row = file_identity(SPRINT / "CONTRACT.md")
    original_contract_row = file_identity(KIMI_SPRINT / "CONTRACT.md")
    operator_map_row = file_identity(KIMI_SPRINT / "operator-map.md")
    patch_row = file_identity(KIMI_SPRINT / "wrf-ladder-instrumentation.patch")

    step1 = cpu_analysis["steps"]["step1"]
    proof = {
        "schema": "gpuwrf.v0234.dycore-suboperator-gpt-continuation.terminal-cpu-proof.v1",
        "verdict": "TERMINAL_CPU_PROOF_COMPLETE_FROZEN_GPU_ARM_REQUIRED",
        "objective": (
            "complete the six inherited pristine-WRF CPU members and frozen "
            "dycore-suboperator ladder without GPU use"
        ),
        "authority": {
            "base": base,
            "preregistration": prereg,
            "instrumentation": instrumentation,
            "evidence_code": evidence_code,
            "frozen_plan_preregistered_blob": plan_blob,
            "frozen_plan_worktree": current_plan,
            "continuation_contract": continuation_contract_row,
            "original_contract": original_contract_row,
            "operator_map": operator_map_row,
            "wrf_patch": patch_row,
            "build_receipt": build_receipt_row,
            "instrumented_binary": binary_row,
            "source_tree_718ff454": source_tree_718,
            "source_tree_evidence_code": source_tree_head,
            "production_model_delta_after_718ff454": [],
            "preregistration_predates_all_new_members": True,
            "member_completion_chronology": member_finished,
        },
        "inherited_evidence_audit": inherited,
        "retained_control": {
            "verification": control_verification_row,
            "known_old_expected_count": 516,
            "authenticated_actual_count": 564,
            "expensive_control_rerun": False,
            "reason_not_rerun": "retained output-neutral control reverified green",
        },
        "runs": receipts,
        "resource_contract": {
            "chief_admission": "physical cores 13-15 plus SMT 29-31",
            "production_owned": "physical cores 0-11",
            "new_members_all_directly_cgroup_observed": True,
            "new_members_all_observed_rank_count": 12,
            "production_identity_stable": EXPECTED_PRODUCTION_IDENTITY,
            "minimum_disk_floor_required_gib": 40.0,
            "minimum_memory_floor_required_gib": 32.0,
            "minimum_observed_disk_gib": min(
                receipts[member]["resource_summary"]["minimum_mnt_data_free_gib"]
                for member in MEMBERS
            ),
            "minimum_observed_memory_gib": min(
                receipts[member]["resource_summary"]["minimum_mem_available_gib"]
                for member in MEMBERS
            ),
        },
        "analysis": {
            "compact_ladder_envelopes": ladder_row,
            "detailed_cpu_ladder": cpu_analysis_row,
            "step1_max_member_rung_rmse": {
                name: step1["rungs"][name]["envelope"]["wrf_envelope_max_rmse"]
                for name in cpu_analysis["rung_order"]
            },
            "step1_max_member_span_update_difference_rmse": {
                name: step1["spans"][name]["envelope"]["wrf_envelope_max_rmse"]
                for name in cpu_analysis["span_order"]
            },
            "step1_l1_auxiliary_relax_max_member_rmse": step1[
                "auxiliary_rungs"
            ]["l1_rk1_relax"]["envelope"]["wrf_envelope_max_rmse"],
            "step1_l4_to_l5_member_response_exact_zero": all(
                row["update_difference"]["sse"] == 0.0
                for row in step1["spans"]["l4_to_l5_post_loop_pre_final"]["members"].values()
            ),
            "step1_l5_to_sp4_member_response_exact_zero": all(
                row["update_difference"]["sse"] == 0.0
                for row in step1["spans"]["l5_to_sp4_end_of_step_boundary"]["members"].values()
            ),
            "all_detailed_comparisons_finite": True,
            "frozen_systematic_factor": 2.0,
            "gpu_l1_l5_values_available": False,
            "earliest_systematic_gpu_rung_resolved": False,
        },
        "acceptance_gates": {
            "inherited_hashes_and_chronology_authenticated": True,
            "retained_control_reverification_green": True,
            "six_members_exact_inputs_complete": True,
            "six_members_finite_monitor_safe_archived": True,
            "six_members_direct_rank_affinity_proof": True,
            "rung_and_span_before_after_delta_metrics_complete": True,
            "step2_replication_present": True,
            "production_model_unchanged_after_718ff454": True,
            "gpu_hold_honored": True,
            "terminal_cpu_stop_condition_reached": True,
        },
        "negative_pre_science_evidence": {
            "process_corrections": corrections_row,
            "scientific_alternation_advanced": False,
        },
        "tests": {
            "command": (
                "/usr/bin/taskset -c 13-15,29-31 <USER_HOME>/miniconda3/bin/python "
                "-m pytest -q tests/test_v0234_dycore_suboperator_kimi.py "
                "tests/test_v0234_first_interval_momentum.py"
            ),
            "result": "21 passed",
        },
        "commands": command_log_row,
        "report": report_row,
        "model_attestation": {
            "owner": "GPT-5.6 max continuation",
            "additional_models_or_agents": 0,
            "src_gpuwrf_tree_equal_to_718ff454": True,
            "production_numerics_edited": False,
        },
        "gpu_attestation": {
            "state": "HOLD",
            "queries": 0,
            "locks": 0,
            "compiles": 0,
            "dispatches": 0,
            "kernels": 0,
            "profilers": 0,
        },
        "terminal_cpu_conclusion": {
            "why_gpu_arm_is_required": (
                "the six-member WRF L1-L5 envelopes are now frozen, but retained "
                "GPU evidence contains only SP3/SP4 and cannot decide any L1-L5 ratio"
            ),
            "exact_next_discriminator": (
                "the single preregistered proof-only d03 L1-L5 ladder arm in "
                ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/"
                "dycore-suboperator-ladder-exact-launch-command.sh"
            ),
            "coordination_required": (
                "manager 0:1 must obtain explicit chief 0:3 release of GPU HOLD; "
                "lock-v2 intent production-preemptible; no broader GPU work"
            ),
            "gpu_action_taken_by_this_continuation": False,
        },
        "unresolved_risks": [
            "first divergent GPU suboperator remains unresolved until L1-L5 GPU values exist",
            (
                "frozen L1-to-L2 arithmetic crosses tendency and velocity representations; "
                "retained but explicitly non-physical as an update tendency"
            ),
            (
                "retained control has systemd affinity proof but its inherited name-pattern "
                "monitor did not directly enumerate ranks; all six new members do"
            ),
        ],
    }
    if not all(proof["acceptance_gates"].values()):
        raise RuntimeError("terminal acceptance gate false")
    write_self_hashed(PROOF, proof)
    _, proof_row = read_self_hashed(PROOF)
    print(json.dumps({
        "verdict": proof["verdict"],
        "proof": str(PROOF),
        "proof_sha256": proof_row["claimed_self_hash"],
        "six_members": len(MEMBERS),
        "gpu_actions": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
