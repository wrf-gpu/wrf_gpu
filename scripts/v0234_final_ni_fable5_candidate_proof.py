"""Aggregate CPU candidate proof for the v0.23.4 PH base-operand correction.

Requires (and authenticates) the sprint's component proofs:

- authority-proof.json                (frozen inputs green)
- carry-localization-proof.json      (interior ph freeze confirmed from carries)
- source-invariant-proof.json        (pristine WRF discrepancy proven)
- interface-transfer-proof.json      (candidate OFF e0d0-exact; ON clean HLO)
- contract-amendment-01/02.json      (Fable + GPT-MAX ownership)
- CPU A/B arm proofs                 (falsified reproduces; corrected passes)

Runs focused and proportional acoustic suites, compile/diff/static gates, and
the correction source audit (no callback or loop-transfer tokens in the
correction diff), then emits candidate-cpu-proof.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "gpuwrf.v0234.final-ni-fable5-candidate-cpu.v1"
REPO = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
CPUAB_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_final_ni_fable5_cpuab"
)
CORRECTION_PARENT = "60659a2e02e33153d87256ee78bf1380f5eabe76"
CORRECTED_CPU_COMMIT = "8fde9c941ba31f292d9921455dde9ea6e8387fc5"
GPU_EXECUTION_COMMIT = "66a5fbde1a1591cfadbcce74e481bbbc28f5057b"
ARM_DIRS = {"falsified": "falsified-stop200", "corrected": "corrected"}
MODEL_FILES = (
    "src/gpuwrf/coupling/boundary_apply.py",
    "src/gpuwrf/dynamics/core/acoustic.py",
)
FORBIDDEN_SOURCE = (
    "jax.device_get",
    "jax.pure_callback",
    "io_callback",
    "host_callback",
    "debug.callback",
    "np.asarray(",
    "numpy.asarray(",
)
FOCUSED_TEST_FILES = (
    "tests/test_v0234_1500_scientific_rca_candidate.py",
    "tests/test_v0234_final_ni_fable5_ph_base.py",
    "tests/test_v0234_1500_scientific_rca_source_oracle.py",
    "tests/test_v0234_1500_scientific_rca_trajectory.py",
    "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py",
    "tests/test_v0234_nested_boundary_critic_repair.py",
    "tests/test_v014_specified_bdy_cadence.py",
    "tests/test_m6_boundary_apply.py",
    "tests/test_v014_pre_halo_capture.py",
)
PROPORTIONAL_TEST_FILES = (
    "tests/test_m4_acoustic.py",
    "tests/test_m6b4_acoustic_recurrence_parity.py",
    "tests/test_m6b_dycore_rk_acoustic_fix.py",
    "tests/test_m6b_fix_rk1_acoustic_loop.py",
    "tests/test_m6x_c2_acoustic.py",
    "tests/test_m6x_vertical_acoustic_oracle.py",
    "tests/test_v020_acoustic_substeps_env.py",
    "tests/test_v0234_nested_boundary_science_repair.py",
    "tests/test_v0234_final_ni_fable5_cpu_probe.py",
    "tests/test_v0234_final_ni_fable5_gpu_policy.py",
)
PY_COMPILE_FILES = (
    "scripts/v0234_final_ni_fable5_authority_proof.py",
    "scripts/v0234_final_ni_fable5_candidate_proof.py",
    "scripts/v0234_final_ni_fable5_carry_probe.py",
    "scripts/v0234_final_ni_fable5_cpu_ab_probe.py",
    "scripts/v0234_final_ni_fable5_final_proof.py",
    "scripts/v0234_final_ni_fable5_gpu_policy_proof.py",
    "scripts/v0234_final_ni_fable5_gpu_probe.py",
    "scripts/v0234_final_ni_fable5_source_proof.py",
)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_component(name: str, expected_verdict: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = SPRINT_DIR / name
    payload = json.loads(path.read_text())
    digest = canonical_hash(payload)
    if payload.get("proof_sha256") != digest:
        raise RuntimeError(f"{name}: canonical hash mismatch")
    if payload.get("verdict") != expected_verdict:
        raise RuntimeError(f"{name}: verdict {payload.get('verdict')!r}")
    return payload, {"path": str(path), "proof_sha256": digest, "verdict": payload["verdict"]}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def git_file_sha256(commit: str, path: str) -> str:
    data = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(data).hexdigest()


def correction_source_audit(model_commit: str) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", CORRECTION_PARENT, model_commit, "--", *MODEL_FILES],
        check=True,
        capture_output=True,
    ).stdout
    added = [
        line[1:]
        for line in diff.decode().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    hits = [token for token in FORBIDDEN_SOURCE for line in added if token in line]
    if hits:
        raise RuntimeError(f"correction adds callback/transfer source: {hits}")
    blob_lineage = {
        path: {
            "corrected_cpu_commit_sha256": git_file_sha256(CORRECTED_CPU_COMMIT, path),
            "gpu_execution_commit_sha256": git_file_sha256(GPU_EXECUTION_COMMIT, path),
            "aggregate_head_sha256": git_file_sha256(model_commit, path),
        }
        for path in MODEL_FILES
    }
    blob_identity = all(len(set(row.values())) == 1 for row in blob_lineage.values())
    if not blob_identity:
        raise RuntimeError(f"model blob lineage differs across executions: {blob_lineage}")
    return {
        "correction_parent": git("rev-parse", CORRECTION_PARENT),
        "model_commit": model_commit,
        "model_tree": git("rev-parse", f"{model_commit}^{{tree}}"),
        "correction_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "added_lines_scanned": len(added),
        "forbidden_source_hits": hits,
        "model_blob_lineage": blob_lineage,
        "model_blob_identity_across_cpu_gpu_and_aggregate": blob_identity,
        "operational_carry_source_unchanged": (
            git("rev-parse", f"{CORRECTION_PARENT}:src/gpuwrf/runtime/operational_state.py")
            == git("rev-parse", f"{model_commit}:src/gpuwrf/runtime/operational_state.py")
        ),
    }


def run_tests(label: str, test_files: tuple[str, ...]) -> dict[str, Any]:
    command = [
        "/usr/bin/taskset",
        "-c",
        "0-3",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "JAX_PLATFORMS=cpu",
        "JAX_PLATFORM_NAME=cpu",
        "JAX_ENABLE_X64=1",
        "JAX_ENABLE_COMPILATION_CACHE=false",
        "PYTHONPATH=.:src",
        "python3",
        "-m",
        "pytest",
        "-q",
        *test_files,
    ]
    completed = subprocess.run(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    normalized = re.sub(
        r"\s+in\s+[0-9]+(?:\.[0-9]+)?s(?:\s+\([^\n)]*\))?(?=\n?$)",
        " in <elapsed>s",
        completed.stdout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"focused tests failed:\n{completed.stdout[-8000:]}")
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": normalized[-2000:],
    }


def run_static_gates() -> dict[str, Any]:
    compile_command = [sys.executable, "-m", "py_compile", *PY_COMPILE_FILES]
    compiled = subprocess.run(
        compile_command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if compiled.returncode != 0:
        raise RuntimeError(f"py_compile failed:\n{compiled.stdout}")
    diff_command = ["git", "-C", str(REPO), "diff", "--check"]
    diff_check = subprocess.run(
        diff_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if diff_check.returncode != 0:
        raise RuntimeError(f"git diff --check failed:\n{diff_check.stdout}")
    return {
        "py_compile": {
            "command": compile_command,
            "returncode": compiled.returncode,
            "stdout": compiled.stdout,
        },
        "git_diff_check": {
            "command": diff_command,
            "returncode": diff_check.returncode,
            "stdout": diff_check.stdout,
        },
    }


def load_arm(arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = CPUAB_ROOT / ARM_DIRS[arm] / f"cpu-ab-{arm}-proof.json"
    payload = json.loads(path.read_text())
    if payload.get("proof_sha256") != canonical_hash(payload):
        raise RuntimeError(f"{arm} arm proof hash mismatch")
    if (
        payload.get("arm") != arm
        or payload.get("schema") != "gpuwrf.v0234.final-ni-fable5-cpu-ab-arm.v1"
    ):
        raise RuntimeError(f"{arm} arm identity mismatch")
    return payload, {
        "path": str(path),
        "proof_sha256": payload["proof_sha256"],
    }


def main() -> int:
    model_commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--", *MODEL_FILES):
        raise RuntimeError("model files must be committed")

    _, authority_row = load_component("authority-proof.json", "AUTHORITY_GREEN")
    _, localization_row = load_component(
        "carry-localization-proof.json", "INTERIOR_PH_FREEZE_CONFIRMED"
    )
    _, source_row = load_component("source-invariant-proof.json", "SOURCE_DISCREPANCY_PROVEN")
    interface, interface_row = load_component(
        "interface-transfer-proof.json",
        "CANDIDATE_OFF_E0D0_EXACT_CANDIDATE_ON_INTERFACE_CLEAN",
    )
    amendment1_path = SPRINT_DIR / "contract-amendment-01.json"
    amendment1 = json.loads(amendment1_path.read_text())
    if amendment1.get("verdict") != "OWNERSHIP_AMENDED_FAIL_CLOSED":
        raise RuntimeError("Fable contract amendment not green")
    amendment2_path = SPRINT_DIR / "contract-amendment-02.json"
    amendment2 = json.loads(amendment2_path.read_text())
    if amendment2.get("proof_sha256") != canonical_hash(amendment2):
        raise RuntimeError("GPT-MAX contract amendment canonical hash mismatch")
    if amendment2.get("verdict") != "GPTMAX_CONTINUATION_OWNERSHIP_AMENDED_FAIL_CLOSED":
        raise RuntimeError("GPT-MAX contract amendment not green")
    amendment3_path = SPRINT_DIR / "contract-amendment-03.json"
    amendment3 = json.loads(amendment3_path.read_text())
    if amendment3.get("proof_sha256") != canonical_hash(amendment3):
        raise RuntimeError("CPU control repair amendment canonical hash mismatch")
    if amendment3.get("verdict") != "CPU_CONTROL_STOP_AFTER_ALARM_REPAIR_AMENDED_FAIL_CLOSED":
        raise RuntimeError("CPU control repair amendment not green")

    falsified, falsified_row = load_arm("falsified")
    corrected, corrected_row = load_arm("corrected")

    fals_obs = falsified["observation"]
    corr_obs = corrected["observation"]
    fals_cold = falsified["cold_start_vs_retained_step0"]
    corr_cold = corrected["cold_start_vs_retained_step0"]
    ab_gates = {
        "exact_arm_commits": (
            falsified["model"]["commit"] == CORRECTION_PARENT
            and corrected["model"]["commit"] == CORRECTED_CPU_COMMIT
        ),
        "cold_start_cpu_arms_exactly_equal_with_qke_device_seam_only": (
            fals_cold["compared_leaves"] == corr_cold["compared_leaves"]
            and fals_cold["compared_leaves"] > 0
            and fals_cold["mismatched_leaves"] == corr_cold["mismatched_leaves"]
            and corr_cold["mismatched_leaves"] == ["qke"]
        ),
        "falsified_guard_raised_nonfinite": bool(fals_obs["finite_guard"]["raised"]),
        "falsified_ni_nonfinite": "Ni" in fals_obs["nonfinite_fields"],
        "falsified_ni_first_index_reproduced": (
            fals_obs["ni_first_nonfinite_scan_index"] == [0, 1, 1]
        ),
        "falsified_ph_beyond_ring0_bit_frozen": bool(
            fals_obs["ph_beyond_ring0_bit_frozen"]
        ),
        "falsified_stopped_only_after_completed_step200_alarm": (
            falsified.get("runner_termination")
            == {
                "stop_after_alarm_requested": True,
                "stopped_after_completed_alarm": True,
                "last_completed_d03_step": 200,
                "model_or_numerical_change": False,
            }
        ),
        "corrected_guard_green": not corr_obs["finite_guard"]["raised"],
        "corrected_all_finite": not corr_obs["nonfinite_fields"],
        "corrected_interior_ph_advances": (
            corr_obs["ph_interior_max_abs_change"] > 0.0
            and not corr_obs["ph_beyond_ring0_bit_frozen"]
        ),
        "same_platform_same_inputs": (
            falsified["backend"] == corrected["backend"] == "cpu"
            and falsified["input_dir"] == corrected["input_dir"]
            and falsified["root_steps"] == corrected["root_steps"]
            and falsified["environment"] == corrected["environment"]
        ),
    }
    if not all(ab_gates.values()):
        raise RuntimeError(f"CPU A/B gates red: {ab_gates}")

    source_audit = correction_source_audit(model_commit)
    focused_tests = run_tests("focused_candidate", FOCUSED_TEST_FILES)
    proportional_tests = run_tests("proportional_acoustic", PROPORTIONAL_TEST_FILES)
    static_gates = run_static_gates()

    proof = {
        "schema": SCHEMA,
        "verdict": "CPU_SOURCE_CANDIDATE_GREEN_BOUNDED_GPU_PENDING",
        "model_commit": model_commit,
        "frozen_inputs": {
            "authority": authority_row,
            "carry_localization": localization_row,
            "source_invariant": source_row,
            "interface_transfer": interface_row,
            "contract_amendments": {
                "fable": {
                    "path": str(amendment1_path),
                    "file_sha256": hashlib.sha256(amendment1_path.read_bytes()).hexdigest(),
                    "verdict": amendment1["verdict"],
                },
                "gptmax": {
                    "path": str(amendment2_path),
                    "proof_sha256": amendment2["proof_sha256"],
                    "verdict": amendment2["verdict"],
                },
                "cpu_control_repair": {
                    "path": str(amendment3_path),
                    "proof_sha256": amendment3["proof_sha256"],
                    "verdict": amendment3["verdict"],
                },
            },
        },
        "correction": {
            **source_audit,
            "mechanism": (
                "give the nested in-loop spec_bdyupdate_ph an explicit advanced-base "
                "operand: advance_w's ph_next is kept everywhere outside the spec ring "
                "(WRF updates grid%ph_2 in place) and the ring walks from the "
                "pre-advance loop-carried value; ring algebra, cadence, ownership, and "
                "the U/V/T/MU/MUTS/W walks are unchanged"
            ),
            "generality": (
                "all live nested domains using the coupled frozen-WRF boundary bundle; "
                "no clamp, mask, sanitizer, threshold move, case coordinate, or output workaround"
            ),
        },
        "cpu_ab_discriminator": {
            "falsified_arm": {**falsified_row, "observation": fals_obs},
            "corrected_arm": {**corrected_row, "observation": corr_obs},
            "gates": ab_gates,
        },
        "operator_gates": {
            "candidate_off_exact": interface["candidate_off"]["exact"],
            "ordinary_candidate_on_finite": interface["candidate_on"]["output_finite"],
            "forbidden_hlo_tokens": interface["candidate_on"]["forbidden_hlo_tokens"],
            "production_carry_leaf_count": interface["production_interface"]["leaf_count"],
            "new_loop_host_device_transfer": False,
            "observer_or_callback_added": False,
        },
        "focused_tests": focused_tests,
        "proportional_tests": proportional_tests,
        "static_gates": static_gates,
        "commands": [
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_authority_proof.py",
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_carry_probe.py",
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_source_proof.py",
            "python scripts/v0234_candidate_off_e0d0_proof.py --output .agent/sprints/2026-07-14-v0234-final-ni-fable5/interface-transfer-proof.json",
            "scripts/v0234_final_ni_fable5_cpu_ab_probe.py (both arms, env in command-log)",
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_candidate_proof.py",
        ],
        "procedural_admission": {
            "gpu_arm_was_already_running_when_gptmax_took_ownership": True,
            "gpu_launch_preceded_cpu_aggregate_completion": True,
            "gpu_results_admissible_only_after_this_cpu_proof_is_green": True,
            "additional_gpu_arms_launched_by_gptmax": 0,
            "critic_review_required": True,
        },
    }
    proof["proof_sha256"] = canonical_hash(proof)
    out_path = SPRINT_DIR / "candidate-cpu-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"verdict={proof['verdict']}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
