#!/usr/bin/env python3
"""Seal the CPU/static proof for the v0.23.4 spec-ring cadence correction."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO_ROOT / ".agent/sprints/2026-07-14-v0234-1500-scientific-rca"
CORRECTION_PARENT = "7465711d"
MODEL_COMMIT = "0b4374a5"
MODEL_FILES = (
    "src/gpuwrf/coupling/boundary_apply.py",
    "src/gpuwrf/dynamics/core/acoustic.py",
    "src/gpuwrf/runtime/operational_mode.py",
)
FORBIDDEN_SOURCE = (
    "jax.device_get",
    "jax.pure_callback",
    "io_callback",
    "host_callback",
    "debug.callback",
    "outside_compilation",
)
TEST_FILES = (
    "tests/test_v0234_1500_scientific_rca_candidate.py",
    "tests/test_v0234_1500_scientific_rca_source_oracle.py",
    "tests/test_v0234_1500_scientific_rca_trajectory.py",
    "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py",
    "tests/test_v0234_nested_boundary_critic_repair.py",
    "tests/test_v014_specified_bdy_cadence.py",
    "tests/test_m6_boundary_apply.py",
    "tests/test_v014_pre_halo_capture.py",
)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated_proof(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = SPRINT_DIR / name
    payload = json.loads(path.read_text())
    expected = payload.get("proof_sha256")
    actual = canonical_hash(payload)
    if expected != actual:
        raise RuntimeError(f"{name} canonical proof changed: {expected} != {actual}")
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "proof_sha256": actual,
    }


def git(*args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(REPO_ROOT), *args), text=True
    ).strip()


def correction_source_audit(model_commit: str) -> dict[str, Any]:
    changed = tuple(
        line
        for line in git(
            "diff", "--name-only", CORRECTION_PARENT, model_commit, "--", "src/gpuwrf"
        ).splitlines()
        if line
    )
    if set(changed) != set(MODEL_FILES):
        raise RuntimeError(f"unexpected model correction surface: {changed}")
    diff = subprocess.check_output(
        (
            "git", "-C", str(REPO_ROOT), "diff", "--binary",
            CORRECTION_PARENT, model_commit, "--", *MODEL_FILES,
        )
    )
    text = diff.decode(errors="replace")
    added = tuple(
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    hits = {
        token: [line for line in added if token in line]
        for token in FORBIDDEN_SOURCE
    }
    if any(hits.values()):
        raise RuntimeError(f"correction adds callback/transfer source: {hits}")
    return {
        "parent": git("rev-parse", CORRECTION_PARENT),
        "model_commit": model_commit,
        "model_tree": git("rev-parse", f"{model_commit}^{{tree}}"),
        "changed_model_files": list(changed),
        "correction_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_blobs": {
            path: git("rev-parse", f"{model_commit}:{path}") for path in MODEL_FILES
        },
        "operational_carry_source_unchanged": (
            git("rev-parse", f"{CORRECTION_PARENT}:src/gpuwrf/runtime/operational_state.py")
            == git("rev-parse", f"{model_commit}:src/gpuwrf/runtime/operational_state.py")
        ),
        "added_lines_scanned": len(added),
        "forbidden_source_hits": hits,
    }


def run_tests() -> dict[str, Any]:
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "JAX_PLATFORMS=cpu",
        "JAX_PLATFORM_NAME=cpu",
        "JAX_ENABLE_X64=1",
        "JAX_ENABLE_COMPILATION_CACHE=false",
        "PYTHONPATH=.:src",
        "python",
        "-m",
        "pytest",
        "-q",
        *TEST_FILES,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    normalized = re.sub(
        r"\s+in\s+[0-9]+(?:\.[0-9]+)?s(?:\s+\([^\n)]*\))?(?=\n?$)",
        " in <elapsed>s",
        completed.stdout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"focused candidate tests failed:\n{completed.stdout[-8000:]}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "stdout_tail": normalized[-4000:],
    }


def build_proof() -> dict[str, Any]:
    model_commit = git("rev-parse", MODEL_COMMIT)
    if git("status", "--porcelain", "--", *MODEL_FILES):
        raise RuntimeError("model source must be committed before candidate proof")
    authority, authority_row = authenticated_proof("authority-proof.json")
    trajectory, trajectory_row = authenticated_proof("trajectory-spatial-proof.json")
    source, source_row = authenticated_proof("source-invariant-proof.json")
    amendment, amendment_row = authenticated_proof("contract-amendment-01.json")
    interface, interface_row = authenticated_proof("interface-transfer-proof.json")
    if authority.get("verdict") != "AUTHORITY_GREEN":
        raise RuntimeError("authority proof is not green")
    if trajectory.get("verdict") != "TRAJECTORY_RECONSTRUCTED":
        raise RuntimeError("trajectory proof is not complete")
    if source.get("verdict") != "SOURCE_DISCREPANCY_PROVEN":
        raise RuntimeError("source discrepancy is not proven")
    if amendment.get("verdict") != "OWNERSHIP_AMENDED_FAIL_CLOSED":
        raise RuntimeError("contract amendment is not green")
    if (
        interface.get("verdict")
        != "CANDIDATE_OFF_E0D0_EXACT_CANDIDATE_ON_INTERFACE_CLEAN"
        or interface["candidate_off"]["exact"] is not True
        or interface["production_interface"]["leaf_count"] != 106
        or interface["candidate_on"]["output_finite"] is not True
        or any(interface["candidate_on"]["forbidden_hlo_tokens"].values())
    ):
        raise RuntimeError("ordinary candidate interface/HLO proof is not green")

    source_audit = correction_source_audit(model_commit)
    if not source_audit["operational_carry_source_unchanged"]:
        raise RuntimeError("OperationalCarry source changed")
    tests = run_tests()
    return {
        "schema": "gpuwrf.v0234.1500-scientific-rca-candidate-cpu.v1",
        "verdict": "CPU_SOURCE_CANDIDATE_GREEN_DYNAMIC_FULL_HISTORY_PENDING",
        "frozen_inputs": {
            "authority": authority_row,
            "trajectory": trajectory_row,
            "source_invariant": source_row,
            "contract_amendment": amendment_row,
            "interface_transfer": interface_row,
        },
        "correction": {
            **source_audit,
            "mechanism": "replace nested RK-stage endpoint hard pins with pristine additive pre-substep spec updates for U/V/T/MU/MUTS/W and the exact mass-reweighted PH equation; retain advance_mu_t muave",
            "generality": "all live nested domains using the coupled frozen-WRF boundary bundle; no case coordinate, variable threshold, clamp, or output path",
            "candidate_flag": "nested_frozen_wrf_boundary_bundle",
        },
        "operator_gates": {
            "old_candidate_reproduced_by_independent_algebra": True,
            "corrected_u_v_t_mu_muts_w_additive_walk_matches_wrf": True,
            "corrected_ph_coupled_conservation_oracle": source["conservation_oracle"],
            "muave_absent_from_nested_spec_update": True,
            "stagger_and_corner_partitions": "PASS_U_V_MASS_FULLLEVEL_Y_SIDE_OWNER",
            "parent_record_cadence": "PASS_TWO_RECORD_PARENT_DT_TENDENCY",
            "candidate_off_exact": interface["candidate_off"]["exact"],
            "ordinary_candidate_on_finite": interface["candidate_on"]["output_finite"],
            "ordinary_candidate_on_forbidden_hlo_tokens": interface["candidate_on"]["forbidden_hlo_tokens"],
            "production_carry_leaf_count": interface["production_interface"]["leaf_count"],
            "new_loop_host_device_transfer": False,
            "observer_or_callback_added": False,
        },
        "focused_tests": tests,
        "dynamic_gates": {
            "gpu_used": False,
            "gpu_compile_count": 0,
            "gpu_dispatch_count": 0,
            "gpu_wall_seconds": 0.0,
            "d03_1500_no_worse": "UNREACHABLE_FROM_RETAINED_STATE_WITHOUT_REPLAYING_CORRECTED_HISTORY",
            "steps_9313_9314_9315_9405": "UNREACHABLE_FROM_RETAINED_STATE_WITHOUT_REPLAYING_CORRECTED_HISTORY",
            "reason": "retained carries already contain 14h40m of the disproven endpoint-pin trajectory; splicing the corrected operator there cannot test cumulative 15:00 accuracy or late-Ni stability",
            "policy": "explicit critic/full-replay risk; no inferred pass and no contaminated same-carry GPU arm",
        },
        "causal_verdict": {
            "boundary_application": "PROVEN_SOURCE_CAUSE_AND_CORRECTED",
            "dry_mass_pressure": "PROVEN_DOWNSTREAM_PATH",
            "forcedown_sint": "FALSIFIED_AS_EARLIEST_CAUSE",
            "corner_interpolation": "FALSIFIED_AS_EARLIEST_CAUSE",
            "ni": "SEPARATE_LATE_MECHANISM; corrected-history 9405 gate pending",
            "v10": "SEPARATE_METRIC; 15:00 no-worse gate pending",
        },
        "unresolved_risks": [
            "A fresh corrected-history replay is required to test every frozen 15:00 no-worse maximum.",
            "The same replay must demonstrate finite steps 9313/9314/9315/9405 and preserved late-Ni stability.",
            "An independent science critic must accept the source/cadence correction before that GPU replay.",
        ],
        "commands": [
            "python scripts/v0234_1500_scientific_rca_source_oracle.py",
            "python scripts/v0234_candidate_off_e0d0_proof.py --output .agent/sprints/2026-07-14-v0234-1500-scientific-rca/interface-transfer-proof.json",
            "python scripts/v0234_1500_scientific_rca_candidate_proof.py",
            "git diff --check",
        ],
        "gpu_commands": 0,
        "jax_backend_for_cpu_gates": "cpu",
    }


def main() -> int:
    payload = build_proof()
    payload["proof_sha256"] = canonical_hash(payload)
    output = SPRINT_DIR / "candidate-cpu-proof.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "proof_sha256": payload["proof_sha256"],
                "verdict": payload["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
