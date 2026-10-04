"""Pre-import runner bindings for the admitted 04478354 theta candidate.

This module mutates only runner authority/constants before admission.  It does
not import JAX/gpuwrf and does not alter any production model callable.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping


CANDIDATE = "044783549697caf8c80d094f2b3e73f5ef340537"
CANDIDATE_TREE = "87b0242e0e7af3b0c6cdcf49bd25c18fe7c42f2a"
PARENT_CANDIDATE = "571e4a4208cd01941ec91c9bea4b8ae6ae4806c5"
PARENT_MODEL = "cb46ef1b3179871382c1277adc09056ade5c5cec"
PARTIAL_WIND = "2c13b73112d9877d603324d66b127ecad60bf7e3"
NAMESPACE = "nested_theta_sixth_order_04478354_full18h_discriminator1"
LOCK_LABEL = "v0234-nested-theta-sixth-full18h"


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_THETA_SIXTH_PROFILE_APPLIED", False):
        return

    old_candidate = runner.CANDIDATE_COMMIT
    old_tree = runner.CANDIDATE_TREE
    old_sources = dict(runner.ACCEPTED_SOURCE_HASHES)
    old_namespace = runner.FULL_REPLAY_NAMESPACE
    prior_authority = runner.assert_final_candidate_proof_authority

    final_dir = runner.FINAL_NI_DIR
    amendment14 = final_dir / "contract-amendment-14.json"
    amendment15 = final_dir / "contract-amendment-15.json"
    source_oracle = final_dir / "nested-theta-sixth-order-source-oracle.json"
    complete_ab = final_dir / "nested-theta-sixth-order-complete-cpu-ab-proof.json"
    parent_auth = final_dir / "nested-theta-sixth-order-actual-parent-auth.json"
    candidate_proof = final_dir / "nested-theta-sixth-order-candidate-proof.json"

    proof_constants = {
        "amendment14": (
            amendment14,
            "1d354afa65b63ba17d7e1055edc0bc6c88c32bd53aae44ef4182c3eabed2f0ad",
            "67479e5faf493d496fa32340d02e783b420b14e84ae0ca97caf18c3b120cda98",
            False,
        ),
        "amendment15": (
            amendment15,
            "0daaa4514a68cf8222d616f11a1ed561d25c4af1e146ab56559f3cf6f5826621",
            "36ed6b56097ed57392b2fbe45b9d63a68656f55015c9083be3f98f95e90f2519",
            False,
        ),
        "source_oracle": (
            source_oracle,
            "7afa34dfd983113d0b294a281f3d597819f7469f31202e78f8fedfd340eb6b75",
            "150d38a525e2e35fc425abb527bbfddd27df02da922b9f7b96baafd06b1c09a6",
            True,
        ),
        "complete_cpu_ab": (
            complete_ab,
            "643595907584e845efdf34c1443b1486078e6bdebd3f39261e7c2396684538d3",
            "f89469af26eba6f96655d35401ce81db4493b023d7893db3cb9f1bd6bcd602b6",
            True,
        ),
        "actual_parent_auth": (
            parent_auth,
            "2d6d3701065e76f53720a08cccb91fe645faabfcef52e3a8aefddbd3936e86e6",
            "b3d6f06d9543bc7a9c027246aa9d30de54d381392ed8cc952c28e78e57cfc549",
            True,
        ),
        "candidate_proof": (
            candidate_proof,
            "151b3baca7ae45f51406c81838c05a8339e04c7ba9ff030f152318510978e0f7",
            "8bbac566bf584717df4af9e06b28cd3ad4888e5dd731c35c24413a9802f86c3e",
            True,
        ),
    }

    runner.SCHEMA = "gpuwrf.v0234.nested-theta-sixth-order-full18h.v1"
    runner.CPU_PROOF_SCHEMA = (
        "gpuwrf.v0234.nested-theta-sixth-order-runner-cpu-audit.v1"
    )
    runner.AUDIT_ADMISSION = "READY_FOR_NESTED_THETA_SIXTH_GPU_REPLAY"
    runner.CANDIDATE_COMMIT = CANDIDATE
    runner.CANDIDATE_TREE = CANDIDATE_TREE
    runner.FULL_REPLAY_NAMESPACE = NAMESPACE
    runner.LOCK_LABEL = LOCK_LABEL
    runner.LAUNCH_COMMAND = runner.SPRINT_DIR / (
        "nested-theta-sixth-order-full18h-exact-launch-command.txt"
    )
    runner.RUNNER_CPU_AUDIT = runner.SPRINT_DIR / (
        "nested-theta-sixth-order-full18h-runner-cpu-proof.json"
    )
    runner.EARLY_CAUSAL_SCALAR_PARENT = runner.LINEAGE_WORK_DIR / (
        "nested_t_source_cb46ef1b_full18h_discriminator1/output/"
        "wrfout_d03_2025-03-01_00:20:00"
    )
    runner.EARLY_CAUSAL_SCALAR_PARENT_SHA256 = (
        "b24a928f1a85c7034cc15e64bbf01cc9aeb5aad6dc0486d294f08289d6299701"
    )
    runner.EARLY_CAUSAL_RING1_BASELINE = {
        "T": {
            "prior_vs_cpu": 0.07439330567219023,
            "prior_vs_retry20": 0.05252442272211308,
            "partial_vs_cpu": 0.07753516846062372,
            "partial_vs_retry20": 0.0588941045970705,
            "scalar_vs_cpu": 0.07677568686572497,
            "scalar_vs_retry20": 0.05897056871683568,
        },
        "U": {
            "prior_vs_cpu": 0.2857256114890511,
            "prior_vs_retry20": 0.27772695352886917,
            "partial_vs_cpu": 0.20247207650654608,
            "partial_vs_retry20": 0.19592398176235976,
            "scalar_vs_cpu": 0.19936177084893247,
            "scalar_vs_retry20": 0.1926764711328736,
        },
    }
    runner.ACCEPTED_SOURCE_HASHES = {
        **old_sources,
        "src/gpuwrf/dynamics/explicit_diffusion.py": (
            "8fc7726b2de017cf5ff4896aa85eed67e419ad9fd011b00addb50568279d2f61"
        ),
        "src/gpuwrf/runtime/operational_mode.py": (
            "db43339801830ed4196d301239445ef8f4352b9cc3041de2e1d4dc618873afcc"
        ),
    }
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        **runner.CANDIDATE_CLEAN_AUTHORITY,
        "theta_sixth_candidate_commit": CANDIDATE,
        "theta_sixth_candidate_proof_sha256": (
            "8bbac566bf584717df4af9e06b28cd3ad4888e5dd731c35c24413a9802f86c3e"
        ),
        "theta_sixth_actual_parent_auth_sha256": (
            "b3d6f06d9543bc7a9c027246aa9d30de54d381392ed8cc952c28e78e57cfc549"
        ),
        "theta_sixth_direct_momentum_source_changed": False,
        "theta_sixth_cpu_tests": "98 passed",
    }

    def _read_new_proof(
        name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        path, file_hash, payload_hash, embedded = proof_constants[name]
        if embedded:
            return runner._authenticated_canonical_json(
                path, file_hash, payload_hash, f"THETA_SIXTH_{name.upper()}"
            )
        payload, row = runner.read_authenticated_json(
            path, file_hash, f"THETA_SIXTH_{name.upper()}_FILE"
        )
        observed = runner.canonical_digest(payload)
        if observed != payload_hash:
            raise runner.RunnerGateError(
                f"THETA_SIXTH_{name.upper()}_CANONICAL",
                f"expected={payload_hash} observed={observed}",
            )
        return payload, {**row, "canonical_payload_sha256": observed}

    def assert_theta_candidate_authority() -> dict[str, Any]:
        current_candidate = runner.CANDIDATE_COMMIT
        current_tree = runner.CANDIDATE_TREE
        current_sources = runner.ACCEPTED_SOURCE_HASHES
        current_namespace = runner.FULL_REPLAY_NAMESPACE
        runner.CANDIDATE_COMMIT = old_candidate
        runner.CANDIDATE_TREE = old_tree
        runner.ACCEPTED_SOURCE_HASHES = old_sources
        runner.FULL_REPLAY_NAMESPACE = old_namespace
        try:
            prior = prior_authority()
        finally:
            runner.CANDIDATE_COMMIT = current_candidate
            runner.CANDIDATE_TREE = current_tree
            runner.ACCEPTED_SOURCE_HASHES = current_sources
            runner.FULL_REPLAY_NAMESPACE = current_namespace

        payloads = {}
        rows = {}
        for name in proof_constants:
            payloads[name], rows[name] = _read_new_proof(name)
        source = payloads["source_oracle"]
        ab = payloads["complete_cpu_ab"]
        parent = payloads["actual_parent_auth"]
        candidate = payloads["candidate_proof"]
        amendment = payloads["amendment15"]
        candidate_row = candidate.get("candidate") or {}
        candidate_checks = candidate.get("checks") or {}
        source_checks = source.get("checks") or {}
        parent_checks = parent.get("checks") or {}
        false_ab = sorted(
            key for key, value in (ab.get("checks") or {}).items() if not value
        )
        if (
            payloads["amendment14"].get("verdict")
            != "NESTED_RK1_THETA_SIXTH_ORDER_AMENDED_FAIL_CLOSED"
            or amendment.get("verdict")
            != "NESTED_THETA_SIXTH_ORDER_GATE_SEMANTICS_AMENDED_FAIL_CLOSED"
            or source.get("verdict")
            != "NESTED_THETA_SIXTH_ORDER_SOURCE_ORACLE_GREEN"
            or not source_checks
            or not all(source_checks.values())
            or false_ab != ["retained_parent_hlo_authenticated"]
            or parent.get("verdict")
            != "NESTED_THETA_SIXTH_ORDER_ACTUAL_PARENT_AUTH_GREEN"
            or not parent_checks
            or not all(parent_checks.values())
            or candidate.get("schema")
            != "gpuwrf.v0234.nested-theta-sixth-order-candidate.v1"
            or candidate.get("verdict")
            != "NESTED_THETA_SIXTH_ORDER_GPU_STEP200_ADMITTED"
            or candidate_row.get("commit") != CANDIDATE
            or candidate_row.get("tree") != CANDIDATE_TREE
            or candidate_row.get("parent_falsified_commit") != PARENT_CANDIDATE
            or candidate_row.get("partial_wind_commit") != PARTIAL_WIND
            or not candidate_checks
            or not all(candidate_checks.values())
            or candidate.get("causal_scope", {}).get("direct_momentum_source_changed")
            is not False
            or candidate.get("causal_scope", {}).get("partial_wind_mechanism_retained")
            is not True
            or candidate.get("causal_scope", {}).get("v10_link_claimed") is not False
            or candidate.get("complete_cpu_ab", {}).get("leaf_count") != 106
            or candidate.get("complete_cpu_ab", {}).get("all_finite") is not True
            or candidate.get("complete_cpu_ab", {}).get("forbidden_callback_targets")
            != []
            or candidate.get("gpu_discriminator", {}).get(
                "gpu_commands_run_for_candidate"
            )
            != 0
            or candidate.get("gpu_discriminator", {}).get(
                "full_forecasts_run_for_candidate"
            )
            != 0
        ):
            raise runner.RunnerGateError(
                "THETA_SIXTH_CANDIDATE_AUTHORITY_SEMANTICS",
                repr(
                    {
                        "candidate": candidate_row,
                        "candidate_verdict": candidate.get("verdict"),
                        "false_ab": false_ab,
                        "source_verdict": source.get("verdict"),
                        "parent_verdict": parent.get("verdict"),
                    }
                ),
            )
        return {
            **prior,
            "prior_t_source_candidate_commit": prior["t_source_candidate_commit"],
            "t_source_candidate_commit": CANDIDATE,
            "theta_sixth_amendment14": rows["amendment14"],
            "theta_sixth_amendment15": rows["amendment15"],
            "theta_sixth_source_oracle": rows["source_oracle"],
            "theta_sixth_complete_cpu_ab": rows["complete_cpu_ab"],
            "theta_sixth_actual_parent_auth": rows["actual_parent_auth"],
            "theta_sixth_candidate_proof": rows["candidate_proof"],
            "theta_sixth_candidate_commit": CANDIDATE,
            "theta_sixth_parent_model_commit": PARENT_MODEL,
            "earliest_causal_gate": "d03 step 200 / 00:20",
            "full_18h_authorized_only_after_step200_green": True,
        }

    def assert_theta_source_authority(
        environment: Mapping[str, str], *, require_clean: bool,
    ) -> dict[str, Any]:
        head = runner._git(runner.REPO_ROOT, "rev-parse", "HEAD")
        approved_runner = environment.get("GPUWRF_NESTED_BUNDLE_RUNNER_SHA")
        if approved_runner and approved_runner != head:
            raise runner.RunnerGateError(
                "RUNNER_HEAD", f"approved={approved_runner} head={head}"
            )
        dirty = runner._git(runner.REPO_ROOT, "status", "--porcelain")
        if require_clean and dirty:
            raise runner.RunnerGateError("WORKTREE_DIRTY", dirty)
        tree = runner._git(
            runner.REPO_ROOT, "rev-parse", f"{CANDIDATE}^{{tree}}"
        )
        if tree != CANDIDATE_TREE:
            raise runner.RunnerGateError("CANDIDATE_TREE", tree)
        ancestor = subprocess.run(
            (
                "git",
                "-C",
                str(runner.REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                CANDIDATE,
                head,
            ),
            check=False,
        ).returncode == 0
        if not ancestor:
            raise runner.RunnerGateError(
                "CANDIDATE_ANCESTRY", f"{CANDIDATE} !<= {head}"
            )
        model_diff = runner._git(
            runner.REPO_ROOT,
            "diff",
            "--name-only",
            CANDIDATE,
            "--",
            "src/gpuwrf",
        )
        if model_diff:
            raise runner.RunnerGateError("ACCEPTED_MODEL_CHANGED", model_diff)
        delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", CANDIDATE, head
        ).splitlines()
        allowed_exact = {
            "scripts/v0234_nested_frozen_wrf_boundary_window.py",
            "scripts/v0234_nested_theta_sixth_order_cpu_ab.py",
            "scripts/v0234_nested_theta_sixth_order_parent_auth.py",
            "scripts/v0234_nested_theta_sixth_order_candidate_proof.py",
            "scripts/v0234_nested_theta_sixth_order_runner_profile.py",
        }
        allowed_prefixes = (
            str(runner.SPRINT_DIR.relative_to(runner.REPO_ROOT)) + "/",
            str(runner.FINAL_NI_DIR.relative_to(runner.REPO_ROOT)) + "/",
        )
        unexpected = sorted(
            path
            for path in delta
            if path not in allowed_exact
            and not any(path.startswith(prefix) for prefix in allowed_prefixes)
        )
        if unexpected:
            raise runner.RunnerGateError("RUNNER_ONLY_DELTA", repr(unexpected))
        source_rows = {
            relative: runner._require_sha(
                runner.REPO_ROOT / relative,
                expected,
                "ACCEPTED_SOURCE_HASH",
            )
            for relative, expected in runner.ACCEPTED_SOURCE_HASHES.items()
        }
        return {
            "runner_head": head,
            "approved_runner_head": approved_runner,
            "candidate_commit": CANDIDATE,
            "candidate_tree": tree,
            "candidate_is_ancestor": ancestor,
            "accepted_model_diff_empty": True,
            "runner_only_delta": delta,
            "unexpected_runner_delta": unexpected,
            "accepted_model_tree": runner._git(
                runner.REPO_ROOT, "rev-parse", f"{CANDIDATE}:src/gpuwrf"
            ),
            "accepted_sources": source_rows,
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "worktree_clean": not bool(dirty),
        }

    runner.assert_final_candidate_proof_authority = assert_theta_candidate_authority
    runner.assert_candidate_source_authority = assert_theta_source_authority
    runner._THETA_SIXTH_PROFILE_APPLIED = True
