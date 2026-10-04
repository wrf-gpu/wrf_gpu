"""Pre-import runner bindings for admitted 395fb800 WRF h_sca order candidate."""

from __future__ import annotations

import subprocess
from typing import Any, Mapping


CANDIDATE = "395fb800df0bb619462db8e312f734ca0c538161"
CANDIDATE_TREE = "23b6cc942d4b0b0b32cf86bf88ebdec1c693fd4e"
PARENT = "dc3fecdd12d0e55d095c6f110d178d7a2a6c3ef5"
PARENT_MODEL = "18d97595c59ca01840081f11109780c291dfcae8"
PARTIAL_WIND = "2c13b73112d9877d603324d66b127ecad60bf7e3"
NAMESPACE = "nested_h_sca_order_395fb800_full18h_discriminator1"
LOCK_LABEL = "v0234-nested-h-sca-order-full18h"


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_H_SCA_ORDER_PROFILE_APPLIED", False):
        return

    # Preserve every admitted 18d ancestry gate, then layer only the WRF
    # h_sca_adv_order loader correction.  This module is import/JAX free.
    from scripts.v0234_nested_scalar_sixth_order_runner_profile import (
        apply_profile as apply_scalar_profile,
    )

    apply_scalar_profile(runner)
    old_sources = dict(runner.ACCEPTED_SOURCE_HASHES)
    prior_authority = runner.assert_final_candidate_proof_authority

    final_dir = runner.FINAL_NI_DIR
    amendment18 = final_dir / "contract-amendment-18.json"
    source_oracle = final_dir / "nested-h-sca-order-source-oracle.json"
    hlo_audit = final_dir / "nested-h-sca-order-lowered-hlo-audit.json"
    complete_ab = final_dir / "nested-h-sca-order-complete-cpu-ab-proof.json"
    candidate_proof = final_dir / "nested-h-sca-order-candidate-proof.json"
    proof_constants = {
        "amendment18": (
            amendment18,
            "661f3873a9a010be8c355a3b54c003b715cdfa7270bbe85a395442b881644024",
            "16cf5876c00490f903f18672241ebbabe5ae634f386910907627780f6e4b7c10",
            False,
        ),
        "source_oracle": (
            source_oracle,
            "fa978ff6757610a28c1c4ab921db416bbb68f47b054ef85600597217c1812e3f",
            "6f1eb7fc778c5b72e120d3ff30ed1d55a8af020371d1c6b3b7edbab8e44b2637",
            True,
        ),
        "hlo_audit": (
            hlo_audit,
            "1eb9b82291305e9c45f01f52bd56193e1c9290fc89d4eed39f438fc16cf52fd0",
            "78077e56e6b88b38d831031a6ab3c863fd766676268cf103eec120d56c408689",
            True,
        ),
        "complete_cpu_ab": (
            complete_ab,
            "b573c51c4ad1ca99ab4b27f72f7381b221796a2f152e327c4f704a39fca74ac6",
            "4ff8e5a5aa430cc82fbd5c0bffbd7ba00c4b1f31f750c6bc6a39011b45877153",
            True,
        ),
        "candidate_proof": (
            candidate_proof,
            "1c59b2c539a48dd8287b772b9cea47df6d6bfe08534fb49ef472fef2e53e74e2",
            "f4017b84716f395798a31eb5d944b3977c79b92fbff32c520ed5ae03c7982154",
            True,
        ),
    }

    runner.SCHEMA = "gpuwrf.v0234.nested-h-sca-order-full18h.v1"
    runner.CPU_PROOF_SCHEMA = "gpuwrf.v0234.nested-h-sca-order-runner-cpu-audit.v1"
    runner.AUDIT_ADMISSION = "READY_FOR_NESTED_H_SCA_ORDER_GPU_REPLAY"
    runner.CANDIDATE_COMMIT = CANDIDATE
    runner.CANDIDATE_TREE = CANDIDATE_TREE
    runner.FULL_REPLAY_NAMESPACE = NAMESPACE
    runner.LOCK_LABEL = LOCK_LABEL
    runner.LAUNCH_COMMAND = runner.SPRINT_DIR / (
        "nested-h-sca-order-full18h-exact-launch-command.txt"
    )
    runner.RUNNER_CPU_AUDIT = runner.SPRINT_DIR / (
        "nested-h-sca-order-full18h-runner-cpu-proof.json"
    )
    runner.EARLY_CAUSAL_SCALAR_PARENT = runner.LINEAGE_WORK_DIR / (
        "nested_scalar_sixth_order_18d97595_full18h_discriminator1/output/"
        "wrfout_d03_2025-03-01_00:20:00"
    )
    runner.EARLY_CAUSAL_SCALAR_PARENT_SHA256 = (
        "951ba3fb51ef09913fa88272123699e332150815c9495526dc6f756180d71b45"
    )
    runner.EARLY_CAUSAL_RING1_BASELINE = {
        "T": {
            "prior_vs_cpu": 0.07439330567219023,
            "prior_vs_retry20": 0.05252442272211308,
            "partial_vs_cpu": 0.07753516846062372,
            "partial_vs_retry20": 0.0588941045970705,
            "scalar_vs_cpu": 0.07919418720753015,
            "scalar_vs_retry20": 0.06328271928135847,
        },
        "U": {
            "prior_vs_cpu": 0.2857256114890511,
            "prior_vs_retry20": 0.27772695352886917,
            "partial_vs_cpu": 0.20247207650654608,
            "partial_vs_retry20": 0.19592398176235976,
            "scalar_vs_cpu": 0.19926174599231458,
            "scalar_vs_retry20": 0.19249874544304876,
        },
    }
    runner.ACCEPTED_SOURCE_HASHES = {
        **old_sources,
        "src/gpuwrf/integration/nested_pipeline.py": (
            "48c6d4663bb7387265ea205a7f295228c3f737a9fd793c2dd55d27cdcf2edaad"
        ),
    }
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        **runner.CANDIDATE_CLEAN_AUTHORITY,
        "h_sca_order_candidate_commit": CANDIDATE,
        "h_sca_order_candidate_proof_sha256": (
            "f4017b84716f395798a31eb5d944b3977c79b92fbff32c520ed5ae03c7982154"
        ),
        "h_sca_order_rhs_ph_arithmetic_changed": False,
        "h_sca_order_cpu_tests": "87 passed",
        "h_sca_order_v10_link_claimed": False,
    }

    def _read_new_proof(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        path, file_hash, payload_hash, embedded = proof_constants[name]
        if embedded:
            return runner._authenticated_canonical_json(
                path, file_hash, payload_hash, f"H_SCA_ORDER_{name.upper()}"
            )
        payload, row = runner.read_authenticated_json(
            path, file_hash, f"H_SCA_ORDER_{name.upper()}_FILE"
        )
        observed = runner.canonical_digest(payload)
        if observed != payload_hash:
            raise runner.RunnerGateError(
                f"H_SCA_ORDER_{name.upper()}_CANONICAL",
                f"expected={payload_hash} observed={observed}",
            )
        return payload, {**row, "canonical_payload_sha256": observed}

    def assert_h_sca_candidate_authority() -> dict[str, Any]:
        prior = prior_authority()
        payloads: dict[str, dict[str, Any]] = {}
        rows: dict[str, dict[str, Any]] = {}
        for name in proof_constants:
            payloads[name], rows[name] = _read_new_proof(name)

        amendment = payloads["amendment18"]
        source = payloads["source_oracle"]
        hlo = payloads["hlo_audit"]
        ab = payloads["complete_cpu_ab"]
        candidate = payloads["candidate_proof"]
        sealed = amendment.get("sealed_candidate") or {}
        source_checks = source.get("checks") or {}
        hlo_checks = hlo.get("checks") or {}
        ab_checks = ab.get("checks") or {}
        candidate_checks = candidate.get("checks") or {}
        candidate_row = candidate.get("candidate") or {}
        causal = candidate.get("causal_verdict") or {}
        gpu = candidate.get("gpu_discriminator") or {}
        if (
            amendment.get("verdict")
            != "NESTED_H_SCA_ORDER_CANDIDATE_SEALED_GPU_HELD"
            or sealed.get("commit") != CANDIDATE
            or sealed.get("tree") != CANDIDATE_TREE
            or sealed.get("parent") != PARENT
            or sealed.get("arithmetic_kernel_edits") != 0
            or sealed.get("carry_or_result_leaf_edits") != 0
            or source.get("verdict") != "NESTED_H_SCA_ORDER_SOURCE_ORACLE_GREEN"
            or source.get("candidate_commit") != CANDIDATE
            or source.get("candidate_tree") != CANDIDATE_TREE
            or not source_checks
            or not all(source_checks.values())
            or hlo.get("verdict") != "NESTED_H_SCA_ORDER_HLO_AUDIT_GREEN"
            or not hlo_checks
            or not all(hlo_checks.values())
            or hlo.get("configuration")
            != {"h_sca_adv_order": 5, "leaf_count": 106, "moist_adv_opt": 0, "scalar_adv_opt": 0}
            or hlo.get("lowering", {}).get("compile_calls") != 0
            or hlo.get("lowering", {}).get("dispatch_calls") != 0
            or hlo.get("lowering", {}).get("forbidden_custom_targets") != []
            or hlo.get("lowering", {}).get("unknown_custom_targets") != []
            or ab.get("verdict") != "NESTED_H_SCA_ORDER_COMPLETE_CPU_AB_GREEN"
            or ab.get("candidate_commit") != CANDIDATE
            or ab.get("parent_commit") != PARENT
            or ab.get("partial_wind_commit") != PARTIAL_WIND
            or not ab_checks
            or not all(ab_checks.values())
            or ab.get("input", {}).get("leaf_count") != 106
            or ab.get("candidate_B", {}).get("audit", {}).get("interface_identity") is not True
            or ab.get("candidate_B", {}).get("audit", {}).get("forbidden_targets") != []
            or ab.get("candidate_B", {}).get("configuration")
            != {"h_sca_adv_order": 5, "moist_adv_opt": 0, "scalar_adv_opt": 0}
            or ab.get("candidate_B", {}).get("audit", {}).get("stablehlo_sha256")
            != "adb3c3eb4946fd78a8855bca5713cdef6d0fcf7bdfe72d2a94f6655e614732e6"
            or candidate.get("schema") != "gpuwrf.v0234.nested-h-sca-order-candidate.v1"
            or candidate.get("verdict")
            != "NESTED_H_SCA_ORDER_STEP200_LAUNCH_READY_HELD"
            or not candidate_checks
            or not all(candidate_checks.values())
            or candidate_row.get("commit") != CANDIDATE
            or candidate_row.get("tree") != CANDIDATE_TREE
            or candidate_row.get("parent") != PARENT
            or candidate.get("complete_cpu_ab", {}).get("leaf_count") != 106
            or candidate.get("complete_cpu_ab", {}).get("all_finite") is not True
            or causal.get("primary_lane") != "THM/PH"
            or causal.get("v10_link_claimed") is not False
            or causal.get("full_forecast_claim") is not False
            or gpu.get("gpu_commands_run_for_candidate") != 0
            or gpu.get("full_forecasts_run_for_candidate") != 0
            or gpu.get("launch_ready_only") is not True
        ):
            raise runner.RunnerGateError(
                "H_SCA_ORDER_CANDIDATE_AUTHORITY_SEMANTICS",
                repr({
                    "amendment": amendment.get("verdict"),
                    "source": source.get("verdict"),
                    "hlo": hlo.get("verdict"),
                    "ab": ab.get("verdict"),
                    "candidate": candidate.get("verdict"),
                    "candidate_row": candidate_row,
                }),
            )
        return {
            **prior,
            "prior_scalar_sixth_candidate_commit": PARENT_MODEL,
            "t_source_candidate_commit": CANDIDATE,
            "h_sca_order_amendment18": rows["amendment18"],
            "h_sca_order_source_oracle": rows["source_oracle"],
            "h_sca_order_hlo_audit": rows["hlo_audit"],
            "h_sca_order_complete_cpu_ab": rows["complete_cpu_ab"],
            "h_sca_order_candidate_proof": rows["candidate_proof"],
            "h_sca_order_candidate_commit": CANDIDATE,
            "h_sca_order_parent_model_commit": PARENT_MODEL,
            "earliest_causal_gate": "d03 step 200 / 00:20",
            "full_18h_authorized_only_after_step200_green": True,
        }

    def assert_h_sca_source_authority(
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
        tree = runner._git(runner.REPO_ROOT, "rev-parse", f"{CANDIDATE}^{{tree}}")
        if tree != CANDIDATE_TREE:
            raise runner.RunnerGateError("CANDIDATE_TREE", tree)
        ancestor = subprocess.run(
            (
                "git", "-C", str(runner.REPO_ROOT), "merge-base",
                "--is-ancestor", CANDIDATE, head,
            ),
            check=False,
        ).returncode == 0
        if not ancestor:
            raise runner.RunnerGateError(
                "CANDIDATE_ANCESTRY", f"{CANDIDATE} !<= {head}"
            )
        model_diff = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", CANDIDATE, "--", "src/gpuwrf"
        )
        if model_diff:
            raise runner.RunnerGateError("ACCEPTED_MODEL_CHANGED", model_diff)
        delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", CANDIDATE, head
        ).splitlines()
        allowed_exact = {
            "scripts/v0234_nested_frozen_wrf_boundary_window.py",
            "scripts/v0234_nested_h_sca_order_candidate_proof.py",
            "scripts/v0234_nested_h_sca_order_hlo_audit.py",
            "scripts/v0234_nested_h_sca_order_runner_profile.py",
            "scripts/v0234_nested_scalar_controls_cpu_ranking.py",
            "tests/test_v0234_nested_h_sca_order_proofs.py",
        }
        allowed_prefixes = (
            str(runner.SPRINT_DIR.relative_to(runner.REPO_ROOT)) + "/",
            str(runner.FINAL_NI_DIR.relative_to(runner.REPO_ROOT)) + "/",
        )
        unexpected = sorted(
            path for path in delta
            if path not in allowed_exact
            and not any(path.startswith(prefix) for prefix in allowed_prefixes)
        )
        if unexpected:
            raise runner.RunnerGateError("RUNNER_ONLY_DELTA", repr(unexpected))
        source_rows = {
            relative: runner._require_sha(
                runner.REPO_ROOT / relative, expected, "ACCEPTED_SOURCE_HASH"
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

    runner.assert_final_candidate_proof_authority = assert_h_sca_candidate_authority
    runner.assert_candidate_source_authority = assert_h_sca_source_authority
    runner._H_SCA_ORDER_PROFILE_APPLIED = True
