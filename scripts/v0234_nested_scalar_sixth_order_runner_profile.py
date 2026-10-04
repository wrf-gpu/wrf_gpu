"""Pre-import runner bindings for admitted 18d97595 scalar diff6 candidate."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping


CANDIDATE = "18d97595c59ca01840081f11109780c291dfcae8"
CANDIDATE_TREE = "ac570ee89be7d18dcc6eccfc5bcfa45c32d30dd6"
PARENT = "835d5b6f07c357d5296716df7f2c99bcbb66a10a"
PARENT_MODEL = "044783549697caf8c80d094f2b3e73f5ef340537"
PARTIAL_WIND = "2c13b73112d9877d603324d66b127ecad60bf7e3"
NAMESPACE = "nested_scalar_sixth_order_18d97595_full18h_discriminator1"
LOCK_LABEL = "v0234-nested-scalar-sixth-full18h"


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_SCALAR_SIXTH_PROFILE_APPLIED", False):
        return

    # Preserve and authenticate the full admitted 044 ancestry before layering
    # the one-file scalar candidate.  The theta profile performs no model import.
    from scripts.v0234_nested_theta_sixth_order_runner_profile import (
        apply_profile as apply_theta_profile,
    )

    apply_theta_profile(runner)
    old_candidate = runner.CANDIDATE_COMMIT
    old_tree = runner.CANDIDATE_TREE
    old_sources = dict(runner.ACCEPTED_SOURCE_HASHES)
    old_namespace = runner.FULL_REPLAY_NAMESPACE
    prior_authority = runner.assert_final_candidate_proof_authority

    final_dir = runner.FINAL_NI_DIR
    amendment16 = final_dir / "contract-amendment-16.json"
    source_oracle = final_dir / "nested-scalar-sixth-order-source-oracle.json"
    complete_ab = final_dir / "nested-scalar-sixth-order-complete-cpu-ab-proof.json"
    candidate_proof = final_dir / "nested-scalar-sixth-order-candidate-proof.json"
    proof_constants = {
        "amendment16": (
            amendment16,
            "4275c7dea50a5903b4b6f6274fcd878b5ebdfdf1f7401a5b8f81e6131271b110",
            "9ef825d9fefaf10e40c5ea5acfe05517f3cb57432639de79dbf42cafbc1ab748",
            False,
        ),
        "source_oracle": (
            source_oracle,
            "1d0b0c792c1a2be40f477ce60a17835729e0799444ba55dfa57085d0013d556a",
            "eb263213d60df012f5e4d000ba1a790a477c9a7dc8041abe82cd477c021a0d5d",
            True,
        ),
        "complete_cpu_ab": (
            complete_ab,
            "6307cf3d303055886568cf4edb6910dc71eb0f5502922a92aaebc0f5bc6275d6",
            "a21248ad8d1e6aa139db05b3028ee27a4069da4312619d70b5586a62b7d0b1d1",
            True,
        ),
        "candidate_proof": (
            candidate_proof,
            "2d23c2155e1df4fd8f56d6297fa35ed6a8115e13d308072967008540b45b7a26",
            "7d912a3429af62957dc965563148fcf2bf0348fb57578c79538d2cc12c3e6065",
            True,
        ),
    }

    runner.SCHEMA = "gpuwrf.v0234.nested-scalar-sixth-order-full18h.v1"
    runner.CPU_PROOF_SCHEMA = (
        "gpuwrf.v0234.nested-scalar-sixth-order-runner-cpu-audit.v1"
    )
    runner.AUDIT_ADMISSION = "READY_FOR_NESTED_SCALAR_SIXTH_GPU_REPLAY"
    runner.CANDIDATE_COMMIT = CANDIDATE
    runner.CANDIDATE_TREE = CANDIDATE_TREE
    runner.FULL_REPLAY_NAMESPACE = NAMESPACE
    runner.LOCK_LABEL = LOCK_LABEL
    runner.LAUNCH_COMMAND = runner.SPRINT_DIR / (
        "nested-scalar-sixth-order-full18h-exact-launch-command.txt"
    )
    runner.RUNNER_CPU_AUDIT = runner.SPRINT_DIR / (
        "nested-scalar-sixth-order-full18h-runner-cpu-proof.json"
    )
    runner.EARLY_CAUSAL_SCALAR_PARENT = runner.LINEAGE_WORK_DIR / (
        "nested_theta_sixth_order_04478354_full18h_discriminator1/output/"
        "wrfout_d03_2025-03-01_00:20:00"
    )
    runner.EARLY_CAUSAL_SCALAR_PARENT_SHA256 = (
        "a19d793c0fb88d08bd07d1ef55daf8ba51f5d53428461d804338c30a24aff871"
    )
    runner.EARLY_CAUSAL_RING1_BASELINE = {
        "T": {
            "prior_vs_cpu": 0.07439330567219023,
            "prior_vs_retry20": 0.05252442272211308,
            "partial_vs_cpu": 0.07753516846062372,
            "partial_vs_retry20": 0.0588941045970705,
            "scalar_vs_cpu": 0.07895907309242274,
            "scalar_vs_retry20": 0.06330030746547205,
        },
        "U": {
            "prior_vs_cpu": 0.2857256114890511,
            "prior_vs_retry20": 0.27772695352886917,
            "partial_vs_cpu": 0.20247207650654608,
            "partial_vs_retry20": 0.19592398176235976,
            "scalar_vs_cpu": 0.19937246800958175,
            "scalar_vs_retry20": 0.19258729494677282,
        },
    }
    runner.ACCEPTED_SOURCE_HASHES = {
        **old_sources,
        "src/gpuwrf/dynamics/explicit_diffusion.py": (
            "8fc7726b2de017cf5ff4896aa85eed67e419ad9fd011b00addb50568279d2f61"
        ),
        "src/gpuwrf/runtime/operational_mode.py": (
            "fdda05e0ceb977c9a7faafd026beb3f83afea734844dd448d8546cccbf8c2b5e"
        ),
    }
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        **runner.CANDIDATE_CLEAN_AUTHORITY,
        "scalar_sixth_candidate_commit": CANDIDATE,
        "scalar_sixth_candidate_proof_sha256": (
            "7d912a3429af62957dc965563148fcf2bf0348fb57578c79538d2cc12c3e6065"
        ),
        "scalar_sixth_direct_dry_theta_source_changed": False,
        "scalar_sixth_cpu_tests": "141 passed",
    }

    def _read_new_proof(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        path, file_hash, payload_hash, embedded = proof_constants[name]
        if embedded:
            return runner._authenticated_canonical_json(
                path, file_hash, payload_hash, f"SCALAR_SIXTH_{name.upper()}"
            )
        payload, row = runner.read_authenticated_json(
            path, file_hash, f"SCALAR_SIXTH_{name.upper()}_FILE"
        )
        observed = runner.canonical_digest(payload)
        if observed != payload_hash:
            raise runner.RunnerGateError(
                f"SCALAR_SIXTH_{name.upper()}_CANONICAL",
                f"expected={payload_hash} observed={observed}",
            )
        return payload, {**row, "canonical_payload_sha256": observed}

    def assert_scalar_candidate_authority() -> dict[str, Any]:
        prior = prior_authority()
        payloads = {}
        rows = {}
        for name in proof_constants:
            payloads[name], rows[name] = _read_new_proof(name)
        amendment = payloads["amendment16"]
        source = payloads["source_oracle"]
        ab = payloads["complete_cpu_ab"]
        candidate = payloads["candidate_proof"]
        source_checks = source.get("checks") or {}
        ab_checks = ab.get("checks") or {}
        candidate_checks = candidate.get("checks") or {}
        candidate_row = candidate.get("candidate") or {}
        causal = candidate.get("causal_scope") or {}
        if (
            amendment.get("verdict")
            != "NESTED_RK1_MOIST_SCALAR_SIXTH_ORDER_AMENDED_FAIL_CLOSED"
            or source.get("verdict")
            != "NESTED_SCALAR_SIXTH_ORDER_SOURCE_ORACLE_GREEN"
            or not source_checks
            or not all(source_checks.values())
            or ab.get("verdict")
            != "NESTED_SCALAR_SIXTH_ORDER_COMPLETE_CPU_AB_GREEN"
            or not ab_checks
            or not all(ab_checks.values())
            or candidate.get("schema")
            != "gpuwrf.v0234.nested-scalar-sixth-order-candidate.v1"
            or candidate.get("verdict")
            != "NESTED_SCALAR_SIXTH_ORDER_GPU_STEP200_ADMITTED"
            or not candidate_checks
            or not all(candidate_checks.values())
            or candidate_row.get("commit") != CANDIDATE
            or candidate_row.get("tree") != CANDIDATE_TREE
            or candidate_row.get("parent_commit") != PARENT
            or candidate_row.get("parent_model_commit") != PARENT_MODEL
            or candidate_row.get("partial_wind_commit") != PARTIAL_WIND
            or causal.get("direct_dry_momentum_or_theta_source_changed") is not False
            or causal.get("partial_wind_mechanism_retained") is not True
            or causal.get("v10_link_claimed") is not False
            or candidate.get("complete_cpu_ab", {}).get("leaf_count") != 106
            or candidate.get("complete_cpu_ab", {}).get("all_finite") is not True
            or candidate.get("complete_cpu_ab", {}).get(
                "forbidden_callback_targets"
            )
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
                "SCALAR_SIXTH_CANDIDATE_AUTHORITY_SEMANTICS",
                repr(
                    {
                        "amendment": amendment.get("verdict"),
                        "source": source.get("verdict"),
                        "ab": ab.get("verdict"),
                        "candidate": candidate.get("verdict"),
                        "candidate_row": candidate_row,
                    }
                ),
            )
        return {
            **prior,
            "prior_theta_sixth_candidate_commit": old_candidate,
            "t_source_candidate_commit": CANDIDATE,
            "scalar_sixth_amendment16": rows["amendment16"],
            "scalar_sixth_source_oracle": rows["source_oracle"],
            "scalar_sixth_complete_cpu_ab": rows["complete_cpu_ab"],
            "scalar_sixth_candidate_proof": rows["candidate_proof"],
            "scalar_sixth_candidate_commit": CANDIDATE,
            "scalar_sixth_parent_model_commit": PARENT_MODEL,
            "earliest_causal_gate": "d03 step 200 / 00:20",
            "full_18h_authorized_only_after_step200_green": True,
        }

    def assert_scalar_source_authority(
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
            runner.REPO_ROOT, "diff", "--name-only", CANDIDATE,
            "--", "src/gpuwrf",
        )
        if model_diff:
            raise runner.RunnerGateError("ACCEPTED_MODEL_CHANGED", model_diff)
        delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", CANDIDATE, head
        ).splitlines()
        allowed_exact = {
            "scripts/v0234_nested_frozen_wrf_boundary_window.py",
            "scripts/v0234_nested_scalar_sixth_order_cpu_ab.py",
            "scripts/v0234_nested_scalar_sixth_order_candidate_proof.py",
            "scripts/v0234_nested_scalar_sixth_order_runner_profile.py",
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

    runner.assert_final_candidate_proof_authority = assert_scalar_candidate_authority
    runner.assert_candidate_source_authority = assert_scalar_source_authority
    runner._SCALAR_SIXTH_PROFILE_APPLIED = True
