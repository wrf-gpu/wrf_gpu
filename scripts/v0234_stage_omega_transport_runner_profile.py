"""Pre-import runner bindings for the admitted stage-omega transport candidate."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping


CANDIDATE = "470e6111d516479bed4bc0c3b2be1007bb082afd"
CANDIDATE_TREE = "bb0d7c9a4fe7befdf3120bdfdabde11db985682a"
PARENT = "0acd5dab23828a35e110b2aa226b340ad43ee728"
PARENT_MODEL = "395fb800df0bb619462db8e312f734ca0c538161"
NAMESPACE = "nested_stage_omega_transport_470e6111_full18h_toolingrepair2"
FIRST_GPU_NAMESPACE = "nested_stage_omega_transport_470e6111_full18h_discriminator1"
FIRST_TOOLING_REPAIR_NAMESPACE = (
    "nested_stage_omega_transport_470e6111_full18h_toolingrepair1"
)
LOCK_LABEL = "v0234-stage-omega-transport-full18h"
POST_FABLE_BASE = "dc771bab107ba2a9dbc473687cf444c5e07a5bea"
POST_FABLE_BASE_TREE = "b84d1609a69c5b69bd0dd51968fed535f5097178"
PRESERVED_POST_FABLE_NAMESPACE = (
    "nested_stage_omega_transport_470e6111_post_fable_corner_window_gpt56"
)
POST_FABLE_NAMESPACE = f"{PRESERVED_POST_FABLE_NAMESPACE}_resource_retry1"
POST_FABLE_LOCK_LABEL = "v0234-post-fable-fulltree-resource-retry1"
RESOURCE_RESTART_CRITIC_SCHEMA = (
    "gpuwrf.v0234.post-fable-resource-restart-critic.v1"
)
RESOURCE_RESTART_CRITIC_VERDICT = "KIMI_RESOURCE_RESTART_CRITIC_ACCEPT"
RESOURCE_PREEMPT_PROOF_SHA256 = (
    "481bc1ea1120a3efde478c888cbd09f7ca751a88c2b913b8b60803272c78e4e1"
)
RESOURCE_PREEMPT_PROOF_PAYLOAD_SHA256 = (
    "2459ea780ae9bc97ec1b47709290687b0bca807d2448722ab569049148456dd6"
)
KNOWN_V10_NAMESPACE = "nested_stage_omega_transport_470e6111_full18h_toolingrepair2"
KNOWN_V10_ARTIFACT_SHA256 = (
    "38683937ed6be6eb96bb71c3bb7c5e8976d8d91c525021c184712c24eb169448"
)
KNOWN_V10_PAYLOAD_SHA256 = (
    "82ddb7bd13d5a1c6e91827ac15169d6de8822f0675e23ee369e4c93970e925e1"
)
ARM_S_PROOF_SHA256 = (
    "3986508430fe0bb40546abaecf6f978287f6eedef1f702fd6a6b82f15196befb"
)
ARM_S_PROOF_PAYLOAD_SHA256 = (
    "166fb66cb74e76b4d0759fa573ad6478ace030b36839725b09cfafbcdaa96298"
)


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_STAGE_OMEGA_TRANSPORT_PROFILE_APPLIED", False):
        return

    from scripts.v0234_nested_h_sca_order_runner_profile import (
        apply_profile as apply_h_sca_profile,
    )

    apply_h_sca_profile(runner)
    prior_authority = runner.assert_final_candidate_proof_authority
    old_sources = dict(runner.ACCEPTED_SOURCE_HASHES)

    final_dir = runner.FINAL_NI_DIR
    amendment37 = final_dir / "contract-amendment-37.json"
    amendment38 = final_dir / "contract-amendment-38.json"
    amendment39 = final_dir / "contract-amendment-39.json"
    amendment40 = final_dir / "contract-amendment-40.json"
    amendment41 = final_dir / "contract-amendment-41.json"
    source_oracle = final_dir / "stage-omega-transport-source-oracle.json"
    complete_ab = final_dir / "stage-omega-transport-complete-cpu-ab-proof.json"
    first_gpu_run = runner.LINEAGE_WORK_DIR / FIRST_GPU_NAMESPACE
    first_gpu_pair = first_gpu_run / "frame-pairs/d03-step-00200.json"
    first_gpu_blocker = first_gpu_run / "full-run-blocker.json"
    first_gpu_failure = first_gpu_run / "failure/failure-proof.json"
    tooling1_run = runner.LINEAGE_WORK_DIR / FIRST_TOOLING_REPAIR_NAMESPACE
    tooling1_pair = tooling1_run / "frame-pairs/d03-step-00200.json"
    tooling1_blocker = tooling1_run / "full-run-blocker.json"
    tooling1_failure = tooling1_run / "failure/failure-proof.json"
    post_fable_sprint = runner.REPO_ROOT / (
        ".agent/sprints/2026-07-17-v0234-post-fable-corner-window"
    )
    known_v10_artifact = runner.LINEAGE_WORK_DIR / (
        f"{KNOWN_V10_NAMESPACE}/frame-pairs/d03-step-09000.json"
    )
    arm_s_proof = runner.LINEAGE_WORK_DIR / (
        "post_fable_corner_window_gpt56/armS_gpu/arm-S-proof.json"
    )
    profile_source = runner.REPO_ROOT / (
        "scripts/v0234_stage_omega_transport_runner_profile.py"
    )
    resource_preempt_proof = post_fable_sprint / "full-tree-preempted-proof.json"
    preserved_run = runner.LINEAGE_WORK_DIR / PRESERVED_POST_FABLE_NAMESPACE
    resource_retry_run = runner.LINEAGE_WORK_DIR / POST_FABLE_NAMESPACE

    def assert_resource_restart_authority() -> dict[str, Any]:
        proof, proof_row = runner.read_authenticated_json(
            resource_preempt_proof,
            RESOURCE_PREEMPT_PROOF_SHA256,
            "RESOURCE_RESTART_PREEMPT_PROOF",
        )
        unsigned = dict(proof)
        embedded = unsigned.pop("proof_sha256", None)
        observed = runner.canonical_digest(unsigned)
        gpu_run = proof.get("gpu_run") or {}
        preemption = proof.get("preemption") or {}
        release = proof.get("release_evidence") or {}
        retained = proof.get("retained_namespace") or {}
        immutable = proof.get("immutable_code") or {}
        artifact_rows = retained.get("artifacts") or []
        expected_paths = {
            str(path.resolve())
            for path in (
                preserved_run / "ordinary-one-step-lowered-hlo.json",
                preserved_run / "failure/last-healthy-d01-step-0.pkl",
                preserved_run / "failure/.first-failed-d01-step-0.pkl.tmp-1743996",
            )
        }
        proof_paths = {
            str(Path(str(row.get("path", ""))).resolve())
            for row in artifact_rows
            if isinstance(row, dict)
        }
        observed_paths = {
            str(path.resolve()) for path in preserved_run.rglob("*") if path.is_file()
        }
        if (
            proof.get("schema")
            != "gpuwrf.v0234.post-fable-fulltree-preempted.v1"
            or proof.get("verdict")
            != "POST_FABLE_FULLTREE_PREEMPTED_GPU_RELEASED"
            or proof.get("current_contract_state")
            != "POST_FABLE_CORNER_WINDOW_RESOURCE_BLOCKED"
            or embedded != RESOURCE_PREEMPT_PROOF_PAYLOAD_SHA256
            or observed != RESOURCE_PREEMPT_PROOF_PAYLOAD_SHA256
            or gpu_run.get("scientific_result_produced") is not False
            or gpu_run.get("scientific_checkpoint_produced") is not False
            or gpu_run.get("known_1500_v10_reached") is not False
            or gpu_run.get("terminal_output_counts_produced") is not False
            or preemption.get("restart_forbidden_until_fresh_release_gpu") is not True
            or release.get("holder_sidecar_absent") is not True
            or release.get("gpu_work_restarted") is not False
            or immutable.get("model_bytes_changed") is not False
            or immutable.get("src_gpuwrf_tree")
            != "835dcc29bf316c0715b41a72e064985e9cf099df"
            or retained.get("namespace") != str(preserved_run.resolve())
            or retained.get("stable_across_two_post_release_hash_passes") is not True
            or retained.get("terminal_proof_absent") is not True
            or retained.get("frame_pairs_absent") is not True
            or len(artifact_rows) != 3
            or proof_paths != expected_paths
            or observed_paths != expected_paths
            or preserved_run.is_symlink()
            or resource_retry_run.exists()
            or resource_retry_run.is_symlink()
        ):
            raise runner.RunnerGateError(
                "RESOURCE_RESTART_PREEMPT_SEMANTICS",
                repr({
                    "embedded": embedded,
                    "observed": observed,
                    "proof_paths": sorted(proof_paths),
                    "observed_paths": sorted(observed_paths),
                    "retry_exists": resource_retry_run.exists(),
                }),
            )
        authenticated_artifacts = []
        for artifact in artifact_rows:
            path = Path(str(artifact["path"]))
            authority = runner.stable_file_authority(
                path, "RESOURCE_RESTART_PRESERVED_ARTIFACT"
            )
            if (
                authority["sha256"] != artifact.get("sha256")
                or authority["bytes"] != artifact.get("bytes")
            ):
                raise runner.RunnerGateError(
                    "RESOURCE_RESTART_PRESERVED_ARTIFACT_IDENTITY",
                    repr({"proof": artifact, "observed": authority}),
                )
            authenticated_artifacts.append(authority)
        return {
            "required": True,
            "preempt_proof": {
                **proof_row,
                "canonical_payload_sha256": observed,
                "verdict": proof["verdict"],
            },
            "prior_scientific_result_produced": False,
            "preserved_namespace": str(preserved_run.resolve()),
            "preserved_artifacts": authenticated_artifacts,
            "preserved_file_set_exact": True,
            "resource_retry_number": 1,
            "resource_retry_namespace": str(resource_retry_run.resolve()),
            "resource_retry_namespace_absent": True,
        }

    runner.SCHEMA = "gpuwrf.v0234.post-fable-late-ni-fulltree.v1"
    runner.CPU_PROOF_SCHEMA = (
        "gpuwrf.v0234.post-fable-late-ni-runner-cpu-audit.v1"
    )
    runner.AUDIT_ADMISSION = "READY_FOR_POST_FABLE_LATE_NI_GPU_REPLAY"
    runner.CANDIDATE_COMMIT = CANDIDATE
    runner.CANDIDATE_TREE = CANDIDATE_TREE
    runner.FULL_REPLAY_NAMESPACE = POST_FABLE_NAMESPACE
    runner.LOCK_LABEL = POST_FABLE_LOCK_LABEL
    runner.LAUNCH_COMMAND = post_fable_sprint / "full-tree-gpu-exact-launch-command.sh"
    runner.RUNNER_CPU_AUDIT = (
        post_fable_sprint / "resource-retry-runner-cpu-proof.json"
    )
    runner.REQUIRE_KNOWN_1500_V10_RECORD = True
    runner.REQUIRE_TOOLING_CRITIC_ACCEPT = True
    runner.TOOLING_CRITIC_SCHEMA = RESOURCE_RESTART_CRITIC_SCHEMA
    runner.TOOLING_CRITIC_VERDICT = RESOURCE_RESTART_CRITIC_VERDICT
    runner.TOOLING_CRITIC_PROFILE_SOURCE = profile_source
    runner.TOOLING_CRITIC_REVIEWED_PATHS = (profile_source,)
    runner.TOOLING_CRITIC_REQUIRED_PAYLOAD = {
        "resource_restart_preempt_proof_file_sha256": (
            RESOURCE_PREEMPT_PROOF_SHA256
        ),
        "resource_restart_preempt_proof_canonical_sha256": (
            RESOURCE_PREEMPT_PROOF_PAYLOAD_SHA256
        ),
        "preserved_preempted_namespace": str(preserved_run.resolve()),
        "resource_retry_namespace": str(resource_retry_run.resolve()),
        "resource_retry_number": 1,
        "prior_scientific_result_produced": False,
        "resource_retry_gpu_replay_admitted": True,
    }
    runner.TOOLING_CRITIC_AUTHORITY_HOOK = assert_resource_restart_authority
    runner.KNOWN_1500_V10_ARTIFACT = known_v10_artifact
    runner.KNOWN_1500_V10_ARTIFACT_SHA256 = KNOWN_V10_ARTIFACT_SHA256
    runner.KNOWN_1500_V10_PAYLOAD_SHA256 = KNOWN_V10_PAYLOAD_SHA256
    runner.EARLY_CAUSAL_SCALAR_PARENT = runner.LINEAGE_WORK_DIR / (
        "nested_h_sca_order_395fb800_full18h_discriminator1/output/"
        "wrfout_d03_2025-03-01_00:20:00"
    )
    runner.EARLY_CAUSAL_SCALAR_PARENT_SHA256 = (
        "3235053155927d04ee11971e34ddce71ba81572bf9f1b94a043e59c14d0c6c10"
    )
    runner.EARLY_CAUSAL_RING1_BASELINE = {
        "T": {
            "prior_vs_cpu": 0.07439330567219023,
            "prior_vs_retry20": 0.05252442272211308,
            "partial_vs_cpu": 0.07753516846062372,
            "partial_vs_retry20": 0.0588941045970705,
            "scalar_vs_cpu": 0.07906161911226972,
            "scalar_vs_retry20": 0.06323878503953201,
        },
        "U": {
            "prior_vs_cpu": 0.2857256114890511,
            "prior_vs_retry20": 0.27772695352886917,
            "partial_vs_cpu": 0.20247207650654608,
            "partial_vs_retry20": 0.19592398176235976,
            "scalar_vs_cpu": 0.1992207926622843,
            "scalar_vs_retry20": 0.19252577923290928,
        },
    }
    runner.ACCEPTED_SOURCE_HASHES = {
        **old_sources,
        "src/gpuwrf/runtime/operational_mode.py": (
            "cb431d035e1d1a1d584e56585499747ba679f7e4e58fe659a8082ff27b321a4c"
        ),
    }
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        **runner.CANDIDATE_CLEAN_AUTHORITY,
        "stage_omega_transport_candidate_commit": CANDIDATE,
        "stage_omega_transport_candidate_tree": CANDIDATE_TREE,
        "stage_omega_transport_source_oracle_sha256": (
            "346b281b0ab5216bd7535c216005410061b3a9f0c575596eedfb4f6da69bec71"
        ),
        "stage_omega_transport_cpu_ab_sha256": (
            "d93ac8b3035212591502361f353d2b37133ceb2b89f07d76100ac5c74fb8e690"
        ),
        "stage_omega_transport_cpu_tests": "13 passed, 1 skipped",
        "stage_omega_transport_v10_link_claimed": False,
        "waiting_source_gate_repair_payload_sha256": (
            "e767a780106c01033ca0e0d8f2827c2c6f29c3981cad03517bb365e0f1d6254e"
        ),
        "step200_literal_recompute_repair_payload_sha256": (
            "504a3eb79dd542ce974aa4089b7a499461bb7e25c9da415fc0d90daac1aedddd"
        ),
        "process_marker_boundary_repair_payload_sha256": (
            "19231837b2551a28596e37ff4d0e162f775d4f8adff165d982e5fc2db9690151"
        ),
        "post_fable_base_commit": POST_FABLE_BASE,
        "post_fable_base_tree": POST_FABLE_BASE_TREE,
        "post_fable_arm_s_proof_sha256": ARM_S_PROOF_PAYLOAD_SHA256,
        "known_1500_v10_artifact_sha256": KNOWN_V10_PAYLOAD_SHA256,
        "known_1500_v10_record_is_not_waiver": True,
        "resource_restart_preempt_proof_sha256": (
            RESOURCE_PREEMPT_PROOF_PAYLOAD_SHA256
        ),
        "resource_restart_prior_scientific_result_produced": False,
        "resource_restart_retry_number": 1,
        "resource_restart_namespace": str(resource_retry_run.resolve()),
        "resource_restart_requires_focused_kimi_accept": True,
    }

    def _read_plain_contract(
        path: Any, file_hash: str, payload_hash: str, code: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, row = runner.read_authenticated_json(path, file_hash, code)
        observed = runner.canonical_digest(payload)
        if observed != payload_hash:
            raise runner.RunnerGateError(
                f"{code}_CANONICAL", f"expected={payload_hash} observed={observed}"
            )
        return payload, {**row, "canonical_payload_sha256": observed}

    def assert_stage_omega_candidate_authority() -> dict[str, Any]:
        prior = prior_authority()
        contract37, contract37_row = _read_plain_contract(
            amendment37,
            "8897fee95f66251a0a443ed9f0c52bfd8a5d549eab753e96e4a2f314dbf5f115",
            "0a0ef0a21f789c3b6ad142ad06d7e7c935af8e916e71d3cdb7381133541f2dad",
            "STAGE_OMEGA_AMENDMENT37",
        )
        contract38, contract38_row = _read_plain_contract(
            amendment38,
            "e6e174550b20f81557b13091f9ac758a98a9ee6d68bfff6b1e85d65d4103ae8b",
            "ba77e257cd334e57c815bcc9cd2aaa9ae2eafadcc32ef34edb008cba500330c8",
            "STAGE_OMEGA_AMENDMENT38",
        )
        contract39, contract39_row = _read_plain_contract(
            amendment39,
            "0b7d4076c15b6f0666cbd27c6edb460184bc58ea1af6d4cc741a24435e27e77e",
            "e767a780106c01033ca0e0d8f2827c2c6f29c3981cad03517bb365e0f1d6254e",
            "STAGE_OMEGA_AMENDMENT39",
        )
        contract40, contract40_row = _read_plain_contract(
            amendment40,
            "bb15599a6087795296582a84df5df52d3356bd3bdb2dac636f390aac9dd8a2d2",
            "504a3eb79dd542ce974aa4089b7a499461bb7e25c9da415fc0d90daac1aedddd",
            "STAGE_OMEGA_AMENDMENT40",
        )
        contract41, contract41_row = _read_plain_contract(
            amendment41,
            "1da3a9b3cf2c98a59227ab782ef75bab483e732cac9ff62acc62244ea594fc85",
            "19231837b2551a28596e37ff4d0e162f775d4f8adff165d982e5fc2db9690151",
            "STAGE_OMEGA_AMENDMENT41",
        )
        pair, pair_row = runner._authenticated_canonical_json(
            first_gpu_pair,
            "dd56b40a20d2655bb3e9875e926bb05b45164d2b14cc5e794b799732f7eed1fe",
            "7d8498587f7f0d8f41cb6bd5b64b243b1757e8c639df41b3d747e89292178848",
            "STAGE_OMEGA_FIRST_GPU_PAIR",
        )
        blocker40, blocker40_row = runner._authenticated_canonical_json(
            first_gpu_blocker,
            "c596db97d6b48a207e54c2b965ab0f52392ddd325bf73b76ede6c26478ebccc0",
            "d062e812aded6dd3b9119095ad172019fcd8517c1895602b58e9d44e2953a7ba",
            "STAGE_OMEGA_FIRST_GPU_BLOCKER",
        )
        failure40, failure40_row = runner._authenticated_canonical_json(
            first_gpu_failure,
            "a6bbd58a158ce5fe0d5fdf14d4603dfe5bc54559c86740b730ff8411a02dc53e",
            "af50005bbc0ac0d2f565209907031ee6acc0e9da81a1ba24c53c09a5b36a2257",
            "STAGE_OMEGA_FIRST_GPU_FAILURE",
        )
        tooling1_pair_payload, tooling1_pair_row = runner._authenticated_canonical_json(
            tooling1_pair,
            "fe851239f2369f260f97dc821c9fdc24b44578b0409c81fa9f9ac3e1cfdcf72c",
            "d9c6441e28a5a4c134a81980d8b0dcaefabc626ec2ebf74ce06a95233468a81c",
            "STAGE_OMEGA_TOOLING1_PAIR",
        )
        tooling1_blocker_payload, tooling1_blocker_row = (
            runner._authenticated_canonical_json(
                tooling1_blocker,
                "c89472c231751dc37d1203222a1e532964c9aafd53b6414ba932885474c1a0d5",
                "a249776f1e3e13e9ca98268f2642cac5bb728a1bbed68eef6a5080dd5f8124f4",
                "STAGE_OMEGA_TOOLING1_BLOCKER",
            )
        )
        tooling1_failure_payload, tooling1_failure_row = (
            runner._authenticated_canonical_json(
                tooling1_failure,
                "dd63e16073d5d1acbb5c104e292c9504daefe57f15d16caaf33d7a93369faf42",
                "810814528f9d7c8f86ab1516b0304adb31b9e13c2095e3a7f7fac1851fa94a1d",
                "STAGE_OMEGA_TOOLING1_FAILURE",
            )
        )
        source, source_row = runner._authenticated_canonical_json(
            source_oracle,
            "8dc8136b4d358dd2b81d8c248abf63b9ab79b3125df65762bb8f66b8e2c74586",
            "346b281b0ab5216bd7535c216005410061b3a9f0c575596eedfb4f6da69bec71",
            "STAGE_OMEGA_SOURCE_ORACLE",
        )
        ab, ab_row = runner._authenticated_canonical_json(
            complete_ab,
            "ac3c9feb0a0ecded2d605f007940b9ad2c51c0ce77743a17e0a922c74d3721d0",
            "d93ac8b3035212591502361f353d2b37133ceb2b89f07d76100ac5c74fb8e690",
            "STAGE_OMEGA_COMPLETE_CPU_AB",
        )
        arm_s, arm_s_row = runner._authenticated_canonical_json(
            arm_s_proof,
            ARM_S_PROOF_SHA256,
            ARM_S_PROOF_PAYLOAD_SHA256,
            "POST_FABLE_ARM_S_GPU_PROOF",
        )
        arm_rows = {row["step"]: row for row in arm_s.get("rows", [])}
        arm_checkpoints = (9016, 9042, 9075)
        if (
            arm_s.get("schema")
            != "gpuwrf.v0234.final-holistic-late-window-corner-discriminator.v1"
            or arm_s.get("verdict") != "WINDOW_COMPLETED_BOUNDED"
            or arm_s.get("arm") != "S"
            or arm_s.get("backend") != "gpu"
            or arm_s.get("requested_backend") != "cuda"
            or arm_s.get("start_step") != 9001
            or arm_s.get("requested_steps") != 405
            or arm_s.get("completed_steps") != 405
            or arm_s.get("head_commit") != POST_FABLE_BASE
            or arm_s.get("head_src_tree")
            != "835dcc29bf316c0715b41a72e064985e9cf099df"
            or arm_s.get("model_bytes_changed") is not False
            or arm_s.get("hot_loop_full_carry_transfers") != 0
            or arm_s.get("first_red_step") is not None
            or arm_s.get("first_red_reason") is not None
            or arm_s.get("input_sha256")
            != "1b1521090539cc38a5660712ff1fe57a4251c3b790f7d619a176a8f32037d055"
            or len(arm_s.get("rows", [])) != 405
            or set(arm_rows) != set(range(9001, 9406))
            or any(row.get("nonfinite_total") != 0 for row in arm_rows.values())
            or any(row.get("corner_nonfinite") != 0 for row in arm_rows.values())
            or any(
                row.get("corner_mu_maxabs", float("inf")) >= 5.0e4
                for row in arm_rows.values()
            )
            or any(
                (arm_rows[step].get("cpu_prefix_comparison") or {}).get("passed")
                is not True
                for step in arm_checkpoints
            )
            or arm_rows[9314].get("corner_mu_maxabs", float("inf")) >= 5.0e4
            or arm_rows[9405].get("corner_mu_maxabs", float("inf")) >= 5.0e4
            or (arm_s.get("final_carry") or {}).get("sha256")
            != "aadc7cd7733f1641f8814b4381842e011be73768cb46905671a2f037a86adf9b"
            or (arm_s.get("lock_v2") or {}).get("commit")
            != "8152309aff1e85e1052d44d549a5a5409e710bdd"
            or (arm_s.get("lock_v2") or {}).get("intent")
            != "production-preemptible"
            or (arm_s.get("lock_v2") or {}).get("live_verified") is not True
        ):
            raise runner.RunnerGateError(
                "POST_FABLE_ARM_S_AUTHORITY_SEMANTICS", str(arm_s_proof)
            )
        sealed = contract38.get("sealed_candidate") or {}
        binding = contract38.get("runner_binding") or {}
        repair = contract39.get("authorized_runner_only_repair") or {}
        blocker = contract39.get("zero_model_blocker") or {}
        tooling_repair40 = contract40.get("authorized_runner_only_repair") or {}
        exact_blocker = contract40.get("exact_runner_blocker") or {}
        first_gpu = contract40.get("first_gpu_arm") or {}
        tooling1_arm = contract41.get("toolingrepair1_arm") or {}
        process_cause = contract41.get("exact_runner_cause") or {}
        tooling_repair41 = contract41.get("authorized_runner_only_repair") or {}
        source_checks = source.get("checks") or {}
        ab_checks = ab.get("checks") or {}
        if (
            contract37.get("verdict")
            != "STAGE_OMEGA_TRANSPORT_OWNERSHIP_CPU_CANDIDATE_AMENDED_FAIL_CLOSED"
            or contract38.get("verdict")
            != "STAGE_OMEGA_TRANSPORT_GPU_STEP200_AND_DIRECT_FULL18H_BOUND_FAIL_CLOSED"
            or contract39.get("verdict")
            != "WAITING_SOURCE_PREEMPT_GATE_REPAIR_AUTHORIZED_FAIL_CLOSED"
            or contract40.get("verdict")
            != "STAGE_OMEGA_STEP200_LITERAL_RECOMPUTE_REPAIR_AUTHORIZED_FAIL_CLOSED"
            or contract41.get("verdict")
            != "STAGE_OMEGA_PROCESS_MARKER_BOUNDARY_REPAIR_AUTHORIZED_FAIL_CLOSED"
            or sealed.get("commit") != CANDIDATE
            or sealed.get("tree") != CANDIDATE_TREE
            or sealed.get("parent_commit") != PARENT
            or sealed.get("parent_model_commit") != PARENT_MODEL
            or sealed.get("only_model_file")
            != "src/gpuwrf/runtime/operational_mode.py"
            or sealed.get("model_file_sha256")
            != "cb431d035e1d1a1d584e56585499747ba679f7e4e58fe659a8082ff27b321a4c"
            or sealed.get("new_carry_or_result_leaves") != 0
            or sealed.get("new_loop_host_device_transfers") != 0
            or sealed.get("observer_or_callback_added") is not False
            or binding.get("namespace") != FIRST_GPU_NAMESPACE
            or binding.get("first_gate")
            != "fresh canonical full-history d03 step 200 / 00:20"
            or binding.get("manager_pause") is not False
            or blocker.get("classification") != "OVERBROAD_WAITING_SOURCE_POLICY"
            or blocker.get("compile_calls") != 0
            or blocker.get("dispatch_calls") != 0
            or blocker.get("nightly_status") != "waiting_source"
            or repair.get("candidate_commit") != CANDIDATE
            or repair.get("model_file_edits") != 0
            or repair.get("gpu_query_added") is not False
            or first_gpu.get("namespace") != FIRST_GPU_NAMESPACE
            or first_gpu.get("classification")
            != "SCIENTIFIC_STEP200_GREEN_RUNNER_BASELINE_LITERAL_EXACTNESS_RED"
            or first_gpu.get("directional_science_gates_green") is not True
            or first_gpu.get("finite_identity_green") is not True
            or first_gpu.get("step200_carry_leaf_count") != 106
            or first_gpu.get("step200_carry_all_finite") is not True
            or exact_blocker.get("field") != "U"
            or exact_blocker.get("quantity") != "scalar_vs_retry20"
            or exact_blocker.get("stored_literal") != 0.19252577923290926
            or exact_blocker.get("sha_bound_recompute") != 0.19252577923290928
            or exact_blocker.get("only_failed_gate") != "baseline_authenticated"
            or tooling_repair40.get("literal_after_recompute")
            != runner.EARLY_CAUSAL_RING1_BASELINE["U"]["scalar_vs_retry20"]
            or tooling_repair40.get("fresh_namespace")
            != FIRST_TOOLING_REPAIR_NAMESPACE
            or tooling_repair40.get("model_file_edits") != 0
            or tooling_repair40.get("threshold_changes") != 0
            or tooling_repair40.get("science_hypothesis_changes") != 0
            or tooling_repair40.get("additional_critic") is not False
            or pair.get("passed") is not False
            or pair.get("finite_identity", {}).get("passed") is not True
            or pair.get("early_causal_00_20", {}).get("fields", {}).get("U", {}).get(
                "gates", {}
            ).get("baseline_authenticated") is not False
            or blocker40.get("failure_code") != "INCREMENTAL_FRAME_PAIR"
            or blocker40.get("model_or_numerical_edit") is not False
            or failure40.get("failure", {}).get("code") != "INCREMENTAL_FRAME_PAIR"
            or failure40.get("first_failed", {}).get("manifest", {}).get("leaf_count")
            != 106
            or failure40.get("first_failed", {}).get("manifest", {}).get(
                "floating_nonfinite_count"
            ) != 0
            or tooling1_arm.get("namespace") != FIRST_TOOLING_REPAIR_NAMESPACE
            or tooling1_arm.get("classification")
            != "RUNNER_ONLY_PRODUCTION_PROCESS_SUBSTRING_FALSE_POSITIVE"
            or tooling1_arm.get("step200_science_green") is not True
            or tooling1_arm.get("last_paired_d03_step") != 3000
            or tooling1_arm.get("paired_frame_counts_before_stop")
            != {"d01": 5, "d02": 6, "d03": 16}
            or process_cause.get("gate") != "PREEMPT_PRODUCTION_GPU_ACTIVE"
            or process_cause.get("detected_cpu_only_call")
            != "alisios.pipeline.nightly_profiles.wn2_remote_preflight"
            or process_cause.get("raw_marker") != "ncu"
            or process_cause.get("false_match_container") != "concurrent.futures"
            or tooling_repair41.get("fresh_namespace") != NAMESPACE
            or tooling_repair41.get("model_file_edits") != 0
            or tooling_repair41.get("threshold_changes") != 0
            or tooling_repair41.get("science_hypothesis_changes") != 0
            or tooling_repair41.get("additional_critic") is not False
            or tooling1_pair_payload.get("passed") is not True
            or tooling1_pair_payload.get("finite_identity", {}).get("passed") is not True
            or tooling1_pair_payload.get("early_causal_00_20", {}).get("passed")
            is not True
            or tooling1_blocker_payload.get("failure_code")
            != "PREEMPT_PRODUCTION_GPU_ACTIVE"
            or tooling1_blocker_payload.get("model_or_numerical_edit") is not False
            or tooling1_failure_payload.get("failure", {}).get("code")
            != "PREEMPT_PRODUCTION_GPU_ACTIVE"
            or "concurrent.futures"
            not in tooling1_failure_payload.get("failure", {}).get("detail", "")
            or tooling1_failure_payload.get("last_healthy", {}).get("manifest", {}).get(
                "floating_nonfinite_count"
            ) != 0
            or tooling1_failure_payload.get("first_failed", {}).get("manifest", {}).get(
                "floating_nonfinite_count"
            ) != 0
            or source.get("verdict") != "STAGE_OMEGA_TRANSPORT_SOURCE_ORACLE_GREEN"
            or not source_checks
            or not all(source_checks.values())
            or source.get("gpu_commands") != 0
            or source.get("gpu_queries") != 0
            or ab.get("verdict") != "STAGE_OMEGA_TRANSPORT_CPU_AB_GREEN"
            or not ab_checks
            or not all(ab_checks.values())
            or ab.get("input", {}).get("leaf_count") != 106
            or ab.get("effective_runtime", {}).get(
                "jax_cpu_enable_async_dispatch"
            ) is not False
            or ab.get("retained_A", {}).get("stablehlo_sha256")
            != "c6f79887129a2ad3c84559a91abe6ff2cf9ac5595f136ad906b211bba54358b0"
            or ab.get("retained_A", {}).get("manifest", {}).get("sha256")
            != "8c5fd350b01da801dd9b12eddac6a747a9e822f0bb2c70ba12a37af93f530e2c"
            or ab.get("candidate_B", {}).get("stablehlo_sha256")
            != "8ba0a8b61291b6578eb7ff91cdf0f301ec68380a415002ac26006e3f117be0e1"
            or ab.get("candidate_B", {}).get("manifest", {}).get("sha256")
            != "102e0bc50debb290e815c317fccae38a80d94a2e5632c913c693544ad2db0ba2"
            or ab.get("gpu_commands") != 0
            or ab.get("gpu_queries") != 0
        ):
            raise runner.RunnerGateError(
                "STAGE_OMEGA_CANDIDATE_AUTHORITY_SEMANTICS",
                repr(
                    {
                        "contract37": contract37.get("verdict"),
                        "contract38": contract38.get("verdict"),
                        "contract39": contract39.get("verdict"),
                        "contract40": contract40.get("verdict"),
                        "contract41": contract41.get("verdict"),
                        "sealed": sealed,
                        "source": source.get("verdict"),
                        "ab": ab.get("verdict"),
                    }
                ),
            )
        return {
            **prior,
            "stage_omega_amendment37": contract37_row,
            "stage_omega_amendment38": contract38_row,
            "stage_omega_amendment39": contract39_row,
            "stage_omega_amendment40": contract40_row,
            "stage_omega_amendment41": contract41_row,
            "stage_omega_first_gpu_pair": pair_row,
            "stage_omega_first_gpu_blocker": blocker40_row,
            "stage_omega_first_gpu_failure": failure40_row,
            "stage_omega_tooling1_pair": tooling1_pair_row,
            "stage_omega_tooling1_blocker": tooling1_blocker_row,
            "stage_omega_tooling1_failure": tooling1_failure_row,
            "stage_omega_source_oracle": source_row,
            "stage_omega_complete_cpu_ab": ab_row,
            "post_fable_arm_s_gpu": arm_s_row,
            "post_fable_arm_s_green_through_step": 9405,
            "t_source_candidate_commit": CANDIDATE,
            "stage_omega_candidate_commit": CANDIDATE,
            "stage_omega_parent_model_commit": PARENT_MODEL,
            "earliest_causal_gate": "d03 step 200 / 00:20",
            "full_18h_authorized_only_after_step200_green": True,
            "additional_review_or_manager_authority_required": False,
        }

    def assert_stage_omega_source_authority(
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
        post_fable_tree = runner._git(
            runner.REPO_ROOT, "rev-parse", f"{POST_FABLE_BASE}^{{tree}}"
        )
        post_fable_ancestor = subprocess.run(
            (
                "git",
                "-C",
                str(runner.REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                POST_FABLE_BASE,
                head,
            ),
            check=False,
        ).returncode == 0
        if post_fable_tree != POST_FABLE_BASE_TREE or not post_fable_ancestor:
            raise runner.RunnerGateError(
                "POST_FABLE_BASE_AUTHORITY",
                repr({"tree": post_fable_tree, "ancestor": post_fable_ancestor}),
            )
        model_diff = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", CANDIDATE, "--", "src/gpuwrf"
        )
        if model_diff:
            raise runner.RunnerGateError("ACCEPTED_MODEL_CHANGED", model_diff)
        delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", POST_FABLE_BASE, head
        ).splitlines()
        allowed_exact = {
            str(runner.RUNNER_SOURCE.relative_to(runner.REPO_ROOT)),
            "scripts/v0234_stage_omega_transport_runner_profile.py",
            "tests/test_v0234_nested_frozen_wrf_boundary_window.py",
            "tests/test_v0234_post_fable_corner_window.py",
            "tests/test_v0234_post_fable_fulltree_tooling.py",
        }
        allowed_prefixes = (
            str(post_fable_sprint.relative_to(runner.REPO_ROOT)) + "/",
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
            "post_fable_base_commit": POST_FABLE_BASE,
            "post_fable_base_tree": post_fable_tree,
            "post_fable_base_is_ancestor": post_fable_ancestor,
            "accepted_model_diff_empty": True,
            "post_fable_tooling_delta": delta,
            "runner_only_delta": delta,
            "unexpected_runner_delta": unexpected,
            "accepted_model_tree": runner._git(
                runner.REPO_ROOT, "rev-parse", f"{CANDIDATE}:src/gpuwrf"
            ),
            "accepted_sources": source_rows,
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "worktree_clean": not bool(dirty),
        }

    runner.assert_final_candidate_proof_authority = (
        assert_stage_omega_candidate_authority
    )
    runner.assert_candidate_source_authority = assert_stage_omega_source_authority
    runner._STAGE_OMEGA_TRANSPORT_PROFILE_APPLIED = True
