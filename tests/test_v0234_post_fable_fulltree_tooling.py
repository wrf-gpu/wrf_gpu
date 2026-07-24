from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


os.environ["GPUWRF_NESTED_BUNDLE_APPROVED_SHA"] = (
    "470e6111d516479bed4bc0c3b2be1007bb082afd"
)

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402


def _write_self_hashed_json(path: Path, payload: dict[str, object]) -> tuple[str, str]:
    unsigned = dict(payload)
    unsigned.pop("proof_sha256", None)
    canonical = runner.canonical_digest(unsigned)
    signed = {**unsigned, "proof_sha256": canonical}
    path.write_text(json.dumps(signed, sort_keys=True, indent=2) + "\n")
    return runner.sha256_file(path), canonical


def test_profile_binds_post_fable_namespace_lock_and_hard_critic_gate() -> None:
    assert runner.SCHEMA == "gpuwrf.v0234.post-fable-late-ni-fulltree.v1"
    assert runner.REQUIRE_KNOWN_1500_V10_RECORD is True
    assert runner.REQUIRE_TOOLING_CRITIC_ACCEPT is True
    assert runner.FULL_REPLAY_NAMESPACE == (
        "nested_stage_omega_transport_470e6111_"
        "post_fable_corner_window_gpt56_resource_retry1"
    )
    assert runner.LOCK_LABEL == "v0234-post-fable-fulltree-resource-retry1"
    assert runner.RUNNER_CPU_AUDIT.name == "resource-retry-runner-cpu-proof.json"
    assert runner.TOOLING_CRITIC_SCHEMA == (
        "gpuwrf.v0234.post-fable-resource-restart-critic.v1"
    )
    assert runner.TOOLING_CRITIC_VERDICT == "KIMI_RESOURCE_RESTART_CRITIC_ACCEPT"
    audit = runner.audit_exact_launch_command(runner.LAUNCH_COMMAND)
    assert audit["passed"] is True
    assert audit["known_1500_v10_record_required"] is True
    assert audit["tooling_critic_accept_required"] is True
    text = runner.LAUNCH_COMMAND.read_text()
    assert "--record-known-1500-v10-red" in text
    assert 'GPUWRF_POST_FABLE_TOOLING_CRITIC="$CRITIC"' in text
    assert 'CRITIC="$SPRINT/kimi-resource-restart-critic-accept.json"' in text
    assert 'AUDIT="$SPRINT/resource-retry-runner-cpu-proof.json"' in text
    assert f"RUN_DIR={runner.LINEAGE_WORK_DIR / runner.FULL_REPLAY_NAMESPACE}" in text
    assert "--intent production-preemptible" in text
    assert "nvidia-smi" not in text
    assert text.count("/scripts/with_gpu_lock.sh") == 1
    assert text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1


def test_resource_restart_authority_preserves_preempted_namespace_exactly() -> None:
    assert runner.TOOLING_CRITIC_AUTHORITY_HOOK is not None
    authority = runner.TOOLING_CRITIC_AUTHORITY_HOOK()
    assert authority["required"] is True
    assert authority["prior_scientific_result_produced"] is False
    assert authority["preserved_file_set_exact"] is True
    assert authority["resource_retry_number"] == 1
    assert authority["resource_retry_namespace_absent"] is True
    assert authority["preempt_proof"]["sha256"] == (
        "481bc1ea1120a3efde478c888cbd09f7ca751a88c2b913b8b60803272c78e4e1"
    )
    assert authority["preempt_proof"]["canonical_payload_sha256"] == (
        "2459ea780ae9bc97ec1b47709290687b0bca807d2448722ab569049148456dd6"
    )
    artifacts = authority["preserved_artifacts"]
    assert len(artifacts) == 3
    assert {row["sha256"] for row in artifacts} == {
        "15575931876ca3126e91e5b1c7cbcd85360e7cc2754354f262c5c82556d3a308",
        "52e455921d02c3047e6c251b59a907685cb97deeecf994583d9dff5372920cef",
    }


def test_resource_restart_launcher_requires_new_proofs_before_lock() -> None:
    text = runner.LAUNCH_COMMAND.read_text()
    lock_offset = text.index("/scripts/with_gpu_lock.sh")
    assert text.index('AUDIT_SHA="$(sha256sum "$AUDIT"') < lock_offset
    assert text.index('CRITIC_SHA="$(sha256sum "$CRITIC"') < lock_offset
    assert "kimi-tooling-critic-accept.json" not in text
    assert "full-tree-runner-cpu-proof.json" not in text
    assert "_resource_retry1" in text
    assert "--intent production-preemptible" in text


def test_arm_s_green_authority_is_authenticated_before_replay() -> None:
    authority = runner.assert_final_candidate_proof_authority()
    arm = authority["post_fable_arm_s_gpu"]
    assert arm["sha256"] == (
        "3986508430fe0bb40546abaecf6f978287f6eedef1f702fd6a6b82f15196befb"
    )
    assert arm["canonical_payload_sha256"] == (
        "166fb66cb74e76b4d0759fa573ad6478ace030b36839725b09cfafbcdaa96298"
    )
    assert authority["post_fable_arm_s_green_through_step"] == 9405


def test_known_v10_authority_is_exact_and_remains_a_release_blocker() -> None:
    authority, row = runner.authenticate_known_1500_v10_observation()
    assert row["sha256"] == (
        "38683937ed6be6eb96bb71c3bb7c5e8976d8d91c525021c184712c24eb169448"
    )
    assert row["canonical_payload_sha256"] == (
        "82ddb7bd13d5a1c6e91827ac15169d6de8822f0675e23ee369e4c93970e925e1"
    )
    result = runner.classify_exact_known_1500_v10_red(
        authority["decisive_1500"],
        candidate_sha256=authority["candidate"]["sha256"],
        authority=authority,
    )
    assert result["passed"] is True
    assert result["classification"] == "KNOWN_V10_ONLY_RED_EXACT"
    assert result["waiver_or_reclassification"] is False
    assert result["release_blocker_remains"] is True


def test_known_v10_record_rejects_metric_or_frame_drift() -> None:
    authority, _row = runner.authenticate_known_1500_v10_observation()
    changed = copy.deepcopy(authority["decisive_1500"])
    changed["strict_fields"]["V10"]["candidate_rmse"] += 1.0e-12
    metric_drift = runner.classify_exact_known_1500_v10_red(
        changed,
        candidate_sha256=authority["candidate"]["sha256"],
        authority=authority,
    )
    assert metric_drift["passed"] is False
    assert metric_drift["decisive_payload_exact"] is False
    frame_drift = runner.classify_exact_known_1500_v10_red(
        authority["decisive_1500"],
        candidate_sha256="0" * 64,
        authority=authority,
    )
    assert frame_drift["passed"] is False
    assert frame_drift["candidate_frame_sha256_exact"] is False


def test_real_retained_1500_pair_is_recorded_red_and_returns_for_late_ni(
    tmp_path: Path,
) -> None:
    import numpy as np
    from netCDF4 import Dataset

    from scripts import v0234_corrected_fullbuffer_gate as comparator

    authority, _row = runner.authenticate_known_1500_v10_observation()
    runtime = SimpleNamespace(np=np, Dataset=Dataset, comparator=comparator)
    pairer = runner.IncrementalFramePairer(
        runtime,
        tmp_path / "frame-pairs",
        record_known_1500_v10_red=True,
    )
    result = pairer.pair(
        "d03",
        runner.DECISIVE_D03_STEP,
        Path(authority["candidate"]["path"]),
    )
    assert result["scientific_pair_pass"] is False
    assert result["finite_identity_pass"] is True
    assert result["decisive_1500"]["passed"] is False
    assert result["known_1500_v10_record"]["passed"] is True
    assert result["recorded_known_v10_red_and_continued"] is True
    artifact = json.loads(Path(result["artifact"]).read_text())
    assert artifact["passed"] is False
    assert artifact["recorded_known_v10_red_and_continued"] is True
    assert artifact["known_1500_v10_record"]["waiver_or_reclassification"] is False


def test_known_v10_authority_rejects_a_second_red_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _row = runner.authenticate_known_1500_v10_observation()
    changed = copy.deepcopy(authority)
    changed["decisive_1500"]["strict_fields"]["U"]["no_worse"] = False
    path = tmp_path / "changed-known-red.json"
    file_sha, payload_sha = _write_self_hashed_json(path, changed)
    monkeypatch.setattr(runner, "KNOWN_1500_V10_ARTIFACT", path)
    monkeypatch.setattr(runner, "KNOWN_1500_V10_ARTIFACT_SHA256", file_sha)
    monkeypatch.setattr(runner, "KNOWN_1500_V10_PAYLOAD_SHA256", payload_sha)
    with pytest.raises(runner.RunnerGateError, match="KNOWN_1500_V10_AUTHORITY_SEMANTICS"):
        runner.authenticate_known_1500_v10_observation()


def test_tooling_critic_acceptance_is_required_and_exact(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.RunnerGateError, match="TOOLING_CRITIC_ENV"):
        runner.validate_tooling_critic_acceptance({}, required=True)
    candidate = runner._git(runner.REPO_ROOT, "rev-parse", "HEAD")
    payload: dict[str, object] = {
        "schema": runner.TOOLING_CRITIC_SCHEMA,
        "verdict": runner.TOOLING_CRITIC_VERDICT,
        "candidate_commit": candidate,
        "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
        "exact_launcher_sha256": runner.sha256_file(runner.LAUNCH_COMMAND),
        "profile_source_sha256": runner.sha256_file(
            runner.TOOLING_CRITIC_PROFILE_SOURCE
        ),
        "model_code_changed": False,
        "full_tree_gpu_replay_admitted": True,
        "critic": "Kimi K3 thinking-max",
        **runner.TOOLING_CRITIC_REQUIRED_PAYLOAD,
    }
    path = tmp_path / "critic.json"
    file_sha, payload_sha = _write_self_hashed_json(path, payload)
    accepted = runner.validate_tooling_critic_acceptance(
        {
            "GPUWRF_POST_FABLE_TOOLING_CRITIC": str(path),
            "GPUWRF_POST_FABLE_TOOLING_CRITIC_SHA256": file_sha,
        },
        required=True,
    )
    assert accepted["checked"] is True
    assert accepted["canonical_payload_sha256"] == payload_sha
    assert accepted["candidate_is_ancestor"] is True
    assert accepted["required_payload_exact"] is True
    assert accepted["extra_authority"]["preserved_file_set_exact"] is True
    assert str(runner.TOOLING_CRITIC_PROFILE_SOURCE.resolve()) in accepted["reviewed_paths"]

    payload["full_tree_gpu_replay_admitted"] = False
    bad_sha, _ = _write_self_hashed_json(path, payload)
    with pytest.raises(runner.RunnerGateError, match="TOOLING_CRITIC_SEMANTICS"):
        runner.validate_tooling_critic_acceptance(
            {
                "GPUWRF_POST_FABLE_TOOLING_CRITIC": str(path),
                "GPUWRF_POST_FABLE_TOOLING_CRITIC_SHA256": bad_sha,
            },
            required=True,
        )

    payload["full_tree_gpu_replay_admitted"] = True
    payload["resource_retry_number"] = 2
    bad_resource_sha, _ = _write_self_hashed_json(path, payload)
    with pytest.raises(runner.RunnerGateError, match="TOOLING_CRITIC_SEMANTICS"):
        runner.validate_tooling_critic_acceptance(
            {
                "GPUWRF_POST_FABLE_TOOLING_CRITIC": str(path),
                "GPUWRF_POST_FABLE_TOOLING_CRITIC_SHA256": bad_resource_sha,
            },
            required=True,
        )

    payload["resource_retry_number"] = 1
    payload["profile_source_sha256"] = "0" * 64
    bad_profile_sha, _ = _write_self_hashed_json(path, payload)
    with pytest.raises(runner.RunnerGateError, match="TOOLING_CRITIC_SEMANTICS"):
        runner.validate_tooling_critic_acceptance(
            {
                "GPUWRF_POST_FABLE_TOOLING_CRITIC": str(path),
                "GPUWRF_POST_FABLE_TOOLING_CRITIC_SHA256": bad_profile_sha,
            },
            required=True,
        )


def test_cli_cannot_cross_known_red_or_critic_scope_accidentally(tmp_path: Path) -> None:
    with pytest.raises(runner.RunnerGateError, match="KNOWN_1500_V10_SCOPE"):
        runner.main([
            "--run-dir", str(tmp_path / "wrong-scope"),
            "--proof-output", str(tmp_path / "wrong-scope.json"),
            "--record-known-1500-v10-red",
        ])
    with pytest.raises(runner.RunnerGateError, match="KNOWN_1500_V10_RECORD_REQUIRED"):
        runner.main([
            "--run-dir", str(tmp_path / runner.FULL_REPLAY_NAMESPACE),
            "--proof-output", str(tmp_path / "missing-record.json"),
            "--direct-terminal",
        ])


def test_late_window_retention_and_terminal_red_semantics_are_source_bound() -> None:
    assert runner.LATE_WINDOW_RETAIN_D03_STEPS == (9313, 9314, 9405)
    retain_source = inspect.getsource(runner.retain_late_window_carry_and_frame)
    window_source = inspect.getsource(runner._run_window)
    terminal_source = inspect.getsource(runner._runtime_main)
    assert "_retain_complete_carry(" in retain_source
    assert "all_numeric_nonfinite" in retain_source
    assert "static_identity" in retain_source
    assert "explicit CONTRACT.md retention" in retain_source
    assert "LATE_WINDOW_RETAIN_D03_STEPS" in window_source
    assert "retain_late_window_carry_and_frame(" in window_source
    assert "FULL_18H_LATE_NI_COMPLETE_KNOWN_V10_RED_REMAINS" in terminal_source
    assert 'output_dir = args.run_dir / "gpu-output"' in terminal_source
    assert '"all_incremental_pairs_passed": not args.record_known_1500_v10_red' in terminal_source
    assert '"known_1500_v10_red_remains_release_blocker"' in terminal_source
    audit = runner.static_source_audit()
    assert audit["passed"] is True
    assert audit["known_1500_v10_record_fail_closed"] is True
    assert audit["tooling_critic_acceptance_preimport_gate"] is True
    assert audit["late_window_9313_9314_9405_retention_fail_closed"] is True
    assert audit["continuation_first_new_red_retained"] is True
    assert audit["canonical_fresh_gpu_output_path"] is True
    assert audit["terminal_v10_red_semantics_honest"] is True


def test_late_window_retention_writes_carry_frame_and_field_manifests(
    tmp_path: Path,
) -> None:
    import hashlib
    import pickle

    import numpy as np
    from netCDF4 import Dataset

    from scripts import v0234_corrected_fullbuffer_gate as comparator

    def write_frame(path: Path, stamp: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(path, "w") as dataset:
            dataset.createDimension("Time", 1)
            dataset.createDimension("DateStrLen", 19)
            dataset.createDimension("y", 2)
            dataset.createDimension("x", 3)
            times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
            times[0, :] = np.asarray(list(stamp), dtype="S1")
            for index, field in enumerate(runner.STRICT_FIELDS):
                variable = dataset.createVariable(field, "f8", ("Time", "y", "x"))
                variable[0, :, :] = np.full((2, 3), index + 1.0)
            for index, field in enumerate(runner.STATIC_FIELDS):
                variable = dataset.createVariable(field, "f8", ("Time", "y", "x"))
                variable[0, :, :] = np.full((2, 3), index + 100.0)

    reference_stamp = "2025-03-01_15:40:00"
    step = 9313
    candidate_stamp = (
        runner.RUN_START
        + runner.timedelta(seconds=step * runner.DT_SECONDS["d03"])
    ).strftime("%Y-%m-%d_%H:%M:%S")
    reference = tmp_path / f"wrfout_d03_{reference_stamp}"
    candidate = tmp_path / f"wrfout_d03_{candidate_stamp}"
    write_frame(reference, reference_stamp)
    write_frame(candidate, candidate_stamp)
    reference_authority = {
        "path": str(reference),
        "bytes": reference.stat().st_size,
        "sha256": runner.sha256_file(reference),
    }
    pairer = SimpleNamespace(
        cpu_index={("d03", reference_stamp): reference_authority}
    )

    def host_manifest(value: object) -> dict[str, object]:
        digest = hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()
        return {
            "leaf_count": 1,
            "floating_nonfinite_count": 0,
            "manifest_sha256": digest,
        }

    ordinary = SimpleNamespace(
        host_tree_manifest=host_manifest,
        compare_manifests=lambda left, right: {
            "all_leaf_bytes_equal": left == right,
        },
    )
    runtime = SimpleNamespace(
        np=np,
        Dataset=Dataset,
        comparator=comparator,
        jax=SimpleNamespace(device_get=lambda value: value),
        ordinary=ordinary,
    )
    frame_writer = lambda _domain, _step, _carry: {"wrfout": str(candidate)}
    row = runner.retain_late_window_carry_and_frame(
        runtime,
        pairer,
        frame_writer,
        tmp_path / "run",
        step=step,
        carry={"theta": np.asarray([300.0])},
        health={"passed": True, "complete_carry_nonfinite_count": 0},
    )
    assert row["passed"] is True
    assert row["frame"]["decoded_time"] == candidate_stamp
    assert row["frame"]["stable_after_inspection"] is True
    assert row["all_numeric_nonfinite"] == []
    assert all(row["static_identity"].values())
    assert set(row["strict_field_manifests"]) == set(runner.STRICT_FIELDS)
    assert Path(row["carry"]["path"]).is_file()
    assert Path(row["frame"]["path"]).is_file()


def test_module_remains_preimport_stdlib_only() -> None:
    assert "jax" not in sys.modules
    assert not any(name.startswith("gpuwrf") for name in sys.modules)
