"""CPU-only capture-parent wiring for ADR-036 evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import m0_vram_sampler as mvs  # noqa: E402
import prepare_m0_matched_pair as pair_prep  # noqa: E402
import step1_driver as drv  # noqa: E402
import step1_dryrun as dry  # noqa: E402
import step1_stub as stub  # noqa: E402


RUN_ID = "m0-parent-hook-0001"


def _locked_env(tmp_path: Path) -> dict[str, str]:
    holder = tmp_path / "holder.txt"
    holder.write_text("holder=baseline-census pid=1 token=tok cmd=y\n")
    return dict(
        os.environ,
        GPUWRF_GPU_LOCK_HELD="1",
        GPUWRF_GPU_LOCK_TOKEN="tok",
        GPUWRF_GPU_LOCK_HOLDER_FILE=str(holder),
        GPUWRF_GPU_LOCK_LABEL="baseline-census",
    )


def _identity(path: Path, **updates) -> Path:
    payload = {
        "schema": mvs.IDENTITY_SCHEMA,
        "run_id": RUN_ID,
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
        "device_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "workload_identity_sha256": "4" * 64,
        "integration_scope_sha256": "5" * 64,
        "event_mix_sha256": "6" * 64,
        **updates,
    }
    path.write_text(json.dumps(payload))
    return path


class FakeSampler:
    def __init__(
        self,
        *,
        identity,
        allocator_sidecar,
        output_sidecar,
        stderr_path,
        fail_finish=False,
        events=None,
    ):
        self.identity = identity
        self.allocator_sidecar = allocator_sidecar
        self.output_sidecar = output_sidecar
        self.stderr_path = stderr_path
        self.fail_finish = fail_finish
        self.events = events if events is not None else []
        self.root_pid = None

    def start(self):
        self.events.append("sampler_start")

    def attach(self, *, root_pid, process_group_id):
        self.events.append("sampler_attach")
        self.root_pid = root_pid
        self.process_group_id = process_group_id

    def finish(self):
        self.events.append("sampler_finish")
        if self.fail_finish:
            raise mvs.ResidencyEvidenceError("mutated sampler evidence")
        allocator = {
            "schema": mvs.ALLOCATOR_SCHEMA,
            "emitter_process_role": "forecast_process",
            **self.identity.binding(),
            "forecast_pid": self.root_pid,
        }
        total = {
            "schema": mvs.SAMPLER_SCHEMA,
            "sampler_process_role": "lock_owner_parent",
            **self.identity.binding(),
            "forecast_pid": self.root_pid,
        }
        drv._atomic_json(self.allocator_sidecar, allocator)
        drv._atomic_json(self.output_sidecar, total)
        return total

    def abort(self):
        self.events.append("sampler_abort")


def _factory(events, *, fail_finish=False):
    def build(**kwargs):
        return FakeSampler(
            **kwargs, fail_finish=fail_finish, events=events
        )

    return build


def test_default_stub_capture_keeps_evidence_hook_off(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    out = tmp_path / "run-default"
    drv.run_capture(
        out_root=out,
        run_id=RUN_ID,
        receipt_path=receipt,
        env=_locked_env(tmp_path),
        stub_stage=stub.stub_stage_command(out, run_id=RUN_ID),
        ledger_hint=tmp_path / "ledger.json",
    )
    seen = json.loads((out / "stub_env.json").read_text())
    assert seen.get("GPUWRF_M0_EVIDENCE") is None
    assert not (out / "forecast_allocator.json").exists()
    outcome = json.loads((out / "capture_outcome.json").read_text())
    assert outcome["evidence_hook"] == {
        "enabled": False,
        "default_off": True,
        "identity_path": None,
    }


def test_opt_in_identity_reaches_child_and_parent_sampler_lifecycle(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    identity = _identity(tmp_path / "identity.json")
    out = tmp_path / "run-enabled"
    events: list[str] = []
    outcome = drv.run_capture(
        out_root=out,
        run_id=RUN_ID,
        receipt_path=receipt,
        evidence_identity_path=identity,
        env=_locked_env(tmp_path),
        stub_stage=stub.stub_stage_command(out, run_id=RUN_ID),
        ledger_hint=tmp_path / "ledger.json",
        sampler_factory=_factory(events),
    )

    seen = json.loads((out / "stub_env.json").read_text())
    assert seen["GPUWRF_M0_EVIDENCE"] == "1"
    assert seen["GPUWRF_M0_RUN_ID"] == RUN_ID
    assert seen["GPUWRF_M0_EVIDENCE_PATH"] == str(
        (out / "forecast_allocator.json").resolve()
    )
    assert events == ["sampler_start", "sampler_attach", "sampler_finish"]
    assert outcome["status"] == "OK"
    assert outcome["residency_evidence"]["status"] == "OK"
    manifest = json.loads((out / "capture_artifacts.json").read_text())
    assert {
        "evidence_identity",
        "forecast_allocator",
        "lock_owner_total_residency",
    } <= set(manifest["artifacts"])


def test_sampler_mutation_turns_successful_child_into_failed_capture(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    identity = _identity(tmp_path / "identity.json")
    out = tmp_path / "run-failed-evidence"
    outcome = drv.run_capture(
        out_root=out,
        run_id=RUN_ID,
        receipt_path=receipt,
        evidence_identity_path=identity,
        env=_locked_env(tmp_path),
        stub_stage=stub.stub_stage_command(out, run_id=RUN_ID),
        ledger_hint=tmp_path / "ledger.json",
        sampler_factory=_factory([], fail_finish=True),
    )
    assert outcome["returncode"] == 0
    assert outcome["status"] == "FAILED"
    assert outcome["residency_evidence"]["status"] == "FAILED"
    assert "mutated sampler evidence" in outcome["residency_evidence"]["error"]


def test_bad_identity_fails_before_output_creation_or_receipt_spend(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    identity = _identity(tmp_path / "identity.json", source_sha256="short")
    out = tmp_path / "run-bad-identity"
    ledger = tmp_path / "ledger.json"
    with pytest.raises(mvs.ResidencyEvidenceError, match="source_sha256"):
        drv.run_capture(
            out_root=out,
            run_id=RUN_ID,
            receipt_path=receipt,
            evidence_identity_path=identity,
            env=_locked_env(tmp_path),
            stub_stage=["true"],
            ledger_hint=ledger,
        )
    assert not out.exists()
    assert not ledger.exists()


def test_production_capture_without_identity_refuses_before_any_launch(tmp_path):
    with pytest.raises(drv.Step1Blocked, match="requires --evidence-identity"):
        drv.run_capture(
            out_root=tmp_path / "never-created",
            run_id=RUN_ID,
            receipt_path=tmp_path / "unused-receipt.json",
            stub_stage=None,
        )
    assert not (tmp_path / "never-created").exists()


def test_production_orchestration_without_identity_refuses_before_lock(tmp_path):
    called = []
    with pytest.raises(drv.Step1Blocked, match="requires --evidence-identity"):
        drv.run_orchestrate(
            out_root=tmp_path / "never-created",
            run_id=RUN_ID,
            receipt_path=tmp_path / "unused-receipt.json",
            wrapper_runner=lambda command: called.append(command),
        )
    assert called == []


def test_separate_pair_is_prepared_as_exact_held_session_callbacks(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("x = 1\n")
    run_dir = tmp_path / "case"
    run_dir.mkdir()
    (run_dir / "namelist.input").write_text("&time_control\n/\n")
    (run_dir / "wrfout_d01_0000").write_bytes(b"initial")
    (run_dir / "wrfout_d01_0100").write_bytes(b"final")
    cache_seed = tmp_path / "cache-seed"
    cache_seed.mkdir()
    (cache_seed / "jit_exact-cache").write_bytes(b"prepared executable")
    cache_identity = mvs.directory_tree_identity(cache_seed)
    fast_pair = tmp_path / "fast-pair.json"
    fast_pair.write_text(json.dumps({"schema": "test.fast-pair.v1"}))
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({
        "schema": "wrf_gpu2.v025.m0.autotune0_qualification.v1",
        "status": "AUTOTUNE0_QUALIFIED",
        "run_id": pair_prep.exact_parent.WINDOWS["W1"]["run_id"],
        "cache_seed_path": str(cache_seed.resolve()),
        "cache_seed_sha256": cache_identity["sha256"],
        "cold_readiness_seconds": 599.0,
        "cached_readiness_seconds": 59.0,
        "warm_integration_seconds": 299.0,
        "selected_cold_stage": "cold_empty_cache_readiness_1",
        "selected_cold_attempt_index": 1,
        "fast_pair_path": str(fast_pair),
        "fast_pair_sha256": pair_prep._sha256(fast_pair),
        "w1_exact_boundary": {
            "run_id": pair_prep.exact_parent.WINDOWS["W1"]["run_id"],
            "cold_result_sha256": "1" * 64,
            "cached_result_sha256": "2" * 64,
            "gpu_result_sha256": "3" * 64,
        },
        "pair_completeness": {
            "fresh_cpu_arm_this_invocation": True,
            "fresh_gpu_arm_this_invocation": True,
            "comparator_result_present": True,
            "provenance_present": True,
            "arms_non_overlapping": True,
            "completeness_percent": 100,
        },
        "integration_clock": (
            "synchronized-integration-only-excluding-compile-cache-load-and-io"
        ),
    }))

    output = tmp_path / "prepared"
    cache_snapshots = tmp_path / "cache-snapshots"
    monkeypatch.setattr(pair_prep, "CANONICAL_OUTPUT_DIR", output)
    monkeypatch.setattr(pair_prep, "CANONICAL_CACHE_SEED", cache_seed)
    monkeypatch.setattr(
        pair_prep, "CANONICAL_CACHE_SNAPSHOT_ROOT", cache_snapshots
    )
    monkeypatch.setattr(pair_prep, "CANONICAL_QUALIFICATION", qualification)
    monkeypatch.setattr(pair_prep, "CANONICAL_SOURCE_ROOT", source)
    monkeypatch.setattr(pair_prep, "CANONICAL_RUN_DIR", run_dir)
    monkeypatch.setattr(
        pair_prep, "CANONICAL_PROFILED_RUN_ID", "pair-profiled-test-0001"
    )
    monkeypatch.setattr(
        pair_prep, "CANONICAL_CLEAN_RUN_ID", "pair-clean-test-0001"
    )
    session_receipt = tmp_path / "future-session-receipt.json"
    plan = pair_prep.prepare(
        output_dir=output,
        device_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        profiled_run_id="pair-profiled-test-0001",
        clean_run_id="pair-clean-test-0001",
        profiled_receipt=session_receipt,
        clean_receipt=session_receipt,
        cache_seed=cache_seed,
        cache_snapshot_root=cache_snapshots,
        qualification_manifest=qualification,
        source_root=source,
        run_dir=run_dir,
        raw_root=tmp_path / "future-raw",
    )
    assert plan["status"] == "PREPARED_NOT_AUTHORISED_NOT_RUN"
    assert plan["manager_binding"]["status"] == (
        "FROZEN_MANAGER_BINDING_CONFIRMED"
    )
    assert all(plan["manager_binding"]["checks"].values())
    assert plan["profiler_gate_evidence_status"] == "MISSING"
    assert plan["held_session_execution"] == {
        "label": "m0-core-w1-w2-w3-session",
        "receipt_path": str(session_receipt),
        "receipt_spends": 1,
        "canonical_lock_acquisitions": 1,
        "callbacks": ["W2_PROFILED_CAPTURE", "W3_CLEAN_MATCHED_ARM"],
        "second_wrapper_reachable": False,
    }
    assert plan["profiled"]["command"] == [
        "HELD_SESSION_CALLBACK",
        "m0-core-w1-w2-w3-session",
        "W2",
        str(session_receipt),
    ]
    assert plan["clean"]["command"] == [
        "HELD_SESSION_CALLBACK",
        "m0-core-w1-w2-w3-session",
        "W3",
        str(session_receipt),
    ]
    assert plan["profiled"]["run_id"] != plan["clean"]["run_id"]
    assert plan["profiled"]["instrumentation"] == "nsys"
    assert plan["clean"]["instrumentation"] == "none"
    assert plan["profiled"]["evidence_hook"] == (
        "sprint-local exact compiled boundary"
    )
    assert plan["clean"]["evidence_hook"].startswith("no production hook")
    assert plan["identical_fields"]["workload_identity_sha256"]
    assert plan["prepared_cache"]["profiled_identity"] == (
        plan["prepared_cache"]["clean_identity"]
    )
    assert plan["prepared_cache"]["private_directory_per_arm"] is True
    assert (
        plan["prepared_cache"]["new_miss_files_private_by_directory"] is True
    )
    assert plan["prepared_cache"]["ram_scaling"].startswith("O(1)")
    assert plan["prepared_cache"]["disk_scaling"].startswith("O(seed bytes")
    assert Path(plan["profiled"]["prepared_cache_path"]).is_dir()
    assert Path(plan["clean"]["prepared_cache_path"]).is_dir()
    assert plan["prepared_cache"]["qualification"]["status"] == (
        "AUTOTUNE0_QUALIFIED"
    )
    assert json.loads((output / "plan.json").read_text()) == plan


def test_pair_preparation_rejects_paths_the_manager_would_ignore(tmp_path):
    with pytest.raises(RuntimeError, match="frozen W2/W3 manager graph"):
        pair_prep.prepare(
            output_dir=tmp_path / "wrong-output",
            device_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            profiled_run_id=pair_prep.CANONICAL_PROFILED_RUN_ID,
            clean_run_id=pair_prep.CANONICAL_CLEAN_RUN_ID,
            profiled_receipt=tmp_path / "profiled-receipt.json",
            clean_receipt=tmp_path / "clean-receipt.json",
            cache_seed=pair_prep.CANONICAL_CACHE_SEED,
            cache_snapshot_root=pair_prep.CANONICAL_CACHE_SNAPSHOT_ROOT,
            qualification_manifest=pair_prep.CANONICAL_QUALIFICATION,
            source_root=pair_prep.CANONICAL_SOURCE_ROOT,
            run_dir=pair_prep.CANONICAL_RUN_DIR,
        )
    assert not (tmp_path / "wrong-output").exists()


def test_pair_snapshots_cannot_exist_before_w1_qualification(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("x = 1\n")
    run_dir = tmp_path / "case"
    run_dir.mkdir()
    (run_dir / "namelist.input").write_text("&time_control\n/\n")
    (run_dir / "wrfout_d01_0000").write_bytes(b"initial")
    (run_dir / "wrfout_d01_0100").write_bytes(b"final")
    cache_seed = tmp_path / "cache-seed"
    cache_seed.mkdir()
    (cache_seed / "jit-cache").write_bytes(b"cache")
    snapshots = tmp_path / "snapshots"
    output = tmp_path / "prepared"
    missing_qualification = tmp_path / "missing-qualification.json"
    monkeypatch.setattr(pair_prep, "CANONICAL_OUTPUT_DIR", output)
    monkeypatch.setattr(pair_prep, "CANONICAL_CACHE_SEED", cache_seed)
    monkeypatch.setattr(
        pair_prep, "CANONICAL_CACHE_SNAPSHOT_ROOT", snapshots
    )
    monkeypatch.setattr(
        pair_prep, "CANONICAL_QUALIFICATION", missing_qualification
    )
    monkeypatch.setattr(pair_prep, "CANONICAL_SOURCE_ROOT", source)
    monkeypatch.setattr(pair_prep, "CANONICAL_RUN_DIR", run_dir)
    monkeypatch.setattr(
        pair_prep, "CANONICAL_PROFILED_RUN_ID", "pair-profiled-test-0002"
    )
    monkeypatch.setattr(
        pair_prep, "CANONICAL_CLEAN_RUN_ID", "pair-clean-test-0002"
    )

    with pytest.raises(RuntimeError, match="qualification manifest is missing"):
        pair_prep.prepare(
            output_dir=output,
            device_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            profiled_run_id="pair-profiled-test-0002",
            clean_run_id="pair-clean-test-0002",
            profiled_receipt=tmp_path / "profiled-receipt.json",
            clean_receipt=tmp_path / "clean-receipt.json",
            cache_seed=cache_seed,
            cache_snapshot_root=snapshots,
            qualification_manifest=missing_qualification,
            source_root=source,
            run_dir=run_dir,
        )

    assert not snapshots.exists()
    assert not output.exists()


def test_pair_print_plan_is_non_consuming_and_needs_no_existing_inputs(tmp_path):
    missing_root = tmp_path / "must-remain-missing"
    args = pair_prep.argparse.Namespace(
        output_dir=missing_root / "output",
        device_uuid=None,
        profiled_run_id="future-profiled-0001",
        clean_run_id="future-clean-0001",
        profiled_receipt=missing_root / "receipt-profiled.json",
        clean_receipt=missing_root / "receipt-clean.json",
        cache_seed=missing_root / "cache-seed",
        cache_snapshot_root=missing_root / "snapshots",
        qualification_manifest=missing_root / "qualification.json",
    )
    plan = pair_prep.inspection_plan(args)

    assert plan["status"] == "INSPECTION_ONLY_NO_FILES_READ_OR_WRITTEN"
    assert plan["receipts_consumed"] == []
    assert not missing_root.exists()


def test_superseded_pair_refuses_before_output_or_receipt_spend(tmp_path):
    calls: list[str] = []
    out = tmp_path / "never-created"
    with pytest.raises(drv.Step1Blocked, match="mechanically superseded"):
        drv.run_capture(
            out_root=out,
            run_id="m0-pair-profiled-20260728-r1",
            receipt_path=tmp_path / "unused-receipt.json",
            stub_stage=["true"],
            authorise=lambda *args, **kwargs: calls.append("authorised"),
        )
    assert calls == []
    assert not out.exists()


def test_copied_superseded_identity_refuses_before_receipt_spend(tmp_path):
    relative = "proofs/v025/m0/prepared_profiler_pair/profiled_identity.json"
    old = subprocess.run(
        ["git", "show", f"5638bedd:{relative}"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    identity = tmp_path / "copied-old-identity.json"
    identity.write_bytes(old)
    calls: list[str] = []

    with pytest.raises(drv.Step1Blocked, match="identity_sha256"):
        drv.run_capture(
            out_root=tmp_path / "never-created",
            run_id="fresh-profiled-run-0001",
            receipt_path=tmp_path / "unused-receipt.json",
            evidence_identity_path=identity,
            stub_stage=["true"],
            authorise=lambda *args, **kwargs: calls.append("authorised"),
        )
    assert calls == []
    assert not (tmp_path / "never-created").exists()


def test_copied_superseded_plan_and_cache_hash_are_rejected(tmp_path):
    relative = "proofs/v025/m0/prepared_profiler_pair/plan.json"
    old_plan = subprocess.run(
        ["git", "show", f"5638bedd:{relative}"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    copied = tmp_path / "copied-old-plan.json"
    copied.write_bytes(old_plan)

    with pytest.raises(pair_prep.stale_pair.StalePairError, match="plan_sha256"):
        pair_prep.stale_pair.reject_stale_prepared_pair(plan_path=copied)
    with pytest.raises(
        pair_prep.stale_pair.StalePairError,
        match="prepared_cache_sha256",
    ):
        pair_prep.stale_pair.reject_stale_prepared_pair(
            prepared_cache_sha256=(
                "825980d919b383833d88c80e416c20cb3f47c0e36b434ed65167288d7409c618"
            )
        )


def test_prepared_cache_identity_fails_closed_on_partial_or_mutated_state(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "entry-cache").write_bytes(b"one")
    identity = mvs.directory_tree_identity(cache)
    assert drv.validate_prepared_cache(cache, identity["sha256"]) == identity

    with pytest.raises(drv.Step1Blocked, match="supplied together"):
        drv.validate_prepared_cache(cache, None)
    (cache / "entry-cache").write_bytes(b"two")
    with pytest.raises(drv.Step1Blocked, match="does not match"):
        drv.validate_prepared_cache(cache, identity["sha256"])


def test_clean_pair_capture_keeps_hook_and_sampler_off(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    identity = _identity(tmp_path / "identity.json")
    out = tmp_path / "run-clean"
    events: list[str] = []
    outcome = drv.run_capture(
        out_root=out,
        run_id=RUN_ID,
        receipt_path=receipt,
        evidence_identity_path=identity,
        profiled=False,
        env=_locked_env(tmp_path),
        stub_stage=stub.stub_stage_command(out, run_id=RUN_ID),
        ledger_hint=tmp_path / "ledger.json",
        sampler_factory=lambda **kwargs: events.append("constructed"),
    )
    seen = json.loads((out / "stub_env.json").read_text())
    assert seen["GPUWRF_M0_EVIDENCE"] is None
    assert outcome["evidence_hook"]["enabled"] is False
    assert outcome["evidence_hook"]["default_off"] is True
    assert outcome["residency_evidence"]["status"] == "NOT_ENABLED"
    assert events == []


def test_clean_pair_capture_rejects_inherited_evidence_environment(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    identity = _identity(tmp_path / "identity.json")
    hostile = _locked_env(tmp_path)
    hostile["GPUWRF_M0_EVIDENCE"] = "1"
    with pytest.raises(drv.Step1Blocked, match="requires every GPUWRF_M0"):
        drv.run_capture(
            out_root=tmp_path / "never-created",
            run_id=RUN_ID,
            receipt_path=receipt,
            evidence_identity_path=identity,
            profiled=False,
            env=hostile,
            stub_stage=["true"],
        )
