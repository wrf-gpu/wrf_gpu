"""CPU-only proof of the Amendment-6 one-owner execution graph."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts" / "v025"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m0_three_window_executor as executor  # noqa: E402
import m0_core_session_protocol as core_session  # noqa: E402
import m0_exact_boundary_contract as exact_contract  # noqa: E402


ORDERED_STAGE_KEYS = [
    "CPU_PREFLIGHT:fresh_12_rank_cpu_wrf_preflight",
    "W1:cold_two_of_three_decision",
    "W1:cached_readiness_and_warm_integration",
    "W2:profiled_cached_readiness_and_integration",
    "W2:profiled_artifact_capture_integrity",
    "W3:clean_cached_readiness_and_integration",
    "W3:profiled_clean_exact_identity_gate",
]


def _result_stage_keys(payload):
    keys = []
    for phase in payload["results"]:
        for stage in phase["result"]["stages"]:
            keys.append((f"{phase['phase']}:{stage['name']}", stage["status"]))
    return keys


def test_plan_freezes_three_unique_capped_windows_and_exact_commands():
    plan = executor.build_plan()

    assert plan["status"] == "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
    assert plan["device_touched"] is False
    assert plan["receipts_consumed"] == []
    assert plan["maximum_windows"] == 3
    assert len(plan["windows"]) == 3
    assert len({window["label"] for window in plan["windows"]}) == 3
    assert len({window["run_id"] for window in plan["windows"]}) == 3
    assert len({window["receipt"] for window in plan["windows"]}) == 1
    assert plan["session_receipt"] == executor.SESSION_RECEIPT
    assert plan["maximum_lock_acquisitions"] == 1
    assert (
        plan["amendment_5_session_protocol"]["label"]
        == "m0-core-w1-w2-w3-session"
    )
    owner_command = plan["manager_session_command"]
    assert owner_command == executor._prospective_manager_session_command(
        receipt=executor.SESSION_RECEIPT
    )
    assert "--manager-core-session-owner" in owner_command
    assert plan["manager_session_command_sha256"] == executor._command_sha256(
        owner_command
    )
    held = plan["amendment_5_held_session"]
    session_command = held["held_wrapper_command"]
    assert session_command[:6] == [
        str(executor.LOCK_WRAPPER),
        "--timeout",
        "0",
        "--label",
        core_session.SESSION_LABEL,
        "--",
    ]
    assert held["held_wrapper_command_sha256"] == executor._command_sha256(
        session_command
    )

    deadlines = {"W1": 2160.0, "W2": 1400.0, "W3": 600.0}
    for window in plan["windows"]:
        assert window["deadline_seconds"] == deadlines[window["id"]]
        assert window["budget"]["deadline_seconds"] == deadlines[window["id"]]
        assert window["budget"]["headroom_seconds"] >= 0.0
        assert window["manager_command_invocations"] == 0
        assert window["manager_command_sha256"] == executor._command_sha256(
            window["manager_command"]
        )
        assert window["manager_command"] == owner_command
        for stage in window["stages"]:
            assert stage["command_sha256"] == executor._command_sha256(
                stage["command"]
            )
            assert stage["command"] == window["manager_command"]
            assert (
                stage["command_scope"]
                == "ONE_INVOCATION_FOR_W1_W2_W3_SESSION"
            )
            assert stage["dry_stub_command_sha256"] == executor._command_sha256(
                stage["dry_stub_command"]
            )
            assert stage["dry_stub_command"] == [
                "CPU_DRY_STUB_ONLY",
                window["id"],
                stage["name"],
            ]
    cpu_preflight = plan["prelock_cpu_comparator"]
    assert cpu_preflight["deadline_seconds"] == 420.0
    assert cpu_preflight["device_touched"] is False
    for stage in cpu_preflight["stages"]:
        assert stage["command_sha256"] == executor._command_sha256(stage["command"])
        assert "PYTHONPATH" not in stage["command"]
    assert "scripts/v025/run_cpu_arm.py" in " ".join(
        cpu_preflight["stages"][0]["command"]
    )
    assert "IMPLEMENTED" in cpu_preflight[
        "exact_result_wrfout_adapter"
    ]
    assert held["held_budget"]["reservation_seconds"] == 4220.0
    assert held["held_budget"]["headroom_seconds"] == 280.0
    assert all(
        "pallas" not in stage["name"].lower()
        for window in plan["windows"]
        for stage in window["stages"]
    )

    assert plan["profiled_pair_order"] == [
        "m0-autotune0-profiled-20260728-r3",
        "m0-autotune0-clean-20260728-r3",
    ]
    assert plan["mechanical_order"].index("W2_PROFILED_CAPTURE") < (
        plan["mechanical_order"].index("W3_CLEAN_MATCHED_ARM")
    )
    assert (
        plan["timing_boundaries"]["current_gate_eligibility"]
        == "CPU_PROVEN_GPU_MEASUREMENTS_MISSING"
    )
    assert (
        plan["manager_window_commands"]["status"]
        == "SUPERSEDED_BY_ONE_SESSION_COMMAND"
    )
    assert (
        plan["manager_window_commands"]["independent_authorisation_allowed"]
        is False
    )
    assert list(plan["manager_post_commands"]) == ["W1", "W2", "W3"]
    for window_id, command in plan["manager_post_commands"].items():
        assert command[:2] == ["taskset", "-c"]
        marker = command.index("--manager-post-stage")
        assert command[marker + 1] == window_id
        assert "--lock-release-proof" in command
        assert "--output" in command
    legacy = plan["legacy_amendment4_outer_graph_commands"]
    assert legacy["status"] == "NON_AUTHORITATIVE_SUPERSEDED"
    assert list(legacy["commands"]) == ["W1", "W2", "W3"]
    for window_id, command in legacy["commands"].items():
        marker = command.index("--manager-window-graph")
        assert command[marker + 1] == window_id
        assert "--receipt" in command
        assert "--result" in command
    assert list(plan["manager_window_inspection"]) == ["W1", "W2", "W3"]
    assert all(
        graph["device_imports"] == []
        for graph in plan["manager_window_inspection"].values()
    )
    assert plan["stale_pair"]["authorisation_eligible"] is False
    assert exact_contract.validate_plan(plan)["lock_acquisitions"] == 1


def test_build_and_print_plan_do_not_create_deferred_cache_or_identity_paths(tmp_path):
    plan = executor.build_plan()
    deferred = plan["identity_and_cache_provenance"]
    paths = [
        Path(record["path"])
        for record in deferred.values()
        if isinstance(record, dict) and "path" in record
    ]
    before = {path: os.path.lexists(path) for path in paths}

    env = os.environ.copy()
    env.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(
        [sys.executable, str(executor.SCRIPT), "--print-plan"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    printed = json.loads(completed.stdout)
    assert printed == plan
    assert {path: os.path.lexists(path) for path in paths} == before
    assert list(tmp_path.iterdir()) == []


def test_all_three_windows_and_cpu_pair_execute_end_to_end_as_cpu_stubs():
    payload = executor.dry_run_all()

    assert payload["status"] == "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
    assert payload["scenario"] == "ALL_STUBS_OK"
    assert payload["device_touched"] is False
    assert payload["device_queries"] == []
    assert payload["device_imports"] == []
    assert payload["receipts_read"] == []
    assert payload["receipts_consumed"] == []
    assert payload["cache_snapshots_created"] == []
    assert [entry["phase"] for entry in payload["results"]] == [
        "CPU_PREFLIGHT",
        "W1",
        "W2",
        "W3",
    ]
    assert payload["executed_stub_stage_keys"] == ORDERED_STAGE_KEYS
    assert all(
        stage_status == "OK"
        for _key, stage_status in _result_stage_keys(payload)
    )


@pytest.mark.parametrize("failure_key", ORDERED_STAGE_KEYS)
def test_every_injected_failure_suppresses_all_later_gpu_stages(failure_key):
    payload = executor.dry_run_all(inject_failure=failure_key)
    index = ORDERED_STAGE_KEYS.index(failure_key)

    assert payload["executed_stub_stage_keys"] == ORDERED_STAGE_KEYS[: index + 1]
    observed = dict(_result_stage_keys(payload))
    assert observed[failure_key] == "FAILED"
    for later in ORDERED_STAGE_KEYS[index + 1 :]:
        assert observed[later] == "SUPPRESSED"
    assert payload["device_touched"] is False
    assert payload["receipts_consumed"] == []
    assert payload["gpu_evidence_status"] == "MISSING"


def test_device_stage_refuses_before_device_import_or_writing_result(tmp_path):
    receipt = tmp_path / "must-not-be-read.json"
    result = tmp_path / "must-not-be-written.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(executor.SCRIPT),
            "--manager-device-stage",
            "W1",
            "--run-id",
            "future-run-0001",
            "--receipt",
            str(receipt),
            "--result",
            str(result),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["receipt_consumed"] is False
    assert payload["device_touched"] is False
    assert payload["jax_imported"] is False


def test_legacy_outer_graph_refuses_without_reading_receipt(tmp_path):
    receipt = tmp_path / "must-not-be-read.json"
    result = tmp_path / "must-not-be-written.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(executor.SCRIPT),
            "--manager-window-graph",
            "W1",
            "--receipt",
            str(receipt),
            "--result",
            str(result),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["receipt_read"] is False
    assert payload["receipt_consumed"] is False
    assert payload["device_touched"] is False
    assert payload["jax_imported"] is False
    assert not result.exists()
    assert payload["gpuwrf_imported"] is False
    assert not receipt.exists()
    assert not result.exists()
