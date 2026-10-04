"""CPU-only gates for manager amendments 01 and 5."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m0_core_session_protocol as session  # noqa: E402
import m0_exact_boundary_contract as exact  # noqa: E402
import m0_long_run_controls as longrun  # noqa: E402


def _cold(
    index: int,
    *,
    passed: bool,
    readiness: float = 599.0,
) -> dict:
    return {
        "process_id": 10_000 + index,
        "cache_path": f"/unique-empty-normal-cache-{index}",
        "readiness_seconds": readiness if passed else None,
        "threshold_stop": not passed,
    }


@pytest.fixture
def pure_session(monkeypatch):
    """The full suite intentionally imports gpuwrf in unrelated earlier tests."""

    monkeypatch.setattr(session, "assert_accelerator_free", lambda: None)


def test_session_entrypoint_is_accelerator_free_in_a_fresh_process():
    code = f"""
import json,sys
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_core_session_protocol as s
value=s.classify_cold_attempts([
 {{"process_id":1,"cache_path":"/a","readiness_seconds":1.0,
   "threshold_stop":False}},
 {{"process_id":2,"cache_path":"/b","readiness_seconds":2.0,
   "threshold_stop":False}}
])
print(json.dumps({{"decision":value["decision"],
 "contaminated":s._loaded if hasattr(s,"_loaded") else []}}))
"""
    environment = dict(os.environ)
    environment.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["decision"] == "PASS"


@pytest.mark.parametrize(
    ("outcomes", "decision", "attempts"),
    [
        ((True, True), "PASS", 2),
        ((False, False), "FAIL", 2),
        ((True, False, True), "PASS", 3),
        ((False, True, False), "FAIL", 3),
    ],
)
def test_frozen_two_of_three_cold_decision(
    outcomes, decision, attempts, pure_session
):
    payload = session.classify_cold_attempts(
        [_cold(index, passed=value) for index, value in enumerate(outcomes, 1)]
    )
    assert payload["decision"] == decision
    assert payload["attempts_used"] == attempts
    assert payload["threshold_seconds"] == 600.0


def test_third_cold_process_is_forbidden_after_first_two_decide(pure_session):
    with pytest.raises(session.SessionRefusal, match="third cold process"):
        session.classify_cold_attempts(
            [_cold(1, passed=True), _cold(2, passed=True), _cold(3, passed=False)]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda attempts: attempts[1].update(
            process_id=attempts[0]["process_id"]
        ),
        lambda attempts: attempts[1].update(
            cache_path=attempts[0]["cache_path"]
        ),
        lambda attempts: attempts[0].update(
            readiness_seconds=600.000001
        ),
        lambda attempts: attempts[0].update(
            readiness_seconds=None, threshold_stop=False
        ),
        lambda attempts: attempts[0].update(threshold_stop=True),
    ],
)
def test_cold_identity_and_threshold_mutations_fail_closed(
    mutation, pure_session
):
    attempts = [_cold(1, passed=True), _cold(2, passed=False)]
    mutation(attempts)
    with pytest.raises(session.SessionRefusal):
        session.classify_cold_attempts(attempts)


def test_first_cold_miss_continues_but_definitive_w1_failure_suppresses(
    pure_session,
):
    outcomes = [False, True, True]
    observed: list[int | str] = []

    def cold_runner(index):
        observed.append(index)
        return _cold(index, passed=outcomes[index - 1])

    def stage_runner(name):
        observed.append(name)
        return {"status": "CPU_STUB_PASS"}

    payload = session.execute_session_graph(
        cold_runner=cold_runner,
        stage_runner=stage_runner,
    )
    assert payload["status"] == "PASS"
    assert observed[:3] == [1, 2, 3]
    assert observed[3:] == [
        "W1_CACHED_AND_CORRECTNESS",
        session.C1_STAGE,
        "W2_PROFILED_CAPTURE",
        "W3_CLEAN_MATCHED_ARM",
    ]

    observed.clear()
    outcomes[:] = [False, False, True]
    payload = session.execute_session_graph(
        cold_runner=cold_runner,
        stage_runner=stage_runner,
    )
    assert payload["status"] == "BLOCKED"
    assert observed == [1, 2]
    assert payload["suppressed"] == [
        "W1_CACHED_AND_CORRECTNESS",
        session.C1_STAGE,
        "W2",
        "W3",
    ]


def test_safety_failure_always_suppresses_later_session_stages(pure_session):
    payload = session.execute_session_graph(
        cold_runner=lambda index: _cold(index, passed=True),
        stage_runner=lambda name: (
            (_ for _ in ()).throw(RuntimeError("identity changed"))
            if name == "W2_PROFILED_CAPTURE"
            else {"status": "PASS"}
        ),
    )
    assert payload["status"] == "BLOCKED"
    assert payload["first_failure"]["stage"] == "W2_PROFILED_CAPTURE"
    assert payload["suppressed"] == ["W3"]


def _receipt() -> dict:
    payload = {
        "window": session.SESSION_LABEL,
        "requested_at_utc": "2026-07-28T00:00:00+00:00",
        "replies": {
            manager: {
                "affirmative": True,
                "verbatim": f"{manager} approves exact session",
                "received_at_utc": "2026-07-28T00:00:01+00:00",
            }
            for manager in session.REQUIRED_MANAGERS
        },
        "spent": None,
    }
    return payload


def test_session_receipt_requires_both_verbatim_and_is_content_addressed(
    pure_session,
):
    receipt = _receipt()
    accepted = session.validate_receipt_shape(receipt)
    assert accepted["status"] == "PASS"
    receipt["fingerprint"] = accepted["fingerprint"]
    assert session.validate_receipt_shape(receipt) == accepted

    for manager in session.REQUIRED_MANAGERS:
        mutated = copy.deepcopy(receipt)
        mutated["replies"][manager]["verbatim"] = ""
        with pytest.raises(session.SessionRefusal, match=manager):
            session.validate_receipt_shape(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["spent"] = {"spent_at_utc": "2026-07-28T00:00:02+00:00"}
    with pytest.raises(session.SessionRefusal, match="already spent"):
        session.validate_receipt_shape(mutated)


def test_every_exact_long_run_payload_validator_passes_representative_fixture():
    result = exact.validate_representative_long_run_fixtures()
    assert result["status"] == "PASS"
    assert len(result["validators_exercised"]) == 6
    assert all(
        value["status"] == "PASS" for value in result["results"].values()
    )


def test_long_run_inhibitor_and_clock_continuity_are_exact():
    command, record = longrun.inhibited_command(
        ["CPU_ONLY_LONG_ORACLE"],
        estimated_seconds=600.0,
    )
    assert command[:3] == [
        longrun.SYSTEMD_INHIBIT,
        "--what=sleep:idle:handle-lid-switch",
        "--mode=block",
    ]
    assert command[-1] == "CPU_ONLY_LONG_ORACLE"
    assert record["status"] == "PASS"

    start = {
        "boot_id": "boot-a",
        "boottime_minus_monotonic_ns": 100,
        "monotonic_midpoint_ns": 1_000,
    }
    end = {
        "boot_id": "boot-a",
        "boottime_minus_monotonic_ns": 100,
        "monotonic_midpoint_ns": 2_000,
    }
    assert longrun.validate_clock_continuity(start, end)["status"] == "PASS"
    end["boottime_minus_monotonic_ns"] += (
        longrun.CLOCK_OFFSET_TOLERANCE_NS + 1
    )
    with pytest.raises(longrun.LongRunRefusal, match="CLOCK_BOOTTIME"):
        longrun.validate_clock_continuity(start, end)


def test_long_run_memory_preflight_rejects_both_resource_failures(
    tmp_path,
    monkeypatch,
):
    meminfo = tmp_path / "meminfo"
    proc = tmp_path / "proc"
    proc.mkdir()
    meminfo.write_text(
        "MemTotal:       100000 kB\n"
        "MemAvailable:    90000 kB\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        longrun, "_lineage_pids", lambda _pid=None: {os.getpid()}
    )
    passed = longrun.memory_preflight(
        expected_peak_bytes=40_000_000,
        meminfo_path=meminfo,
        proc_root=proc,
    )
    assert passed["status"] == "PASS"
    assert passed["required_available_bytes"] == 80_000_000

    meminfo.write_text(
        "MemTotal:       100000 kB\n"
        "MemAvailable:    70000 kB\n",
        encoding="utf-8",
    )
    with pytest.raises(longrun.LongRunRefusal, match="MemAvailable"):
        longrun.memory_preflight(
            expected_peak_bytes=40_000_000,
            meminfo_path=meminfo,
            proc_root=proc,
        )

    meminfo.write_text(
        "MemTotal:       100000 kB\n"
        "MemAvailable:    90000 kB\n",
        encoding="utf-8",
    )
    foreign = proc / "424242"
    foreign.mkdir()
    (foreign / "status").write_text(
        "Name:\tforeign\nPid:\t424242\nPPid:\t1\nVmRSS:\t25000 kB\n",
        encoding="utf-8",
    )
    with pytest.raises(longrun.LongRunRefusal, match="foreign process"):
        longrun.memory_preflight(
            expected_peak_bytes=40_000_000,
            meminfo_path=meminfo,
            proc_root=proc,
        )


def test_exact_output_directory_preflight_rejects_precreated_target(tmp_path):
    target = tmp_path / "wrfout"
    assert longrun.output_directory_preflight(target)["status"] == "PASS"
    target.mkdir()
    with pytest.raises(longrun.LongRunRefusal, match="must be absent"):
        longrun.output_directory_preflight(target)
