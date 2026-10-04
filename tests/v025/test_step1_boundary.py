"""The LIVE boundary: wrapper -> capture -> observed release -> separate analysis.

Manager review on main `8d27c1b3`. The previous driver simulated release, never
installed a `PreemptionGuard`, called `receipt.check` directly with a fourth
non-frozen label, and compared unlike clocks in H1. These tests execute the real
three-process path with stubs and then mutate each boundary condition.

`tests/v025/test_step1_driver.py` covers the artifact plumbing; this file covers
the boundary: coordination, lock lifecycle, preemption/process groups, and the
clock H1 is allowed to divide by.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import h1_discriminator as h1  # noqa: E402
import nvtx_exclusive as nx  # noqa: E402
import run_gpu_arm as arm  # noqa: E402
import step1_driver as drv  # noqa: E402
import step1_dryrun as dry  # noqa: E402
import step1_stub as stub  # noqa: E402

RUN_ID = "dryrun-boundary-0001"

@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    """ONE real execution of the whole three-process path."""
    out = tmp_path_factory.mktemp("boundary")
    proofs = tmp_path_factory.mktemp("proofs")
    original = drv.PROOFS
    # The analysis is a separate process, so the redirect must go through the
    # environment as well as the module attribute.
    os.environ["GPUWRF_STEP1_PROOFS"] = str(proofs)
    drv.PROOFS = proofs
    try:
        record = dry.run(out, run_id=RUN_ID)
    finally:
        drv.PROOFS = original
        os.environ.pop("GPUWRF_STEP1_PROOFS", None)
    return record, Path(record["run_directory"]), proofs


# --------------------------------------------------------------------------- #
# 1. frozen window label, canonical authorisation                              #
# --------------------------------------------------------------------------- #
def test_the_window_label_is_frozen_and_whitelisted():
    assert drv.WINDOW_LABEL == "baseline-census"
    assert drv.WINDOW_LABEL in arm.WINDOWS


def test_the_driver_does_not_call_receipt_check_directly():
    """§13 goes through `arm.authorise`, which also whitelists the window."""
    source = (REPO / "scripts/v025/step1_driver.py").read_text()
    assert "receipt.check(" not in source
    assert "arm.authorise" in source


def test_capture_authorises_through_the_canonical_path(executed):
    _, out, _ = executed
    outcome = json.loads((out / "capture_outcome.json").read_text())
    assert outcome["authorisation"]["window"] == "baseline-census"
    assert outcome["authorisation"]["consumed"] is True
    assert outcome["authorisation"]["lock"]["token_matched"] is True


def test_an_unknown_window_is_refused_by_the_canonical_path(tmp_path):
    with pytest.raises(arm.WindowNotAuthorised, match="unknown window"):
        arm.authorise("step1-attribution", receipt_path=tmp_path / "r.json")


def test_capture_without_a_receipt_refuses(tmp_path):
    with pytest.raises(drv.Step1Blocked, match="no coordination receipt"):
        drv.run_capture(
            out_root=tmp_path / "run-no-receipt", run_id="capture-no-receipt",
            receipt_path=None, stub_stage=["true"],
        )


def test_a_spent_receipt_is_refused_the_second_time(tmp_path):
    """One receipt authorises ONE window."""
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    ledger = tmp_path / "ledger.json"
    holder = tmp_path / "holder.txt"
    holder.write_text("holder=baseline-census pid=1 token=tok cmd=y\n")
    env = {"GPUWRF_GPU_LOCK_HELD": "1", "GPUWRF_GPU_LOCK_TOKEN": "tok",
           "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
           "GPUWRF_GPU_LOCK_LABEL": "baseline-census"}
    arm.authorise(drv.WINDOW_LABEL, receipt_path=receipt, env=env, ledger_path=ledger)
    with pytest.raises(arm.WindowNotAuthorised):
        arm.authorise(drv.WINDOW_LABEL, receipt_path=receipt, env=env, ledger_path=ledger)


def test_capture_refuses_without_the_canonical_lock(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    with pytest.raises(arm.WindowNotAuthorised, match="with_gpu_lock|LOCK_HELD"):
        drv.run_capture(
            out_root=tmp_path / "run-no-lock", run_id="capture-no-lock",
            receipt_path=receipt, stub_stage=["true"],
            env={"PATH": os.environ.get("PATH", "")},
        )


# --------------------------------------------------------------------------- #
# 2. real lock lifecycle: release is OBSERVED, not claimed                      #
# --------------------------------------------------------------------------- #
def test_the_orchestrator_wraps_capture_in_the_canonical_lock_script():
    command = drv.locked_capture_command(
        out_root=Path("/tmp/x"), receipt=Path("/tmp/r"), run_id=RUN_ID
    )
    assert command[0].endswith("with_gpu_lock.sh")
    assert "--label" in command and "baseline-census" in command
    assert "--capture" in command


def test_release_is_evidenced_by_the_wrapper_returning(executed):
    record, _, _ = executed
    assert record["lock_wrapper"]["returned"] is True
    assert record["lock_wrapper"]["returncode"] == 0
    assert record["gpu_released"]["released"] is True
    assert "exited" in record["gpu_released"]["evidence"]
    assert len(record["lock_wrapper"]["log"]["sha256"]) == 64


def test_no_release_callback_survives_in_the_driver():
    """Release must not be something the driver asserts on its own authority."""
    source = (REPO / "scripts/v025/step1_driver.py").read_text()
    assert "release_gpu" not in source


def test_analysis_runs_only_after_the_wrapper_returned(executed):
    record, _, _ = executed
    assert record["analysis"]["ran_after_release"] is True
    assert record["analysis"]["returncode"] == 1
    assert record["execution_status"] == "COMPLETED"
    assert record["gate_status"] == "PARTIAL"


def test_a_wrapper_that_fails_blocks_before_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    ran = []
    record = drv.run_orchestrate(
        out_root=tmp_path / "run", run_id=RUN_ID,
        receipt_path=tmp_path / "r.json", dry_run=True,
        wrapper_runner=lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "boom"),
        analysis_runner=lambda cmd, env: ran.append(cmd) or
        subprocess.CompletedProcess(cmd, 0, "", ""))
    assert record["status"] == "BLOCKED"
    assert record["analysis"]["status"] == "NOT_RUN"
    assert not ran, "analysis must not run when the locked capture failed"


def test_the_analysis_environment_is_sanitised_to_cpu():
    hostile = {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin"}
    env = drv.analysis_environment(hostile)
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_the_analysis_is_cpu_pinned_and_a_separate_process():
    command = drv.analyse_command(out_root=Path("/tmp/x"), run_id=RUN_ID)
    assert command[:2] == ["taskset", "-c"]
    assert "--analyse" in command


def test_the_driver_is_a_coordinated_entry_point_not_a_guarded_one():
    source = (REPO / "scripts/v025/step1_driver.py").read_text()
    assert "import cpu_guard" not in source


def test_successful_outer_runner_preserves_rc_zero_and_hashes_log(tmp_path):
    result = drv._group_runner(
        ["bash", "-c", "printf success"],
        timeout_s=5.0,
        log_path=tmp_path / "outer.log",
    )
    assert result.returncode == 0
    assert result.stdout == "success"
    assert len(result.log_sha256) == 64


# --------------------------------------------------------------------------- #
# 3. preemption and process groups                                             #
# --------------------------------------------------------------------------- #
def _locked_env(tmp_path):
    holder = tmp_path / "holder.txt"
    holder.write_text("holder=baseline-census pid=1 token=tok cmd=y\n")
    return dict(os.environ, GPUWRF_GPU_LOCK_HELD="1", GPUWRF_GPU_LOCK_TOKEN="tok",
                GPUWRF_GPU_LOCK_HOLDER_FILE=str(holder),
                GPUWRF_GPU_LOCK_LABEL="baseline-census")


def test_the_stage_runs_in_its_own_process_group(executed):
    """Killing only the direct child would orphan the forecast on the GPU."""
    source = (REPO / "scripts/v025/step1_driver.py").read_text()
    assert "start_new_session=True" in source
    assert "os.killpg" in source


def test_a_timeout_kills_the_whole_group_and_leaves_no_orphan(tmp_path):
    """A child that spawns a grandchild and ignores SIGTERM must still die."""
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    script = tmp_path / "spawner.sh"
    script.write_text(
        "#!/bin/bash\ntrap '' TERM\n"        # ignore SIGTERM: only SIGKILL ends it
        "sleep 300 &\n"                       # a grandchild in the same group
        "sleep 300\n")
    script.chmod(0o755)

    outcome = drv.run_capture(out_root=tmp_path / "run-timeout", run_id="capture-timeout",
                              receipt_path=receipt,
                              env=_locked_env(tmp_path), stub_stage=[str(script)],
                              timeout_s=2.0, ledger_hint=tmp_path / "ledger.json")
    assert outcome["killed"] is True
    assert "budget" in outcome["killed_reason"]
    assert outcome["orphans"] == [], f"orphans survived: {outcome['orphans']}"
    assert outcome["status"] == "FAILED"


def test_the_receipt_stays_spent_after_a_killed_capture(tmp_path):
    """A window that is killed still consumed its coordination."""
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    ledger = tmp_path / "ledger.json"
    outcome = drv.run_capture(out_root=tmp_path / "run-spent", run_id="capture-spent",
                              receipt_path=receipt,
                              env=_locked_env(tmp_path), stub_stage=["sleep", "300"],
                              timeout_s=1.5, ledger_hint=ledger)
    assert outcome["killed"] is True
    assert outcome["receipt_remains_spent"] is True
    assert json.loads(ledger.read_text()), "the spend ledger must carry the fingerprint"
    with pytest.raises(arm.WindowNotAuthorised):
        arm.authorise(drv.WINDOW_LABEL, receipt_path=receipt,
                      env=_locked_env(tmp_path), ledger_path=ledger)


def test_a_preemption_signal_kills_the_group(tmp_path):
    """The standing rule: relinquish the GPU the moment it is requested."""
    receipt = dry.write_receipt(tmp_path / "receipt.json")

    class Preempted(arm.PreemptionGuard):
        def check(self):
            raise RuntimeError("manager requested the GPU")

    outcome = drv.run_capture(out_root=tmp_path / "run-preempt", run_id="capture-preempt",
                              receipt_path=receipt,
                              env=_locked_env(tmp_path), stub_stage=["sleep", "300"],
                              timeout_s=60.0, ledger_hint=tmp_path / "ledger.json",
                              guard=Preempted())
    assert outcome["killed"] is True
    assert "preemption" in outcome["killed_reason"]
    assert outcome["orphans"] == []


def test_large_capture_stdout_cannot_fill_a_pipe_and_false_timeout(tmp_path):
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    command = [
        sys.executable, "-c",
        "import sys; sys.stdout.write('x' * 2_000_000); sys.stdout.flush()",
    ]
    out = tmp_path / "run-large-log"
    outcome = drv.run_capture(
        out_root=out, run_id="capture-large-log", receipt_path=receipt,
        env=_locked_env(tmp_path), stub_stage=command, timeout_s=10.0,
        ledger_hint=tmp_path / "ledger.json",
    )
    assert outcome["status"] == "OK"
    assert outcome["capture_log"]["bytes"] == 2_000_000
    assert len(outcome["capture_log"]["sha256"]) == 64


def test_a_killed_capture_blocks_the_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    out = tmp_path / "run"
    out.mkdir(parents=True)
    stub.write_trace(out / "step1_pushpop.csv")
    stub.write_hlo_dump(out / "dump")
    (out / "capture_outcome.json").write_text(json.dumps(
        {"status": "FAILED", "killed": True, "killed_reason": "budget exceeded",
         "orphans": [], "run_id": RUN_ID}))
    record = drv.run_analyse(out_root=out, run_id=RUN_ID, dry_run=True)
    assert record["status"] == "BLOCKED"
    for key in ("H1", "H2", "H3"):
        assert record[key]["verdict"] == "SUPPRESSED"


def test_an_orphan_blocks_the_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    out = tmp_path / "run"
    out.mkdir(parents=True)
    stub.write_trace(out / "step1_pushpop.csv")
    stub.write_hlo_dump(out / "dump")
    (out / "capture_outcome.json").write_text(json.dumps(
        {"status": "OK", "killed": False, "orphans": [12345], "run_id": RUN_ID}))
    record = drv.run_analyse(out_root=out, run_id=RUN_ID, dry_run=True)
    assert record["status"] == "BLOCKED"
    assert "orphan" in record["blocked_reason"]


def test_a_missing_capture_outcome_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    out = tmp_path / "run"
    out.mkdir(parents=True)
    record = drv.run_analyse(out_root=out, run_id=RUN_ID, dry_run=True)
    assert record["status"] == "BLOCKED"
    assert "capture_outcome" in record["blocked_reason"]


def test_existing_run_directory_is_refused_before_receipt_spend(tmp_path):
    out = tmp_path / "existing-run"
    out.mkdir()
    (out / "stale.txt").write_text("old evidence")
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    ledger = tmp_path / "ledger.json"
    with pytest.raises(drv.Step1Blocked, match="already exists"):
        drv.run_capture(
            out_root=out, run_id="capture-stale-run", receipt_path=receipt,
            env=_locked_env(tmp_path), stub_stage=["true"], ledger_hint=ledger,
        )
    assert not ledger.exists(), "invalid output scope must fail before spending consent"


# --------------------------------------------------------------------------- #
# 4. H1's clock                                                                 #
# --------------------------------------------------------------------------- #
def test_h1_blocks_when_t_off_is_on_the_wrong_clock():
    assert h1.verdict(100.0, completed=True, t_off_basis="nvtx-span")["verdict"] == "BLOCKED"
    assert h1.verdict(100.0, completed=True)["verdict"] == "BLOCKED"


def test_the_required_basis_matches_between_the_two_modules():
    assert nx.SESSION_ORIGIN_UPPER_BOUND_BASIS == h1.UPPER_BOUND_BASIS


def test_t_off_keeps_process_startup(executed):
    """The defect: an NVTX span drops startup and inflates the removable share."""
    _, _, proofs = executed
    record = json.loads((proofs / "step1_attribution_dryrun.json").read_text())
    derivation = record["H1"]["t_off_derivation"]
    assert derivation["startup_before_first_range_seconds"] > 0, (
        "the stub must contain startup, or this proves nothing"
    )
    assert record["H1"]["t_off_seconds"] == pytest.approx(
        derivation["first_to_last_range_span_seconds"]
        + derivation["startup_before_first_range_seconds"])
    assert record["H1"]["t_off_seconds"] > derivation["first_to_last_range_span_seconds"]


def test_the_clocks_are_named_in_the_verdict(executed):
    _, _, proofs = executed
    record = json.loads((proofs / "step1_attribution_dryrun.json").read_text())
    assert record["H1"]["t_off_basis"] == h1.UPPER_BOUND_BASIS
    assert record["H1"]["measurement_kind"] == "upper_bound"
    assert record["H1"]["product_compile_gate_eligible"] is False
    assert record["H1"]["t_on_basis"] == "process-launch-to-kill"


def test_dropping_startup_would_change_the_verdict_direction():
    """Why the units error mattered: it moves the answer, not just the decimals."""
    span_only = h1.removable_lower_bound(420.0, h1.T_ON_LOWER_BOUND_S)
    with_startup = h1.removable_lower_bound(420.0 + 120.0, h1.T_ON_LOWER_BOUND_S)
    assert span_only >= h1.BAND_PARTIAL > with_startup


# --------------------------------------------------------------------------- #
# 5. baseline-census from the same paid capture                                 #
# --------------------------------------------------------------------------- #
def test_the_census_is_produced_by_the_same_window(executed):
    """One window, both deliverables -- and it reports BLOCKED honestly."""
    _, _, proofs = executed
    record = json.loads((proofs / "step1_attribution_dryrun.json").read_text())
    census = record["baseline_census"]
    # Dry scope/reference fixtures are not production evidence, and the capture
    # takes no matched profiler pair or residency series. BLOCKED is correct.
    assert census["status"] == "BLOCKED"
    assert set(census["gates_not_ok"]) == {
        "production_representativeness",
        "transfer_audit",
        "profiler_perturbation",
        "vram",
    }


def test_dry_arithmetic_runs_but_production_evidence_stays_missing(executed):
    _, _, proofs = executed
    record = json.loads((proofs / "step1_attribution_dryrun.json").read_text())
    census = record["baseline_census"]
    assert record["integration_scope"]["boundary_source"] == "nvtx_pushpop_trace"
    assert record["integration_scope"]["dry_run_stub"] is True
    assert census["device_time_attribution"]["status"] == "OK"
    assert census["production_representativeness"]["status"] == "MISSING"
    assert "dry-run fixture" in census["production_representativeness"]["reason"]
    assert census["a6_static_proxy_rank_diagnostic"]["status"] == "DIAGNOSTIC"
    assert census["transfer_audit"]["status"] == "MISSING"
    assert "dry-run fixture" in census["transfer_audit"]["reason"]


def test_the_census_checks_both_95_percent_bars(executed):
    _, _, proofs = executed
    attribution = (json.loads((proofs / "step1_attribution_dryrun.json").read_text())
                   ["baseline_census"]["device_time_attribution"])
    assert attribution["attributed_launch_share"] >= 0.95
    assert attribution["attributed_device_time_share"] >= 0.95
    assert attribution["meets_attribution_bar"] is True


def test_the_census_carries_the_static_proxy_discriminator(executed):
    _, _, proofs = executed
    attribution = (json.loads((proofs / "step1_attribution_dryrun.json").read_text())
                   ["baseline_census"]["device_time_attribution"])
    assert attribution["static_proxy_reference"]["physics_share"] == 0.817
    assert attribution["measured_physics_share_of_known"] is not None


def test_a_blocked_census_makes_the_analysis_partial_not_ok(executed):
    """OK must not be reachable while a deliverable is missing."""
    _, _, proofs = executed
    record = json.loads((proofs / "step1_attribution_dryrun.json").read_text())
    assert record["status"] == "PARTIAL"
    assert "baseline_census" in record["deliverables_not_ok"]


def test_top_level_status_cannot_hide_missing_required_gates(executed):
    record, _, _ = executed
    assert record["execution_status"] == "COMPLETED"
    assert record["gate_status"] == "PARTIAL"
    assert record["status"] == "PARTIAL"
    assert record["analysis"]["returncode"] != 0


def test_the_live_exporter_was_invoked_not_bypassed(executed):
    """The stub replaces the nsys PROCESS; the export path itself must run."""
    _, _, proofs = executed
    export = json.loads((proofs / "step1_attribution_dryrun.json").read_text())["nsys_export"]
    assert export["status"] == "OK"
    for name, entry in export["exports"].items():
        assert entry["command"].startswith("nsys stats --report ")
        assert len(entry["sha256"]) == 64


def test_low_attribution_blocks_the_census_rather_than_scoring_it():
    import baseline_census as bc
    kernels = [{"name": "mangled_kernel_xyz", "device_time_ns": 10e9, "launches": 100},
               {"name": "rrtmg_lw_kernel", "device_time_ns": 1e6, "launches": 1}]
    census = bc.build(kernels=kernels)
    assert census["status"] == "BLOCKED"
    assert census["device_time_attribution"]["status"] == "FAILED"
    assert census["production_representativeness"]["status"] == "MISSING"


def test_the_nsys_command_requests_memory_usage():
    command = drv.nsys_command(["true"], out_root=Path("/tmp/x"))
    assert "--cuda-memory-usage" in command


def test_current_window_explicitly_does_not_budget_a_perturbation_pair():
    plan = drv.profiler_matched_pair_plan()
    assert plan["sum_of_frozen_caps_seconds"] == 1640.0
    assert plan["current_window_deadline_seconds"] == 1500.0
    assert plan["current_window_eligible"] is False


def test_vram_plan_keeps_allocator_read_in_forecast_process():
    plan = drv.vram_measurement_plan()
    assert "forecast-process" in plan["decomposition"]
    assert "lock owner" in plan["product_metric"]
    assert "must not import JAX" in plan["forbidden"]




# --------------------------------------------------------------------------- #
# 6. reproducibility of the committed artifact                                  #
# --------------------------------------------------------------------------- #
def test_the_emitted_object_contains_no_absolute_run_paths(executed):
    """A reviewer's run with a different --out-root must not dirty the tree."""
    _, out, proofs = executed
    for name in ("step1_attribution_dryrun.json", "step1_orchestration_dryrun.json"):
        text = (proofs / name).read_text()
        assert str(out) not in text
        assert "<OUT_ROOT>" in text


# --------------------------------------------------------------------------- #
# 7. the OUTER wrapper is fail-safe too (manager review, main c227cfc7)         #
# --------------------------------------------------------------------------- #
def test_the_lock_is_acquired_with_zero_wait():
    """After an explicit handover, contention is a coordination failure.

    Waiting would silently queue and then take whatever gap opened next -- the
    §13 violation already committed once in this sprint.
    """
    assert drv.LOCK_TIMEOUT_S == 0
    command = drv.locked_capture_command(
        out_root=Path("/tmp/x"), receipt=Path("/tmp/r"), run_id=RUN_ID
    )
    assert command[1] == "--timeout" and command[2] == "0"


def test_the_outer_wrapper_runs_in_its_own_process_group():
    source = (REPO / "scripts/v025/step1_driver.py").read_text()
    assert "_group_runner" in source
    assert source.count("start_new_session=True") >= 2, (
        "both the inner capture AND the outer wrapper need their own group"
    )


def test_the_outer_runner_kills_a_trapping_child_and_its_grandchild(tmp_path):
    """The same mutation proof as the inner boundary, one level up.

    `subprocess.run(timeout=...)` would kill only `with_gpu_lock.sh`, leaving the
    capture -- and the forecast under it -- alive on the GPU with nobody holding
    the lock.
    """
    marker = tmp_path / "grandchild_alive"
    script = tmp_path / "wrapper.sh"
    script.write_text(
        "#!/bin/bash\ntrap '' TERM\n"
        f"( while true; do touch {marker}; sleep 0.2; done ) &\n"
        "sleep 300\n")
    script.chmod(0o755)

    started = time.monotonic()
    result = drv._group_runner([str(script)], timeout_s=2.0)
    assert time.monotonic() - started < 30, "the deadline must actually fire"
    assert result.returncode != 0
    assert "killed the wrapper process group" in result.stdout

    # the grandchild must be gone: no further touches after the kill
    marker.unlink(missing_ok=True)
    time.sleep(1.0)
    assert not marker.exists(), "a grandchild survived the outer group kill"
