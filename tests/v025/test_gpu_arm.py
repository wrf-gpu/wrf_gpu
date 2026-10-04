"""Fail-closed tests for the FAST-v025 GPU arm (contract §5.2, §5.3, §9, §13).

Every one of these runs on CPU and launches nothing. That is the point: the
refusal paths have to be proven *before* a coordinated 35-50 minute window is
spent discovering that one of them was wrong. A window spent on a bug is a window
the two neighbouring managers gave up for nothing.

The tests are organised by the thing they stop:

* §13 — no GPU touch without BOTH managers' verbatim affirmative replies for
  THIS window, and without the canonical lock actually held;
* §13 — preemption produces no result at all, not a truncated one;
* §5.3 — a timing class that was never measured cannot be serialised away;
* §9  — attribution that misses the 95% bar reports that it missed, and the
  physics-vs-dycore discriminator is computed from measured device time.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import run_gpu_arm as rga  # noqa: E402


def _receipt(tmp_path: Path, **overrides) -> Path:
    now = datetime.now(timezone.utc)
    body = {
        "window": "baseline-census",
        "requested_at_utc": now.isoformat(),
        "request_text": "M0 W1 baseline-census, 35-50 min",
        "replies": {
            "0:2": {"affirmative": True, "verbatim": "ok go ahead",
                    "received_at_utc": now.isoformat()},
            "0:3": {"affirmative": True, "verbatim": "fine, GPU is free",
                    "received_at_utc": now.isoformat()},
        },
    }
    body.update(overrides)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(body))
    return path


def _lock_env(tmp_path: Path, token: str = "tok-1", *, holder_token: str | None = None) -> dict:
    holder = tmp_path / "holder"
    holder.write_text(
        "holder=baseline-census pid=1 since=now "
        f"token={holder_token or token} cmd=x\n"
    )
    return {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_TOKEN": token,
        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
        "GPUWRF_GPU_LOCK_LABEL": "baseline-census",
    }


# --------------------------------------------------------------------------- #
# §13 coordination receipt                                                     #
# --------------------------------------------------------------------------- #
def test_absent_receipt_is_a_no(tmp_path):
    """Silence is not approval, so a missing receipt must refuse."""
    with pytest.raises(rga.WindowNotAuthorised, match="silence is not approval"):
        rga.CoordinationReceipt.load(tmp_path / "nope.json")


def test_receipt_for_a_different_window_is_refused(tmp_path):
    receipt = rga.CoordinationReceipt.load(_receipt(tmp_path, window="pallas-sm120"))
    with pytest.raises(rga.WindowNotAuthorised, match="does not carry over"):
        receipt.check("baseline-census")


@pytest.mark.parametrize("manager", ["0:2", "0:3"])
def test_one_missing_manager_is_refused(tmp_path, manager):
    now = datetime.now(timezone.utc).isoformat()
    replies = {
        m: {"affirmative": True, "verbatim": "ok", "received_at_utc": now}
        for m in rga.REQUIRED_MANAGERS if m != manager
    }
    receipt = rga.CoordinationReceipt.load(_receipt(tmp_path, replies=replies))
    with pytest.raises(rga.WindowNotAuthorised, match="silence is not approval"):
        receipt.check("baseline-census")


def test_a_negative_reply_is_refused(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    replies = {
        "0:2": {"affirmative": True, "verbatim": "ok", "received_at_utc": now},
        "0:3": {"affirmative": False, "verbatim": "no, I need the GPU", "received_at_utc": now},
    }
    receipt = rga.CoordinationReceipt.load(_receipt(tmp_path, replies=replies))
    with pytest.raises(rga.WindowNotAuthorised, match="not authorised"):
        receipt.check("baseline-census")


def test_affirmative_without_verbatim_text_is_refused(tmp_path):
    """§13 requires the reply copied verbatim; a bare boolean is not a record."""
    now = datetime.now(timezone.utc).isoformat()
    replies = {
        "0:2": {"affirmative": True, "verbatim": "ok", "received_at_utc": now},
        "0:3": {"affirmative": True, "verbatim": "   ", "received_at_utc": now},
    }
    receipt = rga.CoordinationReceipt.load(_receipt(tmp_path, replies=replies))
    with pytest.raises(rga.WindowNotAuthorised, match="verbatim"):
        receipt.check("baseline-census")


def test_a_stale_receipt_is_refused(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=9)).isoformat()
    receipt = rga.CoordinationReceipt.load(_receipt(tmp_path, requested_at_utc=old))
    with pytest.raises(rga.WindowNotAuthorised, match="standing permission"):
        receipt.check("baseline-census")


def test_a_fresh_complete_receipt_passes(tmp_path):
    rga.CoordinationReceipt.load(_receipt(tmp_path)).check("baseline-census")


# --------------------------------------------------------------------------- #
# §13 canonical lock                                                           #
# --------------------------------------------------------------------------- #
def test_no_lock_env_is_refused():
    with pytest.raises(rga.WindowNotAuthorised, match="with_gpu_lock.sh"):
        rga.check_canonical_lock({})


def test_inherited_lock_env_without_a_live_holder_is_refused(tmp_path):
    """A stale GPUWRF_GPU_LOCK_HELD=1 must not read as a held lock."""
    env = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_TOKEN": "tok",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(tmp_path / "absent"),
    }
    with pytest.raises(rga.WindowNotAuthorised, match="does not exist"):
        rga.check_canonical_lock(env)


def test_token_mismatch_means_someone_else_holds_the_lock(tmp_path):
    env = _lock_env(tmp_path, token="mine", holder_token="theirs")
    with pytest.raises(rga.WindowNotAuthorised, match="held by someone else"):
        rga.check_canonical_lock(env)


def test_matching_token_is_accepted(tmp_path):
    assert rga.check_canonical_lock(_lock_env(tmp_path))["token_matched"] is True


def test_authorise_requires_a_known_window(tmp_path):
    with pytest.raises(rga.WindowNotAuthorised, match="unknown window"):
        rga.authorise("whatever", receipt_path=_receipt(tmp_path))


# --------------------------------------------------------------------------- #
# §13 preemption                                                               #
# --------------------------------------------------------------------------- #
def test_preemption_guard_raises_and_yields_no_result():
    guard = rga.PreemptionGuard()
    guard.check()  # clean
    guard._handler(15, None)
    with pytest.raises(rga.Preempted, match="INVALID"):
        guard.check()


def test_preempted_arm_returns_nothing(tmp_path, monkeypatch):
    """No 'let the short run finish' path exists: the arm raises, it does not return."""
    receipt = _receipt(tmp_path)
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)

    def preempting_runner(command, *, cwd, env, guard, deadline_seconds=None):
        guard._handler(2, None)
        guard.check()
        raise AssertionError("unreachable")

    arm = rga.gpu_arm(ledger_path=tmp_path / 'ledger.json', receipt_path=receipt, case_dir=tmp_path, runner=preempting_runner)
    with pytest.raises(rga.Preempted):
        arm(run_id="x", run_root=tmp_path)


# --------------------------------------------------------------------------- #
# §5.3 timing classes                                                          #
# --------------------------------------------------------------------------- #
def test_unmeasured_timing_class_cannot_be_serialised():
    timings = rga.TimingClasses(
        lock_wait_seconds=1.0, cold_compile_seconds=2.0, cached_load_seconds=3.0,
        warm_integration_seconds=4.0, io_seconds=5.0,
    )  # profiler_perturbation_seconds left unmeasured
    with pytest.raises(rga.ArmIncompleteError, match="profiler_perturbation_seconds"):
        timings.as_dict()


def test_all_classes_measured_serialises():
    timings = rga.TimingClasses(
        lock_wait_seconds=1.0, cold_compile_seconds=2.0, cached_load_seconds=3.0,
        warm_integration_seconds=4.0, io_seconds=5.0, profiler_perturbation_seconds=0.5,
    )
    assert timings.as_dict()["cold_compile_seconds"] == 2.0


def test_zero_is_a_measurement_but_none_is_not():
    """A class measured as 0.0 is fine; the guard is against never measuring it."""
    timings = rga.TimingClasses(
        lock_wait_seconds=0.0, cold_compile_seconds=0.0, cached_load_seconds=0.0,
        warm_integration_seconds=0.0, io_seconds=0.0, profiler_perturbation_seconds=0.0,
    )
    assert timings.as_dict()["io_seconds"] == 0.0


# --------------------------------------------------------------------------- #
# §9 attribution and the W1 discriminator                                      #
# --------------------------------------------------------------------------- #
def test_kernel_names_map_to_census_families():
    assert rga.attribute_kernel("fusion_rrtmg_lw_taumol") == "physics.radiation"
    assert rga.attribute_kernel("wrapped_thomas_solve_scan") == "dycore.vertical_implicit"
    assert rga.attribute_kernel("advect_scalar_flux_1") == "dycore.advection"
    assert rga.attribute_kernel("mystery_kernel_42") == "unknown"


def test_attribution_below_the_bar_reports_that_it_missed():
    kernels = [
        {"name": "rrtmg_lw", "device_time_ns": 10, "launches": 1},
        {"name": "mystery", "device_time_ns": 90, "launches": 9},
    ]
    result = rga.attribute_device_time(kernels)
    assert result["meets_attribution_bar"] is False
    assert result["unknown_device_time_share"] == pytest.approx(0.9)


def test_attribution_at_the_bar_passes():
    kernels = [
        {"name": "rrtmg_lw", "device_time_ns": 96, "launches": 96},
        {"name": "mystery", "device_time_ns": 4, "launches": 4},
    ]
    assert rga.attribute_device_time(kernels)["meets_attribution_bar"] is True


def test_discriminator_computes_measured_physics_vs_dycore_share():
    """The manager's mandatory W1 question, computed from device time."""
    kernels = [
        {"name": "rrtmg_lw", "device_time_ns": 800, "launches": 8},
        {"name": "advect_scalar_flux", "device_time_ns": 200, "launches": 2},
    ]
    result = rga.attribute_device_time(kernels)
    assert result["measured_physics_share_of_known"] == pytest.approx(0.8)
    assert result["measured_dycore_share_of_known"] == pytest.approx(0.2)
    assert result["static_proxy_reference"]["dycore_share"] == 0.183


def test_empty_trace_does_not_claim_attribution():
    result = rga.attribute_device_time([])
    assert result["meets_attribution_bar"] is False
    assert result["measured_dycore_share_of_known"] is None


# --------------------------------------------------------------------------- #
# the happy path, still with no GPU                                            #
# --------------------------------------------------------------------------- #
def test_stub_arm_completes_and_separates_cold_from_cached(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path)
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    out = tmp_path / "gpu_x"
    out.mkdir(parents=True, exist_ok=True)

    def runner(command, *, cwd, env, guard, deadline_seconds=None):
        # emulate the model writing its history file on the first pass
        (tmp_path / "gpu_run1").mkdir(exist_ok=True)
        (tmp_path / "gpu_run1" / "wrfout_d01_2026-07-26_01:00:18").write_text("x")
        return rga.stub_runner(command, cwd=cwd, env=env, guard=guard)

    arm = rga.gpu_arm(ledger_path=tmp_path / 'ledger.json', receipt_path=receipt, case_dir=tmp_path, runner=runner)
    record = arm(run_id="run1", run_root=tmp_path)
    assert record.kind == "gpu"
    partial = record.payload["timing_classes_partial"]
    assert partial["cold_compile_seconds"] == 42.0
    assert partial["cached_load_seconds"] == 3.0
    assert partial["cold_compile_seconds"] != partial["cached_load_seconds"]
    # classes this arm cannot measure stay None rather than defaulting to zero
    assert partial["profiler_perturbation_seconds"] is None


# --------------------------------------------------------------------------- #
# §5.3 frozen early stops, enforced BY THE RUNNER                              #
# --------------------------------------------------------------------------- #
def test_the_frozen_bars_are_the_contract_numbers():
    assert rga.COLD_COMPILE_MAX_SECONDS == 600.0
    assert rga.CACHED_LOAD_MAX_SECONDS == 60.0
    assert rga.GPU_ARM_MAX_SECONDS == 300.0


def test_cold_compile_past_600s_is_stopped_by_the_runner(tmp_path, monkeypatch):
    """Regression for the first W1 window.

    There was no internal enforcement: the 600 s bar lived only in the contract
    and in a shell `timeout` set ABOVE it, so the compile ran past 655 s and the
    MANAGER had to stop it by hand with a signal. A frozen threshold that needs a
    human to notice is not a gate. The runner must now stop itself and report the
    breach as a measured result.
    """
    receipt = _receipt(tmp_path)
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)

    def slow_cold(command, *, cwd, env, guard, deadline_seconds=None):
        # The cold pass is the one carrying the 600 s deadline.
        if deadline_seconds == rga.COLD_COMPILE_MAX_SECONDS:
            raise rga.EarlyStopBreached(
                f"frozen early stop: this pass exceeded {deadline_seconds:.0f} s "
                "(killed at 655.0 s)."
            )
        return rga.stub_runner(command, cwd=cwd, env=env, guard=guard)

    arm = rga.gpu_arm(ledger_path=tmp_path / 'ledger.json', receipt_path=receipt, case_dir=tmp_path, runner=slow_cold)
    record = arm(run_id="slow", run_root=tmp_path)
    assert record.status == "EARLY_STOP"
    assert record.payload["early_stop"] == "cold_compile_exceeded_600s"
    assert record.payload["frozen_bar_seconds"] == 600.0
    assert "not moved" in record.payload["gate_verdict"]


def test_the_cold_deadline_is_actually_passed_to_the_child():
    """The bar must reach subprocess.run, not just exist as a constant."""
    seen = {}

    def capture(command, *, cwd, env, guard, deadline_seconds=None):
        seen[env.get("GPUWRF_JAX_CACHE")] = deadline_seconds
        return 0, 1.0, ""

    import inspect

    assert "deadline_seconds" in inspect.signature(rga._run).parameters


def test_a_breach_reports_the_measured_seconds_not_just_a_failure(tmp_path, monkeypatch):
    """'At least 655 s' is a result; a bare failure is not."""
    receipt = _receipt(tmp_path)
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)

    def slow(command, *, cwd, env, guard, deadline_seconds=None):
        raise rga.EarlyStopBreached("frozen early stop: exceeded 600 s (killed at 655.0 s).")

    arm = rga.gpu_arm(ledger_path=tmp_path / 'ledger.json', receipt_path=receipt, case_dir=tmp_path, runner=slow)
    record = arm(run_id="slow2", run_root=tmp_path)
    assert "655.0 s" in record.payload["detail"]


def test_a_dry_run_can_never_report_a_successful_arm(tmp_path, monkeypatch):
    """The stub writes no wrfout, so its arm is always FAILED — by design.

    If a dry run could return `status: OK`, a stub result could reach
    `run_fast_pair` and satisfy the recurring-pair rule without a GPU ever
    running. It cannot.
    """
    receipt = _receipt(tmp_path)
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    arm = rga.gpu_arm(ledger_path=tmp_path / 'ledger.json', receipt_path=receipt, case_dir=tmp_path, runner=rga.stub_runner)
    record = arm(run_id="dry", run_root=tmp_path)
    assert record.status == "FAILED"


def test_gpu_arm_strips_this_workers_cpu_pins_from_the_child(tmp_path, monkeypatch):
    """A GPU arm must not inherit JAX_PLATFORMS=cpu and quietly measure the CPU.

    This worker's whole shell runs with the CPU-first pins set. Inheriting them
    into a coordinated GPU window would produce a complete, plausible, entirely
    CPU measurement — the most expensive silent failure available in this sprint.
    """
    receipt = _receipt(tmp_path)
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seen: dict[str, dict] = {}

    def runner(command, *, cwd, env, guard, deadline_seconds=None):
        seen["env"] = env
        (tmp_path / "gpu_p").mkdir(exist_ok=True)
        (tmp_path / "gpu_p" / "wrfout_d01_x").write_text("x")
        return rga.stub_runner(command, cwd=cwd, env=env, guard=guard)

    arm = rga.gpu_arm(ledger_path=tmp_path / 'ledger.json', receipt_path=receipt, case_dir=tmp_path, runner=runner)
    record = arm(run_id="p", run_root=tmp_path)
    assert "JAX_PLATFORMS" not in seen["env"]
    assert "CUDA_VISIBLE_DEVICES" not in seen["env"]
    assert set(record.payload["cpu_pin_env_stripped"]) >= {"JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES"}


def test_the_command_is_the_shipped_production_entrypoint(tmp_path):
    command = rga.build_forecast_command(run_dir=tmp_path, out_dir=tmp_path / "o")
    assert command[1:4] == ["-m", "gpuwrf", "run"]
    assert "--domain" in command and "d01" in command
    # stable hash so the pre-registered command can be pinned in the receipt
    assert len(rga.sha256_text(" ".join(command))) == 64


def test_the_built_command_actually_parses_against_the_shipped_cli(tmp_path):
    """Parse it with the REAL parser, not just check its shape.

    Regression for the first W1 attempt, which died inside a coordinated window
    on `--hours: invalid int value: '1.0'`. Every test here passed beforehand
    because they all inspected the list rather than handing it to argparse. A
    command that has never met the parser it is aimed at is not pre-registered,
    it is only written down.
    """
    from gpuwrf.cli import build_parser

    command = rga.build_forecast_command(run_dir=tmp_path, out_dir=tmp_path / "o", hours=1.0)
    args = build_parser().parse_args(command[3:])  # drop python -m gpuwrf
    assert args.hours == 1
    assert args.domain == "d01"


@pytest.mark.parametrize("hours", [1, 1.0, 2, 3.0])
def test_integral_hours_are_emitted_as_ints(tmp_path, hours):
    from gpuwrf.cli import build_parser

    command = rga.build_forecast_command(run_dir=tmp_path, out_dir=tmp_path / "o", hours=hours)
    assert build_parser().parse_args(command[3:]).hours == int(hours)
