"""CPU-only proofs for the Amendment-6 outer owner and held session.

Every test here runs device-denied.  The device callbacks are injected CPU
stubs, the receipts and ledgers live only in ``tmp_path``, and the mutation
matrix asserts that each attack is *refused*, not merely reported.
"""

from __future__ import annotations

import fcntl
import json
import hashlib
import os
import subprocess
import sys
import time
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m0_core_session_protocol as session  # noqa: E402
import m0_exact_boundary_contract as exact  # noqa: E402
import m0_hlo_live_buffers as livebuf  # noqa: E402
import m0_postlock_census as postlock  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402
import m0_window_parent as window_parent  # noqa: E402
import nsys_export as nex  # noqa: E402
import parse_profiler as pp  # noqa: E402


DEVICE_FREE_ENV = {
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_SOURCE_AUTHORITY_CACHE = None
_TEST_LOCK_FDS: list[int] = []


@pytest.fixture(scope="module")
def frozen_session_git(tmp_path_factory):
    """The held-session protocol retains its original production-tree pin."""
    archived = tmp_path_factory.mktemp("m0-held-source") / "repository.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", "--shared", str(REPO), str(archived)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(archived), "update-ref", "HEAD",
         "ba9dc2331d14c7f0598b2fa9028c669236a79b6e"], check=True,
    )
    assert subprocess.check_output(
        ["git", "-C", str(archived), "rev-parse", "HEAD:src/gpuwrf"], text=True,
    ).strip() == "a6885ceded260df2f5777d7366d75a5d38947cb7"
    return archived


@pytest.fixture(autouse=True)
def archived_session_git_context(frozen_session_git, monkeypatch):
    # Check the real pinned git object without requiring current model HEAD to
    # equal a terminal session's historical candidate.
    monkeypatch.setattr(executor, "REPO", frozen_session_git)


@pytest.fixture(autouse=True)
def _close_private_lock_fds():
    start = len(_TEST_LOCK_FDS)
    yield
    for fd in reversed(_TEST_LOCK_FDS[start:]):
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except OSError:
            pass
    del _TEST_LOCK_FDS[start:]


@pytest.fixture
def pure_session(monkeypatch, tmp_path):
    """Unrelated earlier modules import gpuwrf into this monolithic process.

    The real accelerator-free invariant is proven separately, in fresh
    processes, by ``test_session_modules_import_no_accelerator_in_a_hostile_environment``.
    """

    monkeypatch.setattr(session, "assert_accelerator_free", lambda: None)
    monkeypatch.setattr(window_parent, "_assert_parent_accelerator_free", lambda: None)
    monkeypatch.setattr(postlock, "assert_accelerator_free", lambda: None)
    import wrf_source_authority as wsa

    global _SOURCE_AUTHORITY_CACHE
    if _SOURCE_AUTHORITY_CACHE is None:
        _SOURCE_AUTHORITY_CACHE = wsa.build_source_authority(
            namelist_path=window_parent.FAST_RUN_DIR / "namelist.input",
            environ={
                variable: str(wsa.CANONICAL_ROOT)
                for variable in wsa.ROOT_ENV_VARS
            },
        )
    authority_path = tmp_path / "fixture-wrf-source-authority.json"
    wsa.write_authority(authority_path, _SOURCE_AUTHORITY_CACHE)
    monkeypatch.setenv(wsa.AUTHORITY_ENV_VAR, str(authority_path))
    for variable in wsa.ROOT_ENV_VARS:
        monkeypatch.setenv(variable, str(wsa.CANONICAL_ROOT))


def _receipt_payload(
    *,
    window: str = session.SESSION_LABEL,
    requested_at: datetime | None = None,
    managers: tuple[str, ...] = session.REQUIRED_MANAGERS,
    affirmative: bool = True,
    verbatim: str = "yes, proceed with the one M0-CORE session",
    lock_path: Path | None = None,
) -> dict:
    stamp = (requested_at or datetime.now(timezone.utc)).isoformat()
    path = Path(
        lock_path
        or os.environ.get("GPUWRF_GPU_LOCK_FILE", "")
        or executor.CANONICAL_GPU_LOCK
    ).absolute()
    observed = os.lstat(path)
    major = os.major(observed.st_dev)
    minor = os.minor(observed.st_dev)
    identity = {
        "path": str(path),
        "mode": observed.st_mode,
        "device_decimal": observed.st_dev,
        "device_major": major,
        "device_minor": minor,
        "inode": observed.st_ino,
        "proc_locks_key": f"{major:02x}:{minor:02x}:{observed.st_ino}",
        "is_regular": stat.S_ISREG(observed.st_mode),
        "is_symlink": stat.S_ISLNK(observed.st_mode),
    }
    return {
        "schema": "wrf_gpu2.v025.m0.gpu_coordination_receipt.v1",
        "window": window,
        "requested_at_utc": stamp,
        "request_text": "M0-CORE W1->C1->W2->W3, one lock, one receipt",
        "replies": {
            manager: {
                "affirmative": affirmative,
                "verbatim": f"{manager}: {verbatim}",
                "received_at_utc": stamp,
            }
            for manager in managers
        },
        "kernel_lock_binding": {
            "preflight": {
                "status": "PASS",
                "kernel_identity": identity,
                "vacancy_grants_permission": False,
            },
            "immediate_live_recheck": {
                "status": "PASS",
                "identity": dict(identity),
                "vacancy_grants_permission": False,
            },
            "vacancy_grants_permission": False,
        },
    }


def _write_receipt(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "session_receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _held_lock(tmp_path: Path, monkeypatch) -> None:
    """Simulate running under scripts/with_gpu_lock.sh without any device."""

    holder = tmp_path / "lock.holder"
    lock = tmp_path / "private-gpu.lock"
    lock.touch()
    lock_fd = os.open(lock, os.O_RDWR | os.O_APPEND)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _TEST_LOCK_FDS.append(lock_fd)
    token = "gpuwrf-lock-cpu-test-token"
    holder.write_text(
        f"holder={session.SESSION_LABEL} pid={os.getpid()} "
        f"token={token} cmd=cpu-test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(executor, "CANONICAL_GPU_LOCK", lock)
    monkeypatch.setenv("GPUWRF_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("GPUWRF_GPU_LOCK_TOKEN", token)
    monkeypatch.setenv("GPUWRF_GPU_LOCK_HOLDER_FILE", str(holder))
    monkeypatch.setenv("GPUWRF_GPU_LOCK_LABEL", session.SESSION_LABEL)
    monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", str(lock_fd))
    monkeypatch.setenv("GPUWRF_GPU_LOCK_FILE", str(lock.absolute()))


def _write_preflight_fixture(
    proof_root: Path,
    receipt_path: Path,
) -> tuple[Path, Path]:
    """Publish a content-addressed CPU-only preflight for held-session tests."""

    import m0_w1_fast_pair as pairmod

    proof_root.mkdir(parents=True, exist_ok=True)
    identity_path = proof_root / "session_preflight_identity.json"
    import wrf_source_authority as wsa

    authority = wsa.build_source_authority(
        namelist_path=window_parent.FAST_RUN_DIR / "namelist.input",
        environ={
            variable: str(wsa.CANONICAL_ROOT)
            for variable in wsa.ROOT_ENV_VARS
        },
    )
    authority_path = proof_root / "wrf_source_authority.json"
    wsa.write_authority(authority_path, authority)
    boundary = {
        "schema": "wrf_gpu2.v025.m0.cpu_real_boundary_preflight.v1",
        "status": "PASS",
        "device_action": False,
        "platforms": ["cpu"],
        "native_bundle": {"field_count": 29},
        "authority": {
            "authority_sha256": authority["authority_sha256"],
        },
        "lowered_program": {
            "sha256": "7" * 64,
            "integration_trip_count": {
                "status": "PASS",
                "method": "exact-lowered-trip-count",
                "entry_function": "main",
                "loop_count": 4,
                "segment_trip_counts": [179, 1, 179, 1],
                "steps": 360,
                "segments": [],
                "stablehlo_sha256": "7" * 64,
                "configured_cross_check": {
                    "derivation":
                        "Fraction(str(hours))*3600/Fraction(str(dt_s))",
                    "hours": 1.0,
                    "forecast_interval_seconds": 3600.0,
                    "timestep_seconds": 10.0,
                    "exact_steps": 360,
                    "integral": True,
                },
                "count_matches_configuration": True,
            },
        },
        "observations": {
            "native_loader_calls": 1,
            "fast_argument_builder_calls": 1,
            "wrapper_preparation_calls": 1,
            "exact_lower_calls": 1,
            "compile_calls": 0,
            "device_invocations": 0,
            "receipt_reads": 0,
            "ledger_reads": 0,
            "ledger_writes": 0,
            "lock_checks": 0,
            "wrapper_calls": 0,
        },
    }
    boundary["boundary_sha256"] = executor._canonical_sha256(boundary)
    boundary_path = proof_root / "cpu_real_boundary_preflight.json"
    boundary_path.write_text(
        json.dumps(boundary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    import m0_review10_fallback_capability as fallback_capability

    capability = fallback_capability.build_declaration(
        boundary, boundary_path=boundary_path, environ={}
    )
    capability_path = proof_root / "capture_capability_declaration.json"
    capability_path.write_text(
        json.dumps(capability, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    owner = executor._prospective_manager_session_command(
        receipt=str(receipt_path)
    )
    wrapper = executor._held_session_wrapper_command(receipt=str(receipt_path))
    identity = {
        "schema": executor.SESSION_IDENTITY_SCHEMA,
        "status": "PASS",
        "src_gpuwrf_tree": "a6885ceded260df2f5777d7366d75a5d38947cb7",
        "wrf_source_authority": {
            "path": str(authority_path),
            "sha256": postlock.sha256_file(authority_path),
            "content_sha256": authority["authority_sha256"],
        },
        "cpu_real_boundary_preflight": {
            "path": str(boundary_path),
            "sha256": postlock.sha256_file(boundary_path),
            "content_sha256": boundary["boundary_sha256"],
            "observations": boundary["observations"],
        },
        "capture_capability_declaration": {
            "path": str(capability_path),
            "sha256": postlock.sha256_file(capability_path),
            "content_sha256": capability["content_address"]["sha256"],
        },
        "owner_command_normalized": executor._normalize_command(owner),
        "owner_command_sha256": executor._command_sha256(owner),
        "held_wrapper_command_normalized": executor._normalize_command(wrapper),
        "held_wrapper_command_sha256": executor._command_sha256(wrapper),
        "device_action": False,
    }
    identity["identity_sha256"] = executor._canonical_sha256(identity)
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    revalidation = {
        "schema": executor.SESSION_REVALIDATION_SCHEMA,
        "status": "PASS",
        "session_identity_content_sha256": identity["identity_sha256"],
        "observed": {"cpu_fixture": True},
        "device_action": False,
    }
    revalidation["revalidation_sha256"] = executor._canonical_sha256(
        revalidation
    )
    (proof_root / "prelock_revalidation.json").write_text(
        json.dumps(revalidation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    wrfout = proof_root / "cpu-wrfout"
    wrfout.write_bytes(b"cpu-only-preflight-output")
    output_sha256 = postlock.sha256_file(wrfout)
    invocation = [
        "taskset",
        "-c",
        "16-27",
        "mpirun",
        "-np",
        "12",
        "wrf.exe",
    ]
    cpu_record = {
        "kind": "cpu",
        "run_id": "m0-core-prelock-fresh-cpu",
        "started_at_utc": "2026-07-28T00:00:00+00:00",
        "finished_at_utc": "2026-07-28T00:00:01+00:00",
        "status": "OK",
        "payload": {
            "command": invocation,
            "cpu_list": "16-27",
            "ranks": 12,
            "launcher_wallclock_seconds": 1.0,
            "final_wrfout_path": str(wrfout),
            "final_wrfout_sha256": output_sha256,
        },
    }
    preflight = {
        "schema": pairmod.CPU_PREFLIGHT_SCHEMA,
        "status": "PASS",
        "session_identity_path": str(identity_path.resolve()),
        "session_identity_sha256": postlock.sha256_file(identity_path),
        "session_identity_content_sha256": identity["identity_sha256"],
        "started_monotonic_ns": 1,
        "finished_monotonic_ns": 2,
        "cpu_list": "16-27",
        "ranks": 12,
        "cpu_record": cpu_record,
        "cpu_record_sha256": postlock.canonical_sha256(cpu_record),
        "cpu_invocation_argv": invocation,
        "cpu_invocation_sha256": hashlib.sha256(
            "\0".join(invocation).encode("utf-8")
        ).hexdigest(),
        "verified_output": {
            "final_wrfout_path": str(wrfout),
            "final_wrfout_bytes": wrfout.stat().st_size,
            "final_wrfout_sha256": output_sha256,
        },
        "device_action": False,
        "lock_environment_absent": True,
    }
    preflight["preflight_sha256"] = postlock.canonical_sha256(preflight)
    cpu_path = proof_root / "cpu_preflight.json"
    cpu_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return identity_path, cpu_path


def _stub_hooks(record: list[str], *, fail: str | None = None) -> dict:
    def make(stage: str):
        def run():
            record.append(stage)
            if stage == fail:
                raise window_parent.WindowRefusal(f"injected {stage} failure")
            return {"stage": stage, "cpu_stub": True}

        return run

    return {
        "W1": make("W1"),
        session.C1_STAGE: make(session.C1_STAGE),
        "W2": make("W2"),
        "W3": make("W3"),
    }


def _run_session(tmp_path, monkeypatch, *, hooks, receipt=None, **kwargs):
    _held_lock(tmp_path, monkeypatch)
    payload = dict(receipt) if receipt is not None else _receipt_payload()
    if receipt is not None:
        payload["kernel_lock_binding"] = _receipt_payload()[
            "kernel_lock_binding"
        ]
    receipt_path = _write_receipt(tmp_path, payload)
    proof_root = tmp_path / "proof"
    identity_path, cpu_path = _write_preflight_fixture(proof_root, receipt_path)
    return executor.run_held_session(
        receipt_path=receipt_path,
        ledger_path=tmp_path / "spend_ledger.json",
        proof_root=proof_root,
        session_identity_path=identity_path,
        cpu_preflight_path=cpu_path,
        stage_hooks=hooks,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# R1 - the frozen outer owner and held wrapper are both executable             #
# --------------------------------------------------------------------------- #
def test_frozen_amendment6_commands_parse_byte_for_byte(tmp_path):
    absent = tmp_path / "absent.json"
    owner = executor._prospective_manager_session_command(receipt=str(absent))
    assert owner == [
        sys.executable,
        str(executor.SCRIPT),
        "--manager-core-session-owner",
        "--receipt",
        str(absent),
    ]
    command = executor._held_session_wrapper_command(
        receipt=str(absent)
    )
    assert command[:6] == [
        str(executor.LOCK_WRAPPER),
        "--timeout",
        "0",
        "--label",
        session.SESSION_LABEL,
        "--",
    ]
    assert command[6:] == [
        sys.executable,
        str(executor.SCRIPT),
        "--manager-core-session-held",
        "--receipt",
        str(absent),
    ]
    inner = command[6:]
    environment = dict(os.environ)
    environment.update(DEVICE_FREE_ENV)
    completed = subprocess.run(
        inner,
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    # argparse must accept the mode: the refusal has to come from the missing
    # receipt, not from an unrecognised argument (Kimi major M1).
    assert "unrecognized arguments" not in completed.stderr
    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["receipt_consumed"] is False
    assert payload["device_touched"] is False
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False

    outer = subprocess.run(
        owner,
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert outer.returncode == 2, outer.stderr
    outer_payload = json.loads(outer.stdout)
    assert outer_payload["status"] == "BLOCKED"
    assert outer_payload["receipt_consumed"] is False
    assert outer_payload["device_touched"] is False


def test_session_plan_freezes_one_receipt_one_lock_and_no_escape_paths():
    plan = executor.held_session_plan()
    assert plan["entrypoint"] == "--manager-core-session-owner"
    assert plan["held_entrypoint"] == "--manager-core-session-held"
    assert plan["entrypoint_implemented"] is True
    assert plan["receipt_spends"] == 1
    assert plan["lock_acquisitions"] == 1
    assert plan["global_deadline_seconds"] == 4500.0
    assert plan["stage_order"] == [
        "W1_COLD_DECISION",
        "W1_CACHED_AND_CORRECTNESS",
        session.C1_STAGE,
        "W2_PROFILED_CAPTURE",
        "W3_CLEAN_MATCHED_ARM",
        "LOCK_RELEASE",
        "POSTLOCK_CPU_ANALYSIS",
        "POSTLOCK_M0_CORE_FINALIZER",
    ]
    assert set(plan["forbidden_paths"]) == {
        "override",
        "retry",
        "queueing",
        "receipt_refund",
        "threshold_movement",
        "second_lock_acquisition",
    }
    assert plan["held_budget"]["reservation_seconds"] == 4220.0
    assert plan["held_budget"]["headroom_seconds"] == 280.0
    assert plan["coordination_costs"] == {
        "cpu_preflight_before_lock_seconds": 420.0,
        "c1_compare_and_pair_prepare_in_lock_seconds": 60.0,
        "disclosed_to_both_managers": True,
        "cpu_preflight_inside_global_deadline": False,
        "c1_inside_global_deadline": True,
    }
    # The plan the manager already froze must still validate.
    assert exact.validate_plan(executor.build_plan())["lock_acquisitions"] == 1


def test_held_session_runs_the_frozen_order_and_spends_exactly_one_receipt(
    tmp_path, monkeypatch, pure_session
):
    order: list[str] = []
    payload = _run_session(tmp_path, monkeypatch, hooks=_stub_hooks(order))

    assert payload["status"] == "PASS", json.dumps(
        payload, indent=2, sort_keys=True, default=str
    )
    assert order == ["W1", session.C1_STAGE, "W2", "W3"]
    assert payload["graph"]["stage_order_executed"] == list(
        session.POST_COLD_STAGE_ORDER
    )
    assert payload["lock_acquisitions"] == 1
    assert payload["authorization"]["spent_after_cpu_preflight"] is True
    assert (
        payload["authorization"]["spent_after_input_revalidation"] is True
    )
    assert payload["authorization"]["spent_before_any_device_child"] is True
    assert payload["receipt_refund_path"] is None
    assert payload["retry_path"] is None
    assert payload["queue_path"] is None
    ledger = json.loads((tmp_path / "spend_ledger.json").read_text())
    assert len(ledger["spent"]) == 1


def test_real_w1_c1_w2_w3_callbacks_reach_w3_with_only_children_injected(
    tmp_path, monkeypatch, pure_session
):
    """Traverse the live callbacks; replace only device/CPU child execution."""

    import prepare_m0_matched_pair as preparer

    raw_root = tmp_path / "raw"
    pair_cache_root = tmp_path / "pair-cache"
    proof_dir = tmp_path / "window-proofs"
    prepared_dir = proof_dir / "prepared"
    qualification_path = proof_dir / "qualification.json"
    windows = {name: dict(value) for name, value in window_parent.WINDOWS.items()}
    windows["W1"].update(
        {
            "cache_path": raw_root / windows["W1"]["run_id"] / "qualified-cache",
            "result": proof_dir / "w1.window.json",
        }
    )
    windows["W2"].update(
        {
            "cache_path": pair_cache_root / windows["W2"]["run_id"],
            "identity_path": prepared_dir / "profiled_identity.json",
            "result": proof_dir / "w2.window.json",
        }
    )
    windows["W3"].update(
        {
            "cache_path": pair_cache_root / windows["W3"]["run_id"],
            "identity_path": prepared_dir / "clean_identity.json",
            "result": proof_dir / "w3.window.json",
        }
    )
    monkeypatch.setattr(window_parent, "RAW_ROOT", raw_root)
    monkeypatch.setattr(window_parent, "PAIR_CACHE_ROOT", pair_cache_root)
    monkeypatch.setattr(window_parent, "PROOF_ROOT", proof_dir)
    monkeypatch.setattr(window_parent, "QUALIFICATION", qualification_path)
    monkeypatch.setattr(window_parent, "WINDOWS", windows)
    monkeypatch.setattr(preparer, "CANONICAL_OUTPUT_DIR", prepared_dir)
    monkeypatch.setattr(
        preparer, "CANONICAL_CACHE_SEED", Path(windows["W1"]["cache_path"])
    )
    monkeypatch.setattr(
        preparer, "CANONICAL_CACHE_SNAPSHOT_ROOT", pair_cache_root
    )
    monkeypatch.setattr(
        preparer, "CANONICAL_QUALIFICATION", qualification_path
    )
    monkeypatch.setattr(
        preparer, "CANONICAL_PROFILED_RUN_ID", windows["W2"]["run_id"]
    )
    monkeypatch.setattr(
        preparer, "CANONICAL_CLEAN_RUN_ID", windows["W3"]["run_id"]
    )

    child_order: list[str] = []
    next_pid = {"value": 1000}
    exact_digest = "c" * 64
    identity = {
        "argument_identity": "argument-identity",
        "lowered_program_sha256": "d" * 64,
        "jit_identity": "jit-identity",
        "case_preparation_sequence": ["case"],
        "wrapper_preparation_sequence": ["wrapper"],
    }

    def fake_expensive_child(**kwargs):
        stage = kwargs["stage"]
        mode = kwargs["mode"]
        run_id = str(kwargs["run_id"])
        stage_root = Path(kwargs["run_root"]) / stage
        stage_root.mkdir(parents=True, exist_ok=False)
        cache = Path(kwargs["cache_path"])
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "compiled-cache-entry").write_bytes(b"same-cache-bytes")
        next_pid["value"] += 1
        child_order.append(stage)
        timing = {
            "readiness_seconds": 10.0 if mode == "compile-only" else 1.0,
            "integration_seconds": None if mode == "compile-only" else 1.0,
            "child_executable_ready_monotonic_ns": 10,
            "integration_start_monotonic_ns": (
                None if mode == "compile-only" else 20
            ),
            "integration_end_monotonic_ns": (
                None if mode == "compile-only" else 30
            ),
            "derived_by_phase_subtraction": False,
        }
        if mode == "profiled":
            allocator_path = stage_root / "forecast_allocator.json"
            allocator_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
            instrumentation = {
                "mode": "profiled",
                "nvtx_range": "GPUWRF_M0_FORECAST_INTEGRATION",
                "allocator_sidecar": str(allocator_path),
            }
            allocator = {"status": "PASS"}
        else:
            allocator_path = None
            instrumentation = {
                "mode": "clean",
                "nvtx_range": None,
                "allocator_sidecar": None,
                "clean_has_profiler_range": False,
                "clean_has_allocator_read": False,
                "invocation_helper":
                    "m0_exact_boundary_child._invoke_and_synchronize",
            }
            allocator = None
        exact_payload = {
            "schema": "cpu-real-callback-fixture",
            "status": "OK",
            "run_id": run_id,
            "pid": next_pid["value"],
            "call": {
                "argument_identity": identity["argument_identity"],
                "lowered_program_sha256": identity[
                    "lowered_program_sha256"
                ],
            },
            "production_binding": {
                "jit_identity": identity["jit_identity"],
                "case_preparation_sequence": identity[
                    "case_preparation_sequence"
                ],
                "wrapper_preparation_sequence": identity[
                    "wrapper_preparation_sequence"
                ],
            },
            "timing": timing,
            "result": {"exact_value_sha256": exact_digest},
            "instrumentation": instrumentation,
            "allocator": allocator,
        }
        if mode != "compile-only":
            wrfout_dir = stage_root / "wrfout"
            wrfout_dir.mkdir()
            wrfout = wrfout_dir / "wrfout_d01_fixture"
            wrfout.write_bytes(b"same-wrfout-bytes")
            exact_payload["wrfout"] = {
                "status": "PASS",
                "run_id": run_id,
                "final_wrfout_path": str(wrfout),
                "final_wrfout_sha256": postlock.sha256_file(wrfout),
                "result_exact_value_sha256": exact_digest,
                "binding_sha256": "e" * 64,
            }
        result_path = stage_root / "exact_boundary.json"
        result_path.write_text(
            json.dumps(exact_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log_path = stage_root / "child.log"
        log_path.write_text("CPU fixture replaced expensive child\n", encoding="utf-8")
        sampler = None
        if mode == "profiled":
            (stage_root / "step1_autotune_off.nsys-rep").write_bytes(
                b"CPU-only profiler fixture"
            )
            sampler = {"status": "PASS", "peak_device_memory_bytes": 1}
            (stage_root / "lock_owner_total_residency.json").write_text(
                json.dumps(sampler, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return {
            "name": stage,
            "status": "OK",
            "fresh_process": True,
            "pid": next_pid["value"],
            "result_path": str(result_path),
            "result_sha256": postlock.sha256_file(result_path),
            "result": exact_payload,
            "sampler": sampler,
            "log_path": str(log_path),
            "log_sha256": postlock.sha256_file(log_path),
        }

    monkeypatch.setattr(
        window_parent, "_launch_authorized_child", fake_expensive_child
    )

    def fake_c1_process(
        command,
        *,
        stage,
        timeout_seconds,
        environment,
        log_path,
        registry,
        cwd=REPO,
        deadline=None,
    ):
        assert stage == session.C1_STAGE
        assert timeout_seconds == 60.0
        child_order.append("C1_COMPARE_AND_PREPARE")
        fast_pair_path = Path(command[command.index("--output") + 1])
        w1 = json.loads(Path(windows["W1"]["result"]).read_text())
        stages = {item["name"]: item for item in w1["stages"]}
        selected = w1["session"]["selected_cold_stage"]
        cold = stages[selected]
        cached = stages["cached_readiness_and_warm_integration"]
        cached_result = json.loads(Path(cached["result_path"]).read_text())
        pair_payload = {
            "schema": "wrf_gpu2.v025.m0.fast_case_qualification.v1",
            "status": "PASS",
            "c1_subgate": {
                "both_green": True,
                "cpu_arm_pre_staged_before_receipt_and_lock": True,
            },
            "w1_exact_boundary": {
                "run_id": windows["W1"]["run_id"],
                "cold_result_sha256": cold["result_sha256"],
                "cached_result_sha256": cached["result_sha256"],
                "gpu_result_sha256": exact_digest,
                "gpu_wrfout_path":
                    cached_result["wrfout"]["final_wrfout_path"],
                "gpu_wrfout_sha256":
                    cached_result["wrfout"]["final_wrfout_sha256"],
                "gpu_wrfout_binding_sha256":
                    cached_result["wrfout"]["binding_sha256"],
            },
            "pair_completeness": {
                "fresh_cpu_arm_this_invocation": True,
                "fresh_gpu_arm_this_invocation": True,
                "comparator_result_present": True,
                "provenance_present": True,
                "arms_non_overlapping": True,
                "completeness_percent": 100,
            },
            "economy_gates": {
                "cold_compile_seconds": {"value": 10.0},
                "cached_load_seconds": {"value": 1.0},
                "gpu_arm_seconds": {"value": 1.0},
                "cpu_arm_seconds": {"value": 1.0},
                "pair_seconds_excl_lock_wait": {"value": 2.0},
            },
        }
        fast_pair_path.write_text(
            json.dumps(pair_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        window_parent._post_w1_qualification(
            w1_result_path=Path(windows["W1"]["result"]),
            fast_pair_path=fast_pair_path,
            output_path=qualification_path,
        )
        preparer.prepare(
            output_dir=prepared_dir,
            device_uuid=window_parent.EXPECTED_DEVICE_UUID,
            profiled_run_id=windows["W2"]["run_id"],
            clean_run_id=windows["W3"]["run_id"],
            profiled_receipt=Path(command[command.index("--receipt") + 1]),
            clean_receipt=Path(command[command.index("--receipt") + 1]),
            cache_seed=Path(windows["W1"]["cache_path"]),
            cache_snapshot_root=pair_cache_root,
            qualification_manifest=qualification_path,
            source_root=REPO / "src/gpuwrf",
            run_dir=window_parent.FAST_RUN_DIR,
        )
        Path(log_path).write_text(
            "CPU fixture replaced expensive C1 process\n", encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(executor, "_run_registered_command", fake_c1_process)
    _held_lock(tmp_path, monkeypatch)
    receipt_path = _write_receipt(tmp_path, _receipt_payload())
    root = tmp_path / "session"
    identity_path, cpu_path = _write_preflight_fixture(root, receipt_path)
    payload = executor.run_held_session(
        receipt_path=receipt_path,
        ledger_path=tmp_path / "ledger.json",
        proof_root=root,
        session_identity_path=identity_path,
        cpu_preflight_path=cpu_path,
    )

    assert payload["status"] == "PASS", json.dumps(
        payload, indent=2, sort_keys=True, default=str
    )
    assert payload["graph"]["stage_order_executed"] == list(
        session.POST_COLD_STAGE_ORDER
    )
    assert child_order == [
        "cold_empty_cache_readiness_1",
        "cold_empty_cache_readiness_2",
        "cached_readiness_and_warm_integration",
        "C1_COMPARE_AND_PREPARE",
        "profiled_cached_readiness_and_integration",
        "clean_cached_readiness_and_integration",
    ]
    assert payload["stage_details"][session.C1_STAGE][
        "qualification"
    ]["selected_cold_stage"] == "cold_empty_cache_readiness_1"
    assert Path(windows["W2"]["result"]).is_file()
    assert Path(windows["W3"]["result"]).is_file()
    assert (prepared_dir / "plan.json").is_file()


# --------------------------------------------------------------------------- #
# R2 - C1 is first and suppressing                                             #
# --------------------------------------------------------------------------- #
def test_c1_failure_makes_w2_and_w3_mechanically_unreachable(
    tmp_path, monkeypatch, pure_session
):
    order: list[str] = []
    payload = _run_session(
        tmp_path, monkeypatch, hooks=_stub_hooks(order, fail=session.C1_STAGE)
    )

    assert payload["status"] == "BLOCKED"
    assert order == ["W1", session.C1_STAGE]
    assert "W2" not in order and "W3" not in order
    assert payload["first_failure"]["stage"] == session.C1_STAGE
    assert payload["first_failure"]["kind"] == "C1_SUBGATE_FAILURE"
    assert payload["graph"]["suppressed"] == ["W2", "W3"]


def test_w1_failure_suppresses_c1_and_every_device_stage(
    tmp_path, monkeypatch, pure_session
):
    order: list[str] = []
    payload = _run_session(tmp_path, monkeypatch, hooks=_stub_hooks(order, fail="W1"))

    assert payload["status"] == "BLOCKED"
    assert order == ["W1"]
    assert payload["graph"]["suppressed"] == [session.C1_STAGE, "W2", "W3"]


def test_c1_subgate_names_both_components_and_discloses_its_lock_cost():
    contract = session.session_contract()
    subgate = contract["c1_subgate"]
    assert contract["stage_order"].index(session.C1_STAGE) == (
        contract["stage_order"].index("W1_CACHED_AND_CORRECTNESS") + 1
    )
    assert contract["stage_order"].index(session.C1_STAGE) < contract[
        "stage_order"
    ].index("W2_PROFILED_CAPTURE")
    assert subgate["components"] == [
        "w1_same_result_wrfout_binding_verify_file",
        "fresh_12_rank_cpu_wrf_comparator",
    ]
    assert subgate["both_required"] is True
    assert subgate["runs_inside_the_one_held_lock"] is True
    assert subgate["expected_lock_hold_seconds"] == 60.0
    assert subgate["expected_lock_hold_seconds"] < contract[
        "global_deadline_seconds"
    ]
    assert contract["suppression"]["c1_comparator_failure"] == ["W2", "W3"]
    assert contract["suppression"]["c1_wrfout_binding_failure"] == ["W2", "W3"]
    assert contract["machine_authority"]["fields"] == [
        "cpu_closure_gate",
        "milestone_partition",
    ]


def test_in_session_comparator_requires_the_held_lock_not_a_release_proof(
    tmp_path, monkeypatch, pure_session
):
    import m0_w1_fast_pair as pairmod

    w1 = tmp_path / "w1.json"
    w1.write_text(json.dumps({"status": "OK", "window": "W1"}), encoding="utf-8")
    with pytest.raises(pairmod.W1PairRefusal, match="exactly one"):
        pairmod.build_pair(
            w1_result_path=w1,
            release_path=tmp_path / "release.json",
            cpu_run_root=tmp_path / "cpu",
            output_path=tmp_path / "pair.json",
            in_session=True,
        )
    with pytest.raises(pairmod.W1PairRefusal, match="exactly one"):
        pairmod.build_pair(
            w1_result_path=w1,
            cpu_run_root=tmp_path / "cpu",
            output_path=tmp_path / "pair.json",
        )
    # In-session mode refuses when the canonical lock is NOT held.
    for key in (
        "GPUWRF_GPU_LOCK_HELD",
        "GPUWRF_GPU_LOCK_TOKEN",
        "GPUWRF_GPU_LOCK_HOLDER_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    import run_gpu_arm as gpu_auth

    with pytest.raises(gpu_auth.WindowNotAuthorised):
        pairmod._in_session_provenance(w1)


# --------------------------------------------------------------------------- #
# R1 mutation matrix - receipts, lock, deadline, queueing, leaks               #
# --------------------------------------------------------------------------- #
def test_receipt_reuse_is_refused_after_the_single_spend(
    tmp_path, monkeypatch, pure_session
):
    _held_lock(tmp_path, monkeypatch)
    receipt_path = _write_receipt(tmp_path, _receipt_payload())
    ledger = tmp_path / "spend_ledger.json"
    identity_1, cpu_1 = _write_preflight_fixture(
        tmp_path / "proof-1", receipt_path
    )
    first = executor.run_held_session(
        receipt_path=receipt_path,
        ledger_path=ledger,
        proof_root=tmp_path / "proof-1",
        session_identity_path=identity_1,
        cpu_preflight_path=cpu_1,
        stage_hooks=_stub_hooks([]),
    )
    assert first["status"] == "PASS"
    identity_2, cpu_2 = _write_preflight_fixture(
        tmp_path / "proof-2", receipt_path
    )
    with pytest.raises(Exception) as excinfo:
        executor.run_held_session(
            receipt_path=receipt_path,
            ledger_path=ledger,
            proof_root=tmp_path / "proof-2",
            session_identity_path=identity_2,
            cpu_preflight_path=cpu_2,
            stage_hooks=_stub_hooks([]),
        )
    assert "SPENT" in str(excinfo.value) or "already spent" in str(excinfo.value)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"window": "some-other-window"}, "exact M0-CORE session"),
        ({"replies": {}}, "affirmative verbatim"),
    ],
)
def test_receipt_shape_mutations_fail_closed(mutation, match, pure_session):
    payload = _receipt_payload()
    payload.update(mutation)
    with pytest.raises(session.SessionRefusal, match=match):
        session.validate_receipt_shape(payload)


def test_missing_manager_text_is_refused(pure_session):
    payload = _receipt_payload()
    for manager in session.REQUIRED_MANAGERS:
        payload["replies"][manager]["verbatim"] = "   "
    with pytest.raises(session.SessionRefusal, match="affirmative verbatim"):
        session.validate_receipt_shape(payload)
    payload = _receipt_payload(managers=("0:2",))
    with pytest.raises(session.SessionRefusal, match="0:3"):
        session.validate_receipt_shape(payload)


def test_non_affirmative_manager_is_refused(pure_session):
    payload = _receipt_payload(affirmative=False)
    with pytest.raises(session.SessionRefusal, match="affirmative verbatim"):
        session.validate_receipt_shape(payload)


def test_content_address_mismatch_is_refused(pure_session):
    payload = _receipt_payload()
    payload["fingerprint"] = "0" * 64
    with pytest.raises(session.SessionRefusal, match="content address"):
        session.validate_receipt_shape(payload)


def test_session_refuses_without_the_already_held_canonical_lock(
    tmp_path, monkeypatch, pure_session
):
    for key in (
        "GPUWRF_GPU_LOCK_HELD",
        "GPUWRF_GPU_LOCK_TOKEN",
        "GPUWRF_GPU_LOCK_HOLDER_FILE",
        "GPUWRF_GPU_LOCK_LABEL",
    ):
        monkeypatch.delenv(key, raising=False)
    receipt_path = _write_receipt(tmp_path, _receipt_payload())
    identity, cpu = _write_preflight_fixture(tmp_path / "proof", receipt_path)
    with pytest.raises(Exception, match="with_gpu_lock"):
        executor.run_held_session(
            receipt_path=receipt_path,
            ledger_path=tmp_path / "ledger.json",
            proof_root=tmp_path / "proof",
            session_identity_path=identity,
            cpu_preflight_path=cpu,
            stage_hooks=_stub_hooks([]),
        )
    # The refusal happened before any spend: the ledger was never written.
    assert not (tmp_path / "ledger.json").exists()


def test_lock_token_mismatch_is_refused(tmp_path, monkeypatch, pure_session):
    _held_lock(tmp_path, monkeypatch)
    monkeypatch.setenv("GPUWRF_GPU_LOCK_TOKEN", "someone-elses-token")
    receipt_path = _write_receipt(tmp_path, _receipt_payload())
    identity, cpu = _write_preflight_fixture(tmp_path / "proof", receipt_path)
    with pytest.raises(Exception, match="held by someone else"):
        executor.run_held_session(
            receipt_path=receipt_path,
            ledger_path=tmp_path / "ledger.json",
            proof_root=tmp_path / "proof",
            session_identity_path=identity,
            cpu_preflight_path=cpu,
            stage_hooks=_stub_hooks([]),
        )
    assert not (tmp_path / "ledger.json").exists()


@pytest.mark.parametrize("mutation", ["exported", "holder"])
def test_wrong_session_lock_label_is_refused_before_receipt_spend(
    tmp_path, monkeypatch, pure_session, mutation
):
    _held_lock(tmp_path, monkeypatch)
    holder = Path(os.environ["GPUWRF_GPU_LOCK_HOLDER_FILE"])
    if mutation == "exported":
        monkeypatch.setenv("GPUWRF_GPU_LOCK_LABEL", "WRONG-LABEL")
    else:
        holder.write_text(
            "holder=WRONG-LABEL pid=1 "
            f"token={os.environ['GPUWRF_GPU_LOCK_TOKEN']} cmd=cpu-test\n",
            encoding="utf-8",
        )
    receipt_path = _write_receipt(tmp_path, _receipt_payload())
    identity, cpu = _write_preflight_fixture(tmp_path / "proof", receipt_path)
    with pytest.raises(Exception, match="label"):
        executor.run_held_session(
            receipt_path=receipt_path,
            ledger_path=tmp_path / "ledger.json",
            proof_root=tmp_path / "proof",
            session_identity_path=identity,
            cpu_preflight_path=cpu,
            stage_hooks=_stub_hooks([]),
        )
    assert not (tmp_path / "ledger.json").exists()


def test_cpu_preflight_failure_never_launches_wrapper_or_spends_receipt(
    tmp_path, monkeypatch, pure_session
):
    receipt_path = _write_receipt(tmp_path, _receipt_payload())
    seen: list[str] = []

    def identity_builder(*, output_path, receipt_path, tool_runner=None):
        output_path.write_text('{"fixture":true}\n', encoding="utf-8")
        return {"fixture": True}

    def fail_cpu(
        command,
        *,
        stage,
        timeout_seconds,
        environment,
        log_path,
        registry,
        cwd=REPO,
        deadline=None,
    ):
        seen.append(stage)
        assert stage == "CPU_PREFLIGHT"
        assert timeout_seconds == 420.0
        assert not (tmp_path / "ledger.json").exists()
        Path(log_path).write_text("injected CPU preflight failure\n", encoding="utf-8")
        return 19

    monkeypatch.setattr(
        executor, "build_session_preflight_identity", identity_builder
    )
    monkeypatch.setattr(executor, "_run_registered_command", fail_cpu)
    payload = executor.run_session_owner(
        receipt_path=receipt_path,
        ledger_path=tmp_path / "ledger.json",
        proof_root=tmp_path / "owner",
        cpu_run_root=tmp_path / "cpu",
    )

    assert payload["status"] == "BLOCKED"
    assert seen == ["CPU_PREFLIGHT"]
    assert payload["first_failure"]["stage"] == "CPU_PREFLIGHT"
    assert payload["release_proof"] is None
    assert not (tmp_path / "ledger.json").exists()


def test_outer_owner_source_mechanically_reaches_release_posts_and_finalizer():
    source = (SCRIPTS / "m0_three_window_executor.py").read_text(encoding="utf-8")
    owner = source[source.index("def run_session_owner("):]
    owner = owner[: owner.index("\ndef _execute_live_graph(")]
    ordered = [
        "build_session_preflight_identity(",
        "_cpu_preflight_command(",
        "revalidate_session_preflight_inputs(",
        "_held_session_wrapper_command(",
        "build_session_lock_release_proof(",
        '"POSTLOCK_W3_PAIR"',
        '"POSTLOCK_W2_EXPORT_CENSUS"',
        '"POSTLOCK_M0_CORE_FINALIZER"',
    ]
    positions = [owner.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)
    assert owner.count("_held_session_wrapper_command(") == 1
    assert "timeout_seconds=core_session.GLOBAL_DEADLINE_SECONDS" in owner


def test_prelock_revalidation_accepts_symlinked_tools_but_rejects_retarget(
    tmp_path, monkeypatch
):
    import fast_case
    import m0_vram_sampler as mvs

    first = tmp_path / "tool-v1"
    second = tmp_path / "tool-v2"
    first.write_bytes(b"frozen-tool-v1")
    second.write_bytes(b"mutated-tool-v2")
    links = {}
    tools = {}
    for role in ("python", "nsys", "mpirun", "wrf_exe"):
        link = tmp_path / f"{role}-link"
        link.symlink_to(first)
        links[role] = link
        tools[role] = executor._tool_identity(link)
        assert tools[role]["invoked_via_symlink"] is True
        assert tools[role]["path"] == str(first.resolve())

    config_path = tmp_path / "namelist.input"
    config_path.write_text("&time_control\n/\n", encoding="utf-8")
    source_inputs = {"input": "1" * 64}
    descriptor = {"case": "CPU-fixture"}
    cache = {
        "w1_seed_path": str(window_parent.WINDOWS["W1"]["cache_path"]),
        "required_state": "ABSENT",
        "observed_absent": True,
    }
    import wrf_source_authority as wsa

    authority = wsa.build_source_authority(
        namelist_path=window_parent.FAST_RUN_DIR / "namelist.input",
        environ={
            variable: str(wsa.CANONICAL_ROOT)
            for variable in wsa.ROOT_ENV_VARS
        },
    )
    authority_path = tmp_path / "wrf_source_authority.json"
    wsa.write_authority(authority_path, authority)
    boundary = {
        "schema": "wrf_gpu2.v025.m0.cpu_real_boundary_preflight.v1",
        "status": "PASS",
        "device_action": False,
        "platforms": ["cpu"],
        "native_bundle": {"field_count": 29},
        "authority": {"authority_sha256": authority["authority_sha256"]},
        "lowered_program": {
            "sha256": "7" * 64,
            "integration_trip_count": {
                "status": "PASS",
                "method": "exact-lowered-trip-count",
                "steps": 360,
                "stablehlo_sha256": "7" * 64,
                "configured_cross_check": {
                    "exact_steps": 360,
                },
                "count_matches_configuration": True,
            },
        },
        "observations": {
            "native_loader_calls": 1,
            "fast_argument_builder_calls": 1,
            "wrapper_preparation_calls": 1,
            "exact_lower_calls": 1,
            "compile_calls": 0,
            "device_invocations": 0,
            "receipt_reads": 0,
            "ledger_reads": 0,
            "ledger_writes": 0,
            "lock_checks": 0,
            "wrapper_calls": 0,
        },
    }
    boundary["boundary_sha256"] = executor._canonical_sha256(boundary)
    boundary_path = tmp_path / "cpu_real_boundary_preflight.json"
    boundary_path.write_text(
        json.dumps(boundary, sort_keys=True) + "\n", encoding="utf-8"
    )
    import m0_review10_fallback_capability as fallback_capability

    capability = fallback_capability.build_declaration(
        boundary, boundary_path=boundary_path, environ={}
    )
    capability_path = tmp_path / "capture_capability_declaration.json"
    capability_path.write_text(
        json.dumps(capability, sort_keys=True) + "\n", encoding="utf-8"
    )
    identity = {
        "identity_sha256": "a" * 64,
        "src_gpuwrf_tree": "a6885ceded260df2f5777d7366d75a5d38947cb7",
        "source_inputs": source_inputs,
        "source_inputs_sha256": executor._canonical_sha256(source_inputs),
        "config": {
            "path": str(config_path),
            "sha256": postlock.sha256_file(config_path),
            "case_descriptor_sha256":
                executor._canonical_sha256(descriptor),
        },
        "input_manifest_sha256": "b" * 64,
        "cache": cache,
        "tools": tools,
        "wrf_source_authority": {
            "path": str(authority_path),
            "content_sha256": authority["authority_sha256"],
        },
        "cpu_real_boundary_preflight": {
            "path": str(boundary_path),
            "sha256": postlock.sha256_file(boundary_path),
            "content_sha256": boundary["boundary_sha256"],
        },
        "capture_capability_declaration": {
            "path": str(capability_path),
            "sha256": postlock.sha256_file(capability_path),
            "content_sha256": capability["content_address"]["sha256"],
        },
    }
    monkeypatch.setattr(
        fast_case, "verify_source_inputs", lambda: source_inputs
    )
    monkeypatch.setattr(fast_case, "case_descriptor", lambda: descriptor)
    monkeypatch.setattr(
        mvs, "input_manifest_sha256", lambda _path: "b" * 64
    )
    monkeypatch.setattr(
        executor,
        "_git_object",
        lambda _name: "a6885ceded260df2f5777d7366d75a5d38947cb7",
    )
    monkeypatch.setattr(
        window_parent,
        "validate_pre_authorization",
        lambda _window: {"unique_empty_cache_absent": True},
    )
    proof = executor.revalidate_session_preflight_inputs(identity)
    assert proof["status"] == "PASS"

    links["python"].unlink()
    links["python"].symlink_to(second)
    with pytest.raises(Exception, match="tools"):
        executor.revalidate_session_preflight_inputs(
            identity, output_path=tmp_path / "must-not-exist.json"
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_cpu_preflight_binds_the_exact_mpi_and_wrf_executables(
    tmp_path, pure_session
):
    import m0_w1_fast_pair as pair_gate
    import run_fast_pair as fast_pair

    tool_paths = {}
    tools = {}
    for role in ("python", "nsys", "mpirun", "wrf_exe"):
        path = tmp_path / role
        path.write_bytes(f"{role}-cpu-fixture".encode())
        tool_paths[role] = path
        tools[role] = executor._tool_identity(path)
    identity = {
        "schema": executor.SESSION_IDENTITY_SCHEMA,
        "status": "PASS",
        "tools": tools,
        "device_action": False,
    }
    identity["identity_sha256"] = postlock.canonical_sha256(identity)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    wrfout = tmp_path / "wrfout"
    wrfout.write_bytes(b"fresh-cpu-wrfout")
    output_hash = postlock.sha256_file(wrfout)

    def cpu_runner(**_kwargs):
        return fast_pair.ArmRecord(
            kind="cpu",
            run_id="m0-core-prelock-fresh-cpu",
            started_at_utc="2026-07-28T00:00:00+00:00",
            finished_at_utc="2026-07-28T00:00:01+00:00",
            status="OK",
            payload={
                "command": [
                    "taskset",
                    "-c",
                    "16-27",
                    str(tool_paths["mpirun"]),
                    "-np",
                    "12",
                    str(tool_paths["wrf_exe"]),
                ],
                "cpu_list": "16-27",
                "ranks": 12,
                "launcher_wallclock_seconds": 1.0,
                "final_wrfout_path": str(wrfout),
                "final_wrfout_sha256": output_hash,
                "mpirun_realpath": tools["mpirun"]["path"],
                "mpirun_sha256": tools["mpirun"]["sha256"],
                "wrf_exe_realpath": tools["wrf_exe"]["path"],
                "wrf_exe_sha256": tools["wrf_exe"]["sha256"],
            },
        )

    payload = pair_gate.stage_cpu_preflight(
        session_identity_path=identity_path,
        cpu_run_root=tmp_path / "unused-cpu-root",
        output_path=tmp_path / "cpu-preflight.json",
        cpu_runner=cpu_runner,
    )
    assert payload["status"] == "PASS"

    def mutated_runner(**kwargs):
        record = cpu_runner(**kwargs)
        record.payload["wrf_exe_sha256"] = "0" * 64
        return record

    with pytest.raises(
        pair_gate.W1PairRefusal, match="frozen MPI/WRF tools"
    ):
        pair_gate.stage_cpu_preflight(
            session_identity_path=identity_path,
            cpu_run_root=tmp_path / "unused-mutated-root",
            output_path=tmp_path / "must-not-exist.json",
            cpu_runner=mutated_runner,
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_mutated_prelock_revalidation_is_refused_before_receipt_spend(
    tmp_path, monkeypatch, pure_session
):
    _held_lock(tmp_path, monkeypatch)
    receipt_path = _write_receipt(tmp_path, _receipt_payload())
    root = tmp_path / "proof"
    identity, cpu = _write_preflight_fixture(root, receipt_path)
    revalidation_path = root / "prelock_revalidation.json"
    revalidation = json.loads(revalidation_path.read_text(encoding="utf-8"))
    revalidation["observed"]["cpu_fixture"] = False
    revalidation_path.write_text(
        json.dumps(revalidation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="stale or malformed"):
        executor.run_held_session(
            receipt_path=receipt_path,
            ledger_path=tmp_path / "ledger.json",
            proof_root=root,
            session_identity_path=identity,
            cpu_preflight_path=cpu,
            stage_hooks=_stub_hooks([]),
        )
    assert not (tmp_path / "ledger.json").exists()


def test_normalized_commands_are_identical_across_worktrees_and_python_paths():
    first = [
        "/opt/venv-a/bin/python3.13",
        "/tmp/worktree-a/scripts/v025/m0_three_window_executor.py",
        "--manager-core-session-owner",
        "--receipt",
        "/tmp/worktree-a/.agent/sprints/receipt.json",
    ]
    second = [
        "/another/venv/bin/python3.12",
        "/srv/worktree-b/scripts/v025/m0_three_window_executor.py",
        "--manager-core-session-owner",
        "--receipt",
        "/srv/worktree-b/.agent/sprints/receipt.json",
    ]
    expected = [
        "<PYTHON>",
        "<REPO>/scripts/v025/m0_three_window_executor.py",
        "--manager-core-session-owner",
        "--receipt",
        "<REPO>/.agent/sprints/receipt.json",
    ]
    assert executor._normalize_command(first) == expected
    assert executor._normalize_command(second) == expected
    assert executor._command_sha256(first) == executor._command_sha256(second)
    assert postlock.normalize_command(first) == expected
    assert postlock.normalized_command_sha256(
        first
    ) == postlock.normalized_command_sha256(second)


def test_stale_receipt_beyond_its_max_age_is_refused(
    tmp_path, monkeypatch, pure_session
):
    import run_gpu_arm as gpu_auth

    old = datetime.now(timezone.utc) - (gpu_auth.RECEIPT_MAX_AGE + timedelta(minutes=5))
    with pytest.raises(Exception, match="old"):
        _run_session(
            tmp_path,
            monkeypatch,
            hooks=_stub_hooks([]),
            receipt=_receipt_payload(requested_at=old),
        )


def test_global_deadline_is_one_monotonic_4500s_budget(pure_session):
    with pytest.raises(session.SessionRefusal):
        session.SessionDeadline(budget_seconds=4500.001)
    with pytest.raises(session.SessionRefusal):
        session.SessionDeadline(budget_seconds=0.0)

    clock = {"ns": 0}
    deadline = session.SessionDeadline(
        budget_seconds=4500.0, clock=lambda: clock["ns"]
    )
    deadline.require("W1_COLD_1", 600.0)
    clock["ns"] = 4_000 * 1_000_000_000
    # 500s left cannot fit a 600s cold attempt: no extension path exists.
    with pytest.raises(session.SessionRefusal, match="no extension path"):
        deadline.require("W1_COLD_2", 600.0)
    clock["ns"] = 4_500 * 1_000_000_000
    with pytest.raises(session.SessionRefusal, match="deadline exceeded"):
        deadline.check("W2_PROFILED_CAPTURE")


def test_worst_branch_budget_mutation_is_refused_before_coordination(
    monkeypatch, pure_session
):
    budget = session.validate_held_budget()
    assert budget["reservation_seconds"] == 4220.0
    assert budget["headroom_seconds"] == 280.0
    mutated = tuple(session.HELD_STAGE_RESERVATIONS[:-1]) + (
        ("W3_CLEAN_MATCHED_ARM", 881.0),
    )
    monkeypatch.setattr(session, "HELD_STAGE_RESERVATIONS", mutated)
    with pytest.raises(session.SessionRefusal, match="reservations changed"):
        session.validate_held_budget()


@pytest.mark.parametrize("leader_exits_cleanly", [False, True])
def test_registered_process_group_kills_and_reaps_descendants(
    tmp_path, leader_exits_cleanly
):
    pid_path = tmp_path / "descendant.pid"
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM,lambda *_:None);"
        "time.sleep(300)"
    )
    leader_tail = "raise SystemExit(0)" if leader_exits_cleanly else "time.sleep(300)"
    leader_code = (
        "import subprocess,sys,time;"
        "from pathlib import Path;"
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"Path({str(pid_path)!r}).write_text(str(child.pid));"
        f"{leader_tail}"
    )
    registry = executor.SessionProcessGroups()
    kwargs = {
        "stage": "CPU_ORPHAN_MUTATION",
        "timeout_seconds": 2.0 if leader_exits_cleanly else 0.15,
        "environment": dict(os.environ),
        "log_path": tmp_path / "child.log",
        "registry": registry,
        "cwd": REPO,
    }
    if leader_exits_cleanly:
        assert executor._run_registered_command(
            [sys.executable, "-c", leader_code], **kwargs
        ) == 0
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            executor._run_registered_command(
                [sys.executable, "-c", leader_code], **kwargs
            )
    deadline = time.monotonic() + 2.0
    while not pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    descendant = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)
    assert registry.proof()["status"] == "PASS"
    assert registry.proof()["all_registered_groups_empty"] is True


def test_deadline_exhaustion_suppresses_every_later_stage(pure_session):
    clock = {"ns": 0}
    deadline = session.SessionDeadline(
        budget_seconds=4500.0, clock=lambda: clock["ns"]
    )
    seen: list[str] = []

    def stage_runner(stage):
        seen.append(stage)
        clock["ns"] += 2_400 * 1_000_000_000
        return {"ok": True}

    payload = session.execute_session_graph(
        cold_runner=lambda index: {
            "process_id": 100 + index,
            "cache_path": f"/cache-{index}",
            "readiness_seconds": 10.0,
            "threshold_stop": False,
        },
        stage_runner=stage_runner,
        deadline=deadline,
    )
    assert payload["status"] == "BLOCKED"
    assert seen == ["W1_CACHED_AND_CORRECTNESS", session.C1_STAGE]
    assert payload["first_failure"]["stage"] == session.C1_STAGE
    assert payload["suppressed"] == ["W2", "W3"]


def test_no_override_retry_queue_or_refund_path_exists_in_the_driver():
    import ast

    source = (SCRIPTS / "m0_three_window_executor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_held_session"
    )
    # Compare executable code only: prose about the forbidden paths is exactly
    # what this driver should contain, and comments are not control flow.
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    documented = "\n".join(ast.unparse(node) for node in body)

    class _BlankStrings(ast.NodeTransformer):
        """Prose is not control flow: compare executable tokens only."""

        def visit_Constant(self, node):  # noqa: N802 - ast visitor name
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

    executable = "\n".join(
        ast.unparse(_BlankStrings().visit(node))
        for node in ast.parse(documented).body
    )
    # No loop can re-attempt a stage, and nothing here may re-enter the lock
    # wrapper, refund a receipt, or edit a frozen threshold.
    assert not [node for node in ast.walk(function) if isinstance(node, ast.While)]
    for forbidden in (
        "retry",
        "requeue",
        "refund",
        "unspend",
        "flock",
        "with_gpu_lock",
        "LOCK_WRAPPER",
        "_prospective_manager_session_command",
        "_manager_window_command",
        "COLD_THRESHOLD_SECONDS =",
        "GLOBAL_DEADLINE_SECONDS =",
    ):
        assert forbidden not in executable, (
            f"held session gained a {forbidden!r} path"
        )
    # The payload still has to declare, in writing, that those paths are absent.
    for field in (
        "'retry_path': None",
        "'queue_path': None",
        "'receipt_refund_path': None",
    ):
        assert field in documented
    assert executable.count("gpu_auth.authorise(") == 1
    assert executable.index("receipt.check(") < executable.index(
        "gpu_auth.authorise("
    )


def test_preemption_or_stage_exception_still_sweeps_and_fails_closed(
    tmp_path, monkeypatch, pure_session
):
    swept: list[int] = []
    monkeypatch.setattr(executor, "_sweep_group", lambda group: swept.append(group))

    def exploding():
        raise KeyboardInterrupt("manager preemption")

    payload = _run_session(
        tmp_path,
        monkeypatch,
        hooks={
            "W1": lambda: {"ok": True},
            session.C1_STAGE: exploding,
            "W2": lambda: pytest.fail("W2 ran after preemption"),
            "W3": lambda: pytest.fail("W3 ran after preemption"),
        },
    )
    assert payload["status"] == "BLOCKED"
    assert payload["orphan_sweep_completed"] is True
    assert payload["first_failure"]["stage"] == session.C1_STAGE


def test_cold_timeout_is_censored_data_not_a_safety_failure():
    assert issubclass(window_parent.WindowTimeout, window_parent.WindowRefusal)
    parent = (SCRIPTS / "m0_window_parent.py").read_text(encoding="utf-8")
    launcher = parent[parent.index("def run_session_cold_attempts("):]
    launcher = launcher[: launcher.index("\ndef run_session_w1(")]
    assert "except WindowTimeout as exc:" in launcher
    assert '"threshold_stop": True' in launcher
    assert '"readiness_seconds": None' in launcher
    # A definitive FAIL suppresses everything; a single miss never does.
    assert "definitively FAILED" in launcher
    assert 'decision["decision"] != "CONTINUE"' in launcher


def test_only_cache_key_neutral_dump_flags_may_be_added(pure_session):
    flags = window_parent.cold_dump_flags(Path("/tmp/dump"))
    assert flags == ("--xla_dump_to=/tmp/dump", "--xla_dump_hlo_as_text")
    environment = window_parent._base_child_environment(
        Path("/tmp/cache"), extra_xla_flags=flags
    )
    assert environment["XLA_FLAGS"].startswith("--xla_gpu_autotune_level=0")
    with pytest.raises(window_parent.WindowRefusal, match="non-cache-neutral"):
        window_parent._base_child_environment(
            Path("/tmp/cache"), extra_xla_flags=("--xla_gpu_autotune_level=4",)
        )


def test_prepared_cache_uses_zero_payload_copy_hardlinks_and_read_on_hit_policy(
    tmp_path, monkeypatch, pure_session
):
    import prepare_m0_matched_pair as preparer

    seed = tmp_path / "seed"
    seed.mkdir()
    source = seed / "compiled-cache-entry"
    source.write_bytes(b"immutable compiled executable")
    profiled = tmp_path / "profiled"
    clean = tmp_path / "clean"
    seed_identity = preparer.mvs.directory_tree_identity(seed)
    monkeypatch.setattr(
        preparer.mvs,
        "directory_tree_identity",
        lambda _path: pytest.fail(
            "hard-link snapshots must not re-read an already-hashed payload"
        ),
    )
    assert preparer._clone_cache(
        seed, profiled, seed_identity=seed_identity
    ) == seed_identity
    assert preparer._clone_cache(
        seed, clean, seed_identity=seed_identity
    ) == seed_identity
    assert (profiled / source.name).stat().st_ino == source.stat().st_ino
    assert (clean / source.name).stat().st_ino == source.stat().st_ino
    (profiled / "private-miss").write_bytes(b"profiled-only")
    assert not (clean / "private-miss").exists()
    assert not (seed / "private-miss").exists()

    environment = window_parent._base_child_environment(profiled)
    assert environment["GPUWRF_JAX_CACHE_LOCK"] == "0"
    assert environment["JAX_COMPILATION_CACHE_MAX_SIZE"] == "-1"
    assert environment["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] == "none"
    assert environment["GPUWRF_XLA_AUTOTUNE_CACHE"] == "0"


# --------------------------------------------------------------------------- #
# R4a - stale export and unverified nsys preamble                              #
# --------------------------------------------------------------------------- #
def test_stale_sqlite_notice_is_refused_by_the_parsers():
    text = (
        "NOTICE: Existing SQLite export found: /raw/nsys_baseline.sqlite\n"
        '"Time (%)","Total Time (ns)","Instances","Name"\n'
        '"100.0","5","1","kernel_a"\n'
    )
    with pytest.raises(pp.ProfilerParseError, match="reused an existing SQLite"):
        pp.parse_nsys_kernel_summary(text)
    trace = (
        "NOTICE: Existing SQLite export found: /raw/nsys_baseline.sqlite\n"
        '"Start (ns)","Duration (ns)","Name"\n'
        '"1","2","kernel_a"\n'
    )
    with pytest.raises(pp.ProfilerParseError, match="reused an existing SQLite"):
        pp.parse_cuda_gpu_trace(trace)


def test_verified_preamble_is_tolerated_but_unknown_preamble_is_not():
    ok = (
        "Generating SQLite file /private/export.sqlite from /raw/report.nsys-rep\n"
        "Processing [/private/export.sqlite] with "
        "[/opt/nvidia/nsight-systems/host-linux-x64/Reports/"
        "cuda_gpu_kern_sum.py]...\n"
        '"Time (%)","Total Time (ns)","Instances","Name"\n'
        '"100.0","5","1","kernel_a"\n'
    )
    assert pp.parse_nsys_kernel_summary(ok)[0]["name"] == "kernel_a"
    bad = (
        "kernel_a,999\n"
        '"Time (%)","Total Time (ns)","Instances","Name"\n'
        '"100.0","5","1","kernel_a"\n'
    )
    with pytest.raises(pp.ProfilerParseError, match="unverified nsys preamble"):
        pp.parse_nsys_kernel_summary(bad)


def test_export_forces_a_fresh_private_sqlite_and_refuses_a_reused_one(tmp_path):
    rep = tmp_path / "capture.nsys-rep"
    rep.write_bytes(b"nsys-rep-bytes")
    commands: list[list[str]] = []

    class _Completed:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def runner(command):
        commands.append(list(command))
        if command[:2] == ["nsys", "--version"]:
            return _Completed("NVIDIA Nsight Systems version 2025.5.2.266")
        if "--sqlite" in command:
            Path(command[command.index("--sqlite") + 1]).write_bytes(
                b"fresh-private-sqlite"
            )
        return _Completed(
            '"Start (ns)","Duration (ns)","Name","Instances","Total Time (ns)"\n'
            '"1","2","kernel_a","1","2"\n'
        )

    payload = nex.export_all(
        rep, tmp_path / "exports", runner=runner, run_id="run-1"
    )
    assert payload["status"] == "OK"
    assert payload["stale_sibling_sqlite_reuse_possible"] is False
    private = str(tmp_path / "exports" / "private_export.sqlite")
    assert payload["private_sqlite_export"] == private
    assert payload["forced_sqlite_export_count"] == 1
    assert payload["sqlite_report_read_count"] == len(nex.REPORTS) - 1
    report_commands = commands[1:]
    assert "--force-export=true" in report_commands[0]
    assert report_commands[0][report_commands[0].index("--sqlite") + 1] == private
    for command in report_commands[1:]:
        assert "--force-export=true" not in command
        assert "--sqlite" not in command
        assert command[-1] == private

    def stale_runner(command):
        if command[:2] == ["nsys", "--version"]:
            return _Completed("NVIDIA Nsight Systems version 2025.5.2.266")
        if "--sqlite" in command:
            Path(command[command.index("--sqlite") + 1]).write_bytes(
                b"fresh-private-sqlite"
            )
        return _Completed(
            "NOTICE: Existing SQLite export found: /raw/other_run.sqlite\n"
            '"Start (ns)","Duration (ns)","Name","Instances","Total Time (ns)"\n'
            '"1","2","kernel_a","1","2"\n'
        )

    stale = nex.export_all(
        rep, tmp_path / "exports-stale", runner=stale_runner, run_id="run-1"
    )
    assert stale["status"] == "BLOCKED"
    assert all(
        record["status"] == "BLOCKED_STALE_EXPORT"
        for record in stale["exports"].values()
    )


# --------------------------------------------------------------------------- #
# R4b/R4c - malformed transfers and the no-replacement production path         #
# --------------------------------------------------------------------------- #
def test_aggregate_transfer_rows_produce_a_named_gate_not_a_keyerror():
    aggregate = [{"name": "[CUDA memcpy HtoD]", "bytes": 4096.0}]
    gate = postlock._timestamped_row_gate(aggregate)
    assert gate["status"] == "BLOCKED"
    assert gate["unplaceable_rows"] == 1
    assert "aggregate" in gate["reason"]
    assert postlock._timestamped_row_gate(None)["status"] == "MISSING"
    good = [{"name": "k", "start_ns": 1.0, "duration_ns": 2.0}]
    assert postlock._timestamped_row_gate(good)["status"] == "PASS"


def test_census_survives_untimestamped_rows_and_blocks(pure_session):
    import m0_c1_c2_cpu_proofs as proofs

    inputs = proofs.complete_census_inputs()
    inputs["cuda_rows"] = [
        {"name": "[CUDA memcpy HtoD]", "bytes": 4096.0},
    ]
    census = postlock.build_census(**inputs)
    assert census["status"] == "BLOCKED"
    assert "cuda_trace_row_integrity" in census["gates_not_pass"]
    postlock.validate_census(census)


def test_replacement_is_reserved_for_the_legacy_dry_path(tmp_path):
    rep = tmp_path / "capture.nsys-rep"
    rep.write_bytes(b"bytes")
    with pytest.raises(nex.ExportFailed, match="legacy dry path"):
        nex.export_all(
            rep, tmp_path / "out", runner=lambda command: None, allow_replace=True
        )
    driver = (SCRIPTS / "step1_driver.py").read_text(encoding="utf-8")
    assert "legacy_dry_path=True" in driver


# --------------------------------------------------------------------------- #
# R4d - bounded compiler live-buffer sidecar                                   #
# --------------------------------------------------------------------------- #
def _dump(tmp_path: Path, *, buffer_text: str, live_text: str) -> Path:
    root = tmp_path / "xla_dump"
    root.mkdir(parents=True, exist_ok=True)
    (root / "module_0000.fast-buffer-assignment.txt").write_text(
        buffer_text, encoding="utf-8"
    )
    (root / "module_0000.fast-live-range.txt").write_text(
        live_text, encoding="utf-8"
    )
    return root


BUFFER_TEXT = """BufferAssignment:
allocation 0: size 400, parameter 0, shape |f32[100]| at ShapeIndex {}:
 value: <0 param.1> (size=400,offset=0): f32[100]
allocation 1: size 800, preallocated temp
 value: <1 fusion.2{0} (phi)> (size=600,offset=0): f32[150]
 value: <2 add.3> (size=200,offset=600): f32[50]

BufferAssignment stats:
             parameter allocation:        400B
              constant allocation:          0B
        maybe_live_out allocation:        200B
     preallocated temp allocation:        800B
  preallocated temp fragmentation:          0B (0.00%)
                 total allocation:       1200B
Total bytes used: 1200
"""

LIVE_TEXT = """HloLiveRange (max 4):
  InstructionSequence:
    0:param.1
    1:fusion.2
    2:add.3
  BufferLiveRange:
    param.1{}:0-1
    fusion.2{0}:1-2
    add.3{}:2-3
"""


def test_sidecar_computes_real_peak_live_bytes_from_compiler_evidence(tmp_path):
    root = _dump(tmp_path, buffer_text=BUFFER_TEXT, live_text=LIVE_TEXT)
    payload = livebuf.collect_sidecar(root, run_id="cpu-fixture")
    assert payload["status"] == "PASS"
    # param.1 (400) and fusion.2 (600) are simultaneously live at index 1.
    assert payload["peak_live_buffers"] == 1000
    assert (
        payload["peak_live_buffers_source"]
        == "compiler_buffer_assignment_and_live_range_dump"
    )
    assert payload["aggregate_temporary_bytes_is_not_peak_live"] is True
    assert livebuf.validate_sidecar(payload)["peak_live_buffers"] == 1000
    # It is emphatically not the aggregate temp allocation.
    assert payload["peak_live_buffers"] != 800


def test_sidecar_refuses_an_incomplete_join_instead_of_understating_the_peak(
    tmp_path,
):
    root = _dump(
        tmp_path,
        buffer_text=BUFFER_TEXT,
        live_text=LIVE_TEXT + "    unknown.9{}:0-3\n",
    )
    with pytest.raises(livebuf.SidecarRefusal) as excinfo:
        livebuf.collect_sidecar(root, run_id="cpu-fixture")
    assert excinfo.value.status == "BLOCKED_SIDECAR_EVIDENCE"
    assert "join is incomplete" in str(excinfo.value)


def test_sidecar_bounds_refuse_oversized_or_overlong_dumps(tmp_path):
    root = _dump(tmp_path, buffer_text=BUFFER_TEXT, live_text=LIVE_TEXT)
    tiny = livebuf.SidecarBounds(max_file_bytes=16)
    with pytest.raises(livebuf.SidecarRefusal) as excinfo:
        livebuf.collect_sidecar(root, run_id="cpu-fixture", bounds=tiny)
    assert excinfo.value.status == "BLOCKED_SIDECAR_BOUNDS"
    few_lines = livebuf.SidecarBounds(max_lines=2)
    with pytest.raises(livebuf.SidecarRefusal) as excinfo:
        livebuf.collect_sidecar(root, run_id="cpu-fixture", bounds=few_lines)
    assert excinfo.value.status == "BLOCKED_SIDECAR_BOUNDS"
    few_files = livebuf.SidecarBounds(max_files=1)
    with pytest.raises(livebuf.SidecarRefusal) as excinfo:
        livebuf.collect_sidecar(root, run_id="cpu-fixture", bounds=few_files)
    assert excinfo.value.status == "BLOCKED_SIDECAR_BOUNDS"


def test_toolchain_discriminators_are_verified_against_the_installed_xla():
    toolchain = livebuf.verify_toolchain()
    assert toolchain["status"] == "PASS"
    assert toolchain["dump_flags_excluded_from_cache_key"] is True
    assert Path(toolchain["library"]).is_file()


def test_missing_toolchain_discriminator_returns_blocked_toolchain_evidence(
    tmp_path,
):
    fake = tmp_path / "libjax_common.so"
    fake.write_bytes(b"no dump format strings here")
    with pytest.raises(livebuf.SidecarRefusal) as excinfo:
        livebuf.verify_toolchain(library=fake)
    assert excinfo.value.status == "BLOCKED_TOOLCHAIN_EVIDENCE"
    assert "buffer_assignment_dump_suffix" in excinfo.value.detail[
        "missing_discriminator"
    ]
    refusal = livebuf.refusal_payload(excinfo.value, run_id="cpu-fixture")
    assert refusal["peak_live_buffers"] == "MISSING"
    assert refusal["gpu_session_permitted"] is False


def test_aggregate_temporary_bytes_can_never_be_relabelled_as_peak(tmp_path):
    root = _dump(tmp_path, buffer_text=BUFFER_TEXT, live_text=LIVE_TEXT)
    payload = livebuf.collect_sidecar(root, run_id="cpu-fixture")
    payload["peak_live_buffers_source"] = "compiled_memory_analysis_temporary_bytes"
    payload["sidecar_sha256"] = livebuf.canonical_sha256(
        {key: value for key, value in payload.items() if key != "sidecar_sha256"}
    )
    with pytest.raises(livebuf.SidecarRefusal, match="aggregate"):
        livebuf.validate_sidecar(payload)


def test_census_hlo_gate_defers_peak_live_and_ignores_the_future_sidecar(
    tmp_path, pure_session
):
    import m0_c1_c2_cpu_proofs as proofs

    inputs = proofs.complete_census_inputs()
    green = postlock.build_census(**inputs)
    assert green["required_gates"]["hlo_completeness"]["status"] == "PASS"
    assert green["hlo"]["peak_live_buffers"] == "MISSING"
    assert green["required_gates"]["hlo_completeness"][
        "peak_live_buffers"
    ] == "MISSING"
    assert "deferred, waived=false" in green["required_gates"][
        "hlo_completeness"
    ]["peak_live_buffers_rule"]
    postlock.validate_census(green)

    root = _dump(tmp_path, buffer_text=BUFFER_TEXT, live_text=LIVE_TEXT)
    with pytest.raises(TypeError, match="live_buffer_sidecar"):
        postlock.build_census(
            **inputs,
            live_buffer_sidecar=livebuf.collect_sidecar(
                root, run_id="cpu-fixture"
            ),
        )
    source = (SCRIPTS / "m0_three_window_executor.py").read_text(encoding="utf-8")
    held = source[source.index("def run_held_session("):]
    held = held[: held.index("\ndef run_session_owner(")]
    owner = source[source.index("def run_session_owner("):]
    owner = owner[: owner.index("\ndef _execute_live_graph(")]
    assert "m0_hlo_live_buffers" not in held + owner
    assert "xla_dump" not in held.lower() + owner.lower()


# --------------------------------------------------------------------------- #
# R3 - the post-lock M0-CORE finalizer                                         #
# --------------------------------------------------------------------------- #
def test_m0_core_gate_subset_excludes_the_deferred_diagnostic():
    assert "production_representativeness" not in postlock.M0_CORE_CENSUS_GATES
    assert postlock.M0_DEFERRED_CENSUS_GATES == (
        "production_representativeness",
    )
    assert not set(postlock.M0_CORE_CENSUS_GATES) & set(
        postlock.M0_DEFERRED_CENSUS_GATES
    )
    assert "hlo_completeness" in postlock.M0_CORE_CENSUS_GATES
    assert "cuda_trace_row_integrity" in postlock.M0_CORE_CENSUS_GATES
    assert set(postlock.M0_DEFERRED_FIELDS) == {
        "instruction_counter_E_baseline_T_ceiling",
        "native_pallas_sm120",
        "compiler_peak_live_buffers",
        "ncu_registers_occupancy_stalls_dram_l2",
        "two_domain_rho_family_coverage_trace",
        "multi_size_vram_scaling_fit",
    }


def _m0_core_manifest(**overrides) -> dict:
    payload = {
        "schema": postlock.M0_CORE_SCHEMA,
        "status": "M0_CORE_EVIDENCE_COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "machine_authority": ["cpu_closure_gate", "milestone_partition"],
        "cpu_closure_gate": "PASS",
        "m0_core_blocking_fields": [],
        "m0_core_blocking_census_gates": [],
        "may_open_m1": False,
        "milestone_partition": {
            "M0_CORE": {
                "status": "PASS",
                "census_gate_subset": list(postlock.M0_CORE_CENSUS_GATES),
                "census_gate_status": {
                    name: "PASS" for name in postlock.M0_CORE_CENSUS_GATES
                },
                "may_open_m1": False,
            },
            "M0_DEFERRED": {
                "status": "MISSING_BY_NAMED_FUTURE_MILESTONE",
                "waived": False,
                "fields": dict(postlock.M0_DEFERRED_FIELDS),
                "census_gates": {"production_representativeness": "MISSING"},
                "governed_by_m0_core_verdict": False,
            },
        },
        "full_census_status": "BLOCKED",
        "full_census_gates_not_pass": ["production_representativeness"],
        "release_gate": "MISSING",
        "session_root": {
            "session": {"sha256": "a" * 64},
            "bidirectional_bindings": "PASS",
        },
        "artifacts": {
            "held_session_root": {"sha256": "a" * 64},
            "session_lock_release": {"sha256": "b" * 64},
        },
        "jax_imported": False,
        "device_action": False,
    }
    payload.update(overrides)
    payload["manifest_sha256"] = postlock.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    return payload


def test_m0_core_verdict_is_reachable_while_deferred_evidence_is_missing():
    verdict = postlock.validate_m0_core_manifest(_m0_core_manifest())
    assert verdict["m0_core_gate"] == "PASS"
    assert verdict["blocking_census_gates"] == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            {"milestone_partition": None},
            "M0-CORE census gate subset changed",
        ),
        ({"may_open_m1": True}, "open M1"),
        ({"release_gate": "OK"}, "release gate"),
        (
            {"machine_authority": ["status"]},
            "machine authority changed",
        ),
        (
            {"full_census_status": "OK"},
            "contradicts",
        ),
        (
            {"session_root": None},
            "held-session/bidirectional root binding",
        ),
        (
            {
                "artifacts": {
                    "held_session_root": {"sha256": "f" * 64},
                    "session_lock_release": {"sha256": "b" * 64},
                }
            },
            "held-session/bidirectional root binding",
        ),
        (
            {"jax_imported": True},
            "held-session/bidirectional root binding",
        ),
    ],
)
def test_m0_core_manifest_mutations_fail_closed(mutation, match):
    payload = _m0_core_manifest(**mutation)
    with pytest.raises(postlock.PostlockRefusal, match=match):
        postlock.validate_m0_core_manifest(payload)


def test_deferred_evidence_cannot_be_waived_or_promoted():
    payload = _m0_core_manifest()
    payload["milestone_partition"]["M0_DEFERRED"]["waived"] = True
    payload["manifest_sha256"] = postlock.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    with pytest.raises(postlock.PostlockRefusal, match="waived"):
        postlock.validate_m0_core_manifest(payload)

    payload = _m0_core_manifest()
    payload["milestone_partition"]["M0_DEFERRED"]["governed_by_m0_core_verdict"] = True
    payload["manifest_sha256"] = postlock.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    with pytest.raises(postlock.PostlockRefusal, match="promoted"):
        postlock.validate_m0_core_manifest(payload)


def test_blocking_gate_cannot_be_hidden_behind_a_green_string():
    payload = _m0_core_manifest()
    payload["milestone_partition"]["M0_CORE"]["census_gate_status"][
        "hlo_completeness"
    ] = "MISSING"
    payload["manifest_sha256"] = postlock.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    with pytest.raises(postlock.PostlockRefusal, match="inventory is stale"):
        postlock.validate_m0_core_manifest(payload)

    payload["m0_core_blocking_census_gates"] = ["hlo_completeness"]
    payload["manifest_sha256"] = postlock.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    with pytest.raises(postlock.PostlockRefusal, match="hides a blocking"):
        postlock.validate_m0_core_manifest(payload)


def test_finalizer_stale_hash_is_refused():
    payload = _m0_core_manifest()
    payload["cpu_closure_gate"] = "PASS"
    payload["status"] = "M0_CORE_EVIDENCE_COMPLETE_PENDING_INDEPENDENT_REVIEW"
    payload["manifest_sha256"] = "0" * 64
    with pytest.raises(postlock.PostlockRefusal, match="content hash changed"):
        postlock.validate_m0_core_manifest(payload)


def test_finalizer_source_rehashes_session_and_all_retained_raw_artifacts():
    source = (SCRIPTS / "m0_postlock_census.py").read_text(encoding="utf-8")
    finalizer = source[source.index("def finalize_m0_core("):]
    finalizer = finalizer[: finalizer.index("\ndef validate_m0_core_manifest(")]
    assert "_validate_session_chain(" in finalizer
    assert "_rehash_census_raw(census)" in finalizer
    chain = source[source.index("def _validate_session_chain("):]
    chain = chain[: chain.index("\ndef finalize_m0_core(")]
    for binding in (
        "session_proof_sha256",
        "receipt_fingerprint",
        "selected_cold_stage",
        "cpu_preflight",
        "prelock_revalidation",
        "window_bindings",
        "session_inputs",
        "prepared_cache_sha256",
    ):
        assert binding in chain


# --------------------------------------------------------------------------- #
# hostile-environment / device-denied proof                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module",
    [
        "m0_core_session_protocol",
        "m0_hlo_live_buffers",
        "m0_postlock_census",
        "m0_three_window_executor",
        "m0_window_parent",
    ],
)
def test_session_modules_import_no_accelerator_in_a_hostile_environment(module):
    """Even with CUDA advertised and JAX told to use it, nothing reaches it."""

    code = f"""
import json, sys
sys.path.insert(0, {str(SCRIPTS)!r})
import {module}  # noqa: F401
print(json.dumps({{
    "jax": any(name == "jax" or name.startswith("jax.") for name in sys.modules),
    "jaxlib": any(
        name == "jaxlib" or name.startswith("jaxlib.") for name in sys.modules
    ),
    "gpuwrf": any(
        name == "gpuwrf" or name.startswith("gpuwrf.") for name in sys.modules
    ),
}}))
"""
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cuda",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        ["taskset", "-c", ",".join(map(str, sorted(os.sched_getaffinity(0)))),
         sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "jax": False,
        "jaxlib": False,
        "gpuwrf": False,
    }


def test_executable_dry_run_of_the_whole_session_in_a_fresh_process(
    tmp_path, frozen_session_git
):
    """Drive the real driver end to end in a device-denied fresh process."""

    dry_lock = tmp_path / "dry-run-gpu.lock"
    dry_lock.touch()
    receipt_path = _write_receipt(
        tmp_path, _receipt_payload(lock_path=dry_lock)
    )
    identity_path, cpu_path = _write_preflight_fixture(
        tmp_path / "proof", receipt_path
    )
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import m0_core_session_protocol as session
import m0_three_window_executor as executor
executor.REPO = Path({str(frozen_session_git)!r})

root = Path({str(tmp_path)!r})
lock = Path({str(dry_lock)!r})
holder = root / "lock.holder"
token = "gpuwrf-lock-dry-token"
import fcntl, os
lock_fd = os.open(lock, os.O_RDWR | os.O_APPEND)
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
holder.write_text(
    "holder=" + session.SESSION_LABEL
    + " pid=" + str(os.getpid()) + " token=" + token + " cmd=cpu-test\\n"
)
executor.CANONICAL_GPU_LOCK = lock
os.environ["GPUWRF_GPU_LOCK_HELD"] = "1"
os.environ["GPUWRF_GPU_LOCK_TOKEN"] = token
os.environ["GPUWRF_GPU_LOCK_HOLDER_FILE"] = str(holder)
os.environ["GPUWRF_GPU_LOCK_LABEL"] = session.SESSION_LABEL
os.environ["GPUWRF_GPU_LOCK_FD"] = str(lock_fd)
os.environ["GPUWRF_GPU_LOCK_FILE"] = str(lock.absolute())

order = []


def stub(name):
    def run():
        order.append(name)
        return {{"cpu_stub": True}}
    return run


payload = executor.run_held_session(
    receipt_path=Path({str(receipt_path)!r}),
    ledger_path=root / "ledger.json",
    proof_root=root / "proof",
    session_identity_path=Path({str(identity_path)!r}),
    cpu_preflight_path=Path({str(cpu_path)!r}),
    stage_hooks={{
        "W1": stub("W1"),
        session.C1_STAGE: stub(session.C1_STAGE),
        "W2": stub("W2"),
        "W3": stub("W3"),
    }},
)
print(json.dumps({{
    "status": payload["status"],
    "order": order,
    "spends": len(json.loads((root / "ledger.json").read_text())["spent"]),
    "proof_written": Path(payload.get("session_proof_path", "")).is_file(),
    "elapsed_within_cap": payload["elapsed_seconds"] < 4500.0,
    "jax": any(n == "jax" or n.startswith("jax.") for n in sys.modules),
    "gpuwrf": any(n == "gpuwrf" or n.startswith("gpuwrf.") for n in sys.modules),
}}))
"""
    environment = dict(os.environ)
    environment.update(DEVICE_FREE_ENV)
    completed = subprocess.run(
        ["taskset", "-c", ",".join(map(str, sorted(os.sched_getaffinity(0)))),
         sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "status": "PASS",
        "order": ["W1", session.C1_STAGE, "W2", "W3"],
        "spends": 1,
        "proof_written": True,
        "elapsed_within_cap": True,
        "jax": False,
        "gpuwrf": False,
    }


def test_session_source_contract_still_holds():
    result = exact.validate_sources(
        child_path=SCRIPTS / "m0_exact_boundary_child.py",
        parent_path=SCRIPTS / "m0_window_parent.py",
        parent_death_guard_path=SCRIPTS / "m0_parent_death_guard.py",
        executor_path=SCRIPTS / "m0_three_window_executor.py",
        pallas_path=SCRIPTS / "pallas_sm120_spike.py",
    )
    assert result["status"] == "PASS"
