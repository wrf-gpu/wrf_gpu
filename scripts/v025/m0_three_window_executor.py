#!/usr/bin/env python3
"""Amendment-6 one-owner plan, CPU dry executor, and manager entrypoint.

The inspect/dry paths remain JAX-free and never read a receipt.  Amendment 6
supersedes the former per-window CLI entrypoints: they now refuse before
reading a receipt, while the one outer owner delegates held callbacks directly
to :mod:`m0_window_parent`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
LOCK_WRAPPER = REPO / "scripts/with_gpu_lock.sh"
SCRIPT_DIR = SCRIPT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_window as rw  # noqa: E402
import m0_core_session_protocol as core_session  # noqa: E402
import m0_window_parent as window_parent  # noqa: E402


STATUS = "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
SCHEMA = "wrf_gpu2.v025.m0.autotune0_three_window_plan.v1"
DRY_SCHEMA = "wrf_gpu2.v025.m0.autotune0_three_window_dry_run.v1"
WINDOW_DEADLINE_SECONDS = 1500.0
SPRINT = ".agent/sprints/2026-07-27-v0250-m0-setup"
PROOF_ROOT = "proofs/v025/m0/autotune0_three_window_r5"
RAW_ROOT = "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-three-window-r5"
PAIR_CACHE_ROOT = (
    "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-qualified-pair-cache-r5"
)
R5_SPRINT = ".agent/sprints/2026-08-09-v0250-m0-r5-kernel-lock-authority"
# The fixed R5 receipt *template* path: absent today, written only when fresh
# coordination is granted.  The authoritative object stays the fingerprint-keyed
# global spend ledger; this pathname carries no authority by itself.
SESSION_RECEIPT = f"{R5_SPRINT}/M0_CORE_W1_W2_W3_SESSION_RECEIPT_R5.json"
CANONICAL_GPU_LOCK = Path("/tmp/wrf_gpu2_gpu.lock")


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        "\0".join(_normalize_command(command)).encode("utf-8")
    ).hexdigest()


def _normalize_command(command: Sequence[str]) -> list[str]:
    """Remove worktree and interpreter installation paths from semantics."""

    normalized: list[str] = []
    repo_prefix = f"{REPO.resolve()}{os.sep}"
    portable_repo_markers = (
        "/.agent/",
        "/proofs/",
        "/scripts/v025/",
        "/scripts/with_gpu_lock.sh",
        "/src/gpuwrf/",
        "/tests/v025/",
    )
    for raw in command:
        value = str(raw)
        executable_name = Path(value).name
        if Path(value) == Path(sys.executable) or (
            Path(value).is_absolute()
            and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable_name)
        ):
            normalized.append("<PYTHON>")
        elif value == str(REPO.resolve()):
            normalized.append("<REPO>")
        elif value.startswith(repo_prefix):
            normalized.append(f"<REPO>/{value[len(repo_prefix):]}")
        else:
            marker = next(
                (
                    candidate
                    for candidate in portable_repo_markers
                    if candidate in value
                ),
                None,
            )
            normalized.append(
                f"<REPO>{value[value.index(marker):]}"
                if marker is not None and Path(value).is_absolute()
                else value
            )
    return normalized


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _manager_window_inner_command(
    *,
    window_id: str,
    receipt: str,
    result_path: Path | None = None,
) -> list[str]:
    """Return the one inner command that spends one receipt for one window."""

    return [
        sys.executable,
        str(SCRIPT),
        "--manager-device-stage",
        window_id,
        "--receipt",
        receipt,
        "--result",
        str(result_path or window_parent.WINDOWS[window_id]["result"]),
    ]


def _manager_window_command(
    *,
    window_id: str,
    receipt: str,
    result_path: Path | None = None,
) -> list[str]:
    """Return the exact zero-wait lock-wrapper command for a whole window."""

    return [
        str(LOCK_WRAPPER),
        "--timeout",
        "0",
        "--label",
        str(window_parent.WINDOWS[window_id]["label"]),
        "--",
        *_manager_window_inner_command(
            window_id=window_id,
            receipt=receipt,
            result_path=result_path,
        ),
    ]


def _manager_outer_graph_command(*, window_id: str, receipt: str) -> list[str]:
    """The manager invokes this JAX-free owner around the exact lock command."""

    return [
        sys.executable,
        str(SCRIPT),
        "--manager-window-graph",
        window_id,
        "--receipt",
        receipt,
        "--result",
        str(window_parent.WINDOWS[window_id]["result"]),
    ]


def _held_session_wrapper_command(*, receipt: str) -> list[str]:
    """The exact zero-wait wrapper launched by the outer owner."""

    return [
        str(LOCK_WRAPPER),
        "--timeout",
        "0",
        "--label",
        core_session.SESSION_LABEL,
        "--",
        sys.executable,
        str(SCRIPT),
        "--manager-core-session-held",
        "--receipt",
        receipt,
    ]


def _prospective_manager_session_command(*, receipt: str) -> list[str]:
    """The one JAX-free outer owner command the manager invokes."""

    return [
        sys.executable,
        str(SCRIPT),
        "--manager-core-session-owner",
        "--receipt",
        receipt,
    ]


def _release_proof_path(window_id: str) -> Path:
    result = Path(window_parent.WINDOWS[window_id]["result"])
    return result.with_name(f"{result.stem}.lock-release.json")


def _post_cpu_list() -> str:
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("post-lock process has no available CPU affinity")
    return ",".join(str(cpu) for cpu in available[:4])


def _manager_post_command(
    window_id: str,
    *,
    result_path: Path | None = None,
    release_path: Path | None = None,
    output_path: Path | None = None,
    fast_pair_path: Path | None = None,
    session_proof_path: Path | None = None,
    pair_post_path: Path | None = None,
    w3_result_path: Path | None = None,
    w1_session_result_path: Path | None = None,
) -> list[str]:
    """Return the JAX-free command run only after the lock wrapper returns."""

    command = [
        "taskset",
        "-c",
        _post_cpu_list(),
        sys.executable,
        str(SCRIPT),
        "--manager-post-stage",
        window_id,
        "--w1-result",
        str(result_path or window_parent.WINDOWS[window_id]["result"]),
        "--lock-release-proof",
        str(release_path or _release_proof_path(window_id)),
    ]
    if session_proof_path is not None:
        command.extend(["--session-proof", str(session_proof_path)])
    if window_id == "W1":
        command.extend(
            [
                "--fast-pair",
                str(fast_pair_path or f"{PROOF_ROOT}/autotune0_fast_pair.json"),
                "--output",
                str(output_path or window_parent.QUALIFICATION),
            ]
        )
    else:
        command.extend(
            [
                "--output",
                str(output_path or (
                    f"{PROOF_ROOT}/"
                    f"{window_parent.WINDOWS[window_id]['run_id']}.post.json"
                )),
            ]
        )
        if window_id == "W2":
            if pair_post_path is not None:
                command.extend(["--pair-post", str(pair_post_path)])
            if w3_result_path is not None:
                command.extend(["--w3-result", str(w3_result_path)])
            if w1_session_result_path is not None:
                command.extend(
                    ["--w1-session-result", str(w1_session_result_path)]
                )
    return command


def _w1_pair_command(
    *,
    result_path: Path,
    release_path: Path,
    fast_pair_path: Path,
    cpu_run_root: Path,
) -> list[str]:
    return [
        "taskset",
        "-c",
        _post_cpu_list(),
        sys.executable,
        str(SCRIPT_DIR / "m0_w1_fast_pair.py"),
        "--w1-result",
        str(result_path),
        "--lock-release-proof",
        str(release_path),
        "--cpu-run-root",
        str(cpu_run_root),
        "--output",
        str(fast_pair_path),
    ]


def _postlock_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "GPUWRF_GPU_LOCK_HELD",
            "GPUWRF_GPU_LOCK_TOKEN",
            "GPUWRF_GPU_LOCK_HOLDER_FILE",
            "GPUWRF_GPU_LOCK_LABEL",
            "GPUWRF_GPU_LOCK_FD",
            "GPUWRF_GPU_LOCK_FILE",
            "GPUWRF_M0_EVIDENCE",
            "GPUWRF_M0_EVIDENCE_PATH",
            "GPUWRF_M0_RUN_ID",
            "GPUWRF_M0_SOURCE_SHA256",
            "GPUWRF_M0_CONFIG_SHA256",
            "GPUWRF_M0_INPUT_MANIFEST_SHA256",
            "GPUWRF_M0_DEVICE_UUID",
            "CUDA_VISIBLE_DEVICES",
            "JAX_PLATFORMS",
            "XLA_FLAGS",
        }
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _sweep_group(process_group: int) -> None:
    for signal_number, wait_seconds in (
        (signal.SIGTERM, 0.25),
        (signal.SIGKILL, 2.0),
    ):
        try:
            os.killpg(process_group, signal_number)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)


def run_outer_window_graph(
    *,
    window_id: str,
    receipt_path: Path,
    result_path: Path | None = None,
    post_output_path: Path | None = None,
    fast_pair_path: Path | None = None,
    cpu_run_root: Path | None = None,
) -> dict[str, Any]:
    """Run one lock wrapper, observe return, then launch fresh CPU post work."""

    window_parent._assert_parent_accelerator_free()
    if window_id not in window_parent.WINDOWS:
        raise window_parent.WindowRefusal(f"unknown window {window_id!r}")
    window = window_parent.WINDOWS[window_id]
    result = Path(result_path or window["result"])
    release_path = _release_proof_path(window_id)
    if result_path is not None:
        release_path = result.with_name(f"{result.stem}.lock-release.json")
    command = _manager_window_command(
        window_id=window_id,
        receipt=str(receipt_path),
        result_path=result,
    )
    log_path = result.with_name(f"{result.stem}.outer-wrapper.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(log_path):
        raise window_parent.WindowRefusal(
            f"outer wrapper log already exists: {log_path}"
        )
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(timezone.utc).isoformat()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=dict(os.environ),
            start_new_session=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process_group = os.getpgid(process.pid)
        try:
            returncode = process.wait(
                timeout=float(window["deadline_seconds"]) + 30.0
            )
        except subprocess.TimeoutExpired:
            _sweep_group(process_group)
            raise window_parent.WindowRefusal(
                f"{window_id} outer lock wrapper exceeded its deadline"
            )
        finally:
            log.flush()
            os.fsync(log.fileno())
    if returncode != 0:
        _sweep_group(process_group)
        raise window_parent.WindowRefusal(
            f"{window_id} outer lock wrapper returned {returncode}; "
            f"log tail: {log_path.read_text(errors='replace')[-2000:]}"
        )
    # The wrapper leader is reaped, but fail closed if it left anything in its
    # session.  The release endpoint is recorded only after that final sweep.
    _sweep_group(process_group)
    returned_ns = time.monotonic_ns()
    returned_utc = datetime.now(timezone.utc).isoformat()

    import m0_postlock_census as postlock

    release = postlock.build_lock_release_proof(
        window_id=window_id,
        run_id=str(window["run_id"]),
        wrapper_command=command,
        wrapper_started_monotonic_ns=started_ns,
        wrapper_returned_monotonic_ns=returned_ns,
        wrapper_returncode=returncode,
        window_result_path=result,
        output_path=release_path,
        wrapper_started_at_utc=started_utc,
        wrapper_returned_at_utc=returned_utc,
    )
    environment = _postlock_environment()
    pair_command = None
    if window_id == "W1":
        pair_path = Path(
            fast_pair_path or f"{PROOF_ROOT}/autotune0_fast_pair.json"
        )
        cpu_root = Path(
            cpu_run_root
            or "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-w1-fresh-cpu-r3"
        )
        pair_command = _w1_pair_command(
            result_path=result,
            release_path=release_path,
            fast_pair_path=pair_path,
            cpu_run_root=cpu_root,
        )
        pair_process = subprocess.run(
            pair_command,
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=600.0,
        )
        if pair_process.returncode != 0:
            raise window_parent.WindowRefusal(
                "post-W1 fresh CPU pair failed: "
                f"{(pair_process.stdout + pair_process.stderr)[-2000:]}"
            )
        fast_pair_path = pair_path
    post_command = _manager_post_command(
        window_id,
        result_path=result,
        release_path=release_path,
        output_path=post_output_path,
        fast_pair_path=fast_pair_path,
    )
    post = subprocess.run(
        post_command,
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800.0,
    )
    if post.returncode != 0:
        raise window_parent.WindowRefusal(
            f"{window_id} post-lock process returned {post.returncode}: "
            f"{(post.stdout + post.stderr)[-2000:]}"
        )
    return {
        "schema": "wrf_gpu2.v025.m0.outer_window_graph.v1",
        "status": "PASS",
        "window": window_id,
        "run_id": window["run_id"],
        "wrapper_command": command,
        "wrapper_log_path": str(log_path),
        "wrapper_log_sha256": window_parent.sha256_file(log_path),
        "lock_release_proof": release,
        "lock_release_path": str(release_path),
        "pair_command": pair_command,
        "post_command": post_command,
        "post_returncode": post.returncode,
        "release_before_analysis": True,
    }


# --------------------------------------------------------------------------- #
# Amendment-6 held session: outer owner plus exact held entrypoint              #
# --------------------------------------------------------------------------- #
SESSION_SCHEMA = "wrf_gpu2.v025.m0.core_session_execution.v1"
SESSION_OWNER_SCHEMA = "wrf_gpu2.v025.m0.core_session_owner.v1"
SESSION_IDENTITY_SCHEMA = "wrf_gpu2.v025.m0.session_preflight_identity.v1"
SESSION_REVALIDATION_SCHEMA = (
    "wrf_gpu2.v025.m0.session_prelock_revalidation.v1"
)
SESSION_PROOF_ROOT = f"{PROOF_ROOT}/held_session"
C1_CPU_RUN_ROOT = (
    "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-c1-prelock-cpu-r5"
)


def _c1_command(
    *,
    result_path: Path,
    fast_pair_path: Path,
    cpu_preflight_path: Path,
    session_identity_path: Path,
    receipt_path: Path,
) -> list[str]:
    """The 60-second in-lock comparison of the pre-staged CPU arm."""

    return [
        "taskset",
        "-c",
        "16-27",
        sys.executable,
        str(SCRIPT_DIR / "m0_w1_fast_pair.py"),
        "--w1-result",
        str(result_path),
        "--in-session-held-lock",
        "--pre-staged-cpu",
        str(cpu_preflight_path),
        "--session-identity",
        str(session_identity_path),
        "--prepare-session-pair",
        "--receipt",
        str(receipt_path),
        "--cpu-run-root",
        str(Path(C1_CPU_RUN_ROOT)),
        "--output",
        str(fast_pair_path),
    ]


def _cpu_preflight_command(
    *,
    session_identity_path: Path,
    cpu_preflight_path: Path,
    cpu_run_root: Path,
) -> list[str]:
    return [
        "taskset",
        "-c",
        "16-27",
        sys.executable,
        str(SCRIPT_DIR / "m0_w1_fast_pair.py"),
        "--prestage-cpu",
        "--session-identity",
        str(session_identity_path),
        "--cpu-run-root",
        str(cpu_run_root),
        "--output",
        str(cpu_preflight_path),
    ]


def _finalizer_command(
    *,
    census_path: Path,
    output_path: Path,
    fast_pair_path: Path | None = None,
    pair_post_path: Path | None = None,
    release_path: Path | None = None,
    session_proof_path: Path | None = None,
) -> list[str]:
    """The JAX-free M0-CORE finalizer, run after the lock wrapper returns."""

    return [
        "taskset",
        "-c",
        _post_cpu_list(),
        sys.executable,
        str(SCRIPT),
        "--manager-m0-core-finalizer",
        "--census",
        str(census_path),
        "--fast-pair",
        str(
            fast_pair_path
            or REPO / f"{SESSION_PROOF_ROOT}/c1_fast_pair.json"
        ),
        "--pair-post",
        str(
            pair_post_path
            or Path(
                f"{PROOF_ROOT}/"
                f"{window_parent.WINDOWS['W3']['run_id']}.post.json"
            )
        ),
        "--lock-release-proof",
        str(release_path or _release_proof_path("W3")),
        "--session-proof",
        str(
            session_proof_path
            or REPO / f"{SESSION_PROOF_ROOT}/m0_core_session.json"
        ),
        "--output",
        str(output_path),
    ]


def held_session_plan(*, receipt: str = SESSION_RECEIPT) -> dict[str, Any]:
    """The executable contract for the one held session, without running it."""

    owner_command = _prospective_manager_session_command(receipt=receipt)
    wrapper_command = _held_session_wrapper_command(receipt=receipt)
    contract = core_session.session_contract()
    return {
        "schema": SESSION_SCHEMA,
        "status": "PROSPECTIVE_NOT_RUN",
        "manager_session_command": owner_command,
        "manager_session_command_normalized": _normalize_command(owner_command),
        "manager_session_command_sha256": _command_sha256(owner_command),
        "held_wrapper_command": wrapper_command,
        "held_wrapper_command_normalized": _normalize_command(wrapper_command),
        "held_wrapper_command_sha256": _command_sha256(wrapper_command),
        "session_contract": contract,
        "held_budget": core_session.validate_held_budget(),
        "entrypoint": "--manager-core-session-owner",
        "held_entrypoint": "--manager-core-session-held",
        "entrypoint_implemented": True,
        "receipt": receipt,
        "receipt_spends": 1,
        "lock_acquisitions": 1,
        "lock_acquired_by": "the outer owner's exact zero-wait wrapper",
        "global_deadline_seconds": core_session.GLOBAL_DEADLINE_SECONDS,
        "stage_order": contract["stage_order"],
        "suppression": contract["suppression"],
        "c1_subgate": contract["c1_subgate"],
        "coordination_costs": {
            "cpu_preflight_before_lock_seconds":
                core_session.CPU_PREFLIGHT_TIMEOUT_SECONDS,
            "c1_compare_and_pair_prepare_in_lock_seconds":
                core_session.C1_COMPARATOR_LOCK_HOLD_SECONDS,
            "disclosed_to_both_managers": True,
            "cpu_preflight_inside_global_deadline": False,
            "c1_inside_global_deadline": True,
        },
        "forbidden_paths": [
            "override",
            "retry",
            "queueing",
            "receipt_refund",
            "threshold_movement",
            "second_lock_acquisition",
        ],
        "postlock": {
            "outer_owner_observes_wrapper_return": True,
            "release_proof": True,
            "jax_free": True,
            "census": "m0_postlock_census.analyze_w2",
            "finalizer": "m0_postlock_census.finalize_m0_core",
        },
        "m0_core_evidence_prerequisites": {
            "compiler_peak_live_buffers": {
                "gate": "hlo_completeness",
                "status": "MISSING",
                "waived": False,
                "owner":
                    "M1 same-set candidate diagnosis / M2 backend bake-off",
                "mechanically_reachable_from_m0_core": False,
                "dump_flags_in_w1": [],
            },
            "hlo_completeness": {
                "requires": [
                    "hash-bound static dtype/convert census",
                    "compiled executable memory_analysis",
                    "W2 measured GPU VRAM",
                    "W2 measured host RSS",
                ],
                "aggregate_temporary_is_peak_live": False,
            }
        },
    }


def _load_session_receipt(receipt_path: Path) -> dict[str, Any]:
    """Validate receipt shape and content address before anything else runs."""

    path = Path(receipt_path)
    if path.is_symlink() or not path.is_file():
        raise core_session.SessionRefusal(
            f"session receipt is missing/not a regular file: {path}"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise core_session.SessionRefusal(
            f"session receipt is malformed: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise core_session.SessionRefusal("session receipt is not a JSON object")
    return raw


def _validate_r5_kernel_lock_receipt_binding(
    receipt_raw: dict[str, Any],
    *,
    require_held_fd: bool,
    expected_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Bind both vacant snapshots and, in-lock, the inherited wrapper FD.

    The pre-receipt ``/proc/locks`` snapshots establish only that one exact
    lock object was vacant.  They never grant permission.  The held child must
    additionally prove that fd 9 (or the wrapper-exported equivalent) still
    names that same device/inode before the one coordination receipt is spent.
    """

    expected_path = str(
        Path(expected_lock_path or CANONICAL_GPU_LOCK).absolute()
    )
    binding = receipt_raw.get("kernel_lock_binding")
    if not isinstance(binding, dict):
        raise core_session.SessionRefusal(
            "R5 receipt lacks the required kernel-lock binding"
        )
    if binding.get("vacancy_grants_permission") is not False:
        raise core_session.SessionRefusal(
            "R5 receipt must state that kernel vacancy grants no permission"
        )
    preflight = binding.get("preflight")
    live = binding.get("immediate_live_recheck")
    if not isinstance(preflight, dict) or preflight.get("status") != "PASS":
        raise core_session.SessionRefusal(
            "R5 receipt lacks a passing pre-request kernel-lock preflight"
        )
    if not isinstance(live, dict) or live.get("status") != "PASS":
        raise core_session.SessionRefusal(
            "R5 receipt lacks a passing immediate kernel-lock recheck"
        )
    if (
        preflight.get("vacancy_grants_permission") is not False
        or live.get("vacancy_grants_permission") is not False
    ):
        raise core_session.SessionRefusal(
            "R5 kernel-lock snapshots incorrectly grant device permission"
        )

    pre_identity = preflight.get("kernel_identity")
    live_identity = live.get("identity")
    if not isinstance(pre_identity, dict) or not isinstance(live_identity, dict):
        raise core_session.SessionRefusal(
            "R5 receipt kernel-lock identity is missing"
        )
    identity_fields = (
        "path",
        "mode",
        "device_decimal",
        "device_major",
        "device_minor",
        "inode",
        "proc_locks_key",
    )
    if any(
        pre_identity.get(field) != live_identity.get(field)
        for field in identity_fields
    ):
        raise core_session.SessionRefusal(
            "R5 receipt preflight and immediate lock identities differ"
        )
    if (
        pre_identity.get("path") != expected_path
        or pre_identity.get("is_regular") is not True
        or pre_identity.get("is_symlink") is not False
    ):
        raise core_session.SessionRefusal(
            "R5 receipt does not bind the exact expected regular lock file"
        )
    try:
        device = int(pre_identity["device_decimal"])
        major = int(pre_identity["device_major"])
        minor = int(pre_identity["device_minor"])
        inode = int(pre_identity["inode"])
        mode = int(pre_identity["mode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise core_session.SessionRefusal(
            "R5 receipt lock identity has malformed numeric fields"
        ) from exc
    if (
        os.major(device) != major
        or os.minor(device) != minor
        or pre_identity.get("proc_locks_key")
        != f"{major:02x}:{minor:02x}:{inode}"
    ):
        raise core_session.SessionRefusal(
            "R5 receipt lock identity is internally inconsistent"
        )

    result = {
        "status": "PASS",
        "expected_path": expected_path,
        "device_decimal": device,
        "inode": inode,
        "proc_locks_key": pre_identity["proc_locks_key"],
        "held_fd_checked": False,
    }
    if not require_held_fd:
        return result

    exported_path = os.environ.get("GPUWRF_GPU_LOCK_FILE", "")
    fd_text = os.environ.get("GPUWRF_GPU_LOCK_FD", "")
    if exported_path != expected_path or not fd_text:
        raise core_session.SessionRefusal(
            "held wrapper does not export the receipt-bound lock path/fd"
        )
    try:
        fd = int(fd_text)
        held_stat = os.fstat(fd)
        path_stat = os.lstat(expected_path)
    except (TypeError, ValueError, OSError) as exc:
        raise core_session.SessionRefusal(
            f"cannot verify the held wrapper lock fd: {exc}"
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise core_session.SessionRefusal(
            "held wrapper lock path is no longer one regular file"
        )
    observed = (held_stat.st_dev, held_stat.st_ino, held_stat.st_mode)
    path_observed = (path_stat.st_dev, path_stat.st_ino, path_stat.st_mode)
    expected = (device, inode, mode)
    if observed != expected or path_observed != expected:
        raise core_session.SessionRefusal(
            "held wrapper fd/path no longer names the receipt-bound lock inode"
        )
    import m0_kernel_lock_authority as lock_authority

    held_snapshot = lock_authority.capture_kernel_lock_status(
        Path(expected_path)
    )
    held_records = held_snapshot.get("matching_records") or []
    if (
        held_snapshot.get("verdict") != "BLOCKED_OCCUPIED"
        or not held_records
        or not any(
            record.get("kind") == "FLOCK"
            and record.get("mode") == "WRITE"
            and record.get("device_major") == major
            and record.get("device_minor") == minor
            and record.get("inode") == inode
            for record in held_records
        )
    ):
        raise core_session.SessionRefusal(
            "held wrapper fd has no exact kernel WRITE/FLOCK record"
        )
    result.update(
        {
            "held_fd_checked": True,
            "held_fd": fd,
            "held_kernel_record_count": len(held_records),
            "held_kernel_snapshot_content_address": held_snapshot.get(
                "content_address"
            ),
        }
    )
    return result


def _git_object(object_name: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", object_name],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=10.0,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise core_session.SessionRefusal(
            f"cannot resolve frozen git identity {object_name!r}"
        )
    return value


def _tool_identity(invoked_path: Path) -> dict[str, Any]:
    """Bind the executable actually reached through an invocation path.

    Python, nsys and WRF are installed behind symlinks on the target host.
    Rejecting the symlink makes the frozen command unexecutable; trusting only
    the symlink leaves a retargeting gap.  Record both, hash the resolved regular
    file, and require the same resolution again immediately before the lock.
    """

    invoked = Path(invoked_path)
    try:
        resolved = invoked.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise core_session.SessionRefusal(
            f"preflight tool path cannot be resolved: {invoked}: {exc}"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise core_session.SessionRefusal(
            f"preflight tool target is missing/not regular: {resolved}"
        )
    before = resolved.stat()
    digest = window_parent.sha256_file(resolved)
    after = resolved.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if tuple(getattr(before, field) for field in stable_fields) != tuple(
        getattr(after, field) for field in stable_fields
    ):
        raise core_session.SessionRefusal(
            f"preflight tool changed while hashing: {resolved}"
        )
    return {
        "invoked_path": str(invoked),
        "path": str(resolved),
        "invoked_via_symlink": invoked.is_symlink(),
        "device": after.st_dev,
        "inode": after.st_ino,
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def build_session_preflight_identity(
    *,
    output_path: Path,
    receipt_path: Path,
    tool_runner: Callable[[Sequence[str]], Any] | None = None,
) -> dict[str, Any]:
    """Bind every non-device input before the CPU preflight or receipt spend."""

    window_parent._assert_parent_accelerator_free()
    core_session.assert_accelerator_free()
    budget = core_session.validate_held_budget()
    w1_precondition = window_parent.validate_pre_authorization("W1")
    import fast_case
    import m0_vram_sampler as mvs
    import m0_review10_fallback_capability as fallback_capability
    import wrf_source_authority as wsa

    source_inputs = fast_case.verify_source_inputs()
    run_dir = Path(window_parent.FAST_RUN_DIR)
    namelist = run_dir / "namelist.input"
    authority_environment = dict(os.environ)
    for variable in wsa.ROOT_ENV_VARS:
        authority_environment.setdefault(variable, str(wsa.CANONICAL_ROOT))
    authority = wsa.build_source_authority(
        namelist_path=namelist,
        environ=authority_environment,
    )
    authority_path = Path(output_path).parent / "wrf_source_authority.json"
    if os.path.lexists(authority_path):
        raise core_session.SessionRefusal(
            f"source authority artifact already exists: {authority_path}"
        )
    wsa.write_authority(authority_path, authority)
    boundary_path = Path(output_path).parent / "cpu_real_boundary_preflight.json"
    boundary_command = [
        sys.executable,
        str(SCRIPT_DIR / "m0_cpu_boundary_preflight.py"),
        "--authority",
        str(authority_path),
        "--run-dir",
        str(run_dir),
        "--output",
        str(boundary_path),
    ]
    boundary_environment = {
        key: value
        for key, value in authority_environment.items()
        if not key.startswith("GPUWRF_GPU_LOCK_")
    }
    boundary_environment.update(wsa.child_environment_binding(authority))
    boundary_environment.update(
        {
            wsa.AUTHORITY_ENV_VAR: str(authority_path),
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "GPUWRF_JAX_CACHE": "0",
            "GPUWRF_JAX_CACHE_LOCK": "0",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    boundary_run = subprocess.run(
        boundary_command,
        cwd=REPO,
        env=boundary_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300.0,
    )
    if boundary_run.returncode != 0 or not boundary_path.is_file():
        raise core_session.SessionRefusal(
            "real CPU boundary preflight failed before receipt/lock/device: "
            + (boundary_run.stderr or boundary_run.stdout)[-2000:]
        )
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    if (
        boundary.get("status") != "PASS"
        or boundary.get("device_action") is not False
        or boundary.get("authority", {}).get("authority_sha256")
        != authority["authority_sha256"]
    ):
        raise core_session.SessionRefusal(
            "real CPU boundary preflight produced invalid evidence"
        )
    capability = fallback_capability.build_declaration(
        boundary,
        boundary_path=boundary_path,
        environ=boundary_environment,
    )
    capability_path = (
        Path(output_path).parent / "capture_capability_declaration.json"
    )
    window_parent._atomic_json_no_replace(capability_path, capability)
    # These three artifacts are the one-path session authorities.  Make later
    # accidental writes fail at the filesystem boundary as well as by hash.
    for immutable_path in (authority_path, boundary_path, capability_path):
        os.chmod(immutable_path, 0o444)
    input_manifest = mvs.input_manifest_sha256(run_dir)
    src_tree = _git_object("HEAD:src/gpuwrf")
    expected_tree = "a6885ceded260df2f5777d7366d75a5d38947cb7"
    if src_tree != expected_tree:
        raise core_session.SessionRefusal(
            f"src/gpuwrf tree changed: {src_tree} != {expected_tree}"
        )
    runner = tool_runner or (
        lambda command: subprocess.run(
            list(command),
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    )
    nsys = shutil.which("nsys")
    if nsys is None:
        raise core_session.SessionRefusal("nsys is absent from PATH")
    version = runner(["nsys", "--version"])
    if version.returncode != 0:
        raise core_session.SessionRefusal("nsys --version failed in preflight")
    tools = {}
    for role, path in {
        "python": Path(sys.executable),
        "nsys": Path(nsys),
        "mpirun": Path(fast_case.MPIRUN),
        "wrf_exe": Path(fast_case.WRF_EXE),
    }.items():
        tools[role] = _tool_identity(path)
    owner_command = _prospective_manager_session_command(
        receipt=str(receipt_path)
    )
    wrapper_command = _held_session_wrapper_command(
        receipt=str(receipt_path)
    )
    payload = {
        "schema": SESSION_IDENTITY_SCHEMA,
        "status": "PASS",
        "src_gpuwrf_tree": src_tree,
        "source_inputs": source_inputs,
        "source_inputs_sha256": _canonical_sha256(source_inputs),
        "wrf_source_authority": {
            "path": str(authority_path),
            "sha256": window_parent.sha256_file(authority_path),
            "content_sha256": authority["authority_sha256"],
        },
        "cpu_real_boundary_preflight": {
            "path": str(boundary_path),
            "sha256": window_parent.sha256_file(boundary_path),
            "content_sha256": boundary["boundary_sha256"],
            "observations": boundary["observations"],
            "wall_seconds": boundary["timing"]["wall_seconds"],
            "peak_rss_bytes": boundary["peak_rss"]["peak_bytes"],
        },
        "capture_capability_declaration": {
            "path": str(capability_path),
            "sha256": window_parent.sha256_file(capability_path),
            "content_sha256": capability["content_address"]["sha256"],
            "denominator_method": capability["denominator"]["method"],
            "denominator_steps": capability["denominator"]["steps"],
            "available": list(capability["available"]),
            "missing": list(capability["missing"]),
        },
        "config": {
            "path": str(namelist),
            "sha256": window_parent.sha256_file(namelist),
            "case_descriptor_sha256": _canonical_sha256(
                fast_case.case_descriptor()
            ),
        },
        "input_manifest_sha256": input_manifest,
        "cache": {
            "w1_seed_path": str(window_parent.WINDOWS["W1"]["cache_path"]),
            "required_state": "ABSENT",
            "observed_absent": w1_precondition["unique_empty_cache_absent"],
        },
        "tools": tools,
        "nsys_version": (version.stdout or version.stderr or "").strip(),
        "budget": budget,
        "owner_command_normalized": _normalize_command(owner_command),
        "owner_command_sha256": _command_sha256(owner_command),
        "held_wrapper_command_normalized": _normalize_command(wrapper_command),
        "held_wrapper_command_sha256": _command_sha256(wrapper_command),
        "receipt_path": str(Path(receipt_path)),
        "device_action": False,
    }
    payload["identity_sha256"] = _canonical_sha256(payload)
    window_parent._atomic_json_no_replace(output_path, payload)
    return payload


def validate_session_preflight_identity(
    path: Path,
    *,
    receipt_path: Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise core_session.SessionRefusal("session preflight identity is not an object")
    expected = _canonical_sha256(
        {key: value for key, value in payload.items()
         if key != "identity_sha256"}
    )
    wrapper = _held_session_wrapper_command(receipt=str(receipt_path))
    owner = _prospective_manager_session_command(receipt=str(receipt_path))
    if (
        payload.get("schema") != SESSION_IDENTITY_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("identity_sha256") != expected
        or payload.get("src_gpuwrf_tree")
        != "a6885ceded260df2f5777d7366d75a5d38947cb7"
        or payload.get("src_gpuwrf_tree") != _git_object("HEAD:src/gpuwrf")
        or payload.get("owner_command_normalized") != _normalize_command(owner)
        or payload.get("owner_command_sha256") != _command_sha256(owner)
        or payload.get("held_wrapper_command_normalized")
        != _normalize_command(wrapper)
        or payload.get("held_wrapper_command_sha256")
        != _command_sha256(wrapper)
        or payload.get("device_action") is not False
    ):
        raise core_session.SessionRefusal(
            "session preflight identity is stale or binds a different command"
        )
    core_session.validate_held_budget()
    import m0_review10_fallback_capability as fallback_capability
    import wrf_source_authority as wsa

    authority_record = payload.get("wrf_source_authority") or {}
    boundary_record = payload.get("cpu_real_boundary_preflight") or {}
    capability_record = payload.get("capture_capability_declaration") or {}
    try:
        authority_path = Path(str(authority_record["path"]))
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        wsa.validate_source_authority(authority, require_env_match=False)
        boundary_path = Path(str(boundary_record["path"]))
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        fallback_capability.validate_real_boundary(
            boundary,
            expected_authority_sha256=authority["authority_sha256"],
        )
        capability_path = Path(str(capability_record["path"]))
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        fallback_capability.validate_declaration(capability)
    except (
        KeyError,
        OSError,
        json.JSONDecodeError,
        wsa.SourceAuthorityRefusal,
        fallback_capability.CapabilityRefusal,
    ) as exc:
        raise core_session.SessionRefusal(
            f"WRF/boundary identity is unavailable: {exc}"
        ) from exc
    if (
        window_parent.sha256_file(authority_path) != authority_record.get("sha256")
        or authority.get("authority_sha256")
        != authority_record.get("content_sha256")
        or window_parent.sha256_file(boundary_path) != boundary_record.get("sha256")
        or boundary.get("boundary_sha256")
        != boundary_record.get("content_sha256")
        or boundary.get("status") != "PASS"
        or window_parent.sha256_file(capability_path)
        != capability_record.get("sha256")
        or (capability.get("content_address") or {}).get("sha256")
        != capability_record.get("content_sha256")
        or capability.get("status") != "PASS"
    ):
        raise core_session.SessionRefusal("WRF/boundary preflight identity drifted")
    return payload


def revalidate_session_preflight_inputs(
    identity: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Re-derive every frozen CPU input immediately before lock acquisition."""

    import fast_case
    import m0_vram_sampler as mvs
    import m0_review10_fallback_capability as fallback_capability
    import wrf_source_authority as wsa

    expected_tools = identity.get("tools")
    if not isinstance(expected_tools, dict) or set(expected_tools) != {
        "python",
        "nsys",
        "mpirun",
        "wrf_exe",
    }:
        raise core_session.SessionRefusal(
            "session identity has an incomplete tool manifest"
        )
    observed_source_inputs = fast_case.verify_source_inputs()
    config = identity.get("config")
    if not isinstance(config, dict):
        raise core_session.SessionRefusal(
            "session identity has no frozen config identity"
        )
    config_path = Path(str(config.get("path", "")))
    observed_tools = {
        role: _tool_identity(Path(str(expected["invoked_path"])))
        for role, expected in expected_tools.items()
        if isinstance(expected, dict) and "invoked_path" in expected
    }
    observed_precondition = window_parent.validate_pre_authorization("W1")
    authority_record = identity.get("wrf_source_authority") or {}
    boundary_record = identity.get("cpu_real_boundary_preflight") or {}
    capability_record = identity.get("capture_capability_declaration") or {}
    authority_path = Path(str(authority_record.get("path", "")))
    try:
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        observed_authority = wsa.validate_source_authority(
            authority, require_env_match=False
        )
        boundary_path = Path(str(boundary_record.get("path", "")))
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        fallback_capability.validate_real_boundary(
            boundary,
            expected_authority_sha256=observed_authority["authority_sha256"],
        )
        capability_path = Path(str(capability_record.get("path", "")))
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        fallback_capability.validate_declaration(capability)
    except (
        OSError,
        json.JSONDecodeError,
        wsa.SourceAuthorityRefusal,
        fallback_capability.CapabilityRefusal,
    ) as exc:
        raise core_session.SessionRefusal(
            f"WRF source authority revalidation failed: {exc}"
        ) from exc
    observed = {
        "src_gpuwrf_tree": _git_object("HEAD:src/gpuwrf"),
        "source_inputs": observed_source_inputs,
        "source_inputs_sha256": _canonical_sha256(observed_source_inputs),
        "config_sha256": window_parent.sha256_file(config_path),
        "case_descriptor_sha256": _canonical_sha256(
            fast_case.case_descriptor()
        ),
        "input_manifest_sha256": mvs.input_manifest_sha256(
            Path(window_parent.FAST_RUN_DIR)
        ),
        "cache": {
            "w1_seed_path": str(window_parent.WINDOWS["W1"]["cache_path"]),
            "required_state": "ABSENT",
            "observed_absent":
                observed_precondition["unique_empty_cache_absent"],
        },
        "tools": observed_tools,
        "wrf_source_authority_content_sha256":
            observed_authority["authority_sha256"],
        "cpu_real_boundary_file_sha256":
            window_parent.sha256_file(boundary_path),
        "cpu_real_boundary_content_sha256": boundary.get("boundary_sha256"),
        "capture_capability_file_sha256":
            window_parent.sha256_file(capability_path),
        "capture_capability_content_sha256":
            (capability.get("content_address") or {}).get("sha256"),
    }
    expected = {
        "src_gpuwrf_tree": identity.get("src_gpuwrf_tree"),
        "source_inputs": identity.get("source_inputs"),
        "source_inputs_sha256": identity.get("source_inputs_sha256"),
        "config_sha256": config.get("sha256"),
        "case_descriptor_sha256": config.get("case_descriptor_sha256"),
        "input_manifest_sha256": identity.get("input_manifest_sha256"),
        "cache": identity.get("cache"),
        "tools": expected_tools,
        "wrf_source_authority_content_sha256":
            authority_record.get("content_sha256"),
        "cpu_real_boundary_file_sha256": boundary_record.get("sha256"),
        "cpu_real_boundary_content_sha256":
            boundary_record.get("content_sha256"),
        "capture_capability_file_sha256": capability_record.get("sha256"),
        "capture_capability_content_sha256":
            capability_record.get("content_sha256"),
    }
    if observed != expected:
        mismatches = sorted(
            key for key in expected if observed.get(key) != expected.get(key)
        )
        raise core_session.SessionRefusal(
            "frozen preflight inputs changed during the CPU comparator: "
            + ", ".join(mismatches)
        )
    payload = {
        "schema": SESSION_REVALIDATION_SCHEMA,
        "status": "PASS",
        "session_identity_content_sha256": identity["identity_sha256"],
        "observed": observed,
        "device_action": False,
    }
    payload["revalidation_sha256"] = _canonical_sha256(payload)
    if output_path is not None:
        window_parent._atomic_json_no_replace(output_path, payload)
    return payload


def validate_session_prelock_revalidation(
    path: Path,
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate the owner's last CPU-only check without re-hashing in-lock."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise core_session.SessionRefusal(
            f"session pre-lock revalidation is unavailable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise core_session.SessionRefusal(
            "session pre-lock revalidation is not an object"
        )
    expected_hash = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "revalidation_sha256"
        }
    )
    if (
        payload.get("schema") != SESSION_REVALIDATION_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("revalidation_sha256") != expected_hash
        or payload.get("session_identity_content_sha256")
        != identity.get("identity_sha256")
        or payload.get("device_action") is not False
    ):
        raise core_session.SessionRefusal(
            "session pre-lock revalidation is stale or malformed"
        )
    for role, tool in (identity.get("tools") or {}).items():
        try:
            invoked = Path(str(tool["invoked_path"]))
            resolved = invoked.resolve(strict=True)
            stat = resolved.stat()
        except (KeyError, OSError, RuntimeError) as exc:
            raise core_session.SessionRefusal(
                f"frozen tool {role} cannot be resolved at held entry: {exc}"
            ) from exc
        if (
            str(resolved) != tool.get("path")
            or stat.st_dev != tool.get("device")
            or stat.st_ino != tool.get("inode")
            or stat.st_size != tool.get("bytes")
            or stat.st_mtime_ns != tool.get("mtime_ns")
        ):
            raise core_session.SessionRefusal(
                f"frozen tool {role} changed between preflight and held entry"
            )
    return payload


class SessionProcessGroups:
    """Register-before-wait process groups and prove every group empty."""

    def __init__(self) -> None:
        self.active: dict[int, str] = {}
        self.completed: list[dict[str, Any]] = []

    @staticmethod
    def exists(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def register(self, process_group: int, stage: str) -> None:
        if (
            isinstance(process_group, bool)
            or not isinstance(process_group, int)
            or process_group <= 0
            or process_group == os.getpgrp()
            or process_group in self.active
        ):
            raise core_session.SessionRefusal(
                f"invalid/duplicate process group for {stage}: {process_group}"
            )
        self.active[process_group] = stage

    def complete(self, process_group: int, stage: str) -> None:
        if self.active.get(process_group) != stage:
            raise core_session.SessionRefusal(
                f"unregistered process group completion for {stage}: "
                f"{process_group}"
            )
        if self.exists(process_group):
            raise core_session.SessionRefusal(
                f"{stage} process group {process_group} is not empty"
            )
        self.completed.append(
            {
                "process_group": process_group,
                "stage": stage,
                "empty": True,
            }
        )
        del self.active[process_group]

    def sweep(self) -> None:
        failures = []
        for process_group, stage in list(self.active.items()):
            _sweep_group(process_group)
            if self.exists(process_group):
                failures.append((process_group, stage))
                continue
            self.complete(process_group, stage)
        if failures:
            raise core_session.SessionRefusal(
                f"process-group sweep left descendants: {failures!r}"
            )

    def proof(self) -> dict[str, Any]:
        return {
            "status": "PASS" if not self.active else "BLOCKED",
            "registered": len(self.completed) + len(self.active),
            "completed": list(self.completed),
            "active": [
                {"process_group": group, "stage": stage}
                for group, stage in sorted(self.active.items())
            ],
            "all_registered_groups_empty": not self.active,
        }


def _run_registered_command(
    command: Sequence[str],
    *,
    stage: str,
    timeout_seconds: float,
    environment: dict[str, str],
    log_path: Path,
    registry: SessionProcessGroups,
    cwd: Path = REPO,
    deadline: Any | None = None,
) -> int:
    """Run one fresh process group, clamped to a deadline, then reap it."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(log_path):
        raise core_session.SessionRefusal(f"process log already exists: {log_path}")
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            start_new_session=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # start_new_session=True makes the child PID the process-group/session
        # leader by construction.  Using the PID avoids a getpgid race when a
        # deliberately hostile fixture exits between Popen and registration.
        process_group = process.pid
        registry.register(process_group, stage)
        try:
            timeout = (
                deadline.timeout_for(stage, timeout_seconds)
                if deadline is not None
                else timeout_seconds
            )
            returncode = process.wait(timeout=timeout)
            if deadline is not None:
                deadline.check_after(stage)
        except BaseException:
            _sweep_group(process_group)
            try:
                process.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
            if not registry.exists(process_group):
                registry.complete(process_group, stage)
            raise
        finally:
            stream.flush()
            os.fsync(stream.fileno())
        _sweep_group(process_group)
        try:
            process.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
        registry.complete(process_group, stage)
        return returncode


def run_held_session(
    *,
    receipt_path: Path,
    ledger_path: Path = window_parent.LEDGER,
    deadline: Any | None = None,
    proof_root: Path | None = None,
    cpu_preflight_path: Path | None = None,
    session_identity_path: Path | None = None,
    stage_hooks: dict[str, Callable[[], Any]] | None = None,
    process_groups: SessionProcessGroups | None = None,
) -> dict[str, Any]:
    """Execute the whole M0-CORE session inside one already-held lock.

    Order is mechanical: receipt shape, content-addressed spend, held-lock
    proof, then W1 (two-of-three cold + cached), then the C1 sub-gate, then W2,
    then W3 -- every later stage unreachable the moment an earlier one fails.
    This function has no escape hatch of any kind and it never acquires a lock:
    the outer owner's wrapper already holds the only one. Amendment 6's forbidden
    paths are enumerated in :func:`held_session_plan`, and a source-level test
    asserts none of them exists here.
    """

    started_ns = time.monotonic_ns()
    started_utc = datetime.now(timezone.utc).isoformat()
    window_parent._assert_parent_accelerator_free()
    core_session.assert_accelerator_free()
    budget = core_session.validate_held_budget()
    deadline = deadline or core_session.SessionDeadline()
    root = Path(
        proof_root
        or os.environ.get(
            "GPUWRF_M0_SESSION_PROOF_ROOT",
            str(REPO / SESSION_PROOF_ROOT),
        )
    )
    identity_path = Path(
        session_identity_path
        or os.environ.get(
            "GPUWRF_M0_SESSION_IDENTITY",
            str(root / "session_preflight_identity.json"),
        )
    )
    cpu_path = Path(
        cpu_preflight_path
        or os.environ.get(
            "GPUWRF_M0_CPU_PREFLIGHT",
            str(root / "cpu_preflight.json"),
        )
    )
    revalidation_path = Path(
        os.environ.get(
            "GPUWRF_M0_PREFLIGHT_REVALIDATION",
            str(root / "prelock_revalidation.json"),
        )
    )
    identity = validate_session_preflight_identity(
        identity_path, receipt_path=Path(receipt_path)
    )
    prelock_revalidation = validate_session_prelock_revalidation(
        revalidation_path, identity=identity
    )
    import m0_w1_fast_pair as w1_pair

    cpu_preflight, _cpu_record = w1_pair.validate_cpu_preflight(
        cpu_path, expected_session_identity_path=identity_path
    )
    registry = process_groups or SessionProcessGroups()

    receipt_raw = _load_session_receipt(Path(receipt_path))
    receipt_shape = core_session.validate_receipt_shape(receipt_raw)
    import run_gpu_arm as gpu_auth

    # Validate the canonical wrapper's token/label first, without spending, so
    # an unwrapped or stale-environment invocation retains its precise refusal.
    gpu_auth.check_canonical_lock(expected_label=core_session.SESSION_LABEL)
    kernel_lock_binding = _validate_r5_kernel_lock_receipt_binding(
        receipt_raw, require_held_fd=True
    )
    # The already-held zero-wait lock is proven BEFORE the receipt is spent, so
    # an unwrapped invocation can never consume the one coordination.
    receipt = gpu_auth.CoordinationReceipt.load(Path(receipt_path))
    receipt.check(core_session.SESSION_LABEL, ledger_path=Path(ledger_path))
    if receipt.fingerprint() != receipt_shape["fingerprint"]:
        raise core_session.SessionRefusal(
            "session receipt content address does not match its coordination"
        )
    authorization_result = gpu_auth.authorise(
        core_session.SESSION_LABEL,
        receipt_path=Path(receipt_path),
        env=None,
        ledger_path=Path(ledger_path),
    )
    lock = authorization_result["lock"]
    spend = authorization_result["spend_record"]
    authorization = {
        "window": "M0_CORE_SESSION",
        "label": core_session.SESSION_LABEL,
        "receipt_path": str(Path(receipt_path)),
        "receipt_fingerprint": receipt.fingerprint(),
        "receipt_spent_at_utc": spend["spent_at_utc"],
        "spend_record": spend,
        "canonical_lock": lock,
        "kernel_lock_binding": kernel_lock_binding,
        "lock_acquisitions": 1,
        "cpu_preflight_path": str(cpu_path.resolve()),
        "cpu_preflight_sha256": window_parent.sha256_file(cpu_path),
        "cpu_preflight_content_sha256": cpu_preflight["preflight_sha256"],
        "session_identity_path": str(identity_path.resolve()),
        "session_identity_sha256": window_parent.sha256_file(identity_path),
        "session_identity_content_sha256": identity["identity_sha256"],
        "prelock_revalidation_path": str(revalidation_path.resolve()),
        "prelock_revalidation_sha256": window_parent.sha256_file(
            revalidation_path
        ),
        "prelock_revalidation_content_sha256":
            prelock_revalidation["revalidation_sha256"],
        "spent_after_cpu_preflight": True,
        "spent_after_input_revalidation": True,
        "spent_before_any_device_child": True,
    }

    stage_details: dict[str, Any] = {}
    hooks = stage_hooks or {}
    w1: dict[str, Any] = {}

    def cold_and_cached() -> Any:
        if "W1" in hooks:
            return hooks["W1"]()
        result = window_parent.run_session_w1(
            authorization=authorization,
            session=core_session,
            deadline=deadline,
            prerequisite=window_parent.validate_pre_authorization("W1"),
            register_process_group=registry.register,
            complete_process_group=registry.complete,
        )
        w1.update(result)
        result_path = Path(window_parent.WINDOWS["W1"]["result"])
        return {
            "run_id": result["payload"]["run_id"],
            "result_path": str(result_path),
            "result_bytes": result_path.stat().st_size,
            "result_sha256": window_parent.sha256_file(result_path),
            "cold_decision": result["cold"]["decision"],
            "selected_cold_stage": result["cold"]["selected_cold_stage"],
            "selected_cold_attempt_index":
                result["cold"]["qualified_attempt_index"],
        }

    def c1_subgate() -> Any:
        if core_session.C1_STAGE in hooks:
            return hooks[core_session.C1_STAGE]()
        deadline.require(
            core_session.C1_STAGE,
            core_session.C1_COMPARATOR_LOCK_HOLD_SECONDS,
        )
        result_path = Path(window_parent.WINDOWS["W1"]["result"])
        fast_pair_path = root / "c1_fast_pair.json"
        command = _c1_command(
            result_path=result_path,
            fast_pair_path=fast_pair_path,
            cpu_preflight_path=cpu_path,
            session_identity_path=identity_path,
            receipt_path=Path(receipt_path),
        )
        log_path = root / "c1_compare.log"
        returncode = _run_registered_command(
            command,
            stage=core_session.C1_STAGE,
            timeout_seconds=core_session.C1_COMPARATOR_LOCK_HOLD_SECONDS,
            environment=dict(os.environ),
            log_path=log_path,
            registry=registry,
            deadline=deadline,
        )
        if returncode != 0:
            raise window_parent.WindowRefusal(
                "C1 sub-gate failed; W2 and W3 are unreachable: "
                f"{log_path.read_text(errors='replace')[-2000:]}"
            )
        pair = json.loads(fast_pair_path.read_text(encoding="utf-8"))
        if (
            (pair.get("c1_subgate") or {}).get("both_green") is not True
            or (pair.get("c1_subgate") or {}).get(
                "cpu_arm_pre_staged_before_receipt_and_lock"
            )
            is not True
        ):
            raise window_parent.WindowRefusal(
                "C1 sub-gate did not report both components green"
            )
        qualification = json.loads(
            window_parent.QUALIFICATION.read_text(encoding="utf-8")
        )
        prepared_path = (
            Path(window_parent.WINDOWS["W2"]["identity_path"]).parent
            / "plan.json"
        )
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        deadline.check_after(core_session.C1_STAGE)
        return {
            "command": command,
            "command_normalized": _normalize_command(command),
            "command_sha256": _command_sha256(command),
            "log_path": str(log_path),
            "log_sha256": window_parent.sha256_file(log_path),
            "fast_pair_path": str(fast_pair_path),
            "fast_pair_sha256": window_parent.sha256_file(fast_pair_path),
            "qualification_path": str(window_parent.QUALIFICATION),
            "qualification_sha256": window_parent.sha256_file(
                window_parent.QUALIFICATION
            ),
            "qualification": qualification,
            "prepared_pair_path": str(prepared_path),
            "prepared_pair_sha256": window_parent.sha256_file(prepared_path),
            "prepared_pair": prepared,
            "lock_held_during_comparator": True,
            "cpu_arm_ran_before_lock": True,
        }

    def device_window(window_id: str) -> Callable[[], Any]:
        def run() -> Any:
            if window_id in hooks:
                return hooks[window_id]()
            result = window_parent.run_session_window(
                window_id=window_id,
                authorization=authorization,
                session=core_session,
                deadline=deadline,
                register_process_group=registry.register,
                complete_process_group=registry.complete,
            )
            result_path = Path(window_parent.WINDOWS[window_id]["result"])
            return {
                "run_id": result["payload"]["run_id"],
                "result_path": str(result_path),
                "result_bytes": result_path.stat().st_size,
                "result_sha256": window_parent.sha256_file(result_path),
            }

        return run

    runners = {
        "W1_CACHED_AND_CORRECTNESS": cold_and_cached,
        core_session.C1_STAGE: c1_subgate,
        "W2_PROFILED_CAPTURE": device_window("W2"),
        "W3_CLEAN_MATCHED_ARM": device_window("W3"),
    }

    def cold_runner(index: int) -> dict[str, Any]:
        # W1's cold protocol is executed inside cold_and_cached, which owns the
        # per-attempt caches. The graph-level cold callback exists only for the
        # injected CPU-stub proofs; a live session records one decided protocol.
        if "COLD" in hooks:
            return hooks["COLD"](index)
        raise core_session.SessionRefusal(
            "live sessions decide the cold protocol inside W1, not through the "
            "graph-level cold callback"
        )

    current_stage: dict[str, Any] = {"stage": None}

    def stage_runner(stage: str) -> Any:
        current_stage["stage"] = stage
        detail = runners[stage]()
        stage_details[stage] = detail
        return detail

    sweep_error: str | None = None
    try:
        if "COLD" in hooks:
            graph = core_session.execute_session_graph(
                cold_runner=cold_runner,
                stage_runner=stage_runner,
                deadline=deadline,
            )
        else:
            graph = _execute_live_graph(
                stage_runner=stage_runner, deadline=deadline
            )
        status = graph.get("status", "BLOCKED")
        error = graph.get("first_failure")
    except BaseException as exc:  # noqa: BLE001 - preemption fails closed too
        # A manager SIGINT arrives as KeyboardInterrupt, which is a BaseException.
        # Amendment 6 makes preemption invalidate the session, so it must land
        # here and publish a refusal rather than unwinding past the sweep.
        preempted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        graph = {"status": "BLOCKED", "stages": [], "suppressed": ["every_later_stage"]}
        status = "BLOCKED"
        error = {
            "kind": "PREEMPTED" if preempted else "SESSION_REFUSAL",
            "stage": current_stage["stage"],
            "error": f"{type(exc).__name__}: {exc}",
            "receipt_invalidated": True,
        }
    finally:
        try:
            registry.sweep()
        except BaseException as exc:  # noqa: BLE001 - orphan proof is fatal
            sweep_error = f"{type(exc).__name__}: {exc}"

    elapsed = (time.monotonic_ns() - started_ns) / 1e9
    if elapsed > core_session.GLOBAL_DEADLINE_SECONDS:
        status = "BLOCKED"
        error = {
            "kind": "GLOBAL_DEADLINE_EXCEEDED",
            "error": f"session ran {elapsed:.3f}s > "
            f"{core_session.GLOBAL_DEADLINE_SECONDS:g}s",
        }
    if sweep_error is not None or registry.proof()["status"] != "PASS":
        status = "BLOCKED"
        error = {
            "kind": "PROCESS_GROUP_LEAK",
            "stage": current_stage["stage"],
            "error": sweep_error or "a registered process group remains active",
        }
    group_proof = registry.proof()
    payload = {
        "schema": SESSION_SCHEMA,
        "status": status,
        "label": core_session.SESSION_LABEL,
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "global_deadline_seconds": core_session.GLOBAL_DEADLINE_SECONDS,
        "held_budget": budget,
        "authorization": authorization,
        "session_contract": core_session.session_contract(),
        "graph": graph,
        "stage_details": stage_details,
        "first_failure": error,
        "lock_acquisitions": 1,
        "lock_released_by": "the manager's with_gpu_lock.sh on process exit",
        "postlock_analysis_runs_after_release": True,
        "receipt_refund_path": None,
        "retry_path": None,
        "queue_path": None,
        "held_process_group": os.getpgrp(),
        "process_group_proof": group_proof,
        "orphan_sweep_completed": (
            group_proof["status"] == "PASS"
            and group_proof["all_registered_groups_empty"] is True
        ),
        "session_identity": {
            "path": str(identity_path.resolve()),
            "sha256": window_parent.sha256_file(identity_path),
            "content_sha256": identity["identity_sha256"],
        },
        "prelock_revalidation": {
            "path": str(revalidation_path.resolve()),
            "sha256": window_parent.sha256_file(revalidation_path),
            "content_sha256":
                prelock_revalidation["revalidation_sha256"],
        },
        "cpu_preflight": {
            "path": str(cpu_path.resolve()),
            "sha256": window_parent.sha256_file(cpu_path),
            "content_sha256": cpu_preflight["preflight_sha256"],
        },
        "compiler_dump_flags": [],
        "native_pallas_reachable": False,
    }
    payload["session_sha256"] = _canonical_sha256(payload)
    output = root / "m0_core_session.json"
    try:
        window_parent._atomic_json_no_replace(output, payload)
        payload["session_proof_path"] = str(output)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "BLOCKED"
        payload["session_proof_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def run_session_owner(
    *,
    receipt_path: Path,
    ledger_path: Path = window_parent.LEDGER,
    proof_root: Path | None = None,
    cpu_run_root: Path = Path(C1_CPU_RUN_ROOT),
) -> dict[str, Any]:
    """Own preflight, one wrapper lifecycle, post-lock work, and finalization."""

    window_parent._assert_parent_accelerator_free()
    core_session.assert_accelerator_free()
    import m0_postlock_census as postlock
    import run_gpu_arm as gpu_auth

    postlock.assert_lock_environment_absent()
    receipt_raw = _load_session_receipt(Path(receipt_path))
    receipt_shape = core_session.validate_receipt_shape(receipt_raw)
    _validate_r5_kernel_lock_receipt_binding(
        receipt_raw, require_held_fd=False
    )
    receipt = gpu_auth.CoordinationReceipt.load(Path(receipt_path))
    receipt.check(core_session.SESSION_LABEL, ledger_path=Path(ledger_path))
    if receipt.fingerprint() != receipt_shape["fingerprint"]:
        raise core_session.SessionRefusal(
            "outer-owner receipt content address differs from coordination receipt"
        )
    root = Path(proof_root or REPO / SESSION_PROOF_ROOT)
    if os.path.lexists(root):
        raise core_session.SessionRefusal(
            f"session owner proof root already exists: {root}"
        )
    root.mkdir(parents=True, exist_ok=False)
    registry = SessionProcessGroups()
    stage = "SESSION_IDENTITY_PREFLIGHT"
    started_utc = datetime.now(timezone.utc).isoformat()
    owner_started_ns = time.monotonic_ns()
    commands: list[dict[str, Any]] = []
    session_path = root / "m0_core_session.json"
    release_path = root / "session_lock_release.json"
    terminal_path = root / "m0_core_terminal.json"
    owner_path = root / "session_owner.json"

    try:
        identity_path = root / "session_preflight_identity.json"
        identity = build_session_preflight_identity(
            output_path=identity_path,
            receipt_path=Path(receipt_path),
        )
        stage = "CPU_PREFLIGHT"
        cpu_path = root / "cpu_preflight.json"
        cpu_command = _cpu_preflight_command(
            session_identity_path=identity_path,
            cpu_preflight_path=cpu_path,
            cpu_run_root=Path(cpu_run_root),
        )
        cpu_log = root / "cpu_preflight.log"
        cpu_rc = _run_registered_command(
            cpu_command,
            stage=stage,
            timeout_seconds=core_session.CPU_PREFLIGHT_TIMEOUT_SECONDS,
            environment=_postlock_environment(),
            log_path=cpu_log,
            registry=registry,
        )
        commands.append(
            {
                "stage": stage,
                "command": cpu_command,
                "command_normalized": _normalize_command(cpu_command),
                "command_sha256": _command_sha256(cpu_command),
                "returncode": cpu_rc,
                "log_path": str(cpu_log),
                "log_sha256": window_parent.sha256_file(cpu_log),
            }
        )
        if cpu_rc != 0:
            raise core_session.SessionRefusal(
                "fresh pre-lock CPU-WRF comparator failed before receipt spend: "
                f"{cpu_log.read_text(errors='replace')[-2000:]}"
            )
        import m0_w1_fast_pair as w1_pair

        w1_pair.validate_cpu_preflight(
            cpu_path, expected_session_identity_path=identity_path
        )
        stage = "PRELOCK_INPUT_REVALIDATION"
        revalidation_path = root / "prelock_revalidation.json"
        prelock_revalidation = revalidate_session_preflight_inputs(
            identity, output_path=revalidation_path
        )

        stage = "HELD_ZERO_WAIT_WRAPPER"
        wrapper_command = _held_session_wrapper_command(
            receipt=str(receipt_path)
        )
        wrapper_log = root / "held_wrapper.log"
        wrapper_environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "GPUWRF_M0_SESSION_IDENTITY",
                "GPUWRF_M0_CPU_PREFLIGHT",
                "GPUWRF_M0_PREFLIGHT_REVALIDATION",
                "GPUWRF_M0_SESSION_PROOF_ROOT",
            }
        }
        wrapper_environment.update(
            {
                "GPUWRF_M0_SESSION_IDENTITY": str(identity_path),
                "GPUWRF_M0_CPU_PREFLIGHT": str(cpu_path),
                "GPUWRF_M0_PREFLIGHT_REVALIDATION": str(revalidation_path),
                "GPUWRF_M0_SESSION_PROOF_ROOT": str(root),
            }
        )
        import wrf_source_authority as wsa

        authority_path = Path(identity["wrf_source_authority"]["path"])
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        wrapper_environment.update(wsa.child_environment_binding(authority))
        wrapper_environment[wsa.AUTHORITY_ENV_VAR] = str(authority_path)
        wrapper_started_ns = time.monotonic_ns()
        wrapper_started_utc = datetime.now(timezone.utc).isoformat()
        wrapper_rc = _run_registered_command(
            wrapper_command,
            stage=stage,
            timeout_seconds=core_session.GLOBAL_DEADLINE_SECONDS,
            environment=wrapper_environment,
            log_path=wrapper_log,
            registry=registry,
        )
        wrapper_returned_ns = time.monotonic_ns()
        wrapper_returned_utc = datetime.now(timezone.utc).isoformat()
        wrapper_group = next(
            item["process_group"]
            for item in reversed(registry.completed)
            if item["stage"] == stage
        )
        commands.append(
            {
                "stage": stage,
                "command": wrapper_command,
                "command_normalized": _normalize_command(wrapper_command),
                "command_sha256": _command_sha256(wrapper_command),
                "returncode": wrapper_rc,
                "log_path": str(wrapper_log),
                "log_sha256": window_parent.sha256_file(wrapper_log),
            }
        )
        if not session_path.is_file():
            raise core_session.SessionRefusal(
                f"held wrapper returned {wrapper_rc} without a session proof"
            )
        session_payload = json.loads(session_path.read_text(encoding="utf-8"))
        release = postlock.build_session_lock_release_proof(
            session_proof_path=session_path,
            wrapper_command=wrapper_command,
            wrapper_log_path=wrapper_log,
            wrapper_started_monotonic_ns=wrapper_started_ns,
            wrapper_returned_monotonic_ns=wrapper_returned_ns,
            wrapper_returncode=wrapper_rc,
            wrapper_process_group=wrapper_group,
            process_group_empty=not registry.exists(wrapper_group),
            holder_file=Path("/tmp/wrf_gpu2_gpu.lock.holder"),
            output_path=release_path,
            wrapper_started_at_utc=wrapper_started_utc,
            wrapper_returned_at_utc=wrapper_returned_utc,
        )
        if wrapper_rc != 0 or session_payload.get("status") != "PASS":
            raise core_session.SessionRefusal(
                "held session failed; post-lock census/finalizer suppressed"
            )

        post_environment = _postlock_environment()
        pair_post_path = root / "w3_pair_post.json"
        stage = "POSTLOCK_W3_PAIR"
        w3_command = _manager_post_command(
            "W3",
            result_path=Path(window_parent.WINDOWS["W3"]["result"]),
            release_path=release_path,
            output_path=pair_post_path,
            session_proof_path=session_path,
        )
        w3_log = root / "w3_post.log"
        w3_rc = _run_registered_command(
            w3_command,
            stage=stage,
            timeout_seconds=300.0,
            environment=post_environment,
            log_path=w3_log,
            registry=registry,
        )
        commands.append(
            {
                "stage": stage,
                "command": w3_command,
                "command_normalized": _normalize_command(w3_command),
                "command_sha256": _command_sha256(w3_command),
                "returncode": w3_rc,
                "log_path": str(w3_log),
                "log_sha256": window_parent.sha256_file(w3_log),
            }
        )
        if w3_rc != 0:
            raise core_session.SessionRefusal(
                f"W3 post-lock pair failed: {w3_log.read_text(errors='replace')[-2000:]}"
            )

        census_path = root / "w2_census.json"
        stage = "POSTLOCK_W2_EXPORT_CENSUS"
        w2_command = _manager_post_command(
            "W2",
            result_path=Path(window_parent.WINDOWS["W2"]["result"]),
            release_path=release_path,
            output_path=census_path,
            session_proof_path=session_path,
            pair_post_path=pair_post_path,
            w3_result_path=Path(window_parent.WINDOWS["W3"]["result"]),
            w1_session_result_path=Path(window_parent.WINDOWS["W1"]["result"]),
        )
        w2_log = root / "w2_post.log"
        w2_rc = _run_registered_command(
            w2_command,
            stage=stage,
            timeout_seconds=1_800.0,
            environment=post_environment,
            log_path=w2_log,
            registry=registry,
        )
        commands.append(
            {
                "stage": stage,
                "command": w2_command,
                "command_normalized": _normalize_command(w2_command),
                "command_sha256": _command_sha256(w2_command),
                "returncode": w2_rc,
                "log_path": str(w2_log),
                "log_sha256": window_parent.sha256_file(w2_log),
            }
        )
        if w2_rc != 0:
            raise core_session.SessionRefusal(
                f"W2 post-lock census failed: {w2_log.read_text(errors='replace')[-2000:]}"
            )

        stage = "POSTLOCK_M0_CORE_FINALIZER"
        finalizer_command = _finalizer_command(
            census_path=census_path,
            output_path=terminal_path,
            fast_pair_path=root / "c1_fast_pair.json",
            pair_post_path=pair_post_path,
            release_path=release_path,
            session_proof_path=session_path,
        )
        finalizer_log = root / "finalizer.log"
        finalizer_rc = _run_registered_command(
            finalizer_command,
            stage=stage,
            timeout_seconds=300.0,
            environment=post_environment,
            log_path=finalizer_log,
            registry=registry,
        )
        commands.append(
            {
                "stage": stage,
                "command": finalizer_command,
                "command_normalized": _normalize_command(finalizer_command),
                "command_sha256": _command_sha256(finalizer_command),
                "returncode": finalizer_rc,
                "log_path": str(finalizer_log),
                "log_sha256": window_parent.sha256_file(finalizer_log),
            }
        )
        if finalizer_rc != 0 or not terminal_path.is_file():
            raise core_session.SessionRefusal(
                "M0-CORE finalizer failed: "
                f"{finalizer_log.read_text(errors='replace')[-2000:]}"
            )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        status = (
            "PASS" if terminal.get("cpu_closure_gate") == "PASS" else "BLOCKED"
        )
        error = None
    except BaseException as exc:  # noqa: BLE001 - one terminal suppression root
        try:
            registry.sweep()
        except BaseException as sweep_exc:  # noqa: BLE001
            sweep_error = f"{type(sweep_exc).__name__}: {sweep_exc}"
        else:
            sweep_error = None
        status = "BLOCKED"
        error = {
            "stage": stage,
            "kind": (
                "PREEMPTED"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "SESSION_OWNER_REFUSAL"
            ),
            "error": f"{type(exc).__name__}: {exc}",
            "process_group_sweep_error": sweep_error,
        }
        terminal = {
            "schema": SESSION_OWNER_SCHEMA,
            "status": "M0_CORE_BLOCKED",
            "cpu_closure_gate": "BLOCKED",
            "may_open_m1": False,
            "first_failure": error,
            "receipt_consumed": (
                bool(
                    (json.loads(session_path.read_text()).get("authorization"))
                    if session_path.is_file()
                    else False
                )
            ),
            "later_stages_suppressed": True,
            "release_proof_path": (
                str(release_path) if release_path.is_file() else None
            ),
            "device_action": False if stage in {
                "SESSION_IDENTITY_PREFLIGHT",
                "CPU_PREFLIGHT",
                "PRELOCK_INPUT_REVALIDATION",
            } else "HELD_WRAPPER_MAY_HAVE_RUN",
        }
        terminal["manifest_sha256"] = _canonical_sha256(terminal)
        if not terminal_path.exists():
            window_parent._atomic_json_no_replace(terminal_path, terminal)

    owner_payload = {
        "schema": SESSION_OWNER_SCHEMA,
        "status": status,
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (time.monotonic_ns() - owner_started_ns) / 1e9,
        "receipt_path": str(Path(receipt_path)),
        "commands": commands,
        "prelock_revalidation": (
            {
                "path": str(revalidation_path),
                "sha256": window_parent.sha256_file(revalidation_path),
                "content_sha256":
                    prelock_revalidation["revalidation_sha256"],
            }
            if "prelock_revalidation" in locals()
            and revalidation_path.is_file()
            else None
        ),
        "process_group_proof": registry.proof(),
        "session_proof": (
            {
                "path": str(session_path),
                "sha256": window_parent.sha256_file(session_path),
            }
            if session_path.is_file()
            else None
        ),
        "release_proof": (
            {
                "path": str(release_path),
                "sha256": window_parent.sha256_file(release_path),
            }
            if release_path.is_file()
            else None
        ),
        "terminal_manifest": {
            "path": str(terminal_path),
            "sha256": window_parent.sha256_file(terminal_path),
            "cpu_closure_gate": terminal.get("cpu_closure_gate"),
        },
        "first_failure": error,
        "retry_path": None,
        "receipt_refund_path": None,
    }
    owner_payload["owner_sha256"] = _canonical_sha256(owner_payload)
    window_parent._atomic_json_no_replace(owner_path, owner_payload)
    owner_payload["owner_path"] = str(owner_path)
    return owner_payload


def _execute_live_graph(
    *,
    stage_runner: Callable[[str], Any],
    deadline: Any,
) -> dict[str, Any]:
    """Run the frozen post-cold stage order with live callbacks.

    The live W1 callback owns the cold protocol (it holds the per-attempt caches
    and the threshold censoring), so the graph here starts at the stage order
    Amendment 6 freezes and stops at the first failure.
    """

    executed: list[dict[str, Any]] = []
    for stage in core_session.POST_COLD_STAGE_ORDER:
        try:
            if stage == "W2_PROFILED_CAPTURE":
                # W2 is not useful without its frozen W3 matched arm.  Refuse
                # before starting W2 unless both remaining reservations fit.
                deadline.require(
                    "W2_AND_W3_REMAINING_BUDGET",
                    core_session.W2_W3_REQUIRED_SECONDS,
                )
            deadline.check(stage)
            detail = stage_runner(stage)
            deadline.check_after(stage)
        except Exception as exc:  # noqa: BLE001 - suppression is the point
            return {
                "schema": core_session.SCHEMA,
                "status": "BLOCKED",
                "stages": executed,
                "first_failure": {
                    "stage": stage,
                    "kind": (
                        "C1_SUBGATE_FAILURE"
                        if stage == core_session.C1_STAGE
                        else "SAFETY_CORRECTNESS_OR_IDENTITY_FAILURE"
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "suppressed": core_session.LATER_STAGES[stage],
            }
        executed.append({"name": stage, "status": "PASS", "detail": detail})
    return {
        "schema": core_session.SCHEMA,
        "status": "PASS",
        "stages": executed,
        "stage_order_executed": [item["name"] for item in executed],
        "first_failure": None,
        "suppressed": [],
    }


def _stage(
    *,
    window_id: str,
    name: str,
    run_id: str,
    receipt: str,
    timeout_seconds: float,
    gpu: bool = True,
    condition: str = "always",
) -> dict[str, Any]:
    result = f"{PROOF_ROOT}/{run_id}.{name}.json"
    identity_path = f"{PROOF_ROOT}/{run_id}.identity.json"
    cache_path = {
        "W1": (
            f"{RAW_ROOT}/m0-autotune0-qualify-20260728-r3/unique-empty-cache"
        ),
        "W2": (
            f"{PAIR_CACHE_ROOT}/m0-autotune0-profiled-20260728-r3"
        ),
        "W3": f"{PAIR_CACHE_ROOT}/m0-autotune0-clean-20260728-r3",
    }[window_id]
    command = _manager_window_command(window_id=window_id, receipt=receipt)
    dry_stub_command = ["CPU_DRY_STUB_ONLY", window_id, name]
    return {
        "name": name,
        "gpu": gpu,
        "timeout_seconds": timeout_seconds,
        "condition": condition,
        "command": command,
        "command_sha256": _command_sha256(command),
        "command_scope": "ONE_INVOCATION_FOR_WHOLE_WINDOW",
        "dry_stub_command": dry_stub_command,
        "dry_stub_command_sha256": _command_sha256(dry_stub_command),
        "result": result,
        "identity_path": identity_path,
        "cache_path": cache_path,
        "xla_flag": "--xla_gpu_autotune_level=0",
    }


def _window(
    *,
    window_id: str,
    label: str,
    run_id: str,
    receipt: str,
    overhead_seconds: float,
    stages: list[dict[str, Any]],
    prerequisites: list[str],
    deadline_seconds: float = WINDOW_DEADLINE_SECONDS,
) -> dict[str, Any]:
    manager_commands = {tuple(stage["command"]) for stage in stages}
    if len(manager_commands) != 1:
        raise ValueError(f"{window_id} stages do not share one manager command")
    manager_command = list(next(iter(manager_commands)))
    budget = rw.validate_budget(
        [
            rw.Stage(
                stage["name"],
                stage["dry_stub_command"],
                gpu=stage["gpu"],
                timeout_seconds=stage["timeout_seconds"],
            )
            for stage in stages
        ],
        deadline_seconds=deadline_seconds,
        overhead_seconds=overhead_seconds,
    )
    return {
        "id": window_id,
        "label": label,
        "run_id": run_id,
        "receipt": receipt,
        "receipt_scope": (
            "fresh dual-manager affirmative receipt for this label and this "
            "one zero-wait canonical-lock wrapper invocation"
        ),
        "deadline_seconds": deadline_seconds,
        "overhead_seconds": overhead_seconds,
        "budget": budget,
        "prerequisites": prerequisites,
        "manager_command": manager_command,
        "manager_command_sha256": _command_sha256(manager_command),
        "manager_command_invocations": 1,
        "stages": stages,
    }


def _cpu_preflight_plan() -> dict[str, Any]:
    cpu_run_id = "m0-autotune0-cpu-reference-20260728-r3"
    cpu_output = f"{RAW_ROOT}/{cpu_run_id}"
    cpu_result = f"{PROOF_ROOT}/{cpu_run_id}.json"
    pair_path = f"{PROOF_ROOT}/autotune0_fast_pair.json"
    cpu_command = [
        sys.executable,
        str(REPO / "scripts/v025/run_cpu_arm.py"),
        "--run-root",
        cpu_output,
        "--label",
        cpu_run_id,
        "--cpu-list",
        "16-27",
        "--hours",
        "1",
        "--out",
        cpu_result,
    ]
    stages = [
        {
            "name": "fresh_12_rank_cpu_wrf_preflight",
            "gpu": False,
            "timeout_seconds": core_session.CPU_PREFLIGHT_TIMEOUT_SECONDS,
            "command": cpu_command,
            "command_sha256": _command_sha256(cpu_command),
        }
    ]
    budget = rw.validate_budget(
        [
            rw.Stage(
                item["name"],
                item["command"],
                gpu=False,
                timeout_seconds=item["timeout_seconds"],
            )
            for item in stages
        ],
        deadline_seconds=core_session.CPU_PREFLIGHT_TIMEOUT_SECONDS,
    )
    return {
        "id": "CPU_PREFLIGHT",
        "status": "NOT_RUN",
        "device_touched": False,
        "run_id": cpu_run_id,
        "prerequisite": "all frozen non-device identities are validated",
        "sequence": [
            "validate receipt without spending it",
            "fresh CPU reference starts in a new CPU-pinned process",
            "bind the output and invocation hashes",
            "only then launch the zero-wait lock wrapper",
        ],
        "deadline_seconds": core_session.CPU_PREFLIGHT_TIMEOUT_SECONDS,
        "cpu_arm_max_seconds": 300.0,
        "budget": budget,
        "stages": stages,
        "clock_rule": (
            "the outer process hard-caps the entire fresh CPU preflight at 420s; "
            "the WRF launcher itself must remain <=300s"
        ),
        "exact_result_wrfout_adapter": (
            "IMPLEMENTED: the cached W1 child publishes the same synchronized "
            "compiled result through the production writer, and the GPU-first "
            "post-lock pair consumes its path/hash/result binding mechanically"
        ),
    }


def build_plan() -> dict[str, Any]:
    """Return the deterministic plan without probing any path or receipt."""

    w1_run = "m0-autotune0-qualify-20260728-r3"
    w2_run = "m0-autotune0-profiled-20260728-r3"
    w3_run = "m0-autotune0-clean-20260728-r3"
    # Amendment 5 replaces the three Amendment-4 receipts with one
    # content-addressed, single-use session receipt.  The window records below
    # remain useful measurement/stage definitions; none is independently
    # authorisable.
    w1_receipt = SESSION_RECEIPT
    w2_receipt = SESSION_RECEIPT
    w3_receipt = SESSION_RECEIPT

    w1 = _window(
        window_id="W1",
        label="m0-autotune0-qualification",
        run_id=w1_run,
        receipt=w1_receipt,
        overhead_seconds=0.0,
        deadline_seconds=2_160.0,
        prerequisites=[
            "unique empty cache path does not exist",
            "one fresh session receipt, not yet spent",
            "outer owner acquired the exact zero-wait session lock",
            "XLA flag exactly --xla_gpu_autotune_level=0",
        ],
        stages=[
            _stage(
                window_id="W1",
                name="cold_two_of_three_decision",
                run_id=w1_run,
                receipt=w1_receipt,
                timeout_seconds=1800.0,
            ),
            _stage(
                window_id="W1",
                name="cached_readiness_and_warm_integration",
                run_id=w1_run,
                receipt=w1_receipt,
                timeout_seconds=360.0,
            ),
        ],
    )
    w2 = _window(
        window_id="W2",
        label="m0-autotune0-profiled-census",
        run_id=w2_run,
        receipt=w2_receipt,
        overhead_seconds=180.0,
        deadline_seconds=1400.0,
        prerequisites=[
            "W1 AUTOTUNE0_QUALIFIED manifest exists and is hash-valid",
            "fresh profiled identity derived after W1",
            "profiled cache snapshot created only after W1",
            "fresh receipt for W2 label",
            "zero-wait canonical lock",
        ],
        stages=[
            _stage(
                window_id="W2",
                name="profiled_cached_readiness_and_integration",
                run_id=w2_run,
                receipt=w2_receipt,
                timeout_seconds=1100.0,
            ),
            _stage(
                window_id="W2",
                name="profiled_artifact_capture_integrity",
                run_id=w2_run,
                receipt=w2_receipt,
                timeout_seconds=120.0,
                gpu=False,
            ),
        ],
    )
    w3 = _window(
        window_id="W3",
        label="m0-autotune0-clean-matched-arm",
        run_id=w3_run,
        receipt=w3_receipt,
        overhead_seconds=180.0,
        prerequisites=[
            "W2 profiled result and identity are mechanically accepted",
            "fresh clean identity derived after W1 and matches W2 binding",
            "clean cache snapshot byte-identical to W2 snapshot",
            "frozen order is exactly [profiled.run_id, clean.run_id]",
            "fresh receipt for W3 label",
            "zero-wait canonical lock",
        ],
        stages=[
            _stage(
                window_id="W3",
                name="clean_cached_readiness_and_integration",
                run_id=w3_run,
                receipt=w3_receipt,
                timeout_seconds=360.0,
            ),
            _stage(
                window_id="W3",
                name="profiled_clean_exact_identity_gate",
                run_id=w3_run,
                receipt=w3_receipt,
                timeout_seconds=60.0,
                gpu=False,
            ),
        ],
        deadline_seconds=600.0,
    )
    session_command = _prospective_manager_session_command(
        receipt=SESSION_RECEIPT
    )
    for window in (w1, w2, w3):
        window["receipt_scope"] = (
            "one shared Amendment-5 session receipt; this window cannot be "
            "authorized or locked independently"
        )
        window["manager_command"] = session_command
        window["manager_command_sha256"] = _command_sha256(session_command)
        window["manager_command_invocations"] = 0
        for stage in window["stages"]:
            stage["command"] = session_command
            stage["command_sha256"] = _command_sha256(session_command)
            stage["command_scope"] = "ONE_INVOCATION_FOR_W1_W2_W3_SESSION"

    plan = {
        "schema": SCHEMA,
        "status": STATUS,
        "device_touched": False,
        "receipts_consumed": [],
        "production_binding": {
            "candidate_commit": "dedff5bbb4ac4fb852bdd9d42b50c1da25a61dfb",
            "src_gpuwrf_tree": "a6885ceded260df2f5777d7366d75a5d38947cb7",
            "xla_flag": "--xla_gpu_autotune_level=0",
        },
        "amendment": (
            ".agent/patches/2026-07-28-v025-m0-exact-executable-boundary.md"
        ),
        "mechanical_order": [
            "SESSION_IDENTITY_PREFLIGHT",
            "CPU_PREFLIGHT_BEFORE_SPEND_OR_LOCK",
            "ONE_ZERO_WAIT_HELD_WRAPPER",
            "W1_COLD_DECISION",
            "W1_CACHED_AND_CORRECTNESS",
            "C1_COMPARE_AND_PAIR_PREPARATION",
            "W2_PROFILED_CAPTURE",
            "W3_CLEAN_MATCHED_ARM",
            "LOCK_RELEASE",
            "W3_POST_ANALYSIS",
            "W2_EXPORT_AND_CENSUS",
            "M0_CORE_FINALIZER",
        ],
        "maximum_windows": 3,
        "maximum_lock_acquisitions": 1,
        "session_receipt": SESSION_RECEIPT,
        "manager_session_command": session_command,
        "manager_session_command_sha256": _command_sha256(session_command),
        "amendment_5_session_protocol": core_session.session_contract(),
        "amendment_5_held_session": held_session_plan(),
        "window_deadline_max_seconds": WINDOW_DEADLINE_SECONDS,
        "windows": [w1, w2, w3],
        "prelock_cpu_comparator": _cpu_preflight_plan(),
        "profiled_pair_order": [w2_run, w3_run],
        "identity_and_cache_provenance": {
            "qualification_manifest": {
                "path": f"{PROOF_ROOT}/autotune0_qualification.json",
                "status": "MISSING_UNTIL_W1",
            },
            "qualification_cache": {
                "path": f"{RAW_ROOT}/{w1_run}/unique-empty-cache",
                "pre_w1_required_state": "MUST_NOT_EXIST",
                "created_by": "W1 cold stage only after receipt and lock",
            },
            "profiled_identity": {
                "path": (
                    f"{PROOF_ROOT}/prepared_profiler_pair_r3/"
                    "profiled_identity.json"
                ),
                "status": "MUST_NOT_EXIST_UNTIL_W1_QUALIFIES",
            },
            "clean_identity": {
                "path": (
                    f"{PROOF_ROOT}/prepared_profiler_pair_r3/"
                    "clean_identity.json"
                ),
                "status": "MUST_NOT_EXIST_UNTIL_W1_QUALIFIES",
            },
            "profiled_cache_snapshot": {
                "path": f"{PAIR_CACHE_ROOT}/{w2_run}",
                "status": "MUST_NOT_EXIST_UNTIL_W1_QUALIFIES",
            },
            "clean_cache_snapshot": {
                "path": f"{PAIR_CACHE_ROOT}/{w3_run}",
                "status": "MUST_NOT_EXIST_UNTIL_W1_QUALIFIES",
            },
            "snapshot_rule": (
                "derive identities and clone two private byte-identical snapshots "
                "from the W1-qualified cache only after the W1 manifest passes"
            ),
        },
        "timing_boundaries": {
            "cold_readiness": (
                "process launch to mechanically observed executable-ready event; "
                "excludes forecast integration"
            ),
            "cached_readiness": (
                "fresh process launch to the same executable-ready event from the "
                "W1-populated cache"
            ),
            "warm_profiled_clean_integration": (
                "synchronized integration-only start/end around the same forecast "
                "range; excludes compile, cache load, input staging, output I/O, and "
                "lock queue"
            ),
            "current_gate_eligibility": "CPU_PROVEN_GPU_MEASUREMENTS_MISSING",
            "implemented_boundary": [
                "exact private production JIT lower(...).compile() return is the "
                "executable-ready event",
                "the same compiled callable and exact arguments are invoked after "
                "readiness in clean and profiled arms",
                "integration start/end directly enclose compiled invocation plus "
                "block_until_ready; no first-call clock or phase subtraction",
            ],
            "rule": (
                "No subtraction, first-call wall clock, or relabelling may satisfy "
                "either missing boundary."
            ),
        },
        "stale_pair": {
            "status": "MECHANICALLY_SUPERSEDED",
            "run_ids": [
                "m0-pair-profiled-20260728-r1",
                "m0-pair-clean-20260728-r1",
            ],
            "old_plan_sha256": (
                "213d09a6bc703294753c26613cfe93e14c2e6dcee453d92b4531a50fb6ab736e"
            ),
            "old_cache_sha256": (
                "825980d919b383833d88c80e416c20cb3f47c0e36b434ed65167288d7409c618"
            ),
            "authorisation_eligible": False,
        },
        "manager_window_commands": {
            "status": "SUPERSEDED_BY_ONE_SESSION_COMMAND",
            "independent_authorisation_allowed": False,
            "command": session_command,
        },
        "manager_post_commands": {
            window_id: _manager_post_command(window_id)
            for window_id in ("W1", "W2", "W3")
        },
        "legacy_amendment4_outer_graph_commands": {
            "status": "NON_AUTHORITATIVE_SUPERSEDED",
            "reason": (
                "retained only to preserve the tested release-before-analysis "
                "implementation; Amendment 5 permits one lock acquisition"
            ),
            "commands": {
            window_id: _manager_outer_graph_command(
                window_id=window_id,
                receipt={
                    "W1": w1_receipt,
                    "W2": w2_receipt,
                    "W3": w3_receipt,
                }[window_id],
            )
            for window_id in ("W1", "W2", "W3")
            },
        },
        "manager_window_inspection": {
            window_id: window_parent.inspect_window(window_id)
            for window_id in ("W1", "W2", "W3")
        },
        "live_execution_status": (
            "AMENDMENT5_SESSION_CONTROL_CPU_PROVEN_DEVICE_CALLBACKS_NOT_RUN"
        ),
    }
    return plan


def _rw_stages(items: list[dict[str, Any]]) -> list[rw.Stage]:
    return [
        rw.Stage(
            name=item["name"],
            command=item.get("dry_stub_command", item["command"]),
            gpu=item["gpu"],
            timeout_seconds=item["timeout_seconds"],
        )
        for item in items
    ]


def _suppressed_window(window: dict[str, Any], reason: str) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).isoformat()
    return {
        "schema": "wrf_gpu2.v025.m0.gpu_window.v1",
        "label": window["label"],
        "started_at_utc": stamp,
        "finished_at_utc": stamp,
        "ok": False,
        "first_failure": None,
        "first_failure_status": "SUPPRESSED",
        "gpu_stages_suppressed": [
            stage["name"] for stage in window["stages"] if stage["gpu"]
        ],
        "cpu_stages_suppressed": [
            stage["name"] for stage in window["stages"] if not stage["gpu"]
        ],
        "stages": [
            {
                "name": stage["name"],
                "status": rw.SUPPRESSED,
                "gpu": stage["gpu"],
                "returncode": None,
                "seconds": None,
                "detail": reason,
                "log_tail": "",
            }
            for stage in window["stages"]
        ],
        "suppression_rule": "suppressed by an earlier phase; no command executed",
    }


def dry_run_all(
    *,
    inject_failure: str | None = None,
    runner_observer: Callable[[Sequence[str]], None] | None = None,
) -> dict[str, Any]:
    """Exercise all windows and cross-window suppression without device action."""

    plan = build_plan()
    command_key: dict[tuple[str, ...], str] = {}
    for window in plan["windows"]:
        for stage in window["stages"]:
            command_key[tuple(stage.get("dry_stub_command", stage["command"]))] = (
                f"{window['id']}:{stage['name']}"
            )
    for stage in plan["prelock_cpu_comparator"]["stages"]:
        command_key[tuple(stage["command"])] = f"CPU_PREFLIGHT:{stage['name']}"

    executed: list[str] = []

    def runner(command, *, cwd, env, timeout):
        key = command_key[tuple(command)]
        executed.append(key)
        if runner_observer is not None:
            runner_observer(command)
        if key == inject_failure:
            return 97, f"[CPU stub injected failure {key}]"
        return 0, f"[CPU stub success {key}; no subprocess/device/receipt]"

    results: list[dict[str, Any]] = []
    stop_reason: str | None = None
    cpu = plan["prelock_cpu_comparator"]
    cpu_result = rw.run_window(
        label="cpu-preflight-before-held-wrapper",
        stages=_rw_stages(cpu["stages"]),
        cwd=REPO,
        env={},
        runner=runner,
        deadline_seconds=core_session.CPU_PREFLIGHT_TIMEOUT_SECONDS,
    ).as_dict()
    if not cpu_result["ok"]:
        stop_reason = (
            "CPU preflight failed before receipt spend/lock; every held stage "
            "is suppressed"
        )
    results.append({"phase": "CPU_PREFLIGHT", "result": cpu_result})
    for index, window in enumerate(plan["windows"]):
        if stop_reason is not None:
            window_result = _suppressed_window(window, stop_reason)
        else:
            window_result = rw.run_window(
                label=window["label"],
                stages=_rw_stages(window["stages"]),
                cwd=REPO,
                env={},
                runner=runner,
                deadline_seconds=window["deadline_seconds"],
                overhead_seconds=window["overhead_seconds"],
            ).as_dict()
            if not window_result["ok"]:
                stop_reason = (
                    f"{window['id']} failed; every later GPU stage/window is suppressed"
                )
        results.append({"phase": window["id"], "result": window_result})

    known = set(command_key.values())
    if inject_failure is not None and inject_failure not in known:
        raise ValueError(
            f"unknown injected failure {inject_failure!r}; expected one of {sorted(known)}"
        )
    return {
        "schema": DRY_SCHEMA,
        "status": STATUS,
        "scenario": (
            "ALL_STUBS_OK"
            if inject_failure is None
            else f"INJECTED_FAILURE:{inject_failure}"
        ),
        "device_touched": False,
        "device_queries": [],
        "device_imports": [],
        "receipts_read": [],
        "receipts_consumed": [],
        "cache_snapshots_created": [],
        "plan_sha256": _canonical_sha256(plan),
        "executed_stub_stage_keys": executed,
        "results": results,
        "suppression_verified": True,
        "gpu_evidence_status": "MISSING",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--dry-run-all", action="store_true")
    mode.add_argument("--manager-device-stage")
    mode.add_argument("--manager-post-stage")
    mode.add_argument("--manager-window-graph")
    mode.add_argument(
        "--manager-core-session-owner",
        action="store_true",
        help=(
            "Amendment-6 outer owner: preflight CPU-WRF before spend/lock, "
            "launch one exact zero-wait held wrapper, then finalize JAX-free"
        ),
    )
    mode.add_argument(
        "--manager-core-session-held",
        action="store_true",
        help=(
            "Amendment-6 inner entrypoint: run W1 -> C1 -> W2 -> W3 "
            "inside the lock the manager's with_gpu_lock.sh already holds"
        ),
    )
    mode.add_argument("--print-session-plan", action="store_true")
    mode.add_argument("--manager-m0-core-finalizer", action="store_true")
    parser.add_argument("--census", type=Path)
    parser.add_argument("--pair-post", type=Path)
    parser.add_argument("--session-proof", type=Path)
    parser.add_argument("--w3-result", type=Path)
    parser.add_argument("--w1-session-result", type=Path)
    parser.add_argument("--inject-failure")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--xla-flag")
    parser.add_argument("--qualification-manifest", type=Path)
    parser.add_argument("--identity-path", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--ledger-path", type=Path, default=window_parent.LEDGER)
    parser.add_argument("--w1-result", type=Path)
    parser.add_argument("--fast-pair", type=Path)
    parser.add_argument("--lock-release-proof", type=Path)
    parser.add_argument("--cpu-run-root", type=Path)
    args = parser.parse_args()

    if args.print_plan:
        print(json.dumps(build_plan(), indent=2, sort_keys=True))
        return 0
    if args.dry_run_all:
        payload = dry_run_all(inject_failure=args.inject_failure)
        if args.output is not None:
            _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.print_session_plan:
        print(json.dumps(held_session_plan(), indent=2, sort_keys=True))
        return 0

    if args.manager_core_session_owner:
        if args.receipt is None:
            raise SystemExit("--manager-core-session-owner requires --receipt")
        try:
            payload = run_session_owner(
                receipt_path=args.receipt,
                ledger_path=args.ledger_path,
                cpu_run_root=Path(args.cpu_run_root or C1_CPU_RUN_ROOT),
            )
        except Exception as exc:  # noqa: BLE001 - fail closed before device use
            refusal = {
                "schema": SESSION_OWNER_SCHEMA,
                "status": "BLOCKED",
                "cpu_closure_gate": "BLOCKED",
                "receipt_consumed": False,
                "device_touched": False,
                "jax_imported": "jax" in sys.modules,
                "gpuwrf_imported": "gpuwrf" in sys.modules,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(json.dumps(refusal, indent=2, sort_keys=True))
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if payload.get("status") == "PASS" else 1

    if args.manager_core_session_held:
        if args.receipt is None:
            raise SystemExit("--manager-core-session-held requires --receipt")
        try:
            payload = run_held_session(
                receipt_path=args.receipt,
                ledger_path=args.ledger_path,
            )
        except Exception as exc:  # noqa: BLE001 - refusal stays pre-import
            refusal = {
                "schema": SESSION_SCHEMA,
                "status": "REFUSED_PRE_DEVICE_IMPORT",
                "label": core_session.SESSION_LABEL,
                "receipt_consumed": False,
                "device_touched": False,
                "jax_imported": "jax" in sys.modules,
                "gpuwrf_imported": "gpuwrf" in sys.modules,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(json.dumps(refusal, indent=2, sort_keys=True))
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if payload.get("status") == "PASS" else 1

    if args.manager_m0_core_finalizer:
        if (
            args.census is None
            or args.output is None
            or args.session_proof is None
        ):
            raise SystemExit(
                "--manager-m0-core-finalizer requires --census, "
                "--session-proof, and --output"
            )
        import m0_postlock_census as postlock

        try:
            payload = postlock.finalize_m0_core(
                w1_result_path=Path(window_parent.WINDOWS["W1"]["result"]),
                w2_result_path=Path(window_parent.WINDOWS["W2"]["result"]),
                w3_result_path=Path(window_parent.WINDOWS["W3"]["result"]),
                census_path=args.census,
                release_path=(
                    args.lock_release_proof or _release_proof_path("W3")
                ),
                fast_pair_path=(
                    args.fast_pair or REPO / f"{SESSION_PROOF_ROOT}/c1_fast_pair.json"
                ),
                qualification_path=window_parent.QUALIFICATION,
                pair_post_path=(
                    args.pair_post
                    or Path(
                        f"{PROOF_ROOT}/"
                        f"{window_parent.WINDOWS['W3']['run_id']}.post.json"
                    )
                ),
                session_proof_path=args.session_proof,
                output_path=args.output,
            )
        except Exception as exc:  # noqa: BLE001 - terminal boundary fails closed
            refusal = {
                "schema": postlock.M0_CORE_SCHEMA,
                "status": "M0_CORE_BLOCKED",
                "cpu_closure_gate": "BLOCKED",
                "may_open_m1": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "jax_imported": "jax" in sys.modules,
                "device_action": False,
            }
            print(json.dumps(refusal, indent=2, sort_keys=True))
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if payload.get("cpu_closure_gate") == "PASS" else 1

    if args.manager_post_stage:
        try:
            payload = window_parent.post_analysis(
                phase=args.manager_post_stage,
                w1_result=args.w1_result,
                fast_pair=args.fast_pair,
                output=args.output,
                lock_release_proof=args.lock_release_proof,
                session_proof=args.session_proof,
                pair_post=args.pair_post,
                w3_result=args.w3_result,
                w1_session_result=args.w1_session_result,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
            refusal = {
                "schema": "wrf_gpu2.v025.m0.manager_post_stage.v1",
                "status": "BLOCKED",
                "stage": args.manager_post_stage,
                "jax_imported": "jax" in sys.modules,
                "gpuwrf_imported": "gpuwrf" in sys.modules,
                "device_touched": False,
                "reason": str(exc),
            }
            print(json.dumps(refusal, indent=2, sort_keys=True))
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    if args.manager_window_graph:
        refusal = {
            "schema": "wrf_gpu2.v025.m0.legacy_window_refusal.v1",
            "status": "REFUSED_PRE_DEVICE_IMPORT",
            "window": args.manager_window_graph,
            "receipt_read": False,
            "receipt_consumed": False,
            "device_touched": False,
            "jax_imported": "jax" in sys.modules,
            "gpuwrf_imported": "gpuwrf" in sys.modules,
            "reason": (
                "Amendment 6 superseded independent window graphs; invoke "
                "--manager-core-session-owner exactly once"
            ),
        }
        print(json.dumps(refusal, indent=2, sort_keys=True))
        return 2

    refusal = {
        "schema": "wrf_gpu2.v025.m0.legacy_window_refusal.v1",
        "status": "REFUSED_PRE_DEVICE_IMPORT",
        "window": args.manager_device_stage,
        "receipt_read": False,
        "receipt_consumed": False,
        "jax_imported": "jax" in sys.modules,
        "gpuwrf_imported": "gpuwrf" in sys.modules,
        "device_touched": False,
        "reason": (
            "Amendment 6 superseded independent device stages; invoke "
            "--manager-core-session-owner exactly once"
        ),
    }
    print(json.dumps(refusal, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
