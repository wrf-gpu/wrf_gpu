#!/usr/bin/env python3
"""Fail-closed process-group guard for M0 forecast and Pallas children.

The lock-owning manager parent launches this guard as a new session leader and
arms ``PR_SET_PDEATHSIG`` before ``exec``.  Once started, the guard installs its
group-kill handlers before it launches the real child in the same process
group.  Parent death therefore kills the guard, the child, and every
non-daemonized grandchild even when the workload traps ``SIGTERM``.

This module is deliberately JAX/gpuwrf-free and performs no device operation.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
from functools import partial
from typing import NoReturn, Sequence


PR_SET_PDEATHSIG = 1
LOCK_OWNER_PID_ENV = "GPUWRF_M0_LOCK_OWNER_PID"


class ParentDeathGuardError(RuntimeError):
    """The guarded process boundary could not be established."""


def _die_now() -> NoReturn:
    os.kill(os.getpid(), signal.SIGKILL)
    os._exit(128 + int(signal.SIGKILL))


def arm_parent_death_signal(
    expected_parent_pid: int,
    death_signal: signal.Signals = signal.SIGTERM,
) -> None:
    """Arm Linux parent-death signaling without the fork-to-prctl race."""

    if os.getppid() != expected_parent_pid:
        _die_now()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(death_signal), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent_pid:
        _die_now()


def _kill_guard_process_group(
    _signal_number: int,
    _frame: object,
) -> NoReturn:
    """Use an untrappable signal so a SIGTERM-trapping workload cannot survive."""

    os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(128 + int(signal.SIGKILL))


def run_guard(command: Sequence[str]) -> int:
    """Launch one workload in this guard's process group and proxy its status."""

    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ParentDeathGuardError("guarded command must be a non-empty argv")
    if os.getpid() != os.getpgrp() or os.getsid(0) != os.getpid():
        raise ParentDeathGuardError(
            "parent-death guard must be launched with start_new_session=True"
        )
    try:
        expected_lock_owner_pid = int(os.environ[LOCK_OWNER_PID_ENV])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParentDeathGuardError(
            f"{LOCK_OWNER_PID_ENV} is missing or invalid"
        ) from exc
    if os.getppid() != expected_lock_owner_pid:
        _die_now()

    # Arm from this deliberately single-threaded guard, not from the manager's
    # Popen preexec_fn.  The profiled manager has live telemetry threads by the
    # time it launches us; running Python after fork from that threaded process
    # can deadlock on inherited locks.  The double-PPID check in
    # arm_parent_death_signal closes the launch-to-prctl race before any
    # workload exists.
    arm_parent_death_signal(expected_lock_owner_pid, signal.SIGTERM)
    for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signal_number, _kill_guard_process_group)
    if os.getppid() != expected_lock_owner_pid:
        _kill_guard_process_group(signal.SIGTERM, None)

    child_environment = dict(os.environ)
    child_environment.pop(LOCK_OWNER_PID_ENV, None)
    guard_pid = os.getpid()
    child = subprocess.Popen(
        list(command),
        env=child_environment,
        start_new_session=False,
        preexec_fn=partial(
            arm_parent_death_signal,
            guard_pid,
            signal.SIGKILL,
        ),
    )
    returncode = child.wait()
    return 128 + abs(returncode) if returncode < 0 else returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    return run_guard(command)


if __name__ == "__main__":
    raise SystemExit(main())
