#!/usr/bin/env python3
"""Read-only, cross-user kernel-flock evidence for the M0 authority boundary.

The interactive holder sidecar is not authoritative: another user or a
Production broker can hold the same inode without writing it.  This module
therefore derives the lock's exact device/inode identity with ``lstat(2)`` and
matches it against one complete ``/proc/locks`` snapshot.  It never acquires a
lock and every result explicitly grants no GPU permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


CANONICAL_LOCK = Path("/tmp/wrf_gpu2_gpu.lock")
CANONICAL_HOLDER = Path("/tmp/wrf_gpu2_gpu.lock.holder")
PROC_LOCKS = Path("/proc/locks")
SCHEMA = "wrf_gpu2.v025.m0.kernel_lock_snapshot.v1"

_LOCK_LINE = re.compile(
    r"^\s*(?P<record_id>\d+):\s+"
    r"(?:(?P<waiter>->)\s+)?"
    r"(?P<kind>\S+)\s+(?P<scope>\S+)\s+(?P<mode>\S+)\s+"
    r"(?P<pid>-?\d+)\s+"
    r"(?P<major>[0-9a-fA-F]+):(?P<minor>[0-9a-fA-F]+):(?P<inode>\d+)\s+"
    r"(?P<start>\S+)\s+(?P<end>\S+)\s*$"
)
_UNIT = re.compile(r"[^/]+\.(?:service|scope|slice)")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_lock(path: Path) -> dict[str, Any]:
    """Return the immutable identity fields used by ``/proc/locks``."""

    observed = os.lstat(path)
    major = os.major(observed.st_dev)
    minor = os.minor(observed.st_dev)
    return {
        "path": str(path),
        "mode": observed.st_mode,
        "is_regular": stat.S_ISREG(observed.st_mode),
        "is_symlink": stat.S_ISLNK(observed.st_mode),
        "device_decimal": observed.st_dev,
        "device_major": major,
        "device_minor": minor,
        "inode": observed.st_ino,
        "proc_locks_key": f"{major:02x}:{minor:02x}:{observed.st_ino}",
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
    }


def _same_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in (
            "path",
            "mode",
            "device_decimal",
            "device_major",
            "device_minor",
            "inode",
        )
    )


def parse_proc_lock_line(line: str) -> dict[str, Any] | None:
    """Parse one current Linux ``/proc/locks`` record."""

    match = _LOCK_LINE.fullmatch(line)
    if match is None:
        return None
    groups = match.groupdict()
    return {
        "raw": line,
        "record_id": int(groups["record_id"]),
        "waiter": groups["waiter"] == "->",
        "kind": groups["kind"],
        "scope": groups["scope"],
        "mode": groups["mode"],
        "pid": int(groups["pid"]),
        "device_major": int(groups["major"], 16),
        "device_minor": int(groups["minor"], 16),
        "inode": int(groups["inode"]),
        "start": groups["start"],
        "end": groups["end"],
    }


def _matches_identity(record: dict[str, Any], identity: dict[str, Any]) -> bool:
    return all(
        record[field] == identity[field]
        for field in ("device_major", "device_minor", "inode")
    )


def _proc_text(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except (OSError, UnicodeError) as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _read_small(path: Path, maximum: int = 16_384) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {
            "status": "MISSING",
            "error": f"{type(exc).__name__}: {exc}",
            "text": None,
        }
    if len(raw) > maximum:
        return {
            "status": "MISSING",
            "error": f"refused {len(raw)} bytes above {maximum}-byte diagnostic ceiling",
            "text": None,
        }
    return {
        "status": "PRESENT",
        "error": None,
        "text": raw.decode("utf-8", errors="replace"),
    }


def holder_provenance(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Collect best-effort process/cgroup identity; never weaken occupancy."""

    if pid <= 0:
        return {
            "pid": pid,
            "status": "MISSING",
            "reason": "kernel/OFD record has no userspace PID",
            "comm": None,
            "uids": None,
            "cgroup_lines": [],
            "systemd_units": [],
            "resource_job_receipt": "MISSING_NOT_SUPPLIED",
        }

    root = Path(proc_root) / str(pid)
    comm = _read_small(root / "comm", 4_096)
    status_record = _read_small(root / "status")
    cgroup = _read_small(root / "cgroup")
    uids: list[int] | None = None
    if status_record["text"] is not None:
        uid_line = next(
            (
                line
                for line in status_record["text"].splitlines()
                if line.startswith("Uid:")
            ),
            None,
        )
        if uid_line is not None:
            try:
                uids = [int(value) for value in uid_line.split()[1:]]
            except ValueError:
                uids = None
    cgroup_lines = (
        cgroup["text"].splitlines() if cgroup["text"] is not None else []
    )
    units = sorted(
        {
            component
            for line in cgroup_lines
            for component in line.split("/")
            if _UNIT.fullmatch(component)
        }
    )
    return {
        "pid": pid,
        "status": (
            "BOUND"
            if comm["text"] is not None or uids is not None or cgroup_lines
            else "MISSING"
        ),
        "comm": comm["text"].strip() if comm["text"] is not None else None,
        "comm_error": comm["error"],
        "uids": uids,
        "status_error": status_record["error"],
        "cgroup_lines": cgroup_lines,
        "cgroup_error": cgroup["error"],
        "systemd_units": units,
        "resource_job_receipt": "MISSING_NOT_SUPPLIED",
    }


def _command(argv: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def live_legacy_diagnostics(
    lock_path: Path, *, holder_path: Path | None = None
) -> dict[str, Any]:
    """Retain non-authoritative legacy views for falsification/disclosure."""

    holder_path = holder_path or lock_path.with_name(f"{lock_path.name}.holder")
    holder = _read_small(holder_path, 8_192)
    fuser = _command(["fuser", "-v", str(lock_path)])
    lslocks = _command(["lslocks", "-o", "COMMAND,PID,TYPE,MODE,PATH"])
    matching_lslocks = [
        line for line in lslocks["stdout"].splitlines() if str(lock_path) in line
    ]
    return {
        "authoritative": False,
        "cannot_promote_vacancy": True,
        "holder_sidecar": {"path": str(holder_path), **holder},
        "fuser": fuser,
        "lslocks": {
            "argv": lslocks["argv"],
            "returncode": lslocks["returncode"],
            "matching_lines": matching_lslocks,
            "stderr": lslocks["stderr"],
        },
    }


def _finalize(proof: dict[str, Any]) -> dict[str, Any]:
    proof["content_address"] = canonical_sha256(proof)
    return proof


def capture_kernel_lock_status(
    lock_path: Path = CANONICAL_LOCK,
    *,
    proc_locks_path: Path = PROC_LOCKS,
    proc_root: Path = Path("/proc"),
    proc_locks_text: str | None = None,
    legacy_observations: dict[str, Any] | None = None,
    include_live_legacy: bool = False,
) -> dict[str, Any]:
    """Capture one fail-closed lock snapshot without acquiring the lock."""

    path = Path(lock_path)
    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "permission_granted": False,
        "canonical_lock_reacquire_or_probe": False,
        "gpu_query_import_context_compile_run_profile": False,
        "lock_path_requested": str(path),
        "proc_locks_source": str(proc_locks_path),
        "snapshot_injected_for_test": proc_locks_text is not None,
        "legacy_observations": legacy_observations,
        "reasons": [],
    }
    if not path.is_absolute():
        proof.update(
            verdict="BLOCKED_UNVERIFIABLE",
            lock_identity_before=None,
            lock_identity_after=None,
            matching_records=[],
            malformed_matching_lines=[],
        )
        proof["reasons"].append("lock path is not absolute")
        return _finalize(proof)

    try:
        before = _stat_lock(path)
    except OSError as exc:
        proof.update(
            verdict="BLOCKED_UNVERIFIABLE",
            lock_identity_before=None,
            lock_identity_after=None,
            matching_records=[],
            malformed_matching_lines=[],
        )
        proof["reasons"].append(f"lstat before snapshot failed: {type(exc).__name__}: {exc}")
        return _finalize(proof)

    proof["lock_identity_before"] = before
    if before["is_symlink"] or not before["is_regular"]:
        proof.update(
            verdict="BLOCKED_UNVERIFIABLE",
            lock_identity_after=before,
            matching_records=[],
            malformed_matching_lines=[],
        )
        proof["reasons"].append("lock path is symlink or not a regular file")
        return _finalize(proof)

    canonical_requested = path.resolve(strict=False) == CANONICAL_LOCK.resolve(strict=False)
    if proc_locks_text is not None and canonical_requested:
        proof.update(
            verdict="BLOCKED_UNVERIFIABLE",
            lock_identity_after=before,
            matching_records=[],
            malformed_matching_lines=[],
        )
        proof["reasons"].append("an injected snapshot cannot adjudicate the canonical lock")
        return _finalize(proof)

    if proc_locks_text is None:
        snapshot, read_error = _proc_text(Path(proc_locks_path))
    else:
        snapshot, read_error = proc_locks_text, None
    proof["proc_locks_read_error"] = read_error
    proof["proc_locks_sha256"] = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    proof["proc_locks_bytes"] = len(snapshot.encode("utf-8"))
    proof["proc_locks_line_count"] = len(snapshot.splitlines())

    try:
        after = _stat_lock(path)
    except OSError as exc:
        after = None
        proof["reasons"].append(
            f"lstat after snapshot failed: {type(exc).__name__}: {exc}"
        )
    proof["lock_identity_after"] = after

    matching: list[dict[str, Any]] = []
    malformed_matching: list[str] = []
    key = before["proc_locks_key"]
    for line in snapshot.splitlines():
        parsed = parse_proc_lock_line(line)
        if parsed is not None and _matches_identity(parsed, before):
            matching.append(parsed)
        elif parsed is None and key in line:
            malformed_matching.append(line)
    proof["matching_records"] = matching
    proof["malformed_matching_lines"] = malformed_matching
    proof["holder_provenance"] = [
        holder_provenance(record["pid"], proc_root=proc_root) for record in matching
    ]
    if include_live_legacy:
        proof["legacy_observations"] = live_legacy_diagnostics(path)
    if proof["legacy_observations"] is None:
        proof["legacy_observations"] = {
            "authoritative": False,
            "cannot_promote_vacancy": True,
            "status": "NOT_CAPTURED",
        }

    if read_error is not None:
        verdict = "BLOCKED_UNVERIFIABLE"
        proof["reasons"].append("kernel lock snapshot could not be read")
    elif after is None or not _same_identity(before, after):
        verdict = "BLOCKED_UNVERIFIABLE"
        proof["reasons"].append("lock identity changed during snapshot")
    elif malformed_matching:
        verdict = "BLOCKED_UNVERIFIABLE"
        proof["reasons"].append("a line naming the exact key was malformed")
    elif matching:
        verdict = "BLOCKED_OCCUPIED"
        proof["reasons"].append(
            f"{len(matching)} kernel lock record(s) match {key}"
        )
    else:
        verdict = "VACANT_UNAUTHORIZED"
        proof["reasons"].append(
            "zero kernel records match; ownership and receipt gates remain missing"
        )
    proof["verdict"] = verdict
    proof["matching_record_count"] = len(matching)
    proof["legacy_cannot_change_verdict"] = True
    return _finalize(proof)


def private_flock_control(root: Path, *, wrapper_path: Path) -> dict[str, Any]:
    """Prove actual device/inode detection on a fresh non-canonical flock."""

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "private.lock"
    lock.write_bytes(b"")
    if lock == CANONICAL_LOCK or root == CANONICAL_LOCK.parent:
        raise RuntimeError("private control resolved onto the canonical lock scope")
    wrapper_before = sha256_file(wrapper_path)
    child_code = (
        "import fcntl,sys; "
        "f=open(sys.argv[1],'a'); "
        "fcntl.flock(f.fileno(), fcntl.LOCK_EX); "
        "print('READY', flush=True); "
        "sys.stdin.readline()"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    occupied: dict[str, Any] | None = None
    release_error: str | None = None
    try:
        readable, _, _ = select.select(
            [child.stdout] if child.stdout is not None else [], [], [], 5
        )
        ready = child.stdout.readline().strip() if readable else ""
        if ready != "READY":
            stderr = child.stderr.read() if child.stderr is not None else ""
            raise RuntimeError(f"private lock child did not become ready: {stderr}")
        occupied = capture_kernel_lock_status(
            lock,
            legacy_observations={
                "authoritative": False,
                "cannot_promote_vacancy": True,
                "holder_sidecar": {"text": ""},
                "fuser": {"stdout": "", "stderr": "", "returncode": 1},
                "lslocks": {"matching_lines": []},
            },
        )
    finally:
        try:
            if child.stdin is not None:
                child.stdin.write("\n")
                child.stdin.flush()
                child.stdin.close()
            child.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
            release_error = f"{type(exc).__name__}: {exc}"
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
    vacant = capture_kernel_lock_status(
        lock,
        legacy_observations={
            "authoritative": False,
            "cannot_promote_vacancy": True,
            "holder_sidecar": {"text": ""},
            "fuser": {"stdout": "", "stderr": "", "returncode": 1},
            "lslocks": {"matching_lines": []},
        },
    )
    wrapper_after = sha256_file(wrapper_path)
    detected_pid = bool(
        occupied
        and any(record["pid"] == child.pid for record in occupied["matching_records"])
    )
    status = (
        "PASS"
        if (
            occupied is not None
            and occupied["verdict"] == "BLOCKED_OCCUPIED"
            and detected_pid
            and vacant["verdict"] == "VACANT_UNAUTHORIZED"
            and child.returncode == 0
            and release_error is None
            and wrapper_before == wrapper_after
        )
        else "BLOCKED"
    )
    return {
        "schema": "wrf_gpu2.v025.m0.private_kernel_flock_control.v1",
        "status": status,
        "private_lock": str(lock),
        "child_pid": child.pid,
        "child_returncode": child.returncode,
        "release_error": release_error,
        "occupied_snapshot": occupied,
        "detected_exact_child_pid": detected_pid,
        "post_release_snapshot": vacant,
        "canonical_wrapper": str(wrapper_path),
        "canonical_wrapper_sha256_before": wrapper_before,
        "canonical_wrapper_sha256_after": wrapper_after,
        "canonical_wrapper_unchanged": wrapper_before == wrapper_after,
        "canonical_lock_touched": False,
        "gpu_action": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=CANONICAL_LOCK)
    parser.add_argument("--proc-locks", type=Path, default=PROC_LOCKS)
    parser.add_argument("--legacy-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("exact device denial is required")
    if any(name.startswith("GPUWRF_GPU_LOCK_") for name in os.environ):
        raise SystemExit("inherited lock authority is forbidden")
    proof = capture_kernel_lock_status(
        args.lock,
        proc_locks_path=args.proc_locks,
        include_live_legacy=args.legacy_diagnostics,
    )
    print(json.dumps(proof, indent=2, sort_keys=True))
    return {
        "VACANT_UNAUTHORIZED": 0,
        "BLOCKED_OCCUPIED": 3,
        "BLOCKED_UNVERIFIABLE": 4,
    }[proof["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
