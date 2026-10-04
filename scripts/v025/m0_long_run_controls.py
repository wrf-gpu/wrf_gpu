#!/usr/bin/env python3
"""JAX-free eligibility controls for expensive M0 CPU/device subprocesses.

The manager amendment requires every run estimated at ten minutes or longer
to fail before launch when publication, host-memory, or suspend-safety
preconditions are not met.  This module intentionally imports no accelerator
or model package and performs no device discovery.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence


LONG_RUN_THRESHOLD_SECONDS = 600.0
EXPECTED_EXACT_PROCESS_TREE_PEAK_BYTES = 21_639_462_912
MEMORY_HEADROOM_RATIO = 2.0
FOREIGN_PROCESS_PHYSICAL_RAM_RATIO = 0.25
CLOCK_OFFSET_TOLERANCE_NS = 100_000_000
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
MEMINFO_PATH = Path("/proc/meminfo")
SYSTEMD_INHIBIT = "/usr/bin/systemd-inhibit"
INHIBIT_WHAT = "sleep:idle:handle-lid-switch"


class LongRunRefusal(RuntimeError):
    """One pre-registered long-run eligibility invariant failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def boot_clock_sample() -> dict[str, Any]:
    """Sample boot identity and suspend-sensitive clock offset."""

    if not hasattr(time, "CLOCK_BOOTTIME"):
        raise LongRunRefusal("CLOCK_BOOTTIME is unavailable")
    if BOOT_ID_PATH.is_symlink() or not BOOT_ID_PATH.is_file():
        raise LongRunRefusal(f"boot ID is unavailable: {BOOT_ID_PATH}")
    monotonic_before_ns = time.monotonic_ns()
    boottime_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    monotonic_after_ns = time.monotonic_ns()
    monotonic_midpoint_ns = (
        monotonic_before_ns + monotonic_after_ns
    ) // 2
    boot_id = BOOT_ID_PATH.read_text(encoding="utf-8").strip()
    if not boot_id:
        raise LongRunRefusal("boot ID is empty")
    return {
        "boot_id": boot_id,
        "boottime_ns": boottime_ns,
        "monotonic_before_ns": monotonic_before_ns,
        "monotonic_after_ns": monotonic_after_ns,
        "monotonic_midpoint_ns": monotonic_midpoint_ns,
        "boottime_minus_monotonic_ns":
            boottime_ns - monotonic_midpoint_ns,
        "sampling_uncertainty_ns":
            monotonic_after_ns - monotonic_before_ns,
    }


def validate_clock_continuity(
    start: dict[str, Any],
    end: dict[str, Any],
    *,
    tolerance_ns: int = CLOCK_OFFSET_TOLERANCE_NS,
) -> dict[str, Any]:
    """Reject a reboot or suspend discontinuity during one long run."""

    if (
        not isinstance(tolerance_ns, int)
        or isinstance(tolerance_ns, bool)
        or tolerance_ns < 0
    ):
        raise LongRunRefusal("clock continuity tolerance is invalid")
    if start.get("boot_id") != end.get("boot_id"):
        raise LongRunRefusal("boot ID changed during the long run")
    start_midpoint = start.get("monotonic_midpoint_ns")
    end_midpoint = end.get("monotonic_midpoint_ns")
    if (
        not isinstance(start_midpoint, int)
        or isinstance(start_midpoint, bool)
        or not isinstance(end_midpoint, int)
        or isinstance(end_midpoint, bool)
        or end_midpoint <= start_midpoint
    ):
        raise LongRunRefusal("long-run monotonic endpoints are invalid")
    start_offset = start.get("boottime_minus_monotonic_ns")
    end_offset = end.get("boottime_minus_monotonic_ns")
    if (
        not isinstance(start_offset, int)
        or isinstance(start_offset, bool)
        or not isinstance(end_offset, int)
        or isinstance(end_offset, bool)
    ):
        raise LongRunRefusal("long-run boot/monotonic offsets are invalid")
    offset_change_ns = end_offset - start_offset
    if abs(offset_change_ns) > tolerance_ns:
        raise LongRunRefusal(
            "CLOCK_BOOTTIME-CLOCK_MONOTONIC changed by "
            f"{offset_change_ns} ns > {tolerance_ns} ns"
        )
    return {
        "status": "PASS",
        "boot_id": start["boot_id"],
        "start": start,
        "end": end,
        "elapsed_monotonic_ns": end_midpoint - start_midpoint,
        "offset_change_ns": offset_change_ns,
        "tolerance_ns": tolerance_ns,
        "suspend_or_reboot_detected": False,
    }


def _meminfo_bytes(path: Path = MEMINFO_PATH) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LongRunRefusal(f"cannot read host memory information: {exc}") from exc
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {"MemTotal:", "MemAvailable:"}:
            try:
                values[fields[0][:-1]] = int(fields[1]) * 1024
            except ValueError as exc:
                raise LongRunRefusal(
                    f"malformed host memory value: {line!r}"
                ) from exc
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise LongRunRefusal("MemTotal/MemAvailable are missing from /proc/meminfo")
    return values


def _lineage_pids(pid: int | None = None) -> set[int]:
    current = int(os.getpid() if pid is None else pid)
    lineage: set[int] = set()
    while current > 0 and current not in lineage:
        lineage.add(current)
        status = Path(f"/proc/{current}/status")
        try:
            lines = status.read_text(encoding="utf-8").splitlines()
        except OSError:
            break
        parent = 0
        for line in lines:
            if line.startswith("PPid:"):
                try:
                    parent = int(line.split()[1])
                except (IndexError, ValueError):
                    parent = 0
                break
        current = parent
    return lineage


def _process_record(path: Path) -> dict[str, Any] | None:
    try:
        pid = int(path.name)
        lines = (path / "status").read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    values: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = value.strip()
    rss_fields = values.get("VmRSS", "").split()
    if not rss_fields:
        return None
    try:
        rss_bytes = int(rss_fields[0]) * 1024
    except ValueError:
        return None
    command = ""
    try:
        command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    except OSError:
        pass
    return {
        "pid": pid,
        "ppid": int(values.get("PPid", "0") or 0),
        "name": values.get("Name", ""),
        "rss_bytes": rss_bytes,
        "command": command[:512],
    }


def memory_preflight(
    *,
    expected_peak_bytes: int = EXPECTED_EXACT_PROCESS_TREE_PEAK_BYTES,
    meminfo_path: Path = MEMINFO_PATH,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Require two measured peaks free and reject one huge foreign process."""

    if (
        not isinstance(expected_peak_bytes, int)
        or isinstance(expected_peak_bytes, bool)
        or expected_peak_bytes <= 0
    ):
        raise LongRunRefusal("expected process-tree peak is invalid")
    memory = _meminfo_bytes(meminfo_path)
    required_available_bytes = int(
        expected_peak_bytes * MEMORY_HEADROOM_RATIO
    )
    if memory["MemAvailable"] < required_available_bytes:
        raise LongRunRefusal(
            "MemAvailable is below 2x the measured process-tree peak: "
            f"{memory['MemAvailable']} < {required_available_bytes}"
        )

    foreign_limit_bytes = int(
        memory["MemTotal"] * FOREIGN_PROCESS_PHYSICAL_RAM_RATIO
    )
    lineage = _lineage_pids()
    records = []
    for path in Path(proc_root).iterdir():
        if not path.name.isdigit():
            continue
        record = _process_record(path)
        if record is not None and record["pid"] not in lineage:
            records.append(record)
    records.sort(key=lambda record: int(record["rss_bytes"]), reverse=True)
    offenders = [
        record
        for record in records
        if int(record["rss_bytes"]) >= foreign_limit_bytes
    ]
    if offenders:
        first = offenders[0]
        raise LongRunRefusal(
            "foreign process RSS is at least 25% of physical RAM: "
            f"pid={first['pid']} rss={first['rss_bytes']} "
            f"limit={foreign_limit_bytes}"
        )
    return {
        "status": "PASS",
        "mem_total_bytes": memory["MemTotal"],
        "mem_available_bytes": memory["MemAvailable"],
        "expected_process_tree_peak_bytes": expected_peak_bytes,
        "required_available_bytes": required_available_bytes,
        "headroom_ratio": MEMORY_HEADROOM_RATIO,
        "foreign_process_limit_bytes": foreign_limit_bytes,
        "foreign_process_physical_ram_ratio":
            FOREIGN_PROCESS_PHYSICAL_RAM_RATIO,
        "foreign_process_offenders": [],
        "largest_foreign_processes": records[:8],
        "lineage_pids_excluded": sorted(lineage),
    }


def output_directory_preflight(target: Path) -> dict[str, Any]:
    """Require an existing parent and an entirely absent publication target."""

    target = Path(target)
    parent = target.resolve(strict=False).parent
    if target.is_symlink() or os.path.lexists(target):
        raise LongRunRefusal(
            f"publication target must be absent before long run: {target}"
        )
    if parent.is_symlink() or not parent.is_dir():
        raise LongRunRefusal(
            f"publication target parent is missing/not a directory: {parent}"
        )
    return {
        "status": "PASS",
        "target": str(target.resolve(strict=False)),
        "target_absent": True,
        "parent": str(parent),
        "parent_is_directory": True,
        "checked_before_long_run": True,
    }


def inhibited_command(
    command: Sequence[str],
    *,
    estimated_seconds: float,
    who: str = "wrf_gpu2-v025-long-run",
    why: str = "protect a hash-bound M0 long-running evidence process",
) -> tuple[list[str], dict[str, Any]]:
    """Wrap a long command with the mandatory system sleep/idle inhibitor."""

    raw = [str(item) for item in command]
    if not raw:
        raise LongRunRefusal("long-run command is empty")
    if (
        isinstance(estimated_seconds, bool)
        or not isinstance(estimated_seconds, (int, float))
        or float(estimated_seconds) <= 0
    ):
        raise LongRunRefusal("long-run duration estimate is invalid")
    is_long = float(estimated_seconds) >= LONG_RUN_THRESHOLD_SECONDS
    if not is_long:
        return raw, {
            "status": "NOT_REQUIRED",
            "estimated_seconds": float(estimated_seconds),
            "threshold_seconds": LONG_RUN_THRESHOLD_SECONDS,
            "command": raw,
        }
    resolved = shutil.which("systemd-inhibit")
    if resolved is None or str(Path(resolved).resolve()) != SYSTEMD_INHIBIT:
        raise LongRunRefusal(
            f"mandatory systemd-inhibit is unavailable at {SYSTEMD_INHIBIT}"
        )
    wrapped = [
        SYSTEMD_INHIBIT,
        f"--what={INHIBIT_WHAT}",
        "--mode=block",
        f"--who={who}",
        f"--why={why}",
        "--",
        *raw,
    ]
    return wrapped, {
        "status": "PASS",
        "estimated_seconds": float(estimated_seconds),
        "threshold_seconds": LONG_RUN_THRESHOLD_SECONDS,
        "systemd_inhibit_path": SYSTEMD_INHIBIT,
        "systemd_inhibit_sha256": _sha256_file(Path(SYSTEMD_INHIBIT)),
        "what": INHIBIT_WHAT,
        "mode": "block",
        "who": who,
        "why": why,
        "raw_command": raw,
        "wrapped_command": wrapped,
    }


__all__ = [
    "CLOCK_OFFSET_TOLERANCE_NS",
    "EXPECTED_EXACT_PROCESS_TREE_PEAK_BYTES",
    "FOREIGN_PROCESS_PHYSICAL_RAM_RATIO",
    "INHIBIT_WHAT",
    "LONG_RUN_THRESHOLD_SECONDS",
    "LongRunRefusal",
    "MEMORY_HEADROOM_RATIO",
    "SYSTEMD_INHIBIT",
    "boot_clock_sample",
    "inhibited_command",
    "memory_preflight",
    "output_directory_preflight",
    "validate_clock_continuity",
]
