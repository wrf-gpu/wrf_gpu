#!/usr/bin/env python3
"""External, JAX-free process-tree RSS sampler for M0 child stages."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "wrf_gpu2.v025.m0.process_tree_rss.v1"


class HostRssError(RuntimeError):
    """The process-tree RSS measurement was incomplete."""


def _accelerator_modules() -> list[str]:
    roots = ("jax", "jaxlib", "gpuwrf")
    return sorted(
        name for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in roots)
    )


def _children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        text = path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [int(item) for item in text.split() if item.isdigit()]


def _process_tree(root_pid: int) -> set[int]:
    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid <= 0 or pid in observed:
            continue
        observed.add(pid)
        pending.extend(_children(pid))
    return observed


def _rss_bytes(pid: int) -> int | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(
            encoding="ascii", errors="replace"
        ).splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    for line in lines:
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) * 1024
    return None


def _atomic_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise HostRssError(f"refusing to replace host RSS sidecar: {path}")
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


class ProcessTreeRssSampler:
    """Poll aggregate RSS without importing or attaching to the accelerator."""

    def __init__(
        self,
        *,
        root_pid: int,
        output_path: Path,
        cadence_ms: int = 50,
        run_id: str | None = None,
        stage: str | None = None,
        cache_path: Path | None = None,
    ) -> None:
        if root_pid <= 0 or cadence_ms <= 0:
            raise HostRssError("root PID and cadence must be positive")
        if _accelerator_modules():
            raise HostRssError("host RSS sampler process imported accelerator roots")
        self.root_pid = int(root_pid)
        self.output_path = Path(output_path)
        self.cadence_ms = int(cadence_ms)
        self.run_id = run_id
        self.stage = stage
        self.cache_path = (
            str(Path(cache_path).resolve()) if cache_path is not None else None
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples = 0
        self._misses = 0
        self._peak = 0
        self._peak_pids: list[int] = []
        self._observed_pids: set[int] = set()
        self._thread_cpu_ns = 0
        self._started_ns: int | None = None
        self._started_utc: str | None = None

    def _sample_once(self) -> None:
        pids = _process_tree(self.root_pid)
        total = 0
        readable: list[int] = []
        for pid in sorted(pids):
            rss = _rss_bytes(pid)
            if rss is None:
                continue
            total += rss
            readable.append(pid)
        with self._lock:
            if not readable:
                self._misses += 1
                return
            self._samples += 1
            self._observed_pids.update(readable)
            if total > self._peak:
                self._peak = total
                self._peak_pids = readable

    def _run(self) -> None:
        cpu_start = time.thread_time_ns()
        try:
            while not self._stop.wait(self.cadence_ms / 1000.0):
                self._sample_once()
        finally:
            self._thread_cpu_ns = time.thread_time_ns() - cpu_start

    def start(self) -> None:
        if self._thread is not None:
            raise HostRssError("host RSS sampler was started twice")
        self._started_ns = time.monotonic_ns()
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run, name="m0-host-rss", daemon=True
        )
        self._thread.start()

    def abort(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def finish(self) -> dict[str, Any]:
        if self._started_ns is None or self._thread is None:
            raise HostRssError("host RSS sampler never started")
        self._sample_once()
        self.abort()
        finished_ns = time.monotonic_ns()
        finished_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            samples = self._samples
            misses = self._misses
            peak = self._peak
            peak_pids = list(self._peak_pids)
            observed = sorted(self._observed_pids)
        if samples <= 0 or self.root_pid not in observed:
            raise HostRssError("host RSS sampler never observed the root process")
        payload = {
            "schema": SCHEMA,
            "status": "PASS",
            "sampler_process_role": "JAX-free-manager-parent",
            "jax_imported_by_sampler": False,
            "root_pid": self.root_pid,
            "run_id": self.run_id,
            "stage": self.stage,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cache_path": self.cache_path,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "JAX_PLATFORMS",
                    "CUDA_VISIBLE_DEVICES",
                    "XLA_FLAGS",
                    "GPUWRF_JAX_CACHE",
                    "GPUWRF_JAX_CACHE_DIR",
                    "GPUWRF_JAX_CACHE_LOCK",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "sample_cadence_ms": self.cadence_ms,
            "samples": samples,
            "sampling_misses": misses,
            "peak_process_tree_rss_bytes": peak,
            "peak_process_tree_pids": peak_pids,
            "observed_process_tree_pids": observed,
            "measurement_start_monotonic_ns": self._started_ns,
            "measurement_end_monotonic_ns": finished_ns,
            "measurement_start_utc": self._started_utc,
            "measurement_end_utc": finished_utc,
            "sampler_thread_cpu_seconds": self._thread_cpu_ns / 1e9,
            "measurement_scope": (
                "whole authorized child lifecycle; aggregate simultaneous VmRSS "
                "over the live descendant tree"
            ),
            "sampling_limitations": [
                "Polling can miss RSS spikes shorter than sample_cadence_ms.",
                "A descendant reparented after the root exits can disappear before the final sample.",
            ],
            "sampler_ram_scaling": "O(unique process IDs); no time series retained",
            "device_action": False,
        }
        _atomic_json_no_replace(self.output_path, payload)
        return payload


def main() -> int:
    """Launch one command and externally sample its complete live process tree."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--cadence-ms", type=int, default=50)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command after -- is required")
    if _accelerator_modules():
        raise HostRssError("sampler launcher imported accelerator roots")

    process = subprocess.Popen(command)
    sampler = ProcessTreeRssSampler(
        root_pid=process.pid,
        output_path=args.output,
        cadence_ms=args.cadence_ms,
        run_id=args.run_id,
        stage=args.stage,
        cache_path=args.cache_path,
    )
    sampler.start()
    try:
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        sampler.abort()
        raise
    payload = sampler.finish()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "command": command,
                "command_returncode": returncode,
                "output": str(args.output),
                "peak_process_tree_rss_bytes": payload[
                    "peak_process_tree_rss_bytes"
                ],
                "device_action": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return returncode


__all__ = ["HostRssError", "ProcessTreeRssSampler", "SCHEMA"]


if __name__ == "__main__":
    raise SystemExit(main())
