#!/usr/bin/env python3
"""External M0 residency sampler; this module must never import JAX.

The forecast process owns JAX allocator counters.  The lock-owning capture
parent owns this sampler and measures the product metric: baseline-subtracted
total residency on one pre-bound physical GPU.  Production sampling uses two
long-lived ``nvidia-smi --loop-ms`` streams, avoiding a new process and NVML
initialisation on every sample.  Tests inject a backend and never touch a
device.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


IDENTITY_SCHEMA = "wrf_gpu2.v025.m0.evidence_identity.v1"
ALLOCATOR_SCHEMA = "wrf_gpu2.v025.m0.forecast_allocator.v1"
EXACT_ALLOCATOR_SCHEMA = "wrf_gpu2.v025.m0.exact_allocator.v1"
SAMPLER_SCHEMA = "wrf_gpu2.v025.m0.lock_owner_total_residency.v1"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEVICE_UUID_PATTERN = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
DEFAULT_CADENCE_MS = 100
MAX_SAMPLING_MISS_FRACTION = 0.05
SAMPLING_LIMITATIONS = (
    "Polling can miss a residency spike shorter than sample_cadence_ms.",
    "The metric is invalid if inferred or malformed misses exceed 5% of "
    "stream samples plus misses.",
    "Pre-launch competing contexts are rejected, but a context that appears and "
    "disappears wholly between process-telemetry samples cannot be observed.",
)


def _loaded_jax_modules() -> tuple[str, ...]:
    """Return already-loaded JAX modules without importing or querying JAX."""

    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == "jax"
            or name.startswith("jax.")
            or name == "jaxlib"
            or name.startswith("jaxlib.")
        )
    )


def _sampling_miss_fraction(samples: int, misses: int) -> float:
    """Compute misses over the complete expected-observation denominator."""

    if samples <= 0 or misses < 0:
        raise ResidencyEvidenceError(
            "sampling coverage requires positive samples and non-negative misses"
        )
    return misses / (samples + misses)


def _sampling_coverage_is_acceptable(samples: int, misses: int) -> bool:
    """Mechanical 95% coverage gate; equality at the 5% limit is accepted."""

    return _sampling_miss_fraction(samples, misses) <= MAX_SAMPLING_MISS_FRACTION


def _monotonic_interval_encloses(
    outer_start_ns: int,
    inner_start_ns: int,
    inner_end_ns: int,
    outer_end_ns: int,
) -> bool:
    """True only for one strictly ordered inner interval inside its sampler."""

    return outer_start_ns <= inner_start_ns < inner_end_ns <= outer_end_ns


def _terminate_if_parent_dies() -> None:
    """Linux child hook: prevent a telemetry loop surviving lock-owner death."""

    import ctypes

    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


class ResidencyEvidenceError(RuntimeError):
    """The sampler could not produce identity-safe total-residency evidence."""


@dataclass(frozen=True)
class EvidenceIdentity:
    run_id: str
    source_sha256: str
    config_sha256: str
    input_manifest_sha256: str
    device_uuid: str
    workload_identity_sha256: str
    integration_scope_sha256: str
    event_mix_sha256: str

    def binding(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "input_manifest_sha256": self.input_manifest_sha256,
            "device_uuid": self.device_uuid,
            "workload_identity_sha256": self.workload_identity_sha256,
            "integration_scope_sha256": self.integration_scope_sha256,
            "event_mix_sha256": self.event_mix_sha256,
        }


def load_identity(path: Path, *, expected_run_id: str | None = None) -> EvidenceIdentity:
    """Load a pre-launch identity object and reject aliases or partial hashes."""

    if not path.is_file():
        raise ResidencyEvidenceError(f"evidence identity file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidencyEvidenceError(f"invalid evidence identity JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResidencyEvidenceError("evidence identity JSON must be an object")
    if payload.get("schema") != IDENTITY_SCHEMA:
        raise ResidencyEvidenceError(
            f"evidence identity schema must be {IDENTITY_SCHEMA}"
        )

    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ResidencyEvidenceError("evidence identity has an invalid run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ResidencyEvidenceError(
            f"evidence identity run_id {run_id!r} does not match {expected_run_id!r}"
        )

    digest_fields = (
        "source_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "workload_identity_sha256",
        "integration_scope_sha256",
        "event_mix_sha256",
    )
    expected_keys = {"schema", "run_id", "device_uuid", *digest_fields}
    if set(payload) != expected_keys:
        raise ResidencyEvidenceError(
            "evidence identity fields are incomplete or unrecognised"
        )
    values: dict[str, str] = {}
    for field in digest_fields:
        value = payload.get(field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise ResidencyEvidenceError(
                f"evidence identity {field} must be a lowercase full SHA-256"
            )
        values[field] = value

    device_uuid = payload.get("device_uuid")
    if not isinstance(device_uuid, str) or not DEVICE_UUID_PATTERN.fullmatch(device_uuid):
        raise ResidencyEvidenceError(
            "evidence identity device_uuid must be a full GPU- UUID"
        )
    return EvidenceIdentity(
        run_id=run_id,
        device_uuid=device_uuid,
        **values,
    )


def _stable_file_identity(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    observed_bytes = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            observed_bytes += len(block)
            digest.update(block)
    after = path.stat()
    if (
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        or observed_bytes != before.st_size
    ):
        raise ResidencyEvidenceError(f"file changed while it was hashed: {path}")
    return observed_bytes, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _stable_file_identity(path)[1]


def _canonical_sha256(value: Any) -> str:
    material = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def source_tree_sha256(source_root: Path) -> str:
    """Content-address the executable source tree, excluding runtime bytecode."""

    if not source_root.is_dir():
        raise ResidencyEvidenceError(f"source tree is missing: {source_root}")
    entries = sorted(
        path
        for path in source_root.rglob("*")
        if "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        raise ResidencyEvidenceError(
            f"source tree identity may not traverse symlinks: {symlinks[0]}"
        )
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ResidencyEvidenceError(f"source tree is empty: {source_root}")
    manifest = []
    for path in files:
        size, digest = _stable_file_identity(path)
        manifest.append({
            "path": path.relative_to(source_root).as_posix(),
            "bytes": size,
            "sha256": digest,
        })
    final_files = sorted(
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    )
    if final_files != [record["path"] for record in manifest]:
        raise ResidencyEvidenceError(
            f"source tree file set changed while it was hashed: {source_root}"
        )
    return _canonical_sha256(manifest)


def directory_tree_identity(root: Path) -> dict[str, Any]:
    """Hash every regular file in a cache snapshot without retaining its bytes."""

    if not root.is_dir():
        raise ResidencyEvidenceError(f"directory snapshot is missing: {root}")
    entries = sorted(root.rglob("*"))
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        raise ResidencyEvidenceError(
            f"directory snapshot may not contain symlinks: {symlinks[0]}"
        )
    directories = [path for path in entries if path.is_dir()]
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ResidencyEvidenceError(f"directory snapshot is empty: {root}")
    manifest: list[dict[str, Any]] = [
        {
            "path": path.relative_to(root).as_posix(),
            "kind": "directory",
        }
        for path in directories
    ]
    total_bytes = 0
    for path in files:
        before = path.stat()
        digest = hashlib.sha256()
        observed_bytes = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                observed_bytes += len(block)
                digest.update(block)
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or observed_bytes != before.st_size:
            raise ResidencyEvidenceError(
                f"directory snapshot changed while it was hashed: {path}"
            )
        total_bytes += observed_bytes
        manifest.append({
            "path": path.relative_to(root).as_posix(),
            "kind": "file",
            "bytes": observed_bytes,
            "sha256": digest.hexdigest(),
        })
    manifest.sort(key=lambda record: (record["path"], record["kind"]))
    final_layout = sorted(
        (
            path.relative_to(root).as_posix(),
            "directory" if path.is_dir() else "file",
        )
        for path in root.rglob("*")
        if (path.is_file() or path.is_dir()) and not path.is_symlink()
    )
    if final_layout != [
        (record["path"], record["kind"]) for record in manifest
    ]:
        raise ResidencyEvidenceError(
            f"directory snapshot layout changed while it was hashed: {root}"
        )
    return {
        "schema": "wrf_gpu2.v025.m0.directory_tree_identity.v1",
        "files": len(files),
        "directories": len(directories),
        "bytes": total_bytes,
        "sha256": _canonical_sha256(manifest),
    }


def input_manifest_sha256(run_dir: Path) -> str:
    """Hash exactly the FAST replay inputs consumed by the production arm."""

    namelist = run_dir / "namelist.input"
    histories = sorted(run_dir.glob("wrfout_d01_*"))
    files = [namelist, *histories]
    if not namelist.is_file() or len(histories) != 2:
        raise ResidencyEvidenceError(
            "FAST replay identity requires namelist.input and exactly two wrfout_d01 histories"
        )
    manifest = []
    for path in files:
        size, digest = _stable_file_identity(path)
        manifest.append({
            "path": path.name,
            "bytes": size,
            "sha256": digest,
        })
    if sorted(run_dir.glob("wrfout_d01_*")) != histories:
        raise ResidencyEvidenceError(
            "FAST replay input set changed while it was hashed"
        )
    return _canonical_sha256(manifest)


def derive_identity(
    *,
    run_id: str,
    device_uuid: str,
    source_root: Path,
    run_dir: Path,
    hours: int = 1,
) -> EvidenceIdentity:
    """Derive every non-device identity from the exact pre-launch workload."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ResidencyEvidenceError("derived identity run_id is invalid")
    if not DEVICE_UUID_PATTERN.fullmatch(device_uuid):
        raise ResidencyEvidenceError("derived identity device_uuid is invalid")
    if hours != 1:
        raise ResidencyEvidenceError("M0 evidence identity is frozen to one forecast hour")
    config_path = run_dir / "namelist.input"
    if not config_path.is_file():
        raise ResidencyEvidenceError(f"FAST namelist is missing: {config_path}")
    source_hash = source_tree_sha256(source_root)
    config_hash = _sha256_file(config_path)
    input_hash = input_manifest_sha256(run_dir)
    integration_scope_hash = _canonical_sha256({
        "entrypoint": "gpuwrf.runtime.operational_mode.run_forecast_operational",
        "hours": hours,
        "range_name": "GPUWRF_M0_FORECAST_INTEGRATION",
        "scope": "after-input-staging-through-final-integration-synchronization",
    })
    event_mix_hash = _canonical_sha256({
        "config_sha256": config_hash,
        "input_manifest_sha256": input_hash,
        "cadence_source": "production namelist and runtime cadence dispatch",
        "hours": hours,
    })
    workload_hash = _canonical_sha256({
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "input_manifest_sha256": input_hash,
        "integration_scope_sha256": integration_scope_hash,
        "event_mix_sha256": event_mix_hash,
        "hours": hours,
    })
    return EvidenceIdentity(
        run_id=run_id,
        source_sha256=source_hash,
        config_sha256=config_hash,
        input_manifest_sha256=input_hash,
        device_uuid=device_uuid,
        workload_identity_sha256=workload_hash,
        integration_scope_sha256=integration_scope_hash,
        event_mix_sha256=event_mix_hash,
    )


def validate_identity_against_workload(
    identity: EvidenceIdentity,
    *,
    source_root: Path,
    run_dir: Path,
    hours: int = 1,
) -> None:
    """Re-derive before receipt spend so a stale claimed identity cannot launch."""

    observed = derive_identity(
        run_id=identity.run_id,
        device_uuid=identity.device_uuid,
        source_root=source_root,
        run_dir=run_dir,
        hours=hours,
    )
    mismatches = [
        key
        for key, expected in identity.binding().items()
        if observed.binding()[key] != expected
    ]
    if mismatches:
        raise ResidencyEvidenceError(
            f"evidence identity does not match pre-launch workload: {mismatches}"
        )


def hook_environment(
    identity: EvidenceIdentity,
    *,
    allocator_sidecar: Path,
) -> dict[str, str]:
    """Build the private opt-in environment consumed by the forecast process."""

    if not allocator_sidecar.is_absolute():
        raise ResidencyEvidenceError("allocator sidecar path must be absolute")
    if not allocator_sidecar.parent.is_dir():
        raise ResidencyEvidenceError("allocator sidecar parent must already exist")
    if os.path.lexists(allocator_sidecar):
        raise ResidencyEvidenceError(
            f"allocator sidecar target already exists: {allocator_sidecar}"
        )
    return {
        "GPUWRF_M0_EVIDENCE": "1",
        "GPUWRF_M0_EVIDENCE_PATH": str(allocator_sidecar),
        "GPUWRF_M0_RUN_ID": identity.run_id,
        "GPUWRF_M0_SOURCE_SHA256": identity.source_sha256,
        "GPUWRF_M0_CONFIG_SHA256": identity.config_sha256,
        "GPUWRF_M0_INPUT_MANIFEST_SHA256": identity.input_manifest_sha256,
        "GPUWRF_M0_DEVICE_UUID": identity.device_uuid,
    }


def _atomic_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise ResidencyEvidenceError(f"refusing to replace evidence: {path}")
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


def _parse_mib(value: str) -> int:
    try:
        mib = float(value.strip())
    except ValueError as exc:
        raise ResidencyEvidenceError(f"invalid nvidia-smi MiB value: {value!r}") from exc
    if not math.isfinite(mib) or mib < 0:
        raise ResidencyEvidenceError(f"invalid nvidia-smi MiB value: {value!r}")
    return int(round(mib * (1 << 20)))


def _parse_global_row(line: str, expected_uuid: str) -> int:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 2 or fields[0] != expected_uuid:
        raise ResidencyEvidenceError(f"unexpected total-residency row: {line!r}")
    return _parse_mib(fields[1])


def _parse_compute_row(line: str, expected_uuid: str) -> tuple[int, int]:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 3 or fields[1] != expected_uuid:
        raise ResidencyEvidenceError(f"unexpected compute-process row: {line!r}")
    try:
        pid = int(fields[0])
    except ValueError as exc:
        raise ResidencyEvidenceError(f"invalid compute-process PID: {line!r}") from exc
    if pid <= 0:
        raise ResidencyEvidenceError(f"invalid compute-process PID: {line!r}")
    return pid, _parse_mib(fields[2])


class SamplingBackend(Protocol):
    """Backend boundary used by production streams and CPU-only fake tests."""

    commands: dict[str, list[str]]

    def baseline(self) -> tuple[int, list[tuple[int, int]]]:
        ...

    def start(
        self,
        total_callback: Callable[[int, int], None],
        process_callback: Callable[[int, int, int], None],
    ) -> None:
        ...

    def stop(self) -> None:
        ...


class NvidiaSmiStreamingBackend:
    """One-shot baseline plus persistent total/process telemetry streams."""

    def __init__(
        self,
        device_uuid: str,
        *,
        cadence_ms: int,
        stderr_path: Path,
        executable: str = "nvidia-smi",
    ) -> None:
        if cadence_ms <= 0:
            raise ResidencyEvidenceError("sample cadence must be positive")
        self.device_uuid = device_uuid
        self.cadence_ms = int(cadence_ms)
        self.stderr_path = stderr_path
        selector = f"--id={device_uuid}"
        interval = f"--loop-ms={self.cadence_ms}"
        self.commands = {
            "baseline_total": [
                executable,
                selector,
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            "baseline_processes": [
                executable,
                selector,
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            "stream_total": [
                executable,
                selector,
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
                interval,
            ],
            "stream_processes": [
                executable,
                selector,
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                interval,
            ],
        }
        self._processes: list[subprocess.Popen[str]] = []
        self._threads: list[threading.Thread] = []
        self._stderr_stream = None
        self._first_total = threading.Event()
        self._stopping = threading.Event()
        self._reader_errors: list[str] = []

    def _run_once(self, command: Sequence[str], *, allow_empty: bool) -> list[str]:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
            preexec_fn=_terminate_if_parent_dies,
        )
        lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and "No running processes found" not in line
        ]
        no_processes = "No running processes found" in (
            completed.stdout + completed.stderr
        )
        if completed.returncode != 0 and not (allow_empty and no_processes):
            raise ResidencyEvidenceError(
                f"external sampler command failed rc={completed.returncode}: "
                f"{completed.stderr[-500:]}"
            )
        if not lines and not allow_empty:
            raise ResidencyEvidenceError("external sampler returned no total-memory row")
        return lines

    def baseline(self) -> tuple[int, list[tuple[int, int]]]:
        total_lines = self._run_once(self.commands["baseline_total"], allow_empty=False)
        if len(total_lines) != 1:
            raise ResidencyEvidenceError(
                f"expected one selected-GPU baseline row, got {len(total_lines)}"
            )
        total = _parse_global_row(total_lines[0], self.device_uuid)
        process_lines = self._run_once(
            self.commands["baseline_processes"], allow_empty=True
        )
        processes = [
            _parse_compute_row(line, self.device_uuid) for line in process_lines
        ]
        return total, processes

    def _read_total(
        self,
        proc: subprocess.Popen[str],
        callback: Callable[[int, int], None],
    ) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if (
                    not line.strip()
                    or "No running processes found" in line
                ):
                    continue
                value = _parse_global_row(line.strip(), self.device_uuid)
                callback(time.monotonic_ns(), value)
                self._first_total.set()
        except Exception as exc:  # noqa: BLE001 - transferred to lock owner
            self._reader_errors.append(f"total stream: {exc}")
            self._first_total.set()
        finally:
            if not self._stopping.is_set():
                self._reader_errors.append("total stream ended before sampler stop")
                self._first_total.set()

    def _read_processes(
        self,
        proc: subprocess.Popen[str],
        callback: Callable[[int, int, int], None],
    ) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if (
                    not line.strip()
                    or "No running processes found" in line
                ):
                    continue
                pid, used = _parse_compute_row(line.strip(), self.device_uuid)
                callback(time.monotonic_ns(), pid, used)
        except Exception as exc:  # noqa: BLE001 - transferred to lock owner
            self._reader_errors.append(f"process stream: {exc}")
        finally:
            if not self._stopping.is_set():
                self._reader_errors.append("process stream ended before sampler stop")

    def start(
        self,
        total_callback: Callable[[int, int], None],
        process_callback: Callable[[int, int, int], None],
    ) -> None:
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_stream = self.stderr_path.open("x", encoding="utf-8")
        total = None
        try:
            total = subprocess.Popen(
                self.commands["stream_total"],
                stdout=subprocess.PIPE,
                stderr=self._stderr_stream,
                text=True,
                bufsize=1,
                preexec_fn=_terminate_if_parent_dies,
            )
            processes = subprocess.Popen(
                self.commands["stream_processes"],
                stdout=subprocess.PIPE,
                stderr=self._stderr_stream,
                text=True,
                bufsize=1,
                preexec_fn=_terminate_if_parent_dies,
            )
        except Exception:
            if total is not None and total.poll() is None:
                total.terminate()
                try:
                    total.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    total.kill()
                    total.wait(timeout=5.0)
            self._stderr_stream.close()
            raise
        self._processes = [total, processes]
        self._threads = [
            threading.Thread(
                target=self._read_total,
                args=(total, total_callback),
                daemon=True,
                name="m0-vram-total",
            ),
            threading.Thread(
                target=self._read_processes,
                args=(processes, process_callback),
                daemon=True,
                name="m0-vram-processes",
            ),
        ]
        for thread in self._threads:
            thread.start()
        if not self._first_total.wait(timeout=10.0):
            self.stop()
            raise ResidencyEvidenceError(
                "persistent total-residency stream did not produce a sample"
            )
        if self._reader_errors:
            errors = "; ".join(self._reader_errors)
            self.stop()
            raise ResidencyEvidenceError(errors)
        exited = [
            (label, proc.returncode)
            for label, proc in (
                ("total", total),
                ("process", processes),
            )
            if proc.poll() is not None
        ]
        if exited:
            self.stop()
            raise ResidencyEvidenceError(
                f"persistent residency stream exited before forecast: {exited}"
            )

    def stop(self) -> None:
        self._stopping.set()
        for proc in self._processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in self._processes:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        for thread in self._threads:
            thread.join(timeout=5.0)
        if self._stderr_stream is not None and not self._stderr_stream.closed:
            self._stderr_stream.flush()
            os.fsync(self._stderr_stream.fileno())
            self._stderr_stream.close()


def _pid_in_group_or_tree(pid: int, *, root_pid: int, process_group_id: int) -> bool:
    """Use live /proc ancestry and PGID without retaining process handles."""

    try:
        if os.getpgid(pid) == process_group_id:
            return True
    except (ProcessLookupError, PermissionError):
        pass
    current = pid
    visited: set[int] = set()
    while current > 1 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return False
        match = re.search(r"^PPid:\s+(\d+)$", status, re.MULTILINE)
        if match is None:
            return False
        current = int(match.group(1))
    return False


def validate_allocator_sidecar(
    payload: dict[str, Any],
    identity: EvidenceIdentity,
) -> dict[str, Any]:
    """Validate forecast-owned evidence before binding parent telemetry to it."""

    if not isinstance(payload, dict):
        raise ResidencyEvidenceError("allocator sidecar must be a JSON object")
    if payload.get("schema") == EXACT_ALLOCATOR_SCHEMA:
        return _validate_exact_boundary_allocator_sidecar(payload, identity)
    expected_keys = {
        "schema",
        "emitter_process_role",
        "instrumentation",
        "run_id",
        "forecast_pid",
        "source_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "device_uuid",
        "measurement_start_ns",
        "measurement_end_ns",
        "measurement_start_utc",
        "measurement_end_utc",
        "peak_bytes_in_use",
        "peak_bytes_reserved",
        "device_platform",
        "device_local_ordinal",
    }
    if set(payload) != expected_keys:
        raise ResidencyEvidenceError(
            "allocator sidecar fields are incomplete or unrecognised"
        )
    if payload.get("schema") != ALLOCATOR_SCHEMA:
        raise ResidencyEvidenceError(
            f"allocator sidecar schema must be {ALLOCATOR_SCHEMA}"
        )
    if payload.get("emitter_process_role") != "forecast_process":
        raise ResidencyEvidenceError("allocator sidecar has the wrong emitter role")
    for key, expected in identity.binding().items():
        if key in payload and payload.get(key) != expected:
            raise ResidencyEvidenceError(f"allocator sidecar identity mismatch: {key}")
    for key in (
        "run_id",
        "source_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "device_uuid",
    ):
        if payload.get(key) != identity.binding()[key]:
            raise ResidencyEvidenceError(f"allocator sidecar is missing binding: {key}")
    instrumentation = payload.get("instrumentation") or {}
    expected_instrumentation = {
        "enabled": True,
        "opt_in_environment": "GPUWRF_M0_EVIDENCE=1",
        "default_when_unset": "original-direct-call-no-range-no-sidecar",
        "range_name": "GPUWRF_M0_FORECAST_INTEGRATION",
        "range_scope": "full-forecast-integration",
        "final_synchronization": "jax.block_until_ready(result)",
        "allocator_read": "after-range-on-result-device",
    }
    if instrumentation != expected_instrumentation:
        raise ResidencyEvidenceError("allocator sidecar provenance is incomplete")
    try:
        forecast_pid = int(payload["forecast_pid"])
        start_ns = int(payload["measurement_start_ns"])
        end_ns = int(payload["measurement_end_ns"])
        peak_in_use = int(payload["peak_bytes_in_use"])
        peak_reserved = int(payload["peak_bytes_reserved"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResidencyEvidenceError("allocator sidecar numeric fields are incomplete") from exc
    if (
        forecast_pid <= 0
        or start_ns <= 0
        or end_ns <= start_ns
        or peak_in_use < 0
        or peak_reserved < peak_in_use
    ):
        raise ResidencyEvidenceError("allocator sidecar numeric fields are invalid")
    integer_fields = (
        "forecast_pid",
        "measurement_start_ns",
        "measurement_end_ns",
        "peak_bytes_in_use",
        "peak_bytes_reserved",
    )
    if any(
        isinstance(payload.get(field), bool)
        or not isinstance(payload.get(field), int)
        for field in integer_fields
    ):
        raise ResidencyEvidenceError(
            "allocator sidecar numeric fields must be exact integers"
        )
    platform = payload.get("device_platform")
    ordinal = payload.get("device_local_ordinal")
    if not isinstance(platform, str) or not platform:
        raise ResidencyEvidenceError("allocator sidecar device platform is invalid")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, (int, str, type(None)))
    ):
        raise ResidencyEvidenceError("allocator sidecar device ordinal is invalid")
    utc_values: dict[str, datetime] = {}
    for field in ("measurement_start_utc", "measurement_end_utc"):
        value = payload.get(field)
        try:
            stamp = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ResidencyEvidenceError(f"allocator sidecar {field} is invalid") from exc
        if stamp.tzinfo is None:
            raise ResidencyEvidenceError(f"allocator sidecar {field} lacks timezone")
        if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            raise ResidencyEvidenceError(f"allocator sidecar {field} is not UTC")
        utc_values[field] = stamp
    if utc_values["measurement_end_utc"] <= utc_values["measurement_start_utc"]:
        raise ResidencyEvidenceError("allocator sidecar UTC interval is invalid")
    return payload


def _validate_exact_boundary_allocator_sidecar(
    payload: dict[str, Any],
    identity: EvidenceIdentity,
) -> dict[str, Any]:
    """Validate the Amendment-4 sprint-local child's allocator sidecar.

    The exact-boundary child deliberately does not enable ADR-036's production
    hook: doing so would add a second range and a second allocator writer around
    the same invocation.  Its smaller sidecar instead binds the child PID,
    run ID, exact range, synchronized monotonic interval, result digest, and
    allocator peaks.  The JAX-free lock owner separately validates the full
    workload/device identity before launch and owns the physical-UUID sampler.
    """

    expected_keys = {
        "schema",
        "run_id",
        "pid",
        "range_name",
        "measurement_start_ns",
        "measurement_end_ns",
        "result_sha256",
        "peak_bytes_in_use",
        "peak_bytes_reserved",
        "device_platform",
        "device_local_ordinal",
    }
    if set(payload) != expected_keys:
        raise ResidencyEvidenceError(
            "exact-boundary allocator fields are incomplete or unrecognised"
        )
    if payload.get("run_id") != identity.run_id:
        raise ResidencyEvidenceError(
            "exact-boundary allocator run_id does not match the prepared identity"
        )
    if payload.get("range_name") != "GPUWRF_M0_FORECAST_INTEGRATION":
        raise ResidencyEvidenceError(
            "exact-boundary allocator range identity changed"
        )
    result_sha256 = payload.get("result_sha256")
    if not isinstance(result_sha256, str) or not SHA256_PATTERN.fullmatch(
        result_sha256
    ):
        raise ResidencyEvidenceError(
            "exact-boundary allocator result digest is invalid"
        )
    integer_fields = (
        "pid",
        "measurement_start_ns",
        "measurement_end_ns",
        "peak_bytes_in_use",
        "peak_bytes_reserved",
    )
    if any(
        isinstance(payload.get(field), bool)
        or not isinstance(payload.get(field), int)
        for field in integer_fields
    ):
        raise ResidencyEvidenceError(
            "exact-boundary allocator numeric fields must be exact integers"
        )
    forecast_pid = payload["pid"]
    start_ns = payload["measurement_start_ns"]
    end_ns = payload["measurement_end_ns"]
    peak_in_use = payload["peak_bytes_in_use"]
    peak_reserved = payload["peak_bytes_reserved"]
    if (
        forecast_pid <= 0
        or start_ns <= 0
        or end_ns <= start_ns
        or peak_in_use < 0
        or peak_reserved < peak_in_use
    ):
        raise ResidencyEvidenceError(
            "exact-boundary allocator numeric fields are invalid"
        )
    if payload.get("device_platform") != "gpu":
        raise ResidencyEvidenceError(
            "exact-boundary allocator did not execute on the GPU platform"
        )
    ordinal = payload.get("device_local_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, (int, str, type(None))):
        raise ResidencyEvidenceError(
            "exact-boundary allocator device ordinal is invalid"
        )
    return {
        **payload,
        "forecast_pid": forecast_pid,
        "identity_binding": {
            **identity.binding(),
            "source": (
                "lock-owner preauthorization plus selected-UUID external sampler"
            ),
        },
    }


class LockOwnerResidencySampler:
    """Lifecycle owned by the authorised capture parent, outside the forecast."""

    def __init__(
        self,
        *,
        identity: EvidenceIdentity,
        allocator_sidecar: Path,
        output_sidecar: Path,
        stderr_path: Path,
        cadence_ms: int = DEFAULT_CADENCE_MS,
        backend: SamplingBackend | None = None,
    ) -> None:
        if cadence_ms <= 0:
            raise ResidencyEvidenceError("sample cadence must be positive")
        if os.path.lexists(output_sidecar):
            raise ResidencyEvidenceError(
                f"residency sidecar target already exists: {output_sidecar}"
            )
        self.identity = identity
        self.allocator_sidecar = allocator_sidecar
        self.output_sidecar = output_sidecar
        self.cadence_ms = int(cadence_ms)
        self.backend = backend or NvidiaSmiStreamingBackend(
            identity.device_uuid,
            cadence_ms=self.cadence_ms,
            stderr_path=stderr_path,
        )
        self._lock = threading.Lock()
        self._samples = 0
        self._peak_absolute_bytes = 0
        self._seen_process_pids: set[int] = set()
        self._tree_process_pids: set[int] = set()
        self._unexpected_process_pids: set[int] = set()
        self._pending_process_pids: set[int] = set()
        self._malformed_samples = 0
        self._baseline_bytes: int | None = None
        self._started_ns: int | None = None
        self._stream_started_ns: int | None = None
        self._started_utc: str | None = None
        self._root_pid: int | None = None
        self._process_group_id: int | None = None
        self._jax_modules_at_start: tuple[str, ...] | None = None
        self._stopped = False

    def _total_sample(self, sampled_ns: int, used_bytes: int) -> None:
        with self._lock:
            if sampled_ns <= 0 or used_bytes < 0:
                self._malformed_samples += 1
                return
            self._samples += 1
            self._peak_absolute_bytes = max(self._peak_absolute_bytes, used_bytes)

    def _process_sample(self, sampled_ns: int, pid: int, used_bytes: int) -> None:
        with self._lock:
            if sampled_ns <= 0 or pid <= 0 or used_bytes < 0:
                self._malformed_samples += 1
                return
            if pid in self._seen_process_pids:
                return
            self._seen_process_pids.add(pid)
            if self._root_pid is None or self._process_group_id is None:
                self._pending_process_pids.add(pid)
            elif _pid_in_group_or_tree(
                pid,
                root_pid=self._root_pid,
                process_group_id=self._process_group_id,
            ):
                self._tree_process_pids.add(pid)
            else:
                self._unexpected_process_pids.add(pid)

    def start(self) -> None:
        if self._started_ns is not None:
            raise ResidencyEvidenceError("residency sampler was started twice")
        loaded_jax = _loaded_jax_modules()
        if loaded_jax:
            raise ResidencyEvidenceError(
                "lock-owner sampler process already loaded JAX/JAXLIB before "
                f"telemetry start: {loaded_jax[:8]}"
            )
        self._jax_modules_at_start = loaded_jax
        self._started_ns = time.monotonic_ns()
        self._started_utc = datetime.now(timezone.utc).isoformat()
        baseline, baseline_processes = self.backend.baseline()
        if baseline < 0:
            raise ResidencyEvidenceError("negative baseline total residency")
        if baseline_processes:
            raise ResidencyEvidenceError(
                "selected GPU had compute contexts before forecast launch: "
                + ",".join(str(pid) for pid, _used in baseline_processes)
        )
        self._baseline_bytes = baseline
        with self._lock:
            # The one-shot baseline is not a persistent-stream observation and
            # therefore cannot inflate stream coverage.
            self._peak_absolute_bytes = baseline
        try:
            self._stream_started_ns = time.monotonic_ns()
            self.backend.start(self._total_sample, self._process_sample)
        except Exception:
            self.backend.stop()
            raise

    def attach(self, *, root_pid: int, process_group_id: int) -> None:
        if self._started_ns is None:
            raise ResidencyEvidenceError("cannot attach an unstarted residency sampler")
        if root_pid <= 0 or process_group_id <= 0:
            raise ResidencyEvidenceError("sampler target PID/PGID must be positive")
        if self._root_pid is not None:
            raise ResidencyEvidenceError("residency sampler target was attached twice")
        self._root_pid = root_pid
        self._process_group_id = process_group_id
        with self._lock:
            for pid in self._pending_process_pids:
                if _pid_in_group_or_tree(
                    pid,
                    root_pid=root_pid,
                    process_group_id=process_group_id,
                ):
                    self._tree_process_pids.add(pid)
                else:
                    self._unexpected_process_pids.add(pid)
            self._pending_process_pids.clear()

    def abort(self) -> None:
        if not self._stopped:
            self.backend.stop()
            self._stopped = True

    def finish(self) -> dict[str, Any]:
        if self._started_ns is None or self._baseline_bytes is None:
            raise ResidencyEvidenceError("residency sampler never started")
        if self._root_pid is None or self._process_group_id is None:
            raise ResidencyEvidenceError("residency sampler never attached to the forecast")
        if not self._stopped:
            self.backend.stop()
            self._stopped = True
        backend_errors = list(getattr(self.backend, "_reader_errors", []))
        if backend_errors:
            raise ResidencyEvidenceError(
                "external sampler reader failed: " + "; ".join(backend_errors)
            )
        finished_ns = time.monotonic_ns()
        finished_utc = datetime.now(timezone.utc).isoformat()

        if not self.allocator_sidecar.is_file():
            raise ResidencyEvidenceError(
                f"forecast allocator sidecar is missing: {self.allocator_sidecar}"
            )
        try:
            allocator = json.loads(self.allocator_sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResidencyEvidenceError(f"invalid allocator sidecar JSON: {exc}") from exc
        allocator = validate_allocator_sidecar(allocator, self.identity)
        forecast_pid = int(allocator["forecast_pid"])

        with self._lock:
            samples = self._samples
            peak_absolute = self._peak_absolute_bytes
            seen_pids = set(self._seen_process_pids)
            tree_pids = set(self._tree_process_pids)
            unexpected = set(self._unexpected_process_pids)
            malformed = self._malformed_samples
        if samples <= 0:
            raise ResidencyEvidenceError("total-residency stream emitted no samples")
        if not _monotonic_interval_encloses(
            self._started_ns,
            int(allocator["measurement_start_ns"]),
            int(allocator["measurement_end_ns"]),
            finished_ns,
        ):
            raise ResidencyEvidenceError(
                "sampler interval does not enclose forecast integration interval"
            )

        if forecast_pid not in seen_pids:
            raise ResidencyEvidenceError(
                "forecast PID was never observed in selected-GPU process telemetry"
            )
        tree_pids.add(forecast_pid)
        unexpected.discard(forecast_pid)
        if unexpected:
            raise ResidencyEvidenceError(
                "unexpected competing GPU contexts: "
                + ",".join(str(pid) for pid in sorted(unexpected))
            )

        peak_subtracted = max(0, peak_absolute - self._baseline_bytes)
        if self._stream_started_ns is None:
            raise ResidencyEvidenceError("residency telemetry stream never became ready")
        duration_ms = (finished_ns - self._stream_started_ns) / 1_000_000.0
        expected_samples = max(1, int(duration_ms // self.cadence_ms) + 1)
        misses = max(0, expected_samples - samples) + malformed
        miss_fraction = _sampling_miss_fraction(samples, misses)
        if not _sampling_coverage_is_acceptable(samples, misses):
            raise ResidencyEvidenceError(
                f"residency sampling coverage invalid: {samples} samples, "
                f"{misses} misses, fraction={miss_fraction:.6f}, "
                f"limit={MAX_SAMPLING_MISS_FRACTION:.6f}"
            )
        if self._jax_modules_at_start != ():
            raise ResidencyEvidenceError(
                "sampler JAX import check did not complete before telemetry"
            )

        commands = getattr(self.backend, "commands", {})
        command_hashes = {
            name: hashlib.sha256(
                "\0".join(command).encode("utf-8")
            ).hexdigest()
            for name, command in sorted(commands.items())
        }
        payload = {
            "schema": SAMPLER_SCHEMA,
            "sampler_process_role": "lock_owner_parent",
            "sampler_backend": "external-persistent-nvidia-smi",
            "jax_imported_by_sampler": False,
            "jax_import_check": {
                "method": "sys.modules-prefix-scan-before-baseline-or-stream",
                "status": "PASS",
                "loaded_modules": [],
            },
            "orphan_control": "linux-prctl-pdeathsig-sigterm-plus-parent-stop",
            **self.identity.binding(),
            "forecast_pid": forecast_pid,
            "capture_root_pid": self._root_pid,
            "capture_process_group_id": self._process_group_id,
            "baseline_absolute_bytes": self._baseline_bytes,
            "peak_absolute_bytes": peak_absolute,
            "peak_baseline_subtracted_bytes": peak_subtracted,
            "sample_cadence_ms": self.cadence_ms,
            "samples": samples,
            "sampling_misses": misses,
            "sampling_quality": {
                "baseline_counted_as_stream_sample": False,
                "denominator": "stream_samples_plus_inferred_or_malformed_misses",
                "observed_miss_fraction": miss_fraction,
                "maximum_miss_fraction": MAX_SAMPLING_MISS_FRACTION,
                "status": "PASS",
            },
            "sampling_limitations": list(SAMPLING_LIMITATIONS),
            "malformed_samples": malformed,
            "observed_process_tree_pids": sorted(tree_pids),
            "unexpected_competing_contexts": [],
            "measurement_start_ns": self._started_ns,
            "measurement_end_ns": finished_ns,
            "measurement_start_utc": self._started_utc,
            "measurement_end_utc": finished_utc,
            "command_sha256": command_hashes,
            "product_metric": (
                "baseline-subtracted peak total selected-device residency"
            ),
            "allocator_role": "decomposition-cross-check-only",
            "sampler_ram_scaling": "O(unique process IDs); sample values aggregated online",
        }
        _atomic_json_no_replace(self.output_sidecar, payload)
        return payload


def identity_template(run_id: str) -> dict[str, Any]:
    """Print-only template used to prepare a future manager-owned window."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ResidencyEvidenceError("template run_id is invalid")
    return {
        "schema": IDENTITY_SCHEMA,
        "run_id": run_id,
        "source_sha256": "<FULL_SHA256>",
        "config_sha256": "<FULL_SHA256>",
        "input_manifest_sha256": "<FULL_SHA256>",
        "device_uuid": "GPU-<FULL_UUID>",
        "workload_identity_sha256": "<FULL_SHA256>",
        "integration_scope_sha256": "<FULL_SHA256>",
        "event_mix_sha256": "<FULL_SHA256>",
        "note": "Populate and hash-bind before requesting the manager-owned device executor.",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-identity-template", metavar="RUN_ID")
    mode.add_argument("--build-identity", metavar="RUN_ID")
    parser.add_argument("--device-uuid")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "src/gpuwrf",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.print_identity_template:
        print(json.dumps(identity_template(args.print_identity_template), indent=2))
        return 0
    if not args.device_uuid or args.run_dir is None or args.output is None:
        parser.error(
            "--build-identity requires --device-uuid, --run-dir, and --output"
        )
    identity = derive_identity(
        run_id=args.build_identity,
        device_uuid=args.device_uuid,
        source_root=args.source_root,
        run_dir=args.run_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_no_replace(
        args.output, {"schema": IDENTITY_SCHEMA, **identity.binding()}
    )
    print(json.dumps({
        "status": "WROTE",
        "path": str(args.output),
        "sha256": _sha256_file(args.output),
        "identity": identity.binding(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
