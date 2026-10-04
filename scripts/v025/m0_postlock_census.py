#!/usr/bin/env python3
"""JAX-free, post-lock reduction of one hash-bound M0 Nsight report.

This module is intentionally import-safe: it never imports JAX or ``gpuwrf``.
The live entrypoint accepts only a successful outer-lock return proof and the
unchanged W2 manager result, exports the existing ``.nsys-rep`` on CPU, and
builds a fail-closed §9 census.  Missing W3 or hardware-counter facts remain
``MISSING``; they are never filled by estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCHEMA = "wrf_gpu2.v025.m0.postlock_kernel_census.v1"
RELEASE_SCHEMA = "wrf_gpu2.v025.m0.lock_release_proof.v1"
SESSION_RELEASE_SCHEMA = "wrf_gpu2.v025.m0.session_lock_release_proof.v1"
STATUS = "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
RANGE_NAME = "GPUWRF_M0_FORECAST_INTEGRATION"
STEP_RANGE_PREFIX = "GPUWRF_M0_TIMESTEP"
MIN_ATTRIBUTION = 0.95
A6_CENSUS = REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STEP_INDEX_RE = re.compile(r"(?:step|index)[=:](\d+)", re.I)
ACCELERATOR_ROOTS = ("jax", "jaxlib", "gpuwrf")
LOCK_ENVIRONMENT = (
    "GPUWRF_GPU_LOCK_HELD",
    "GPUWRF_GPU_LOCK_TOKEN",
    "GPUWRF_GPU_LOCK_HOLDER_FILE",
    "GPUWRF_GPU_LOCK_LABEL",
    "GPUWRF_GPU_LOCK_FD",
    "GPUWRF_GPU_LOCK_FILE",
)
NSYS_M1_BLOCKING_FIELDS = (
    "ordinary_and_cadence_step_classes",
    "kernels_and_device_time_per_step",
    "family_launch_and_device_time_shares",
    "top_kernel_duration_distributions",
    "inter_kernel_gaps_and_tail_kernels",
    "timestamped_integration_range_transfer_audit",
    "launch_and_device_time_attribution",
    "profiler_perturbation",
    "cold_cached_and_clean_timing",
    "single_case_vram_and_host_rss",
)
NCU_M2_DEFERRED_FIELDS = (
    "registers_per_thread",
    "achieved_occupancy",
    "stall_distribution",
    "dram_bytes",
    "l2_bytes",
    "source_location",
)


class PostlockRefusal(RuntimeError):
    """Release provenance, capture identity, or census input failed closed."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_command(command: Sequence[str]) -> list[str]:
    """Make command identity independent of interpreter/worktree location."""

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


def normalized_command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        "\0".join(normalize_command(command)).encode("utf-8")
    ).hexdigest()


def _atomic_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise PostlockRefusal(f"refusing to replace post-lock proof: {path}")
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
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


def _load_json(path: Path, *, what: str) -> dict[str, Any]:
    if Path(path).is_symlink() or not Path(path).is_file():
        raise PostlockRefusal(f"{what} is missing/not a regular file: {path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostlockRefusal(f"{what} is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise PostlockRefusal(f"{what} is not a JSON object")
    return payload


def _loaded_accelerator_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.")
               for root in ACCELERATOR_ROOTS)
    )


def assert_accelerator_free() -> None:
    contaminated = _loaded_accelerator_modules()
    if contaminated:
        raise PostlockRefusal(
            "post-lock analysis process already imported accelerator roots: "
            f"{contaminated[:12]}"
        )


def assert_lock_environment_absent() -> None:
    inherited = [key for key in LOCK_ENVIRONMENT if key in os.environ]
    if inherited:
        raise PostlockRefusal(
            "post-lock process inherited canonical lock environment: "
            f"{inherited}"
        )


def build_lock_release_proof(
    *,
    window_id: str,
    run_id: str,
    wrapper_command: Sequence[str],
    wrapper_started_monotonic_ns: int,
    wrapper_returned_monotonic_ns: int,
    wrapper_returncode: int,
    window_result_path: Path,
    output_path: Path,
    wrapper_started_at_utc: str,
    wrapper_returned_at_utc: str,
) -> dict[str, Any]:
    """Record the successful return before any analysis process can launch."""

    assert_accelerator_free()
    assert_lock_environment_absent()
    if wrapper_returncode != 0:
        raise PostlockRefusal(
            f"lock wrapper returned {wrapper_returncode}; post-analysis forbidden"
        )
    if not (
        isinstance(wrapper_started_monotonic_ns, int)
        and isinstance(wrapper_returned_monotonic_ns, int)
        and 0 < wrapper_started_monotonic_ns < wrapper_returned_monotonic_ns
    ):
        raise PostlockRefusal("lock-wrapper monotonic endpoints are invalid")
    result = _load_json(window_result_path, what=f"{window_id} manager result")
    if result.get("status") != "OK":
        raise PostlockRefusal(f"{window_id} manager result is not OK")
    if result.get("window") != window_id or result.get("run_id") != run_id:
        raise PostlockRefusal("manager result window/run identity mismatch")
    for stamp in (wrapper_started_at_utc, wrapper_returned_at_utc):
        parsed = datetime.fromisoformat(stamp)
        if parsed.tzinfo is None:
            raise PostlockRefusal("lock-wrapper UTC endpoint lacks a timezone")
    payload = {
        "schema": RELEASE_SCHEMA,
        "status": "LOCK_WRAPPER_RETURNED",
        "window": window_id,
        "run_id": run_id,
        "wrapper_command": list(wrapper_command),
        "wrapper_command_sha256": hashlib.sha256(
            "\0".join(wrapper_command).encode("utf-8")
        ).hexdigest(),
        "wrapper_started_monotonic_ns": wrapper_started_monotonic_ns,
        "wrapper_returned_monotonic_ns": wrapper_returned_monotonic_ns,
        "wrapper_returncode": wrapper_returncode,
        "wrapper_started_at_utc": wrapper_started_at_utc,
        "wrapper_returned_at_utc": wrapper_returned_at_utc,
        "window_result_path": str(Path(window_result_path).resolve()),
        "window_result_bytes": Path(window_result_path).stat().st_size,
        "window_result_sha256": sha256_file(window_result_path),
        "lock_wrapper_process_exited": True,
        "analysis_launched": False,
        "created_by_pid": os.getpid(),
        "device_action": False,
    }
    _atomic_json_no_replace(output_path, payload)
    return payload


def build_session_lock_release_proof(
    *,
    session_proof_path: Path,
    wrapper_command: Sequence[str],
    wrapper_log_path: Path,
    wrapper_started_monotonic_ns: int,
    wrapper_returned_monotonic_ns: int,
    wrapper_returncode: int,
    wrapper_process_group: int,
    process_group_empty: bool,
    holder_file: Path,
    output_path: Path,
    wrapper_started_at_utc: str,
    wrapper_returned_at_utc: str,
) -> dict[str, Any]:
    """Bind the one wrapper return/release to the held-session root."""

    assert_accelerator_free()
    assert_lock_environment_absent()
    if not (
        isinstance(wrapper_started_monotonic_ns, int)
        and isinstance(wrapper_returned_monotonic_ns, int)
        and 0 < wrapper_started_monotonic_ns < wrapper_returned_monotonic_ns
        and isinstance(wrapper_process_group, int)
        and not isinstance(wrapper_process_group, bool)
        and wrapper_process_group > 0
        and process_group_empty is True
    ):
        raise PostlockRefusal("session wrapper endpoints/process-group proof invalid")
    session = _load_json(session_proof_path, what="held-session root proof")
    expected_session_hash = canonical_sha256(
        {key: value for key, value in session.items()
         if key != "session_sha256"}
    )
    if (
        session.get("schema") != "wrf_gpu2.v025.m0.core_session_execution.v1"
        or session.get("label") != "m0-core-w1-w2-w3-session"
        or session.get("orphan_sweep_completed") is not True
        or session.get("session_sha256") != expected_session_hash
    ):
        raise PostlockRefusal("held-session root proof is incomplete")
    log_path = Path(wrapper_log_path)
    if log_path.is_symlink() or not log_path.is_file():
        raise PostlockRefusal("session wrapper log is missing/not regular")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    label = "m0-core-w1-w2-w3-session"
    expected_prefix = [
        str(REPO / "scripts/with_gpu_lock.sh"),
        "--timeout",
        "0",
        "--label",
        label,
        "--",
        sys.executable,
        str(REPO / "scripts/v025/m0_three_window_executor.py"),
        "--manager-core-session-held",
        "--receipt",
    ]
    if (
        len(wrapper_command) != len(expected_prefix) + 1
        or list(wrapper_command[: len(expected_prefix)]) != expected_prefix
        or not str(wrapper_command[-1])
    ):
        raise PostlockRefusal("outer owner did not launch the exact held wrapper")
    acquired_marker = f"[with_gpu_lock] {label} ACQUIRED GPU lock"
    released_marker = f"[with_gpu_lock] {label} released GPU lock"
    if (
        log.count(acquired_marker) != 1
        or log.count(released_marker) != 1
        or log.index(acquired_marker) >= log.index(released_marker)
    ):
        raise PostlockRefusal(
            "wrapper log does not prove exactly one ordered acquire/release"
        )
    holder = Path(holder_file)
    if not holder.is_file() or holder.read_text(encoding="utf-8").strip():
        raise PostlockRefusal("canonical holder file is not empty after wrapper return")
    windows: dict[str, Any] = {}
    for window, detail_key in (
        ("W1", "W1_CACHED_AND_CORRECTNESS"),
        ("W2", "W2_PROFILED_CAPTURE"),
        ("W3", "W3_CLEAN_MATCHED_ARM"),
    ):
        detail = (session.get("stage_details") or {}).get(detail_key) or {}
        path_text = detail.get("result_path")
        if path_text is None:
            continue
        path = Path(str(path_text))
        result = _load_json(path, what=f"{window} session result")
        if result.get("window") != window or result.get("status") != "OK":
            raise PostlockRefusal(f"{window} session result is ineligible")
        windows[window] = {
            "run_id": result.get("run_id"),
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if session.get("status") == "PASS" and set(windows) != {"W1", "W2", "W3"}:
        raise PostlockRefusal("green held session lacks all three window bindings")
    payload = {
        "schema": SESSION_RELEASE_SCHEMA,
        "status": "LOCK_WRAPPER_RETURNED_AND_RELEASED",
        "label": label,
        "wrapper_command": list(wrapper_command),
        "wrapper_command_normalized": normalize_command(wrapper_command),
        "wrapper_command_sha256":
            normalized_command_sha256(wrapper_command),
        "wrapper_started_monotonic_ns": wrapper_started_monotonic_ns,
        "wrapper_returned_monotonic_ns": wrapper_returned_monotonic_ns,
        "wrapper_returncode": wrapper_returncode,
        "wrapper_started_at_utc": wrapper_started_at_utc,
        "wrapper_returned_at_utc": wrapper_returned_at_utc,
        "wrapper_process_group": wrapper_process_group,
        "wrapper_process_group_empty": True,
        "wrapper_log_path": str(log_path.resolve()),
        "wrapper_log_sha256": sha256_file(log_path),
        "acquire_count": 1,
        "release_count": 1,
        "holder_file": str(holder.resolve()),
        "holder_file_empty": True,
        "session_proof_path": str(Path(session_proof_path).resolve()),
        "session_proof_bytes": Path(session_proof_path).stat().st_size,
        "session_proof_sha256": sha256_file(session_proof_path),
        "session_status": session.get("status"),
        "windows": windows,
        "lock_wrapper_process_exited": True,
        "analysis_launched": False,
        "device_action": False,
    }
    payload["release_sha256"] = canonical_sha256(payload)
    _atomic_json_no_replace(output_path, payload)
    return payload


def validate_lock_release_proof(
    *,
    release_path: Path,
    window_result_path: Path,
    analysis_started_monotonic_ns: int,
    expected_window: str,
) -> dict[str, Any]:
    release = _load_json(release_path, what="lock-release proof")
    if release.get("schema") == SESSION_RELEASE_SCHEMA:
        if (
            release.get("status") != "LOCK_WRAPPER_RETURNED_AND_RELEASED"
            or release.get("label") != "m0-core-w1-w2-w3-session"
            or release.get("lock_wrapper_process_exited") is not True
            or release.get("wrapper_process_group_empty") is not True
            or release.get("holder_file_empty") is not True
            or release.get("acquire_count") != 1
            or release.get("release_count") != 1
            or release.get("wrapper_returncode") != 0
            or release.get("session_status") != "PASS"
            or release.get("wrapper_command_sha256")
            != normalized_command_sha256(
                release.get("wrapper_command") or []
            )
            or release.get("wrapper_command_normalized")
            != normalize_command(release.get("wrapper_command") or [])
            or release.get("analysis_launched") is not False
            or release.get("release_sha256")
            != canonical_sha256(
                {key: value for key, value in release.items()
                 if key != "release_sha256"}
            )
        ):
            raise PostlockRefusal("session lock-release proof is incomplete")
        returned_ns = release.get("wrapper_returned_monotonic_ns")
        if (
            isinstance(returned_ns, bool)
            or not isinstance(returned_ns, int)
            or returned_ns >= analysis_started_monotonic_ns
        ):
            raise PostlockRefusal(
                "analysis launch is not strictly after session-wrapper return"
            )
        binding = (release.get("windows") or {}).get(expected_window)
        if not isinstance(binding, dict):
            raise PostlockRefusal(
                f"session release lacks {expected_window} result binding"
            )
        path = Path(window_result_path)
        if (
            binding.get("path") != str(path.resolve())
            or not path.is_file()
            or binding.get("bytes") != path.stat().st_size
            or binding.get("sha256") != sha256_file(path)
        ):
            raise PostlockRefusal(
                f"session release {expected_window} result binding changed"
            )
        session_path = Path(str(release.get("session_proof_path", "")))
        if (
            not session_path.is_file()
            or release.get("session_proof_bytes") != session_path.stat().st_size
            or release.get("session_proof_sha256") != sha256_file(session_path)
        ):
            raise PostlockRefusal("held-session root changed after lock release")
        return {**release, "window": expected_window, "run_id": binding["run_id"]}
    if (
        release.get("schema") != RELEASE_SCHEMA
        or release.get("status") != "LOCK_WRAPPER_RETURNED"
        or release.get("window") != expected_window
        or release.get("wrapper_returncode") != 0
        or release.get("lock_wrapper_process_exited") is not True
        or release.get("analysis_launched") is not False
    ):
        raise PostlockRefusal("lock-release proof is incomplete or ineligible")
    returned_ns = release.get("wrapper_returned_monotonic_ns")
    if (
        isinstance(returned_ns, bool)
        or not isinstance(returned_ns, int)
        or returned_ns >= analysis_started_monotonic_ns
    ):
        raise PostlockRefusal(
            "analysis launch is not strictly after observed lock-wrapper return"
        )
    expected_path = str(Path(window_result_path).resolve())
    if release.get("window_result_path") != expected_path:
        raise PostlockRefusal("release proof names a different manager result")
    if (
        not Path(window_result_path).is_file()
        or release.get("window_result_bytes") != Path(window_result_path).stat().st_size
        or release.get("window_result_sha256") != sha256_file(window_result_path)
    ):
        raise PostlockRefusal("manager result changed after lock-wrapper return")
    return release


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"status": "MISSING", "count": 0}

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "status": "PASS",
        "count": len(ordered),
        "min": ordered[0],
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def derive_integration_scope(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    source_rep_sha256: str,
    production_derived: bool,
) -> dict[str, Any]:
    matches = [row for row in rows if row.get("name") == RANGE_NAME]
    if len(matches) != 1:
        return {
            "schema": "wrf_gpu2.v025.m0.integration_scope.v1",
            "status": "MISSING",
            "reason": (
                f"expected exactly one {RANGE_NAME!r} push/pop range; "
                f"observed {len(matches)}"
            ),
        }
    match = matches[0]
    start = float(match.get("start_ns", -1))
    duration = float(match.get("duration_ns", -1))
    end = start + duration
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        return {
            "schema": "wrf_gpu2.v025.m0.integration_scope.v1",
            "status": "MISSING",
            "reason": "integration NVTX range has invalid timestamps",
        }
    return {
        "schema": "wrf_gpu2.v025.m0.integration_scope.v1",
        "status": "OK",
        "run_id": run_id,
        "capture_run_id": run_id,
        "source_rep_sha256": source_rep_sha256,
        "production_derived": bool(production_derived),
        "mechanically_verified": True,
        "boundary_kind": "integration",
        "boundary_source": "nvtx_pushpop_trace",
        "range_name": RANGE_NAME,
        "start_ns": start,
        "end_ns": end,
        "duration_ns": duration,
    }


def _inside(row: dict[str, Any], start: float, end: float) -> bool:
    row_start = float(row["start_ns"])
    row_end = float(row.get("end_ns", row_start + float(row["duration_ns"])))
    return row_start >= start and row_end <= end


def _is_memory_event(row: dict[str, Any]) -> bool:
    name = str(row.get("name", "")).lower()
    return bool(
        row.get("source_memory_kind")
        or row.get("destination_memory_kind")
        or "memcpy" in name
        or "memset" in name
    )


def _explicit_step_ranges(
    nvtx_rows: list[dict[str, Any]],
    *,
    expected_steps: int,
    integration_start: float,
    integration_end: float,
) -> list[dict[str, Any]] | None:
    candidates = [
        row for row in nvtx_rows
        if str(row.get("name", "")).startswith(STEP_RANGE_PREFIX)
    ]
    if not candidates:
        return None
    indexed: dict[int, dict[str, Any]] = {}
    for row in candidates:
        match = STEP_INDEX_RE.search(str(row["name"]))
        if match is None:
            return None
        index = int(match.group(1))
        if index in indexed:
            return None
        indexed[index] = row
    if sorted(indexed) != list(range(expected_steps)):
        return None
    ranges = []
    previous_end = integration_start
    for index in range(expected_steps):
        row = indexed[index]
        start = float(row["start_ns"])
        end = start + float(row["duration_ns"])
        if start < previous_end or start < integration_start or end > integration_end:
            return None
        ranges.append({"index": index, "start_ns": start, "end_ns": end})
        previous_end = end
    return ranges


def _gap_step_ranges(
    kernels: list[dict[str, Any]],
    *,
    expected_steps: int,
    integration_start: float,
    integration_end: float,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    if expected_steps < 1 or len(kernels) < expected_steps:
        return None, {"reason": "fewer in-range kernels than expected timesteps"}
    gaps = []
    for index in range(len(kernels) - 1):
        gap = max(
            0.0,
            float(kernels[index + 1]["start_ns"])
            - float(kernels[index]["end_ns"]),
        )
        gaps.append((gap, index))
    if expected_steps == 1:
        cut_indices: list[int] = []
        boundary_gaps = []
        interior_gaps = [gap for gap, _index in gaps]
    else:
        positive = sorted(gaps, key=lambda item: (-item[0], item[1]))
        if len(positive) < expected_steps - 1:
            return None, {"reason": "not enough inter-kernel gaps"}
        selected = positive[: expected_steps - 1]
        if any(gap <= 0 for gap, _index in selected):
            return None, {"reason": "step boundary gaps are not positive"}
        cut_indices = sorted(index for _gap, index in selected)
        boundary_gaps = [gap for gap, index in gaps if index in set(cut_indices)]
        interior_gaps = [gap for gap, index in gaps if index not in set(cut_indices)]
        largest_interior = max(interior_gaps, default=0.0)
        smallest_boundary = min(boundary_gaps)
        if smallest_boundary < max(1.0, largest_interior * 1.5):
            return None, {
                "reason": "inter-step gaps are not separated from intra-step gaps",
                "smallest_boundary_gap_ns": smallest_boundary,
                "largest_interior_gap_ns": largest_interior,
                "required_ratio": 1.5,
            }
    chunks: list[list[dict[str, Any]]] = []
    begin = 0
    for cut in cut_indices:
        chunks.append(kernels[begin : cut + 1])
        begin = cut + 1
    chunks.append(kernels[begin:])
    if len(chunks) != expected_steps or any(not chunk for chunk in chunks):
        return None, {"reason": "gap segmentation did not yield nonempty steps"}
    ranges = []
    for index, chunk in enumerate(chunks):
        start = integration_start if index == 0 else float(chunk[0]["start_ns"])
        end = (
            integration_end
            if index == expected_steps - 1
            else float(chunk[-1]["end_ns"])
        )
        ranges.append({"index": index, "start_ns": start, "end_ns": end})
    return ranges, {
        "boundary_gap_ns": _distribution(boundary_gaps),
        "interior_gap_ns": _distribution(interior_gaps),
    }


def derive_step_census(
    *,
    nvtx_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    integration_scope: dict[str, Any],
    expected_steps: int,
    radiation_cadence_steps: int,
    family_for_name: Callable[[str], str],
) -> dict[str, Any]:
    if integration_scope.get("status") != "OK":
        return {"status": "MISSING", "reason": "integration scope is missing"}
    start = float(integration_scope["start_ns"])
    end = float(integration_scope["end_ns"])
    ordered = sorted(
        [row for row in kernel_rows if _inside(row, start, end)],
        key=lambda row: (float(row["start_ns"]), float(row["duration_ns"])),
    )
    crossing = [
        row for row in kernel_rows
        if float(row["start_ns"]) < end
        and float(row["end_ns"]) > start
        and not _inside(row, start, end)
    ]
    if crossing:
        return {
            "status": "MISSING",
            "reason": f"{len(crossing)} kernels cross the integration boundary",
        }
    ranges = _explicit_step_ranges(
        nvtx_rows,
        expected_steps=expected_steps,
        integration_start=start,
        integration_end=end,
    )
    derivation: dict[str, Any]
    if ranges is not None:
        derivation = {
            "method": "explicit-indexed-production-NVTX-step-ranges",
            "marker_prefix": STEP_RANGE_PREFIX,
        }
    else:
        ranges, gap_detail = _gap_step_ranges(
            ordered,
            expected_steps=expected_steps,
            integration_start=start,
            integration_end=end,
        )
        if ranges is None:
            return {
                "status": "MISSING",
                "reason": gap_detail.get("reason"),
                "gap_discriminator": gap_detail,
                "whole_process_fallback_used": False,
            }
        derivation = {
            "method": "expected-count-largest-gap-segmentation",
            "gap_discriminator": gap_detail,
            "whole_process_fallback_used": False,
        }

    steps = []
    for step_range in ranges:
        members = [
            row for row in ordered
            if _inside(row, step_range["start_ns"], step_range["end_ns"])
        ]
        if not members:
            return {
                "status": "MISSING",
                "reason": f"derived step {step_range['index']} contains no kernels",
            }
        family_counts: Counter[str] = Counter()
        family_time: defaultdict[str, float] = defaultdict(float)
        for row in members:
            family = family_for_name(str(row["name"]))
            family_counts[family] += 1
            family_time[family] += float(row["duration_ns"])
        steps.append(
            {
                **step_range,
                "kernel_launches": len(members),
                "device_time_ns": sum(float(row["duration_ns"]) for row in members),
                "families": {
                    name: {
                        "launches": family_counts[name],
                        "device_time_ns": family_time[name],
                    }
                    for name in sorted(family_counts)
                },
                "has_radiation": family_counts["physics.radiation"] > 0,
            }
        )
    ordinary = [step for step in steps if not step["has_radiation"]]
    radiation = [step for step in steps if step["has_radiation"]]
    expected_radiation = max(
        1, math.ceil(expected_steps / max(1, radiation_cadence_steps))
    )
    classes_status = (
        "PASS"
        if ordinary and len(radiation) == expected_radiation
        else "MISSING"
    )

    def class_record(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "steps": len(items),
            "indices": [item["index"] for item in items],
            "kernels_per_step": _distribution(
                item["kernel_launches"] for item in items
            ),
            "device_time_per_step_ns": _distribution(
                item["device_time_ns"] for item in items
            ),
        }

    return {
        "status": classes_status,
        "reason": (
            None
            if classes_status == "PASS"
            else "ordinary or expected radiation/cadence class is absent"
        ),
        "expected_steps": expected_steps,
        "derived_steps": len(steps),
        "radiation_cadence_steps": radiation_cadence_steps,
        "expected_radiation_steps": expected_radiation,
        "derivation": derivation,
        "classes": {
            "ordinary": class_record(ordinary),
            "radiation": class_record(radiation),
        },
        "steps": steps,
    }


def _top_kernel_distributions(
    kernels: list[dict[str, Any]],
    *,
    family_for_name: Callable[[str], str],
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in kernels:
        grouped[str(row["name"])].append(float(row["duration_ns"]))
    ranked = sorted(
        grouped.items(), key=lambda item: (-sum(item[1]), item[0])
    )
    return [
        {
            "name": name,
            "family": family_for_name(name),
            "launches": len(durations),
            "duration_ns": _distribution(durations),
            "registers_per_thread": "MISSING",
            "achieved_occupancy": "MISSING",
            "dram_bytes": "MISSING",
            "l2_bytes": "MISSING",
            "stall_distribution": "MISSING",
            "source_location": "MISSING",
        }
        for name, durations in ranked[:25]
    ]


def _timestamped_row_gate(
    cuda_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Name the aggregate-substitution refusal instead of raising ``KeyError``.

    Kimi minor m2: an aggregate ``cuda_gpu_mem_time_sum`` table has no
    ``start_ns``, and the scoping helper indexed it before the transfer gate
    could reject it, so the census died with a raw ``KeyError`` instead of an
    explicit, diagnosable gate result.  Fail-closed either way, but a named gate
    is what a post-lock operator can act on inside a coordinated session.
    """

    if cuda_rows is None:
        return {
            "status": "MISSING",
            "reason": "no timestamped CUDA trace was exported",
            "required": "cuda_gpu_trace rows with Start (ns) and Duration (ns)",
            "rows": 0,
        }
    unplaceable = 0
    for row in cuda_rows:
        start = row.get("start_ns")
        duration = row.get("duration_ns")
        if (
            isinstance(start, bool)
            or isinstance(duration, bool)
            or not isinstance(start, (int, float))
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(duration))
            or float(start) < 0.0
            or float(duration) < 0.0
        ):
            unplaceable += 1
    if unplaceable:
        return {
            "status": "BLOCKED",
            "reason": (
                f"{unplaceable} of {len(cuda_rows)} CUDA rows carry no usable "
                "timestamp; an aggregate summary can never place a transfer "
                "inside or outside the integration range"
            ),
            "required": "cuda_gpu_trace rows with Start (ns) and Duration (ns)",
            "rows": len(cuda_rows),
            "unplaceable_rows": unplaceable,
        }
    return {
        "status": "PASS",
        "rows": len(cuda_rows),
        "unplaceable_rows": 0,
        "rule": "every scoped row carries a finite start and duration",
    }


def _a6_hlo_binding(
    exact_child: dict[str, Any] | None,
) -> dict[str, Any]:
    if not A6_CENSUS.is_file():
        return {"status": "MISSING", "reason": f"A6 census missing: {A6_CENSUS}"}
    a6 = _load_json(A6_CENSUS, what="frozen A6 HLO census")
    operators = a6.get("operators")
    if (
        not isinstance(operators, list)
        or len(operators) != 49
        or len({item.get("name") for item in operators}) != 49
    ):
        return {"status": "MISSING", "reason": "frozen A6 census is not 49 operators"}
    static_complete = all(
        isinstance(item, dict)
        and isinstance(item.get("dtype_tokens_stablehlo"), dict)
        and isinstance(item.get("dtype_tokens_optimized"), dict)
        and isinstance(item.get("converts_stablehlo"), dict)
        and isinstance(item.get("converts_optimized"), dict)
        and item.get("status") in {"LOWERED", "NOT_LOWERED"}
        for item in operators
    )
    largest = sorted(
        (
            {
                "operator": item.get("name"),
                "family": item.get("family"),
                "temporary_bytes": (
                    (item.get("memory_analysis") or {}).get("temporary_bytes")
                ),
            }
            for item in operators
        ),
        key=lambda item: -(item["temporary_bytes"] or 0),
    )[:20]
    child_call = (exact_child or {}).get("call") or {}
    stablehlo = child_call.get("stablehlo")
    stablehlo_complete = (
        isinstance(stablehlo, dict)
        and SHA256_RE.fullmatch(str(stablehlo.get("sha256", ""))) is not None
        and isinstance(stablehlo.get("bytes"), int)
        and not isinstance(stablehlo.get("bytes"), bool)
        and stablehlo["bytes"] > 0
        and isinstance(stablehlo.get("dtype_token_counts"), dict)
        and isinstance(stablehlo.get("convert_operation_count"), int)
        and not isinstance(stablehlo.get("convert_operation_count"), bool)
        and stablehlo["convert_operation_count"] >= 0
    )
    compiled_memory = child_call.get("compiled_memory_analysis")
    memory_fields = (
        compiled_memory.get("fields")
        if isinstance(compiled_memory, dict)
        else None
    )
    required_memory_fields = (
        "temp_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
    )
    compiled_memory_complete = (
        isinstance(compiled_memory, dict)
        and compiled_memory.get("status") == "OK"
        and isinstance(memory_fields, dict)
        and all(
            isinstance(memory_fields.get(name), int)
            and not isinstance(memory_fields.get(name), bool)
            and memory_fields[name] >= 0
            for name in required_memory_fields
        )
    )
    hlo_status = (
        "PASS"
        if static_complete and stablehlo_complete and compiled_memory_complete
        else "MISSING"
    )
    return {
        "status": hlo_status,
        "a6_path": str(A6_CENSUS),
        "a6_sha256": sha256_file(A6_CENSUS),
        "operator_count": 49,
        "operator_names_sha256": canonical_sha256(
            [item.get("name") for item in operators]
        ),
        "family_labels": sorted(
            {str(item.get("family")) for item in operators}
        ),
        "dtype_and_convert_census": {
            "status": "PASS" if static_complete and stablehlo_complete else "MISSING",
            "per_operator": [
                {
                    "name": item.get("name"),
                    "family": item.get("family"),
                    "dtype_tokens_stablehlo":
                        item.get("dtype_tokens_stablehlo"),
                    "dtype_tokens_optimized":
                        item.get("dtype_tokens_optimized"),
                    "converts_stablehlo": item.get("converts_stablehlo"),
                    "converts_optimized": item.get("converts_optimized"),
                }
                for item in operators
            ],
            "exact_compiled_program": stablehlo or "MISSING",
        },
        "largest_temporaries": largest,
        "compiled_memory_analysis": compiled_memory or "MISSING",
        "compiled_memory_analysis_gate": {
            "status": "PASS" if compiled_memory_complete else "MISSING",
            "required_fields": list(required_memory_fields),
            "aggregate_sizes_are_not_peak_live": True,
        },
        "peak_live_buffers": "MISSING",
        "peak_live_buffers_reason": (
            "deferred by Amendment 6; M0 uses production-normal flags and "
            "does not collect compiler dumps"
        ),
        "peak_live_buffers_evidence": "MISSING",
        "peak_live_buffers_deferred": {
            "status": "MISSING",
            "waived": False,
            "owner": (
                "M1 same-set candidate diagnosis / M2 backend bake-off"
            ),
            "mechanically_reachable_from_m0_core": False,
        },
        "aggregate_temporary_bytes_is_not_peak_live": True,
    }


def _resource_evidence(
    *,
    exact_child: dict[str, Any] | None,
    residency: dict[str, Any] | None,
    host_rss: dict[str, Any] | None,
) -> dict[str, Any]:
    allocator = (exact_child or {}).get("allocator")
    exact_run_id = (exact_child or {}).get("run_id")
    vram_status = "PASS"
    reasons = []
    if not isinstance(allocator, dict):
        vram_status = "MISSING"
        reasons.append("same-process allocator record missing")
    if not isinstance(residency, dict):
        vram_status = "MISSING"
        reasons.append("lock-owner residency record missing")
    elif (
        residency.get("schema")
        != "wrf_gpu2.v025.m0.lock_owner_total_residency.v1"
        or (residency.get("sampling_quality") or {}).get("status") != "PASS"
        or residency.get("run_id") != exact_run_id
    ):
        vram_status = "MISSING"
        reasons.append("lock-owner residency schema/coverage/run binding invalid")
    host_status = (
        "PASS"
        if isinstance(host_rss, dict)
        and host_rss.get("status") == "PASS"
        and isinstance(host_rss.get("peak_process_tree_rss_bytes"), int)
        and host_rss.get("run_id") == exact_run_id
        else "MISSING"
    )
    return {
        "vram": {
            "status": vram_status,
            "reason": "; ".join(reasons) if reasons else None,
            "allocator": allocator if allocator is not None else "MISSING",
            "lock_owner_total_residency":
                residency if residency is not None else "MISSING",
        },
        "host_ram": {
            "status": host_status,
            "evidence": host_rss if host_status == "PASS" else "MISSING",
            "reason": (
                None
                if host_status == "PASS"
                else "future W2 capture must supply process-tree RSS sampling"
            ),
        },
    }


def _export_integrity_gate(
    export: dict[str, Any],
    *,
    run_id: str,
    source_hash: str,
) -> dict[str, Any]:
    required = export.get("required")
    records = export.get("exports")
    problems: list[str] = []
    if export.get("status") != "OK":
        problems.append("export status is not OK")
    if export.get("run_id") != run_id:
        problems.append("export run ID differs")
    if not isinstance(required, list) or not required:
        problems.append("required export inventory is missing")
        required = []
    if not isinstance(records, dict):
        problems.append("export records are missing")
        records = {}
    private_path = Path(str(export.get("private_sqlite_export", "")))
    private_hash = export.get("private_sqlite_sha256")
    fixture_private = (
        export.get("fixture_complete") is True
        and SHA256_RE.fullmatch(str(private_hash or "")) is not None
        and isinstance(export.get("private_sqlite_bytes"), int)
        and not isinstance(export.get("private_sqlite_bytes"), bool)
        and export["private_sqlite_bytes"] > 0
    )
    if (
        export.get("forced_sqlite_export_count") != 1
        or export.get("sqlite_report_read_count")
        != max(0, len(records) - 1)
        or (
            not fixture_private
            and (
                private_path.is_symlink()
                or not private_path.is_file()
                or private_path.stat().st_size
                != export.get("private_sqlite_bytes")
                or sha256_file(private_path) != private_hash
            )
        )
    ):
        problems.append("private SQLite one-export binding is invalid")
    for name in required:
        record = records.get(name)
        if not isinstance(record, dict):
            problems.append(f"{name}: record missing")
            continue
        if record.get("status") != "OK":
            problems.append(f"{name}: status is not OK")
        if record.get("run_id") != run_id:
            problems.append(f"{name}: run ID differs")
        if record.get("source_rep_sha256") != source_hash:
            problems.append(f"{name}: source report hash differs")
        if record.get("private_sqlite_sha256") != private_hash:
            problems.append(f"{name}: private SQLite hash differs")
        if (
            not SHA256_RE.fullmatch(str(record.get("sha256", "")))
            or not isinstance(record.get("data_rows"), int)
            or isinstance(record.get("data_rows"), bool)
            or int(record.get("data_rows", 0)) <= 0
        ):
            problems.append(f"{name}: output hash/row count is invalid")
    return {
        "status": "PASS" if not problems else "BLOCKED",
        "problems": problems,
        "required": list(required),
        "manifest_sha256": export.get("manifest_sha256"),
    }


def build_census(
    *,
    export: dict[str, Any],
    nvtx_rows: list[dict[str, Any]] | None,
    cuda_rows: list[dict[str, Any]] | None,
    kernel_summary: list[dict[str, Any]] | None,
    run_id: str,
    exact_child: dict[str, Any] | None,
    release_binding: dict[str, Any],
    production_derived: bool,
    residency: dict[str, Any] | None = None,
    host_rss: dict[str, Any] | None = None,
    matched_profiler: dict[str, Any] | None = None,
    clean_exact_child: dict[str, Any] | None = None,
    production_reference: dict[str, Any] | None = None,
    candidate_coverage: dict[str, Any] | None = None,
    cold_readiness_seconds: float | None = None,
    hlo_evidence_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure arithmetic layer used by live analysis and adversarial fixtures."""

    assert_accelerator_free()
    import baseline_census as baseline
    import run_gpu_arm as arm

    source_hash = str(export.get("source_rep_sha256", ""))
    scope = (
        derive_integration_scope(
            nvtx_rows or [],
            run_id=run_id,
            source_rep_sha256=source_hash,
            production_derived=production_derived,
        )
        if nvtx_rows is not None
        else {"status": "MISSING", "reason": "NVTX push/pop table unavailable"}
    )
    row_integrity = _timestamped_row_gate(cuda_rows)
    if (
        scope.get("status") == "OK"
        and cuda_rows is not None
        and row_integrity["status"] == "PASS"
    ):
        in_range = [
            row for row in cuda_rows
            if _inside(row, float(scope["start_ns"]), float(scope["end_ns"]))
        ]
        kernel_rows = [row for row in in_range if not _is_memory_event(row)]
    else:
        in_range = []
        kernel_rows = []
    trace_kernels = [
        {
            "name": row["name"],
            "device_time_ns": float(row["duration_ns"]),
            "launches": 1,
        }
        for row in kernel_rows
    ]
    attribution = (
        arm.attribute_device_time(
            trace_kernels, min_attribution=MIN_ATTRIBUTION
        )
        if trace_kernels
        else {
            "meets_attribution_bar": False,
            "attributed_device_time_share": 0.0,
            "attributed_launch_share": 0.0,
            "unknown_device_time_share": 1.0,
            "unknown_launch_share": 1.0,
            "families": {},
            "total_device_time_ns": 0.0,
            "total_launches": 0,
        }
    )
    attribution["status"] = (
        "PASS" if attribution.get("meets_attribution_bar") else "BLOCKED"
    )
    summary_crosscheck = {
        "status": "MISSING",
        "reason": "kernel summary unavailable",
    }
    if kernel_summary:
        summary_launches = sum(int(row["launches"]) for row in kernel_summary)
        summary_time = sum(float(row["device_time_ns"]) for row in kernel_summary)
        scoped_launches = len(kernel_rows)
        scoped_time = sum(float(row["duration_ns"]) for row in kernel_rows)
        summary_crosscheck = {
            "status": (
                "PASS"
                if summary_launches >= scoped_launches
                and summary_time + 1.0 >= scoped_time
                else "BLOCKED"
            ),
            "whole_capture_launches": summary_launches,
            "whole_capture_device_time_ns": summary_time,
            "scoped_launches": scoped_launches,
            "scoped_device_time_ns": scoped_time,
            "rule": "whole-capture summary totals must enclose exact-range trace totals",
        }

    case_namelist = (
        (((exact_child or {}).get("case") or {}).get("case_metadata") or {})
        .get("namelist")
        or {}
    )
    hours = float(((exact_child or {}).get("call") or {}).get("hours", 1.0))
    dt_s = float(case_namelist.get("dt_s", 0.0) or 0.0)
    radiation_cadence = int(
        case_namelist.get("radiation_cadence_steps", 0) or 0
    )
    expected_steps = (
        int(round(hours * 3600.0 / dt_s))
        if dt_s > 0 and abs(hours * 3600.0 / dt_s - round(hours * 3600.0 / dt_s))
        <= 1e-8
        else 0
    )
    step_census = (
        derive_step_census(
            nvtx_rows=nvtx_rows or [],
            kernel_rows=kernel_rows,
            integration_scope=scope,
            expected_steps=expected_steps,
            radiation_cadence_steps=radiation_cadence,
            family_for_name=arm.attribute_kernel,
        )
        if expected_steps > 0
        else {
            "status": "MISSING",
            "reason": "exact child lacks an integral hours/dt timestep count",
        }
    )
    transfer = baseline.transfer_gate(cuda_rows, scope)
    if transfer.get("status") == "OK":
        transfer["status"] = "PASS"
    profiler = baseline.profiler_gate(matched_profiler)
    if profiler.get("status") == "OK":
        profiler["status"] = "PASS"
    resources = _resource_evidence(
        exact_child=exact_child,
        residency=residency,
        host_rss=host_rss,
    )
    hlo = _a6_hlo_binding(exact_child)
    if hlo_evidence_override is not None:
        if hlo_evidence_override.get("fixture_complete") is not True:
            raise PostlockRefusal(
                "HLO evidence override lacks explicit complete-fixture provenance"
            )
        hlo = {
            **hlo,
            **hlo_evidence_override,
            "cpu_fixture_override": True,
        }
    taxonomy = {
        "status": "PASS" if hlo.get("status") == "PASS" else "MISSING",
        "a6_operator_count": hlo.get("operator_count"),
        "a6_sha256": hlo.get("a6_sha256"),
        "family_patterns": [list(item) for item in arm.FAMILY_PATTERNS],
        "family_patterns_sha256": canonical_sha256(
            [list(item) for item in arm.FAMILY_PATTERNS]
        ),
        "attribution_threshold": MIN_ATTRIBUTION,
        "independent_launch_and_time_bars": True,
    }
    gaps = []
    ordered = sorted(kernel_rows, key=lambda row: float(row["start_ns"]))
    for left, right in zip(ordered, ordered[1:]):
        gaps.append(max(0.0, float(right["start_ns"]) - float(left["end_ns"])))
    top_kernels = _top_kernel_distributions(
        kernel_rows, family_for_name=arm.attribute_kernel
    )
    clean_status = (
        "PASS"
        if isinstance(clean_exact_child, dict)
        and clean_exact_child.get("status") == "OK"
        else "MISSING"
    )
    timing = {
        "cold_readiness": (
            float(cold_readiness_seconds)
            if isinstance(cold_readiness_seconds, (int, float))
            and not isinstance(cold_readiness_seconds, bool)
            and math.isfinite(float(cold_readiness_seconds))
            and float(cold_readiness_seconds) > 0
            else "MISSING"
        ),
        "cached_readiness": (
            ((exact_child or {}).get("timing") or {}).get(
                "readiness_seconds", "MISSING"
            )
        ),
        "profiled_warm_integration": (
            ((exact_child or {}).get("timing") or {}).get(
                "integration_seconds", "MISSING"
            )
        ),
        "clean_warm_integration": (
            ((clean_exact_child or {}).get("timing") or {}).get(
                "integration_seconds", "MISSING"
            )
            if clean_status == "PASS"
            else "MISSING"
        ),
        "clean_status": clean_status,
        "profiler_perturbation": profiler,
    }
    export_gate = _export_integrity_gate(
        export,
        run_id=run_id,
        source_hash=source_hash,
    )
    exact_run = (exact_child or {}).get("run_id")
    candidate_source = (candidate_coverage or {}).get("source_rep_sha256")
    source_binding = {
        "status": (
            "PASS"
            if SHA256_RE.fullmatch(source_hash)
            and export.get("run_id") == run_id
            and release_binding.get("run_id") == run_id
            and exact_run == run_id
            and (
                candidate_coverage is None
                or (
                    candidate_coverage.get("run_id") == run_id
                    and candidate_coverage.get("capture_run_id") == run_id
                    and candidate_source == source_hash
                )
            )
            else "BLOCKED"
        ),
        "run_id": run_id,
        "source_rep_sha256": source_hash,
        "exact_child_run_id": exact_run,
        "candidate_source_rep_sha256": candidate_source,
        "release_binding": release_binding,
    }
    if production_reference is not None and candidate_coverage is not None:
        production_gate = baseline.production_representativeness_gate(
            attribution.get("families", {}),
            production_reference,
            candidate_coverage,
        )
        if production_gate.get("status") == "OK":
            production_gate["status"] = "PASS"
    else:
        production_gate = {
            "status": "MISSING",
            "reason": (
                "matched two-domain production trace is not yet bound to this "
                "W2 capture; A6 remains a static diagnostic, not a substitute"
            ),
        }
    timing_values = (
        timing["cold_readiness"],
        timing["cached_readiness"],
        timing["profiled_warm_integration"],
        timing["clean_warm_integration"],
    )
    timing_gate = {
        "status": (
            "PASS"
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0
                for value in timing_values
            )
            else "MISSING"
        ),
        "values": timing,
    }
    hlo_gate = {
        "status": (
            "PASS"
            if hlo.get("status") == "PASS"
            and (hlo.get("dtype_and_convert_census") or {}).get("status")
            == "PASS"
            and (hlo.get("compiled_memory_analysis_gate") or {}).get("status")
            == "PASS"
            and resources["vram"].get("status") == "PASS"
            and resources["host_ram"].get("status") == "PASS"
            else "MISSING"
        ),
        "peak_live_buffers": "MISSING",
        "peak_live_buffers_rule": (
            "deferred, waived=false; aggregate temporary_bytes is not peak "
            "live and does not influence M0-CORE"
        ),
        "requires": [
            "hash-bound static dtype/convert census",
            "exact executable stablehlo identity",
            "compiled memory_analysis temp/output/alias fields",
            "W2 measured GPU VRAM",
            "W2 measured host RSS",
        ],
        "evidence": hlo,
    }
    required = {
        "export": export_gate,
        "source_binding": source_binding,
        "cuda_trace_row_integrity": row_integrity,
        "integration_scope": {
            **scope,
            "status": "PASS" if scope.get("status") == "OK" else scope.get(
                "status", "MISSING"
            ),
        },
        "step_classes": step_census,
        "taxonomy": taxonomy,
        "attribution": attribution,
        "summary_crosscheck": summary_crosscheck,
        "transfer_audit": transfer,
        "vram": resources["vram"],
        "host_ram": resources["host_ram"],
        "profiler_perturbation": profiler,
        "clean_w3": {"status": clean_status},
        "timing_completeness": timing_gate,
        "hlo_completeness": hlo_gate,
        "production_representativeness": production_gate,
    }
    not_pass = [
        name for name, gate in required.items()
        if gate.get("status") != "PASS"
    ]
    payload = {
        "schema": SCHEMA,
        "status": "OK" if not not_pass else "BLOCKED",
        "cpu_closure_status": STATUS,
        "run_id": run_id,
        "production_derived": bool(production_derived),
        "source_binding": source_binding,
        "release_provenance": release_binding,
        "export": export,
        "integration_scope": scope,
        "step_census": step_census,
        "family_census": attribution.get("families", {}),
        "device_time_attribution": attribution,
        "kernel_summary_crosscheck": summary_crosscheck,
        "top_kernel_distributions": top_kernels,
        "launch_duration_ns": _distribution(
            float(row["duration_ns"]) for row in kernel_rows
        ),
        "inter_kernel_gap_ns": _distribution(gaps),
        "tail_kernels": sorted(
            (
                {
                    "name": row["name"],
                    "duration_ns": float(row["duration_ns"]),
                    "start_ns": float(row["start_ns"]),
                }
                for row in kernel_rows
            ),
            key=lambda row: (-row["duration_ns"], row["name"]),
        )[:25],
        "transfer_audit": transfer,
        "hlo": hlo,
        "taxonomy": taxonomy,
        "timing": timing,
        "resources": resources,
        "production_representativeness": production_gate,
        "measurement_partition": {
            "m0_core_m1_blocking": {
                "source": "nsys plus bound sidecars/clean arm",
                "fields": list(NSYS_M1_BLOCKING_FIELDS),
                "governed_by_required_gates": True,
            },
            "m0_deferred_m2": {
                "source": "ncu",
                "fields": list(NCU_M2_DEFERRED_FIELDS),
                "status": "MISSING_UNTIL_M2",
                "governed_by_required_gates": False,
                "waived": False,
            },
        },
        "raw_artifacts": {
            "source_rep": export.get("source_rep"),
            "source_rep_sha256": source_hash,
            "private_sqlite": {
                "path": export.get("private_sqlite_export"),
                "bytes": export.get("private_sqlite_bytes"),
                "sha256": export.get("private_sqlite_sha256"),
                "forced_export_count":
                    export.get("forced_sqlite_export_count"),
            },
            "exports": {
                name: {
                    "path": record.get("path"),
                    "sha256": record.get("sha256"),
                    "data_rows": record.get("data_rows"),
                }
                for name, record in (export.get("exports") or {}).items()
            },
        },
        "required_gates": required,
        "gates_not_pass": not_pass,
        "honest_missing_policy": (
            "no top-level OK while any required C1/C2 sub-gate is "
            "MISSING/BLOCKED"
        ),
    }
    payload["census_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "census_sha256"}
    )
    return payload


def validate_census(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject a stale hash or a top-level success hiding any missing gate."""

    if payload.get("schema") != SCHEMA:
        raise PostlockRefusal("post-lock census schema changed")
    partition = payload.get("measurement_partition")
    if not isinstance(partition, dict):
        raise PostlockRefusal("post-lock census measurement partition is missing")
    core = partition.get("m0_core_m1_blocking")
    deferred = partition.get("m0_deferred_m2")
    if (
        not isinstance(core, dict)
        or core.get("source") != "nsys plus bound sidecars/clean arm"
        or core.get("fields") != list(NSYS_M1_BLOCKING_FIELDS)
        or core.get("governed_by_required_gates") is not True
    ):
        raise PostlockRefusal(
            "nsys-derived M0-CORE/M1-blocking measurement partition changed"
        )
    if (
        not isinstance(deferred, dict)
        or deferred.get("source") != "ncu"
        or deferred.get("fields") != list(NCU_M2_DEFERRED_FIELDS)
        or deferred.get("status") != "MISSING_UNTIL_M2"
        or deferred.get("governed_by_required_gates") is not False
        or deferred.get("waived") is not False
    ):
        raise PostlockRefusal(
            "ncu-only M0-DEFERRED/M2 measurement partition changed"
        )
    required = payload.get("required_gates")
    if not isinstance(required, dict) or not required:
        raise PostlockRefusal("post-lock census required-gate inventory is missing")
    observed = sorted(
        name for name, gate in required.items()
        if not isinstance(gate, dict) or gate.get("status") != "PASS"
    )
    recorded = sorted(payload.get("gates_not_pass") or [])
    if recorded != observed:
        raise PostlockRefusal("post-lock census missing-gate inventory is stale")
    expected_status = "OK" if not observed else "BLOCKED"
    if payload.get("status") != expected_status:
        raise PostlockRefusal(
            "top-level census status hides a MISSING/BLOCKED required gate"
        )
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "census_sha256"}
    )
    if payload.get("census_sha256") != expected_hash:
        raise PostlockRefusal("post-lock census content hash changed")
    return {
        "status": "PASS",
        "census_status": expected_status,
        "gates_not_pass": observed,
    }


def _unique_stage(window: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        stage for stage in window.get("stages", [])
        if stage.get("name") == name
    ]
    if len(matches) != 1:
        raise PostlockRefusal(
            f"manager result has {len(matches)} stages named {name!r}"
        )
    return matches[0]


def _verified_artifact(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise PostlockRefusal(f"W2 {role} artifact path/bytes/hash changed")
    return path


def analyze_w2(
    *,
    w2_result_path: Path,
    release_path: Path,
    output_path: Path,
    export_root: Path,
    exporter: Callable[..., dict[str, Any]] | None = None,
    pair_post_path: Path | None = None,
    w3_result_path: Path | None = None,
    w1_result_path: Path | None = None,
    session_proof_path: Path | None = None,
) -> dict[str, Any]:
    """Validate release/capture provenance, export, and write one census."""

    analysis_started_ns = time.monotonic_ns()
    analysis_started_utc = datetime.now(timezone.utc).isoformat()
    assert_accelerator_free()
    assert_lock_environment_absent()
    release = validate_lock_release_proof(
        release_path=release_path,
        window_result_path=w2_result_path,
        analysis_started_monotonic_ns=analysis_started_ns,
        expected_window="W2",
    )
    w2 = _load_json(w2_result_path, what="W2 manager result")
    if (
        w2.get("schema") != "wrf_gpu2.v025.m0.manager_window.v1"
        or w2.get("status") != "OK"
        or w2.get("window") != "W2"
        or w2.get("run_id") != release.get("run_id")
    ):
        raise PostlockRefusal("W2 manager result schema/status/run is ineligible")
    integrity = _unique_stage(w2, "profiled_artifact_capture_integrity")
    if (
        integrity.get("status") != "PASS"
        or integrity.get("postlock_export_and_census_status") != "MISSING"
    ):
        raise PostlockRefusal("W2 capture did not stop at the frozen MISSING gate")
    artifacts = integrity.get("artifacts") or {}
    required_roles = {
        "exact_boundary",
        "source_rep",
        "allocator",
        "lock_owner_total_residency",
    }
    if set(artifacts) != required_roles:
        raise PostlockRefusal("W2 capture artifact inventory changed")
    paths = {
        role: _verified_artifact(artifacts[role], role=role)
        for role in sorted(required_roles)
    }
    exact_child = _load_json(paths["exact_boundary"], what="W2 exact child")
    if (
        exact_child.get("status") != "OK"
        or exact_child.get("run_id") != w2["run_id"]
        or (exact_child.get("instrumentation") or {}).get("mode") != "profiled"
        or (exact_child.get("instrumentation") or {}).get("nvtx_range")
        != RANGE_NAME
    ):
        raise PostlockRefusal("W2 exact child identity/instrumentation changed")
    residency = _load_json(
        paths["lock_owner_total_residency"], what="W2 residency"
    )
    matched_profiler = None
    clean_exact_child = None
    cold_readiness_seconds = None
    session_inputs: dict[str, Any] | None = None
    if release.get("schema") == SESSION_RELEASE_SCHEMA:
        required_paths = {
            "pair_post": pair_post_path,
            "W3 result": w3_result_path,
            "W1 result": w1_result_path,
            "session proof": session_proof_path,
        }
        missing = sorted(
            name for name, path in required_paths.items() if path is None
        )
        if missing:
            raise PostlockRefusal(
                "session W2 census lacks post-W3 bindings: " + ", ".join(missing)
            )
        pair_post = _load_json(Path(pair_post_path), what="W3 matched-pair post")
        expected_pair_hash = canonical_sha256(
            {key: value for key, value in pair_post.items()
             if key != "pair_post_sha256"}
        )
        if (
            pair_post.get("schema")
            != "wrf_gpu2.v025.m0.pair_post_analysis.v1"
            or pair_post.get("status")
            != "PROFILED_CLEAN_IDENTITY_AND_PERTURBATION_CONFIRMED"
            or pair_post.get("pair_post_sha256") != expected_pair_hash
        ):
            raise PostlockRefusal("W3 matched-pair post is incomplete or stale")
        w3_path = Path(w3_result_path)
        w1_path = Path(w1_result_path)
        w3 = _load_json(w3_path, what="W3 manager result")
        w1 = _load_json(w1_path, what="W1 manager result")
        for window, path, payload in (
            ("W2", Path(w2_result_path), w2),
            ("W3", w3_path, w3),
        ):
            binding = (pair_post.get("window_bindings") or {}).get(window) or {}
            if (
                binding.get("run_id") != payload.get("run_id")
                or binding.get("path") != str(path.resolve())
                or binding.get("sha256") != sha256_file(path)
            ):
                raise PostlockRefusal(
                    f"W3 matched-pair post does not bind this {window} result"
                )
        clean_stage = _unique_stage(
            w3, "clean_cached_readiness_and_integration"
        )
        clean_path = _verified_artifact(
            {
                "path": clean_stage.get("result_path"),
                "bytes": Path(str(clean_stage.get("result_path", ""))).stat().st_size
                if Path(str(clean_stage.get("result_path", ""))).is_file()
                else None,
                "sha256": clean_stage.get("result_sha256"),
            },
            role="W3 clean exact-boundary",
        )
        clean_exact_child = _load_json(clean_path, what="W3 clean exact child")
        selected = (w1.get("session") or {}).get("selected_cold_stage")
        cold_stage = _unique_stage(w1, str(selected))
        cold_child = _load_json(
            _verified_artifact(
                {
                    "path": cold_stage.get("result_path"),
                    "bytes": Path(
                        str(cold_stage.get("result_path", ""))
                    ).stat().st_size
                    if Path(
                        str(cold_stage.get("result_path", ""))
                    ).is_file()
                    else None,
                    "sha256": cold_stage.get("result_sha256"),
                },
                role="W1 selected cold exact-boundary",
            ),
            what="W1 selected cold exact child",
        )
        cold_readiness_seconds = (cold_child.get("timing") or {}).get(
            "readiness_seconds"
        )
        matched_profiler = pair_post.get("matched_pair")
        session_path = Path(session_proof_path)
        if (
            str(session_path.resolve()) != release.get("session_proof_path")
            or sha256_file(session_path) != release.get("session_proof_sha256")
            or (pair_post.get("session_root") or {}).get("sha256")
            != release.get("session_proof_sha256")
        ):
            raise PostlockRefusal(
                "W2 census/pair post does not bind the released session root"
            )
        session_inputs = {
            "session_proof": {
                "path": str(session_path.resolve()),
                "sha256": sha256_file(session_path),
            },
            "w1_result": {
                "path": str(w1_path.resolve()),
                "sha256": sha256_file(w1_path),
                "run_id": w1.get("run_id"),
                "selected_cold_stage": selected,
            },
            "w3_result": {
                "path": str(w3_path.resolve()),
                "sha256": sha256_file(w3_path),
                "run_id": w3.get("run_id"),
            },
            "pair_post": {
                "path": str(Path(pair_post_path).resolve()),
                "sha256": sha256_file(Path(pair_post_path)),
                "content_sha256": pair_post["pair_post_sha256"],
            },
        }

    if exporter is None:
        import nsys_export
        exporter = nsys_export.export_all
    export = exporter(
        paths["source_rep"],
        export_root,
        run_id=w2["run_id"],
        expected_source_sha256=artifacts["source_rep"]["sha256"],
    )
    nvtx_rows = None
    cuda_rows = None
    kernel_summary = None
    parse_errors: dict[str, str] = {}
    try:
        import nvtx_exclusive
        path = Path(export["exports"]["nvtx_pushpop_trace"]["path"])
        nvtx_rows = nvtx_exclusive.parse_pushpop_trace(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if not nvtx_rows:
            raise ValueError("NVTX push/pop table parsed to zero ranges")
    except Exception as exc:  # noqa: BLE001 - becomes an explicit missing gate
        parse_errors["nvtx_pushpop_trace"] = f"{type(exc).__name__}: {exc}"
    try:
        import parse_profiler
        path = Path(export["exports"]["cuda_gpu_trace"]["path"])
        cuda_rows = parse_profiler.parse_cuda_gpu_trace(
            path.read_text(encoding="utf-8", errors="replace")
        )
    except Exception as exc:  # noqa: BLE001
        parse_errors["cuda_gpu_trace"] = f"{type(exc).__name__}: {exc}"
    try:
        import parse_profiler
        path = Path(export["exports"]["cuda_gpu_kern_sum"]["path"])
        kernel_summary = parse_profiler.parse_nsys_kernel_summary(
            path.read_text(encoding="utf-8", errors="replace")
        )
    except Exception as exc:  # noqa: BLE001
        parse_errors["cuda_gpu_kern_sum"] = f"{type(exc).__name__}: {exc}"

    release_binding = {
        "schema": release.get("schema"),
        "run_id": release["run_id"],
        "release_path": str(Path(release_path).resolve()),
        "release_sha256": sha256_file(release_path),
        "window_result_sha256": release["window_result_sha256"],
        "wrapper_returned_monotonic_ns":
            release["wrapper_returned_monotonic_ns"],
        "analysis_started_monotonic_ns": analysis_started_ns,
        "release_before_analysis": True,
        "session_proof_sha256": release.get("session_proof_sha256"),
    }
    census = build_census(
        export=export,
        nvtx_rows=nvtx_rows,
        cuda_rows=cuda_rows,
        kernel_summary=kernel_summary,
        run_id=w2["run_id"],
        exact_child=exact_child,
        release_binding=release_binding,
        production_derived=True,
        residency=residency,
        host_rss=(
            _load_json(
                Path(str(
                    _unique_stage(
                        w2, "profiled_cached_readiness_and_integration"
                    ).get("host_rss_path", "")
                )),
                what="W2 host RSS",
            )
            if _unique_stage(
                w2, "profiled_cached_readiness_and_integration"
            ).get("host_rss_path")
            else None
        ),
        matched_profiler=matched_profiler,
        clean_exact_child=clean_exact_child,
        cold_readiness_seconds=cold_readiness_seconds,
    )
    census["session_inputs"] = session_inputs
    census["analysis_process"] = {
        "pid": os.getpid(),
        "started_monotonic_ns": analysis_started_ns,
        "started_at_utc": analysis_started_utc,
        "finished_monotonic_ns": time.monotonic_ns(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "accelerator_modules_imported": _loaded_accelerator_modules(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "GPUWRF_JAX_CACHE",
                "GPUWRF_JAX_CACHE_DIR",
                "GPUWRF_JAX_CACHE_LOCK",
            )
        },
        "lock_environment_present": {
            key: key in os.environ for key in LOCK_ENVIRONMENT
        },
        "device_action": False,
    }
    if any(census["analysis_process"]["lock_environment_present"].values()):
        raise PostlockRefusal("post-lock lock-environment invariant changed")
    if census["analysis_process"]["accelerator_modules_imported"]:
        raise PostlockRefusal("post-lock analysis imported accelerator modules")
    census["parse_errors"] = parse_errors
    census["census_sha256"] = canonical_sha256(
        {key: value for key, value in census.items() if key != "census_sha256"}
    )
    validate_census(census)
    _atomic_json_no_replace(output_path, census)
    return census


# --------------------------------------------------------------------------- #
# M0-CORE finalizer: the one JAX-free verdict after W3                          #
# --------------------------------------------------------------------------- #
M0_CORE_SCHEMA = "wrf_gpu2.v025.m0.m0_core_terminal_manifest.v1"

#: Amendment 5's M0-CORE items 1-5 expressed as the census/session gate subset
#: that actually blocks M1. Top-level census `OK` is deliberately NOT the
#: criterion: `production_representativeness` is an Amendment-5 M1/M2 diagnostic,
#: so requiring it here would make a perfect W1-W3 session logically unable to
#: reach an M0-CORE verdict.
M0_CORE_CENSUS_GATES = (
    "export",
    "source_binding",
    "cuda_trace_row_integrity",
    "integration_scope",
    "step_classes",
    "taxonomy",
    "attribution",
    "summary_crosscheck",
    "transfer_audit",
    "vram",
    "host_ram",
    "profiler_perturbation",
    "clean_w3",
    "timing_completeness",
    "hlo_completeness",
)

#: Named, non-waived, and explicitly not part of the M0-CORE verdict.
M0_DEFERRED_FIELDS = {
    "instruction_counter_E_baseline_T_ceiling": "M2_ENTRY",
    "native_pallas_sm120": "M2_BACKEND_BAKEOFF_ENTRY",
    "compiler_peak_live_buffers":
        "M1_SAME_SET_CANDIDATE_DIAGNOSIS_OR_M2_BACKEND_BAKEOFF",
    "ncu_registers_occupancy_stalls_dram_l2": "M2",
    "two_domain_rho_family_coverage_trace": "M1_M2_PRODUCTION_DIAGNOSTIC",
    "multi_size_vram_scaling_fit": "RELEASE_SCALING_ACCEPTANCE",
}

#: Census gates that are real and required, but by a later milestone.
M0_DEFERRED_CENSUS_GATES = ("production_representativeness",)


def _rehash_window(window_id: str, result_path: Path) -> dict[str, Any]:
    """Re-read one manager-window artifact and re-hash every raw child record."""

    result = _load_json(result_path, what=f"{window_id} manager result")
    if (
        result.get("schema") != "wrf_gpu2.v025.m0.manager_window.v1"
        or result.get("status") != "OK"
        or result.get("window") != window_id
    ):
        raise PostlockRefusal(f"{window_id} manager result is ineligible")
    retained: dict[str, dict[str, Any]] = {}

    def retain(
        *,
        role: str,
        path_value: Any,
        expected_sha256: Any,
        expected_bytes: Any = None,
    ) -> dict[str, Any]:
        path = Path(str(path_value or ""))
        if path.is_symlink() or not path.is_file():
            raise PostlockRefusal(
                f"{window_id}/{role} is missing/not regular: {path}"
            )
        observed_bytes = path.stat().st_size
        observed_hash = sha256_file(path)
        if expected_sha256 != observed_hash or (
            expected_bytes is not None and expected_bytes != observed_bytes
        ):
            raise PostlockRefusal(
                f"{window_id}/{role} bytes/hash changed before final publication"
            )
        key = str(path.resolve())
        prior = retained.get(key)
        if prior is not None and (
            prior["sha256"] != observed_hash or prior["bytes"] != observed_bytes
        ):
            raise PostlockRefusal(
                f"{window_id}/{role} aliases a contradictory retained artifact"
            )
        record = {
            "path": key,
            "bytes": observed_bytes,
            "sha256": observed_hash,
        }
        retained[key] = record
        return record

    stages: list[dict[str, Any]] = []
    for stage in result.get("stages") or []:
        record = {"name": stage.get("name"), "status": stage.get("status")}
        raw_path = stage.get("result_path")
        if raw_path:
            artifact = retain(
                role=f"{stage.get('name')}/result",
                path_value=raw_path,
                expected_sha256=stage.get("result_sha256"),
            )
            path = Path(artifact["path"])
            child = _load_json(path, what=f"{window_id} child result")
            if (
                child.get("status") != "OK"
                or child.get("run_id") != result.get("run_id")
            ):
                raise PostlockRefusal(
                    f"{window_id}/{stage.get('name')} child status/run is invalid"
                )
            record.update(
                {
                    "result_path": artifact["path"],
                    "result_bytes": artifact["bytes"],
                    "result_sha256": artifact["sha256"],
                    "run_id": child.get("run_id"),
                    "exact_value_sha256": (child.get("result") or {}).get(
                        "exact_value_sha256"
                    ),
                }
            )
        for role, path_key, hash_key in (
            ("log", "log_path", "log_sha256"),
            ("host_rss", "host_rss_path", "host_rss_sha256"),
            ("final_wrfout", "final_wrfout_path", "final_wrfout_sha256"),
        ):
            if stage.get(path_key) is not None:
                record[role] = retain(
                    role=f"{stage.get('name')}/{role}",
                    path_value=stage[path_key],
                    expected_sha256=stage.get(hash_key),
                )
        artifact_records = stage.get("artifacts")
        if isinstance(artifact_records, dict):
            record["artifacts"] = {
                role: retain(
                    role=f"{stage.get('name')}/{role}",
                    path_value=artifact.get("path"),
                    expected_sha256=artifact.get("sha256"),
                    expected_bytes=artifact.get("bytes"),
                )
                for role, artifact in sorted(artifact_records.items())
                if isinstance(artifact, dict)
            }
        stages.append(record)
    return {
        "window": window_id,
        "run_id": result.get("run_id"),
        "result_path": str(Path(result_path).resolve()),
        "result_sha256": sha256_file(result_path),
        "stages": stages,
        "retained_raw_artifacts": [
            retained[path] for path in sorted(retained)
        ],
    }


def _rehash_declared_file(
    record: dict[str, Any],
    *,
    role: str,
    path_key: str = "path",
    hash_key: str = "sha256",
    bytes_key: str = "bytes",
) -> dict[str, Any]:
    path = Path(str(record.get(path_key, "")))
    if path.is_symlink() or not path.is_file():
        raise PostlockRefusal(f"{role} is missing/not regular: {path}")
    observed = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if (
        record.get(hash_key) != observed["sha256"]
        or (
            record.get(bytes_key) is not None
            and record.get(bytes_key) != observed["bytes"]
        )
    ):
        raise PostlockRefusal(f"{role} bytes/hash changed before final publication")
    return observed


def _rehash_census_raw(census: dict[str, Any]) -> dict[str, Any]:
    raw = census.get("raw_artifacts") or {}
    source = _rehash_declared_file(
        {
            "path": raw.get("source_rep"),
            "sha256": raw.get("source_rep_sha256"),
        },
        role="W2 source nsys report",
    )
    private = _rehash_declared_file(
        raw.get("private_sqlite") or {},
        role="W2 private SQLite export",
    )
    exports = {
        name: _rehash_declared_file(
            record, role=f"W2 profiler export {name}"
        )
        for name, record in sorted((raw.get("exports") or {}).items())
    }
    if (
        (raw.get("private_sqlite") or {}).get("forced_export_count") != 1
        or not exports
    ):
        raise PostlockRefusal("W2 raw profiler inventory is not one-export complete")
    return {
        "source_rep": source,
        "private_sqlite": private,
        "exports": exports,
    }


def _validate_session_chain(
    *,
    session_proof_path: Path,
    release_path: Path,
    release: dict[str, Any],
    windows: dict[str, dict[str, Any]],
    census_path: Path,
    census: dict[str, Any],
    fast_pair_path: Path,
    pair: dict[str, Any],
    qualification_path: Path,
    qualification: dict[str, Any],
    pair_post_path: Path,
    pair_post: dict[str, Any],
) -> dict[str, Any]:
    """Re-derive every Amendment-6 run/root direction before publication."""

    import m0_core_session_protocol as core_session
    import m0_vram_sampler as mvs

    session_path = Path(session_proof_path)
    session = _load_json(session_path, what="held-session root")
    expected_session_content = canonical_sha256(
        {key: value for key, value in session.items()
         if key != "session_sha256"}
    )
    expected_order = list(core_session.POST_COLD_STAGE_ORDER)
    if (
        session.get("schema") != "wrf_gpu2.v025.m0.core_session_execution.v1"
        or session.get("status") != "PASS"
        or session.get("label") != core_session.SESSION_LABEL
        or session.get("session_sha256") != expected_session_content
        or (session.get("graph") or {}).get("status") != "PASS"
        or (session.get("graph") or {}).get("stage_order_executed")
        != expected_order
        or session.get("lock_acquisitions") != 1
        or session.get("retry_path") is not None
        or session.get("queue_path") is not None
        or session.get("receipt_refund_path") is not None
        or session.get("orphan_sweep_completed") is not True
        or (session.get("process_group_proof") or {}).get("status") != "PASS"
    ):
        raise PostlockRefusal("held-session root is not a terminal green graph")
    budget = core_session.validate_held_budget()
    if session.get("held_budget") != budget:
        raise PostlockRefusal("held-session root does not bind the 4220/4500 budget")

    authorization = session.get("authorization") or {}
    spend = authorization.get("spend_record") or {}
    lock = authorization.get("canonical_lock") or {}
    fingerprint = authorization.get("receipt_fingerprint")
    if (
        not SHA256_RE.fullmatch(str(fingerprint or ""))
        or authorization.get("label") != core_session.SESSION_LABEL
        or authorization.get("lock_acquisitions") != 1
        or authorization.get("spent_after_cpu_preflight") is not True
        or authorization.get("spent_after_input_revalidation") is not True
        or authorization.get("spent_before_any_device_child") is not True
        or spend.get("window") != core_session.SESSION_LABEL
        or spend.get("managers") != list(core_session.REQUIRED_MANAGERS)
        or lock.get("label_matched") is not True
        or lock.get("exported_label") != core_session.SESSION_LABEL
    ):
        raise PostlockRefusal("session receipt/spend/exact-lock binding is invalid")

    session_sha = sha256_file(session_path)
    if (
        release.get("schema") != SESSION_RELEASE_SCHEMA
        or release.get("session_proof_path") != str(session_path.resolve())
        or release.get("session_proof_sha256") != session_sha
        or release.get("session_status") != "PASS"
    ):
        raise PostlockRefusal("release proof does not point back to this session root")

    detail_names = {
        "W1": "W1_CACHED_AND_CORRECTNESS",
        "W2": "W2_PROFILED_CAPTURE",
        "W3": "W3_CLEAN_MATCHED_ARM",
    }
    details = session.get("stage_details") or {}
    for window, detail_name in detail_names.items():
        detail = details.get(detail_name) or {}
        release_window = (release.get("windows") or {}).get(window) or {}
        current = windows[window]
        if (
            detail.get("run_id") != current["run_id"]
            or str(Path(str(detail.get("result_path", ""))).resolve())
            != current["result_path"]
            or detail.get("result_sha256") != current["result_sha256"]
            or release_window.get("run_id") != current["run_id"]
            or release_window.get("path") != current["result_path"]
            or release_window.get("sha256") != current["result_sha256"]
        ):
            raise PostlockRefusal(
                f"session/release/finalizer disagree on {window}"
            )
        manager = _load_json(
            Path(current["result_path"]), what=f"{window} manager result"
        )
        manager_auth = manager.get("authorization") or {}
        if (
            manager_auth.get("receipt_fingerprint") != fingerprint
            or manager_auth.get("spend_record") != spend
            or (manager_auth.get("canonical_lock") or {}).get("label_matched")
            is not True
        ):
            raise PostlockRefusal(
                f"{window} does not point back to the session spend/lock"
            )

    c1 = details.get(core_session.C1_STAGE) or {}
    c1_files = (
        ("fast_pair", fast_pair_path, "fast_pair_path", "fast_pair_sha256"),
        (
            "qualification",
            qualification_path,
            "qualification_path",
            "qualification_sha256",
        ),
    )
    for role, path, path_field, hash_field in c1_files:
        if (
            str(Path(str(c1.get(path_field, ""))).resolve())
            != str(Path(path).resolve())
            or c1.get(hash_field) != sha256_file(path)
        ):
            raise PostlockRefusal(f"C1 does not bind the final {role}")
    prepared_path = Path(str(c1.get("prepared_pair_path", "")))
    prepared = _load_json(prepared_path, what="C1 prepared pair plan")
    if c1.get("prepared_pair_sha256") != sha256_file(prepared_path):
        raise PostlockRefusal("C1 prepared-pair plan changed after W1")

    w1_session = _load_json(
        Path(windows["W1"]["result_path"]), what="W1 manager result"
    ).get("session") or {}
    selected = w1_session.get("selected_cold_stage")
    selected_index = w1_session.get("qualified_attempt_index")
    if (
        selected != f"cold_empty_cache_readiness_{selected_index}"
        or c1.get("qualification", {}).get("selected_cold_stage") != selected
        or qualification.get("selected_cold_stage") != selected
        or qualification.get("selected_cold_attempt_index") != selected_index
        or (prepared.get("prepared_cache") or {}).get("qualification", {}).get(
            "selected_cold_stage"
        )
        != selected
    ):
        raise PostlockRefusal("selected numbered W1 cold attempt is not canonical")

    if (
        qualification.get("fast_pair_path") != str(fast_pair_path)
        or qualification.get("fast_pair_sha256") != sha256_file(fast_pair_path)
        or (prepared.get("prepared_cache") or {}).get(
            "qualification_manifest_sha256"
        )
        != sha256_file(qualification_path)
    ):
        raise PostlockRefusal("pair/qualification/prepared-pair direction is broken")

    pair_provenance = pair.get("lock_release_provenance") or {}
    identity_record = session.get("session_identity") or {}
    cpu_record = session.get("cpu_preflight") or {}
    if (
        pair_provenance.get("cpu_preflight_sha256")
        != cpu_record.get("sha256")
        or pair_provenance.get("cpu_preflight_content_sha256")
        != cpu_record.get("content_sha256")
        or pair_provenance.get("session_identity_sha256")
        != identity_record.get("sha256")
        or pair_provenance.get("session_identity_content_sha256")
        != identity_record.get("content_sha256")
        or (pair.get("c1_subgate") or {}).get("both_green") is not True
    ):
        raise PostlockRefusal("C1 pair does not point back to CPU/session preflight")

    identity_path = Path(str(identity_record.get("path", "")))
    identity = _load_json(identity_path, what="session preflight identity")
    if (
        identity_record.get("sha256") != sha256_file(identity_path)
        or identity_record.get("content_sha256")
        != identity.get("identity_sha256")
        or identity.get("identity_sha256")
        != canonical_sha256(
            {key: value for key, value in identity.items()
             if key != "identity_sha256"}
        )
        or identity.get("src_gpuwrf_tree")
        != "a6885ceded260df2f5777d7366d75a5d38947cb7"
    ):
        raise PostlockRefusal("session preflight identity changed")
    for role, tool in (identity.get("tools") or {}).items():
        try:
            invoked_resolution = Path(
                str(tool.get("invoked_path", ""))
            ).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PostlockRefusal(
                f"session tool {role} invocation path is unavailable: {exc}"
            ) from exc
        if str(invoked_resolution) != tool.get("path"):
            raise PostlockRefusal(
                f"session tool {role} invocation path was retargeted"
            )
        _rehash_declared_file(tool, role=f"session tool {role}")
    _rehash_declared_file(
        identity.get("config") or {}, role="session namelist"
    )
    run_dir = Path(str((identity.get("config") or {}).get("path", ""))).parent
    if mvs.input_manifest_sha256(run_dir) != identity.get(
        "input_manifest_sha256"
    ):
        raise PostlockRefusal("session input manifest changed")

    revalidation_record = session.get("prelock_revalidation") or {}
    revalidation_path = Path(str(revalidation_record.get("path", "")))
    revalidation = _load_json(
        revalidation_path, what="pre-lock input revalidation"
    )
    if (
        revalidation_record.get("sha256")
        != sha256_file(revalidation_path)
        or authorization.get("prelock_revalidation_path")
        != str(revalidation_path.resolve())
        or authorization.get("prelock_revalidation_sha256")
        != revalidation_record.get("sha256")
        or authorization.get("prelock_revalidation_content_sha256")
        != revalidation_record.get("content_sha256")
        or revalidation.get("schema")
        != "wrf_gpu2.v025.m0.session_prelock_revalidation.v1"
        or revalidation.get("status") != "PASS"
        or revalidation.get("session_identity_content_sha256")
        != identity_record.get("content_sha256")
        or revalidation.get("revalidation_sha256")
        != revalidation_record.get("content_sha256")
        or revalidation.get("revalidation_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in revalidation.items()
                if key != "revalidation_sha256"
            }
        )
        or revalidation.get("device_action") is not False
    ):
        raise PostlockRefusal(
            "pre-lock input revalidation changed or is not session-bound"
        )

    cpu_path = Path(str(cpu_record.get("path", "")))
    cpu = _load_json(cpu_path, what="CPU preflight")
    if (
        cpu_record.get("sha256") != sha256_file(cpu_path)
        or cpu_record.get("content_sha256") != cpu.get("preflight_sha256")
        or cpu.get("preflight_sha256")
        != canonical_sha256(
            {key: value for key, value in cpu.items()
             if key != "preflight_sha256"}
        )
        or cpu.get("session_identity_sha256") != identity_record.get("sha256")
    ):
        raise PostlockRefusal("CPU preflight changed after receipt spend")
    cpu_output = _rehash_declared_file(
        cpu.get("verified_output") or {},
        role="fresh CPU-WRF output",
        path_key="final_wrfout_path",
        hash_key="final_wrfout_sha256",
        bytes_key="final_wrfout_bytes",
    )

    pair_cpu = ((pair.get("pair") or {}).get("cpu_arm") or {})
    pair_gpu = ((pair.get("pair") or {}).get("gpu_arm") or {})
    for role, arm in (("C1 CPU arm", pair_cpu), ("C1 GPU arm", pair_gpu)):
        _rehash_declared_file(
            arm,
            role=role,
            path_key="final_wrfout_path",
            hash_key="final_wrfout_sha256",
            bytes_key="final_wrfout_bytes",
        )
    if pair_cpu.get("final_wrfout_sha256") != cpu_output["sha256"]:
        raise PostlockRefusal("C1 CPU arm does not bind the preflight output")

    pair_post_expected = canonical_sha256(
        {key: value for key, value in pair_post.items()
         if key != "pair_post_sha256"}
    )
    if (
        pair_post.get("pair_post_sha256") != pair_post_expected
        or (pair_post.get("session_root") or {}).get("sha256") != session_sha
        or [
            ((pair_post.get("window_bindings") or {}).get(name) or {}).get(
                "run_id"
            )
            for name in ("W2", "W3")
        ]
        != [windows["W2"]["run_id"], windows["W3"]["run_id"]]
    ):
        raise PostlockRefusal("matched-pair post does not bind session/W2/W3")

    census_inputs = census.get("session_inputs") or {}
    if (
        (census_inputs.get("session_proof") or {}).get("sha256") != session_sha
        or (census_inputs.get("pair_post") or {}).get("sha256")
        != sha256_file(pair_post_path)
        or (census_inputs.get("w1_result") or {}).get("sha256")
        != windows["W1"]["result_sha256"]
        or (census_inputs.get("w3_result") or {}).get("sha256")
        != windows["W3"]["result_sha256"]
        or census.get("run_id") != windows["W2"]["run_id"]
    ):
        raise PostlockRefusal("census does not point back to session/W1/W2/W3")

    prepared_cache = prepared.get("prepared_cache") or {}
    for arm_name in ("profiled", "clean"):
        arm = prepared.get(arm_name) or {}
        identity_file = _rehash_declared_file(
            {
                "path": arm.get("identity_path"),
                "sha256": arm.get("identity_sha256"),
            },
            role=f"prepared {arm_name} identity",
        )
        cache_identity = mvs.directory_tree_identity(
            Path(str(arm.get("prepared_cache_path", "")))
        )
        if (
            cache_identity.get("sha256") != arm.get("prepared_cache_sha256")
            or identity_file["sha256"] != arm.get("identity_sha256")
        ):
            raise PostlockRefusal(
                f"prepared {arm_name} identity/cache changed"
            )
    if (
        (prepared.get("held_session_execution") or {}).get(
            "canonical_lock_acquisitions"
        )
        != 1
        or (prepared.get("held_session_execution") or {}).get(
            "second_wrapper_reachable"
        )
        is not False
        or prepared_cache.get("qualification_manifest_sha256")
        != sha256_file(qualification_path)
    ):
        raise PostlockRefusal("prepared pair is not the one-held-session plan")

    return {
        "session": {
            "path": str(session_path.resolve()),
            "bytes": session_path.stat().st_size,
            "sha256": session_sha,
            "content_sha256": session["session_sha256"],
        },
        "receipt_fingerprint": fingerprint,
        "spend_record": spend,
        "selected_cold_stage": selected,
        "selected_cold_attempt_index": selected_index,
        "cpu_preflight": {
            "path": str(cpu_path.resolve()),
            "sha256": sha256_file(cpu_path),
            "output": cpu_output,
        },
        "prelock_revalidation": {
            "path": str(revalidation_path.resolve()),
            "sha256": sha256_file(revalidation_path),
            "content_sha256": revalidation["revalidation_sha256"],
        },
        "prepared_pair": {
            "path": str(prepared_path.resolve()),
            "sha256": sha256_file(prepared_path),
            "arm_order": prepared.get("frozen_arm_order"),
        },
        "bidirectional_bindings": "PASS",
    }


def finalize_m0_core(
    *,
    w1_result_path: Path,
    w2_result_path: Path,
    w3_result_path: Path,
    census_path: Path,
    release_path: Path,
    fast_pair_path: Path,
    qualification_path: Path,
    pair_post_path: Path,
    session_proof_path: Path,
    output_path: Path,
    analysis_started_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Publish the one terminal M0-CORE manifest after W3.

    The machine authority is ``cpu_closure_gate`` plus ``milestone_partition``;
    the descriptive ``status`` string is never a gate.  A green M0-CORE verdict
    does not promote the full census or the release bar, and deferred evidence
    stays ``MISSING`` with a named milestone and ``waived: false``.
    """

    started_ns = (
        int(analysis_started_monotonic_ns)
        if analysis_started_monotonic_ns is not None
        else time.monotonic_ns()
    )
    assert_accelerator_free()
    assert_lock_environment_absent()
    if Path(output_path).parent.resolve() != Path(session_proof_path).parent.resolve():
        raise PostlockRefusal(
            "terminal M0-CORE manifest must live in the held-session proof root"
        )
    release = validate_lock_release_proof(
        release_path=release_path,
        window_result_path=w3_result_path,
        analysis_started_monotonic_ns=started_ns,
        expected_window="W3",
    )
    windows = {
        "W1": _rehash_window("W1", Path(w1_result_path)),
        "W2": _rehash_window("W2", Path(w2_result_path)),
        "W3": _rehash_window("W3", Path(w3_result_path)),
    }
    if len({record["run_id"] for record in windows.values()}) != 3:
        raise PostlockRefusal("the three windows do not carry three run IDs")

    census = _load_json(Path(census_path), what="post-lock census")
    validate_census(census)
    required = census.get("required_gates") or {}
    partitioned = set(M0_CORE_CENSUS_GATES) | set(M0_DEFERRED_CENSUS_GATES)
    if partitioned != set(required):
        raise PostlockRefusal(
            "census gate inventory does not partition exactly into the "
            "M0-CORE subset and the named deferred gates: "
            f"{sorted(set(required) ^ partitioned)}"
        )
    core_gate_status = {
        name: (required.get(name) or {}).get("status", "MISSING")
        for name in M0_CORE_CENSUS_GATES
    }
    core_not_pass = sorted(
        name for name, status in core_gate_status.items() if status != "PASS"
    )

    pair = _load_json(Path(fast_pair_path), what="C1 fresh FAST pair")
    qualification = _load_json(
        Path(qualification_path), what="W1 qualification manifest"
    )
    pair_post = _load_json(Path(pair_post_path), what="W3 matched-pair analysis")
    c1_subgate = pair.get("c1_subgate") or {}
    c1_status = (
        "PASS"
        if (
            pair.get("schema") == "wrf_gpu2.v025.m0.fast_case_qualification.v1"
            and pair.get("status") in {"PASS", "FAST_QUALIFIED"}
            and (pair.get("pair_completeness") or {}).get("completeness_percent")
            == 100
            and c1_subgate.get("both_green") is True
            and qualification.get("status") == "AUTOTUNE0_QUALIFIED"
            and qualification.get("run_id") == windows["W1"]["run_id"]
            and (qualification.get("w1_exact_boundary") or {}).get(
                "gpu_wrfout_sha256"
            )
            == (pair.get("w1_exact_boundary") or {}).get("gpu_wrfout_sha256")
        )
        else "BLOCKED"
    )
    if (
        pair_post.get("status")
        != "PROFILED_CLEAN_IDENTITY_AND_PERTURBATION_CONFIRMED"
    ):
        raise PostlockRefusal("W3 matched-pair analysis is not confirmed")
    # The matched pair must be the pair this session actually produced: bind its
    # digest to the re-hashed W3 clean child, not to itself.
    clean_digests = {
        stage.get("exact_value_sha256")
        for stage in windows["W3"]["stages"]
        if stage.get("name") == "clean_cached_readiness_and_integration"
    }
    if clean_digests != {pair_post.get("result_sha256")}:
        raise PostlockRefusal(
            "matched-pair analysis is not bound to this session's W3 clean result"
        )
    session_chain = _validate_session_chain(
        session_proof_path=Path(session_proof_path),
        release_path=Path(release_path),
        release=release,
        windows=windows,
        census_path=Path(census_path),
        census=census,
        fast_pair_path=Path(fast_pair_path),
        pair=pair,
        qualification_path=Path(qualification_path),
        qualification=qualification,
        pair_post_path=Path(pair_post_path),
        pair_post=pair_post,
    )
    raw_artifacts = _rehash_census_raw(census)
    analysis_process = census.get("analysis_process") or {}
    c2_status = (
        "PASS"
        if (
            census.get("schema") == SCHEMA
            and analysis_process.get("device_action") is False
            and analysis_process.get("accelerator_modules_imported") == []
            and not any(
                (analysis_process.get("lock_environment_present") or {}).values()
            )
        )
        else "BLOCKED"
    )
    window_exact_children = [
        stage
        for window in windows.values()
        for stage in window["stages"]
        if stage.get("result_path")
    ]
    exact_boundary_status = (
        "PASS"
        if window_exact_children
        and all(
            stage.get("run_id") == window["run_id"]
            for window in windows.values()
            for stage in window["stages"]
            if stage.get("result_path")
        )
        else "BLOCKED"
    )

    core_fields = {
        "accepted_exact_cpu_boundary": exact_boundary_status,
        "c1_same_result_wrfout_runtime": c1_status,
        "c1_fresh_cpu_wrf_comparator": c1_status,
        "c2_jax_free_postlock_cpu_path": c2_status,
        "frozen_command_source_input_cache_tool_manifest": (
            "PASS"
            if session_chain.get("bidirectional_bindings") == "PASS"
            else "BLOCKED"
        ),
        "one_session_w1_w2_w3": (
            "PASS"
            if release.get("acquire_count") == release.get("release_count") == 1
            else "BLOCKED"
        ),
        "fast_correctness_and_resource_gates": (
            "PASS" if not core_not_pass else "BLOCKED"
        ),
        "different_model_review": "MISSING_MANAGER_OWNED",
    }
    blocking = sorted(
        name
        for name, status in core_fields.items()
        if status != "PASS" and name != "different_model_review"
    )
    core_status = "PASS" if not blocking and not core_not_pass else "BLOCKED"
    payload = {
        "schema": M0_CORE_SCHEMA,
        # Descriptive only. Automation must key on cpu_closure_gate and
        # milestone_partition, never on this string.
        "status": (
            "M0_CORE_EVIDENCE_COMPLETE_PENDING_INDEPENDENT_REVIEW"
            if core_status == "PASS"
            else "M0_CORE_BLOCKED"
        ),
        "status_semantics": (
            "descriptive label; the machine authority is cpu_closure_gate plus "
            "milestone_partition"
        ),
        "machine_authority": ["cpu_closure_gate", "milestone_partition"],
        "cpu_closure_gate": core_status,
        "m0_core_gate": core_status,
        "m0_core_blocking_fields": blocking,
        "m0_core_blocking_census_gates": core_not_pass,
        "may_open_m1": False,
        "may_open_m1_requires": (
            "M0-CORE green AND one different-model blocker/major review with "
            "zero blockers and zero majors; this manifest never grants it"
        ),
        "milestone_partition": {
            "M0_CORE": {
                "status": core_status,
                "required_before": "M1_MECHANISM_WORK",
                "census_gate_subset": list(M0_CORE_CENSUS_GATES),
                "census_gate_status": core_gate_status,
                "fields": core_fields,
                "may_open_m1": False,
            },
            "M0_DEFERRED": {
                "status": "MISSING_BY_NAMED_FUTURE_MILESTONE",
                "waived": False,
                "fields": dict(M0_DEFERRED_FIELDS),
                "census_gates": {
                    name: (required.get(name) or {}).get("status", "MISSING")
                    for name in M0_DEFERRED_CENSUS_GATES
                },
                "governed_by_m0_core_verdict": False,
            },
        },
        "full_census_status": census.get("status"),
        "full_census_gates_not_pass": census.get("gates_not_pass"),
        "release_gate": "MISSING",
        "release_gate_reason": (
            "a green M0-CORE verdict is milestone-scoped; it does not promote "
            "the full census, the multi-size VRAM scaling fit, or the release "
            "gate to OK"
        ),
        "windows": windows,
        "session_root": session_chain,
        "release_provenance": {
            "path": str(Path(release_path).resolve()),
            "sha256": sha256_file(release_path),
            "wrapper_returned_monotonic_ns": release[
                "wrapper_returned_monotonic_ns"
            ],
            "finalizer_started_monotonic_ns": started_ns,
            "release_before_finalizer": True,
        },
        "artifacts": {
            "held_session_root": session_chain["session"],
            "session_lock_release": {
                "path": str(Path(release_path).resolve()),
                "sha256": sha256_file(release_path),
                "content_sha256": release.get("release_sha256"),
            },
            "census": {
                "path": str(Path(census_path).resolve()),
                "sha256": sha256_file(census_path),
                "census_sha256": census.get("census_sha256"),
            },
            "fast_pair": {
                "path": str(Path(fast_pair_path).resolve()),
                "sha256": sha256_file(fast_pair_path),
            },
            "qualification": {
                "path": str(Path(qualification_path).resolve()),
                "sha256": sha256_file(qualification_path),
            },
            "matched_pair_post": {
                "path": str(Path(pair_post_path).resolve()),
                "sha256": sha256_file(pair_post_path),
            },
            "prepared_pair": session_chain["prepared_pair"],
            "retained_raw": raw_artifacts,
        },
        "jax_imported": False,
        "device_action": False,
    }
    payload["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    validate_m0_core_manifest(payload)
    _atomic_json_no_replace(Path(output_path), payload)
    return payload


def validate_m0_core_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Refuse a promoted deferred field, a stale hash, or a string-keyed gate."""

    if payload.get("schema") != M0_CORE_SCHEMA:
        raise PostlockRefusal("M0-CORE manifest schema changed")
    if payload.get("machine_authority") != ["cpu_closure_gate", "milestone_partition"]:
        raise PostlockRefusal("M0-CORE manifest machine authority changed")
    partition = payload.get("milestone_partition") or {}
    core = partition.get("M0_CORE") or {}
    deferred = partition.get("M0_DEFERRED") or {}
    if core.get("census_gate_subset") != list(M0_CORE_CENSUS_GATES):
        raise PostlockRefusal("M0-CORE census gate subset changed")
    if (
        deferred.get("waived") is not False
        or deferred.get("status") != "MISSING_BY_NAMED_FUTURE_MILESTONE"
        or deferred.get("fields") != dict(M0_DEFERRED_FIELDS)
        or deferred.get("governed_by_m0_core_verdict") is not False
    ):
        raise PostlockRefusal(
            "M0-DEFERRED evidence was waived, renamed, or promoted into the "
            "M0-CORE verdict"
        )
    if set(M0_CORE_CENSUS_GATES) & set(M0_DEFERRED_CENSUS_GATES):
        raise PostlockRefusal("a gate is both M0-CORE and deferred")
    observed = sorted(
        name
        for name, status in (core.get("census_gate_status") or {}).items()
        if status != "PASS"
    )
    if observed != sorted(payload.get("m0_core_blocking_census_gates") or []):
        raise PostlockRefusal("M0-CORE blocking-gate inventory is stale")
    expected_gate = (
        "PASS"
        if not observed and not (payload.get("m0_core_blocking_fields") or [])
        else "BLOCKED"
    )
    if (
        payload.get("cpu_closure_gate") != expected_gate
        or core.get("status") != expected_gate
    ):
        raise PostlockRefusal(
            "M0-CORE gate hides a blocking field or census gate"
        )
    if payload.get("may_open_m1") is not False:
        raise PostlockRefusal("no manifest may itself open M1")
    if payload.get("release_gate") != "MISSING":
        raise PostlockRefusal(
            "an M0-CORE verdict may not promote the release gate"
        )
    session_root = payload.get("session_root") or {}
    session_artifact = (
        (payload.get("artifacts") or {}).get("held_session_root") or {}
    )
    release_artifact = (
        (payload.get("artifacts") or {}).get("session_lock_release") or {}
    )
    if (
        session_root.get("bidirectional_bindings") != "PASS"
        or not SHA256_RE.fullmatch(
            str((session_root.get("session") or {}).get("sha256", ""))
        )
        or session_artifact != session_root.get("session")
        or not SHA256_RE.fullmatch(
            str(release_artifact.get("sha256", ""))
        )
        or payload.get("jax_imported") is not False
        or payload.get("device_action") is not False
    ):
        raise PostlockRefusal(
            "M0-CORE manifest lost its held-session/bidirectional root binding"
        )
    # The narrower verdict may never rewrite the wider one: if the full census
    # still lists a gate that is not PASS, its status stays BLOCKED regardless
    # of how green the M0-CORE subset is.
    full_not_pass = payload.get("full_census_gates_not_pass")
    if not isinstance(full_not_pass, list):
        raise PostlockRefusal("M0-CORE manifest lost the full-census inventory")
    if payload.get("full_census_status") != ("BLOCKED" if full_not_pass else "OK"):
        raise PostlockRefusal(
            "M0-CORE manifest reports a full-census status that contradicts its "
            "own missing-gate inventory"
        )
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    if payload.get("manifest_sha256") != expected_hash:
        raise PostlockRefusal("M0-CORE manifest content hash changed")
    return {
        "status": "PASS",
        "m0_core_gate": expected_gate,
        "blocking_census_gates": observed,
        "blocking_fields": sorted(payload.get("m0_core_blocking_fields") or []),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2-result", type=Path, required=True)
    parser.add_argument("--lock-release-proof", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = analyze_w2(
            w2_result_path=args.w2_result,
            release_path=args.lock_release_proof,
            output_path=args.output,
            export_root=args.export_root,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed terminal boundary
        refusal = {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "jax_imported": "jax" in sys.modules,
            "gpuwrf_imported": "gpuwrf" in sys.modules,
            "device_action": False,
        }
        print(json.dumps(refusal, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    # A successful reduction can honestly remain BLOCKED until W3; that is not
    # a process failure. The caller gates the payload status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
