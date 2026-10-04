#!/usr/bin/env python3
"""Bounded, content-addressed compiler buffer-assignment/live-range sidecar.

P9 recorded the exact hole this closes: the frozen A6 census carries aggregate
``temporary_bytes`` from JAX's public ``memory_analysis()``, and that number is
**not** peak live buffers.  It is the size of one preallocated arena after XLA
has already reused memory; relabelling it would overstate how much of the M0-CORE
HLO evidence exists.  So this module never derives peak from it.

Instead it reduces the compiler's own primary evidence, which this exact
toolchain emits next to the executable when ``--xla_dump_to`` is set:

* ``*-buffer-assignment.txt`` — one ``allocation N: size B, ...`` block per
  allocation with a ``value: <id name{index}> (size=B,offset=O): shape`` line per
  assigned HLO value;
* ``*-live-range.txt`` — ``HloLiveRange (max T):`` with an ``InstructionSequence``
  and a ``BufferLiveRange`` section giving every value a ``start-end`` interval
  on the flattened schedule.

Peak live bytes is then a real computation over primary evidence: sweep the
schedule and take the maximum total size of simultaneously live values.  Nothing
is estimated; if the join between the two files is not total and unambiguous, or
if any bound is exceeded, the sidecar refuses and the M0-CORE HLO gate stays
MISSING.

Three refusal classes, all fail-closed:

``BLOCKED_TOOLCHAIN_EVIDENCE``
    static inspection of the installed jaxlib cannot find a format discriminator
    this parser depends on; the exact missing discriminator is named;
``BLOCKED_SIDECAR_BOUNDS``
    the dump would exceed a file-count/size/line/value bound.  Truncating would
    understate the peak, so the collector refuses instead;
``BLOCKED_SIDECAR_EVIDENCE``
    the dump exists but is incomplete, ambiguous, or internally inconsistent.

This module imports neither JAX nor ``gpuwrf`` and opens no device.  The
toolchain check is a byte scan of an installed shared library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "wrf_gpu2.v025.m0.hlo_live_buffer_sidecar.v1"
TOOLCHAIN_SCHEMA = "wrf_gpu2.v025.m0.hlo_dump_toolchain_identity.v1"

#: Format discriminators emitted by the exact installed XLA.  Each one is a
#: literal this parser depends on; if the shipped library stops containing it,
#: the format assumption is no longer proven and the collector refuses.
TOOLCHAIN_DISCRIMINATORS: tuple[tuple[str, str], ...] = (
    ("buffer_assignment_dump_suffix", "-buffer-assignment.txt"),
    ("live_range_dump_suffix", "-live-range.txt"),
    ("allocation_line", "allocation %d: size %d"),
    ("assigned_value_line", " value: %s (size=%d,offset=%d): %s"),
    ("buffer_assignment_stats_header", "BufferAssignment stats:"),
    ("preallocated_temp_line", "     preallocated temp allocation: %10s"),
    ("total_allocation_line", "                 total allocation: %10s"),
    ("live_range_header", "HloLiveRange (max %d):"),
    ("live_range_sequence_header", "  InstructionSequence:"),
    ("live_range_buffer_header", "  BufferLiveRange:"),
    ("live_range_entry_line", "    %s%s:%d-%d"),
)

#: JAX excludes both dump flags from the persistent-compilation-cache key, so
#: enabling them on a cold compile cannot make a later cached stage miss.  These
#: two literals are re-verified statically before the flags may be used.
CACHE_KEY_EXCLUSION_DISCRIMINATORS: tuple[tuple[str, str], ...] = (
    ("cache_key_excludes_dump_to", '"--xla_dump_to",'),
    ("cache_key_excludes_dump_as_text", '"--xla_dump_hlo_as_text",'),
    ("compile_options_clears_dump_to", 'debug_options.xla_dump_to = ""'),
    (
        "compile_options_clears_dump_as_text",
        "debug_options.xla_dump_hlo_as_text = False",
    ),
)


@dataclass(frozen=True)
class SidecarBounds:
    """Hard ceilings so the collector can never violate compile/RAM budgets."""

    max_files: int = 8
    max_file_bytes: int = 64 << 20
    max_total_bytes: int = 192 << 20
    max_lines: int = 2_000_000
    max_values: int = 400_000
    max_retained_records: int = 64

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_lines": self.max_lines,
            "max_values": self.max_values,
            "max_retained_records": self.max_retained_records,
        }


BOUNDS = SidecarBounds()


class SidecarRefusal(RuntimeError):
    """The sidecar could not be produced from proven primary evidence."""

    def __init__(self, status: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


_ALLOCATION_RE = re.compile(r"^allocation\s+(\d+):\s+size\s+(\d+)\b(.*)$")
# `HloValue::ToShortString()` is `<%d %s%s%s%s>`: id, instruction name, the
# shape index for tuple shapes, and up to two trailing annotations such as
# " (phi)". Name and index are parsed strictly; trailing annotations are
# tolerated because they carry no size or liveness information.
_VALUE_RE = re.compile(
    r"^\s*value:\s+<(?P<id>\d+)\s+(?P<name>[^\s<>{}]+)(?P<index>\{[^}]*\})?"
    r"(?P<extra>[^>]*)>\s+\(size=(?P<size>\d+),offset=(?P<offset>\d+)\):"
)
_LIVE_RANGE_RE = re.compile(
    r"^\s{4}(?P<name>\S+?)(?P<index>\{[^}]*\})?:(?P<start>\d+)-(?P<end>\d+)\s*$"
)
_LIVE_RANGE_HEADER_RE = re.compile(r"^HloLiveRange\s+\(max\s+(\d+)\):\s*$")
_STATS_RE = re.compile(r"^\s*(?P<label>[a-z_ ]+) allocation:\s+(?P<value>\S+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _default_library() -> Path | None:
    """Locate the installed jaxlib without importing it."""

    for entry in sys.path:
        candidate = Path(entry) / "jaxlib" / "libjax_common.so"
        if candidate.is_file():
            return candidate
    for entry in sys.path:
        root = Path(entry) / "jaxlib"
        if root.is_dir():
            libraries = sorted(root.glob("*.so"), key=lambda p: -p.stat().st_size)
            if libraries:
                return libraries[0]
    return None


def _default_cache_key_module() -> Path | None:
    for entry in sys.path:
        candidate = Path(entry) / "jax" / "_src" / "cache_key.py"
        if candidate.is_file():
            return candidate
    return None


def _contains_all(
    path: Path,
    discriminators: Sequence[tuple[str, str]],
    *,
    chunk_bytes: int = 8 << 20,
) -> list[str]:
    """Stream a file once and report which literals were never observed.

    Streaming with an overlap keeps peak RAM at a few MiB even for the 300 MB+
    shared library, so the toolchain check cannot itself breach a RAM ceiling.
    """

    needles = [literal.encode("utf-8") for _name, literal in discriminators]
    found = [False] * len(needles)
    overlap = max(len(needle) for needle in needles) - 1 if needles else 0
    tail = b""
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            window = tail + block
            for index, needle in enumerate(needles):
                if not found[index] and needle in window:
                    found[index] = True
            if all(found):
                break
            tail = window[-overlap:] if overlap else b""
    return [
        name
        for (name, _literal), seen in zip(discriminators, found)
        if not seen
    ]


def verify_toolchain(
    *,
    library: Path | None = None,
    cache_key_module: Path | None = None,
) -> dict[str, Any]:
    """Statically prove this exact toolchain emits the format we parse."""

    library = Path(library) if library is not None else _default_library()
    if library is None or not Path(library).is_file():
        raise SidecarRefusal(
            "BLOCKED_TOOLCHAIN_EVIDENCE",
            "no installed jaxlib shared library was found to inspect; the dump "
            "format cannot be proven for this toolchain",
            missing_discriminator="jaxlib_shared_library",
        )
    missing = _contains_all(library, TOOLCHAIN_DISCRIMINATORS)
    if missing:
        raise SidecarRefusal(
            "BLOCKED_TOOLCHAIN_EVIDENCE",
            "the installed XLA does not contain the buffer-assignment/live-range "
            f"format discriminators this parser depends on: {missing}",
            missing_discriminator=missing,
            library=str(library),
        )
    module = (
        Path(cache_key_module)
        if cache_key_module is not None
        else _default_cache_key_module()
    )
    cache_key_missing: list[str] = ["jax_cache_key_module"]
    if module is not None and module.is_file():
        cache_key_missing = _contains_all(
            module, CACHE_KEY_EXCLUSION_DISCRIMINATORS
        )
    if cache_key_missing:
        raise SidecarRefusal(
            "BLOCKED_TOOLCHAIN_EVIDENCE",
            "this JAX no longer proves that the dump flags are excluded from "
            f"the persistent compilation cache key: {cache_key_missing}",
            missing_discriminator=cache_key_missing,
        )
    version_path = Path(library).parent / "version.py"
    return {
        "schema": TOOLCHAIN_SCHEMA,
        "status": "PASS",
        "library": str(library),
        "library_bytes": Path(library).stat().st_size,
        "library_version_sha256": (
            sha256_file(version_path) if version_path.is_file() else None
        ),
        "cache_key_module": str(module) if module is not None else None,
        "discriminators": {
            name: literal for name, literal in TOOLCHAIN_DISCRIMINATORS
        },
        "discriminators_sha256": canonical_sha256(
            [list(item) for item in TOOLCHAIN_DISCRIMINATORS]
        ),
        "dump_flags_excluded_from_cache_key": True,
        "cache_key_discriminators": {
            name: literal for name, literal in CACHE_KEY_EXCLUSION_DISCRIMINATORS
        },
    }


def _select_dump_files(
    dump_dir: Path, bounds: SidecarBounds
) -> dict[str, list[Path]]:
    root = Path(dump_dir)
    if root.is_symlink() or not root.is_dir():
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            f"XLA dump directory is missing/not a directory: {root}",
        )
    selected = {
        "buffer_assignment": sorted(root.glob("*-buffer-assignment.txt")),
        "live_range": sorted(root.glob("*-live-range.txt")),
    }
    for role, paths in selected.items():
        if not paths:
            raise SidecarRefusal(
                "BLOCKED_SIDECAR_EVIDENCE",
                f"XLA dump contains no {role.replace('_', ' ')} artifact",
                dump_dir=str(root),
            )
    total_files = sum(len(paths) for paths in selected.values())
    if total_files > bounds.max_files:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_BOUNDS",
            f"{total_files} candidate dump files exceed the "
            f"{bounds.max_files}-file bound; refusing rather than sampling",
        )
    total_bytes = 0
    for paths in selected.values():
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise SidecarRefusal(
                    "BLOCKED_SIDECAR_EVIDENCE",
                    f"dump member is not a regular file: {path}",
                )
            size = path.stat().st_size
            if size > bounds.max_file_bytes:
                raise SidecarRefusal(
                    "BLOCKED_SIDECAR_BOUNDS",
                    f"{path.name} is {size} bytes, above the "
                    f"{bounds.max_file_bytes}-byte per-file bound",
                )
            total_bytes += size
    if total_bytes > bounds.max_total_bytes:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_BOUNDS",
            f"dump aggregate {total_bytes} bytes exceeds the "
            f"{bounds.max_total_bytes}-byte bound",
        )
    return selected


def _iter_lines(path: Path, bounds: SidecarBounds) -> Iterable[str]:
    count = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            count += 1
            if count > bounds.max_lines:
                raise SidecarRefusal(
                    "BLOCKED_SIDECAR_BOUNDS",
                    f"{path.name} exceeds the {bounds.max_lines}-line parse bound",
                )
            yield line.rstrip("\n")


def parse_buffer_assignment(
    path: Path, *, bounds: SidecarBounds = BOUNDS
) -> dict[str, Any]:
    """Reduce one ``*-buffer-assignment.txt`` to sized values plus stats."""

    values: dict[str, int] = {}
    duplicates: list[str] = []
    allocations: list[dict[str, Any]] = []
    stats: dict[str, str] = {}
    current: dict[str, Any] | None = None
    saw_stats_header = False
    for line in _iter_lines(path, bounds):
        allocation = _ALLOCATION_RE.match(line)
        if allocation is not None:
            current = {
                "index": int(allocation.group(1)),
                "size_bytes": int(allocation.group(2)),
                "attributes": allocation.group(3).strip(", "),
                "value_count": 0,
            }
            allocations.append(current)
            continue
        if line.strip() == "BufferAssignment stats:":
            saw_stats_header = True
            continue
        if saw_stats_header:
            stat = _STATS_RE.match(line)
            if stat is not None:
                stats[stat.group("label").strip().replace(" ", "_")] = stat.group(
                    "value"
                )
        value = _VALUE_RE.match(line)
        if value is None:
            continue
        key = f"{value.group('name')}{value.group('index') or '{}'}"
        size = int(value.group("size"))
        if key in values:
            if values[key] != size:
                duplicates.append(key)
            continue
        values[key] = size
        if len(values) > bounds.max_values:
            raise SidecarRefusal(
                "BLOCKED_SIDECAR_BOUNDS",
                f"{path.name} assigns more than {bounds.max_values} values",
            )
        if current is not None:
            current["value_count"] += 1
    if not allocations or not values:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            f"{path.name} has no parsable allocation/value records",
        )
    if duplicates:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            f"{path.name} assigns conflicting sizes to {duplicates[:8]}",
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": Path(path).stat().st_size,
        "allocation_count": len(allocations),
        "allocation_total_bytes": sum(item["size_bytes"] for item in allocations),
        "value_sizes": values,
        "stats": stats,
    }


def parse_live_range(
    path: Path, *, bounds: SidecarBounds = BOUNDS
) -> dict[str, Any]:
    """Reduce one ``*-live-range.txt`` to per-value schedule intervals."""

    intervals: dict[str, tuple[int, int]] = {}
    schedule_end: int | None = None
    section: str | None = None
    for line in _iter_lines(path, bounds):
        header = _LIVE_RANGE_HEADER_RE.match(line)
        if header is not None:
            schedule_end = int(header.group(1))
            continue
        if line == "  InstructionSequence:":
            section = "sequence"
            continue
        if line == "  BufferLiveRange:":
            section = "buffers"
            continue
        if section != "buffers":
            continue
        entry = _LIVE_RANGE_RE.match(line)
        if entry is None:
            continue
        key = f"{entry.group('name')}{entry.group('index') or '{}'}"
        start = int(entry.group("start"))
        end = int(entry.group("end"))
        if start > end:
            raise SidecarRefusal(
                "BLOCKED_SIDECAR_EVIDENCE",
                f"{path.name} has an inverted live range for {key}",
            )
        if key in intervals and intervals[key] != (start, end):
            raise SidecarRefusal(
                "BLOCKED_SIDECAR_EVIDENCE",
                f"{path.name} gives {key} two different live ranges",
            )
        intervals[key] = (start, end)
        if len(intervals) > bounds.max_values:
            raise SidecarRefusal(
                "BLOCKED_SIDECAR_BOUNDS",
                f"{path.name} declares more than {bounds.max_values} live ranges",
            )
    if schedule_end is None or not intervals:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            f"{path.name} has no HloLiveRange header or no buffer live ranges",
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": Path(path).stat().st_size,
        "schedule_end": schedule_end,
        "intervals": intervals,
    }


def compute_peak_live_bytes(
    *,
    value_sizes: dict[str, int],
    intervals: dict[str, tuple[int, int]],
    max_retained: int = BOUNDS.max_retained_records,
) -> dict[str, Any]:
    """Sweep the schedule and return the true maximum simultaneous live bytes.

    Every live-ranged value must have a size from buffer assignment; a partial
    join would silently understate the peak, so it refuses instead.
    """

    unsized = sorted(key for key in intervals if key not in value_sizes)
    if unsized:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            "buffer assignment does not size every live-ranged value; the join "
            f"is incomplete for {unsized[:8]} ({len(unsized)} total)",
            missing_discriminator="buffer_assignment_live_range_join",
        )
    events: list[tuple[int, int, int]] = []
    for key, (start, end) in intervals.items():
        size = int(value_sizes[key])
        if size < 0:
            raise SidecarRefusal(
                "BLOCKED_SIDECAR_EVIDENCE", f"negative buffer size for {key}"
            )
        # `end` is inclusive on the flattened schedule, so the value is freed at
        # end + 1. Frees are ordered before allocations at the same index
        # because XLA's own heap simulator lets a buffer starting at t reuse the
        # space of one whose live range ended at t-1. Ordering them the other
        # way would report a peak that no schedule ever reaches.
        events.append((start, 1, size))
        events.append((end + 1, 0, -size))
    events.sort(key=lambda item: (item[0], item[1]))
    live = 0
    peak = 0
    peak_time = 0
    for time_index, _kind, delta in events:
        live += delta
        if live > peak:
            peak = live
            peak_time = time_index
    if live != 0:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            "live-range sweep did not return to zero; the dump is inconsistent",
        )
    live_at_peak = sorted(
        (
            {"value": key, "size_bytes": int(value_sizes[key]),
             "start": start, "end": end}
            for key, (start, end) in intervals.items()
            if start <= peak_time <= end
        ),
        key=lambda item: (-item["size_bytes"], item["value"]),
    )
    return {
        "peak_live_bytes": peak,
        "peak_schedule_index": peak_time,
        "live_value_count_at_peak": len(live_at_peak),
        "largest_live_values_at_peak": live_at_peak[:max_retained],
        "values_joined": len(intervals),
        "method": (
            "schedule sweep over compiler live ranges with buffer-assignment "
            "sizes; never derived from aggregate temporary_bytes"
        ),
    }


def collect_sidecar(
    dump_dir: Path,
    *,
    run_id: str,
    bounds: SidecarBounds = BOUNDS,
    library: Path | None = None,
    cache_key_module: Path | None = None,
) -> dict[str, Any]:
    """Produce one bounded, content-addressed live-buffer evidence object."""

    toolchain = verify_toolchain(
        library=library, cache_key_module=cache_key_module
    )
    selected = _select_dump_files(Path(dump_dir), bounds)
    assignments = [
        parse_buffer_assignment(path, bounds=bounds)
        for path in selected["buffer_assignment"]
    ]
    ranges = [
        parse_live_range(path, bounds=bounds) for path in selected["live_range"]
    ]
    if len(assignments) != 1 or len(ranges) != 1:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            "expected exactly one buffer-assignment and one live-range artifact; "
            f"found {len(assignments)} and {len(ranges)}",
        )
    assignment = assignments[0]
    live_range = ranges[0]
    peak = compute_peak_live_bytes(
        value_sizes=assignment["value_sizes"],
        intervals=live_range["intervals"],
        max_retained=bounds.max_retained_records,
    )
    if peak["peak_live_bytes"] > assignment["allocation_total_bytes"]:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            f"computed peak {peak['peak_live_bytes']} exceeds the assigned "
            f"total {assignment['allocation_total_bytes']}; evidence is "
            "internally inconsistent",
        )
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "run_id": run_id,
        "dump_dir": str(Path(dump_dir).resolve()),
        "toolchain": toolchain,
        "bounds": bounds.as_dict(),
        "peak_live_buffers": int(peak["peak_live_bytes"]),
        "peak_live_buffers_source": (
            "compiler_buffer_assignment_and_live_range_dump"
        ),
        "peak_live_buffers_detail": peak,
        "buffer_assignment": {
            key: value
            for key, value in assignment.items()
            if key != "value_sizes"
        },
        "live_range": {
            key: value for key, value in live_range.items() if key != "intervals"
        },
        "aggregate_temporary_bytes_is_not_peak_live": True,
        "device_action": False,
    }
    payload["sidecar_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "sidecar_sha256"}
    )
    return payload


def validate_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject a promoted aggregate, a stale hash, or a missing provenance."""

    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE", "live-buffer sidecar schema changed"
        )
    if payload.get("status") != "PASS":
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            f"live-buffer sidecar status is {payload.get('status')!r}",
        )
    if (
        payload.get("peak_live_buffers_source")
        != "compiler_buffer_assignment_and_live_range_dump"
    ):
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            "peak live buffers must come from the compiler dump; an aggregate "
            "temporary_bytes substitution is never a peak-live-buffer proof",
        )
    if payload.get("aggregate_temporary_bytes_is_not_peak_live") is not True:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE",
            "sidecar dropped the aggregate-vs-peak distinction",
        )
    peak = payload.get("peak_live_buffers")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE", "peak live buffers is not a positive integer"
        )
    toolchain = payload.get("toolchain") or {}
    if (
        toolchain.get("schema") != TOOLCHAIN_SCHEMA
        or toolchain.get("status") != "PASS"
        or toolchain.get("discriminators_sha256")
        != canonical_sha256([list(item) for item in TOOLCHAIN_DISCRIMINATORS])
    ):
        raise SidecarRefusal(
            "BLOCKED_TOOLCHAIN_EVIDENCE",
            "sidecar is not bound to the statically verified dump format",
        )
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "sidecar_sha256"}
    )
    if payload.get("sidecar_sha256") != expected:
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE", "live-buffer sidecar content hash changed"
        )
    return {
        "status": "PASS",
        "peak_live_buffers": peak,
        "sidecar_sha256": expected,
    }


def refusal_payload(exc: SidecarRefusal, *, run_id: str | None = None) -> dict[str, Any]:
    """The honest object a refusal publishes instead of an invented metric."""

    return {
        "schema": SCHEMA,
        "status": exc.status,
        "run_id": run_id,
        "reason": str(exc),
        "peak_live_buffers": "MISSING",
        "peak_live_buffers_source": None,
        "aggregate_temporary_bytes_is_not_peak_live": True,
        "gpu_session_permitted": False,
        "device_action": False,
        **exc.detail,
    }


def _atomic_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise SidecarRefusal(
            "BLOCKED_SIDECAR_EVIDENCE", f"refusing to replace sidecar: {path}"
        )
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
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-toolchain-only", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_toolchain_only:
            payload = verify_toolchain()
        else:
            if args.dump_dir is None or args.run_id is None:
                parser.error("--dump-dir and --run-id are required")
            payload = collect_sidecar(args.dump_dir, run_id=args.run_id)
            validate_sidecar(payload)
    except SidecarRefusal as exc:
        refusal = refusal_payload(exc, run_id=args.run_id)
        print(json.dumps(refusal, indent=2, sort_keys=True, default=str))
        return 2
    if args.output is not None:
        _atomic_json_no_replace(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "BOUNDS",
    "CACHE_KEY_EXCLUSION_DISCRIMINATORS",
    "SCHEMA",
    "SidecarBounds",
    "SidecarRefusal",
    "TOOLCHAIN_DISCRIMINATORS",
    "collect_sidecar",
    "compute_peak_live_bytes",
    "parse_buffer_assignment",
    "parse_live_range",
    "refusal_payload",
    "validate_sidecar",
    "verify_toolchain",
]


if __name__ == "__main__":
    raise SystemExit(main())
