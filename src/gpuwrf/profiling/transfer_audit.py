"""Transfer-audit helpers for the M3 dummy loop."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import jax


TRANSFER_RE = re.compile(r"(memcpy|transfer|host_to_device|device_to_host|h2d|d2h)", re.IGNORECASE)
H2D_RE = re.compile(r"(host_to_device|h2d|memcpyh2d)", re.IGNORECASE)
D2H_RE = re.compile(r"(device_to_host|d2h|memcpyd2h)", re.IGNORECASE)
SIZE_RE = re.compile(r"(?:^|[\s,{])(?:bytes|byte_size|size|num_bytes|NumBytes)\s*[:=]\s*(\d+)", re.IGNORECASE)


def block_until_ready(value: Any) -> None:
    """Synchronizes a pytree; reused by timing and transfer-audit call sites."""

    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, value)


def visible_gpu_name() -> str:
    """Reports the selected GPU name for machine-readable audit metadata."""

    for device in jax.devices():
        if device.platform == "gpu":
            return str(device)
    return "none"


def _read_trace(path: Path) -> str:
    """Reads plain or gzipped profiler trace chunks from JAX trace output."""

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _flatten_text(value: Any) -> str:
    """Serializes nested trace args so memcpy_details direction and size are visible."""

    if isinstance(value, dict):
        return " ".join(f"{key}:{_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _largest_size(value: Any) -> int:
    """Extracts byte counts from trace args, including nested memcpy_details payloads."""

    size = 0
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"bytes", "byte size", "byte_size", "size", "numbytes", "num_bytes"}:
                try:
                    size = max(size, int(item))
                except (TypeError, ValueError):
                    pass
            size = max(size, _largest_size(item))
        return size
    if isinstance(value, (list, tuple)):
        for item in value:
            size = max(size, _largest_size(item))
        return size
    if isinstance(value, str):
        for match in SIZE_RE.finditer(value):
            size = max(size, int(match.group(1)))
    return size


def count_transfer_bytes(trace_dir: Path) -> tuple[int, int, list[str]]:
    """Scans profiler trace text for post-init memcpy events and byte counts."""

    h2d = 0
    d2h = 0
    matched: list[str] = []
    for path in sorted(trace_dir.rglob("*")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix not in (".json", ".gz", ".trace", ".pb"):
            continue
        text = _read_trace(path)
        if not TRANSFER_RE.search(text):
            continue
        matched.append(str(path))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        events = payload.get("traceEvents", []) if isinstance(payload, dict) else []
        if not events:
            size = _largest_size(text)
            if H2D_RE.search(text):
                h2d += size
            if D2H_RE.search(text):
                d2h += size
            continue
        for event in events:
            name = str(event.get("name", ""))
            args = event.get("args", {}) if isinstance(event, dict) else {}
            detail_text = f"{name} {_flatten_text(args)}"
            size = _largest_size(args)
            if H2D_RE.search(detail_text):
                h2d += size
            elif D2H_RE.search(detail_text):
                d2h += size
    return h2d, d2h, matched


def write_transfer_audit(path: Path, iterations: int, trace_dir: Path) -> dict[str, Any]:
    """Writes the M3 transfer-audit JSON after a traced warmed dummy-loop run."""

    h2d, d2h, matches = count_transfer_bytes(trace_dir)
    payload = {
        "host_to_device_bytes_post_init": int(h2d),
        "device_to_host_bytes_post_init": int(d2h),
        "iterations": int(iterations),
        "method": "jax.profiler.trace scanned for post-init memcpy events",
        "jax_version": jax.__version__,
        "gpu_name": visible_gpu_name(),
        "trace_dir": str(trace_dir),
        "trace_transfer_event_files": matches,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
