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
        for event in events:
            name = str(event.get("name", ""))
            args = event.get("args", {}) if isinstance(event, dict) else {}
            size = 0
            if isinstance(args, dict):
                for key in ("bytes", "Byte Size", "size", "NumBytes"):
                    if key in args:
                        try:
                            size = max(size, int(args[key]))
                        except (TypeError, ValueError):
                            pass
            if H2D_RE.search(name):
                h2d += size
            elif D2H_RE.search(name):
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
