#!/usr/bin/env python3
"""Summarize transfer and launch timeline evidence from replay proofs.

Purpose: check whether diagnostic or replay runs preserve device residency and
avoid host callbacks inside the post-init timestep path.
Answers: JAXPR callback-free flag, H2D/D2H bytes, launch count if provided, and
peak memory.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `src/gpuwrf/integration/d02_replay.py:549-618` implements
static and trace transfer audits, and `src/gpuwrf/integration/d02_replay.py:621-632`
records peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    audit = payload.get("transfer_audit", payload.get("audit", {}))
    static = audit.get("static", {})
    trace = audit.get("trace", {})
    launch_count = payload.get("launch_count", payload.get("kernel_launch_count"))
    peak = payload.get("peak_gpu_memory", payload.get("peak_memory", {}))
    h2d = _int_or_none(trace.get("host_to_device_bytes_post_init", trace.get("h2d_bytes", 0))) or 0
    d2h = _int_or_none(trace.get("device_to_host_bytes_post_init", trace.get("d2h_bytes", 0))) or 0
    callback_free = bool(static.get("host_callback_free", payload.get("jaxpr_callback_free", True)))

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "transfer_launch_timeline", "purpose": "Summarize residency and launch evidence."},
        "input": {"path": str(input_path)},
        "measurements": {
            "jaxpr_callback_free": callback_free,
            "host_to_device_bytes_post_init": h2d,
            "device_to_host_bytes_post_init": d2h,
            "post_init_total_transfer_bytes": h2d + d2h,
            "launch_count": _int_or_none(launch_count),
            "peak_memory": peak,
            "trace_transfer_event_files": trace.get("trace_transfer_event_files", []),
        },
        "units": {"bytes": "B", "launch_count": "count"},
        "artifacts": {},
        "status": "OK" if callback_free and h2d + d2h == 0 else "TRANSFER_OR_CALLBACK_RISK",
        "source_citations": [
            "src/gpuwrf/integration/d02_replay.py:549-618",
            "src/gpuwrf/integration/d02_replay.py:621-632",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
