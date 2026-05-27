#!/usr/bin/env python
"""Classify D2H transfers in an M7 Nsight Systems trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _quote_list(cols: list[str]) -> str:
    return ", ".join(f'"{col}"' for col in cols)


def _run_export(report: Path, sqlite_path: Path) -> dict[str, Any]:
    if sqlite_path.exists():
        return {"returncode": 0, "stdout": "", "stderr": "", "reused_existing_sqlite": True}
    cmd = [
        "nsys",
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(report),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "reused_existing_sqlite": False,
    }


def _string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _tables(conn):
        return {}
    cols = _columns(conn, "StringIds")
    value_col = "value" if "value" in cols else "string" if "string" in cols else None
    if value_col is None or "id" not in cols:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in conn.execute(f'SELECT "id", "{value_col}" FROM "StringIds"').fetchall()
    }


def _resolve_string(value: Any, strings: dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, int) and value in strings:
        return strings[value]
    return str(value)


def _copy_kind_map(conn: sqlite3.Connection) -> dict[int, str]:
    fallback = {1: "HtoD", 2: "DtoH", 3: "DtoD", 4: "HtoA", 5: "AtoH", 6: "AtoA", 7: "AtoD", 8: "DtoA"}
    for table in ("ENUM_CUDA_MEMCPY_OPER", "ENUM_CUPTI_ACTIVITY_MEMCPY_KIND"):
        if table not in _tables(conn):
            continue
        cols = _columns(conn, table)
        id_col = "id" if "id" in cols else "value" if "value" in cols else None
        label_col = "label" if "label" in cols else "name" if "name" in cols else None
        if id_col is None or label_col is None:
            continue
        return {
            int(row[0]): str(row[1])
            for row in conn.execute(f'SELECT "{id_col}", "{label_col}" FROM "{table}"').fetchall()
        }
    return fallback


def _kernel_window(conn: sqlite3.Connection) -> dict[str, Any]:
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in _tables(conn):
        return {"available": False, "start_ns": None, "end_ns": None, "count": 0}
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    if "start" not in cols or "end" not in cols:
        return {"available": False, "start_ns": None, "end_ns": None, "count": 0}
    start, end, count = conn.execute(
        'SELECT MIN("start"), MAX("end"), COUNT(*) FROM "CUPTI_ACTIVITY_KIND_KERNEL"'
    ).fetchone()
    return {
        "available": start is not None and end is not None,
        "start_ns": int(start) if start is not None else None,
        "end_ns": int(end) if end is not None else None,
        "count": int(count or 0),
    }


def _nvtx_window(conn: sqlite3.Connection, marker_regex: str) -> dict[str, Any]:
    if "NVTX_EVENTS" not in _tables(conn):
        return {"available": False, "start_ns": None, "end_ns": None, "matches": []}
    strings = _string_ids(conn)
    cols = _columns(conn, "NVTX_EVENTS")
    if "start" not in cols or "end" not in cols:
        return {"available": False, "start_ns": None, "end_ns": None, "matches": []}
    name_candidates = [col for col in ("text", "message", "name", "registeredString", "domainId") if col in cols]
    selected = ["start", "end"] + name_candidates
    pattern = re.compile(marker_regex)
    matches: list[dict[str, Any]] = []
    for raw in conn.execute(f'SELECT {_quote_list(selected)} FROM "NVTX_EVENTS"').fetchall():
        row = dict(zip(selected, raw, strict=True))
        names = [_resolve_string(row.get(col), strings) for col in name_candidates]
        label = " ".join(name for name in names if name)
        if not pattern.search(label):
            continue
        if row["start"] is None or row["end"] is None:
            continue
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        matches.append({"label": label, "start_ns": start, "end_ns": end})
    if not matches:
        return {"available": False, "start_ns": None, "end_ns": None, "matches": []}
    return {
        "available": True,
        "start_ns": min(item["start_ns"] for item in matches),
        "end_ns": max(item["end_ns"] for item in matches),
        "matches": matches,
    }


def _memcpy_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_MEMCPY" not in _tables(conn):
        return []
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
    required = [col for col in ("start", "end", "copyKind") if col in cols]
    if len(required) != 3:
        return []
    bytes_col = "bytes" if "bytes" in cols else None
    selected = required + ([bytes_col] if bytes_col else [])
    kinds = _copy_kind_map(conn)
    rows: list[dict[str, Any]] = []
    for raw in conn.execute(f'SELECT {_quote_list(selected)} FROM "CUPTI_ACTIVITY_KIND_MEMCPY"').fetchall():
        row = dict(zip(selected, raw, strict=True))
        kind = int(row["copyKind"])
        name = kinds.get(kind, f"copyKind:{kind}")
        rows.append(
            {
                "start_ns": int(row["start"]),
                "end_ns": int(row["end"]),
                "copyKind": kind,
                "copyKind_name": name,
                "bytes": int(row.get(bytes_col, 0) or 0) if bytes_col else 0,
                "direction": _direction(kind, name),
            }
        )
    return rows


def _direction(kind: int, name: str) -> str:
    lowered = name.lower()
    if kind == 2 or "dtoh" in lowered or "device_to_host" in lowered or "device to host" in lowered:
        return "D2H"
    if kind == 1 or "htod" in lowered or "host_to_device" in lowered or "host to device" in lowered:
        return "H2D"
    return "OTHER"


def audit_trace(report: Path, sqlite_path: Path, marker_regex: str) -> dict[str, Any]:
    export = _run_export(report, sqlite_path)
    if int(export["returncode"]) != 0:
        return {
            "artifact_type": "m7_d2h_audit",
            "status": "BLOCKED-PROFILER",
            "nsys_report": str(report),
            "sqlite_export": str(sqlite_path),
            "export": export,
            "error": "nsys export failed",
        }
    conn = sqlite3.connect(str(sqlite_path))
    kernel = _kernel_window(conn)
    nvtx = _nvtx_window(conn, marker_regex)
    rows = _memcpy_rows(conn)
    conn.close()

    loop_start = nvtx["start_ns"] if nvtx["available"] else kernel["start_ns"]
    loop_end = nvtx["end_ns"] if nvtx["available"] else kernel["end_ns"]
    method = "nvtx_marker" if nvtx["available"] else "kernel_timeline_process_of_elimination"
    if loop_start is None or loop_end is None:
        loop_start = 0
        loop_end = -1
        method = "no_kernel_window_available"

    def inside(row: dict[str, Any]) -> bool:
        return int(loop_start) <= int(row["start_ns"]) <= int(loop_end)

    d2h_rows = [row for row in rows if row["direction"] == "D2H"]
    h2d_rows = [row for row in rows if row["direction"] == "H2D"]
    d2h_inside = [row for row in d2h_rows if inside(row)]
    h2d_inside = [row for row in h2d_rows if inside(row)]

    k_start = kernel["start_ns"]
    k_end = kernel["end_ns"]
    if k_start is None or k_end is None:
        pre_kernel = d2h_inside
        inter_kernel: list[dict[str, Any]] = []
        post_kernel: list[dict[str, Any]] = []
    else:
        pre_kernel = [row for row in d2h_inside if int(row["start_ns"]) < int(k_start)]
        inter_kernel = [row for row in d2h_inside if int(k_start) <= int(row["start_ns"]) <= int(k_end)]
        post_kernel = [row for row in d2h_inside if int(row["start_ns"]) > int(k_end)]

    clusters: dict[int, dict[str, int]] = {}
    for row in inter_kernel:
        bucket = clusters.setdefault(int(row["bytes"]), {"bytes": int(row["bytes"]), "count": 0})
        bucket["count"] += 1

    status = "PASS" if len(inter_kernel) == 0 and method != "no_kernel_window_available" else "BLOCKED-D2H"
    return {
        "artifact_type": "m7_d2h_audit",
        "status": status,
        "nsys_report": str(report),
        "sqlite_export": str(sqlite_path),
        "export": export,
        "method": method,
        "nvtx_marker_regex": marker_regex,
        "nvtx_window": nvtx,
        "kernel_window": kernel,
        "loop_window_ns": {"start": loop_start, "end": loop_end},
        "counts": {
            "memcpy_total": len(rows),
            "d2h_total_trace": len(d2h_rows),
            "h2d_total_trace": len(h2d_rows),
            "d2h_inside_loop_window": len(d2h_inside),
            "h2d_inside_loop_window": len(h2d_inside),
            "d2h_pre_kernel_inside_window": len(pre_kernel),
            "d2h_inter_kernel_inside_window": len(inter_kernel),
            "d2h_post_kernel_inside_window": len(post_kernel),
        },
        "bytes": {
            "d2h_inter_kernel_inside_window": sum(int(row["bytes"]) for row in inter_kernel),
            "d2h_inside_loop_window": sum(int(row["bytes"]) for row in d2h_inside),
            "h2d_inside_loop_window": sum(int(row["bytes"]) for row in h2d_inside),
        },
        "inter_kernel_d2h_byte_clusters": sorted(clusters.values(), key=lambda item: -item["count"]),
        "classification": {
            "hard_invariant": "D2H inter-kernel/intra-step path must be 0.",
            "pass_condition": "counts.d2h_inter_kernel_inside_window == 0",
            "io_at_step_boundaries_allowed": True,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help=".nsys-rep or .qdrep file to audit")
    parser.add_argument("--sqlite", type=Path, default=None, help="Optional sqlite export path")
    parser.add_argument("--output", type=Path, required=True, help="Audit JSON path")
    parser.add_argument("--marker-regex", default="m7_profile_window", help="NVTX marker regex for timestep loop")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sqlite_path = args.sqlite or args.report.with_suffix(".sqlite")
    payload = audit_trace(args.report, sqlite_path, args.marker_regex)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
