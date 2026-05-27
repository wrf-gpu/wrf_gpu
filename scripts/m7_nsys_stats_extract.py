#!/usr/bin/env python
"""Extract M7 Nsight Systems kernel/API summary from an nsys report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [str(row[1]) for row in rows]


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


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


def _quote_list(cols: list[str]) -> str:
    return ", ".join(f'"{col}"' for col in cols)


def _resolve_name(row: dict[str, Any], strings: dict[int, str], candidates: tuple[str, ...], fallback: str) -> str:
    for col in candidates:
        if col not in row:
            continue
        value = row[col]
        if value is None:
            continue
        if isinstance(value, int) and value in strings:
            resolved = strings[value]
        else:
            resolved = str(value)
        if resolved and resolved.lower() != "none":
            return resolved
    return fallback


def _kernel_rows(conn: sqlite3.Connection, strings: dict[int, str]) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in _tables(conn):
        return []
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    required = [col for col in ("start", "end") if col in cols]
    if len(required) != 2:
        return []
    name_candidates = (
        "demangledName",
        "shortName",
        "mangledName",
        "name",
        "nameId",
    )
    selected = required + [col for col in name_candidates if col in cols]
    rows: list[dict[str, Any]] = []
    for raw in conn.execute(f'SELECT {_quote_list(selected)} FROM "CUPTI_ACTIVITY_KIND_KERNEL"').fetchall():
        row = dict(zip(selected, raw, strict=True))
        start = int(row["start"])
        end = int(row["end"])
        rows.append(
            {
                "name": _resolve_name(row, strings, name_candidates, "unknown_kernel"),
                "start_ns": start,
                "end_ns": end,
                "duration_ns": max(0, end - start),
            }
        )
    return rows


def _runtime_summary(conn: sqlite3.Connection, strings: dict[int, str]) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in _tables(conn):
        return []
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if "start" not in cols or "end" not in cols:
        return []
    name_candidates = ("nameId", "name", "cbid")
    selected = ["start", "end"] + [col for col in name_candidates if col in cols]
    grouped: dict[str, dict[str, Any]] = {}
    for raw in conn.execute(f'SELECT {_quote_list(selected)} FROM "CUPTI_ACTIVITY_KIND_RUNTIME"').fetchall():
        row = dict(zip(selected, raw, strict=True))
        name = _resolve_name(row, strings, name_candidates, "unknown_cuda_api")
        duration = max(0, int(row["end"]) - int(row["start"]))
        bucket = grouped.setdefault(name, {"name": name, "count": 0, "total_duration_ns": 0})
        bucket["count"] += 1
        bucket["total_duration_ns"] += duration
    rows = list(grouped.values())
    rows.sort(key=lambda item: (-int(item["count"]), -int(item["total_duration_ns"]), str(item["name"])))
    for row in rows:
        row["total_duration_ms"] = float(row["total_duration_ns"]) / 1.0e6
    return rows


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


def _memcpy_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    if "CUPTI_ACTIVITY_KIND_MEMCPY" not in _tables(conn):
        return {"available": False, "by_kind": []}
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
    if "copyKind" not in cols:
        return {"available": False, "by_kind": []}
    bytes_col = "bytes" if "bytes" in cols else None
    kind_names = _copy_kind_map(conn)
    selected = ["copyKind"] + ([bytes_col] if bytes_col else [])
    grouped: dict[int, dict[str, Any]] = {}
    for raw in conn.execute(f'SELECT {_quote_list(selected)} FROM "CUPTI_ACTIVITY_KIND_MEMCPY"').fetchall():
        row = dict(zip(selected, raw, strict=True))
        kind = int(row["copyKind"])
        bucket = grouped.setdefault(
            kind,
            {"copyKind": kind, "name": kind_names.get(kind, f"copyKind:{kind}"), "count": 0, "bytes": 0},
        )
        bucket["count"] += 1
        if bytes_col:
            bucket["bytes"] += int(row[bytes_col] or 0)
    by_kind = list(grouped.values())
    by_kind.sort(key=lambda item: (-int(item["count"]), str(item["name"])))
    return {"available": True, "by_kind": by_kind}


def extract_summary(report: Path, sqlite_path: Path) -> dict[str, Any]:
    export = _run_export(report, sqlite_path)
    if int(export["returncode"]) != 0:
        return {
            "artifact_type": "m7_nsys_summary",
            "status": "BLOCKED-PROFILER",
            "nsys_report": str(report),
            "sqlite_export": str(sqlite_path),
            "export": export,
            "error": "nsys export failed",
        }

    conn = sqlite3.connect(str(sqlite_path))
    strings = _string_ids(conn)
    kernels = _kernel_rows(conn, strings)
    runtime = _runtime_summary(conn, strings)
    memcpy = _memcpy_summary(conn)
    conn.close()

    kernels.sort(key=lambda item: int(item["duration_ns"]), reverse=True)
    top = []
    for rank, row in enumerate(kernels[:10], start=1):
        top.append(
            {
                "rank": rank,
                "name": row["name"],
                "duration_ns": int(row["duration_ns"]),
                "duration_ms": float(row["duration_ns"]) / 1.0e6,
                "start_ns": int(row["start_ns"]),
                "end_ns": int(row["end_ns"]),
                "occupancy": None,
                "occupancy_note": "Nsight Systems trace does not expose achieved occupancy; see ncu_hot_kernels.json.",
            }
        )

    total_gpu_ns = sum(int(row["duration_ns"]) for row in kernels)
    return {
        "artifact_type": "m7_nsys_summary",
        "status": "PASS" if kernels else "BLOCKED-PROFILER",
        "nsys_report": str(report),
        "sqlite_export": str(sqlite_path),
        "export": export,
        "total_gpu_time_ns": int(total_gpu_ns),
        "total_gpu_time_s": float(total_gpu_ns) / 1.0e9,
        "kernel_count": len(kernels),
        "longest_10_kernels": top,
        "cuda_api_call_counts": runtime,
        "memcpy_summary": memcpy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help=".nsys-rep or .qdrep file to summarize")
    parser.add_argument("--sqlite", type=Path, default=None, help="Optional sqlite export path")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = args.report
    sqlite_path = args.sqlite or report.with_suffix(".sqlite")
    payload = extract_summary(report, sqlite_path)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
