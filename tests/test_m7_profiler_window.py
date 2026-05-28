"""M7 profiler-window D2H audit tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts" / "m7_d2h_audit.py"

spec = importlib.util.spec_from_file_location("m7_d2h_audit", AUDIT_PATH)
assert spec is not None
m7_d2h_audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m7_d2h_audit)


def _sqlite_with_windows(path: Path, *, explicit_marker: bool, xla_module: bool) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            """
            CREATE TABLE NVTX_EVENTS (
                start INTEGER NOT NULL,
                end INTEGER,
                text TEXT,
                textId INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                copyKind INTEGER NOT NULL,
                bytes INTEGER NOT NULL
            )
            """
        )
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 400)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (10, 20, 2, 1024)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (150, 160, 1, 2048)")
        if explicit_marker:
            conn.execute("INSERT INTO NVTX_EVENTS VALUES (50, 350, 'm7_profile_window', NULL)")
        if xla_module:
            conn.execute(
                "INSERT INTO StringIds VALUES (?, ?)",
                (1, "XlaModule:#hlo_module=jit_run_forecast_operational,program_id=7#"),
            )
            conn.execute("INSERT INTO NVTX_EVENTS VALUES (100, 300, NULL, 1)")


def test_xla_module_fallback_excludes_pre_forecast_d2h(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    _sqlite_with_windows(sqlite_path, explicit_marker=False, xla_module=True)

    payload = m7_d2h_audit.audit_trace(tmp_path / "missing.nsys-rep", sqlite_path, "m7_profile_window")

    assert payload["status"] == "PASS"
    assert payload["method"] == "xla_module_nvtx"
    assert payload["window_provenance"] == "xla-module-fallback"
    assert payload["loop_window_ns"] == {"start": 100, "end": 300}
    assert payload["counts"]["d2h_inter_kernel_inside_window"] == 0
    assert payload["counts"]["h2d_inside_loop_window"] == 1


def test_kernel_timeline_fallback_is_tagged_broad(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    _sqlite_with_windows(sqlite_path, explicit_marker=False, xla_module=False)

    payload = m7_d2h_audit.audit_trace(tmp_path / "missing.nsys-rep", sqlite_path, "m7_profile_window")

    assert payload["status"] == "BLOCKED-D2H"
    assert payload["method"] == "kernel_timeline_process_of_elimination"
    assert payload["window_provenance"] == "broad-fallback"
    assert payload["counts"]["d2h_inter_kernel_inside_window"] == 1

