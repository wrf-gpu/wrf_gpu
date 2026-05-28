#!/usr/bin/env python
"""Shared helpers for publication-test proof objects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SPRINT_DIR = ROOT / ".agent" / "sprints" / "2026-05-27-testing-plan-execution-redo"
WRF_SOURCE = Path("/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF")
WRF_ENV = Path("/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh")
STABLE_WRF_EXE = Path("/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe")
CANARY_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")

HIGH_TEST_FILES = {
    "IDEALIZED-WARMBUBBLE": "idealized_warmbubble.json",
    "IDEALIZED-DENSITY-CURRENT": "idealized_density_current.json",
    "IDEALIZED-MOUNTAIN-WAVE": "idealized_mountain_wave.json",
    "CONSERVATION-MASS-24H": "conservation_mass_24h.json",
    "CONSERVATION-ENERGY-24H": "conservation_energy_24h.json",
    "STABILITY-CFL-SWEEP": "stability_cfl_sweep.json",
    "STABILITY-ACOUSTIC-SUBSTEP-SWEEP": "stability_acoustic_substep.json",
    "DETERMINISM-REPEAT": "determinism_repeat.json",
    "SAVEPOINT-PARITY-DEEP": "savepoint_parity_deep.json",
    "CANARY-MULTIDAY-SIDE-BY-SIDE": "canary_multiday_skill.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str], *, cwd: Path = ROOT, timeout_s: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "timeout_s": int(timeout_s),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "timeout_s": int(timeout_s),
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def git_head(path: Path = ROOT) -> str | None:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=path, timeout_s=10)
    if result["returncode"] != 0:
        return None
    return str(result["stdout"]).strip()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrf_provenance() -> dict[str, Any]:
    source_commit = git_head(WRF_SOURCE) if WRF_SOURCE.exists() else None
    return {
        "wrf_source_path": str(WRF_SOURCE),
        "wrf_source_exists": WRF_SOURCE.exists(),
        "wrf_source_commit": source_commit,
        "env_script": str(WRF_ENV),
        "env_script_exists": WRF_ENV.exists(),
        "stable_wrf_exe": str(STABLE_WRF_EXE),
        "stable_wrf_exe_exists": STABLE_WRF_EXE.exists(),
        "stable_wrf_exe_executable": STABLE_WRF_EXE.exists() and bool(STABLE_WRF_EXE.stat().st_mode & 0o111),
        "stable_wrf_exe_sha256": sha256_file(STABLE_WRF_EXE),
    }


def gpu_probe(*, skip: bool = False, timeout_s: int = 5) -> dict[str, Any]:
    if skip:
        return {"available": None, "skipped": True, "reason": "skip-gpu-probe requested"}
    result = run_command(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
        timeout_s=int(timeout_s),
    )
    return {
        "available": result["returncode"] == 0,
        "skipped": False,
        "command": result,
    }


def proof_header(test_id: str, verdict: str, status: str) -> dict[str, Any]:
    return {
        "schema": "PublicationHighPriorityProof",
        "schema_version": 1,
        "test_id": test_id,
        "verdict": verdict,
        "status": status,
        "generated_utc": utc_now(),
        "repo_commit": git_head(ROOT),
    }


def write_case_summary(path: Path, case: Any) -> dict[str, Any]:
    summary = case.summary()
    write_json(path, summary)
    return summary


def finite_stats(summary: dict[str, Any]) -> bool:
    return all(bool(row.get("finite")) for row in summary.get("array_stats", {}).values())


def threshold_rows(rows: dict[str, tuple[float | None, str, bool | None]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (value, threshold, passed) in rows.items():
        out[name] = {"value": value, "threshold": threshold, "passed": passed}
    return out


def summarize_repeatability(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        return {"path": str(path), "exists": False}
    fields = data.get("comparison", {}).get("fields", {})
    max_delta = 0.0
    for field in fields.values():
        if isinstance(field, dict):
            max_delta = max(max_delta, float(field.get("max_abs_delta") or 0.0))
    return {
        "path": str(path),
        "exists": True,
        "status": data.get("status"),
        "run_count": 2 if data.get("run1_final_wrfout") and data.get("run2_final_wrfout") else None,
        "field_count": data.get("comparison", {}).get("field_count"),
        "max_abs_delta": max_delta,
    }


def summarize_restart(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        return {"path": str(path), "exists": False}
    fields = data.get("fields", {})
    return {
        "path": str(path),
        "exists": True,
        "verdict": data.get("verdict"),
        "field_count": len(fields),
        "max_delta": max((float(row.get("max_delta") or 0.0) for row in fields.values() if isinstance(row, dict)), default=0.0),
        "all_fields_pass": all(bool(row.get("pass")) for row in fields.values() if isinstance(row, dict)),
    }


def summarize_m6b6(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        return {"path": str(path), "exists": False}
    tier_counts = {}
    for tier_name, tier in data.get("tiers", {}).items():
        if isinstance(tier, dict):
            tier_counts[tier_name] = tier.get("savepoint_count") or len(tier.get("results", []))
    return {
        "path": str(path),
        "exists": True,
        "passed": data.get("passed"),
        "outcome": data.get("outcome"),
        "tier_savepoint_counts": tier_counts,
    }


def summarize_skill(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        return {"path": str(path), "exists": False}
    variables = {}
    for name, row in data.get("aggregate_comparison", {}).get("variables", {}).items():
        variables[name] = {
            "within_20pct_all_metrics": row.get("within_20pct_all_metrics"),
            "rmse_relative_delta": row.get("metrics", {}).get("rmse", {}).get("relative_delta"),
        }
    return {
        "path": str(path),
        "exists": True,
        "verdict": data.get("verdict"),
        "common_valid_time_count": data.get("common_valid_time_count"),
        "station_count_scored": data.get("station_count_scored"),
        "variables": variables,
    }


def discover_canary_cases(run_root: Path = CANARY_RUN_ROOT, *, window_days: int = 14) -> dict[str, Any]:
    by_day: dict[str, Path] = {}
    for path in sorted(run_root.glob("*_18z_l3_24h_*")):
        if not path.is_dir():
            continue
        day = path.name[:8]
        if len(day) == 8 and day.isdigit():
            by_day[day] = path
    days = sorted(by_day)
    selected: list[str] = []
    for start in range(max(len(days) - int(window_days) + 1, 0)):
        window = days[start : start + int(window_days)]
        parsed = [datetime.strptime(day, "%Y%m%d").date() for day in window]
        if all((parsed[i + 1] - parsed[i]).days == 1 for i in range(len(parsed) - 1)):
            selected = window
            break
    if not selected:
        selected = days[: int(window_days)]
    return {
        "run_root": str(run_root),
        "available_day_count": len(days),
        "selected_window_days": len(selected),
        "continuous": len(selected) == int(window_days)
        and all(
            (datetime.strptime(selected[i + 1], "%Y%m%d").date() - datetime.strptime(selected[i], "%Y%m%d").date()).days == 1
            for i in range(len(selected) - 1)
        ),
        "cases": [{"day": day, "run_dir": str(by_day[day])} for day in selected],
    }


def parse_proof_dir(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-dir", type=Path, default=SPRINT_DIR)
    return parser.parse_args(argv)


def write_summary_md(path: Path, title: str, payload: dict[str, Any], lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", "", f"Verdict: {payload.get('verdict')}", f"Status: {payload.get('status')}", ""]
    body.extend(lines)
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
