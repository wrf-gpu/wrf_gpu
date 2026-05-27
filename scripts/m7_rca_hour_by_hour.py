#!/usr/bin/env python
"""Hour-by-hour GPU-vs-CPU WRF output deviation diagnostics for M7 RCA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.data_inventory import parse_wrfout_valid_time  # noqa: E402


SPRINT_DIR = ROOT / ".agent/sprints/2026-05-27-m7-skill-regression-rca-codex"
DEFAULT_GPU_ROOT = Path("/tmp/m7_pipeline_runs/20260521")
DEFAULT_CPU_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
DEFAULT_HOURLY_OUTPUT = SPRINT_DIR / "hour_by_hour_deviation.json"
DEFAULT_FIRST_HOUR_OUTPUT = SPRINT_DIR / "first_hour_diff.json"
DEFAULT_FIELDS = ("T2", "U10", "V10", "T", "U", "V", "QVAPOR", "P", "PSFC")


@dataclass(frozen=True)
class WrfoutPair:
    output_index: int
    lead_hour: int
    valid_time: pd.Timestamp
    gpu_path: Path
    cpu_path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def discover_wrfouts(root: str | Path) -> list[Path]:
    source = Path(root)
    files = sorted(path for path in source.glob("wrfout_d02_*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no wrfout_d02_* files found under {source}")
    return files


def _timestamp(path: Path) -> pd.Timestamp:
    stamp = pd.Timestamp(parse_wrfout_valid_time(path))
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _time_map(files: Iterable[Path]) -> dict[pd.Timestamp, Path]:
    return {_timestamp(path): path for path in files}


def build_wrfout_pairs(gpu_root: str | Path, cpu_root: str | Path) -> list[WrfoutPair]:
    gpu_by_time = _time_map(discover_wrfouts(gpu_root))
    cpu_files = discover_wrfouts(cpu_root)
    cpu_by_time = _time_map(cpu_files)
    common_times = sorted(set(gpu_by_time) & set(cpu_by_time))
    if not common_times:
        raise ValueError(f"no common wrfout valid times for GPU={gpu_root} and CPU={cpu_root}")

    cpu_run_start = min(cpu_by_time)
    pairs: list[WrfoutPair] = []
    for index, valid_time in enumerate(common_times):
        lead_hours = int(round((valid_time - cpu_run_start).total_seconds() / 3600.0))
        pairs.append(
            WrfoutPair(
                output_index=index,
                lead_hour=lead_hours,
                valid_time=valid_time,
                gpu_path=gpu_by_time[valid_time],
                cpu_path=cpu_by_time[valid_time],
            )
        )
    return pairs


def load_field(path: str | Path, field: str) -> np.ndarray:
    with Dataset(path, "r") as dataset:
        if field not in dataset.variables:
            raise KeyError(field)
        variable = dataset.variables[field]
        values = variable[:]
        if np.ma.isMaskedArray(values):
            values = values.filled(np.nan)
        array = np.asarray(values)
        if array.shape[:1] == (1,):
            array = array[0]
        return np.asarray(array, dtype=np.float64)


def numeric_variable_names(path: str | Path) -> list[str]:
    names: list[str] = []
    with Dataset(path, "r") as dataset:
        for name, variable in dataset.variables.items():
            if np.issubdtype(np.dtype(variable.dtype), np.number):
                names.append(str(name))
    return sorted(names)


def field_stats(gpu_values: np.ndarray, cpu_values: np.ndarray) -> dict[str, Any]:
    if gpu_values.shape != cpu_values.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "gpu_shape": list(gpu_values.shape),
            "cpu_shape": list(cpu_values.shape),
        }

    diff = np.asarray(gpu_values, dtype=np.float64) - np.asarray(cpu_values, dtype=np.float64)
    finite = np.isfinite(diff) & np.isfinite(gpu_values) & np.isfinite(cpu_values)
    valid_count = int(finite.sum())
    if valid_count == 0:
        return {
            "status": "NO_FINITE_VALUES",
            "shape": list(diff.shape),
            "sample_count": int(diff.size),
            "finite_count": 0,
            "mean_diff": None,
            "max_abs_diff": None,
            "correlation": None,
        }

    gpu_flat = np.asarray(gpu_values[finite], dtype=np.float64)
    cpu_flat = np.asarray(cpu_values[finite], dtype=np.float64)
    diff_flat = np.asarray(diff[finite], dtype=np.float64)
    gpu_std = float(np.std(gpu_flat))
    cpu_std = float(np.std(cpu_flat))
    correlation = None
    if valid_count >= 2 and gpu_std > 0.0 and cpu_std > 0.0:
        correlation = float(np.corrcoef(gpu_flat, cpu_flat)[0, 1])

    return {
        "status": "OK",
        "shape": list(diff.shape),
        "sample_count": int(diff.size),
        "finite_count": valid_count,
        "nonfinite_count": int(diff.size - valid_count),
        "mean_diff": float(np.mean(diff_flat)),
        "max_abs_diff": float(np.max(np.abs(diff_flat))),
        "rmse": float(np.sqrt(np.mean(diff_flat * diff_flat))),
        "mae": float(np.mean(np.abs(diff_flat))),
        "gpu_mean": float(np.mean(gpu_flat)),
        "cpu_mean": float(np.mean(cpu_flat)),
        "correlation": correlation,
    }


def compare_field(gpu_path: str | Path, cpu_path: str | Path, field: str) -> dict[str, Any]:
    try:
        gpu_values = load_field(gpu_path, field)
    except KeyError:
        return {"status": "MISSING_GPU_FIELD"}
    try:
        cpu_values = load_field(cpu_path, field)
    except KeyError:
        return {"status": "MISSING_CPU_FIELD"}
    return field_stats(gpu_values, cpu_values)


def compare_pair(pair: WrfoutPair, fields: Sequence[str]) -> dict[str, Any]:
    return {
        "output_index": int(pair.output_index),
        "lead_hour": int(pair.lead_hour),
        "valid_time_utc": pair.valid_time.isoformat(),
        "gpu_path": str(pair.gpu_path),
        "cpu_path": str(pair.cpu_path),
        "fields": {field: compare_field(pair.gpu_path, pair.cpu_path, field) for field in fields},
    }


def build_hourly_payload(
    *,
    gpu_root: str | Path,
    cpu_root: str | Path,
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> dict[str, Any]:
    pairs = build_wrfout_pairs(gpu_root, cpu_root)
    rows = [compare_pair(pair, fields) for pair in pairs]
    return {
        "schema": "M7SkillRegressionHourByHourDeviation",
        "schema_version": 1,
        "gpu_root": str(gpu_root),
        "cpu_root": str(cpu_root),
        "field_order": list(fields),
        "pair_count": int(len(pairs)),
        "lead_hours": [int(pair.lead_hour) for pair in pairs],
        "hour_index_note": "output_index 0 is the first GPU wrfout, CPU lead_hour 1.",
        "rows": rows,
    }


def build_first_hour_payload(
    *,
    gpu_root: str | Path,
    cpu_root: str | Path,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    pairs = build_wrfout_pairs(gpu_root, cpu_root)
    if not pairs:
        raise ValueError("no wrfout pairs available")
    first = pairs[0]
    requested = list(fields or sorted(set(numeric_variable_names(first.gpu_path)) & set(numeric_variable_names(first.cpu_path))))
    field_results = {field: compare_field(first.gpu_path, first.cpu_path, field) for field in requested}
    ranked = sorted(
        (
            {"field": field, **stats}
            for field, stats in field_results.items()
            if stats.get("status") == "OK" and stats.get("max_abs_diff") is not None
        ),
        key=lambda item: float(item["max_abs_diff"]),
        reverse=True,
    )
    top = ranked[0] if ranked else None
    return {
        "schema": "M7SkillRegressionFirstHourDiff",
        "schema_version": 1,
        "gpu_root": str(gpu_root),
        "cpu_root": str(cpu_root),
        "lead_hour": int(first.lead_hour),
        "valid_time_utc": first.valid_time.isoformat(),
        "gpu_path": str(first.gpu_path),
        "cpu_path": str(first.cpu_path),
        "field_count": int(len(field_results)),
        "fields": field_results,
        "ranked_by_max_abs_diff": ranked,
        "largest_field": top,
        "already_diverged_at_first_output": bool(top and float(top["max_abs_diff"]) > 1.0e-6),
        "note": "All matching numeric wrfout variables are used as the available on-disk proxy for State fields.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT)
    parser.add_argument("--cpu-root", type=Path, default=DEFAULT_CPU_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_HOURLY_OUTPUT)
    parser.add_argument("--first-hour-output", type=Path, default=DEFAULT_FIRST_HOUR_OUTPUT)
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hourly = build_hourly_payload(gpu_root=args.gpu_root, cpu_root=args.cpu_root, fields=args.fields)
    first_hour = build_first_hour_payload(gpu_root=args.gpu_root, cpu_root=args.cpu_root)
    write_json(args.output, hourly)
    write_json(args.first_hour_output, first_hour)
    summary = {
        "hourly_output": str(args.output),
        "first_hour_output": str(args.first_hour_output),
        "pair_count": hourly["pair_count"],
        "lead_hours": hourly["lead_hours"],
        "first_hour_largest_field": None if first_hour["largest_field"] is None else first_hour["largest_field"]["field"],
        "first_hour_largest_max_abs_diff": None if first_hour["largest_field"] is None else first_hour["largest_field"]["max_abs_diff"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
