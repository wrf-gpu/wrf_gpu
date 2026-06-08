"""Shared CPU-only helpers for Canary existing-data inventory and scoring."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset


WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")
CASE_RE = re.compile(r"(20\d{6}_\d{2}z)")

SURFACE_VARS = ("T2", "U10", "V10", "PSFC", "RAINNC")
THREED_VARS = ("T", "U", "V", "W", "QVAPOR", "QCLOUD", "QRAIN", "P", "PB", "PH", "PHB")
DEFAULT_SCORE_VARS = ("T2", "U10", "V10")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_wrfout_time(path: Path) -> datetime | None:
    m = WRFOUT_RE.match(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def parse_case_id(text: str) -> str | None:
    m = CASE_RE.search(text)
    return m.group(1) if m else None


def parse_iso_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_init_from_case_id(case_id: str | None) -> datetime | None:
    if not case_id:
        return None
    try:
        date_s, hour_s = case_id.split("_")
        hour = int(hour_s.replace("z", ""))
        return datetime(
            int(date_s[:4]),
            int(date_s[4:6]),
            int(date_s[6:8]),
            hour,
            tzinfo=timezone.utc,
        )
    except Exception:
        return None


def list_wrfout_files(run_dir: Path, domain: str) -> list[tuple[datetime, Path]]:
    out: list[tuple[datetime, Path]] = []
    for path in sorted(run_dir.glob(f"wrfout_{domain}_*")):
        if not path.is_file():
            continue
        valid = parse_wrfout_time(path)
        if valid is not None:
            out.append((valid, path))
    return sorted(out)


def read_var(path: Path, name: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        if name not in ds.variables:
            raise KeyError(f"{name} not present in {path}")
        var = ds.variables[name]
        data = np.asarray(np.ma.filled(var[:], np.nan), dtype=np.float64)
        if var.dimensions and var.dimensions[0] == "Time":
            data = data[0]
        return data


def trim_boundary(arr: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return arr
    if arr.ndim < 2:
        return arr
    if arr.shape[-1] <= 2 * width or arr.shape[-2] <= 2 * width:
        return arr[...,:0,:0]
    return arr[..., width:-width, width:-width]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize_deltas(delta: np.ndarray) -> dict[str, Any]:
    if delta.size == 0:
        return {"status": "NO_DATA", "n": 0}
    abs_delta = np.abs(delta)
    return {
        "status": "OK",
        "n": int(delta.size),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "bias": float(np.mean(delta)),
        "mae": float(np.mean(abs_delta)),
        "p95_abs": float(np.percentile(abs_delta, 95)),
        "p99_abs": float(np.percentile(abs_delta, 99)),
        "max_abs": float(np.max(abs_delta)),
    }


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
