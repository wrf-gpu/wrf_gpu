#!/usr/bin/env python3
"""v0.22 small-grid operational-relaxed skill-band harness.

This is a model-free post-processor: it reads existing WRF-style wrfout NetCDF
directories and optional run-side guard JSONs. It does not import JAX, CUDA, or
gpuwrf forecast code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


BANDS_PATH = SRC / "gpuwrf" / "validation" / "operational_relaxed_bands.json"
DIVERGENCE_METRIC_PATH = ROOT / "proofs" / "perf" / "v015" / "fp32_oracles" / "divergence_growth_metric.py"
WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")
CASE_INIT_RE = re.compile(r"(20\d{6})_(\d{2})z", re.IGNORECASE)
DEFAULT_FIELDS = ("T2", "U10", "V10")
DEFAULT_LEADS = (24, 72, 120)
SURFACE_UNITS = {"T2": "K", "U10": "m s-1", "V10": "m s-1", "PSFC": "Pa", "PBLH": "m"}


def _load_divergence_metric() -> Any:
    spec = importlib.util.spec_from_file_location("v022_divergence_growth_metric", DIVERGENCE_METRIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import divergence metric from {DIVERGENCE_METRIC_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return clean_float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_wrfout_time(path: Path) -> datetime | None:
    match = WRFOUT_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def parse_init_from_text(text: str) -> datetime | None:
    match = CASE_INIT_RE.search(text)
    if not match:
        return None
    date_s, hour_s = match.groups()
    return datetime(int(date_s[:4]), int(date_s[4:6]), int(date_s[6:8]), int(hour_s), tzinfo=timezone.utc)


def parse_init(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    if "_" in text and "T" not in text:
        out = datetime.strptime(text, "%Y-%m-%d_%H:%M:%S")
        return out.replace(tzinfo=timezone.utc)
    out = datetime.fromisoformat(text)
    return out.replace(tzinfo=timezone.utc) if out.tzinfo is None else out.astimezone(timezone.utc)


def discover_wrfouts(run_dir: Path, domain: str) -> dict[datetime, Path]:
    out: dict[datetime, Path] = {}
    for path in sorted(run_dir.glob(f"wrfout_{domain}_*")):
        if not path.is_file():
            continue
        valid = parse_wrfout_time(path)
        if valid is not None:
            out[valid] = path
    return out


def infer_init_time(
    *,
    explicit: str | None,
    directories: list[Path],
    maps: list[dict[datetime, Path]],
) -> tuple[datetime, str]:
    if explicit:
        return parse_init(explicit), "--init"
    for directory in directories:
        parsed = parse_init_from_text(str(directory))
        if parsed is not None:
            return parsed, "directory-name"
    all_times = sorted({time for mapping in maps for time in mapping})
    if not all_times:
        raise RuntimeError("no wrfout files found for init-time inference")
    return all_times[0], "earliest-wrfout-fallback"


def lead_time(valid_time: datetime, init_time: datetime) -> int:
    return int(round((valid_time - init_time).total_seconds() / 3600.0))


def file_for_lead(files: dict[datetime, Path], init_time: datetime, lead_hour: int) -> Path | None:
    return files.get(init_time + timedelta(hours=int(lead_hour)))


def read_var(path: Path, name: str) -> np.ndarray:
    with Dataset(path, "r") as dataset:
        if name not in dataset.variables:
            raise KeyError(f"{path.name} missing {name}")
        variable = dataset.variables[name]
        raw = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
        return np.asarray(np.ma.filled(raw, np.nan))


def list_numeric_variables(path: Path) -> list[str]:
    with Dataset(path, "r") as dataset:
        return [
            name
            for name, variable in dataset.variables.items()
            if np.dtype(variable.dtype).kind in {"b", "i", "u", "f", "c"}
        ]


def _read_2d(dataset: Dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    raw = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
    data = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{name} is not 2-D after Time squeeze: shape={data.shape}")
    return data


def _window_sum(mask: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    radius = window_size // 2
    padded = np.pad(mask.astype(np.float64), ((radius, radius), (radius, radius)), mode="constant", constant_values=0.0)
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    y0 = np.arange(mask.shape[0])
    x0 = np.arange(mask.shape[1])
    y1 = y0 + window_size
    x1 = x0 + window_size
    return integral[y1[:, None], x1[None, :]] - integral[y0[:, None], x1[None, :]] - integral[y1[:, None], x0[None, :]] + integral[y0[:, None], x0[None, :]]


def compute_fractions_skill_score(
    forecast_precip_mm: np.ndarray,
    observed_precip_mm: np.ndarray,
    *,
    threshold_mm: float,
    window_size: int,
) -> dict[str, Any]:
    """Dense-grid FSS, formula-compatible with forecast_vs_obs.compute_fractions_skill_score."""

    forecast = np.asarray(forecast_precip_mm, dtype=np.float64)
    observed = np.asarray(observed_precip_mm, dtype=np.float64)
    if forecast.shape != observed.shape:
        raise ValueError(f"forecast and observed grids must have the same shape, got {forecast.shape} and {observed.shape}")
    finite_forecast = np.isfinite(forecast)
    finite_observed = np.isfinite(observed)
    forecast_mask = (forecast >= threshold_mm) & finite_forecast
    observed_mask = (observed >= threshold_mm) & finite_observed
    denominator_cells = float(window_size * window_size)
    forecast_fraction = _window_sum(forecast_mask, window_size) / denominator_cells
    observed_fraction = _window_sum(observed_mask, window_size) / denominator_cells
    numerator = float(np.mean((forecast_fraction - observed_fraction) ** 2))
    denominator = float(np.mean(forecast_fraction**2 + observed_fraction**2))
    if denominator == 0.0:
        fss = 1.0 if numerator == 0.0 else 0.0
    else:
        fss = 1.0 - numerator / denominator
    return {
        "schema": "M7PrecipFSS",
        "schema_version": 1,
        "threshold_mm": float(threshold_mm),
        "window_size_cells": int(window_size),
        "forecast_grid_shape": [int(value) for value in forecast.shape],
        "forecast_exceedance_cells": int(forecast_mask.sum()),
        "observed_exceedance_cells": int(observed_mask.sum()),
        "observed_finite_cells": int(finite_observed.sum()),
        "fss": float(np.clip(fss, 0.0, 1.0)),
        "status": "OK",
    }


def read_wrf_precip_delta(start_wrfout_path: Path, end_wrfout_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense WRF RAINNC+RAINC accumulation delta and lat/lon grid."""

    def total(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with Dataset(path, "r") as dataset:
            lats = _read_2d(dataset, "XLAT")
            lons = _read_2d(dataset, "XLONG")
            rain = np.zeros_like(_read_2d(dataset, "RAINNC"))
            for name in ("RAINNC", "RAINC"):
                if name in dataset.variables:
                    rain = rain + _read_2d(dataset, name)
            return rain, lats, lons

    start_total, lats, lons = total(start_wrfout_path)
    end_total, end_lats, end_lons = total(end_wrfout_path)
    if start_total.shape != end_total.shape or lats.shape != end_lats.shape or lons.shape != end_lons.shape:
        raise ValueError("start and end wrfout grids do not match")
    return np.maximum(end_total - start_total, 0.0), lats, lons


def field_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "pass": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
        }
    finite = np.isfinite(candidate) & np.isfinite(reference)
    count = int(finite.sum())
    if count == 0:
        return {"status": "NO_FINITE_PAIRS", "pass": False, "shape": list(candidate.shape), "sample_count": 0}
    c = candidate[finite]
    r = reference[finite]
    diff = c - r
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_std = float(np.std(r))
    cand_std = float(np.std(c))
    if count >= 2 and ref_std > 0.0 and cand_std > 0.0:
        pearson = float(np.corrcoef(c, r)[0, 1])
    else:
        pearson = None
    return {
        "status": "OK",
        "pass": True,
        "shape": list(candidate.shape),
        "sample_count": count,
        "finite_pair_fraction": float(count / candidate.size) if candidate.size else 1.0,
        "rmse": rmse,
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "pearson_r": pearson,
        "reference_std": ref_std,
        "candidate_std": cand_std,
        "nrmse": None if ref_std <= 0.0 else float(rmse / ref_std),
    }


def finite_summary(paths: list[Path], *, fields: tuple[str, ...] | None = None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_nan = 0
    total_inf = 0
    for path in paths:
        file_record: dict[str, Any] = {"path": str(path), "fields": {}, "nan": 0, "inf": 0, "readable": False}
        try:
            names = list(fields) if fields else list_numeric_variables(path)
            for name in names:
                try:
                    values = read_var(path, name)
                except KeyError:
                    file_record["fields"][name] = {"status": "MISSING", "nan": 0, "inf": 0}
                    continue
                nan_count = int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0
                inf_count = int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.number) else 0
                file_record["fields"][name] = {"status": "OK", "nan": nan_count, "inf": inf_count}
                file_record["nan"] += nan_count
                file_record["inf"] += inf_count
            file_record["readable"] = True
        except Exception as exc:
            file_record["error"] = f"{type(exc).__name__}: {exc}"
        total_nan += int(file_record["nan"])
        total_inf += int(file_record["inf"])
        files.append(file_record)
    return {
        "status": "PASS" if total_nan == 0 and total_inf == 0 and all(row["readable"] for row in files) else "FAIL",
        "nan": total_nan,
        "inf": total_inf,
        "pass": bool(total_nan == 0 and total_inf == 0 and all(row["readable"] for row in files)),
        "file_count": int(len(files)),
        "files": files,
    }


def _total_from_file(path: Path, names: tuple[str, ...]) -> float | None:
    total = 0.0
    with Dataset(path, "r") as dataset:
        for name in names:
            if name not in dataset.variables:
                return None
            variable = dataset.variables[name]
            raw = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
            total += float(np.nansum(np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)))
    return total


def _mass_array(path: Path) -> np.ndarray | None:
    try:
        mu = read_var(path, "MU")
        mub = read_var(path, "MUB")
    except KeyError:
        return None
    return np.asarray(mu, dtype=np.float64) + np.asarray(mub, dtype=np.float64)


def conservation_proxy(files: dict[datetime, Path]) -> dict[str, Any]:
    ordered = [path for _, path in sorted(files.items())]
    if len(ordered) < 2:
        return {"status": "UNAVAILABLE", "pass": False, "reason": "fewer than two wrfout files"}
    first = ordered[0]
    last = ordered[-1]
    dry0 = _total_from_file(first, ("MU", "MUB"))
    dry1 = _total_from_file(last, ("MU", "MUB"))
    if dry0 is None or dry1 is None:
        return {"status": "UNAVAILABLE", "pass": False, "reason": "MU/MUB missing from wrfout"}
    dry_resid = 0.0 if dry0 == 0.0 else abs(dry1 - dry0) / abs(dry0)

    water_resid = None
    energy_resid = None
    mass0 = _mass_array(first)
    mass1 = _mass_array(last)
    if mass0 is not None and mass1 is not None:
        try:
            qv0 = read_var(first, "QVAPOR")
            qv1 = read_var(last, "QVAPOR")
            water0 = float(np.nansum(np.asarray(qv0, dtype=np.float64) * mass0[np.newaxis, :, :]))
            water1 = float(np.nansum(np.asarray(qv1, dtype=np.float64) * mass1[np.newaxis, :, :]))
            water_resid = None if water0 == 0.0 else abs(water1 - water0) / abs(water0)
        except Exception:
            water_resid = None
        try:
            t0 = read_var(first, "T")
            t1 = read_var(last, "T")
            theta0 = np.asarray(t0, dtype=np.float64) + 300.0
            theta1 = np.asarray(t1, dtype=np.float64) + 300.0
            energy0 = float(np.nansum(theta0 * mass0[np.newaxis, :, :]))
            energy1 = float(np.nansum(theta1 * mass1[np.newaxis, :, :]))
            energy_resid = None if energy0 == 0.0 else abs(energy1 - energy0) / abs(energy0)
        except Exception:
            energy_resid = None

    return {
        "status": "PASS",
        "method": "wrfout_proxy_domain_integral",
        "pass": True,
        "initial_wrfout": str(first),
        "final_wrfout": str(last),
        "dry_mass_relative_residual": float(dry_resid),
        "water_relative_residual": clean_float(water_resid),
        "moist_static_energy_relative_residual": clean_float(energy_resid),
        "note": "Proxy from wrfout MU/MUB, QVAPOR, and T. Full on-device closure should be supplied via --candidate-guards.",
    }


def compare_conservation(candidate: dict[str, Any], strict: dict[str, Any] | None, bands: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("status") not in ("PASS", "OK"):
        return {**candidate, "pass": False}
    margins = bands["hard_guards"]["conservation"]
    fields = [
        ("dry_mass_relative_residual", "dry_mass_relative_residual_margin_vs_strict"),
        ("water_relative_residual", "water_relative_residual_margin_vs_strict"),
        ("moist_static_energy_relative_residual", "moist_static_energy_relative_residual_margin_vs_strict"),
    ]
    out = dict(candidate)
    out["field_pass"] = {}
    all_pass = True
    for field, margin_key in fields:
        value = clean_float(candidate.get(field))
        if value is None:
            out["field_pass"][field] = {"status": "UNAVAILABLE", "pass": False}
            all_pass = False
            continue
        strict_value = clean_float(strict.get(field)) if strict else None
        if strict_value is None:
            threshold = float(margins[margin_key])
            basis = "absolute_margin_no_strict_guard"
        else:
            threshold = strict_value + float(margins[margin_key])
            basis = "strict_plus_margin"
        passed = bool(value <= threshold)
        out["field_pass"][field] = {"value": value, "threshold": threshold, "basis": basis, "pass": passed}
        all_pass = all_pass and passed
    out["pass"] = bool(all_pass)
    out["status"] = "PASS" if all_pass else "FAIL"
    return out


def compare_wrfout_exact(left: Path, right: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    passed = True
    with Dataset(left, "r") as left_ds, Dataset(right, "r") as right_ds:
        for name in sorted(set(left_ds.variables) | set(right_ds.variables)):
            if name not in left_ds.variables or name not in right_ds.variables:
                fields[name] = {"status": "MISSING", "pass": False}
                passed = False
                continue
            left_var = left_ds.variables[name]
            right_var = right_ds.variables[name]
            if tuple(left_var.shape) != tuple(right_var.shape):
                fields[name] = {
                    "status": "SHAPE_MISMATCH",
                    "left_shape": list(left_var.shape),
                    "right_shape": list(right_var.shape),
                    "pass": False,
                }
                passed = False
                continue
            left_values = np.asarray(np.ma.filled(left_var[:], np.nan))
            right_values = np.asarray(np.ma.filled(right_var[:], np.nan))
            if left_values.dtype.kind in {"S", "U"} or right_values.dtype.kind in {"S", "U"}:
                equal = bool(np.array_equal(left_values, right_values))
                max_abs = 0.0 if equal else float("inf")
            else:
                delta = right_values.astype(np.float64) - left_values.astype(np.float64)
                equal = bool(np.array_equal(left_values, right_values, equal_nan=True))
                max_abs = float(np.nanmax(np.abs(delta))) if delta.size else 0.0
            fields[name] = {"status": "OK", "max_abs": max_abs, "pass": equal}
            passed = passed and equal
    return {"status": "PASS" if passed else "FAIL", "left": str(left), "right": str(right), "pass": bool(passed), "fields": fields}


def restart_equivalence(
    *,
    candidate_files: dict[datetime, Path],
    restart_files: dict[datetime, Path] | None,
    init_time: datetime,
    leads: tuple[int, ...],
    guard_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    guard_restart = ((guard_payload or {}).get("hard_guards") or {}).get("restart_equivalence")
    if guard_restart is None:
        guard_restart = (guard_payload or {}).get("restart_equivalence")
    if guard_restart is not None:
        out = dict(guard_restart)
        out.setdefault("source", "candidate-guards")
        out["pass"] = bool(out.get("pass", out.get("status") == "PASS"))
        out["status"] = "PASS" if out["pass"] else "FAIL"
        return out
    if restart_files is None:
        return {"status": "UNAVAILABLE", "pass": False, "reason": "--restart-dir or guard restart_equivalence not supplied"}
    present = [lead for lead in leads if file_for_lead(candidate_files, init_time, lead) and file_for_lead(restart_files, init_time, lead)]
    if not present:
        return {"status": "UNAVAILABLE", "pass": False, "reason": "no common candidate/restart lead wrfout"}
    lead = max(present)
    comparison = compare_wrfout_exact(file_for_lead(candidate_files, init_time, lead), file_for_lead(restart_files, init_time, lead))  # type: ignore[arg-type]
    return {**comparison, "lead_hours": int(lead)}


def normalize_clamp_guard(candidate_guard: dict[str, Any] | None, strict_guard: dict[str, Any] | None) -> dict[str, Any]:
    source = ((candidate_guard or {}).get("hard_guards") or {}).get("clamp_limiter_audit")
    if source is None:
        source = (candidate_guard or {}).get("clamp_limiter_audit")
    if source is None:
        return {"status": "UNAVAILABLE", "pass": False, "reason": "--candidate-guards missing clamp_limiter_audit"}
    out = dict(source)
    out.setdefault("source", "candidate-guards")
    out["pass"] = bool(out.get("pass", out.get("status") == "PASS"))
    out["status"] = "PASS" if out["pass"] else "FAIL"
    return out


def normalize_transfer_guard(candidate_guard: dict[str, Any] | None) -> dict[str, Any]:
    source = ((candidate_guard or {}).get("hard_guards") or {}).get("transfer_audit")
    if source is None:
        source = (candidate_guard or {}).get("transfer_audit")
    if source is None:
        return {"status": "UNAVAILABLE", "pass": False, "reason": "--candidate-guards missing transfer_audit"}
    out = dict(source)
    transfers = int(out.get("new_in_loop_transfers", 0))
    out["pass"] = bool(out.get("pass", transfers == 0))
    out["status"] = "PASS" if out["pass"] else "FAIL"
    return out


def hard_guards(
    *,
    candidate_files: dict[datetime, Path],
    strict_files: dict[datetime, Path] | None,
    restart_files: dict[datetime, Path] | None,
    init_time: datetime,
    leads: tuple[int, ...],
    fields: tuple[str, ...],
    finite_scope: str,
    bands: dict[str, Any],
    candidate_guard: dict[str, Any] | None,
    strict_guard: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_files = [file_for_lead(candidate_files, init_time, lead) for lead in leads]
    selected_existing = [path for path in selected_files if path is not None]
    finite_fields = fields if finite_scope == "required" else None
    finite = finite_summary(selected_existing, fields=finite_fields)
    candidate_conservation = ((candidate_guard or {}).get("hard_guards") or {}).get("conservation")
    if candidate_conservation is None:
        candidate_conservation = (candidate_guard or {}).get("conservation")
    if candidate_conservation is None:
        candidate_conservation = conservation_proxy(candidate_files)
    strict_conservation = None
    if strict_guard:
        strict_conservation = ((strict_guard.get("hard_guards") or {}).get("conservation") or strict_guard.get("conservation"))
    if strict_conservation is None and strict_files is not None:
        strict_conservation = conservation_proxy(strict_files)
    conservation = compare_conservation(dict(candidate_conservation), dict(strict_conservation) if strict_conservation else None, bands)
    restart = restart_equivalence(
        candidate_files=candidate_files,
        restart_files=restart_files,
        init_time=init_time,
        leads=leads,
        guard_payload=candidate_guard,
    )
    clamp = normalize_clamp_guard(candidate_guard, strict_guard)
    transfer = normalize_transfer_guard(candidate_guard)
    output_cadence = {
        "status": "PASS" if len(selected_existing) == len(leads) else "FAIL",
        "pass": bool(len(selected_existing) == len(leads)),
        "requested_leads": list(leads),
        "present_leads": [lead for lead in leads if file_for_lead(candidate_files, init_time, lead) is not None],
    }
    all_pass = bool(
        finite.get("pass")
        and conservation.get("pass")
        and restart.get("pass")
        and clamp.get("pass")
        and transfer.get("pass")
        and output_cadence.get("pass")
    )
    return {
        "finite": finite,
        "conservation": conservation,
        "clamp_limiter_audit": clamp,
        "restart_equivalence": restart,
        "output_cadence": output_cadence,
        "transfer_audit": transfer,
        "all_hard_guards_pass": all_pass,
    }


def _band_value(table: dict[str, Any], field: str, lead: int, key: str | None = None) -> float | None:
    node = table.get(field)
    if node is None:
        return None
    if key is not None:
        node = node.get(key)
        if node is None:
            return None
    if isinstance(node, dict) and "lead_hours" in node:
        node = node["lead_hours"]
    if isinstance(node, dict):
        return clean_float(node.get(str(int(lead))))
    return clean_float(node)


def score_field_at_lead(
    *,
    field: str,
    lead: int,
    candidate_path: Path | None,
    cpu_path: Path | None,
    strict_path: Path | None,
    bands: dict[str, Any],
    divergence: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "field": field,
        "lead_hours": int(lead),
        "units": SURFACE_UNITS.get(field, ""),
        "candidate_wrfout": str(candidate_path) if candidate_path else None,
        "cpu_wrf_wrfout": str(cpu_path) if cpu_path else None,
        "strict_wrfout": str(strict_path) if strict_path else None,
    }
    if candidate_path is None:
        record.update({"status": "MISSING_CANDIDATE", "validation_band_pass": False, "tier_o_pass": False})
        return record
    if cpu_path is None:
        record.update({"status": "MISSING_CPU_WRF", "validation_band_pass": False, "tier_o_pass": False})
        return record
    try:
        candidate = read_var(candidate_path, field)
        cpu = read_var(cpu_path, field)
    except KeyError as exc:
        record.update({"status": "MISSING_FIELD", "reason": str(exc), "validation_band_pass": False, "tier_o_pass": False})
        return record
    vs_cpu = field_stats(candidate, cpu)
    record["vs_cpu_wrf"] = vs_cpu
    cpu_limit = _band_value(bands["surface_cpu_wrf_rmse_bands"], field, lead)
    cpu_rmse = clean_float(vs_cpu.get("rmse"))
    validation_pass = bool(cpu_rmse is not None and cpu_limit is not None and cpu_rmse <= cpu_limit and vs_cpu.get("pass"))
    record["validation_band"] = {"rmse_max": cpu_limit, "pass": validation_pass}
    record["validation_band_pass"] = validation_pass

    tier_reasons: list[str] = []
    strict_vs_cpu = None
    vs_strict = None
    if strict_path is not None:
        try:
            strict = read_var(strict_path, field)
            vs_strict = field_stats(candidate, strict)
            strict_vs_cpu = field_stats(strict, cpu)
            record["vs_strict"] = vs_strict
            record["strict_vs_cpu_wrf"] = strict_vs_cpu
        except KeyError as exc:
            tier_reasons.append(f"strict missing {field}: {exc}")
    else:
        tier_reasons.append("strict baseline unavailable")

    tier_pass = validation_pass
    tier_table = bands.get("tier_o_surface_delta_bands", {})
    if field in tier_table:
        delta_limit = _band_value(tier_table, field, lead, "delta_rmse_max")
        delta_rmse = clean_float((vs_strict or {}).get("rmse"))
        delta_pass = bool(delta_rmse is not None and delta_limit is not None and delta_rmse <= delta_limit)
        record["tier_o_delta_band"] = {"delta_rmse": delta_rmse, "delta_rmse_max": delta_limit, "pass": delta_pass}
        tier_pass = tier_pass and delta_pass
        if not delta_pass:
            tier_reasons.append(f"{field} delta_rmse={delta_rmse} > {delta_limit}")
        if field == "T2":
            pearson_min = _band_value(tier_table, field, lead, "pearson_min_vs_cpu_wrf")
            pearson = clean_float(vs_cpu.get("pearson_r"))
            pearson_pass = True if pearson_min is None else bool(pearson is not None and pearson >= pearson_min)
            nrmse_max = _band_value(tier_table, field, lead, "nrmse_max_vs_cpu_wrf")
            nrmse = clean_float(vs_cpu.get("nrmse"))
            nrmse_pass = True if nrmse_max is None else bool(nrmse is not None and nrmse <= nrmse_max)
            record["tier_o_cpu_shape_band"] = {
                "pearson_r": pearson,
                "pearson_min": pearson_min,
                "pearson_pass": pearson_pass,
                "nrmse": nrmse,
                "nrmse_max": nrmse_max,
                "nrmse_pass": nrmse_pass,
            }
            tier_pass = tier_pass and pearson_pass and nrmse_pass
        if field in ("U10", "V10"):
            ratio_cap = clean_float(tier_table[field].get("rmse_ratio_to_strict_max"))
            strict_rmse = clean_float((strict_vs_cpu or {}).get("rmse"))
            ratio = None
            if strict_rmse is not None and strict_rmse > 0.0 and cpu_rmse is not None:
                ratio = cpu_rmse / strict_rmse
            elif strict_rmse == 0.0 and cpu_rmse == 0.0:
                ratio = 1.0
            ratio_pass = bool(ratio is not None and ratio_cap is not None and ratio <= ratio_cap)
            div_record = divergence.get(field, {})
            div_regime = div_record.get("regime")
            div_pass = bool(div_regime == bands["divergence_growth"]["required_regime"])
            record["tier_o_wind_band"] = {
                "candidate_cpu_rmse": cpu_rmse,
                "strict_cpu_rmse": strict_rmse,
                "rmse_ratio_to_strict": ratio,
                "rmse_ratio_to_strict_max": ratio_cap,
                "ratio_pass": ratio_pass,
                "divergence_regime": div_regime,
                "divergence_pass": div_pass,
            }
            tier_pass = tier_pass and ratio_pass and div_pass
            if not ratio_pass:
                tier_reasons.append(f"{field} rmse_ratio_to_strict={ratio} > {ratio_cap}")
            if not div_pass:
                tier_reasons.append(f"{field} divergence_regime={div_regime} != BOUNDED")
    record["tier_o_pass"] = bool(tier_pass)
    record["tier_o_reject_reasons"] = tier_reasons
    record["status"] = "PASS" if validation_pass else "FAIL"
    return record


def common_series(
    *,
    candidate_files: dict[datetime, Path],
    reference_files: dict[datetime, Path],
    init_time: datetime,
    fields: tuple[str, ...],
    max_lead: int,
) -> dict[str, Any]:
    dgm = _load_divergence_metric()
    leads = sorted(
        lead_time(time, init_time)
        for time in set(candidate_files) & set(reference_files)
        if 0 <= lead_time(time, init_time) <= int(max_lead)
    )
    out: dict[str, Any] = {"leads": leads, "fields": {}, "status": "OK" if leads else "NO_COMMON_LEADS"}
    if not leads:
        return out
    for field in fields:
        candidate_arrays = []
        reference_arrays = []
        missing = False
        for lead in leads:
            cpath = file_for_lead(candidate_files, init_time, lead)
            rpath = file_for_lead(reference_files, init_time, lead)
            try:
                candidate_arrays.append(read_var(cpath, field))  # type: ignore[arg-type]
                reference_arrays.append(read_var(rpath, field))  # type: ignore[arg-type]
            except Exception as exc:
                out["fields"][field] = {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}", "pass": False}
                missing = True
                break
        if missing:
            continue
        candidate_stack = np.stack(candidate_arrays, axis=0)
        reference_stack = np.stack(reference_arrays, axis=0)
        divergence = dgm.rmse_series(candidate_stack, reference_stack)
        envelope = max(float(np.nanmedian(np.maximum(divergence, 0.0))), float(np.nanstd(reference_stack)) * 1.0e-6, 1.0e-12)
        result = dgm.classify_divergence_growth(
            np.asarray(leads, dtype=np.float64),
            np.asarray(divergence, dtype=np.float64),
            envelope=envelope,
            field_name=field,
        )
        out["fields"][field] = {
            "status": "OK",
            "regime": result.regime,
            "passes_flag": bool(result.passes),
            "pass": bool(result.regime == "BOUNDED"),
            "divergence_series": [float(value) for value in result.divergence],
            "envelope": float(result.envelope),
            "early_slope_per_lead": float(result.early_slope),
            "late_slope_per_lead": float(result.late_slope),
            "late_slope_ratio": float(result.late_slope_ratio),
            "final_over_envelope": float(result.final_over_envelope),
            "max_over_envelope": float(result.max_over_envelope),
            "detail": result.detail,
        }
    return out


def score_precip_fss(
    *,
    lead: int,
    candidate_files: dict[datetime, Path],
    cpu_files: dict[datetime, Path],
    init_time: datetime,
    bands: dict[str, Any],
    grid_resolution: str,
) -> dict[str, Any]:
    start_candidate = file_for_lead(candidate_files, init_time, 0)
    end_candidate = file_for_lead(candidate_files, init_time, lead)
    start_cpu = file_for_lead(cpu_files, init_time, 0)
    end_cpu = file_for_lead(cpu_files, init_time, lead)
    if not (start_candidate and end_candidate and start_cpu and end_cpu):
        return {"status": "UNAVAILABLE", "pass": False, "reason": "lead-0 and target lead wrfouts are required for dense precip delta FSS"}
    try:
        candidate_precip, _, _ = read_wrf_precip_delta(start_candidate, end_candidate)
        cpu_precip, _, _ = read_wrf_precip_delta(start_cpu, end_cpu)
    except Exception as exc:
        return {"status": "UNAVAILABLE", "pass": False, "reason": f"{type(exc).__name__}: {exc}"}
    fss_bands = bands["precip_fss"]
    window = int(fss_bands["window_size_cells"].get(grid_resolution, fss_bands["window_size_cells"]["default"]))
    minima = fss_bands["minimum"][str(int(lead))]
    rows = []
    all_pass = True
    for threshold, minimum in zip(fss_bands["threshold_mm"], minima):
        result = compute_fractions_skill_score(candidate_precip, cpu_precip, threshold_mm=float(threshold), window_size=window)
        passed = bool(result["fss"] >= float(minimum))
        rows.append({**result, "minimum": float(minimum), "pass": passed})
        all_pass = all_pass and passed
    return {"status": "PASS" if all_pass else "FAIL", "pass": bool(all_pass), "fss": rows, "baseline": "dense_cpu_wrf"}


def build_domain_payload(args: argparse.Namespace, domain: str, bands: dict[str, Any]) -> dict[str, Any]:
    candidate_dir = Path(args.candidate_dir)
    cpu_dir = Path(args.cpu_wrf_dir)
    strict_dir = Path(args.strict_dir) if args.strict_dir else None
    restart_dir = Path(args.restart_dir) if args.restart_dir else None
    candidate_files = discover_wrfouts(candidate_dir, domain)
    cpu_files = discover_wrfouts(cpu_dir, domain)
    strict_files = discover_wrfouts(strict_dir, domain) if strict_dir else None
    restart_files = discover_wrfouts(restart_dir, domain) if restart_dir else None
    dirs = [candidate_dir, cpu_dir] + ([strict_dir] if strict_dir else [])
    maps = [candidate_files, cpu_files] + ([strict_files] if strict_files else [])
    init_time, init_source = infer_init_time(explicit=args.init, directories=[path for path in dirs if path], maps=[m for m in maps if m])
    leads = tuple(int(value) for value in args.leads)
    fields = tuple(args.fields)
    candidate_guard = load_json(args.candidate_guards)
    strict_guard = load_json(args.strict_guards)
    guards = hard_guards(
        candidate_files=candidate_files,
        strict_files=strict_files,
        restart_files=restart_files,
        init_time=init_time,
        leads=leads,
        fields=fields,
        finite_scope=args.finite_scope,
        bands=bands,
        candidate_guard=candidate_guard,
        strict_guard=strict_guard,
    )
    divergence_reference = strict_files if strict_files else cpu_files
    divergence = common_series(
        candidate_files=candidate_files,
        reference_files=divergence_reference,
        init_time=init_time,
        fields=tuple(field for field in fields if field in bands["divergence_growth"]["fields"]),
        max_lead=max(leads),
    )
    divergence_fields = divergence.get("fields", {})
    lead_payloads: list[dict[str, Any]] = []
    reject_reasons: list[str] = []
    for lead in leads:
        field_rows: dict[str, Any] = {}
        for field in fields:
            if field == "RAINNC":
                field_rows[field] = score_precip_fss(
                    lead=lead,
                    candidate_files=candidate_files,
                    cpu_files=cpu_files,
                    init_time=init_time,
                    bands=bands,
                    grid_resolution=args.grid_resolution,
                )
                continue
            field_rows[field] = score_field_at_lead(
                field=field,
                lead=lead,
                candidate_path=file_for_lead(candidate_files, init_time, lead),
                cpu_path=file_for_lead(cpu_files, init_time, lead),
                strict_path=file_for_lead(strict_files, init_time, lead) if strict_files else None,
                bands=bands,
                divergence=divergence_fields,
            )
        validation_pass = all(bool(row.get("validation_band_pass", row.get("pass"))) for row in field_rows.values())
        tier_o_pass = all(bool(row.get("tier_o_pass", row.get("pass"))) for row in field_rows.values())
        lead_reasons = []
        for field, row in field_rows.items():
            if not bool(row.get("validation_band_pass", row.get("pass"))):
                lead_reasons.append(f"lead {lead} {field} validation band failed")
            if args.gate_mode == "operational-relaxed" and not bool(row.get("tier_o_pass", row.get("pass"))):
                lead_reasons.extend([f"lead {lead} {field}: {reason}" for reason in row.get("tier_o_reject_reasons", [])])
        lead_payload = {
            "schema": "OperationalRelaxedGate",
            "schema_version": 1,
            "lever_id": args.lever_id,
            "lever_config": json.loads(args.lever_config_json),
            "case_id": args.case_id,
            "domain": domain,
            "lead_hours": int(lead),
            "baselines_present": [
                name
                for name, present in {
                    "cpu_wrf": bool(cpu_files),
                    "strict_gpu": bool(strict_files),
                    "aemet_obs": False,
                }.items()
                if present
            ],
            "baselines_unavailable": [
                name
                for name, present in {
                    "cpu_wrf": bool(cpu_files),
                    "strict_gpu": bool(strict_files),
                    "aemet_obs": False,
                }.items()
                if not present
            ],
            "hard_guards": guards,
            "skill_bands": field_rows,
            "divergence_growth": divergence_fields,
            "validation_band_pass": bool(validation_pass),
            "tier_o_band_pass": bool(tier_o_pass),
            "verdict": "TIER_O_ACCEPTED" if validation_pass and tier_o_pass and guards["all_hard_guards_pass"] else "TIER_O_REJECTED",
            "reject_reasons": lead_reasons,
            "enric_signoff": {"required": args.gate_mode == "operational-relaxed", "recorded": False, "ref": None},
            "provenance": {
                "candidate_dir": str(candidate_dir),
                "cpu_wrf_dir": str(cpu_dir),
                "strict_gpu_dir": str(strict_dir) if strict_dir else None,
                "restart_dir": str(restart_dir) if restart_dir else None,
                "init_time_utc": init_time.isoformat(),
                "init_time_source": init_source,
                "bands_path": str(BANDS_PATH),
                "contract_ref": bands["contract_ref"],
                "wrf_fortran_reference": args.wrf_fortran_ref,
            },
        }
        if lead_reasons:
            reject_reasons.extend(lead_reasons)
        lead_payloads.append(lead_payload)

    backlog_pass = all(item["validation_band_pass"] for item in lead_payloads)
    tier_o_pass = all(item["tier_o_band_pass"] for item in lead_payloads) and bool(guards["all_hard_guards_pass"])
    if args.gate_mode == "operational-relaxed" and not guards["all_hard_guards_pass"]:
        reject_reasons.append("one or more hard guards failed/unavailable")
    verdict_pass = tier_o_pass if args.gate_mode == "operational-relaxed" else backlog_pass
    return {
        "schema": "V022OperationalRelaxedGateDomainRollup",
        "schema_version": 1,
        "gate_mode": args.gate_mode,
        "case_id": args.case_id,
        "domain": domain,
        "lever_id": args.lever_id,
        "lead_hours": list(leads),
        "fields": list(fields),
        "init_time_utc": init_time.isoformat(),
        "init_time_source": init_source,
        "candidate_dir": str(candidate_dir),
        "cpu_wrf_dir": str(cpu_dir),
        "strict_dir": str(strict_dir) if strict_dir else None,
        "hard_guards": guards,
        "divergence_growth": divergence,
        "lead_payloads": lead_payloads,
        "validation_backlog_pass": bool(backlog_pass),
        "tier_o_pass": bool(tier_o_pass),
        "gate_pass": bool(verdict_pass),
        "verdict": "PASS" if verdict_pass else "FAIL",
        "reject_reasons": sorted(set(reject_reasons)),
    }


def write_report(out_dir: Path, rollup: dict[str, Any]) -> None:
    lines = [
        "# v0.22 Operational-Relaxed Small-Grid Gate",
        "",
        f"- gate_mode: `{rollup['gate_mode']}`",
        f"- lever_id: `{rollup['lever_id']}`",
        f"- case_id: `{rollup['case_id']}`",
        f"- verdict: `{rollup['verdict']}`",
        "",
        "| domain | lead h | hard guards | validation band | tier-o band | verdict |",
        "|---|---:|---|---|---|---|",
    ]
    for domain in rollup["domains"]:
        for lead in domain["lead_payloads"]:
            lines.append(
                "| {domain} | {lead} | {guards} | {validation} | {tier} | {verdict} |".format(
                    domain=domain["domain"],
                    lead=lead["lead_hours"],
                    guards="PASS" if lead["hard_guards"]["all_hard_guards_pass"] else "FAIL/UNAVAILABLE",
                    validation="PASS" if lead["validation_band_pass"] else "FAIL",
                    tier="PASS" if lead["tier_o_band_pass"] else "FAIL/UNAVAILABLE",
                    verdict=lead["verdict"],
                )
            )
    if rollup.get("reject_reasons"):
        lines.extend(["", "## Reject Reasons", ""])
        lines.extend(f"- {reason}" for reason in rollup["reject_reasons"])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The metric path is CPU-only and reads existing wrfout directories.",
            "- AEMET/obs scoring is recorded as unavailable in this first wiring pass.",
            "- Full on-device conservation and clamp/transfer audits should be supplied by the run via `--candidate-guards`.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_timestr(valid: datetime) -> str:
    return valid.strftime("%Y-%m-%d_%H:%M:%S")


def _write_one_synthetic_wrfout(path: Path, valid: datetime, lead: int, *, perturb_t2: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx, nz = 5, 6, 3
    lat = 27.5 + np.arange(ny, dtype=np.float64)[:, None] * 0.03 + np.zeros((ny, nx))
    lon = -16.8 + np.arange(nx, dtype=np.float64)[None, :] * 0.03 + np.zeros((ny, nx))
    y = np.arange(ny, dtype=np.float64)[:, None]
    x = np.arange(nx, dtype=np.float64)[None, :]
    t2 = 290.0 + 0.05 * lead + 0.2 * y + 0.1 * x + perturb_t2
    u10 = 4.0 + 0.01 * lead + 0.05 * x
    v10 = -2.0 + 0.008 * lead + 0.04 * y
    mu = np.full((ny, nx), 90000.0, dtype=np.float64)
    mub = np.full((ny, nx), 1000.0, dtype=np.float64)
    qv = np.full((nz, ny, nx), 0.008, dtype=np.float64)
    theta_pert = np.full((nz, ny, nx), -5.0 + 0.001 * lead, dtype=np.float64)
    precip = np.maximum(lead - 12, 0) * (0.1 + 0.01 * y + 0.005 * x)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", ny)
        dataset.createDimension("west_east", nx)
        dataset.createDimension("bottom_top", nz)
        dataset.START_DATE = "2026-05-21_18:00:00"
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list(_make_timestr(valid)), dtype="S1")
        for name, values in {
            "XLAT": lat,
            "XLONG": lon,
            "T2": t2,
            "U10": u10,
            "V10": v10,
            "MU": mu,
            "MUB": mub,
            "PSFC": mu + mub,
            "RAINNC": precip,
            "RAINC": np.zeros((ny, nx), dtype=np.float64),
        }.items():
            var = dataset.createVariable(name, "f8", ("Time", "south_north", "west_east"))
            var[0, :, :] = values
        for name, values in {"QVAPOR": qv, "T": theta_pert}.items():
            var = dataset.createVariable(name, "f8", ("Time", "bottom_top", "south_north", "west_east"))
            var[0, :, :, :] = values


def write_synthetic_example(root: Path, *, leads: tuple[int, ...], perturb_t2: float = 0.0) -> dict[str, Path]:
    init = datetime(2026, 5, 21, 18, tzinfo=timezone.utc)
    dirs = {
        "candidate": root / "candidate",
        "cpu_wrf": root / "cpu_wrf",
        "strict": root / "strict_gpu",
        "restart": root / "restart_probe",
    }
    all_leads = sorted(set([0, *leads]))
    for lead in all_leads:
        valid = init + timedelta(hours=int(lead))
        name = f"wrfout_d02_{_make_timestr(valid)}"
        _write_one_synthetic_wrfout(dirs["cpu_wrf"] / name, valid, lead)
        _write_one_synthetic_wrfout(dirs["strict"] / name, valid, lead)
        _write_one_synthetic_wrfout(dirs["restart"] / name, valid, lead, perturb_t2=perturb_t2)
        _write_one_synthetic_wrfout(dirs["candidate"] / name, valid, lead, perturb_t2=perturb_t2)
    guard = {
        "hard_guards": {
            "conservation": {
                "status": "PASS",
                "dry_mass_relative_residual": 0.0,
                "water_relative_residual": 0.0,
                "moist_static_energy_relative_residual": 0.0,
                "pass": True
            },
            "clamp_limiter_audit": {
                "status": "PASS",
                "new_term_keys": [],
                "dynamic_total_count": 0,
                "pass": True
            },
            "transfer_audit": {
                "status": "PASS",
                "new_in_loop_transfers": 0,
                "pass": True
            }
        }
    }
    write_json(root / "candidate_guards.json", guard)
    write_json(root / "strict_guards.json", guard)
    dirs["candidate_guards"] = root / "candidate_guards.json"
    dirs["strict_guards"] = root / "strict_guards.json"
    return dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--cpu-wrf-dir", type=Path)
    parser.add_argument("--strict-dir", type=Path)
    parser.add_argument("--restart-dir", type=Path)
    parser.add_argument("--candidate-guards", type=Path)
    parser.add_argument("--strict-guards", type=Path)
    parser.add_argument("--out", type=Path, default=Path("proofs/v022/operational_relaxed/example"))
    parser.add_argument("--case-id", default="CANARY-L2-D02")
    parser.add_argument("--lever-id", default="baseline_validation")
    parser.add_argument("--lever-config-json", default="{}")
    parser.add_argument("--domains", nargs="+", default=["d02"])
    parser.add_argument("--leads", nargs="+", type=int, default=list(DEFAULT_LEADS))
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument("--init", default=None, help="forecast init time, e.g. 2026-05-21_18:00:00")
    parser.add_argument("--gate-mode", choices=("cpu-wrf-backlog", "operational-relaxed"), default="cpu-wrf-backlog")
    parser.add_argument("--finite-scope", choices=("all", "required"), default="all")
    parser.add_argument("--grid-resolution", choices=("default", "3km", "1km"), default="default")
    parser.add_argument("--bands", type=Path, default=BANDS_PATH)
    parser.add_argument("--wrf-fortran-ref", default="<DATA_ROOT>/src/wrf_pristine")
    parser.add_argument("--synthetic-example-root", type=Path, help="write a tiny Canary-shaped wrfout fixture and score it")
    parser.add_argument("--synthetic-perturb-t2", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        json.loads(args.lever_config_json)
    except json.JSONDecodeError as exc:
        print(f"--lever-config-json is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if args.synthetic_example_root:
        dirs = write_synthetic_example(
            args.synthetic_example_root,
            leads=tuple(int(value) for value in args.leads),
            perturb_t2=float(args.synthetic_perturb_t2),
        )
        args.candidate_dir = args.candidate_dir or dirs["candidate"]
        args.cpu_wrf_dir = args.cpu_wrf_dir or dirs["cpu_wrf"]
        args.strict_dir = args.strict_dir or dirs["strict"]
        args.restart_dir = args.restart_dir or dirs["restart"]
        args.candidate_guards = args.candidate_guards or dirs["candidate_guards"]
        args.strict_guards = args.strict_guards or dirs["strict_guards"]
        args.case_id = args.case_id or "CANARY-L2-D02-SYNTHETIC"
        args.init = args.init or "2026-05-21_18:00:00"
    if args.candidate_dir is None or args.cpu_wrf_dir is None:
        print("--candidate-dir and --cpu-wrf-dir are required unless --synthetic-example-root is used", file=sys.stderr)
        return 2
    bands = json.loads(Path(args.bands).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = []
    reject_reasons: list[str] = []
    try:
        for domain in args.domains:
            payload = build_domain_payload(args, domain, bands)
            domains.append(payload)
            reject_reasons.extend(payload.get("reject_reasons", []))
            domain_dir = out_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            write_json(domain_dir / "rollup.json", payload)
            for lead_payload in payload["lead_payloads"]:
                write_json(domain_dir / f"lead_{int(lead_payload['lead_hours']):03d}.json", lead_payload)
    except Exception as exc:
        error_payload = {
            "schema": "V022OperationalRelaxedGateError",
            "schema_version": 1,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(out_dir / "error.json", error_payload)
        print(error_payload["error"], file=sys.stderr)
        return 2
    gate_pass = all(bool(domain["gate_pass"]) for domain in domains)
    rollup = {
        "schema": "V022OperationalRelaxedGateRollup",
        "schema_version": 1,
        "gate_mode": args.gate_mode,
        "case_id": args.case_id,
        "lever_id": args.lever_id,
        "domains": domains,
        "verdict": "PASS" if gate_pass else "FAIL",
        "reject_reasons": sorted(set(reject_reasons)),
    }
    write_json(out_dir / "rollup.json", rollup)
    write_report(out_dir, rollup)
    print(json.dumps({"verdict": rollup["verdict"], "out": str(out_dir)}, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
