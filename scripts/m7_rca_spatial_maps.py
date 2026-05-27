#!/usr/bin/env python
"""Spatial and boundary/interior GPU-vs-CPU deviation diagnostics for M7 RCA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from m7_rca_hour_by_hour import (  # noqa: E402
    DEFAULT_CPU_ROOT,
    DEFAULT_FIELDS,
    DEFAULT_GPU_ROOT,
    SPRINT_DIR,
    WrfoutPair,
    build_wrfout_pairs,
    load_field,
    write_json,
    _json_default,
)


DEFAULT_ARTIFACT_DIR = Path("/tmp/m7_rca_artifacts")
DEFAULT_SPATIAL_OUTPUT = SPRINT_DIR / "spatial_deviation_summary.json"
DEFAULT_BOUNDARY_OUTPUT = SPRINT_DIR / "boundary_vs_interior.json"
DEFAULT_MAP_FIELDS = ("T2", "U10", "V10", "PSFC")
DEFAULT_MAP_LEADS = (1, 6, 12, 24)
BOUNDARY_WIDTH = 5


def _finite_flat(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return flat[np.isfinite(flat)]


def percentile_stats(values: np.ndarray) -> dict[str, Any]:
    finite = _finite_flat(values)
    if finite.size == 0:
        return {
            "finite_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "abs_p50": None,
            "abs_p90": None,
            "abs_p95": None,
            "abs_p99": None,
        }
    abs_values = np.abs(finite)
    return {
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "abs_mean": float(np.mean(abs_values)),
        "abs_p50": float(np.percentile(abs_values, 50)),
        "abs_p90": float(np.percentile(abs_values, 90)),
        "abs_p95": float(np.percentile(abs_values, 95)),
        "abs_p99": float(np.percentile(abs_values, 99)),
        "max_abs": float(np.max(abs_values)),
    }


def horizontal_boundary_mask(shape: Sequence[int], width: int = BOUNDARY_WIDTH) -> np.ndarray:
    if len(shape) < 2:
        raise ValueError(f"field shape must have at least two horizontal dimensions, got {shape}")
    ny = int(shape[-2])
    nx = int(shape[-1])
    mask_2d = np.zeros((ny, nx), dtype=bool)
    w_y = min(int(width), ny)
    w_x = min(int(width), nx)
    mask_2d[:w_y, :] = True
    mask_2d[-w_y:, :] = True
    mask_2d[:, :w_x] = True
    mask_2d[:, -w_x:] = True
    if len(shape) == 2:
        return mask_2d
    return np.broadcast_to(mask_2d, tuple(shape)).copy()


def split_boundary_interior(diff: np.ndarray, width: int = BOUNDARY_WIDTH) -> dict[str, Any]:
    values = np.asarray(diff, dtype=np.float64)
    mask = horizontal_boundary_mask(values.shape, width=width)
    boundary = values[mask]
    interior = values[~mask]
    boundary_stats = percentile_stats(boundary)
    interior_stats = percentile_stats(interior)
    boundary_abs = boundary_stats.get("abs_mean")
    interior_abs = interior_stats.get("abs_mean")
    ratio = None
    concentration = "UNCLASSIFIED"
    if boundary_abs is not None and interior_abs is not None:
        if float(interior_abs) <= 1.0e-12:
            ratio = None if float(boundary_abs) <= 1.0e-12 else float("inf")
        else:
            ratio = float(boundary_abs) / float(interior_abs)
        if ratio == float("inf") or (ratio is not None and ratio >= 2.0):
            concentration = "BOUNDARY_CONCENTRATED"
        elif ratio is not None and ratio <= 0.5:
            concentration = "INTERIOR_CONCENTRATED"
        else:
            concentration = "MIXED_OR_UNIFORM"
    return {
        "boundary_width_cells": int(width),
        "boundary_cell_count": int(mask.sum()),
        "interior_cell_count": int((~mask).sum()),
        "boundary": boundary_stats,
        "interior": interior_stats,
        "boundary_to_interior_abs_mean_ratio": ratio,
        "concentration": concentration,
    }


def classify_spatial_pattern(diff: np.ndarray, boundary_split: Mapping[str, Any]) -> str:
    stats = percentile_stats(diff)
    finite = _finite_flat(diff)
    if finite.size == 0:
        return "NO_FINITE_VALUES"
    max_abs = float(stats["max_abs"])
    p50 = float(stats["abs_p50"])
    p95 = float(stats["abs_p95"])
    mean = float(stats["mean"])
    same_sign_fraction = float(np.mean(np.sign(finite) == np.sign(mean))) if abs(mean) > 0.0 else 0.0
    concentration = str(boundary_split.get("concentration"))
    if concentration == "BOUNDARY_CONCENTRATED":
        return "LOCALIZED_BOUNDARY"
    if p50 > 0.0 and max_abs >= 8.0 * p50 and p95 >= 3.0 * p50:
        return "LOCALIZED_HOTSPOT"
    if p95 > 0.0 and abs(mean) >= 0.7 * p95 and same_sign_fraction >= 0.9:
        return "SPATIALLY_UNIFORM_BIAS"
    return "MIXED_OR_BROAD"


def _max_location(diff: np.ndarray, lat: np.ndarray | None = None, lon: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(diff, dtype=np.float64)
    abs_values = np.abs(values)
    if not np.isfinite(abs_values).any():
        return {"status": "NO_FINITE_VALUES"}
    index = tuple(int(item) for item in np.unravel_index(np.nanargmax(abs_values), values.shape))
    location: dict[str, Any] = {
        "status": "OK",
        "index": list(index),
        "diff": float(values[index]),
        "abs_diff": float(abs_values[index]),
    }
    if lat is not None and lon is not None and lat.shape == values.shape[-2:] and lon.shape == values.shape[-2:]:
        yx = index[-2:]
        location["lat"] = float(lat[yx])
        location["lon"] = float(lon[yx])
    return location


def _diff_field(pair: WrfoutPair, field: str) -> np.ndarray:
    gpu = load_field(pair.gpu_path, field)
    cpu = load_field(pair.cpu_path, field)
    if gpu.shape != cpu.shape:
        raise ValueError(f"{field} shape mismatch: GPU={gpu.shape} CPU={cpu.shape}")
    return np.asarray(gpu - cpu, dtype=np.float64)


def _field_dims(field: str, shape: Sequence[int]) -> tuple[str, ...]:
    if len(shape) == 2:
        if field == "U":
            return ("south_north", "west_east_stag")
        if field == "V":
            return ("south_north_stag", "west_east")
        return ("south_north", "west_east")
    if len(shape) == 3:
        if field == "U":
            return ("bottom_top", "south_north", "west_east_stag")
        if field == "V":
            return ("bottom_top", "south_north_stag", "west_east")
        return ("bottom_top", "south_north", "west_east")
    return tuple(f"dim_{idx}" for idx in range(len(shape)))


def write_diff_netcdf(pair: WrfoutPair, fields: Sequence[str], artifact_dir: Path) -> tuple[Path, dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"lead_{pair.lead_hour:02d}_surface_deviation.nc"
    data_vars: dict[str, Any] = {}
    field_summaries: dict[str, Any] = {}
    lat = None
    lon = None
    try:
        lat = load_field(pair.cpu_path, "XLAT")
        lon = load_field(pair.cpu_path, "XLONG")
        data_vars["XLAT"] = (("south_north", "west_east"), lat.astype(np.float32))
        data_vars["XLONG"] = (("south_north", "west_east"), lon.astype(np.float32))
    except Exception:
        lat = None
        lon = None

    for field in fields:
        try:
            diff = _diff_field(pair, field)
        except Exception as exc:
            field_summaries[field] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
            continue
        dims = _field_dims(field, diff.shape)
        data_vars[f"{field}_diff"] = (dims, diff.astype(np.float32))
        split = split_boundary_interior(diff)
        field_summaries[field] = {
            "status": "OK",
            "shape": list(diff.shape),
            "pattern_classification": classify_spatial_pattern(diff, split),
            "max_error_location": _max_location(diff, lat=lat, lon=lon),
            "percentiles": percentile_stats(diff),
            "boundary_vs_interior": split,
        }
    dataset = xr.Dataset(
        data_vars=data_vars,
        attrs={
            "schema": "M7SkillRegressionSpatialDeviationMap",
            "gpu_path": str(pair.gpu_path),
            "cpu_path": str(pair.cpu_path),
            "lead_hour": int(pair.lead_hour),
            "valid_time_utc": pair.valid_time.isoformat(),
            "note": "Variables are GPU minus CPU.",
        },
    )
    encoding = {name: {"zlib": True, "complevel": 1} for name in dataset.data_vars if name.endswith("_diff")}
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path, field_summaries


def build_spatial_payload(
    *,
    gpu_root: str | Path,
    cpu_root: str | Path,
    artifact_dir: Path,
    leads: Sequence[int] = DEFAULT_MAP_LEADS,
    fields: Sequence[str] = DEFAULT_MAP_FIELDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = build_wrfout_pairs(gpu_root, cpu_root)
    pairs_by_lead = {int(pair.lead_hour): pair for pair in pairs}

    map_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for lead in leads:
        pair = pairs_by_lead.get(int(lead))
        if pair is None:
            map_rows.append({"lead_hour": int(lead), "status": "MISSING_LEAD"})
            continue
        nc_path, summaries = write_diff_netcdf(pair, fields, artifact_dir)
        map_rows.append(
            {
                "lead_hour": int(pair.lead_hour),
                "output_index": int(pair.output_index),
                "valid_time_utc": pair.valid_time.isoformat(),
                "gpu_path": str(pair.gpu_path),
                "cpu_path": str(pair.cpu_path),
                "netcdf_artifact": str(nc_path),
                "fields": summaries,
            }
        )

    for pair in pairs:
        field_entries: dict[str, Any] = {}
        for field in DEFAULT_FIELDS:
            try:
                diff = _diff_field(pair, field)
                split = split_boundary_interior(diff)
                field_entries[field] = {
                    "status": "OK",
                    "shape": list(diff.shape),
                    **split,
                }
            except Exception as exc:
                field_entries[field] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        boundary_rows.append(
            {
                "lead_hour": int(pair.lead_hour),
                "output_index": int(pair.output_index),
                "valid_time_utc": pair.valid_time.isoformat(),
                "fields": field_entries,
            }
        )

    spatial_payload = {
        "schema": "M7SkillRegressionSpatialDeviationSummary",
        "schema_version": 1,
        "gpu_root": str(gpu_root),
        "cpu_root": str(cpu_root),
        "artifact_dir": str(artifact_dir),
        "lead_hours": [int(lead) for lead in leads],
        "fields": list(fields),
        "rows": map_rows,
    }
    boundary_payload = {
        "schema": "M7SkillRegressionBoundaryVsInterior",
        "schema_version": 1,
        "gpu_root": str(gpu_root),
        "cpu_root": str(cpu_root),
        "boundary_width_cells": BOUNDARY_WIDTH,
        "field_order": list(DEFAULT_FIELDS),
        "rows": boundary_rows,
    }
    return spatial_payload, boundary_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT)
    parser.add_argument("--cpu-root", type=Path, default=DEFAULT_CPU_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--spatial-output", type=Path, default=DEFAULT_SPATIAL_OUTPUT)
    parser.add_argument("--boundary-output", type=Path, default=DEFAULT_BOUNDARY_OUTPUT)
    parser.add_argument("--leads", type=int, nargs="+", default=list(DEFAULT_MAP_LEADS))
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_MAP_FIELDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spatial, boundary = build_spatial_payload(
        gpu_root=args.gpu_root,
        cpu_root=args.cpu_root,
        artifact_dir=args.artifact_dir,
        leads=args.leads,
        fields=args.fields,
    )
    write_json(args.spatial_output, spatial)
    write_json(args.boundary_output, boundary)
    print(
        json.dumps(
            {
                "spatial_output": str(args.spatial_output),
                "boundary_output": str(args.boundary_output),
                "artifact_dir": str(args.artifact_dir),
                "lead_count": len(args.leads),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
