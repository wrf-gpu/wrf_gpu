#!/usr/bin/env python3
"""Create spatial error maps and stratified divergence summaries.

Purpose: locate whether forecast-reference errors concentrate near boundaries,
terrain, land/sea masks, or elevation bands.
Answers: `.npz` error maps plus JSON summaries by boundary band, terrain
quartile, land/sea class, and elevation quartile.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `artifacts`, `status`, and `source_citations`.
Source citations: `src/gpuwrf/integration/d02_replay.py:515-546` defines
reference field comparisons, and `src/gpuwrf/integration/d02_replay.py:142-206`
documents boundary side-history context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name].tolist() for name in data.files}
    return json.loads(path.read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _rmse(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(finite * finite)))


def _boundary_distance(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape)
    return np.minimum.reduce((y, x, shape[0] - 1 - y, shape[1] - 1 - x))


def _band_summary(error: np.ndarray, distance: np.ndarray) -> list[dict[str, Any]]:
    bands = [(0, 5, "0-5"), (5, 10, "5-10"), (10, 20, "10-20"), (20, None, "interior")]
    out = []
    for lo, hi, label in bands:
        mask = distance >= lo if hi is None else (distance >= lo) & (distance < hi)
        out.append({"band_grid_cells": label, "sample_count": int(np.count_nonzero(mask)), "rmse": _rmse(error[mask])})
    return out


def _quartile_summary(error: np.ndarray, driver: np.ndarray, label: str) -> list[dict[str, Any]]:
    flat = driver[np.isfinite(driver)]
    if flat.size == 0:
        return []
    cuts = np.nanquantile(flat, [0.0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for index in range(4):
        lo = cuts[index]
        hi = cuts[index + 1]
        mask = (driver >= lo) & (driver <= hi if index == 3 else driver < hi)
        out.append({f"{label}_quartile": index + 1, "range": [float(lo), float(hi)], "sample_count": int(mask.sum()), "rmse": _rmse(error[mask])})
    return out


def build_payload(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    forecast = payload.get("forecast", {})
    reference = payload.get("reference", {})
    field = payload.get("field") or next((name for name in forecast if name in reference), None)
    if field is None:
        error = np.zeros((0, 0), dtype=np.float64)
    else:
        error = _array(forecast[field]) - _array(reference[field])
    if error.ndim != 2:
        error = np.squeeze(error)
    if error.ndim != 2:
        error = np.atleast_2d(error)
    elevation = _array(payload.get("elevation_m", np.zeros_like(error)))
    landmask = _array(payload.get("landmask", np.ones_like(error)))
    terrain = _array(payload.get("terrain_metric", elevation))
    distance = _boundary_distance(error.shape) if error.size else np.zeros_like(error, dtype=int)
    npz_path = output_path.with_suffix(".npz")
    np.savez(npz_path, field=np.asarray(field or ""), error=error, elevation_m=elevation, landmask=landmask)

    land_summary = []
    for value, label in ((0.0, "sea"), (1.0, "land")):
        mask = landmask == value
        land_summary.append({"class": label, "sample_count": int(mask.sum()), "rmse": _rmse(error[mask])})

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "spatial_divergence_map", "purpose": "Map and stratify forecast-reference error."},
        "input": {"path": str(input_path), "field": field, "shape": list(error.shape)},
        "measurements": {
            "global_rmse": _rmse(error),
            "boundary_bands": _band_summary(error, distance),
            "terrain_quartiles": _quartile_summary(error, terrain, "terrain"),
            "land_sea": land_summary,
            "elevation_quartiles": _quartile_summary(error, elevation, "elevation_m"),
        },
        "units": {"error": "field-dependent", "elevation_m": "m", "distance": "grid cells"},
        "artifacts": {"spatial_error_npz": str(npz_path)},
        "status": "OK" if field is not None else "NO_COMPARABLE_FIELD",
        "source_citations": [
            "src/gpuwrf/integration/d02_replay.py:515-546",
            "src/gpuwrf/integration/d02_replay.py:142-206",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    proof = build_payload(args.input, args.output)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
