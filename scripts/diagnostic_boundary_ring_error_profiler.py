#!/usr/bin/env python3
"""Profile forecast-reference error by distance from lateral boundaries.

Purpose: separate boundary-forcing error from interior solver drift in d02
replay outputs.
Answers: RMSE by 0-5, 5-10, 10-20, and interior grid-cell rings.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `src/gpuwrf/integration/d02_replay.py:99-140` packs W/E/S/N
side histories, and `src/gpuwrf/coupling/boundary_apply.py:31-77` applies
specified and relaxation-zone boundaries.
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
    return json.loads(path.read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _distance(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape)
    return np.minimum.reduce((y, x, shape[0] - 1 - y, shape[1] - 1 - x))


def _rmse(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.nanmean(values * values)))


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    forecast = payload.get("forecast", {})
    reference = payload.get("reference", {})
    field = payload.get("field") or next((name for name in forecast if name in reference), None)
    error = _array(forecast[field]) - _array(reference[field]) if field else np.zeros((0, 0))
    error = np.atleast_2d(np.squeeze(error))
    distance = _distance(error.shape) if error.size else np.zeros_like(error, dtype=int)
    bands = [(0, 5, "0-5"), (5, 10, "5-10"), (10, 20, "10-20"), (20, None, "interior")]
    ring_rmse = []
    for lo, hi, label in bands:
        mask = distance >= lo if hi is None else (distance >= lo) & (distance < hi)
        ring_rmse.append({"band_grid_cells": label, "sample_count": int(mask.sum()), "rmse": _rmse(error[mask])})

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "boundary_ring_error_profiler", "purpose": "Quantify error by boundary distance."},
        "input": {"path": str(input_path), "field": field, "shape": list(error.shape)},
        "measurements": {"ring_rmse": ring_rmse},
        "units": {"distance": "grid cells", "rmse": "field-dependent"},
        "artifacts": {},
        "status": "OK" if field else "NO_COMPARABLE_FIELD",
        "source_citations": [
            "src/gpuwrf/integration/d02_replay.py:99-140",
            "src/gpuwrf/coupling/boundary_apply.py:31-77",
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
