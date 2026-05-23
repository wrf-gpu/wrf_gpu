#!/usr/bin/env python3
"""Track activation of the temporary mu-continuity limiter.

Purpose: measure how often `_mu_continuity_increment` saturates raw dry-column
mass updates and where the largest column update occurs.
Answers: max raw dmu, max bounded dmu, saturation fraction, and max-column
location per step.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `src/gpuwrf/dynamics/acoustic_wrf.py:456-475` implements the
tanh dry-column limiter, and `src/gpuwrf/dynamics/acoustic_wrf.py:872-876`
applies it in the acoustic substep.
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


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("limiter_steps"), list):
        return list(payload["limiter_steps"])
    if isinstance(payload.get("steps"), list):
        return [row for row in payload["steps"] if "raw_dmu" in row or "bounded_dmu" in row]
    return []


def _array(value: Any) -> np.ndarray:
    return np.asarray(value if value is not None else [], dtype=np.float64)


def _location(index: int, shape: tuple[int, ...]) -> dict[str, int | None]:
    if not shape:
        return {"i": None, "j": None, "k": None}
    coords = np.unravel_index(index, shape)
    if len(coords) == 1:
        return {"i": int(coords[0]), "j": None, "k": None}
    if len(coords) == 2:
        return {"i": int(coords[1]), "j": int(coords[0]), "k": None}
    return {"i": int(coords[-1]), "j": int(coords[-2]), "k": int(coords[-3])}


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    rows = _rows(payload)
    per_step: list[dict[str, Any]] = []
    any_activation = False
    for index, row in enumerate(rows):
        raw = _array(row.get("raw_dmu"))
        bounded = _array(row.get("bounded_dmu"))
        if raw.shape != bounded.shape:
            bounded = np.resize(bounded, raw.shape) if raw.size else bounded
        if raw.size == 0:
            max_index = 0
            saturated = np.asarray([], dtype=bool)
        else:
            max_index = int(np.nanargmax(np.abs(raw)))
            saturated = np.abs(raw) > (np.abs(bounded) + 1.0e-12)
        activation = float(np.count_nonzero(saturated) / saturated.size) if saturated.size else 0.0
        any_activation = any_activation or activation > 0.0
        per_step.append(
            {
                "step": row.get("step", index + 1),
                "time_s": row.get("time_s"),
                "max_raw_dmu_pa": float(np.nanmax(np.abs(raw))) if raw.size else 0.0,
                "max_bounded_dmu_pa": float(np.nanmax(np.abs(bounded))) if bounded.size else 0.0,
                "saturation_fraction": activation,
                "max_column_location": _location(max_index, raw.shape),
            }
        )

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "limiter_activation_tracker", "purpose": "Quantify dry-mass limiter saturation."},
        "input": {"path": str(input_path), "step_count": len(rows)},
        "measurements": {
            "per_step": per_step,
            "max_saturation_fraction": max((row["saturation_fraction"] for row in per_step), default=0.0),
            "limiter_active": any_activation,
        },
        "units": {"dmu": "Pa", "time_s": "s", "saturation_fraction": "1"},
        "artifacts": {},
        "status": "LIMITER_ACTIVE" if any_activation else "OK",
        "source_citations": [
            "src/gpuwrf/dynamics/acoustic_wrf.py:456-475",
            "src/gpuwrf/dynamics/acoustic_wrf.py:872-876",
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
