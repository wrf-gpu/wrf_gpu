#!/usr/bin/env python3
"""Extract vertical-column profiles and phase-space traces.

Purpose: show whether selected columns exhibit oscillatory gravity-wave
behavior, monotone blowup, or bounded settling.
Answers: per-column time series, vertical profiles, and compact phase portraits
for w/theta/p/mu pairs.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `scripts/diagnostic_warm_bubble_vs_slice.py:151-213` is the
reference center-column probe pattern, and `scripts/m6_warm_bubble_test.py:88-177`
builds the warm-bubble column state used by that probe.
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
    return np.asarray(value if value is not None else [], dtype=np.float64)


def _profile_summary(values: Any) -> dict[str, Any]:
    arr = _array(values)
    if arr.size == 0:
        return {"levels": 0, "min": None, "max": None, "mean": None}
    return {"levels": int(arr.size), "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)), "mean": float(np.nanmean(arr))}


def _phase(x: Any, y: Any) -> list[list[float]]:
    xa = _array(x).reshape(-1)
    ya = _array(y).reshape(-1)
    n = min(xa.size, ya.size)
    return [[float(xa[i]), float(ya[i])] for i in range(n)]


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    columns = []
    for index, column in enumerate(payload.get("columns", [])):
        profiles = column.get("profiles", {})
        series = column.get("time_series", {})
        columns.append(
            {
                "name": column.get("name", f"column_{index}"),
                "i": column.get("i"),
                "j": column.get("j"),
                "vertical_profiles": {name: _profile_summary(values) for name, values in profiles.items()},
                "time_series": series,
                "phase_portraits": {
                    "w_vs_theta": _phase(series.get("w"), series.get("theta")),
                    "w_vs_p": _phase(series.get("w"), series.get("p")),
                    "mu_vs_w": _phase(series.get("mu"), series.get("w")),
                },
            }
        )

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "vertical_column_phase_space", "purpose": "Summarize selected-column vertical dynamics."},
        "input": {"path": str(input_path), "column_count": len(columns)},
        "measurements": {"columns": columns},
        "units": {"w": "m s-1", "theta": "K", "p": "Pa", "mu": "Pa", "height": "m"},
        "artifacts": {},
        "status": "OK" if columns else "NO_COLUMNS",
        "source_citations": [
            "scripts/diagnostic_warm_bubble_vs_slice.py:151-213",
            "scripts/m6_warm_bubble_test.py:88-177",
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
