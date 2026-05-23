#!/usr/bin/env python3
"""Build a field RMSE/bias/max-error timeline against Gen2-style references.

Purpose: identify which surface or column field diverges first by lead time.
Answers: RMSE, bias, and max absolute error for T2, U10, V10, qv2, w, and
theta when forecast/reference arrays are available.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `src/gpuwrf/integration/d02_replay.py:515-546` defines the
Gen2 forecast comparison fields, and `scripts/diagnostic_gen2_rmse_baseline.py:1-9`
shows the Gen2 RMSE-noise-floor analysis pattern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FIELDS = ("T2", "U10", "V10", "qv2", "w", "theta", "w_k20", "theta_k20")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _stats(forecast: Any, reference: Any) -> dict[str, float]:
    error = _array(forecast) - _array(reference)
    return {
        "rmse": float(np.sqrt(np.nanmean(error * error))),
        "bias": float(np.nanmean(error)),
        "max_abs_error": float(np.nanmax(np.abs(error))),
    }


def _from_existing_comparison(payload: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict) or "rmse" not in comparison:
        return []
    fields = {}
    for name, item in comparison.get("rmse", {}).items():
        fields.setdefault(name, {})["rmse"] = item.get("value")
    for name, item in comparison.get("spatial_mean_drift", {}).items():
        fields.setdefault(name, {})["bias"] = item.get("value")
    for name, item in comparison.get("max_abs_error", {}).items():
        fields.setdefault(name, {})["max_abs_error"] = item.get("value")
    return [
        {
            "lead_time_s": float(comparison.get("lead_hours", 0.0)) * 3600.0,
            "lead_hours": comparison.get("lead_hours"),
            "fields": fields,
            "reference": comparison.get("gen2_reference_path"),
        }
    ]


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    timeline = _from_existing_comparison(payload)
    for lead in payload.get("leads", []):
        forecast = lead.get("forecast", {})
        reference = lead.get("reference", {})
        fields = {}
        for name in DEFAULT_FIELDS:
            if name in forecast and name in reference:
                fields[name] = _stats(forecast[name], reference[name])
        timeline.append(
            {
                "lead_time_s": lead.get("lead_time_s"),
                "lead_hours": lead.get("lead_hours"),
                "fields": fields,
                "reference": lead.get("reference_path"),
            }
        )

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "field_rmse_timeline", "purpose": "Track forecast-reference error by lead."},
        "input": {"path": str(input_path), "lead_count": len(timeline)},
        "measurements": {"timeline": timeline, "fields_requested": list(DEFAULT_FIELDS)},
        "units": {"T2": "K", "theta": "K", "U10": "m s-1", "V10": "m s-1", "w": "m s-1", "qv2": "kg kg-1"},
        "artifacts": {},
        "status": "OK" if timeline else "NO_COMPARABLE_FIELDS",
        "source_citations": [
            "src/gpuwrf/integration/d02_replay.py:515-546",
            "scripts/diagnostic_gen2_rmse_baseline.py:1-9",
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
