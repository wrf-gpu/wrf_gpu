#!/usr/bin/env python3
"""Trace the first physical-bound violation in diagnostic time series.

Purpose: find the first field and location where a replay, fixture, or
operator-sanity proof leaves documented finite/physical bounds.
Answers: which `(field, step, time_s, value, bound, i, j, k)` fails first,
plus the first nonfinite sample if one appears.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `scripts/m6_warm_bubble_test.py:72-78` defines physical
bound fields, and `scripts/m6_warm_bubble_test.py:275-304` reports violations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BOUNDS = {
    "theta_perturbation_max_K": {"max": 50.0, "units": "K"},
    "theta_perturbation_min_K": {"min": -50.0, "units": "K"},
    "p_perturbation_max_Pa": {"max": 50000.0, "units": "Pa"},
    "p_perturbation_min_Pa": {"min": -50000.0, "units": "Pa"},
    "mu_perturbation_max_Pa": {"max": 50000.0, "units": "Pa"},
    "theta_K": {"min": 150.0, "max": 400.0, "units": "K"},
    "qv": {"min": 0.0, "max": 0.1, "units": "kg kg-1"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _iter_series(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bounds = dict(DEFAULT_BOUNDS)
    bounds.update(payload.get("bounds", {}))
    if isinstance(payload.get("series"), list):
        return payload["series"], bounds
    if isinstance(payload.get("samples"), dict):
        rows: list[dict[str, Any]] = []
        for label, sample in payload["samples"].items():
            row = dict(sample)
            row.setdefault("label", label)
            row.setdefault("step", sample.get("step"))
            row.setdefault("time_s", sample.get("time_s"))
            rows.append(row)
        return rows, bounds
    if isinstance(payload.get("fields"), dict):
        rows = []
        times = payload.get("time_s") or list(range(max(len(v) for v in payload["fields"].values())))
        for step, time_s in enumerate(times):
            row = {"step": step, "time_s": time_s}
            for name, values in payload["fields"].items():
                if step < len(values):
                    row[name] = values[step]
            rows.append(row)
        return rows, bounds
    return [], bounds


def _record_violation(field: str, row: dict[str, Any], value: float, comparator: str, bound: float, units: str) -> dict[str, Any]:
    return {
        "field": field,
        "step": row.get("step"),
        "time_s": row.get("time_s"),
        "value": value,
        "bound": bound,
        "comparator": comparator,
        "units": units,
        "i": row.get("i"),
        "j": row.get("j"),
        "k": row.get("k"),
    }


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load_json(input_path)
    rows, bounds = _iter_series(payload)
    violations: list[dict[str, Any]] = []
    first_nonfinite: dict[str, Any] | None = None

    for row in rows:
        for field, spec in bounds.items():
            if field not in row:
                continue
            value = _as_float(row[field])
            if value is None:
                continue
            if not math.isfinite(value):
                if first_nonfinite is None:
                    first_nonfinite = _record_violation(field, row, value, "finite", math.nan, spec.get("units", "unknown"))
                continue
            units = str(spec.get("units", "unknown"))
            if "max" in spec and value > float(spec["max"]):
                violations.append(_record_violation(field, row, value, "<=", float(spec["max"]), units))
            if "min" in spec and value < float(spec["min"]):
                violations.append(_record_violation(field, row, value, ">=", float(spec["min"]), units))

    if payload.get("bound_violations") and not violations:
        violations = list(payload["bound_violations"])

    first = violations[0] if violations else None
    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {
            "name": "bound_violation_tracer",
            "purpose": "Locate first finite or physical-bound violation.",
        },
        "input": {"path": str(input_path), "rows": len(rows), "bound_count": len(bounds)},
        "measurements": {
            "first_violation": first,
            "violation_count": len(violations),
            "violations": violations,
            "first_nonfinite": first_nonfinite,
        },
        "units": {"time_s": "s", "value": "field-dependent"},
        "artifacts": {},
        "status": "VIOLATION" if first or first_nonfinite else "OK",
        "source_citations": [
            "scripts/m6_warm_bubble_test.py:72-78",
            "scripts/m6_warm_bubble_test.py:275-304",
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
