#!/usr/bin/env python3
"""Summarize per-term operator tendency budgets.

Purpose: expose which named RHS term dominates w/theta/mu changes before a
replay or warm-bubble diagnostic fails.
Answers: per-term max absolute value, mean, L2 norm, sign balance, and shape
for buoyancy, pressure restoring, density coupling, theta transport, Rayleigh,
smdiv, and boundary forcing terms.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md:71-90`
records the first-substep RHS decomposition, and `src/gpuwrf/dynamics/acoustic_wrf.py:735-747`
contains the current named recurrence terms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_TERMS = (
    "buoyancy",
    "pressure_restoring",
    "density_coupling",
    "theta_transport",
    "rayleigh",
    "smdiv",
    "boundary_forcing",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    return np.asarray(value if value is not None else [], dtype=np.float64)


def _budget(values: Any) -> dict[str, Any]:
    arr = _array(values)
    if arr.size == 0:
        return {"present": False, "shape": [], "max_abs": None, "mean": None, "l2": None, "positive_fraction": None}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"present": True, "shape": list(arr.shape), "max_abs": None, "mean": None, "l2": None, "positive_fraction": None}
    return {
        "present": True,
        "shape": list(arr.shape),
        "max_abs": float(np.nanmax(np.abs(finite))),
        "mean": float(np.nanmean(finite)),
        "l2": float(np.sqrt(np.nansum(finite * finite))),
        "positive_fraction": float(np.count_nonzero(finite > 0.0) / finite.size),
    }


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    terms = payload.get("terms", payload.get("operator_terms", {}))
    per_term = {name: _budget(terms.get(name)) for name in EXPECTED_TERMS}
    extra_terms = {name: _budget(value) for name, value in terms.items() if name not in per_term}
    ranking = sorted(
        [{"term": name, "max_abs": stats["max_abs"]} for name, stats in {**per_term, **extra_terms}.items() if stats["max_abs"] is not None],
        key=lambda item: item["max_abs"],
        reverse=True,
    )

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "operator_term_budget_tracer", "purpose": "Rank named operator terms by magnitude."},
        "input": {"path": str(input_path), "term_count": len(terms)},
        "measurements": {"per_term": per_term, "extra_terms": extra_terms, "ranking": ranking},
        "units": {"term_value": "field-dependent tendency units", "l2": "same units times sqrt(samples)"},
        "artifacts": {},
        "status": "OK" if terms else "NO_TERMS",
        "source_citations": [
            ".agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md:71-90",
            "src/gpuwrf/dynamics/acoustic_wrf.py:735-747",
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
