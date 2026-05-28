#!/usr/bin/env python3
"""Track conservation-like totals over replay or fixture states.

Purpose: report mass, water, kinetic-energy, and dry-static-energy drift before
RMSE failures become large.
Answers: per-step totals, source/sink and boundary terms when provided, and
relative drift from the first sample.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `<development-history-not-included-in-public-repo>/milestones/ROADMAP.md:78-88` names M4 invariants,
and `<development-history-not-included-in-public-repo>/milestones/ROADMAP.md:89-98` carries those checks into M6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


GRAVITY_M_S2 = 9.80665
CP_DRY_AIR = 1004.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(value: Any) -> np.ndarray:
    return np.asarray(value if value is not None else 0.0, dtype=np.float64)


def _sum(value: Any) -> float:
    return float(np.nansum(_array(value)))


def _totals(state: dict[str, Any]) -> dict[str, float]:
    if "totals" in state:
        return {name: float(value) for name, value in state["totals"].items()}
    mass = _array(state.get("mass", state.get("mu", 1.0)))
    qv = _array(state.get("qv", 0.0))
    u = _array(state.get("u", 0.0))
    v = _array(state.get("v", 0.0))
    w = _array(state.get("w", 0.0))
    theta = _array(state.get("theta", 0.0))
    height = _array(state.get("height_m", 0.0))
    return {
        "mass": _sum(mass),
        "water": _sum(mass * qv),
        "kinetic_energy": float(0.5 * np.nansum(mass * (u * u + v * v + w * w))),
        "dry_static_energy": float(np.nansum(mass * (CP_DRY_AIR * theta + GRAVITY_M_S2 * height))),
    }


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    states = payload.get("states", [])
    time_series = []
    baseline: dict[str, float] | None = None
    for index, state in enumerate(states):
        totals = _totals(state)
        if baseline is None:
            baseline = totals
        drift = {}
        for name, value in totals.items():
            base = baseline.get(name, 0.0)
            drift[name] = None if base == 0.0 else float((value - base) / abs(base))
        time_series.append(
            {
                "step": state.get("step", index),
                "time_s": state.get("time_s"),
                "totals": totals,
                "relative_drift": drift,
                "source_terms": state.get("source_terms", {}),
                "boundary_terms": state.get("boundary_terms", {}),
            }
        )

    max_abs_drift = {}
    for row in time_series:
        for name, value in row["relative_drift"].items():
            if value is not None:
                max_abs_drift[name] = max(max_abs_drift.get(name, 0.0), abs(value))

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "conservation_tracker", "purpose": "Track integral invariant drift."},
        "input": {"path": str(input_path), "state_count": len(states)},
        "measurements": {"time_series": time_series, "max_abs_relative_drift": max_abs_drift},
        "units": {"mass": "kg or Pa-column equivalent", "water": "kg", "kinetic_energy": "J", "dry_static_energy": "J"},
        "artifacts": {},
        "status": "OK" if states else "NO_STATES",
        "source_citations": ["<development-history-not-included-in-public-repo>/milestones/ROADMAP.md:78-88", "<development-history-not-included-in-public-repo>/milestones/ROADMAP.md:89-98"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
