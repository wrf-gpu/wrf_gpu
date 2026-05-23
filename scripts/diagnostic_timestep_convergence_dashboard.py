#!/usr/bin/env python3
"""Assemble a placeholder timestep-convergence dashboard structure.

Purpose: standardize the S4 Tier-3 dt-refinement output shape before the actual
controlled convergence runner exists.
Answers: norms by variable, lead, and dt pair, plus an explicit placeholder
verdict that prevents this sidecar from claiming Tier-3 success.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `.agent/milestones/ROADMAP.md:89-98` defines M6 Tier-3 needs,
and `.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md:115-123`
specifies the controlled dt-refinement artifact shape.
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


def _norms(lhs: Any, rhs: Any) -> dict[str, float]:
    diff = _array(lhs) - _array(rhs)
    return {
        "l2": float(np.sqrt(np.nanmean(diff * diff))),
        "linf": float(np.nanmax(np.abs(diff))),
        "mean_bias": float(np.nanmean(diff)),
    }


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    pairs = []
    for pair in payload.get("dt_pairs", payload.get("pairs", [])):
        variables = {}
        coarse = pair.get("coarse", {})
        fine = pair.get("fine", {})
        for name in sorted(set(coarse) & set(fine)):
            variables[name] = _norms(coarse[name], fine[name])
        pairs.append(
            {
                "lead_time_s": pair.get("lead_time_s"),
                "dt_coarse_s": pair.get("dt_coarse_s"),
                "dt_fine_s": pair.get("dt_fine_s"),
                "variables": variables,
            }
        )

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "timestep_convergence_dashboard", "purpose": "Standardize Tier-3 dt-pair norms."},
        "input": {"path": str(input_path), "pair_count": len(pairs)},
        "measurements": {
            "dt_pairs": pairs,
            "convergence_verdict": "PLACEHOLDER_PENDING_S4",
            "verdict_note": "This S1 sidecar records structure only; S4 owns pass/fail convergence evidence.",
        },
        "units": {"dt": "s", "lead_time": "s", "norms": "variable-dependent"},
        "artifacts": {},
        "status": "STRUCTURE_ONLY",
        "source_citations": [
            ".agent/milestones/ROADMAP.md:89-98",
            ".agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md:115-123",
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
