#!/usr/bin/env python3
"""Audit whether sanitizer/finite guards are masking replay failures.

Purpose: summarize pre-sanitize nonfinite, clip, and changed-value counts.
Answers: first bad candidate step, total nonfinite/clip/change counts, and
whether post-sanitize finiteness alone is hiding candidate instability.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `src/gpuwrf/integration/d02_replay.py:313-322` defines
per-step sanitizer diagnostics, and `src/gpuwrf/integration/d02_replay.py:468-487`
summarizes candidate guard counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("sanitizer_steps"), list):
        return list(payload["sanitizer_steps"])
    if isinstance(payload.get("steps"), list):
        return list(payload["steps"])
    diagnostics = payload.get("diagnostics", {})
    if {"candidate_nonfinite_count_total", "candidate_clip_count_total", "candidate_changed_count_total"} <= set(diagnostics):
        return [
            {
                "step": None,
                "time_s": None,
                "candidate_nonfinite_count": diagnostics.get("candidate_nonfinite_count_total", 0),
                "candidate_clip_count": diagnostics.get("candidate_clip_count_total", 0),
                "candidate_changed_count": diagnostics.get("candidate_changed_count_total", 0),
            }
        ]
    return []


def _int(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def build_payload(input_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    rows = _step_rows(payload)
    per_step = []
    first_bad = None
    totals = {"candidate_nonfinite_count": 0, "candidate_clip_count": 0, "candidate_changed_count": 0}
    for index, row in enumerate(rows):
        item = {
            "step": row.get("step", index + 1),
            "time_s": row.get("time_s"),
            "candidate_nonfinite_count": _int(row, "candidate_nonfinite_count"),
            "candidate_clip_count": _int(row, "candidate_clip_count"),
            "candidate_changed_count": _int(row, "candidate_changed_count"),
            "finite_after_sanitize": bool(row.get("finite_after_sanitize", True)),
        }
        for key in totals:
            totals[key] += item[key]
        if first_bad is None and (
            item["candidate_nonfinite_count"] > 0 or item["candidate_clip_count"] > 0 or item["candidate_changed_count"] > 0
        ):
            first_bad = item
        per_step.append(item)

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "sanitizer_audit", "purpose": "Expose pre-sanitize candidate failures."},
        "input": {"path": str(input_path), "step_count": len(rows)},
        "measurements": {
            "per_step": per_step,
            "first_bad_candidate_step": first_bad,
            "totals": totals,
            "post_sanitize_only_pass_risk": bool(first_bad),
        },
        "units": {"counts": "count", "time_s": "s"},
        "artifacts": {},
        "status": "FAIL_SANITIZER_MASKING" if first_bad else "OK",
        "source_citations": [
            "src/gpuwrf/integration/d02_replay.py:313-322",
            "src/gpuwrf/integration/d02_replay.py:468-487",
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
