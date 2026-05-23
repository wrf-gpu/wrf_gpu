#!/usr/bin/env python3
"""Scan stabilizer-like code for source-backed provenance.

Purpose: classify named stabilizers as source-backed, experiment-backed, or
reject before operator changes consume them.
Answers: source lines, matched stabilizer keywords/constants, and provenance
classification.
Expected output schema: JSON with top-level `schema_version`, `diagnostic`,
`input`, `measurements`, `units`, `status`, and `source_citations`.
Source citations: `scripts/m6_warm_bubble_test.py:326-360` implements the
anti-clamp scanner, and `src/gpuwrf/dynamics/acoustic_wrf.py:456-475` documents
the temporary mu-continuity limiter.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STABILIZER_RE = re.compile(r"(tanh|clip|clamp|limit|limiter|rayleigh|smdiv|damping|drag|sponge|filter)", re.IGNORECASE)
TARGET_RE = re.compile(r"(9\.0|10\.0|5\.0|target|positive-only|lift_bias|updraft_drag)", re.IGNORECASE)
SOURCE_RE = re.compile(r"(wrf source|mpas|module_small_step_em|mpas_atm_time_integration|klemp|source anchor)", re.IGNORECASE)
EXPERIMENT_RE = re.compile(r"(temporary|validation|prototype|experiment|diagnostic)", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return [path]
            if isinstance(payload.get("source_files"), list):
                return [Path(item) for item in payload["source_files"]]
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*.py") if item.is_file())
    return []


def _classify(line: str, context: str) -> str:
    text = f"{context}\n{line}"
    if TARGET_RE.search(text):
        return "reject"
    if SOURCE_RE.search(text):
        return "source-backed"
    if EXPERIMENT_RE.search(text):
        return "experiment-backed"
    return "experiment-backed"


def build_payload(input_path: Path) -> dict[str, Any]:
    findings = []
    for path in _candidate_files(input_path):
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines, start=1):
            if not STABILIZER_RE.search(line):
                continue
            context = "\n".join(lines[max(0, index - 4) : min(len(lines), index + 3)])
            findings.append(
                {
                    "path": str(path),
                    "line": index,
                    "snippet": line.strip(),
                    "classification": _classify(line, context),
                }
            )
    counts: dict[str, int] = {"source-backed": 0, "experiment-backed": 0, "reject": 0}
    for item in findings:
        counts[item["classification"]] += 1

    return {
        "schema_version": "m6x-s1-diagnostic-sidecar-v1",
        "diagnostic": {"name": "stabilizer_provenance_scanner", "purpose": "Classify stabilizer provenance."},
        "input": {"path": str(input_path), "files_scanned": len(_candidate_files(input_path))},
        "measurements": {"findings": findings, "classification_counts": counts},
        "units": {"findings": "count"},
        "artifacts": {},
        "status": "REJECT_FOUND" if counts["reject"] else "OK",
        "source_citations": [
            "scripts/m6_warm_bubble_test.py:326-360",
            "src/gpuwrf/dynamics/acoustic_wrf.py:456-475",
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
