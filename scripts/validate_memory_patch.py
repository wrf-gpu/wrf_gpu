#!/usr/bin/env python3
"""Validate a proposed memory or skill patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED_MARKERS = {
    "scope": ["scope"],
    "evidence": ["evidence", "proof"],
    "proposed_destination": ["proposed destination", "destination"],
    "reviewer_status": ["reviewer status", "review status"],
}


def validate(path: Path) -> dict:
    errors: list[str] = []
    if not path.exists():
        return {"ok": False, "errors": [f"missing file: {path}"]}
    text = path.read_text(encoding="utf-8").lower()
    if not text.strip():
        errors.append("patch is empty")
    for label, markers in REQUIRED_MARKERS.items():
        if not any(marker in text for marker in markers):
            errors.append(f"missing {label}")
    if "approved" in text and "evidence" not in text:
        errors.append("approval without evidence is invalid")
    return {"ok": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", type=Path)
    args = parser.parse_args()
    result = validate(args.patch)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
