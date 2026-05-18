#!/usr/bin/env python3
"""Validate that an ADR contains required fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED = [
    "## Status",
    "## Context",
    "## Decision",
    "## Alternatives",
    "## Consequences",
    "## Proof Objects",
    "## Review",
]


def validate(path: Path) -> dict:
    errors: list[str] = []
    if not path.exists():
        return {"ok": False, "errors": [f"missing file: {path}"]}
    text = path.read_text(encoding="utf-8")
    for marker in REQUIRED:
        if marker not in text:
            errors.append(f"missing {marker}")
    return {"ok": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("adr", type=Path)
    args = parser.parse_args()
    result = validate(args.adr)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
