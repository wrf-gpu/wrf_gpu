#!/usr/bin/env python3
"""Check whether a sprint has the required closeout files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED = [
    "sprint-contract.md",
    "worker-report.md",
    "reviewer-report.md",
    "tester-report.md",
    "manager-closeout.md",
    "memory-patch.md",
]

# Each report must include a decision token so reviewer/tester/manager can't
# be passed off as untouched template stubs (the templates have empty H2 sections).
DECISION_TOKENS = {
    "reviewer-report.md": ("Decision:",),
    "tester-report.md": ("Decision:",),
    "manager-closeout.md": ("Merge Decision:",),
    "memory-patch.md": ("Reviewer Status:",),
    "worker-report.md": ("Summary:",),
}

# Minimum body size after frontmatter — generous, just defeats the empty-template
# (200-byte) case without prescribing a specific report length.
MIN_BYTES = 400


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sprint_path", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for name in REQUIRED:
        path = args.sprint_path / name
        if not path.exists():
            errors.append(f"missing {name}")
            continue
        size = path.stat().st_size
        if size == 0:
            errors.append(f"empty {name}")
            continue
        if size < MIN_BYTES and name != "sprint-contract.md":
            errors.append(f"{name} too short ({size} < {MIN_BYTES} bytes) — likely an unfinished stub")
            continue
        text = path.read_text(errors="replace")
        tokens = DECISION_TOKENS.get(name, ())
        if tokens and not any(tok in text for tok in tokens):
            errors.append(
                f"{name} missing decision token (one of {list(tokens)}) — looks like an unfilled template"
            )
    artifacts = args.sprint_path / "artifacts"
    if not artifacts.exists():
        errors.append("missing artifacts directory")
    result = {"ok": not errors, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
