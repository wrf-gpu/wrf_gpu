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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sprint_path", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for name in REQUIRED:
        path = args.sprint_path / name
        if not path.exists():
            errors.append(f"missing {name}")
        elif path.stat().st_size == 0:
            errors.append(f"empty {name}")
    artifacts = args.sprint_path / "artifacts"
    if not artifacts.exists():
        errors.append("missing artifacts directory")
    result = {"ok": not errors, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
