#!/usr/bin/env python3
"""Compare two small JSON numeric arrays with tolerances."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def flatten(value):
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten(item))
        return out
    return [float(value)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    expected = flatten(json.loads(args.expected.read_text(encoding="utf-8")))
    actual = flatten(json.loads(args.actual.read_text(encoding="utf-8")))
    errors = []
    if len(expected) != len(actual):
        errors.append(f"length mismatch {len(expected)} != {len(actual)}")
    else:
        for index, (lhs, rhs) in enumerate(zip(expected, actual)):
            if not math.isclose(lhs, rhs, abs_tol=args.atol, rel_tol=args.rtol):
                errors.append(f"index {index}: expected {lhs}, actual {rhs}")
                break
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
