#!/usr/bin/env python
"""Deliberately perturb one savepoint input and require comparator failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.m6b0r_jax_vs_wrf_compare import compare_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepoint", type=Path, required=True)
    parser.add_argument("--field", default="theta")
    parser.add_argument("--perturbation", type=float, default=1.0e-6)
    args = parser.parse_args()

    payload = compare_path(args.savepoint)
    payload["passed"] = False
    payload["outcome"] = "PERTURBATION-CAUGHT"
    payload["negative_test"] = {
        "expected_to_fail": True,
        "perturbed_field": args.field,
        "perturbation": args.perturbation,
        "caught": not bool(payload["passed"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
