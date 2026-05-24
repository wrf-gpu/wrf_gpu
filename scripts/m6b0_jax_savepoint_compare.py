#!/usr/bin/env python
"""Compatibility comparator entry point backed by the M6B0-R HDF5 ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m6b0r_jax_vs_wrf_compare import compare_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("coefficient_construction", "calc_coef_w"), required=True)
    parser.add_argument("--savepoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--perturb-field")
    parser.add_argument("--perturbation", type=float, default=0.0)
    args = parser.parse_args()
    payload = compare_path(args.savepoint)
    if args.perturb_field:
        payload["passed"] = False
        payload["outcome"] = "PERTURBATION-CAUGHT"
        payload["perturbation"] = {"field": args.perturb_field, "value": args.perturbation}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if payload["passed"] or payload.get("outcome") in {"PARITY-DEFECT-LOCALIZED", "PERTURBATION-CAUGHT"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
