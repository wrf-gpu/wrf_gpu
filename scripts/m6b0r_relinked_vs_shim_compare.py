#!/usr/bin/env python
"""Compare M6B0-R shim savepoints against the relinked-lane savepoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.validation.savepoint_io import read_savepoint


SPRINT = ROOT / ".agent/sprints/2026-05-24-m6b0r-relink-completion"
SHIM_ROOT = ROOT / ".agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints"
LEGACY_SHIM_ROOT = (
    Path("/tmp/wrf_gpu2_m6b0r")
    / ".agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints"
)
RELINKED_ROOT = ROOT / "external/wrf_savepoint_patch/savepoints/relinked"


def _compare_pair(shim_path: Path, relinked_path: Path) -> dict[str, object]:
    shim = read_savepoint(shim_path)
    relinked = read_savepoint(relinked_path)
    fields: dict[str, object] = {}
    passed = True
    for name in sorted(set(shim.arrays) & set(relinked.arrays)):
        lhs = np.asarray(shim.arrays[name])
        rhs = np.asarray(relinked.arrays[name])
        if lhs.shape != rhs.shape:
            fields[name] = {
                "passed": False,
                "reason": "shape mismatch",
                "shim_shape": list(lhs.shape),
                "relinked_shape": list(rhs.shape),
            }
            passed = False
            continue
        delta = rhs - lhs
        max_abs = float(np.nanmax(np.abs(delta))) if delta.size else 0.0
        field_passed = bool(np.isfinite(max_abs) and max_abs < 1.0e-12)
        fields[name] = {
            "passed": field_passed,
            "max_abs_delta": max_abs,
            "shim_shape": list(lhs.shape),
            "relinked_shape": list(rhs.shape),
            "dtype": str(rhs.dtype),
        }
        passed = passed and field_passed
    return {
        "shim_path": str(shim_path),
        "relinked_path": str(relinked_path),
        "boundary": shim.metadata.boundary,
        "operator": shim.metadata.operator,
        "rk_stage_index": shim.metadata.rk_stage_index,
        "acoustic_substep_index": shim.metadata.acoustic_substep_index,
        "passed": bool(passed),
        "fields": fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column",), required=True)
    parser.add_argument("--boundary", default="calc_coef_w_pre")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--shim-root", type=Path, default=SHIM_ROOT)
    parser.add_argument("--relinked-root", type=Path, default=RELINKED_ROOT)
    parser.add_argument("--output", type=Path, default=SPRINT / "proof_shim_vs_relinked_delta.json")
    args = parser.parse_args()

    name = f"{args.boundary}_step{args.step:03d}.h5"
    shim_path = args.shim_root / args.tier / name
    if not shim_path.exists() and args.shim_root == SHIM_ROOT:
        shim_path = LEGACY_SHIM_ROOT / args.tier / name
    result = _compare_pair(shim_path, args.relinked_root / args.tier / name)
    payload = {
        "status": "PASS" if result["passed"] else "SHIM-DIVERGENCE-DETECTED",
        "tier": args.tier,
        "comparison": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
