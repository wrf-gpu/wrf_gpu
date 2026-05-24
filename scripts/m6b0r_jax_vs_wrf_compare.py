#!/usr/bin/env python
"""Compare JAX calc_coef_w coefficients against M6B0-R WRF savepoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients
from gpuwrf.validation.savepoint_io import read_savepoint
from gpuwrf.validation.savepoint_schema import load_tolerance_ladder


SPRINT = ROOT / ".agent/sprints/2026-05-24-m6b0r-real-fortran-emission"
COMPARE_FIELDS = ("a", "alpha", "gamma")


def _post_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("calc_coef_w_post_step*.h5"))


def _jax_calc(savepoint) -> dict[str, np.ndarray]:
    coeffs = build_epssm_column_coefficients(
        jnp.asarray(savepoint.arrays["theta"]),
        jnp.asarray(savepoint.arrays["dz_m"]),
        dt=savepoint.metadata.dt_seconds,
        epssm=0.1,
    )
    _cofrz, _cofwr, _cofwz, _coftz, _cofwt, _rdzw, tri_a, tri_b, tri_c = [
        np.asarray(jax.device_get(item)) for item in coeffs
    ]
    return {
        "a": tri_a,
        "alpha": 1.0 / tri_b,
        "gamma": tri_c / tri_b,
    }


def _threshold(entry: dict[str, object], expected: np.ndarray) -> float:
    abs_tol = float(entry["abs"]) if entry.get("abs") is not None else 0.0
    rel_tol = float(entry["rel"]) if entry.get("rel") is not None else 0.0
    scale = float(np.nanmax(np.abs(expected))) if expected.size else 1.0
    return max(abs_tol, rel_tol * max(scale, 1.0))


def compare_savepoint(path: Path, ladder: dict[str, object]) -> dict[str, object]:
    savepoint = read_savepoint(path)
    actual = _jax_calc(savepoint)
    fields: dict[str, object] = {}
    passed = True
    for name in COMPARE_FIELDS:
        expected = np.asarray(savepoint.arrays[name])
        got = np.asarray(actual[name])
        common_shape = tuple(min(a, b) for a, b in zip(expected.shape, got.shape))
        slices = tuple(slice(0, dim) for dim in common_shape)
        delta = got[slices] - expected[slices]
        max_abs = float(np.nanmax(np.abs(delta)))
        flat_index = int(np.nanargmax(np.abs(delta)))
        location = np.unravel_index(flat_index, delta.shape)
        entry = dict(ladder["fields"][name])  # type: ignore[index]
        tol = _threshold(entry, expected[slices])
        field_passed = bool(np.isfinite(max_abs) and max_abs <= tol and expected.shape == got.shape)
        fields[name] = {
            "max_abs_delta": max_abs,
            "tolerance": tol,
            "passed": field_passed,
            "location": [int(item) for item in location],
            "expected_shape": list(expected.shape),
            "actual_shape": list(got.shape),
            "units": entry["units"],
            "dtype": entry["dtype"],
            "abs_threshold": entry["abs"],
            "rel_threshold": entry["rel"],
            "ulp_threshold": entry["ulp"],
        }
        passed = passed and field_passed
    return {
        "path": str(path),
        "run_id": savepoint.metadata.run_id,
        "tier": savepoint.metadata.tier,
        "operator": savepoint.metadata.operator,
        "boundary": savepoint.metadata.boundary,
        "passed": bool(passed),
        "fields": fields,
    }


def compare_path(path: Path) -> dict[str, object]:
    ladder = load_tolerance_ladder()
    files = _post_files(path)
    if not files:
        raise ValueError(f"no calc_coef_w_post savepoints found under {path}")
    results = [compare_savepoint(file_path, ladder) for file_path in files]
    passed = all(bool(item["passed"]) for item in results)
    return {
        "operator": "calc_coef_w",
        "passed": bool(passed),
        "outcome": "PASS" if passed else "PARITY-DEFECT-LOCALIZED",
        "savepoint_count": len(results),
        "results": results,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads HDF5 on host before the JAX coefficient call; no timestep loop is executed.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("calc_coef_w",), required=True)
    parser.add_argument("--tier", choices=("column", "patch16", "golden", "all"), required=True)
    parser.add_argument("--savepoint-root", type=Path, default=SPRINT / "savepoints")
    parser.add_argument("--output", type=Path, default=SPRINT / "proof_real_coefficient_parity.json")
    args = parser.parse_args()

    tiers = ("column", "patch16", "golden") if args.tier == "all" else (args.tier,)
    tier_results = {tier: compare_path(args.savepoint_root / tier) for tier in tiers}
    passed = all(bool(item["passed"]) for item in tier_results.values())
    payload = {
        "operator": args.operator,
        "passed": bool(passed),
        "outcome": "PASS" if passed else "PARITY-DEFECT-LOCALIZED",
        "tiers": tier_results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
