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

import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.metrics import load_wrfinput_metrics
from gpuwrf.validation.comparator_common import field_tolerance
from gpuwrf.validation.savepoint_io import read_savepoint
from gpuwrf.validation.savepoint_schema import load_tolerance_ladder


# Backwards-compat alias for any script that still imports `_threshold`.
_threshold = field_tolerance


SPRINT = ROOT / ".agent/sprints/2026-05-24-m6b0r-real-fortran-emission"
COMPARE_FIELDS = ("a", "alpha", "gamma")


def _post_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("calc_coef_w_post_step*.h5"))


_METRICS_CACHE: dict[str, object] = {}


def _metrics_for_source(path: str):
    if path not in _METRICS_CACHE:
        _METRICS_CACHE[path] = load_wrfinput_metrics(path)
    return _METRICS_CACHE[path]


def _jax_calc(savepoint) -> dict[str, np.ndarray]:
    # WRF calc_coef_w uses hybrid-coordinate mass denominators, not dz/theta
    # metric coefficients. Source: module_small_step_em.F:624-649.
    # The Canary d02 namelist has TOP_LID=F, so WRF line 620 leaves lid_flag=1.
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        jnp.asarray(savepoint.arrays["mut"]),
        _metrics_for_source(savepoint.metadata.source_path),
        dt=savepoint.metadata.dt_seconds,
        epssm=0.1,
        top_lid=False,
    )
    return {
        "a": np.asarray(a),
        "alpha": np.asarray(alpha),
        "gamma": np.asarray(gamma),
    }


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
        tol = field_tolerance(entry, expected[slices])
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
