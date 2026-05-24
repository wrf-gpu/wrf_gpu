#!/usr/bin/env python
"""Compare JAX coefficient construction against M6B0 savepoints."""

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
from gpuwrf.validation.savepoint_schema import Savepoint


COEFFICIENT_FIELDS = ("cofrz", "cofwr", "cofwz", "coftz", "cofwt", "rdzw", "tri_a", "tri_b", "tri_c")
TOLERANCE_LADDER = {
    "cofrz": 1.0e-12,
    "cofwr": 1.0e-12,
    "cofwz": 1.0e-11,
    "coftz": 1.0e-11,
    "cofwt": 1.0e-12,
    "rdzw": 1.0e-12,
    "tri_a": 1.0e-11,
    "tri_b": 1.0e-11,
    "tri_c": 1.0e-11,
}


def _iter_savepoints(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("coefficient_construction_step*.npz"))


def _compare_one(
    savepoint: Savepoint,
    *,
    perturb_field: str | None = None,
    perturbation: float = 0.0,
) -> dict[str, object]:
    arrays = dict(savepoint.arrays)
    if perturb_field:
        if perturb_field not in arrays:
            raise ValueError(f"perturb_field not present: {perturb_field}")
        perturbed = np.array(arrays[perturb_field], copy=True)
        perturbed.flat[0] += perturbation
        arrays[perturb_field] = perturbed
    coeffs = build_epssm_column_coefficients(
        jnp.asarray(arrays["theta"]),
        jnp.asarray(arrays["dz_m"]),
        dt=savepoint.metadata.dt_seconds,
        epssm=0.1,
    )
    expected = {name: np.asarray(savepoint.arrays[name]) for name in COEFFICIENT_FIELDS}
    actual = {name: np.asarray(jax.device_get(value)) for name, value in zip(COEFFICIENT_FIELDS, coeffs)}
    fields = {}
    passed = True
    for name in COEFFICIENT_FIELDS:
        delta = np.asarray(actual[name] - expected[name])
        max_abs = float(np.nanmax(np.abs(delta)))
        tol = float(TOLERANCE_LADDER[name])
        field_passed = bool(np.isfinite(max_abs) and max_abs <= tol)
        fields[name] = {
            "max_abs_delta": max_abs,
            "tolerance": tol,
            "passed": field_passed,
            "shape": list(actual[name].shape),
        }
        passed = passed and field_passed
    return {
        "run_id": savepoint.metadata.run_id,
        "tier": savepoint.metadata.tier,
        "operator": savepoint.metadata.operator,
        "boundary": savepoint.metadata.boundary,
        "passed": bool(passed),
        "fields": fields,
        "perturbation": {"field": perturb_field, "value": perturbation} if perturb_field else None,
    }


def compare_path(
    path: Path,
    *,
    perturb_field: str | None = None,
    perturbation: float = 0.0,
) -> dict[str, object]:
    files = _iter_savepoints(path)
    if not files:
        raise ValueError(f"no coefficient_construction savepoints found under {path}")
    results = []
    passed = True
    for file_path in files:
        savepoint = read_savepoint(file_path)
        if savepoint.metadata.operator != "coefficient_construction":
            continue
        item = _compare_one(savepoint, perturb_field=perturb_field, perturbation=perturbation)
        item["path"] = str(file_path)
        results.append(item)
        passed = passed and bool(item["passed"])
    if not results:
        raise ValueError(f"no coefficient_construction operator savepoints found under {path}")
    return {
        "operator": "coefficient_construction",
        "passed": bool(passed),
        "savepoint_count": len(results),
        "tolerance_ladder": TOLERANCE_LADDER,
        "results": results,
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads NPZ on host before JAX coefficient call; no timestep loop is executed.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("coefficient_construction",), required=True)
    parser.add_argument("--savepoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--perturb-field")
    parser.add_argument("--perturbation", type=float, default=0.0)
    args = parser.parse_args()

    payload = compare_path(args.savepoint, perturb_field=args.perturb_field, perturbation=args.perturbation)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
