#!/usr/bin/env python
"""Fail-closed synthetic HDF5 dry-run for the M6B0-R savepoint layer."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import h5py

from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import SCHEMA_VERSION, Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder


SPRINT = ROOT / ".agent/sprints/2026-05-24-m6b0r-real-fortran-emission"


def _metadata(arrays: dict[str, np.ndarray]) -> SavepointMetadata:
    return SavepointMetadata(
        run_id="m6b0r-synthetic-dryrun",
        wrf_version="synthetic",
        wrf_commit="115e5756f98ee2370d62b6709baac6417d8f7338",
        namelist_hash="synthetic-namelist",
        source_path="synthetic://m6b0r",
        domain_index=2,
        tier="column",
        operator="calc_coef_w",
        boundary="calc_coef_w_post",
        dt_seconds=6.0,
        rk_stage_index=1,
        acoustic_substep_index=1,
        map_factors={"MAPFAC_M": {"min": 1.0, "max": 1.0}},
        vertical_grid={"kind": "synthetic", "nz": 4},
        variables={
            name: VariableMetadata(
                name=name,
                dtype=str(array.dtype),
                shape=tuple(int(dim) for dim in array.shape),
                stagger="w" if name in {"a", "alpha", "gamma"} else "mass",
                units="dimensionless" if name in {"a", "alpha", "gamma"} else "K",
                provenance="synthetic dry-run",
                role="expected" if name in {"a", "alpha", "gamma"} else "input",
            )
            for name, array in arrays.items()
        },
        created_utc=datetime.now(timezone.utc).isoformat(),
    )


def _compare_self(savepoint: Savepoint, *, perturb: bool = False) -> dict[str, object]:
    ladder = load_tolerance_ladder()
    fields: dict[str, object] = {}
    passed = True
    for name in ("a", "alpha", "gamma"):
        expected = np.asarray(savepoint.arrays[name])
        actual = np.array(expected, copy=True)
        if perturb and name == "alpha":
            actual.flat[0] += 1.0e-3
        entry = ladder["fields"][name]
        tol = max(float(entry["abs"]), float(entry["rel"]) * max(float(np.nanmax(np.abs(expected))), 1.0))
        max_abs = float(np.nanmax(np.abs(actual - expected)))
        field_passed = bool(max_abs <= tol)
        fields[name] = {"max_abs_delta": max_abs, "tolerance": tol, "passed": field_passed}
        passed = passed and field_passed
    return {"passed": bool(passed), "fields": fields}


def main() -> int:
    out_dir = SPRINT / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (4, 4, 4)
    arrays = {
        "theta": np.full(shape, 300.0, dtype=np.float64),
        "a": np.zeros((5, 4, 4), dtype=np.float64),
        "alpha": np.ones((5, 4, 4), dtype=np.float64),
        "gamma": np.zeros((5, 4, 4), dtype=np.float64),
    }
    path = out_dir / "synthetic_calc_coef_w_post.h5"
    write_savepoint(path, Savepoint(metadata=_metadata(arrays), arrays=arrays))
    loaded = read_savepoint(path)
    clean = _compare_self(loaded)
    perturbed = _compare_self(loaded, perturb=True)

    version_failed = False
    try:
        read_savepoint(path, expected_schema_version="m6b0-savepoint-v0")
    except ValueError as exc:
        version_failed = "unsupported savepoint schema" in str(exc)

    tamper_path = out_dir / "synthetic_tampered.h5"
    tamper_path.write_bytes(path.read_bytes())
    with h5py.File(tamper_path, "a") as handle:
        data = handle["fields/alpha"]
        data[0, 0, 0] = data[0, 0, 0] + 1.0e-3
    tamper_failed = False
    try:
        read_savepoint(tamper_path)
    except ValueError:
        tamper_failed = True

    payload = {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "clean_self_compare_passed": clean["passed"],
        "perturbation_caught": not perturbed["passed"],
        "schema_version_mismatch_caught": version_failed,
        "tamper_detection_caught": tamper_failed,
        "clean": clean,
        "perturbed": perturbed,
    }
    payload["passed"] = all(
        [
            payload["clean_self_compare_passed"],
            payload["perturbation_caught"],
            payload["schema_version_mismatch_caught"],
            payload["tamper_detection_caught"],
        ]
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    (SPRINT / "proof_synthetic_dryrun.json").write_text(text + "\n")
    print(text)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
