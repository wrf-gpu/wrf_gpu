#!/usr/bin/env python
"""Emit and compare M6B2 WRF-shaped Thomas-solve savepoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.tridiag_solve import thomas_back_scan, thomas_forward_scan
from gpuwrf.validation.comparator_common import field_compare, field_tolerance
from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder


# Sibling-script import without the `sys.path.insert(0, scripts)` hack. The
# m6b0r extractor is not a package yet (queued in
# `m6b0r-fortran-hook-abi-followup`); use importlib so we don't pollute
# sys.path with a non-package directory.
def _import_m6b0r_extract():
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "m6b0r_wrf_savepoint_extract",
        ROOT / "scripts" / "m6b0r_wrf_savepoint_extract.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("m6b0r_wrf_savepoint_extract.py not found in scripts/")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_m6b0r = _import_m6b0r_extract()
SOURCE_WRFOUT = _m6b0r.SOURCE_WRFOUT
WRF_COMMIT = _m6b0r.WRF_COMMIT
_load_state = _m6b0r._load_state
_sha256_path = _m6b0r._sha256_path
_wrf_calc_coef_w = _m6b0r._wrf_calc_coef_w


# Backwards-compat alias for any reference to the historical helper.
_threshold = field_tolerance
_field_compare = field_compare


SPRINT = ROOT / ".agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity"
COMPARE_FWD_FIELDS = ("tri_fwd",)
COMPARE_BACK_FIELDS = ("tri_solution",)
ALL_COMPARE_FIELDS = COMPARE_FWD_FIELDS + COMPARE_BACK_FIELDS


def _m6b2_run_id(source_run_id: str, tier: str) -> str:
    if tier == "golden":
        return source_run_id.replace("m6b0r", "m6b1", 1)
    return source_run_id.replace("m6b0r", "m6b2", 1)


def _initial_w(state: dict[str, object]) -> np.ndarray:
    attrs = dict(state["attrs"])  # type: ignore[arg-type]
    ys = slice(*attrs["slice_y"])
    xs = slice(*attrs["slice_x"])
    with Dataset(SOURCE_WRFOUT) as ds:
        if "W" in ds.variables:
            return np.asarray(ds.variables["W"][0, :, ys, xs], dtype=np.float64)
    theta = np.asarray(state["theta"], dtype=np.float64)
    return np.zeros((theta.shape[0] + 1,) + theta.shape[1:], dtype=np.float64)


def _wrf_forward_numpy(a: np.ndarray, alpha: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    out = np.array(rhs, copy=True, dtype=np.float64)
    for k in range(1, out.shape[0]):
        out[k] = (out[k] - a[k] * out[k - 1]) * alpha[k]
    return out


def _wrf_back_numpy(gamma: np.ndarray, w_fwd: np.ndarray) -> np.ndarray:
    out = np.array(w_fwd, copy=True, dtype=np.float64)
    for k in range(out.shape[0] - 2, 0, -1):
        out[k] = out[k] - gamma[k] * out[k + 1]
    return out


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    meta = {}
    for name, array in arrays.items():
        arr = np.asarray(array)
        units = "operator-native"
        stagger = "scalar"
        if name in {"tri_a", "tri_alpha", "tri_gamma"}:
            units = "dimensionless"
            stagger = "w"
        elif name in {"tri_rhs", "tri_fwd", "tri_solution"}:
            units = "m s-1"
            stagger = "w"
        elif name == "mut":
            units = "Pa"
            stagger = "mass"
        meta[name] = VariableMetadata(
            name=name,
            dtype=str(arr.dtype),
            shape=tuple(int(dim) for dim in arr.shape),
            stagger=stagger,
            units=units,
            provenance="WRF dyn_em/module_small_step_em.F advance_w Thomas sweep lines 1533-1550",
            role=roles.get(name, "input"),
        )
    return meta


def _savepoint(
    *,
    tier: str,
    boundary: str,
    step: int,
    state: dict[str, object],
    arrays: dict[str, np.ndarray],
    roles: dict[str, str],
) -> Savepoint:
    attrs = dict(state["attrs"])  # type: ignore[arg-type]
    metadata_attrs = {k: v for k, v in attrs.items() if k != "dims"}
    return Savepoint(
        metadata=SavepointMetadata(
            run_id=f"{_m6b2_run_id(str(state['run_id']), tier)}-step{step:03d}-{boundary}",
            wrf_version="WRF-Gen2-artifact",
            wrf_commit=WRF_COMMIT,
            namelist_hash=hashlib.sha256(json.dumps(metadata_attrs, sort_keys=True).encode()).hexdigest(),
            source_path=str(SOURCE_WRFOUT),
            domain_index=2,
            tier=tier,
            operator="advance_w",
            boundary=boundary,
            dt_seconds=float(attrs["dt"]),
            rk_stage_index=1,
            acoustic_substep_index=step,
            map_factors={"MAPFAC_M": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
            vertical_grid={
                "kind": "wrf-hybrid-eta",
                "nz": int(np.asarray(arrays["tri_rhs" if "tri_rhs" in arrays else "tri_fwd"]).shape[0] - 1),
                "advance_w_tridiag_attrs": metadata_attrs,
            },
            variables=_var_meta(arrays, roles),
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Sanitizer-off M6B2 WRF-shaped Thomas-solve extraction from real Canary d02 wrfout. "
                "Forward/back expected arrays are emitted by a NumPy transcription of WRF lines 1533-1550."
            ),
        ),
        arrays=arrays,
    )


def emit_tier(tier: str, steps: int, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    state = _load_state(tier)
    attrs = dict(state["attrs"])  # type: ignore[arg-type]
    coeffs = _wrf_calc_coef_w(state, dts=float(attrs["dt"]), epssm=float(attrs["epssm"]))
    a = np.asarray(coeffs["a"], dtype=np.float64)
    alpha = np.asarray(coeffs["alpha"], dtype=np.float64)
    gamma = np.asarray(coeffs["gamma"], dtype=np.float64)
    rhs = _initial_w(state)
    files = []
    for step in range(1, steps + 1):
        fwd_pre_arrays = {
            "tri_a": a,
            "tri_alpha": alpha,
            "tri_gamma": gamma,
            "tri_rhs": np.asarray(rhs, dtype=np.float64),
            "mut": np.asarray(state["mut"], dtype=np.float64),
        }
        fwd_pre_path = output / f"advance_w_tridiag_fwd_pre_step{step:03d}.h5"
        write_savepoint(
            fwd_pre_path,
            _savepoint(tier=tier, boundary="advance_w_tridiag_fwd_pre", step=step, state=state, arrays=fwd_pre_arrays, roles={}),
        )
        files.append(fwd_pre_path)

        fwd = _wrf_forward_numpy(a, alpha, rhs)
        fwd_post_arrays = {**fwd_pre_arrays, "tri_fwd": fwd}
        fwd_post_path = output / f"advance_w_tridiag_fwd_post_step{step:03d}.h5"
        write_savepoint(
            fwd_post_path,
            _savepoint(
                tier=tier,
                boundary="advance_w_tridiag_fwd_post",
                step=step,
                state=state,
                arrays=fwd_post_arrays,
                roles={"tri_fwd": "expected"},
            ),
        )
        files.append(fwd_post_path)

        back_pre_arrays = {"tri_gamma": gamma, "tri_fwd": fwd}
        back_pre_path = output / f"advance_w_tridiag_back_pre_step{step:03d}.h5"
        write_savepoint(
            back_pre_path,
            _savepoint(tier=tier, boundary="advance_w_tridiag_back_pre", step=step, state=state, arrays=back_pre_arrays, roles={}),
        )
        files.append(back_pre_path)

        solution = _wrf_back_numpy(gamma, fwd)
        back_post_arrays = {**back_pre_arrays, "tri_solution": solution}
        back_post_path = output / f"advance_w_tridiag_back_post_step{step:03d}.h5"
        write_savepoint(
            back_post_path,
            _savepoint(
                tier=tier,
                boundary="advance_w_tridiag_back_post",
                step=step,
                state=state,
                arrays=back_post_arrays,
                roles={"tri_solution": "expected"},
            ),
        )
        files.append(back_post_path)
        rhs = solution

    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "run_id": _m6b2_run_id(str(state["run_id"]), tier),
        "steps": list(range(1, steps + 1)),
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "attrs": attrs,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
        "direct_relinked_wrf": False,
        "wrf_source_lines": "module_small_step_em.F:1533-1550",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def compare_step(root: Path, step: int, ladder: dict[str, object]) -> dict[str, object]:
    fwd_pre = read_savepoint(root / f"advance_w_tridiag_fwd_pre_step{step:03d}.h5")
    fwd_post = read_savepoint(root / f"advance_w_tridiag_fwd_post_step{step:03d}.h5")
    back_pre = read_savepoint(root / f"advance_w_tridiag_back_pre_step{step:03d}.h5")
    back_post = read_savepoint(root / f"advance_w_tridiag_back_post_step{step:03d}.h5")

    fwd_actual = np.asarray(
        thomas_forward_scan(
            jnp.asarray(fwd_pre.arrays["tri_a"]),
            jnp.asarray(fwd_pre.arrays["tri_alpha"]),
            jnp.asarray(fwd_pre.arrays["tri_rhs"]),
        )
    )
    back_actual = np.asarray(thomas_back_scan(jnp.asarray(back_pre.arrays["tri_gamma"]), jnp.asarray(back_pre.arrays["tri_fwd"])))
    fwd_fields = {"tri_fwd": _field_compare("tri_fwd", fwd_actual, np.asarray(fwd_post.arrays["tri_fwd"]), ladder)}
    back_fields = {
        "tri_solution": _field_compare("tri_solution", back_actual, np.asarray(back_post.arrays["tri_solution"]), ladder)
    }
    fwd_passed = all(bool(item["passed"]) for item in fwd_fields.values())
    back_passed = all(bool(item["passed"]) for item in back_fields.values())
    return {
        "step": step,
        "tier": fwd_post.metadata.tier,
        "fwd": {
            "path": str(root / f"advance_w_tridiag_fwd_post_step{step:03d}.h5"),
            "boundary": fwd_post.metadata.boundary,
            "passed": bool(fwd_passed),
            "fields": fwd_fields,
        },
        "back": {
            "path": str(root / f"advance_w_tridiag_back_post_step{step:03d}.h5"),
            "boundary": back_post.metadata.boundary,
            "passed": bool(back_passed),
            "fields": back_fields,
        },
        "passed": bool(fwd_passed and back_passed),
    }


def compare_tier(tier: str, steps: int, savepoint_root: Path) -> dict[str, object]:
    output = savepoint_root / tier
    manifest = emit_tier(tier, steps, output)
    ladder = load_tolerance_ladder()
    results = [compare_step(output, step, ladder) for step in range(1, steps + 1)]
    passed = all(bool(item["passed"]) for item in results)
    fwd_failed = any(not bool(item["fwd"]["passed"]) for item in results)
    back_failed = any(not bool(item["back"]["passed"]) for item in results)
    if passed:
        outcome = "PASS"
    elif fwd_failed:
        outcome = "PARITY-DEFECT-LOCALIZED-IN-TRIDIAG-FWD"
    elif back_failed:
        outcome = "PARITY-DEFECT-LOCALIZED-IN-TRIDIAG-BACK"
    else:
        outcome = "PARITY-DEFECT-LOCALIZED-IN-TRIDIAG"
    return {
        "operator": "advance_w_tridiag",
        "tier": tier,
        "passed": bool(passed),
        "outcome": outcome,
        "savepoint_count": len(results),
        "manifest": manifest,
        "results": results,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads HDF5 before isolated JAX scan calls; no production timestep loop is executed.",
        },
    }


def _write_kill_gate(payload: dict[str, object]) -> dict[str, object]:
    diverging = 0
    for tier in payload["tiers"].values():  # type: ignore[union-attr]
        first = tier["results"][0]  # type: ignore[index]
        diverging += sum(1 for item in first["fwd"]["fields"].values() if not bool(item["passed"]))  # type: ignore[index]
        diverging += sum(1 for item in first["back"]["fields"].values() if not bool(item["passed"]))  # type: ignore[index]
    status = {
        "operator": "advance_w_tridiag",
        "substep": 1,
        "diverging_field_count": int(diverging),
        "threshold": 15,
        "passed": bool(diverging <= 15),
        "decision": "PROCEED_TO_M6B3" if diverging <= 15 else "STOP_ESCALATE_M6B2",
    }
    text = json.dumps(status, indent=2, sort_keys=True)
    (SPRINT / "proof_kill_gate_status.txt").write_text(text + "\n")
    return status


def synthetic_dryrun() -> dict[str, object]:
    a = np.zeros((5, 2, 1), dtype=np.float64)
    alpha = np.ones_like(a)
    gamma = np.zeros_like(a)
    rhs = np.arange(10, dtype=np.float64).reshape(5, 2, 1) * 0.01
    a[2:, :, :] = -0.05
    alpha[1:, :, :] = 0.97
    gamma[1:-1, :, :] = -0.04
    fwd_expected = _wrf_forward_numpy(a, alpha, rhs)
    sol_expected = _wrf_back_numpy(gamma, fwd_expected)
    fwd_actual = np.asarray(thomas_forward_scan(jnp.asarray(a), jnp.asarray(alpha), jnp.asarray(rhs)))
    sol_actual = np.asarray(thomas_back_scan(jnp.asarray(gamma), jnp.asarray(fwd_expected)))

    ladder = load_tolerance_ladder()
    fwd_clean = _field_compare("tri_fwd", fwd_actual, fwd_expected, ladder)
    back_clean = _field_compare("tri_solution", sol_actual, sol_expected, ladder)
    fwd_tol = float(fwd_clean["tolerance"])
    back_tol = float(back_clean["tolerance"])
    fwd_perturbed = np.array(fwd_expected, copy=True)
    sol_perturbed = np.array(sol_expected, copy=True)
    fwd_perturbed.flat[0] += 20.0 * fwd_tol
    sol_perturbed.flat[0] += 20.0 * back_tol
    fwd_perturb = _field_compare("tri_fwd", fwd_actual, fwd_perturbed, ladder)
    back_perturb = _field_compare("tri_solution", sol_actual, sol_perturbed, ladder)
    payload = {
        "operator": "advance_w_tridiag",
        "clean_self_compare_passed": bool(fwd_clean["passed"] and back_clean["passed"]),
        "fwd_and_back_perturbations_caught": bool(not fwd_perturb["passed"] and not back_perturb["passed"]),
        "clean": {"tri_fwd": fwd_clean, "tri_solution": back_clean},
        "perturbed": {"tri_fwd": fwd_perturb, "tri_solution": back_perturb},
        "passed": bool(fwd_clean["passed"] and back_clean["passed"] and not fwd_perturb["passed"] and not back_perturb["passed"]),
        "sanitizer_mode": "off",
        "wrf_source_lines": "module_small_step_em.F:1533-1550",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (SPRINT / "proof_synthetic_dryrun_m6b2.json").write_text(text + "\n")
    (SPRINT / "proof_synthetic_dryrun_m6b2.txt").write_text(text + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden", "all"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--savepoint-root", type=Path, default=SPRINT / "savepoints")
    parser.add_argument("--output", type=Path, default=SPRINT / "proof_tridiag_solve_parity.json")
    parser.add_argument("--synthetic-dryrun", action="store_true")
    args = parser.parse_args()

    if args.synthetic_dryrun:
        payload = synthetic_dryrun()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 2
    if args.tier is None:
        parser.error("--tier is required unless --synthetic-dryrun is set")

    tiers = ("column", "patch16", "golden") if args.tier == "all" else (args.tier,)
    tier_results = {tier: compare_tier(tier, args.steps, args.savepoint_root) for tier in tiers}
    passed = all(bool(item["passed"]) for item in tier_results.values())
    fwd_failed = any("TRIDIAG-FWD" in str(item["outcome"]) for item in tier_results.values())
    back_failed = any("TRIDIAG-BACK" in str(item["outcome"]) for item in tier_results.values())
    if passed:
        outcome = "PASS"
    elif fwd_failed:
        outcome = "PARITY-DEFECT-LOCALIZED-IN-TRIDIAG-FWD"
    elif back_failed:
        outcome = "PARITY-DEFECT-LOCALIZED-IN-TRIDIAG-BACK"
    else:
        outcome = "PARITY-DEFECT-LOCALIZED-IN-TRIDIAG"
    payload = {
        "operator": "advance_w_tridiag",
        "passed": bool(passed),
        "outcome": outcome,
        "tiers": tier_results,
    }
    if args.tier == "all":
        payload["kill_gate"] = _write_kill_gate(payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
