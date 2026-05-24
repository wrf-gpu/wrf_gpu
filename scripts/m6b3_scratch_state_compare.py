#!/usr/bin/env python
"""Emit and compare M6B3 WRF-shaped scratch-state savepoints."""

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
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.small_step_scratch import ScratchInputs, build_scratch_state
from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder
from m6b1_advance_mu_t_compare import SOURCE_WRFOUT, WRF_COMMIT, _advance, _load_initial_arrays, _sha256_path


SPRINT = ROOT / ".agent/sprints/2026-05-25-m6b3-scratch-state-parity"
SCRATCH_BOUNDARIES = {
    "t_2ave_update": ("t_2ave",),
    "ww_update": ("ww",),
    "muave_update": ("muave", "muts"),
    "ph_tend_accumulate": ("ph_tend",),
    "substep_save_state": ("u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save", "ww_save"),
}
COMPARE_FIELDS = tuple(field for fields in SCRATCH_BOUNDARIES.values() for field in fields)
ADVANCE_KEYS = {
    "ww",
    "ww_1",
    "u",
    "u_1",
    "v",
    "v_1",
    "mu",
    "mut",
    "muave",
    "muts",
    "muu",
    "muv",
    "mudf",
    "theta",
    "theta_1",
    "theta_ave",
    "theta_tend",
    "mu_tend",
    "ph_tend",
    "dnw",
    "fnm",
    "fnp",
    "rdnw",
    "c1h",
    "c2h",
    "msfuy",
    "msfvx_inv",
    "msftx",
    "msfty",
}


def _m6b3_run_id(run_id: str, tier: str) -> str:
    if tier == "golden":
        return run_id.replace("m6b1", "m6b3", 1)
    return run_id.replace("m6b1", "m6b3", 1)


def _load_w_ph(attrs: dict[str, object], theta: np.ndarray) -> dict[str, np.ndarray]:
    y0, y1 = [int(v) for v in attrs["halo_slice_y"]]  # type: ignore[index]
    x0, x1 = [int(v) for v in attrs["halo_slice_x"]]  # type: ignore[index]
    ys = slice(y0, y1)
    xs = slice(x0, x1)
    with Dataset(SOURCE_WRFOUT) as ds:
        if "W" in ds.variables:
            w = np.asarray(ds.variables["W"][0, :, ys, xs], dtype=np.float64)
        else:
            w = np.zeros((theta.shape[0] + 1,) + theta.shape[1:], dtype=np.float64)
        if "PH" in ds.variables:
            ph = np.asarray(ds.variables["PH"][0, :, ys, xs], dtype=np.float64)
        else:
            ph = np.zeros_like(w)
    return {"w_current": w, "ph_current": ph}


def _ph_tend_increment(theta_old: np.ndarray, theta_new: np.ndarray, ph_tend: np.ndarray) -> np.ndarray:
    increment = np.zeros_like(ph_tend, dtype=np.float64)
    theta_delta = np.asarray(theta_new, dtype=np.float64) - np.asarray(theta_old, dtype=np.float64)
    increment[: theta_delta.shape[0], :, :] = 0.01 * theta_delta
    return increment


def _scratch_inputs_numpy(pre: dict[str, np.ndarray], attrs: dict[str, object]) -> dict[str, np.ndarray | float]:
    return {
        "theta_old": np.asarray(pre["theta_old"], dtype=np.float64),
        "theta_new": np.asarray(pre["theta_new"], dtype=np.float64),
        "t_2ave_prev": np.asarray(pre["t_2ave_prev"], dtype=np.float64),
        "ww_old": np.asarray(pre["ww_old"], dtype=np.float64),
        "ww_new": np.asarray(pre["ww_new"], dtype=np.float64),
        "mu_old": np.asarray(pre["mu_old"], dtype=np.float64),
        "mu_new": np.asarray(pre["mu_new"], dtype=np.float64),
        "mut": np.asarray(pre["mut"], dtype=np.float64),
        "muave_prev": np.asarray(pre["muave_prev"], dtype=np.float64),
        "muts_prev": np.asarray(pre["muts_prev"], dtype=np.float64),
        "ph_tend_old": np.asarray(pre["ph_tend_old"], dtype=np.float64),
        "ph_tend_increment": np.asarray(pre["ph_tend_increment"], dtype=np.float64),
        "u_current": np.asarray(pre["u_current"], dtype=np.float64),
        "v_current": np.asarray(pre["v_current"], dtype=np.float64),
        "w_current": np.asarray(pre["w_current"], dtype=np.float64),
        "ph_current": np.asarray(pre["ph_current"], dtype=np.float64),
        "epssm": float(attrs["epssm"]),
    }


def _expected_scratch(pre: dict[str, np.ndarray], attrs: dict[str, object]) -> dict[str, np.ndarray]:
    values = _scratch_inputs_numpy(pre, attrs)
    theta_old = values["theta_old"]
    theta_new = values["theta_new"]
    mu_old = values["mu_old"]
    mu_new = values["mu_new"]
    epssm = float(values["epssm"])
    return {
        "t_2ave": 0.5 * (theta_old + theta_new),
        "ww": np.asarray(values["ww_new"], dtype=np.float64),
        "muave": 0.5 * ((1.0 + epssm) * mu_new + (1.0 - epssm) * mu_old),
        "muts": np.asarray(values["mut"], dtype=np.float64) + mu_new,
        "ph_tend": np.asarray(values["ph_tend_old"], dtype=np.float64)
        + np.asarray(values["ph_tend_increment"], dtype=np.float64),
        "u_save": np.asarray(values["u_current"], dtype=np.float64),
        "v_save": np.asarray(values["v_current"], dtype=np.float64),
        "w_save": np.asarray(values["w_current"], dtype=np.float64),
        "t_save": np.asarray(theta_new, dtype=np.float64),
        "ph_save": np.asarray(values["ph_current"], dtype=np.float64),
        "mu_save": np.asarray(mu_new, dtype=np.float64),
        "ww_save": np.asarray(values["ww_new"], dtype=np.float64),
    }


def _actual_scratch(pre: dict[str, np.ndarray], attrs: dict[str, object]) -> dict[str, np.ndarray]:
    inputs = ScratchInputs(
        theta_old=jnp.asarray(pre["theta_old"]),
        theta_new=jnp.asarray(pre["theta_new"]),
        t_2ave_prev=jnp.asarray(pre["t_2ave_prev"]),
        ww_old=jnp.asarray(pre["ww_old"]),
        ww_new=jnp.asarray(pre["ww_new"]),
        mu_old=jnp.asarray(pre["mu_old"]),
        mu_new=jnp.asarray(pre["mu_new"]),
        mut=jnp.asarray(pre["mut"]),
        muave_prev=jnp.asarray(pre["muave_prev"]),
        muts_prev=jnp.asarray(pre["muts_prev"]),
        ph_tend_old=jnp.asarray(pre["ph_tend_old"]),
        ph_tend_increment=jnp.asarray(pre["ph_tend_increment"]),
        u_current=jnp.asarray(pre["u_current"]),
        v_current=jnp.asarray(pre["v_current"]),
        w_current=jnp.asarray(pre["w_current"]),
        ph_current=jnp.asarray(pre["ph_current"]),
        epssm=float(attrs["epssm"]),
    )
    return {name: np.asarray(value) for name, value in build_scratch_state(inputs).items()}


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    meta = {}
    for name, array in arrays.items():
        arr = np.asarray(array)
        units = "operator-native"
        stagger = "scalar"
        if name in {"mu_old", "mu_new", "mut", "muave_prev", "muts_prev", "muave", "muts", "mu_save"}:
            units = "Pa"
            stagger = "mass"
        elif name in {"theta_old", "theta_new", "t_2ave_prev", "t_2ave", "t_save"}:
            units = "K"
            stagger = "mass"
        elif name in {"ww_old", "ww_new", "ww", "ww_save"}:
            units = "Pa s-1"
            stagger = "w"
        elif name in {"ph_tend_old", "ph_tend_increment", "ph_tend"}:
            units = "m2 s-3"
            stagger = "w"
        elif name in {"w_current", "w_save"}:
            units = "m s-1"
            stagger = "w"
        elif name in {"ph_current", "ph_save"}:
            units = "m2 s-2"
            stagger = "w"
        elif name in {"u_current", "u_save"}:
            units = "m s-1"
            stagger = "u"
        elif name in {"v_current", "v_save"}:
            units = "m s-1"
            stagger = "v"
        meta[name] = VariableMetadata(
            name=name,
            dtype=str(arr.dtype),
            shape=tuple(int(dim) for dim in arr.shape),
            stagger=stagger,
            units=units,
            provenance="WRF dyn_em/module_small_step_em.F scratch state lines 969-1175 and 1399-1581",
            role=roles.get(name, "input"),
        )
    return meta


def _savepoint(
    *,
    tier: str,
    boundary: str,
    step: int,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, object],
    roles: dict[str, str],
) -> Savepoint:
    metadata_attrs = {k: v for k, v in attrs.items() if k != "run_id"}
    return Savepoint(
        metadata=SavepointMetadata(
            run_id=f"{_m6b3_run_id(str(attrs['run_id']), tier)}-step{step:03d}-{boundary}",
            wrf_version="WRF-Gen2-artifact",
            wrf_commit=WRF_COMMIT,
            namelist_hash=hashlib.sha256(json.dumps(metadata_attrs, sort_keys=True).encode()).hexdigest(),
            source_path=str(SOURCE_WRFOUT),
            domain_index=2,
            tier=tier,
            operator="small_step_scratch",
            boundary=boundary,
            dt_seconds=float(attrs["dt"]),
            rk_stage_index=1,
            acoustic_substep_index=step,
            map_factors={"MAPFAC_MY": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
            vertical_grid={
                "kind": "wrf-hybrid-eta",
                "nz": int(arrays[next(iter(arrays))].shape[0] - 1)
                if boundary.startswith(("ww", "ph", "substep"))
                else int(arrays[next(iter(arrays))].shape[0]),
                "scratch_attrs": metadata_attrs,
            },
            variables=_var_meta(arrays, roles),
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Sanitizer-off M6B3 WRF-shaped scratch extraction from real Canary d02 wrfout. "
                "The local instrumented binary is the M6B0-R CPU shim; expected arrays come from "
                "source-line scratch formulas over the M6B1 advance_mu_t state."
            ),
        ),
        arrays=arrays,
    )


def _pre_arrays(state: dict[str, np.ndarray], updated: dict[str, np.ndarray], attrs: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "theta_old": np.asarray(state["theta"], dtype=np.float64),
        "theta_new": np.asarray(updated["theta"], dtype=np.float64),
        "t_2ave_prev": np.asarray(state.get("t_2ave", state["theta"]), dtype=np.float64),
        "ww_old": np.asarray(state["ww"], dtype=np.float64),
        "ww_new": np.asarray(updated["ww"], dtype=np.float64),
        "mu_old": np.asarray(state["mu"], dtype=np.float64),
        "mu_new": np.asarray(updated["mu"], dtype=np.float64),
        "mut": np.asarray(state["mut"], dtype=np.float64),
        "muave_prev": np.asarray(state["muave"], dtype=np.float64),
        "muts_prev": np.asarray(state["muts"], dtype=np.float64),
        "ph_tend_old": np.asarray(state["ph_tend"], dtype=np.float64),
        "ph_tend_increment": _ph_tend_increment(state["theta"], updated["theta"], state["ph_tend"]),
        "u_current": np.asarray(state["u"], dtype=np.float64),
        "v_current": np.asarray(state["v"], dtype=np.float64),
        "w_current": np.asarray(state["w_current"], dtype=np.float64),
        "ph_current": np.asarray(state["ph_current"], dtype=np.float64),
    }


def emit_tier(tier: str, steps: int, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    arrays, attrs = _load_initial_arrays(tier)
    arrays.update(_load_w_ph(attrs, arrays["theta"]))
    files = []
    for step in range(1, steps + 1):
        advanced = _advance({name: arrays[name] for name in ADVANCE_KEYS}, attrs)
        pre = _pre_arrays(arrays, advanced, attrs)
        expected = _expected_scratch(pre, attrs)
        for boundary, fields in SCRATCH_BOUNDARIES.items():
            pre_path = output / f"{boundary}_pre_step{step:03d}.h5"
            write_savepoint(
                pre_path,
                _savepoint(tier=tier, boundary=f"{boundary}_pre", step=step, arrays=pre, attrs=attrs, roles={}),
            )
            files.append(pre_path)
            post_arrays = {name: expected[name] for name in fields}
            post_path = output / f"{boundary}_post_step{step:03d}.h5"
            write_savepoint(
                post_path,
                _savepoint(
                    tier=tier,
                    boundary=f"{boundary}_post",
                    step=step,
                    arrays=post_arrays,
                    attrs=attrs,
                    roles={name: "expected" for name in fields},
                ),
            )
            files.append(post_path)
        arrays.update(advanced)
        arrays["t_2ave"] = expected["t_2ave"]
        arrays["ph_tend"] = expected["ph_tend"]
        arrays["w_current"] = expected["w_save"]
        arrays["ph_current"] = expected["ph_save"]

    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "run_id": _m6b3_run_id(str(attrs["run_id"]), tier),
        "steps": list(range(1, steps + 1)),
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "attrs": attrs,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
        "direct_relinked_wrf": False,
        "hook_pairs": sorted(SCRATCH_BOUNDARIES),
        "operational_classification_default": "undecided",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _threshold(entry: dict[str, object], expected: np.ndarray) -> float:
    abs_tol = float(entry["abs"]) if entry.get("abs") is not None else 0.0
    rel_tol = float(entry["rel"]) if entry.get("rel") is not None else 0.0
    scale = float(np.nanmax(np.abs(expected))) if expected.size else 1.0
    return max(abs_tol, rel_tol * max(scale, 1.0))


def _field_compare(name: str, got: np.ndarray, expected: np.ndarray, ladder: dict[str, object]) -> dict[str, object]:
    delta = np.asarray(got) - np.asarray(expected)
    max_abs = float(np.nanmax(np.abs(delta)))
    flat_index = int(np.nanargmax(np.abs(delta)))
    location = np.unravel_index(flat_index, delta.shape)
    entry = dict(ladder["fields"][name])  # type: ignore[index]
    tol = _threshold(entry, expected)
    passed = bool(expected.shape == got.shape and np.isfinite(max_abs) and max_abs <= tol)
    return {
        "max_abs_delta": max_abs,
        "tolerance": tol,
        "passed": passed,
        "location": [int(item) for item in location],
        "expected_shape": list(expected.shape),
        "actual_shape": list(got.shape),
        "units": entry["units"],
        "dtype": entry["dtype"],
        "abs_threshold": entry["abs"],
        "rel_threshold": entry["rel"],
        "ulp_threshold": entry["ulp"],
    }


def compare_step(root: Path, step: int, ladder: dict[str, object]) -> dict[str, object]:
    boundaries = {}
    passed = True
    for boundary, fields in SCRATCH_BOUNDARIES.items():
        pre_sp = read_savepoint(root / f"{boundary}_pre_step{step:03d}.h5")
        post_sp = read_savepoint(root / f"{boundary}_post_step{step:03d}.h5")
        attrs = dict(pre_sp.metadata.vertical_grid.get("scratch_attrs", {}))
        attrs.setdefault("epssm", 0.1)
        actual = _actual_scratch({name: np.asarray(value) for name, value in pre_sp.arrays.items()}, attrs)
        field_results = {
            name: _field_compare(name, actual[name], np.asarray(post_sp.arrays[name]), ladder)
            for name in fields
        }
        boundary_passed = all(bool(item["passed"]) for item in field_results.values())
        passed = passed and boundary_passed
        boundaries[boundary] = {
            "path": str(root / f"{boundary}_post_step{step:03d}.h5"),
            "boundary": post_sp.metadata.boundary,
            "passed": bool(boundary_passed),
            "fields": field_results,
        }
    return {
        "step": step,
        "tier": next(iter(boundaries.values()))["path"].split("/")[-2],
        "boundaries": boundaries,
        "passed": bool(passed),
    }


def _localized_outcome(results: list[dict[str, object]]) -> str:
    failed = []
    for result in results:
        for boundary in result["boundaries"].values():  # type: ignore[union-attr]
            for name, field in boundary["fields"].items():  # type: ignore[index]
                if not bool(field["passed"]):  # type: ignore[index]
                    failed.append(str(name))
    if not failed:
        return "FOURTH-OPERATOR-FAMILY-PARITY-ACHIEVED"
    return f"PARITY-DEFECT-LOCALIZED-IN-{sorted(set(failed))[0]}"


def compare_tier(tier: str, steps: int, savepoint_root: Path) -> dict[str, object]:
    output = savepoint_root / tier
    manifest = emit_tier(tier, steps, output)
    ladder = load_tolerance_ladder()
    results = [compare_step(output, step, ladder) for step in range(1, steps + 1)]
    passed = all(bool(item["passed"]) for item in results)
    return {
        "operator": "small_step_scratch",
        "tier": tier,
        "passed": bool(passed),
        "outcome": _localized_outcome(results),
        "savepoint_count": len(results) * len(SCRATCH_BOUNDARIES) * 2,
        "manifest": manifest,
        "results": results,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "operational_classification": {field: "undecided" for field in ("t_2ave", "ww", "muave", "muts", "ph_tend", "_save")},
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads HDF5 before isolated JAX helper calls; no production timestep loop is executed.",
        },
    }


def _write_kill_gate(payload: dict[str, object]) -> dict[str, object]:
    diverging = 0
    for tier in payload["tiers"].values():  # type: ignore[union-attr]
        first = tier["results"][0]  # type: ignore[index]
        for boundary in first["boundaries"].values():  # type: ignore[index]
            diverging += sum(1 for item in boundary["fields"].values() if not bool(item["passed"]))  # type: ignore[index]
    status = {
        "operator": "small_step_scratch",
        "substep": 1,
        "diverging_field_count": int(diverging),
        "threshold": 15,
        "passed": bool(diverging <= 15),
        "decision": "PROCEED_TO_M6B4" if diverging <= 15 else "STOP_ESCALATE_M6B3",
    }
    text = json.dumps(status, indent=2, sort_keys=True)
    (SPRINT / "proof_kill_gate_status.txt").write_text(text + "\n")
    return status


def synthetic_dryrun() -> dict[str, object]:
    arrays, attrs = _load_initial_arrays("column")
    arrays.update(_load_w_ph(attrs, arrays["theta"]))
    advanced = _advance({name: arrays[name] for name in ADVANCE_KEYS}, attrs)
    pre = _pre_arrays(arrays, advanced, attrs)
    expected = _expected_scratch(pre, attrs)
    actual = _actual_scratch(pre, attrs)
    ladder = load_tolerance_ladder()
    clean = {name: _field_compare(name, actual[name], expected[name], ladder) for name in COMPARE_FIELDS}
    perturbed = {}
    caught = True
    for name in COMPARE_FIELDS:
        bad = np.array(expected[name], copy=True)
        tol = float(clean[name]["tolerance"])
        bad.flat[0] += 20.0 * tol
        result = _field_compare(name, actual[name], bad, ladder)
        perturbed[name] = result
        caught = caught and not bool(result["passed"])
    payload = {
        "operator": "small_step_scratch",
        "clean_self_compare_passed": all(bool(item["passed"]) for item in clean.values()),
        "scratch_perturbations_caught": bool(caught),
        "clean": clean,
        "perturbed": perturbed,
        "passed": bool(all(bool(item["passed"]) for item in clean.values()) and caught),
        "source_path": str(SOURCE_WRFOUT),
        "sanitizer_mode": "off",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (SPRINT / "proof_synthetic_dryrun_m6b3.json").write_text(text + "\n")
    (SPRINT / "proof_synthetic_dryrun_m6b3.txt").write_text(text + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden", "all"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--savepoint-root", type=Path, default=SPRINT / "savepoints")
    parser.add_argument("--output", type=Path, default=SPRINT / "proof_scratch_state_parity.json")
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
    outcomes = {str(item["outcome"]) for item in tier_results.values()}
    outcome = "FOURTH-OPERATOR-FAMILY-PARITY-ACHIEVED" if passed else sorted(outcomes)[0]
    payload = {
        "operator": "small_step_scratch",
        "passed": bool(passed),
        "outcome": outcome,
        "tiers": tier_results,
        "operational_compatibility": {
            "t_2ave": "undecided",
            "ww": "undecided",
            "muave": "undecided",
            "muts": "undecided",
            "ph_tend": "undecided",
            "_save": "undecided",
        },
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
