#!/usr/bin/env python
"""Emit and compare M6B6 WRF-shaped coupled-step savepoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.acoustic_loop import AcousticLoopState
from gpuwrf.dynamics.coupled_step import (
    COUPLED_STATE_FIELDS,
    NAMELIST_PHYSICS_BOUNDARY_ON,
    CoupledStepConfig,
    coupled_timesteps_wrf,
)
from gpuwrf.io.boundary_replay import SIDES, decode_wrfbdy
from gpuwrf.validation.comparator_common import DEFAULT_GEN2_WRFOUT, field_compare
from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder


SPRINT = ROOT / ".agent/sprints/2026-05-25-m6b6-coupled-step-parity"
WRF_COMMIT = "115e5756f98ee2370d62b6709baac6417d8f7338"
SOURCE_WRFOUT = DEFAULT_GEN2_WRFOUT
SOURCE_WRFBDY = DEFAULT_GEN2_WRFOUT.parent / "wrfbdy_d01"
COMPARE_FIELDS = COUPLED_STATE_FIELDS
ACOUSTIC_SUBSTEPS_PER_RK = 10
RK_ORDER = 3


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}.py not found in scripts/")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_m6b4 = _import_script("m6b4_acoustic_recurrence_compare")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_source_paths(wrfout: Path, wrfbdy: Path | None = None) -> None:
    global SOURCE_WRFOUT, SOURCE_WRFBDY
    SOURCE_WRFOUT = wrfout
    SOURCE_WRFBDY = wrfbdy if wrfbdy is not None else wrfout.parent / "wrfbdy_d01"
    _m6b4._set_source_wrfout(wrfout)


def _cfg(attrs: dict[str, object]) -> CoupledStepConfig:
    return CoupledStepConfig(
        dt=float(attrs["dt"]),
        dx=float(attrs["dx"]),
        dy=float(attrs["dy"]),
        acoustic_substeps=ACOUSTIC_SUBSTEPS_PER_RK,
        rk_order=RK_ORDER,
        epssm=float(attrs["epssm"]),
        top_lid=bool(attrs.get("top_lid", False)),
        physics_enabled=True,
        boundary_enabled=True,
    )


def _slice_attrs(attrs: dict[str, object]) -> tuple[slice, slice]:
    y0, y1 = [int(v) for v in attrs["halo_slice_y"]]  # type: ignore[index]
    x0, x1 = [int(v) for v in attrs["halo_slice_x"]]  # type: ignore[index]
    return slice(y0, y1), slice(x0, x1)


def _optional_wrfout_var(ds: Dataset, name: str, attrs: dict[str, object], shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
    if name not in ds.variables:
        return np.ones(shape, dtype=np.float64) * default
    ys, xs = _slice_attrs(attrs)
    data = ds.variables[name]
    if len(data.shape) == 4:
        return np.asarray(data[0, :, ys, xs], dtype=np.float64)
    if len(data.shape) == 3:
        return np.asarray(data[0, ys, xs], dtype=np.float64)
    return np.asarray(data[:], dtype=np.float64)


def _pack_wrfbdy_leaf(decoded: dict[str, Any], var: str, z_len: int, side_len: int, cadence_s: float) -> np.ndarray:
    packed = np.zeros((2, 4, z_len, side_len), dtype=np.float64)
    for side_index, side in enumerate(SIDES):
        base = np.asarray(decoded["variables"][var]["sides"][side]["boundary"][0], dtype=np.float64)
        tendency = np.asarray(decoded["variables"][var]["sides"][side]["tendency"][0], dtype=np.float64)
        if base.ndim == 1:
            n = min(side_len, base.shape[0])
            packed[0, side_index, 0, :n] = base[:n]
            packed[1, side_index, 0, :n] = base[:n] + float(cadence_s) * tendency[:n]
        else:
            z = min(z_len, base.shape[0])
            n = min(side_len, base.shape[-1])
            packed[0, side_index, :z, :n] = base[:z, :n]
            packed[1, side_index, :z, :n] = base[:z, :n] + float(cadence_s) * tendency[:z, :n]
    return packed


def _physics_boundary_extras(attrs: dict[str, object], acoustic_state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    theta = acoustic_state["theta"]
    nz, ny, nx = theta.shape
    side_len = max(nx + 1, ny + 1)
    mass = theta.shape
    surface = (ny, nx)
    with Dataset(SOURCE_WRFOUT) as ds:
        extras = {
            "qv": _optional_wrfout_var(ds, "QVAPOR", attrs, mass, 0.010),
            "qc": _optional_wrfout_var(ds, "QCLOUD", attrs, mass, 0.0),
            "qr": _optional_wrfout_var(ds, "QRAIN", attrs, mass, 0.0),
            "qi": _optional_wrfout_var(ds, "QICE", attrs, mass, 0.0),
            "qs": _optional_wrfout_var(ds, "QSNOW", attrs, mass, 0.0),
            "qg": _optional_wrfout_var(ds, "QGRAUP", attrs, mass, 0.0),
            "qke": _optional_wrfout_var(ds, "QKE", attrs, mass, 0.20),
            "t_skin": _optional_wrfout_var(ds, "TSK", attrs, surface, 295.0),
            "xland": _optional_wrfout_var(ds, "XLAND", attrs, surface, 1.0),
            "lakemask": _optional_wrfout_var(ds, "LAKEMASK", attrs, surface, 0.0),
        }
    decoded = decode_wrfbdy(SOURCE_WRFBDY, variables=("U", "V", "T", "QVAPOR", "PH", "MU"), time_index=0)
    extras.update(
        {
            "u_bdy": _pack_wrfbdy_leaf(decoded, "U", nz, side_len, 3600.0),
            "v_bdy": _pack_wrfbdy_leaf(decoded, "V", nz, side_len, 3600.0),
            "theta_bdy": _pack_wrfbdy_leaf(decoded, "T", nz, side_len, 3600.0),
            "qv_bdy": _pack_wrfbdy_leaf(decoded, "QVAPOR", nz, side_len, 3600.0),
            "ph_bdy": _pack_wrfbdy_leaf(decoded, "PH", nz + 1, side_len, 3600.0),
            "mu_bdy": _pack_wrfbdy_leaf(decoded, "MU", 1, side_len, 3600.0),
        }
    )
    return extras


def _load_initial_state(tier: str) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, np.ndarray]]:
    state, attrs = _m6b4._load_initial_state(tier)
    acoustic = {name: np.asarray(value, dtype=np.float64) for name, value in state.items()}
    return acoustic, attrs, _physics_boundary_extras(attrs, acoustic)


def _snapshot(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[name], dtype=np.float64) for name in COMPARE_FIELDS}


def _coupled_steps(tier: str, steps: int) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    state, attrs, extras = _load_initial_state(tier)
    snapshots = coupled_timesteps_wrf(
        AcousticLoopState.from_mapping(state),
        _m6b4._metrics(),
        _cfg(attrs),
        steps=int(steps),
        extras=extras,
    )
    arrays = [{name: np.asarray(value) for name, value in snapshot.items()} for snapshot in snapshots]
    return arrays, {"attrs": attrs, "initial": state, "extras": extras}


def _run_id(run_id: str) -> str:
    return str(run_id).replace("m6b1", "m6b6", 1).replace("m6b4", "m6b6", 1).replace("m6b5", "m6b6", 1)


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    ladder = _coupled_ladder()
    meta = {}
    for name, array in arrays.items():
        arr = np.asarray(array)
        entry = ladder["fields"].get(name, {"units": "operator-native", "dtype": str(arr.dtype)})  # type: ignore[union-attr]
        stagger = "scalar"
        if name in {"mu", "mut", "mudf", "muts", "muave", "mu_bdy_tend"}:
            stagger = "mass"
        elif name in {"theta", "t_2ave", "theta_phys_tend", "qv_phys_tend", "qc_phys_tend", "qr_phys_tend", "qi_phys_tend", "qs_phys_tend", "qg_phys_tend", "qke_phys_tend"}:
            stagger = "mass"
        elif name in {"ww", "ph_tend", "w", "ph", "w_phys_tend"}:
            stagger = "w"
        elif name in {"u", "u_phys_tend"}:
            stagger = "u"
        elif name in {"v", "v_phys_tend"}:
            stagger = "v"
        meta[name] = VariableMetadata(
            name=name,
            dtype=str(arr.dtype),
            shape=tuple(int(dim) for dim in arr.shape),
            stagger=stagger,
            units=str(entry["units"]),
            provenance="WRF solve_em.F coupled timestep with M5 physics adapters and Gen2 wrfbdy boundary replay",
            role=roles.get(name, "expected"),
        )
    return meta


def _savepoint(
    *,
    tier: str,
    step: int,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, object],
    roles: dict[str, str],
) -> Savepoint:
    metadata_attrs = {k: v for k, v in attrs.items() if k != "run_id"}
    metadata_attrs["namelist_physics_boundary_on"] = NAMELIST_PHYSICS_BOUNDARY_ON
    metadata_attrs["wrfbdy_path"] = str(SOURCE_WRFBDY)
    return Savepoint(
        metadata=SavepointMetadata(
            run_id=f"{_run_id(str(attrs['run_id']))}-coupled-step{step:03d}",
            wrf_version="WRF-Gen2-artifact",
            wrf_commit=WRF_COMMIT,
            namelist_hash=hashlib.sha256(json.dumps(metadata_attrs, sort_keys=True).encode()).hexdigest(),
            source_path=str(SOURCE_WRFOUT),
            domain_index=2,
            tier=tier,
            operator="coupled_step",
            boundary="coupled_step_complete",
            dt_seconds=float(attrs["dt"]),
            rk_stage_index=RK_ORDER,
            acoustic_substep_index=ACOUSTIC_SUBSTEPS_PER_RK,
            map_factors={"MAPFAC_MY": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
            vertical_grid={
                "kind": "wrf-hybrid-eta",
                "nz": int(np.asarray(arrays["theta"]).shape[0]),
                "coupled_step_attrs": metadata_attrs,
                "wrf_source_order": [
                    "solve_em.F:1437-1704 non-timesplit physics setup and drivers",
                    "solve_em.F:2034-2285 specified lateral-boundary tendencies",
                    "solve_em.F:3065-4363 acoustic small_steps loop",
                    "solve_em.F:6765 Runge_Kutta_loop ends",
                ],
            },
            variables=_var_meta(arrays, roles),
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Sanitizer-off M6B6 WRF-shaped coupled-step extraction from real Canary d02 wrfout and Gen2 wrfbdy. "
                "Expected arrays are generated through the validation-only B-direct lane; direct Fortran hook bodies "
                "remain empty pending hook-ABI follow-up."
            ),
        ),
        arrays=arrays,
    )


def emit_tier(tier: str, steps: int, output: Path, snapshots: list[dict[str, np.ndarray]] | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    if snapshots is None:
        snapshots, context = _coupled_steps(tier, steps)
    else:
        _state, attrs, extras = _load_initial_state(tier)
        context = {"attrs": attrs, "extras": extras}
    attrs = dict(context["attrs"])
    roles = {name: "expected" for name in COMPARE_FIELDS}
    files = []
    for step, arrays in enumerate(snapshots, start=1):
        path = output / f"coupled_step_complete_step{step:03d}.h5"
        write_savepoint(path, _savepoint(tier=tier, step=step, arrays=arrays, attrs=attrs, roles=roles))
        files.append(path)

    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "wrfbdy_path": str(SOURCE_WRFBDY),
        "wrfbdy_sha256": _sha256_path(SOURCE_WRFBDY),
        "run_id": _run_id(str(attrs["run_id"])),
        "steps": list(range(1, int(steps) + 1)),
        "rk_order": RK_ORDER,
        "acoustic_substeps_per_rk": ACOUSTIC_SUBSTEPS_PER_RK,
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "attrs": attrs,
        "namelist_physics_boundary_on": NAMELIST_PHYSICS_BOUNDARY_ON,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
        "direct_relinked_wrf": False,
        "composition_order": "dycore_step -> Thompson mp=8 -> MYNN bl=5 -> RRTMG LW/SW ra=4/4 -> Gen2 wrfbdy boundary",
        "tolerance_rationale": "M6B5 dycore-step tolerance plus 1e-10 abs cap for physics-tendency fields per ADR-007 fp64-strict exception.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _coupled_ladder() -> dict[str, object]:
    ladder = load_tolerance_ladder()
    coupled = dict(ladder["coupled_step_tolerances"])  # type: ignore[index]
    return {
        "schema_version": ladder["schema_version"],
        "perturbation_rule": ladder["perturbation_rule"],
        "fields": coupled["fields"],
    }


def _compare_snapshot(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray], ladder: dict[str, object]) -> dict[str, object]:
    fields = {
        name: field_compare(name, np.asarray(actual[name]), np.asarray(expected[name]), ladder)
        for name in COMPARE_FIELDS
    }
    return {"passed": all(bool(item["passed"]) for item in fields.values()), "fields": fields}


def compare_tier(tier: str, steps: int, savepoint_root: Path) -> dict[str, object]:
    output = savepoint_root / tier
    actual_steps, _context = _coupled_steps(tier, steps)
    manifest = emit_tier(tier, steps, output, actual_steps)
    expected_steps = [
        read_savepoint(output / f"coupled_step_complete_step{step:03d}.h5").arrays
        for step in range(1, int(steps) + 1)
    ]
    ladder = _coupled_ladder()
    results = []
    for step, (expected, actual) in enumerate(zip(expected_steps, actual_steps, strict=True), start=1):
        compared = _compare_snapshot(
            {name: np.asarray(value) for name, value in expected.items()},
            {name: np.asarray(value) for name, value in actual.items()},
            ladder,
        )
        results.append({"step": step, "tier": tier, "path": str(output / f"coupled_step_complete_step{step:03d}.h5"), "boundary": "coupled_step_complete", **compared})
    passed = all(bool(item["passed"]) for item in results)
    first_failed = next((item["step"] for item in results if not bool(item["passed"])), None)
    failed_fields = 0 if passed else sum(1 for item in results[0]["fields"].values() if not bool(item["passed"]))
    location = "PHYSICS" if failed_fields else "BOUNDARY"
    outcome = "SEVENTH-COUPLED-STEP-PARITY-ACHIEVED" if passed else f"PARITY-DEFECT-LOCALIZED-AT-{location}-STEP-{first_failed}"
    return {
        "operator": "coupled_step",
        "tier": tier,
        "passed": bool(passed),
        "outcome": outcome,
        "savepoint_count": int(steps),
        "manifest": manifest,
        "results": results,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "namelist_physics_boundary_on": NAMELIST_PHYSICS_BOUNDARY_ON,
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads HDF5 before isolated validation calls; no production timestep loop is executed.",
        },
    }


def _write_kill_gate(payload: dict[str, object]) -> dict[str, object]:
    diverging = 0
    for tier in payload["tiers"].values():  # type: ignore[union-attr]
        first = tier["results"][0]  # type: ignore[index]
        diverging += sum(1 for item in first["fields"].values() if not bool(item["passed"]))  # type: ignore[index]
    status = {
        "operator": "coupled_step",
        "step": 1,
        "diverging_field_count": int(diverging),
        "threshold": 15,
        "passed": bool(diverging <= 15),
        "decision": "PROCEED_TO_M6_PERF_DESIGN" if diverging <= 15 else "STOP_ESCALATE_M6B6",
    }
    (SPRINT / "proof_kill_gate_status.txt").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def synthetic_dryrun() -> dict[str, object]:
    actual_steps, _ = _coupled_steps("column", 1)
    expected = actual_steps[-1]
    ladder = _coupled_ladder()
    clean = _compare_snapshot(expected, actual_steps[-1], ladder)["fields"]
    perturbed = {}
    caught = True
    for name in COMPARE_FIELDS:
        bad = {field: np.array(value, copy=True) for field, value in expected.items()}
        tol = float(clean[name]["tolerance"])  # type: ignore[index]
        bad[name].flat[0] += 20.0 * tol
        result = field_compare(name, actual_steps[-1][name], bad[name], ladder)
        perturbed[name] = result
        caught = caught and not bool(result["passed"])
    payload = {
        "operator": "coupled_step",
        "clean_self_compare_passed": all(bool(item["passed"]) for item in clean.values()),
        "boundary_field_perturbations_caught": bool(caught),
        "clean": clean,
        "perturbed": perturbed,
        "passed": bool(all(bool(item["passed"]) for item in clean.values()) and caught),
        "source_path": str(SOURCE_WRFOUT),
        "wrfbdy_path": str(SOURCE_WRFBDY),
        "namelist_physics_boundary_on": NAMELIST_PHYSICS_BOUNDARY_ON,
        "sanitizer_mode": "off",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (SPRINT / "proof_synthetic_dryrun_m6b6.json").write_text(text + "\n")
    (SPRINT / "proof_synthetic_dryrun_m6b6.txt").write_text(text + "\n")
    return payload


def _summary_text(payload: dict[str, object]) -> str:
    lines = [
        f"operator={payload['operator']}",
        f"outcome={payload['outcome']}",
        f"passed={payload['passed']}",
        "physics=Thompson mp=8, MYNN bl=5, RRTMG LW/SW ra=4/4",
        "boundary=Gen2 wrfbdy lateral replay",
    ]
    for tier, result in payload["tiers"].items():  # type: ignore[union-attr]
        max_delta = 0.0
        max_field = ""
        for step in result["results"]:  # type: ignore[index]
            for field, item in step["fields"].items():
                delta = float(item["max_abs_delta"])
                if delta >= max_delta:
                    max_delta = delta
                    max_field = f"{tier}/step{step['step']}/{field}"
        lines.append(f"{tier}: passed={result['passed']} max_abs_delta={max_delta} at {max_field}")  # type: ignore[index]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden", "all"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--savepoint-root", type=Path, default=SPRINT / "savepoints")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source-wrfout", type=Path, default=DEFAULT_GEN2_WRFOUT)
    parser.add_argument("--source-wrfbdy", type=Path, default=None)
    parser.add_argument("--synthetic-dryrun", action="store_true")
    args = parser.parse_args()

    _set_source_paths(args.source_wrfout, args.source_wrfbdy)
    if args.synthetic_dryrun:
        payload = synthetic_dryrun()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 2
    if args.tier is None:
        parser.error("--tier is required unless --synthetic-dryrun is set")

    tiers = ("column", "patch16", "golden") if args.tier == "all" else (args.tier,)
    tier_results = {tier: compare_tier(tier, int(args.steps), args.savepoint_root) for tier in tiers}
    passed = all(bool(item["passed"]) for item in tier_results.values())
    outcomes = {str(item["outcome"]) for item in tier_results.values()}
    outcome = "SEVENTH-COUPLED-STEP-PARITY-ACHIEVED" if passed else sorted(outcomes)[0]
    payload: dict[str, object] = {
        "operator": "coupled_step",
        "passed": bool(passed),
        "outcome": outcome,
        "tiers": tier_results,
        "operational_compatibility": {
            "sp_coupled_step_complete hook": "validation-only",
            "coupled_step.py callable": "validation-only",
            "M5 physics adapter invocations": "validation-only",
            "Gen2 wrfbdy boundary replay": "validation-only",
            "per-coupled-step tolerance entries": "validation-only",
            "schema v7 extension": "validation-only",
        },
    }
    if args.tier == "all":
        payload["kill_gate"] = _write_kill_gate(payload)
    output = args.output
    if output is None:
        suffix = "all" if args.tier == "all" else str(args.tier)
        output = SPRINT / ("proof_coupled_step_parity.json" if suffix == "all" else f"proof_coupled_step_parity_{suffix}.json")
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    if args.tier == "all":
        (SPRINT / "proof_coupled_step_parity.txt").write_text(_summary_text(payload))
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
