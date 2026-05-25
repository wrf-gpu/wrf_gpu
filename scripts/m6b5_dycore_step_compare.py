#!/usr/bin/env python
"""Emit and compare M6B5 WRF-shaped full dycore timestep savepoints."""

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

from gpuwrf.dynamics.acoustic_loop import AcousticLoopState
from gpuwrf.dynamics.dycore_step import DycoreStepConfig, FULL_STATE_FIELDS, dycore_timesteps_wrf
from gpuwrf.validation.comparator_common import DEFAULT_GEN2_WRFOUT, field_compare
from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder


SPRINT = ROOT / ".agent/sprints/2026-05-25-m6b5-full-dycore-step-parity"
WRF_COMMIT = "115e5756f98ee2370d62b6709baac6417d8f7338"
COMPARE_FIELDS = FULL_STATE_FIELDS
SOURCE_WRFOUT = DEFAULT_GEN2_WRFOUT
ACOUSTIC_SUBSTEPS_PER_RK = 10
RK_ORDER = 3
NAMELIST_DISABLED = {
    "mp_physics": 0,
    "bl_pbl_physics": 0,
    "ra_lw_physics": 0,
    "ra_sw_physics": 0,
    "cu_physics": 0,
    "sf_sfclay_physics": 0,
    "sf_surface_physics": 0,
    "specified": False,
}


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


def _set_source_wrfout(path: Path) -> None:
    global SOURCE_WRFOUT
    SOURCE_WRFOUT = path
    _m6b4._set_source_wrfout(path)


def _cfg(attrs: dict[str, object]) -> DycoreStepConfig:
    return DycoreStepConfig(
        dt=float(attrs["dt"]),
        dx=float(attrs["dx"]),
        dy=float(attrs["dy"]),
        acoustic_substeps=ACOUSTIC_SUBSTEPS_PER_RK,
        rk_order=RK_ORDER,
        epssm=float(attrs["epssm"]),
        top_lid=bool(attrs.get("top_lid", False)),
        physics_enabled=False,
        boundary_enabled=False,
    )


def _snapshot(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[name], dtype=np.float64) for name in COMPARE_FIELDS}


def _expected_steps(tier: str, steps: int) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    state, attrs = _m6b4._load_initial_state(tier)
    current = {name: np.asarray(value, dtype=np.float64) for name, value in state.items()}
    snapshots = []
    for _step in range(1, int(steps) + 1):
        for _rk_stage in range(1, RK_ORDER + 1):
            coeffs = _m6b4._coefficients_numpy(current, attrs)
            for _substep in range(ACOUSTIC_SUBSTEPS_PER_RK):
                current = _m6b4._expected_substep(current, attrs, coeffs)
        snapshots.append(_snapshot(current))
    return snapshots, {"attrs": attrs, "initial": state}


def _actual_steps(tier: str, steps: int) -> list[dict[str, np.ndarray]]:
    state, attrs = _m6b4._load_initial_state(tier)
    step_snapshots, _rk_snapshots = dycore_timesteps_wrf(
        AcousticLoopState.from_mapping(state),
        _m6b4._metrics(),
        _cfg(attrs),
        steps=int(steps),
    )
    return [{name: np.asarray(value) for name, value in snapshot.items()} for snapshot in step_snapshots]


def _run_id(run_id: str) -> str:
    return str(run_id).replace("m6b1", "m6b5", 1).replace("m6b4", "m6b5", 1)


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    base = _m6b4._var_meta(arrays, roles)
    return {
        name: VariableMetadata(
            name=item.name,
            dtype=item.dtype,
            shape=item.shape,
            stagger=item.stagger,
            units=item.units,
            provenance="WRF solve_em.F RK3 Runge_Kutta_loop wrapping acoustic small_steps; physics/boundary disabled",
            role=item.role,
        )
        for name, item in base.items()
    }


def _savepoint(
    *,
    tier: str,
    step: int,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, object],
    roles: dict[str, str],
) -> Savepoint:
    metadata_attrs = {k: v for k, v in attrs.items() if k != "run_id"}
    metadata_attrs["namelist_disabled"] = NAMELIST_DISABLED
    return Savepoint(
        metadata=SavepointMetadata(
            run_id=f"{_run_id(str(attrs['run_id']))}-dycore-step{step:03d}",
            wrf_version="WRF-Gen2-artifact",
            wrf_commit=WRF_COMMIT,
            namelist_hash=hashlib.sha256(json.dumps(metadata_attrs, sort_keys=True).encode()).hexdigest(),
            source_path=str(SOURCE_WRFOUT),
            domain_index=2,
            tier=tier,
            operator="dycore_step",
            boundary="dycore_step_complete",
            dt_seconds=float(attrs["dt"]),
            rk_stage_index=RK_ORDER,
            acoustic_substep_index=ACOUSTIC_SUBSTEPS_PER_RK,
            map_factors={"MAPFAC_MY": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
            vertical_grid={
                "kind": "wrf-hybrid-eta",
                "nz": int(np.asarray(arrays["theta"]).shape[0]),
                "dycore_step_attrs": metadata_attrs,
                "wrf_source_order": [
                    "solve_em.F:1447 Runge_Kutta_loop begins",
                    "solve_em.F:2409-2738 calc_coef_w per RK stage",
                    "solve_em.F:3065-4363 acoustic small_steps loop",
                    "solve_em.F:6765 Runge_Kutta_loop ends",
                    "solve_em.F:8174 after_all_rk_steps follows dycore step",
                ],
            },
            variables=_var_meta(arrays, roles),
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Sanitizer-off M6B5 WRF-shaped dycore-step extraction from real Canary d02 wrfout. "
                "Physics and lateral boundary application are disabled by metadata contract; direct "
                "Fortran hook bodies remain empty pending hook-ABI follow-up."
            ),
        ),
        arrays=arrays,
    )


def emit_tier(tier: str, steps: int, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    step_snapshots, context = _expected_steps(tier, steps)
    attrs = dict(context["attrs"])
    files = []
    roles = {name: "expected" for name in COMPARE_FIELDS}
    for step, arrays in enumerate(step_snapshots, start=1):
        path = output / f"dycore_step_complete_step{step:03d}.h5"
        write_savepoint(path, _savepoint(tier=tier, step=step, arrays=arrays, attrs=attrs, roles=roles))
        files.append(path)

    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "run_id": _run_id(str(attrs["run_id"])),
        "steps": list(range(1, int(steps) + 1)),
        "rk_order": RK_ORDER,
        "acoustic_substeps_per_rk": ACOUSTIC_SUBSTEPS_PER_RK,
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "attrs": attrs,
        "namelist_disabled": NAMELIST_DISABLED,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
        "direct_relinked_wrf": False,
        "composition_order": "10 timesteps x RK3 x acoustic_loop(calc_coef_w -> repeated acoustic substeps)",
        "geometric_growth_bound": "M6B4 per-substep tolerance x 10 acoustic substeps x 3 RK stages x 10 timesteps = 300x.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _dycore_ladder() -> dict[str, object]:
    ladder = load_tolerance_ladder()
    dycore = dict(ladder["dycore_step_tolerances"])  # type: ignore[index]
    return {
        "schema_version": ladder["schema_version"],
        "perturbation_rule": ladder["perturbation_rule"],
        "fields": dycore["fields"],
    }


def _compare_snapshot(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    ladder: dict[str, object],
) -> dict[str, object]:
    fields = {
        name: field_compare(name, np.asarray(actual[name]), np.asarray(expected[name]), ladder)
        for name in COMPARE_FIELDS
    }
    return {"passed": all(bool(item["passed"]) for item in fields.values()), "fields": fields}


def compare_tier(tier: str, steps: int, savepoint_root: Path) -> dict[str, object]:
    output = savepoint_root / tier
    manifest = emit_tier(tier, steps, output)
    expected_steps = [
        read_savepoint(output / f"dycore_step_complete_step{step:03d}.h5").arrays
        for step in range(1, int(steps) + 1)
    ]
    actual_steps = _actual_steps(tier, steps)
    ladder = _dycore_ladder()
    results = []
    for step, (expected, actual) in enumerate(zip(expected_steps, actual_steps, strict=True), start=1):
        compared = _compare_snapshot(
            {name: np.asarray(value) for name, value in expected.items()},
            {name: np.asarray(value) for name, value in actual.items()},
            ladder,
        )
        results.append(
            {
                "step": step,
                "tier": tier,
                "path": str(output / f"dycore_step_complete_step{step:03d}.h5"),
                "boundary": "dycore_step_complete",
                **compared,
            }
        )
    passed = all(bool(item["passed"]) for item in results)
    first_failed = next((item["step"] for item in results if not bool(item["passed"])), None)
    outcome = "SIXTH-DYCORE-STEP-PARITY-ACHIEVED" if passed else f"PARITY-DEFECT-LOCALIZED-AT-STEP-{first_failed}"
    return {
        "operator": "dycore_step",
        "tier": tier,
        "passed": bool(passed),
        "outcome": outcome,
        "savepoint_count": int(steps),
        "manifest": manifest,
        "results": results,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "namelist_disabled": NAMELIST_DISABLED,
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
        "operator": "dycore_step",
        "step": 1,
        "diverging_field_count": int(diverging),
        "threshold": 15,
        "passed": bool(diverging <= 15),
        "decision": "PROCEED_TO_M6B6" if diverging <= 15 else "STOP_ESCALATE_M6B5",
    }
    (SPRINT / "proof_kill_gate_status.txt").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def synthetic_dryrun() -> dict[str, object]:
    expected_steps, _ = _expected_steps("column", 1)
    actual_steps = _actual_steps("column", 1)
    ladder = _dycore_ladder()
    clean = _compare_snapshot(expected_steps[-1], actual_steps[-1], ladder)["fields"]
    perturbed = {}
    caught = True
    for name in COMPARE_FIELDS:
        bad = {field: np.array(value, copy=True) for field, value in expected_steps[-1].items()}
        tol = float(clean[name]["tolerance"])  # type: ignore[index]
        bad[name].flat[0] += 20.0 * tol
        result = field_compare(name, actual_steps[-1][name], bad[name], ladder)
        perturbed[name] = result
        caught = caught and not bool(result["passed"])
    payload = {
        "operator": "dycore_step",
        "clean_self_compare_passed": all(bool(item["passed"]) for item in clean.values()),
        "boundary_field_perturbations_caught": bool(caught),
        "clean": clean,
        "perturbed": perturbed,
        "passed": bool(all(bool(item["passed"]) for item in clean.values()) and caught),
        "source_path": str(SOURCE_WRFOUT),
        "namelist_disabled": NAMELIST_DISABLED,
        "sanitizer_mode": "off",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (SPRINT / "proof_synthetic_dryrun_m6b5.json").write_text(text + "\n")
    (SPRINT / "proof_synthetic_dryrun_m6b5.txt").write_text(text + "\n")
    return payload


def _summary_text(payload: dict[str, object]) -> str:
    lines = [
        f"operator={payload['operator']}",
        f"outcome={payload['outcome']}",
        f"passed={payload['passed']}",
        "geometric_growth_bound=M6B4 per-substep tolerance x 300",
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
    parser.add_argument("--synthetic-dryrun", action="store_true")
    args = parser.parse_args()

    _set_source_wrfout(args.source_wrfout)
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
    outcome = "SIXTH-DYCORE-STEP-PARITY-ACHIEVED" if passed else sorted(outcomes)[0]
    payload: dict[str, object] = {
        "operator": "dycore_step",
        "passed": bool(passed),
        "outcome": outcome,
        "tiers": tier_results,
        "operational_compatibility": {
            "sp_dycore_step_complete hook": "validation-only",
            "dycore_step.py callable": "validation-only",
            "per-timestep tolerance entries": "validation-only",
            "schema v6 extension": "validation-only",
        },
    }
    if args.tier == "all":
        payload["kill_gate"] = _write_kill_gate(payload)
    output = args.output
    if output is None:
        suffix = "all" if args.tier == "all" else str(args.tier)
        output = SPRINT / ("proof_dycore_step_parity.json" if suffix == "all" else f"proof_dycore_step_parity_{suffix}.json")
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    if args.tier == "all":
        (SPRINT / "proof_dycore_step_parity.txt").write_text(_summary_text(payload))
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
