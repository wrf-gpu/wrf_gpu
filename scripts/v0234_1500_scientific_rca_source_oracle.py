#!/usr/bin/env python3
"""CPU-only source/algebra oracle for the v0.23.4 nested-ring RCA.

This module deliberately imports neither JAX nor gpuwrf.  It captures the
smallest scalar form of pristine WRF's ``spec_bdyupdate`` and
``spec_bdyupdate_ph`` equations so the candidate correction can be tested
against an implementation-independent NumPy oracle.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO_ROOT / ".agent/sprints/2026-07-14-v0234-1500-scientific-rca"
WRF_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
CANDIDATE_COMMIT = "985f5714b533d2c38da9c03ba305d8f8b3c59ed7"


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_proof(path: Path, payload: dict[str, Any]) -> str:
    payload["proof_sha256"] = canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload["proof_sha256"]


def wrf_spec_walk(start, tendency, dts: float, substeps: int) -> np.ndarray:
    """Pristine ``spec_bdyupdate`` trajectory, including the stage entry."""

    value = np.asarray(start, dtype=np.float64)
    tend = np.asarray(tendency, dtype=np.float64)
    rows = [value.copy()]
    for _ in range(int(substeps)):
        value = value + float(dts) * tend
        rows.append(value.copy())
    return np.stack(rows)


def endpoint_pin_walk(start, tendency, dts: float, substeps: int) -> np.ndarray:
    """Accepted candidate trajectory: stage endpoint is pinned every substep."""

    start_value = np.asarray(start, dtype=np.float64)
    endpoint = start_value + float(dts) * int(substeps) * np.asarray(
        tendency, dtype=np.float64
    )
    return np.stack([start_value, *[endpoint] * int(substeps)])


def wrf_spec_ph_update(
    ph_work,
    ph_save,
    field_tend,
    mu_tend,
    muts,
    c1f,
    c2f,
    dts: float,
) -> np.ndarray:
    """Literal vector form of WRF ``module_bc_em.F::spec_bdyupdate_ph``."""

    ph_work = np.asarray(ph_work, dtype=np.float64)
    ph_save = np.asarray(ph_save, dtype=np.float64)
    field_tend = np.asarray(field_tend, dtype=np.float64)
    mu_tend = np.asarray(mu_tend, dtype=np.float64)
    muts = np.asarray(muts, dtype=np.float64)
    c1f = np.asarray(c1f, dtype=np.float64)
    c2f = np.asarray(c2f, dtype=np.float64)
    mu_old = muts - float(dts) * mu_tend
    mass_old = c1f * mu_old + c2f
    mass_new = c1f * muts + c2f
    ratio = mass_old / mass_new
    return (
        ph_work * ratio
        + float(dts) * field_tend / mass_new
        + ph_save * (ratio - 1.0)
    )


def coupled_ph_after_update(
    ph_work,
    ph_save,
    field_tend,
    mu_tend,
    muts,
    c1f,
    c2f,
    dts: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both sides of the conservation identity closed by the WRF update."""

    ph_work = np.asarray(ph_work, dtype=np.float64)
    ph_save = np.asarray(ph_save, dtype=np.float64)
    field_tend = np.asarray(field_tend, dtype=np.float64)
    mu_tend = np.asarray(mu_tend, dtype=np.float64)
    muts = np.asarray(muts, dtype=np.float64)
    c1f = np.asarray(c1f, dtype=np.float64)
    c2f = np.asarray(c2f, dtype=np.float64)
    updated = wrf_spec_ph_update(
        ph_work, ph_save, field_tend, mu_tend, muts, c1f, c2f, dts
    )
    mass_new = c1f * muts + c2f
    mass_old = c1f * (muts - float(dts) * mu_tend) + c2f
    lhs = mass_new * (updated + ph_save)
    rhs = mass_old * (ph_work + ph_save) + float(dts) * field_tend
    return lhs, rhs


def _git_show(commit: str, path: str, *, cwd: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(cwd), "show", f"{commit}:{path}"), text=True
    )


def _git_blob(commit: str, path: str, *, cwd: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(cwd), "rev-parse", f"{commit}:{path}"), text=True
    ).strip()


def build_source_proof() -> dict[str, Any]:
    module_bc = _git_show(WRF_COMMIT, "share/module_bc.F", cwd=WRF_ROOT)
    module_bc_em = _git_show(WRF_COMMIT, "dyn_em/module_bc_em.F", cwd=WRF_ROOT)
    solve_em = _git_show(WRF_COMMIT, "dyn_em/solve_em.F", cwd=WRF_ROOT)
    candidate_acoustic = _git_show(
        CANDIDATE_COMMIT, "src/gpuwrf/dynamics/core/acoustic.py", cwd=REPO_ROOT
    )
    candidate_operational = _git_show(
        CANDIDATE_COMMIT, "src/gpuwrf/runtime/operational_mode.py", cwd=REPO_ROOT
    )

    required = {
        "wrf_additive_equation": "field(i,k,j) = field(i,k,j) + dt*field_tend(i,k,j)",
        "wrf_mass_old": "MU_OLD(i,j) = MUTS(i,j) - dt*MU_TEND(i,j)",
        "wrf_mu_update": "CALL spec_bdyupdate(grid%mu_2, mu_tend, dts_rk",
        "wrf_muts_update": "CALL spec_bdyupdate(grid%muts, mu_tend, dts_rk",
        "candidate_muave_pin": "muave_new = _pin_ring(muave_new, state.muave_spec_target)",
        "candidate_ph_hard_pin": "ph_next = _pin_spec_ring_wrf_owned(",
        "candidate_stage_endpoint": "spec-zone ring-0 targets at the STAGE-END lead",
    }
    checks = {
        "wrf_additive_equation": required["wrf_additive_equation"] in module_bc,
        "wrf_mass_old": required["wrf_mass_old"] in module_bc_em,
        "wrf_mu_update": required["wrf_mu_update"] in solve_em,
        "wrf_muts_update": required["wrf_muts_update"] in solve_em,
        "wrf_has_no_muave_spec_update": "spec_bdyupdate(muave" not in solve_em.lower(),
        "candidate_muave_pin": required["candidate_muave_pin"] in candidate_acoustic,
        "candidate_ph_hard_pin": required["candidate_ph_hard_pin"] in candidate_acoustic,
        "candidate_stage_endpoint": required["candidate_stage_endpoint"] in candidate_operational,
    }
    if not all(checks.values()):
        raise RuntimeError(f"source discriminator did not close: {checks}")

    start = np.asarray([2.0, -7.0])
    tendency = np.asarray([0.25, -0.4])
    substeps = 10
    dts = 0.6
    wrf_walk = wrf_spec_walk(start, tendency, dts, substeps)
    pin_walk = endpoint_pin_walk(start, tendency, dts, substeps)
    first_step_error = pin_walk[1] - wrf_walk[1]

    ph_args = {
        "ph_work": np.asarray([3.0, -2.0, 0.5]),
        "ph_save": np.asarray([100.0, 200.0, -50.0]),
        "field_tend": np.asarray([14.0, -5.0, 2.0]),
        "mu_tend": np.asarray([0.7, -0.2, 0.1]),
        "muts": np.asarray([9000.0, 11000.0, 8000.0]),
        "c1f": np.asarray([0.3, 0.7, 1.0]),
        "c2f": np.asarray([2.0, 3.0, 4.0]),
        "dts": 0.6,
    }
    lhs, rhs = coupled_ph_after_update(**ph_args)
    conservation_max_abs = float(np.max(np.abs(lhs - rhs)))
    conservation_tolerance = 3.0e-10
    if conservation_max_abs > conservation_tolerance:
        raise RuntimeError("independent coupled-ph conservation oracle failed")

    return {
        "schema": "gpuwrf.v0234.1500-scientific-rca-source-invariant.v1",
        "verdict": "SOURCE_DISCREPANCY_PROVEN",
        "frozen_inputs": {
            "candidate_commit": CANDIDATE_COMMIT,
            "pristine_wrf_commit": WRF_COMMIT,
            "source_blobs": {
                "wrf_share_module_bc": _git_blob(
                    WRF_COMMIT, "share/module_bc.F", cwd=WRF_ROOT
                ),
                "wrf_dyn_em_module_bc_em": _git_blob(
                    WRF_COMMIT, "dyn_em/module_bc_em.F", cwd=WRF_ROOT
                ),
                "wrf_dyn_em_solve_em": _git_blob(
                    WRF_COMMIT, "dyn_em/solve_em.F", cwd=WRF_ROOT
                ),
                "candidate_acoustic": _git_blob(
                    CANDIDATE_COMMIT,
                    "src/gpuwrf/dynamics/core/acoustic.py",
                    cwd=REPO_ROOT,
                ),
                "candidate_operational": _git_blob(
                    CANDIDATE_COMMIT,
                    "src/gpuwrf/runtime/operational_mode.py",
                    cwd=REPO_ROOT,
                ),
            },
        },
        "source_checks": checks,
        "equations": {
            "ordinary_spec": "field[n+1] = field[n] + dts * boundary_tendency",
            "ph_spec": "M_new*(ph_work_new+ph_save) = M_old*(ph_work_old+ph_save) + dts*ph_boundary_tendency",
            "muave": "advance_mu_t output is retained; pristine solve_em has no spec_bdyupdate(muave)",
        },
        "algebraic_discriminator": {
            "substeps": substeps,
            "dts": dts,
            "wrf_final": wrf_walk[-1].tolist(),
            "candidate_final": pin_walk[-1].tolist(),
            "first_substep_candidate_minus_wrf": first_step_error.tolist(),
            "first_substep_premature_fraction_of_stage_increment": (
                float(substeps - 1) / float(substeps)
            ),
            "final_values_match": bool(np.array_equal(wrf_walk[-1], pin_walk[-1])),
            "intermediate_values_match": bool(np.array_equal(wrf_walk[1:-1], pin_walk[1:-1])),
        },
        "conservation_oracle": {
            "coupled_ph_max_abs_residual": conservation_max_abs,
            "tolerance": conservation_tolerance,
            "passed": True,
        },
        "causal_prediction": {
            "first_affected_topology": "specified ring then spec-adjacent relaxation ring",
            "first_affected_families": ["U", "V", "T", "W", "dry_mass", "PH"],
            "pressure_signature": "dry-mass/PSFC split; diagnostic residual remains secondary",
            "v10_policy": "metric retained but mechanistically separate absent direct linkage",
        },
        "ranked_hypotheses": [
            {
                "id": "H-boundary-application",
                "status": "SUPPORTED_CAUSAL_SOURCE_DISCREPANCY",
                "reason": "exact stage-end hard pin and muave pin contradict pristine additive cadence/order",
            },
            {
                "id": "H-pressure-mass-coupling",
                "status": "SUPPORTED_DOWNSTREAM_MECHANISM",
                "reason": "premature mu/muts/ph ring forcing injects an acoustic dry-mass/pressure transient",
            },
            {
                "id": "H-forcedown-SINT",
                "status": "FALSIFIED_AS_EARLIEST_CAUSE",
                "reason": "accepted source/index/corner oracles close record construction; discrepancy occurs during consumption",
            },
            {
                "id": "H-interpolation-corner",
                "status": "FALSIFIED_AS_EARLIEST_CAUSE",
                "reason": "side/corner ownership closes; cadence error applies across the complete ring",
            },
            {
                "id": "H-noncandidate-or-unresolved",
                "status": "FALSIFIED_FOR_SOURCE_LOCALIZATION",
                "reason": "candidate-only nested branch contains an exact contradictory operator with the observed signature",
            },
        ],
        "commands": [
            "python scripts/v0234_1500_scientific_rca_source_oracle.py",
            "pytest -q tests/test_v0234_1500_scientific_rca_source_oracle.py",
        ],
        "gpu_commands": 0,
        "jax_imported": False,
    }


def main() -> int:
    proof = build_source_proof()
    digest = write_proof(SPRINT_DIR / "source-invariant-proof.json", proof)
    print(json.dumps({"verdict": proof["verdict"], "proof_sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
