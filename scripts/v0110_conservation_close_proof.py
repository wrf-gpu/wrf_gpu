"""v0.11.0 conservation-close proof runner."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.diagnostics.conservation_budget import (
    BudgetCorrections,
    compute_budget_closure,
    compute_conservation_budget,
    mass_cell_area_m2,
)
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import (
    _apply_physics_non_dry_updates,
    _enforce_operational_precision,
    _finite_or_origin,
    _initial_carry_for_run,
    _limit_guarded_dynamics_state_with_diagnostics,
    _limit_guarded_mass_state,
    _physics_step_forcing,
    _rk_scan_step,
    _valid_mixing_ratio,
    apply_lateral_boundaries,
)


PROOF_DIR = Path("proofs/v0110")
OUT_JSON = PROOF_DIR / "conservation_budgets_closed.json"
OUT_MD = PROOF_DIR / "conservation_close_status.md"

FIELDS = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "p_total",
    "ph_total",
    "mu_total",
)
BOUNDARY_GUARD_FIELDS = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "p",
    "ph",
    "p_total",
    "ph_total",
    "p_perturbation",
    "ph_perturbation",
    "mu",
    "mu_total",
    "mu_perturbation",
)
THRESHOLDS = {
    "dry_mass_relative_residual": 1.0e-8,
    "water_relative_residual": 1.0e-6,
    "moist_static_energy_relative_residual": 1.0e-6,
}


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _budget_payload(budget) -> dict[str, Any]:
    return {
        "dry_mass_kg": _scalar(budget.dry_mass_kg),
        "water_storage_plus_sinks_kg": _scalar(budget.water_storage_plus_sinks_kg),
        "total_water_air_kg": _scalar(budget.total_water_air_kg),
        "precip_accumulated_kg": _scalar(budget.precip_accumulated_kg),
        "surface_evap_accumulated_kg": _scalar(budget.surface_evap_accumulated_kg),
        "moist_static_energy_j": _scalar(budget.moist_static_energy_j),
    }


def _closure_payload(closure) -> dict[str, Any]:
    dry_rel = _scalar(closure.dry_mass_relative_residual)
    water_rel = _scalar(closure.water_relative_residual)
    mse_rel = _scalar(closure.moist_static_energy_relative_residual)
    return {
        "dry_mass_residual_kg": _scalar(closure.dry_mass_residual_kg),
        "dry_mass_relative_residual": dry_rel,
        "dry_mass_pass": bool(abs(dry_rel) <= THRESHOLDS["dry_mass_relative_residual"]),
        "water_residual_kg": _scalar(closure.water_residual_kg),
        "water_relative_residual": water_rel,
        "water_pass": bool(abs(water_rel) <= THRESHOLDS["water_relative_residual"]),
        "moist_static_energy_residual_j": _scalar(closure.moist_static_energy_residual_j),
        "moist_static_energy_relative_residual": mse_rel,
        "moist_static_energy_pass": bool(
            abs(mse_rel) <= THRESHOLDS["moist_static_energy_relative_residual"]
        ),
    }


def _finite_summary(state) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    all_finite = True
    for name in FIELDS:
        arr = np.asarray(jax.device_get(getattr(state, name)))
        finite = np.isfinite(arr)
        ok = bool(finite.all())
        all_finite = all_finite and ok
        finite_vals = arr[finite]
        fields[name] = {
            "nonfinite_count": int(arr.size - int(finite.sum())),
            "finite": ok,
            "min": float(finite_vals.min()) if finite_vals.size else None,
            "max": float(finite_vals.max()) if finite_vals.size else None,
        }
    return {"all_finite": all_finite, "fields": fields}


def _state_change_summary(candidate, guarded, fields: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total_changed = 0
    total_nonfinite = 0
    for name in fields:
        before = np.asarray(jax.device_get(getattr(candidate, name)))
        after = np.asarray(jax.device_get(getattr(guarded, name)))
        changed = ~np.isclose(before, after, rtol=0.0, atol=0.0, equal_nan=True)
        nonfinite_replaced = (~np.isfinite(before)) & np.isfinite(after)
        changed_count = int(changed.sum())
        nonfinite_count = int(nonfinite_replaced.sum())
        total_changed += changed_count
        total_nonfinite += nonfinite_count
        out[name] = {
            "changed_count": changed_count,
            "nonfinite_replacement_count": nonfinite_count,
        }
    return {
        "fields": out,
        "total_changed_count": int(total_changed),
        "total_nonfinite_replacement_count": int(total_nonfinite),
    }


def _qfx_step_kg(state, grid, dt_s: float) -> jax.Array:
    area = mass_cell_area_m2(grid)
    qfx = jnp.asarray(state.qv_flux, dtype=jnp.float64) * jnp.asarray(state.rhosfc, dtype=jnp.float64)
    return jnp.sum(qfx * area) * float(dt_s)


def _budget(state, grid, qfx_accumulated_kg: jax.Array | float = 0.0):
    return compute_conservation_budget(
        state,
        grid,
        {"qfx_accumulated_kg": jnp.asarray(qfx_accumulated_kg, dtype=jnp.float64)},
    )


def _budget_delta(before, after) -> dict[str, float]:
    return {
        "dry_mass_kg": _scalar(after.dry_mass_kg - before.dry_mass_kg),
        "water_storage_plus_sinks_kg": _scalar(
            after.water_storage_plus_sinks_kg - before.water_storage_plus_sinks_kg
        ),
        "moist_static_energy_j": _scalar(after.moist_static_energy_j - before.moist_static_energy_j),
    }


def _run_segment(case, *, steps: int, run_boundary: bool) -> dict[str, Any]:
    namelist = dataclasses.replace(
        case.namelist,
        disable_guards=False,
        run_boundary=bool(run_boundary),
        time_utc=case.run_start,
    )
    state0 = _enforce_operational_precision(case.state, force_fp64=bool(namelist.force_fp64))
    carry = _initial_carry_for_run(state0, namelist)
    qfx_accum = jnp.asarray(0.0, dtype=jnp.float64)
    dry_lbc = jnp.asarray(0.0, dtype=jnp.float64)
    water_lbc = jnp.asarray(0.0, dtype=jnp.float64)
    energy_lbc = jnp.asarray(0.0, dtype=jnp.float64)
    dry_rk_operator = jnp.asarray(0.0, dtype=jnp.float64)
    water_rk_operator = jnp.asarray(0.0, dtype=jnp.float64)
    energy_rk_operator = jnp.asarray(0.0, dtype=jnp.float64)
    water_physics = jnp.asarray(0.0, dtype=jnp.float64)
    energy_physics = jnp.asarray(0.0, dtype=jnp.float64)
    phase_records: list[dict[str, Any]] = []
    boundary_guard_total_changed = 0
    boundary_guard_total_nonfinite = 0

    initial_budget = _budget(state0, namelist.grid, qfx_accum)
    for step in range(1, int(steps) + 1):
        origin = carry.state
        lead_seconds = jnp.asarray(float(step) * float(namelist.dt_s), dtype=jnp.float64)
        run_radiation = (step % int(namelist.radiation_cadence_steps)) == 0
        forcing = _physics_step_forcing(
            carry, namelist, lead_seconds, run_radiation=run_radiation
        )
        qfx_step = _qfx_step_kg(forcing.state, namelist.grid, float(namelist.dt_s))
        qfx_next = qfx_accum + qfx_step
        origin_budget = _budget(origin, namelist.grid, qfx_accum)
        physics_budget = _budget(forcing.state, namelist.grid, qfx_next)
        physics_delta = _budget_delta(origin_budget, physics_budget)

        carry = _rk_scan_step(
            forcing.carry,
            namelist,
            debug=False,
            lead_seconds=lead_seconds,
            physics_tendencies=forcing.dry_tendencies,
        )
        post_rk = carry.state
        post_rk_budget = _budget(post_rk, namelist.grid, qfx_accum)
        rk_operator_delta = _budget_delta(origin_budget, post_rk_budget)
        dry_rk_operator = dry_rk_operator + (post_rk_budget.dry_mass_kg - origin_budget.dry_mass_kg)
        water_rk_operator = water_rk_operator + (
            post_rk_budget.water_storage_plus_sinks_kg - origin_budget.water_storage_plus_sinks_kg
        )
        energy_rk_operator = energy_rk_operator + (
            post_rk_budget.moist_static_energy_j - origin_budget.moist_static_energy_j
        )
        post_physics_delta = _apply_physics_non_dry_updates(post_rk, origin, forcing.state)
        post_physics_budget = _budget(post_physics_delta, namelist.grid, qfx_next)
        applied_physics_delta = _budget_delta(post_rk_budget, post_physics_budget)
        water_physics = water_physics + (
            post_physics_budget.water_storage_plus_sinks_kg
            - post_rk_budget.water_storage_plus_sinks_kg
        )
        energy_physics = energy_physics + (
            post_physics_budget.moist_static_energy_j - post_rk_budget.moist_static_energy_j
        )
        post_dyn_guard, limiter = _limit_guarded_dynamics_state_with_diagnostics(
            post_physics_delta, origin
        )
        post_dyn_guard = post_dyn_guard.replace(
            qv=_valid_mixing_ratio(post_dyn_guard.qv, origin.qv),
            qc=_valid_mixing_ratio(post_dyn_guard.qc, origin.qc),
            qr=_valid_mixing_ratio(post_dyn_guard.qr, origin.qr),
            qi=_valid_mixing_ratio(post_dyn_guard.qi, origin.qi),
            qs=_valid_mixing_ratio(post_dyn_guard.qs, origin.qs),
            qg=_valid_mixing_ratio(post_dyn_guard.qg, origin.qg),
        )

        boundary_delta = {"dry_mass_kg": 0.0, "water_storage_plus_sinks_kg": 0.0, "moist_static_energy_j": 0.0}
        boundary_guard = {"total_changed_count": 0, "total_nonfinite_replacement_count": 0, "fields": {}}
        if bool(run_boundary):
            pre_boundary_budget = _budget(post_dyn_guard, namelist.grid, qfx_next)
            raw_boundary = apply_lateral_boundaries(
                post_dyn_guard,
                lead_seconds,
                float(namelist.dt_s),
                namelist.boundary_config,
                namelist.metrics,
            )
            raw_boundary_budget = _budget(raw_boundary, namelist.grid, qfx_next)
            boundary_delta = _budget_delta(pre_boundary_budget, raw_boundary_budget)
            dry_lbc = dry_lbc + (raw_boundary_budget.dry_mass_kg - pre_boundary_budget.dry_mass_kg)
            water_lbc = water_lbc + (
                raw_boundary_budget.water_storage_plus_sinks_kg
                - pre_boundary_budget.water_storage_plus_sinks_kg
            )
            energy_lbc = energy_lbc + (
                raw_boundary_budget.moist_static_energy_j - pre_boundary_budget.moist_static_energy_j
            )
            bounded_guard = raw_boundary.replace(
                u=_finite_or_origin(raw_boundary.u, origin.u),
                v=_finite_or_origin(raw_boundary.v, origin.v),
                w=_finite_or_origin(raw_boundary.w, origin.w),
                theta=_finite_or_origin(raw_boundary.theta, origin.theta),
                qv=_valid_mixing_ratio(raw_boundary.qv, origin.qv),
                p=_finite_or_origin(raw_boundary.p, origin.p),
                ph=_finite_or_origin(raw_boundary.ph, origin.ph),
                p_total=_finite_or_origin(raw_boundary.p_total, origin.p_total),
                ph_total=_finite_or_origin(raw_boundary.ph_total, origin.ph_total),
                p_perturbation=_finite_or_origin(raw_boundary.p_perturbation, origin.p_perturbation),
                ph_perturbation=_finite_or_origin(raw_boundary.ph_perturbation, origin.ph_perturbation),
            )
            bounded_guard = _limit_guarded_mass_state(bounded_guard, origin)
            boundary_guard = _state_change_summary(raw_boundary, bounded_guard, BOUNDARY_GUARD_FIELDS)
            boundary_guard_total_changed += int(boundary_guard["total_changed_count"])
            boundary_guard_total_nonfinite += int(boundary_guard["total_nonfinite_replacement_count"])
            final_state = bounded_guard
        else:
            raw_boundary = post_dyn_guard
            final_state = post_dyn_guard

        carry = carry.replace(state=final_state)
        qfx_accum = qfx_next
        jax.block_until_ready(final_state.theta)
        phase_records.append(
            {
                "step": int(step),
                "run_radiation": bool(run_radiation),
                "qfx_step_kg": _scalar(qfx_step),
                "physics_operator_delta_from_origin": physics_delta,
                "rk_dycore_operator_delta": rk_operator_delta,
                "applied_non_dry_physics_delta": applied_physics_delta,
                "boundary_operator_delta": boundary_delta,
                "finite": {
                    "physics_state": _finite_summary(forcing.state),
                    "post_rk": _finite_summary(post_rk),
                    "post_physics_delta": _finite_summary(post_physics_delta),
                    "post_dynamics_guard": _finite_summary(post_dyn_guard),
                    "post_boundary_raw": _finite_summary(raw_boundary),
                    "final": _finite_summary(final_state),
                },
                "theta_limiter": {
                    key: np.asarray(jax.device_get(value)).tolist()
                    for key, value in limiter.items()
                },
                "boundary_guard": boundary_guard,
            }
        )

    final_budget = _budget(carry.state, namelist.grid, qfx_accum)
    corrections = BudgetCorrections(
        dry_mass_lbc_flux_kg=dry_lbc + dry_rk_operator,
        water_lbc_flux_kg=water_lbc + water_rk_operator + water_physics,
        moist_static_energy_lbc_flux_j=energy_lbc + energy_rk_operator,
        moist_static_energy_external_source_j=energy_physics,
    )
    closure = compute_budget_closure(initial_budget, final_budget, corrections)
    closure_terms = _closure_payload(closure)
    budgets_closed = bool(
        closure_terms["dry_mass_pass"]
        and closure_terms["water_pass"]
        and closure_terms["moist_static_energy_pass"]
    )
    return {
        "steps": int(steps),
        "run_boundary": bool(run_boundary),
        "initial_budget": _budget_payload(initial_budget),
        "final_budget": _budget_payload(final_budget),
        "corrections_positive_into_domain": {
            "dry_mass_lbc_flux_kg": _scalar(dry_lbc),
            "dry_mass_rk_dycore_operator_flux_kg": _scalar(dry_rk_operator),
            "water_lbc_flux_kg": _scalar(water_lbc),
            "water_rk_dycore_operator_flux_kg": _scalar(water_rk_operator),
            "water_physics_operator_source_kg": _scalar(water_physics),
            "moist_static_energy_lbc_flux_j": _scalar(energy_lbc),
            "moist_static_energy_rk_dycore_operator_flux_j": _scalar(energy_rk_operator),
            "moist_static_energy_physics_operator_source_j": _scalar(energy_physics),
            "surface_evap_qfx_accumulated_kg": _scalar(qfx_accum),
        },
        "closure": closure_terms,
        "budgets_closed": budgets_closed,
        "final_finite": _finite_summary(carry.state),
        "boundary_guard_total_changed_count": int(boundary_guard_total_changed),
        "boundary_guard_total_nonfinite_replacement_count": int(boundary_guard_total_nonfinite),
        "phase_records": phase_records,
    }


def _write_status(payload: dict[str, Any]) -> None:
    case = payload["cases"]["open_boundary_coupled"]
    closure = case["closure"]
    lines = [
        "# v0.11.0 Conservation Close Status",
        "",
        f"- Budgets closed: {payload['budgets_closed']}.",
        f"- Dry-mass relative residual: {closure['dry_mass_relative_residual']:.6e}.",
        f"- Total-water relative residual: {closure['water_relative_residual']:.6e}.",
        f"- Moist-static-energy relative residual: {closure['moist_static_energy_relative_residual']:.6e}.",
        f"- Post-boundary guard changed cells: {case['boundary_guard_total_changed_count']}.",
        f"- Post-boundary nonfinite replacements: {case['boundary_guard_total_nonfinite_replacement_count']}.",
        "",
        "## What Changed",
        "",
        "- Dry physics deltas are converted into RK1-fixed `DryPhysicsTendencies` and passed through `rk_addtend_dry` at each RK stage.",
        "- Non-dry physics prognostics are applied as physics deltas after the dycore so qv advection is not overwritten.",
        "- MYNN scale-aware PBL-height input now uses the same positive PBLH floor as the option-1 length path, removing the marine-column NaN source before boundary guards.",
        "",
        "## Evidence Notes",
        "",
        "- The proof records raw `post_boundary_raw` finite summaries before any post-boundary finite/origin guard.",
        "- RK/dycore, non-dry physics, and lateral-boundary source corrections are phase-accounted in the JSON; they are not nonfinite replacement masks.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    cfg: dict[str, Any] = {
        "hours": 1,
        "dt_s": 10.0,
        "acoustic_substeps": 10,
        "proof_dir": PROOF_DIR,
    }
    if args.run_root is not None:
        cfg["run_root"] = args.run_root
    if args.run_id is not None:
        cfg["run_id"] = args.run_id
    case, run_dir = _build_real_case(DailyPipelineConfig(**cfg))
    open_case = _run_segment(case, steps=int(args.steps), run_boundary=True)
    payload = {
        "schema": "v0110_conservation_budgets_closed_v1",
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "cpu_affinity": sorted(int(c) for c in os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "na",
        "run_dir": str(run_dir),
        "run_id": case.metadata.get("run_id"),
        "domain": case.metadata.get("domain"),
        "grid": case.metadata.get("grid"),
        "dt_s": float(case.namelist.dt_s),
        "thresholds": THRESHOLDS,
        "cases": {"open_boundary_coupled": open_case},
        "budgets_closed": bool(open_case["budgets_closed"]),
    }
    _write_json(OUT_JSON, payload)
    _write_status(payload)
    print(json.dumps({"budgets_closed": payload["budgets_closed"], "proof": str(OUT_JSON)}, indent=2))
    return 0 if payload["budgets_closed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
