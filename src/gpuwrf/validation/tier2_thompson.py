"""Tier-2 invariant checks for the Thompson column source/sink subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_column import ThompsonColumnState, _step_thompson_column_impl
from gpuwrf.validation.tier1_thompson import OUTPUT_FIELDS, load_fixture_state


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m5" / "tier2_thompson_invariants.json"
WATER_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER_FIELDS = ("Ni", "Nr")


def _total_water(state: ThompsonColumnState):
    """Computes column total water mixing ratio for the sedimentation-free budget."""

    return state.qv + state.qc + state.qr + state.qi + state.qs + state.qg


def _trajectory_metrics(state: ThompsonColumnState, dt: float, n_steps: int):
    """Runs a scan and carries trajectory-wide invariant counters."""

    water0 = jnp.sum(_total_water(state))
    T0 = state.T

    def body(carry, _):
        """Advances one source/sink step and accumulates violations."""

        current, pos_bad, finite_bad, heat_max = carry
        next_state = _step_thompson_column_impl(current, dt, False)
        hydro_bad = sum(jnp.sum(getattr(next_state, field) < 0.0) for field in WATER_FIELDS)
        number_bad = sum(jnp.sum(getattr(next_state, field) < 0.0) for field in NUMBER_FIELDS)
        finite = sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(next_state))
        heat = jnp.max(jnp.abs(next_state.T - T0))
        return (next_state, pos_bad + hydro_bad + number_bad, finite_bad + finite, jnp.maximum(heat_max, heat)), None

    init = (state, jnp.asarray(0, dtype=jnp.int64), jnp.asarray(0, dtype=jnp.int64), jnp.asarray(0.0, dtype=jnp.float64))
    final, pos_bad, finite_bad, heat_max = jax.lax.scan(body, init, xs=None, length=int(n_steps))[0]
    water1 = jnp.sum(_total_water(final))
    residual = jnp.abs(water1 - water0) / jnp.maximum(jnp.abs(water0), jnp.finfo(jnp.float64).tiny)
    return final, pos_bad, finite_bad, heat_max, residual


def _wrf_one_step_budget(state: ThompsonColumnState, dt: float, expected: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compares aggregate candidate budgets against the WRF-linked harness output."""

    candidate = _step_thompson_column_impl(state, dt, False)
    water_initial = np.asarray(sum(np.asarray(getattr(state, field), dtype=np.float64) for field in WATER_FIELDS), dtype=np.float64)
    water_candidate = np.asarray(sum(np.asarray(getattr(candidate, field), dtype=np.float64) for field in WATER_FIELDS), dtype=np.float64)
    water_wrf = np.asarray(sum(expected[field] for field in WATER_FIELDS), dtype=np.float64)
    water_delta_residual = np.abs((water_candidate - water_initial) - (water_wrf - water_initial))
    per_field_abs = {
        field: float(np.max(np.abs(np.asarray(getattr(candidate, field), dtype=np.float64) - expected[field])))
        for field in OUTPUT_FIELDS
    }
    tracked_number_abs = max(per_field_abs["Ni"], per_field_abs["Nr"])
    return {
        "source": "WRF linked Fortran harness fixture output",
        "water_delta_max_abs": float(np.max(water_delta_residual)),
        "water_delta_tolerance": 1.0e-8,
        "water_delta_pass": bool(np.max(water_delta_residual) <= 1.0e-8),
        "tracked_number_max_abs": float(tracked_number_abs),
        "tracked_number_tolerance": 1.0e5,
        "tracked_number_pass": bool(tracked_number_abs <= 1.0e5),
        "per_field_max_abs_err": per_field_abs,
        "number_caveat": "WRF finalizes Ns/Ng/Qb internally but this M5 harness exposes only Ni/Nr; tracked-number tolerance is carry-forward diagnostic.",
    }


def invariant_record(state: ThompsonColumnState | None = None, dt: float | None = None, n_steps: int = 10) -> dict[str, Any]:
    """Computes the M5-S1 trajectory-wide Tier-2 invariant result."""

    if state is None or dt is None:
        loaded_state, loaded_dt, expected = load_fixture_state()
        state = loaded_state
        dt = loaded_dt
    else:
        _loaded_state, _loaded_dt, expected = load_fixture_state()
    final, pos_bad, finite_bad, heat_max, residual = _trajectory_metrics(state, float(dt), int(n_steps))
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, final)
    positivity_violations = int(np.asarray(pos_bad))
    nonfinite = int(np.asarray(finite_bad))
    max_heat = float(np.asarray(heat_max))
    water_residual = float(np.asarray(residual))
    record = {
        "iterations": int(n_steps),
        "dt_s": float(dt),
        "positivity": {"violations": positivity_violations, "pass": positivity_violations == 0},
        "water_budget": {"relative_residual": water_residual, "tolerance": 1.0e-8, "pass": water_residual <= 1.0e-8},
        "finite_latent_heating": {"max_abs_delta_T_K": max_heat, "bound_K": 100.0, "pass": max_heat < 100.0},
        "nan_inf": {"violations": nonfinite, "pass": nonfinite == 0},
    }
    wrf_budget = _wrf_one_step_budget(state, float(dt), expected)
    record["wrf_harness_one_step_budget"] = wrf_budget
    record["pass"] = bool(
        positivity_violations == 0
        and nonfinite == 0
        and water_residual <= 1.0e-8
        and max_heat < 100.0
        and wrf_budget["water_delta_pass"]
        and wrf_budget["tracked_number_pass"]
    )
    return record


def run_tier2(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the required tier-2 Thompson invariant proof JSON."""

    record = invariant_record()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
