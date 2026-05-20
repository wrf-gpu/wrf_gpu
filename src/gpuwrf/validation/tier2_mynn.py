"""Tier-2 budget checks for the M5-S2 MYNN PBL column kernel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.mynn_constants import TKE_EPS
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, _mynn_budget_diagnostics
from gpuwrf.validation.tier1_mynn import load_fixture_state


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m5" / "tier2_mynn_invariants.json"


def _mass_by_column(field, state: MynnPBLColumnState):
    """Computes rho-dz weighted integrals independently per scenario."""

    return jnp.sum(field * state.rho * state.dz, axis=-1)


def _trajectory_metrics(state: MynnPBLColumnState, dt: float, n_steps: int):
    """Runs a scan and accumulates MYNN Tier-2 budget residuals."""

    def body(carry, _):
        current, u_res, v_res, theta_res, qv_res, tke_res, pos_bad, finite_bad = carry
        next_state, budget = _mynn_budget_diagnostics(current, dt)

        u_expected = dt * (budget["surface_u"] + budget["top_u"])
        v_expected = dt * (budget["surface_v"] + budget["top_v"])
        theta_expected = dt * (budget["surface_theta"] + budget["top_theta"])
        qv_expected = dt * (budget["surface_qv"] + budget["top_qv"])
        tke_expected = dt * _mass_by_column(
            budget["tke_production"] + budget["tke_transport"] - budget["tke_dissipation"],
            current,
        )

        u_delta = _mass_by_column(next_state.u, current) - _mass_by_column(current.u, current)
        v_delta = _mass_by_column(next_state.v, current) - _mass_by_column(current.v, current)
        theta_delta = _mass_by_column(next_state.theta, current) - _mass_by_column(current.theta, current)
        qv_delta = _mass_by_column(next_state.qv, current) - _mass_by_column(current.qv, current)
        tke_delta = _mass_by_column(next_state.tke, current) - _mass_by_column(current.tke, current)

        u_res = jnp.maximum(u_res, jnp.max(jnp.abs(u_delta - u_expected) / jnp.maximum(jnp.abs(u_expected), 1.0)))
        v_res = jnp.maximum(v_res, jnp.max(jnp.abs(v_delta - v_expected) / jnp.maximum(jnp.abs(v_expected), 1.0)))
        theta_res = jnp.maximum(theta_res, jnp.max(jnp.abs(theta_delta - theta_expected) / jnp.maximum(jnp.abs(theta_expected), 1.0)))
        qv_res = jnp.maximum(qv_res, jnp.max(jnp.abs(qv_delta - qv_expected) / jnp.maximum(jnp.abs(qv_expected), 1.0e-9)))
        tke_res = jnp.maximum(tke_res, jnp.max(jnp.abs(tke_delta - tke_expected) / jnp.maximum(jnp.abs(tke_expected), 1.0e-9)))

        pos = jnp.sum(next_state.tke < TKE_EPS) + jnp.sum(next_state.qv < 0.0)
        finite = sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(next_state))
        return (next_state, u_res, v_res, theta_res, qv_res, tke_res, pos_bad + pos, finite_bad + finite), None

    zeros = tuple(jnp.asarray(0.0, dtype=jnp.float64) for _ in range(5))
    init = (state, *zeros, jnp.asarray(0, dtype=jnp.int64), jnp.asarray(0, dtype=jnp.int64))
    return jax.lax.scan(body, init, xs=None, length=int(n_steps))[0]


def invariant_record(state: MynnPBLColumnState | None = None, dt: float | None = None, n_steps: int = 10) -> dict[str, Any]:
    """Computes the M5-S2 trajectory-wide Tier-2 budget result."""

    if state is None or dt is None:
        loaded_state, loaded_dt, _ = load_fixture_state()
        state = loaded_state
        dt = loaded_dt
    final, u_res, v_res, theta_res, qv_res, tke_res, pos_bad, finite_bad = _trajectory_metrics(state, float(dt), int(n_steps))
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, final)

    positivity_violations = int(np.asarray(pos_bad))
    nonfinite = int(np.asarray(finite_bad))
    momentum_residual = float(np.asarray(jnp.maximum(u_res, v_res)))
    heat_residual = float(np.asarray(theta_res))
    moisture_residual = float(np.asarray(qv_res))
    tke_residual = float(np.asarray(tke_res))
    tol = 5.0e-5
    record = {
        "iterations": int(n_steps),
        "dt_s": float(dt),
        "positivity": {"violations": positivity_violations, "tke_floor": TKE_EPS, "pass": positivity_violations == 0},
        "momentum_budget": {"relative_residual": momentum_residual, "tolerance": tol, "pass": momentum_residual <= tol},
        "heat_budget": {"theta_relative_residual": heat_residual, "tolerance": tol, "pass": heat_residual <= tol},
        "moisture_budget": {"qv_relative_residual": moisture_residual, "tolerance": tol, "pass": moisture_residual <= tol},
        "tke_budget": {
            "relative_residual": tke_residual,
            "tolerance": 2.0e-2,
            "terms": "surface/shear/buoyancy production + vertical transport - dissipation",
            "pass": tke_residual <= 2.0e-2,
        },
        "nan_inf": {"violations": nonfinite, "pass": nonfinite == 0},
        "pass": bool(
            positivity_violations == 0
            and nonfinite == 0
            and momentum_residual <= tol
            and heat_residual <= tol
            and moisture_residual <= tol
            and tke_residual <= 2.0e-2
        ),
    }
    return record


def run_tier2(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the required tier-2 MYNN budget proof JSON."""

    record = invariant_record()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
