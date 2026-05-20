"""Tier-2 invariant checks for the M5-S2 MYNN PBL column kernel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.mynn_constants import TKE_EPS
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, _step_mynn_pbl_impl
from gpuwrf.validation.tier1_mynn import load_fixture_state


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m5" / "tier2_mynn_invariants.json"


def _sum(field, dz):
    """Computes a layer-depth-weighted column integral."""

    return jnp.sum(field * dz)


def _trajectory_metrics(state: MynnPBLColumnState, dt: float, n_steps: int):
    """Runs a scan and accumulates MYNN tier-2 invariant counters."""

    u0 = _sum(state.u, state.dz)
    v0 = _sum(state.v, state.dz)
    theta0 = _sum(state.theta, state.dz)
    qv0 = _sum(state.qv, state.dz)

    def body(carry, _):
        current, pos_bad, finite_bad = carry
        next_state = _step_mynn_pbl_impl(current, dt, False)
        pos = jnp.sum(next_state.tke < TKE_EPS) + jnp.sum(next_state.qv < 0.0)
        finite = sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(next_state))
        return (next_state, pos_bad + pos, finite_bad + finite), None

    init = (state, jnp.asarray(0, dtype=jnp.int64), jnp.asarray(0, dtype=jnp.int64))
    final, pos_bad, finite_bad = jax.lax.scan(body, init, xs=None, length=int(n_steps))[0]
    u_res = jnp.abs(_sum(final.u, final.dz) - u0) / jnp.maximum(jnp.abs(u0), jnp.finfo(jnp.float64).tiny)
    v_res = jnp.abs(_sum(final.v, final.dz) - v0) / jnp.maximum(jnp.abs(v0), jnp.finfo(jnp.float64).tiny)
    theta_res = jnp.abs(_sum(final.theta, final.dz) - theta0) / jnp.maximum(jnp.abs(theta0), jnp.finfo(jnp.float64).tiny)
    qv_res = jnp.abs(_sum(final.qv, final.dz) - qv0) / jnp.maximum(jnp.abs(qv0), jnp.finfo(jnp.float64).tiny)
    return final, pos_bad, finite_bad, u_res, v_res, theta_res, qv_res


def invariant_record(state: MynnPBLColumnState | None = None, dt: float | None = None, n_steps: int = 10) -> dict[str, Any]:
    """Computes the M5-S2 trajectory-wide Tier-2 invariant result."""

    if state is None or dt is None:
        loaded_state, loaded_dt, _ = load_fixture_state()
        state = loaded_state
        dt = loaded_dt
    final, pos_bad, finite_bad, u_res, v_res, theta_res, qv_res = _trajectory_metrics(state, float(dt), int(n_steps))
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, final)
    positivity_violations = int(np.asarray(pos_bad))
    nonfinite = int(np.asarray(finite_bad))
    momentum_residual = float(np.asarray(jnp.maximum(u_res, v_res)))
    theta_residual = float(np.asarray(theta_res))
    mass_residual = float(np.asarray(qv_res))
    tol = 1.0e-10
    record = {
        "iterations": int(n_steps),
        "dt_s": float(dt),
        "positivity": {"violations": positivity_violations, "tke_floor": TKE_EPS, "pass": positivity_violations == 0},
        "momentum_conservation": {"relative_residual": momentum_residual, "tolerance": tol, "pass": momentum_residual <= tol},
        "energy_conservation": {"theta_relative_residual": theta_residual, "tolerance": tol, "pass": theta_residual <= tol},
        "mass_conservation": {"qv_relative_residual": mass_residual, "tolerance": tol, "pass": mass_residual <= tol},
        "nan_inf": {"violations": nonfinite, "pass": nonfinite == 0},
        "pass": bool(positivity_violations == 0 and nonfinite == 0 and momentum_residual <= tol and theta_residual <= tol and mass_residual <= tol),
    }
    return record


def run_tier2(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the required tier-2 MYNN invariant proof JSON."""

    record = invariant_record()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
