"""Tier-2 invariant checks for the reduced M4 dycore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.dynamics.rk3 import rk3_step
from gpuwrf.profiling.transfer_audit import block_until_ready


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m4" / "tier2_invariants.json"


def make_ideal_grid(
    nz: int = 40,
    ny: int = 80,
    nx: int = 80,
    dx_m: float = 250.0,
    dy_m: float = 250.0,
    top_m: float = 4000.0,
) -> GridSpec:
    """Constructs the M4 idealized flat grid reused by scripts and tests."""

    projection = Projection("lambert", 0.0, 0.0, float(dx_m), float(dy_m), int(nx), int(ny))
    terrain = TerrainProvenance(
        source_path="analytic://m4-flat",
        sha256="analytic-m4-flat",
        shape=(int(ny), int(nx)),
        units="m",
        projection_transform="identity",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta_levels = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), float(top_m), eta_levels)
    bc = BCMetadata("ideal", ("u", "v", "theta", "qv", "p"), 0, "linear", False)
    terrain_height = jnp.zeros((int(ny), int(nx)), dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


def density_current_state(grid: GridSpec) -> tuple[State, Tendencies]:
    """Creates the M4 nontrivial tracer-translation state for invariant validation."""

    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    x0 = 0.30 * float(grid.nx)
    z0 = 0.50 * float(grid.nz - 1)
    radius_x = 0.10 * float(grid.nx)
    radius_z = 0.18 * float(grid.nz)
    bump = jnp.exp(-(((x - x0) / radius_x) ** 2 + ((z - z0) / radius_z) ** 2)) + y * 0.0
    theta = 300.0 + 8.0 * bump
    qv = theta * 0.0 + 1.0e-3
    p = theta * 0.0
    mu = jnp.ones((grid.ny, grid.nx), dtype=jnp.float64)
    u = jnp.ones_like(state.u) * 5.0
    return state.replace(u=u, theta=theta, qv=qv, p=p, mu=mu), tendencies


def _trajectory_with_checks(state: State, tendencies: Tendencies, grid: GridSpec, dt: float, n_steps: int, n_acoustic: int):
    """Runs the RK loop while accumulating trajectory-wide finiteness checks."""

    def body(carry, _):
        """Advances one step and accumulates validation counters."""

        current, qv_bad, finite_bad = carry
        next_state = rk3_step(current, tendencies, grid, dt, n_acoustic, False)
        qv_bad = qv_bad + jnp.sum(next_state.qv < 0.0)
        finite_bad = finite_bad + sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(next_state))
        return (next_state, qv_bad, finite_bad), None

    init = (state, jnp.asarray(0, dtype=jnp.int64), jnp.asarray(0, dtype=jnp.int64))
    return jax.lax.scan(body, init, xs=None, length=int(n_steps))[0]


def invariant_record(grid: GridSpec, n_steps: int = 100, dt: float = 2.0, n_acoustic: int = 4) -> dict[str, Any]:
    """Runs the dycore and computes the required tier-2 invariant scalars."""

    state, tendencies = density_current_state(grid)
    mass0 = jnp.sum(state.theta)
    final, qv_violations, nan_inf_violations = _trajectory_with_checks(
        state,
        tendencies,
        grid,
        dt,
        int(n_steps),
        int(n_acoustic),
    )
    block_until_ready(final)
    mass1 = jnp.sum(final.theta)
    mass_residual = jnp.abs(mass1 - mass0) / mass0
    max_theta_delta = jnp.max(jnp.abs(final.theta - state.theta))
    rms_velocity = jnp.sqrt(jnp.mean(final.u[:, :, :-1] ** 2 + final.v[:, :-1, :] ** 2) + jnp.mean(final.w[1:, :, :] ** 2))
    qv_bad = int(qv_violations)
    finite_bad = int(nan_inf_violations)
    record = {
        "case": "m4-linear-tracer-translation",
        "iterations": int(n_steps),
        "dt_s": float(dt),
        "mass_residual_relative": float(mass_residual),
        "mass_field": "theta_total",
        "qv_positivity_violations": qv_bad,
        "nan_inf_violations": finite_bad,
        "max_theta_delta": float(max_theta_delta),
        "rms_velocity": float(rms_velocity),
        "final_state_differs_from_initial": bool(float(max_theta_delta) > 0.1),
        "pass": bool(
            float(mass_residual) <= 1.0e-10
            and qv_bad == 0
            and finite_bad == 0
            and bool(float(max_theta_delta) > 0.1)
        ),
    }
    return record


def run_tier2(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the tier-2 invariant proof JSON for the contract command."""

    record = invariant_record(make_ideal_grid())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
