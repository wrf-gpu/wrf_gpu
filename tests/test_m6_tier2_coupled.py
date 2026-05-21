from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG
from gpuwrf.coupling.driver import BoundarySnapshot, run_forecast_segment
from gpuwrf.validation.tier2_coupled import (
    boundary_flux_closure,
    dry_mass_residual,
    hydrometeor_positivity,
    nan_inf_count,
    tke_positivity,
    water_budget_residual,
)


def _physical_state() -> tuple[State, Tendencies, GridSpec]:
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    z_interfaces = jnp.linspace(0.0, 9000.0, grid.nz + 1, dtype=jnp.float64)[:, None, None]
    ph = jnp.broadcast_to(z_interfaces * 9.80665, state.ph.shape)
    state = state.replace(
        theta=jnp.ones_like(state.theta) * 300.0,
        qv=jnp.ones_like(state.qv) * 0.01,
        p=jnp.ones_like(state.p) * 90000.0,
        ph=ph,
        mu=jnp.ones_like(state.mu) * 80000.0,
        t_skin=jnp.ones_like(state.t_skin) * 295.0,
        soil_moisture=jnp.ones_like(state.soil_moisture) * 0.2,
        xland=jnp.ones_like(state.xland),
        lakemask=jnp.zeros_like(state.lakemask),
        mavail=jnp.ones_like(state.mavail) * 0.2,
        roughness_m=jnp.ones_like(state.roughness_m) * 0.05,
    )
    return state, Tendencies.zeros(grid), grid


def test_tier2_kernels_pass_on_unchanged_physical_state():
    state, _tendencies, _grid = _physical_state()

    assert dry_mass_residual(state, state, 60.0)["max_abs"] == 0.0
    assert water_budget_residual(state, state, 60.0)["max_abs"] == 0.0
    assert hydrometeor_positivity(state)["violations"] == 0
    assert tke_positivity(state)["violations"] == 0
    assert nan_inf_count(state)["violations"] == 0


def test_boundary_flux_closure_uses_pre_boundary_tendency_snapshot():
    state, _tendencies, _grid = _physical_state()
    dt = 60.0
    tendency = BoundarySnapshot(
        jnp.zeros_like(state.u),
        jnp.zeros_like(state.v),
        jnp.zeros_like(state.theta),
        jnp.zeros_like(state.qv),
        jnp.zeros_like(state.ph),
        jnp.ones_like(state.mu) * 2.0,
    )
    state_next = state.replace(mu=state.mu + dt * tendency.mu)

    record = boundary_flux_closure(
        state,
        state_next,
        dt,
        {"pre_boundary": BoundarySnapshot(state.u, state.v, state.theta, state.qv, state.ph, state.mu), "tendency": tendency},
    )

    assert record["max_abs"] == 0.0
    assert record["per_leaf"]["mu"]["max_abs"] == 0.0


def test_forecast_segment_pre_sanitize_tap_returns_one_state_per_step():
    state, tendencies, grid = _physical_state()

    final_state, tap = run_forecast_segment(
        state,
        tendencies,
        grid,
        1.0,
        2,
        start_step=0,
        total_steps=2,
        n_acoustic=1,
        radiation_cadence_steps=10,
        final_radiation=False,
        boundary_config=DEFAULT_BOUNDARY_CONFIG,
        capture_pre_sanitize=True,
    )

    assert final_state.mu.shape == state.mu.shape
    assert tap.state.mu.shape[0] == 2
    assert tap.pre_boundary.mu.shape[0] == 2
    assert tap.boundary_tendency.mu.shape[0] == 2
