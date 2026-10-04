"""Driver hold/clock wiring only; column physics is checked by WRF oracles."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.diagnostics import census
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry

from test_initial_held_output import fixture


@pytest.mark.parametrize("counted", [False, True])
def test_one_driver_solve_holds_heating_and_fluxes_with_separate_solar_clock(monkeypatch, counted):
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = fixture._state(grid)
    # Simple numerical leaves stand for live heating/flux fields. No mocked
    # atmospheric profile is being compared with WRF here.
    held = initial_operational_carry(state).replace(
        radiation_diagnostics=(jnp.ones((2, 2)),),
        census=census.initial_census() if counted else None)
    nml = SimpleNamespace(
        radiation_interval_s=1800., dt_s=54., radiation_cadence_steps=33,
        time_utc=None, grid=grid, radiation_static=None, topo_shading=0,
        slope_rad=0, topo_shadow_length_m=25000.)
    seen = []
    def solve(state, grid, **kwargs):
        assert kwargs["_with_diagnostics"] is True
        seen.append(1)
        # Expose each clock independently so a shared midpoint clock fails.
        return jnp.full_like(state.theta, kwargs["lead_seconds"]), (
            jnp.full_like(state.t_skin, kwargs["solar_lead_seconds"]),)
    monkeypatch.setattr(op, "rrtmg_theta_tendency", solve)
    run = jax.jit(lambda carry, pred, lead: op._refresh_rrtmg_driver(
        carry, nml, lead, pred, None))
    refreshed = run(held, jnp.asarray(True), jnp.asarray(43200.))
    np.testing.assert_array_equal(refreshed.rthraten, 43200.)
    np.testing.assert_array_equal(refreshed.radiation_diagnostics[0], 44100.)
    between = run(refreshed, jnp.asarray(False), jnp.asarray(43254.))
    np.testing.assert_array_equal(between.rthraten, refreshed.rthraten)
    np.testing.assert_array_equal(between.radiation_diagnostics[0], refreshed.radiation_diagnostics[0])
    if counted:
        work = dict(zip(census.WORK, map(int, jax.device_get(between.census.work))))
        assert work["radiation_tendency_calls"] == 1
        assert work["radiation_surface_calls"] == 0
        assert work["radiation_output_calls"] == 0
    assert len(seen) == 1  # One traced solve supplies both heating and fluxes.
