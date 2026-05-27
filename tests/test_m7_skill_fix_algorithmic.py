from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp

from gpuwrf.contracts.state import Tendencies
from gpuwrf.coupling.physics_couplers import _surface_flux_column_inputs, _to_columns, mynn_adapter
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, hour_steps
from gpuwrf.physics.surface_constants import CP_D
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _limit_guarded_dynamics_state,
    _limit_theta_by_level,
    _physics_boundary_step,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_run_dummy_coupled.py"
_SPEC = importlib.util.spec_from_file_location("m6_run_dummy_coupled", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_dummy_grid = _MODULE.make_dummy_grid
make_initial_state = _MODULE.make_initial_state


def test_guarded_physics_boundary_step_keeps_rk3_theta_mu_update():
    grid = make_dummy_grid(4, 4, 8)
    state = make_initial_state(grid)
    tendencies = Tendencies.zeros(grid).replace(
        theta=jnp.ones_like(state.theta) * 0.01,
        mu=jnp.ones_like(state.mu) * 2.0,
    )
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=tendencies,
        metrics=grid.metrics,
        dt_s=1.0,
        acoustic_substeps=1,
        run_physics=False,
        run_boundary=False,
        use_vertical_solver=False,
        disable_guards=False,
    )

    out = _physics_boundary_step(
        initial_operational_carry(state),
        namelist,
        jnp.asarray(1, dtype=jnp.int32),
        run_radiation=False,
    ).state
    block_until_ready(out)

    assert float(jnp.max(jnp.abs(out.theta - state.theta))) > 0.0
    assert float(jnp.max(jnp.abs(out.mu_perturbation - state.mu_perturbation))) > 0.0


def test_mynn_adapter_consumes_nonzero_surface_flux_inputs():
    grid = make_dummy_grid(4, 4, 8)
    state = make_initial_state(grid)
    rho = jnp.ones_like(state.rhosfc) * 1.2
    theta_flux_w_m2 = 200.0
    theta_flux_kinematic = theta_flux_w_m2 / (rho * CP_D)
    fluxed = state.replace(
        theta_flux=theta_flux_kinematic,
        qv_flux=jnp.zeros_like(state.qv_flux),
        tau_u=jnp.zeros_like(state.tau_u),
        tau_v=jnp.zeros_like(state.tau_v),
        rhosfc=rho,
    )
    neutral = state.replace(
        theta_flux=jnp.zeros_like(state.theta_flux),
        qv_flux=jnp.zeros_like(state.qv_flux),
        tau_u=jnp.zeros_like(state.tau_u),
        tau_v=jnp.zeros_like(state.tau_v),
        rhosfc=rho,
    )

    theta_bc, qv_bc, tau_u_bc, tau_v_bc = _surface_flux_column_inputs(fluxed, _to_columns(fluxed.theta))
    assert float(jnp.max(theta_bc[..., 0])) > 0.0
    assert float(jnp.max(jnp.abs(theta_bc[..., 1:]))) == 0.0
    assert float(jnp.max(jnp.abs(qv_bc))) == 0.0
    assert float(jnp.max(jnp.abs(tau_u_bc))) == 0.0
    assert float(jnp.max(jnp.abs(tau_v_bc))) == 0.0

    heated = mynn_adapter(fluxed, 10.0, grid)
    baseline = mynn_adapter(neutral, 10.0, grid)
    block_until_ready((heated, baseline))

    lowest_delta = heated.theta[0] - baseline.theta[0]
    assert float(jnp.max(lowest_delta)) > 1.0e-4


def test_guard_limiter_clamps_theta_and_preserves_positive_mu_total():
    origin = jnp.ones((44, 1, 1), dtype=jnp.float64) * 300.0
    origin = origin.at[0, 0, 0].set(150.0).at[31, 0, 0].set(800.0)
    candidate = origin.at[0, 0, 0].set(100.0).at[31, 0, 0].set(800.0).at[2, 0, 0].set(jnp.nan)
    limited = _limit_theta_by_level(candidate, origin)

    assert float(limited[0, 0, 0]) == 200.0
    assert float(limited[31, 0, 0]) == 700.0
    assert float(limited[2, 0, 0]) == 300.0

    grid = make_dummy_grid(4, 4, 8)
    origin = make_initial_state(grid)
    bad_mu = jnp.ones_like(origin.mu_total) * -5.0
    bad = origin.replace(mu=bad_mu, mu_total=bad_mu, mu_perturbation=bad_mu)
    guarded = _limit_guarded_dynamics_state(bad, origin)

    assert float(jnp.min(guarded.mu_total)) >= 1.0
    assert float(jnp.min(guarded.mu)) >= 1.0


def test_daily_pipeline_default_radiation_cadence_is_wrf_30_minutes():
    config = DailyPipelineConfig()

    assert config.radiation_cadence_steps == 180
    assert hour_steps(24, config.dt_s) // config.radiation_cadence_steps == 48
