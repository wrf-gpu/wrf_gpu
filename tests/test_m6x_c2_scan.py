from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig, acoustic_substep
from gpuwrf.dynamics.damping import SmdivConfig
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.dynamics.orchestrator import OrchestratorConfig, run_scan


def _analytic_state(grid: GridSpec) -> State:
    arrays = {
        field: jnp.zeros(shape, dtype=jnp.float64)
        for field, shape in _state_field_shapes(grid).items()
    }
    arrays["p"] = jnp.ones_like(arrays["p"]) * 1000.0
    arrays["p_total"] = arrays["p"]
    arrays["p_perturbation"] = arrays["p"]
    arrays["ph_total"] = arrays["ph"]
    arrays["ph_perturbation"] = arrays["ph"]
    arrays["theta"] = jnp.ones_like(arrays["theta"]) * 300.0
    arrays["mu"] = jnp.ones_like(arrays["mu"]) * 90000.0
    arrays["mu_total"] = arrays["mu"]
    arrays["mu_perturbation"] = arrays["mu"]
    return State(**arrays)


def test_state_decomposition_round_trips_against_base_state():
    grid = GridSpec.canary_3km_template()
    state = _analytic_state(grid)
    base = BaseState(
        pb=jnp.ones_like(state.p_total) * 90000.0,
        phb=jnp.ones_like(state.ph_total) * 10.0,
        mub=jnp.ones_like(state.mu_total) * 85000.0,
        t0=jnp.ones_like(state.theta) * 300.0,
        theta_base=jnp.ones_like(state.theta) * 300.0,
    )
    state = state.replace(
        p_total=base.pb + 250.0,
        p_perturbation=jnp.ones_like(state.p_perturbation) * 250.0,
        ph_total=base.phb + 3.0,
        ph_perturbation=jnp.ones_like(state.ph_perturbation) * 3.0,
        mu_total=base.mub + 125.0,
        mu_perturbation=jnp.ones_like(state.mu_perturbation) * 125.0,
    )

    assert jnp.max(jnp.abs(state.p_total - (state.p_perturbation + base.pb))) < 1.0e-12
    assert jnp.max(jnp.abs(state.ph_total - (state.ph_perturbation + base.phb))) < 1.0e-12
    assert jnp.max(jnp.abs(state.mu_total - (state.mu_perturbation + base.mub))) < 1.0e-12


def test_acoustic_substep_carries_previous_pressure_and_smdiv_effect():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _analytic_state(grid)
    previous_pressure = state.p_total - 1.0

    next_state, next_previous = acoustic_substep(
        state,
        previous_pressure,
        metrics,
        AcousticConfig(smdiv=SmdivConfig(enabled=True, coefficient=0.1)),
    )

    assert jnp.allclose(next_previous, state.p_total)
    assert not jnp.allclose(next_state.p_total, state.p_total)
    assert jnp.all(jnp.isfinite(next_state.p_total))


def test_outer_and_nested_scan_run_without_host_callback_primitives():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _analytic_state(grid)
    config = OrchestratorConfig(acoustic=AcousticConfig(n_substeps=2))

    carry = run_scan(state, metrics, config, 1.0, 3)

    assert carry.state.p_total.shape == state.p_total.shape
    assert carry.previous_pressure.shape == state.p_total.shape
    assert jnp.all(jnp.isfinite(carry.state.theta))

    jaxpr = str(jax.make_jaxpr(run_scan, static_argnums=(2, 3, 4))(state, metrics, config, 1.0, 3)).lower()
    assert "scan" in jaxpr
    assert "host_callback" not in jaxpr
    assert "io_callback" not in jaxpr
    assert "pure_callback" not in jaxpr
