from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, _state_field_shapes
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
    arrays["theta"] = jnp.ones_like(arrays["theta"]) * 300.0
    arrays["mu"] = jnp.ones_like(arrays["mu"]) * 90000.0
    return State(**arrays)


def test_acoustic_substep_carries_previous_pressure_and_smdiv_effect():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _analytic_state(grid)
    previous_pressure = state.p - 1.0

    next_state, next_previous = acoustic_substep(
        state,
        previous_pressure,
        metrics,
        AcousticConfig(smdiv=SmdivConfig(enabled=True, coefficient=0.1)),
    )

    assert jnp.allclose(next_previous, state.p)
    assert not jnp.allclose(next_state.p, state.p)
    assert jnp.all(jnp.isfinite(next_state.p))


def test_outer_and_nested_scan_run_without_host_callback_primitives():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _analytic_state(grid)
    config = OrchestratorConfig(acoustic=AcousticConfig(n_substeps=2))

    carry = run_scan(state, metrics, config, 1.0, 3)

    assert carry.state.p.shape == state.p.shape
    assert carry.previous_pressure.shape == state.p.shape
    assert jnp.all(jnp.isfinite(carry.state.theta))

    jaxpr = str(jax.make_jaxpr(run_scan, static_argnums=(2, 3, 4))(state, metrics, config, 1.0, 3)).lower()
    assert "scan" in jaxpr
    assert "host_callback" not in jaxpr
    assert "io_callback" not in jaxpr
    assert "pure_callback" not in jaxpr
