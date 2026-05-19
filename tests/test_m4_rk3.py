from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.rk3 import rk3_scalar_decay, rk3_stage
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid


def _decay_error(n_steps: int) -> float:
    dt = 1.0 / float(n_steps)
    got = float(rk3_scalar_decay(1.0, dt, n_steps))
    return abs(got - float(np.exp(-1.0)))


def test_rk3_scalar_helper_is_third_order():
    e1 = _decay_error(20)
    e2 = _decay_error(40)
    assert np.log2(e1 / e2) > 2.8


def test_rk3_stage_preserves_state_shapes_and_dtypes():
    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    out = rk3_stage(state, tendencies, grid, 0.25)
    for before, after in zip(jnp.asarray(state.theta).shape, jnp.asarray(out.theta).shape, strict=True):
        assert before == after
    assert out.theta.dtype == state.theta.dtype


def test_rk3_stage_keeps_positive_qv_for_still_air_case():
    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    out = rk3_stage(state, tendencies, grid, 0.25)
    assert int(jnp.sum(out.qv < 0.0)) == 0
