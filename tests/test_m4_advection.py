from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.advection import advect_mass_scalar, derivative3_upwind, derivative5_upwind, fixture_reference_update, mass_face_velocities
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid


def _sine_error(n: int) -> float:
    x = jnp.arange(n, dtype=jnp.float64) / float(n)
    field = jnp.sin(2.0 * jnp.pi * x)
    velocity = field * 0.0 + 1.0
    got = derivative5_upwind(field, velocity, 1.0 / float(n), axis=0)
    expect = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)
    return float(jnp.max(jnp.abs(got - expect)))


def test_fifth_order_horizontal_derivative_converges_on_periodic_sine():
    coarse = _sine_error(64)
    fine = _sine_error(128)
    assert np.log2(coarse / fine) > 4.5


def test_third_order_vertical_derivative_preserves_constant_field():
    field = jnp.ones((8,), dtype=jnp.float64)
    velocity = field
    got = derivative3_upwind(field, velocity, 100.0, axis=0)
    assert float(jnp.max(jnp.abs(got))) == 0.0


def test_mass_scalar_advection_is_conservative_for_constant_velocity():
    grid = make_ideal_grid(6, 8, 8)
    state, _ = density_current_state(grid)
    u_mass, v_mass, w_mass = mass_face_velocities(state)
    tendency = advect_mass_scalar(state.theta, u_mass, v_mass, w_mass, grid)
    assert abs(float(jnp.sum(tendency))) < 1.0e-10


def test_fixture_reference_wrapper_matches_committed_phi_next():
    with np.load("fixtures/samples/analytic-stencil-3d-advdiff-v1.npz", allow_pickle=False) as loaded:
        got = np.asarray(
            fixture_reference_update(
                jnp.asarray(loaded["phi_initial"]),
                jnp.asarray(loaded["u_face"], dtype=jnp.float64),
                jnp.asarray(loaded["v_face"], dtype=jnp.float64),
                jnp.asarray(loaded["w_face"], dtype=jnp.float64),
                3.0,
            )
        )
        assert np.max(np.abs(got - loaded["phi_next"])) <= 1.0e-10
