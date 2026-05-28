from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.dynamics.acoustic import acoustic_once, divergence_cgrid, forward_backward_acoustic
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid


def test_cgrid_divergence_is_zero_for_still_air():
    grid = make_ideal_grid(4, 6, 6)
    state, _ = density_current_state(grid)
    div = divergence_cgrid(state, grid)
    assert float(jnp.max(jnp.abs(div))) == 0.0


def test_acoustic_once_preserves_shapes():
    grid = make_ideal_grid(4, 6, 6)
    state, _ = density_current_state(grid)
    out = acoustic_once(state, grid, 0.1)
    assert out.u.shape == state.u.shape
    assert out.v.shape == state.v.shape
    assert out.w.shape == state.w.shape
    assert out.p.shape == state.p.shape


def test_acoustic_scan_keeps_state_finite():
    grid = make_ideal_grid(4, 6, 6)
    state, _ = density_current_state(grid)
    out = forward_backward_acoustic(state, grid, 0.25, 2)
    leaves = [out.u, out.v, out.w, out.theta, out.qv, out.p, out.ph, out.mu]
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
