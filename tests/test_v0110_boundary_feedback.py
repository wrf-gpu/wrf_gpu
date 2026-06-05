from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import numpy as np
import jax.numpy as jnp

from gpuwrf.coupling.boundary_feedback import (
    apply_feedback,
    build_feedback_weights,
    feedback_mask,
    feedback_overlap_conservation,
)


def _weights(stagger: str = ""):
    return build_feedback_weights(
        parent_grid_ratio=3,
        i_parent_start=2,
        j_parent_start=2,
        parent_we=12,
        parent_sn=12,
        child_we=24,
        child_sn=24,
        stagger=stagger,
        spec_zone=1,
    )


def test_feedback_identity_when_gate_off():
    weights = _weights()
    parent = jnp.arange(12 * 12, dtype=jnp.float64).reshape(12, 12)
    child = jnp.ones((24, 24), dtype=jnp.float64)
    out = apply_feedback(parent, child, weights, feedback=False)
    assert np.array_equal(np.asarray(out), np.asarray(parent))


def test_mass_feedback_matches_static_child_average():
    weights = _weights()
    parent = jnp.zeros((12, 12), dtype=jnp.float64)
    child = jnp.arange(24 * 24, dtype=jnp.float64).reshape(24, 24)
    out = np.asarray(apply_feedback(parent, child, weights, feedback=True)).reshape(-1)

    first_parent = int(np.asarray(weights.parent_lin)[0])
    first_donors = np.asarray(weights.child_lin)[0]
    expected = float(np.mean(np.asarray(child).reshape(-1)[first_donors]))
    assert abs(out[first_parent] - expected) < 1.0e-12


def test_feedback_conserves_overlap_integral_for_3d_field():
    weights = _weights()
    child = jnp.arange(4 * 24 * 24, dtype=jnp.float64).reshape(4, 24, 24)
    result = feedback_overlap_conservation(child, weights, leaf="theta")
    assert result.conserved is True
    assert result.rel_residual <= 1.0e-12
    assert result.n_cells == int(np.sum(np.asarray(feedback_mask(weights))))


def test_u_and_v_staggered_feedback_have_expected_stencil():
    wu = _weights("U")
    wv = _weights("V")
    assert wu.stencil == 3
    assert wv.stencil == 3
    assert wu.pwe == 13 and wu.psn == 12
    assert wv.pwe == 12 and wv.psn == 13
