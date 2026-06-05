from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import numpy as np
import jax.numpy as jnp

from gpuwrf.coupling.boundary_feedback import (
    StateFeedbackWeights,
    apply_feedback,
    apply_state_feedback,
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


class _DuckState:
    _FIELDS = (
        "u",
        "v",
        "w",
        "theta",
        "qv",
        "p_perturbation",
        "p_total",
        "p",
        "ph_perturbation",
        "ph_total",
        "ph",
        "mu_perturbation",
        "mu_total",
        "mu",
        "qke",
    )

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def replace(self, _cast=True, **updates):
        del _cast
        values = {name: getattr(self, name) for name in self._FIELDS}
        values.update(updates)
        return _DuckState(**values)


def _duck_state(ny: int, nx: int, *, fill: float, base: float) -> _DuckState:
    z = 2
    mass = jnp.full((z, ny, nx), fill, dtype=jnp.float64)
    mass2 = jnp.full((ny, nx), fill, dtype=jnp.float64)
    p_pert = mass
    ph_pert = mass
    mu_pert = mass2
    p_base = jnp.full_like(mass, base)
    ph_base = jnp.full_like(mass, base + 1000.0)
    mu_base = jnp.full_like(mass2, base + 2000.0)
    return _DuckState(
        u=jnp.full((z, ny, nx + 1), fill, dtype=jnp.float64),
        v=jnp.full((z, ny + 1, nx), fill, dtype=jnp.float64),
        w=mass,
        theta=mass,
        qv=mass,
        p_perturbation=p_pert,
        p_total=p_base + p_pert,
        p=p_base + p_pert,
        ph_perturbation=ph_pert,
        ph_total=ph_base + ph_pert,
        ph=ph_base + ph_pert,
        mu_perturbation=mu_pert,
        mu_total=mu_base + mu_pert,
        mu=mu_base + mu_pert,
        qke=mass,
    )


def test_state_feedback_gate_updates_parent_overlap_and_rebuilds_totals():
    weights = StateFeedbackWeights(mass=_weights(""), u=_weights("U"), v=_weights("V"))
    parent = _duck_state(12, 12, fill=0.0, base=100.0)
    child = _duck_state(24, 24, fill=9.0, base=900.0)

    off = apply_state_feedback(parent, child, weights, feedback=False)
    assert off is parent

    out = apply_state_feedback(parent, child, weights, feedback=True)
    mask = np.asarray(feedback_mask(weights.mass), dtype=bool)
    theta = np.asarray(out.theta)
    qke = np.asarray(out.qke)
    p_base = np.asarray(parent.p_total - parent.p_perturbation)

    assert np.all(theta[:, mask] == 9.0)
    assert np.all(theta[:, ~mask] == 0.0)
    assert np.all(qke[:, mask] == 9.0)
    assert np.all(np.asarray(out.p_total) == p_base + np.asarray(out.p_perturbation))
