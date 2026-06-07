"""WRF sm121 feedback-zone smoother (two-way nesting) — faithfulness tests.

These pin the JAX :func:`sm121_smooth` against an independent NumPy oracle that
re-implements ``share/interp_fcn.F::sm121`` index-for-index (serial single-tile,
``smooth_passes = 1``), and check that :func:`apply_state_feedback` runs the
copy_fcn feedback THEN the smoother over the nest-overlap region only.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import numpy as np
import jax.numpy as jnp

from gpuwrf.coupling.boundary_feedback import (
    StateFeedbackWeights,
    apply_feedback,
    apply_state_feedback,
    build_feedback_smoother,
    build_feedback_weights,
    build_state_feedback_weights,
    feedback_mask,
    sm121_smooth,
)


# --- WRF sm121 NumPy oracle (1-based -> 0-based, serial single-tile) ----------


def _sm121_oracle(
    field2d: np.ndarray,
    *,
    ratio: int,
    ipos: int,
    jpos: int,
    child_we: int,
    child_sn: int,
    stagger: str,
) -> np.ndarray:
    """Index-faithful NumPy port of WRF sm121 for one 2D (ny, nx) parent field.

    Axis 0 is the WRF ``j`` (south-north), axis 1 is the WRF ``i`` (west-east).
    """

    cfld = field2d.astype(np.float64).copy()
    n_parent_i = child_we // ratio
    n_parent_j = child_sn // ratio
    istag = 0 if stagger == "U" else 1
    jstag = 0 if stagger == "V" else 1

    # WRF 1-based inclusive bounds (serial: cits/cite collapse to domain).
    i_lo_1, i_hi_1 = ipos + 2, ipos + n_parent_i - 2 - istag
    j_lo_1, j_hi_1 = jpos + 2, jpos + n_parent_j - 2 - jstag

    # cfldnew = raw cfld (the 3-cell halo init is just a copy of cfld here).
    cfldnew = cfld.copy()

    # j-pass first (WRF: cfldnew(i,j) = 0.25*(cfld(i,j+1)+2cfld(i,j)+cfld(i,j-1))).
    for i1 in range(i_lo_1, i_hi_1 + 1):
        for j1 in range(j_lo_1, j_hi_1 + 1):
            i, j = i1 - 1, j1 - 1
            cfldnew[j, i] = 0.25 * (cfld[j + 1, i] + 2.0 * cfld[j, i] + cfld[j - 1, i])

    # i-pass last (WRF: cfld(i,j) = 0.25*(cfldnew(i+1,j)+2cfldnew(i,j)+cfldnew(i-1,j))).
    for j1 in range(j_lo_1, j_hi_1 + 1):
        for i1 in range(i_lo_1, i_hi_1 + 1):
            i, j = i1 - 1, j1 - 1
            cfld[j, i] = 0.25 * (cfldnew[j, i + 1] + 2.0 * cfldnew[j, i] + cfldnew[j, i - 1])

    return cfld


_RATIO = 3
_IPOS = 2
_JPOS = 2
_PWE = _PSN = 16
_CWE = _CSN = 24  # 8 parent cells covered


def _smoother(stagger: str):
    return build_feedback_smoother(
        parent_grid_ratio=_RATIO,
        i_parent_start=_IPOS,
        j_parent_start=_JPOS,
        child_we=_CWE,
        child_sn=_CSN,
        stagger=stagger,
    )


def _parent_shape(stagger: str) -> tuple[int, int]:
    ny = _PSN + (1 if stagger == "V" else 0)
    nx = _PWE + (1 if stagger == "U" else 0)
    return ny, nx


def test_sm121_matches_wrf_oracle_mass_2d():
    rng = np.random.default_rng(0)
    ny, nx = _parent_shape("")
    field = rng.standard_normal((ny, nx))
    got = np.asarray(sm121_smooth(jnp.asarray(field), _smoother("")))
    want = _sm121_oracle(
        field, ratio=_RATIO, ipos=_IPOS, jpos=_JPOS, child_we=_CWE, child_sn=_CSN, stagger=""
    )
    assert np.allclose(got, want, atol=1e-12), float(np.max(np.abs(got - want)))


def test_sm121_matches_wrf_oracle_mass_3d():
    rng = np.random.default_rng(1)
    ny, nx = _parent_shape("")
    field = rng.standard_normal((5, ny, nx))
    got = np.asarray(sm121_smooth(jnp.asarray(field), _smoother("")))
    want = np.stack(
        [
            _sm121_oracle(
                field[k], ratio=_RATIO, ipos=_IPOS, jpos=_JPOS,
                child_we=_CWE, child_sn=_CSN, stagger="",
            )
            for k in range(field.shape[0])
        ]
    )
    assert np.allclose(got, want, atol=1e-12), float(np.max(np.abs(got - want)))


def test_sm121_matches_wrf_oracle_u_and_v_stagger():
    rng = np.random.default_rng(2)
    for stagger in ("U", "V"):
        ny, nx = _parent_shape(stagger)
        field = rng.standard_normal((ny, nx))
        got = np.asarray(sm121_smooth(jnp.asarray(field), _smoother(stagger)))
        want = _sm121_oracle(
            field, ratio=_RATIO, ipos=_IPOS, jpos=_JPOS,
            child_we=_CWE, child_sn=_CSN, stagger=stagger,
        )
        assert np.allclose(got, want, atol=1e-12), (stagger, float(np.max(np.abs(got - want))))


def test_sm121_only_touches_interior_and_preserves_outside():
    """The smoother must mutate only the WRF interior, leaving every other cell."""
    ny, nx = _parent_shape("")
    field = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
    sm = _smoother("")
    got = np.asarray(sm121_smooth(jnp.asarray(field), sm))
    changed = ~np.isclose(got, field)
    # Every changed cell is inside [j_lo:j_hi, i_lo:i_hi].
    js, is_ = np.where(changed)
    if js.size:
        assert js.min() >= sm.j_lo and js.max() < sm.j_hi
        assert is_.min() >= sm.i_lo and is_.max() < sm.i_hi


def test_constant_field_is_smoother_invariant():
    """A 1-2-1 pass over a constant field is the identity (sum of weights = 1)."""
    ny, nx = _parent_shape("")
    field = np.full((ny, nx), 7.25, dtype=np.float64)
    got = np.asarray(sm121_smooth(jnp.asarray(field), _smoother("")))
    assert np.allclose(got, 7.25, atol=1e-12)


# --- apply_state_feedback: copy_fcn THEN sm121 over the overlap ---------------


class _DuckState:
    _FIELDS = (
        "u", "v", "w", "theta", "qv",
        "p_perturbation", "p_total", "p",
        "ph_perturbation", "ph_total", "ph",
        "mu_perturbation", "mu_total", "mu", "qke",
    )

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def replace(self, _cast=True, **updates):
        del _cast
        values = {name: getattr(self, name) for name in self._FIELDS}
        values.update(updates)
        return _DuckState(**values)


def _duck_state(ny: int, nx: int, *, fill, base: float) -> _DuckState:
    z = 2
    if callable(fill):
        mass = jnp.asarray(np.stack([fill(ny, nx) for _ in range(z)]))
        mass2 = jnp.asarray(fill(ny, nx))
        u = jnp.asarray(np.stack([fill(ny, nx + 1) for _ in range(z)]))
        v = jnp.asarray(np.stack([fill(ny + 1, nx) for _ in range(z)]))
    else:
        mass = jnp.full((z, ny, nx), fill, dtype=jnp.float64)
        mass2 = jnp.full((ny, nx), fill, dtype=jnp.float64)
        u = jnp.full((z, ny, nx + 1), fill, dtype=jnp.float64)
        v = jnp.full((z, ny + 1, nx), fill, dtype=jnp.float64)
    p_base = jnp.full_like(mass, base)
    ph_base = jnp.full_like(mass, base + 1000.0)
    mu_base = jnp.full_like(mass2, base + 2000.0)
    return _DuckState(
        u=u, v=v, w=mass, theta=mass, qv=mass,
        p_perturbation=mass, p_total=p_base + mass, p=p_base + mass,
        ph_perturbation=mass, ph_total=ph_base + mass, ph=ph_base + mass,
        mu_perturbation=mass2, mu_total=mu_base + mass2, mu=mu_base + mass2,
        qke=mass,
    )


def _edge_weights():
    common = dict(
        parent_grid_ratio=_RATIO,
        i_parent_start=_IPOS,
        j_parent_start=_JPOS,
        parent_we=_PWE,
        parent_sn=_PSN,
        child_we=_CWE,
        child_sn=_CSN,
        spec_zone=1,
    )
    return StateFeedbackWeights(
        mass=build_feedback_weights(stagger="", **common),
        u=build_feedback_weights(stagger="U", **common),
        v=build_feedback_weights(stagger="V", **common),
        mass_smooth=_smoother(""),
        u_smooth=_smoother("U"),
        v_smooth=_smoother("V"),
        smooth_option=1,
    )


def test_state_feedback_applies_copyfcn_then_sm121():
    """The fed-back parent overlap must equal copy_fcn-then-sm121, not copy_fcn alone."""
    weights = _edge_weights()
    rng = np.random.default_rng(7)
    parent = _duck_state(_PSN, _PWE, fill=0.0, base=100.0)
    child = _duck_state(_CSN, _CWE, fill=lambda a, b: rng.standard_normal((a, b)), base=900.0)

    out = apply_state_feedback(parent, child, weights, feedback=True)

    # Reference: copy_fcn feedback alone, then the smoother applied explicitly.
    fed_only = apply_feedback(parent.theta, child.theta, weights.mass, feedback=True)
    smoothed = sm121_smooth(fed_only, weights.mass_smooth)
    assert np.allclose(np.asarray(out.theta), np.asarray(smoothed), atol=1e-12)
    # And the smoother must actually have changed the copy_fcn-only result.
    assert not np.allclose(np.asarray(out.theta), np.asarray(fed_only))


def test_smooth_option_zero_disables_smoother():
    weights_off = build_state_feedback_weights(
        parent_grid_ratio=_RATIO,
        i_parent_start=_IPOS,
        j_parent_start=_JPOS,
        parent_grid=type("G", (), {"nx": _PWE, "ny": _PSN})(),
        child_grid=type("G", (), {"nx": _CWE, "ny": _CSN})(),
        smooth_option=0,
    )
    rng = np.random.default_rng(9)
    parent = _duck_state(_PSN, _PWE, fill=0.0, base=100.0)
    child = _duck_state(_CSN, _CWE, fill=lambda a, b: rng.standard_normal((a, b)), base=900.0)
    out = apply_state_feedback(parent, child, weights_off, feedback=True)
    fed_only = apply_feedback(parent.theta, child.theta, weights_off.mass, feedback=True)
    assert np.allclose(np.asarray(out.theta), np.asarray(fed_only), atol=1e-12)


def test_feedback_changes_only_overlap_region_in_parent():
    """Two-way feedback must leave the parent OUTSIDE the nest overlap untouched."""
    weights = _edge_weights()
    rng = np.random.default_rng(11)
    parent = _duck_state(_PSN, _PWE, fill=lambda a, b: rng.standard_normal((a, b)), base=100.0)
    child = _duck_state(_CSN, _CWE, fill=lambda a, b: rng.standard_normal((a, b)), base=900.0)

    out = apply_state_feedback(parent, child, weights, feedback=True)
    # The copy_fcn overlap mask bounds where the parent may change; the smoother
    # only ever writes inside that same overlap interior, so the union of changed
    # cells must stay within the overlap mask.
    mask = np.asarray(feedback_mask(weights.mass), dtype=bool)
    changed = ~np.isclose(np.asarray(out.theta), np.asarray(parent.theta))
    assert np.all(changed[:, ~mask] == False)  # noqa: E712 -- explicit boolean compare
    assert np.any(changed[:, mask])
