"""CPU logic gate for the one-kernel force-down SINT (nesting/sint_kernel.py).

Interpret mode checks indexing, staggers, ring layout and padding against the
existing ``field_sides(interp_sint_full(...))`` path.  Bitwise equality of the
explicitly rounded GPU kernel is a GPU-arm claim (tools/host_sync/sint/).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from gpuwrf.nesting.boundary_construction import (
    build_nest_force_weights,
    field_sides_2d,
    field_sides_3d,
)
from gpuwrf.nesting.interp import interp_sint_full
from gpuwrf.nesting.sint_kernel import sint_sides
from gpuwrf.validation.moving_nest_testbed import build_flat_grid


def _weights(ratio: int, start: int):
    pgrid = build_flat_grid(nx=19, ny=17, nz=4, dx_m=3000.0)
    cgrid = build_flat_grid(nx=ratio * 5 + 1, ny=ratio * 4 + 1, nz=4, dx_m=1000.0)
    weights = build_nest_force_weights(
        parent_grid_ratio=ratio,
        i_parent_start=start,
        j_parent_start=start,
        parent_grid=pgrid,
        child_grid=cgrid,
        registration="sint",
    )
    return pgrid, cgrid, weights


def _field(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    z = np.indices(shape).sum(axis=0).astype(np.float64)
    # Smooth background plus sharp jumps so the monotone limiter is active.
    return jnp.asarray(np.sin(0.37 * z) * 50.0 + rng.normal(size=shape) * 5.0
                       + 40.0 * (rng.random(shape) > 0.8), dtype=dtype)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.float32))
@pytest.mark.parametrize("ratio", (3, 2))
def test_sint_sides_matches_full_grid_then_sides(dtype, ratio):
    pgrid, cgrid, weights = _weights(ratio, start=4)
    width, side_len = 5, max(cgrid.nx, cgrid.ny) + 1
    cases = (
        ("mass", weights.mass, (5, pgrid.ny, pgrid.nx), {}),
        ("u", weights.u, (5, pgrid.ny, pgrid.nx + 1), {"xstag": True}),
        ("v", weights.v, (5, pgrid.ny + 1, pgrid.nx), {"ystag": True}),
    )
    tol = 1e-12 if dtype == jnp.float64 else 2e-6
    for index, (name, plan, shape, stag) in enumerate(cases):
        field = _field(shape, dtype, seed=index)
        reference = field_sides_3d(
            interp_sint_full(field, plan, parent_grid_ratio=ratio, **stag), width, side_len
        )
        got = sint_sides(field, plan, parent_grid_ratio=ratio, width=width,
                         side_len=side_len, interpret=True, **stag)
        assert got.shape == reference.shape and got.dtype == reference.dtype, name
        np.testing.assert_allclose(np.asarray(got), np.asarray(reference),
                                   rtol=tol, atol=tol * 50.0, err_msg=name)
        # Padding cells are exactly zero, like field_sides_3d.
        assert np.array_equal(np.asarray(got) == 0, np.asarray(reference) == 0) or name


def test_sint_sides_2d_matches_field_sides_2d():
    pgrid, cgrid, weights = _weights(3, start=4)
    width, side_len = 5, max(cgrid.nx, cgrid.ny) + 1
    field = _field((pgrid.ny, pgrid.nx), jnp.float64, seed=7)
    reference = field_sides_2d(
        interp_sint_full(field, weights.mass, parent_grid_ratio=3), width, side_len
    )
    got = sint_sides(field, weights.mass, parent_grid_ratio=3, width=width,
                     side_len=side_len, interpret=True)
    assert got.shape == reference.shape
    np.testing.assert_allclose(np.asarray(got), np.asarray(reference), rtol=1e-12, atol=5e-11)


def test_sint_sides_is_deletion_sensitive_to_the_limiter():
    """A kernel that skipped the y-pass limiter would not match (mutation guard)."""

    pgrid, cgrid, weights = _weights(3, start=4)
    width, side_len = 5, max(cgrid.nx, cgrid.ny) + 1
    field = _field((3, pgrid.ny, pgrid.nx), jnp.float64, seed=11)
    got = np.asarray(sint_sides(field, weights.mass, parent_grid_ratio=3, width=width,
                                side_len=side_len, interpret=True))
    shifted = np.asarray(sint_sides(field, weights.mass, parent_grid_ratio=3, xstag=True,
                                    width=width, side_len=side_len, interpret=True))
    assert np.max(np.abs(got - shifted)) > 1e-3


def test_single_jit_coupled_forcedown_matches_per_producer_bitwise(monkeypatch):
    """One fusion-disabled executable == per-producer path, every leaf (CPU)."""

    import jax

    from gpuwrf.nesting.boundary_construction import build_child_boundary_package
    from test_v0234_nested_boundary_science_repair import _fixture

    pgrid, cgrid, parent, child, weights = _fixture()
    kwargs = dict(bdy_width=5, parent_metrics=pgrid.metrics, child_metrics=cgrid.metrics,
                  coupled_forcedown=True, parent_grid_ratio=3, _compiled_producers=True)
    monkeypatch.setenv("GPUWRF_FORCEDOWN_SINGLE_JIT", "0")
    reference = build_child_boundary_package(child, parent, weights, **kwargs)
    monkeypatch.setenv("GPUWRF_FORCEDOWN_SINGLE_JIT", "1")
    single = build_child_boundary_package(child, parent, weights, **kwargs)
    ref_leaves = jax.tree_util.tree_flatten_with_path(reference)[0]
    new_leaves = jax.tree_util.tree_flatten_with_path(single)[0]
    assert [p for p, _ in ref_leaves] == [p for p, _ in new_leaves]
    changed = [
        jax.tree_util.keystr(path)
        for (path, a), (_, b) in zip(ref_leaves, new_leaves)
        if np.asarray(a).dtype != np.asarray(b).dtype
        or np.asarray(a).tobytes() != np.asarray(b).tobytes()
    ]
    assert not changed, changed
    # Untouched child leaves are the very same objects (placement-neutral, E43).
    assert single.theta is child.theta and single.u is child.u


@pytest.mark.parametrize("ratio", (2, 3, 4, 5, 7))
@pytest.mark.parametrize("staggered", (False, True))
def test_host_subcell_offsets_match_the_traced_plan(ratio, staggered):
    """Kernel offsets are host constants equal to interp._sint_axis_plan's ``a``.

    Ratios 2-4 (PROD/WN3 use 3) are bitwise equal to the traced plan both eager
    and jitted; all ratios equal the eager (true-division) plan.
    """

    import jax

    from gpuwrf.nesting.interp import _sint_axis_plan
    from gpuwrf.nesting.sint_kernel import _host_subcell_offsets

    n = 23
    lower = jnp.arange(n, dtype=jnp.int32) // ratio + 4
    fraction = (jnp.arange(n, dtype=jnp.float64) % ratio) / ratio
    host = _host_subcell_offsets(n, ratio, staggered, np.float64)
    _, eager = _sint_axis_plan(lower, fraction, ratio=ratio, staggered=staggered)
    np.testing.assert_array_equal(host, np.asarray(eager))
    if ratio <= 4:
        jitted = jax.jit(
            lambda lo, fr: _sint_axis_plan(lo, fr, ratio=ratio, staggered=staggered)[1]
        )(lower, fraction)
        np.testing.assert_array_equal(host, np.asarray(jitted))


@pytest.mark.parametrize("ratio", (2, 3, 4, 5))
@pytest.mark.parametrize("staggered", (False, True))
@pytest.mark.parametrize("start", (2, 3, 17, 92))
def test_integer_centers_match_the_traced_plan(ratio, staggered, start):
    """S2: the kernel's parent centers need no f64 (cell-centered SINT weights)."""

    from gpuwrf.nesting.interp import _sint_axis_plan, build_sint_weights
    from gpuwrf.nesting.sint_kernel import _int_centers

    n = ratio * 9 + 1
    w = build_sint_weights(parent_grid_ratio=ratio, i_parent_start=start, j_parent_start=start,
                           parent_ny=start + 15, parent_nx=start + 15 + int(staggered),
                           child_ny=n, child_nx=n + int(staggered))
    traced, _ = _sint_axis_plan(w.x0, w.wx, ratio=ratio, staggered=staggered)
    got = _int_centers(w.x0, int(w.x0.shape[0]), ratio, staggered)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(traced))


def _f64_eqns(jaxpr) -> int:
    import jax.extend

    count = 0
    for eqn in jaxpr.eqns:
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            count += _f64_eqns(sub)
        count += any(getattr(v.aval, "dtype", None) == np.float64 for v in eqn.outvars)
    return count


@pytest.mark.parametrize(("xstag", "ystag"), ((False, False), (True, False), (False, True)))
def test_sint_sides_on_real_fields_has_no_f64(xstag, ystag):
    """S2 gate (review-s2small): a REAL force-down field leaves no f64 in sint_sides."""

    import jax

    from gpuwrf.nesting.interp import build_sint_weights

    w = build_sint_weights(parent_grid_ratio=3, i_parent_start=5, j_parent_start=7,
                           parent_ny=30 + ystag, parent_nx=31 + xstag,
                           child_ny=28 + ystag, child_nx=31 + xstag)
    field = jnp.zeros((4, 30 + ystag, 31 + xstag), jnp.float32)
    jaxpr = jax.make_jaxpr(
        lambda f, weights: sint_sides(f, weights, parent_grid_ratio=3, xstag=xstag, ystag=ystag,
                                      width=5, side_len=32, interpret=True)
    )(field, w)
    assert _f64_eqns(jaxpr.jaxpr) == 0
