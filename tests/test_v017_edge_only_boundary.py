"""v0.17 edge-only (ring-only) boundary interpolation -- CPU bit-identity gates.

The child-boundary forcing used to interpolate the FULL child grid
(``_interp``) and then slice the width-N W/E/S/N ring strips
(``field_sides_3d``/``_2d``).  The edge-only path
(:func:`gpuwrf.nesting.boundary_construction.field_sides_3d_edgeonly`) gathers ONLY
the ring cells by subsetting the precomputed parent->child weights to the ring
rows/cols and re-running the IDENTICAL per-cell bilinear gather.

Because that gather is independent per output cell, the edge-only strips MUST be
bit-for-bit identical to the full-grid->slice strips -- only the wasted interior
interp work is removed.  These tests assert exactly that (``np.array_equal``):

  1. the low-level side builders (3-D + 2-D, both registrations, mass/u/v
     staggerings) equal ``field_sides_*(_interp(...))`` byte-for-byte;
  2. the same holds under ``jax.jit`` (surfaces any XLA FMA/contraction
     difference for the smaller array -- on CPU there is none);
  3. the FULL boundary package (every ``*_bdy`` two-time leaf) from
     ``build_child_boundary_package`` is byte-identical with edge-only ON vs OFF,
     including the padding regime where ``width >= child_dim``;
  4. the env gate selects edge-only by default and the OFF path reproduces the
     reference output.

GPU-free (CPU JAX, fp64).  The parent runs the GPU A/B bit-compare separately.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.nesting import interp as I
from gpuwrf.nesting import boundary_construction as BC


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_p0_1a_nesting.py).
# ---------------------------------------------------------------------------


class _DuckState:
    """Minimal duck-typed state exposing only what the package builder reads."""

    _FIELDS = ("u", "v", "w", "theta", "qv", "p_perturbation", "p_total",
               "ph_perturbation", "ph_total", "mu_perturbation", "mu_total")
    _BDY = ("u_bdy", "v_bdy", "w_bdy", "theta_bdy", "qv_bdy", "ph_bdy", "phb_bdy",
            "p_bdy", "pb_bdy", "mu_bdy", "mub_bdy")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def replace(self, **updates):
        d = {f: getattr(self, f) for f in self._FIELDS}
        d.update({b: getattr(self, b) for b in self._BDY})
        d.update(updates)
        return _DuckState(**d)


class _Grid:
    def __init__(self, ny, nx):
        self.ny = ny
        self.nx = nx


def _smooth_field(nz, ny, nx, seed=0):
    """A smooth, non-constant 3-D field (varies in z/y/x) for a real bit-compare."""

    rng = np.random.default_rng(seed)
    z = np.arange(nz)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    base = (
        10.0
        + 0.7 * x
        + 0.5 * y
        + 0.3 * z
        + 4.0 * np.sin(0.21 * x + 0.13 * y)
        + 2.0 * np.cos(0.17 * y - 0.09 * z)
    )
    base = base + 0.05 * rng.standard_normal((nz, ny, nx))
    return jnp.asarray(base, dtype=jnp.float64)


def _make_state(nz, ny, nx, width, side_len, *, seed=0):
    """A child/parent-like duck state with smooth, distinct prognostic fields."""

    def f3(shape, s):
        return _smooth_field(*shape, seed=s)

    theta = f3((nz, ny, nx), seed + 1)
    qv = f3((nz, ny, nx), seed + 2)
    w = f3((nz + 1, ny, nx), seed + 3)
    p_pert = f3((nz, ny, nx), seed + 4)
    p_tot = p_pert + f3((nz, ny, nx), seed + 40)
    ph_pert = f3((nz + 1, ny, nx), seed + 5)
    ph_tot = ph_pert + f3((nz + 1, ny, nx), seed + 50)
    u = f3((nz, ny, nx + 1), seed + 6)
    v = f3((nz, ny + 1, nx), seed + 7)
    mu_pert = _smooth_field(1, ny, nx, seed=seed + 8)[0]
    mu_tot = mu_pert + _smooth_field(1, ny, nx, seed=seed + 80)[0]

    bdy3 = jnp.zeros((1, 4, width, nz, side_len))
    bdy3f = jnp.zeros((1, 4, width, nz + 1, side_len))
    bdy2 = jnp.zeros((1, 4, width, 1, side_len))
    return _DuckState(
        u=u, v=v, w=w, theta=theta, qv=qv,
        p_perturbation=p_pert, p_total=p_tot,
        ph_perturbation=ph_pert, ph_total=ph_tot,
        mu_perturbation=mu_pert, mu_total=mu_tot,
        u_bdy=jnp.zeros((1, 4, width, nz, side_len)),
        v_bdy=jnp.zeros((1, 4, width, nz, side_len)),
        w_bdy=bdy3f, theta_bdy=bdy3, qv_bdy=bdy3,
        ph_bdy=bdy3f, phb_bdy=bdy3f,
        p_bdy=bdy3, pb_bdy=bdy3, mu_bdy=bdy2, mub_bdy=bdy2,
    )


_GEOM = dict(ratio=3, ips=5, jps=4)


def _weights(pny, pnx, cny, cnx, registration):
    return BC.build_nest_force_weights(
        parent_grid_ratio=_GEOM["ratio"],
        i_parent_start=_GEOM["ips"],
        j_parent_start=_GEOM["jps"],
        parent_grid=_Grid(pny, pnx),
        child_grid=_Grid(cny, cnx),
        registration=registration,
    )


# ---------------------------------------------------------------------------
# 1. low-level side builders: edge-only == full-grid->slice, byte for byte.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("registration", ["sint", "bilinear"])
@pytest.mark.parametrize(
    "stagger", ["mass", "u", "v"]
)
def test_field_sides_3d_edgeonly_bit_identical(registration, stagger):
    pny, pnx = 40, 44
    cny, cnx = 15, 18
    width = 5
    side_len = max(cny, cnx) + 2
    nz = 7
    nfw = _weights(pny, pnx, cny, cnx, registration)
    w = {"mass": nfw.mass, "u": nfw.u, "v": nfw.v}[stagger]
    # parent extent matches the staggering the weights were built for.
    pny_s = pny + (1 if stagger == "v" else 0)
    pnx_s = pnx + (1 if stagger == "u" else 0)
    parent = _smooth_field(nz, pny_s, pnx_s, seed=11)

    full_child = BC._interp(parent, w, registration)
    ref = np.asarray(BC.field_sides_3d(full_child, width, side_len))
    edge = np.asarray(
        BC.field_sides_3d_edgeonly(parent, w, registration, width, side_len)
    )
    assert edge.shape == ref.shape
    assert np.array_equal(edge, ref), (
        f"edge-only 3-D strips differ ({registration}/{stagger}); "
        f"max|d|={np.max(np.abs(edge - ref))}"
    )


@pytest.mark.parametrize("registration", ["sint", "bilinear"])
def test_field_sides_2d_edgeonly_bit_identical(registration):
    pny, pnx = 38, 41
    cny, cnx = 14, 16
    width = 5
    side_len = max(cny, cnx) + 3
    nfw = _weights(pny, pnx, cny, cnx, registration)
    parent = _smooth_field(1, pny, pnx, seed=21)[0]

    full_child = BC._interp(parent, nfw.mass, registration)
    ref = np.asarray(BC.field_sides_2d(full_child, width, side_len))
    edge = np.asarray(
        BC.field_sides_2d_edgeonly(parent, nfw.mass, registration, width, side_len)
    )
    assert edge.shape == ref.shape
    assert np.array_equal(edge, ref), (
        f"edge-only 2-D strips differ ({registration}); "
        f"max|d|={np.max(np.abs(edge - ref))}"
    )


# ---------------------------------------------------------------------------
# 2. under jax.jit (surfaces any XLA FMA/contraction difference).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("registration", ["sint", "bilinear"])
def test_edgeonly_bit_identical_under_jit(registration):
    pny, pnx = 40, 40
    cny, cnx = 15, 15
    width = 5
    side_len = max(cny, cnx) + 1
    nz = 6
    nfw = _weights(pny, pnx, cny, cnx, registration)
    parent = _smooth_field(nz, pny, pnx, seed=31)

    def full(p, weights):
        return BC.field_sides_3d(BC._interp(p, weights, registration), width, side_len)

    def edge(p, weights):
        return BC.field_sides_3d_edgeonly(p, weights, registration, width, side_len)

    full_j = jax.jit(full, static_argnums=())
    edge_j = jax.jit(edge, static_argnums=())
    ref = np.asarray(full_j(parent, nfw.mass))
    got = np.asarray(edge_j(parent, nfw.mass))
    assert np.array_equal(got, ref), (
        f"edge-only differs from full under jit ({registration}); "
        f"max|d|={np.max(np.abs(got - ref))}"
    )


# ---------------------------------------------------------------------------
# 3. FULL boundary package: every *_bdy leaf byte-identical, edge-only ON vs OFF.
# ---------------------------------------------------------------------------


_BDY_LEAVES = (
    "u_bdy", "v_bdy", "w_bdy", "theta_bdy", "qv_bdy",
    "ph_bdy", "phb_bdy", "p_bdy", "pb_bdy", "mu_bdy", "mub_bdy",
)


def _build_package_both(pny, pnx, cny, cnx, width, registration, monkeypatch):
    nz = 8
    side_len = max(cny, cnx) + 2
    nfw = _weights(pny, pnx, cny, cnx, registration)
    parent = _make_state(nz, pny, pnx, width, side_len, seed=100)
    child = _make_state(nz, cny, cnx, width, side_len, seed=200)

    monkeypatch.setenv("GPUWRF_EDGE_ONLY_BOUNDARY", "0")
    ref = BC.build_child_boundary_package(child, parent, nfw, bdy_width=width)
    monkeypatch.setenv("GPUWRF_EDGE_ONLY_BOUNDARY", "1")
    edge = BC.build_child_boundary_package(child, parent, nfw, bdy_width=width)
    return ref, edge


@pytest.mark.parametrize("registration", ["sint", "bilinear"])
def test_full_package_bit_identical(registration, monkeypatch):
    ref, edge = _build_package_both(40, 44, 15, 18, 5, registration, monkeypatch)
    for leaf in _BDY_LEAVES:
        a = np.asarray(getattr(ref, leaf))
        b = np.asarray(getattr(edge, leaf))
        assert a.shape == b.shape, f"{leaf} shape mismatch"
        assert np.array_equal(a, b), (
            f"{leaf} edge-only differs ({registration}); max|d|={np.max(np.abs(a - b))}"
        )


def test_full_package_bit_identical_width_exceeds_child(monkeypatch):
    """Padding regime: child smaller than the ring width (width=5, child=4/3)."""

    ref, edge = _build_package_both(40, 40, 4, 3, 5, "sint", monkeypatch)
    for leaf in _BDY_LEAVES:
        a = np.asarray(getattr(ref, leaf))
        b = np.asarray(getattr(edge, leaf))
        assert np.array_equal(a, b), f"{leaf} differs in padding regime"


# ---------------------------------------------------------------------------
# 4. env gate behaviour.
# ---------------------------------------------------------------------------


def test_edge_only_default_on(monkeypatch):
    monkeypatch.delenv("GPUWRF_EDGE_ONLY_BOUNDARY", raising=False)
    assert BC._edge_only_enabled() is True


@pytest.mark.parametrize("val,expected", [
    ("0", False), ("false", False), ("OFF", False), ("no", False),
    ("1", True), ("true", True), ("on", True), ("", True),
])
def test_edge_only_env_gate(monkeypatch, val, expected):
    monkeypatch.setenv("GPUWRF_EDGE_ONLY_BOUNDARY", val)
    assert BC._edge_only_enabled() is expected


# ---------------------------------------------------------------------------
# 5. sanity: edge-only really restricts the gather (not a no-op alias).
# ---------------------------------------------------------------------------


def test_edgeonly_strips_are_actual_ring_values():
    """The edge-only W/E/S/N strips equal the corresponding full-child-grid cells."""

    pny, pnx = 40, 40
    cny, cnx = 16, 16
    width = 5
    side_len = max(cny, cnx)
    nz = 5
    nfw = _weights(pny, pnx, cny, cnx, "sint")
    parent = _smooth_field(nz, pny, pnx, seed=7)
    full_child = np.asarray(BC._interp(parent, nfw.mass, "sint"))
    edge = np.asarray(
        BC.field_sides_3d_edgeonly(parent, nfw.mass, "sint", width, side_len)
    )
    # West outer strip (index 0) == child column 0 (all rows, level 0).
    assert np.array_equal(edge[0, 0, 0, :cny], full_child[0, :, 0])
    # East outer strip (index 0) == child LAST column (flipped so outer == edge).
    assert np.array_equal(edge[1, 0, 0, :cny], full_child[0, :, cnx - 1])
    # South outer strip (index 0) == child row 0.
    assert np.array_equal(edge[2, 0, 0, :cnx], full_child[0, 0, :])
    # North outer strip (index 0) == child LAST row.
    assert np.array_equal(edge[3, 0, 0, :cnx], full_child[0, cny - 1, :])
