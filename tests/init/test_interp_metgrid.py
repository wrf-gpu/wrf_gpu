"""S3 interp-kernel oracle tests.

Grades the JAX port (``gpuwrf.init.interp_metgrid``) against the REAL WPS
``interp_module.F`` kernels, compiled into ``proofs/v030/s3_oracle/liboracle.so``
(built on demand by this test if missing). Sprint AC: each kernel reproduces its
``interp_module.F`` counterpart to ``<= 1e-6`` rel on a unit-test grid with a
known analytic field; the ``+``-chain dispatcher matches ``interp_sequence``
fall-through; masking (water/both) + ``search`` match.

The residual is the Fortran single-precision (``real``) rounding vs the JAX fp64
math; it sits at ~1e-7 rel, comfortably under 1e-6.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

# Keep JAX quiet + deterministic on CPU for the oracle compare.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp  # noqa: E402

from gpuwrf.init import interp_metgrid as im  # noqa: E402

_HERE = os.path.dirname(__file__)
_ORACLE_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "proofs", "v030", "s3_oracle"))
_LIB = os.path.join(_ORACLE_DIR, "liboracle.so")

# Sprint-predeclared kernel tolerance vs interp_module.F.
KERNEL_REL_TOL = 1e-6


def _ensure_oracle():
    if not os.path.exists(_LIB):
        subprocess.run(["bash", os.path.join(_ORACLE_DIR, "build.sh")], check=True)
    if _ORACLE_DIR not in sys.path:
        sys.path.insert(0, _ORACLE_DIR)


@pytest.fixture(scope="module")
def oracle():
    _ensure_oracle()
    from oracle import Oracle  # imported after path insert

    return Oracle()


def _relmax(jax_vals, ora_vals, *, only_finite=True):
    j = np.asarray(jax_vals, dtype=np.float64)
    o = np.asarray(ora_vals, dtype=np.float64)
    if only_finite:
        m = (np.abs(o) < 1e29) & (np.abs(j) < 1e29)
    else:
        m = np.ones_like(o, dtype=bool)
    if m.sum() == 0:
        return 0.0
    return float(np.max(np.abs(j[m] - o[m]) / (np.abs(o[m]) + 1e-9)))


# --- analytic source fields --------------------------------------------------
def _linear_field(nx, ny):
    ii = np.arange(1, nx + 1)[:, None]
    jj = np.arange(1, ny + 1)[None, :]
    return (3.0 + 0.7 * ii + 0.4 * jj).astype(np.float64) * np.ones((nx, ny))


def _smooth_field(nx, ny):
    ii = np.arange(1, nx + 1)[:, None]
    jj = np.arange(1, ny + 1)[None, :]
    return (
        290.0
        + 5.0 * np.sin(0.15 * ii)
        + 3.0 * np.cos(0.2 * jj)
        + 0.05 * ii * jj
    ).astype(np.float64) * np.ones((nx, ny))


def _interior_points(nx, ny, n, seed):
    rng = np.random.default_rng(seed)
    rx = rng.uniform(3.0, nx - 3.0, n)
    ry = rng.uniform(3.0, ny - 3.0, n)
    return rx, ry


# --- the individual-kernel oracle gate ---------------------------------------
SINGLE_METHODS = {
    "nearest_neighbor": [(im.N_NEIGHBOR, 0)],
    "four_pt": [(im.FOUR_POINT, 0)],
    "sixteen_pt": [(im.SIXTEEN_POINT, 0)],
    "four_pt_average": [(im.AVERAGE4, 0)],
    "sixteen_pt_average": [(im.AVERAGE16, 0)],
    "wt_four_pt_average": [(im.W_AVERAGE4, 0)],
    "wt_sixteen_pt_average": [(im.W_AVERAGE16, 0)],
}


@pytest.mark.parametrize("method_name", list(SINGLE_METHODS))
@pytest.mark.parametrize("field_name", ["linear", "smooth"])
def test_single_kernel_vs_fortran(oracle, method_name, field_name):
    nx, ny = 28, 22
    slab = _linear_field(nx, ny) if field_name == "linear" else _smooth_field(nx, ny)
    rx, ry = _interior_points(nx, ny, 400, seed=hash(method_name) % 1000)
    chain = SINGLE_METHODS[method_name]
    o = oracle.interp(slab, rx, ry, chain)
    j = im.interp_sequence(jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab), chain)
    rel = _relmax(j, o)
    assert rel <= KERNEL_REL_TOL, f"{method_name}/{field_name} rel={rel:.3e}"


def test_oned_vs_fortran(oracle):
    """The 1-D overlapping parabola itself, incl the a/d==0 sub-branches."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, 500)
    a = rng.uniform(1.0, 5.0, 500)
    b = rng.uniform(1.0, 5.0, 500)
    c = rng.uniform(1.0, 5.0, 500)
    d = rng.uniform(1.0, 5.0, 500)
    # force some a==0 / d==0 / both branches
    a[:100] = 0.0
    d[100:200] = 0.0
    a[200:250] = 0.0
    d[200:250] = 0.0
    o = oracle.oned(x, a, b, c, d)
    j = np.asarray(im.oned(jnp.asarray(x), jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)))
    rel = _relmax(j, o, only_finite=False)
    assert rel <= KERNEL_REL_TOL, f"oned rel={rel:.3e}"


def test_oned_x_endpoints(oracle):
    """x==0 -> b, x==1 -> c exact-endpoint branch."""
    x = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    a = np.array([2.0, 2.0, 0.0, 0.0], dtype=float)
    b = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)
    c = np.array([9.0, 9.0, 9.0, 9.0], dtype=float)
    d = np.array([3.0, 3.0, 0.0, 0.0], dtype=float)
    o = oracle.oned(x, a, b, c, d)
    j = np.asarray(im.oned(jnp.asarray(x), jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)))
    assert _relmax(j, o, only_finite=False) <= KERNEL_REL_TOL


# --- the +-chain dispatcher fall-through -------------------------------------
def test_chain_dispatcher_matches_sequence(oracle):
    """The TT/UU/VV/GHT/SPECHUMD chain sixteen_pt+four_pt+average_4pt."""
    nx, ny = 30, 24
    slab = _smooth_field(nx, ny)
    rx, ry = _interior_points(nx, ny, 500, seed=11)
    chain = im.parse_interp_string("sixteen_pt+four_pt+average_4pt")
    assert chain == [(im.SIXTEEN_POINT, 0), (im.FOUR_POINT, 0), (im.AVERAGE4, 0)]
    o = oracle.interp(slab, rx, ry, chain)
    j = im.interp_sequence(jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab), chain)
    assert _relmax(j, o) <= KERNEL_REL_TOL


def test_chain_falls_back_at_boundary(oracle):
    """Near the array edge sixteen_pt declines (needs +-2 cells) and four_pt
    takes over -- the exact interp_sequence fall-through. We probe target points
    in the 1.5..2.5 band where sixteen_pt's far-enough test (ifx>=2 & <=nx-2)
    fails but four_pt still works."""
    nx, ny = 20, 18
    slab = _smooth_field(nx, ny)
    # rx in [1.2, 1.9] -> floor=1 -> sixteen_pt far_enough fails (needs ifx>=2)
    rng = np.random.default_rng(3)
    rx = rng.uniform(1.2, 1.9, 120)
    ry = rng.uniform(4.0, 12.0, 120)
    chain = im.parse_interp_string("sixteen_pt+four_pt+average_4pt")
    o = oracle.interp(slab, rx, ry, chain)
    j = im.interp_sequence(jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab), chain)
    assert _relmax(j, o) <= KERNEL_REL_TOL


def test_parse_interp_string_search_depth():
    assert im.parse_interp_string("search") == [(im.SEARCH, im.DEFAULT_SEARCH_DEPTH)]
    assert im.parse_interp_string("search(50)") == [(im.SEARCH, 50)]
    soil = im.parse_interp_string("sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search")
    assert soil == [
        (im.SIXTEEN_POINT, 0),
        (im.FOUR_POINT, 0),
        (im.W_AVERAGE4, 0),
        (im.W_AVERAGE16, 0),
        (im.SEARCH, im.DEFAULT_SEARCH_DEPTH),
    ]
    # unrecognized tokens skipped
    assert im.parse_interp_string("average_gcell+four_pt") == [(im.FOUR_POINT, 0)]


# --- masking (soil water-mask) + search --------------------------------------
def test_masked_soil_chain_vs_fortran(oracle):
    """ST/SM policy: sixteen_pt+four_pt+wt4+wt16+search with interp_mask=LANDSEA(0),
    equality exclusion (water source cells excluded). Coastal coverage."""
    nx, ny = 26, 22
    ii = np.arange(1, nx + 1)[:, None]
    jj = np.arange(1, ny + 1)[None, :]
    slab = (281.0 + 0.4 * ii + 0.25 * jj) * np.ones((nx, ny))
    landsea = np.where(ii <= 9, 0.0, 1.0) * np.ones((nx, ny))
    msg = -1.0e30
    slab_m = slab.copy()
    slab_m[landsea == 0] = msg
    chain = im.parse_interp_string(
        "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search"
    )
    rx, ry = _interior_points(nx, ny, 400, seed=21)
    o = oracle.interp(slab_m, rx, ry, chain, msgval=msg, mask_array=landsea, maskval=0.0, mask_relational=" ")
    j = im.interp_sequence(
        jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab_m), chain,
        msgval=msg, mask_array=jnp.asarray(landsea), maskval=0.0, mask_relational=" ",
    )
    o = np.asarray(o)
    j = np.asarray(j)
    # finite/missing classification must agree everywhere
    fin_o = np.abs(o) < 1e29
    fin_j = np.abs(j) < 1e29
    assert np.array_equal(fin_o, fin_j), "finite/missing mask disagreement"
    assert _relmax(j, o) <= KERNEL_REL_TOL


def test_pure_search_island_in_water(oracle):
    """Force the search branch: a land patch surrounded by water; deep-water
    targets where 16pt/4pt/wt all decline so search must find the nearest land
    donor. Validates the BFS nearest-usable tie-break."""
    nx, ny = 32, 32
    msg = -1.0e30
    slab = np.full((nx, ny), msg)
    landsea = np.zeros((nx, ny))
    for i in range(20, 24):
        for j in range(20, 24):
            landsea[i, j] = 1.0
            slab[i, j] = 300.0 + i + j
    chain = im.parse_interp_string(
        "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search"
    )
    rx = np.array([10.3, 12.0, 8.7, 15.2, 18.9, 21.5, 5.0, 27.0], dtype=float)
    ry = np.array([10.6, 9.0, 12.2, 14.0, 17.1, 21.4, 5.0, 27.0], dtype=float)
    o = np.asarray(
        oracle.interp(slab, rx, ry, chain, msgval=msg, mask_array=landsea, maskval=0.0, mask_relational=" ")
    )
    j = np.asarray(
        im.interp_sequence(
            jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab), chain,
            msgval=msg, mask_array=jnp.asarray(landsea), maskval=0.0, mask_relational=" ",
        )
    )
    assert _relmax(j, o) <= KERNEL_REL_TOL


def test_nearest_neighbor_landsea_categorical(oracle):
    """LANDSEA uses nearest_neighbor; a 0/1 field must round-trip exactly."""
    nx, ny = 20, 16
    ii = np.arange(1, nx + 1)[:, None]
    slab = np.where(ii <= 10, 0.0, 1.0) * np.ones((nx, ny))
    rx, ry = _interior_points(nx, ny, 300, seed=31)
    chain = [(im.N_NEIGHBOR, 0)]
    o = np.asarray(oracle.interp(slab, rx, ry, chain))
    j = np.asarray(im.interp_sequence(jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab), chain))
    assert np.array_equal(o, j)


# --- source lat/lon -> fractional index mapping ------------------------------
def test_latlon_to_source_xy_aifs():
    """The regular_ll lltoxy for the AIFS grid. lon0=0, dlon=0.25, lat0=90,
    dlat=-0.25, nx=1440, ny=721 (i=lon, j=lat)."""
    grid = im.LatLonSourceGrid(
        lon0_deg=0.0, dlon_deg=0.25, lat0_deg=90.0, dlat_deg=-0.25, nx=1440, ny=721
    )
    # equator, 0E -> ry at lat 0 = (0-90)/-0.25 + 1 = 361 ; rx at 0E = 1
    rx, ry = im.latlon_to_source_xy(np.array([0.0]), np.array([0.0]), grid)
    assert np.isclose(float(rx[0]), 1.0)
    assert np.isclose(float(ry[0]), 361.0)
    # 90N pole -> ry = 1
    rx, ry = im.latlon_to_source_xy(np.array([90.0]), np.array([10.0]), grid)
    assert np.isclose(float(ry[0]), 1.0)
    assert np.isclose(float(rx[0]), 10.0 / 0.25 + 1.0)
    # negative lon wraps: -16.4E -> 343.6E -> rx = 343.6/0.25 + 1
    rx, ry = im.latlon_to_source_xy(np.array([28.0]), np.array([-16.4]), grid)
    assert np.isclose(float(rx[0]), (360.0 - 16.4) / 0.25 + 1.0)


def test_interp_field_to_grid_shape_and_value():
    """End-to-end per-slab helper: a constant source field interpolates to the
    same constant on every target point (sanity), correct output shape."""
    grid = im.LatLonSourceGrid(
        lon0_deg=0.0, dlon_deg=0.25, lat0_deg=90.0, dlat_deg=-0.25, nx=1440, ny=721
    )
    src = np.full((1440, 721), 273.15)
    ny_t, nx_t = 5, 7
    tlat = np.linspace(28.0, 29.0, ny_t)[:, None] * np.ones((ny_t, nx_t))
    tlon = np.linspace(-17.0, -16.0, nx_t)[None, :] * np.ones((ny_t, nx_t))
    chain = im.parse_interp_string("sixteen_pt+four_pt+average_4pt")
    out = np.asarray(im.interp_field_to_grid(jnp.asarray(src), tlat, tlon, grid, chain))
    assert out.shape == (ny_t, nx_t)
    assert np.allclose(out, 273.15, atol=1e-6)
