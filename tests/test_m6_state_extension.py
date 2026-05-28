from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import PRECISION_MATRIX
from gpuwrf.contracts.state import State


NEW_3D = ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng", "qke")
SURFACE_2D = (
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "rhosfc",
    "fltv",
    "t_skin",
    "soil_moisture",
    "xland",
    "lakemask",
    "mavail",
    "roughness_m",
    "lu_index",
    "rain_acc",
    "snow_acc",
    "graupel_acc",
    "ice_acc",
)
BOUNDARY = (
    "u_bdy",
    "v_bdy",
    "theta_bdy",
    "qv_bdy",
    "ph_bdy",
    "mu_bdy",
    "w_bdy",
    "p_bdy",
    "pb_bdy",
    "phb_bdy",
    "mub_bdy",
)


def _platform(array) -> str:
    return array.devices().copy().pop().platform


def test_m6_new_state_leaves_are_device_arrays_with_expected_shape_and_dtype():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)

    for field in NEW_3D:
        leaf = getattr(state, field)
        assert isinstance(leaf, jax.Array)
        assert _platform(leaf) == "gpu"
        assert leaf.shape == (grid.nz, grid.ny, grid.nx)
        assert leaf.dtype == PRECISION_MATRIX[field][0]

    for field in SURFACE_2D:
        leaf = getattr(state, field)
        assert isinstance(leaf, jax.Array)
        assert _platform(leaf) == "gpu"
        assert leaf.shape == (grid.ny, grid.nx)
        assert leaf.dtype == PRECISION_MATRIX[field][0]

    assert state.xland.dtype == jnp.float32
    assert state.lakemask.dtype == jnp.float32
    assert state.mavail.dtype == jnp.float32
    assert state.roughness_m.dtype == jnp.float64
    assert state.lu_index.dtype == jnp.int32

    side = max(grid.nx + 1, grid.ny + 1)
    width = 5
    assert state.u_bdy.shape == (1, 4, width, grid.nz, side)
    assert state.v_bdy.shape == (1, 4, width, grid.nz, side)
    assert state.theta_bdy.shape == (1, 4, width, grid.nz, side)
    assert state.qv_bdy.shape == (1, 4, width, grid.nz, side)
    assert state.ph_bdy.shape == (1, 4, width, grid.nz + 1, side)
    assert state.mu_bdy.shape == (1, 4, width, 1, side)
    assert state.w_bdy.shape == state.ph_bdy.shape
    assert state.p_bdy.shape == state.theta_bdy.shape
    assert state.pb_bdy.shape == state.theta_bdy.shape
    assert state.phb_bdy.shape == state.ph_bdy.shape
    assert state.mub_bdy.shape == state.mu_bdy.shape
    for field in BOUNDARY:
        leaf = getattr(state, field)
        assert isinstance(leaf, jax.Array)
        assert _platform(leaf) == "gpu"
        assert leaf.dtype == PRECISION_MATRIX[field][0]


def test_m6_existing_state_leaves_preserve_c_grid_shapes():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)

    assert state.u.shape == (grid.nz, grid.ny, grid.nx + 1)
    assert state.v.shape == (grid.nz, grid.ny + 1, grid.nx)
    assert state.w.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.theta.shape == (grid.nz, grid.ny, grid.nx)
    assert state.qv.shape == (grid.nz, grid.ny, grid.nx)
    assert state.p.shape == (grid.nz, grid.ny, grid.nx)
    assert state.p_total.shape == (grid.nz, grid.ny, grid.nx)
    assert state.p_perturbation.shape == (grid.nz, grid.ny, grid.nx)
    assert state.ph.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.ph_total.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.ph_perturbation.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.mu.shape == (grid.ny, grid.nx)
    assert state.mu_total.shape == (grid.ny, grid.nx)
    assert state.mu_perturbation.shape == (grid.ny, grid.nx)
