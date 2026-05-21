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
    "rain_acc",
    "snow_acc",
    "graupel_acc",
    "ice_acc",
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
        assert leaf.dtype == jnp.float64


def test_m6_existing_state_leaves_preserve_c_grid_shapes():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)

    assert state.u.shape == (grid.nz, grid.ny, grid.nx + 1)
    assert state.v.shape == (grid.nz, grid.ny + 1, grid.nx)
    assert state.w.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.theta.shape == (grid.nz, grid.ny, grid.nx)
    assert state.qv.shape == (grid.nz, grid.ny, grid.nx)
    assert state.p.shape == (grid.nz, grid.ny, grid.nx)
    assert state.ph.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.mu.shape == (grid.ny, grid.nx)
