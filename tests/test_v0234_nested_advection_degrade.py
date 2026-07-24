"""CPU/source gates for WRF's nested advection-boundary stencil selection."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.flux_advection import (
    CoupledVelocities,
    advect_scalar_flux,
)
from gpuwrf.runtime.operational_mode import (
    _specified_adv_degrade_active,
    _stage_transport_velocities,
)
from gpuwrf.validation.moving_nest_testbed import (
    build_domain_namelist,
    build_flat_grid,
    build_neutral_state,
)


PRISTINE_ADVECT = Path(
    "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F"
)


def _child_namelist(*, bundle: bool, legacy_flag: bool = False):
    grid = build_flat_grid(nx=14, ny=13, nz=4, dx_m=1000.0)
    grid = dataclasses.replace(
        grid,
        bc=dataclasses.replace(grid.bc, source="parent-live-test"),
    )
    namelist = build_domain_namelist(
        grid,
        dt_s=2.0,
        is_child=True,
        parent_dt_s=6.0,
    )
    boundary = dataclasses.replace(
        namelist.boundary_config,
        force_geopotential=False,
        nested_frozen_wrf_boundary_bundle=bool(bundle),
    )
    return dataclasses.replace(
        namelist,
        boundary_config=boundary,
        specified_adv_degrade=bool(legacy_flag),
    )


def test_pristine_wrf_selects_degraded_stencils_for_specified_or_nested():
    source = PRISTINE_ADVECT.read_text()
    selector = (
        "if(config_flags%specified .or. config_flags%nested) "
        "specified = .true."
    )
    # U, V, scalar, W, PD, WENO scalar, WENO-PD scalar, and monotonic scalar.
    assert source.count(selector) == 8
    for routine in (
        "advect_u",
        "advect_v",
        "advect_scalar",
        "advect_w",
        "advect_scalar_pd",
        "advect_scalar_weno",
        "advect_scalar_wenopd",
        "advect_scalar_mono",
    ):
        assert f"SUBROUTINE {routine}" in source


def test_frozen_live_child_activates_existing_degraded_branch_without_legacy_flag():
    candidate = _child_namelist(bundle=True, legacy_flag=False)
    released = _child_namelist(bundle=False, legacy_flag=False)
    assert _specified_adv_degrade_active(candidate) is True
    assert _specified_adv_degrade_active(released) is False

    state = build_neutral_state(candidate.grid, u0_m_s=7.0, v0_m_s=-2.0)
    velocities = _stage_transport_velocities(state, candidate)
    assert velocities.specified is True
    assert velocities.ru_full is not None
    assert velocities.rv_full is not None
    assert velocities.ru_full.shape[-1] == candidate.grid.nx + 1
    assert velocities.rv_full.shape[-2] == candidate.grid.ny + 1


def test_legacy_root_gate_and_candidate_off_semantics_are_unchanged():
    child = _child_namelist(bundle=False, legacy_flag=False)
    specified_root = dataclasses.replace(
        child,
        specified_adv_degrade=True,
        boundary_config=dataclasses.replace(
            child.boundary_config,
            force_geopotential=True,
            nested_frozen_wrf_boundary_bundle=False,
        ),
    )
    assert _specified_adv_degrade_active(specified_root) is True
    assert _specified_adv_degrade_active(
        dataclasses.replace(specified_root, specified_adv_degrade=False)
    ) is False
    assert _specified_adv_degrade_active(
        dataclasses.replace(specified_root, run_boundary=False)
    ) is False


def _scalar_tendency(*, specified: bool) -> np.ndarray:
    nz, ny, nx = 4, 9, 12
    field = np.zeros((nz, ny, nx), dtype=np.float64)
    # An east outer-ring signal must not enter the west ring through a periodic
    # fifth-order stencil on a WRF nested domain.
    field[:, :, -1] = 1.0
    velocities = CoupledVelocities(
        ru=jnp.ones((nz, ny, nx), dtype=jnp.float64),
        rv=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        rom=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        msftx=jnp.ones((ny, nx), dtype=jnp.float64),
        specified=specified,
    )
    return np.asarray(
        advect_scalar_flux(
            jnp.asarray(field),
            velocities,
            mut=jnp.ones((ny, nx), dtype=jnp.float64),
            c1=jnp.ones((nz,), dtype=jnp.float64),
            rdx=1.0,
            rdy=1.0,
            rdzw=jnp.ones((nz,), dtype=jnp.float64),
            fzm=jnp.full((nz + 1,), 0.5, dtype=jnp.float64),
            fzp=jnp.full((nz + 1,), 0.5, dtype=jnp.float64),
        )
    )


def test_nested_degraded_stencil_blocks_opposite_edge_wrap_at_ring_one():
    periodic = _scalar_tendency(specified=False)
    nested = _scalar_tendency(specified=True)
    assert float(np.max(np.abs(periodic[:, :, 1]))) == 0.25
    np.testing.assert_array_equal(nested[:, :, 1], 0.0)
    # The east-local ring response remains nonzero: this is a stencil/ownership
    # correction, not a boundary tendency mask or blanket advection disable.
    assert float(np.max(np.abs(nested[:, :, -2]))) == 0.5
