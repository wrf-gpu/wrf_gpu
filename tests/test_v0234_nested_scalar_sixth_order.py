"""Focused source/algebra gates for the nested moist/scalar diff6 bundle."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import NESTED_BOUNDARY_SCALAR_SPECIES
from gpuwrf.dynamics.explicit_diffusion import wrf_sixth_order_scalar_tendf
from gpuwrf.runtime.operational_mode import (
    _nested_scalar_sixth_order_tendencies,
    _nested_scalar_stage_tendencies,
    _rk_scan_step,
)


WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")


def _fixture():
    nz, ny, nx = 3, 13, 15
    kk, yy, xx = np.indices((nz, ny, nx), dtype=np.float64)
    fields = {}
    for index, name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES, start=1):
        fields[name] = jnp.asarray(
            index * 1.0e-6 * (
                np.sin(0.7 * xx) + np.cos(0.4 * yy) + 0.1 * kk * xx
            )
        )
    mu = 78_000.0 + 15.0 * yy[0] + 7.0 * xx[0]
    metrics = SimpleNamespace(
        c1h=jnp.asarray(np.linspace(0.4, 0.9, nz)),
        c2h=jnp.asarray(np.linspace(100.0, 300.0, nz)),
        msftx=jnp.asarray(0.97 + 2.0e-4 * yy[0] + 1.0e-4 * xx[0]),
        msfty=jnp.asarray(0.97 + 2.0e-4 * yy[0] + 1.0e-4 * xx[0]),
    )
    state = SimpleNamespace(mu_total=jnp.asarray(mu), **fields)
    namelist = SimpleNamespace(
        metrics=metrics,
        dt_s=6.0,
        rk_order=3,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
    )
    return state, namelist


def test_pristine_wrf_moist_and_other_scalar_diff6_is_rk1_frozen():
    module_em = (WRF / "dyn_em/module_em.F").read_text()
    solve_em = (WRF / "dyn_em/solve_em.F").read_text()
    first_rk = (WRF / "dyn_em/module_first_rk_step_part1.F").read_text()
    assert "CALL init_zero_tendency" in first_rk
    assert "moist_tend,chem_tend,scalar_tend" in first_rk
    start = module_em.index("rk_step_1: IF( rk_step == 1 ) THEN")
    end = module_em.index("ENDIF rk_step_1", start)
    rk1 = module_em[start:end]
    assert "CALL sixth_order_diffusion( 'm', scalar" in rk1
    assert "config_flags%moist_mix6_off" in solve_em
    assert "config_flags%scalar_mix6_off" in solve_em
    assert "dt_rk = grid%dt/3." in solve_em
    assert "tendency(i,k,j) = tendency(i,k,j) + sc_tend(i,k,j,im)" in module_em

    source = inspect.getsource(_rk_scan_step)
    build = source.index("rk1_forward_diff6_scalar = _nested_scalar_sixth_order_tendencies")
    stages = source.index("def advance_stage")
    consume = source.index("_nested_scalar_stage_tendencies", stages)
    assert build < stages < consume
    assert source.count("_nested_scalar_sixth_order_tendencies(") == 1


def test_bundle_order_dt_over_three_and_exact_scalar_values():
    state, namelist = _fixture()
    actual = _nested_scalar_sixth_order_tendencies(state, namelist)
    assert len(actual) == len(NESTED_BOUNDARY_SCALAR_SPECIES) == 8
    for name, value in zip(NESTED_BOUNDARY_SCALAR_SPECIES, actual, strict=True):
        expected = wrf_sixth_order_scalar_tendf(
            getattr(state, name),
            state.mu_total,
            c1=namelist.metrics.c1h,
            c2=namelist.metrics.c2h,
            msftx=namelist.metrics.msftx,
            msfty=namelist.metrics.msfty,
            dt=2.0,
            diff_6th_factor=0.12,
            monotonic=True,
            specified_or_nested=True,
        )
        np.testing.assert_array_equal(np.asarray(value), np.asarray(expected))
        assert np.count_nonzero(np.asarray(value)) > 0


def test_bundle_preserves_source_owned_rings_and_never_reads_opposite_edge():
    state, namelist = _fixture()
    qv = np.zeros_like(np.asarray(state.qv))
    qv[:, 3:-3, -7:] = 1.0e-4 * np.asarray([0, 1, 0, 1, 0, 1, 0])
    state.qv = jnp.asarray(qv)
    value = np.asarray(_nested_scalar_sixth_order_tendencies(state, namelist)[0])
    ny, nx = state.mu_total.shape
    yy, xx = np.indices((ny, nx))
    ring = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
    np.testing.assert_array_equal(value[:, ring < 3], 0.0)
    np.testing.assert_array_equal(value[:, 3:-3, 3], 0.0)
    assert np.any(value[:, 3:-3, -4] != 0.0)


def test_frozen_diffusion_is_unscaled_sc_tend_outside_advection_spec_zone():
    nz, ny, nx = 1, 13, 15
    shape = (nz, ny, nx)
    advected = tuple(jnp.ones(shape) for _ in NESTED_BOUNDARY_SCALAR_SPECIES)
    frozen = []
    for index, _name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES, start=1):
        value = np.zeros(shape, dtype=np.float64)
        value[:, 3, 7] = float(index)
        value[:, 6, 7] = float(index)
        frozen.append(jnp.asarray(value))
    species, merged = _nested_scalar_stage_tendencies(
        advected,
        NESTED_BOUNDARY_SCALAR_SPECIES,
        tuple(frozen),
        SimpleNamespace(spec_zone=5),
        jnp.full((ny, nx), 2.0),
    )
    assert species == NESTED_BOUNDARY_SCALAR_SPECIES
    for index, value in enumerate(merged, start=1):
        array = np.asarray(value)
        # Ring 3 is outside the advection rectangle but WRF sc_tend still acts.
        assert array[0, 3, 7] == float(index)
        # Interior receives msfty*advection plus the same unscaled sc_tend.
        assert array[0, 6, 7] == 2.0 + float(index)
        assert array[0, 4, 7] == 0.0


def test_zero_members_remain_bit_exact_zero():
    state, namelist = _fixture()
    zeros = jnp.zeros_like(state.qv)
    for name in NESTED_BOUNDARY_SCALAR_SPECIES:
        setattr(state, name, zeros)
    for value in _nested_scalar_sixth_order_tendencies(state, namelist):
        np.testing.assert_array_equal(np.asarray(value), 0.0)
