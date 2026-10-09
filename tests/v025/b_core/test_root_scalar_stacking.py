"""S3: root scalar transport in one call + species-stacked native plain stencil (CPU).

(a) moist + Ni/Nr share one _scalar_transport_coupled_tendencies call when both WRF loops use
the same option; (b) the native REAL specified plain stencil runs all species in one Pallas
launch (species loop inside each program, BC48).  Both must be bitwise identical to the
per-species form (GPU: phase2b/BC48_stackloop/probe.json).
"""
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree
from gpuwrf.contracts import state as state_contract
from gpuwrf.dynamics.metrics import load_wrfinput_metrics
from gpuwrf.validation.tier2 import make_ideal_grid
from gpuwrf.coupling.boundary_apply import BoundaryConfig

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr")


def _assert_transport_equal(merged, split, single, name):
    assert all(np.isfinite(np.asarray(value)).all() for value in (merged, split, single)), f'nonfinite scalar {name}'
    assert np.array_equal(np.asarray(merged), np.asarray(split)), name
    assert np.array_equal(np.asarray(merged), np.asarray(single)), name


@pytest.mark.parametrize('name', SPECIES)
@pytest.mark.parametrize('role', ('merged', 'split', 'single', 'all'))
@pytest.mark.parametrize('bad', (np.nan, np.inf, -np.inf))
def test_transport_comparison_rejects_nonfinite(name, role, bad):
    arrays = {key: np.array([.25, .5]) for key in ('merged', 'split', 'single')}
    for key, value in arrays.items():
        if role in (key, 'all'):
            value[-1] = bad
    with pytest.raises(AssertionError, match=f'nonfinite scalar {name}'):
        _assert_transport_equal(*arrays.values(), name)


def _fixture():
    """Real PROD window, freshly seeded in the active carry precision.

    These transport/launch-count gates need the dycore, not an initial radiation
    solve over both full domains. Keep the 44 real levels and specified stencil.
    """
    path = Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725/wrfinput_d01')
    assert path.exists(), 'registered real PROD input required'
    ny = nx = 12
    y, x = 29, 48
    with Dataset(path) as ds:
        full_ny, full_nx = len(ds.dimensions['south_north']), len(ds.dimensions['west_east'])
        nz = len(ds.dimensions['bottom_top'])

        def read(name):
            var = ds[name]
            ey = int(var.dimensions[-2] == 'south_north_stag')
            ex = int(var.dimensions[-1] == 'west_east_stag')
            return jnp.asarray(np.asarray(var[0, ..., y:y+ny+ey, x:x+nx+ex]), jnp.float64)

        a = {name: read(name) for name in ('PB','PHB','MUB','P','PH','MU','U','V','W','T','QVAPOR','HGT')}
        dx, dy = float(ds.DX), float(ds.DY)
    metrics = load_wrfinput_metrics(path)
    updates = {}
    for name in metrics._array_names():
        value = getattr(metrics, name)
        if value.ndim == 2:
            ey, ex = int(value.shape[0] == full_ny+1), int(value.shape[1] == full_nx+1)
            updates[name] = value[y:y+ny+ey, x:x+nx+ex]
    metrics = replace(metrics, **updates)
    grid = make_ideal_grid(nz, ny, nx, dx_m=dx, dy_m=dy)
    grid = replace(grid, metrics=metrics, terrain_height=a['HGT'], bc=replace(grid.bc, source='wrfbdy'))
    guard = state_contract._gpu_device
    state_contract._gpu_device = lambda: jax.devices('cpu')[0]
    try:
        state = state_contract.State.zeros(grid).replace(_cast=False,
            u=a['U'], v=a['V'], w=a['W'], qv=a['QVAPOR'],
            theta=(a['T']+300)*(1+(461.6/287)*a['QVAPOR']),
            p_total=a['PB']+a['P'], p_perturbation=a['P'],
            ph_total=a['PHB']+a['PH'], ph_perturbation=a['PH'],
            mu_total=a['MUB']+a['MU'], mu_perturbation=a['MU'])
    finally:
        state_contract._gpu_device = guard
    nml = replace(op.OperationalNamelist.from_grid(grid, metrics=metrics, force_fp64=True),
        run_physics=False, mp_physics=8, cu_physics=0, bl_pbl_physics=0,
        sf_sfclay_physics=0, sf_surface_physics=0, ra_sw_physics=0, ra_lw_physics=0,
        use_noahmp=False, use_flux_advection=True, specified_adv_degrade=True,
        boundary_config=BoundaryConfig(), moist_adv_opt=1, scalar_adv_opt=1)
    return nml, op._initial_carry_for_run(state, nml)


def _case():
    nml, carry = _fixture()
    state = carry.state
    rng = np.random.default_rng(7)
    fields = {name: jnp.asarray(np.abs(rng.standard_normal(state.qv.shape)) * scale, state.qv.dtype)
              for name, scale in zip(SPECIES, (1e-2, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e4, 1e3))}
    origin = apply_halo(state.replace(**fields), halo_spec(nml.grid))
    shifted = {name: jnp.roll(value, 1, axis=-1) for name, value in fields.items()}
    current = apply_halo(state.replace(**shifted), halo_spec(nml.grid))
    return nml, origin, current


def _transport(nml, current, origin, species, rk_step):
    return jax.jit(lambda cur, org: op._scalar_transport_coupled_tendencies(
        cur, nml, rk_step=rk_step, step_origin=org, species=species,
        advection_opt=int(nml.moist_adv_opt),
        transport_velocities=op._stage_transport_velocities(cur, nml)))(current, origin)


@pytest.mark.parametrize("native", ["0", "1"])
@pytest.mark.parametrize("rk_step", [1, 3])
def test_merged_and_stacked_equal_per_species(monkeypatch, cpu_pallas_interpret, native, rk_step):
    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", native)
    jax.clear_caches()
    nml, origin, current = _case()
    assert int(nml.moist_adv_opt) == int(nml.scalar_adv_opt) == 1
    assert op._stage_transport_velocities(current, nml).specified
    merged = _transport(nml, current, origin, SPECIES, rk_step)
    split = (_transport(nml, current, origin, SPECIES[:6], rk_step)
             + _transport(nml, current, origin, SPECIES[6:], rk_step))
    single = tuple(_transport(nml, current, origin, (name,), rk_step)[0] for name in SPECIES)
    for name, a, b, c in zip(SPECIES, merged, split, single):
        _assert_transport_equal(a, b, c, name)
    jax.clear_caches()


def test_root_own_step_traces_one_stacked_stencil_per_plain_stage(monkeypatch, cpu_pallas_interpret):
    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", "1")
    nml, carry = _fixture()
    if os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1":
        assert carry.state.theta.dtype == jnp.float32
        assert carry.base_state is not None
    if op._h_diabatic_pair_enabled(nml) and carry.h_diabatic is None:
        carry = carry.replace(h_diabatic=jnp.zeros_like(carry.state.theta))
    from gpuwrf.kernels import dyn_flux_fp32
    original = dyn_flux_fp32.advect_scalar_flux_fp32
    original_stacked = dyn_flux_fp32.advect_scalar_flux_fp32_stacked
    shapes, stacked_shapes = [], []

    def spy(field, *args, **kwargs):
        shapes.append(tuple(field.shape))
        return original(field, *args, **kwargs)

    def spy_stacked(fields, *args, **kwargs):
        stacked_shapes.append(tuple(fields.shape))
        return original_stacked(fields, *args, **kwargs)

    dyn_flux_fp32.advect_scalar_flux_fp32 = spy
    dyn_flux_fp32.advect_scalar_flux_fp32_stacked = spy_stacked
    try:
        jax.clear_caches()
        abstract = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), op.build_clock_base(nml), n_steps=1,
            cadence=int(nml.radiation_cadence_steps)), abstract)
    finally:
        dyn_flux_fp32.advect_scalar_flux_fp32 = original
        dyn_flux_fp32.advect_scalar_flux_fp32_stacked = original_stacked
        jax.clear_caches()
    nz, ny, nx = carry.state.theta.shape
    # theta: one call per RK stage (3); scalars: ONE stacked launch per plain stage (RK1, RK2)
    # instead of 8 per stage (was 3 + 16 = 19 launches).
    assert shapes == [(nz, ny, nx)] * 3, shapes
    assert stacked_shapes == [(len(SPECIES), nz, ny, nx)] * 2, stacked_shapes
