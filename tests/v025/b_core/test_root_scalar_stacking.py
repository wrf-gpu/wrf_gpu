"""S3: root scalar transport in one call + species-stacked native plain stencil (CPU).

(a) moist + Ni/Nr share one _scalar_transport_coupled_tendencies call when both WRF loops use
the same option; (b) the native REAL specified plain stencil runs all species in one Pallas
launch (species loop inside each program, BC48).  Both must be bitwise identical to the
per-species form (GPU: phase2b/BC48_stackloop/probe.json).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prod_inputs import prod_domains

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr")


def _case():
    hierarchy, bundles, _, _, _, carries = prod_domains()
    nml = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).domains["d01"].namelist
    state = carries["d01"].state
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
def test_merged_and_stacked_equal_per_species(monkeypatch, native, rk_step):
    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", native)
    jax.clear_caches()
    nml, origin, current = _case()
    assert int(nml.moist_adv_opt) == int(nml.scalar_adv_opt) == 1
    merged = _transport(nml, current, origin, SPECIES, rk_step)
    split = (_transport(nml, current, origin, SPECIES[:6], rk_step)
             + _transport(nml, current, origin, SPECIES[6:], rk_step))
    single = tuple(_transport(nml, current, origin, (name,), rk_step)[0] for name in SPECIES)
    for name, a, b, c in zip(SPECIES, merged, split, single):
        assert np.array_equal(np.asarray(a), np.asarray(b)), name
        assert np.array_equal(np.asarray(a), np.asarray(c)), name
    jax.clear_caches()


def test_root_own_step_traces_one_stacked_stencil_per_plain_stage(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", "1")
    if os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1":
        # Import-time REAL families (KF column, sfclay, ...) cannot be reverted per test, and
        # the conftest-legacy PROD carry is the f64 carry (no base_state, f64 KF held rates).
        pytest.skip("legacy f64 carry fixture; the v0.3 REAL carry needs a REAL-flag process "
                    "(b-core) -- release-flag stacking is covered by the LW10 GPU cold + BC48 gate")
    hierarchy, bundles, _, _, _, carries = prod_domains()
    nml = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).domains["d01"].namelist
    carry = carries["d01"]
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
