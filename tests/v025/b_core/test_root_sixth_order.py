"""BC44: the specified root uses WRF's RK1-frozen sixth-order diffusion (CPU, abstract trace).

WRF sixth_order_diffusion treats specified exactly like nested (module_big_step_utilities_em.F:6334)
and forms u/v/w/theta (module_em.F:882-919) and moist/scalar (:1423) contributions once at RK1.
Numerical evidence vs a pristine serial WRF d01 run: tests/v025/b_core/root_sixth_order_evidence.json.
"""

import jax
import jax.numpy as jnp
import pytest
from prod_inputs import prod_domains

from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree


def _root_case():
    hierarchy, bundles, _, _, _, carries = prod_domains()
    nml = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).domains["d01"].namelist
    carry = carries["d01"]
    if op._h_diabatic_pair_enabled(nml) and carry.h_diabatic is None:
        carry = carry.replace(h_diabatic=jnp.zeros_like(carry.state.theta))
    return nml, carry


def _trace(nml, carry):
    calls = {"uvw": 0, "scalar": 0, "periodic": 0}
    names = {"uvw": "wrf_sixth_order_uvw_tendf", "scalar": "wrf_sixth_order_scalar_tendf",
             "periodic": "sixth_order_diffusion_tendency"}
    originals = {key: getattr(op, name) for key, name in names.items()}

    def spy(key):
        def wrapped(*args, **kwargs):
            calls[key] += 1
            return originals[key](*args, **kwargs)
        return wrapped

    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    for key, name in names.items():
        setattr(op, name, spy(key))
    try:
        jax.clear_caches()
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1,
            cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        for key, name in names.items():
            setattr(op, name, originals[key])
        jax.clear_caches()
    return calls


def test_specified_root_uses_frozen_wrf_sixth_order():
    nml, carry = _root_case()
    assert jax.devices()[0].platform == "cpu"
    assert int(nml.diff_6th_opt) == 2 and op._acoustic_lateral_bc_flags(nml)[1]
    calls = _trace(nml, carry)
    # theta + 8 moist/number scalars through the scalar form, u/v/w once; no periodic form.
    assert calls == {"uvw": 1, "scalar": 9, "periodic": 0}


def test_periodic_programs_keep_released_sixth_order():
    nml, carry = _root_case()
    import dataclasses
    periodic = dataclasses.replace(nml, run_boundary=False)
    assert not any(op._acoustic_lateral_bc_flags(periodic)[1:])
    calls = _trace(periodic, carry)
    assert calls["uvw"] == 0 and calls["scalar"] == 0 and calls["periodic"] > 0


@pytest.mark.parametrize("domain", ["d01", "d02"])
def test_scalar_sixth_order_uses_rk1_post_acoustic_muts(domain):
    """rk_scalar_tend receives grid%muts (solve_em.F:2303): the RK1 post-acoustic total mass."""
    hierarchy, bundles, _, _, _, carries = prod_domains()
    nml = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).domains[domain].namelist
    carry = carries[domain]
    if op._h_diabatic_pair_enabled(nml) and carry.h_diabatic is None:
        carry = carry.replace(h_diabatic=jnp.zeros_like(carry.state.theta))
    finished, masses = [], []
    original_finish = op._carry_from_finished_stage
    original_scalar = op.wrf_sixth_order_scalar_tendf
    rk1_dt = float(nml.dt_s) / float(nml.rk_order)

    def spy_finish(*args, **kwargs):
        out = original_finish(*args, **kwargs)
        finished.append(out.state.mu_total)
        return out

    def spy_scalar(field, mu_total, **kwargs):
        if abs(float(kwargs["dt"]) - rk1_dt) < 1e-9:
            masses.append(mu_total)
        return original_scalar(field, mu_total, **kwargs)

    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    op._carry_from_finished_stage = spy_finish
    op.wrf_sixth_order_scalar_tendf = spy_scalar
    try:
        jax.clear_caches()
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), op.build_clock_base(nml), n_steps=1,
            cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        op._carry_from_finished_stage = original_finish
        op.wrf_sixth_order_scalar_tendf = original_scalar
        jax.clear_caches()
    assert len(finished) == int(nml.rk_order)
    # Post-acoustic bundle: once per step (RK1), reused in RK2/RK3.  The nested
    # child's released C1 arm also builds a time-t bundle (rk1_reference mass).
    post = [m for m in masses if m is finished[0]]
    assert len(post) == 8
    assert len(masses) in (8, 16)
