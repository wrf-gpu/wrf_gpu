"""B38: WRF h_diabatic pair under WRF-order MP (CPU, abstract trace of the real root own-step).

WRF (use_theta_m=1) keeps the previous post-RK microphysics theta_m heating rate
(moist_physics_finish_em, module_big_step_utilities_em.F:5735-5741), adds
(c1*mut+c2)*h_diabatic/msfty to t_tend at EVERY RK stage (rk_addtend_dry,
module_em.F:1079) and removes dts*N*(c1*mut+c2)*h_diabatic at the final-stage
small_step_finish (module_small_step_em.F:417-423).  Numerical pristine evidence:
tests/v025/b_core/h_diabatic_evidence.json.
"""

import jax
import jax.numpy as jnp
import pytest
from prod_inputs import prod_domains

from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree


def _root_case():
    hierarchy, bundles, _, _, _, carries = prod_domains()
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    return tree.domains["d01"].namelist, carries["d01"]


def _trace(nml, carry, *, finish=None, addtend=None):
    seen = {"addtend": [], "finish": []}
    original_addtend = op.rk_addtend_dry
    original_finish = op.small_step_finish_wrf

    def spy_addtend(tendencies, physics, **kwargs):
        seen["addtend"].append(getattr(physics, "h_diabatic", None) is not None)
        return (addtend or original_addtend)(tendencies, physics, **kwargs)

    def spy_finish(prep, acoustic, **kwargs):
        seen["finish"].append(kwargs.get("h_diabatic") is not None)
        return (finish or original_finish)(prep, acoustic, **kwargs)

    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    op.rk_addtend_dry = spy_addtend
    op.small_step_finish_wrf = spy_finish
    try:
        jax.clear_caches()
        out = jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1,
            cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        op.rk_addtend_dry = original_addtend
        op.small_step_finish_wrf = original_finish
        jax.clear_caches()
    return seen, out


def test_pair_dispatch_follows_wrf_order(monkeypatch):
    nml, carry = _root_case()
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    monkeypatch.delenv("GPUWRF_MP_H_DIABATIC", raising=False)
    assert op._h_diabatic_pair_enabled(nml)
    seeded = carry.replace(h_diabatic=jnp.zeros_like(carry.state.theta))
    seen, out = _trace(nml, seeded)
    stages = int(nml.rk_order)
    assert seen["addtend"] == [True] * stages
    assert seen["finish"] == [False] * (stages - 1) + [True]
    assert out.h_diabatic.shape == carry.state.theta.shape
    assert out.h_diabatic.dtype == carry.state.theta.dtype

    # Legacy order and explicit opt-out keep the released carry (no leaf, no pair).
    monkeypatch.setenv("GPUWRF_MP_H_DIABATIC", "0")
    assert not op._h_diabatic_pair_enabled(nml)
    monkeypatch.setenv("GPUWRF_MP_H_DIABATIC", "1")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "0")
    assert not op._h_diabatic_pair_enabled(nml)
    # The loader seeds the leaf under the WRF-order default; the released legacy carry has none.
    seen, out = _trace(nml, carry.replace(h_diabatic=None))
    assert seen["addtend"] == [False] * stages and seen["finish"] == [False] * stages
    assert out.h_diabatic is None


def test_restart_roundtrip_keeps_h_diabatic(tmp_path, monkeypatch):
    """WRF restarts h_diabatic (Registry 'rdu'); the wrfrst carry must round-trip it exactly."""
    import numpy as np
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.contracts.state import State
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry

    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices()[0])
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    heat = jnp.asarray(np.linspace(-1e-3, 2e-3, state.theta.size).reshape(state.theta.shape), state.theta.dtype)
    kwargs = dict(valid_time="2026-06-03_00:30:00", run_start="2026-06-03_00:00:00", step_index=1)
    for value in (heat, None):
        carry = initial_operational_carry(state).replace(h_diabatic=value)
        path = tmp_path / ("with" if value is not None else "without")
        write_wrfrst_carry(carry, grid, {}, path, **kwargs)
        restored, _ = read_wrfrst_carry(path)
        if value is None:
            assert restored.h_diabatic is None
        else:
            assert restored.h_diabatic.dtype == heat.dtype
            assert np.array_equal(np.asarray(restored.h_diabatic), np.asarray(heat))
