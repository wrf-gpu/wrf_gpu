"""E85 compiled two-step carry seam; structural evidence, not forecast fidelity.

Numerical RK/radiation/MP drivers are substituted. The real operational
dispatcher, MP writeback/producer, non-dry updates, precision enforcement and
final carry reconstruction execute on real RE01 columns under CPU JIT.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import rrtmg_mp_re as R
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.validation.tier2 import make_ideal_grid
from test_mp_re import _real_state


@pytest.mark.parametrize("native_carry", [False, True])
def test_compiled_step2_radiation_reads_step1_post_mp_radii(monkeypatch, native_carry):
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setattr(R, "_MP_RE", True)
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "1")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    for name in ("GPUWRF_DYN_FP32", "GPUWRF_DYN_RK_FP32", "GPUWRF_DYN_CARRY_FP32", "GPUWRF_CARRY_REAL_ALL", "GPUWRF_DYN_REAL_ALL"):
        monkeypatch.setenv(name, str(int(native_carry)))
    monkeypatch.setattr(op, "_W_SURFACE_RESET", False)
    state, mp = _real_state()
    state = state.ensure_conditional_leaves(mp_physics=8).replace(mu_total=jnp.full_like(state.mu_total, 90000.))
    grid = make_ideal_grid(*state.theta.shape)
    nml = op.OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None,
        dt_s=54., mp_physics=8, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=0,
        ra_sw_physics=4, ra_lw_physics=4, use_mp_re=1, radiation_cadence_steps=1, rad_rk_tendf=1,
        run_physics=True, run_boundary=False, disable_guards=True, force_fp64=True,
        time_utc="2026-02-28_00:00:00")
    if native_carry:
        from gpuwrf.kernels.dyn_carry_fp32 import real_state, real_scratch, real_carry
        state = real_state(state)
        carry = real_carry(real_scratch(initial_operational_carry(state, base_state=op._base_state_from_totals(state))))
    else:
        carry = initial_operational_carry(state)
    # Alter current hydrometeors in RK: a radiation-time recomputation cannot
    # accidentally agree with the previous-step saved radii.
    monkeypatch.setattr(op, "_rk_scan_step", lambda c, *a, **kw: c.replace(state=c.state.replace(
        qc=jnp.zeros_like(c.state.qc), qi=jnp.zeros_like(c.state.qi), qs=jnp.zeros_like(c.state.qs))))
    monkeypatch.setattr(op, "thompson_adapter", lambda s, *a, **kw: C._state_from_thompson_output(s, mp))

    def two_steps(c):
        observed = []
        def radiation(s, *args, **kwargs):
            sw, lw, *_ = C._rrtmg_column_inputs(s, None, time_utc=nml.time_utc,
                                               use_mp_re=kwargs["use_mp_re"])
            observed.append(tuple(getattr(column, name) for column in (sw, lw)
                                  for name in ("re_cloud", "re_ice", "re_snow")))
            return jnp.zeros_like(s.theta)
        # The observer stays inside this trace; its arrays are returned as
        # device outputs. There is no debug callback or transfer in the step.
        with monkeypatch.context() as local:
            local.setattr(op, "rrtmg_theta_tendency", radiation)
            first = op._physics_boundary_step(c, nml, jnp.int32(1), run_radiation=True)
            second = op._physics_boundary_step(first, nml, jnp.int32(2), run_radiation=True)
        return first, second, tuple(observed)

    first, second, observed = jax.jit(two_steps)(carry)
    assert len(observed) == 2
    for index, name in enumerate(("re_cloud", "re_ice", "re_snow")):
        saved = getattr(first.state, name)
        assert saved.dtype == jnp.float32
        assert getattr(second.state, name).dtype == jnp.float32
        for offset in (0, 3):
            np.testing.assert_array_equal(observed[0][offset + index], C._to_columns(getattr(carry.state, name)))
            np.testing.assert_array_equal(observed[1][offset + index], C._to_columns(saved))
    assert np.any(np.asarray(first.state.re_cloud) != np.float32(2.49e-6))
