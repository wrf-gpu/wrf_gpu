"""REAL4 literal oracle for the source-leaf physics coupling (GPUWRF_CARRY_REAL_ALL).

Transcribes pristine WRF in Fortran order, one float32 rounding per operation:
calculate_phy_tend (dyn_em/module_em.F:2235-2245: R*TEN = (c1h*mut+c2h)*R*TEN, each source
separately), update_phy_ten (phys/module_physics_addtendc.F add_a2a: rt_tendf = 0 + RTHRATEN,
then + RTHBLTEN; add_a2c_u/v: 0 + 0.5*(r(i)+r(i-1)) on the specified-domain faces) and
conv_t_tendf_to_moist (module_big_step_utilities_em.F:6697-6702) with REAL r_v/r_d.
"""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as om

F = np.float32
NZ, NY, NX = 6, 5, 7


def _operands(seed=11):
    rng = np.random.default_rng(seed)
    m = types.SimpleNamespace(c1h=rng.uniform(0.0, 1.0, NZ).astype(F), c2h=rng.uniform(0.0, 5000.0, NZ).astype(F))
    mut = rng.uniform(8.0e4, 9.5e4, (NY, NX)).astype(F)
    shape = (NZ, NY, NX)
    rthraten = rng.normal(0.0, 2e-5, shape).astype(F)
    rthblten = rng.normal(0.0, 5e-4, shape).astype(F)
    rqvblten = rng.normal(0.0, 2e-7, shape).astype(F)
    rublten = rng.normal(0.0, 1e-3, shape).astype(F)
    rvblten = rng.normal(0.0, 1e-3, shape).astype(F)
    qv = rng.uniform(1e-4, 1.5e-2, shape).astype(F)
    theta = rng.uniform(290.0, 330.0, shape).astype(F)
    return mut, rthraten, rthblten, rqvblten, rublten, rvblten, qv, theta, m


def _wrf_literal(mut, rthraten, rthblten, rqvblten, rublten, rvblten, qv, theta, m):
    mass = m.c1h[:, None, None] * mut[None] + m.c2h[:, None, None]
    t_tendf = (F(0.0) + mass * rthraten) + mass * rthblten
    qv_tend = F(0.0) + mass * rqvblten
    rvrd = F(461.6) / F(287.0)
    one = F(1.0)
    t_m = (one + rvrd * qv) * t_tendf + ((rvrd * theta) / (one + rvrd * qv)) * qv_tend
    rub, rvb = mass * rublten, mass * rvblten
    ru = np.zeros((NZ, NY, NX + 1), F); rv = np.zeros((NZ, NY + 1, NX), F)
    for i in range(1, NX):            # add_a2c_u: faces ids+1..ide-1, rows jds+1..jde-2
        ru[:, 1:NY - 1, i] = F(0.0) + F(0.5) * (rub[:, 1:NY - 1, i] + rub[:, 1:NY - 1, i - 1])
    for j in range(1, NY):            # add_a2c_v: faces jds+1..jde-1, cols ids+1..ide-2
        rv[:, j, 1:NX - 1] = F(0.0) + F(0.5) * (rvb[:, j, 1:NX - 1] + rvb[:, j - 1, 1:NX - 1])
    return t_m, ru, rv


def _run(mut, rthraten, rthblten, rqvblten, rublten, rvblten, qv, theta, m):
    jm = types.SimpleNamespace(c1h=jnp.asarray(m.c1h), c2h=jnp.asarray(m.c2h), precision="fp32")
    def f(*a):
        out = om._source_leaf_dry_tendencies(*a, jm, jnp.float32, real_glue=True)
        return out.t_tendf, out.ru_tendf, out.rv_tendf
    out = jax.jit(f)(*(jnp.asarray(x) for x in (mut, rthraten, rthblten, rqvblten, rublten, rvblten, qv, theta)))
    return tuple(np.asarray(x) for x in out)


def _within(got, ref, scale):
    return np.abs(got.astype(np.float64) - ref.astype(np.float64)) <= 8 * np.finfo(F).eps * scale + 1e-30


def _scales(mut, rthraten, rthblten, rqvblten, rublten, rvblten, qv, theta, m):
    d = lambda a: np.abs(np.asarray(a, np.float64))
    mass = d(m.c1h)[:, None, None] * d(mut)[None] + d(m.c2h)[:, None, None]
    t = (1 + 1.61 * d(qv)) * mass * (d(rthraten) + d(rthblten)) + 1.61 * d(theta) * mass * d(rqvblten)
    ru = np.zeros((NZ, NY, NX + 1)); rv = np.zeros((NZ, NY + 1, NX))
    ru[:, :, 1:NX] = mass[:, :, 1:] * d(rublten)[:, :, 1:] + mass[:, :, :-1] * d(rublten)[:, :, :-1]
    rv[:, 1:NY, :] = mass[:, 1:] * d(rvblten)[:, 1:] + mass[:, :-1] * d(rvblten)[:, :-1]
    return t, ru, rv


def test_real_constant_is_wrf_real():
    assert om._RVRD_REAL == F(461.6) / F(287.0) and om._RVRD_REAL.dtype == np.float32


def test_source_leaf_coupling_matches_wrf_literal():
    ops = _operands()
    got, ref, scales = _run(*ops), _wrf_literal(*ops), _scales(*ops)
    for g, r, sc in zip(got, ref, scales):
        assert g.dtype == np.float32 and g.shape == r.shape
        assert np.all(_within(g, r, sc)), float(np.max(np.abs(g - r)))
    assert np.array_equal(got[1][:, [0, NY - 1], :], np.zeros((NZ, 2, NX + 1), F))  # specified-edge exclusion


@pytest.mark.parametrize("mutant", ["no_pbl_theta", "no_moist_conversion", "rvrd_p608", "a2c_shift"])
def test_oracle_rejects_mutants(mutant, monkeypatch):
    ops = list(_operands())
    ref, scales = _wrf_literal(*ops), _scales(*ops)
    if mutant == "no_pbl_theta":
        ops[2] = np.zeros_like(ops[2])
    elif mutant == "no_moist_conversion":
        ops[6] = np.zeros_like(ops[6])
    elif mutant == "rvrd_p608":
        monkeypatch.setattr(om, "_RVRD_REAL", F(0.608))
    else:
        ops[4] = np.roll(ops[4], 1, axis=2)
    got = _run(*ops)
    assert not all(np.all(_within(g, r, sc)) for g, r, sc in zip(got, ref, scales))
