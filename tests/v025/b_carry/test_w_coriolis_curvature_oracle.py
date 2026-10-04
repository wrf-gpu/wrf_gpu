"""REAL4 literal oracle for the rw_tend Coriolis + curvature glue (GPUWRF_CARRY_REAL_ALL).

The oracle transcribes pristine WRF in Fortran evaluation order, one float32 rounding per
operation: couple_momentum (module_big_step_utilities_em.F:369-385: ru = u*(c1h*muu+c2h)/msfu,
rv = v*(c1h*muv+c2h)*msfv_inv), coriolis (:3833-3842) and curvature (:4453-4463) for rw_tend
on faces k = 2..kte-1, with module_em passing fnm/fnp as fzm/fzp.
"""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime.operational_mode import _w_coriolis_curvature

F = np.float32
NZ, NY, NX = 9, 6, 7


def _operands(seed=3):
    rng = np.random.default_rng(seed)
    m = types.SimpleNamespace(
        c1h=rng.uniform(0.0, 1.0, NZ).astype(F), c2h=rng.uniform(0.0, 5000.0, NZ).astype(F),
        fnm=rng.uniform(0.4, 0.6, NZ).astype(F), fnp=rng.uniform(0.4, 0.6, NZ).astype(F),
        msfuy=rng.uniform(0.98, 1.02, (NY, NX + 1)).astype(F), msfvx=rng.uniform(0.98, 1.02, (NY + 1, NX)).astype(F),
        msftx=rng.uniform(0.98, 1.02, (NY, NX)).astype(F), msfty=rng.uniform(0.98, 1.02, (NY, NX)).astype(F),
        e=rng.uniform(0.9e-4, 1.1e-4, (NY, NX)).astype(F), cosa=rng.uniform(0.98, 1.0, (NY, NX)).astype(F),
        sina=rng.uniform(-0.1, 0.1, (NY, NX)).astype(F),
    )
    u = rng.normal(0.0, 15.0, (NZ, NY, NX + 1)).astype(F)
    v = rng.normal(0.0, 15.0, (NZ, NY + 1, NX)).astype(F)
    muu = rng.uniform(8.0e4, 9.5e4, (NY, NX + 1)).astype(F)
    muv = rng.uniform(8.0e4, 9.5e4, (NY + 1, NX)).astype(F)
    return u, v, muu, muv, m


def _wrf_literal(u, v, muu, muv, m):
    ru = np.empty_like(u); rv = np.empty_like(v)
    for k in range(NZ):  # couple_momentum
        ru[k] = (u[k] * (m.c1h[k] * muu + m.c2h[k])) / m.msfuy
        rv[k] = (v[k] * (m.c1h[k] * muv + m.c2h[k])) * (F(1.0) / m.msfvx)
    rer = F(1.0) / F(6370.0e3)
    half = F(0.5)
    mxy = m.msftx / m.msfty
    out = np.zeros((NZ - 1, NY, NX), F)
    for k in range(1, NZ):
        fzm, fzp = m.fnm[k], m.fnp[k]
        xru = fzm * (ru[k, :, :-1] + ru[k, :, 1:]) + fzp * (ru[k - 1, :, :-1] + ru[k - 1, :, 1:])
        xrv = fzm * (rv[k, :-1, :] + rv[k, 1:, :]) + fzp * (rv[k - 1, :-1, :] + rv[k - 1, 1:, :])
        xu = fzm * (u[k, :, :-1] + u[k, :, 1:]) + fzp * (u[k - 1, :, :-1] + u[k - 1, :, 1:])
        xv = fzm * (v[k, :-1, :] + v[k, 1:, :]) + fzp * (v[k - 1, :-1, :] + v[k - 1, 1:, :])
        cor = m.e * ((m.cosa * half) * xru - ((mxy * m.sina) * half) * xrv)
        curv = rer * (((half * xru) * half) * xu + (((mxy * half) * xrv) * half) * xv)
        out[k - 1] = (F(0.0) + cor) + curv
    return out


def _scale(u, v, muu, muv, m):
    """Sum of |term| magnitudes (f64) bounding the REAL rounding of each face value."""
    d = lambda a: np.asarray(a, np.float64)
    ru = d(u) * (d(m.c1h)[:, None, None] * d(muu) + d(m.c2h)[:, None, None]) / d(m.msfuy)
    rv = d(v) * (d(m.c1h)[:, None, None] * d(muv) + d(m.c2h)[:, None, None]) / d(m.msfvx)
    ruc = np.abs(0.5 * (ru[:, :, :-1] + ru[:, :, 1:])); rvc = np.abs(0.5 * (rv[:, :-1] + rv[:, 1:]))
    uc = np.abs(0.5 * (d(u)[:, :, :-1] + d(u)[:, :, 1:])); vc = np.abs(0.5 * (d(v)[:, :-1] + d(v)[:, 1:]))
    face = lambda f: f[1:] + f[:-1]
    return np.abs(d(m.e)) * (face(ruc) + face(rvc)) + (face(ruc) * face(uc) + face(rvc) * face(vc)) / 6370.0e3


def _run(u, v, muu, muv, m, **kw):
    jm = types.SimpleNamespace(**{k: jnp.asarray(val) for k, val in vars(m).items()})
    fn = jax.jit(lambda a, b, c, d: _w_coriolis_curvature(a, b, c, d, jm, jnp.float32, **kw))
    return np.asarray(fn(jnp.asarray(u), jnp.asarray(v), jnp.asarray(muu), jnp.asarray(muv)))


def test_real_glue_matches_wrf_literal():
    ops = _operands()
    got = _run(*ops, wrf_real=True)
    ref = _wrf_literal(*ops)
    assert got.dtype == np.float32 and got.shape == ref.shape
    tol = 8 * np.finfo(np.float32).eps * _scale(*ops)
    assert np.all(np.abs(got.astype(np.float64) - ref) <= tol), float(np.max(np.abs(got - ref) / tol))
    # XLA:CPU contracts FMAs (E107), so ulps differ; measured max |diff| = 0.10 of this bound.


@pytest.mark.parametrize("mutant", ["swap_fnm_fnp", "no_curvature", "inverse_mapfac", "no_coriolis"])
def test_oracle_rejects_mutants(mutant, monkeypatch):
    u, v, muu, muv, m = _operands()
    ref = _wrf_literal(u, v, muu, muv, m)
    if mutant == "swap_fnm_fnp":
        m = types.SimpleNamespace(**{**vars(m), "fnm": m.fnp, "fnp": m.fnm})
    elif mutant == "inverse_mapfac":
        m = types.SimpleNamespace(**{**vars(m), "msftx": m.msfty, "msfty": m.msftx})
    elif mutant == "no_coriolis":
        m = types.SimpleNamespace(**{**vars(m), "e": np.zeros_like(m.e)})
    else:
        import gpuwrf.runtime.operational_mode as om
        monkeypatch.setattr(om, "_W_RERADIUS", 0.0)
    got = _run(u, v, muu, muv, m, wrf_real=True)
    tol = 8 * np.finfo(np.float32).eps * _scale(u, v, muu, muv, m)
    assert np.any(np.abs(got.astype(np.float64) - ref) > tol)
