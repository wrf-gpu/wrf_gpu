"""JAX MYNN2.5 PBL column kernel for M5-S2."""

from __future__ import annotations

from functools import partial
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.mynn_constants import (
    A1,
    A2,
    B1,
    C1,
    C2,
    C5,
    CKMOD,
    CTAU,
    E1C,
    E2C,
    E3C,
    E4C,
    E5C,
    G1,
    G2,
    GRAV,
    GTR,
    KARMAN,
    LOCAL_ALP1,
    LOCAL_ALP2,
    LOCAL_ALP3,
    LOCAL_ELF_SOFT_MAX,
    LOCAL_ELT_MAX,
    LOCAL_ELT_MIN,
    MAX_PBLH_TRANSITION,
    MIN_PBLH,
    P608,
    QKEMIN,
    QMIN,
    SQFAC,
    TKE_EPS,
)
from gpuwrf.physics.mynn_surface_stub import surface_layer
from gpuwrf.physics.tridiagonal_solver import solve_tridiagonal


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class MynnPBLColumnState:
    """Pytree for a batch of independent MYNN PBL columns on mass levels."""

    __slots__ = ("u", "v", "w", "theta", "qv", "tke", "p", "rho", "dz", "km", "kh", "el")

    def __init__(self, u, v, w, theta, qv, tke, p, rho, dz, km, kh, el) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.tke = tke
        self.p = p
        self.rho = rho
        self.dz = dz
        self.km = km
        self.kh = kh
        self.el = el

    def replace(self, **updates) -> "MynnPBLColumnState":
        """Returns a same-layout pytree with named fields replaced."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents all column arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the state after a JAX transform."""

        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for cache/debug tests."""

        if not isinstance(other, MynnPBLColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        """Hashes small column states outside the physics hot path."""

        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


def _leaves(state: MynnPBLColumnState) -> Iterable[jax.Array]:
    """Centralizes leaf iteration for equality, hashing, and byte accounting."""

    return (getattr(state, name) for name in MynnPBLColumnState.__slots__)


def _clip_state(state: MynnPBLColumnState) -> MynnPBLColumnState:
    """Applies WRF-like lower bounds before column physics."""

    return state.replace(
        qv=jnp.maximum(state.qv, 0.0),
        tke=jnp.maximum(state.tke, TKE_EPS),
        rho=jnp.maximum(state.rho, 1.0e-4),
        dz=jnp.maximum(state.dz, 1.0),
    )


def _zero_like(x):
    return x * 0.0


def _zero_edge_like(x):
    """Returns one zero edge column matching the batch dimensions of `x`."""

    return x[..., :1] * 0.0


def _edge_heights(dz):
    """Builds WRF-style interface heights with `zw(kts)=0`."""

    return jnp.concatenate((_zero_edge_like(dz), jnp.cumsum(dz, axis=-1)), axis=-1)


def _wrf_zw(dz):
    """Returns the WRF `zw(kts:kte)` subset used by MYNN internals."""

    return _edge_heights(dz)[..., :-1]


def _interface_dz(dz):
    """Returns `0.5*(dz(k)+dz(k-1))` with the surface slot unused."""

    interior = 0.5 * (dz[..., 1:] + dz[..., :-1])
    return jnp.concatenate((dz[..., :1], interior), axis=-1)


def _virtual_potential(theta, qv):
    """Computes the dry virtual potential temperature used by MYNN gradients."""

    return theta * (1.0 + P608 * qv)


def _surface_terms(state: MynnPBLColumnState, surface=None):
    """Builds the surface fluxes used by the column kernel.

    When ``surface`` (a :class:`mynn_surface_stub.SurfaceFluxes`) is provided —
    the operational coupling path, where the real WRF revised surface layer
    (``physics.surface_layer.surface_layer``) ran first in ``surface_adapter`` —
    the kernel consumes those fluxes directly. When it is ``None`` (the analytic
    Tier-1/Tier-2 MYNN fixture path) the legacy neutral-bulk stub is used so the
    standalone MYNN fixture stays a pure PBL test. The surface→MYNN order and
    flux hand-off are the FROZEN Gate-1 coupling (coupler_interface.md §3)."""

    flux = surface_layer(state) if surface is None else surface
    wind = jnp.maximum(jnp.sqrt(state.u[..., 0] * state.u[..., 0] + state.v[..., 0] * state.v[..., 0]), 0.2)
    return flux, wind, flux.fltv, flux.rhosfc


def _flux_richardson(ri, ri1, ri2, ri3, ri4, rfc):
    """WRF-faithful flux Richardson formula with an unguarded SQRT radicand."""

    radicand = ri * ri - ri3 * ri + ri4
    return jnp.minimum(ri1 * (ri + ri2 - jnp.sqrt(radicand)), rfc)


def _mym_level2(state: MynnPBLColumnState):
    """WRF MYNN `mym_level2`: level-2 gradients and stability functions."""

    thlv = _virtual_potential(state.theta, state.qv)
    dzk = _interface_dz(state.dz)[..., 1:]
    duz = ((state.u[..., 1:] - state.u[..., :-1]) ** 2 + (state.v[..., 1:] - state.v[..., :-1]) ** 2) / (dzk * dzk)
    dtv_i = (thlv[..., 1:] - thlv[..., :-1]) / dzk
    gm_i = duz
    gh_i = -dtv_i * GTR

    ri = -gh_i / jnp.maximum(duz, 1.0e-10)
    a2fac = jnp.where(CKMOD == 1.0, 1.0 / (1.0 + jnp.maximum(ri, 0.0)), 1.0)
    rfc = G1 / (G1 + G2)
    f1 = B1 * (G1 - C1) + 3.0 * A2 * a2fac * (1.0 - C2) * (1.0 - C5) + 2.0 * A1 * (3.0 - 2.0 * C2)
    f2 = B1 * (G1 + G2) - 3.0 * A1 * (1.0 - C2)
    rf1 = B1 * (G1 - C1) / f1
    rf2 = B1 * G1 / f2
    smc = A1 / (A2 * a2fac) * f1 / f2
    shc = 3.0 * (A2 * a2fac) * (G1 + G2)

    ri1 = 0.5 / smc
    ri2 = rf1 * smc
    ri3 = 4.0 * rf2 * smc - 2.0 * ri2
    ri4 = ri2 * ri2
    rf = _flux_richardson(ri, ri1, ri2, ri3, ri4, rfc)
    sh_i = shc * (rfc - rf) / (1.0 - rf)
    sm_i = smc * (rf1 - rf) / (rf2 - rf) * sh_i

    z = _zero_edge_like(state.theta)
    dtv = jnp.concatenate((z, dtv_i), axis=-1)
    gm = jnp.concatenate((z, gm_i), axis=-1)
    gh = jnp.concatenate((z, gh_i), axis=-1)
    sm = jnp.concatenate((z, sm_i), axis=-1)
    sh = jnp.concatenate((z, sh_i), axis=-1)
    return dtv, gm, gh, sm, sh


def _first_or_fallback(candidates, fallback):
    """Returns the lowest-height finite candidate along the column."""

    value = jnp.min(candidates, axis=-1)
    return jnp.where(jnp.isfinite(value), value, fallback)


def _get_pblh(state: MynnPBLColumnState, qke):
    """Dry JAX transcription of WRF `GET_PBLH` for MYNN length-scale input."""

    thv = _virtual_potential(state.theta, state.qv)
    zw = _wrf_zw(state.dz)
    nz = state.dz.shape[-1]
    idx = jnp.arange(nz)
    lowest_200 = (idx >= 1) & (zw <= 200.0)
    minthv = jnp.min(jnp.where(lowest_200, thv, jnp.inf), axis=-1)
    minthv = jnp.where(jnp.isfinite(minthv), minthv, thv[..., 0])
    delt_thv = 1.0
    theta_hit = (idx >= 1) & (idx <= nz - 2) & (thv >= (minthv[..., None] + delt_thv))
    prev_thv = jnp.concatenate((thv[..., :1], thv[..., :-1]), axis=-1)
    prev_dz = jnp.concatenate((state.dz[..., :1], state.dz[..., :-1]), axis=-1)
    theta_candidate = zw - prev_dz * jnp.minimum((thv - (minthv[..., None] + delt_thv)) / jnp.maximum(thv - prev_thv, 1.0e-6), 1.0)
    theta_pblh = _first_or_fallback(jnp.where(theta_hit, theta_candidate, jnp.inf), zw[..., 1])

    maxqke = jnp.maximum(qke[..., 0], 0.0)
    tke_eps = jnp.maximum(maxqke / 40.0, 0.01)
    qtke = jnp.maximum(0.5 * qke, 0.0)
    qtkem1 = jnp.concatenate((qtke[..., :1], qtke[..., :-1]), axis=-1)
    tke_hit = (idx >= 1) & (idx <= nz - 2) & (qtke <= tke_eps[..., None])
    tke_candidate = zw - prev_dz * jnp.minimum((tke_eps[..., None] - qtke) / jnp.maximum(qtkem1 - qtke, 1.0e-6), 1.0)
    tke_pblh = jnp.maximum(_first_or_fallback(jnp.where(tke_hit, tke_candidate, jnp.inf), zw[..., 1]), zw[..., 1])
    tke_pblh = jnp.minimum(tke_pblh, theta_pblh + 350.0)
    tke_pblh = jnp.maximum(tke_pblh, jnp.maximum(theta_pblh - 350.0, 10.0))
    wt = 0.5 * jnp.tanh((theta_pblh - 200.0) / 400.0) + 0.5
    return jnp.where(maxqke <= tke_eps, theta_pblh, tke_pblh * (1.0 - wt) + theta_pblh * wt)


def _mym_length_option2(state: MynnPBLColumnState, qke, dtv, fltv):
    """WRF option-2 MYNN master length scale with EDMF/cloud terms disabled."""

    dz = state.dz
    zw = _wrf_zw(dz)
    dzk = _interface_dz(dz)
    nz = dz.shape[-1]
    idx = jnp.arange(nz)
    afk_i = dz[..., 1:] / (dz[..., 1:] + dz[..., :-1])
    abk_i = 1.0 - afk_i
    qkw_i = jnp.sqrt(jnp.maximum(qke[..., 1:] * abk_i + qke[..., :-1] * afk_i, QKEMIN))
    qkw = jnp.concatenate((jnp.sqrt(jnp.maximum(qke[..., :1], QKEMIN)), qkw_i), axis=-1)
    qtke = jnp.concatenate((jnp.maximum(0.5 * qke[..., :1], 0.5 * QKEMIN), 0.5 * qkw_i * qkw_i), axis=-1)

    pblh = _get_pblh(state, qke)
    pblh2 = jnp.maximum(pblh, MIN_PBLH)
    h1 = jnp.minimum(jnp.maximum(0.3 * pblh2, 300.0), MAX_PBLH_TRANSITION)
    h2 = 0.5 * h1
    pblh_plus_ent = jnp.maximum(pblh + h1, 100.0)

    integrate_mask = (idx >= 1) & (zw <= pblh_plus_ent[..., None])
    qdz = jnp.minimum(jnp.maximum(qkw - QMIN, 0.03), 30.0) * dzk
    elt_num = 1.0e-5 + jnp.sum(jnp.where(integrate_mask, qdz * zw, 0.0), axis=-1)
    elt_den = 1.0e-5 + jnp.sum(jnp.where(integrate_mask, qdz, 0.0), axis=-1)
    elt = jnp.minimum(jnp.maximum(LOCAL_ALP1 * elt_num / elt_den, LOCAL_ELT_MIN), LOCAL_ELT_MAX)
    vsc = (GTR * elt * jnp.maximum(fltv, 0.0)) ** (1.0 / 3.0)

    bv = jnp.maximum(jnp.sqrt(jnp.maximum(GTR * dtv, 0.0)), 0.001)
    stable = dtv > 0.0
    elb_mf_stable = (LOCAL_ALP2 * qkw / bv) * (1.0 + LOCAL_ALP3 * jnp.sqrt(vsc[..., None] / jnp.maximum(bv * elt[..., None], 1.0e-12)))
    wstar_stable = 1.25 * (GTR * pblh[..., None] * jnp.maximum(fltv[..., None], 1.0e-4)) ** (1.0 / 3.0)
    wt = 0.5 * jnp.tanh((zw - (pblh2 + h1)[..., None]) / h2[..., None]) + 0.5
    tau_stable = jnp.minimum(jnp.maximum(CTAU * wstar_stable / GRAV, 30.0), 150.0)
    tau_stable = tau_stable * (1.0 - wt) + 50.0 * wt
    elf_stable = jnp.minimum(tau_stable * jnp.sqrt(jnp.minimum(qtke, 40.0)), zw)

    wstar_unstable = wstar_stable
    tau_unstable = jnp.minimum(jnp.maximum(CTAU * wstar_unstable / GRAV, 50.0), 200.0)
    tau_unstable = tau_unstable * (1.0 - wt) + jnp.maximum(100.0, 0.25 * dzk) * wt
    elb_unstable = jnp.minimum(tau_unstable * jnp.sqrt(jnp.minimum(qtke, 40.0)), zw)
    elf_unstable = elb_unstable

    elb_mf = jnp.maximum(jnp.where(stable, elb_mf_stable, elb_unstable), 0.01)
    elf = jnp.where(stable, elf_stable, elf_unstable)
    elf = elf / (1.0 + elf / LOCAL_ELF_SOFT_MAX)
    els = KARMAN * zw
    el = jnp.sqrt((els * els) / (1.0 + (els * els) / (elt[..., None] * elt[..., None]) + (els * els) / (elb_mf * elb_mf)))
    el = el * (1.0 - wt) + elf * wt
    el = jnp.where(idx == 0, 0.0, el)
    return qkw, el, pblh


def _mym_turbulence(state: MynnPBLColumnState, qke, fltv):
    """WRF MYNN `mym_turbulence` dry level-2.5 path."""

    dtv, gm, gh, sm20_raw, sh20_raw = _mym_level2(state)
    qkw, el, pblh = _mym_length_option2(state, qke, dtv, fltv)
    dzk = _interface_dz(state.dz)
    elsq = el * el
    q3sq_initial = qkw * qkw
    q2sq = B1 * elsq * (sm20_raw * gm + sh20_raw * gh)
    sh = jnp.maximum(sh20_raw, 1.0e-5)
    sm = sm20_raw

    ri = -gh / jnp.maximum(gm, 1.0e-10)
    a2fac = jnp.where(CKMOD == 1.0, 1.0 / (1.0 + jnp.maximum(ri, 0.0)), 1.0)
    prlim = jnp.where(
        ri >= 1.0,
        7.0 * ri,
        jnp.where((ri >= 0.01) & (ri <= 1.0), 6.873 * ri + 1.0 / (6.873 * ri), 5.0),
    )
    gmel = gm * elsq
    ghel = gh * elsq
    q3sq = jnp.where((elsq > 0.0) & ((q3sq_initial / jnp.maximum(elsq, 1.0e-20)) < -gh), -elsq * gh, q3sq_initial)
    q3sq = jnp.maximum(q3sq, QKEMIN)

    qdiv = jnp.sqrt(jnp.maximum(q3sq / jnp.maximum(q2sq, 1.0e-20), 0.0))
    helfand = (q2sq > 0.0) & (q3sq < q2sq)
    sh_hl = sh * qdiv
    sm_hl = sm * qdiv

    e1_hl = q3sq - E1C * ghel * a2fac * qdiv * qdiv
    e2_hl = q3sq - E2C * ghel * a2fac * qdiv * qdiv
    e3_hl = e1_hl + E3C * ghel * a2fac * a2fac * qdiv * qdiv
    e4_hl = e1_hl - E4C * ghel * a2fac * qdiv * qdiv
    eden_hl = jnp.maximum(e2_hl * e4_hl + e3_hl * E5C * gmel * qdiv * qdiv, 1.0e-20)
    del e1_hl, e2_hl, e3_hl, e4_hl, eden_hl

    e1 = q3sq - E1C * ghel * a2fac
    e2 = q3sq - E2C * ghel * a2fac
    e3 = e1 + E3C * ghel * a2fac * a2fac
    e4 = e1 - E4C * ghel * a2fac
    eden = jnp.maximum(e2 * e4 + e3 * E5C * gmel, 1.0e-20)
    sm25 = q3sq * A1 * (e3 - 3.0 * C1 * e4) / eden
    sh25 = q3sq * (A2 * a2fac) * (e2 + 3.0 * C1 * E5C * gmel) / eden
    sm = jnp.where(helfand, sm_hl, sm25)
    sh = jnp.where(helfand, sh_hl, sh25)
    sh = jnp.minimum(jnp.maximum(sh, 0.0), 4.0)
    sm = jnp.minimum(sm, prlim * jnp.maximum(sh, 0.02))
    sm = jnp.maximum(sm, 0.0)

    elq = el * qkw
    pdk = elq * (sm * gm + sh * gh)
    pdt = elq * sh * 0.0
    pdq = elq * sh * 0.0
    pdc = elq * sh * 0.0
    tcd = _zero_like(pdk)
    qcd = _zero_like(pdk)
    dfm = jnp.where(dzk > 0.0, elq * sm / dzk, 0.0)
    dfh = jnp.where(dzk > 0.0, elq * sh / dzk, 0.0)
    dfq = dfm
    qshear = elq * sm * gm
    qbuoy = elq * sh * gh
    zeros = _zero_edge_like(state.tke)
    dfm = jnp.concatenate((zeros, dfm[..., 1:]), axis=-1)
    dfh = jnp.concatenate((zeros, dfh[..., 1:]), axis=-1)
    dfq = jnp.concatenate((zeros, dfq[..., 1:]), axis=-1)
    pdk = jnp.concatenate((zeros, pdk[..., 1:]), axis=-1)
    qshear = jnp.concatenate((zeros, qshear[..., 1:]), axis=-1)
    qbuoy = jnp.concatenate((zeros, qbuoy[..., 1:]), axis=-1)
    return {
        "qkw": qkw,
        "el": el,
        "pblh": pblh,
        "dfm": dfm,
        "dfh": dfh,
        "dfq": dfq,
        "pdk": pdk,
        "pdt": pdt,
        "pdq": pdq,
        "pdc": pdc,
        "tcd": tcd,
        "qcd": qcd,
        "qshear": qshear,
        "qbuoy": qbuoy,
    }


def _rho_interfaces(state: MynnPBLColumnState, diffusivity):
    """Builds WRF rho-weighted interface diffusion factors."""

    dz = state.dz
    rho = state.rho
    rhoz_i = (rho[..., 1:] * dz[..., :-1] + rho[..., :-1] * dz[..., 1:]) / (dz[..., :-1] + dz[..., 1:])
    rhoz_i = jnp.maximum(rhoz_i, 1.0e-4)
    rhoz = jnp.concatenate((rho[..., :1], rhoz_i, rhoz_i[..., -1:]), axis=-1)
    diff_ext = jnp.concatenate((diffusivity, diffusivity[..., -1:]), axis=-1)
    return rhoz * diff_ext


def _solve_tridiagonal(a, b, c, d):
    """Uses XLA's tridiagonal primitive for the production vertical solves."""

    return solve_tridiagonal(a, b, c, d)


def _mym_predict_qke(state: MynnPBLColumnState, qke, turb, dt, ustar, flux):
    """WRF `mym_predict` dry level-2.5 qke equation."""

    del flux
    dz = state.dz
    rho = state.rho
    dtz = dt / dz
    rhoinv = 1.0 / jnp.maximum(rho, 1.0e-4)
    qkw_mass = jnp.sqrt(jnp.maximum(qke, 0.0))
    df3q = SQFAC * turb["dfq"]
    kqdz = _rho_interfaces(state, df3q)
    vkz = KARMAN * 0.5 * dz[..., 0]
    pdk1 = 2.0 * ustar * ustar * ustar / jnp.maximum(vkz, 1.0e-6)
    pdk = jnp.concatenate(((pdk1 - turb["pdk"][..., 1])[..., None], turb["pdk"][..., 1:]), axis=-1)

    el_pair = 0.5 * (turb["el"][..., 1:] + turb["el"][..., :-1])
    b1l = jnp.maximum(B1 * el_pair, 1.0e-6)
    bp_i = 2.0 * qkw_mass[..., :-1] / b1l
    rp_i = pdk[..., 1:] + pdk[..., :-1]
    lower = -dtz[..., :-1] * kqdz[..., :-2] * rhoinv[..., :-1]
    diag = 1.0 + dtz[..., :-1] * (kqdz[..., :-2] + kqdz[..., 1:-1]) * rhoinv[..., :-1] + bp_i * dt
    upper = -dtz[..., :-1] * kqdz[..., 1:-1] * rhoinv[..., :-1]
    rhs = rp_i * dt + qke[..., :-1]

    a = jnp.concatenate((lower, _zero_edge_like(qke)), axis=-1)
    b = jnp.concatenate((diag, jnp.ones_like(qke[..., -1:])), axis=-1)
    c = jnp.concatenate((upper, _zero_edge_like(qke)), axis=-1)
    d = jnp.concatenate((rhs, qke[..., -1:]), axis=-1)
    qke_new = jnp.minimum(jnp.maximum(_solve_tridiagonal(a, b, c, d), QKEMIN), 150.0)

    tke_up = 0.5 * qke_new
    qwt_bottom = (kqdz[..., 1] * (tke_up[..., 1] - tke_up[..., 0]) - kqdz[..., 0] * tke_up[..., 0]) / dz[..., 0]
    qwt_mid = (
        kqdz[..., 2:-1] * (tke_up[..., 2:] - tke_up[..., 1:-1])
        - kqdz[..., 1:-2] * (tke_up[..., 1:-1] - tke_up[..., :-2])
    ) / dz[..., 1:-1]
    qwt_top = (-kqdz[..., -2] * (tke_up[..., -1] - tke_up[..., -2])) / dz[..., -1]
    qwt = jnp.concatenate((qwt_bottom[..., None], qwt_mid, qwt_top[..., None]), axis=-1)
    bp = jnp.concatenate((bp_i, _zero_edge_like(qke)), axis=-1)
    qdiss = bp * tke_up
    return qke_new, qwt, qdiss, pdk


def _diffusion_solve_with_surface(x, diffusivity, state, dt, bottom_rhs, bottom_drag=0.0):
    """WRF `mynn_tendencies` tridiagonal coefficients for one dry scalar."""

    dz = state.dz
    dtz = dt / dz
    rhoinv = 1.0 / jnp.maximum(state.rho, 1.0e-4)
    kdz = _rho_interfaces(state, diffusivity)
    bottom_drag = jnp.zeros_like(bottom_rhs) + bottom_drag
    lower0 = -dtz[..., :1] * kdz[..., :1] * rhoinv[..., :1]
    diag0 = 1.0 + dtz[..., :1] * (kdz[..., 1:2] + kdz[..., :1] + bottom_drag[..., None]) * rhoinv[..., :1]
    upper0 = -dtz[..., :1] * kdz[..., 1:2] * rhoinv[..., :1]
    rhs0 = x[..., :1] + bottom_rhs[..., None]

    lower_i = -dtz[..., 1:-1] * kdz[..., 1:-2] * rhoinv[..., 1:-1]
    diag_i = 1.0 + dtz[..., 1:-1] * (kdz[..., 1:-2] + kdz[..., 2:-1]) * rhoinv[..., 1:-1]
    upper_i = -dtz[..., 1:-1] * kdz[..., 2:-1] * rhoinv[..., 1:-1]
    rhs_i = x[..., 1:-1]

    a = jnp.concatenate((lower0, lower_i, _zero_edge_like(x)), axis=-1)
    b = jnp.concatenate((diag0, diag_i, jnp.ones_like(x[..., -1:])), axis=-1)
    c = jnp.concatenate((upper0, upper_i, _zero_edge_like(x)), axis=-1)
    d = jnp.concatenate((rhs0, rhs_i, x[..., -1:]), axis=-1)
    return _solve_tridiagonal(a, b, c, d)


def _apply_mean_tendencies(state: MynnPBLColumnState, turb, dt, flux, wind, rhosfc):
    """Applies WRF-style dry U/V/theta/qv implicit tendency solves."""

    rhoinv0 = 1.0 / jnp.maximum(state.rho[..., 0], 1.0e-4)
    dtz0 = dt / state.dz[..., 0]
    bottom_drag = rhosfc * flux.ustar * flux.ustar / wind
    u = _diffusion_solve_with_surface(state.u, turb["dfm"], state, dt, jnp.zeros_like(wind), bottom_drag)
    v = _diffusion_solve_with_surface(state.v, turb["dfm"], state, dt, jnp.zeros_like(wind), bottom_drag)
    theta_rhs = dtz0 * rhosfc * flux.theta_flux * rhoinv0
    qv_flux = jnp.maximum(flux.qv_flux, jnp.minimum(0.9 * state.qv[..., 0] - 1.0e-8, 0.0) / jnp.maximum(dtz0, 1.0e-12))
    qv_rhs = dtz0 * rhosfc * qv_flux * rhoinv0
    theta = _diffusion_solve_with_surface(state.theta, turb["dfh"], state, dt, theta_rhs)
    qv = _diffusion_solve_with_surface(state.qv, turb["dfh"], state, dt, qv_rhs)
    return u, v, theta, jnp.maximum(qv, 0.0)


def _retrieve_exchange_coeffs(state: MynnPBLColumnState, turb):
    """WRF `retrieve_exchange_coeffs`: converts `dfm/dfh` back to K units."""

    dzk = _interface_dz(state.dz)
    km = turb["dfm"] * dzk
    kh = turb["dfh"] * dzk
    km = jnp.concatenate((_zero_edge_like(km), km[..., 1:]), axis=-1)
    kh = jnp.concatenate((_zero_edge_like(kh), kh[..., 1:]), axis=-1)
    return km, kh


def _step_mynn_pbl_impl_with_pblh(state: MynnPBLColumnState, dt: float, debug: bool, surface=None):
    """Unjitted implementation; returns the advanced state and the MYNN PBLH."""

    state = _clip_state(state)
    flux, wind, fltv, rhosfc = _surface_terms(state, surface)
    qke = 2.0 * state.tke
    turb = _mym_turbulence(state, qke, fltv)
    qke_new, _qwt, qdiss, _pdk = _mym_predict_qke(state, qke, turb, dt, flux.ustar, flux)
    u, v, theta, qv = _apply_mean_tendencies(state, turb, dt, flux, wind, rhosfc)
    km, kh = _retrieve_exchange_coeffs(state, turb)
    tke = 0.5 * qke_new

    u = assert_finite(u, "mynn_u", enabled=debug)
    v = assert_finite(v, "mynn_v", enabled=debug)
    theta = assert_finite(theta, "mynn_theta", enabled=debug)
    qv = assert_physical_bounds(qv, 0.0, 0.1, "mynn_qv", enabled=debug)
    tke = assert_physical_bounds(tke, TKE_EPS, 200.0, "mynn_tke", enabled=debug)
    km = assert_finite(km, "mynn_km", enabled=debug)
    kh = assert_finite(kh, "mynn_kh", enabled=debug)
    el = assert_finite(turb["el"], "mynn_el", enabled=debug)
    del qdiss
    return state.replace(u=u, v=v, theta=theta, qv=qv, tke=tke, km=km, kh=kh, el=el), turb["pblh"]


def _step_mynn_pbl_impl(state: MynnPBLColumnState, dt: float, debug: bool, surface=None) -> MynnPBLColumnState:
    """Unjitted implementation shared by production and stripped entry points."""

    next_state, _pblh = _step_mynn_pbl_impl_with_pblh(state, dt, debug, surface)
    return next_state


def _mynn_budget_diagnostics(state: MynnPBLColumnState, dt: float, surface=None):
    """Returns one-step budget diagnostics for Tier-2 validation."""

    state = _clip_state(state)
    flux, wind, fltv, rhosfc = _surface_terms(state, surface)
    qke = 2.0 * state.tke
    turb = _mym_turbulence(state, qke, fltv)
    qke_new, qwt, qdiss, pdk = _mym_predict_qke(state, qke, turb, dt, flux.ustar, flux)
    u, v, theta, qv = _apply_mean_tendencies(state, turb, dt, flux, wind, rhosfc)
    km, kh = _retrieve_exchange_coeffs(state, turb)
    next_state = state.replace(u=u, v=v, theta=theta, qv=qv, tke=0.5 * qke_new, km=km, kh=kh, el=turb["el"])
    drag = rhosfc * flux.ustar * flux.ustar / wind
    kmdz = _rho_interfaces(state, turb["dfm"])
    khdz = _rho_interfaces(state, turb["dfh"])
    prod_mass = jnp.concatenate((0.5 * (pdk[..., :-1] + pdk[..., 1:]), _zero_edge_like(pdk)), axis=-1)
    return next_state, {
        "surface_u": -drag * u[..., 0],
        "surface_v": -drag * v[..., 0],
        "surface_theta": rhosfc * flux.theta_flux,
        "surface_qv": rhosfc * flux.qv_flux,
        "top_u": kmdz[..., -2] * (u[..., -1] - u[..., -2]),
        "top_v": kmdz[..., -2] * (v[..., -1] - v[..., -2]),
        "top_theta": khdz[..., -2] * (theta[..., -1] - theta[..., -2]),
        "top_qv": khdz[..., -2] * (qv[..., -1] - qv[..., -2]),
        "tke_production": prod_mass,
        "tke_transport": qwt,
        "tke_dissipation": qdiss,
    }


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_mynn_pbl_column(
    state: MynnPBLColumnState, dt: float, *, debug: bool = False, surface=None
) -> MynnPBLColumnState:
    """Advances one fused MYNN2.5 column step.

    ``surface`` (a :class:`SurfaceFluxes` pytree) is the operational coupling
    hand-off from the WRF revised surface layer; ``None`` uses the standalone
    neutral-bulk stub for the analytic MYNN fixture."""

    return _step_mynn_pbl_impl(state, dt, debug, surface)


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_mynn_pbl_column_with_pblh(
    state: MynnPBLColumnState, dt: float, *, debug: bool = False, surface=None
):
    """Advance one MYNN column step and also return the diagnosed PBL height.

    Used by the operational ``mynn_adapter`` to emit the PBLH operational
    diagnostic (coupler_interface.md §4) without adding a prognostic State leaf."""

    return _step_mynn_pbl_impl_with_pblh(state, dt, debug, surface)


@partial(jax.jit, static_argnames=("dt",))
def step_mynn_pbl_column_debug_stripped(state: MynnPBLColumnState, dt: float) -> MynnPBLColumnState:
    """Hand-stripped sibling used for the debug-vs-production HLO identity proof."""

    return _step_mynn_pbl_impl(state, dt, False)
