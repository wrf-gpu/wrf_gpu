"""WRF revised-MM5/Jimenez surface layer for ``sf_sfclay_physics=1``.

This is the v0.6.0 per-scheme lane port of WRF's
``phys/module_sf_sfclayrev.F`` wrapper and ``physics_mmm/sf_sfclayrev.F90``
core. It implements the default ARW path used by the generated oracle:
surface fluxes enabled, no SCM-forced flux, no alternate water heat-transfer
coefficient option, and no land ``iz0tlnd`` thermal-roughness option.

The public column function returns the frozen S0 ``PhysicsStepResult``. Surface
flux handles are state replacements because the WRF surface-layer driver writes
them directly for the later PBL call.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from typing import Iterable, NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency
from gpuwrf.physics.surface_constants import (
    CP_D,
    CZO,
    EP1,
    EP2,
    G,
    KARMAN,
    MIN_WIND_M_S,
    OZO,
    P0_PA,
    PRT,
    R_D,
    R_D_OVER_CP,
    SALINITY_FACTOR,
    SFCLAYREV_TABLE_DZOL,
    SFCLAYREV_TABLE_N,
    SVP1_KPA,
    SVP2,
    SVP3_K,
    SVPT0_K,
    VCONVC,
    XKA,
    XLV,
    ZOLRI_BR_CAP,
    ZOLRI_MAX_ITER,
)


configure_jax_x64()


def _psim_stable_full(zolf):
    return -6.1 * jnp.log(zolf + (1.0 + zolf**2.5) ** (1.0 / 2.5))


def _psih_stable_full(zolf):
    return -5.3 * jnp.log(zolf + (1.0 + zolf**1.1) ** (1.0 / 1.1))


def _psim_unstable_full(zolf):
    x = (1.0 - 16.0 * zolf) ** 0.25
    psimk = (
        2.0 * jnp.log(0.5 * (1.0 + x))
        + jnp.log(0.5 * (1.0 + x * x))
        - 2.0 * jnp.arctan(x)
        + 2.0 * jnp.arctan(1.0)
    )
    ym = (1.0 - 10.0 * zolf) ** 0.33
    psimc = (
        1.5 * jnp.log((ym * ym + ym + 1.0) / 3.0)
        - jnp.sqrt(3.0) * jnp.arctan((2.0 * ym + 1.0) / jnp.sqrt(3.0))
        + 4.0 * jnp.arctan(1.0) / jnp.sqrt(3.0)
    )
    return (psimk + zolf * zolf * psimc) / (1.0 + zolf * zolf)


def _psih_unstable_full(zolf):
    y = (1.0 - 16.0 * zolf) ** 0.5
    psihk = 2.0 * jnp.log((1.0 + y) / 2.0)
    yh = (1.0 - 34.0 * zolf) ** 0.33
    psihc = (
        1.5 * jnp.log((yh * yh + yh + 1.0) / 3.0)
        - jnp.sqrt(3.0) * jnp.arctan((2.0 * yh + 1.0) / jnp.sqrt(3.0))
        + 4.0 * jnp.arctan(1.0) / jnp.sqrt(3.0)
    )
    return (psihk + zolf * zolf * psihc) / (1.0 + zolf * zolf)


_N = SFCLAYREV_TABLE_N
_np_z = np.arange(0, _N + 1, dtype=np.float64) * SFCLAYREV_TABLE_DZOL
_PSIM_STAB_TABLE = jnp.asarray(np.asarray(_psim_stable_full(_np_z)), dtype=jnp.float64)
_PSIH_STAB_TABLE = jnp.asarray(np.asarray(_psih_stable_full(_np_z)), dtype=jnp.float64)
_PSIM_UNSTAB_TABLE = jnp.asarray(np.asarray(_psim_unstable_full(-_np_z)), dtype=jnp.float64)
_PSIH_UNSTAB_TABLE = jnp.asarray(np.asarray(_psih_unstable_full(-_np_z)), dtype=jnp.float64)
del _np_z


def _table_lookup(coord, table, full_fn, zolf):
    nzol = jnp.floor(coord).astype(jnp.int32)
    rzol = coord - nzol.astype(coord.dtype)
    in_table = (nzol + 1) < _N
    nzol_c = jnp.clip(nzol, 0, _N - 1)
    base = table[nzol_c]
    nxt = table[jnp.clip(nzol_c + 1, 0, _N)]
    interp = base + rzol * (nxt - base)
    return jnp.where(in_table, interp, full_fn(zolf))


def _psim_stable(zolf):
    return _table_lookup(zolf * 100.0, _PSIM_STAB_TABLE, _psim_stable_full, zolf)


def _psih_stable(zolf):
    return _table_lookup(zolf * 100.0, _PSIH_STAB_TABLE, _psih_stable_full, zolf)


def _psim_unstable(zolf):
    return _table_lookup(-zolf * 100.0, _PSIM_UNSTAB_TABLE, _psim_unstable_full, zolf)


def _psih_unstable(zolf):
    return _table_lookup(-zolf * 100.0, _PSIH_UNSTAB_TABLE, _psih_unstable_full, zolf)


def _zolri2(zol2, ri2, z, z0):
    zol2 = jnp.where(zol2 * ri2 < 0.0, 0.0, zol2)
    zol20 = zol2 * z0 / z
    zol3 = zol2 + zol20
    log_term = jnp.log((z + z0) / z0)
    psix_u = log_term - (_psim_unstable(zol3) - _psim_unstable(zol20))
    psih_u = log_term - (_psih_unstable(zol3) - _psih_unstable(zol20))
    psix_s = log_term - (_psim_stable(zol3) - _psim_stable(zol20))
    psih_s = log_term - (_psih_stable(zol3) - _psih_stable(zol20))
    unstable = ri2 < 0.0
    psix = jnp.where(unstable, psix_u, psix_s)
    psih = jnp.where(unstable, psih_u, psih_s)
    return zol2 * psih / (psix * psix) - ri2


def _zolri(ri, z, z0):
    unstable = ri < 0.0
    x1 = jnp.where(unstable, -5.0, 0.0)
    x2 = jnp.where(unstable, 0.0, 5.0)
    fx1 = _zolri2(x1, ri, z, z0)
    fx2 = _zolri2(x2, ri, z, z0)
    zolri = x2

    def body(_, carry):
        x1, x2, fx1, fx2, zolri = carry
        active = (jnp.abs(x1 - x2) > 0.01) & (fx1 != fx2)
        use_x1 = jnp.abs(fx2) < jnp.abs(fx1)
        x1_new = x1 - fx1 / (fx2 - fx1) * (x2 - x1)
        x2_new = x2 - fx2 / (fx2 - fx1) * (x2 - x1)
        fx1_new = _zolri2(x1_new, ri, z, z0)
        fx2_new = _zolri2(x2_new, ri, z, z0)
        x1 = jnp.where(active & use_x1, x1_new, x1)
        fx1 = jnp.where(active & use_x1, fx1_new, fx1)
        x2 = jnp.where(active & (~use_x1), x2_new, x2)
        fx2 = jnp.where(active & (~use_x1), fx2_new, fx2)
        zolri = jnp.where(active, jnp.where(use_x1, x1_new, x2_new), zolri)
        return x1, x2, fx1, fx2, zolri

    return jax.lax.fori_loop(0, ZOLRI_MAX_ITER, body, (x1, x2, fx1, fx2, zolri))[4]


def _depth_dependent_z0(water_depth, z0, ust):
    effective_depth = jnp.where(water_depth < 10.0, 10.0, jnp.where(water_depth > 100.0, 100.0, water_depth))
    depth_b = (1.0 / 30.0) * jnp.log(1260.0 / effective_depth)
    out = jnp.exp((2.7 * ust - 1.8 / depth_b) / (ust + 0.17 / depth_b))
    return jnp.minimum(out, 0.1)


@jax.tree_util.register_pytree_node_class
class SfclayRevisedMM5ColumnState:
    """Pytree for independent lowest-level surface-layer columns."""

    __slots__ = (
        "u",
        "v",
        "temperature",
        "qv",
        "pressure",
        "dz",
        "psfc",
        "tsk",
        "xland",
        "lakemask",
        "mavail",
        "znt",
        "ust",
        "mol",
        "hfx",
        "qfx",
        "qsfc",
        "pblh",
        "dx",
        "water_depth",
    )

    def __init__(
        self,
        u,
        v,
        temperature,
        qv,
        pressure,
        dz,
        psfc,
        tsk,
        xland,
        lakemask,
        mavail,
        znt,
        ust,
        mol,
        hfx,
        qfx,
        qsfc,
        pblh,
        dx,
        water_depth,
    ) -> None:
        self.u = u
        self.v = v
        self.temperature = temperature
        self.qv = qv
        self.pressure = pressure
        self.dz = dz
        self.psfc = psfc
        self.tsk = tsk
        self.xland = xland
        self.lakemask = lakemask
        self.mavail = mavail
        self.znt = znt
        self.ust = ust
        self.mol = mol
        self.hfx = hfx
        self.qfx = qfx
        self.qsfc = qsfc
        self.pblh = pblh
        self.dx = dx
        self.water_depth = water_depth

    def replace(self, **updates) -> "SfclayRevisedMM5ColumnState":
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SfclayRevisedMM5ColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


def _leaves(state: SfclayRevisedMM5ColumnState) -> Iterable[jax.Array]:
    return (getattr(state, name) for name in SfclayRevisedMM5ColumnState.__slots__)


class SfclayRevisedMM5Output(NamedTuple):
    ust: object
    tstar: object
    qstar: object
    theta_flux: object
    qv_flux: object
    tau_u: object
    tau_v: object
    rhosfc: object
    fltv: object
    hfx: object
    qfx: object
    lh: object
    u10: object
    v10: object
    th2: object
    t2: object
    q2: object
    chs: object
    chs2: object
    cqs2: object
    flhc: object
    flqc: object
    ck: object
    cka: object
    cd: object
    cda: object
    qsfc: object
    qgh: object
    znt: object
    zol: object
    mol: object
    rmol: object
    regime: object
    psim: object
    psih: object
    fm: object
    fh: object
    br: object
    wspd: object
    gz1oz0: object


def _arr(value):
    return jnp.asarray(value, dtype=jnp.float64)


def sfclay_revised_mm5_run(
    state: SfclayRevisedMM5ColumnState,
    *,
    isfflx: bool = True,
    shalwater_z0: bool = False,
    isftcflx: int = 0,
    iz0tlnd: int = 0,
    scm_force_flux: bool = False,
) -> SfclayRevisedMM5Output:
    """Run the vectorized WRF ``sf_sfclayrev_run`` default path."""

    if isftcflx != 0:
        raise NotImplementedError("sfclayrev isftcflx options are not in the v0.6.0 option-1 oracle lane")
    if iz0tlnd != 0:
        raise NotImplementedError("sfclayrev iz0tlnd land thermal-roughness options are not in this lane")

    ux = _arr(state.u)
    vx = _arr(state.v)
    t1d = _arr(state.temperature)
    qx = _arr(state.qv)
    p1d = _arr(state.pressure)
    dz = _arr(state.dz)
    psfcpa = _arr(state.psfc)
    tsk = _arr(state.tsk)
    xland = _arr(state.xland)
    lakemask = _arr(state.lakemask)
    mavail = _arr(state.mavail)
    znt_in = jnp.maximum(_arr(state.znt), 1.0e-9)
    ust_in = jnp.maximum(_arr(state.ust), 0.0)
    mol_in = _arr(state.mol)
    hfx_prev = _arr(state.hfx)
    qfx_prev = _arr(state.qfx)
    qsfc_in = _arr(state.qsfc)
    pblh = jnp.maximum(_arr(state.pblh), 1.0)
    dx = jnp.maximum(_arr(state.dx), 1.0)
    water_depth = _arr(state.water_depth)

    psfc = psfcpa / 1000.0
    thgb = tsk * (P0_PA / psfcpa) ** R_D_OVER_CP
    pl = p1d / 1000.0
    thx = t1d * (P0_PA * 0.001 / pl) ** R_D_OVER_CP
    tvcon = 1.0 + EP1 * qx
    thvx = thx * tvcon
    scr4 = t1d * tvcon
    e1_ground = SVP1_KPA * jnp.exp(SVP2 * (tsk - SVPT0_K) / (tsk - SVP3_K))
    is_water = (xland - 1.5) >= 0.0
    is_land = ~is_water
    e1_ground = jnp.where(is_water & (lakemask == 0.0), e1_ground * SALINITY_FACTOR, e1_ground)
    qsfc = jnp.where(is_water | (qsfc_in <= 0.0), EP2 * e1_ground / (psfc - e1_ground), qsfc_in)
    e1_air = SVP1_KPA * jnp.exp(SVP2 * (t1d - SVPT0_K) / (t1d - SVP3_K))
    qgh = EP2 * e1_air / (pl - e1_air)
    cpm = CP_D * (1.0 + 0.8 * qx)

    rhox = psfc * 1000.0 / (R_D * scr4)
    za = 0.5 * dz
    govrth = G / thx
    gz1oz0 = jnp.log((za + znt_in) / znt_in)
    gz2oz0 = jnp.log((2.0 + znt_in) / znt_in)
    gz10oz0 = jnp.log((10.0 + znt_in) / znt_in)
    wind_raw = jnp.sqrt(ux * ux + vx * vx)
    tskv = thgb * (1.0 + EP1 * qsfc)
    dthvdz = thvx - tskv
    fluxc = jnp.maximum(hfx_prev / rhox / CP_D + EP1 * tskv * qfx_prev / rhox, 0.0)
    vconv_land = VCONVC * (G / tsk * pblh * fluxc) ** 0.33
    vconv_water = jnp.sqrt(jnp.maximum(-dthvdz, 0.0))
    vconv = jnp.where(is_land, vconv_land, vconv_water)
    vsgd = 0.32 * (jnp.maximum(dx / 5000.0 - 1.0, 0.0)) ** 0.33
    wspd = jnp.maximum(jnp.sqrt(wind_raw * wind_raw + vconv * vconv + vsgd * vsgd), MIN_WIND_M_S)
    br = govrth * za * dthvdz / (wspd * wspd)
    br = jnp.where(mol_in < 0.0, jnp.minimum(br, 0.0), br)

    zol_seed = jnp.zeros_like(br)
    zol_pos = _zolri(jnp.minimum(br, ZOLRI_BR_CAP), za, znt_in)
    zol_neg_capped = _zolri(jnp.maximum(br, -ZOLRI_BR_CAP), za, znt_in)
    zol_neg = jnp.where(ust_in < 0.001, br * gz1oz0, zol_neg_capped)
    zol = jnp.where(br > 0.0, zol_pos, jnp.where(br < 0.0, zol_neg, zol_seed))

    zolzz = zol * (za + znt_in) / za
    zol10 = zol * (10.0 + znt_in) / za
    zol2 = zol * (2.0 + znt_in) / za
    zol0 = zol * znt_in / za
    zl2 = 2.0 * zol / za
    zl10 = 10.0 * zol / za
    zl = jnp.where(is_land, 0.01 * zol / za, zol0)

    stable = br > 0.0
    neutral = br == 0.0
    unstable = br < 0.0
    zeros = jnp.zeros_like(br)

    psim_s = _psim_stable(zolzz) - _psim_stable(zol0)
    psih_s = _psih_stable(zolzz) - _psih_stable(zol0)
    psim10_s = _psim_stable(zol10) - _psim_stable(zol0)
    psih10_s = _psih_stable(zol10) - _psih_stable(zol0)
    psim2_s = _psim_stable(zol2) - _psim_stable(zol0)
    psih2_s = _psih_stable(zol2) - _psih_stable(zol0)
    pq_s = _psih_stable(zol) - _psih_stable(zl)
    pq2_s = _psih_stable(zl2) - _psih_stable(zl)
    pq10_s = _psih_stable(zl10) - _psih_stable(zl)

    psim_u = _psim_unstable(zolzz) - _psim_unstable(zol0)
    psih_u = _psih_unstable(zolzz) - _psih_unstable(zol0)
    psim10_u = _psim_unstable(zol10) - _psim_unstable(zol0)
    psih10_u = _psih_unstable(zol10) - _psih_unstable(zol0)
    psim2_u = _psim_unstable(zol2) - _psim_unstable(zol0)
    psih2_u = _psih_unstable(zol2) - _psih_unstable(zol0)
    pq_u = _psih_unstable(zol) - _psih_unstable(zl)
    pq2_u = _psih_unstable(zl2) - _psih_unstable(zl)
    pq10_u = _psih_unstable(zl10) - _psih_unstable(zl)

    psim = jnp.where(stable, psim_s, jnp.where(unstable, psim_u, zeros))
    psih = jnp.where(stable, psih_s, jnp.where(unstable, psih_u, zeros))
    psim10 = jnp.where(stable, psim10_s, jnp.where(unstable, psim10_u, zeros))
    psih10 = jnp.where(stable, psih10_s, jnp.where(unstable, psih10_u, zeros))
    psim2 = jnp.where(stable, psim2_s, jnp.where(unstable, psim2_u, zeros))
    psih2 = jnp.where(stable, psih2_s, jnp.where(unstable, psih2_u, zeros))
    pq = jnp.where(stable, pq_s, jnp.where(unstable, pq_u, zeros))
    pq2 = jnp.where(stable, pq2_s, jnp.where(unstable, pq2_u, zeros))
    pq10 = jnp.where(stable, pq10_s, jnp.where(unstable, pq10_u, zeros))

    psih = jnp.where(unstable, jnp.minimum(psih, 0.9 * gz1oz0), psih)
    psim = jnp.where(unstable, jnp.minimum(psim, 0.9 * gz1oz0), psim)
    psih2 = jnp.where(unstable, jnp.minimum(psih2, 0.9 * gz2oz0), psih2)
    psim10 = jnp.where(unstable, jnp.minimum(psim10, 0.9 * gz10oz0), psim10)
    psih10 = jnp.where(unstable, jnp.minimum(psih10, 0.9 * gz10oz0), psih10)
    regime = jnp.where(stable, 1.0, jnp.where(neutral, 3.0, 4.0))
    zol = jnp.where(neutral, 0.0, zol)
    rmol = zol / za

    dtg = thx - thgb
    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    psit = gz1oz0 - psih
    psit2 = gz2oz0 - psih2
    zq_base = jnp.where(is_water, znt_in, 0.01)
    psiq = jnp.log(KARMAN * ust_in * za / XKA + za / zq_base) - pq
    psiq2 = jnp.log(KARMAN * ust_in * 2.0 / XKA + 2.0 / zq_base) - pq2
    psiq10 = jnp.log(KARMAN * ust_in * 10.0 / XKA + 10.0 / zq_base) - pq10

    visc = (1.32 + 0.009 * (t1d - 273.15)) * 1.0e-5
    restar = ust_in * znt_in / visc
    z0t = jnp.clip(5.5e-5 * restar ** (-0.60), 2.0e-9, 1.0e-4)
    z0q = z0t

    def heat_psih(z0):
        zz = zol * (za + z0) / za
        z10 = zol * (10.0 + z0) / za
        z2 = zol * (2.0 + z0) / za
        zbase = zol * z0 / za
        p1_s = _psih_stable(zz) - _psih_stable(zbase)
        p10_s = _psih_stable(z10) - _psih_stable(zbase)
        p2_s = _psih_stable(z2) - _psih_stable(zbase)
        p1_u = _psih_unstable(zz) - _psih_unstable(zbase)
        p10_u = _psih_unstable(z10) - _psih_unstable(zbase)
        p2_u = _psih_unstable(z2) - _psih_unstable(zbase)
        p1 = jnp.where(zol > 0.0, p1_s, jnp.where(zol == 0.0, zeros, p1_u))
        p10 = jnp.where(zol > 0.0, p10_s, jnp.where(zol == 0.0, zeros, p10_u))
        p2 = jnp.where(zol > 0.0, p2_s, jnp.where(zol == 0.0, zeros, p2_u))
        return p1, p10, p2

    psih_t, psih10_t, psih2_t = heat_psih(z0t)
    psit_water = jnp.log((za + z0t) / z0t) - psih_t
    psit2_water = jnp.log((2.0 + z0t) / z0t) - psih2_t
    psih_q, psih10_q, psih2_q = heat_psih(z0q)
    psiq_water = jnp.log((za + z0q) / z0q) - psih_q
    psiq2_water = jnp.log((2.0 + z0q) / z0q) - psih2_q
    psiq10_water = jnp.log((10.0 + z0q) / z0q) - psih10_q
    psih = jnp.where(is_water, psih_q, psih)
    psih10 = jnp.where(is_water, psih10_q, psih10)
    psih2 = jnp.where(is_water, psih2_q, psih2)
    psit = jnp.where(is_water, psit_water, psit)
    psit2 = jnp.where(is_water, psit2_water, psit2)
    psiq = jnp.where(is_water, psiq_water, psiq)
    psiq2 = jnp.where(is_water, psiq2_water, psiq2)
    psiq10 = jnp.where(is_water, psiq10_water, psiq10)

    ck = (KARMAN / psix10) * (KARMAN / psiq10)
    cd = (KARMAN / psix10) * (KARMAN / psix10)
    cka = (KARMAN / psix) * (KARMAN / psiq)
    cda = (KARMAN / psix) * (KARMAN / psix)

    ust = 0.5 * ust_in + 0.5 * KARMAN * wspd / psix
    u10 = ux * psix10 / psix
    v10 = vx * psix10 / psix
    th2 = thgb + dtg * psit2 / psit
    q2 = qsfc + (qx - qsfc) * psiq2 / psiq
    t2 = th2 * (psfcpa / P0_PA) ** R_D_OVER_CP
    ust = jnp.where(is_land, jnp.maximum(ust, 0.001), ust)
    mol = KARMAN * dtg / psit / PRT
    denomq = psiq
    denomq2 = psiq2
    denomt2 = psit2
    fm = psix
    fh = psit

    hfx_zero = jnp.zeros_like(hfx_prev)
    qfx_zero = jnp.zeros_like(qfx_prev)
    hfx_work = jnp.where(scm_force_flux, hfx_prev, hfx_zero)
    qfx_work = jnp.where(scm_force_flux, qfx_prev, qfx_zero)

    znt_charnock = CZO * ust * ust / G + 0.11 * 1.5e-5 / ust
    znt_charnock = jnp.minimum(znt_charnock, 2.85e-3)
    znt_shallow = _depth_dependent_z0(water_depth, znt_in, ust)
    znt_water = jnp.where(shalwater_z0, znt_shallow, znt_charnock)
    znt = jnp.where(is_water & isfflx, znt_water, znt_in)

    flqc = rhox * mavail * ust * KARMAN / denomq
    flhc = jnp.where(jnp.abs(dtg) > 1.0e-5, cpm * rhox * ust * mol / dtg, 0.0)
    qfx_calc = flqc * (qsfc - qx)
    hfx_calc = flhc * (thgb - thx)
    compute_fluxes = bool(isfflx) and not bool(scm_force_flux)
    qfx = jnp.where(compute_fluxes, qfx_calc, qfx_work)
    hfx = jnp.where(compute_fluxes, hfx_calc, hfx_work)
    lh = XLV * qfx
    chs = ust * KARMAN / denomq
    cqs2 = ust * KARMAN / denomq2
    chs2 = ust * KARMAN / denomt2

    theta_flux = hfx / jnp.maximum(rhox * cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(rhox, 1.0e-12)
    wind_for_tau = jnp.maximum(wind_raw, MIN_WIND_M_S)
    tau_u = -(ust * ust) * ux / wind_for_tau
    tau_v = -(ust * ust) * vx / wind_for_tau
    fltv = (1.0 + EP1 * qx) * theta_flux + EP1 * thx * qv_flux
    tstar = mol
    qstar = -qfx / jnp.maximum(rhox * ust, 1.0e-12)

    return SfclayRevisedMM5Output(
        ust=ust,
        tstar=tstar,
        qstar=qstar,
        theta_flux=theta_flux,
        qv_flux=qv_flux,
        tau_u=tau_u,
        tau_v=tau_v,
        rhosfc=rhox,
        fltv=fltv,
        hfx=hfx,
        qfx=qfx,
        lh=lh,
        u10=u10,
        v10=v10,
        th2=th2,
        t2=t2,
        q2=q2,
        chs=chs,
        chs2=chs2,
        cqs2=cqs2,
        flhc=flhc,
        flqc=flqc,
        ck=ck,
        cka=cka,
        cd=cd,
        cda=cda,
        qsfc=qsfc,
        qgh=qgh,
        znt=znt,
        zol=zol,
        mol=mol,
        rmol=rmol,
        regime=regime,
        psim=psim,
        psih=psih,
        fm=fm,
        fh=fh,
        br=br,
        wspd=wspd,
        gz1oz0=gz1oz0,
    )


def step_sfclay_revised_mm5_column(
    u,
    v,
    temperature,
    qv,
    pressure,
    dz,
    *,
    psfc,
    tsk,
    xland,
    lakemask=0.0,
    mavail=1.0,
    znt=0.1,
    ust=0.1,
    mol=0.0,
    hfx=0.0,
    qfx=0.0,
    qsfc=-1.0,
    pblh=1000.0,
    dx=3000.0,
    water_depth=50.0,
    isfflx=True,
    shalwater_z0=False,
) -> PhysicsStepResult:
    """Return the frozen S0 adapter payload for one surface-layer call."""

    state = SfclayRevisedMM5ColumnState(
        u,
        v,
        temperature,
        qv,
        pressure,
        dz,
        psfc,
        tsk,
        xland,
        lakemask,
        mavail,
        znt,
        ust,
        mol,
        hfx,
        qfx,
        qsfc,
        pblh,
        dx,
        water_depth,
    )
    out = sfclay_revised_mm5_run(state, isfflx=isfflx, shalwater_z0=shalwater_z0)
    tendency = PhysicsTendency(
        state_replacements={
            "ustar": out.ust,
            "theta_flux": out.theta_flux,
            "qv_flux": out.qv_flux,
            "tau_u": out.tau_u,
            "tau_v": out.tau_v,
            "rhosfc": out.rhosfc,
            "fltv": out.fltv,
        },
        diagnostics={
            "HFX": out.hfx,
            "QFX": out.qfx,
            "LH": out.lh,
            "TSTAR": out.tstar,
            "QSTAR": out.qstar,
            "T2": out.t2,
            "TH2": out.th2,
            "Q2": out.q2,
            "U10": out.u10,
            "V10": out.v10,
            "ZNT": out.znt,
        },
    )
    tendency.validate_keys()
    diagnostics = PhysicsDiagnostics(
        surface_layer={
            "UST": out.ust,
            "TSTAR": out.tstar,
            "QSTAR": out.qstar,
            "T2": out.t2,
            "TH2": out.th2,
            "Q2": out.q2,
            "U10": out.u10,
            "V10": out.v10,
            "HFX": out.hfx,
            "QFX": out.qfx,
            "LH": out.lh,
            "ZNT": out.znt,
            "CHS": out.chs,
            "CHS2": out.chs2,
            "CQS2": out.cqs2,
            "FLHC": out.flhc,
            "FLQC": out.flqc,
            "CK": out.ck,
            "CKA": out.cka,
            "CD": out.cd,
            "CDA": out.cda,
            "QSFC": out.qsfc,
            "QGH": out.qgh,
            "MOL": out.mol,
            "RMOL": out.rmol,
            "ZOL": out.zol,
            "REGIME": out.regime,
            "PSIM": out.psim,
            "PSIH": out.psih,
            "FM": out.fm,
            "FH": out.fh,
            "BR": out.br,
            "WSPD": out.wspd,
            "GZ1OZ0": out.gz1oz0,
        }
    )
    return PhysicsStepResult(tendency=tendency, diagnostics=diagnostics)


__all__ = [
    "SfclayRevisedMM5ColumnState",
    "SfclayRevisedMM5Output",
    "sfclay_revised_mm5_run",
    "step_sfclay_revised_mm5_column",
]
