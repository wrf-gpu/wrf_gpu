"""WRF Pleim-Xiu surface layer for ``sf_sfclay_physics=7``.

This is a vectorized JAX transcription of the WRF v4 ARW routine
``phys/module_sf_pxsfclay.F:PXSFCLAY1D`` from the pristine WRF tree on this
workstation.  The routine is a Monin-Obukhov surface-layer parameterization used
with the Pleim-Xiu land-surface model and ACM2 PBL.

The public adapter returns the frozen v0.6.0 ``PhysicsStepResult`` payload:
surface-layer quantities that WRF writes in-place are exposed as state
replacements, and WRF diagnostics/carry-style arrays are exposed through the
surface-layer diagnostic mapping.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency
from gpuwrf.physics.surface_constants import (
    CP_D,
    EP1,
    EP2,
    G,
    KARMAN,
    MIN_WIND_M_S,
    P0_PA,
    R_D,
    R_D_OVER_CP,
    SVP1_KPA,
    SVP2,
    SVP3_K,
    SVPT0_K,
    VCONVC,
    XLV,
)


# module_sf_pxsfclay.F module parameters.
PX_RICRIT = 0.25
PX_BETAH = 5.0
PX_BETAM = 5.0
PX_BM = 13.0
PX_BH = 15.7
PX_GAMAM = 19.3
PX_GAMAH = 11.6
PX_PR0 = 0.95
PX_CZO = 0.032
PX_OZO = 1.0e-4


@jax.tree_util.register_pytree_node_class
class PleimXiuSfclayColumnState:
    """Pytree holding independent lowest-level Pleim-Xiu columns."""

    __slots__ = (
        "u",
        "v",
        "temperature",
        "theta",
        "qv",
        "pressure",
        "dz",
        "psfc",
        "tsk",
        "xland",
        "mavail",
        "znt",
        "ust",
        "mol",
        "hfx",
        "qfx",
        "qsfc",
        "pblh",
        "dx",
        "itimestep",
    )

    def __init__(
        self,
        u,
        v,
        temperature,
        theta,
        qv,
        pressure,
        dz,
        psfc,
        tsk,
        xland,
        mavail,
        znt,
        ust,
        mol,
        hfx,
        qfx,
        qsfc,
        pblh,
        dx,
        itimestep,
    ) -> None:
        self.u = u
        self.v = v
        self.temperature = temperature
        self.theta = theta
        self.qv = qv
        self.pressure = pressure
        self.dz = dz
        self.psfc = psfc
        self.tsk = tsk
        self.xland = xland
        self.mavail = mavail
        self.znt = znt
        self.ust = ust
        self.mol = mol
        self.hfx = hfx
        self.qfx = qfx
        self.qsfc = qsfc
        self.pblh = pblh
        self.dx = dx
        self.itimestep = itimestep

    def replace(self, **updates) -> "PleimXiuSfclayColumnState":
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
        if not isinstance(other, PleimXiuSfclayColumnState):
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


def _leaves(state: PleimXiuSfclayColumnState) -> Iterable[jax.Array]:
    return (getattr(state, name) for name in PleimXiuSfclayColumnState.__slots__)


class PleimXiuSfclayOutput(NamedTuple):
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
    cpm: object
    flhc: object
    flqc: object
    cqs: object
    qsfc: object
    qgh: object
    znt: object
    zol: object
    mol: object
    rmol: object
    regime: object
    psim: object
    psih: object
    br: object
    wspd: object
    gz1oz0: object
    ra: object
    rbh: object
    rbw: object
    molength: object


def _arr(value):
    return jnp.asarray(value, dtype=jnp.float64)


def _svp_ground_px(tsk):
    """Saturation vapor pressure over ice/water, module_sf_pxsfclay.F:360-368."""

    ice = SVP1_KPA * jnp.exp(
        4648.0 * (1.0 / SVPT0_K - 1.0 / tsk)
        - 11.64 * jnp.log(SVPT0_K / tsk)
        + 0.02265 * (SVPT0_K - tsk)
    )
    water = SVP1_KPA * jnp.exp(SVP2 * (tsk - SVPT0_K) / (tsk - SVP3_K))
    return jnp.where(tsk < SVPT0_K, ice, water)


def _svp_water(temp):
    return SVP1_KPA * jnp.exp(SVP2 * (temp - SVPT0_K) / (temp - SVP3_K))


def _pxsfclay_regime(br, gz1oz0):
    ricriti = 1.0 / PX_RICRIT
    ricut = 1.0 / (ricriti + gz1oz0)

    zoll_very_stable = br * gz1oz0 / (1.0 - ricriti * ricut)
    psim_very_stable = 1.0 - PX_BETAM - zoll_very_stable
    psih_very_stable = 1.0 - PX_BETAH - zoll_very_stable

    zoll_stable = br * gz1oz0 / (1.0 - ricriti * br)
    psim_stable = -PX_BETAM * zoll_stable
    psih_stable = -PX_BETAH * zoll_stable

    am = 0.031 + 0.276 * jnp.log(gz1oz0)
    ah = 0.04 + 0.355 * jnp.log(gz1oz0)
    sqlnzz0 = jnp.sqrt(gz1oz0)
    psim_unstable = am * jnp.log(1.0 - PX_BM * sqlnzz0 * br)
    psih_unstable = ah * jnp.log(1.0 - PX_BH * sqlnzz0 * br)

    very_stable = br >= ricut
    stable = (br >= 0.0) & (~very_stable)
    regime = jnp.where(very_stable, 1.0, jnp.where(stable, 2.0, 3.0))
    psim = jnp.where(very_stable, psim_very_stable, jnp.where(stable, psim_stable, psim_unstable))
    psih = jnp.where(very_stable, psih_very_stable, jnp.where(stable, psih_stable, psih_unstable))
    return regime, psim, psih, ricut


def pxsfclay_run(state: PleimXiuSfclayColumnState, *, isfflx: bool = True) -> PleimXiuSfclayOutput:
    """Run vectorized WRF ``PXSFCLAY1D`` default path.

    ``isfflx`` is accepted for adapter symmetry with other surface-layer ports.
    WRF's Pleim-Xiu routine receives this flag but does not branch on it.
    """

    del isfflx
    ux = _arr(state.u)
    vx = _arr(state.v)
    t1d = _arr(state.temperature)
    theta = _arr(state.theta)
    qx = _arr(state.qv)
    p1d = _arr(state.pressure)
    dz = _arr(state.dz)
    psfcpa = _arr(state.psfc)
    tsk = _arr(state.tsk)
    xland = _arr(state.xland)
    mavail = _arr(state.mavail)
    znt_in = _arr(state.znt)
    ust_in = _arr(state.ust)
    hfx_prev = _arr(state.hfx)
    qfx_prev = _arr(state.qfx)
    qsfc_in = _arr(state.qsfc)
    pblh = _arr(state.pblh)
    dx = _arr(state.dx)
    itimestep = _arr(state.itimestep)

    psfc = psfcpa / 1000.0
    tvcon = 1.0 + EP1 * qx
    thetav1 = theta * tvcon
    rhox = psfcpa / (R_D * t1d * tvcon)

    e1_ground = _svp_ground_px(tsk)
    recompute_qsfc = (xland > 1.5) | (qsfc_in <= 0.0) | (itimestep == 1.0)
    qsfc = jnp.where(recompute_qsfc, EP2 * e1_ground / (psfc - e1_ground), qsfc_in)

    e1_air = _svp_water(t1d)
    pl = p1d / 1000.0
    qgh = EP2 * e1_air / (pl - e1_air)
    cpm = CP_D * (1.0 + 0.8 * qx)

    tv0 = tsk * (1.0 + EP1 * qsfc)
    cpot = (100.0 / psfc) ** R_D_OVER_CP
    th0 = tv0 * cpot
    thetag = cpot * tsk

    za = 0.5 * dz
    ws = jnp.sqrt(ux * ux + vx * vx)
    gz1oz0_initial = jnp.log(za / znt_in)
    dthvdz = thetav1 - th0
    fluxc = jnp.maximum(hfx_prev / rhox / CP_D + EP1 * th0 * qfx_prev / rhox, 0.0)
    vconv = VCONVC * (G / tsk * pblh * fluxc) ** 0.33
    vsgd = 0.32 * jnp.maximum(dx / 5000.0 - 1.0, 0.0) ** 0.33
    wspd = jnp.maximum(jnp.sqrt(ws * ws + vconv * vconv + vsgd * vsgd), MIN_WIND_M_S)
    govrth = G / theta
    br = govrth * za * dthvdz / (wspd * wspd)

    regime, psim, psih, _ricut = _pxsfclay_regime(br, gz1oz0_initial)

    dtg = theta - thetag
    psix_initial = gz1oz0_initial - psim
    ust_first = 0.5 * ust_in + 0.5 * KARMAN * wspd / psix_initial
    is_water = xland >= 1.5
    znt_water = PX_CZO * ust_first * ust_first / G + PX_OZO
    znt = jnp.where(is_water, znt_water, znt_in)
    gz1oz0 = jnp.where(is_water, jnp.log(za / znt), gz1oz0_initial)
    psix = gz1oz0 - psim
    ust = jnp.where(is_water, KARMAN * wspd / psix, ust_first)

    ra = PX_PR0 * (gz1oz0 - psih) / (KARMAN * ust)
    rbh = 5.0 / ust
    rbw = 4.503 / ust
    chs = 1.0 / (ra + rbh)
    cqs = 1.0 / (ra + rbw)
    mol = dtg * chs / ust
    tstv = (thetav1 - th0) * chs / ust
    tstv = jnp.where(jnp.abs(tstv) < 1.0e-5, 1.0e-5, tstv)
    molength = thetav1 * ust * ust / (KARMAN * G * tstv)

    xmol = jnp.where(molength > 0.0, jnp.maximum(molength, 2.0), molength)
    rmol = 1.0 / xmol
    zol = za * rmol
    zobol = 1.5 * rmol
    z10ol = 10.0 * rmol
    zntol = znt * rmol

    ynt = (1.0 - PX_GAMAH * zntol) ** 0.5
    yob = (1.0 - PX_GAMAH * zobol) ** 0.5
    psih2_unstable = 2.0 * jnp.log((yob + 1.0) / (ynt + 1.0))
    x1 = (1.0 - PX_GAMAM * z10ol) ** 0.25
    x2 = (1.0 - PX_GAMAM * zntol) ** 0.25
    psim10_unstable = (
        2.0 * jnp.log((1.0 + x1) / (1.0 + x2))
        + jnp.log((1.0 + x1 * x1) / (1.0 + x2 * x2))
        - 2.0 * jnp.arctan(x1)
        + 2.0 * jnp.arctan(x2)
    )

    psih2_stable_near = -PX_BETAH * (zobol - zntol)
    psih2_stable_far = 1.0 - PX_BETAH - (zobol - zntol)
    psih2_stable = jnp.where((zobol - zntol) <= 1.0, psih2_stable_near, psih2_stable_far)
    psim10_stable_near = -PX_BETAM * (z10ol - zntol)
    psim10_stable_far = 1.0 - PX_BETAM - (z10ol - zntol)
    psim10_stable = jnp.where((z10ol - zntol) <= 1.0, psim10_stable_near, psim10_stable_far)

    unstable_mol = xmol < 0.0
    psih2 = jnp.where(unstable_mol, psih2_unstable, psih2_stable)
    psim10 = jnp.where(unstable_mol, psim10_unstable, psim10_stable)

    g2oz0 = jnp.log(1.5 / znt)
    g10oz0 = jnp.log(10.0 / znt)
    ra2 = PX_PR0 * (g2oz0 - psih2) / (KARMAN * ust)
    chs2 = 1.0 / (ra2 + rbh)
    cqs2 = 1.0 / (ra2 + rbw)
    u10 = ux * (g10oz0 - psim10) / psix
    v10 = vx * (g10oz0 - psim10) / psix

    flhc = cpm * rhox * chs
    flqc = rhox * cqs * mavail
    qfx = jnp.maximum(flqc * (qsfc - qx), 0.0)
    lh = XLV * qfx
    hfx_raw = -flhc * dtg
    hfx = jnp.where(is_water, hfx_raw, jnp.maximum(hfx_raw, -250.0))

    theta_flux = hfx / jnp.maximum(rhox * cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(rhox, 1.0e-12)
    wind_for_tau = jnp.maximum(ws, MIN_WIND_M_S)
    tau_u = -(ust * ust) * ux / wind_for_tau
    tau_v = -(ust * ust) * vx / wind_for_tau
    fltv = (1.0 + EP1 * qx) * theta_flux + EP1 * theta * qv_flux
    qstar = -qfx / jnp.maximum(rhox * ust, 1.0e-12)

    rho_diag = psfcpa / (R_D * tsk)
    q2 = jnp.where(cqs2 < 1.0e-5, qsfc, qsfc - qfx / jnp.maximum(rho_diag * cqs2, 1.0e-12))
    t2 = jnp.where(chs2 < 1.0e-5, tsk, tsk - hfx / jnp.maximum(rho_diag * CP_D * chs2, 1.0e-12))
    th2 = t2 * (P0_PA / psfcpa) ** R_D_OVER_CP

    return PleimXiuSfclayOutput(
        ust=ust,
        tstar=mol,
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
        cpm=cpm,
        flhc=flhc,
        flqc=flqc,
        cqs=cqs,
        qsfc=qsfc,
        qgh=qgh,
        znt=znt,
        zol=zol,
        mol=mol,
        rmol=rmol,
        regime=regime,
        psim=psim,
        psih=psih,
        br=br,
        wspd=wspd,
        gz1oz0=gz1oz0,
        ra=ra,
        rbh=rbh,
        rbw=rbw,
        molength=molength,
    )


def step_pxsfclay_column(
    u,
    v,
    temperature,
    qv,
    pressure,
    dz,
    *,
    theta=None,
    psfc,
    tsk,
    xland,
    mavail=1.0,
    znt=0.1,
    ust=0.1,
    mol=0.0,
    hfx=0.0,
    qfx=0.0,
    qsfc=-1.0,
    pblh=1000.0,
    dx=3000.0,
    itimestep=1,
    isfflx=True,
) -> PhysicsStepResult:
    """Return the frozen S0 adapter payload for one Pleim-Xiu SL call."""

    if theta is None:
        theta = _arr(temperature) * (P0_PA / _arr(pressure)) ** R_D_OVER_CP
    state = PleimXiuSfclayColumnState(
        u,
        v,
        temperature,
        theta,
        qv,
        pressure,
        dz,
        psfc,
        tsk,
        xland,
        mavail,
        znt,
        ust,
        mol,
        hfx,
        qfx,
        qsfc,
        pblh,
        dx,
        itimestep,
    )
    out = pxsfclay_run(state, isfflx=isfflx)
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
            "CPM": out.cpm,
            "FLHC": out.flhc,
            "FLQC": out.flqc,
            "CQS": out.cqs,
            "QSFC": out.qsfc,
            "QGH": out.qgh,
            "MOL": out.mol,
            "RMOL": out.rmol,
            "ZOL": out.zol,
            "REGIME": out.regime,
            "PSIM": out.psim,
            "PSIH": out.psih,
            "BR": out.br,
            "WSPD": out.wspd,
            "GZ1OZ0": out.gz1oz0,
            "RA": out.ra,
            "RBH": out.rbh,
            "RBW": out.rbw,
            "MOLENGTH": out.molength,
        }
    )
    return PhysicsStepResult(tendency=tendency, diagnostics=diagnostics)


__all__ = [
    "PleimXiuSfclayColumnState",
    "PleimXiuSfclayOutput",
    "pxsfclay_run",
    "step_pxsfclay_column",
]
