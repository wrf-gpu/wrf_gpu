"""WRF YSU PBL column kernel for the v0.6.0 PBL lane.

This module ports the scalar-column path of WRF ``bl_ysu_run`` used by
``module_bl_ysu.F`` for the v0.6.0 savepoint gate:

* no BEP/BEM urban canopy forcing,
* no chemical/passive tracer mixing,
* no cloud-water/ice top-down branch in the generated oracle cases,
* ``ctopo=ctopo2=1`` so the topo-wind correction is algebraically neutral.

The public entry returns the frozen S0 ``PhysicsStepResult`` with tendencies for
``u``, ``v``, ``theta``, and ``qv`` plus YSU diagnostics. The implementation is
column-level and CPU-parity oriented; dispatcher/domain batching is a later
manager-owned integration task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency


config.update("jax_enable_x64", True)


G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
XLV = 2.5e6
ROVCP = R_D / CP
EP1 = R_V / R_D - 1.0
EP2 = R_D / R_V
KARMAN = 0.4

XKZMINM = 0.1
XKZMINH = 0.01
XKZMAX = 1000.0
RIMIN = -100.0
RLAM = 30.0
PRMIN = 0.25
PRMAX = 4.0
BRCR_UB = 0.0
BRCR_SB = 0.25
CORI = 1.0e-4
AFAC = 6.8
BFAC = 6.8
PFAC = 2.0
PFAC_Q = 2.0
PHIFAC = 8.0
SFCFRAC = 0.1
D1 = 0.02
D2 = 0.05
D3 = 0.001
H1 = 0.33333335
H2 = 0.6666667
ZFMIN = 1.0e-8
APHI5 = 5.0
APHI16 = 16.0
TMIN = 1.0e-2
GAMCRT = 3.0
GAMCRQ = 2.0e-3
RCL = 1.0


@jax.tree_util.register_pytree_node_class
class YSUColumnState:
    """Pytree for one independent YSU PBL column on mass levels."""

    __slots__ = ("u", "v", "temperature", "qv", "pressure", "pressure_interface", "exner", "dz")

    def __init__(self, u, v, temperature, qv, pressure, pressure_interface, exner, dz) -> None:
        self.u = u
        self.v = v
        self.temperature = temperature
        self.qv = qv
        self.pressure = pressure
        self.pressure_interface = pressure_interface
        self.exner = exner
        self.dz = dz

    def replace(self, **updates) -> "YSUColumnState":
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
        if not isinstance(other, YSUColumnState):
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


@dataclass(frozen=True)
class YSUDiagnostics:
    pblh: jax.Array
    kpbl: jax.Array
    exch_h: jax.Array
    exch_m: jax.Array
    wstar: jax.Array
    delta: jax.Array


def _leaves(state: YSUColumnState) -> Iterable[jax.Array]:
    return (getattr(state, name) for name in YSUColumnState.__slots__)


def _as1d(value, *, length: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} length {arr.shape[0]} does not match {length}")
    return arr.copy()


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas solve matching WRF ``tridin_ysu``/``tridi2n`` indexing."""

    n = rhs.shape[0]
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, n - 1):
        denom = diag[k] - lower[k] * cp[k - 1]
        cp[k] = upper[k] / denom
        dp[k] = (rhs[k] - lower[k] * dp[k - 1]) / denom
    denom = diag[n - 1] - lower[n - 1] * cp[n - 2]
    dp[n - 1] = (rhs[n - 1] - lower[n - 1] * dp[n - 2]) / denom

    out = np.zeros(n, dtype=np.float64)
    out[n - 1] = dp[n - 1]
    for k in range(n - 2, -1, -1):
        out[k] = dp[k] - cp[k] * out[k + 1]
    return out


def _build_matrix(
    nz: int,
    delp: np.ndarray,
    p: np.ndarray,
    dza: np.ndarray,
    coeff: np.ndarray,
    dt2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.zeros(nz, dtype=np.float64)
    diag = np.zeros(nz, dtype=np.float64)
    upper = np.zeros(nz, dtype=np.float64)
    diag[0] = 1.0
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        dsdz2 = dsig * coeff[k] * rdz * rdz
        upper[k] = -dtodsd * dsdz2
        lower[k + 1] = -dtodsu * dsdz2
        diag[k] = diag[k] - upper[k]
        diag[k + 1] = 1.0 - lower[k + 1]
    return lower, diag, upper


def _first_pbl_guess(thv: np.ndarray, thermal: float, za: np.ndarray, br: float, brcr: float, u: np.ndarray, v: np.ndarray):
    brup = br
    brdn = br
    stable = False
    kpbl = 1
    for k in range(1, thv.shape[0]):
        if not stable:
            brdn = brup
            spdk2 = max(u[k] * u[k] + v[k] * v[k], 1.0)
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
            kpbl = k + 1
            stable = brup > brcr
    return kpbl, brdn, brup


def _interp_pblh(kpbl: int, brdn: float, brup: float, brcr: float, za: np.ndarray, zq: np.ndarray):
    if brdn >= brcr:
        brint = 0.0
    elif brup <= brcr:
        brint = 1.0
    else:
        brint = (brcr - brdn) / (brup - brdn)
    k0 = max(kpbl - 1, 1)
    hpbl = za[k0 - 1] + brint * (za[k0] - za[k0 - 1])
    if hpbl < zq[1]:
        kpbl = 1
    pblflg = kpbl > 1
    return hpbl, kpbl, pblflg


def _ysu_numpy(
    state: YSUColumnState,
    *,
    psfc: float,
    znt: float,
    ust: float,
    hfx: float,
    qfx: float,
    wspd: float,
    br: float,
    psim: float,
    psih: float,
    dt: float,
    xland: float,
    u10: float,
    v10: float,
    uoce: float,
    voce: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | np.ndarray]]:
    u = _as1d(state.u, name="u")
    v = _as1d(state.v, length=u.shape[0], name="v")
    tx = _as1d(state.temperature, length=u.shape[0], name="temperature")
    qv = _as1d(state.qv, length=u.shape[0], name="qv")
    p = _as1d(state.pressure, length=u.shape[0], name="pressure")
    pdi = _as1d(state.pressure_interface, length=u.shape[0] + 1, name="pressure_interface")
    pi = _as1d(state.exner, length=u.shape[0], name="exner")
    dz = _as1d(state.dz, length=u.shape[0], name="dz")
    nz = u.shape[0]

    th = tx / pi
    thli = th.copy()
    thv = th * (1.0 + EP1 * qv)
    rhox = psfc / (R_D * tx[0] * (1.0 + EP1 * qv[0]))
    govrth = G / th[0]

    zq = np.zeros(nz + 1, dtype=np.float64)
    rhox2 = np.zeros(nz, dtype=np.float64)
    for k in range(nz):
        zq[k + 1] = dz[k] + zq[k]
        rhox2[k] = p[k] / (R_D * tx[k] * (1.0 + EP1 * qv[k]))
    za = 0.5 * (zq[:-1] + zq[1:])
    delp = pdi[:-1] - pdi[1:]
    dza = np.empty(nz, dtype=np.float64)
    dza[0] = za[0]
    dza[1:] = za[1:] - za[:-1]

    xkzh = np.zeros(nz, dtype=np.float64)
    xkzm = np.zeros(nz, dtype=np.float64)
    xkzq = np.zeros(nz, dtype=np.float64)
    xkzhl = np.zeros(nz, dtype=np.float64)
    xkzml = np.zeros(nz, dtype=np.float64)
    xkzom = np.full(nz, XKZMINM, dtype=np.float64)
    xkzoh = np.full(nz, XKZMINH, dtype=np.float64)
    xkzom[-1] = 0.0
    xkzoh[-1] = 0.0
    exch_h = np.zeros(nz, dtype=np.float64)
    exch_m = np.zeros(nz, dtype=np.float64)
    zfac = np.zeros(nz, dtype=np.float64)
    zfacent = np.zeros(nz, dtype=np.float64)
    entfac = np.full(nz, 1.0e30, dtype=np.float64)
    wscalek = np.zeros(nz, dtype=np.float64)
    wscalek2 = np.zeros(nz, dtype=np.float64)

    dt2 = 2.0 * dt
    rdt = 1.0 / dt2
    cont = CP / G
    conpr = BFAC * KARMAN * SFCFRAC

    wspd1 = np.sqrt((u[0] - uoce) * (u[0] - uoce) + (v[0] - voce) * (v[0] - voce)) + 1.0e-9
    sflux = hfx / rhox / CP + qfx / rhox * EP1 * th[0]
    sfcflg = not (br > 0.0)
    thermal = float(thv[0])
    thermalli = float(thli[0])
    zl1 = float(za[0])

    kpbl, brdn, brup = _first_pbl_guess(thv, thermal, za, br, BRCR_UB, u, v)
    hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)

    fm = psim
    fh = psih
    zol1 = max(br * fm * fm / fh, RIMIN)
    zol1 = min(zol1, -ZFMIN) if sfcflg else max(zol1, ZFMIN)
    hol1 = zol1 * hpbl / zl1 * SFCFRAC
    if sfcflg:
        phim = (1.0 - APHI16 * hol1) ** (-0.25)
        phih = (1.0 - APHI16 * hol1) ** (-0.5)
        bfx0 = max(sflux, 0.0)
        wstar3 = govrth * bfx0 * hpbl
        wstar = wstar3**H1
    else:
        phim = 1.0 + APHI5 * hol1
        phih = phim
        wstar = 0.0
        wstar3 = 0.0

    ust3 = ust**3
    wscale = (ust3 + PHIFAC * KARMAN * wstar3 * 0.5) ** H1
    wscale = min(wscale, ust * APHI16)
    wscale = max(wscale, ust / APHI5)

    hgamt = 0.0
    hgamq = 0.0
    hgamu = 0.0
    hgamv = 0.0
    wstar3_2 = 0.0
    if sfcflg and sflux > 0.0:
        gamfac = BFAC / rhox / wscale
        hgamt = min(gamfac * hfx / CP, GAMCRT)
        hgamq = min(gamfac * qfx, GAMCRQ)
        vpert = (hgamt + EP1 * th[0] * hgamq) / BFAC * AFAC
        thermal = thermal + max(vpert, 0.0) * min(za[0] / (SFCFRAC * hpbl), 1.0)
        thermalli = thermalli + max(vpert, 0.0) * min(za[0] / (SFCFRAC * hpbl), 1.0)
        hgamt = max(hgamt, 0.0)
        hgamq = max(hgamq, 0.0)
        brint = -15.9 * ust * ust / wspd * wstar3 / (wscale**4)
        hgamu = brint * u[0]
        hgamv = brint * v[0]
    else:
        pblflg = False

    if pblflg:
        kpbl = 1
        hpbl = zq[0]
        kpbl, brdn, brup = _first_pbl_guess(thv, thermal, za, br, BRCR_UB, u, v)
        hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)

    if (not sfcflg) and hpbl < zq[1]:
        stable = False
        brup = br
    else:
        stable = True
        brup = br

    brcr = BRCR_UB
    if (not stable) and ((xland - 1.5) >= 0.0):
        wspd10 = np.sqrt(u10 * u10 + v10 * v10)
        ross = wspd10 / (CORI * znt)
        brcr = min(0.16 * (1.0e-7 * ross) ** (-0.18), 0.3)
    elif not stable:
        brcr = BRCR_SB

    for k in range(1, nz):
        if not stable:
            brdn = brup
            spdk2 = max(u[k] * u[k] + v[k] * v[k], 1.0)
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
            kpbl = k + 1
            stable = brup > brcr

    if (not sfcflg) and hpbl < zq[1]:
        hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, brcr, za, zq)

    cloudflg = False
    wm2 = 0.0
    we = 0.0
    hfxpbl = 0.0
    qfxpbl = 0.0
    ufxpbl = 0.0
    vfxpbl = 0.0
    delta = 0.0
    prpbl = 1.0
    if pblflg:
        k = kpbl - 2
        wm3 = wstar3 + 5.0 * ust3
        wm2 = wm3**H2
        bfxpbl = -0.15 * thv[0] / G * wm3 / hpbl
        dthvx = max(thv[k + 1] - thv[k], TMIN)
        we = max(bfxpbl / dthvx, -np.sqrt(wm2))
        dthx = max(th[k + 1] - th[k], TMIN)
        dqx = min(qv[k + 1] - qv[k], 0.0)
        hfxpbl = we * dthx
        qfxpbl = we * dqx
        dux = u[k + 1] - u[k]
        dvx = v[k + 1] - v[k]
        if dux > TMIN:
            ufxpbl = max(prpbl * we * dux, -ust * ust)
        elif dux < -TMIN:
            ufxpbl = min(prpbl * we * dux, ust * ust)
        if dvx > TMIN:
            vfxpbl = max(prpbl * we * dvx, -ust * ust)
        elif dvx < -TMIN:
            vfxpbl = min(prpbl * we * dvx, ust * ust)
        delb = govrth * D3 * hpbl
        delta = min(D1 * hpbl + D2 * wm2 / delb, 100.0)

    for k in range(nz):
        if pblflg and (k + 1) >= kpbl:
            entfac[k] = ((zq[k + 1] - hpbl) / delta) ** 2

    for k in range(nz):
        if (k + 1) < kpbl:
            zfac[k] = min(max(1.0 - (zq[k + 1] - zl1) / (hpbl - zl1), ZFMIN), 1.0)
            zfacent[k] = (1.0 - zfac[k]) ** 3
            wscalek[k] = (ust3 + PHIFAC * KARMAN * wstar3 * (1.0 - zfac[k])) ** H1
            wscalek2[k] = (PHIFAC * KARMAN * wstar3_2 * zfac[k]) ** H1
            if sfcflg:
                prfac = conpr
                denom = 1.0 + 4.0 * KARMAN * (wstar3 + wstar3_2) / ust3
                prfac2 = 15.9 * (wstar3 + wstar3_2) / ust3 / denom
                prnumfac = -3.0 * max(zq[k + 1] - SFCFRAC * hpbl, 0.0) ** 2 / hpbl**2
            else:
                prfac = 0.0
                prfac2 = 0.0
                prnumfac = 0.0
                phim8z = 1.0 + APHI5 * zol1 * zq[k + 1] / zl1
                wscalek[k] = max(ust / phim8z, 0.001)
            prnum0 = max(min(phih / phim + prfac, PRMAX), PRMIN)
            xkzm[k] = (
                wscalek[k] * KARMAN * zq[k + 1] * zfac[k] ** PFAC
                + wscalek2[k] * KARMAN * (hpbl - zq[k + 1]) * (1.0 - zfac[k]) ** PFAC
            )
            if (k + 1) == kpbl - 1 and cloudflg and we < 0.0:
                xkzm[k] = 0.0
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            xkzq[k] = xkzm[k] / prnum * zfac[k] ** (PFAC_Q - PFAC)
            prnum0 = prnum0 / (1.0 + prfac2 * KARMAN * SFCFRAC)
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            xkzh[k] = xkzm[k] / prnum
            xkzm[k] = min(xkzm[k] + xkzom[k], XKZMAX)
            xkzh[k] = min(xkzh[k] + xkzoh[k], XKZMAX)
            xkzq[k] = min(xkzq[k] + xkzoh[k], XKZMAX)

    for k in range(nz - 1):
        if (k + 1) >= kpbl:
            ss = ((u[k + 1] - u[k]) ** 2 + (v[k + 1] - v[k]) ** 2) / (dza[k + 1] ** 2) + 1.0e-9
            govrthv = G / (0.5 * (thv[k + 1] + thv[k]))
            ri = govrthv * (thv[k + 1] - thv[k]) / (ss * dza[k + 1])
            zk = KARMAN * zq[k + 1]
            rlamdz = min(max(0.1 * dza[k + 1], RLAM), 300.0)
            rlamdz = min(dza[k + 1], rlamdz)
            rl2 = (zk * rlamdz / (rlamdz + zk)) ** 2
            dk = rl2 * np.sqrt(ss)
            if ri < 0.0:
                ri = max(ri, RIMIN)
                sri = np.sqrt(-ri)
                xkzm[k] = dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.746 * sri))
                xkzh[k] = dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.286 * sri))
            else:
                xkzh[k] = dk / (1.0 + 5.0 * ri) ** 2
                prnum = min(1.0 + 2.1 * ri, PRMAX)
                xkzm[k] = xkzh[k] * prnum
            xkzm[k] = min(xkzm[k] + xkzom[k], XKZMAX)
            xkzh[k] = min(xkzh[k] + xkzoh[k], XKZMAX)
            xkzml[k] = xkzm[k]
            xkzhl[k] = xkzh[k]

    # Heat solve.
    rhs = np.zeros(nz, dtype=np.float64)
    rhs[0] = th[0] - 300.0 + hfx / cont / delp[0] * dt2
    lower = np.zeros(nz, dtype=np.float64)
    diag = np.zeros(nz, dtype=np.float64)
    upper = np.zeros(nz, dtype=np.float64)
    diag[0] = 1.0
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzh[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzt = tem1 * (-hgamt / hpbl - hfxpbl * zfacent[k] / xkzh[k])
            rhs[k] = rhs[k] + dtodsd * dsdzt
            rhs[k + 1] = th[k + 1] - 300.0 - dtodsu * dsdzt
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzh[k] = -we * dza[kpbl - 1] * np.exp(-entfac[k])
            xkzh[k] = np.sqrt(xkzh[k] * xkzhl[k])
            xkzh[k] = min(max(xkzh[k], xkzoh[k]), XKZMAX)
            rhs[k + 1] = th[k + 1] - 300.0
        else:
            rhs[k + 1] = th[k + 1] - 300.0
        tem1 = dsig * xkzh[k] * rdz
        dsdz2 = tem1 * rdz
        upper[k] = -dtodsd * dsdz2
        lower[k + 1] = -dtodsu * dsdz2
        diag[k] = diag[k] - upper[k]
        diag[k + 1] = 1.0 - lower[k + 1]
        exch_h[k + 1] = xkzh[k]
    theta_sol = _solve_tridiagonal(lower, diag, upper, rhs)
    theta_tend = (theta_sol - th + 300.0) * rdt

    # Water vapor solve.
    for k in range(nz - 1):
        if (k + 1) >= kpbl:
            xkzq[k] = xkzh[k]
    rhs_q = np.zeros(nz, dtype=np.float64)
    lower_q = np.zeros(nz, dtype=np.float64)
    diag_q = np.zeros(nz, dtype=np.float64)
    upper_q = np.zeros(nz, dtype=np.float64)
    diag_q[0] = 1.0
    rhs_q[0] = qv[0] + qfx * G / delp[0] * dt2
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzq[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzq = tem1 * (-qfxpbl * zfacent[k] / xkzq[k])
            rhs_q[k] = rhs_q[k] + dtodsd * dsdzq
            rhs_q[k + 1] = qv[k + 1] - dtodsu * dsdzq
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzq[k] = -we * dza[kpbl - 1] * np.exp(-entfac[k])
            xkzq[k] = np.sqrt(xkzq[k] * xkzhl[k])
            xkzq[k] = min(max(xkzq[k], xkzoh[k]), XKZMAX)
            rhs_q[k + 1] = qv[k + 1]
        else:
            rhs_q[k + 1] = qv[k + 1]
        tem1 = dsig * xkzq[k] * rdz
        dsdz2 = tem1 * rdz
        upper_q[k] = -dtodsd * dsdz2
        lower_q[k + 1] = -dtodsu * dsdz2
        diag_q[k] = diag_q[k] - upper_q[k]
        diag_q[k + 1] = 1.0 - lower_q[k + 1]
    qv_sol = _solve_tridiagonal(lower_q, diag_q, upper_q, rhs_q)
    qv_tend = (qv_sol - qv) * rdt

    # Momentum solve.
    rhs_u = np.zeros(nz, dtype=np.float64)
    rhs_v = np.zeros(nz, dtype=np.float64)
    lower_m = np.zeros(nz, dtype=np.float64)
    diag_m = np.zeros(nz, dtype=np.float64)
    upper_m = np.zeros(nz, dtype=np.float64)
    fric = ust * ust / wspd1 * rhox * G / delp[0] * dt2 * (wspd1 / wspd) ** 2
    diag_m[0] = 1.0 + fric
    rhs_u[0] = u[0] + uoce * ust * ust * rhox * G / delp[0] * dt2 / wspd1 * (wspd1 / wspd) ** 2
    rhs_v[0] = v[0] + voce * ust * ust * rhox * G / delp[0] * dt2 / wspd1 * (wspd1 / wspd) ** 2
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzm[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzu = tem1 * (-hgamu / hpbl - ufxpbl * zfacent[k] / xkzm[k])
            dsdzv = tem1 * (-hgamv / hpbl - vfxpbl * zfacent[k] / xkzm[k])
            rhs_u[k] = rhs_u[k] + dtodsd * dsdzu
            rhs_u[k + 1] = u[k + 1] - dtodsu * dsdzu
            rhs_v[k] = rhs_v[k] + dtodsd * dsdzv
            rhs_v[k + 1] = v[k + 1] - dtodsu * dsdzv
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzm[k] = prpbl * xkzh[k]
            xkzm[k] = np.sqrt(xkzm[k] * xkzml[k])
            xkzm[k] = min(max(xkzm[k], xkzom[k]), XKZMAX)
            rhs_u[k + 1] = u[k + 1]
            rhs_v[k + 1] = v[k + 1]
        else:
            rhs_u[k + 1] = u[k + 1]
            rhs_v[k + 1] = v[k + 1]
        tem1 = dsig * xkzm[k] * rdz
        dsdz2 = tem1 * rdz
        upper_m[k] = -dtodsd * dsdz2
        lower_m[k + 1] = -dtodsu * dsdz2
        diag_m[k] = diag_m[k] - upper_m[k]
        diag_m[k + 1] = 1.0 - lower_m[k + 1]
        exch_m[k + 1] = xkzm[k]

    u_sol = _solve_tridiagonal(lower_m, diag_m.copy(), upper_m, rhs_u)
    v_sol = _solve_tridiagonal(lower_m, diag_m.copy(), upper_m, rhs_v)
    u_tend = (u_sol - u) * rdt
    v_tend = (v_sol - v) * rdt

    tendencies = {
        "u": u_tend,
        "v": v_tend,
        "theta": theta_tend,
        "qv": qv_tend,
    }
    diagnostics = {
        "pblh": float(hpbl),
        "kpbl": int(kpbl),
        "exch_h": exch_h,
        "exch_m": exch_m,
        "wstar": float(wstar),
        "delta": float(delta),
    }
    return tendencies, diagnostics


def step_ysu_column(
    u,
    v,
    temperature,
    qv,
    pressure,
    pressure_interface,
    exner,
    dz,
    *,
    psfc,
    znt,
    ust,
    hfx,
    qfx,
    wspd,
    br,
    psim,
    psih,
    dt,
    xland=1.0,
    u10=0.0,
    v10=0.0,
    uoce=0.0,
    voce=0.0,
) -> PhysicsStepResult:
    """Run one WRF-YSU column and return frozen S0 tendency/diagnostics.

    Inputs follow the WRF ``module_bl_ysu.F`` wrapper naming: ``temperature`` is
    absolute temperature, ``exner`` is ``pi3d``, ``pressure_interface`` is the
    length ``nz+1`` pressure-at-interface column, and surface fluxes are the
    already-computed surface-layer values consumed by YSU.
    """

    state = YSUColumnState(u, v, temperature, qv, pressure, pressure_interface, exner, dz)
    tendencies_np, diagnostics_np = _ysu_numpy(
        state,
        psfc=_scalar(psfc),
        znt=_scalar(znt),
        ust=_scalar(ust),
        hfx=_scalar(hfx),
        qfx=_scalar(qfx),
        wspd=_scalar(wspd),
        br=_scalar(br),
        psim=_scalar(psim),
        psih=_scalar(psih),
        dt=_scalar(dt),
        xland=_scalar(xland),
        u10=_scalar(u10),
        v10=_scalar(v10),
        uoce=_scalar(uoce),
        voce=_scalar(voce),
    )
    tendencies = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in tendencies_np.items()}
    diagnostics = {
        "pblh": jnp.asarray(diagnostics_np["pblh"], dtype=jnp.float64),
        "kpbl": jnp.asarray(diagnostics_np["kpbl"], dtype=jnp.int32),
        "exch_h": jnp.asarray(diagnostics_np["exch_h"], dtype=jnp.float64),
        "exch_m": jnp.asarray(diagnostics_np["exch_m"], dtype=jnp.float64),
        "wstar": jnp.asarray(diagnostics_np["wstar"], dtype=jnp.float64),
        "delta": jnp.asarray(diagnostics_np["delta"], dtype=jnp.float64),
    }
    tendency = PhysicsTendency(state_tendencies=tendencies)
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        diagnostics=PhysicsDiagnostics(pbl=diagnostics),
    )


__all__ = ["YSUColumnState", "YSUDiagnostics", "step_ysu_column"]
