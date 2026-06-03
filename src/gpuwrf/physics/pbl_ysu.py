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


# ---------------------------------------------------------------------------
# jax.lax.scan-traceable / vmap-batched YSU column kernel (v0.6.0 GPU-op).
#
# This is a 1:1 transcription of ``_ysu_numpy`` above into pure ``jnp`` /
# ``jax.lax`` primitives so the kernel is jit-traceable on a device State and
# ``jax.vmap``-batchable over the ``(ncol,)`` grid. The host-NumPy reference is
# retained for the per-case parity cross-check; the traceable path is what
# ``step_ysu_column`` and the operational scan adapter use. Every Python
# ``if``/``range`` over levels in the reference becomes a ``jnp.where`` masked op
# or a ``jax.lax.scan`` over a static ``nz``; the two Thomas solves use a
# scan-based tridiagonal solver matching ``tridin_ysu``/``tridi2n`` exactly.
# ---------------------------------------------------------------------------


def _thomas_scan(lower: jax.Array, diag: jax.Array, upper: jax.Array, rhs: jax.Array) -> jax.Array:
    """Scan-based Thomas solve matching WRF ``tridin_ysu``/``tridi2n`` indexing.

    Reproduces :func:`_solve_tridiagonal` exactly: forward sweep computes
    ``cp[k] = upper[k]/denom`` and ``dp[k] = (rhs[k]-lower[k]*dp[k-1])/denom``
    for ``k=1..n-1`` (``cp[n-1]`` is computed but unused in back-substitution,
    matching the reference where ``cp`` stops at ``n-2``), back sweep computes
    ``out[k] = dp[k]-cp[k]*out[k+1]``.
    """

    n = rhs.shape[0]
    cp0 = upper[0] / diag[0]
    dp0 = rhs[0] / diag[0]

    def _fwd(carry, k):
        cp_prev, dp_prev = carry
        denom = diag[k] - lower[k] * cp_prev
        cp_k = upper[k] / denom
        dp_k = (rhs[k] - lower[k] * dp_prev) / denom
        return (cp_k, dp_k), (cp_k, dp_k)

    _, (cp_rest, dp_rest) = jax.lax.scan(_fwd, (cp0, dp0), jnp.arange(1, n))
    cp = jnp.concatenate([cp0[None], cp_rest])
    dp = jnp.concatenate([dp0[None], dp_rest])

    def _bwd(x_next, k):
        x_k = dp[k] - cp[k] * x_next
        return x_k, x_k

    x_last = dp[n - 1]
    _, x_rest = jax.lax.scan(_bwd, x_last, jnp.arange(n - 2, -1, -1))
    return jnp.concatenate([x_rest[::-1], x_last[None]])


def _first_pbl_guess_traceable(thv, thermal, za, br, brcr, u, v):
    """Traceable ``_first_pbl_guess``: forward scan freezing on first ``brup>brcr``."""

    def _step(carry, k):
        brdn_c, brup_c, stable_c, kpbl_c = carry
        spdk2 = jnp.maximum(u[k] * u[k] + v[k] * v[k], 1.0)
        brup_new = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
        # Only advance while not yet stable.
        active = jnp.logical_not(stable_c)
        brdn_n = jnp.where(active, brup_c, brdn_c)
        brup_n = jnp.where(active, brup_new, brup_c)
        kpbl_n = jnp.where(active, (k + 1).astype(jnp.int32), kpbl_c)
        stable_n = jnp.where(active, brup_new > brcr, stable_c)
        return (brdn_n, brup_n, stable_n, kpbl_n), None

    init = (br, br, jnp.asarray(False), jnp.asarray(1, jnp.int32))
    (brdn, brup, _stable, kpbl), _ = jax.lax.scan(
        _step, init, jnp.arange(1, thv.shape[0], dtype=jnp.int32)
    )
    return kpbl, brdn, brup


def _interp_pblh_traceable(kpbl, brdn, brup, brcr, za, zq):
    """Traceable ``_interp_pblh``."""

    brint = jnp.where(
        brdn >= brcr,
        0.0,
        jnp.where(brup <= brcr, 1.0, (brcr - brdn) / (brup - brdn)),
    )
    k0 = jnp.maximum(kpbl - 1, 1)
    hpbl = za[k0 - 1] + brint * (za[k0] - za[k0 - 1])
    kpbl_new = jnp.where(hpbl < zq[1], jnp.asarray(1, jnp.int32), kpbl)
    pblflg = kpbl_new > 1
    return hpbl, kpbl_new, pblflg


def _ysu_column_traceable(
    u, v, tx, qv, p, pdi, pi, dz,
    *, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt, xland, u10, v10, uoce, voce,
):
    """jax.lax-traceable single YSU column. 1:1 with :func:`_ysu_numpy`.

    All inputs are ``(nz,)`` arrays (``pdi`` is ``(nz+1,)``) or scalars; output is
    ``(u_tend, v_tend, theta_tend, qv_tend, exch_h, exch_m, pblh, kpbl, wstar,
    delta)``. Vectorize over the grid with ``jax.vmap``.
    """

    nz = u.shape[0]
    kidx = jnp.arange(nz)  # 0-based level index
    kp1 = kidx + 1  # WRF 1-based (k+1) used in the host conditionals

    th = tx / pi
    thli = th
    thv = th * (1.0 + EP1 * qv)
    rhox = psfc / (R_D * tx[0] * (1.0 + EP1 * qv[0]))
    govrth = G / th[0]

    zq = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz)])  # (nz+1,)
    za = 0.5 * (zq[:-1] + zq[1:])
    delp = pdi[:-1] - pdi[1:]
    dza = jnp.concatenate([za[:1], za[1:] - za[:-1]])

    xkzom = jnp.where(kidx == nz - 1, 0.0, XKZMINM)
    xkzoh = jnp.where(kidx == nz - 1, 0.0, XKZMINH)

    dt2 = 2.0 * dt
    rdt = 1.0 / dt2
    cont = CP / G
    conpr = BFAC * KARMAN * SFCFRAC

    wspd1 = jnp.sqrt((u[0] - uoce) ** 2 + (v[0] - voce) ** 2) + 1.0e-9
    sflux = hfx / rhox / CP + qfx / rhox * EP1 * th[0]
    sfcflg = jnp.logical_not(br > 0.0)
    zl1 = za[0]

    thermal0 = thv[0]
    thermalli0 = thli[0]
    kpbl, brdn, brup = _first_pbl_guess_traceable(thv, thermal0, za, br, BRCR_UB, u, v)
    hpbl, kpbl, pblflg = _interp_pblh_traceable(kpbl, brdn, brup, BRCR_UB, za, zq)

    fm = psim
    fh = psih
    zol1 = jnp.maximum(br * fm * fm / fh, RIMIN)
    zol1 = jnp.where(sfcflg, jnp.minimum(zol1, -ZFMIN), jnp.maximum(zol1, ZFMIN))
    hol1 = zol1 * hpbl / zl1 * SFCFRAC

    phim_u = (1.0 - APHI16 * hol1) ** (-0.25)
    phih_u = (1.0 - APHI16 * hol1) ** (-0.5)
    bfx0 = jnp.maximum(sflux, 0.0)
    wstar3_u = govrth * bfx0 * hpbl
    wstar_u = wstar3_u ** H1
    phim_s = 1.0 + APHI5 * hol1
    phim = jnp.where(sfcflg, phim_u, phim_s)
    phih = jnp.where(sfcflg, phih_u, phim_s)
    wstar = jnp.where(sfcflg, wstar_u, 0.0)
    wstar3 = jnp.where(sfcflg, wstar3_u, 0.0)

    ust3 = ust ** 3
    wscale = (ust3 + PHIFAC * KARMAN * wstar3 * 0.5) ** H1
    wscale = jnp.minimum(wscale, ust * APHI16)
    wscale = jnp.maximum(wscale, ust / APHI5)

    wstar3_2 = 0.0  # YSU top-down off (no cloud branch in oracle cases)

    # The sfcflg & sflux>0 convective block updates thermal/hgam* and pblflg.
    conv = jnp.logical_and(sfcflg, sflux > 0.0)
    gamfac = BFAC / rhox / wscale
    hgamt_c = jnp.minimum(gamfac * hfx / CP, GAMCRT)
    hgamq_c = jnp.minimum(gamfac * qfx, GAMCRQ)
    vpert = (hgamt_c + EP1 * th[0] * hgamq_c) / BFAC * AFAC
    bump = jnp.maximum(vpert, 0.0) * jnp.minimum(za[0] / (SFCFRAC * hpbl), 1.0)
    thermal = jnp.where(conv, thermal0 + bump, thermal0)
    thermalli = jnp.where(conv, thermalli0 + bump, thermalli0)
    hgamt = jnp.where(conv, jnp.maximum(hgamt_c, 0.0), 0.0)
    hgamq = jnp.where(conv, jnp.maximum(hgamq_c, 0.0), 0.0)
    brint_m = -15.9 * ust * ust / wspd * wstar3 / (wscale ** 4)
    hgamu = jnp.where(conv, brint_m * u[0], 0.0)
    hgamv = jnp.where(conv, brint_m * v[0], 0.0)
    pblflg = jnp.where(conv, pblflg, jnp.asarray(False))

    # if pblflg: recompute PBL height with the convectively bumped thermal.
    kpbl_r, brdn_r, brup_r = _first_pbl_guess_traceable(thv, thermal, za, br, BRCR_UB, u, v)
    hpbl_r, kpbl_r, pblflg_r = _interp_pblh_traceable(kpbl_r, brdn_r, brup_r, BRCR_UB, za, zq)
    hpbl = jnp.where(pblflg, hpbl_r, hpbl)
    kpbl = jnp.where(pblflg, kpbl_r, kpbl)
    pblflg = jnp.where(pblflg, pblflg_r, pblflg)

    # Stable/unstable RB scan with brcr.
    cond_stable_low = jnp.logical_and(jnp.logical_not(sfcflg), hpbl < zq[1])
    # "stable" flag entering the scan: False when (not sfcflg and hpbl<zq1), else True.
    stable_init = jnp.logical_not(cond_stable_low)
    brup_pre = br

    is_water = (xland - 1.5) >= 0.0
    wspd10 = jnp.sqrt(u10 * u10 + v10 * v10)
    ross = wspd10 / (CORI * znt)
    brcr_water = jnp.minimum(0.16 * (1.0e-7 * ross) ** (-0.18), 0.3)
    # brcr default BRCR_UB; if not stable_init -> water:brcr_water else BRCR_SB.
    brcr = jnp.where(
        stable_init,
        BRCR_UB,
        jnp.where(is_water, brcr_water, BRCR_SB),
    )

    def _rb_step(carry, k):
        brdn_c, brup_c, stable_c, kpbl_c = carry
        active = jnp.logical_not(stable_c)
        spdk2 = jnp.maximum(u[k] * u[k] + v[k] * v[k], 1.0)
        brup_new = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
        brdn_n = jnp.where(active, brup_c, brdn_c)
        brup_n = jnp.where(active, brup_new, brup_c)
        kpbl_n = jnp.where(active, (k + 1).astype(jnp.int32), kpbl_c)
        stable_n = jnp.where(active, brup_new > brcr, stable_c)
        return (brdn_n, brup_n, stable_n, kpbl_n), None

    (brdn2, brup2, _st2, kpbl2), _ = jax.lax.scan(
        _rb_step, (brup_pre, brup_pre, stable_init, kpbl.astype(jnp.int32)),
        jnp.arange(1, nz, dtype=jnp.int32),
    )

    # if (not sfcflg) and hpbl<zq1: re-interp pblh with brcr.
    hpbl_b, kpbl_b, pblflg_b = _interp_pblh_traceable(kpbl2, brdn2, brup2, brcr, za, zq)
    hpbl = jnp.where(cond_stable_low, hpbl_b, hpbl)
    kpbl = jnp.where(cond_stable_low, kpbl_b, kpbl)
    pblflg = jnp.where(cond_stable_low, pblflg_b, pblflg)

    # Entrainment block (pblflg).
    kpbl_f = kpbl  # traced int
    k_ent = kpbl_f - 2  # 0-based level kpbl-2
    wm3 = wstar3 + 5.0 * ust3
    wm2 = wm3 ** H2
    bfxpbl = -0.15 * thv[0] / G * wm3 / hpbl
    thv_kp1 = thv[k_ent + 1]
    thv_k = thv[k_ent]
    dthvx = jnp.maximum(thv_kp1 - thv_k, TMIN)
    we = jnp.maximum(bfxpbl / dthvx, -jnp.sqrt(wm2))
    th_kp1 = th[k_ent + 1]
    th_k = th[k_ent]
    dthx = jnp.maximum(th_kp1 - th_k, TMIN)
    dqx = jnp.minimum(qv[k_ent + 1] - qv[k_ent], 0.0)
    hfxpbl = we * dthx
    qfxpbl = we * dqx
    dux = u[k_ent + 1] - u[k_ent]
    dvx = v[k_ent + 1] - v[k_ent]
    prpbl = 1.0
    ufxpbl = jnp.where(
        dux > TMIN, jnp.maximum(prpbl * we * dux, -ust * ust),
        jnp.where(dux < -TMIN, jnp.minimum(prpbl * we * dux, ust * ust), 0.0),
    )
    vfxpbl = jnp.where(
        dvx > TMIN, jnp.maximum(prpbl * we * dvx, -ust * ust),
        jnp.where(dvx < -TMIN, jnp.minimum(prpbl * we * dvx, ust * ust), 0.0),
    )
    delb = govrth * D3 * hpbl
    delta_c = jnp.minimum(D1 * hpbl + D2 * wm2 / delb, 100.0)
    # Zero the entrainment quantities when not pblflg.
    we = jnp.where(pblflg, we, 0.0)
    hfxpbl = jnp.where(pblflg, hfxpbl, 0.0)
    qfxpbl = jnp.where(pblflg, qfxpbl, 0.0)
    ufxpbl = jnp.where(pblflg, ufxpbl, 0.0)
    vfxpbl = jnp.where(pblflg, vfxpbl, 0.0)
    wm2 = jnp.where(pblflg, wm2, 0.0)
    delta = jnp.where(pblflg, delta_c, 0.0)

    # entfac: pblflg and (k+1)>=kpbl. Default 1e30. delta may be 0 when not pblflg;
    # guard the division (only used where pblflg true).
    delta_safe = jnp.where(pblflg, delta, 1.0)
    entfac = jnp.where(
        jnp.logical_and(pblflg, kp1 >= kpbl),
        ((zq[1:] - hpbl) / delta_safe) ** 2,
        1.0e30,
    )

    # Free-convective K profile for (k+1)<kpbl.
    in_pbl = kp1 < kpbl
    zfac = jnp.clip(1.0 - (zq[1:] - zl1) / (hpbl - zl1), ZFMIN, 1.0)
    zfacent = (1.0 - zfac) ** 3
    wscalek = (ust3 + PHIFAC * KARMAN * wstar3 * (1.0 - zfac)) ** H1
    wscalek2 = (PHIFAC * KARMAN * wstar3_2 * zfac) ** H1
    # sfcflg branch for prfac/prnumfac; else branch overrides wscalek.
    prfac_s = conpr
    denom_pr = 1.0 + 4.0 * KARMAN * (wstar3 + wstar3_2) / ust3
    prfac2_s = 15.9 * (wstar3 + wstar3_2) / ust3 / denom_pr
    prnumfac_s = -3.0 * jnp.maximum(zq[1:] - SFCFRAC * hpbl, 0.0) ** 2 / hpbl ** 2
    phim8z = 1.0 + APHI5 * zol1 * zq[1:] / zl1
    wscalek_ns = jnp.maximum(ust / phim8z, 0.001)
    prfac = jnp.where(sfcflg, prfac_s, 0.0)
    prfac2 = jnp.where(sfcflg, prfac2_s, 0.0)
    prnumfac = jnp.where(sfcflg, prnumfac_s, 0.0)
    wscalek = jnp.where(sfcflg, wscalek, wscalek_ns)
    prnum0 = jnp.clip(phih / phim + prfac, PRMIN, PRMAX)
    xkzm_pbl = (
        wscalek * KARMAN * zq[1:] * zfac ** PFAC
        + wscalek2 * KARMAN * (hpbl - zq[1:]) * (1.0 - zfac) ** PFAC
    )
    # cloudflg always False in the oracle path -> the (k+1)==kpbl-1 zeroing is inert.
    prnum_q = 1.0 + (prnum0 - 1.0) * jnp.exp(prnumfac)
    xkzq_pbl = xkzm_pbl / prnum_q * zfac ** (PFAC_Q - PFAC)
    prnum0b = prnum0 / (1.0 + prfac2 * KARMAN * SFCFRAC)
    prnum_h = 1.0 + (prnum0b - 1.0) * jnp.exp(prnumfac)
    xkzh_pbl = xkzm_pbl / prnum_h
    xkzm_pbl = jnp.minimum(xkzm_pbl + xkzom, XKZMAX)
    xkzh_pbl = jnp.minimum(xkzh_pbl + xkzoh, XKZMAX)
    xkzq_pbl = jnp.minimum(xkzq_pbl + xkzoh, XKZMAX)

    # Local Ri-based K for (k+1)>=kpbl (defined on k=0..nz-2).
    above = jnp.logical_and(kp1 >= kpbl, kidx < nz - 1)
    dza_kp1 = jnp.concatenate([dza[1:], dza[-1:]])  # dza[k+1] for k=0..nz-1 (last invalid)
    u_kp1 = jnp.concatenate([u[1:], u[-1:]])
    v_kp1 = jnp.concatenate([v[1:], v[-1:]])
    thv_next = jnp.concatenate([thv[1:], thv[-1:]])
    th_next = jnp.concatenate([th[1:], th[-1:]])
    ss = ((u_kp1 - u) ** 2 + (v_kp1 - v) ** 2) / (dza_kp1 ** 2) + 1.0e-9
    govrthv = G / (0.5 * (thv_next + thv))
    ri = govrthv * (thv_next - thv) / (ss * dza_kp1)
    zk = KARMAN * zq[1:]
    rlamdz = jnp.minimum(jnp.maximum(0.1 * dza_kp1, RLAM), 300.0)
    rlamdz = jnp.minimum(dza_kp1, rlamdz)
    rl2 = (zk * rlamdz / (rlamdz + zk)) ** 2
    dk = rl2 * jnp.sqrt(ss)
    ri_neg = jnp.maximum(ri, RIMIN)
    sri = jnp.sqrt(jnp.maximum(-ri, 0.0))
    xkzm_un = dk * (1.0 + 8.0 * (-ri_neg) / (1.0 + 1.746 * sri))
    xkzh_un = dk * (1.0 + 8.0 * (-ri_neg) / (1.0 + 1.286 * sri))
    xkzh_st = dk / (1.0 + 5.0 * ri) ** 2
    prnum_st = jnp.minimum(1.0 + 2.1 * ri, PRMAX)
    xkzm_st = xkzh_st * prnum_st
    xkzm_loc = jnp.where(ri < 0.0, xkzm_un, xkzm_st)
    xkzh_loc = jnp.where(ri < 0.0, xkzh_un, xkzh_st)
    xkzm_loc = jnp.minimum(xkzm_loc + xkzom, XKZMAX)
    xkzh_loc = jnp.minimum(xkzh_loc + xkzoh, XKZMAX)

    # Assemble the working xkzh/xkzm/xkzq on levels 0..nz-1.
    # In-PBL free-convective values where (k+1)<kpbl; local Ri values where above.
    xkzh = jnp.where(in_pbl, xkzh_pbl, jnp.where(above, xkzh_loc, 0.0))
    xkzm = jnp.where(in_pbl, xkzm_pbl, jnp.where(above, xkzm_loc, 0.0))
    xkzq = jnp.where(in_pbl, xkzq_pbl, jnp.where(above, xkzh_loc, 0.0))
    # xkzml/xkzhl carry the local (above) values for the entrainment sqrt blend.
    xkzml = jnp.where(above, xkzm_loc, 0.0)
    xkzhl = jnp.where(above, xkzh_loc, 0.0)

    # --- Heat solve ---
    # Per-level (k=0..nz-2) coefficients; build lower/diag/upper/rhs by scatter.
    dza_k1 = dza[1:]  # dza[k+1], length nz-1 valid for k=0..nz-2
    p_k = p[:-1]
    p_k1 = p[1:]
    delp_k = delp[:-1]
    delp_k1 = delp[1:]
    dtodsd = dt2 / delp_k
    dtodsu = dt2 / delp_k1
    dsig = p_k - p_k1
    rdz = 1.0 / dza_k1
    kk = jnp.arange(nz - 1)  # interface index k=0..nz-2 (between level k and k+1)
    kp1_face = kk + 1  # WRF 1-based (k+1)

    xkzh_face = xkzh[:-1]  # xkzh on faces k=0..nz-2
    # Entrainment-modified xkzh on faces where pblflg & (k+1)>=kpbl & entfac<4.6.
    ent_face = jnp.logical_and(
        jnp.logical_and(pblflg, kp1_face >= kpbl), entfac[:-1] < 4.6
    )
    dza_kpblm1 = dza[kpbl - 1]
    xkzh_ent = -we * dza_kpblm1 * jnp.exp(-entfac[:-1])
    xkzh_ent = jnp.sqrt(jnp.maximum(xkzh_ent * xkzhl[:-1], 0.0))
    xkzh_ent = jnp.minimum(jnp.maximum(xkzh_ent, xkzoh[:-1]), XKZMAX)
    in_pbl_face = kp1_face < kpbl
    xkzh_used = jnp.where(ent_face, xkzh_ent, xkzh_face)

    tem1 = dsig * xkzh_used * rdz
    dsdzt = tem1 * (-hgamt / hpbl - hfxpbl * zfacent[:-1] / jnp.where(in_pbl_face, xkzh_used, 1.0))
    dsdz2 = tem1 * rdz
    upper_h = -dtodsd * dsdz2
    lower_h_off = -dtodsu * dsdz2  # this is lower[k+1]

    # WRF sets diag[0]=1 then accumulates diag[k] -= upper[k], diag[k+1] = 1 - lower[k+1].
    # The ones() base + additive (-upper at k, -lower at k+1) reproduces this exactly:
    # diag[0] starts at 1 and gets -upper[0]; every other diag[m] = 1 - lower[m] - upper[m]
    # (upper[m] only contributes for m<=nz-2; lower[m] only for m>=1), matching the loop.
    diag_h = jnp.ones(nz)
    diag_h = diag_h.at[kk].add(-upper_h)
    diag_h = diag_h.at[kk + 1].add(-lower_h_off)
    upper_arr = jnp.zeros(nz).at[kk].set(upper_h)
    lower_arr = jnp.zeros(nz).at[kk + 1].set(lower_h_off)

    rhs_h = th - 300.0
    rhs_h = rhs_h.at[0].add(hfx / cont / delp[0] * dt2)
    # in-PBL counter-gradient contributions to rhs at k and k+1.
    add_k = jnp.where(in_pbl_face, dtodsd * dsdzt, 0.0)
    rhs_h = rhs_h.at[kk].add(add_k)
    # rhs[k+1] = th[k+1]-300 - dtodsu*dsdzt where in_pbl_face; else th[k+1]-300 (already set).
    sub_kp1 = jnp.where(in_pbl_face, -dtodsu * dsdzt, 0.0)
    rhs_h = rhs_h.at[kk + 1].add(sub_kp1)

    theta_sol = _thomas_scan(lower_arr, diag_h, upper_arr, rhs_h)
    theta_tend = (theta_sol - th + 300.0) * rdt
    exch_h = jnp.concatenate([jnp.zeros(1), xkzh_used])

    # --- Water-vapor solve ---
    # WRF: for (k+1)>=kpbl: xkzq[k]=xkzh[k] (the local-Ri value). Then same
    # entrainment override on the face as the heat solve but onto xkzq.
    xkzq_face = jnp.where(kp1_face >= kpbl, xkzh[:-1], xkzq[:-1])
    xkzq_ent = -we * dza_kpblm1 * jnp.exp(-entfac[:-1])
    xkzq_ent = jnp.sqrt(jnp.maximum(xkzq_ent * xkzhl[:-1], 0.0))
    xkzq_ent = jnp.minimum(jnp.maximum(xkzq_ent, xkzoh[:-1]), XKZMAX)
    xkzq_used = jnp.where(ent_face, xkzq_ent, xkzq_face)

    tem1_q = dsig * xkzq_used * rdz
    dsdzq = tem1_q * (-qfxpbl * zfacent[:-1] / jnp.where(in_pbl_face, xkzq_used, 1.0))
    dsdz2_q = tem1_q * rdz
    upper_q = -dtodsd * dsdz2_q
    lower_q_off = -dtodsu * dsdz2_q
    diag_q = jnp.ones(nz)
    diag_q = diag_q.at[kk].add(-upper_q)
    diag_q = diag_q.at[kk + 1].add(-lower_q_off)
    upper_q_arr = jnp.zeros(nz).at[kk].set(upper_q)
    lower_q_arr = jnp.zeros(nz).at[kk + 1].set(lower_q_off)
    rhs_q = qv
    rhs_q = rhs_q.at[0].add(qfx * G / delp[0] * dt2)
    add_kq = jnp.where(in_pbl_face, dtodsd * dsdzq, 0.0)
    rhs_q = rhs_q.at[kk].add(add_kq)
    sub_kp1q = jnp.where(in_pbl_face, -dtodsu * dsdzq, 0.0)
    rhs_q = rhs_q.at[kk + 1].add(sub_kp1q)
    qv_sol = _thomas_scan(lower_q_arr, diag_q, upper_q_arr, rhs_q)
    qv_tend = (qv_sol - qv) * rdt

    # --- Momentum solve ---
    # xkzm face values: in-PBL uses the free-convective xkzm; the entrainment
    # override sets xkzm[k]=sqrt(prpbl*xkzh[k] * xkzml[k]) for the ent faces.
    xkzm_face = xkzm[:-1]
    xkzm_ent = prpbl * xkzh_used  # WRF uses the (entrainment-overwritten) xkzh here
    xkzm_ent = jnp.sqrt(jnp.maximum(xkzm_ent * xkzml[:-1], 0.0))
    xkzm_ent = jnp.minimum(jnp.maximum(xkzm_ent, xkzom[:-1]), XKZMAX)
    xkzm_used = jnp.where(ent_face, xkzm_ent, xkzm_face)

    fric = ust * ust / wspd1 * rhox * G / delp[0] * dt2 * (wspd1 / wspd) ** 2
    tem1_m = dsig * xkzm_used * rdz
    dsdzu = tem1_m * (-hgamu / hpbl - ufxpbl * zfacent[:-1] / jnp.where(in_pbl_face, xkzm_used, 1.0))
    dsdzv = tem1_m * (-hgamv / hpbl - vfxpbl * zfacent[:-1] / jnp.where(in_pbl_face, xkzm_used, 1.0))
    dsdz2_m = tem1_m * rdz
    upper_m = -dtodsd * dsdz2_m
    lower_m_off = -dtodsu * dsdz2_m
    diag_m = jnp.ones(nz)
    diag_m = diag_m.at[0].add(fric)
    diag_m = diag_m.at[kk].add(-upper_m)
    diag_m = diag_m.at[kk + 1].add(-lower_m_off)
    upper_m_arr = jnp.zeros(nz).at[kk].set(upper_m)
    lower_m_arr = jnp.zeros(nz).at[kk + 1].set(lower_m_off)

    rhs_u = u
    rhs_v = v
    rhs_u = rhs_u.at[0].add(uoce * ust * ust * rhox * G / delp[0] * dt2 / wspd1 * (wspd1 / wspd) ** 2)
    rhs_v = rhs_v.at[0].add(voce * ust * ust * rhox * G / delp[0] * dt2 / wspd1 * (wspd1 / wspd) ** 2)
    add_ku = jnp.where(in_pbl_face, dtodsd * dsdzu, 0.0)
    add_kv = jnp.where(in_pbl_face, dtodsd * dsdzv, 0.0)
    rhs_u = rhs_u.at[kk].add(add_ku)
    rhs_v = rhs_v.at[kk].add(add_kv)
    sub_kp1u = jnp.where(in_pbl_face, -dtodsu * dsdzu, 0.0)
    sub_kp1v = jnp.where(in_pbl_face, -dtodsu * dsdzv, 0.0)
    rhs_u = rhs_u.at[kk + 1].add(sub_kp1u)
    rhs_v = rhs_v.at[kk + 1].add(sub_kp1v)

    u_sol = _thomas_scan(lower_m_arr, diag_m, upper_m_arr, rhs_u)
    v_sol = _thomas_scan(lower_m_arr, diag_m, upper_m_arr, rhs_v)
    u_tend = (u_sol - u) * rdt
    v_tend = (v_sol - v) * rdt
    exch_m = jnp.concatenate([jnp.zeros(1), xkzm_used])

    return (
        u_tend, v_tend, theta_tend, qv_tend,
        exch_h, exch_m, hpbl, kpbl.astype(jnp.int32), wstar, delta,
    )


def ysu_columns(
    u, v, temperature, qv, pressure, pressure_interface, exner, dz,
    *, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt, xland, u10, v10, uoce=None, voce=None,
):
    """Batched, jit/vmap-traceable YSU over ``(ncol,)`` columns.

    Profile inputs are ``(ncol, nz)`` (``pressure_interface`` is ``(ncol, nz+1)``);
    surface inputs are ``(ncol,)``. Returns a dict of ``(ncol, nz)`` tendencies and
    ``(ncol, nz)``/``(ncol,)`` diagnostics. This is the operational scan entry.
    """

    ncol = u.shape[0]
    if uoce is None:
        uoce = jnp.zeros(ncol, dtype=u.dtype)
    if voce is None:
        voce = jnp.zeros(ncol, dtype=u.dtype)
    dt_b = jnp.broadcast_to(jnp.asarray(dt, jnp.float64), (ncol,))

    out = jax.vmap(
        lambda *a: _ysu_column_traceable(
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
            psfc=a[8], znt=a[9], ust=a[10], hfx=a[11], qfx=a[12], wspd=a[13],
            br=a[14], psim=a[15], psih=a[16], dt=a[17], xland=a[18], u10=a[19],
            v10=a[20], uoce=a[21], voce=a[22],
        )
    )(
        u, v, temperature, qv, pressure, pressure_interface, exner, dz,
        psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt_b, xland, u10, v10, uoce, voce,
    )
    (u_t, v_t, th_t, qv_t, exch_h, exch_m, pblh, kpbl, wstar, delta) = out
    return {
        "u": u_t, "v": v_t, "theta": th_t, "qv": qv_t,
        "exch_h": exch_h, "exch_m": exch_m,
        "pblh": pblh, "kpbl": kpbl, "wstar": wstar, "delta": delta,
    }


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

    Backed by the ``jax.lax.scan``-traceable :func:`_ysu_column_traceable` (a 1:1
    transcription of the host-NumPy reference; the per-case savepoint parity
    asserts the traceable path equals WRF within the predeclared tolerances).
    """

    out = _ysu_column_traceable(
        jnp.asarray(u, jnp.float64),
        jnp.asarray(v, jnp.float64),
        jnp.asarray(temperature, jnp.float64),
        jnp.asarray(qv, jnp.float64),
        jnp.asarray(pressure, jnp.float64),
        jnp.asarray(pressure_interface, jnp.float64),
        jnp.asarray(exner, jnp.float64),
        jnp.asarray(dz, jnp.float64),
        psfc=jnp.asarray(psfc, jnp.float64),
        znt=jnp.asarray(znt, jnp.float64),
        ust=jnp.asarray(ust, jnp.float64),
        hfx=jnp.asarray(hfx, jnp.float64),
        qfx=jnp.asarray(qfx, jnp.float64),
        wspd=jnp.asarray(wspd, jnp.float64),
        br=jnp.asarray(br, jnp.float64),
        psim=jnp.asarray(psim, jnp.float64),
        psih=jnp.asarray(psih, jnp.float64),
        dt=jnp.asarray(dt, jnp.float64),
        xland=jnp.asarray(xland, jnp.float64),
        u10=jnp.asarray(u10, jnp.float64),
        v10=jnp.asarray(v10, jnp.float64),
        uoce=jnp.asarray(uoce, jnp.float64),
        voce=jnp.asarray(voce, jnp.float64),
    )
    (u_t, v_t, th_t, qv_t, exch_h, exch_m, pblh, kpbl, wstar, delta) = out
    tendencies = {
        "u": u_t,
        "v": v_t,
        "theta": th_t,
        "qv": qv_t,
    }
    diagnostics = {
        "pblh": pblh,
        "kpbl": kpbl.astype(jnp.int32),
        "exch_h": exch_h,
        "exch_m": exch_m,
        "wstar": wstar,
        "delta": delta,
    }
    tendency = PhysicsTendency(state_tendencies=tendencies)
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        diagnostics=PhysicsDiagnostics(pbl=diagnostics),
    )


__all__ = ["YSUColumnState", "YSUDiagnostics", "step_ysu_column", "ysu_columns"]
