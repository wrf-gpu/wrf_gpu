"""WRF ACM2 PBL column kernel for the v0.6.0 PBL lane.

This module ports the scalar-column path of WRF ``module_bl_acm.F`` used for
``bl_pbl_physics=7``. ACM2 combines local downward eddy diffusion with an
asymmetric, non-local upward transilient term in free-convective boundary
layers; stable and neutral columns fall back to the local tridiagonal diffusion
solve.

The public entry returns the frozen S0 ``PhysicsStepResult`` with tendencies for
``u``, ``v``, ``theta``, and ``qv`` plus PBL diagnostics. Dispatcher/domain
batching is deliberately out of scope for this per-scheme savepoint lane.
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


RIC = 0.25
CRANKP = 0.5
KARMAN = 0.4
G_DEFAULT = 9.81
R_D_DEFAULT = 287.0
CP_DEFAULT = 7.0 * R_D_DEFAULT / 2.0
R_V_DEFAULT = 461.6
EP1_DEFAULT = R_V_DEFAULT / R_D_DEFAULT - 1.0


@jax.tree_util.register_pytree_node_class
class ACM2ColumnState:
    """Pytree for one independent ACM2 PBL column on mass levels."""

    __slots__ = ("u", "v", "theta", "temperature", "qv", "qc", "qi", "density", "dz")

    def __init__(self, u, v, theta, temperature, qv, qc, qi, density, dz) -> None:
        self.u = u
        self.v = v
        self.theta = theta
        self.temperature = temperature
        self.qv = qv
        self.qc = qc
        self.qi = qi
        self.density = density
        self.dz = dz

    def replace(self, **updates) -> "ACM2ColumnState":
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
        if not isinstance(other, ACM2ColumnState):
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
class ACM2Diagnostics:
    pblh: jax.Array
    kpbl: jax.Array
    regime: jax.Array
    noconv: jax.Array
    rmol: jax.Array
    exch_h: jax.Array
    exch_m: jax.Array


def _leaves(state: ACM2ColumnState) -> Iterable[jax.Array]:
    return (getattr(state, name) for name in ACM2ColumnState.__slots__)


def _as1d(value, *, length: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} length {arr.shape[0]} does not match {length}")
    return arr.copy()


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def _fortran_sign_magnitude(magnitude: float, sign_source: float) -> float:
    return -abs(magnitude) if np.signbit(sign_source) else abs(magnitude)


def _cuberoot_positive(value: float, exponent: float = 0.333333) -> float:
    if value < 0.0:
        raise ValueError(f"ACM2 expected a non-negative fractional-power input, got {value}")
    return value**exponent


def _tri_solve_1based(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray, n: int) -> np.ndarray:
    """WRF ``TRI`` transcription using 1-based coefficient arrays."""

    nsp = rhs.shape[0]
    gam = np.zeros(n + 1, dtype=np.float64)
    out = np.zeros((nsp, n + 1), dtype=np.float64)

    bet = 1.0 / diag[1]
    out[:, 1] = bet * rhs[:, 1]

    for k in range(2, n + 1):
        gam[k] = bet * upper[k - 1]
        bet = 1.0 / (diag[k] - lower[k] * gam[k])
        out[:, k] = bet * (rhs[:, k] - lower[k] * out[:, k - 1])

    for k in range(n - 1, 0, -1):
        out[:, k] = out[:, k] - gam[k + 1] * out[:, k + 1]
    return out


def _matrix_solve_1based(a: np.ndarray, b: np.ndarray, c: np.ndarray, rhs: np.ndarray, e: np.ndarray, n: int) -> np.ndarray:
    """WRF ``MATRIX`` bordered-band solver transcription for ACM2 non-local mixing."""

    nsp = rhs.shape[0]
    y = np.zeros((nsp, n + 1), dtype=np.float64)
    out = np.zeros((nsp, n + 1), dtype=np.float64)
    lower = np.zeros((n + 1, n + 1), dtype=np.float64)
    uii = np.zeros(n + 1, dtype=np.float64)
    uiip1 = np.zeros(n + 1, dtype=np.float64)
    ruii = np.zeros(n + 1, dtype=np.float64)

    lower[1, 1] = 1.0
    uii[1] = b[1]
    ruii[1] = 1.0 / uii[1]

    for i in range(2, n + 1):
        lower[i, i] = 1.0
        lower[i, 1] = a[i] / b[1]
        uiip1[i - 1] = e[i - 1]
        if i >= 3:
            for j in range(2, i):
                aij = c[i] if i == j + 1 else 0.0
                lower[i, j] = (aij - lower[i, j - 1] * e[j - 1]) / (b[j] - lower[j, j - 1] * e[j - 1])

    for i in range(2, n + 1):
        uii[i] = b[i] - lower[i, i - 1] * e[i - 1]
        ruii[i] = 1.0 / uii[i]

    y[:, 1] = rhs[:, 1]
    for i in range(2, n + 1):
        accum = rhs[:, i].copy()
        for j in range(1, i):
            accum = accum - lower[i, j] * y[:, j]
        y[:, i] = accum

    out[:, n] = y[:, n] * ruii[n]
    for i in range(n - 1, 0, -1):
        out[:, i] = (y[:, i] - uiip1[i] * out[:, i + 1]) * ruii[i]
    return out


def _mix_acm_values(
    *,
    dtpbl: float,
    noconv: int,
    zf: np.ndarray,
    dzh: np.ndarray,
    dzhi: np.ndarray,
    dzfi: np.ndarray,
    klpbl: int,
    pbl: float,
    mol: float,
    ust: float,
    eddy: np.ndarray,
    values: np.ndarray,
    surface_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """ACM/ACMM common semi-implicit solve.

    ``values`` has shape ``(n_species, nz)``. ``eddy`` is copied and returned
    because WRF reduces local eddy coefficients inside the convective branch
    before exporting ``EXCH_H``/``EXCH_M``.
    """

    del ust  # Present for ACM/ACMM signature symmetry; rates use ``mol`` below.
    nsp, nz = values.shape
    kl = nz
    eddy_out = eddy.copy()

    mbarks = np.zeros(nz + 2, dtype=np.float64)
    mdwn = np.zeros(nz + 2, dtype=np.float64)
    dtlim = float(dtpbl)
    fsacm = 0.0
    kcbl = 1

    if noconv == 1:
        kcbl = int(klpbl)
        hovl = -pbl / mol
        fsacm = 1.0 / (1.0 + ((KARMAN / hovl) ** 0.3333) / (0.72 * KARMAN))
        meddy = eddy_out[0] * dzfi[0] / (pbl - zf[1])
        mbar = meddy * fsacm
        for k in range(1, kcbl):
            eddy_out[k - 1] = eddy_out[k - 1] * (1.0 - fsacm)
        for k in range(2, kcbl + 1):
            mbarks[k] = mbar
            mdwn[k] = mbar * (pbl - zf[k - 1]) * dzhi[k - 1]
        mbarks[1] = mbar
        mbarks[kcbl] = mdwn[kcbl]
        mdwn[kcbl + 1] = 0.0

    for k in range(1, kl):
        ekz = eddy_out[k - 1] * dzfi[k - 1] * dzhi[k - 1]
        if ekz > 0.0:
            dtlim = min(0.75 / ekz, dtlim)

    if noconv == 1:
        rz = (zf[kcbl] - zf[1]) * dzhi[0]
        if mbarks[1] * rz > 0.0:
            dtlim = min(0.5 / (mbarks[1] * rz), dtlim)

    nlp = int(dtpbl / dtlim + 1.0)
    dts = dtpbl / nlp
    vci = values.copy()

    for _nl in range(1, nlp + 1):
        ai = np.zeros(kl + 1, dtype=np.float64)
        bi = np.zeros(kl + 1, dtype=np.float64)
        ci = np.zeros(kl + 1, dtype=np.float64)
        ei = np.zeros(kl + 1, dtype=np.float64)
        xplus = np.zeros(kl + 1, dtype=np.float64)
        xminus = np.zeros(kl + 1, dtype=np.float64)

        for k in range(2, kcbl + 1):
            ei[k - 1] = -CRANKP * mdwn[k] * dts * dzh[k - 1] * dzhi[k - 2]
            bi[k] = 1.0 + CRANKP * mdwn[k] * dts
            ai[k] = -CRANKP * mbarks[k] * dts

        ei[1] = ei[1] - eddy_out[0] * CRANKP * dzhi[0] * dzfi[0] * dts
        ai[2] = ai[2] - eddy_out[0] * CRANKP * dzhi[1] * dzfi[0] * dts

        for k in range(kcbl + 1, kl + 1):
            bi[k] = 1.0

        for k in range(2, kl + 1):
            xplus[k] = eddy_out[k - 1] * dzhi[k - 1] * dzfi[k - 1] * dts
            xminus[k] = eddy_out[k - 2] * dzhi[k - 1] * dzfi[k - 2] * dts
            ci[k] = -xminus[k] * CRANKP
            ei[k] = ei[k] - xplus[k] * CRANKP
            bi[k] = bi[k] + xplus[k] * CRANKP + xminus[k] * CRANKP

        if noconv == 1:
            bi[1] = (
                1.0
                + CRANKP * mbarks[1] * (pbl - zf[1]) * dts * dzhi[0]
                + eddy_out[0] * CRANKP * dzhi[0] * dzfi[0] * dts
            )
        else:
            bi[1] = 1.0 + eddy_out[0] * CRANKP * dzhi[0] * dzfi[0] * dts

        di = np.zeros((nsp, kl + 1), dtype=np.float64)
        ui = np.zeros((nsp, kl + 1), dtype=np.float64)

        for k in range(2, kcbl + 1):
            delc = dts * (
                mbarks[k] * vci[:, 0]
                - mdwn[k] * vci[:, k - 1]
                + dzh[k] * dzhi[k - 1] * mdwn[k + 1] * vci[:, k]
            )
            di[:, k] = vci[:, k - 1] + (1.0 - CRANKP) * delc

        for k in range(kcbl + 1, kl + 1):
            di[:, k] = vci[:, k - 1]

        for k in range(2, kl + 1):
            if k == kl:
                di[:, k] = di[:, k] - (1.0 - CRANKP) * xminus[k] * (vci[:, k - 1] - vci[:, k - 2])
            else:
                di[:, k] = (
                    di[:, k]
                    + (1.0 - CRANKP) * xplus[k] * (vci[:, k] - vci[:, k - 1])
                    - (1.0 - CRANKP) * xminus[k] * (vci[:, k - 1] - vci[:, k - 2])
                )

        if noconv == 1:
            di[:, 1] = vci[:, 0] + (
                surface_flux
                - (1.0 - CRANKP)
                * (mbarks[1] * (pbl - zf[1]) * vci[:, 0] - mdwn[2] * vci[:, 1] * dzh[1])
            ) * dzhi[0] * dts
        else:
            di[:, 1] = vci[:, 0] + surface_flux * dzhi[0] * dts

        di[:, 1] = di[:, 1] + (1.0 - CRANKP) * eddy_out[0] * dzhi[0] * dzfi[0] * dts * (vci[:, 1] - vci[:, 0])

        if noconv == 1:
            ui = _matrix_solve_1based(ai, bi, ci, di, ei, kl)
        else:
            ui = _tri_solve_1based(ci, bi, ei, di, kl)

        for k in range(1, kl + 1):
            vci[:, k - 1] = ui[:, k]

    return vci, eddy_out, fsacm


def _diagnose_pbl_height(
    *,
    theta_v: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    za: np.ndarray,
    zf: np.ndarray,
    mol: float,
    xtime: int,
    ust: float,
    wst: float,
    tstv: float,
    g: float,
) -> tuple[float, int, np.ndarray]:
    nz = theta_v.shape[0]
    rib = np.zeros(nz, dtype=np.float64)

    ksrc = 1
    for k in range(1, nz + 1):
        ksrc = k
        if zf[k] > 30.0:
            break

    th1 = float(np.mean(theta_v[:ksrc]))
    zh1 = float(np.mean(za[:ksrc]))
    uh1 = float(np.mean(u[:ksrc]))
    vh1 = float(np.mean(v[:ksrc]))

    if mol < 0.0 and xtime > 1:
        wss = _cuberoot_positive(ust**3 + 0.6 * wst**3)
        tconv = -8.5 * ust * tstv / wss
        th1 = th1 + tconv

    kmix = ksrc
    for k in range(ksrc, nz + 1):
        dtmp = theta_v[k - 1] - th1
        if dtmp < 0.0:
            kmix = k

    if kmix > ksrc:
        if kmix >= nz:
            raise RuntimeError("ACM2 PBL-height interpolation would exceed the top model level")
        fintt = (th1 - theta_v[kmix - 1]) / (theta_v[kmix] - theta_v[kmix - 1])
        zmix = fintt * (za[kmix] - za[kmix - 1]) + za[kmix - 1]
        umix = fintt * (u[kmix] - u[kmix - 1]) + u[kmix - 1]
        vmix = fintt * (v[kmix] - v[kmix - 1]) + v[kmix - 1]
    else:
        zmix = zh1
        umix = uh1
        vmix = vh1

    kpblh = nz
    for k in range(kmix, nz + 1):
        dtmp = theta_v[k - 1] - th1
        tog = 0.5 * (theta_v[k - 1] + th1) / g
        wssq = (u[k - 1] - umix) ** 2 + (v[k - 1] - vmix) ** 2
        if kmix == ksrc:
            wssq = u[k - 1] ** 2 + v[k - 1] ** 2
            wssq = wssq + 100.0 * ust * ust
        wssq = max(wssq, 0.1)
        rib[k - 1] = abs(za[k - 1] - zmix) * dtmp / (tog * wssq)
        if rib[k - 1] >= RIC:
            kpblh = k
            break
    else:
        raise RuntimeError(f"ACM2 RIB never exceeds RIC; top RIB={rib[-1]}")

    if kpblh > ksrc:
        fint = (RIC - rib[kpblh - 2]) / (rib[kpblh - 1] - rib[kpblh - 2])
        if fint > 0.5:
            kpblht = kpblh
            fint = fint - 0.5
        else:
            kpblht = kpblh - 1
            fint = fint + 0.5
        pbl = fint * (zf[kpblht] - zf[kpblht - 1]) + zf[kpblht - 1]
        klpbl = kpblht
    else:
        klpbl = ksrc
        pbl = za[ksrc - 1]

    return float(pbl), int(klpbl), rib


def _eddyx(
    *,
    dtpbl: float,
    zf: np.ndarray,
    za: np.ndarray,
    mol: float,
    pbl: float,
    ust: float,
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    theta_v: np.ndarray,
    density: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    g: float,
    rd: float,
    cpair: float,
) -> tuple[np.ndarray, np.ndarray]:
    del dtpbl, density  # Present in WRF signature; not used on the dry/no-cloud path.
    nz = u.shape[0]
    eddyz = np.zeros(nz, dtype=np.float64)
    eddyzm = np.zeros(nz, dtype=np.float64)

    rv = 461.5
    rlam = 80.0
    gamh = 16.0
    gamm = 16.0
    betah = 5.0
    p_exp = 2.0
    edyz0 = 0.01
    pr = 0.8

    for k in range(1, nz):
        idx = k - 1
        edyz_bl = 0.0
        edyzm_bl = 0.0
        dzf = za[idx + 1] - za[idx]
        kzo = edyz0

        if zf[k] < pbl:
            zovl = zf[k] / mol
            if zovl < 0.0:
                if zf[k] < 0.1 * pbl:
                    phih = 1.0 / np.sqrt(1.0 - gamh * zovl)
                    phim = (1.0 - gamm * zovl) ** (-0.25)
                    wt = ust / phih
                    wm = ust / phim
                else:
                    zsol = 0.1 * pbl / mol
                    phih = 1.0 / np.sqrt(1.0 - gamh * zsol)
                    phim = (1.0 - gamm * zsol) ** (-0.25)
                    wt = ust / phih
                    wm = ust / phim
            elif zovl < 1.0:
                phih = 1.0 + betah * zovl
                wt = ust / phih
                wm = wt
            else:
                phih = betah + zovl
                wt = ust / phih
                wm = wt
            zfunc = zf[k] * (1.0 - zf[k] / pbl) ** p_exp
            edyz_bl = KARMAN * wt * zfunc
            edyzm_bl = KARMAN * wm * zfunc

        ss = ((u[idx + 1] - u[idx]) ** 2 + (v[idx + 1] - v[idx]) ** 2) / (dzf * dzf) + 1.0e-9
        goth = 2.0 * g / (theta_v[idx + 1] + theta_v[idx])
        ri = goth * (theta_v[idx + 1] - theta_v[idx]) / (dzf * ss)

        if (qc[idx] + qi[idx]) > 0.01e-3 or (qc[idx + 1] + qi[idx + 1]) > 0.01e-3:
            qmean = 0.5 * (qv[idx] + qv[idx + 1])
            tmean = 0.5 * (temperature[idx] + temperature[idx + 1])
            xlv = (2.501 - 0.00237 * (tmean - 273.15)) * 1.0e6
            alph = xlv * qmean / rd / tmean
            chi = xlv * xlv * qmean / cpair / rv / tmean / tmean
            ri = (1.0 + alph) * (ri - g * g / ss / tmean / cpair * ((chi - alph) / (1.0 + chi)))

        zk = 0.4 * zf[k]
        sql = (zk * rlam / (rlam + zk)) ** 2
        if ri >= 0.0:
            fh = 1.0 / (1.0 + 10.0 * ri + 50.0 * ri**2 + 5000.0 * ri**4) + 0.0012
            fm = pr * fh + 0.00104
            eddyz[idx] = kzo + np.sqrt(ss) * fh * sql
            eddyzm[idx] = kzo + np.sqrt(ss) * fm * sql
        else:
            eddyz[idx] = kzo + np.sqrt(ss * (1.0 - 25.0 * ri)) * sql
            eddyzm[idx] = eddyz[idx] * pr

        if edyz_bl > eddyz[idx]:
            eddyz[idx] = edyz_bl
            eddyzm[idx] = min(edyzm_bl, edyz_bl * 0.8)

        eddyz[idx] = min(1000.0, eddyz[idx])
        eddyz[idx] = max(kzo, eddyz[idx])
        eddyzm[idx] = min(1000.0, eddyzm[idx])
        eddyzm[idx] = max(kzo, eddyzm[idx])

    eddyz[-1] = 0.0
    eddyzm[-1] = 0.0
    return eddyz, eddyzm


def _acm2_numpy(
    state: ACM2ColumnState,
    *,
    pblh_initial: float,
    ust: float,
    hfx: float,
    qfx: float,
    wspd: float,
    mut: float,
    dt: float,
    xtime: int,
    g: float = G_DEFAULT,
    rd: float = R_D_DEFAULT,
    cpd: float = CP_DEFAULT,
    ep1: float = EP1_DEFAULT,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | np.ndarray]]:
    u = _as1d(state.u, name="u")
    v = _as1d(state.v, length=u.shape[0], name="v")
    theta = _as1d(state.theta, length=u.shape[0], name="theta")
    temperature = _as1d(state.temperature, length=u.shape[0], name="temperature")
    qv = _as1d(state.qv, length=u.shape[0], name="qv")
    qc = _as1d(state.qc, length=u.shape[0], name="qc")
    qi = _as1d(state.qi, length=u.shape[0], name="qi")
    density = _as1d(state.density, length=u.shape[0], name="density")
    dz = _as1d(state.dz, length=u.shape[0], name="dz")
    nz = u.shape[0]

    if nz < 4:
        raise ValueError("ACM2 needs at least four vertical levels to exercise the PBL branch")
    if ust <= 0.0:
        raise ValueError("ACM2 requires positive friction velocity")
    if wspd <= 0.0:
        raise ValueError("ACM2 requires positive lowest-level wind speed")

    cpair = cpd * (1.0 + 0.84 * qv[0])
    tmpfx = hfx / (cpair * density[0])
    tmpvtcon = 1.0 + ep1 * qv[0]
    ws1 = np.sqrt(u[0] ** 2 + v[0] ** 2)
    tst = -tmpfx / ust
    qst = -qfx / (ust * density[0])
    ustm = ust * ws1 / wspd
    theta_v = theta * (1.0 + ep1 * qv)
    thv1 = tmpvtcon * theta[0]
    tstv = tst * tmpvtcon + thv1 * ep1 * qst
    if abs(tstv) < 1.0e-6:
        tstv = _fortran_sign_magnitude(1.0e-6, tstv)
    mol = thv1 * ust**2 / (KARMAN * g * tstv)
    rmol = 1.0 / mol
    wst = ust * (pblh_initial / (KARMAN * abs(mol))) ** 0.333333
    pstar = mut / 1000.0
    del pstar  # Retained for WRF argument parity; unused by ACM/ACMM.

    zf = np.zeros(nz + 1, dtype=np.float64)
    for k in range(nz):
        zf[k + 1] = dz[k] + zf[k]
    za = 0.5 * (zf[:-1] + zf[1:])
    dzh = zf[1:] - zf[:-1]
    dzhi = 1.0 / dzh
    dzfi = np.zeros(nz, dtype=np.float64)
    dzfi[:-1] = 1.0 / (za[1:] - za[:-1])
    dzfi[-1] = dzfi[-2]

    pblh, kpbl, rib = _diagnose_pbl_height(
        theta_v=theta_v,
        u=u,
        v=v,
        za=za,
        zf=zf,
        mol=mol,
        xtime=int(xtime),
        ust=ust,
        wst=wst,
        tstv=tstv,
        g=g,
    )

    noconv = 0
    regime = 0.0
    if pblh / mol < -0.02 and kpbl > 3 and theta_v[0] > theta_v[1] and int(xtime) > 1:
        noconv = 1
        regime = 4.0

    eddyz, eddyzm = _eddyx(
        dtpbl=dt,
        zf=zf,
        za=za,
        mol=mol,
        pbl=pblh,
        ust=ust,
        u=u,
        v=v,
        temperature=temperature,
        theta_v=theta_v,
        density=density,
        qv=qv,
        qc=qc,
        qi=qi,
        g=g,
        rd=rd,
        cpair=cpair,
    )

    scalar_values = np.vstack([theta, qv, qc, qi])
    scalar_flux = np.asarray([-ust * tst, -ust * qst, 0.0, 0.0], dtype=np.float64)
    scalar_new, exch_h, fsacm_h = _mix_acm_values(
        dtpbl=dt,
        noconv=noconv,
        zf=zf,
        dzh=dzh,
        dzhi=dzhi,
        dzfi=dzfi,
        klpbl=kpbl,
        pbl=pblh,
        mol=mol,
        ust=ust,
        eddy=eddyz,
        values=scalar_values,
        surface_flux=scalar_flux,
    )

    momentum_values = np.vstack([u, v])
    fm = -ustm * ustm
    wind1 = np.sqrt(u[0] * u[0] + v[0] * v[0]) + 1.0e-9
    momentum_flux = np.asarray([fm * u[0] / wind1, fm * v[0] / wind1], dtype=np.float64)
    momentum_new, exch_m, fsacm_m = _mix_acm_values(
        dtpbl=dt,
        noconv=noconv,
        zf=zf,
        dzh=dzh,
        dzhi=dzhi,
        dzfi=dzfi,
        klpbl=kpbl,
        pbl=pblh,
        mol=mol,
        ust=ust,
        eddy=eddyzm,
        values=momentum_values,
        surface_flux=momentum_flux,
    )

    rdt = 1.0 / dt
    tendencies = {
        "u": (momentum_new[0] - u) * rdt,
        "v": (momentum_new[1] - v) * rdt,
        "theta": (scalar_new[0] - theta) * rdt,
        "qv": (scalar_new[1] - qv) * rdt,
        "qc": (scalar_new[2] - qc) * rdt,
        "qi": (scalar_new[3] - qi) * rdt,
    }
    diagnostics = {
        "pblh": float(pblh),
        "kpbl": int(kpbl),
        "regime": float(regime),
        "noconv": int(noconv),
        "rmol": float(rmol),
        "exch_h": exch_h,
        "exch_m": exch_m,
        "rib": rib,
        "fsacm_h": float(fsacm_h),
        "fsacm_m": float(fsacm_m),
    }
    return tendencies, diagnostics


def step_acm2_column(
    u,
    v,
    theta,
    temperature,
    qv,
    density,
    dz,
    *,
    pblh_initial,
    ust,
    hfx,
    qfx,
    wspd,
    mut,
    dt,
    xtime=60,
    qc=None,
    qi=None,
    g=G_DEFAULT,
    rd=R_D_DEFAULT,
    cpd=CP_DEFAULT,
    ep1=EP1_DEFAULT,
) -> PhysicsStepResult:
    """Run one WRF-ACM2 column and return frozen S0 tendency/diagnostics.

    Inputs follow ``module_bl_acm.F:ACMPBL`` naming. ``pblh_initial`` is the
    incoming WRF ``PBLH`` value used by ACM2 before it diagnoses the new PBL
    height; this is part of the scheme state and must come from the savepoint.
    ``density`` is WRF ``RR3D`` dry-air density, and ``mut`` is total ``MU`` in
    Pa. ``qc`` and ``qi`` default to zero arrays for the v0.6.0 ACM2 PBL gate.
    """

    u_np = _as1d(u, name="u")
    nz = u_np.shape[0]
    qc_arr = np.zeros(nz, dtype=np.float64) if qc is None else _as1d(qc, length=nz, name="qc")
    qi_arr = np.zeros(nz, dtype=np.float64) if qi is None else _as1d(qi, length=nz, name="qi")
    state = ACM2ColumnState(u_np, v, theta, temperature, qv, qc_arr, qi_arr, density, dz)
    tendencies_np, diagnostics_np = _acm2_numpy(
        state,
        pblh_initial=_scalar(pblh_initial),
        ust=_scalar(ust),
        hfx=_scalar(hfx),
        qfx=_scalar(qfx),
        wspd=_scalar(wspd),
        mut=_scalar(mut),
        dt=_scalar(dt),
        xtime=int(_scalar(xtime)),
        g=_scalar(g),
        rd=_scalar(rd),
        cpd=_scalar(cpd),
        ep1=_scalar(ep1),
    )

    tendencies = {
        "u": jnp.asarray(tendencies_np["u"], dtype=jnp.float64),
        "v": jnp.asarray(tendencies_np["v"], dtype=jnp.float64),
        "theta": jnp.asarray(tendencies_np["theta"], dtype=jnp.float64),
        "qv": jnp.asarray(tendencies_np["qv"], dtype=jnp.float64),
    }
    diagnostics = {
        "pblh": jnp.asarray(diagnostics_np["pblh"], dtype=jnp.float64),
        "kpbl": jnp.asarray(diagnostics_np["kpbl"], dtype=jnp.int32),
        "regime": jnp.asarray(diagnostics_np["regime"], dtype=jnp.float64),
        "noconv": jnp.asarray(diagnostics_np["noconv"], dtype=jnp.int32),
        "rmol": jnp.asarray(diagnostics_np["rmol"], dtype=jnp.float64),
        "exch_h": jnp.asarray(diagnostics_np["exch_h"], dtype=jnp.float64),
        "exch_m": jnp.asarray(diagnostics_np["exch_m"], dtype=jnp.float64),
        "fsacm_h": jnp.asarray(diagnostics_np["fsacm_h"], dtype=jnp.float64),
        "fsacm_m": jnp.asarray(diagnostics_np["fsacm_m"], dtype=jnp.float64),
    }
    tendency = PhysicsTendency(state_tendencies=tendencies)
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        diagnostics=PhysicsDiagnostics(pbl=diagnostics),
    )


__all__ = ["ACM2ColumnState", "ACM2Diagnostics", "step_acm2_column"]
