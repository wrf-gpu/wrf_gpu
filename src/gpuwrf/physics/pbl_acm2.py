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

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency


configure_jax_x64()


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


# ---------------------------------------------------------------------------
# jax.lax.scan-traceable / vmap-batched ACM2 column kernel (v0.6.0 GPU-op).
#
# 1:1 transcription of the host-NumPy reference above into pure ``jnp`` /
# ``jax.lax`` so the kernel is jit-traceable on a device State and
# ``jax.vmap``-batchable over ``(ncol,)``. The Fortran ``TRI``/``MATRIX`` solvers
# and the PBL-height/eddy diagnostics use STATIC (nz-bounded) Python loops which
# unroll at trace time into a fixed graph; the only DATA-dependent control flow
# in the reference -- the ``break`` index searches, the ``if noconv`` /
# ``if kmix>ksrc`` branches, and the substep count ``nlp`` -- becomes masked
# ``jnp.where`` / argmax / a fixed-max ``jax.lax.scan`` with per-column masking.
#
# nlp note: WRF chooses a per-column number of semi-implicit substeps from a CFL
# limit. That count is data-dependent, so the batched kernel scans a STATIC
# ``_NLP_MAX`` substeps and freezes a column once it has done its own ``nlp``
# (``jnp.where(i < nlp, advanced, vci)``) -- exact for any column with
# ``nlp <= _NLP_MAX``. _NLP_MAX is asserted large enough at trace time.
# ---------------------------------------------------------------------------

_NLP_MAX = 16


def _mean_first_k(arr: jax.Array, ksrc: jax.Array) -> jax.Array:
    """Mean of ``arr[:ksrc]`` (1-based ksrc) as a traced scalar."""

    n = arr.shape[0]
    idx = jnp.arange(n)
    mask = (idx < ksrc).astype(arr.dtype)
    return jnp.sum(arr * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _diagnose_pbl_height_traceable(theta_v, u, v, za, zf, mol, xtime, ust, wst, tstv, g):
    """Traceable :func:`_diagnose_pbl_height`. Returns ``(pbl, klpbl, rib)``."""

    nz = theta_v.shape[0]
    # ksrc (1-based): first k in [1,nz] with zf[k]>30, else nz.
    zf_k = zf[1:]  # zf[1..nz]
    gt30 = zf_k > 30.0
    any_gt = jnp.any(gt30)
    first_gt = jnp.argmax(gt30)  # index into [0..nz-1] -> k = first_gt+1
    ksrc = jnp.where(any_gt, (first_gt + 1).astype(jnp.int32), jnp.asarray(nz, jnp.int32))

    th1 = _mean_first_k(theta_v, ksrc)
    zh1 = _mean_first_k(za, ksrc)
    uh1 = _mean_first_k(u, ksrc)
    vh1 = _mean_first_k(v, ksrc)

    wss = (ust ** 3 + 0.6 * wst ** 3) ** 0.333333
    tconv = -8.5 * ust * tstv / wss
    conv_correction = jnp.logical_and(mol < 0.0, xtime > 1)
    th1 = jnp.where(conv_correction, th1 + tconv, th1)

    # kmix (1-based): last k in [ksrc,nz] with theta_v[k-1]<th1, else ksrc.
    kidx1 = jnp.arange(1, nz + 1)  # 1-based level indices
    in_range = kidx1 >= ksrc
    dtmp_all = theta_v - th1  # theta_v[k-1] for k=1..nz
    unstable = jnp.logical_and(in_range, dtmp_all < 0.0)
    any_unstable = jnp.any(unstable)
    # last index where unstable: argmax over reversed.
    last_unstable = nz - 1 - jnp.argmax(unstable[::-1])  # 0-based
    kmix = jnp.where(any_unstable, (last_unstable + 1).astype(jnp.int32), ksrc)

    kmix_gt_ksrc = kmix > ksrc
    # interpolation uses theta_v[kmix-1], theta_v[kmix], za[kmix-1], za[kmix] (0-based
    # = theta_v[kmix-1] is index kmix-1). gather with the traced kmix.
    tvm1 = theta_v[kmix - 1]
    tvm = theta_v[kmix]
    fintt = (th1 - tvm1) / (tvm - tvm1)
    zmix_i = fintt * (za[kmix] - za[kmix - 1]) + za[kmix - 1]
    umix_i = fintt * (u[kmix] - u[kmix - 1]) + u[kmix - 1]
    vmix_i = fintt * (v[kmix] - v[kmix - 1]) + v[kmix - 1]
    zmix = jnp.where(kmix_gt_ksrc, zmix_i, zh1)
    umix = jnp.where(kmix_gt_ksrc, umix_i, uh1)
    vmix = jnp.where(kmix_gt_ksrc, vmix_i, vh1)

    # rib over k in [kmix,nz]; kpblh = first k with rib>=RIC (else nz).
    in_rib = kidx1 >= kmix
    tog = 0.5 * (theta_v + th1) / g
    wssq_general = (u - umix) ** 2 + (v - vmix) ** 2
    wssq_src = u ** 2 + v ** 2 + 100.0 * ust * ust
    wssq = jnp.where(kmix == ksrc, wssq_src, wssq_general)
    wssq = jnp.maximum(wssq, 0.1)
    rib_full = jnp.abs(za - zmix) * dtmp_all / (tog * wssq)
    rib = jnp.where(in_rib, rib_full, 0.0)

    ge_ric = jnp.logical_and(in_rib, rib >= RIC)
    any_ric = jnp.any(ge_ric)
    first_ric = jnp.argmax(ge_ric)  # 0-based
    kpblh = jnp.where(any_ric, (first_ric + 1).astype(jnp.int32), jnp.asarray(nz, jnp.int32))

    kpblh_gt_ksrc = kpblh > ksrc
    fint0 = (RIC - rib[kpblh - 2]) / (rib[kpblh - 1] - rib[kpblh - 2])
    fint_gt = fint0 > 0.5
    kpblht = jnp.where(fint_gt, kpblh, kpblh - 1)
    fint = jnp.where(fint_gt, fint0 - 0.5, fint0 + 0.5)
    pbl_i = fint * (zf[kpblht] - zf[kpblht - 1]) + zf[kpblht - 1]
    klpbl_i = kpblht
    pbl = jnp.where(kpblh_gt_ksrc, pbl_i, za[ksrc - 1])
    klpbl = jnp.where(kpblh_gt_ksrc, klpbl_i, ksrc).astype(jnp.int32)
    return pbl, klpbl, rib


def _eddyx_traceable(zf, za, mol, pbl, ust, u, v, temperature, theta_v, qv, qc, qi, g, rd, cpair):
    """Traceable :func:`_eddyx`. Returns ``(eddyz, eddyzm)`` length nz (top=0)."""

    nz = u.shape[0]
    rv = 461.5
    rlam = 80.0
    gamh = 16.0
    gamm = 16.0
    betah = 5.0
    p_exp = 2.0
    edyz0 = 0.01
    pr = 0.8
    kzo = edyz0

    # operate on interface index k=1..nz-1 (0-based idx=k-1 = 0..nz-2).
    idx = jnp.arange(nz - 1)  # 0..nz-2
    k = idx + 1  # 1..nz-1
    zfk = zf[1:nz]  # zf[1..nz-1]
    dzf = za[idx + 1] - za[idx]

    # BL eddy (zf[k]<pbl).
    in_bl = zfk < pbl
    zovl = zfk / mol
    # zovl<0 branch with z< or >= 0.1*pbl.
    near = zfk < 0.1 * pbl
    zsol = 0.1 * pbl / mol
    phih_neg_near = 1.0 / jnp.sqrt(1.0 - gamh * zovl)
    phim_neg_near = (1.0 - gamm * zovl) ** (-0.25)
    phih_neg_far = 1.0 / jnp.sqrt(1.0 - gamh * zsol)
    phim_neg_far = (1.0 - gamm * zsol) ** (-0.25)
    wt_neg = jnp.where(near, ust / phih_neg_near, ust / phih_neg_far)
    wm_neg = jnp.where(near, ust / phim_neg_near, ust / phim_neg_far)
    # zovl in [0,1).
    phih_mid = 1.0 + betah * zovl
    wt_mid = ust / phih_mid
    # zovl>=1.
    phih_hi = betah + zovl
    wt_hi = ust / phih_hi
    wt = jnp.where(zovl < 0.0, wt_neg, jnp.where(zovl < 1.0, wt_mid, wt_hi))
    wm = jnp.where(zovl < 0.0, wm_neg, jnp.where(zovl < 1.0, wt_mid, wt_hi))
    zfunc = zfk * (1.0 - zfk / pbl) ** p_exp
    edyz_bl = jnp.where(in_bl, KARMAN * wt * zfunc, 0.0)
    edyzm_bl = jnp.where(in_bl, KARMAN * wm * zfunc, 0.0)

    # local Ri.
    ss = ((u[idx + 1] - u[idx]) ** 2 + (v[idx + 1] - v[idx]) ** 2) / (dzf * dzf) + 1.0e-9
    goth = 2.0 * g / (theta_v[idx + 1] + theta_v[idx])
    ri0 = goth * (theta_v[idx + 1] - theta_v[idx]) / (dzf * ss)
    # cloud correction.
    cloud = jnp.logical_or((qc[idx] + qi[idx]) > 0.01e-3, (qc[idx + 1] + qi[idx + 1]) > 0.01e-3)
    qmean = 0.5 * (qv[idx] + qv[idx + 1])
    tmean = 0.5 * (temperature[idx] + temperature[idx + 1])
    xlv = (2.501 - 0.00237 * (tmean - 273.15)) * 1.0e6
    alph = xlv * qmean / rd / tmean
    chi = xlv * xlv * qmean / cpair / rv / tmean / tmean
    ri_cloud = (1.0 + alph) * (ri0 - g * g / ss / tmean / cpair * ((chi - alph) / (1.0 + chi)))
    ri = jnp.where(cloud, ri_cloud, ri0)

    zk = 0.4 * zfk
    sql = (zk * rlam / (rlam + zk)) ** 2
    fh = 1.0 / (1.0 + 10.0 * ri + 50.0 * ri ** 2 + 5000.0 * ri ** 4) + 0.0012
    fm = pr * fh + 0.00104
    eddyz_pos = kzo + jnp.sqrt(ss) * fh * sql
    eddyzm_pos = kzo + jnp.sqrt(ss) * fm * sql
    eddyz_neg = kzo + jnp.sqrt(ss * (1.0 - 25.0 * ri)) * sql
    eddyzm_neg = eddyz_neg * pr
    eddyz = jnp.where(ri >= 0.0, eddyz_pos, eddyz_neg)
    eddyzm = jnp.where(ri >= 0.0, eddyzm_pos, eddyzm_neg)

    # BL override when edyz_bl > eddyz.
    bl_wins = edyz_bl > eddyz
    eddyzm = jnp.where(bl_wins, jnp.minimum(edyzm_bl, edyz_bl * 0.8), eddyzm)
    eddyz = jnp.where(bl_wins, edyz_bl, eddyz)

    eddyz = jnp.clip(eddyz, kzo, 1000.0)
    eddyzm = jnp.clip(eddyzm, kzo, 1000.0)

    # append top level = 0.
    eddyz = jnp.concatenate([eddyz, jnp.zeros(1)])
    eddyzm = jnp.concatenate([eddyzm, jnp.zeros(1)])
    return eddyz, eddyzm


def _tri_solve_1based_traceable(lower, diag, upper, rhs, n):
    """Traceable WRF ``TRI`` solver. ``rhs`` is ``(nsp, n+1)``; returns ``(nsp, n+1)``.

    Static (n-bounded) Python recurrence; unrolls at trace time. Index 0 of each
    1-based array is unused padding (kept for index parity with the reference).
    """

    nsp = rhs.shape[0]
    gam = [jnp.asarray(0.0)] * (n + 1)
    out = [jnp.zeros(nsp)] * (n + 1)  # 1-based; out[0] padding
    bet = 1.0 / diag[1]
    out[1] = bet * rhs[:, 1]
    for kk in range(2, n + 1):
        gam[kk] = bet * upper[kk - 1]
        bet = 1.0 / (diag[kk] - lower[kk] * gam[kk])
        out[kk] = bet * (rhs[:, kk] - lower[kk] * out[kk - 1])
    for kk in range(n - 1, 0, -1):
        out[kk] = out[kk] - gam[kk + 1] * out[kk + 1]
    return jnp.stack([jnp.zeros(nsp)] + out[1:], axis=1)  # (nsp, n+1)


def _matrix_solve_1based_traceable(a, b, c, rhs, e, n):
    """Traceable WRF ``MATRIX`` bordered-band solver. ``rhs`` is ``(nsp, n+1)``.

    Static (n-bounded) Python loops; unrolls at trace time. ``lower`` is built as a
    dict of traced scalars indexed by (i,j) -- only the (i,1), (i,i), (i,i-1), and
    the recurrence (i,j) entries are ever populated, matching the reference.
    """

    nsp = rhs.shape[0]
    lower = {}  # (i,j) -> traced scalar
    uii = [jnp.asarray(0.0)] * (n + 1)
    uiip1 = [jnp.asarray(0.0)] * (n + 1)
    ruii = [jnp.asarray(0.0)] * (n + 1)

    lower[(1, 1)] = jnp.asarray(1.0)
    uii[1] = b[1]
    ruii[1] = 1.0 / uii[1]

    for i in range(2, n + 1):
        lower[(i, i)] = jnp.asarray(1.0)
        lower[(i, 1)] = a[i] / b[1]
        uiip1[i - 1] = e[i - 1]
        if i >= 3:
            for j in range(2, i):
                aij = c[i] if i == j + 1 else 0.0
                lower[(i, j)] = (aij - lower[(i, j - 1)] * e[j - 1]) / (
                    b[j] - lower[(j, j - 1)] * e[j - 1]
                )

    for i in range(2, n + 1):
        uii[i] = b[i] - lower[(i, i - 1)] * e[i - 1]
        ruii[i] = 1.0 / uii[i]

    y = [jnp.zeros(nsp)] * (n + 1)
    y[1] = rhs[:, 1]
    for i in range(2, n + 1):
        accum = rhs[:, i]
        for j in range(1, i):
            lij = lower.get((i, j))
            if lij is not None:
                accum = accum - lij * y[j]
        y[i] = accum

    out = [jnp.zeros(nsp)] * (n + 1)
    out[n] = y[n] * ruii[n]
    for i in range(n - 1, 0, -1):
        out[i] = (y[i] - uiip1[i] * out[i + 1]) * ruii[i]
    return jnp.stack([jnp.zeros(nsp)] + out[1:], axis=1)  # (nsp, n+1)


def _mix_acm_values_traceable(dtpbl, noconv, zf, dzh, dzhi, dzfi, klpbl, pbl, mol, ust, eddy, values, surface_flux):
    """Traceable :func:`_mix_acm_values`. ``values`` is ``(nsp, nz)``; ``noconv`` traced 0/1.

    Returns ``(vci, eddy_out, fsacm)``. The semi-implicit substep loop runs a
    fixed ``_NLP_MAX`` iterations, freezing each column after its own ``nlp``.
    """

    del ust
    nsp, nz = values.shape
    kl = nz
    eddy_out = eddy

    # noconv-branch mass-flux coefficients (1-based arrays length nz+2).
    kcbl = jnp.where(noconv == 1, klpbl, jnp.asarray(1, jnp.int32))
    hovl = -pbl / mol
    fsacm = jnp.where(noconv == 1, 1.0 / (1.0 + ((KARMAN / hovl) ** 0.3333) / (0.72 * KARMAN)), 0.0)
    meddy = eddy_out[0] * dzfi[0] / (pbl - zf[1])
    mbar = meddy * fsacm

    # eddy_out[k-1] *= (1-fsacm) for k in 1..kcbl-1 (0-based 0..kcbl-2), noconv only.
    kidx0 = jnp.arange(nz)
    reduce_mask = jnp.logical_and(noconv == 1, kidx0 <= (kcbl - 2))
    eddy_out = jnp.where(reduce_mask, eddy_out * (1.0 - fsacm), eddy_out)

    # mbarks[k], mdwn[k] for k in 2..kcbl (1-based), plus mbarks[1]=mbar,
    # mbarks[kcbl]=mdwn[kcbl], mdwn[kcbl+1]=0. Length nz+2 1-based arrays built by a
    # STATIC k-loop with a traced `2<=k<=kcbl` mask (mirrors range(2,kcbl+1)).
    nl2 = nz + 2
    jpos = jnp.arange(nl2)  # 1-based position 0..nz+1
    in_band = jnp.logical_and(noconv == 1, jnp.logical_and(jpos >= 2, jpos <= kcbl))
    # mdwn[k] = mbar*(pbl - zf[k-1])*dzhi[k-1]; zf len nz+1 (idx 0..nz), dzhi len nz.
    zf_km1 = zf[jnp.clip(jpos - 1, 0, nz)]
    dzhi_km1 = dzhi[jnp.clip(jpos - 1, 0, nz - 1)]
    mdwn = jnp.where(in_band, mbar * (pbl - zf_km1) * dzhi_km1, 0.0)
    mbarks = jnp.where(in_band, mbar, 0.0)
    # mbarks[1]=mbar (noconv); else 0.
    mbarks = mbarks.at[1].set(jnp.where(noconv == 1, mbar, 0.0))
    # mbarks[kcbl]=mdwn[kcbl]  (noconv; kcbl in [1,nz]).
    mbarks = mbarks.at[kcbl].set(jnp.where(noconv == 1, mdwn[kcbl], mbarks[kcbl]))
    # mdwn[kcbl+1]=0 (already 0 outside the band; explicit for parity).
    mdwn = mdwn.at[jnp.minimum(kcbl + 1, nz + 1)].set(
        jnp.where(noconv == 1, 0.0, mdwn[jnp.minimum(kcbl + 1, nz + 1)])
    )

    # dtlim from local eddy CFL: for k in 1..kl-1 (0-based 0..kl-2): ekz=eddy[k-1]*dzfi*dzhi.
    ek_idx = jnp.arange(nz - 1)  # 0..nz-2 = (k-1) for k=1..nz-1
    ekz = eddy_out[ek_idx] * dzfi[ek_idx] * dzhi[ek_idx]
    cfl_lim = jnp.where(ekz > 0.0, 0.75 / ekz, jnp.inf)
    dtlim = jnp.minimum(dtpbl, jnp.min(cfl_lim))
    # noconv mass-flux CFL: rz=(zf[kcbl]-zf[1])*dzhi[0].
    rz = (zf[kcbl] - zf[1]) * dzhi[0]
    mf_lim = jnp.where(jnp.logical_and(noconv == 1, mbarks[1] * rz > 0.0), 0.5 / (mbarks[1] * rz), jnp.inf)
    dtlim = jnp.minimum(dtlim, mf_lim)

    nlp = (dtpbl / dtlim + 1.0).astype(jnp.int32)
    dts = dtpbl / nlp.astype(jnp.float64)

    is_nc = noconv == 1  # traced bool

    def _substep(vci, i):
        # i in 0.._NLP_MAX-1; active when i < nlp. The coefficient build mirrors the
        # reference's STATIC k-loops (range(2,kcbl+1)/range(kcbl+1,kl+1)/range(2,kl+1))
        # by iterating k over the full static range with a traced `k<=kcbl` mask, so
        # the kcbl-dependent band membership is data-dependent but the loop bound is
        # static (unrolls into a fixed graph). 1-based coefficient lists, index 0
        # padding (matches the WRF arrays).
        active = i < nlp
        z = jnp.zeros(nsp)
        ai = [jnp.asarray(0.0)] * (kl + 1)
        bi = [jnp.asarray(0.0)] * (kl + 1)
        ci = [jnp.asarray(0.0)] * (kl + 1)
        ei = [jnp.asarray(0.0)] * (kl + 1)
        di = [z] * (kl + 1)

        # --- mass-flux band: for k in range(2, kcbl+1) ---  (masked by k<=kcbl & noconv)
        for k in range(2, kl + 1):
            in_band = jnp.logical_and(is_nc, k <= kcbl)
            # ei[k-1] -= CRANKP*mdwn[k]*dts*dzh[k-1]*dzhi[k-2]
            ei[k - 1] = ei[k - 1] + jnp.where(
                in_band, -CRANKP * mdwn[k] * dts * dzh[k - 1] * dzhi[k - 2], 0.0
            )
            bi[k] = bi[k] + jnp.where(in_band, 1.0 + CRANKP * mdwn[k] * dts, 0.0)
            ai[k] = ai[k] + jnp.where(in_band, -CRANKP * mbarks[k] * dts, 0.0)

        # ei[1] -= eddy[0]*CRANKP*dzhi[0]*dzfi[0]*dts ; ai[2] -= eddy[0]*...*dzhi[1].
        # UNCONDITIONAL in WRF (lines outside the `if noconv` block): ei[1] couples
        # level 1<->0 in BOTH the TRI (noconv=0) and MATRIX (noconv=1) solves. (ai is
        # only read by MATRIX, but set unconditionally for exact parity.)
        ei[1] = ei[1] - eddy_out[0] * CRANKP * dzhi[0] * dzfi[0] * dts
        ai[2] = ai[2] - eddy_out[0] * CRANKP * dzhi[1] * dzfi[0] * dts

        # --- bi[k]=1 for k in range(kcbl+1, kl+1) ---  (UNCONDITIONAL in WRF: this
        # loop is OUTSIDE the `if noconv` block). For noconv=0, kcbl=1 so it sets
        # bi[k]=1 for every k=2..kl (the local-diffusion base); for noconv=1 it sets
        # the above-PBL part. Mask is k>=kcbl+1 only -- NOT gated by noconv.
        for k in range(2, kl + 1):
            bi[k] = jnp.where(k >= (kcbl + 1), 1.0, bi[k])

        # --- diffusion band: for k in range(2, kl+1) ---
        xplus = [jnp.asarray(0.0)] * (kl + 1)
        xminus = [jnp.asarray(0.0)] * (kl + 1)
        for k in range(2, kl + 1):
            xplus[k] = eddy_out[k - 1] * dzhi[k - 1] * dzfi[k - 1] * dts
            xminus[k] = eddy_out[k - 2] * dzhi[k - 1] * dzfi[k - 2] * dts
            ci[k] = -xminus[k] * CRANKP
            ei[k] = ei[k] - xplus[k] * CRANKP
            bi[k] = bi[k] + xplus[k] * CRANKP + xminus[k] * CRANKP

        # --- bi[1] ---
        bi1_nc = 1.0 + CRANKP * mbarks[1] * (pbl - zf[1]) * dts * dzhi[0] + eddy_out[0] * CRANKP * dzhi[0] * dzfi[0] * dts
        bi1_loc = 1.0 + eddy_out[0] * CRANKP * dzhi[0] * dzfi[0] * dts
        bi[1] = jnp.where(is_nc, bi1_nc, bi1_loc)

        # --- RHS di ---
        # k in range(2, kcbl+1): mass-flux explicit part (noconv & k<=kcbl);
        # k in range(kcbl+1, kl+1): di[k]=vci[k-1].
        for k in range(2, kl + 1):
            in_mf = jnp.logical_and(is_nc, k <= kcbl)
            delc = dts * (
                mbarks[k] * vci[:, 0]
                - mdwn[k] * vci[:, k - 1]
                + dzh[k] * dzhi[k - 1] * mdwn[k + 1] * vci[:, k]
            )
            di_mf = vci[:, k - 1] + (1.0 - CRANKP) * delc
            di[k] = jnp.where(in_mf, di_mf, vci[:, k - 1])

        # diffusion explicit for k in range(2, kl+1).
        for k in range(2, kl + 1):
            if k == kl:
                add = -(1.0 - CRANKP) * xminus[k] * (vci[:, k - 1] - vci[:, k - 2])
            else:
                add = (1.0 - CRANKP) * xplus[k] * (vci[:, k] - vci[:, k - 1]) - (1.0 - CRANKP) * xminus[k] * (vci[:, k - 1] - vci[:, k - 2])
            di[k] = di[k] + add

        # di[1].
        di1_nc = vci[:, 0] + (
            surface_flux - (1.0 - CRANKP) * (mbarks[1] * (pbl - zf[1]) * vci[:, 0] - mdwn[2] * vci[:, 1] * dzh[1])
        ) * dzhi[0] * dts
        di1_loc = vci[:, 0] + surface_flux * dzhi[0] * dts
        di[1] = jnp.where(is_nc, di1_nc, di1_loc)
        di[1] = di[1] + (1.0 - CRANKP) * eddy_out[0] * dzhi[0] * dzfi[0] * dts * (vci[:, 1] - vci[:, 0])

        ai_a = jnp.stack(ai)
        bi_a = jnp.stack(bi)
        ci_a = jnp.stack(ci)
        ei_a = jnp.stack(ei)
        di_a = jnp.stack(di, axis=1)  # (nsp, kl+1)

        ui_mat = _matrix_solve_1based_traceable(ai_a, bi_a, ci_a, di_a, ei_a, kl)
        ui_tri = _tri_solve_1based_traceable(ci_a, bi_a, ei_a, di_a, kl)
        ui = jnp.where(is_nc, ui_mat, ui_tri)

        vci_new = ui[:, 1:]  # ui[:,1..kl] -> vci[:,0..kl-1]
        vci_out = jnp.where(active, vci_new, vci)
        return vci_out, None

    vci, _ = jax.lax.scan(_substep, values, jnp.arange(_NLP_MAX, dtype=jnp.int32))
    return vci, eddy_out, fsacm


def _acm2_column_traceable(
    u, v, theta, temperature, qv, qc, qi, density, dz,
    *, pblh_initial, ust, hfx, qfx, wspd, mut, dt, xtime, g, rd, cpd, ep1,
):
    """jax.lax-traceable single ACM2 column. 1:1 with :func:`_acm2_numpy`."""

    nz = u.shape[0]
    cpair = cpd * (1.0 + 0.84 * qv[0])
    tmpfx = hfx / (cpair * density[0])
    tmpvtcon = 1.0 + ep1 * qv[0]
    ws1 = jnp.sqrt(u[0] ** 2 + v[0] ** 2)
    tst = -tmpfx / ust
    qst = -qfx / (ust * density[0])
    ustm = ust * ws1 / wspd
    theta_v = theta * (1.0 + ep1 * qv)
    thv1 = tmpvtcon * theta[0]
    tstv0 = tst * tmpvtcon + thv1 * ep1 * qst
    # Fortran SIGN(1e-6, tstv) when |tstv|<1e-6.
    tstv = jnp.where(
        jnp.abs(tstv0) < 1.0e-6,
        jnp.where(jnp.signbit(tstv0), -1.0e-6, 1.0e-6),
        tstv0,
    )
    mol = thv1 * ust ** 2 / (KARMAN * g * tstv)
    rmol = 1.0 / mol
    wst = ust * (pblh_initial / (KARMAN * jnp.abs(mol))) ** 0.333333

    zf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz)])  # (nz+1,)
    za = 0.5 * (zf[:-1] + zf[1:])
    dzh = zf[1:] - zf[:-1]
    dzhi = 1.0 / dzh
    dzfi = jnp.concatenate([1.0 / (za[1:] - za[:-1]), (1.0 / (za[-1] - za[-2]))[None]])

    pblh, kpbl, rib = _diagnose_pbl_height_traceable(
        theta_v, u, v, za, zf, mol, xtime, ust, wst, tstv, g
    )

    noconv = jnp.where(
        jnp.logical_and(
            jnp.logical_and(pblh / mol < -0.02, kpbl > 3),
            jnp.logical_and(theta_v[0] > theta_v[1], xtime > 1),
        ),
        jnp.asarray(1, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )
    regime = jnp.where(noconv == 1, 4.0, 0.0)

    eddyz, eddyzm = _eddyx_traceable(
        zf, za, mol, pblh, ust, u, v, temperature, theta_v, qv, qc, qi, g, rd, cpair
    )

    scalar_values = jnp.stack([theta, qv, qc, qi])  # (4, nz)
    scalar_flux = jnp.stack([-ust * tst, -ust * qst, jnp.asarray(0.0), jnp.asarray(0.0)])
    scalar_new, exch_h, fsacm_h = _mix_acm_values_traceable(
        dt, noconv, zf, dzh, dzhi, dzfi, kpbl, pblh, mol, ust, eddyz, scalar_values, scalar_flux
    )

    momentum_values = jnp.stack([u, v])  # (2, nz)
    fm = -ustm * ustm
    wind1 = jnp.sqrt(u[0] * u[0] + v[0] * v[0]) + 1.0e-9
    momentum_flux = jnp.stack([fm * u[0] / wind1, fm * v[0] / wind1])
    momentum_new, exch_m, fsacm_m = _mix_acm_values_traceable(
        dt, noconv, zf, dzh, dzhi, dzfi, kpbl, pblh, mol, ust, eddyzm, momentum_values, momentum_flux
    )

    rdt = 1.0 / dt
    u_tend = (momentum_new[0] - u) * rdt
    v_tend = (momentum_new[1] - v) * rdt
    theta_tend = (scalar_new[0] - theta) * rdt
    qv_tend = (scalar_new[1] - qv) * rdt
    return (
        u_tend, v_tend, theta_tend, qv_tend,
        exch_h, exch_m, pblh, kpbl.astype(jnp.int32), regime, noconv, rmol, fsacm_h, fsacm_m,
    )


def acm2_columns(
    u, v, theta, temperature, qv, density, dz,
    *, pblh_initial, ust, hfx, qfx, wspd, mut, dt, xtime,
    qc=None, qi=None, g=G_DEFAULT, rd=R_D_DEFAULT, cpd=CP_DEFAULT, ep1=EP1_DEFAULT,
):
    """Batched, jit/vmap-traceable ACM2 over ``(ncol,)`` columns.

    Profile inputs ``(ncol, nz)``; surface inputs ``(ncol,)``. Returns a dict of
    ``(ncol, nz)`` tendencies and diagnostics. This is the operational scan entry.
    """

    ncol = u.shape[0]
    nz = u.shape[1]
    if qc is None:
        qc = jnp.zeros((ncol, nz), jnp.float64)
    if qi is None:
        qi = jnp.zeros((ncol, nz), jnp.float64)
    xtime_b = jnp.broadcast_to(jnp.asarray(xtime, jnp.float64), (ncol,))

    out = jax.vmap(
        lambda *a: _acm2_column_traceable(
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
            pblh_initial=a[9], ust=a[10], hfx=a[11], qfx=a[12], wspd=a[13],
            mut=a[14], dt=dt, xtime=a[15], g=g, rd=rd, cpd=cpd, ep1=ep1,
        )
    )(
        u, v, theta, temperature, qv, qc, qi, density, dz,
        pblh_initial, ust, hfx, qfx, wspd, mut, xtime_b,
    )
    (u_t, v_t, th_t, qv_t, exch_h, exch_m, pblh, kpbl, regime, noconv, rmol, fsacm_h, fsacm_m) = out
    return {
        "u": u_t, "v": v_t, "theta": th_t, "qv": qv_t,
        "exch_h": exch_h, "exch_m": exch_m,
        "pblh": pblh, "kpbl": kpbl, "regime": regime, "noconv": noconv,
        "rmol": rmol, "fsacm_h": fsacm_h, "fsacm_m": fsacm_m,
    }


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

    Backed by the ``jax.lax.scan``-traceable :func:`_acm2_column_traceable` (a 1:1
    transcription of the host-NumPy reference; the per-case savepoint parity
    asserts the traceable path equals WRF within the predeclared tolerances).
    """

    u_a = jnp.asarray(u, jnp.float64)
    nz = u_a.shape[0]
    qc_a = jnp.zeros(nz, jnp.float64) if qc is None else jnp.asarray(qc, jnp.float64)
    qi_a = jnp.zeros(nz, jnp.float64) if qi is None else jnp.asarray(qi, jnp.float64)
    out = _acm2_column_traceable(
        u_a,
        jnp.asarray(v, jnp.float64),
        jnp.asarray(theta, jnp.float64),
        jnp.asarray(temperature, jnp.float64),
        jnp.asarray(qv, jnp.float64),
        qc_a,
        qi_a,
        jnp.asarray(density, jnp.float64),
        jnp.asarray(dz, jnp.float64),
        pblh_initial=jnp.asarray(pblh_initial, jnp.float64),
        ust=jnp.asarray(ust, jnp.float64),
        hfx=jnp.asarray(hfx, jnp.float64),
        qfx=jnp.asarray(qfx, jnp.float64),
        wspd=jnp.asarray(wspd, jnp.float64),
        mut=jnp.asarray(mut, jnp.float64),
        dt=jnp.asarray(dt, jnp.float64),
        xtime=jnp.asarray(xtime, jnp.float64),
        g=jnp.asarray(g, jnp.float64),
        rd=jnp.asarray(rd, jnp.float64),
        cpd=jnp.asarray(cpd, jnp.float64),
        ep1=jnp.asarray(ep1, jnp.float64),
    )
    (u_t, v_t, th_t, qv_t, exch_h, exch_m, pblh, kpbl, regime, noconv, rmol, fsacm_h, fsacm_m) = out

    tendencies = {
        "u": u_t,
        "v": v_t,
        "theta": th_t,
        "qv": qv_t,
    }
    diagnostics = {
        "pblh": pblh,
        "kpbl": kpbl.astype(jnp.int32),
        "regime": regime,
        "noconv": noconv.astype(jnp.int32),
        "rmol": rmol,
        "exch_h": exch_h,
        "exch_m": exch_m,
        "fsacm_h": fsacm_h,
        "fsacm_m": fsacm_m,
    }
    tendency = PhysicsTendency(state_tendencies=tendencies)
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        diagnostics=PhysicsDiagnostics(pbl=diagnostics),
    )


__all__ = ["ACM2ColumnState", "ACM2Diagnostics", "step_acm2_column", "acm2_columns"]
