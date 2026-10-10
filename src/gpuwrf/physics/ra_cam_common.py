"""Shared constants, tables and column set-up of the WRF CAM radiation port (``ra_lw/sw_physics = 3``).

Source: pristine WRF V4.7.1 ``phys/module_ra_cam.F`` + ``phys/module_ra_cam_support.F`` (CAM3 radiation, Collins et al.).
CAM radiation computes in ``real(r8)`` (DOUBLE) inside WRF, so the port is float64 like WRF; the WRF-side REAL (float32)
operands and REAL expressions of the ``camrad`` wrapper are reproduced in float32 before widening.

Precision rule used throughout (gfortran, no ``-fdefault-real-8``): an unsuffixed literal such as ``1.66`` or a
``data``/``parameter`` initialiser of an ``r8`` entity is a single-precision constant widened to double -> :func:`lit`.

Arrays follow CAM's vertical order inside the scheme (index 0 = model top, ``pver-1`` = lowest layer; interfaces
``pverp = pver + 1``) and carry the column batch on the leading axis.  ``ntoplw = 1`` (WRF ``radini``) everywhere.

Oracle: ``proofs/cam_rad`` (CAM01, pristine ``camrad`` true caller on real WN3 columns).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gpuwrf.config.paths import wrf_run_path

f32 = np.float32


def lit(x) -> float:
    """A gfortran default-REAL literal used in an ``r8`` context: rounded to float32, widened exactly."""

    return float(np.float32(x))


# --------------------------------------------------------------------------------------------------------------------- #
# Constants (module_model_constants REAL -> camradinit -> radini -> radaeini -> esinti), computed with WRF's precision.  #
# --------------------------------------------------------------------------------------------------------------------- #
_R_D = f32(287.0)
_R_V = f32(461.6)
_CP = f32(f32(7.0) * _R_D) / f32(2.0)
_G = f32(9.81)
_STBOLT = f32(5.67051e-8)
_EP_2 = _R_D / _R_V
_PICONST = f32(3.1415926535897932384626433)
DEGRAD = float(_PICONST / f32(180.0))


class CamConstants(NamedTuple):
    # radini (cgs units)
    gravit: float
    rga: float
    gravmks: float
    cpair: float
    epsilo: float
    sslp: float
    stebol: float
    rgsslp: float
    dpfo3: float
    dpfco2: float
    dayspy: float
    pie: float
    cplos: float
    cplol: float
    # camradinit physconst
    mwdry: float
    mwco2: float
    mwh2o: float
    mwch4: float
    mwn2o: float
    mwf11: float
    mwf12: float
    cappa: float
    rair: float
    tmelt: float
    r_universal: float
    latvap: float
    latice: float
    zvir: float
    rh2o: float
    epsqs: float
    # radaeini
    p0: float
    amd: float
    amco2: float
    c16: float
    c17: float
    c26: float
    c27: float
    c28: float
    c29: float
    c30: float
    c31: float
    fwcoef: float
    fwc1: float
    fwc2: float
    fc1: float
    # module default trace gas (radtpl/radems/radabs read the MODULE co2vmr, never updated by camrad)
    co2vmr_module: float


def _cam_constants() -> CamConstants:
    gravit = float(f32(100.0) * _G)
    sslp = lit(1.013250e6)
    v0 = lit(22.4136)
    p0_local = lit(0.1) * sslp
    amd_local = lit(28.9644)
    goz = float(_G)
    mwdry = lit(28.966)
    coefj = np.asarray([[2.82096e-02, 2.47836e-04, 1.16904e-06], [9.27379e-02, 8.04454e-04, 6.88844e-06]], f32).astype(
        np.float64)
    coefk = np.asarray([[2.48852e-01, 2.09667e-03, 2.60377e-06], [1.03594e+00, 6.58620e-03, 4.04456e-06]], f32).astype(
        np.float64)
    return CamConstants(
        gravit=gravit,
        rga=1.0 / gravit,
        gravmks=float(_G),
        cpair=float(f32(1.0e4) * _CP),
        epsilo=float(_EP_2),
        sslp=sslp,
        stebol=float(f32(1.0e3) * _STBOLT),
        rgsslp=0.5 / (gravit * sslp),
        dpfo3=lit(2.5e-3),
        dpfco2=lit(5.0e-3),
        dayspy=365.0,
        pie=float(f32(4.0) * np.arctan(f32(1.0))),
        cplos=v0 / (amd_local * goz) * 100.0,
        cplol=v0 / (amd_local * goz * p0_local) * 0.5 * 100.0,
        mwdry=mwdry,
        mwco2=44.0,
        mwh2o=lit(18.016),
        mwch4=16.0,
        mwn2o=44.0,
        mwf11=136.0,
        mwf12=120.0,
        cappa=float(_R_D / _CP),
        rair=float(_R_D),
        tmelt=lit(273.16),
        r_universal=float(f32(6.02214e26) * _STBOLT),
        latvap=lit(2.501e6),
        latice=lit(3.336e5),
        zvir=float(_R_V / _R_D - f32(1.0)),
        rh2o=float(_R_V),
        epsqs=float(_EP_2),
        p0=101325.0 * 10.0,
        amd=mwdry,
        amco2=44.0,
        c16=coefj[0, 2] / coefj[0, 1],
        c17=coefk[0, 2] / coefk[0, 1],
        c26=coefj[1, 2] / coefj[1, 1],
        c27=coefk[1, 2] / coefk[1, 1],
        c28=lit(0.5),
        c29=lit(0.002053),
        c30=lit(0.1),
        c31=lit(3.0e-5),
        fwcoef=lit(0.1),
        fwc1=lit(0.30),
        fwc2=lit(4.5),
        fc1=lit(2.6),
        co2vmr_module=lit(3.550e-4),
    )


CONST = _cam_constants()

# radae.F90 (module_ra_cam_support.F:87-117) table geometry -- r8 parameters from REAL literals.
MIN_TP_H2O = lit(160.0)
MAX_TP_H2O = lit(349.999999)
DTP_H2O = lit(21.111111111111)
MIN_TE_H2O = lit(-120.0)
MAX_TE_H2O = lit(79.999999)
DTE_H2O = lit(10.0)
MIN_RH_H2O = lit(0.0)
MAX_RH_H2O = lit(1.19999999)
DRH_H2O = lit(0.2)
MIN_LU_H2O = lit(-8.0)
MIN_U_H2O = lit(1.0e-8)
MAX_LU_H2O = lit(3.9999999)
DLU_H2O = lit(0.5)
MIN_LP_H2O = lit(-3.0)
MIN_P_H2O = lit(1.0e-3)
MAX_LP_H2O = lit(-0.0000001)
DLP_H2O = lit(0.3333333333333)
N_U, N_P, N_TP, N_TE, N_RH = 25, 10, 10, 21, 7
TABLE_SHAPE = (N_P, N_TP, N_U, N_TE, N_RH)

# coefh/coefj/coefk (Ramanathan & Downey 1986), fat/fet (module_ra_cam_support.F:125-171), Fortran (i, j) -> [i-1, j-1].
COEFH = np.asarray([[5.46557e+01, -7.30387e-02], [1.09311e+02, -1.46077e-01], [5.11479e+01, -6.82615e-02],
                    [1.02296e+02, -1.36523e-01]], f32).astype(np.float64).T  # coefh(2,4)
COEFJ = np.asarray([[2.82096e-02, 2.47836e-04, 1.16904e-06], [9.27379e-02, 8.04454e-04, 6.88844e-06]], f32).astype(
    np.float64).T  # coefj(3,2)
COEFK = np.asarray([[2.48852e-01, 2.09667e-03, 2.60377e-06], [1.03594e+00, 6.58620e-03, 4.04456e-06]], f32).astype(
    np.float64).T  # coefk(3,2)
FAT = np.asarray([[-1.06665373E-01, 2.90617375E-02, -2.70642049E-04, 1.07595511E-06, -1.97419681E-09, 1.37763374E-12],
                  [1.10666537E+00, -2.90617375E-02, 2.70642049E-04, -1.07595511E-06, 1.97419681E-09, -1.37763374E-12]],
                 f32).astype(np.float64).T  # fat(o_fa, nbands)
FET = np.asarray([[3.46148163E-01, 1.51240299E-02, -1.21846479E-04, 4.04970123E-07, -6.15368936E-10, 3.52415071E-13],
                  [6.53851837E-01, -1.51240299E-02, 1.21846479E-04, -4.04970123E-07, 6.15368936E-10, -3.52415071E-13]],
                 f32).astype(np.float64).T  # fet(o_fe, nbands)

_PSI_R = np.asarray([[5.65308452E-01, -7.30087891E+01], [4.07519005E-03, 1.22199547E+00],
                     [-1.04347237E-05, -7.12256227E-03], [1.23765354E-08, 1.47852825E-05]], f32).astype(np.float64)
_PHI_R = np.asarray([[9.60917711E-01, -2.21031342E+01], [4.86076751E-04, 4.24062610E-01],
                     [-1.84806265E-06, -2.95543415E-03], [2.11239959E-09, 7.52470896E-06]], f32).astype(np.float64)

RETAB = np.asarray([
    5.92779, 6.26422, 6.61973, 6.99539, 7.39234, 7.81177, 8.25496, 8.72323, 9.21800, 9.74075, 10.2930, 10.8765, 11.4929,
    12.1440, 12.8317, 13.5581, 14.2319, 15.0351, 15.8799, 16.7674, 17.6986, 18.6744, 19.6955, 20.7623, 21.8757, 23.0364,
    24.2452, 25.5034, 26.8125, 27.7895, 28.6450, 29.4167, 30.1088, 30.7306, 31.2943, 31.8151, 32.3077, 32.7870, 33.2657,
    33.7540, 34.2601, 34.7892, 35.3442, 35.9255, 36.5316, 37.1602, 37.8078, 38.4720, 39.1508, 39.8442, 40.5552, 41.2912,
    42.0635, 42.8876, 43.7863, 44.7853, 45.9170, 47.2165, 48.7221, 50.4710, 52.4980, 54.8315, 57.4898, 60.4785, 63.7898,
    65.5604, 71.2885, 75.4113, 79.7368, 84.2351, 88.8833, 93.6658, 98.5739, 103.603, 108.752, 114.025, 119.424, 124.954,
    130.630, 136.457, 142.446, 148.608, 154.956, 161.503, 168.262, 175.248, 182.473, 189.952, 197.699, 205.728, 214.055,
    222.694, 231.661, 240.971, 250.639], f32).astype(np.float64)


def psi(tpx, iband: int):
    """module_ra_cam_support.F psi(tpx, iband); iband is Fortran 1/2."""

    r0, r1, r2, r3 = (_PSI_R[j, iband - 1] for j in range(4))
    return (((r3 * tpx) + r2) * tpx + r1) * tpx + r0


def phi(tpx, iband: int):
    r0, r1, r2, r3 = (_PHI_R[j, iband - 1] for j in range(4))
    return (((r3 * tpx) + r2) * tpx + r1) * tpx + r0


def fh2oself(temp):
    return jnp.power(lit(2.0727484), (lit(296.0) - temp) / lit(36.0))


# --------------------------------------------------------------------------------------------------------------------- #
# Tables                                                                                                                #
# --------------------------------------------------------------------------------------------------------------------- #
class CamAbsTables(NamedTuple):
    ah2onw: np.ndarray
    eh2onw: np.ndarray
    ah2ow: np.ndarray
    cn_ah2ow: np.ndarray
    cn_eh2ow: np.ndarray
    ln_ah2ow: np.ndarray
    ln_eh2ow: np.ndarray
    estblh2o: np.ndarray  # (ntemp + 1,) = estblh2o(0:192)


def _gffgch_water(t: np.ndarray) -> np.ndarray:
    """gffgch(t, es, itype=0): Goff-Gratch over water, r8, host libm (glibc pow/log10 like gfortran)."""

    ps = lit(1013.246)
    ts = lit(373.16)
    e1 = lit(11.344) * (1.0 - t / ts)
    e2 = lit(-3.49149) * (ts / t - 1.0)
    f1 = lit(-7.90298) * (ts / t - 1.0)
    f2 = lit(5.02808) * np.log10(ts / t)
    f3 = lit(-1.3816) * (np.power(10.0, e1) - 1.0) / 10000000.0
    f4 = lit(8.1328) * (np.power(10.0, e2) - 1.0) / 1000.0
    f5 = np.log10(ps)
    f = f1 + f2 + f3 + f4 + f5
    return (np.power(10.0, f)) * 100.0


def _read_fortran_records(path: Path, count: int) -> list[np.ndarray]:
    """Sequential unformatted, big-endian 4-byte record markers (WRF -fconvert=big-endian), float64 payloads."""

    raw = path.read_bytes()
    out, pos = [], 0
    for _ in range(count):
        n = int(np.frombuffer(raw, ">i4", 1, pos)[0])
        pos += 4
        out.append(np.frombuffer(raw, ">f8", n // 8, pos).astype(np.float64))
        pos += n
        if int(np.frombuffer(raw, ">i4", 1, pos)[0]) != n:
            raise RuntimeError(f"CAM_ABS_DATA record marker mismatch in {path}")
        pos += 4
    if pos != len(raw):
        raise RuntimeError(f"CAM_ABS_DATA trailing bytes in {path}")
    return out


@lru_cache(maxsize=2)
def load_cam_abs_tables(path: str | None = None) -> CamAbsTables:
    """radaeini: the seven water-vapour abs/ems tables of ``run/CAM_ABS_DATA`` + the estblh2o Goff-Gratch table."""

    p = Path(path) if path is not None else wrf_run_path("CAM_ABS_DATA")
    recs = _read_fortran_records(p, 7)
    tabs = [r.reshape(TABLE_SHAPE, order="F") for r in recs]
    tmin = int(np.rint(MIN_TP_H2O))
    tmax = int(np.rint(MAX_TP_H2O)) + 1
    est = np.zeros(193, np.float64)
    temps = np.arange(tmin, tmax + 1, dtype=np.float64)
    est[: temps.size] = _gffgch_water(temps)
    return CamAbsTables(*tabs, est)


# --------------------------------------------------------------------------------------------------------------------- #
# Ozone: oznini (REAL latitude interpolation) -> oznint (r8 time) -> radozn (r8 pressure)                                #
# --------------------------------------------------------------------------------------------------------------------- #
_DATE_OZ = np.asarray([16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350], np.float64)
LEVSIZ = 59


def cam_ozmixm(xlat_deg) -> tuple[jnp.ndarray, jnp.ndarray]:
    """oznini: REAL lin_interpol2 of the CAM monthly ozone to each column latitude.

    Returns ``ozmixm (ncol, 12, levsiz)`` float32 (months Jan..Dec = WRF slices 2..13) and ``pin (levsiz,)`` float32 Pa.
    """

    from gpuwrf.physics.wrf_cam_ozone import _load_wrf_cam_ozone_asset

    lat_tab, p_tab, oz_tab = (jnp.asarray(a, jnp.float32) for a in _load_wrf_cam_ozone_asset())
    lat = jnp.asarray(xlat_deg, jnp.float32)
    k = jnp.clip(jnp.searchsorted(lat_tab, lat, side="left") - 1, 0, lat_tab.shape[0] - 2)
    x0 = jnp.take(lat_tab, k, mode="clip")
    x1 = jnp.take(lat_tab, k + 1, mode="clip")
    f0 = jnp.moveaxis(jnp.take(oz_tab, k, axis=1, mode="clip"), 1, 0)       # (ncol, 12, levsiz)
    f1 = jnp.moveaxis(jnp.take(oz_tab, k + 1, axis=1, mode="clip"), 1, 0)
    a = (f1 - f0) / (x1 - x0)[:, None, None]
    return (f0 + a * (lat - x0)[:, None, None]).astype(jnp.float32), p_tab


def oznint_factors(julian):
    """oznint/getfactors(ozncyc=.true.): month indices (0-based) and r8 factors for WRF's zero-based REAL julian.

    Elementwise in ``julian`` (scalar or per column)."""

    jul = jnp.asarray(julian, jnp.float32).astype(jnp.float64) + 1.0
    ijul = jul.astype(jnp.int32)
    jul = jul - ijul.astype(jnp.float32).astype(jnp.float64)
    ijul = jnp.mod(ijul, 365)
    ijul = jnp.where(ijul == 0, 365, ijul)
    jul = jul + ijul.astype(jnp.float64)
    dates = jnp.asarray(_DATE_OZ)
    after = dates > jul[..., None]
    np1 = jnp.where(jnp.any(after, axis=-1), jnp.argmax(after, axis=-1), 0)   # 0-based; Fortran default np1 = 1
    cdayp = jnp.take(dates, np1, mode="clip")
    nm = jnp.where(np1 > 0, np1 - 1, 11)
    cdaym = jnp.take(dates, nm, mode="clip")
    dpy = 365.0
    cyc = np1 == 0
    deltat = jnp.where(cyc, cdayp + dpy - cdaym, cdayp - cdaym)
    fact1 = jnp.where(cyc, jnp.where(jul > cdayp, (cdayp + dpy - jul) / deltat, (cdayp - jul) / deltat),
                      (cdayp - jul) / deltat)
    fact2 = jnp.where(cyc, jnp.where(jul > cdayp, (jul - cdaym) / deltat, (jul + dpy - cdaym) / deltat),
                      (jul - cdaym) / deltat)
    return nm, np1, fact1, fact2


def oznint(ozmixm, julian):
    """``ozmix (ncol, levsiz)`` r8 = ozmixmj(nm)*fact1 + ozmixmj(np)*fact2 (ozmixmj = REAL ozmixm widened)."""

    nm, np1, fact1, fact2 = oznint_factors(julian)
    oz = jnp.asarray(ozmixm).astype(jnp.float64)
    if nm.ndim == 0:
        return jnp.take(oz, nm, axis=1, mode="clip") * fact1 + jnp.take(oz, np1, axis=1, mode="clip") * fact2
    pick = lambda m: jnp.take_along_axis(oz, m[:, None, None], axis=1, mode="clip")[:, 0]
    return pick(nm) * fact1[:, None] + pick(np1) * fact2[:, None]


def radozn(pmid, pin, ozmix):
    """radozn: r8 linear-in-p ozone at layer midpoints (pmid in Pa, CAM order); pin REAL widened."""

    pin8 = jnp.asarray(pin, jnp.float32).astype(jnp.float64)
    kk = jnp.clip(jnp.searchsorted(pin8, pmid, side="left") - 1, 0, LEVSIZ - 2)  # pin(kk) < p <= pin(kk+1)
    p0 = jnp.take(pin8, kk, mode="clip")
    p1 = jnp.take(pin8, kk + 1, mode="clip")
    o0 = jnp.take_along_axis(ozmix, kk, axis=-1, mode="clip")
    o1 = jnp.take_along_axis(ozmix, kk + 1, axis=-1, mode="clip")
    dpu = pmid - p0
    dpl = p1 - pmid
    inside = (o0 * dpl + o1 * dpu) / (dpl + dpu)
    above = ozmix[:, :1] * pmid / pin8[0]
    below = ozmix[:, -1:]
    return jnp.where(pmid < pin8[0], above, jnp.where(pmid > pin8[-1], below, inside))


def radinp(pmid, pint, o3vmr):
    """radinp: cgs pressures (pbr, pnm), eccf = 1, o3mmr = (amo/amd)*o3vmr."""

    vmmr = 48.0 / CONST.amd
    return pmid * 10.0, pint * 10.0, 1.0, vmmr * o3vmr


# --------------------------------------------------------------------------------------------------------------------- #
# Trace gases: trcmix_clwrf (ghg_input = 1)                                                                             #
# --------------------------------------------------------------------------------------------------------------------- #
def trcmix_clwrf(pmid, clat, n2ovmr, ch4vmr, f11vmr, f12vmr):
    """CLWRF trace-gas mass mixing ratios (n2o, ch4, cfc11, cfc12), r8, CAM order."""

    c = CONST
    rmwn2o = c.mwn2o / c.mwdry
    rmwch4 = c.mwch4 / c.mwdry
    rmwf11 = c.mwf11 / c.mwdry
    rmwf12 = c.mwf12 / c.mwdry
    coslat = jnp.cos(clat)[:, None]
    ch40 = rmwch4 * ch4vmr
    n2o0 = rmwn2o * n2ovmr
    cfc110 = rmwf11 * f11vmr
    cfc120 = rmwf12 * f12vmr
    dlat = jnp.abs(lit(57.2958) * clat)[:, None]
    lo = dlat <= 45.0
    xn2o = jnp.where(lo, lit(0.3478) + lit(0.00116) * dlat, lit(0.4000) + lit(0.013333) * (dlat - 45))
    xch4 = jnp.where(lo, lit(0.2353), lit(0.2353) + lit(0.0225489) * (dlat - 45))
    xcfc11 = jnp.where(lo, lit(0.7273) + lit(0.00606) * dlat, 1.00 + lit(0.013333) * (dlat - 45))
    xcfc12 = jnp.where(lo, lit(0.4000) + lit(0.00222) * dlat, lit(0.50) + lit(0.024444) * (dlat - 45))
    ptrop = lit(250.0e2) - lit(150.0e2) * (coslat * coslat)  # gfortran folds x**2.0 -> x*x
    trop = pmid >= ptrop
    pratio = pmid / ptrop
    ch4 = jnp.where(trop, ch40, ch40 * jnp.power(pratio, xch4))
    n2o = jnp.where(trop, n2o0, n2o0 * jnp.power(pratio, xn2o))
    cfc11 = jnp.where(trop, cfc110, cfc110 * jnp.power(pratio, xcfc11))
    cfc12 = jnp.where(trop, cfc120, cfc120 * jnp.power(pratio, xcfc12))
    return n2o, ch4, cfc11, cfc12


# --------------------------------------------------------------------------------------------------------------------- #
# camrad preprocessing (module_ra_cam.F:512-700) and param_cldoptics_calc                                                #
# --------------------------------------------------------------------------------------------------------------------- #
class CamColumns(NamedTuple):
    """radctl operands in CAM order (r8)."""

    q1: jnp.ndarray       # specific humidity (ncol, pver)
    qliq: jnp.ndarray
    qice: jnp.ndarray
    cld: jnp.ndarray
    pmid: jnp.ndarray     # Pa
    pint: jnp.ndarray     # Pa (ncol, pverp)
    lnpmid: jnp.ndarray
    lnpint: jnp.ndarray
    pdel: jnp.ndarray
    t: jnp.ndarray
    ps: jnp.ndarray
    lwups: jnp.ndarray    # stebol*emiss*tsk**4 (cgs)
    landfrac: jnp.ndarray
    landm: jnp.ndarray
    snowh: jnp.ndarray
    icefrac: jnp.ndarray
    clat: jnp.ndarray
    coszrs: jnp.ndarray
    albedo: jnp.ndarray


def camrad_prepare(t_phy, p_phy, p8w, qv, qc, qi, qs, cldfra, xland, xice, snow, emiss, tsk, xlat, coszen, albedo):
    """WRF-order (k = 0 bottom) REAL operands ``(ncol, nz)`` / ``(ncol, nz+1)`` -> CAM-order r8 radctl operands.

    Moisture conversion is the Thompson/WSM branch ``F_QI .and. F_QC .and. F_QS`` (qice = qi + qs), in REAL.
    """

    r4 = lambda a: jnp.asarray(a, jnp.float32)
    w = lambda a: r4(a).astype(jnp.float64)
    flip = lambda a: a[:, ::-1]
    qv4 = r4(qv)
    one = jnp.float32(1.0)
    q1 = jnp.maximum(jnp.float32(1.0e-10), qv4 / (one + qv4))
    qliq = jnp.maximum(jnp.float32(0.0), r4(qc) / (one + qv4))
    qice = jnp.maximum(jnp.float32(0.0), (r4(qi) + r4(qs)) / (one + qv4))
    pint = flip(w(p8w))
    pmid = flip(w(p_phy))
    lnpint = jnp.log(pint)
    tsk4 = r4(tsk)
    tsk2 = tsk4 * tsk4
    landfrac = (jnp.float32(2.0) - r4(xland)).astype(jnp.float64)
    return CamColumns(
        q1=flip(q1.astype(jnp.float64)),
        qliq=flip(qliq.astype(jnp.float64)),
        qice=flip(qice.astype(jnp.float64)),
        cld=flip(w(cldfra)),
        pmid=pmid,
        pint=pint,
        lnpmid=jnp.log(pmid),
        lnpint=lnpint,
        pdel=pint[:, 1:] - pint[:, :-1],
        t=flip(w(t_phy)),
        ps=pint[:, -1],
        lwups=CONST.stebol * w(emiss) * (tsk2 * tsk2).astype(jnp.float64),
        landfrac=landfrac,
        landm=landfrac,
        snowh=(jnp.float32(0.001) * r4(snow)).astype(jnp.float64),
        icefrac=w(xice),
        clat=(r4(xlat) * jnp.float32(DEGRAD)).astype(jnp.float64),
        coszrs=w(coszen),
        albedo=w(albedo),
    )


class CloudOptics(NamedTuple):
    cicewp: jnp.ndarray
    cliqwp: jnp.ndarray
    emis: jnp.ndarray
    rel: jnp.ndarray
    rei: jnp.ndarray
    pmxrgn: jnp.ndarray   # (ncol, pverp), dynes/cm2
    nmxrgn: jnp.ndarray   # (ncol,) int


def reltab(t, landfrac, landm, icefrac, snowh):
    rliqocean, rliqice, rliqland = 14.0, 14.0, 8.0
    one = 1.0
    rel = rliqland + (rliqocean - rliqland) * jnp.minimum(one, jnp.maximum(0.0, (CONST.tmelt - t) * lit(0.05)))
    rel = rel + (rliqocean - rel) * jnp.minimum(one, jnp.maximum(0.0, snowh[:, None] * 10.0))
    rel = rel + (rliqocean - rel) * jnp.minimum(one, jnp.maximum(0.0, 1.0 - landm[:, None]))
    rel = rel + (rliqice - rel) * jnp.minimum(one, jnp.maximum(0.0, icefrac[:, None]))
    return rel


def reitab(t):
    index = jnp.clip(jnp.trunc(t - 179.0).astype(jnp.int32), 1, 94)
    corr = t - jnp.trunc(t)
    tab = jnp.asarray(RETAB)
    return jnp.take(tab, index - 1, mode="clip") * (1.0 - corr) + jnp.take(tab, index, mode="clip") * corr


def cldovrlap(pint, cld):
    """cldovrlap: max-overlap regions = contiguous cloudy blocks; boundary at the first clear layer after a block
    (interface pressure, dynes) if any cloud remains below; the last region ends at the surface."""

    pnm = pint * 10.0
    ncol, pver = cld.shape
    layer = cld > 0.0
    prev = jnp.concatenate([jnp.zeros((ncol, 1), bool), layer[:, :-1]], axis=1)
    below = jnp.flip(jnp.cumsum(jnp.flip(layer, axis=1), axis=1), axis=1)  # count(cld_layer(k:pver))
    boundary = (~layer) & prev & (below > 0)
    nb = jnp.sum(boundary, axis=1)
    rank = jnp.cumsum(boundary, axis=1) - 1
    slot = jnp.where(boundary, rank, pver + 1)      # out-of-range slots are dropped by the scatter
    pmx = jnp.zeros((ncol, pver + 1), jnp.float64)
    rows = jnp.arange(ncol)[:, None]
    pmx = pmx.at[rows, slot].set(pnm[:, :-1], mode="drop")
    pmx = pmx.at[jnp.arange(ncol), nb].set(pnm[:, -1])
    return pmx, (nb + 1).astype(jnp.int32)


def param_cldoptics_calc(cols: CamColumns) -> CloudOptics:
    c = CONST
    gicewp = cols.qice * cols.pdel / c.gravmks * 1000.0
    gliqwp = cols.qliq * cols.pdel / c.gravmks * 1000.0
    cicewp = gicewp / jnp.maximum(0.01, cols.cld)
    cliqwp = gliqwp / jnp.maximum(0.01, cols.cld)
    ficemr = cols.qice / jnp.maximum(1.0e-10, (cols.qice + cols.qliq))
    cwp = cicewp + cliqwp
    rel = reltab(cols.t, cols.landfrac, cols.landm, cols.icefrac, cols.snowh)
    rei = reitab(cols.t)
    kabsl = lit(0.090361)
    kabsi = lit(0.005) + 1.0 / rei
    kabs = kabsl * (1.0 - ficemr) + kabsi * ficemr
    emis = 1.0 - jnp.exp(-(lit(1.66) * kabs) * cwp)
    pmxrgn, nmxrgn = cldovrlap(cols.pint, cols.cld)
    return CloudOptics(cicewp, cliqwp, emis, rel, rei, pmxrgn, nmxrgn)


__all__ = [
    "CONST", "CamAbsTables", "CamColumns", "CloudOptics", "DEGRAD", "LEVSIZ", "camrad_prepare", "cam_ozmixm",
    "cldovrlap", "fh2oself", "lit", "load_cam_abs_tables", "oznint", "oznint_factors", "param_cldoptics_calc", "phi",
    "psi", "radinp", "radozn", "reitab", "reltab", "trcmix_clwrf",
]
