"""WRF CAM shortwave radiation (``ra_sw_physics = 3``): the ``dosw`` branch of ``radctl``.

Source: pristine WRF V4.7.1 ``phys/module_ra_cam.F`` (``radctl`` 1546-2073, ``get_aerosol`` 1105-1289, ``radcswmx``
5687-7563, ``raddedmx`` 7565-7839, module DATA 1-197) and ``phys/module_ra_cam_support.F`` (``aqsat``, ``gestbl``/
``esinti``/``gffgch``/``estblf``, ``background``, ``scale_aerosols``, ``get_int_scales``, ``vert_interpolate``,
``getfactors``, ``aerosol_init``, ``aer_optics_initialize``, ``findvalue``).

Only the radiatively active path is ported: ``radforce = .false.`` (the second ``get_aerosol``/``radcswmx`` call),
``aerosol_indirect`` has no outputs, and the aerosol optical diagnostics (aertau/aerssa/...) are not radctl outputs.

Precision: CAM computes in ``real(r8)``; every unsuffixed literal and every ``data``/``parameter`` initialiser of an
``r8`` entity is a single-precision constant widened (:func:`lit`), ``_r8`` literals are double.  REAL WRF state
(``solcon``, ``julian``, ``m_hybi``, the ``aerosolc`` climatology) is reproduced in float32 before widening.  The
Fortran evaluation order (left to right, parentheses as written) is kept so that the port differs from WRF only by
XLA contraction/transcendental rounding (~1e-15 relative).

Layout: CAM vertical order (index 0 = model top), the column batch on the leading axis, ``pver`` layers,
``pverp = pver + 1`` interfaces.  Inside ``radcswmx`` the extra layer above the model top is layer 0 of the
``0:pver`` arrays (as in WRF).

Structure (jit-traceable, no host callbacks): the 19-interval spectral loop is a ``lax.scan`` whose carry holds
WRF's spectral accumulators (same summation order); level recurrences of the adding method are ``lax.scan``s.  The
max-random overlap is solved per configuration (up to ``nconfgmax = 15`` per column): WRF's binary-tree sharing of
identical partial configurations computes the same numbers from the same operands, so solving each configuration
on its own is bitwise equivalent.  The configuration selection (stream weights, odometer enumeration, the streaming
top-15 table with ``findvalue``'s quickselect tie-breaking and its persistent ``ptrc`` permutation, the
``totwgt = 0`` maximum-overlap second pass) is a per-column ``lax.while_loop`` (vmapped).  Subtrees whose largest
attainable weight is below ``areamin`` are skipped: they cannot change WRF's state, the visited configurations keep
WRF's order.

Parity (CPU, f64, tests/test_ra_cam_sw_oracle.py): rh and aerosol bitwise (eager); fluxes <= 2.5e-13 relative;
diffuse (total - direct) and cloud-forcing differences <= 1.1e-11; heating rates <= 8.1e-11 (flux-divergence
cancellation of the <= 2-ulp XLA-vs-glibc ``exp`` differences); camrad's REAL heating rate bitwise on CAM01.

WRF quirks reproduced (all verified against the CAM01 pristine oracle):

* ``indxsl`` (cloud-optics band index): ``wavmin(ns) == 0.700_r8`` etc. compare a single-precision-widened DATA value
  with a double literal and are never true, so bands 10-16 keep the previous band's ``indxsl = 1`` (visible Slingo /
  Ebert-Curry coefficients for the near-IR water bands); bands 17-19 get 4 through ``wavmin > 2.38_r8``.
* ``tauxcl``/``tauxci`` (radctl outputs) are those of the LAST spectral interval (``ns = 19``, ``indxsl = 4``).
* ``fsdndir``/``fsdndif`` profiles are returned in cgs (radctl does not rescale them); ``sols``..``solld`` are
  scaled by ``0.001_r8`` inside radcswmx, the other fluxes by the REAL literal ``1.e-3`` in radctl.
* The aerosol species slots come from the Registry scalar indices ``P_SUL..P_VOLC``.  In a real WRF run
  ``set_scalar_indices_from_config`` sets them to 2..13 (:data:`WRF_AEROSOL_INDICES`, the CAM01 oracle since its
  index fix).  A caller that never runs it leaves every ``P_*`` at its declared default 1
  (:data:`UNSET_AEROSOL_INDICES`, kept for diagnostics only): then
  ``aerosol_init`` writes SUL..BCPHI all into slot 1 and DUST2..4 into slots 2..4, ``background``/VOLC zero slot 1, and
  radcswmx reads sulfate, sea salt, carbon, background and volcanic from that zero slot -- only dust bins 2..4
  radiate.  ``naer`` (=11) passed as ``naer_c`` only truncates the trailing species dimension (identical memory
  layout for the leading 11 slots), it does not shift anything.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config.paths import wrf_run_path
from gpuwrf.physics.ra_cam_common import CONST, lit, radinp

f32 = np.float32

NSPINT = 19
NRH = 1000
NDSTSZ = 4
PAERLEV = 29
NAER_ALL = 13
NAER = 11
IDXVIS = 8           # aer_optics idxVIS (1-based band index)
NCONFGMAX = 15
AREAMIN = 0.01       # areamin = 0.01_r8
CLDMIN = 1.0e-80     # cldmin = 1.0e-80_r8
TAUBACK = 0.0        # tauback = 0._r8


def _lits(values) -> np.ndarray:
    return np.asarray(values, f32).astype(np.float64)


# --------------------------------------------------------------------------------------------------------------------- #
# module_ra_cam DATA (single-precision literals widened to r8)                                                          #
# --------------------------------------------------------------------------------------------------------------------- #
ABARL = _lits([2.817e-02, 2.682e-02, 2.264e-02, 1.281e-02])
BBARL = _lits([1.305, 1.346, 1.454, 1.641])
CBARL = _lits([-5.62e-08, -6.94e-06, 4.64e-04, 0.201])
DBARL = _lits([1.63e-07, 2.35e-05, 1.24e-03, 7.56e-03])
EBARL = _lits([0.829, 0.794, 0.754, 0.826])
FBARL = _lits([2.482e-03, 4.226e-03, 6.560e-03, 4.353e-03])
ABARI = _lits([3.448e-03, 3.448e-03, 3.448e-03, 3.448e-03])
BBARI = _lits([2.431, 2.431, 2.431, 2.431])
CBARI = _lits([1.00e-05, 1.10e-04, 1.861e-02, .46658])
DBARI = _lits([0.0, 1.405e-05, 8.328e-04, 2.05e-05])
EBARI = _lits([0.7661, 0.7730, 0.794, 0.9595])
FBARI = _lits([5.851e-04, 5.665e-04, 7.267e-04, 1.076e-04])
DELTA = lit(0.0014257179260883)
O2MMR = lit(.23143)
FRCSOL = _lits([.001488, .001389, .001290, .001686, .002877, .003869, .026336, .360739, .065392, .526861,
                .526861, .526861, .526861, .526861, .526861, .526861, .006239, .001834, .001834])
WAVMIN = _lits([.200, .245, .265, .275, .285, .295, .305, .350, .640, .700, .701, .701, .701, .701, .702, .702,
                2.630, 4.160, 4.160])
WAVMAX = _lits([.245, .265, .275, .285, .295, .305, .350, .640, .700, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000,
                2.860, 4.550, 4.550])
RAYTAU = _lits([4.020, 2.180, 1.700, 1.450, 1.250, 1.085, 0.730, 0.155208, 0.0392, 0.02899756, 0.01356763,
                0.00537341, 0.00228515, 0.00105028, 0.00046631, 0.00025734, .0001, .0001, .0001])
ABH2O = _lits([0.0] * 9 + [0.00256608, 0.06310504, 0.42287445, 2.45397941, 11.20070807, 47.66091389, 240.19010243]
              + [0.0] * 3)
ABO3 = _lits([5.370e+04, 13.080e+04, 9.292e+04, 4.530e+04, 1.616e+04, 4.441e+03, 1.775e+02, 2.4058030e+01, 2.210e+01]
             + [0.0] * 10)
ABCO2 = _lits([0.0] * 16 + [.094, .196, 1.963])
ABO2 = _lits([0.0] * 8 + [1.11e-05, 6.69e-05] + [0.0] * 9)
PH2O = _lits([0.0] * 9 + [.505, .210, .120, .070, .048, .029, .018] + [0.0] * 3)
PCO2 = _lits([0.0] * 16 + [1.000, .640, .360])
PO2 = _lits([0.0] * 8 + [1.000, 1.000] + [0.0] * 9)

# raddedmx parameters (r8 parameters from REAL literals)
WRAY = lit(0.999999)
GRAY = lit(0.0)
FRAY = lit(0.1)


def _band_tables():
    """Per-band static quantities of radcswmx: indxsl (0-based), psf, visible flag, trayoslp."""

    indxsl = []
    cur = None
    for ns in range(NSPINT):
        if WAVMAX[ns] <= 0.7:
            cur = 1
        elif WAVMIN[ns] == 0.700:
            cur = 2
        elif WAVMIN[ns] == 0.701:
            cur = 3
        elif WAVMIN[ns] == 0.702 or WAVMIN[ns] > 2.38:
            cur = 4
        # else: indxsl keeps the previous interval's value (WRF quirk; see module docstring)
        indxsl.append(cur - 1)
    psf = np.ones(NSPINT)
    for ns in range(NSPINT):
        if PH2O[ns] != 0.0:
            psf[ns] = psf[ns] * PH2O[ns]
        if PCO2[ns] != 0.0:
            psf[ns] = psf[ns] * PCO2[ns]
        if PO2[ns] != 0.0:
            psf[ns] = psf[ns] * PO2[ns]
    wavmid = 0.5 * (WAVMIN + WAVMAX)
    visible = wavmid < 0.7
    trayoslp = RAYTAU / CONST.sslp
    return np.asarray(indxsl, np.int32), psf, visible, trayoslp


INDXSL, PSF, VISIBLE, TRAYOSLP = _band_tables()


# --------------------------------------------------------------------------------------------------------------------- #
# Aerosol species indices (Registry aerosolc scalar indices P_SUL..P_VOLC, 1-based)                                      #
# --------------------------------------------------------------------------------------------------------------------- #
class AerosolIndices(NamedTuple):
    sul: int
    sslt: int
    dust1: int
    ocpho: int
    bcpho: int
    ocphi: int
    bcphi: int
    bg: int
    volc: int


#: ``set_scalar_indices_from_config`` with ra_sw_physics = 3 (Registry aerosolc: dummy slot 1, then sul..volc).
WRF_AEROSOL_INDICES = AerosolIndices(2, 3, 4, 8, 9, 10, 11, 12, 13)
#: Registry defaults ``P_* = 1`` (scalar indices never configured; NOT a WRF run configuration).
UNSET_AEROSOL_INDICES = AerosolIndices(1, 1, 1, 1, 1, 1, 1, 1, 1)


# --------------------------------------------------------------------------------------------------------------------- #
# aer_optics_initialize: CAM_AEROPT_DATA                                                                                #
# --------------------------------------------------------------------------------------------------------------------- #
class CamAeroptTables(NamedTuple):
    ksul: np.ndarray     # (nrh, nspint)
    wsul: np.ndarray
    gsul: np.ndarray
    ksslt: np.ndarray
    wsslt: np.ndarray
    gsslt: np.ndarray
    kcphil: np.ndarray
    wcphil: np.ndarray
    gcphil: np.ndarray
    kcphob: np.ndarray   # (nspint,)
    wcphob: np.ndarray
    gcphob: np.ndarray
    kcb: np.ndarray
    wcb: np.ndarray
    gcb: np.ndarray
    kdst: np.ndarray     # (ndstsz, nspint)
    wdst: np.ndarray
    gdst: np.ndarray
    kbg: np.ndarray
    wbg: np.ndarray
    gbg: np.ndarray
    kvolc: np.ndarray
    wvolc: np.ndarray
    gvolc: np.ndarray


def _read_be_records(path: Path) -> list[np.ndarray]:
    """Sequential unformatted, big-endian 4-byte record markers (WRF -fconvert=big-endian), r8 payloads."""

    raw = path.read_bytes()
    out, pos = [], 0
    while pos < len(raw):
        n = int(np.frombuffer(raw, ">i4", 1, pos)[0])
        pos += 4
        out.append(np.frombuffer(raw, ">f8", n // 8, pos).astype(np.float64))
        pos += n
        if int(np.frombuffer(raw, ">i4", 1, pos)[0]) != n:
            raise RuntimeError(f"CAM_AEROPT_DATA record marker mismatch in {path}")
        pos += 4
    return out


def _find_k(x, y) -> int:
    """exp_interpol/lin_interpol bracket search (0-based k with x[k] < y <= x[k+1])."""

    n = len(x)
    if y <= x[0]:
        return 0
    if y >= x[n - 1]:
        return n - 2
    k = 0
    while y > x[k + 1] and k < n - 1:
        k += 1
    return k


def _exp_interpol(x, f, y) -> float:
    k = _find_k(x, y)
    a = (math.log(f[k + 1] / f[k])) / (x[k + 1] - x[k])
    return f[k] * math.exp(a * (y - x[k]))


def _lin_interpol(x, f, y) -> float:
    k = _find_k(x, y)
    a = (f[k + 1] - f[k]) / (x[k + 1] - x[k])
    return f[k] + a * (y - x[k])


@lru_cache(maxsize=2)
def load_cam_aeropt_tables(path: str | None = None) -> CamAeroptTables:
    """aer_optics_initialize (module_ra_cam_support.F:3712-3905) on host, scalar libm like gfortran."""

    p = Path(path) if path is not None else wrf_run_path("CAM_AEROPT_DATA")
    recs = _read_be_records(p)
    if len(recs) != 26:
        raise RuntimeError(f"CAM_AEROPT_DATA: expected 26 records, found {len(recs)} in {p}")
    m2 = lambda r: r.reshape((8, NSPINT), order="F")          # (irh, nspint)
    d2 = lambda r: r.reshape((NDSTSZ, NSPINT), order="F")     # (ndstsz, nspint)
    rh_opac = recs[1]
    ksul_o, wsul_o, gsul_o = m2(recs[2]), m2(recs[3]), m2(recs[4])
    kssam, wssam, gssam = m2(recs[5]), m2(recs[6]), m2(recs[7])
    ksscm, wsscm, gsscm = m2(recs[8]), m2(recs[9]), m2(recs[10])
    kcphil_o, wcphil_o, gcphil_o = m2(recs[11]), m2(recs[12]), m2(recs[13])
    kcb, wcb, gcb = recs[14], recs[15], recs[16]
    kdst, wdst, gdst = d2(recs[17]), d2(recs[18]), d2(recs[19])
    kbg, wbg, gbg = recs[20], recs[21], recs[22]
    kvolc, wvolc, gvolc = recs[23], recs[24], recs[25]

    wgt = float(f32(6.0) / f32(7.0))     # r8 parameter wgt_sscm = 6.0 / 7.0 (REAL expression)
    one_m = 1.0 - wgt
    ksslt_o = one_m * kssam + wgt * ksscm
    wsslt_o = (one_m * kssam * wssam + wgt * ksscm * wsscm) / ksslt_o
    gsslt_o = (one_m * kssam * wssam * gssam + wgt * ksscm * wsscm * gsscm) / (ksslt_o * wsslt_o)

    kcphob = kcphil_o[0, :].copy()
    wcphob = wcphil_o[0, :].copy()
    gcphob = gcphil_o[0, :].copy()

    x = [float(v) for v in rh_opac]
    out = {name: np.zeros((NRH, NSPINT)) for name in
           ("ksul", "wsul", "gsul", "ksslt", "wsslt", "gsslt", "kcphil", "wcphil", "gcphil")}
    srcs = {"ksul": (ksul_o, _exp_interpol), "wsul": (wsul_o, _lin_interpol), "gsul": (gsul_o, _lin_interpol),
            "ksslt": (ksslt_o, _exp_interpol), "wsslt": (wsslt_o, _lin_interpol), "gsslt": (gsslt_o, _lin_interpol),
            "kcphil": (kcphil_o, _exp_interpol), "wcphil": (wcphil_o, _lin_interpol),
            "gcphil": (gcphil_o, _lin_interpol)}
    for krh in range(1, NRH + 1):
        rh = 1.0 / NRH * (krh - 1)
        for kbnd in range(NSPINT):
            for name, (tab, fn) in srcs.items():
                col = tab[:, kbnd]
                f = [float(v) / float(col[0]) for v in col]
                out[name][krh - 1, kbnd] = fn(x, f, rh) * float(col[0])
    return CamAeroptTables(
        out["ksul"], out["wsul"], out["gsul"], out["ksslt"], out["wsslt"], out["gsslt"],
        out["kcphil"], out["wcphil"], out["gcphil"], kcphob, wcphob, gcphob, kcb, wcb, gcb,
        kdst, wdst, gdst, kbg, wbg, gbg, kvolc, wvolc, gvolc)


# --------------------------------------------------------------------------------------------------------------------- #
# esinti/gestbl/gffgch (itype = -20) -> estbl;  estblf, aqsat, radctl relative humidity                                  #
# --------------------------------------------------------------------------------------------------------------------- #
ES_TMIN = lit(173.16)
ES_TMAX = lit(375.16)
_TRICE = lit(20.00)
PLENEST = 250


def _gffgch_host(t: float, tr: float) -> float:
    """gffgch(t, es, itype = -tr) with module tmelt; scalar libm (gfortran scalar calls)."""

    tmelt = CONST.tmelt
    if t < (tmelt - tr):
        eswtr = None
    else:
        ps = lit(1013.246)
        ts = lit(373.16)
        e1 = lit(11.344) * (1.0 - t / ts)
        e2 = -lit(3.49149) * (ts / t - 1.0)
        f1 = -lit(7.90298) * (ts / t - 1.0)
        f2 = lit(5.02808) * math.log10(ts / t)
        f3 = -lit(1.3816) * (math.pow(10.0, e1) - 1.0) / 10000000.0
        f4 = lit(8.1328) * (math.pow(10.0, e2) - 1.0) / 1000.0
        f5 = math.log10(ps)
        f = f1 + f2 + f3 + f4 + f5
        es = (math.pow(10.0, f)) * 100.0
        eswtr = es
        if t >= tmelt:
            return es
    t0 = tmelt
    term1 = lit(2.01889049) / (t0 / t)
    term2 = lit(3.56654) * math.log(t0 / t)
    term3 = lit(20.947031) * (t0 / t)
    es = lit(575.185606e10) * math.exp(-(term1 + term2 + term3))
    if t < (tmelt - tr):
        return es
    weight = min((tmelt - t) / tr, 1.0)
    return weight * es + (1.0 - weight) * eswtr


@lru_cache(maxsize=1)
def estbl_table() -> np.ndarray:
    """gestbl(tmn = 173.16, tmx = 375.16, trice = 20, ip = .true.): estbl(1:plenest)."""

    lentbl = int(ES_TMAX - ES_TMIN + lit(2.000001))
    itype = int(-_TRICE)
    tr = abs(float(f32(itype)))
    est = np.full(PLENEST, -99999.0)
    t = ES_TMIN - 1.0
    for n in range(lentbl):
        t = t + 1.0
        est[n] = _gffgch_host(t, tr)
    return est


def estblf(td):
    """estblf(td): linear lookup in estbl (r8)."""

    tab = jnp.asarray(estbl_table())
    e = jnp.maximum(jnp.minimum(td, ES_TMAX), ES_TMIN)
    ai = jnp.trunc(e - ES_TMIN)
    i0 = ai.astype(jnp.int32)                         # Fortran i = int(e - tmin) + 1 -> 0-based i0
    return (ES_TMIN + ai - e + 1.0) * jnp.take(tab, i0) - (ES_TMIN + ai - e) * jnp.take(tab, i0 + 1)


def cam_aqsat(t, p):
    """aqsat(t, p, es, qs, ...) over all layers: saturation vapour pressure and specific humidity."""

    epsqs = CONST.epsqs
    omeps = 1.0 - epsqs
    es = estblf(t)
    qs = epsqs * es / (p - omeps * es)
    qs = jnp.minimum(1.0, qs)
    neg = qs < 0.0
    return jnp.where(neg, p, es), jnp.where(neg, 1.0, qs)


def cam_rh(t, pmid, q1):
    """radctl SW relative humidity: ``q / qsat * ((1 - eps) qsat + eps) / ((1 - eps) q + eps)``."""

    _, qsat = cam_aqsat(t, pmid)
    eps = CONST.epsilo
    return q1 / qsat * ((1.0 - eps) * qsat + eps) / ((1.0 - eps) * q1 + eps)


# --------------------------------------------------------------------------------------------------------------------- #
# aerosol_init climatology + get_aerosol (vert_interpolate, getfactors, background, scale_aerosols)                      #
# --------------------------------------------------------------------------------------------------------------------- #
def aerosol_climatology(m_hybi, idx: AerosolIndices = WRF_AEROSOL_INDICES):
    """aerosol_init's uniform upward-cumulative climatology aerosolc(k, slot) (REAL), slots 1..13 -> 0..12.

    Writes are replayed in WRF order so that colliding indices overwrite exactly as in WRF.
    """

    hyb = jnp.asarray(m_hybi, jnp.float32)
    one = jnp.float32(1.0)
    aer = jnp.zeros((hyb.shape[0], NAER_ALL), jnp.float32)
    writes = [(idx.sul, 1.e-7), (idx.sslt, 1.e-22), (idx.dust1, 1.e-7), (idx.dust1 + 1, 1.e-7),
              (idx.dust1 + 2, 1.e-7), (idx.dust1 + 3, 1.e-7), (idx.ocpho, 1.e-7), (idx.bcpho, 1.e-9),
              (idx.ocphi, 1.e-7), (idx.bcphi, 1.e-8)]
    for slot, coef in writes:
        aer = aer.at[:, slot - 1].set(jnp.float32(coef) * (one - hyb))
    return aer


_MID = _lits([16.5, 46.0, 75.5, 106.0, 136.5, 167.0, 197.5, 228.5, 259.0, 289.5, 320.0, 350.5])


def aerosol_time_factors(julian):
    """get_aerosol calendar day + getfactors(cycflag = .true.): (fact1, fact2) r8 per column."""

    jul = jnp.asarray(julian, jnp.float32).astype(jnp.float64) + 1.0
    ijul = jul.astype(jnp.int32)
    jul = jul - ijul.astype(jnp.float32).astype(jnp.float64)
    ijul = jnp.mod(ijul, 365)
    ijul = jnp.where(ijul == 0, 365, ijul)
    cday = jul + ijul.astype(jnp.float64)
    mid = jnp.asarray(_MID)
    later = mid[None, 1:] > cday[:, None]                        # Mid(i) > caldayloc, i = 2..12
    inner = jnp.where(jnp.any(later, axis=1), jnp.argmax(later, axis=1) + 1, 0)   # 0-based mo_nxt
    wrap = (cday < mid[0]) | (cday >= mid[11])
    nxt = jnp.where(wrap, 0, inner)
    prv = jnp.where(wrap, 11, inner - 1)
    cdayp = jnp.take(mid, nxt)
    cdaym = jnp.take(mid, prv)
    dpy = 365.0
    dec_jan = nxt == 0
    deltat = jnp.where(dec_jan, cdayp + dpy - cdaym, cdayp - cdaym)
    dec = cday > cdayp
    fact1 = jnp.where(dec_jan, jnp.where(dec, (cdayp + dpy - cday) / deltat, (cdayp - cday) / deltat),
                      (cdayp - cday) / deltat)
    fact2 = jnp.where(dec_jan, jnp.where(dec, (cday - cdaym) / deltat, (cday + dpy - cdaym) / deltat),
                      (cday - cdaym) / deltat)
    return fact1, fact2


def vert_interpolate(pint, aerc, m_hybi, m_ps=1.0e5):
    """vert_interpolate for species 1..naer: cumulative column mass onto CAM interfaces -> mmr (ncol, pver, naer).

    ``aerc`` (paerlev, naer_all) REAL climatology (identical for every column, as aerosol_init sets it), ``pint`` Pa.
    """

    pint = jnp.asarray(pint, jnp.float64)
    ncol, pverp = pint.shape
    pver = pverp - 1
    hyb = jnp.asarray(m_hybi, jnp.float32).astype(jnp.float64)
    ps = jnp.asarray(m_ps, jnp.float32).astype(jnp.float64)
    ps = jnp.broadcast_to(ps, (ncol,))
    ac = jnp.asarray(aerc, jnp.float32).astype(jnp.float64)[:, :NAER]             # (paerlev, naer)
    bnd = hyb[None, :] * ps[:, None]                                               # (ncol, paerlev) M_hybi*Match_ps
    p = pint[:, 1:pver]                                                            # interfaces k = 2..pver
    kk = jnp.sum(bnd[:, None, :] < p[:, :, None], axis=-1)                         # 1-based kk: bnd(kk) < p
    lo = jnp.clip(kk - 1, 0, PAERLEV - 2)
    b_lo = jnp.take_along_axis(bnd, lo, axis=1)
    b_hi = jnp.take_along_axis(bnd, lo + 1, axis=1)
    dpu = p - b_lo
    dpl = b_hi - p
    a_lo = ac[lo]                                                                  # (ncol, pver-1, naer)
    a_hi = ac[lo + 1]
    interp = (a_lo * dpl[..., None] + a_hi * dpu[..., None]) / (dpl + dpu)[..., None]
    above = p < bnd[:, :1]
    below = p > bnd[:, -1:]
    mid = jnp.where(above[..., None], ac[0][None, None, :], jnp.where(below[..., None], 0.0, interp))
    top = jnp.broadcast_to(ac[0], (ncol, 1, NAER))
    aer = jnp.concatenate([top, mid, jnp.zeros((ncol, 1, NAER))], axis=1)          # (ncol, pverp, naer)
    aer_l = aer[:, :pver]
    aer_l = jnp.where(aer_l < 1.e-40, 0.0, aer_l)
    aer = jnp.concatenate([aer_l, aer[:, pver:]], axis=1)
    diff = aer[:, :pver] - aer[:, 1:]
    diff = jnp.where(jnp.abs(diff) < lit(1e-15) * aer[:, :1], 0.0, diff)
    m_to_mmr = CONST.gravmks / (pint[:, 1:] - pint[:, :pver])
    return diff * m_to_mmr[..., None]


def cam_aerosol(pint, julian, mxaerl, m_hybi, *, idx: AerosolIndices = WRF_AEROSOL_INDICES, m_ps=1.0e5,
                tables: CamAeroptTables | None = None):
    """get_int_scales + get_aerosol: AEROSOLt (ncol, pver, naer_all) r8 on CAM layers (pint in Pa)."""

    tables = tables if tables is not None else load_cam_aeropt_tables()
    pint = jnp.asarray(pint, jnp.float64)
    ncol, pverp = pint.shape
    pver = pverp - 1
    aerc = aerosol_climatology(m_hybi, idx)
    mmr = vert_interpolate(pint, aerc, m_hybi, m_ps)            # aerosolcp == aerosolcn, m_psp == m_psn
    fact1, fact2 = aerosol_time_factors(julian)
    f1 = jnp.broadcast_to(fact1, (ncol,))[:, None, None]
    f2 = jnp.broadcast_to(fact2, (ncol,))[:, None, None]
    aert = jnp.zeros((ncol, pver, NAER_ALL), jnp.float64)
    aert = aert.at[:, :, :NAER].set(mmr * f1 + mmr * f2)
    # background (tauback = 0): mmr = mass2mmr * mass for the bottom mxaerl layers, else 0
    mxaerl = jnp.asarray(mxaerl, jnp.int32)
    kbot = pverp - mxaerl                                      # Fortran pverrp - mxaerl (1-based interface)
    p_top = jnp.take(pint, kbot - 1, axis=1)
    mass2mmr = CONST.gravmks / (pint[:, pver] - p_top)
    mass = TAUBACK / (1.e3 * jnp.asarray(tables.kbg)[IDXVIS - 1])
    k1 = jnp.arange(1, pver + 1)
    bg = jnp.where(k1[None, :] >= kbot, (mass2mmr * mass)[:, None], 0.0)
    aert = aert.at[:, :, idx.bg - 1].set(bg)
    aert = aert.at[:, :, idx.volc - 1].set(0.0)
    scales = get_int_scales(idx)
    return aert * jnp.asarray(scales)[None, None, :]


def get_int_scales(idx: AerosolIndices = WRF_AEROSOL_INDICES) -> np.ndarray:
    scales = np.full(NAER_ALL, 1.0)
    sulscl = carscl = ssltscl = dustscl = volcscl = 1.0
    scales[idx.bg - 1] = 1.0
    scales[idx.sul - 1] = sulscl
    scales[idx.sslt - 1] = ssltscl
    for i in range(idx.ocpho, idx.ocpho + 4):
        scales[i - 1] = carscl
    for i in range(idx.dust1, idx.dust1 + 4):
        scales[i - 1] = dustscl
    scales[idx.volc - 1] = volcscl
    return scales


# --------------------------------------------------------------------------------------------------------------------- #
# radcswmx: absorber paths (band independent), per-interval optics + raddedmx                                          #
# --------------------------------------------------------------------------------------------------------------------- #
def _dedd(tautot, wt, gnum, fnum, coszrs):
    """raddedmx delta-Eddington layer solution (statement functions inlined, WRF evaluation order)."""

    wtot = wt / tautot
    gtot = gnum / wt
    ftot = fnum / wt
    ts = (1.0 - wtot * ftot) * tautot
    ws = (1.0 - ftot) * wtot / (1.0 - wtot * ftot)
    gs = (gtot - ftot) / (1.0 - ftot)
    lm = jnp.sqrt(3.0 * (1.0 - ws) * (1.0 - ws * gs))
    uu = coszrs
    den = 1.0 - lm * lm * uu * uu
    alp = .75 * ws * uu * ((1.0 + gs * (1.0 - ws)) / den)
    gam = .50 * ws * ((3.0 * gs * (1.0 - ws) * uu * uu + 1.0) / den)
    ue = 1.5 * (1.0 - ws * gs) / lm
    arg = jnp.minimum(lm * ts, 25.0)
    extins = jnp.exp(-arg)
    ne = ((ue + 1.0) * (ue + 1.0) / extins) - ((ue - 1.0) * (ue - 1.0) * extins)
    rdif = (ue + 1.0) * (ue - 1.0) * (1.0 / extins - extins) / ne
    tdif = 4.0 * ue / ne
    arg = jnp.minimum(ts / coszrs, 25.0)
    explay = jnp.exp(-arg)
    apg = alp + gam
    amg = alp - gam
    rdir = amg * (tdif * explay - 1.0) + apg * rdif
    tdir = apg * tdif + (amg * rdif - (apg - 1.0)) * explay
    return (jnp.maximum(rdir, 0.0), jnp.maximum(rdif, 0.0), jnp.maximum(tdir, 0.0), jnp.maximum(tdif, 0.0),
            explay)


class _Paths(NamedTuple):
    """Band-independent radcswmx column quantities (layer arrays include the extra layer 0 where noted)."""

    dp: jnp.ndarray       # pflx(k+1) - pflx(k), k = 0..pver  (ncol, pver+1)
    uh2o: jnp.ndarray     # (ncol, pver+1)
    uo3: jnp.ndarray
    uco2: jnp.ndarray
    uo2: jnp.ndarray
    usul: jnp.ndarray     # (ncol, pver)
    ubg: jnp.ndarray
    usslt: jnp.ndarray
    ucphil: jnp.ndarray
    ucphob: jnp.ndarray
    ucb: jnp.ndarray
    uvolc: jnp.ndarray
    udst: jnp.ndarray     # (ndstsz, ncol, pver)
    krh: jnp.ndarray      # 1-based rh bin (ncol, pver)
    wrh: jnp.ndarray


def _column_paths(pnm, h2ommr, rh, o3mmr, aermmr, coszrs, co2mmr, idx) -> _Paths:
    c = CONST
    gravit = c.gravit
    ncol = pnm.shape[0]
    tmp1 = 0.5 / (gravit * c.sslp)
    tmp2 = DELTA / gravit
    sqrco2 = jnp.sqrt(co2mmr)[:, None]
    pflx = jnp.concatenate([jnp.zeros((ncol, 1)), pnm], axis=1)                # pflx(0:pverp)
    zenfac = jnp.sqrt(coszrs)[:, None]
    # extra layer 0
    ptop = pflx[:, 1:2]
    ptho2 = O2MMR * ptop / gravit
    ptho3 = o3mmr[:, :1] * ptop / gravit
    pthco2 = sqrco2 * (ptop / gravit)
    h2ostr = jnp.sqrt(1.0 / h2ommr[:, :1])
    pthh2o = ptop * ptop * tmp1 + (ptop * c.rga) * (h2ostr * zenfac * DELTA)
    uh2o0 = h2ommr[:, :1] * pthh2o
    uco20 = zenfac * pthco2
    uo20 = zenfac * ptho2
    uo30 = ptho3
    # model layers 1..pver
    pdel = pflx[:, 2:] - pflx[:, 1:-1]
    path = pdel / gravit
    ptho2 = O2MMR * path
    ptho3 = o3mmr * path
    pthco2 = sqrco2 * path
    h2ostr = jnp.sqrt(1.0 / h2ommr)
    pthh2o = (pflx[:, 2:] * pflx[:, 2:] - pflx[:, 1:-1] * pflx[:, 1:-1]) * tmp1 + pdel * h2ostr * zenfac * tmp2
    sl = lambda s: aermmr[:, :, s - 1]
    usslt = sl(idx.sslt) * path
    rhtrunc = jnp.minimum(rh, 1.0)
    krh = jnp.minimum(jnp.floor(rhtrunc * NRH).astype(jnp.int32) + 1, NRH - 1)
    return _Paths(
        dp=pflx[:, 1:] - pflx[:, :-1],
        uh2o=jnp.concatenate([uh2o0, h2ommr * pthh2o], axis=1),
        uo3=jnp.concatenate([uo30, ptho3], axis=1),
        uco2=jnp.concatenate([uco20, zenfac * pthco2], axis=1),
        uo2=jnp.concatenate([uo20, zenfac * ptho2], axis=1),
        usul=sl(idx.sul) * path, ubg=sl(idx.bg) * path, usslt=jnp.where(usslt < 0.0, 0.0, usslt),
        ucphil=sl(idx.ocphi) * path, ucphob=sl(idx.ocpho) * path, ucb=(sl(idx.bcpho) + sl(idx.bcphi)) * path,
        uvolc=sl(idx.volc), udst=jnp.stack([sl(idx.dust1 + ksz) * path for ksz in range(NDSTSZ)]),
        krh=krh, wrh=rhtrunc * NRH - krh.astype(jnp.float64))


def _band_constants(tables: CamAeroptTables) -> dict:
    """Per-interval constants (leading axis nspint) scanned by radcswmx's spectral loop."""

    isl = INDXSL
    a = lambda x: jnp.asarray(x)
    t = lambda x: jnp.asarray(x).T
    return dict(
        abarl=a(ABARL[isl]), bbarl=a(BBARL[isl]), cbarl=a(CBARL[isl]), dbarl=a(DBARL[isl]), ebarl=a(EBARL[isl]),
        fbarl=a(FBARL[isl]), abari=a(ABARI[isl]), bbari=a(BBARI[isl]), cbari=a(CBARI[isl]), dbari=a(DBARI[isl]),
        ebari=a(EBARI[isl]), fbari=a(FBARI[isl]),
        frcsol=a(FRCSOL), psf=a(PSF), vis=a(VISIBLE), trayoslp=a(TRAYOSLP),
        abh2o=a(ABH2O), abo3=a(ABO3), abco2=a(ABCO2), abo2=a(ABO2),
        ksul=t(tables.ksul), wsul=t(tables.wsul), gsul=t(tables.gsul), ksslt=t(tables.ksslt),
        wsslt=t(tables.wsslt), gsslt=t(tables.gsslt), kcphil=t(tables.kcphil), wcphil=t(tables.wcphil),
        gcphil=t(tables.gcphil), kcphob=a(tables.kcphob), wcphob=a(tables.wcphob), gcphob=a(tables.gcphob),
        kcb=a(tables.kcb), wcb=a(tables.wcb), gcb=a(tables.gcb), kvolc=a(tables.kvolc), wvolc=a(tables.wvolc),
        gvolc=a(tables.gvolc), kbg=a(tables.kbg), wbg=a(tables.wbg), gbg=a(tables.gbg),
        kdst=t(tables.kdst), wdst=t(tables.wdst), gdst=t(tables.gdst))


def _band_optics(b, p: _Paths, cld, cicewp, cliqwp, rel, rei, coszrs):
    """One spectral interval: cloud + aerosol optics and raddedmx (all-sky, clear-sky), layers 0..pver."""

    ncol = cld.shape[0]
    # cloud particle properties (Slingo / Ebert-Curry, indxsl of this interval)
    tmp1l = b["abarl"] + b["bbarl"] / rel
    tmp2l = 1.0 - b["cbarl"] - b["dbarl"] * rel
    tmp3l = b["fbarl"] * rel
    tmp1i = b["abari"] + b["bbari"] / rei
    tmp2i = 1.0 - b["cbari"] - b["dbari"] * rei
    tmp3i = b["fbari"] * rei
    cloudy = cld >= CLDMIN
    tauxcl = jnp.where(cloudy, cliqwp * tmp1l, 0.0)
    tauxci = jnp.where(cloudy, cicewp * tmp1i, 0.0)
    wcl = jnp.minimum(tmp2l, .999999)
    gcl = b["ebarl"] + tmp3l
    fcl = gcl * gcl
    wci = jnp.minimum(tmp2i, .999999)
    gci = b["ebari"] + tmp3i
    fci = gci * gci

    # aerosol: rh interpolation of the hygroscopic species, external mixing
    krh, wrh = p.krh, p.wrh
    ri = lambda row: row[krh] * (wrh + 1.0) - row[krh - 1] * wrh            # tab(krh+1)*(wrh+1) - tab(krh)*wrh
    ksuli, ksslti, kcphili = ri(b["ksul"]), ri(b["ksslt"]), ri(b["kcphil"])
    wsuli, wsslti, wcphili = ri(b["wsul"]), ri(b["wsslt"]), ri(b["wcphil"])
    gsuli, gsslti, gcphili = ri(b["gsul"]), ri(b["gsslt"]), ri(b["gcphil"])

    tau_sul = 1.e4 * ksuli * p.usul
    tau_sslt = 1.e4 * ksslti * p.usslt
    tau_cphil = 1.e4 * kcphili * p.ucphil
    tau_cphob = 1.e4 * b["kcphob"] * p.ucphob
    tau_cb = 1.e4 * b["kcb"] * p.ucb
    tau_volc = 1.e3 * b["kvolc"] * p.uvolc
    tau_dst = [1.e4 * b["kdst"][ksz] * p.udst[ksz] for ksz in range(NDSTSZ)]
    tau_bg = 1.e4 * b["kbg"] * p.ubg

    tau_w_sul = tau_sul * wsuli
    tau_w_sslt = tau_sslt * wsslti
    tau_w_cphil = tau_cphil * wcphili
    tau_w_cphob = tau_cphob * b["wcphob"]
    tau_w_cb = tau_cb * b["wcb"]
    tau_w_volc = tau_volc * b["wvolc"]
    tau_w_dst = [tau_dst[ksz] * b["wdst"][ksz] for ksz in range(NDSTSZ)]
    tau_w_bg = tau_bg * b["wbg"]

    tau_w_g_sul = tau_w_sul * gsuli
    tau_w_g_sslt = tau_w_sslt * gsslti
    tau_w_g_cphil = tau_w_cphil * gcphili
    tau_w_g_cphob = tau_w_cphob * b["gcphob"]
    tau_w_g_cb = tau_w_cb * b["gcb"]
    tau_w_g_volc = tau_w_volc * b["gvolc"]
    tau_w_g_dst = [tau_w_dst[ksz] * b["gdst"][ksz] for ksz in range(NDSTSZ)]
    tau_w_g_bg = tau_w_bg * b["gbg"]

    tau_w_f_sul = tau_w_sul * (gsuli * gsuli)
    tau_w_f_bg = tau_w_bg * (b["gbg"] * b["gbg"])
    tau_w_f_sslt = tau_w_sslt * (gsslti * gsslti)
    tau_w_f_cphil = tau_w_cphil * (gcphili * gcphili)
    tau_w_f_cphob = tau_w_cphob * (b["gcphob"] * b["gcphob"])
    tau_w_f_cb = tau_w_cb * (b["gcb"] * b["gcb"])
    tau_w_f_volc = tau_w_volc * (b["gvolc"] * b["gvolc"])
    tau_w_f_dst = [tau_w_dst[ksz] * (b["gdst"][ksz] * b["gdst"][ksz]) for ksz in range(NDSTSZ)]

    def ssum(terms):                       # gfortran SUM: sequential, left to right
        s = terms[0]
        for t in terms[1:]:
            s = s + t
        return s

    tau_dst_tot = ssum(tau_dst)
    tau_w_dst_tot = ssum(tau_w_dst)
    tau_w_g_dst_tot = ssum(tau_w_g_dst)
    tau_w_f_dst_tot = ssum(tau_w_f_dst)

    tau_tot = tau_sul + tau_sslt + tau_cphil + tau_cphob + tau_cb + tau_dst_tot
    tau_tot = tau_tot + tau_bg + tau_volc
    tau_w_tot = tau_w_sul + tau_w_sslt + tau_w_cphil + tau_w_cphob + tau_w_cb + tau_w_dst_tot
    tau_w_tot = tau_w_tot + tau_w_bg + tau_w_volc
    tau_w_g_tot = tau_w_g_sul + tau_w_g_sslt + tau_w_g_cphil + tau_w_g_cphob + tau_w_g_cb + tau_w_g_dst_tot
    tau_w_g_tot = tau_w_g_tot + tau_w_g_bg + tau_w_g_volc
    tau_w_f_tot = tau_w_f_sul + tau_w_f_sslt + tau_w_f_cphil + tau_w_f_cphob + tau_w_f_cb + tau_w_f_dst_tot
    tau_w_f_tot = tau_w_f_tot + tau_w_f_bg + tau_w_f_volc

    pos_t = tau_tot > 0.0
    w_tot = jnp.where(pos_t, tau_w_tot / jnp.where(pos_t, tau_tot, 1.0), 0.0)
    pos_w = tau_w_tot > 0.0
    safe_w = jnp.where(pos_w, tau_w_tot, 1.0)
    g_tot = jnp.where(pos_w, tau_w_g_tot / safe_w, 0.0)
    f_tot = jnp.where(pos_w, tau_w_f_tot / safe_w, 0.0)

    # extra layer 0 (no cloud or aerosol above the model top; the other properties are arbitrary)
    top = lambda val, arr: jnp.concatenate([jnp.full((ncol, 1), val, jnp.float64), arr], axis=1)
    tauxcl0, tauxci0 = top(0.0, tauxcl), top(0.0, tauxci)
    wcl0, gcl0, fcl0 = top(0.999999, wcl), top(0.85, gcl), top(0.725, fcl)
    wci0, gci0, fci0 = top(0.999999, wci), top(0.85, gci), top(0.725, fci)
    tauxar0, wa0 = top(0.0, tau_tot), top(0.925, jnp.minimum(w_tot, 0.999999))
    ga0, fa0 = top(0.850, g_tot), top(0.7225, f_tot)

    # raddedmx
    cz = coszrs[:, None]
    tauray = b["trayoslp"] * p.dp
    taugab = b["abh2o"] * p.uh2o + b["abo3"] * p.uo3 + b["abco2"] * p.uco2 + b["abo2"] * p.uo2
    wtau = WRAY * tauray
    tautot = tauxcl0 + tauxci0 + tauray + taugab + tauxar0
    taucsc = tauxcl0 * wcl0 + tauxci0 * wci0 + tauxar0 * wa0
    wt = wtau + taucsc
    gnum = wtau * GRAY + gcl0 * wcl0 * tauxcl0 + gci0 * wci0 * tauxci0 + ga0 * wa0 * tauxar0
    fnum = wtau * FRAY + fcl0 * wcl0 * tauxcl0 + fci0 * wci0 * tauxci0 + fa0 * wa0 * tauxar0
    allsky = _dedd(tautot, wt, gnum, fnum, cz)
    tautot_c = tauray + taugab + tauxar0
    wt_c = wtau + tauxar0 * wa0
    gnum_c = wtau * GRAY + ga0 * wa0 * tauxar0
    fnum_c = wtau * FRAY + fa0 * wa0 * tauxar0
    clear_c = _dedd(tautot_c, wt_c, gnum_c, fnum_c, cz)
    nocloud = (tauxcl0 == 0.0) & (tauxci0 == 0.0)
    clear = tuple(jnp.where(nocloud, x, y) for x, y in zip(allsky, clear_c))
    return allsky, clear, tauxcl, tauxci


# --------------------------------------------------------------------------------------------------------------------- #
# radcswmx: max-random overlap index calculations                                                                       #
# --------------------------------------------------------------------------------------------------------------------- #
def _region_streams(cld, reg, nslots):
    """Streams of every max-overlap region (radcswmx 'Construct wstr, cstr, nstr').

    cld (ncol, pver) r8, reg (ncol, pver) 0-based region of each layer.  Returns
    W (ncol, R, S) stream weights wstr, nstr (ncol, R), rho (ncol, pver) = first stream (0-based) in which a cloudy
    layer is cloudy, cloudy (ncol, pver), nrgn (ncol,).  Streams follow the ascending sort of 1 - cld; equal values
    share a stream boundary exactly as the Fortran 'asort(l) /= cld0' test does.
    """

    ncol, pver = cld.shape
    R = S = nslots
    cloudy = cld >= CLDMIN
    a = 1.0 - cld
    same = (reg[:, :, None] == reg[:, None, :]) & cloudy[:, None, :] & cloudy[:, :, None]   # [k, k']
    eq = a[:, :, None] == a[:, None, :]
    less = a[:, None, :] < a[:, :, None]                                                    # a[k'] < a[k]
    earlier = jnp.asarray(np.tril(np.ones((pver, pver), bool), -1))[None]                   # k' < k
    first = cloudy & ~jnp.any(same & eq & earlier, axis=2)
    any_less = jnp.any(same & less, axis=2)
    creating = first & ((a != 0.0) | any_less)
    rho = jnp.sum(same & creating[:, None, :] & (a[:, None, :] <= a[:, :, None]), axis=2)
    prev = jnp.max(jnp.where(same & less, a[:, None, :], -jnp.inf), axis=2)
    prev = jnp.where(any_less, prev, 0.0)
    wcreate = a - prev
    rows = jnp.arange(ncol)[:, None]
    sidx = jnp.where(creating, rho - 1, S + 1)
    W = jnp.zeros((ncol, R, S), jnp.float64).at[rows, reg, sidx].set(wcreate, mode="drop")
    regoh = reg[:, :, None] == jnp.arange(R)[None, None, :]                                 # (ncol, pver, R)
    m = jnp.sum(regoh & creating[:, :, None], axis=1)                                      # distinct creating values
    nstr = m + 1
    amax = jnp.max(jnp.where(regoh & creating[:, :, None], a[:, :, None], -jnp.inf), axis=1)
    cld0 = jnp.where(m > 0, amax, 0.0)
    W = W.at[jnp.arange(ncol)[:, None], jnp.arange(R)[None, :], m].set(1.0 - cld0, mode="drop")
    nrgn = jnp.max(reg, axis=1) + 1
    return W, nstr, rho, cloudy, nrgn


def _findvalue1(wgtv, ptrc, active):
    """findvalue(1, nconfgmax, wgtv, ptrc) (quickselect for the smallest weight, permuting ptrc in place).

    Inactive calls return ptrc unchanged without iterating.
    """

    n = NCONFGMAX
    pos = jnp.arange(n, dtype=jnp.int32)

    def swap(ix, a, b):
        va, vb = ix[a], ix[b]
        return ix.at[a].set(vb).at[b].set(va)

    def cswap(ix, pred, a, b):
        return jnp.where(pred, swap(ix, a, b), ix)

    def outer_cond(st):
        return ~st[3]

    def outer_body(st):
        il, ir, ix, _ = st
        ain = lambda q: wgtv[ix[q]]
        small = (ir - il) <= 1
        # small branch
        sw = small & ((ir - il) == 1) & (ain(ir) < ain(il))
        ix_small = cswap(ix, sw, il, ir)
        # partition branch
        im = (il + ir) // 2
        ip1 = jnp.minimum(il + 1, n - 1).astype(jnp.int32)
        ixb = swap(ix, im, ip1)
        ixb = cswap(ixb, wgtv[ixb[ip1]] > wgtv[ixb[ir]], ip1, ir)
        ixb = cswap(ixb, wgtv[ixb[il]] > wgtv[ixb[ir]], il, ir)
        ixb = cswap(ixb, wgtv[ixb[ip1]] > wgtv[ixb[il]], ip1, il)
        ia = ixb[il]
        pv = wgtv[ia]

        def pcond(ps):
            return ~ps[3]

        def pbody(ps):
            i, j, ixp, _ = ps
            vals = wgtv[ixp]
            inew = jnp.argmax((pos > i) & (vals >= pv)).astype(jnp.int32)
            jnew = (n - 1 - jnp.argmax(((pos < j) & (vals <= pv))[::-1])).astype(jnp.int32)
            stop = jnew < inew
            ixp = cswap(ixp, ~stop, inew, jnew)
            return inew, jnew, ixp, stop

        i, j, ixb, _ = jax.lax.while_loop(pcond, pbody, (ip1, ir, ixb, small))
        ixb = ixb.at[il].set(ixb[j]).at[j].set(ia)
        ir_n = jnp.where(j >= 0, j - 1, ir)
        il_n = jnp.where(j <= 0, i, il)
        ix_new = jnp.where(small, ix_small, ixb)
        return (jnp.where(small, il, il_n), jnp.where(small, ir, ir_n), ix_new, small)

    _, _, ix, _ = jax.lax.while_loop(outer_cond, outer_body,
                                     (jnp.int32(0), jnp.int32(n - 1), ptrc.astype(jnp.int32), ~active))
    return ix[0], ix


def _suffix_bound(W, nstr, nrgn):
    """prod over regions r' > r of the largest stream weight (upper bound of any completion of a prefix)."""

    R = W.shape[0]
    ridx = jnp.arange(R)
    mx = jnp.where(ridx < nrgn, jnp.max(jnp.where(ridx[None, :] < nstr[:, None], W, -jnp.inf), axis=1), 1.0)
    return jnp.concatenate([jnp.cumprod(mx[::-1])[::-1][1:], jnp.ones((1,))])


def _select_column(W1, nstr1, nrgn1, W2, nstr2, nrgn2, active):
    """Configuration loop of radcswmx for one column ('Construction of totwgt, wgtv, ccon, nconfig').

    Enumerates the configurations (all permutations of streams over the regions, last region fastest) with exact
    sequential weight products, skipping subtrees whose largest attainable weight is below areamin (they cannot
    change the state), and keeps the nconfgmax-slot table with WRF's findvalue replacement.  If the first pass ends
    with totwgt = 0, the second pass runs on the maximum-overlap streams (W2: nmxrgn = 1, pmxrgn(1) = 1e30) without
    resetting wgtv/ptrc/ccon slots, as in WRF.  Returns slot stream choices (nconfgmax, R), wgtv, nconfig, totwgt,
    and whether the second pass was used.
    """

    R = W1.shape[0]
    ridx = jnp.arange(R, dtype=jnp.int32)
    sfx1 = _suffix_bound(W1, nstr1, nrgn1)
    sfx2 = _suffix_bound(W2, nstr2, nrgn2)
    nstr1, nstr2 = nstr1.astype(jnp.int32), nstr2.astype(jnp.int32)
    nrgn1, nrgn2 = nrgn1.astype(jnp.int32), nrgn2.astype(jnp.int32)
    thr = AREAMIN * (1.0 - 1.0e-9)

    state = dict(
        istr=jnp.zeros((R,), jnp.int32), done=~active, phase=jnp.bool_(False), nconfig=jnp.int32(0),
        totwgt=jnp.float64(0.0), wgtv=jnp.zeros((NCONFGMAX,), jnp.float64),
        ptrc=jnp.arange(NCONFGMAX, dtype=jnp.int32), slots=jnp.zeros((NCONFGMAX, R), jnp.int32),
        new_term=jnp.bool_(True), jprev=jnp.int32(0))

    def body(st):
        ph = st["phase"]
        W = jnp.where(ph, W2, W1)
        nstr = jnp.where(ph, nstr2, nstr1)
        nrgn = jnp.where(ph, nrgn2, nrgn1)
        suffix = jnp.where(ph, sfx2, sfx1)
        istr = st["istr"]
        inreg = ridx < nrgn
        fac = jnp.where(inreg, W[ridx, istr], 1.0)
        _, prefix = jax.lax.scan(lambda x, f: (x * f, x * f), jnp.float64(1.0), fac)   # xwgt = xwgt*wstr, in order
        dead = inreg & (prefix * suffix < thr)
        anydead = jnp.any(dead)
        xwgt = prefix[R - 1]
        qualify = (~anydead) & (xwgt >= AREAMIN)

        nconf1 = st["nconfig"] + 1
        fill = qualify & (nconf1 <= NCONFGMAX)
        over = qualify & (nconf1 > NCONFGMAX)
        need_fv = over & st["new_term"]
        jfv, ptrc_fv = _findvalue1(st["wgtv"], st["ptrc"], need_fv)
        jfv = jnp.where(need_fv, jfv, st["jprev"])
        ptrc = jnp.where(need_fv, ptrc_fv, st["ptrc"])
        jfill = jnp.minimum(nconf1, NCONFGMAX) - 1
        ptrc = jnp.where(fill, ptrc.at[jfill].set(jfill), ptrc)
        j = jnp.where(fill, jfill, jfv)
        old = st["wgtv"][j]
        replace = over & (old < xwgt)
        totwgt = jnp.where(replace, st["totwgt"] - old, st["totwgt"])
        new_term = jnp.where(over, replace, st["new_term"])
        write = fill | replace
        wgtv = jnp.where(write, st["wgtv"].at[j].set(xwgt), st["wgtv"])
        totwgt = jnp.where(write, totwgt + xwgt, totwgt)
        slots = jnp.where(write, st["slots"].at[j].set(istr), st["slots"])
        nconfig = jnp.where(qualify, jnp.minimum(nconf1, NCONFGMAX), st["nconfig"])
        jprev = jnp.where(qualify, j, st["jprev"])

        # odometer: advance at the first dead depth (subtree skipped) or at the last region, carrying upward
        d = jnp.where(anydead, jnp.argmax(dead).astype(jnp.int32), nrgn - 1)
        notmax = (ridx <= d) & (istr + 1 < nstr)
        q = (R - 1 - jnp.argmax(notmax[::-1])).astype(jnp.int32)
        inc = jnp.where(ridx < q, istr, jnp.where(ridx == q, istr + 1, 0))
        finished = ~jnp.any(notmax)
        restart = finished & (~ph) & ~(totwgt > 0.0)          # WRF: totwgt = 0 -> maximum overlap, second pass
        return dict(
            istr=jnp.where(restart, 0, inc), done=finished & ~restart, phase=ph | restart,
            nconfig=jnp.where(restart, 0, nconfig), totwgt=jnp.where(restart, 0.0, totwgt), wgtv=wgtv, ptrc=ptrc,
            slots=slots, new_term=jnp.where(restart, True, new_term), jprev=jprev)

    out = jax.lax.while_loop(lambda st: ~st["done"], body, state)
    return out["slots"], out["wgtv"], out["nconfig"], out["totwgt"], out["phase"]


def _overlap_configs(cld, pbr, pmxrgn, nmxrgn, day):
    """Index calculations for max-random overlap -> ccon (ncol, nconfgmax, pver+1), wgtv, nconfig, totwgt."""

    ncol, pver = cld.shape
    R = pver + 1
    rr = jnp.arange(R)
    valid = rr[None, :] < nmxrgn[:, None]
    # region of layer k: number of max-overlap regions whose bottom pressure lies above pmid(k)
    reg1 = jnp.sum(valid[:, None, :] & (pmxrgn[:, None, :R] < pbr[:, :, None]), axis=2)
    reg1 = jnp.minimum(reg1, R - 1)
    reg2 = jnp.zeros_like(reg1)                                # nmxrgn = 1, pmxrgn(1) = 1e30
    W1, nstr1, rho1, cloudy, nrgn1 = _region_streams(cld, reg1, R)
    W2, nstr2, rho2, _, nrgn2 = _region_streams(cld, reg2, R)
    slots, wgtv, nconfig, totwgt, second = jax.vmap(_select_column)(W1, nstr1, nrgn1, W2, nstr2, nrgn2, day)
    reg = jnp.where(second[:, None], reg2, reg1)
    rho = jnp.where(second[:, None], rho2, rho1)
    # ccon(k, j) = cstr(k, istr_j(region(k)))
    stream_of_layer = jnp.take_along_axis(slots, jnp.broadcast_to(reg[:, None, :], (ncol, NCONFGMAX, pver)), axis=2)
    ccon = cloudy[:, None, :] & (stream_of_layer >= rho[:, None, :])
    ccon = jnp.concatenate([jnp.zeros((ncol, NCONFGMAX, 1), bool), ccon], axis=2)
    return ccon, wgtv, nconfig, totwgt


# --------------------------------------------------------------------------------------------------------------------- #
# radcswmx                                                                                                              #
# --------------------------------------------------------------------------------------------------------------------- #
def _adding_allsky(allsky, clear, ccon, albdir, albdif, wgtv, nconfig, totwgt):
    """STEP 2 (adding method per configuration) + STEP 3 (configuration-weighted interface fluxes), one interval.

    Layer arrays (ncol, pver+1); ccon (ncol, C, pver+1).  Returns fluxup, fluxdn, fluxdndir (pverp+1, ncol).
    Each configuration is solved on its own; WRF's binary-tree sharing of identical partial configurations computes
    the same values from the same operands, so this is bitwise equivalent.
    """

    lay = lambda a: jnp.moveaxis(a, -1, 0)[..., None]                       # (pver+1, ncol, 1)
    rdir, rdif, tdir, tdif, explay = map(lay, allsky)
    rdirc, rdifc, tdirc, tdifc, explayc = map(lay, clear)
    cc = jnp.moveaxis(ccon, -1, 0)                                          # (pver+1, ncol, C)
    xs = (cc, rdir, rdif, tdir, tdif, explay, rdirc, rdifc, tdirc, tdifc, explayc)
    ncol, C = ccon.shape[0], ccon.shape[1]

    def down(carry, x):
        xexpt, xrdnd, xtdnt = carry
        cl, rdir_k, rdif_k, tdir_k, tdif_k, ex_k, rdirc_k, rdifc_k, tdirc_k, tdifc_k, exc_k = x
        ytdnd = jnp.where(cl, tdif_k, tdifc_k)
        yrdnd = jnp.where(cl, rdif_k, rdifc_k)
        yrdir = jnp.where(cl, rdir_k, rdirc_k)
        ytdir = jnp.where(cl, tdir_k, tdirc_k)
        yexp = jnp.where(cl, ex_k, exc_k)
        tdnmexp = xtdnt - xexpt
        rdenom = 1.0 / (1.0 - yrdnd * xrdnd)
        rdirexp = yrdir * xexpt
        zexpt = xexpt * yexp
        zrdnd = yrdnd + xrdnd * (ytdnd * ytdnd) * rdenom
        ztdnt = xexpt * ytdir + ytdnd * (tdnmexp + xrdnd * rdirexp) * rdenom
        return (zexpt, zrdnd, ztdnt), (zexpt, zrdnd, ztdnt)

    def up(carry, x):
        xrupd, xrups = carry
        cl, rdir_k, rdif_k, tdir_k, tdif_k, ex_k, rdirc_k, rdifc_k, tdirc_k, tdifc_k, exc_k = x
        yexpt = jnp.where(cl, ex_k, exc_k)
        yrupd = jnp.where(cl, rdif_k, rdifc_k)
        ytupd = jnp.where(cl, tdif_k, tdifc_k)
        yrdir = jnp.where(cl, rdir_k, rdirc_k)
        ytdir = jnp.where(cl, tdir_k, tdirc_k)
        rdenom = 1.0 / (1.0 - yrupd * xrupd)
        tdnmexp = ytdir - yexpt
        rdirexp = xrups * yexpt
        zrupd = yrupd + xrupd * (ytupd * ytupd) * rdenom
        zrups = yrdir + ytupd * (rdirexp + xrupd * tdnmexp) * rdenom
        return (zrupd, zrups), (zrupd, zrups)

    shp = (ncol, C)
    one, zero = jnp.ones(shp), jnp.zeros(shp)
    init_u = (jnp.broadcast_to(albdif[:, None], shp), jnp.broadcast_to(albdir[:, None], shp))
    _, (rupdif, rupdir) = jax.lax.scan(up, init_u, xs, reverse=True)       # interfaces 0..pver
    rupdif = jnp.concatenate([rupdif, init_u[0][None]], axis=0)            # + surface (pverp)
    rupdir = jnp.concatenate([rupdir, init_u[1][None]], axis=0)
    _, (exptdn, rdndif, tdntot) = jax.lax.scan(down, (one, zero, one), xs)  # interfaces 1..pverp
    exptdn = jnp.concatenate([one[None], exptdn], axis=0)
    rdndif = jnp.concatenate([zero[None], rdndif], axis=0)
    tdntot = jnp.concatenate([one[None], tdntot], axis=0)

    rdenom = 1.0 / (1.0 - rdndif * rupdif)
    term_up = (exptdn * rupdir + (tdntot - exptdn) * rupdif) * rdenom
    term_dn = exptdn + (tdntot - exptdn + exptdn * rupdir * rdndif) * rdenom
    fluxup = jnp.zeros(exptdn.shape[:-1])
    fluxdn = jnp.zeros(exptdn.shape[:-1])
    fluxdndir = jnp.zeros(exptdn.shape[:-1])
    for j in range(C):                                                      # iconfig = 1..nconfig, slot order
        on = j < nconfig
        xw = wgtv[:, j]
        fluxup = jnp.where(on, fluxup + xw * term_up[..., j], fluxup)
        fluxdn = jnp.where(on, fluxdn + xw * term_dn[..., j], fluxdn)
        fluxdndir = jnp.where(on, fluxdndir + xw * exptdn[..., j], fluxdndir)
    return fluxup / totwgt, fluxdn / totwgt, fluxdndir / totwgt


def _adding_clear(clear, albdir, albdif):
    """Clear-sky adding method and interface fluxes of radcswmx, one interval -> (pverp+1, ncol) arrays."""

    lay = lambda a: jnp.moveaxis(a, -1, 0)
    rdirc, rdifc, tdirc, tdifc, explayc = map(lay, clear)

    def down_c(carry, x):
        xexpt, xrdnd, xtdnt = carry
        rdirc_k, rdifc_k, tdirc_k, tdifc_k, exc_k = x
        zexpt = xexpt * exc_k
        rdenom = 1.0 / (1.0 - rdifc_k * xrdnd)
        rdirexp = rdirc_k * xexpt
        tdnmexp = xtdnt - xexpt
        ztdnt = xexpt * tdirc_k + tdifc_k * (tdnmexp + xrdnd * rdirexp) * rdenom
        zrdnd = rdifc_k + xrdnd * (tdifc_k * tdifc_k) * rdenom
        return (zexpt, zrdnd, ztdnt), (zexpt, zrdnd, ztdnt)

    def up_c(carry, x):
        xrupd, xrups = carry
        rdirc_k, rdifc_k, tdirc_k, tdifc_k, exc_k = x
        rdenom = 1.0 / (1.0 - rdifc_k * xrupd)
        zrups = rdirc_k + tdifc_k * (xrups * exc_k + xrupd * (tdirc_k - exc_k)) * rdenom
        zrupd = rdifc_k + xrupd * (tdifc_k * tdifc_k) * rdenom
        return (zrupd, zrups), (zrupd, zrups)

    ncol = albdir.shape[0]
    one, zero = jnp.ones((ncol,)), jnp.zeros((ncol,))
    xs = (rdirc, rdifc, tdirc, tdifc, explayc)
    _, (exptdnc, rdndifc, tdntotc) = jax.lax.scan(down_c, (one, zero, one), xs)
    exptdnc = jnp.concatenate([one[None], exptdnc], axis=0)
    rdndifc = jnp.concatenate([zero[None], rdndifc], axis=0)
    tdntotc = jnp.concatenate([one[None], tdntotc], axis=0)
    _, (rupdifc, rupdirc) = jax.lax.scan(up_c, (albdif, albdir), xs, reverse=True)
    rupdifc = jnp.concatenate([rupdifc, albdif[None]], axis=0)
    rupdirc = jnp.concatenate([rupdirc, albdir[None]], axis=0)
    rdenom = 1.0 / (1.0 - rdndifc * rupdifc)
    fluxup = (exptdnc * rupdirc + (tdntotc - exptdnc) * rupdifc) * rdenom
    fluxdn = exptdnc + (tdntotc - exptdnc + exptdnc * rupdirc * rdndifc) * rdenom
    return fluxup, fluxdn, exptdnc


def radcswmx(*, pnm, pbr, h2ommr, rh, o3mmr, aermmr, cld, cicewp, cliqwp, rel, rei, coszrs, solcon, asdir, asdif,
             aldir, aldif, nmxrgn, pmxrgn, co2mmr, tables: CamAeroptTables | None = None,
             idx: AerosolIndices = WRF_AEROSOL_INDICES):
    """radcswmx (cgs): the radcswmx outputs used by radctl, before radctl's unit conversion.

    pnm/pbr dynes/cm2 (radinp), pmxrgn dynes/cm2.  The spectral loop is a ``lax.scan`` over the 19 intervals whose
    carry holds WRF's spectral accumulators (same summation order).  Night columns (coszrs <= 0) return WRF's zero
    initialisation.
    """

    tables = tables if tables is not None else load_cam_aeropt_tables()
    f64 = lambda a: jnp.asarray(a, jnp.float64)
    pnm, pbr, h2ommr, rh, o3mmr, aermmr = map(f64, (pnm, pbr, h2ommr, rh, o3mmr, aermmr))
    cld, cicewp, cliqwp, rel, rei, coszrs, pmxrgn = map(f64, (cld, cicewp, cliqwp, rel, rei, coszrs, pmxrgn))
    ncol, pver = cld.shape
    pverp = pver + 1
    day = coszrs > 0.0
    cz = jnp.where(day, coszrs, 1.0)
    co2mmr = jnp.broadcast_to(f64(co2mmr), (ncol,))
    solcon8 = jnp.broadcast_to(jnp.asarray(solcon, jnp.float32).astype(jnp.float64), (ncol,))
    solin = solcon8 * cz * 1000.
    nmxrgn = jnp.broadcast_to(jnp.asarray(nmxrgn, jnp.int32), (ncol,))
    asdir, asdif, aldir, aldif = (jnp.broadcast_to(f64(a), (ncol,)) for a in (asdir, asdif, aldir, aldif))

    paths = _column_paths(pnm, h2ommr, rh, o3mmr, aermmr, cz, co2mmr, idx)
    ccon, wgtv, nconfig, totwgt = _overlap_configs(cld, pbr, pmxrgn, nmxrgn, day)
    totwgt = jnp.where(day, totwgt, 1.0)
    nconf = nconfig[None, :]

    def interval(acc, b):
        allsky, clear, tauxcl, tauxci = _band_optics(b, paths, cld, cicewp, cliqwp, rel, rei, cz)
        albdir = jnp.where(b["vis"], asdir, aldir)
        albdif = jnp.where(b["vis"], asdif, aldif)
        fu, fd, fdd = _adding_allsky(allsky, clear, ccon, albdir, albdif, wgtv, nconf, totwgt)
        fuc, fdc, exptdnc = _adding_clear(clear, albdir, albdif)
        solflx = solin * b["frcsol"] * b["psf"]
        wexptdn = fdd[pverp]                    # sum_j wgtv(j) * exptdn(pverp, j) / totwgt (same sum as fluxdndir)
        vis = b["vis"]
        sw_dir = wexptdn * solflx * 0.001
        sw_dif = (fd[pverp] - wexptdn) * solflx * 0.001
        new = dict(
            fsntoa=acc["fsntoa"] + solflx * (fd[0] - fu[0]),
            fsns=acc["fsns"] + solflx * (fd[pverp] - fu[pverp]),
            fswup=acc["fswup"] + solflx * fu, fswdn=acc["fswdn"] + solflx * fd,
            fswdndir=acc["fswdndir"] + solflx * fdd,
            sols=jnp.where(vis, acc["sols"] + sw_dir, acc["sols"]),
            solsd=jnp.where(vis, acc["solsd"] + sw_dif, acc["solsd"]),
            soll=jnp.where(vis, acc["soll"], acc["soll"] + sw_dir),
            solld=jnp.where(vis, acc["solld"], acc["solld"] + sw_dif),
            totfld=acc["totfld"] + solflx * ((fd[:-1] - fd[1:]) + (fu[1:] - fu[:-1])),
            fswupc=acc["fswupc"] + solflx * fuc, fswdnc=acc["fswdnc"] + solflx * fdc,
            fswdncdir=acc["fswdncdir"] + solflx * exptdnc,
            totfldc=acc["totfldc"] + solflx * ((fdc[:-1] - fdc[1:]) + (fuc[1:] - fuc[:-1])),
            fsntoac=acc["fsntoac"] + solflx * (fdc[0] - fuc[0]),
            tauxcl=tauxcl, tauxci=tauxci)
        return new, None

    zi, zl, zc = jnp.zeros((pverp + 1, ncol)), jnp.zeros((pverp, ncol)), jnp.zeros((ncol,))
    acc0 = dict(fsntoa=zc, fsns=zc, fswup=zi, fswdn=zi, fswdndir=zi, sols=zc, solsd=zc, soll=zc, solld=zc,
                totfld=zl, fswupc=zi, fswdnc=zi, fswdncdir=zi, totfldc=zl, fsntoac=zc,
                tauxcl=jnp.zeros((ncol, pver)), tauxci=jnp.zeros((ncol, pver)))
    acc, _ = jax.lax.scan(interval, acc0, _band_constants(tables))

    den = (pnm[:, :pver] - pnm[:, 1:]).T
    qrs = (-(lit(1.e-4) * CONST.gravit * acc["totfld"][1:]) / den).T
    qrscs = (-(lit(1.e-4) * CONST.gravit * acc["totfldc"][1:]) / den).T
    prof = lambda a: a[1:].T                                                # interfaces 1..pverp
    fsdndir = prof(acc["fswdndir"])
    fsdndif = prof(acc["fswdn"] - acc["fswdndir"])
    out = dict(
        qrs=qrs, qrscs=qrscs, fsup=prof(acc["fswup"]), fsupc=prof(acc["fswupc"]), fsdn=prof(acc["fswdn"]),
        fsdnc=prof(acc["fswdnc"]), fsdndir=fsdndir, fsdndif=fsdndif, fsdncdir=prof(acc["fswdncdir"]),
        fsdncdif=prof(acc["fswdnc"] - acc["fswdncdir"]), fsns=acc["fsns"], fsntoa=acc["fsntoa"],
        fsntoac=acc["fsntoac"], fsds=acc["fswdn"][pverp], fsdsdir=fsdndir[:, -1], fsdsdif=fsdndif[:, -1],
        sols=acc["sols"], soll=acc["soll"], solsd=acc["solsd"], solld=acc["solld"], solin=solin,
        tauxcl=acc["tauxcl"], tauxci=acc["tauxci"])
    dmask = lambda a: jnp.where(day.reshape((-1,) + (1,) * (a.ndim - 1)), a, 0.0)
    return {k: dmask(v) for k, v in out.items()}


# --------------------------------------------------------------------------------------------------------------------- #
# radctl SW branch                                                                                                      #
# --------------------------------------------------------------------------------------------------------------------- #
def cam_sw_radctl(*, q1, qliq=None, qice=None, cld, pmid, pint, t, cicewp, cliqwp, rel, rei, pmxrgn, nmxrgn, o3vmr,
                  coszrs, solcon, albedo, julian, mxaerl, m_hybi, co2mmr, landfrac=None,
                  tables: CamAeroptTables | None = None, aer_idx: AerosolIndices = WRF_AEROSOL_INDICES):
    """radctl ``if (dosw)`` with radforce = .false.: aqsat -> rh, get_int_scales + get_aerosol, radcswmx, MKS scaling.

    Operands in CAM order (r8, Pa; pmxrgn dynes/cm2 as returned by cldovrlap).  ``qliq``/``qice``/``landfrac`` only
    feed ``aerosol_indirect``, which has no outputs.  Returns the radctl SW outputs with radctl's scaling
    (``fsdndir``/``fsdndif`` stay cgs, as in WRF).
    """

    del qliq, qice, landfrac
    tables = tables if tables is not None else load_cam_aeropt_tables()
    f64 = lambda a: jnp.asarray(a, jnp.float64)
    q1, cld, pmid, pint, t, o3vmr = map(f64, (q1, cld, pmid, pint, t, o3vmr))
    pbr, pnm, _eccf, o3mmr = radinp(pmid, pint, o3vmr)
    rh = cam_rh(t, pmid, q1)
    aerosol = cam_aerosol(pint, julian, mxaerl, m_hybi, idx=aer_idx, tables=tables)
    alb = f64(albedo)
    sw = radcswmx(pnm=pnm, pbr=pbr, h2ommr=q1, rh=rh, o3mmr=o3mmr, aermmr=aerosol, cld=cld, cicewp=cicewp,
                  cliqwp=cliqwp, rel=rel, rei=rei, coszrs=coszrs, solcon=solcon, asdir=alb, asdif=alb, aldir=alb,
                  aldif=alb, nmxrgn=nmxrgn, pmxrgn=pmxrgn, co2mmr=co2mmr, tables=tables, idx=aer_idx)
    s = lit(1.e-3)
    fsntoa = sw["fsntoa"] * s
    fsntoac = sw["fsntoac"] * s
    return dict(
        qrs=sw["qrs"], qrscs=sw["qrscs"],
        fsup=sw["fsup"] * s, fsupc=sw["fsupc"] * s, fsdn=sw["fsdn"] * s, fsdnc=sw["fsdnc"] * s,
        fsdndir=sw["fsdndir"], fsdndif=sw["fsdndif"],
        fsns=sw["fsns"] * s, fsds=sw["fsds"] * s, fsdsdir=sw["fsdsdir"] * s, fsdsdif=sw["fsdsdif"] * s,
        swcftoa=fsntoa - fsntoac,
        sols=sw["sols"], soll=sw["soll"], solsd=sw["solsd"], solld=sw["solld"],
        tauxcl=sw["tauxcl"], tauxci=sw["tauxci"])


__all__ = [
    "AerosolIndices", "CamAeroptTables", "UNSET_AEROSOL_INDICES", "WRF_AEROSOL_INDICES", "aerosol_climatology",
    "aerosol_time_factors", "cam_aerosol", "cam_aqsat", "cam_rh", "cam_sw_radctl", "estbl_table", "estblf",
    "get_int_scales", "load_cam_aeropt_tables", "radcswmx", "vert_interpolate",
]
