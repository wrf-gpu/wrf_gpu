"""JIT/vmap-traceable JAX port of the classic WRF RRTM longwave column kernel.

The host-NumPy driver in :mod:`gpuwrf.physics.ra_lw_rrtm` (``solve_rrtm_lw_column``)
is parity-proven against the unmodified pristine WRF ``phys/module_ra_rrtm.F``
(``proofs/v060/run_rrtm_lw_parity.py``) but is a single-column Python loop with
per-band / per-g-point control flow and ``lru_cache`` table I/O -- it cannot ride
the device ``jax.lax.scan`` operational radiation slot.

This module re-expresses *exactly the same math* in a fully traceable form:

* the AER 16-band / 140-g-point lookup tables are loaded ONCE on the host
  (:func:`gpuwrf.physics.ra_lw_rrtm._load_tables`) and frozen into ``jnp`` device
  constants (:func:`_jax_tables`, cached) -- no host I/O in the trace;
* ``_prepare_atmosphere`` (buffer-layer build, O3 climatology, cloud optical
  depth), ``_setcoef``, the sixteen ``TAUGB`` bands, and the one-angle ``RTRN``
  transfer are vectorised over the layer axis (``nlayers``) with ``jnp.where``
  branch selection and ``jnp.take`` clamped 1-based row gathers, mirroring the
  host ``_row``/``_interp4``/``_binary_lower`` helpers;
* the downward / upward radiance sweeps in ``RTRN`` are sequential recurrences,
  expressed with :func:`jax.lax.scan` over layers;
* the whole column endpoint is ``jax.vmap`` -ed over the ``ncol`` batch axis and
  is JIT-traceable with no ``host_callback``/``io_callback``/``pure_callback``.

The contiguous-from-bottom property of the troposphere layers (pressure is
monotone-decreasing, so ``plog > 4.56`` is a leading run) makes the host's
scalar ``laytrop``/``layswtch``/``laylow`` counts equivalent to the per-layer
boolean masks used here; this equivalence is asserted in the proof harness.

This is the kernel the operational ``ra_lw_physics=1`` scan slot calls. Its
parity is bound by ``proofs/radiation/rrtm_lw_oracle.py`` (pristine-WRF oracle,
NOT a JAX-vs-JAX self-compare).
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.ra_lw_rrtm import (
    DEFAULT_PTOP_PA,
    DELTAP_MB,
    NBANDS,
    NGB,
    NGB_START,
    NGC,
    NGPT,
    NSPA,
    NSPB,
    ONEMINUS,
    SECANG,
    WTNUM,
    RRTMLWColumnResult,
    RRTMLWColumnState,
    _load_tables,
    _nint,
)

# --------------------------------------------------------------------------- #
# Frozen O3 / standard-atmosphere climatology constants (mirror the host).
# --------------------------------------------------------------------------- #
_O3SUM = np.asarray(
    [5.297e-8, 5.852e-8, 6.579e-8, 7.505e-8, 8.577e-8, 9.895e-8, 1.175e-7, 1.399e-7,
     1.677e-7, 2.003e-7, 2.571e-7, 3.325e-7, 4.438e-7, 6.255e-7, 8.168e-7, 1.036e-6,
     1.366e-6, 1.855e-6, 2.514e-6, 3.240e-6, 4.033e-6, 4.854e-6, 5.517e-6, 6.089e-6,
     6.689e-6, 1.106e-5, 1.462e-5, 1.321e-5, 9.856e-6, 5.960e-6, 5.960e-6], dtype=np.float64)
_PPSUM = np.asarray(
    [955.890, 850.532, 754.599, 667.742, 589.841, 519.421, 455.480, 398.085, 347.171, 301.735,
     261.310, 225.360, 193.419, 165.490, 141.032, 120.125, 102.689, 87.829, 75.123, 64.306,
     55.086, 47.209, 40.535, 34.795, 29.865, 19.122, 9.277, 4.660, 2.421, 1.294, 0.647], dtype=np.float64)
_O3WIN = np.asarray(
    [4.629e-8, 4.686e-8, 5.017e-8, 5.613e-8, 6.871e-8, 8.751e-8, 1.138e-7, 1.516e-7,
     2.161e-7, 3.264e-7, 4.968e-7, 7.338e-7, 1.017e-6, 1.308e-6, 1.625e-6, 2.011e-6,
     2.516e-6, 3.130e-6, 3.840e-6, 4.703e-6, 5.486e-6, 6.289e-6, 6.993e-6, 7.494e-6,
     8.197e-6, 9.632e-6, 1.113e-5, 1.146e-5, 9.389e-6, 6.135e-6, 6.135e-6], dtype=np.float64)
_PPWIN = np.asarray(
    [955.747, 841.783, 740.199, 649.538, 568.404, 495.815, 431.069, 373.464, 322.354, 277.190,
     237.635, 203.433, 174.070, 148.949, 127.408, 108.915, 93.114, 79.551, 67.940, 58.072,
     49.593, 42.318, 36.138, 30.907, 26.362, 16.423, 7.583, 3.620, 1.807, 0.938, 0.469], dtype=np.float64)
_PPROF = np.asarray(
    [1000.00, 855.47, 731.82, 626.05, 535.57, 458.16, 391.94, 335.29, 286.83, 245.38,
     209.91, 179.57, 153.62, 131.41, 112.42, 96.17, 82.27, 70.38, 60.21, 51.51,
     44.06, 37.69, 32.25, 27.59, 23.60, 20.19, 17.27, 14.77, 12.64, 10.81,
     9.25, 7.91, 6.77, 5.79, 4.95, 4.24, 3.63, 3.10, 2.65, 2.27, 1.94, 1.66,
     1.42, 1.22, 1.04, 0.89, 0.76, 0.65, 0.56, 0.48, 0.41, 0.35, 0.30, 0.26,
     0.22, 0.19, 0.16, 0.14, 0.12, 0.10], dtype=np.float64)
_TPROF = np.asarray(
    [279.94, 276.16, 270.73, 264.14, 256.71, 249.28, 241.97, 234.91, 228.78, 224.02,
     220.52, 217.31, 215.21, 213.48, 211.63, 211.45, 211.73, 212.71, 213.81, 214.95,
     215.96, 216.73, 217.42, 218.11, 218.89, 219.92, 221.31, 222.84, 224.39, 226.04,
     227.78, 229.73, 231.88, 234.22, 236.82, 239.50, 242.30, 245.21, 248.13, 251.08,
     254.04, 257.02, 259.84, 261.88, 263.38, 264.67, 265.42, 265.34, 264.45, 262.76,
     260.85, 258.78, 256.49, 254.02, 251.07, 248.23, 245.46, 242.77, 239.87, 237.53], dtype=np.float64)

_CO2VMR = (280.0 + 90.0 * np.exp(0.02 * (2009 - 2000))) * 1.0e-6
_N2OVMR = 319.0e-9
_CH4VMR = 1774.0e-9
_AMD, _AMW, _AVGDRO = 28.9644, 18.0154, 6.022e23
_AMDW, _AMDO = 1.607758, 0.603461
_GRAVIT = 9.81 * 100.0


def _o3ann() -> np.ndarray:
    """Annual-mean O3 climatology profile (host ``_o3_average_from_interfaces``)."""

    o3ann = np.zeros(31, dtype=np.float64)
    o3ann[0] = 0.5 * (_O3SUM[0] + _O3WIN[0])
    for k in range(1, 31):
        interp = _O3WIN[k - 1] + (_O3WIN[k] - _O3WIN[k - 1]) / (_PPWIN[k] - _PPWIN[k - 1]) * (_PPSUM[k] - _PPWIN[k - 1])
        o3ann[k] = 0.5 * (interp + _O3SUM[k])
    return o3ann


def _ppwrkh() -> np.ndarray:
    out = np.zeros(32, dtype=np.float64)
    out[0] = 1100.0
    out[1:31] = 0.5 * (_PPSUM[1:] + _PPSUM[:-1])
    return out


# --------------------------------------------------------------------------- #
# Device constant bundle (built once from the host lookup tables).
# --------------------------------------------------------------------------- #
class _JaxTables(NamedTuple):
    # per-band tables kept at native ng (band index is a static python int in
    # every band function, so a python tuple of device arrays is fine and avoids
    # any ng-padding / slicing bugs).
    absa: tuple                # tuple[jnp.ndarray]  (rows_a, ng)
    absb: tuple                # tuple[jnp.ndarray | None]  (rows_b, ng)
    absb_present: tuple
    selfref: tuple             # tuple[jnp.ndarray | None]  (10, ng)
    selfref_present: tuple
    forref: tuple              # tuple[jnp.ndarray | None]  (ng,)
    forref_present: tuple
    fracrefa: tuple            # tuple[jnp.ndarray]  (ng,) or (ng, ncols)
    fracrefa_cols: tuple
    fracrefb: tuple            # tuple[jnp.ndarray | None]  (ng,) or (ng, ncols)
    fracrefb_cols: tuple
    fracrefb_present: tuple
    minor: dict                # name -> jnp array (native length)
    preflog: jnp.ndarray       # (59,)
    tref: jnp.ndarray          # (59,)
    delwave: jnp.ndarray       # (16,)
    totplnk: jnp.ndarray       # (181, 16)
    tau: jnp.ndarray           # (5001,)
    tf: jnp.ndarray
    trans: jnp.ndarray
    corr1: jnp.ndarray         # (201,)
    corr2: jnp.ndarray
    bpade: float
    fluxfac: float
    heatfac: float
    local: dict                # name -> jnp array
    o3ann: jnp.ndarray         # (31,)
    ppwrkh: jnp.ndarray        # (32,)
    ngb: jnp.ndarray           # (140,) band index per g-point


@lru_cache(maxsize=1)
def _jax_tables() -> _JaxTables:
    # The lookup tables are FROZEN host constants. They are stored as NumPy arrays
    # (NOT jnp/device arrays): JAX ops fold NumPy operands as compile-time XLA
    # constants, which is trace-context-INDEPENDENT. Caching jnp arrays instead
    # would poison the lru_cache if the first call happened inside a jit trace
    # (the device arrays would be leaked tracers). NumPy constants avoid that.
    t = _load_tables()

    def J(a):
        return None if a is None else np.asarray(a, dtype=np.float64)

    absa = tuple(J(t.absa[b]) for b in range(NBANDS))
    absb_present = tuple(t.absb[b] is not None for b in range(NBANDS))
    absb = tuple(J(t.absb[b]) for b in range(NBANDS))
    selfref_present = tuple(t.selfref[b] is not None for b in range(NBANDS))
    selfref = tuple(J(t.selfref[b]) for b in range(NBANDS))
    forref_present = tuple(t.forref[b] is not None for b in range(NBANDS))
    forref = tuple(J(t.forref[b]) for b in range(NBANDS))

    def _cols(arr):
        if arr is None:
            return 0
        return 1 if arr.ndim == 1 else int(arr.shape[1])

    fracrefa = tuple(J(t.fracrefa[b]) for b in range(NBANDS))
    fracrefa_cols = tuple(_cols(t.fracrefa[b]) for b in range(NBANDS))
    fracrefb_present = tuple(t.fracrefb[b] is not None for b in range(NBANDS))
    fracrefb = tuple(J(t.fracrefb[b]) for b in range(NBANDS))
    fracrefb_cols = tuple(_cols(t.fracrefb[b]) for b in range(NBANDS))

    minor = {k: np.asarray(v, dtype=np.float64) for k, v in t.minor.items()}
    local = {k: np.asarray(v, dtype=np.float64) for k, v in t.local.items()}

    return _JaxTables(
        absa=absa, absb=absb, absb_present=absb_present,
        selfref=selfref, selfref_present=selfref_present,
        forref=forref, forref_present=forref_present,
        fracrefa=fracrefa, fracrefa_cols=fracrefa_cols,
        fracrefb=fracrefb, fracrefb_cols=fracrefb_cols, fracrefb_present=fracrefb_present,
        minor=minor,
        preflog=np.asarray(t.preflog), tref=np.asarray(t.tref),
        delwave=np.asarray(t.delwave), totplnk=np.asarray(t.totplnk),
        tau=np.asarray(t.tau), tf=np.asarray(t.tf), trans=np.asarray(t.trans),
        corr1=np.asarray(t.corr1), corr2=np.asarray(t.corr2),
        bpade=float(t.bpade), fluxfac=float(t.fluxfac), heatfac=float(t.heatfac),
        local=local,
        o3ann=np.asarray(_o3ann()), ppwrkh=np.asarray(_ppwrkh()),
        ngb=np.asarray(NGB.astype(np.int32)),
    )


# --------------------------------------------------------------------------- #
# Traceable gather helpers (mirror host _row / _interp4 / _binary_lower).
# --------------------------------------------------------------------------- #
def _fint(x):
    """Fortran-style truncation toward zero (host ``_fint``)."""

    return jnp.trunc(x).astype(jnp.int32)


def _row(table, idx_1b):
    """1-based, clamped row gather: ``table[clip(idx-1, 0, rows-1)]``.

    ``table`` is a FROZEN NumPy constant; ``jnp.asarray`` folds it to an XLA
    compile-time constant (concrete, trace-context-independent) so the traced
    integer index gathers without a NumPy-vs-tracer indexing error or a leak.
    """

    rows = table.shape[0]
    idx = jnp.clip(idx_1b - 1, 0, rows - 1)
    return jnp.asarray(table)[idx]


def _at(vec, idx_0b):
    """0-based clamped scalar gather into a 1-D ``vec`` (host ``vec[idx]``)."""

    return jnp.asarray(vec)[jnp.clip(idx_0b, 0, vec.shape[0] - 1)]


def _at1(vec, idx_1b):
    """1-based clamped scalar gather into a 1-D ``vec`` (host ``vec[idx-1]``)."""

    return jnp.asarray(vec)[jnp.clip(idx_1b - 1, 0, vec.shape[0] - 1)]


def _interp4(table, ind0, ind1, fac00, fac10, fac01, fac11):
    return (fac00 * _row(table, ind0) + fac10 * _row(table, ind0 + 1)
            + fac01 * _row(table, ind1) + fac11 * _row(table, ind1 + 1))


def _self_term(selfref, indself, selffrac):
    base = _row(selfref, indself)
    return base + selffrac * (_row(selfref, indself + 1) - base)


def _binary_lower(table, ind0, ind1, nsp, fs, fac00, fac10, fac01, fac11):
    omf = 1.0 - fs
    return (omf * fac00 * _row(table, ind0) + fs * fac00 * _row(table, ind0 + 1)
            + omf * fac10 * _row(table, ind0 + nsp) + fs * fac10 * _row(table, ind0 + nsp + 1)
            + omf * fac01 * _row(table, ind1) + fs * fac01 * _row(table, ind1 + 1)
            + omf * fac11 * _row(table, ind1 + nsp) + fs * fac11 * _row(table, ind1 + nsp + 1))


def _frac_col0(frac, ncols):
    """The first column of a fraction table (host ``fracref[:]`` or ``[:,0]``)."""

    return frac if (frac.ndim == 1) else frac[:, 0]


def _frac_interp_cols(frac, ncols, js, fs):
    """Interpolate a fraction table between columns js-1 and js (1-based).

    Mirrors host ``_frac_interp``: 1-D tables (ncols==1) are returned as-is.
    """

    if ncols == 1 or frac.ndim == 1:
        return jnp.asarray(frac if frac.ndim == 1 else frac[:, 0])
    fj = jnp.asarray(frac)
    j0 = jnp.clip(js - 1, 0, ncols - 1)
    j1 = jnp.clip(js, 0, ncols - 1)
    return fj[:, j0] + fs * (fj[:, j1] - fj[:, j0])


def _binary_params(a, b, strrat, mult):
    speccomb = a + strrat * b
    specparm = jnp.minimum(a / jnp.maximum(speccomb, 1.0e-300), ONEMINUS)
    specmult = mult * specparm
    js = 1 + _fint(specmult)
    fs = jnp.mod(specmult, 1.0)
    return speccomb, js, fs


# --------------------------------------------------------------------------- #
# Atmosphere prep (vectorised; nlayers is static given nz).
# --------------------------------------------------------------------------- #
def _nbuf(ptop_pa: float | None = None) -> int:
    """Above-model-top buffer-layer count (static int; sizes the kernel arrays).

    Mirrors WRF ``module_ra_rrtm.F:6781`` ``nint(p_top*0.01/deltap)``. ``ptop_pa``
    is the grid's real model-top pressure; ``None`` falls back to the legacy
    hardcoded ``DEFAULT_PTOP_PA`` (== 5000 Pa), keeping all existing callers
    bit-identical.  ``ptop_pa`` MUST be a Python float (it sets the buffer count
    and thus traced-array shapes), never a JAX-traced value."""

    ptop = DEFAULT_PTOP_PA if ptop_pa is None else float(ptop_pa)
    return _nint(ptop * 0.01 / DELTAP_MB)


def _o3_average_jax(pz_bottom_up, tab: _JaxTables):
    """O3 layer averages from interface pressures (vectorised host loop)."""

    bottom = pz_bottom_up[:-1]                       # (n,)
    top = pz_bottom_up[1:]
    ppk0 = tab.ppwrkh[:-1]                            # (31,)
    ppk1 = tab.ppwrkh[1:]
    # (n, 31) broadcast
    b = bottom[:, None]
    tp = top[:, None]
    pb1 = jnp.where(-(b - ppk0) >= 0.0, 0.0, b - ppk0)
    pb2 = jnp.where(-(b - ppk1) >= 0.0, 0.0, b - ppk1)
    pt1 = jnp.where(-(tp - ppk0) >= 0.0, 0.0, tp - ppk0)
    pt2 = jnp.where(-(tp - ppk1) >= 0.0, 0.0, tp - ppk1)
    acc = jnp.sum((pb2 - pb1 - pt2 + pt1) * tab.o3ann[None, :], axis=1)
    return acc / jnp.maximum(bottom - top, 1.0e-300)


def _prepare_atmosphere_jax(state_col, tab: _JaxTables, nz: int, ptop_pa: float | None = None):
    """Traceable per-column ``_prepare_atmosphere``. ``state_col`` is a dict of
    (nz,) arrays plus scalars ``emiss``/``tsk``. Returns the layer profiles on
    ``nlayers = nz + nbuf`` model layers (bottom-up). ``ptop_pa`` is the grid's
    real model-top pressure (static float; ``None`` -> legacy 5000 Pa)."""

    nbuf = _nbuf(ptop_pa)
    nlayers = nz + nbuf

    T = state_col["T"]; t8w = state_col["t8w"]; p = state_col["p"]; p8w = state_col["p8w"]
    qv = state_col["qv"]; qc = state_col["qc"]; qr = state_col["qr"]
    qi = state_col["qi"]; qs = state_col["qs"]
    cldfra = state_col["cloud_fraction"]; dz = state_col["dz"]
    emiss = state_col["emiss"]; tsk = state_col["tsk"]

    pprof_top = float(_PPROF[-1])  # static

    # --- pz / tz interfaces (nlayers+1), pavel / tavel (nlayers) ---
    pz = jnp.zeros(nlayers + 1, dtype=jnp.float64)
    pz = pz.at[: nz + 1].set(p8w * 0.01)
    # buffer interfaces above nz: pz[l] = pz[l-1] - DELTAP for l in [nz+1, nlayers-1]
    # then pz[nlayers] = 0. Build statically since the step is constant.
    for l in range(nz + 1, nlayers):
        pz = pz.at[l].set(pz[l - 1] - DELTAP_MB)
    pz = pz.at[nlayers].set(0.0)

    pavel = jnp.zeros(nlayers, dtype=jnp.float64)
    pavel = pavel.at[:nz].set(p * 0.01)
    for l in range(nz + 1, nlayers):
        pavel = pavel.at[l - 1].set(0.5 * (pz[l] + pz[l - 1]))
    pavel = pavel.at[nlayers - 1].set(0.5 * (pz[nlayers] + pz[nlayers - 1]))

    # --- F2 positivity GUARD (NOT a masking clamp) ----------------------------
    # WRF builds the above-model-top buffer as pz[l]=pz[l-1]-deltap; with the
    # buffer correctly sized to the grid p_top (F1) every layer mid-pressure
    # pavel and every interior interface pz[1:nlayers] is strictly positive
    # (pz[nlayers] is intentionally exactly 0, WRF module_ra_rrtm.F:4068). If a
    # mis-configured / pathological column drives any of those non-positive, the
    # atmosphere is unphysical: rather than silently CLAMP it to a finite value
    # (the forbidden masking-clamp pattern -- it would emit plausible-looking but
    # meaningless LW heating that passes the finiteness gate), we TAINT pavel/pz
    # with NaN so the result NaN-propagates and the sanity gate FAILS LOUD. In
    # production (positive pressures) the guard is a no-op and the path is
    # bit-identical.
    pz_interior = pz[1:nlayers]
    bad = jnp.logical_or(jnp.any(pavel <= 0.0), jnp.any(pz_interior <= 0.0))
    taint = jnp.where(bad, jnp.nan, 0.0)
    pavel = pavel + taint
    pz = pz + taint

    tz = jnp.zeros(nlayers + 1, dtype=jnp.float64)
    tz = tz.at[: nz + 1].set(t8w)
    tavel = jnp.zeros(nlayers, dtype=jnp.float64)
    tavel = tavel.at[:nz].set(T)

    # standard-atmosphere temperature interpolation for buffer interface temps.
    pprof = jnp.asarray(_PPROF)
    tprof = jnp.asarray(_TPROF)
    nprof = _PPROF.size
    # varint[l] for l in 1..nlayers: piecewise-linear T(pprof) lookup at pz[l].
    # vectorised: for each interface find klev = last index with pprof[klev] >= pz
    # (host scans pprof[1:] for first pprof[ll] < pz, klev = ll-1).
    pz_q = pz[1: nlayers + 1]                         # (nlayers,)
    # count of pprof entries (index>=1) strictly greater-or-equal handling:
    # host: klev = (#ll in 1..nprof-1 with pprof[ll] >= pz_q) ... equivalently
    # klev = number of pprof[1:] entries >= pz, capped at nprof-1, but only when
    # pprof[-1] < pz (else klev defaults to nprof-1 and uses flat tprof[-1]).
    ge = (pprof[1:][None, :] >= pz_q[:, None])        # (nlayers, nprof-1)
    klev = jnp.sum(ge.astype(jnp.int32), axis=1)      # first ll where pprof[ll]<pz => klev=ll-1
    klev = jnp.clip(klev, 0, nprof - 1)
    interior = pprof_top < pz_q                        # host guard `pprof[-1] < pz[l]`
    k0 = jnp.clip(klev, 0, nprof - 2)
    wght = (pz_q - pprof[k0]) / (pprof[k0 + 1] - pprof[k0])
    interp_v = wght * (tprof[k0 + 1] - tprof[k0]) + tprof[k0]
    flat_v = tprof[jnp.clip(klev, 0, nprof - 1)]
    not_last = klev != (nprof - 1)
    varint_q = jnp.where(interior & not_last, interp_v, flat_v)  # (nlayers,) for l=1..nlayers
    # buffer temps: tz[l] = varint[l] + (tz[nz]-varint[nz]); tavel[l-1]=0.5*(tz[l]+tz[l-1])
    tz_nz = tz[nz]
    varint_nz = varint_q[nz - 1]                       # varint index l=nz -> q-index nz-1
    delta_t = tz_nz - varint_nz
    for l in range(nz + 1, nlayers + 1):
        tz = tz.at[l].set(varint_q[l - 1] + delta_t)
    for l in range(nz + 1, nlayers + 1):
        tavel = tavel.at[l - 1].set(0.5 * (tz[l] + tz[l - 1]))

    # --- coldry + species columns (wkl rows 0,1,2,3,5) ---
    layer = jnp.arange(nlayers)
    in_model = layer < nz
    qv_pad = jnp.zeros(nlayers, dtype=jnp.float64).at[:nz].set(qv)
    h2ovmr_model = jnp.maximum(qv_pad, 1.0e-12) * _AMDW
    h2ovmr = jnp.where(in_model, h2ovmr_model, 5.0e-6)
    amm = (1.0 - h2ovmr) * _AMD + h2ovmr * _AMW
    dpz = pz[:-1] - pz[1:]
    coldry = dpz * 1.0e3 * _AVGDRO / (_GRAVIT * amm * (1.0 + h2ovmr))

    # O3 (annual climatology), independent of model/buffer.
    # host: shifted = _o3_average(pz[1:]) has nlayers-1 entries; o3prof2[:shifted]
    # = shifted, o3prof2[-1] = 6.135e-6 (the topmost layer is the climatology cap).
    shifted = _o3_average_jax(pz[1:], tab)             # (nlayers-1,)
    o3prof2 = jnp.concatenate([shifted, jnp.array([6.135e-6])])  # (nlayers,)

    wkl0 = h2ovmr
    wkl1 = jnp.full(nlayers, _CO2VMR)
    wkl2 = o3prof2 * _AMDO
    wkl3 = jnp.full(nlayers, _N2OVMR)
    wkl5 = jnp.full(nlayers, _CH4VMR)
    # buffer rows 3,5 inherit the topmost model value (host: wkl[3/5,l]=wkl[..,nz-1])
    # for model layers it is the constant; topmost model value is the same const,
    # so the buffer values equal the constants -> nothing to override.
    wkl0c = wkl0 * coldry
    wkl1c = wkl1 * coldry
    wkl2c = wkl2 * coldry
    wkl3c = wkl3 * coldry
    wkl5c = wkl5 * coldry

    # --- cloud optical depth + fraction (model layers only) ---
    p_pa = jnp.zeros(nlayers).at[:nz].set(p)
    T_pad = jnp.zeros(nlayers).at[:nz].set(T)
    qc_pad = jnp.zeros(nlayers).at[:nz].set(qc)
    qi_pad = jnp.zeros(nlayers).at[:nz].set(qi)
    qr_pad = jnp.zeros(nlayers).at[:nz].set(qr)
    qs_pad = jnp.zeros(nlayers).at[:nz].set(qs)
    dz_pad = jnp.zeros(nlayers).at[:nz].set(dz)
    cldfra_pad = jnp.zeros(nlayers).at[:nz].set(cldfra)
    ro = jnp.where(in_model, p_pa / (287.0 * jnp.where(in_model, T_pad, 1.0)), 0.0)
    clwp = ro * qc_pad * dz_pad * 1000.0
    ciwp = ro * qi_pad * dz_pad * 1000.0
    plwp = jnp.power(jnp.maximum(ro * qr_pad, 0.0), 0.75) * dz_pad * 1000.0
    piwp = jnp.power(jnp.maximum(ro * qs_pad, 0.0), 0.75) * dz_pad * 1000.0
    taucloud = jnp.where(in_model, 0.144 * clwp + 0.0735 * ciwp + 0.330e-3 * plwp + 2.34e-3 * piwp, 0.0)
    cloudfrac = jnp.where(taucloud > 0.01, 1.0, jnp.where(in_model, cldfra_pad, 0.0))

    tbound = jnp.minimum(tsk, 339.99)
    semiss = jnp.full(NBANDS, emiss, dtype=jnp.float64)

    wkl = (wkl0c, wkl1c, wkl2c, wkl3c, wkl5c)
    return dict(
        pavel=pavel, tavel=tavel, pz=pz, tz=tz, cloudfrac=cloudfrac, taucloud=taucloud,
        coldry=coldry, wkl=wkl, tbound=tbound, semiss=semiss, nlayers=nlayers, nz=nz,
    )


# --------------------------------------------------------------------------- #
# setcoef (vectorised over layers; laytrop/layswtch/laylow as per-layer masks).
# --------------------------------------------------------------------------- #
class _CoefJax(NamedTuple):
    colh2o: jnp.ndarray
    colco2: jnp.ndarray
    colo3: jnp.ndarray
    coln2o: jnp.ndarray
    colch4: jnp.ndarray
    water: jnp.ndarray
    co2mult: jnp.ndarray
    fac00: jnp.ndarray
    fac01: jnp.ndarray
    fac10: jnp.ndarray
    fac11: jnp.ndarray
    forfac: jnp.ndarray
    selffac: jnp.ndarray
    selffrac: jnp.ndarray
    jp: jnp.ndarray
    jt: jnp.ndarray
    jt1: jnp.ndarray
    indself: jnp.ndarray
    low_mask: jnp.ndarray      # plog > 4.56  (== l < laytrop)
    swtch_mask: jnp.ndarray    # l < layswtch
    laylow: jnp.ndarray        # scalar (1-based count)


def _setcoef_jax(atm, tab: _JaxTables):
    pavel = atm["pavel"]; tavel = atm["tavel"]; coldry = atm["coldry"]
    wkl0, wkl1, wkl2, wkl3, wkl5 = atm["wkl"]
    nlayers = atm["nlayers"]

    # F2: positivity GUARD instead of the forbidden ``jnp.maximum(pavel,1e-300)``
    # masking clamp. A non-positive pavel is unphysical; emit NaN (fail-loud /
    # NaN-propagate) rather than silently log a clamped floor. In production
    # (pavel>0, guaranteed by the F1-sized buffer + the prepare-time guard) this
    # is exactly ``jnp.log(pavel)`` -- bit-identical.
    plog = jnp.where(pavel > 0.0, jnp.log(jnp.where(pavel > 0.0, pavel, 1.0)), jnp.nan)
    jp = jnp.clip(_fint(36.0 - 5.0 * (plog + 0.04)), 1, 58)
    jp1 = jp + 1
    fp = 5.0 * (_at1(tab.preflog, jp) - plog)
    tref_jp = _at1(tab.tref, jp)
    tref_jp1 = _at1(tab.tref, jp1)
    jt = jnp.clip(_fint(3.0 + (tavel - tref_jp) / 15.0), 1, 4)
    ft = ((tavel - tref_jp) / 15.0) - (jt - 3).astype(jnp.float64)
    jt1 = jnp.clip(_fint(3.0 + (tavel - tref_jp1) / 15.0), 1, 4)
    ft1 = ((tavel - tref_jp1) / 15.0) - (jt1 - 3).astype(jnp.float64)
    water = wkl0 / jnp.maximum(coldry, 1.0e-300)
    scalefac = pavel * (296.0 / 1013.0) / tavel

    low_mask = plog > 4.56
    swtch_high = plog > 5.76
    low_low = plog >= 6.62

    factor = (tavel - 188.0) / 7.2
    indself = jnp.where(low_mask, jnp.clip(_fint(factor) - 7, 1, 9), 1)
    selffrac = jnp.where(low_mask, factor - (indself + 7).astype(jnp.float64), 0.0)

    forfac = scalefac / (1.0 + water)
    selffac = water * forfac
    colh2o = 1.0e-20 * wkl0
    co2_raw = 1.0e-20 * wkl1
    colco2 = jnp.where(co2_raw != 0.0, co2_raw, 1.0e-32 * coldry)
    colo3 = 1.0e-20 * wkl2
    n2o_raw = 1.0e-20 * wkl3
    coln2o = jnp.where(n2o_raw != 0.0, n2o_raw, 1.0e-32 * coldry)
    ch4_raw = 1.0e-20 * wkl5
    colch4 = jnp.where(ch4_raw != 0.0, ch4_raw, 1.0e-32 * coldry)
    co2reg = 3.55e-24 * coldry
    co2mult = (colco2 - co2reg) * 272.63 * jnp.exp(-1919.4 / tavel) / (8.7604e-4 * tavel)
    compfp = 1.0 - fp
    fac10 = compfp * ft
    fac00 = compfp * (1.0 - ft)
    fac11 = fp * ft1
    fac01 = fp * (1.0 - ft1)

    # layswtch / laylow scalar counts from the contiguous masks (the troposphere
    # branch uses the per-layer ``low_mask`` directly; the scalar ``laytrop`` count
    # the host carries is not separately needed here).
    layswtch = jnp.sum(swtch_high.astype(jnp.int32))
    laylow = jnp.sum(low_low.astype(jnp.int32))
    laylow = jnp.where(laylow == 0, 1, laylow)
    # host: if layswtch<n and jp[layswtch]<=6: layswtch+=1  (jp here is 0-based idx)
    jp_at_swtch = _at(jp, layswtch)  # jp[layswtch]
    layswtch = jnp.where((layswtch < nlayers) & (jp_at_swtch <= 6), layswtch + 1, layswtch)

    layer = jnp.arange(nlayers)
    swtch_mask = layer < layswtch

    return _CoefJax(
        colh2o=colh2o, colco2=colco2, colo3=colo3, coln2o=coln2o, colch4=colch4,
        water=water, co2mult=co2mult, fac00=fac00, fac01=fac01, fac10=fac10, fac11=fac11,
        forfac=forfac, selffac=selffac, selffrac=selffrac, jp=jp, jt=jt, jt1=jt1,
        indself=indself, low_mask=low_mask, swtch_mask=swtch_mask, laylow=laylow,
    )


# --------------------------------------------------------------------------- #
# Per-band optical depth (TAUGB1..16), vectorised over layers via jax.vmap.
# Each band returns (nlayers, ng_band) tau and (nlayers, ng_band) pfrac.
# --------------------------------------------------------------------------- #
def _gb_pure(tab, c, band, gas, with_self, with_for):
    """Bands 1, 10, 11, 14 (host ``_pure_band``): per-layer vectorised."""

    ng = int(NGC[band]); nspa = int(NSPA[band]); nspb = int(NSPB[band])
    absa = tab.absa[band]
    selfref = tab.selfref[band]; forref = tab.forref[band]
    fa = tab.fracrefa[band]
    fb_present = tab.fracrefb_present[band]
    fb = tab.fracrefb[band]
    absb_present = tab.absb_present[band]
    absb = tab.absb[band] if absb_present else absa  # placeholder when absent
    frac_a0 = _frac_col0(fa, tab.fracrefa_cols[band])
    frac_b0 = _frac_col0(fb, tab.fracrefb_cols[band]) if fb_present else frac_a0

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        # host: low = l < laytrop or absb is None
        low = c.low_mask[idx] if absb_present else jnp.array(True)
        # low path
        ind0_l = ((jp - 1) * 5 + (jt - 1)) * nspa + 1
        ind1_l = (jp * 5 + (jt1 - 1)) * nspa + 1
        tau_l = gas[idx] * _interp4(absa, ind0_l, ind1_l, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        if with_self:
            tau_l = tau_l + gas[idx] * c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx])
        if with_for and tab.forref_present[band]:
            tau_l = tau_l + gas[idx] * c.forfac[idx] * forref
        # high path
        ind0_h = ((jp - 13) * 5 + (jt - 1)) * nspb + 1
        ind1_h = ((jp - 12) * 5 + (jt1 - 1)) * nspb + 1
        tau_h = gas[idx] * _interp4(absb, ind0_h, ind1_h, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        if with_for and tab.forref_present[band]:
            tau_h = tau_h + gas[idx] * c.forfac[idx] * forref
        tau = jnp.where(low, tau_l[:ng], tau_h[:ng])
        frac = jnp.where(low, frac_a0[:ng], frac_b0[:ng])
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _gb2(tab, c):
    band = 1; ng = int(NGC[band]); nspa = int(NSPA[band]); nspb = int(NSPB[band])
    absa = tab.absa[band]; absb = tab.absb[band]
    selfref = tab.selfref[band]; forref = tab.forref[band]
    fa = tab.fracrefa[band]; fb = tab.fracrefb[band]
    refparam = tab.local["TAUGB2_REFPARAM"]

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        fp = c.fac11[idx] + c.fac01[idx]
        # WRF: ``IFP = 2.E2*FP+0.5 ; IF (IFP.LE.0) IFP=0`` with CORR1/2 dim (0:200)
        # and FP guaranteed in [0,1] for a physical column => IFP in [0,200].
        # F2: the upper ``jnp.clip(ifp,0,200)`` was a masking clamp -- on a
        # pathological (NaN/oob) column it silently swallowed the bad index and
        # returned a finite CORR. We still clamp the GATHER index (required to keep
        # it in-bounds for XLA), but multiply by a NaN ``guard`` whenever fp is
        # non-finite or ifp falls outside the valid [0,200] table range, so a bad
        # column NaN-propagates (fail-loud) instead of emitting finite garbage.
        ifp = jnp.maximum(0, _fint(200.0 * fp + 0.5))
        guard = jnp.where(jnp.isfinite(fp) & (ifp >= 0) & (ifp <= 200), 1.0, jnp.nan)
        cr2 = jnp.asarray(tab.corr2)[jnp.clip(ifp, 0, 200)] * guard
        cr1 = jnp.asarray(tab.corr1)[jnp.clip(ifp, 0, 200)] * guard
        fc00 = c.fac00[idx] * cr2; fc10 = c.fac10[idx] * cr2
        fc01 = c.fac01[idx] * cr1; fc11 = c.fac11[idx] * cr1
        # low
        h2oparam = c.water[idx] / (c.water[idx] + 0.002)
        # ifrac = first idx in 2..12 with h2oparam >= refparam[idx-1], else 13
        refp = jnp.asarray(refparam)
        ge = (h2oparam >= refp[1:12])                 # refparam[1..11] -> idx 2..12
        ifrac = jnp.where(jnp.any(ge), 2 + jnp.argmax(ge.astype(jnp.int32)), 13)
        rp0 = refp[jnp.clip(ifrac - 1, 0, refp.shape[0] - 1)]
        rp1 = refp[jnp.clip(ifrac - 2, 0, refp.shape[0] - 1)]
        fracint = (h2oparam - rp0) / (rp1 - rp0)
        ind0_l = ((jp - 1) * 5 + (jt - 1)) * nspa + 1
        ind1_l = (jp * 5 + (jt1 - 1)) * nspa + 1
        tau_l = c.colh2o[idx] * (
            fc00 * _row(absa, ind0_l) + fc10 * _row(absa, ind0_l + 1)
            + fc01 * _row(absa, ind1_l) + fc11 * _row(absa, ind1_l + 1)
            + c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx])
            + c.forfac[idx] * forref)
        faj = jnp.asarray(fa)
        col0 = jnp.clip(ifrac - 1, 0, fa.shape[1] - 1)
        col1 = jnp.clip(ifrac - 2, 0, fa.shape[1] - 1)
        frac_l = faj[:, col0] + fracint * (faj[:, col1] - faj[:, col0])
        # high
        ind0_h = ((jp - 13) * 5 + (jt - 1)) * nspb + 1
        ind1_h = ((jp - 12) * 5 + (jt1 - 1)) * nspb + 1
        tau_h = c.colh2o[idx] * (
            fc00 * _row(absb, ind0_h) + fc10 * _row(absb, ind0_h + 1)
            + fc01 * _row(absb, ind1_h) + fc11 * _row(absb, ind1_h + 1)
            + c.forfac[idx] * forref)
        frac_h = fb            # fracrefb2 is 1-D (16,)
        low = c.low_mask[idx]
        tau = jnp.where(low, tau_l[:ng], tau_h[:ng])
        frac = jnp.where(low, frac_l[:ng], frac_h[:ng])
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _gb3(tab, c):
    band = 2; ng = int(NGC[band]); nspa = int(NSPA[band]); nspb = int(NSPB[band])
    absa = tab.absa[band]; absb = tab.absb[band]
    selfref = tab.selfref[band]; forref = tab.forref[band]
    fa = tab.fracrefa[band]; fa_cols = tab.fracrefa_cols[band]
    fb = tab.fracrefb[band]; fb_cols = tab.fracrefb_cols[band]
    h2oref = tab.local["TAUGB3_H2OREF"]; n2oref = tab.local["TAUGB3_N2OREF"]
    co2ref = tab.local["TAUGB3_CO2REF"]; etaref = tab.local["TAUGB3_ETAREF"]
    strrat = 1.19268
    absn2oa = tab.minor["ABSN2OAC3"]; absn2ob = tab.minor["ABSN2OBC3"]

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        # --- low ---
        speccomb, js, fs = _binary_params(c.colh2o[idx], c.colco2[idx], strrat, 8.0)
        js2, fs2 = jnp.where(js == 8, jnp.where(fs >= 0.9, 9, 8), js), \
            jnp.where(js == 8, jnp.where(fs >= 0.9, 10.0 * (fs - 0.9), fs / 0.9), fs)
        ns = js2 + _fint(fs2 + 0.5)
        fp = c.fac01[idx] + c.fac11[idx]
        eta = _at1(etaref, ns)
        wcomb1 = jnp.where(ns == 10, _at1(h2oref, jp), strrat * _at1(co2ref, jp) / (1.0 - eta))
        wcomb2 = jnp.where(ns == 10, _at1(h2oref, jp1_of(jp)), strrat * _at1(co2ref, jp1_of(jp)) / (1.0 - eta))
        n2o_jp = _at1(n2oref, jp)
        n2o_jp1 = _at1(n2oref, jp1_of(jp))
        ratio = (n2o_jp / wcomb1) + fp * ((n2o_jp1 / wcomb2) - (n2o_jp / wcomb1))
        n2omult = c.coln2o[idx] - speccomb * ratio
        ind0_l = ((jp - 1) * 5 + (jt - 1)) * nspa + js2
        ind1_l = (jp * 5 + (jt1 - 1)) * nspa + js2
        tau_l = speccomb * _binary_lower(absa, ind0_l, ind1_l, nspa, fs2, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        tau_l = tau_l + c.colh2o[idx] * (c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx]) + c.forfac[idx] * forref)
        tau_l = tau_l + n2omult * absn2oa
        frac_l = _frac_interp_cols(fa, fa_cols, js2, fs2)
        # --- high ---
        speccomb_h, js_h, fs_h = _binary_params(c.colh2o[idx], c.colco2[idx], strrat, 4.0)
        ns_h = js_h + _fint(fs_h + 0.5)
        eta_h = _at1(etaref, ns_h)
        wcomb1h = jnp.where(ns_h == 5, _at1(h2oref, jp), strrat * _at1(co2ref, jp) / (1.0 - eta_h))
        wcomb2h = jnp.where(ns_h == 5, _at1(h2oref, jp1_of(jp)), strrat * _at1(co2ref, jp1_of(jp)) / (1.0 - eta_h))
        ratio_h = (n2o_jp / wcomb1h) + fp * ((n2o_jp1 / wcomb2h) - (n2o_jp / wcomb1h))
        n2omult_h = c.coln2o[idx] - speccomb_h * ratio_h
        ind0_h = ((jp - 13) * 5 + (jt - 1)) * nspb + js_h
        ind1_h = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js_h
        tau_h = speccomb_h * _binary_lower(absb, ind0_h, ind1_h, nspb, fs_h, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        tau_h = tau_h + c.colh2o[idx] * c.forfac[idx] * forref + n2omult_h * absn2ob
        frac_h = _frac_interp_cols(fb, fb_cols, js_h, fs_h)
        low = c.low_mask[idx]
        tau = jnp.where(low, tau_l[:ng], tau_h[:ng])
        frac = jnp.where(low, frac_l[:ng], frac_h[:ng])
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def jp1_of(jp):
    return jp + 1


def _gb_simple_binary(tab, c, band, low_a, low_b, strrat_low,
                      high_a=None, high_b=None, strrat_high=None, high_adjust=None):
    """Bands 4, 5 (simple binary; band 5 adds a wx term handled by caller=0)."""

    ng = int(NGC[band]); nspa = int(NSPA[band]); nspb = int(NSPB[band])
    absa = tab.absa[band]; absb = tab.absb[band]
    selfref = tab.selfref[band]
    fa = tab.fracrefa[band]; fa_cols = tab.fracrefa_cols[band]
    fb = tab.fracrefb[band]; fb_cols = tab.fracrefb_cols[band]
    has_high = high_a is not None

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        # low
        speccomb, js, fs = _binary_params(low_a[idx], low_b[idx], strrat_low, 8.0)
        ind0_l = ((jp - 1) * 5 + (jt - 1)) * nspa + js
        ind1_l = (jp * 5 + (jt1 - 1)) * nspa + js
        tau_l = speccomb * _binary_lower(absa, ind0_l, ind1_l, nspa, fs, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        tau_l = tau_l + c.colh2o[idx] * c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx])
        frac_l = _frac_interp_cols(fa, fa_cols, js, fs)
        if not has_high:
            return tau_l[:ng], frac_l[:ng]
        # high
        speccomb_h, js_h, fs_h = _binary_params(high_a[idx], high_b[idx], strrat_high, 4.0)
        if high_adjust == "band4":
            js_adj = jnp.where(js_h > 1, js_h + 1, jnp.where(fs_h >= 0.0024, 2, 1))
            fs_adj = jnp.where(js_h > 1, fs_h, jnp.where(fs_h >= 0.0024, (fs_h - 0.0024) / 0.9976, fs_h / 0.0024))
            js_h, fs_h = js_adj, fs_adj
        ind0_h = ((jp - 13) * 5 + (jt - 1)) * nspb + js_h
        ind1_h = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js_h
        tau_h = speccomb_h * _binary_lower(absb, ind0_h, ind1_h, nspb, fs_h, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        frac_h = _frac_interp_cols(fb, fb_cols, js_h, fs_h)
        low = c.low_mask[idx]
        tau = jnp.where(low, tau_l[:ng], tau_h[:ng])
        frac = jnp.where(low, frac_l[:ng], frac_h[:ng])
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _gb6(tab, c):
    """Band 6: lower H2O + co2mult; wx (CFC) terms are zero in this WRF config."""

    band = 5; ng = int(NGC[band])
    absa = tab.absa[band]; selfref = tab.selfref[band]
    fa0 = _frac_col0(tab.fracrefa[band], tab.fracrefa_cols[band])  # 1-D (ng,)
    absco2 = tab.minor["ABSCO2C6"]

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        ind0 = ((jp - 1) * 5 + (jt - 1)) + 1
        ind1 = (jp * 5 + (jt1 - 1)) + 1
        tau_l = c.colh2o[idx] * (
            _interp4(absa, ind0, ind1, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
            + c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx]))
        tau_l = tau_l + c.co2mult[idx] * absco2
        low = c.low_mask[idx]
        tau = jnp.where(low, tau_l[:ng], jnp.zeros(ng))
        return tau, fa0[:ng]

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _gb7(tab, c):
    band = 6; ng = int(NGC[band]); nspa = int(NSPA[band])
    absa = tab.absa[band]; absb = tab.absb[band]
    selfref = tab.selfref[band]
    fa = tab.fracrefa[band]; fa_cols = tab.fracrefa_cols[band]
    fb = tab.fracrefb[band]
    absco2 = tab.minor["ABSCO2C7"]

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        # low
        speccomb, js, fs = _binary_params(c.colh2o[idx], c.colo3[idx], 8.21104e4, 8.0)
        ind0_l = ((jp - 1) * 5 + (jt - 1)) * nspa + js
        ind1_l = (jp * 5 + (jt1 - 1)) * nspa + js
        tau_l = speccomb * _binary_lower(absa, ind0_l, ind1_l, nspa, fs, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        tau_l = tau_l + c.colh2o[idx] * c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx])
        frac_l = _frac_interp_cols(fa, fa_cols, js, fs)
        # high
        ind0_h = ((jp - 13) * 5 + (jt - 1)) + 1
        ind1_h = ((jp - 12) * 5 + (jt1 - 1)) + 1
        tau_h = c.colo3[idx] * _interp4(absb, ind0_h, ind1_h, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        frac_h = fb            # fracrefb7 is 1-D (ng,)
        low = c.low_mask[idx]
        tau = jnp.where(low, tau_l[:ng], tau_h[:ng]) + c.co2mult[idx] * absco2[:ng]
        frac = jnp.where(low, frac_l[:ng], frac_h[:ng])
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _gb8(tab, c):
    """Band 8: layswtch branch; wx (CFC) terms zero in this WRF config."""

    band = 7; ng = int(NGC[band])
    absa = tab.absa[band]; absb = tab.absb[band]
    selfref = tab.selfref[band]
    fa0 = _frac_col0(tab.fracrefa[band], tab.fracrefa_cols[band])  # 1-D
    fb0 = _frac_col0(tab.fracrefb[band], tab.fracrefb_cols[band])  # 1-D
    h2oref = tab.local["TAUGB8_H2OREF"]; n2oref = tab.local["TAUGB8_N2OREF"]; o3ref = tab.local["TAUGB8_O3REF"]
    absco2a = tab.minor["ABSCO2AC8"]; absco2b = tab.minor["ABSCO2BC8"]
    absn2oa = tab.minor["ABSN2OAC8"]; absn2ob = tab.minor["ABSN2OBC8"]

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        fp = c.fac01[idx] + c.fac11[idx]
        # low (l < layswtch)
        ind0_l = ((jp - 1) * 5 + (jt - 1)) + 1
        ind1_l = (jp * 5 + (jt1 - 1)) + 1
        n2o_jp = _at1(n2oref, jp); n2o_jp1 = _at1(n2oref, jp + 1)
        h2o_jp = _at1(h2oref, jp); h2o_jp1 = _at1(h2oref, jp + 1)
        ratio_l = (n2o_jp / h2o_jp) + fp * ((n2o_jp1 / h2o_jp1) - (n2o_jp / h2o_jp))
        n2omult_l = c.coln2o[idx] - c.colh2o[idx] * ratio_l
        tau_l = c.colh2o[idx] * (
            _interp4(absa, ind0_l, ind1_l, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
            + c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx]))
        tau_l = tau_l + c.co2mult[idx] * absco2a + n2omult_l * absn2oa
        # high
        ind0_h = ((jp - 7) * 5 + (jt - 1)) + 1
        ind1_h = ((jp - 6) * 5 + (jt1 - 1)) + 1
        o3_jp = _at1(o3ref, jp); o3_jp1 = _at1(o3ref, jp + 1)
        ratio_h = (n2o_jp / o3_jp) + fp * ((n2o_jp1 / o3_jp1) - (n2o_jp / o3_jp))
        n2omult_h = c.coln2o[idx] - c.colo3[idx] * ratio_h
        tau_h = c.colo3[idx] * _interp4(absb, ind0_h, ind1_h, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        tau_h = tau_h + c.co2mult[idx] * absco2b + n2omult_h * absn2ob
        low = c.swtch_mask[idx]
        tau = jnp.where(low, tau_l[:ng], tau_h[:ng])
        frac = jnp.where(low, fa0[:ng], fb0[:ng])
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _gb9(tab, c):
    band = 8; ng = int(NGC[band]); nspa = int(NSPA[band])
    absa = tab.absa[band]; absb = tab.absb[band]
    selfref = tab.selfref[band]
    fa = tab.fracrefa[band]; fa_cols = tab.fracrefa_cols[band]
    fb0 = _frac_col0(tab.fracrefb[band], tab.fracrefb_cols[band])  # 1-D
    h2oref = tab.local["TAUGB9_H2OREF"]; n2oref = tab.local["TAUGB9_N2OREF"]
    ch4ref = tab.local["TAUGB9_CH4REF"]; etaref = tab.local["TAUGB9_ETAREF"]
    absn2o = tab.minor["ABSN2OC9"]   # length 3*ng
    strrat = 21.6282

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        # low
        speccomb, js, fs = _binary_params(c.colh2o[idx], c.colch4[idx], strrat, 8.0)
        jfrac = js; ffrac = fs
        # js==8 sub-branches; js==9 reset
        def case8():
            j8 = jnp.where(fs <= 0.68, 8, jnp.where(fs <= 0.92, 9, 10))
            f8 = jnp.where(fs <= 0.68, fs / 0.68, jnp.where(fs <= 0.92, (fs - 0.68) / 0.24, (fs - 0.92) / 0.08))
            return j8, f8, js, fs
        def case9():
            return 10, 1.0, 8, 1.0
        def case_else():
            return js, fs, jfrac, ffrac
        is8 = js == 8; is9 = js == 9
        js2 = jnp.where(is8, case8()[0], jnp.where(is9, 10, js))
        fs2 = jnp.where(is8, case8()[1], jnp.where(is9, 1.0, fs))
        jfrac2 = jnp.where(is9, 8, jfrac)
        ffrac2 = jnp.where(is9, 1.0, ffrac)
        ns = js2 + _fint(fs2 + 0.5)
        # ioff: the host ``ioff`` is a PERSISTENT scalar over the layer loop -- it
        # is set to ng at the layer where ``l+1 == laylow`` and to 2*ng where
        # ``l+1 == layswtch`` and carries forward. With laylow <= layswtch this is,
        # for layer l (0-based): 2*ng if l+1 >= layswtch, else ng if l+1 >= laylow,
        # else 0. (The N2O minor block ABSN2OC9 has three ng-length sub-blocks.)
        layswtch = jnp.sum(c.swtch_mask.astype(jnp.int32))
        ioff = jnp.where(idx + 1 >= layswtch, 2 * ng,
                         jnp.where(idx + 1 >= c.laylow, ng, 0))
        fp = c.fac01[idx] + c.fac11[idx]
        eta = _at1(etaref, ns)
        wcomb1 = jnp.where(ns == 11, _at1(h2oref, jp), strrat * _at1(ch4ref, jp) / (1.0 - eta))
        wcomb2 = jnp.where(ns == 11, _at1(h2oref, jp + 1), strrat * _at1(ch4ref, jp + 1) / (1.0 - eta))
        n2o_jp = _at1(n2oref, jp); n2o_jp1 = _at1(n2oref, jp + 1)
        ratio = (n2o_jp / wcomb1) + fp * ((n2o_jp1 / wcomb2) - (n2o_jp / wcomb1))
        n2omult = c.coln2o[idx] - speccomb * ratio
        ind0_l = ((jp - 1) * 5 + (jt - 1)) * nspa + js2
        ind1_l = (jp * 5 + (jt1 - 1)) * nspa + js2
        tau_l = speccomb * _binary_lower(absa, ind0_l, ind1_l, nspa, fs2, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        tau_l = tau_l + c.colh2o[idx] * c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx])
        # n2omult * absn2o[ioff:ioff+ng] via dynamic_slice
        absn2o_seg = jax.lax.dynamic_slice(jnp.asarray(absn2o), (ioff,), (ng,))
        tau_l = tau_l + n2omult * absn2o_seg
        frac_l = _frac_interp_cols(fa, fa_cols, jfrac2, ffrac2)
        # high
        ind0_h = ((jp - 13) * 5 + (jt - 1)) + 1
        ind1_h = ((jp - 12) * 5 + (jt1 - 1)) + 1
        tau_h = c.colch4[idx] * _interp4(absb, ind0_h, ind1_h, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        frac_h = fb0
        low = c.low_mask[idx]
        tau = jnp.where(low, tau_l[:ng], tau_h[:ng])
        frac = jnp.where(low, frac_l[:ng], frac_h[:ng])
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _gb_lower_only(tab, c, band, a, b, strrat):
    """Bands 12, 13, 15, 16: lower-only binary, zero above troposphere."""

    ng = int(NGC[band]); nspa = int(NSPA[band])
    absa = tab.absa[band]; selfref = tab.selfref[band]
    fa = tab.fracrefa[band]; fa_cols = tab.fracrefa_cols[band]

    def one_layer(idx):
        jp = c.jp[idx]; jt = c.jt[idx]; jt1 = c.jt1[idx]
        speccomb, js, fs = _binary_params(a[idx], b[idx], strrat, 8.0)
        ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js
        ind1 = (jp * 5 + (jt1 - 1)) * nspa + js
        tau_l = speccomb * _binary_lower(absa, ind0, ind1, nspa, fs, c.fac00[idx], c.fac10[idx], c.fac01[idx], c.fac11[idx])
        tau_l = tau_l + c.colh2o[idx] * c.selffac[idx] * _self_term(selfref, c.indself[idx], c.selffrac[idx])
        frac_l = _frac_interp_cols(fa, fa_cols, js, fs)
        low = c.low_mask[idx]
        tau = jnp.where(low, tau_l[:ng], jnp.zeros(ng))
        frac = jnp.where(low, frac_l[:ng], jnp.zeros(ng))
        return tau, frac

    return jax.vmap(one_layer)(jnp.arange(c.colh2o.size))


def _apply_taugb_jax(tab, c):
    n = c.colh2o.size
    taug = jnp.zeros((n, NGPT), dtype=jnp.float64)
    pfrac = jnp.zeros((n, NGPT), dtype=jnp.float64)

    def place(taug, pfrac, band, tau_b, frac_b):
        start = int(NGB_START[band])
        taug = jax.lax.dynamic_update_slice(taug, tau_b, (0, start))
        pfrac = jax.lax.dynamic_update_slice(pfrac, frac_b, (0, start))
        return taug, pfrac

    t1, f1 = _gb_pure(tab, c, 0, c.colh2o, with_self=True, with_for=True)
    taug, pfrac = place(taug, pfrac, 0, t1, f1)
    t, f = _gb2(tab, c); taug, pfrac = place(taug, pfrac, 1, t, f)
    t, f = _gb3(tab, c); taug, pfrac = place(taug, pfrac, 2, t, f)
    t, f = _gb_simple_binary(tab, c, 3, c.colh2o, c.colco2, 850.577, c.colo3, c.colco2, 35.7416, "band4")
    taug, pfrac = place(taug, pfrac, 3, t, f)
    t, f = _gb_simple_binary(tab, c, 4, c.colh2o, c.colco2, 90.4894, c.colo3, c.colco2, 0.900502)
    taug, pfrac = place(taug, pfrac, 4, t, f)  # band5 wx (CCL4) term = 0
    t, f = _gb6(tab, c); taug, pfrac = place(taug, pfrac, 5, t, f)
    t, f = _gb7(tab, c); taug, pfrac = place(taug, pfrac, 6, t, f)
    t, f = _gb8(tab, c); taug, pfrac = place(taug, pfrac, 7, t, f)
    t, f = _gb9(tab, c); taug, pfrac = place(taug, pfrac, 8, t, f)
    t, f = _gb_pure(tab, c, 9, c.colh2o, with_self=False, with_for=False)
    taug, pfrac = place(taug, pfrac, 9, t, f)
    t, f = _gb_pure(tab, c, 10, c.colh2o, with_self=True, with_for=False)
    taug, pfrac = place(taug, pfrac, 10, t, f)
    t, f = _gb_lower_only(tab, c, 11, c.colh2o, c.colco2, 0.009736757); taug, pfrac = place(taug, pfrac, 11, t, f)
    t, f = _gb_lower_only(tab, c, 12, c.colh2o, c.coln2o, 16658.87); taug, pfrac = place(taug, pfrac, 12, t, f)
    t, f = _gb_pure(tab, c, 13, c.colco2, with_self=True, with_for=False)
    taug, pfrac = place(taug, pfrac, 13, t, f)
    t, f = _gb_lower_only(tab, c, 14, c.coln2o, c.colco2, 0.2883201); taug, pfrac = place(taug, pfrac, 14, t, f)
    t, f = _gb_lower_only(tab, c, 15, c.colh2o, c.colch4, 830.411); taug, pfrac = place(taug, pfrac, 15, t, f)
    return taug, pfrac


def _gasabs_jax(tab, c):
    taug, pfrac = _apply_taugb_jax(tab, c)
    odepth = SECANG * taug
    tff = jnp.where(odepth <= 0.0, 0.0, odepth / (tab.bpade + odepth))
    itr = jnp.clip(_fint(5.0e3 * tff + 0.5), 0, 5000)
    return taug, pfrac, itr


# --------------------------------------------------------------------------- #
# Planck + one-angle radiative transfer (RTRN), vectorised + lax.scan.
# --------------------------------------------------------------------------- #
def _planck_band(totplnk, temp):
    """(NBANDS,) Planck-weighted emission at temp (host ``_planck_value`` per band)."""

    tp = jnp.asarray(totplnk)
    idx = jnp.clip(_fint(temp - 159.0), 1, 180)
    frac = temp - _fint(temp).astype(jnp.float64)
    lo = tp[idx - 1]      # (NBANDS,)
    hi = tp[idx]
    return lo + frac * (hi - lo)


def _rtrn_jax(tab, atm, c, itr, pfrac):
    tavel = atm["tavel"]; pz = atm["pz"]; tz = atm["tz"]
    cldfrac = atm["cloudfrac"]; taucloud = atm["taucloud"]
    tbound = atm["tbound"]; semiss = atm["semiss"]
    n = tavel.size
    delwave = tab.delwave            # (NBANDS,)
    ngb = tab.ngb                    # (NGPT,) band index

    # Planck functions per band at bound / levels / layers.
    plankbnd = delwave * _planck_band(tab.totplnk, tbound)             # (NBANDS,)
    plnkemit = semiss * plankbnd
    plvl = jax.vmap(lambda T: delwave * _planck_band(tab.totplnk, T))(tz)   # (n+1, NBANDS)
    play = jax.vmap(lambda T: delwave * _planck_band(tab.totplnk, T))(tavel)  # (n, NBANDS)

    icldlyr = (cldfrac > 0.0)
    odcld = SECANG * taucloud
    abscld = 1.0 - jnp.exp(-odcld)
    efclfrac = abscld * cldfrac

    # per-g-point band map + emission
    semis = semiss[ngb]                                # (NGPT,)
    raduemit = pfrac[0, :] * plnkemit[ngb]             # (NGPT,)
    bglev0 = pfrac[n - 1, :] * plvl[n, ngb]            # (NGPT,) initial (top level)

    # transmission / planck-fraction per layer-gpoint via itr gather
    # (the transmission lookup tables are frozen NumPy constants -> jnp.asarray
    # folds them to XLA constants for the traced ``itr`` gather).
    tauf = jnp.asarray(tab.tf)[itr]                    # (n, NGPT)
    transv = jnp.asarray(tab.trans)[itr]
    abss = 1.0 - transv                                # (n, NGPT)
    tau_itr = jnp.asarray(tab.tau)[itr]                # (n, NGPT)
    bpade = tab.bpade
    play_g = play[:, ngb]                              # (n, NGPT)  band-mapped layer planck
    plvl_g = plvl[:, ngb]                              # (n+1, NGPT)

    # --- downward sweep (lev = n .. 1), recurrence over radld / radclrd ---
    # carry: radld (NGPT,), radclrd (NGPT,), bglev (NGPT,), iclddn (scalar bool)

    def down_step(carry, lev):
        radld, radclrd, bglev, iclddn = carry
        idx = lev - 1
        cloudy = icldlyr[idx]
        iclddn_new = jnp.logical_or(iclddn, cloudy)
        bglay = pfrac[idx, :] * play_g[idx, :]
        delbgup = bglev - bglay
        bbu_l = bglay + tauf[idx, :] * delbgup
        factot = (tau_itr[idx, :] + odcld[idx]) / (bpade + tau_itr[idx, :] + odcld[idx])
        bbutot_l = bglay + factot * delbgup
        bglev_new = pfrac[idx, :] * plvl_g[idx, :]
        delbgdn = bglev_new - bglay
        bbd = bglay + tauf[idx, :] * delbgdn
        a = abss[idx, :]
        bbdlevd = bglay + (tau_itr[idx, :] + odcld[idx]) / (bpade + tau_itr[idx, :] + odcld[idx]) * delbgdn
        atot_l = a + abscld[idx] - a * abscld[idx]
        gassrc = bbd * a
        radld_cloudy = radld - radld * (a + efclfrac[idx] * (1.0 - a)) + gassrc + cldfrac[idx] * (bbdlevd * atot_l - gassrc)
        radclrd_cloudy = radclrd + (bbd - radclrd) * a
        radld_clear = radld + (bbd - radld) * a
        # clear-air radclrd: if iclddn (already saw cloud below) accumulate, else mirror radld
        radclrd_clear = jnp.where(iclddn, radclrd + (bbd - radclrd) * a, radld_clear)
        radld_new = jnp.where(cloudy, radld_cloudy, radld_clear)
        radclrd_new = jnp.where(cloudy, radclrd_cloudy, radclrd_clear)
        drad = jnp.sum(radld_new)
        clrdrad = jnp.where(jnp.logical_or(cloudy, iclddn), jnp.sum(radclrd_new), drad)
        out = (drad * WTNUM, clrdrad * WTNUM, bbu_l, bbutot_l, a, atot_l)
        return (radld_new, radclrd_new, bglev_new, iclddn_new), out

    levs_down = jnp.arange(n, 0, -1)
    init_down = (jnp.zeros(NGPT), jnp.zeros(NGPT), bglev0, jnp.array(False))
    (radld_f, radclrd_f, _, _), down_out = jax.lax.scan(down_step, init_down, levs_down)
    # The clear-sky accumulators (radclrd/radclru, clrdrad/clrurad) are threaded for
    # WRF fidelity but feed only the clear-sky diagnostics (htrc/fnetc) that this
    # endpoint does not emit; only the all-sky fluxes drive heating/GLW/OLR.
    drad_seq, _clrdrad_seq, bbu_seq, bbutot_seq, abss_seq, atot_seq = down_out
    # down_out is ordered lev=n..1 -> totdflux[lev-1]; reverse to model order [0..n-1]
    totdflux_lower = drad_seq[::-1]                    # totdflux[0..n-1]
    # bbu/bbutot/abss/atot recorded at idx=lev-1, reverse to model order
    bbu = bbu_seq[::-1]                                # (n, NGPT)
    bbutot = bbutot_seq[::-1]
    abss_m = abss_seq[::-1]
    atot_m = atot_seq[::-1]

    # --- surface reflection ---
    radlu0 = raduemit + (1.0 - semis) * radld_f
    radclru0 = raduemit + (1.0 - semis) * radclrd_f
    urad0 = jnp.sum(radlu0) * WTNUM

    # --- upward sweep (lev = 1 .. n) ---
    def up_step(carry, idx):
        radlu, radclru = carry
        a = abss_m[idx, :]
        cloudy = icldlyr[idx]
        gassrc = bbu[idx, :] * a
        radlu_cloudy = radlu - radlu * (a + efclfrac[idx] * (1.0 - a)) + gassrc + cldfrac[idx] * (bbutot[idx, :] * atot_m[idx, :] - gassrc)
        radclru_cloudy = radclru + (bbu[idx, :] - radclru) * a
        radlu_clear = radlu + (bbu[idx, :] - radlu) * a
        radclru_clear = radclru + (bbu[idx, :] - radclru) * a
        radlu_new = jnp.where(cloudy, radlu_cloudy, radlu_clear)
        radclru_new = jnp.where(cloudy, radclru_cloudy, radclru_clear)
        urad = jnp.sum(radlu_new) * WTNUM
        clrurad = jnp.sum(radclru_new) * WTNUM
        return (radlu_new, radclru_new), (urad, clrurad)

    (_, _), up_out = jax.lax.scan(up_step, (radlu0, radclru0), jnp.arange(n))
    urad_seq, _clrurad_seq = up_out                    # totuflux[1..n] (clear-sky unused)

    totuflux = jnp.concatenate([jnp.array([urad0]), urad_seq])       # (n+1,)
    totdflux = jnp.concatenate([totdflux_lower, jnp.array([0.0])])   # (n+1,) totdflux[n]=0
    totuflux = totuflux * tab.fluxfac
    totdflux = totdflux * tab.fluxfac

    fnet = totuflux - totdflux
    htr = tab.heatfac * (fnet[:-1] - fnet[1:]) / (pz[:-1] - pz[1:])
    return htr / 86400.0, totdflux[0], totuflux  # heating (K/s, all nlayers), GLW, up-flux


def _solve_one_jax(state_col, tab, nz, ptop_pa: float | None = None):
    atm = _prepare_atmosphere_jax(state_col, tab, nz, ptop_pa)
    c = _setcoef_jax(atm, tab)
    _, pfrac, itr = _gasabs_jax(tab, c)
    htr, glw, totuflux = _rtrn_jax(tab, atm, c, itr, pfrac)
    # OLR is the up-flux at the MODEL top (host ``totuflux[model_layers]`` with
    # ``model_layers = nz``), NOT the buffer top.
    olr = totuflux[nz]
    return htr[:nz], glw, olr


def solve_rrtm_lw_column_jax(state: RRTMLWColumnState) -> RRTMLWColumnResult:
    """Traceable JAX classic-RRTM LW column endpoint (``ra_lw_physics=1``).

    Numerically reproduces :func:`gpuwrf.physics.ra_lw_rrtm.solve_rrtm_lw_column`
    (itself pristine-WRF parity-proven) to fp64 round-off. JIT/vmap-traceable; no
    host callbacks. Inputs/outputs match the host kernel's contract: ``(ncol, nz)``
    columns, ``t8w``/``p8w`` have ``nz+1`` interface entries.
    """

    tab = _jax_tables()
    nz = int(state.T.shape[1])
    # F1: size the above-model-top buffer from the grid's REAL model-top pressure
    # (static float; controls array shapes), not the hardcoded DEFAULT_PTOP_PA.
    # ``None`` -> legacy 5000 Pa, keeping existing callers bit-identical.
    ptop_pa = state.top_pressure_pa

    cols = dict(
        T=jnp.asarray(state.T, dtype=jnp.float64),
        t8w=jnp.asarray(state.t8w, dtype=jnp.float64),
        p=jnp.asarray(state.p, dtype=jnp.float64),
        p8w=jnp.asarray(state.p8w, dtype=jnp.float64),
        qv=jnp.asarray(state.qv, dtype=jnp.float64),
        qc=jnp.asarray(state.qc, dtype=jnp.float64),
        qr=jnp.asarray(state.qr, dtype=jnp.float64),
        qi=jnp.asarray(state.qi, dtype=jnp.float64),
        qs=jnp.asarray(state.qs, dtype=jnp.float64),
        qg=jnp.asarray(state.qg, dtype=jnp.float64),
        cloud_fraction=jnp.asarray(state.cloud_fraction, dtype=jnp.float64),
        dz=jnp.asarray(state.dz, dtype=jnp.float64),
    )
    ncol = cols["T"].shape[0]
    # emiss/tsk may be a per-column array (operational coupler) or a scalar (proof
    # kernel calls). ``jnp.broadcast_to(.reshape(-1), (ncol,))`` handles both and is
    # jit/trace-safe -- a 0-D scalar reshapes to (1,) and broadcasts to (ncol,).
    emiss = jnp.broadcast_to(jnp.asarray(state.emiss, dtype=jnp.float64).reshape(-1), (ncol,))
    tsk = jnp.broadcast_to(jnp.asarray(state.tsk, dtype=jnp.float64).reshape(-1), (ncol,))

    def per_col(col_idx):
        sc = {k: v[col_idx] for k, v in cols.items()}
        sc["emiss"] = emiss[col_idx]
        sc["tsk"] = tsk[col_idx]
        return _solve_one_jax(sc, tab, nz, ptop_pa)

    htr, glw, olr = jax.vmap(per_col)(jnp.arange(ncol))
    return RRTMLWColumnResult(
        heating_rate=htr,
        glw=glw,
        olr=olr,
        flux_down=jnp.zeros_like(htr),   # not consumed by the operational coupler
        flux_up=jnp.zeros_like(htr),
    )


__all__ = ["solve_rrtm_lw_column_jax"]
