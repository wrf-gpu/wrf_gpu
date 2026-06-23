"""GPU-batched (jit/vmap-traceable) WRF modified-Tiedtke cumulus column kernel.

This is the operationalization of the faithful single-column reference in
:mod:`gpuwrf.physics.cumulus_tiedtke` (which itself is a line-faithful NumPy
transcription of pristine ``phys/module_cu_tiedtke.F``). The numerical algorithm
is preserved exactly -- this module only replaces Python control flow / in-place
NumPy mutation with ``jax.lax`` primitives (``fori_loop`` over the fixed KLEV
vertical sweeps, ``cond`` / ``jnp.where`` masking of the data-dependent Fortran
branches) and functional ``.at[].set(...)`` updates, so the whole column traces
to one XLA graph and ``vmap``s across the ``(ny*nx)`` grid columns with ZERO
host/device transfer inside the call.

Design (mirrors the KF GPU port, ``cumulus_kf.py``):

* ``_cumastr_new_jax`` is the WRF ``CUMASTRN`` master routine, with the inner
  ``CUINI`` / ``CUBASE`` / ``CUASC`` / ``CUDLFS`` / ``CUDDRAF`` / ``CUFLX`` /
  ``CUDTDQ`` / ``CUDUDV`` subroutines as pure-JAX helpers. Every level sweep is a
  fixed-trip ``fori_loop`` (trip count = KLEV); every Fortran ``IF(loflag)`` /
  ``CYCLE`` becomes a ``jnp.where`` mask, so no branch of the source is dropped.
* The internal indexing keeps WRF's top-down convention (index 1 = model top,
  index KLEV = surface) with 1-based arrays of physical length ``KLEV+2`` (index
  0 unused, index ``KLEV+1`` = surface interface), exactly as the NumPy reference,
  so the arithmetic is identical level-for-level.
* fp64 throughout (operational release precision); matches the Fortran REAL*4
  oracle to within the predeclared, physically meaningful savepoint tolerance.

The column entry / WRF-orientation boundary (bottom-up WRF/JAX columns <-> the
top-down internal arrays) and the tendency definitions are identical to
``cumulus_tiedtke.tiedtke_column``.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp

configure_jax_x64()

# --- WRF model + Tiedtke constants (verbatim from cumulus_tiedtke.py) ---------
RD = 287.0
RV = 461.6
CPD = 7.0 * RD / 2.0
ALV = 2.5e6
ALS = 2.85e6
ALF = 3.5e5
G = 9.81
RCPD = 1.0 / CPD
VTMP_C1 = RV / RD - 1.0
T000 = 273.15
TMELT = 273.16
HGFR = 233.15
ZRG = 1.0 / G
C1ES = 610.78
C2ES = C1ES * RD / RV
C3LES = 17.269
C3IES = 21.875
C4LES = 35.86
C4IES = 7.66
C5LES = C3LES * (TMELT - C4LES)
C5IES = C3IES * (TMELT - C4IES)

ENTRPEN = 1.0e-4
ENTRSCV = 1.2e-3
ENTRMID = 1.0e-4
ENTRDD = 2.0e-4
CMFCTOP = 0.30
CMFCMAX = 1.0
CMFCMIN = 1.0e-10
CMFDEPS = 0.30
CPRCON = 1.1e-3 / G
ZDNOPRC = 1.5e4
RHC = 0.80
RHM = 1.0
ZBUO0 = 0.50
CRIRH = 0.70
FDBK = 1.0
ZTAU = 2400.0
CEVAPCU1 = 1.93e-6 * 261.0 * 0.5 / G
CEVAPCU2 = 1.0e3 / (38.3 * 0.293)

# Logical switches (all True in the WRF default modified-Tiedtke config).
LMFPEN = True
LMFMID = True
LMFSCV = True
LMFDD = True
LMFDUDV = True


# ----------------------------------------------------------------------------
# Saturation lookup helpers (pointwise, scalar-or-array, branch-free via where)
# ----------------------------------------------------------------------------
def _tlucua(tt):
    warm = tt - TMELT > 0.0
    zcvm3 = jnp.where(warm, C3LES, C3IES)
    zcvm4 = jnp.where(warm, C4LES, C4IES)
    return C2ES * jnp.exp(zcvm3 * (tt - TMELT) / (tt - zcvm4))


def _tlucub(tt):
    z5alvcp = C5LES * ALV / CPD
    z5alscp = C5IES * ALS / CPD
    warm = tt - TMELT > 0.0
    zcvm4 = jnp.where(warm, C4LES, C4IES)
    zcvm5 = jnp.where(warm, z5alvcp, z5alscp)
    return zcvm5 * (1.0 / (tt - zcvm4)) ** 2


def _tlucuc(tt):
    return jnp.where(tt - TMELT > 0.0, ALV / CPD, ALS / CPD)


def _cuadjtq_level(pp, t_kk, q_kk, ldflag, kcall):
    """Functional WRF ``cuadjtq`` for ONE level. Returns (t_kk, q_kk).

    ``ldflag``/``kcall`` are traced; the Fortran early-RETURN for kcall in {1,2}
    with ``not ldflag`` is reproduced by masking the update with ``do_adjust``.
    All four kcall branches (1=positive-limited, 2=negative-limited, 0/4=unlimited)
    are computed and selected with ``jnp.where`` -- identical arithmetic to the
    NumPy reference, no branch dropped.
    """

    def _adjust_once(t_in, q_in, limit):
        # limit: 0 none, 1 positive (max 0), -1 negative (min 0)
        tt = t_in
        zqp = 1.0 / pp
        zqsat0 = jnp.minimum(0.5, _tlucua(tt) * zqp)
        zcor = 1.0 / (1.0 - VTMP_C1 * zqsat0)
        zqsat = zqsat0 * zcor
        zcond = (q_in - zqsat) / (1.0 + zqsat * zcor * _tlucub(tt))
        zcond = jnp.where(limit == 1, jnp.maximum(zcond, 0.0), zcond)
        zcond = jnp.where(limit == -1, jnp.minimum(zcond, 0.0), zcond)
        t_out = t_in + _tlucuc(tt) * zcond
        q_out = q_in - zcond
        return t_out, q_out, zcond

    # do we run the adjustment at all?
    skip = ((kcall == 1) | (kcall == 2)) & (~ldflag)
    do_adjust = ~skip

    # first pass limit by kcall
    limit1 = jnp.where(kcall == 1, 1, jnp.where(kcall == 2, -1, 0))
    t1, q1, zcond1 = _adjust_once(t_kk, q_kk, limit1)
    # second pass (unlimited) conditioned on (zcond1 != 0) or kcall == 4
    do_second = (zcond1 != 0.0) | (kcall == 4)
    t2, q2, _ = _adjust_once(t1, q1, 0)
    t_after = jnp.where(do_second, t2, t1)
    q_after = jnp.where(do_second, q2, q1)

    t_out = jnp.where(do_adjust, t_after, t_kk)
    q_out = jnp.where(do_adjust, q_after, q_kk)
    return t_out, q_out


# ----------------------------------------------------------------------------
# Index helpers: 1-based arrays of physical length klev+2.
# ----------------------------------------------------------------------------
def _zeros(klev):
    return jnp.zeros(klev + 2, dtype=jnp.float64)


def _set(arr, k, val):
    return arr.at[k].set(val)


def _get(arr, k):
    return arr[k]


# ----------------------------------------------------------------------------
# CUINI -- initialize half-level env + draft seed values.
# ----------------------------------------------------------------------------
def _cuini_jax(klev, pten, pqen, pqsen, puen, pven, pverv, pgeo, paph):
    klevm1 = klev - 1
    pgeoh = _zeros(klev)
    ptenh = _zeros(klev)
    pqenh = _zeros(klev)
    pqsenh = _zeros(klev)

    def body1(jk, carry):
        pgeoh, ptenh, pqenh, pqsenh = carry
        pgeoh = pgeoh.at[jk].set(pgeo[jk] + (pgeo[jk - 1] - pgeo[jk]) * 0.5)
        ptenh = ptenh.at[jk].set(
            (jnp.maximum(CPD * pten[jk - 1] + pgeo[jk - 1], CPD * pten[jk] + pgeo[jk]) - pgeoh[jk]) * RCPD
        )
        pqsenh = pqsenh.at[jk].set(pqsen[jk - 1])
        t_adj, q_adj = _cuadjtq_level(paph[jk], ptenh[jk], pqsenh[jk], True, jnp.int32(0))
        ptenh = ptenh.at[jk].set(t_adj)
        pqsenh = pqsenh.at[jk].set(q_adj)
        val = jnp.minimum(pqen[jk - 1], pqsen[jk - 1]) + (pqsenh[jk] - pqsen[jk - 1])
        pqenh = pqenh.at[jk].set(jnp.maximum(val, 0.0))
        return (pgeoh, ptenh, pqenh, pqsenh)

    pgeoh, ptenh, pqenh, pqsenh = jax.lax.fori_loop(2, klev + 1, body1, (pgeoh, ptenh, pqenh, pqsenh))

    ptenh = ptenh.at[klev].set((CPD * pten[klev] + pgeo[klev] - pgeoh[klev]) * RCPD)
    pqenh = pqenh.at[klev].set(pqen[klev])
    ptenh = ptenh.at[1].set(pten[1])
    pqenh = pqenh.at[1].set(pqen[1])
    pgeoh = pgeoh.at[1].set(pgeo[1])

    # smoothing sweep jk = klevm1 .. 2 (descending)
    def body2(idx, ptenh):
        jk = klevm1 - idx  # idx 0 -> klevm1 ; runs while jk >= 2
        valid = jk >= 2
        zzs = jnp.maximum(CPD * ptenh[jk] + pgeoh[jk], CPD * ptenh[jk + 1] + pgeoh[jk + 1])
        new = (zzs - pgeoh[jk]) * RCPD
        return ptenh.at[jk].set(jnp.where(valid, new, ptenh[jk]))

    ptenh = jax.lax.fori_loop(0, klev, body2, ptenh)

    # klwmin: deepest level with most negative pverv, scanning klev..2.
    def body3(idx, carry):
        klwmin, zwmax = carry
        jk = klev - idx
        valid = jk >= 2
        hit = valid & (pverv[jk] < zwmax)
        klwmin = jnp.where(hit, jk, klwmin)
        zwmax = jnp.where(hit, pverv[jk], zwmax)
        return (klwmin, zwmax)

    klwmin, _ = jax.lax.fori_loop(0, klev, body3, (jnp.int32(klev), jnp.float64(0.0)))

    # seed updraft/downdraft arrays
    ptu = _zeros(klev)
    pqu = _zeros(klev)
    ptd = _zeros(klev)
    pqd = _zeros(klev)
    puu = _zeros(klev)
    pvu = _zeros(klev)
    pud = _zeros(klev)
    pvd = _zeros(klev)

    def body4(jk, carry):
        ptu, pqu, ptd, pqd, puu, pvu, pud, pvd = carry
        ik = jnp.where(jk == 1, 1, jk - 1)
        ptu = ptu.at[jk].set(ptenh[jk])
        ptd = ptd.at[jk].set(ptenh[jk])
        pqu = pqu.at[jk].set(pqenh[jk])
        pqd = pqd.at[jk].set(pqenh[jk])
        puu = puu.at[jk].set(puen[ik])
        pud = pud.at[jk].set(puen[ik])
        pvu = pvu.at[jk].set(pven[ik])
        pvd = pvd.at[jk].set(pven[ik])
        return (ptu, pqu, ptd, pqd, puu, pvu, pud, pvd)

    ptu, pqu, ptd, pqd, puu, pvu, pud, pvd = jax.lax.fori_loop(
        1, klev + 1, body4, (ptu, pqu, ptd, pqd, puu, pvu, pud, pvd)
    )

    z = _zeros(klev)
    return {
        "pgeoh": pgeoh, "ptenh": ptenh, "pqenh": pqenh, "pqsenh": pqsenh,
        "ptu": ptu, "pqu": pqu, "ptd": ptd, "pqd": pqd,
        "puu": puu, "pvu": pvu, "pud": pud, "pvd": pvd,
        "pmfu": z, "pmfd": z, "pmfus": z, "pmfds": z, "pmfuq": z, "pmfdq": z,
        "pmful": z, "pdmfup": z, "pdmfdp": z, "pdpmel": z, "plu": z, "plude": z,
        "klab": jnp.zeros(klev + 2, dtype=jnp.int32), "klwmin": klwmin,
    }


# ----------------------------------------------------------------------------
# CUBASE -- cloud base search (ascending-physical = descending index sweep).
# ----------------------------------------------------------------------------
def _cubase_jax(klev, ptenh, pqenh, pgeoh, paph, ptu, pqu, plu, puen, pven, puu, pvu):
    klevm1 = klev - 1
    klab = jnp.zeros(klev + 2, dtype=jnp.int32)
    klab = klab.at[klev].set(1)
    puu = puu.at[klev].set(puen[klev] * (paph[klev + 1] - paph[klev]))
    pvu = pvu.at[klev].set(pven[klev] * (paph[klev + 1] - paph[klev]))

    # sweep jk = klevm1 .. 2 descending
    def body(idx, carry):
        ptu, pqu, plu, puu, pvu, klab, kcbot, ldcum, ldbase = carry
        jk = klevm1 - idx
        valid = jk >= 2
        loflag = (klab[jk + 1] == 1) | (ldcum & (kcbot == jk + 1))
        active = valid & loflag

        # LMFDUDV momentum accumulation while not ldbase
        do_mom = active & LMFDUDV & (~ldbase)
        puu = puu.at[klev].set(jnp.where(do_mom, puu[klev] + puen[jk] * (paph[jk + 1] - paph[jk]), puu[klev]))
        pvu = pvu.at[klev].set(jnp.where(do_mom, pvu[klev] + pven[jk] * (paph[jk + 1] - paph[jk]), pvu[klev]))

        pqu_jk_pre = pqu[jk + 1]
        ptu_jk = (CPD * ptu[jk + 1] + pgeoh[jk + 1] - pgeoh[jk]) * RCPD
        pqu = pqu.at[jk].set(jnp.where(active, pqu_jk_pre, pqu[jk]))
        ptu = ptu.at[jk].set(jnp.where(active, ptu_jk, ptu[jk]))
        zqold = pqu[jk]
        t_adj, q_adj = _cuadjtq_level(paph[jk], ptu[jk], pqu[jk], True, jnp.int32(1))
        ptu = ptu.at[jk].set(jnp.where(active, t_adj, ptu[jk]))
        pqu = pqu.at[jk].set(jnp.where(active, q_adj, pqu[jk]))

        no_cond = pqu[jk] == zqold
        zbuo = ptu[jk] * (1.0 + VTMP_C1 * pqu[jk]) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk]) + ZBUO0
        # branch A: no condensation
        klabA = jnp.where(active & no_cond & (zbuo > 0.0), 1, klab[jk])
        # branch B: condensation
        do_B = active & (~no_cond)
        plu_jk = plu[jk] + (zqold - pqu[jk])
        plu = plu.at[jk].set(jnp.where(do_B, plu_jk, plu[jk]))
        klabB = jnp.where(do_B, 2, klab[jk])
        new_base = do_B & (zbuo > 0.0) & (klab[jk + 1] == 1)
        klab_jk = jnp.where(active & no_cond, klabA, jnp.where(do_B, klabB, klab[jk]))
        klab = klab.at[jk].set(klab_jk)
        kcbot = jnp.where(new_base, jk, kcbot)
        ldcum = ldcum | new_base
        ldbase = ldbase | new_base
        return (ptu, pqu, plu, puu, pvu, klab, kcbot, ldcum, ldbase)

    init = (ptu, pqu, plu, puu, pvu, klab, jnp.int32(klevm1), jnp.bool_(False), jnp.bool_(False))
    ptu, pqu, plu, puu, pvu, klab, kcbot, ldcum, ldbase = jax.lax.fori_loop(0, klev, body, init)

    if LMFDUDV:
        zz = 1.0 / (paph[klev + 1] - paph[kcbot])
        puu_cum = puu[klev] * zz
        pvu_cum = pvu[klev] * zz
        puu = puu.at[klev].set(jnp.where(ldcum, puu_cum, puen[klevm1]))
        pvu = pvu.at[klev].set(jnp.where(ldcum, pvu_cum, pven[klevm1]))
    return ldcum, kcbot, klab, ptu, pqu, plu, puu, pvu


# ----------------------------------------------------------------------------
# CUBASMC -- mid-level convection base (called per-level inside CUASC).
# Returns updated carry tuple of the affected fields.
# ----------------------------------------------------------------------------
def _cubasmc_jax(klev, kk, pten, pqen, pqsen, puen, pven, pverv, pgeo, pgeoh,
                 ldcum, ktype, klab, pmfu, pmfub, pentr, kcbot, ptu, pqu, plu,
                 puu, pvu, pmfus, pmfuq, pmful, pdmfup, pmfuu, pmfuv):
    trigger = (~ldcum) & (klab[kk + 1] == 0) & (pqen[kk] > 0.80 * pqsen[kk])

    ptu_new = (CPD * pten[kk] + pgeo[kk] - pgeoh[kk + 1]) * RCPD
    pqu_new = pqen[kk]
    zzzmb = jnp.minimum(jnp.maximum(CMFCMIN, -pverv[kk] / G), CMFCMAX)
    pmfub_new = zzzmb

    ptu = ptu.at[kk + 1].set(jnp.where(trigger, ptu_new, ptu[kk + 1]))
    pqu = pqu.at[kk + 1].set(jnp.where(trigger, pqu_new, pqu[kk + 1]))
    plu = plu.at[kk + 1].set(jnp.where(trigger, 0.0, plu[kk + 1]))
    pmfu = pmfu.at[kk + 1].set(jnp.where(trigger, pmfub_new, pmfu[kk + 1]))
    pmfus = pmfus.at[kk + 1].set(jnp.where(trigger, pmfub_new * (CPD * ptu[kk + 1] + pgeoh[kk + 1]), pmfus[kk + 1]))
    pmfuq = pmfuq.at[kk + 1].set(jnp.where(trigger, pmfub_new * pqu[kk + 1], pmfuq[kk + 1]))
    pmful = pmful.at[kk + 1].set(jnp.where(trigger, 0.0, pmful[kk + 1]))
    pdmfup = pdmfup.at[kk + 1].set(jnp.where(trigger, 0.0, pdmfup[kk + 1]))
    kcbot = jnp.where(trigger, kk, kcbot)
    klab = klab.at[kk + 1].set(jnp.where(trigger, 1, klab[kk + 1]))
    ktype = jnp.where(trigger, 3, ktype)
    pentr = jnp.where(trigger, ENTRMID, pentr)
    pmfub = jnp.where(trigger, pmfub_new, pmfub)

    do_mom = trigger & LMFDUDV
    puu = puu.at[kk + 1].set(jnp.where(do_mom, puen[kk], puu[kk + 1]))
    pvu = pvu.at[kk + 1].set(jnp.where(do_mom, pven[kk], pvu[kk + 1]))
    pmfuu = jnp.where(do_mom, pmfub * puu[kk + 1], pmfuu)
    pmfuv = jnp.where(do_mom, pmfub * pvu[kk + 1], pmfuv)
    return ldcum, ktype, pmfub, pentr, kcbot, pmfuu, pmfuv, klab, ptu, pqu, plu, pmfu, pmfus, pmfuq, pmful, pdmfup, puu, pvu


# ----------------------------------------------------------------------------
# CUENTR -- entrainment / detrainment for one level kk (returns scalars).
# ----------------------------------------------------------------------------
def _cuentr_jax(klev, kk, ptenh, paph, pap, pgeoh, klwmin, ldcum, ktype,
                kcbot, kctop0, pmfu, pentr, zodetr, khmin):
    zpbase = paph[kcbot]
    zrrho = (RD * ptenh[kk + 1]) / paph[kk + 1]
    zdprho = (paph[kk + 1] - paph[kk]) * ZRG
    zpmid = 0.5 * (zpbase + paph[kctop0])
    zentr = pentr * pmfu[kk + 1] * zdprho * zrrho
    llo1 = (kk < kcbot) & ldcum
    zdmfde = jnp.where(llo1, zentr, 0.0)

    cond2 = llo1 & (ktype == 2) & (((zpbase - paph[kk]) < ZDNOPRC) | (paph[kk] > zpmid))
    iklwmin = jnp.maximum(klwmin, kctop0 + 2)
    cond3 = llo1 & (ktype == 3) & ((kk >= iklwmin) | (pap[kk] > zpmid))
    cond1 = llo1 & (ktype == 1)
    zdmfen = jnp.where(cond2 | cond3 | cond1, zentr, 0.0)

    # organized detrainment (ktype 1)
    zodetr_kk = 0.0
    do_org = llo1 & (ktype == 1) & (kk <= khmin) & (kk >= kctop0)
    ikt = kctop0
    ikh = khmin
    ikh_ok = ikh > ikt
    zzmzk = -(pgeoh[ikh] - pgeoh[kk]) * ZRG
    ztmzk = -(pgeoh[ikh] - pgeoh[ikt]) * ZRG
    # guard ztmzk against zero (only used when do_org & ikh_ok)
    ztmzk_safe = jnp.where(ztmzk == 0.0, 1.0, ztmzk)
    arg = 3.1415 * (zzmzk / ztmzk_safe) * 0.5
    zorgde = jnp.tan(arg) * 3.1415 * 0.5 / ztmzk_safe
    zdprho2 = (paph[kk + 1] - paph[kk]) * (ZRG * zrrho)
    org_val = jnp.minimum(zorgde, 1.0e-3) * pmfu[kk + 1] * zdprho2
    zodetr_kk = jnp.where(do_org & ikh_ok, org_val, 0.0)
    zodetr = zodetr.at[kk].set(zodetr_kk)
    return zpbase, zdmfen, zdmfde, zodetr


# ----------------------------------------------------------------------------
# CUASC -- cloud ascent (the central, most-branching routine).
# ----------------------------------------------------------------------------
def _cuasc_jax(klev, ptenh, pqenh, puen, pven, pten, pqen, pqsen, pgeo, pgeoh,
               pap, paph, pqte, pverv, klwmin, ldcum_in, phcbase, ktype_in, klab_in,
               ptu, pqu, plu, puu, pvu, pmfu, pmfub_in, pentr_in, pmfus, pmfuq,
               pmful, plude, pdmfup, kcbot_in, kctop_in, kctop0_in, ztmst, khmin,
               phhatt, pqsenh):
    klevm1 = klev - 1
    zcons2 = 1.0 / (G * ztmst)
    ktype = jnp.where(ldcum_in, ktype_in, jnp.int32(0))
    ldcum = ldcum_in
    klab = klab_in
    kcbot = kcbot_in
    kctop = kctop_in
    kctop0 = kctop0_in
    pmfub = pmfub_in
    pentr = pentr_in
    zodetr = _zeros(klev)
    zoentr = _zeros(klev)

    # initialization sweep jk = 1..klev
    def init_body(jk, carry):
        plu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup, zodetr, zoentr, klab, kctop0 = carry
        plu = plu.at[jk].set(0.0)
        pmfu = pmfu.at[jk].set(0.0)
        pmfus = pmfus.at[jk].set(0.0)
        pmfuq = pmfuq.at[jk].set(0.0)
        pmful = pmful.at[jk].set(0.0)
        plude = plude.at[jk].set(0.0)
        pdmfup = pdmfup.at[jk].set(0.0)
        zodetr = zodetr.at[jk].set(0.0)
        zoentr = zoentr.at[jk].set(0.0)
        reset_lab = (~ldcum) | (ktype == 3)
        klab = klab.at[jk].set(jnp.where(reset_lab, 0, klab[jk]))
        set_top = (~ldcum) & (paph[jk] < 4.0e4)
        kctop0 = jnp.where(set_top, jk, kctop0)
        return (plu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup, zodetr, zoentr, klab, kctop0)

    (plu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup, zodetr, zoentr, klab, kctop0) = jax.lax.fori_loop(
        1, klev + 1, init_body, (plu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup, zodetr, zoentr, klab, kctop0)
    )

    kctop = jnp.int32(klevm1)
    kcbot = jnp.where(~ldcum, jnp.int32(klevm1), kcbot)
    pmfub = jnp.where(~ldcum, 0.0, pmfub)
    pqu = pqu.at[klev].set(jnp.where(~ldcum, 0.0, pqu[klev]))

    pmfu = pmfu.at[klev].set(pmfub)
    pmfus = pmfus.at[klev].set(pmfub * (CPD * ptu[klev] + pgeoh[klev]))
    pmfuq = pmfuq.at[klev].set(pmfub * pqu[klev])
    zmfuu = jnp.where(LMFDUDV, pmfub * puu[klev], 0.0)
    zmfuv = jnp.where(LMFDUDV, pmfub * pvu[klev], 0.0)

    # ktype 1 zoentr seed at kcbot
    ikb = kcbot
    zbuoy0 = G * ((ptu[ikb] - ptenh[ikb]) / ptenh[ikb] + 0.608 * (pqu[ikb] - pqenh[ikb]))
    do_seed = (ktype == 1) & (zbuoy0 > 0.0) & (ikb > 1)
    zdz0 = (pgeo[ikb - 1] - pgeo[ikb]) * ZRG
    zdz0_safe = jnp.where(zdz0 == 0.0, 1.0, zdz0)
    zdrodz0 = -jnp.log(pten[ikb - 1] / pten[ikb]) / zdz0_safe - G / (RD * ptenh[ikb])
    seed_val = zbuoy0 * 0.5 / (1.0 + zbuoy0 * zdz0) + zdrodz0
    seed_val = jnp.minimum(jnp.maximum(seed_val, 0.0), 1.0e-3)
    zoentr = zoentr.at[ikb - 1].set(jnp.where(do_seed, seed_val, zoentr[ikb - 1]))

    zbuoy_acc = jnp.float64(0.0)

    # main ascent sweep jk = klevm1 .. 2 descending
    def asc_body(idx, carry):
        (ptu, pqu, plu, puu, pvu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup,
         klab, zodetr, zoentr, kcbot, kctop, kctop0, ldcum, ktype, pmfub, pentr,
         zmfuu, zmfuv, zbuoy_acc) = carry
        jk = klevm1 - idx
        valid = jk >= 2

        # CUBASMC (mid level)
        do_basmc = valid & LMFMID & (jk < klevm1) & (jk > klev - 13)
        (ldcum_b, ktype_b, pmfub_b, pentr_b, kcbot_b, zmfuu_b, zmfuv_b, klab_b,
         ptu_b, pqu_b, plu_b, pmfu_b, pmfus_b, pmfuq_b, pmful_b, pdmfup_b, puu_b, pvu_b) = _cubasmc_jax(
            klev, jk, pten, pqen, pqsen, puen, pven, pverv, pgeo, pgeoh,
            ldcum, ktype, klab, pmfu, pmfub, pentr, kcbot, ptu, pqu, plu,
            puu, pvu, pmfus, pmfuq, pmful, pdmfup, zmfuu, zmfuv)
        ldcum = jnp.where(do_basmc, ldcum_b, ldcum)
        ktype = jnp.where(do_basmc, ktype_b, ktype)
        pmfub = jnp.where(do_basmc, pmfub_b, pmfub)
        pentr = jnp.where(do_basmc, pentr_b, pentr)
        kcbot = jnp.where(do_basmc, kcbot_b, kcbot)
        zmfuu = jnp.where(do_basmc, zmfuu_b, zmfuu)
        zmfuv = jnp.where(do_basmc, zmfuv_b, zmfuv)
        klab = jnp.where(do_basmc, klab_b, klab)
        ptu = jnp.where(do_basmc, ptu_b, ptu)
        pqu = jnp.where(do_basmc, pqu_b, pqu)
        plu = jnp.where(do_basmc, plu_b, plu)
        pmfu = jnp.where(do_basmc, pmfu_b, pmfu)
        pmfus = jnp.where(do_basmc, pmfus_b, pmfus)
        pmfuq = jnp.where(do_basmc, pmfuq_b, pmfuq)
        pmful = jnp.where(do_basmc, pmful_b, pmful)
        pdmfup = jnp.where(do_basmc, pdmfup_b, pdmfup)
        puu = jnp.where(do_basmc, puu_b, puu)
        pvu = jnp.where(do_basmc, pvu_b, pvu)

        isum = klab[jk + 1]
        klab = klab.at[jk].set(jnp.where(valid & (klab[jk + 1] == 0), 0, klab[jk]))
        loflag = klab[jk + 1] > 0

        # ktype3 cap at kcbot
        do_cap = valid & (ktype == 3) & (jk == kcbot)
        zmfmax_cap = (paph[jk] - paph[jk - 1]) * zcons2
        cap_active = do_cap & (pmfub > zmfmax_cap) & (pmfub != 0.0)
        zfac = jnp.where(pmfub != 0.0, zmfmax_cap / pmfub, 1.0)
        pmfu = pmfu.at[jk + 1].set(jnp.where(cap_active, pmfu[jk + 1] * zfac, pmfu[jk + 1]))
        pmfus = pmfus.at[jk + 1].set(jnp.where(cap_active, pmfus[jk + 1] * zfac, pmfus[jk + 1]))
        pmfuq = pmfuq.at[jk + 1].set(jnp.where(cap_active, pmfuq[jk + 1] * zfac, pmfuq[jk + 1]))
        zmfuu = jnp.where(cap_active, zmfuu * zfac, zmfuu)
        zmfuv = jnp.where(cap_active, zmfuv * zfac, zmfuv)
        pmfub = jnp.where(cap_active, zmfmax_cap, pmfub)

        proceed = valid & (isum != 0)

        # CUENTR
        zpbase, zdmfen, zdmfde, zodetr = _cuentr_jax(
            klev, jk, ptenh, paph, pap, pgeoh, klwmin, ldcum, ktype,
            kcbot, kctop0, pmfu, pentr, zodetr, khmin)

        # ---- loflag block (only modifies when proceed & loflag) ----
        do_block = proceed & loflag
        below_base = jk < kcbot

        # adjust zdmfen / zdmfde for mass-flux cap
        zmftest = pmfu[jk + 1] + zdmfen - zdmfde
        zmfmax_b = jnp.minimum(zmftest, (paph[jk] - paph[jk - 1]) * zcons2)
        zdmfen_adj = jnp.where(below_base, jnp.maximum(zdmfen - jnp.maximum(zmftest - zmfmax_b, 0.0), 0.0), zdmfen)
        zdmfen = jnp.where(do_block, zdmfen_adj, zdmfen)
        zdmfde = jnp.where(do_block, jnp.minimum(zdmfde, 0.75 * pmfu[jk + 1]), zdmfde)
        pmfu_jk = pmfu[jk + 1] + zdmfen - zdmfde
        pmfu = pmfu.at[jk].set(jnp.where(do_block, pmfu_jk, pmfu[jk]))

        # organized entrainment scaling (below base)
        zdprho = (pgeoh[jk] - pgeoh[jk + 1]) * ZRG
        zoentr_jk = zoentr[jk] * zdprho * pmfu[jk + 1]
        zmftest2 = pmfu[jk] + zoentr_jk - zodetr[jk]
        zmfmax2 = jnp.minimum(zmftest2, (paph[jk] - paph[jk - 1]) * zcons2)
        zoentr_jk = jnp.maximum(zoentr_jk - jnp.maximum(zmftest2 - zmfmax2, 0.0), 0.0)
        do_oentr = do_block & below_base
        zoentr = zoentr.at[jk].set(jnp.where(do_oentr, zoentr_jk, zoentr[jk]))

        # organized detrainment limit (ktype1 below base, jk<=khmin)
        zmse = CPD * ptu[jk + 1] + ALV * pqu[jk + 1] + pgeoh[jk + 1]
        ikt = kctop0
        znevn = (pgeoh[ikt] - pgeoh[jk + 1]) * (zmse - phhatt[jk + 1]) * ZRG
        znevn = jnp.where(znevn <= 0.0, 1.0, znevn)
        zdprho_od = (pgeoh[jk] - pgeoh[jk + 1]) * ZRG
        zodmax = ((phcbase - zmse) / znevn) * zdprho_od * pmfu[jk + 1]
        do_odet = do_block & (ktype == 1) & below_base & (jk <= khmin)
        zodetr = zodetr.at[jk].set(jnp.where(do_odet, jnp.minimum(zodetr[jk], jnp.maximum(zodmax, 0.0)), zodetr[jk]))

        zodetr = zodetr.at[jk].set(jnp.where(do_block, jnp.minimum(zodetr[jk], 0.75 * pmfu[jk]), zodetr[jk]))
        pmfu = pmfu.at[jk].set(jnp.where(do_block, pmfu[jk] + zoentr[jk] - zodetr[jk], pmfu[jk]))

        zqeen = pqenh[jk + 1] * zdmfen + pqenh[jk + 1] * zoentr[jk]
        zseen = (CPD * ptenh[jk + 1] + pgeoh[jk + 1]) * zdmfen + (CPD * ptenh[jk + 1] + pgeoh[jk + 1]) * zoentr[jk]
        zscde = (CPD * ptu[jk + 1] + pgeoh[jk + 1]) * zdmfde
        zga = ALV * pqsenh[jk + 1] / (RV * (ptenh[jk + 1] ** 2))
        zdt = (plu[jk + 1] - 0.608 * (pqsenh[jk + 1] - pqenh[jk + 1])) / (1.0 / ptenh[jk + 1] + 0.608 * zga)
        zscod = CPD * ptenh[jk + 1] + pgeoh[jk + 1] + CPD * zdt
        zscde = zscde + zodetr[jk] * zscod
        zqude = pqu[jk + 1] * zdmfde
        zqcod = pqsenh[jk + 1] + zga * zdt
        zqude = zqude + zodetr[jk] * zqcod
        plude_jk = plu[jk + 1] * zdmfde + plu[jk + 1] * zodetr[jk]
        plude = plude.at[jk].set(jnp.where(do_block, plude_jk, plude[jk]))
        zmfusk = pmfus[jk + 1] + zseen - zscde
        zmfuqk = pmfuq[jk + 1] + zqeen - zqude
        zmfulk = pmful[jk + 1] - plude[jk]
        denom = jnp.maximum(CMFCMIN, pmfu[jk])
        plu = plu.at[jk].set(jnp.where(do_block, zmfulk / denom, plu[jk]))
        pqu = pqu.at[jk].set(jnp.where(do_block, zmfuqk / denom, pqu[jk]))
        ptu_jk = (zmfusk / denom - pgeoh[jk]) * RCPD
        ptu_jk = jnp.minimum(jnp.maximum(ptu_jk, 100.0), 400.0)
        ptu = ptu.at[jk].set(jnp.where(do_block, ptu_jk, ptu[jk]))
        zqold = jnp.where(do_block, pqu[jk], 0.0)

        # CUADJTQ at jk (loflag-gated)
        t_adj, q_adj = _cuadjtq_level(paph[jk], ptu[jk], pqu[jk], loflag, jnp.int32(1))
        ptu = ptu.at[jk].set(jnp.where(valid, t_adj, ptu[jk]))
        pqu = pqu.at[jk].set(jnp.where(valid, q_adj, pqu[jk]))

        cond_cond = valid & loflag & (pqu[jk] != zqold)
        plu_cc = plu[jk] + (zqold - pqu[jk])
        plu = plu.at[jk].set(jnp.where(cond_cond, plu_cc, plu[jk]))
        zbuo = ptu[jk] * (1.0 + VTMP_C1 * pqu[jk] - plu[jk]) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk])
        zbuo = jnp.where(klab[jk + 1] == 1, zbuo + ZBUO0, zbuo)
        new_top = cond_cond & (zbuo > 0.0) & (pmfu[jk] > 0.01 * pmfub) & (jk >= kctop0)
        klab = klab.at[jk].set(jnp.where(cond_cond, 2, klab[jk]))
        # top branch
        zprcon = jnp.where((zpbase - paph[jk]) >= ZDNOPRC, CPRCON, 0.0)
        zlnew = plu[jk] / (1.0 + zprcon * (pgeoh[jk] - pgeoh[jk + 1]))
        pdmfup_top = jnp.maximum(0.0, (plu[jk] - zlnew) * pmfu[jk])
        pdmfup = pdmfup.at[jk].set(jnp.where(new_top, pdmfup_top, pdmfup[jk]))
        plu = plu.at[jk].set(jnp.where(new_top, zlnew, plu[jk]))
        kctop = jnp.where(new_top, jk, kctop)
        ldcum = ldcum | new_top
        # not-top branch (condensation but not new cloud top)
        not_top = cond_cond & (~new_top)
        klab = klab.at[jk].set(jnp.where(not_top, 0, klab[jk]))
        pmfu = pmfu.at[jk].set(jnp.where(not_top, 0.0, pmfu[jk]))

        # update mass-flux integrals (loflag)
        do_int = valid & loflag
        pmful = pmful.at[jk].set(jnp.where(do_int, plu[jk] * pmfu[jk], pmful[jk]))
        pmfus = pmfus.at[jk].set(jnp.where(do_int, (CPD * ptu[jk] + pgeoh[jk]) * pmfu[jk], pmfus[jk]))
        pmfuq = pmfuq.at[jk].set(jnp.where(do_int, pqu[jk] * pmfu[jk], pmfuq[jk]))

        # LMFDUDV momentum
        if LMFDUDV:
            zdmfen_m = zdmfen + zoentr[jk]
            zdmfde_m = zdmfde + zodetr[jk]
            zz_dudv = jnp.where(
                (ktype == 1) | (ktype == 3),
                jnp.where(zdmfen_m <= 1.0e-20, 3.0, 2.0),
                jnp.where(zdmfen_m <= 1.0e-20, 1.0, 0.0),
            )
            zdmfeu = zdmfen_m + zz_dudv * zdmfde_m
            zdmfdu = jnp.minimum(zdmfde_m + zz_dudv * zdmfde_m, 0.75 * pmfu[jk + 1])
            zmfuu_new = zmfuu + zdmfeu * puen[jk] - zdmfdu * puu[jk + 1]
            zmfuv_new = zmfuv + zdmfeu * pven[jk] - zdmfdu * pvu[jk + 1]
            zmfuu = jnp.where(do_int, zmfuu_new, zmfuu)
            zmfuv = jnp.where(do_int, zmfuv_new, zmfuv)
            do_uvset = do_int & (pmfu[jk] > 0.0)
            puu = puu.at[jk].set(jnp.where(do_uvset, zmfuu / pmfu[jk], puu[jk]))
            pvu = pvu.at[jk].set(jnp.where(do_uvset, zmfuv / pmfu[jk], pvu[jk]))

        # ktype1 zoentr seed for next-lower level
        do_seed2 = valid & loflag & (ktype == 1) & (jk > 1)
        zbuoyz = G * ((ptu[jk] - ptenh[jk]) / ptenh[jk] + 0.608 * (pqu[jk] - pqenh[jk]) - plu[jk])
        zbuoyz = jnp.maximum(zbuoyz, 0.0)
        zdz_s = (pgeo[jk - 1] - pgeo[jk]) * ZRG
        zdz_s_safe = jnp.where(zdz_s == 0.0, 1.0, zdz_s)
        zdrodz_s = -jnp.log(pten[jk - 1] / pten[jk]) / zdz_s_safe - G / (RD * ptenh[jk])
        zbuoy_acc_new = zbuoy_acc + zbuoyz * zdz_s
        seed2 = zbuoyz * 0.5 / (1.0 + zbuoy_acc_new) + zdrodz_s
        seed2 = jnp.minimum(jnp.maximum(seed2, 0.0), 1.0e-3)
        zbuoy_acc = jnp.where(do_seed2, zbuoy_acc_new, zbuoy_acc)
        zoentr = zoentr.at[jk - 1].set(jnp.where(do_seed2, seed2, zoentr[jk - 1]))

        return (ptu, pqu, plu, puu, pvu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup,
                klab, zodetr, zoentr, kcbot, kctop, kctop0, ldcum, ktype, pmfub, pentr,
                zmfuu, zmfuv, zbuoy_acc)

    init = (ptu, pqu, plu, puu, pvu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup,
            klab, zodetr, zoentr, kcbot, kctop, kctop0, ldcum, ktype, pmfub, pentr,
            zmfuu, zmfuv, zbuoy_acc)
    (ptu, pqu, plu, puu, pvu, pmfu, pmfus, pmfuq, pmful, plude, pdmfup,
     klab, zodetr, zoentr, kcbot, kctop, kctop0, ldcum, ktype, pmfub, pentr,
     zmfuu, zmfuv, zbuoy_acc) = jax.lax.fori_loop(0, klev, asc_body, init)

    ldcum = jnp.where(kctop == klevm1, jnp.bool_(False), ldcum)
    kcbot = jnp.maximum(kcbot, kctop)
    kcum = jnp.where(ldcum, jnp.int32(1), jnp.int32(0))

    # CUASC final detrainment at cloud top (jk = kctop-1, when kcum != 0)
    jk = kctop - 1
    do_final = (kcum != 0) & (jk >= 1)
    zdmfde_f = (1.0 - CMFCTOP) * pmfu[jk + 1]
    plude = plude.at[jk].set(jnp.where(do_final, zdmfde_f * plu[jk + 1], plude[jk]))
    pmfu_f = pmfu[jk + 1] - zdmfde_f
    pmfu = pmfu.at[jk].set(jnp.where(do_final, pmfu_f, pmfu[jk]))
    pmfus = pmfus.at[jk].set(jnp.where(do_final, (CPD * ptu[jk] + pgeoh[jk]) * pmfu[jk], pmfus[jk]))
    pmfuq = pmfuq.at[jk].set(jnp.where(do_final, pqu[jk] * pmfu[jk], pmfuq[jk]))
    pmful = pmful.at[jk].set(jnp.where(do_final, plu[jk] * pmfu[jk], pmful[jk]))
    plude = plude.at[jk - 1].set(jnp.where(do_final, pmful[jk], plude[jk - 1]))
    pdmfup = pdmfup.at[jk].set(jnp.where(do_final, 0.0, pdmfup[jk]))
    if LMFDUDV:
        puu = puu.at[jk].set(jnp.where(do_final, puu[jk + 1], puu[jk]))
        pvu = pvu.at[jk].set(jnp.where(do_final, pvu[jk + 1], pvu[jk]))

    return {
        "ldcum": ldcum, "ktype": ktype, "kcbot": kcbot, "kctop": kctop,
        "kctop0": kctop0, "kcum": kcum,
        "ptu": ptu, "pqu": pqu, "plu": plu, "puu": puu, "pvu": pvu,
        "pmfu": pmfu, "pmfus": pmfus, "pmfuq": pmfuq, "pmful": pmful,
        "plude": plude, "pdmfup": pdmfup, "klab": klab, "pmfub": pmfub,
    }


# ----------------------------------------------------------------------------
# CUDLFS -- downdraft level-of-free-sink search.
# ----------------------------------------------------------------------------
def _cudlfs_jax(klev, ptenh, pqenh, puen, pven, pgeoh, paph, ptu, pqu, puu, pvu,
                ldcum, kcbot, kctop, pmfub, prfl_in, ptd, pqd, pud, pvd, pmfd, pmfds,
                pmfdq, pdmfdp):
    kdtop = jnp.int32(klev + 1)
    lddraf = jnp.bool_(False)
    prfl = prfl_in
    if not LMFDD:
        return kdtop, lddraf, prfl, ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp

    # jk = 3 .. klev-2 ascending
    def body(jk, carry):
        ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl = carry
        valid = (jk >= 3) & (jk <= klev - 2)
        llo2 = ldcum & (prfl > 0.0) & (~lddraf) & (jk < kcbot) & (jk > kctop)
        active = valid & llo2

        # wet-bulb adjust on copies of env at level jk
        t_wb, q_wb = _cuadjtq_level(paph[jk], ptenh[jk], pqenh[jk], True, jnp.int32(2))
        zttest = 0.5 * (ptu[jk] + t_wb)
        zqtest = 0.5 * (pqu[jk] + q_wb)
        zbuo = zttest * (1.0 + VTMP_C1 * zqtest) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk])
        zcond = pqenh[jk] - q_wb
        zmftop = -CMFDEPS * pmfub
        trigger = active & (zbuo < 0.0) & (prfl > 10.0 * zmftop * zcond)

        kdtop = jnp.where(trigger, jk, kdtop)
        lddraf = lddraf | trigger
        ptd = ptd.at[jk].set(jnp.where(trigger, zttest, ptd[jk]))
        pqd = pqd.at[jk].set(jnp.where(trigger, zqtest, pqd[jk]))
        pmfd = pmfd.at[jk].set(jnp.where(trigger, zmftop, pmfd[jk]))
        pmfds = pmfds.at[jk].set(jnp.where(trigger, pmfd[jk] * (CPD * ptd[jk] + pgeoh[jk]), pmfds[jk]))
        pmfdq = pmfdq.at[jk].set(jnp.where(trigger, pmfd[jk] * pqd[jk], pmfdq[jk]))
        pdmfdp_val = -0.5 * pmfd[jk] * zcond
        pdmfdp = pdmfdp.at[jk - 1].set(jnp.where(trigger, pdmfdp_val, pdmfdp[jk - 1]))
        prfl = jnp.where(trigger, prfl + pdmfdp_val, prfl)

        do_mom = active & LMFDUDV & (pmfd[jk] < 0.0)
        pud = pud.at[jk].set(jnp.where(do_mom, 0.5 * (puu[jk] + puen[jk - 1]), pud[jk]))
        pvd = pvd.at[jk].set(jnp.where(do_mom, 0.5 * (pvu[jk] + pven[jk - 1]), pvd[jk]))
        return (ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl)

    init = (ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl)
    ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl = jax.lax.fori_loop(
        3, klev - 1, body, init
    )
    return kdtop, lddraf, prfl, ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp


# ----------------------------------------------------------------------------
# CUDDRAF -- downdraft computation.
# ----------------------------------------------------------------------------
def _cuddraf_jax(klev, ptenh, pqenh, puen, pven, pgeoh, paph, prfl_in, lddraf,
                 ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp):
    itopde = klev - 2

    def body(jk, carry):
        ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, prfl = carry
        llo2 = lddraf & (pmfd[jk - 1] < 0.0)
        zentr = ENTRDD * pmfd[jk - 1] * RD * ptenh[jk - 1] / (G * paph[jk - 1]) * (paph[jk] - paph[jk - 1])
        above = jk > itopde
        zdmfen = jnp.where(above, 0.0, zentr)
        zdmfde = jnp.where(
            above,
            pmfd[itopde] * (paph[jk] - paph[jk - 1]) / (paph[klev + 1] - paph[itopde]),
            zentr,
        )
        pmfd_jk = pmfd[jk - 1] + zdmfen - zdmfde
        pmfd = pmfd.at[jk].set(jnp.where(llo2, pmfd_jk, pmfd[jk]))
        zseen = (CPD * ptenh[jk - 1] + pgeoh[jk - 1]) * zdmfen
        zqeen = pqenh[jk - 1] * zdmfen
        zsdde = (CPD * ptd[jk - 1] + pgeoh[jk - 1]) * zdmfde
        zqdde = pqd[jk - 1] * zdmfde
        zmfdsk = pmfds[jk - 1] + zseen - zsdde
        zmfdqk = pmfdq[jk - 1] + zqeen - zqdde
        denom = jnp.minimum(-CMFCMIN, pmfd[jk])
        pqd = pqd.at[jk].set(jnp.where(llo2, zmfdqk / denom, pqd[jk]))
        ptd_jk = (zmfdsk / denom - pgeoh[jk]) * RCPD
        ptd_jk = jnp.minimum(jnp.maximum(ptd_jk, 100.0), 400.0)
        ptd = ptd.at[jk].set(jnp.where(llo2, ptd_jk, ptd[jk]))
        zcond0 = pqd[jk]
        t_adj, q_adj = _cuadjtq_level(paph[jk], ptd[jk], pqd[jk], True, jnp.int32(2))
        ptd = ptd.at[jk].set(jnp.where(llo2, t_adj, ptd[jk]))
        pqd = pqd.at[jk].set(jnp.where(llo2, q_adj, pqd[jk]))
        zcond = zcond0 - pqd[jk]
        zbuo = ptd[jk] * (1.0 + VTMP_C1 * pqd[jk]) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk])
        kill = (zbuo >= 0.0) | (prfl <= (pmfd[jk] * zcond))
        pmfd = pmfd.at[jk].set(jnp.where(llo2 & kill, 0.0, pmfd[jk]))
        pmfds = pmfds.at[jk].set(jnp.where(llo2, (CPD * ptd[jk] + pgeoh[jk]) * pmfd[jk], pmfds[jk]))
        pmfdq = pmfdq.at[jk].set(jnp.where(llo2, pqd[jk] * pmfd[jk], pmfdq[jk]))
        pdmfdp_val = -pmfd[jk] * zcond
        pdmfdp = pdmfdp.at[jk - 1].set(jnp.where(llo2, pdmfdp_val, pdmfdp[jk - 1]))
        prfl = jnp.where(llo2, prfl + pdmfdp_val, prfl)

        do_mom = llo2 & LMFDUDV & (pmfd[jk] < 0.0)
        denom_m = jnp.minimum(-CMFCMIN, pmfd[jk])
        zmfduk = pmfd[jk - 1] * pud[jk - 1] + zdmfen * puen[jk - 1] - zdmfde * pud[jk - 1]
        zmfdvk = pmfd[jk - 1] * pvd[jk - 1] + zdmfen * pven[jk - 1] - zdmfde * pvd[jk - 1]
        pud = pud.at[jk].set(jnp.where(do_mom, zmfduk / denom_m, pud[jk]))
        pvd = pvd.at[jk].set(jnp.where(do_mom, zmfdvk / denom_m, pvd[jk]))
        return (ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, prfl)

    init = (ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, prfl_in)
    ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp, prfl = jax.lax.fori_loop(3, klev + 1, body, init)
    return prfl, ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp


# ----------------------------------------------------------------------------
# CUFLX -- mass-flux fluxes + precip + evaporation.
# ----------------------------------------------------------------------------
def _cuflx_jax(klev, pqen, pqsen, ptenh, pqenh, paph, pgeoh, kcbot, kctop,
               kdtop, ktype_in, lddraf_in, ldcum_in, pmfu, pmfd, pmfus, pmfds,
               pmfuq, pmfdq, pmful, plude, pdmfup, pdmfdp, pten, pdpmel, ztmst, sig1):
    zcons1 = CPD / (ALF * G * ztmst)
    zcons2 = 1.0 / (G * ztmst)
    zcucov = 0.05
    ztmelp2 = TMELT + 2.0
    ldcum = ldcum_in
    lddraf = lddraf_in
    ktype = ktype_in

    no_scv = (~LMFSCV) & (ktype == 2)
    ldcum = jnp.where(no_scv, jnp.bool_(False), ldcum)
    lddraf = jnp.where(no_scv, jnp.bool_(False), lddraf)
    itop = kctop
    lddraf = jnp.where((~ldcum) | (kdtop < kctop), jnp.bool_(False), lddraf)
    ktype = jnp.where(~ldcum, jnp.int32(0), ktype)
    ktopm2 = itop - 2

    # sweep1 jk = ktopm2 .. klev ascending (mask jk >= ktopm2)
    def body1(jk, carry):
        pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, pdmfup, pdmfdp, plude = carry
        valid = jk >= ktopm2
        in_cloud = ldcum & (jk >= kctop - 1)
        # branch A: in_cloud
        pmfus_A = pmfus[jk] - pmfu[jk] * (CPD * ptenh[jk] + pgeoh[jk])
        pmfuq_A = pmfuq[jk] - pmfu[jk] * pqenh[jk]
        has_dd = lddraf & (jk >= kdtop)
        pmfds_A = jnp.where(has_dd, pmfds[jk] - pmfd[jk] * (CPD * ptenh[jk] + pgeoh[jk]), 0.0)
        pmfdq_A = jnp.where(has_dd, pmfdq[jk] - pmfd[jk] * pqenh[jk], 0.0)
        pmfd_A = jnp.where(has_dd, pmfd[jk], 0.0)
        pdmfdp_A = jnp.where(has_dd, pdmfdp[jk - 1], 0.0)
        # branch B: not in_cloud -> zero everything
        applyA = valid & in_cloud
        applyB = valid & (~in_cloud)
        pmfus = pmfus.at[jk].set(jnp.where(applyA, pmfus_A, jnp.where(applyB, 0.0, pmfus[jk])))
        pmfuq = pmfuq.at[jk].set(jnp.where(applyA, pmfuq_A, jnp.where(applyB, 0.0, pmfuq[jk])))
        pmfds = pmfds.at[jk].set(jnp.where(applyA, pmfds_A, jnp.where(applyB, 0.0, pmfds[jk])))
        pmfdq = pmfdq.at[jk].set(jnp.where(applyA, pmfdq_A, jnp.where(applyB, 0.0, pmfdq[jk])))
        pmfd = pmfd.at[jk].set(jnp.where(applyA, pmfd_A, jnp.where(applyB, 0.0, pmfd[jk])))
        pdmfdp = pdmfdp.at[jk - 1].set(jnp.where(applyA, pdmfdp_A, jnp.where(applyB, 0.0, pdmfdp[jk - 1])))
        pmfu = pmfu.at[jk].set(jnp.where(applyB, 0.0, pmfu[jk]))
        pmful = pmful.at[jk].set(jnp.where(applyB, 0.0, pmful[jk]))
        pdmfup = pdmfup.at[jk - 1].set(jnp.where(applyB, 0.0, pdmfup[jk - 1]))
        plude = plude.at[jk - 1].set(jnp.where(applyB, 0.0, plude[jk - 1]))
        return (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, pdmfup, pdmfdp, plude)

    (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, pdmfup, pdmfdp, plude) = jax.lax.fori_loop(
        1, klev + 1, body1, (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, pdmfup, pdmfdp, plude)
    )

    # sweep2 jk = ktopm2 .. klev: subsidence below base + precip/snow accounting
    def body2(jk, carry):
        pmfu, pmfus, pmfuq, pmful, pdpmel, prfl, psfl, prain = carry
        valid = jk >= ktopm2
        below = ldcum & (jk > kcbot)
        zzp = (paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[kcbot])
        zzp = jnp.where(ktype == 3, zzp ** 2, zzp)
        pmfu = pmfu.at[jk].set(jnp.where(valid & below, pmfu[kcbot] * zzp, pmfu[jk]))
        pmfus = pmfus.at[jk].set(jnp.where(valid & below, pmfus[kcbot] * zzp, pmfus[jk]))
        pmfuq = pmfuq.at[jk].set(jnp.where(valid & below, pmfuq[kcbot] * zzp, pmfuq[jk]))
        pmful = pmful.at[jk].set(jnp.where(valid & below, pmful[kcbot] * zzp, pmful[jk]))

        do_pr = valid & ldcum
        prain = jnp.where(do_pr, prain + pdmfup[jk], prain)
        warm = pten[jk] > TMELT
        # warm branch
        prfl_w = prfl + pdmfup[jk] + pdmfdp[jk]
        melt = (psfl > 0.0) & (pten[jk] > ztmelp2)
        zfac = zcons1 * (paph[jk + 1] - paph[jk])
        zsnmlt = jnp.minimum(psfl, zfac * (pten[jk] - ztmelp2))
        pdpmel = pdpmel.at[jk].set(jnp.where(do_pr & warm & melt, zsnmlt, pdpmel[jk]))
        psfl_w = jnp.where(melt, psfl - zsnmlt, psfl)
        prfl_w = jnp.where(melt, prfl_w + zsnmlt, prfl_w)
        # cold branch
        psfl_c = psfl + pdmfup[jk] + pdmfdp[jk]
        prfl = jnp.where(do_pr, jnp.where(warm, prfl_w, prfl), prfl)
        psfl = jnp.where(do_pr, jnp.where(warm, psfl_w, psfl_c), psfl)
        return (pmfu, pmfus, pmfuq, pmful, pdpmel, prfl, psfl, prain)

    (pmfu, pmfus, pmfuq, pmful, pdpmel, prfl, psfl, prain) = jax.lax.fori_loop(
        1, klev + 1, body2,
        (pmfu, pmfus, pmfuq, pmful, pdpmel, jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0))
    )

    prfl = jnp.maximum(prfl, 0.0)
    psfl = jnp.maximum(psfl, 0.0)

    # sweep3: below-cloud-base evaporation
    def body3(jk, carry):
        pdmfup, zpsubcl = carry
        valid = jk >= ktopm2
        do_ev = valid & ldcum & (jk >= kcbot) & (zpsubcl > 1.0e-20)
        zrfl = zpsubcl
        cevapcu = CEVAPCU1 * jnp.sqrt(CEVAPCU2 * jnp.sqrt(sig1[jk]))
        zrnew = (jnp.maximum(0.0, jnp.sqrt(zrfl / zcucov) - cevapcu * (paph[jk + 1] - paph[jk]) * jnp.maximum(0.0, pqsen[jk] - pqen[jk]))) ** 2 * zcucov
        zrmin = zrfl - zcucov * jnp.maximum(0.0, 0.8 * pqsen[jk] - pqen[jk]) * zcons2 * (paph[jk + 1] - paph[jk])
        zrnew = jnp.maximum(zrnew, zrmin)
        zrfln = jnp.maximum(zrnew, 0.0)
        zdrfl = jnp.minimum(0.0, zrfln - zrfl)
        pdmfup = pdmfup.at[jk].set(jnp.where(do_ev, pdmfup[jk] + zdrfl, pdmfup[jk]))
        zpsubcl = jnp.where(do_ev, zrfln, zpsubcl)
        return (pdmfup, zpsubcl)

    pdmfup, zpsubcl = jax.lax.fori_loop(1, klev + 1, body3, (pdmfup, prfl + psfl))
    zdpevap = zpsubcl - (prfl + psfl)
    denom = jnp.maximum(1.0e-20, prfl + psfl)
    prfl = prfl + zdpevap * prfl / denom
    psfl = psfl + zdpevap * psfl / denom
    return ktopm2, ktype, lddraf, ldcum, prfl, psfl, prain, pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plude, pdmfup, pdmfdp, pdpmel


# ----------------------------------------------------------------------------
# CUDTDQ -- temperature / moisture tendencies.
# ----------------------------------------------------------------------------
def _cudtdq_jax(klev, ktopm2, paph, ldcum, pten, ptte_in, pqte_in, pmfus, pmfds,
                pmfuq, pmfdq, pmful, pdmfup, pdmfdp, ztmst, pdpmel, prain, prfl,
                psfl, pqen, pqsen, plude, pcte_in):
    ptte = ptte_in
    pqte = pqte_in
    pcte = pcte_in

    def body(jk, carry):
        ptte, pqte, pcte = carry
        valid = (jk >= ktopm2) & ldcum
        zalv = jnp.where(pten[jk] > TMELT, ALV, ALS)
        rhk = jnp.minimum(1.0, pqen[jk] / pqsen[jk])
        rhcoe = jnp.maximum(0.0, (rhk - RHC) / (RHM - RHC))
        pldfd = jnp.maximum(0.0, rhcoe * FDBK * plude[jk])
        is_interior = jk < klev
        g_dp = G / (paph[jk + 1] - paph[jk])
        # interior
        zdtdt_i = g_dp * RCPD * (
            pmfus[jk + 1] - pmfus[jk] + pmfds[jk + 1] - pmfds[jk]
            - ALF * pdpmel[jk] - zalv * (pmful[jk + 1] - pmful[jk] - pldfd - (pdmfup[jk] + pdmfdp[jk]))
        )
        zdqdt_i = g_dp * (
            pmfuq[jk + 1] - pmfuq[jk] + pmfdq[jk + 1] - pmfdq[jk]
            + pmful[jk + 1] - pmful[jk] - pldfd - (pdmfup[jk] + pdmfdp[jk])
        )
        # surface
        zdtdt_s = -g_dp * RCPD * (
            pmfus[jk] + pmfds[jk] + ALF * pdpmel[jk]
            - zalv * (pmful[jk] + pdmfup[jk] + pdmfdp[jk] + pldfd)
        )
        zdqdt_s = -g_dp * (
            pmfuq[jk] + pmfdq[jk] + pldfd + (pmful[jk] + pdmfup[jk] + pdmfdp[jk])
        )
        zdtdt = jnp.where(is_interior, zdtdt_i, zdtdt_s)
        zdqdt = jnp.where(is_interior, zdqdt_i, zdqdt_s)
        ptte = ptte.at[jk].set(jnp.where(valid, ptte[jk] + zdtdt, ptte[jk]))
        pqte = pqte.at[jk].set(jnp.where(valid, pqte[jk] + zdqdt, pqte[jk]))
        pcte = pcte.at[jk].set(jnp.where(valid, g_dp * pldfd, pcte[jk]))
        return (ptte, pqte, pcte)

    ptte, pqte, pcte = jax.lax.fori_loop(1, klev + 1, body, (ptte, pqte, pcte))
    prsfc = jnp.where(ldcum, prfl, 0.0)
    pssfc = jnp.where(ldcum, psfl, 0.0)
    return prsfc, pssfc, ptte, pqte, pcte


# ----------------------------------------------------------------------------
# CUDUDV -- momentum tendencies.
# ----------------------------------------------------------------------------
def _cududv_jax(klev, ktopm2, ktype, kcbot, paph, ldcum, puen, pven,
                pvom_in, pvol_in, puu, pud, pvu, pvd, pmfu, pmfd):
    pvom = pvom_in
    pvol = pvol_in
    zmfuu = _zeros(klev)
    zmfdu = _zeros(klev)
    zmfuv = _zeros(klev)
    zmfdv = _zeros(klev)

    def b1(jk, carry):
        zmfuu, zmfuv, zmfdu, zmfdv = carry
        valid = jk >= ktopm2
        ik = jk - 1
        zmfuu = zmfuu.at[jk].set(jnp.where(valid, pmfu[jk] * (puu[jk] - puen[ik]), zmfuu[jk]))
        zmfuv = zmfuv.at[jk].set(jnp.where(valid, pmfu[jk] * (pvu[jk] - pven[ik]), zmfuv[jk]))
        zmfdu = zmfdu.at[jk].set(jnp.where(valid, pmfd[jk] * (pud[jk] - puen[ik]), zmfdu[jk]))
        zmfdv = zmfdv.at[jk].set(jnp.where(valid, pmfd[jk] * (pvd[jk] - pven[ik]), zmfdv[jk]))
        return (zmfuu, zmfuv, zmfdu, zmfdv)

    zmfuu, zmfuv, zmfdu, zmfdv = jax.lax.fori_loop(1, klev + 1, b1, (zmfuu, zmfuv, zmfdu, zmfdv))

    def b2(jk, carry):
        zmfuu, zmfuv, zmfdu, zmfdv = carry
        valid = (jk >= ktopm2) & (jk > kcbot)
        zzp = (paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[kcbot])
        zzp = jnp.where(ktype == 3, zzp ** 2, zzp)
        zmfuu = zmfuu.at[jk].set(jnp.where(valid, zmfuu[kcbot] * zzp, zmfuu[jk]))
        zmfuv = zmfuv.at[jk].set(jnp.where(valid, zmfuv[kcbot] * zzp, zmfuv[jk]))
        zmfdu = zmfdu.at[jk].set(jnp.where(valid, zmfdu[kcbot] * zzp, zmfdu[jk]))
        zmfdv = zmfdv.at[jk].set(jnp.where(valid, zmfdv[kcbot] * zzp, zmfdv[jk]))
        return (zmfuu, zmfuv, zmfdu, zmfdv)

    zmfuu, zmfuv, zmfdu, zmfdv = jax.lax.fori_loop(1, klev + 1, b2, (zmfuu, zmfuv, zmfdu, zmfdv))

    def b3(jk, carry):
        pvom, pvol = carry
        valid = jk >= ktopm2
        is_interior = jk < klev
        g_dp = G / (paph[jk + 1] - paph[jk])
        zdudt_i = g_dp * (zmfuu[jk + 1] - zmfuu[jk] + zmfdu[jk + 1] - zmfdu[jk])
        zdvdt_i = g_dp * (zmfuv[jk + 1] - zmfuv[jk] + zmfdv[jk + 1] - zmfdv[jk])
        zdudt_s = -g_dp * (zmfuu[jk] + zmfdu[jk])
        zdvdt_s = -g_dp * (zmfuv[jk] + zmfdv[jk])
        zdudt = jnp.where(is_interior, zdudt_i, zdudt_s)
        zdvdt = jnp.where(is_interior, zdvdt_i, zdvdt_s)
        pvom = pvom.at[jk].set(jnp.where(valid, pvom[jk] + zdudt, pvom[jk]))
        pvol = pvol.at[jk].set(jnp.where(valid, pvol[jk] + zdvdt, pvol[jk]))
        return (pvom, pvol)

    do_dudv = ldcum
    pvom2, pvol2 = jax.lax.fori_loop(1, klev + 1, b3, (pvom, pvol))
    pvom = jnp.where(do_dudv, pvom2, pvom)
    pvol = jnp.where(do_dudv, pvol2, pvol)
    return pvom, pvol


# ----------------------------------------------------------------------------
# CUMASTRN master.
# ----------------------------------------------------------------------------
def _cumastr_new_jax(pu, pv, pt, pqv, pqc, pqi, pqvf, pqvbl, poz, pomg, pap, paph,
                     evap, lndj, sig1, dt, klev):
    klevm1 = klev - 1
    ztmst = dt
    pqhfl = evap

    ptte = _zeros(klev)
    pcte = _zeros(klev)
    pvom = _zeros(klev)
    pvol = _zeros(klev)
    ztp1 = pt
    pum1 = pu
    pvm1 = pv
    pverv = pomg
    pgeo = _zeros(klev)
    pqsen = _zeros(klev)
    pqte = _zeros(klev)
    zqq = _zeros(klev)
    zqp1 = _zeros(klev)

    def pre_body(k, carry):
        zqp1, pgeo, pqsen, pqte, zqq = carry
        zqp1 = zqp1.at[k].set(pqv[k] / (1.0 + pqv[k]))
        pgeo = pgeo.at[k].set(G * poz[k])
        qs = jnp.minimum(0.5, _tlucua(ztp1[k]) / pap[k])
        qs = qs / (1.0 - VTMP_C1 * qs)
        pqsen = pqsen.at[k].set(qs)
        pqte = pqte.at[k].set(pqvf[k] + pqvbl[k])
        zqq = zqq.at[k].set(pqte[k])
        return (zqp1, pgeo, pqsen, pqte, zqq)

    zqp1, pgeo, pqsen, pqte, zqq = jax.lax.fori_loop(1, klev + 1, pre_body, (zqp1, pgeo, pqsen, pqte, zqq))

    st = _cuini_jax(klev, ztp1, zqp1, pqsen, pum1, pvm1, pverv, pgeo, paph)
    pgeoh = st["pgeoh"]; ztenh = st["ptenh"]; zqenh = st["pqenh"]; zqsenh = st["pqsenh"]
    ptu = st["ptu"]; pqu = st["pqu"]; ztd = st["ptd"]; zqd = st["pqd"]
    zuu = st["puu"]; zvu = st["pvu"]; zud = st["pud"]; zvd = st["pvd"]
    pmfu = st["pmfu"]; pmfd = st["pmfd"]; zmfus = st["pmfus"]; zmfds = st["pmfds"]
    zmfuq = st["pmfuq"]; zmfdq = st["pmfdq"]; zmful = st["pmful"]
    zdmfup = st["pdmfup"]; zdmfdp = st["pdmfdp"]
    zdpmel = st["pdpmel"]; zlu = st["plu"]; zlude = st["plude"]; ilab = st["klab"]
    ilwmin = st["klwmin"]

    ldcum, kcbot, ilab, ptu, pqu, zlu, zuu, zvu = _cubase_jax(
        klev, ztenh, zqenh, pgeoh, paph, ptu, pqu, zlu, pum1, pvm1, zuu, zvu)

    # zdqcv / zdqpbl accumulation
    def dq_body(jk, carry):
        zdqcv, zdqpbl = carry
        zdqcv = zdqcv + pqte[jk] * (paph[jk + 1] - paph[jk])
        zdqpbl = jnp.where(jk >= kcbot, zdqpbl + pqte[jk] * (paph[jk + 1] - paph[jk]), zdqpbl)
        return (zdqcv, zdqpbl)

    zdqcv0 = pqte[1] * (paph[2] - paph[1])
    zdqcv, zdqpbl = jax.lax.fori_loop(2, klev + 1, dq_body, (zdqcv0, jnp.float64(0.0)))

    ktype = jnp.where(zdqcv > jnp.maximum(0.0, 1.1 * pqhfl * G), jnp.int32(1), jnp.int32(2))
    ikb = kcbot
    zqumqe = pqu[ikb] + zlu[ikb] - zqenh[ikb]
    zdqmin = jnp.maximum(0.01 * zqenh[ikb], 1.0e-10)
    zcons2 = 1.0 / (G * ztmst)
    use_mfub = (zdqpbl > 0.0) & (zqumqe > zdqmin) & ldcum
    zmfub = jnp.where(use_mfub, zdqpbl / (G * jnp.maximum(zqumqe, zdqmin)), 0.01)
    ldcum = ldcum & use_mfub
    zmfmax = (paph[ikb] - paph[ikb - 1]) * zcons2
    zmfub = jnp.minimum(zmfub, zmfmax)

    zhcbase = CPD * ptu[ikb] + pgeoh[ikb] + ALV * pqu[ikb]
    zalvdcp = ALV / CPD
    zqalv = 1.0 / ALV

    # ictop0 search + zhhatt
    zhhatt = _zeros(klev)

    def ictop_body(idx, carry):
        zhhatt, ictop0 = carry
        jk = klevm1 - idx  # klevm1 .. 3 descending (range klevm1..2 exclusive of 2 in py)
        valid = jk >= 3
        zhsat = CPD * ztenh[jk] + pgeoh[jk] + ALV * zqsenh[jk]
        zgam = C5LES * zalvdcp * zqsenh[jk] / ((1.0 - VTMP_C1 * zqsenh[jk]) * (ztenh[jk] - C4LES) ** 2)
        zzz = CPD * ztenh[jk] * 0.608
        zhhat = zhsat - (zzz + zgam * zzz) / (1.0 + zgam * zzz * zqalv) * jnp.maximum(zqsenh[jk] - zqenh[jk], 0.0)
        zhhatt = zhhatt.at[jk].set(jnp.where(valid, zhhat, zhhatt[jk]))
        hit = valid & (jk < ictop0) & (zhcbase > zhhat)
        ictop0 = jnp.where(hit, jk, ictop0)
        return (zhhatt, ictop0)

    ictop0_init = kcbot - 1
    zhhatt, ictop0 = jax.lax.fori_loop(0, klev, ictop_body, (zhhatt, ictop0_init))
    # zhhatt at kcbot
    jk = kcbot
    zhsat = CPD * ztenh[jk] + pgeoh[jk] + ALV * zqsenh[jk]
    zgam = C5LES * zalvdcp * zqsenh[jk] / ((1.0 - VTMP_C1 * zqsenh[jk]) * (ztenh[jk] - C4LES) ** 2)
    zzz = CPD * ztenh[jk] * 0.608
    zhhatt = zhhatt.at[jk].set(zhsat - (zzz + zgam * zzz) / (1.0 + zgam * zzz * zqalv) * jnp.maximum(zqsenh[jk] - zqenh[jk], 0.0))

    # ihmin search
    ihmin0 = jnp.where(ldcum & (ktype == 1), kcbot, jnp.int32(-1))
    zbi = 1.0 / (25.0 * G)

    def ihmin_body(idx, carry):
        ihmin, zhmin = carry
        jk = klev - idx  # klev .. 1 descending
        valid = jk >= 1
        llo1 = ldcum & (ktype == 1) & (ihmin0 == kcbot)
        active = valid & llo1 & (jk < kcbot) & (jk >= ictop0)
        zro = RD * ztenh[jk] / (G * paph[jk])
        zdz = (paph[jk] - paph[jk - 1]) * zro
        dgeo = pgeo[jk - 1] - pgeo[jk]
        dgeo_safe = jnp.where(dgeo == 0.0, 1.0, dgeo)
        zdhdz = (CPD * (ztp1[jk - 1] - ztp1[jk]) + ALV * (zqp1[jk - 1] - zqp1[jk]) + dgeo) * G / dgeo_safe
        zdepth = pgeoh[jk] - pgeoh[ikb]
        zfac = jnp.sqrt(1.0 + zdepth * zbi)
        zhmin_new = zhmin + zdhdz * zfac * zdz
        zrh = -ALV * (zqsenh[jk] - zqenh[jk]) * zfac
        hit = active & (zhmin_new > zrh)
        ihmin = jnp.where(hit, jk, ihmin)
        zhmin = jnp.where(active, zhmin_new, zhmin)
        return (ihmin, zhmin)

    ihmin, _ = jax.lax.fori_loop(0, klev, ihmin_body, (ihmin0, jnp.float64(0.0)))
    ihmin = jnp.where(ldcum & (ktype == 1) & (ihmin < ictop0), ictop0, ihmin)
    zentr = jnp.where(ktype == 1, ENTRPEN, ENTRSCV)
    zentr = jnp.where(lndj == 1, zentr * 1.1, zentr)

    kctop = jnp.int32(klevm1)
    asc1 = _cuasc_jax(
        klev, ztenh, zqenh, pum1, pvm1, ztp1, zqp1, pqsen, pgeo, pgeoh,
        pap, paph, pqte, pverv, ilwmin, ldcum, zhcbase, ktype, ilab,
        ptu, pqu, zlu, zuu, zvu, pmfu, zmfub, zentr, zmfus, zmfuq, zmful,
        zlude, zdmfup, kcbot, kctop, ictop0, ztmst, ihmin, zhhatt, zqsenh)

    ldcum = asc1["ldcum"]; ktype = asc1["ktype"]; kcbot = asc1["kcbot"]
    kctop = asc1["kctop"]; ictop0 = asc1["kctop0"]; icum = asc1["kcum"]
    ptu = asc1["ptu"]; pqu = asc1["pqu"]; zlu = asc1["plu"]; zuu = asc1["puu"]; zvu = asc1["pvu"]
    pmfu = asc1["pmfu"]; zmfus = asc1["pmfus"]; zmfuq = asc1["pmfuq"]; zmful = asc1["pmful"]
    zlude = asc1["plude"]; zdmfup = asc1["pdmfup"]; ilab = asc1["klab"]; zmfub = asc1["pmfub"]

    cum_active = icum != 0

    zpbmpt = paph[kcbot] - paph[kctop]
    ictop0 = jnp.where(ldcum, kctop, ictop0)
    ktype = jnp.where(ldcum & (ktype == 1) & (zpbmpt < ZDNOPRC), jnp.int32(2), ktype)
    zentr = jnp.where(ktype == 2, ENTRSCV * jnp.where(lndj == 1, 1.1, 1.0), zentr)

    # zrfl = sum zdmfup over 1..klev
    zrfl = jnp.sum(jnp.where(jnp.arange(klev + 2) <= klev, zdmfup, 0.0) * (jnp.arange(klev + 2) >= 1))

    if LMFDD:
        idtop, loddraf, zrfl, ztd, zqd, zud, zvd, pmfd, zmfds, zmfdq, zdmfdp = _cudlfs_jax(
            klev, ztenh, zqenh, pum1, pvm1, pgeoh, paph, ptu, pqu, zuu, zvu,
            ldcum, kcbot, kctop, zmfub, zrfl, ztd, zqd, zud, zvd, pmfd, zmfds, zmfdq, zdmfdp)
        zrfl, ztd, zqd, zud, zvd, pmfd, zmfds, zmfdq, zdmfdp = _cuddraf_jax(
            klev, ztenh, zqenh, pum1, pvm1, pgeoh, paph, zrfl, loddraf,
            ztd, zqd, zud, zvd, pmfd, zmfds, zmfdq, zdmfdp)
    else:
        idtop = jnp.int32(klev + 1)
        loddraf = jnp.bool_(False)

    # cape/closure for ktype==1
    ktop0 = jnp.maximum(12, kctop)
    ikb = kcbot

    def cape_body(jk, carry):
        zheat, zcape, zrelh = carry
        in1 = (jk <= kcbot) & (jk > kctop)
        zro = paph[jk] / (RD * ztenh[jk])
        zdz = (paph[jk] - paph[jk - 1]) / (G * zro)
        zheat_add = (((ztp1[jk - 1] - ztp1[jk] + G * zdz / CPD) / ztenh[jk]
                      + 0.608 * (zqp1[jk - 1] - zqp1[jk])) * (pmfu[jk] + pmfd[jk]) * G / zro)
        zcape_add = G * ((ptu[jk] * (1.0 + 0.608 * pqu[jk] - zlu[jk]))
                         / (ztenh[jk] * (1.0 + 0.608 * zqenh[jk])) - 1.0) * zdz
        zheat = jnp.where(in1, zheat + zheat_add, zheat)
        zcape = jnp.where(in1, zcape + zcape_add, zcape)
        in2 = (jk <= kcbot) & (jk > ktop0)
        dept = (paph[jk + 1] - paph[jk]) / (paph[ikb + 1] - paph[ktop0 + 1])
        zrelh = jnp.where(in2, zrelh + dept * zqp1[jk] / pqsen[jk], zrelh)
        return (zheat, zcape, zrelh)

    do_cape = ldcum & (ktype == 1)
    zheat, zcape, zrelh = jax.lax.fori_loop(2, klev + 1, cape_body, (jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0)))
    closure_ok = do_cape & (zrelh >= CRIRH) & (zheat != 0.0)
    zht = jnp.maximum(0.0, zcape) / (ZTAU * jnp.where(zheat != 0.0, zheat, 1.0))
    zmfmax_c = (paph[ikb] - paph[ikb - 1]) * zcons2
    zmfub1_ok = jnp.minimum(jnp.maximum(zmfub * zht, 0.01), zmfmax_c)
    zmfub1 = jnp.where(do_cape, jnp.where(closure_ok, zmfub1_ok, 0.01), zmfub)
    # when ktype1 and closure fails: zmfub=0.01, ldcum=False
    zmfub = jnp.where(do_cape & (~closure_ok), 0.01, zmfub)
    ldcum = jnp.where(do_cape & (~closure_ok), jnp.bool_(False), ldcum)

    # ktype != 1 closure
    not1 = ktype != 1
    ikb = kcbot
    zeps = jnp.where((pmfd[ikb] < 0.0) & loddraf, CMFDEPS, 0.0)
    zqumqe2 = pqu[ikb] + zlu[ikb] - zeps * zqd[ikb] - (1.0 - zeps) * zqenh[ikb]
    zdqmin2 = jnp.maximum(0.01 * zqenh[ikb], 1.0e-10)
    zmfmax2 = (paph[ikb] - paph[ikb - 1]) * zcons2
    use2 = (zdqpbl > 0.0) & (zqumqe2 > zdqmin2) & ldcum & (zmfub < zmfmax2)
    zmfub1_n = jnp.where(use2, zdqpbl / (G * jnp.maximum(zqumqe2, zdqmin2)), zmfub)
    llo1 = (ktype == 2) & (jnp.abs(zmfub1_n - zmfub) < 0.2 * zmfub)
    zmfub1_n = jnp.where(llo1, zmfub1_n, zmfub)
    zmfub1_n = jnp.minimum(zmfub1_n, zmfmax2)
    zmfub1 = jnp.where(not1, zmfub1_n, zmfub1)

    # rescale downdraft mass flux
    def scale_body(jk, carry):
        pmfd, zmfds, zmfdq, zdmfdp = carry
        zfac = zmfub1 / jnp.maximum(zmfub, 1.0e-10)
        pmfd = pmfd.at[jk].set(jnp.where(ldcum, pmfd[jk] * zfac, 0.0))
        zmfds = zmfds.at[jk].set(jnp.where(ldcum, zmfds[jk] * zfac, 0.0))
        zmfdq = zmfdq.at[jk].set(jnp.where(ldcum, zmfdq[jk] * zfac, 0.0))
        zdmfdp = zdmfdp.at[jk].set(jnp.where(ldcum, zdmfdp[jk] * zfac, 0.0))
        return (pmfd, zmfds, zmfdq, zdmfdp)

    pmfd, zmfds, zmfdq, zdmfdp = jax.lax.fori_loop(1, klev + 1, scale_body, (pmfd, zmfds, zmfdq, zdmfdp))
    zmfub = jnp.where(ldcum, zmfub1, 0.0)

    asc2 = _cuasc_jax(
        klev, ztenh, zqenh, pum1, pvm1, ztp1, zqp1, pqsen, pgeo, pgeoh,
        pap, paph, pqte, pverv, ilwmin, ldcum, zhcbase, ktype, ilab,
        ptu, pqu, zlu, zuu, zvu, pmfu, zmfub, zentr, zmfus, zmfuq, zmful,
        zlude, zdmfup, kcbot, kctop, ictop0, ztmst, ihmin, zhhatt, zqsenh)

    ldcum = asc2["ldcum"]; ktype = asc2["ktype"]; kcbot = asc2["kcbot"]
    kctop = asc2["kctop"]; ictop0 = asc2["kctop0"]
    ptu = asc2["ptu"]; pqu = asc2["pqu"]; zlu = asc2["plu"]; zuu = asc2["puu"]; zvu = asc2["pvu"]
    pmfu = asc2["pmfu"]; zmfus = asc2["pmfus"]; zmfuq = asc2["pmfuq"]; zmful = asc2["pmful"]
    zlude = asc2["plude"]; zdmfup = asc2["pdmfup"]; ilab = asc2["klab"]

    (ktopm2, ktype, loddraf, ldcum, prfl, psfl, prain, pmfu, pmfd, zmfus, zmfds,
     zmfuq, zmfdq, zmful, zlude, zdmfup, zdmfdp, zdpmel) = _cuflx_jax(
        klev, zqp1, pqsen, ztenh, zqenh, paph, pgeoh, kcbot, kctop, idtop,
        ktype, loddraf, ldcum, pmfu, pmfd, zmfus, zmfds, zmfuq, zmfdq, zmful,
        zlude, zdmfup, zdmfdp, ztp1, zdpmel, ztmst, sig1)

    prsfc, pssfc, ptte, pqte, pcte = _cudtdq_jax(
        klev, ktopm2, paph, ldcum, ztp1, ptte, pqte, zmfus, zmfds, zmfuq,
        zmfdq, zmful, zdmfup, zdmfdp, ztmst, zdpmel, prain, prfl, psfl, zqp1,
        pqsen, zlude, pcte)

    if LMFDUDV:
        pvom, pvol = _cududv_jax(klev, ktopm2, ktype, kcbot, paph, ldcum, pum1, pvm1,
                                 pvom, pvol, zuu, zud, zvu, zvd, pmfu, pmfd)

    zprecc = jnp.maximum(0.0, (prsfc + pssfc) * ztmst)

    # when icum==0 (no convection from first ascent), output is zero tendencies.
    zprecc = jnp.where(cum_active, zprecc, 0.0)
    ktype_out = jnp.where(cum_active, ktype, jnp.int32(0))
    ptte = jnp.where(cum_active, ptte, jnp.zeros_like(ptte))
    pqte = jnp.where(cum_active, pqte, zqq)  # pqte resets to zqq so (pqte-zqq)=0
    pcte = jnp.where(cum_active, pcte, jnp.zeros_like(pcte))
    pvom = jnp.where(cum_active, pvom, jnp.zeros_like(pvom))
    pvol = jnp.where(cum_active, pvol, jnp.zeros_like(pvol))

    # _finish_tiecnv: liquid/ice split feedback from detrained condensate.
    pqc_out = pqc
    pqi_out = pqi

    def finish_body(k, carry):
        pqc_out, pqi_out, ptte = carry
        do = pcte[k] > 0.0
        ztpp1 = pt[k] + ptte[k] * ztmst
        warm = ztpp1 >= T000
        cold = ztpp1 <= HGFR
        ztc = ztpp1 - T000
        fliq_mid = 0.0059 + 0.9941 * jnp.exp(-0.003102 * ztc * ztc)
        fliq = jnp.where(warm, 1.0, jnp.where(cold, 0.0, fliq_mid))
        zalf = jnp.where(warm, 0.0, ALF)
        fice = 1.0 - fliq
        pqc_out = pqc_out.at[k].set(jnp.where(do, pqc_out[k] + fliq * pcte[k] * ztmst, pqc_out[k]))
        pqi_out = pqi_out.at[k].set(jnp.where(do, pqi_out[k] + fice * pcte[k] * ztmst, pqi_out[k]))
        ptte = ptte.at[k].set(jnp.where(do, ptte[k] - zalf * RCPD * fliq * pcte[k], ptte[k]))
        return (pqc_out, pqi_out, ptte)

    do_finish = FDBK >= 1.0e-9
    if do_finish:
        pqc_out, pqi_out, ptte = jax.lax.fori_loop(1, klev + 1, finish_body, (pqc_out, pqi_out, ptte))

    # final state update
    pt_out = pt
    pqv_out = pqv
    pu_out = pu
    pv_out = pv
    zqp1_out = zqp1

    def upd_body(k, carry):
        pt_out, pqv_out, pu_out, pv_out, zqp1_out = carry
        pt_out = pt_out.at[k].set(ztp1[k] + ptte[k] * ztmst)
        zq = zqp1[k] + (pqte[k] - zqq[k]) * ztmst
        zqp1_out = zqp1_out.at[k].set(zq)
        pqv_out = pqv_out.at[k].set(zq / (1.0 - zq))
        if LMFDUDV:
            pu_out = pu_out.at[k].set(pu_out[k] + pvom[k] * ztmst)
            pv_out = pv_out.at[k].set(pv_out[k] + pvol[k] * ztmst)
        return (pt_out, pqv_out, pu_out, pv_out, zqp1_out)

    pt_out, pqv_out, pu_out, pv_out, zqp1_out = jax.lax.fori_loop(
        1, klev + 1, upd_body, (pt_out, pqv_out, pu_out, pv_out, zqp1_out))

    return {
        "pu": pu_out, "pv": pv_out, "pt": pt_out, "pqv": pqv_out,
        "pqc": pqc_out, "pqi": pqi_out, "zprecc": zprecc, "ktype": ktype_out,
    }


# ----------------------------------------------------------------------------
# Column entry: bottom-up WRF/JAX <-> top-down internal, identical to the
# NumPy reference cumulus_tiedtke.tiedtke_column.
# ----------------------------------------------------------------------------
def tiedtke_column_jax(T, QV, QC, QI, P, P8W, DZ, RHO, PI, U, V, W,
                       QVFTEN, QVPBLTEN, QFX, XLAND, ZNU, dt, *, stepcu=5):
    """One WRF modified-Tiedtke column (GPU-batchable, fully jit/vmap-traceable).

    Inputs are bottom-up WRF/JAX columns of length KLEV (P8W and W are KLEV+1).
    Returns a dict of bottom-up tendency columns + RAINCV/PRATEC/KTYPE, identical
    in meaning to ``cumulus_tiedtke.tiedtke_column``.
    """

    T = jnp.asarray(T, jnp.float64)
    QV = jnp.asarray(QV, jnp.float64)
    QC = jnp.asarray(QC, jnp.float64)
    QI = jnp.asarray(QI, jnp.float64)
    P = jnp.asarray(P, jnp.float64)
    P8W = jnp.asarray(P8W, jnp.float64)
    DZ = jnp.asarray(DZ, jnp.float64)
    RHO = jnp.asarray(RHO, jnp.float64)
    PI = jnp.asarray(PI, jnp.float64)
    U = jnp.asarray(U, jnp.float64)
    V = jnp.asarray(V, jnp.float64)
    W = jnp.asarray(W, jnp.float64)
    QVFTEN = jnp.asarray(QVFTEN, jnp.float64)
    QVPBLTEN = jnp.asarray(QVPBLTEN, jnp.float64)
    ZNU = jnp.asarray(ZNU, jnp.float64)

    klev = T.shape[0]
    delt = float(dt) * int(stepcu)
    rdelt = 1.0 / delt

    # zi/zl heights (bottom-up); matches _prepare_tie_state
    dz_cum = jnp.concatenate([jnp.zeros(1, jnp.float64), jnp.cumsum(DZ)])  # zi[0..klev]
    zi = dz_cum  # zi[k] = sum DZ[0..k-1]; zi[0]=0
    zl = jnp.zeros(klev, jnp.float64)
    # zl[k-1] = 0.5*(zi[k]+zi[k-1]) for k in 1..klev-1 ; top extrapolated
    zl_interior = 0.5 * (zi[1:klev] + zi[0:klev - 1])  # len klev-1 -> indices 0..klev-2
    zl = zl.at[0:klev - 1].set(zl_interior)
    zl = zl.at[klev - 1].set(2.0 * zi[klev - 1] - zl_interior[klev - 2])

    # build 1-based top-down arrays of length klev+2
    pu = _zeros(klev); pv = _zeros(klev); pt = _zeros(klev); pqv = _zeros(klev)
    pqc = _zeros(klev); pqi = _zeros(klev); pqvf = _zeros(klev); pqvbl = _zeros(klev)
    poz = _zeros(klev); pomg = _zeros(klev); pap = _zeros(klev); sig1 = _zeros(klev)
    paph = _zeros(klev)

    kb = jnp.arange(1, klev + 1)
    kt = klev + 1 - kb  # top-down index for each bottom index
    idx0 = kb - 1
    pu = pu.at[kt].set(U[idx0])
    pv = pv.at[kt].set(V[idx0])
    pt = pt.at[kt].set(T[idx0])
    pqv = pqv.at[kt].set(QV[idx0])
    pqc = pqc.at[kt].set(QC[idx0])
    pqi = pqi.at[kt].set(QI[idx0])
    pqvf = pqvf.at[kt].set(QVFTEN[idx0])
    pqvbl = pqvbl.at[kt].set(QVPBLTEN[idx0])
    pomg = pomg.at[kt].set(-0.5 * G * RHO[idx0] * (W[idx0] + W[idx0 + 1]))
    poz = poz.at[kt].set(zl[idx0])
    pap = pap.at[kt].set(P[idx0])
    sig1 = sig1.at[kt].set(ZNU[idx0])

    kb2 = jnp.arange(1, klev + 2)
    kt2 = klev + 2 - kb2
    paph = paph.at[kt2].set(P8W[kb2 - 1])

    lndj = jnp.int32(jnp.abs(jnp.asarray(XLAND, jnp.float64) - 2.0).astype(jnp.int32))
    qfx = jnp.asarray(QFX, jnp.float64)

    out = _cumastr_new_jax(pu, pv, pt, pqv, pqc, pqi, pqvf, pqvbl, poz, pomg, pap, paph,
                           qfx, lndj, sig1, delt, klev)

    # back to bottom-up tendencies
    T0 = T; QV0 = QV; QC0 = QC; QI0 = QI; U0 = U; V0 = V; PI0 = PI
    kt_b = klev + 1 - kb  # top-down index for bottom index kb
    rth = (out["pt"][kt_b] - T0[idx0]) / PI0[idx0] * rdelt
    rqv = (out["pqv"][kt_b] - QV0[idx0]) * rdelt
    rqc = (out["pqc"][kt_b] - QC0[idx0]) * rdelt
    rqi = (out["pqi"][kt_b] - QI0[idx0]) * rdelt
    ru = (out["pu"][kt_b] - U0[idx0]) * rdelt
    rv = (out["pv"][kt_b] - V0[idx0]) * rdelt
    raincv = out["zprecc"] / float(stepcu)
    pratec = out["zprecc"] / (float(stepcu) * float(dt))
    zeros = jnp.zeros_like(rth)
    return {
        "RTHCUTEN": rth, "RQVCUTEN": rqv, "RQCCUTEN": rqc, "RQRCUTEN": zeros,
        "RQICUTEN": rqi, "RQSCUTEN": zeros, "RUCUTEN": ru, "RVCUTEN": rv,
        "RAINCV": raincv, "PRATEC": pratec, "KTYPE": out["ktype"],
    }


def step_tiedtke_column_jax(T, QV, QC, QI, P, P8W, DZ, RHO, PI, U, V, W,
                            QVFTEN, QVPBLTEN, QFX, XLAND, ZNU, dt, *, stepcu=5):
    """Return the frozen v0.6.0 ``PhysicsStepResult`` for one column (JAX kernel).

    Mirrors ``cumulus_tiedtke.step_tiedtke_column`` but uses the jit/vmap-traceable
    :func:`tiedtke_column_jax`. Imported lazily by the contract so that
    ``physics_interfaces`` need not pull JAX.
    """

    from gpuwrf.contracts.physics_interfaces import (
        PhysicsCarry, PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency,
    )

    out = tiedtke_column_jax(T, QV, QC, QI, P, P8W, DZ, RHO, PI, U, V, W,
                             QVFTEN, QVPBLTEN, QFX, XLAND, ZNU, dt, stepcu=stepcu)
    state_tendencies = {
        "theta": out["RTHCUTEN"],
        "qv": out["RQVCUTEN"],
        "qc": out["RQCCUTEN"],
        "qr": out["RQRCUTEN"],
        "qi": out["RQICUTEN"],
        "qs": out["RQSCUTEN"],
    }
    diagnostics = {
        "rthcuten": out["RTHCUTEN"], "rqvcuten": out["RQVCUTEN"],
        "rqccuten": out["RQCCUTEN"], "rqrcuten": out["RQRCUTEN"],
        "rqicuten": out["RQICUTEN"], "rqscuten": out["RQSCUTEN"],
        "rucuten": out["RUCUTEN"], "rvcuten": out["RVCUTEN"],
        "raincv": out["RAINCV"], "pratec": out["PRATEC"], "ktype": out["KTYPE"],
    }
    tendency = PhysicsTendency(
        state_tendencies=state_tendencies,
        accumulator_increments={"rainc_acc": out["RAINCV"]},
    )
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus={}),
        diagnostics=PhysicsDiagnostics(cumulus=diagnostics),
    )


__all__ = ["tiedtke_column_jax", "step_tiedtke_column_jax"]
