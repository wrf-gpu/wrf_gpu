"""GPU-batched (jit/vmap-traceable) WRF New-Tiedtke cumulus column kernel.

Mechanical control-flow translation of the validated fp64 NumPy reference
:mod:`gpuwrf.physics.cumulus_ntiedtke` (itself a line-faithful transcription of
pristine ``phys/physics_mmm/cu_ntiedtke.F90`` + ``phys/module_cu_ntiedtke.F``,
``cu_physics=16``).  The numerical algorithm is FROZEN: every constant
(including the fp32-rounded WRF default-REAL literals and the ``amax1`` REAL*4
demotion of the precip accumulator), every expression's operation order, every
data-dependent branch and every loop bound is reproduced exactly.  This module
only replaces Python control flow / in-place NumPy mutation with

* fixed-trip ``jax.lax.fori_loop`` over the static KLEV vertical sweeps,
* ``jnp.where`` masking of every data-dependent Fortran branch (Fortran
  ``break``/``continue``/early-exit becomes a carried active-flag that gates
  all subsequent writes -- semantically identical because the reference writes
  nothing once the flag is off), and
* functional ``.at[k].set(jnp.where(mask, new, arr[k]))`` updates (safe under
  out-of-range garbage indices in masked-out lanes: the old value is written
  back),

so one column traces to a single XLA graph and ``vmap``s across grid columns.

Internal convention (as the reference): Fortran 1-based TOP-DOWN levels
(index 1 = model top, ``klev`` = lowest level, ``klev+1`` = surface interface);
all internal arrays have physical length ``klev+2`` with a dead 0 slot (kept
exactly 0.0, matching the reference's ``_z1``) plus a dead ``klev+1`` slot for
level arrays.  fp64 throughout (``configure_jax_x64``).

House style: :mod:`gpuwrf.physics.cumulus_tiedtke_jax` (the cu=6 port).
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import numpy as np

import jax
import jax.numpy as jnp

configure_jax_x64()

# --- module cu_ntiedtke_common (verbatim constants, fp32-rounded literals) -----
_F = np.float64


def _f32(x):
    """WRF unsuffixed Fortran literal: default REAL (fp32), promoted to fp64."""
    return float(_F(np.float32(x)))


T13 = float(_F(np.float32(1.0) / np.float32(3.0)))
TMELT = _f32(273.16)
C1ES = _f32(610.78)
C3LES = _f32(17.2693882)
C3IES = _f32(21.875)
C4LES = _f32(35.86)
C4IES = _f32(7.66)

RTWAT = TMELT
RTBER = float(_F(np.float32(273.16) - np.float32(5.0)))
RTICE = float(_F(np.float32(273.16) - np.float32(23.0)))

MOMTRANS = 2
ENTRDD = _f32(2.0e-4)
CMFCMAX = 1.0
CMFCMIN = _f32(1.0e-10)
CMFDEPS = _f32(0.30)
ZDNOPRC = 2.0e4
CPRCON = _f32(1.4e-3)
PGCOEF = _f32(0.7)

NONEQUIL = True
LMFPEN = True
LMFMID = True
LMFSCV = True
LMFDD = True
LMFDUDV = True

# --- cu_ntiedtke_init derived constants (oracle-driver physical constants) ----
# Exactly the _Const defaults of the reference: cp=1004.5, rd=287.0, rv=461.6,
# xlv=2.5e6, xls=2.85e6, xlf=3.5e5, grav=9.81 (all fp64 Python floats, so the
# derived products below are the identical fp64 values).
ALF = 3.5e5
ALS = 2.85e6
ALV = 2.5e6
CPD = 1004.5
G = 9.81
RD = 287.0
RV = 461.6
RCPD = 1.0 / CPD
C2ES = C1ES * RD / RV
C5LES = C3LES * (TMELT - C4LES)
C5IES = C3IES * (TMELT - C4IES)
R5ALVCP = C5LES * ALV * RCPD
R5ALSCP = C5IES * ALS * RCPD
RALVDCP = ALV * RCPD
RALSDCP = ALS * RCPD
RALFDCP = ALF * RCPD
VTMPC1 = RV / RD - 1.0
ZRG = 1.0 / G

# fp32-rounded composites used inside the routines (reference in-line values).
_C_1P5X0P4 = float(_F(np.float32(1.5) * np.float32(0.4)))          # cutypen part1
_ZFACBUO = float(_F(np.float32(0.5) / (np.float32(1.0) + np.float32(0.5))))
_Z_CWDRAG = float(_F(np.float32(3.0) / np.float32(8.0) * np.float32(0.506)
                     / np.float32(0.2)))
_F32_1EM20 = _f32(1.0e-20)
_F32_0P001 = _f32(0.001)
_F32_1P2 = _f32(1.2)
_F32_0P8 = _f32(0.8)
_F32_2EM4 = _f32(2.0e-4)
_F32_5EM3 = _f32(5.0e-3)
_F32_1EM8 = _f32(1.0e-8)
_F32_0P2 = _f32(0.2)
_F32_1EM4 = _f32(1.0e-4)
_F32_1P75EM3 = _f32(1.75e-3)
_F32_0P80 = _f32(0.80)
_F32_0P75EM4 = _f32(0.75e-4)
_F32_0P4 = _f32(0.4)
_F32_1P6 = _f32(1.6)
_F32_0P3 = _f32(0.3)
_F32_1EM10 = _f32(1.0e-10)
_F32_1EM2 = _f32(1.0e-2)
_F32_21P18 = _f32(21.18)
_F32_0P1 = _f32(0.1)
_F32_5EM4 = _f32(5.0e-4)
_F32_3EM4 = _f32(3.0e-4)
_F32_0P05 = _f32(0.05)
_F32_5P44EM4 = _f32(5.44e-4)
_F32_0P7 = _f32(0.7)
_F32_0P9 = _f32(0.9)
_F32_5P09EM3 = _f32(5.09e-3)
_F32_0P5777 = _f32(0.5777)
_F32_0P01 = _f32(0.01)
_F32_0P98 = _f32(0.98)
_F32_1EM15 = _f32(1.0e-15)
_F32_1P06133 = _f32(1.06133)
_F32_1P33EM5 = _f32(1.33e-5)

_I32 = jnp.int32
_B = jnp.bool_


def _f64(x):
    return jnp.asarray(x, jnp.float64)


def _zeros(klev: int):
    """1-based fp64 zeros, physical length klev+2 (slots 0 and klev+1 dead)."""
    return jnp.zeros(klev + 2, dtype=jnp.float64)


def _izeros(klev: int):
    return jnp.zeros(klev + 2, dtype=jnp.int32)


def _mset(arr, idx, mask, val):
    """Masked functional write arr[idx] = val if mask (garbage-index safe)."""
    return arr.at[idx].set(jnp.where(mask, val, arr[idx]))


# --- saturation functions (verbatim) ------------------------------------------
def _foealfa(tt):
    return jnp.minimum(1.0, ((jnp.maximum(RTICE, jnp.minimum(RTWAT, tt)) - RTICE)
                             / (RTWAT - RTICE)) ** 2)


def _foelhm(tt):
    a = _foealfa(tt)
    return a * ALV + (1.0 - a) * ALS


def _foeewm(tt):
    a = _foealfa(tt)
    return C2ES * (a * jnp.exp(C3LES * (tt - TMELT) / (tt - C4LES))
                   + (1.0 - a) * jnp.exp(C3IES * (tt - TMELT) / (tt - C4IES)))


def _foedem(tt):
    a = _foealfa(tt)
    return (a * R5ALVCP * (1.0 / (tt - C4LES) ** 2)
            + (1.0 - a) * R5ALSCP * (1.0 / (tt - C4IES) ** 2))


def _foeldcpm(tt):
    a = _foealfa(tt)
    return a * RALVDCP + (1.0 - a) * RALSDCP


# --- cuadjtqn (functional, one level; kcall is a STATIC Python int) ------------
def _cuadjtqn_level(psp, t, q, ldflag, kcall: int):
    """Saturation adjustment at one level.  Returns (t_new, q_new).

    ``ldflag`` is a traced mask; kcall in {0, 1, 2} selects the same static
    branch structure as the reference (kcall=0 ignores ldflag there; the
    caller masks it).  All updates are masked so a False ``ldflag`` returns
    the inputs unchanged (identical to the Fortran early return).
    """
    if kcall == 1:
        zqp = 1.0 / psp
        zl = 1.0 / (t - C4LES)
        zi = 1.0 / (t - C4IES)
        alfa = _foealfa(t)
        zqsat = C2ES * (alfa * jnp.exp(C3LES * (t - TMELT) * zl)
                        + (1.0 - alfa) * jnp.exp(C3IES * (t - TMELT) * zi))
        zqsat = zqsat * zqp
        zqsat = jnp.minimum(0.5, zqsat)
        zcor = 1.0 - VTMPC1 * zqsat
        zf = (alfa * R5ALVCP * zl ** 2
              + (1.0 - alfa) * R5ALSCP * zi ** 2)
        zcond = (q * zcor ** 2 - zqsat * zcor) / (zcor ** 2 + zqsat * zf)
        do1 = ldflag & (zcond > 0.0)
        t1 = t + _foeldcpm(t) * zcond
        q1 = q - zcond
        zl1 = 1.0 / (t1 - C4LES)
        zi1 = 1.0 / (t1 - C4IES)
        alfa1 = _foealfa(t1)
        zqsat1 = C2ES * (alfa1 * jnp.exp(C3LES * (t1 - TMELT) * zl1)
                         + (1.0 - alfa1) * jnp.exp(C3IES * (t1 - TMELT) * zi1))
        zqsat1 = zqsat1 * zqp
        zqsat1 = jnp.minimum(0.5, zqsat1)
        zcor1 = 1.0 - VTMPC1 * zqsat1
        zf1 = (alfa1 * R5ALVCP * zl1 ** 2
               + (1.0 - alfa1) * R5ALSCP * zi1 ** 2)
        zcond1 = (q1 * zcor1 ** 2 - zqsat1 * zcor1) / (zcor1 ** 2 + zqsat1 * zf1)
        zcond1 = jnp.where(jnp.abs(zcond) < _F32_1EM20, 0.0, zcond1)
        t2 = t1 + _foeldcpm(t1) * zcond1
        q2 = q1 - zcond1
        return jnp.where(do1, t2, t), jnp.where(do1, q2, q)

    if kcall == 2:
        zqp = 1.0 / psp
        zqsat = _foeewm(t) * zqp
        zqsat = jnp.minimum(0.5, zqsat)
        zcor = 1.0 / (1.0 - VTMPC1 * zqsat)
        zqsat = zqsat * zcor
        zcond = (q - zqsat) / (1.0 + zqsat * zcor * _foedem(t))
        zcond = jnp.minimum(zcond, 0.0)
        t1 = t + _foeldcpm(t) * zcond
        q1 = q - zcond
        zqsat1 = _foeewm(t1) * zqp
        zqsat1 = jnp.minimum(0.5, zqsat1)
        zcor1 = 1.0 / (1.0 - VTMPC1 * zqsat1)
        zqsat1 = zqsat1 * zcor1
        zcond1 = (q1 - zqsat1) / (1.0 + zqsat1 * zcor1 * _foedem(t1))
        zcond1 = jnp.where(jnp.abs(zcond) < _F32_1EM20,
                           jnp.minimum(zcond1, 0.0), zcond1)
        t2 = t1 + _foeldcpm(t1) * zcond1
        q2 = q1 - zcond1
        return jnp.where(ldflag, t2, t), jnp.where(ldflag, q2, q)

    if kcall == 0:
        # kcall=0 ignores ldflag in the Fortran; caller masks via ldflag here.
        zqp = 1.0 / psp
        zqsat = _foeewm(t) * zqp
        zqsat = jnp.minimum(0.5, zqsat)
        zcor = 1.0 / (1.0 - VTMPC1 * zqsat)
        zqsat = zqsat * zcor
        zcond1 = (q - zqsat) / (1.0 + zqsat * zcor * _foedem(t))
        t1 = t + _foeldcpm(t) * zcond1
        q1 = q - zcond1
        zqsat1 = _foeewm(t1) * zqp
        zqsat1 = jnp.minimum(0.5, zqsat1)
        zcor1 = 1.0 / (1.0 - VTMPC1 * zqsat1)
        zqsat1 = zqsat1 * zcor1
        zcond2 = (q1 - zqsat1) / (1.0 + zqsat1 * zcor1 * _foedem(t1))
        t2 = t1 + _foeldcpm(t1) * zcond2
        q2 = q1 - zcond2
        return jnp.where(ldflag, t2, t), jnp.where(ldflag, q2, q)

    raise ValueError(f"unsupported kcall={kcall}")


# --- cuinin --------------------------------------------------------------------
def _cuinin_jax(klev, pten, pqen, pqsen, puen, pven, pverv, pgeo, paph, pgeoh):
    """Half-level interpolation + updraft/downdraft init (single column)."""
    klevm1 = klev - 1
    ptenh = _zeros(klev)
    pqenh = _zeros(klev)
    pqsenh = _zeros(klev)

    def body1(jk, carry):
        ptenh, pqenh, pqsenh = carry
        ptenh = ptenh.at[jk].set(
            (jnp.maximum(CPD * pten[jk - 1] + pgeo[jk - 1],
                         CPD * pten[jk] + pgeo[jk]) - pgeoh[jk]) * RCPD)
        pqenh = pqenh.at[jk].set(pqen[jk - 1])
        pqsenh = pqsenh.at[jk].set(pqsen[jk - 1])
        zph = paph[jk]
        # Fortran: cycle when jk >= klev-1 (jk < 2 impossible here)
        adj = ~(jk >= klev - 1)
        t_adj, q_adj = _cuadjtqn_level(zph, ptenh[jk], pqsenh[jk], adj, 0)
        ptenh = ptenh.at[jk].set(t_adj)
        pqsenh = pqsenh.at[jk].set(q_adj)
        val = (jnp.minimum(pqen[jk - 1], pqsen[jk - 1])
               + (pqsenh[jk] - pqsen[jk - 1]))
        val = jnp.maximum(val, 0.0)
        pqenh = _mset(pqenh, jk, adj, val)
        return (ptenh, pqenh, pqsenh)

    ptenh, pqenh, pqsenh = jax.lax.fori_loop(2, klev + 1, body1,
                                             (ptenh, pqenh, pqsenh))

    ptenh = ptenh.at[klev].set((CPD * pten[klev] + pgeo[klev] - pgeoh[klev]) * RCPD)
    pqenh = pqenh.at[klev].set(pqen[klev])
    ptenh = ptenh.at[1].set(pten[1])
    pqenh = pqenh.at[1].set(pqen[1])

    # smoothing sweep jk = klevm1 .. 2 (descending, sequential)
    def body2(idx, ptenh):
        jk = klevm1 - idx
        zzs = jnp.maximum(CPD * ptenh[jk] + pgeoh[jk],
                          CPD * ptenh[jk + 1] + pgeoh[jk + 1])
        return ptenh.at[jk].set((zzs - pgeoh[jk]) * RCPD)

    ptenh = jax.lax.fori_loop(0, klevm1 - 1, body2, ptenh)

    # klwmin: jk = klev .. 3 descending
    def body3(idx, carry):
        klwmin, zwmax = carry
        jk = klev - idx
        hit = pverv[jk] < zwmax
        zwmax = jnp.where(hit, pverv[jk], zwmax)
        klwmin = jnp.where(hit, _I32(jk), klwmin)
        return (klwmin, zwmax)

    klwmin, _ = jax.lax.fori_loop(0, klev - 2, body3, (_I32(klev), _f64(0.0)))

    ptu = _zeros(klev)
    ptd = _zeros(klev)
    pqu = _zeros(klev)
    pqd = _zeros(klev)
    plu = _zeros(klev)
    puu = _zeros(klev)
    pud = _zeros(klev)
    pvu = _zeros(klev)
    pvd = _zeros(klev)
    klab = _izeros(klev)

    def body4(jk, carry):
        ptu, ptd, pqu, pqd, puu, pud, pvu, pvd = carry
        ik = jnp.where(jk == 1, 1, jk - 1)
        ptu = ptu.at[jk].set(ptenh[jk])
        ptd = ptd.at[jk].set(ptenh[jk])
        pqu = pqu.at[jk].set(pqenh[jk])
        pqd = pqd.at[jk].set(pqenh[jk])
        puu = puu.at[jk].set(puen[ik])
        pud = pud.at[jk].set(puen[ik])
        pvu = pvu.at[jk].set(pven[ik])
        pvd = pvd.at[jk].set(pven[ik])
        return (ptu, ptd, pqu, pqd, puu, pud, pvu, pvd)

    ptu, ptd, pqu, pqd, puu, pud, pvu, pvd = jax.lax.fori_loop(
        1, klev + 1, body4, (ptu, ptd, pqu, pqd, puu, pud, pvu, pvd))

    return (ptenh, pqenh, pqsenh, klwmin, ptu, pqu, ptd, pqd,
            puu, pvu, pud, pvd, plu, klab)


# --- cutypen (first-guess updraft, cloud base + convection type) ---------------
def _cutypen_jax(klev, pqen, ptenh, pqenh, pqsenh, pgeoh, paph,
                 hfx, qfx, pgeo, pqsen, pap, pten, lndj,
                 cutu, cuqu, culab, culu):
    """Single-column cutypen (functional).  Returns
    (ldcum, cubot, cutop, ktype, wbase, kdpl, cutu, cuqu, culab, culu)."""
    klevm1 = klev - 1
    lev_idx = jnp.arange(klev + 2)
    lev_valid = (lev_idx >= 1) & (lev_idx <= klev)

    ktype = _I32(0)
    wbase = _f64(0.0)
    ldcum = _B(False)

    # local parcel arrays (reference copies the cuinin-seeded updraft state)
    # copies of the cuinin-seeded updraft state (JAX arrays are immutable, so
    # plain rebinding is an exact copy)
    ptu = cutu
    pqu = cuqu
    plu = culu
    klab = culab
    dh = _zeros(klev)
    dhen = _zeros(klev)
    kup = _zeros(klev)
    vptu = _zeros(klev)
    vten = _zeros(klev)
    zbuo = _zeros(klev)
    abuoy = _zeros(klev)

    kcbot = _I32(klev)
    kctop = _I32(klev)
    lldcum = _B(False)
    loflag = _B(True)

    # ---- shallow test parcel from the surface ---------------------------------
    def shallow_body(idx, carry):
        (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo, abuoy,
         loflag, lldcum, kcbot, kctop) = carry
        jk = klevm1 - idx
        seed = jk == klevm1

        rho = pap[klev] / (RD * (pten[klev] * (1.0 + VTMPC1 * pqen[klev])))
        part1 = _C_1P5X0P4 * pgeo[klev] / (rho * pten[klev])
        part2 = -hfx * RCPD - VTMPC1 * pten[klev] * qfx
        root = _F32_0P001 - part1 * part2
        part2neg = part2 < 0.0
        conw = _F32_1P2 * root ** T13
        deltt = jnp.maximum(1.5 * hfx / (rho * CPD * conw), 0.0)
        deltq = jnp.maximum(1.5 * qfx / (rho * conw), 0.0)
        do_seed = seed & part2neg
        kup = _mset(kup, klev, do_seed, 0.5 * conw ** 2)
        pqu = _mset(pqu, klev, do_seed, pqenh[klev] + deltq)
        dhen = _mset(dhen, klev, do_seed, pgeoh[klev] + ptenh[klev] * CPD)
        dh = _mset(dh, klev, do_seed, dhen[klev] + deltt * CPD)
        ptu = _mset(ptu, klev, do_seed, (dh[klev] - pgeoh[klev]) * RCPD)
        vptu = _mset(vptu, klev, do_seed,
                     ptu[klev] * (1.0 + VTMPC1 * pqu[klev]))
        vten = _mset(vten, klev, do_seed,
                     ptenh[klev] * (1.0 + VTMPC1 * pqenh[klev]))
        zbuo = _mset(zbuo, klev, do_seed,
                     (vptu[klev] - vten[klev]) / vten[klev])
        klab = _mset(klab, klev, do_seed, _I32(1))
        loflag = jnp.where(seed, part2neg, loflag)

        act = loflag  # `if not loflag: break`

        eta = _F32_0P8 / (pgeo[jk] * ZRG) + _F32_2EM4
        dz = (pgeoh[jk] - pgeoh[jk + 1]) * ZRG
        coef = 0.5 * eta * dz
        dhen = _mset(dhen, jk, act, pgeoh[jk] + CPD * ptenh[jk])
        dh = _mset(dh, jk, act, (coef * (dhen[jk + 1] + dhen[jk])
                                 + (1.0 - coef) * dh[jk + 1]) / (1.0 + coef))
        pqu = _mset(pqu, jk, act, (coef * (pqenh[jk + 1] + pqenh[jk])
                                   + (1.0 - coef) * pqu[jk + 1]) / (1.0 + coef))
        ptu = _mset(ptu, jk, act, (dh[jk] - pgeoh[jk]) * RCPD)
        zqold = pqu[jk]
        zph = paph[jk]

        t_adj, q_adj = _cuadjtqn_level(zph, ptu[jk], pqu[jk], act, 1)
        ptu = ptu.at[jk].set(t_adj)
        pqu = pqu.at[jk].set(q_adj)

        zdq = jnp.maximum(zqold - pqu[jk], 0.0)
        plu = _mset(plu, jk, act, plu[jk + 1] + zdq)
        zlglac = zdq * ((1.0 - _foealfa(ptu[jk]))
                        - (1.0 - _foealfa(ptu[jk + 1])))
        plu = _mset(plu, jk, act, jnp.minimum(plu[jk], _F32_5EM3))
        dh = _mset(dh, jk, act, pgeoh[jk] + CPD * (ptu[jk] + RALFDCP * zlglac))
        vptu = _mset(vptu, jk, act,
                     ptu[jk] * (1.0 + VTMPC1 * pqu[jk] - plu[jk])
                     + RALFDCP * zlglac)
        vten = _mset(vten, jk, act, ptenh[jk] * (1.0 + VTMPC1 * pqenh[jk]))
        zbuo = _mset(zbuo, jk, act, (vptu[jk] - vten[jk]) / vten[jk])
        abuoy = _mset(abuoy, jk, act, (zbuo[jk] + zbuo[jk + 1]) * 0.5 * G)
        atop1 = 1.0 - 2.0 * coef
        atop2 = 2.0 * dz * abuoy[jk]
        abot = 1.0 + 2.0 * coef
        kup = _mset(kup, jk, act, (atop1 * kup[jk + 1] + atop2) / abot)

        # cloud-base determination
        cb = act & (plu[jk] > 0.0) & (klab[jk + 1] == 1)
        ik = jk + 1
        zqsu = _foeewm(ptu[ik]) / paph[ik]
        zqsu = jnp.minimum(0.5, zqsu)
        zcor = 1.0 / (1.0 - VTMPC1 * zqsu)
        zqsu = zqsu * zcor
        zdq2 = jnp.minimum(0.0, pqu[ik] - zqsu)
        zalfaw = _foealfa(ptu[ik])
        zfacw = C5LES / ((ptu[ik] - C4LES) ** 2)
        zfaci = C5IES / ((ptu[ik] - C4IES) ** 2)
        zfac = zalfaw * zfacw + (1.0 - zalfaw) * zfaci
        zesdp = _foeewm(ptu[ik]) / paph[ik]
        zcor2 = 1.0 / (1.0 - VTMPC1 * zesdp)
        zdqsdt = zfac * zcor2 * zqsu
        zdtdp = RD * ptu[ik] / (CPD * paph[ik])
        zdp = zdq2 / (zdqsdt * zdtdp)
        zcbase = jnp.trunc(paph[ik] + zdp)  # Fortran INTEGER zcbase
        zpdifftop = zcbase - paph[jk]
        zpdiffbot = paph[jk + 1] - zcbase
        br1 = cb & (zpdifftop > zpdiffbot) & (kup[jk + 1] > 0.0)
        ikb = jnp.minimum(_I32(klev - 1), _I32(jk + 1))
        klab = _mset(klab, ikb, br1, _I32(2))
        klab = _mset(klab, jk, br1, _I32(2))
        kcbot = jnp.where(br1, ikb, kcbot)
        plu = _mset(plu, jk + 1, br1, _F32_1EM8)
        br2 = cb & (zpdifftop <= zpdiffbot) & (kup[jk] > 0.0)
        klab = _mset(klab, jk, br2, _I32(2))
        kcbot = jnp.where(br2, _I32(jk), kcbot)

        kupneg = act & (kup[jk] < 0.0)
        loflag = jnp.where(kupneg, _B(False), loflag)
        kctop = jnp.where(kupneg & (plu[jk + 1] > 0.0), _I32(jk), kctop)
        lldcum = jnp.where(kupneg, plu[jk + 1] > 0.0, lldcum)
        pos = act & ~kupneg
        klab = _mset(klab, jk, pos,
                     jnp.where(plu[jk] > 0.0, _I32(2), _I32(1)))

        return (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo, abuoy,
                loflag, lldcum, kcbot, kctop)

    carry = (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo, abuoy,
             loflag, lldcum, kcbot, kctop)
    carry = jax.lax.fori_loop(0, klev - 2, shallow_body, carry)
    (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo, abuoy,
     loflag, lldcum, kcbot, kctop) = carry

    ikb = kcbot
    ikt = kctop
    lldcum = jnp.where(paph[ikb] - paph[ikt] > ZDNOPRC, _B(False), lldcum)
    ktype = jnp.where(lldcum, _I32(2), ktype)
    ldcum = jnp.where(lldcum, _B(True), ldcum)
    wbase = jnp.where(lldcum, jnp.sqrt(jnp.maximum(2.0 * kup[ikb], 0.0)), wbase)
    cubot = jnp.where(lldcum, ikb, _I32(-1))
    cutop = jnp.where(lldcum, ikt, _I32(-1))
    kdpl = jnp.where(lldcum, _I32(klev), _I32(klev - 1))
    ldcum = jnp.where(lldcum, ldcum, _B(False))

    # writeback for jk >= kctop
    wb = lev_valid & (lev_idx >= kctop)
    culab = jnp.where(wb, klab, culab)
    cutu = jnp.where(wb, ptu, cutu)
    cuqu = jnp.where(wb, pqu, cuqu)
    culu = jnp.where(wb, plu, culu)

    # ---- deep test parcels from elevated departure levels ---------------------
    deltt_d = _F32_0P2
    deltq_d = _F32_1EM4
    deepflag = _B(False)

    def itop_body(idx, itoppacel):
        jk = klev - idx  # jk = klev .. 1
        hit = (paph[klev + 1] - paph[jk]) < 350.0e2
        return jnp.where(hit, _I32(jk), itoppacel)

    itoppacel = jax.lax.fori_loop(0, klev, itop_body, _I32(0))

    n_outer = (klevm1 - 1) - (klev // 2)  # trips of range(klevm1-1, klev//2, -1)

    def deep_outer(oidx, carry):
        (deepflag, ldcum, cubot, cutop, ktype, wbase, kdpl,
         cutu, cuqu, culab, culu) = carry
        levels = (klevm1 - 1) - oidx

        ptu = _zeros(klev)
        pqu = _zeros(klev)
        plu = _zeros(klev)
        dh = _zeros(klev)
        dhen = _zeros(klev)
        kup = _zeros(klev)
        vptu = _zeros(klev)
        vten = _zeros(klev)
        zbuo = _zeros(klev)
        abuoy = _zeros(klev)
        klab = _izeros(klev)

        kcbot = levels
        kctop = levels
        lldcum = _B(False)
        loflag = (~deepflag) & (levels >= itoppacel)

        def deep_inner(idx, icarry):
            (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo, abuoy,
             loflag, lldcum, kcbot, kctop) = icarry
            jk = levels - idx
            valid = jk >= 2
            act0 = loflag & valid  # `if not loflag: break`

            seed = act0 & (jk == levels)
            near = (paph[klev + 1] - paph[jk]) < 60.0e2
            tmix = _f64(0.0)
            qmix = _f64(0.0)
            zmix = _f64(0.0)
            pmix = _f64(0.0)
            for step in range(3):  # nk = jk+2, jk+1, jk
                nk = jk + 2 - step
                take = pmix < 50.0e2
                dp = paph[nk] - paph[nk - 1]
                tmix = jnp.where(take, tmix + dp * ptenh[nk], tmix)
                qmix = jnp.where(take, qmix + dp * pqenh[nk], qmix)
                zmix = jnp.where(take, zmix + dp * pgeoh[nk], zmix)
                pmix = jnp.where(take, pmix + dp, pmix)
            tmix = jnp.where(near, tmix / pmix, ptenh[jk + 1])
            qmix = jnp.where(near, qmix / pmix, pqenh[jk + 1])
            zmix = jnp.where(near, zmix / pmix, pgeoh[jk + 1])

            pqu = _mset(pqu, jk + 1, seed, qmix + deltq_d)
            dhen = _mset(dhen, jk + 1, seed, zmix + tmix * CPD)
            dh = _mset(dh, jk + 1, seed, dhen[jk + 1] + deltt_d * CPD)
            ptu = _mset(ptu, jk + 1, seed,
                        (dh[jk + 1] - pgeoh[jk + 1]) * RCPD)
            kup = _mset(kup, jk + 1, seed, 0.5)
            klab = _mset(klab, jk + 1, seed, _I32(1))
            vptu = _mset(vptu, jk + 1, seed,
                         ptu[jk + 1] * (1.0 + VTMPC1 * pqu[jk + 1]))
            vten = _mset(vten, jk + 1, seed,
                         ptenh[jk + 1] * (1.0 + VTMPC1 * pqenh[jk + 1]))
            zbuo = _mset(zbuo, jk + 1, seed,
                         (vptu[jk + 1] - vten[jk + 1]) / vten[jk + 1])

            act = act0
            fscale = jnp.minimum(1.0, (pqsen[jk] / pqsen[levels]) ** 3)
            eta = _F32_1P75EM3 * fscale
            dz = (pgeoh[jk] - pgeoh[jk + 1]) * ZRG
            coef = 0.5 * eta * dz
            dhen = _mset(dhen, jk, act, pgeoh[jk] + CPD * ptenh[jk])
            dh = _mset(dh, jk, act,
                       (coef * (dhen[jk + 1] + dhen[jk])
                        + (1.0 - coef) * dh[jk + 1]) / (1.0 + coef))
            pqu = _mset(pqu, jk, act,
                        (coef * (pqenh[jk + 1] + pqenh[jk])
                         + (1.0 - coef) * pqu[jk + 1]) / (1.0 + coef))
            ptu = _mset(ptu, jk, act, (dh[jk] - pgeoh[jk]) * RCPD)
            zqold = pqu[jk]
            zph = paph[jk]

            t_adj, q_adj = _cuadjtqn_level(zph, ptu[jk], pqu[jk], act, 1)
            ptu = ptu.at[jk].set(t_adj)
            pqu = pqu.at[jk].set(q_adj)

            zdq = jnp.maximum(zqold - pqu[jk], 0.0)
            plu = _mset(plu, jk, act, plu[jk + 1] + zdq)
            zlglac = zdq * ((1.0 - _foealfa(ptu[jk]))
                            - (1.0 - _foealfa(ptu[jk + 1])))
            plu = _mset(plu, jk, act, 0.5 * plu[jk])
            dh = _mset(dh, jk, act,
                       pgeoh[jk] + CPD * (ptu[jk] + RALFDCP * zlglac))
            vptu = _mset(vptu, jk, act,
                         ptu[jk] * (1.0 + VTMPC1 * pqu[jk] - plu[jk])
                         + RALFDCP * zlglac)
            vten = _mset(vten, jk, act,
                         ptenh[jk] * (1.0 + VTMPC1 * pqenh[jk]))
            zbuo = _mset(zbuo, jk, act, (vptu[jk] - vten[jk]) / vten[jk])
            abuoy = _mset(abuoy, jk, act, (zbuo[jk] + zbuo[jk + 1]) * 0.5 * G)
            atop1 = 1.0 - 2.0 * coef
            atop2 = 2.0 * dz * abuoy[jk]
            abot = 1.0 + 2.0 * coef
            kup = _mset(kup, jk, act, (atop1 * kup[jk + 1] + atop2) / abot)

            cb = act & (plu[jk] > 0.0) & (klab[jk + 1] == 1)
            ik = jk + 1
            zqsu = _foeewm(ptu[ik]) / paph[ik]
            zqsu = jnp.minimum(0.5, zqsu)
            zcor = 1.0 / (1.0 - VTMPC1 * zqsu)
            zqsu = zqsu * zcor
            zdq2 = jnp.minimum(0.0, pqu[ik] - zqsu)
            zalfaw = _foealfa(ptu[ik])
            zfacw = C5LES / ((ptu[ik] - C4LES) ** 2)
            zfaci = C5IES / ((ptu[ik] - C4IES) ** 2)
            zfac = zalfaw * zfacw + (1.0 - zalfaw) * zfaci
            zesdp = _foeewm(ptu[ik]) / paph[ik]
            zcor2 = 1.0 / (1.0 - VTMPC1 * zesdp)
            zdqsdt = zfac * zcor2 * zqsu
            zdtdp = RD * ptu[ik] / (CPD * paph[ik])
            zdp = zdq2 / (zdqsdt * zdtdp)
            zcbase = jnp.trunc(paph[ik] + zdp)  # Fortran INTEGER zcbase
            zpdifftop = zcbase - paph[jk]
            zpdiffbot = paph[jk + 1] - zcbase
            br1 = cb & (zpdifftop > zpdiffbot) & (kup[jk + 1] > 0.0)
            ikb = jnp.minimum(_I32(klev - 1), _I32(jk + 1))
            klab = _mset(klab, ikb, br1, _I32(2))
            klab = _mset(klab, jk, br1, _I32(2))
            kcbot = jnp.where(br1, ikb, kcbot)
            plu = _mset(plu, jk + 1, br1, _F32_1EM8)
            br2 = cb & (zpdifftop <= zpdiffbot) & (kup[jk] > 0.0)
            klab = _mset(klab, jk, br2, _I32(2))
            kcbot = jnp.where(br2, _I32(jk), kcbot)

            kupneg = act & (kup[jk] < 0.0)
            loflag = jnp.where(kupneg, _B(False), loflag)
            kctop = jnp.where(kupneg & (plu[jk + 1] > 0.0), _I32(jk), kctop)
            lldcum = jnp.where(kupneg, plu[jk + 1] > 0.0, lldcum)
            pos = act & ~kupneg
            klab = _mset(klab, jk, pos,
                         jnp.where(plu[jk] > 0.0, _I32(2), _I32(1)))

            return (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo,
                    abuoy, loflag, lldcum, kcbot, kctop)

        icarry = (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo, abuoy,
                  loflag, lldcum, kcbot, kctop)
        icarry = jax.lax.fori_loop(0, klev - 2, deep_inner, icarry)
        (ptu, pqu, plu, klab, dh, dhen, kup, vptu, vten, zbuo, abuoy,
         loflag, lldcum, kcbot, kctop) = icarry

        ikb = kcbot
        ikt = kctop
        lldcum = jnp.where(paph[ikb] - paph[ikt] < ZDNOPRC, _B(False), lldcum)
        trig = lldcum
        ktype = jnp.where(trig, _I32(1), ktype)
        ldcum = jnp.where(trig, _B(True), ldcum)
        deepflag = jnp.where(trig, _B(True), deepflag)
        wbase = jnp.where(trig, jnp.sqrt(jnp.maximum(2.0 * kup[ikb], 0.0)),
                          wbase)
        cubot = jnp.where(trig, ikb, cubot)
        cutop = jnp.where(trig, ikt, cutop)
        kdpl_new = levels + 1
        kdpl = jnp.where(trig, _I32(kdpl_new), kdpl)

        # writeback (resetflag): in [kctop, kdpl] copy parcel, else env, then
        # zero culab below cloud top.
        inrange = (lev_idx <= kdpl_new) & (lev_idx >= kctop)
        w_in = trig & lev_valid & inrange
        w_out = trig & lev_valid & (~inrange)
        culab = jnp.where(w_in, klab, jnp.where(w_out, _I32(1), culab))
        cutu = jnp.where(w_in, ptu, jnp.where(w_out, ptenh, cutu))
        cuqu = jnp.where(w_in, pqu, jnp.where(w_out, pqenh, cuqu))
        culu = jnp.where(w_in, plu, jnp.where(w_out, 0.0, culu))
        culab = jnp.where(trig & lev_valid & (lev_idx < kctop), _I32(0), culab)

        return (deepflag, ldcum, cubot, cutop, ktype, wbase, kdpl,
                cutu, cuqu, culab, culu)

    carry = (deepflag, ldcum, cubot, cutop, ktype, wbase, kdpl,
             cutu, cuqu, culab, culu)
    carry = jax.lax.fori_loop(0, n_outer, deep_outer, carry)
    (deepflag, ldcum, cubot, cutop, ktype, wbase, kdpl,
     cutu, cuqu, culab, culu) = carry

    return ldcum, cubot, cutop, ktype, wbase, kdpl, cutu, cuqu, culab, culu


# --- cuascn (cloud ascent for entraining plume) ---------------------------------
def _cuascn_jax(klev, ztmst, ptenh, pqenh, pten, pqen, pqsen, pgeo, pgeoh,
                pap, paph, pverv, ldcum, ktype, klab, ptu, pqu, plu, pmfub,
                kcbot, kctop, kctop0, lndj, wbase, kdpl):
    """Functional cuascn.  Returns a dict with the updated state (including
    the ptenh/pqenh half-level environment the Fortran mutates in place)."""
    klevm1 = klev - 1
    zcons2 = 3.0 / (G * ztmst)
    zfacbuo = _ZFACBUO
    zprcdgw = CPRCON * ZRG
    z_cldmax = _F32_5EM3
    z_cwifrac = 0.5
    z_cprc2 = 0.5
    z_cwdrag = _Z_CWDRAG

    lev_idx = jnp.arange(klev + 2)
    lev_valid = (lev_idx >= 1) & (lev_idx <= klev)

    # 2. set default values
    llo3 = _B(False)
    zluold = _f64(0.0)
    wup = _f64(0.0)
    zdpmean = _f64(0.0)
    zoentr = _f64(0.0)
    notld = ~ldcum
    ktype = jnp.where(notld, _I32(0), ktype)
    kcbot = jnp.where(notld, _I32(-1), kcbot)
    pmfub = jnp.where(notld, 0.0, pmfub)
    pqu = _mset(pqu, klev, notld, 0.0)

    plu = jnp.where(lev_valid & (lev_idx != kcbot), 0.0, plu)
    pmfu = _zeros(klev)
    pmfus = _zeros(klev)
    pmfuq = _zeros(klev)
    pmful = _zeros(klev)
    plude = _zeros(klev)
    plglac = _zeros(klev)
    pdmfup = _zeros(klev)
    zlrain = _zeros(klev)
    zbuo = _zeros(klev)
    kup = _zeros(klev)
    pmfude_rate = _zeros(klev)
    klab = jnp.where(lev_valid & (notld | (ktype == 3)), _I32(0), klab)

    def top_body(jk, kctop0):
        return jnp.where(notld & (paph[jk] < 4.0e4), _I32(jk), kctop0)

    kctop0 = jax.lax.fori_loop(1, klev + 1, top_body, kctop0)

    ldcum = jnp.where(ktype == 3, _B(False), ldcum)

    # 3. initialize values at cloud base level
    kctop = kcbot
    ikb = kcbot
    kup = _mset(kup, ikb, ldcum, 0.5 * wbase ** 2)
    pmfu = _mset(pmfu, ikb, ldcum, pmfub)
    pmfus = _mset(pmfus, ikb, ldcum, pmfub * (CPD * ptu[ikb] + pgeoh[ikb]))
    pmfuq = _mset(pmfuq, ikb, ldcum, pmfub * pqu[ikb])
    pmful = _mset(pmful, ikb, ldcum, pmfub * plu[ikb])

    # 4. do ascent (jk = klevm1 .. 3 descending)
    def asc_body(idx, carry):
        (ptenh, pqenh, klab, ptu, pqu, plu, pmfu, pmfus, pmfuq, pmful, plude,
         plglac, pdmfup, zlrain, zbuo, kup, pmfude_rate,
         llo3, zluold, wup, zdpmean, zoentr,
         ldcum, ktype, kcbot, kctop, pmfub) = carry
        jk = klevm1 - idx

        # cubasmcn: midlevel convection cloud-base
        mbas = ((~ldcum) & (klab[jk + 1] == 0)
                & (pqen[jk] > _F32_0P80 * pqsen[jk])
                & (pgeo[jk] * ZRG > 5.0e2)
                & (pgeo[jk] * ZRG < 1.0e4))
        ptu = _mset(ptu, jk + 1, mbas,
                    (CPD * pten[jk] + pgeo[jk] - pgeoh[jk + 1]) * RCPD)
        pqu = _mset(pqu, jk + 1, mbas, pqen[jk])
        plu = _mset(plu, jk + 1, mbas, 0.0)
        zzzmb = jnp.maximum(CMFCMIN, -pverv[jk] * ZRG)
        zzzmb = jnp.minimum(zzzmb, CMFCMAX)
        pmfub = jnp.where(mbas, zzzmb, pmfub)
        pmfu = _mset(pmfu, jk + 1, mbas, pmfub)
        pmfus = _mset(pmfus, jk + 1, mbas,
                      pmfub * (CPD * ptu[jk + 1] + pgeoh[jk + 1]))
        pmfuq = _mset(pmfuq, jk + 1, mbas, pmfub * pqu[jk + 1])
        pmful = _mset(pmful, jk + 1, mbas, 0.0)
        pdmfup = _mset(pdmfup, jk + 1, mbas, 0.0)
        kcbot = jnp.where(mbas, _I32(jk), kcbot)
        klab = _mset(klab, jk + 1, mbas, _I32(1))
        zlrain = _mset(zlrain, jk + 1, mbas, 0.0)
        ktype = jnp.where(mbas, _I32(3), ktype)

        zprecip = _f64(0.0)
        llo1 = _B(False)
        is_cnt = klab[jk + 1]
        klab = _mset(klab, jk, klab[jk + 1] == 0, _I32(0))
        loflag = ((ldcum & (klab[jk + 1] == 2))
                  | ((ktype == 3) & (klab[jk + 1] == 1)))
        zph = paph[jk]

        # ktype3 mass-flux cap at cloud base
        mcap = (ktype == 3) & (jk == kcbot)
        zmfmax_c = (paph[jk] - paph[jk - 1]) * zcons2
        over = mcap & (pmfub > zmfmax_c)
        zfac = zmfmax_c / pmfub
        pmfu = _mset(pmfu, jk + 1, over, pmfu[jk + 1] * zfac)
        pmfus = _mset(pmfus, jk + 1, over, pmfus[jk + 1] * zfac)
        pmfuq = _mset(pmfuq, jk + 1, over, pmfuq[jk + 1] * zfac)
        pmfub = jnp.where(over, zmfmax_c, pmfub)
        pmfub = jnp.where(mcap, jnp.minimum(pmfub, zmfmax_c), pmfub)

        llo3 = llo3 | (is_cnt > 0)

        # cuentrn: entrainment/detrainment rates
        zentr = _f64(0.0)
        zdz_e = (pgeoh[jk] - pgeoh[jk + 1]) * ZRG
        zmf_e = pmfu[jk + 1] * zdz_e
        m_ent = llo3 & ldcum & (jk < kcbot)
        zdmfen = jnp.where(m_ent, zentr * zmf_e, 0.0)
        zdmfde = jnp.where(m_ent, _F32_0P75EM4 * zmf_e, 0.0)

        # ---- llo3 block ----
        inA = llo3 & loflag
        zqold = _f64(0.0)

        zdmfde = jnp.where(inA, jnp.minimum(zdmfde, 0.75 * pmfu[jk + 1]),
                           zdmfde)
        at_base = inA & (jk == kcbot)
        zoentr_b = (-_F32_1P75EM3
                    * (jnp.minimum(1.0, pqen[jk] / pqsen[jk]) - 1.0)
                    * (pgeoh[jk] - pgeoh[jk + 1]) * ZRG)
        zoentr_b = jnp.minimum(_F32_0P4, zoentr_b) * pmfu[jk + 1]
        zoentr = jnp.where(at_base, zoentr_b, zoentr)

        below = inA & (jk < kcbot)
        zmfmax = (paph[jk] - paph[jk - 1]) * zcons2
        zxs = jnp.maximum(pmfu[jk + 1] - zmfmax, 0.0)
        wup = jnp.where(below, wup + kup[jk + 1] * (pap[jk + 1] - pap[jk]), wup)
        zdpmean = jnp.where(below, zdpmean + pap[jk + 1] - pap[jk], zdpmean)
        zdmfen_b = zoentr
        ge2 = ktype >= 2
        zdmfen_b = jnp.where(ge2, 2.0 * zdmfen_b, zdmfen_b)
        zdmfde_b = jnp.where(ge2, zdmfen_b, zdmfde)
        zdmfde_b = zdmfde_b * (_F32_1P6 - jnp.minimum(1.0, pqen[jk] / pqsen[jk]))
        zmftest = pmfu[jk + 1] + zdmfen_b - zdmfde_b
        zchange = jnp.maximum(zmftest - zmfmax, 0.0)
        zxe = jnp.maximum(zchange - zxs, 0.0)
        zdmfen_b = zdmfen_b - zxe
        zchange = zchange - zxe
        zdmfde_b = zdmfde_b + zchange
        zdmfen = jnp.where(below, zdmfen_b, zdmfen)
        zdmfde = jnp.where(below, zdmfde_b, zdmfde)

        pmfu = _mset(pmfu, jk, inA, pmfu[jk + 1] + zdmfen - zdmfde)
        zqeen = pqenh[jk + 1] * zdmfen
        zseen = (CPD * ptenh[jk + 1] + pgeoh[jk + 1]) * zdmfen
        zscde = (CPD * ptu[jk + 1] + pgeoh[jk + 1]) * zdmfde
        zqude = pqu[jk + 1] * zdmfde
        plude = _mset(plude, jk, inA, plu[jk + 1] * zdmfde)
        zmfusk = pmfus[jk + 1] + zseen - zscde
        zmfuqk = pmfuq[jk + 1] + zqeen - zqude
        zmfulk = pmful[jk + 1] - plude[jk]
        plu = _mset(plu, jk, inA, zmfulk * (1.0 / jnp.maximum(CMFCMIN, pmfu[jk])))
        pqu = _mset(pqu, jk, inA, zmfuqk * (1.0 / jnp.maximum(CMFCMIN, pmfu[jk])))
        ptu = _mset(ptu, jk, inA,
                    (zmfusk * (1.0 / jnp.maximum(CMFCMIN, pmfu[jk]))
                     - pgeoh[jk]) * RCPD)
        ptu = _mset(ptu, jk, inA, jnp.maximum(100.0, ptu[jk]))
        ptu = _mset(ptu, jk, inA, jnp.minimum(400.0, ptu[jk]))
        zqold = jnp.where(inA, pqu[jk], zqold)
        zlrain = _mset(zlrain, jk, inA,
                       zlrain[jk + 1] * (pmfu[jk + 1] - zdmfde)
                       * (1.0 / jnp.maximum(CMFCMIN, pmfu[jk])))
        zluold = jnp.where(inA, plu[jk], zluold)

        # reset to environmental values if below departure level
        mdpl0 = llo3 & (jk > kdpl)
        ptu = _mset(ptu, jk, mdpl0, ptenh[jk])
        pqu = _mset(pqu, jk, mdpl0, pqenh[jk])
        plu = _mset(plu, jk, mdpl0, 0.0)
        zluold = jnp.where(mdpl0, plu[jk], zluold)

        # cuadjtqn kcall=1 (double-gated on loflag in the reference)
        adj = llo3 & loflag
        t_adj, q_adj = _cuadjtqn_level(zph, ptu[jk], pqu[jk], adj, 1)
        ptu = ptu.at[jk].set(t_adj)
        pqu = pqu.at[jk].set(q_adj)

        # ---- block B (condensation / buoyancy / cloud top) ----
        inB = llo3 & loflag
        changed = pqu[jk] != zqold
        mB = inB & changed
        plglac = _mset(plglac, jk, mB,
                       plu[jk] * ((1.0 - _foealfa(ptu[jk]))
                                  - (1.0 - _foealfa(ptu[jk + 1]))))
        ptu = _mset(ptu, jk, mB, ptu[jk] + RALFDCP * plglac[jk])

        klab = _mset(klab, jk, mB, _I32(2))
        plu = _mset(plu, jk, mB, plu[jk] + zqold - pqu[jk])
        zbc = ptu[jk] * (1.0 + VTMPC1 * pqu[jk] - plu[jk + 1]
                         - zlrain[jk + 1])
        zbe = ptenh[jk] * (1.0 + VTMPC1 * pqenh[jk])
        zbuo = _mset(zbuo, jk, mB, zbc - zbe)

        m3 = mB & (ktype == 3) & (klab[jk + 1] == 1)
        m3a = m3 & (zbuo[jk] > -0.5)
        m3b = m3 & ~(zbuo[jk] > -0.5)
        ldcum = ldcum | m3a
        kctop = jnp.where(m3a, _I32(jk), kctop)
        kup = _mset(kup, jk, m3a, 0.5)
        klab = _mset(klab, jk, m3b, _I32(0))
        pmfu = _mset(pmfu, jk, m3b, 0.0)
        plude = _mset(plude, jk, m3b, 0.0)
        plu = _mset(plu, jk, m3b, 0.0)

        m2 = mB & (klab[jk + 1] == 2)
        neg = m2 & (zbuo[jk] < 0.0)
        ptenh = _mset(ptenh, jk, neg, 0.5 * (pten[jk] + pten[jk - 1]))
        pqenh = _mset(pqenh, jk, neg, 0.5 * (pqen[jk] + pqen[jk - 1]))
        zbuo = _mset(zbuo, jk, neg,
                     zbc - ptenh[jk] * (1.0 + VTMPC1 * pqenh[jk]))
        zbuoc = ((zbuo[jk]
                  / (ptenh[jk] * (1.0 + VTMPC1 * pqenh[jk]))
                  + zbuo[jk + 1]
                  / (ptenh[jk + 1] * (1.0 + VTMPC1 * pqenh[jk + 1])))
                 * 0.5)
        zdkbuo = (pgeoh[jk] - pgeoh[jk + 1]) * zfacbuo * zbuoc
        zdken = jnp.where(
            zdmfen > 0.0,
            jnp.minimum(1.0, (1.0 + z_cwdrag) * zdmfen
                        / jnp.maximum(CMFCMIN, pmfu[jk + 1])),
            jnp.minimum(1.0, (1.0 + z_cwdrag) * zdmfde
                        / jnp.maximum(CMFCMIN, pmfu[jk + 1])))
        kup = _mset(kup, jk, m2,
                    (kup[jk + 1] * (1.0 - zdken) + zdkbuo) / (1.0 + zdken))
        neg2 = m2 & (zbuo[jk] < 0.0)
        zkedke = kup[jk] / jnp.maximum(_F32_1EM10, kup[jk + 1])
        zkedke = jnp.maximum(0.0, jnp.minimum(1.0, zkedke))
        zmfun = jnp.sqrt(zkedke) * pmfu[jk + 1]
        zdmfde = jnp.where(neg2, jnp.maximum(zdmfde, pmfu[jk + 1] - zmfun),
                           zdmfde)
        plude = _mset(plude, jk, neg2, plu[jk + 1] * zdmfde)
        pmfu = _mset(pmfu, jk, neg2, pmfu[jk + 1] + zdmfen - zdmfde)
        mo = m2 & (zbuo[jk] > -_F32_0P2)
        ikb = kcbot
        zoentr_o = (_F32_1P75EM3
                    * (_F32_0P3
                       - (jnp.minimum(1.0, pqen[jk - 1] / pqsen[jk - 1]) - 1.0))
                    * (pgeoh[jk - 1] - pgeoh[jk]) * ZRG
                    * jnp.minimum(1.0, pqsen[jk] / pqsen[ikb]) ** 3)
        zoentr_o = jnp.minimum(_F32_0P4, zoentr_o) * pmfu[jk]
        zoentr = jnp.where(m2, jnp.where(mo, zoentr_o, 0.0), zoentr)
        mdpl = m2 & (jk > kdpl)
        pmfu = _mset(pmfu, jk, mdpl, pmfu[jk + 1])
        kup = _mset(kup, jk, mdpl, 0.5)
        mpos = m2 & (kup[jk] > 0.0) & (pmfu[jk] > 0.0)
        kctop = jnp.where(mpos, _I32(jk), kctop)
        llo1 = llo1 | mpos
        mneg = m2 & ~((kup[jk] > 0.0) & (pmfu[jk] > 0.0))
        klab = _mset(klab, jk, mneg, _I32(0))
        pmfu = _mset(pmfu, jk, mneg, 0.0)
        kup = _mset(kup, jk, mneg, 0.0)
        zdmfde = jnp.where(mneg, pmfu[jk + 1], zdmfde)
        plude = _mset(plude, jk, mneg, plu[jk + 1] * zdmfde)
        mrate = m2 & (pmfu[jk + 1] > 0.0)
        pmfude_rate = _mset(pmfude_rate, jk, mrate, zdmfde)

        mE = inB & (~changed) & (ktype == 2)
        klab = _mset(klab, jk, mE, _I32(0))
        pmfu = _mset(pmfu, jk, mE, 0.0)
        kup = _mset(kup, jk, mE, 0.0)
        zdmfde = jnp.where(mE, pmfu[jk + 1], zdmfde)
        plude = _mset(plude, jk, mE, plu[jk + 1] * zdmfde)
        pmfude_rate = _mset(pmfude_rate, jk, mE, zdmfde)

        # ---- precipitation conversion (Sundqvist) ----
        zdshrd = jnp.where(lndj == 1, _F32_5EM4, _F32_3EM4)
        mP = llo1 & (plu[jk] > zdshrd)
        zwu = jnp.minimum(15.0, jnp.sqrt(2.0 * jnp.maximum(_F32_0P1,
                                                           kup[jk + 1])))
        zprcon = zprcdgw / (0.75 * zwu)
        zdt = jnp.minimum(RTBER - RTICE,
                          jnp.maximum(RTBER - ptu[jk], 0.0))
        zcbf = 1.0 + z_cprc2 * jnp.sqrt(zdt)
        zzco = zprcon * zcbf
        zlcrit = zdshrd / zcbf
        zdfi = pgeoh[jk] - pgeoh[jk + 1]
        zc = plu[jk] - zluold
        zarg = (plu[jk] / zlcrit) ** 2
        zd = jnp.where(zarg < 25.0, zzco * (1.0 - jnp.exp(-zarg)) * zdfi,
                       zzco * zdfi)
        zint = jnp.exp(-zd)
        zlnew = zluold * zint + zc / zd * (1.0 - zint)
        zlnew = jnp.maximum(0.0, jnp.minimum(plu[jk], zlnew))
        zlnew = jnp.minimum(z_cldmax, zlnew)
        zprecip_p = jnp.maximum(0.0, zluold + zc - zlnew)
        pdmfup = _mset(pdmfup, jk, mP, zprecip_p * pmfu[jk])
        zlrain = _mset(zlrain, jk, mP, zlrain[jk] + zprecip_p)
        plu = _mset(plu, jk, mP, zlnew)
        zprecip = jnp.where(mP, zprecip_p, zprecip)

        # ---- rain loading / fallout ----
        mR = llo1 & (zlrain[jk] > 0.0)
        zvw = _F32_21P18 * zlrain[jk] ** _F32_0P2
        zvi = z_cwifrac * zvw
        zalfaw_r = _foealfa(ptu[jk])
        zvv = zalfaw_r * zvw + (1.0 - zalfaw_r) * zvi
        zrold = zlrain[jk] - zprecip
        zc_r = zprecip
        zwu_r = jnp.minimum(15.0, jnp.sqrt(2.0 * jnp.maximum(_F32_0P1,
                                                             kup[jk])))
        zd_r = zvv / zwu_r
        zint_r = jnp.exp(-zd_r)
        zrnew = zrold * zint_r + zc_r / zd_r * (1.0 - zint_r)
        zrnew = jnp.maximum(0.0, jnp.minimum(zlrain[jk], zrnew))
        zlrain = _mset(zlrain, jk, mR, zrnew)

        # ---- flux updates ----
        mF = llo3 & loflag
        pmful = _mset(pmful, jk, mF, plu[jk] * pmfu[jk])
        pmfus = _mset(pmfus, jk, mF, (CPD * ptu[jk] + pgeoh[jk]) * pmfu[jk])
        pmfuq = _mset(pmfuq, jk, mF, pqu[jk] * pmfu[jk])

        return (ptenh, pqenh, klab, ptu, pqu, plu, pmfu, pmfus, pmfuq, pmful,
                plude, plglac, pdmfup, zlrain, zbuo, kup, pmfude_rate,
                llo3, zluold, wup, zdpmean, zoentr,
                ldcum, ktype, kcbot, kctop, pmfub)

    carry = (ptenh, pqenh, klab, ptu, pqu, plu, pmfu, pmfus, pmfuq, pmful,
             plude, plglac, pdmfup, zlrain, zbuo, kup, pmfude_rate,
             llo3, zluold, wup, zdpmean, zoentr,
             ldcum, ktype, kcbot, kctop, pmfub)
    carry = jax.lax.fori_loop(0, klev - 3, asc_body, carry)
    (ptenh, pqenh, klab, ptu, pqu, plu, pmfu, pmfus, pmfuq, pmful,
     plude, plglac, pdmfup, zlrain, zbuo, kup, pmfude_rate,
     llo3, zluold, wup, zdpmean, zoentr,
     ldcum, ktype, kcbot, kctop, pmfub) = carry

    # 5. final calculations
    ldcum = jnp.where(kctop == -1, _B(False), ldcum)
    kcbot = jnp.maximum(kcbot, kctop)
    wup_f = jnp.maximum(_F32_1EM2, wup / jnp.maximum(1.0, zdpmean))
    wup_f = jnp.sqrt(2.0 * wup_f)
    wup = jnp.where(ldcum, wup_f, wup)

    return {
        "ldcum": ldcum, "ktype": ktype, "kcbot": kcbot, "kctop": kctop,
        "kctop0": kctop0, "pmfub": pmfub, "wup": wup,
        "plglac": plglac, "pmfude_rate": pmfude_rate,
        "ptenh": ptenh, "pqenh": pqenh, "klab": klab,
        "ptu": ptu, "pqu": pqu, "plu": plu,
        "pmfu": pmfu, "pmfus": pmfus, "pmfuq": pmfuq, "pmful": pmful,
        "plude": plude, "pdmfup": pdmfup,
    }


# --- cudlfsn (level of free sinking) --------------------------------------------
def _cudlfsn_jax(klev, kcbot, kctop, ldcum, ptenh, pqenh, pten, pqsen,
                 pgeo, pgeoh, paph, ptu, pqu, pmfub, prfl, ptd, pqd):
    """Functional cudlfsn.  Returns
    (pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl, ptd, pqd)."""
    pmfd = _zeros(klev)
    pmfds = _zeros(klev)
    pmfdq = _zeros(klev)
    pdmfdp = _zeros(klev)

    lddraf = _B(False)
    kdtop = _I32(klev + 1)
    ikhsmin = _I32(klev + 1)
    zhsmin = _f64(1.0e8)

    def hs_body(jk, carry):
        zhsmin, ikhsmin = carry
        zhsk = (CPD * pten[jk] + pgeo[jk]
                + _foelhm(pten[jk]) * pqsen[jk])
        hit = zhsk < zhsmin
        ikhsmin = jnp.where(hit, _I32(jk), ikhsmin)
        zhsmin = jnp.where(hit, zhsk, zhsmin)
        return (zhsmin, ikhsmin)

    zhsmin, ikhsmin = jax.lax.fori_loop(3, klev - 1, hs_body,
                                        (zhsmin, ikhsmin))

    ike = klev - 3

    def lfs_body(jk, carry):
        ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl = carry
        ztenwb = ptenh[jk]
        zqenwb = pqenh[jk]
        zph = paph[jk]
        llo2 = (ldcum & (prfl > 0.0) & (~lddraf)
                & ((jk < kcbot) & (jk > kctop)) & (jk >= ikhsmin))

        t_wb, q_wb = _cuadjtqn_level(zph, ztenwb, zqenwb, llo2, 2)

        zttest = 0.5 * (ptu[jk] + t_wb)
        zqtest = 0.5 * (pqu[jk] + q_wb)
        zbuo = (zttest * (1.0 + VTMPC1 * zqtest)
                - ptenh[jk] * (1.0 + VTMPC1 * pqenh[jk]))
        zcond = pqenh[jk] - q_wb
        zmftop = -CMFDEPS * pmfub
        trig = llo2 & (zbuo < 0.0) & (prfl > 10.0 * zmftop * zcond)
        kdtop = jnp.where(trig, _I32(jk), kdtop)
        lddraf = lddraf | trig
        ptd = _mset(ptd, jk, trig, zttest)
        pqd = _mset(pqd, jk, trig, zqtest)
        pmfd = _mset(pmfd, jk, trig, zmftop)
        pmfds = _mset(pmfds, jk, trig, pmfd[jk] * (CPD * ptd[jk] + pgeoh[jk]))
        pmfdq = _mset(pmfdq, jk, trig, pmfd[jk] * pqd[jk])
        pdmfdp_v = -0.5 * pmfd[jk] * zcond
        pdmfdp = _mset(pdmfdp, jk - 1, trig, pdmfdp_v)
        prfl = jnp.where(trig, prfl + pdmfdp[jk - 1], prfl)
        return (ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl)

    carry = (ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl)
    carry = jax.lax.fori_loop(3, ike + 1, lfs_body, carry)
    ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl = carry

    return pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl, ptd, pqd


# --- cuddrafn (downdraft descent) ------------------------------------------------
def _cuddrafn_jax(klev, lddraf, ptenh, pqenh, pgeo, pgeoh, paph, prfl,
                  ptd, pqd, pmfu, pmfd, pmfds, pmfdq, pdmfdp):
    """Functional cuddrafn.  Returns
    (prfl, pmfdde_rate, ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp)."""
    pmfdde_rate = _zeros(klev)

    def itop_body(idx, itopde):
        jk = klev - idx  # jk = klev .. 1
        hit = (paph[klev + 1] - paph[jk]) < 60.0e2
        return jnp.where(hit, _I32(jk), itopde)

    itopde = jax.lax.fori_loop(0, klev, itop_body, _I32(0))

    def dd_body(jk, carry):
        (ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, pmfdde_rate,
         prfl, zoentr, zbuoy) = carry
        zph = paph[jk]
        llo2 = lddraf & (pmfd[jk - 1] < 0.0)

        zentr = (ENTRDD * pmfd[jk - 1]
                 * (pgeoh[jk - 1] - pgeoh[jk]) * ZRG)
        zdmfen = zentr
        zdmfde = zentr

        above = jk > itopde
        zdmfen = jnp.where(above, 0.0, zdmfen)
        zdmfde = jnp.where(above,
                           pmfd[itopde] * (paph[jk] - paph[jk - 1])
                           / (paph[klev + 1] - paph[itopde]),
                           zdmfde)

        ble = jk <= itopde
        zdz = -(pgeoh[jk - 1] - pgeoh[jk]) * ZRG
        zzentr = zoentr * zdz * pmfd[jk - 1]
        zdmfen_c = zdmfen + zzentr
        zdmfen_c = jnp.maximum(zdmfen_c, _F32_0P3 * pmfd[jk - 1])
        zdmfen_c = jnp.maximum(zdmfen_c,
                               -0.75 * pmfu[jk] - (pmfd[jk - 1] - zdmfde))
        zdmfen_c = jnp.minimum(zdmfen_c, 0.0)
        zdmfen = jnp.where(ble, zdmfen_c, zdmfen)

        pmfd = _mset(pmfd, jk, llo2, pmfd[jk - 1] + zdmfen - zdmfde)
        zseen = (CPD * ptenh[jk - 1] + pgeoh[jk - 1]) * zdmfen
        zqeen = pqenh[jk - 1] * zdmfen
        zsdde = (CPD * ptd[jk - 1] + pgeoh[jk - 1]) * zdmfde
        zqdde = pqd[jk - 1] * zdmfde
        zmfdsk = pmfds[jk - 1] + zseen - zsdde
        zmfdqk = pmfdq[jk - 1] + zqeen - zqdde
        pqd = _mset(pqd, jk, llo2,
                    zmfdqk * (1.0 / jnp.minimum(-CMFCMIN, pmfd[jk])))
        ptd = _mset(ptd, jk, llo2,
                    (zmfdsk * (1.0 / jnp.minimum(-CMFCMIN, pmfd[jk]))
                     - pgeoh[jk]) * RCPD)
        ptd = _mset(ptd, jk, llo2, jnp.minimum(400.0, ptd[jk]))
        ptd = _mset(ptd, jk, llo2, jnp.maximum(100.0, ptd[jk]))
        zcond = pqd[jk]

        t_adj, q_adj = _cuadjtqn_level(zph, ptd[jk], pqd[jk], llo2, 2)
        ptd = ptd.at[jk].set(t_adj)
        pqd = pqd.at[jk].set(q_adj)

        zcond = zcond - pqd[jk]
        zbuo = (ptd[jk] * (1.0 + VTMPC1 * pqd[jk])
                - ptenh[jk] * (1.0 + VTMPC1 * pqenh[jk]))
        rain = (prfl > 0.0) & (pmfu[jk] > 0.0)
        zrain = prfl / pmfu[jk]
        zbuo = jnp.where(rain, zbuo - ptd[jk] * zrain, zbuo)
        kill = (zbuo >= 0.0) | (prfl <= (pmfd[jk] * zcond))
        pmfd = _mset(pmfd, jk, llo2 & kill, 0.0)
        zbuo = jnp.where(kill, 0.0, zbuo)
        pmfds = _mset(pmfds, jk, llo2, (CPD * ptd[jk] + pgeoh[jk]) * pmfd[jk])
        pmfdq = _mset(pmfdq, jk, llo2, pqd[jk] * pmfd[jk])
        zdmfdp = -pmfd[jk] * zcond
        pdmfdp = _mset(pdmfdp, jk - 1, llo2, zdmfdp)
        prfl = jnp.where(llo2, prfl + zdmfdp, prfl)

        zbuoyz = zbuo / ptenh[jk]
        zbuoyz = jnp.minimum(zbuoyz, 0.0)
        zdz2 = -(pgeo[jk - 1] - pgeo[jk])
        zbuoy = jnp.where(llo2, zbuoy + zbuoyz * zdz2, zbuoy)
        zoentr = jnp.where(llo2, G * zbuoyz * 0.5 / (1.0 + zbuoy), zoentr)
        pmfdde_rate = _mset(pmfdde_rate, jk, llo2, -zdmfde)
        return (ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, pmfdde_rate,
                prfl, zoentr, zbuoy)

    carry = (ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, pmfdde_rate,
             prfl, _f64(0.0), _f64(0.0))
    carry = jax.lax.fori_loop(3, klev + 1, dd_body, carry)
    (ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp, pmfdde_rate,
     prfl, zoentr, zbuoy) = carry

    return prfl, pmfdde_rate, ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp


# --- cuflxn (final convective fluxes) --------------------------------------------
def _cuflxn_jax(klev, ztmst, pten, pqen, pqsen, ptenh, pqenh, paph, pap,
                pgeoh, lndj, ldcum, kcbot, kctop, kdtop, ktype, lddraf,
                pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plude,
                pdmfup, pdmfdp, plglac, pmfdde_rate):
    """Functional cuflxn.  ktopm2 is the STATIC 2 of the reference.
    Returns (ktopm2, ktype, lddraf, prain, pdpmel, pmflxr, pmflxs) plus all
    threaded arrays (incl. the mutated pqsen)."""
    ztaumel = 18000.0
    zcons1a = CPD / (ALF * G * ztaumel)
    zcons2 = 3.0 / (G * ztmst)
    zcucov = _F32_0P05
    zcpecons = _F32_5P44EM4 / G

    pdpmel = _zeros(klev)
    pmflxr = _zeros(klev)  # physical length klev+2: slots 1..klev+1
    pmflxs = _zeros(klev)

    prain = _f64(0.0)
    lddraf = jnp.where((~ldcum) | (kdtop < kctop), _B(False), lddraf)
    ktype = jnp.where(~ldcum, _I32(0), ktype)
    idbas = _I32(klev)
    rhevap = jnp.where(lndj == 1, _F32_0P7, _F32_0P9)

    ktopm2 = 2

    def flx1_body(jk, carry):
        (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plglac,
         pdmfup, pdmfdp, plude, idbas) = carry
        ikb = jnp.minimum(_I32(jk + 1), _I32(klev))
        mc = ldcum & (jk >= kctop)
        pmfus = _mset(pmfus, jk, mc,
                      pmfus[jk] - pmfu[jk] * (CPD * ptenh[jk] + pgeoh[jk]))
        pmfuq = _mset(pmfuq, jk, mc, pmfuq[jk] - pmfu[jk] * pqenh[jk])
        plglac = _mset(plglac, jk, mc, pmfu[jk] * plglac[jk])
        llddraf = lddraf & (jk >= kdtop)
        mdd = mc & llddraf & (jk >= kdtop)
        pmfds = _mset(pmfds, jk, mdd,
                      pmfds[jk] - pmfd[jk] * (CPD * ptenh[jk] + pgeoh[jk]))
        pmfdq = _mset(pmfdq, jk, mdd, pmfdq[jk] - pmfd[jk] * pqenh[jk])
        mndd = mc & ~(llddraf & (jk >= kdtop))
        pmfd = _mset(pmfd, jk, mndd, 0.0)
        pmfds = _mset(pmfds, jk, mndd, 0.0)
        pmfdq = _mset(pmfdq, jk, mndd, 0.0)
        pdmfdp = _mset(pdmfdp, jk - 1, mndd, 0.0)
        midb = (mc & llddraf & (pmfd[jk] < 0.0)
                & (jnp.abs(pmfd[ikb]) < _F32_1EM20))
        idbas = jnp.where(midb, _I32(jk), idbas)
        mz = ~mc
        pmfu = _mset(pmfu, jk, mz, 0.0)
        pmfd = _mset(pmfd, jk, mz, 0.0)
        pmfus = _mset(pmfus, jk, mz, 0.0)
        pmfds = _mset(pmfds, jk, mz, 0.0)
        pmfuq = _mset(pmfuq, jk, mz, 0.0)
        pmfdq = _mset(pmfdq, jk, mz, 0.0)
        pmful = _mset(pmful, jk, mz, 0.0)
        plglac = _mset(plglac, jk, mz, 0.0)
        pdmfup = _mset(pdmfup, jk - 1, mz, 0.0)
        pdmfdp = _mset(pdmfdp, jk - 1, mz, 0.0)
        plude = _mset(plude, jk - 1, mz, 0.0)
        return (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plglac,
                pdmfup, pdmfdp, plude, idbas)

    carry = (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plglac,
             pdmfup, pdmfdp, plude, idbas)
    carry = jax.lax.fori_loop(ktopm2, klev + 1, flx1_body, carry)
    (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plglac,
     pdmfup, pdmfdp, plude, idbas) = carry

    # cloud-base interpolation to the level below
    ikb = kcbot
    ik = ikb + 1
    zzp = ((paph[klev + 1] - paph[ik]) / (paph[klev + 1] - paph[ikb]))
    zzp = jnp.where(ktype == 3, zzp ** 2, zzp)
    pmfu = _mset(pmfu, ik, ldcum, pmfu[ikb] * zzp)
    pmfus = _mset(pmfus, ik, ldcum,
                  (pmfus[ikb] - _foelhm(ptenh[ikb]) * pmful[ikb]) * zzp)
    pmfuq = _mset(pmfuq, ik, ldcum, (pmfuq[ikb] + pmful[ikb]) * zzp)
    pmful = _mset(pmful, ik, ldcum, 0.0)

    def flx2_body(jk, carry):
        (pmfu, pmfus, pmfuq, pmful, pmfd, pmfds, pmfdq, pmfdde_rate) = carry
        mx = ldcum & (jk > kcbot + 1)
        ikb2 = kcbot + 1
        zzp = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ikb2]))
        zzp = jnp.where(ktype == 3, zzp ** 2, zzp)
        pmfu = _mset(pmfu, jk, mx, pmfu[ikb2] * zzp)
        pmfus = _mset(pmfus, jk, mx, pmfus[ikb2] * zzp)
        pmfuq = _mset(pmfuq, jk, mx, pmfuq[ikb2] * zzp)
        pmful = _mset(pmful, jk, mx, 0.0)
        ikd = idbas
        llddraf = lddraf & (jk > ikd) & (ikd < klev)
        mA = llddraf & (ikd == kcbot + 1)
        zzp2 = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ikd]))
        zzp2 = jnp.where(ktype == 3, zzp2 * zzp2, zzp2)
        pmfd = _mset(pmfd, jk, mA, pmfd[ikd] * zzp2)
        pmfds = _mset(pmfds, jk, mA, pmfds[ikd] * zzp2)
        pmfdq = _mset(pmfdq, jk, mA, pmfdq[ikd] * zzp2)
        pmfdde_rate = _mset(pmfdde_rate, jk, mA, -(pmfd[jk - 1] - pmfd[jk]))
        mB = llddraf & (ikd != kcbot + 1) & (jk == ikd + 1)
        pmfdde_rate = _mset(pmfdde_rate, jk, mB, -(pmfd[jk - 1] - pmfd[jk]))
        return (pmfu, pmfus, pmfuq, pmful, pmfd, pmfds, pmfdq, pmfdde_rate)

    carry = (pmfu, pmfus, pmfuq, pmful, pmfd, pmfds, pmfdq, pmfdde_rate)
    carry = jax.lax.fori_loop(ktopm2, klev + 1, flx2_body, carry)
    (pmfu, pmfus, pmfuq, pmful, pmfd, pmfds, pmfdq, pmfdde_rate) = carry

    # 2. rain/snow fall rates, melting
    def rain_body(jk, carry):
        (prain, pdpmel, pqsen, plglac, pmflxr, pmflxs, pdmfdp) = carry
        mr = ldcum & (jk >= kctop - 1)
        prain = jnp.where(mr, prain + pdmfup[jk], prain)
        melt = mr & (pmflxs[jk] > 0.0) & (pten[jk] > TMELT)
        zcons1 = zcons1a * (1.0 + 0.5 * (pten[jk] - TMELT))
        zfac = zcons1 * (paph[jk + 1] - paph[jk])
        zsnmlt = jnp.minimum(pmflxs[jk], zfac * (pten[jk] - TMELT))
        pdpmel = _mset(pdpmel, jk, melt, zsnmlt)
        pqsen = _mset(pqsen, jk, melt,
                      _foeewm(pten[jk] - zsnmlt / zfac) / pap[jk])
        zalfaw = _foealfa(pten[jk])
        frz = mr & (pten[jk] < TMELT) & (zalfaw > 0.0)
        plglac = _mset(plglac, jk, frz,
                       plglac[jk] + zalfaw * (pdmfup[jk] + pdmfdp[jk]))
        zalfaw = jnp.where(frz, 0.0, zalfaw)
        pmflxr = _mset(pmflxr, jk + 1, mr,
                       pmflxr[jk] + zalfaw * (pdmfup[jk] + pdmfdp[jk])
                       + pdpmel[jk])
        pmflxs = _mset(pmflxs, jk + 1, mr,
                       pmflxs[jk] + (1.0 - zalfaw) * (pdmfup[jk] + pdmfdp[jk])
                       - pdpmel[jk])
        nboth = mr & (pmflxr[jk + 1] + pmflxs[jk + 1] < 0.0)
        pdmfdp = _mset(pdmfdp, jk, nboth,
                       -(pmflxr[jk] + pmflxs[jk] + pdmfup[jk]))
        pmflxr = _mset(pmflxr, jk + 1, nboth, 0.0)
        pmflxs = _mset(pmflxs, jk + 1, nboth, 0.0)
        pdpmel = _mset(pdpmel, jk, nboth, 0.0)
        nr = mr & (~nboth) & (pmflxr[jk + 1] < 0.0)
        pmflxs = _mset(pmflxs, jk + 1, nr, pmflxs[jk + 1] + pmflxr[jk + 1])
        pmflxr = _mset(pmflxr, jk + 1, nr, 0.0)
        ns = mr & (~nboth) & (~nr) & (pmflxs[jk + 1] < 0.0)
        pmflxr = _mset(pmflxr, jk + 1, ns, pmflxr[jk + 1] + pmflxs[jk + 1])
        pmflxs = _mset(pmflxs, jk + 1, ns, 0.0)
        return (prain, pdpmel, pqsen, plglac, pmflxr, pmflxs, pdmfdp)

    carry = (prain, pdpmel, pqsen, plglac, pmflxr, pmflxs, pdmfdp)
    carry = jax.lax.fori_loop(ktopm2, klev + 1, rain_body, carry)
    (prain, pdpmel, pqsen, plglac, pmflxr, pmflxs, pdmfdp) = carry

    # 3. evaporation below cloud base
    def evap_body(jk, carry):
        (pmflxr, pmflxs, pdmfup, pdmfdp, pdpmel) = carry
        me = ldcum & (jk >= kcbot)
        zrfl = pmflxr[jk] + pmflxs[jk]
        big = me & (zrfl > _F32_1EM20)
        zdrfl1 = (zcpecons * jnp.maximum(0.0, pqsen[jk] - pqen[jk]) * zcucov
                  * (jnp.sqrt(paph[jk] / paph[klev + 1]) / _F32_5P09EM3
                     * zrfl / zcucov) ** _F32_0P5777
                  * (paph[jk + 1] - paph[jk]))
        zrnew = zrfl - zdrfl1
        zrmin = (zrfl - zcucov
                 * jnp.maximum(0.0, rhevap * pqsen[jk] - pqen[jk])
                 * zcons2 * (paph[jk + 1] - paph[jk]))
        zrnew = jnp.maximum(zrnew, zrmin)
        zrfln = jnp.maximum(zrnew, 0.0)
        zdrfl = jnp.minimum(0.0, zrfln - zrfl)
        zdenom = 1.0 / jnp.maximum(_F32_1EM20, pmflxr[jk] + pmflxs[jk])
        zalfaw = _foealfa(pten[jk])
        zalfaw = jnp.where(pten[jk] < TMELT, 0.0, zalfaw)
        zpdr = zalfaw * pdmfdp[jk]
        zpds = (1.0 - zalfaw) * pdmfdp[jk]
        pmflxr = _mset(pmflxr, jk + 1, big,
                       pmflxr[jk] + zpdr + pdpmel[jk]
                       + zdrfl * pmflxr[jk] * zdenom)
        pmflxs = _mset(pmflxs, jk + 1, big,
                       pmflxs[jk] + zpds - pdpmel[jk]
                       + zdrfl * pmflxs[jk] * zdenom)
        pdmfup = _mset(pdmfup, jk, big, pdmfup[jk] + zdrfl)
        nboth = big & (pmflxr[jk + 1] + pmflxs[jk + 1] < 0.0)
        pdmfup = _mset(pdmfup, jk, nboth,
                       pdmfup[jk] - (pmflxr[jk + 1] + pmflxs[jk + 1]))
        pmflxr = _mset(pmflxr, jk + 1, nboth, 0.0)
        pmflxs = _mset(pmflxs, jk + 1, nboth, 0.0)
        pdpmel = _mset(pdpmel, jk, nboth, 0.0)
        nr = big & (~nboth) & (pmflxr[jk + 1] < 0.0)
        pmflxs = _mset(pmflxs, jk + 1, nr, pmflxs[jk + 1] + pmflxr[jk + 1])
        pmflxr = _mset(pmflxr, jk + 1, nr, 0.0)
        ns = big & (~nboth) & (~nr) & (pmflxs[jk + 1] < 0.0)
        pmflxr = _mset(pmflxr, jk + 1, ns, pmflxr[jk + 1] + pmflxs[jk + 1])
        pmflxs = _mset(pmflxs, jk + 1, ns, 0.0)
        small = me & (~big)
        pmflxr = _mset(pmflxr, jk + 1, small, 0.0)
        pmflxs = _mset(pmflxs, jk + 1, small, 0.0)
        pdmfdp = _mset(pdmfdp, jk, small, 0.0)
        pdpmel = _mset(pdpmel, jk, small, 0.0)
        return (pmflxr, pmflxs, pdmfup, pdmfdp, pdpmel)

    carry = (pmflxr, pmflxs, pdmfup, pdmfdp, pdpmel)
    carry = jax.lax.fori_loop(ktopm2, klev + 1, evap_body, carry)
    (pmflxr, pmflxs, pdmfup, pdmfdp, pdpmel) = carry

    return (ktopm2, ktype, lddraf, prain, pdpmel, pmflxr, pmflxs, idbas,
            pqsen, pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plude,
            pdmfup, pdmfdp, plglac, pmfdde_rate)


# --- cudtdqn (T/q tendencies) -----------------------------------------------------
def _cudtdqn_jax(klev, ktopm2, ldcum, paph, pten, plglac, plude, pmfu, pmfd,
                 pmfus, pmfds, pmfuq, pmfdq, pmful, pdmfup, pdmfdp, pdpmel,
                 ptent, ptenq, pcte):
    zdp = _zeros(klev)
    zdtdt = _zeros(klev)
    zdqdt = _zeros(klev)

    def dp_body(jk, zdp):
        return _mset(zdp, jk, ldcum, G / (paph[jk + 1] - paph[jk]))

    zdp = jax.lax.fori_loop(1, klev + 1, dp_body, zdp)

    def dtdq_body(jk, carry):
        zdtdt, zdqdt = carry
        zalv = _foelhm(pten[jk])
        below_top = jk < klev
        val_t_a = (zdp[jk] * RCPD
                   * (pmfus[jk + 1] - pmfus[jk] + pmfds[jk + 1]
                      - pmfds[jk] + ALF * plglac[jk]
                      - ALF * pdpmel[jk]
                      - zalv * (pmful[jk + 1] - pmful[jk]
                                - plude[jk] - pdmfup[jk] - pdmfdp[jk])))
        val_q_a = (zdp[jk] * (pmfuq[jk + 1] - pmfuq[jk]
                              + pmfdq[jk + 1] - pmfdq[jk]
                              + pmful[jk + 1] - pmful[jk]
                              - plude[jk] - pdmfup[jk] - pdmfdp[jk]))
        val_t_b = (-zdp[jk] * RCPD
                   * (pmfus[jk] + pmfds[jk] + ALF * pdpmel[jk]
                      - zalv * (pmful[jk] + pdmfup[jk]
                                + pdmfdp[jk] + plude[jk])))
        val_q_b = (-zdp[jk] * (pmfuq[jk] + plude[jk] + pmfdq[jk]
                               + (pmful[jk] + pdmfup[jk] + pdmfdp[jk])))
        zdtdt = _mset(zdtdt, jk, ldcum,
                      jnp.where(below_top, val_t_a, val_t_b))
        zdqdt = _mset(zdqdt, jk, ldcum,
                      jnp.where(below_top, val_q_a, val_q_b))
        return (zdtdt, zdqdt)

    zdtdt, zdqdt = jax.lax.fori_loop(ktopm2, klev + 1, dtdq_body,
                                     (zdtdt, zdqdt))

    def app_body(jk, carry):
        ptent, ptenq, pcte = carry
        ptent = _mset(ptent, jk, ldcum, ptent[jk] + zdtdt[jk])
        ptenq = _mset(ptenq, jk, ldcum, ptenq[jk] + zdqdt[jk])
        pcte = _mset(pcte, jk, ldcum, zdp[jk] * plude[jk])
        return (ptent, ptenq, pcte)

    ptent, ptenq, pcte = jax.lax.fori_loop(ktopm2, klev + 1, app_body,
                                           (ptent, ptenq, pcte))
    return ptent, ptenq, pcte


# --- cududvn (u/v tendencies) ------------------------------------------------------
def _cududvn_jax(klev, ktopm2, ktype, kcbot, kctop, ldcum, paph, puen, pven,
                 pmfu, pmfd, puu, pud, pvu, pvd, ptenu, ptenv):
    zuen = _zeros(klev)
    zven = _zeros(klev)
    zdp = _zeros(klev)
    zmfuu = _zeros(klev)
    zmfdu = _zeros(klev)
    zmfuv = _zeros(klev)
    zmfdv = _zeros(klev)
    zdudt = _zeros(klev)
    zdvdt = _zeros(klev)

    def env_body(jk, carry):
        zuen, zven, zdp = carry
        zuen = _mset(zuen, jk, ldcum, puen[jk])
        zven = _mset(zven, jk, ldcum, pven[jk])
        zdp = _mset(zdp, jk, ldcum, G / (paph[jk + 1] - paph[jk]))
        return (zuen, zven, zdp)

    zuen, zven, zdp = jax.lax.fori_loop(1, klev + 1, env_body,
                                        (zuen, zven, zdp))

    def mf_body(jk, carry):
        zmfuu, zmfuv, zmfdu, zmfdv = carry
        ik = jk - 1
        zmfuu = _mset(zmfuu, jk, ldcum, pmfu[jk] * (puu[jk] - zuen[ik]))
        zmfuv = _mset(zmfuv, jk, ldcum, pmfu[jk] * (pvu[jk] - zven[ik]))
        zmfdu = _mset(zmfdu, jk, ldcum, pmfd[jk] * (pud[jk] - zuen[ik]))
        zmfdv = _mset(zmfdv, jk, ldcum, pmfd[jk] * (pvd[jk] - zven[ik]))
        return (zmfuu, zmfuv, zmfdu, zmfdv)

    zmfuu, zmfuv, zmfdu, zmfdv = jax.lax.fori_loop(
        ktopm2, klev + 1, mf_body, (zmfuu, zmfuv, zmfdu, zmfdv))

    def ext_body(jk, carry):
        zmfuu, zmfuv, zmfdu, zmfdv = carry
        mx = ldcum & (jk > kcbot)
        ikb = kcbot
        zzp = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ikb]))
        zzp = jnp.where(ktype == 3, zzp * zzp, zzp)
        zmfuu = _mset(zmfuu, jk, mx, zmfuu[ikb] * zzp)
        zmfuv = _mset(zmfuv, jk, mx, zmfuv[ikb] * zzp)
        zmfdu = _mset(zmfdu, jk, mx, zmfdu[ikb] * zzp)
        zmfdv = _mset(zmfdv, jk, mx, zmfdv[ikb] * zzp)
        return (zmfuu, zmfuv, zmfdu, zmfdv)

    zmfuu, zmfuv, zmfdu, zmfdv = jax.lax.fori_loop(
        ktopm2, klev + 1, ext_body, (zmfuu, zmfuv, zmfdu, zmfdv))

    def dudv_body(jk, carry):
        zdudt, zdvdt = carry
        ik = jk + 1
        below_top = jk < klev
        val_u_a = zdp[jk] * (zmfuu[ik] - zmfuu[jk] + zmfdu[ik] - zmfdu[jk])
        val_v_a = zdp[jk] * (zmfuv[ik] - zmfuv[jk] + zmfdv[ik] - zmfdv[jk])
        val_u_b = -zdp[jk] * (zmfuu[jk] + zmfdu[jk])
        val_v_b = -zdp[jk] * (zmfuv[jk] + zmfdv[jk])
        zdudt = _mset(zdudt, jk, ldcum,
                      jnp.where(below_top, val_u_a, val_u_b))
        zdvdt = _mset(zdvdt, jk, ldcum,
                      jnp.where(below_top, val_v_a, val_v_b))
        return (zdudt, zdvdt)

    zdudt, zdvdt = jax.lax.fori_loop(ktopm2, klev + 1, dudv_body,
                                     (zdudt, zdvdt))

    def app_body(jk, carry):
        ptenu, ptenv = carry
        ptenu = _mset(ptenu, jk, ldcum, ptenu[jk] + zdudt[jk])
        ptenv = _mset(ptenv, jk, ldcum, ptenv[jk] + zdvdt[jk])
        return (ptenu, ptenv)

    ptenu, ptenv = jax.lax.fori_loop(ktopm2, klev + 1, app_body,
                                     (ptenu, ptenv))
    return ptenu, ptenv


# --- cumastrn (master routine) -----------------------------------------------------
def _cumastrn_jax(klev, pten, pqen, puen, pven, pverv, pqsen, pqhfl, ztmst,
                  pap, paph, pgeo, ptte, pqte, pvom, pvol, phhfl, lndj,
                  zgeoh, dx, scale_fac, scale_fac2):
    """Functional cumastrn.  Returns
    (prsfc, pssfc, ldcum, ktype, pcte, ptte, pqte, pvom, pvol, pqsen)."""
    zcons = 1.0 / (G * ztmst)
    zcons2 = 3.0 / (G * ztmst)
    lev_idx = jnp.arange(klev + 2)
    lev_valid = (lev_idx >= 1) & (lev_idx <= klev)

    pcte = _zeros(klev)

    # 2. cuinin
    (ztenh, zqenh, zqsenh, ilwmin, ptu, pqu, ztd, zqd, zuu, zvu, zud, zvd,
     plu, ilab) = _cuinin_jax(klev, pten, pqen, pqsen, puen, pven, pverv,
                              pgeo, paph, zgeoh)

    # 3.0 cloud base calculations (cutypen)
    (ldcum, kcbot, ictop0, ktype, wbase, kdpl,
     ptu, pqu, ilab, plu) = _cutypen_jax(
        klev, pqen, ztenh, zqenh, zqsenh, zgeoh, paph, phhfl, pqhfl,
        pgeo, pqsen, pap, pten, lndj, ptu, pqu, ilab, plu)

    # 3(b) first-guess cloud-base mass flux
    def pbl_body(jk, carry):
        zdhpbl, upbl = carry
        c = (jk >= kcbot) & ldcum
        zdhpbl = jnp.where(c, zdhpbl + (ALV * pqte[jk] + CPD * ptte[jk])
                           * (paph[jk + 1] - paph[jk]), zdhpbl)
        wspeed = jnp.sqrt(puen[jk] ** 2 + pven[jk] ** 2)
        upbl = jnp.where(c & (lndj == 0),
                         upbl + wspeed * (paph[jk + 1] - paph[jk]), upbl)
        return (zdhpbl, upbl)

    zdhpbl, upbl = jax.lax.fori_loop(2, klev + 1, pbl_body,
                                     (_f64(0.0), _f64(0.0)))
    idtop = _I32(0)

    ikb = kcbot
    zmfmax_fg = (paph[ikb] - paph[ikb - 1]) * zcons2
    t1 = ldcum & (ktype == 1)
    t2 = ldcum & (ktype == 2)
    zqumqe = pqu[ikb] + plu[ikb] - zqenh[ikb]
    zdqmin = jnp.maximum(_F32_0P01 * zqenh[ikb], _F32_1EM10)
    zdh = CPD * (ptu[ikb] - ztenh[ikb]) + ALV * zqumqe
    zdh = G * jnp.maximum(zdh, 1.0e5 * zdqmin)
    pos = zdhpbl > 0.0
    zmfub_t2 = jnp.where(pos, jnp.minimum(zdhpbl / zdh, zmfmax_fg),
                         _F32_0P1 * zmfmax_fg)
    zmfub = jnp.where(t1, _F32_0P1 * zmfmax_fg,
                      jnp.where(t2, zmfub_t2, 0.0))
    ldcum = jnp.where(t2 & (~pos), _B(False), ldcum)

    # 4.0 cloud ascent
    kctop = _I32(0)
    asc = _cuascn_jax(
        klev, ztmst, ztenh, zqenh, pten, pqen, pqsen, pgeo, zgeoh,
        pap, paph, pverv, ldcum, ktype, ilab, ptu, pqu, plu, zmfub,
        kcbot, kctop, ictop0, lndj, wbase, kdpl)
    ldcum = asc["ldcum"]
    ktype = asc["ktype"]
    kcbot = asc["kcbot"]
    kctop = asc["kctop"]
    ictop0 = asc["kctop0"]
    zmfub = asc["pmfub"]
    wup = asc["wup"]
    zlglac = asc["plglac"]
    pmfude_rate = asc["pmfude_rate"]
    ztenh = asc["ptenh"]
    zqenh = asc["pqenh"]
    ilab = asc["klab"]
    ptu = asc["ptu"]
    pqu = asc["pqu"]
    plu = asc["plu"]
    pmfu = asc["pmfu"]
    zmfus = asc["pmfus"]
    zmfuq = asc["pmfuq"]
    zmful = asc["pmful"]
    plude = asc["plude"]
    zdmfup = asc["pdmfup"]

    # 5. cloud depth check, adjust ktype
    zpbmpt = paph[kcbot] - paph[kctop]
    ktype = jnp.where(ldcum & (ktype == 1) & (zpbmpt < ZDNOPRC),
                      _I32(2), ktype)
    ktype = jnp.where(ldcum & (ktype == 2) & (zpbmpt >= ZDNOPRC),
                      _I32(1), ktype)
    ictop0 = jnp.where(ldcum, kctop, ictop0)

    def rfl_body(jk, zrfl):
        return zrfl + zdmfup[jk]

    zrfl = jax.lax.fori_loop(2, klev + 1, rfl_body, zdmfup[1])

    # 6.0 downdrafts (LMFDD is statically True)
    (pmfd, zmfds, zmfdq, zdmfdp, idtop, loddraf, zrfl, ztd, zqd) = \
        _cudlfsn_jax(klev, kcbot, kctop, ldcum, ztenh, zqenh, pten, pqsen,
                     pgeo, zgeoh, paph, ptu, pqu, zmfub, zrfl, ztd, zqd)
    (zrfl, pmfdde_rate, ztd, zqd, pmfd, zmfds, zmfdq, zdmfdp) = \
        _cuddrafn_jax(klev, loddraf, ztenh, zqenh, pgeo, zgeoh, paph, zrfl,
                      ztd, zqd, pmfu, pmfd, zmfds, zmfdq, zdmfdp)

    # 6.1 CAPE closure for deep convection
    zmfub1 = zmfub
    m1 = ldcum & (ktype == 1)
    ikb = kcbot
    ikt = kctop
    ztauc = jnp.where(m1, (zgeoh[ikt] - zgeoh[ikb])
                      / ((2.0 + jnp.minimum(15.0, wup)) * G), 0.0)
    land0 = lndj == 0
    upbl_n = 2.0 + upbl / (paph[klev + 1] - paph[ikb])
    upbl = jnp.where(m1 & land0, upbl_n, upbl)
    ztaubl_l = jnp.minimum(300.0, (zgeoh[ikb] - zgeoh[klev + 1]) / (G * upbl))
    ztaubl = jnp.where(m1, jnp.where(land0, ztaubl_l, ztauc), 0.0)

    def cape_body(jk, carry):
        zheat, zcape1, zcape2 = carry
        llo1 = ldcum & (ktype == 1)
        ca = llo1 & (jk <= kcbot) & (jk > kctop)
        zdz = pgeo[jk - 1] - pgeo[jk]
        zdp_ = pap[jk] - pap[jk - 1]
        zheat = jnp.where(
            ca, zheat + (((pten[jk - 1] - pten[jk] + zdz * RCPD) / ztenh[jk]
                          + VTMPC1 * (pqen[jk - 1] - pqen[jk]))
                         * (G * (pmfu[jk] + pmfd[jk]))), zheat)
        zcape1 = jnp.where(
            ca, zcape1 + (((ptu[jk] - ztenh[jk]) / ztenh[jk]
                           + VTMPC1 * (pqu[jk] - zqenh[jk]) - plu[jk])
                          * zdp_), zcape1)
        cb = llo1 & (jk >= kcbot) & ((paph[klev + 1] - paph[kdpl]) < 50.0e2)
        zdp2 = paph[jk + 1] - paph[jk]
        zcape2 = jnp.where(
            cb, zcape2 + (ztaubl
                          * ((1.0 + VTMPC1 * pqen[jk]) * ptte[jk]
                             + VTMPC1 * pten[jk] * pqte[jk]) * zdp2), zcape2)
        return (zheat, zcape1, zcape2)

    zheat, zcape1, zcape2 = jax.lax.fori_loop(
        1, klev + 1, cape_body, (_f64(0.0), _f64(0.0), _f64(0.0)))

    ztauc_c = jnp.maximum(ztmst, ztauc)
    ztauc_c = jnp.maximum(360.0, ztauc_c)
    ztauc_c = jnp.minimum(10800.0, ztauc_c)
    ztau = ztauc_c * scale_fac
    zcape2_c = jnp.maximum(0.0, zcape2)  # NONEQUIL is statically True
    zcape = jnp.maximum(0.0, jnp.minimum(zcape1 - zcape2_c, 5000.0))
    zheat_c = jnp.maximum(_F32_1EM4, zheat)
    zmfub1_deep = (zcape * zmfub) / (zheat_c * ztau)
    zmfub1_deep = jnp.maximum(zmfub1_deep, _F32_0P001)
    zmfmax_d = (paph[ikb] - paph[ikb - 1]) * zcons2
    zmfub1_deep = jnp.minimum(zmfub1_deep, zmfmax_d)
    zmfub1 = jnp.where(m1, zmfub1_deep, zmfub1)

    # 6.2 shallow closure (moist static energy budget with downdrafts)
    m2c = ldcum & (ktype == 2)
    zeps = jnp.where((pmfd[ikb] < 0.0) & loddraf,
                     -pmfd[ikb] / jnp.maximum(zmfub, CMFCMIN), 0.0)
    zqumqe_s = (pqu[ikb] + plu[ikb]
                - zeps * zqd[ikb] - (1.0 - zeps) * zqenh[ikb])
    zdqmin_s = jnp.maximum(_F32_0P01 * zqenh[ikb], CMFCMIN)
    zmfmax_s = (paph[ikb] - paph[ikb - 1]) * zcons2
    zdh_s = (CPD * (ptu[ikb] - zeps * ztd[ikb]
                    - (1.0 - zeps) * ztenh[ikb]) + ALV * zqumqe_s)
    zdh_s = G * jnp.maximum(zdh_s, 1.0e5 * zdqmin_s)
    zmfub1_sh = jnp.where(zdhpbl > 0.0, zdhpbl / zdh_s, zmfub)
    zmfub1_sh = zmfub1_sh / scale_fac2
    zmfub1_sh = jnp.minimum(zmfub1_sh, zmfmax_s)
    zmfub1 = jnp.where(m2c, zmfub1_sh, zmfub1)

    # 6.3 mid-level convection
    zmfub1 = jnp.where(ldcum & (ktype == 3), zmfub, zmfub1)

    # 6.4 scale the downdraft mass flux
    zfac_dd = zmfub1 / jnp.maximum(zmfub, CMFCMIN)
    dd_mask = lev_valid & ldcum
    pmfd = jnp.where(dd_mask, pmfd * zfac_dd, pmfd)
    zmfds = jnp.where(dd_mask, zmfds * zfac_dd, zmfds)
    zmfdq = jnp.where(dd_mask, zmfdq * zfac_dd, zmfdq)
    zdmfdp = jnp.where(dd_mask, zdmfdp * zfac_dd, zdmfdp)
    pmfdde_rate = jnp.where(dd_mask, pmfdde_rate * zfac_dd, pmfdde_rate)

    # 6.5 scale the updraft mass flux
    zmfs = jnp.where(ldcum, zmfub1 / jnp.maximum(CMFCMIN, zmfub), 1.0)

    def upscale_body(jk, carry):
        pmfu, zmfs = carry
        m = ldcum & (jk >= kctop - 1)
        ikb2 = kcbot
        ab = m & (jk > ikb2)
        zdz = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ikb2]))
        pmfu = _mset(pmfu, jk, ab, pmfu[ikb2] * zdz)
        zmfmax = (paph[jk] - paph[jk - 1]) * zcons2
        hit = m & (pmfu[jk] * zmfs > zmfmax)
        zmfs = jnp.where(hit, jnp.minimum(zmfs, zmfmax / pmfu[jk]), zmfs)
        return (pmfu, zmfs)

    pmfu, zmfs = jax.lax.fori_loop(2, klev + 1, upscale_body, (pmfu, zmfs))

    up_mask = (lev_valid & (lev_idx >= 2) & ldcum
               & (lev_idx <= kcbot) & (lev_idx >= kctop - 1))
    pmfu = jnp.where(up_mask, pmfu * zmfs, pmfu)
    zmfus = jnp.where(up_mask, zmfus * zmfs, zmfus)
    zmfuq = jnp.where(up_mask, zmfuq * zmfs, zmfuq)
    zmful = jnp.where(up_mask, zmful * zmfs, zmful)
    zdmfup = jnp.where(up_mask, zdmfup * zmfs, zdmfup)
    plude = jnp.where(up_mask, plude * zmfs, plude)
    pmfude_rate = jnp.where(up_mask, pmfude_rate * zmfs, pmfude_rate)

    # 6.6 if ktype=2, kcbot=kctop is not allowed
    bad = (ktype == 2) & (kcbot == kctop) & (kcbot >= klev - 1)
    ldcum = jnp.where(bad, _B(False), ldcum)
    ktype = jnp.where(bad, _I32(0), ktype)

    # 6.7 set downdraft mass fluxes to zero above cloud top
    idtop = jnp.where(loddraf & (idtop <= kctop), kctop + 1, idtop)
    dz_mask = lev_valid & (lev_idx >= 2) & loddraf & (lev_idx < idtop)
    pmfd = jnp.where(dz_mask, 0.0, pmfd)
    zmfds = jnp.where(dz_mask, 0.0, zmfds)
    zmfdq = jnp.where(dz_mask, 0.0, zmfdq)
    pmfdde_rate = jnp.where(dz_mask, 0.0, pmfdde_rate)
    zdmfdp = jnp.where(dz_mask, 0.0, zdmfdp)
    de_mask = lev_valid & (lev_idx >= 2) & loddraf & (lev_idx == idtop)
    pmfdde_rate = jnp.where(de_mask, 0.0, pmfdde_rate)

    # 7.0 final convective fluxes in cuflxn
    (itopm2, ktype, loddraf, prain, zdpmel, pmflxr, pmflxs, idbas,
     pqsen, pmfu, pmfd, zmfus, zmfds, zmfuq, zmfdq, zmful, plude,
     zdmfup, zdmfdp, zlglac, pmfdde_rate) = _cuflxn_jax(
        klev, ztmst, pten, pqen, pqsen, ztenh, zqenh, paph, pap, zgeoh,
        lndj, ldcum, kcbot, kctop, idtop, ktype, loddraf, pmfu, pmfd,
        zmfus, zmfds, zmfuq, zmfdq, zmful, plude, zdmfup, zdmfdp,
        zlglac, pmfdde_rate)

    # some adjustments needed
    def adj1_body(jk, zmfs2):
        c = loddraf & (jk >= idtop - 1)
        zmfmax = pmfu[jk] * _F32_0P98
        hit = c & (pmfd[jk] + zmfmax + _F32_1EM15 < 0.0)
        return jnp.where(hit, jnp.minimum(zmfs2, -zmfmax / pmfd[jk]), zmfs2)

    zmfs2 = jax.lax.fori_loop(2, klev + 1, adj1_body, _f64(1.0))

    def adj2_body(jk, carry):
        (pmfd, zmfds, zmfdq, pmfdde_rate, zdmfdp, pmflxr, zmfuub) = carry
        c = (zmfs2 < 1.0) & (jk >= idtop - 1)
        pmfd = _mset(pmfd, jk, c, pmfd[jk] * zmfs2)
        zmfds = _mset(zmfds, jk, c, zmfds[jk] * zmfs2)
        zmfdq = _mset(zmfdq, jk, c, zmfdq[jk] * zmfs2)
        pmfdde_rate = _mset(pmfdde_rate, jk, c, pmfdde_rate[jk] * zmfs2)
        zmfuub = jnp.where(c, zmfuub - (1.0 - zmfs2) * zdmfdp[jk], zmfuub)
        pmflxr = _mset(pmflxr, jk + 1, c, pmflxr[jk + 1] + zmfuub)
        zdmfdp = _mset(zdmfdp, jk, c, zdmfdp[jk] * zmfs2)
        return (pmfd, zmfds, zmfdq, pmfdde_rate, zdmfdp, pmflxr, zmfuub)

    carry = (pmfd, zmfds, zmfdq, pmfdde_rate, zdmfdp, pmflxr, _f64(0.0))
    carry = jax.lax.fori_loop(2, klev + 1, adj2_body, carry)
    (pmfd, zmfds, zmfdq, pmfdde_rate, zdmfdp, pmflxr, zmfuub) = carry

    def adj3_body(jk, carry):
        pmfdde_rate, pmfude_rate, zdmfup, zdmfdp = carry
        c1 = loddraf & (jk >= idtop - 1)
        zerate_d = -pmfd[jk] + pmfd[jk - 1] + pmfdde_rate[jk]
        pmfdde_rate = _mset(pmfdde_rate, jk, c1 & (zerate_d < 0.0),
                            pmfdde_rate[jk] - zerate_d)
        c2 = ldcum & (jk >= kctop - 1)
        zerate_u = pmfu[jk] - pmfu[jk + 1] + pmfude_rate[jk]
        pmfude_rate = _mset(pmfude_rate, jk, c2 & (zerate_u < 0.0),
                            pmfude_rate[jk] - zerate_u)
        zdmfup = _mset(zdmfup, jk, c2,
                       pmflxr[jk + 1] + pmflxs[jk + 1]
                       - pmflxr[jk] - pmflxs[jk])
        zdmfdp = _mset(zdmfdp, jk, c2, 0.0)
        return (pmfdde_rate, pmfude_rate, zdmfup, zdmfdp)

    carry = (pmfdde_rate, pmfude_rate, zdmfup, zdmfdp)
    carry = jax.lax.fori_loop(2, klev, adj3_body, carry)
    (pmfdde_rate, pmfude_rate, zdmfup, zdmfdp) = carry

    # avoid negative humidities at ddraught top
    jkd = idtop
    ikd = jnp.minimum(jkd + 1, _I32(klev))
    hum_hit = loddraf & (zmfdq[jkd] < _F32_0P3 * zmfdq[ikd])
    zmfdq = _mset(zmfdq, jkd, hum_hit, _F32_0P3 * zmfdq[ikd])

    # avoid negative humidities near cloud top
    def hum_body(jk, carry):
        plude, pmfude_rate, pmfdde_rate = carry
        c = ldcum & (jk >= kctop - 1) & (jk < kcbot)
        zdz = ztmst * G / (paph[jk + 1] - paph[jk])
        zmfa = (zmfuq[jk + 1] + zmfdq[jk + 1] - zmfuq[jk] - zmfdq[jk]
                + zmful[jk + 1] - zmful[jk] + zdmfup[jk])
        zmfa = (zmfa - plude[jk]) * zdz
        h1 = c & (pqen[jk] + zmfa < 0.0)
        plude = _mset(plude, jk, h1,
                      plude[jk] + 2.0 * (pqen[jk] + zmfa) / zdz)
        h2 = c & (plude[jk] < 0.0)
        plude = _mset(plude, jk, h2, 0.0)
        pmfude_rate = _mset(pmfude_rate, jk, ~ldcum, 0.0)
        pmfdde_rate = _mset(pmfdde_rate, jk,
                            jnp.abs(pmfd[jk - 1]) < _F32_1EM20, 0.0)
        return (plude, pmfude_rate, pmfdde_rate)

    carry = (plude, pmfude_rate, pmfdde_rate)
    carry = jax.lax.fori_loop(2, klev + 1, hum_body, carry)
    (plude, pmfude_rate, pmfdde_rate) = carry

    prsfc = pmflxr[klev + 1]
    pssfc = pmflxs[klev + 1]

    # 8.0 update T and q tendencies
    ptte, pqte, pcte = _cudtdqn_jax(
        klev, itopm2, ldcum, paph, pten, zlglac, plude, pmfu, pmfd,
        zmfus, zmfds, zmfuq, zmfdq, zmful, zdmfup, zdmfdp, zdpmel,
        ptte, pqte, pcte)

    # 9.0 update u and v tendencies (LMFDUDV is statically True)
    def mom_up_body(idx, carry):
        zuu, zvu = carry
        jk = (klev - 1) - idx
        ik = jk + 1
        mb1 = ldcum & (jk == kcbot) & (ktype < 3)
        ikb = kdpl
        zuu = _mset(zuu, jk, mb1, puen[ikb - 1])
        zvu = _mset(zvu, jk, mb1, pven[ikb - 1])
        mb2 = ldcum & (jk == kcbot) & (ktype == 3)
        zuu = _mset(zuu, jk, mb2, puen[jk - 1])
        zvu = _mset(zvu, jk, mb2, pven[jk - 1])
        mid = ldcum & (jk < kcbot) & (jk >= kctop)
        if MOMTRANS == 1:
            zfac = jnp.where((ktype == 1) | (ktype == 3), 2.0, 0.0)
            zfac = jnp.where((ktype == 1) & (jk <= kctop + 2), 3.0, zfac)
            zerate = (pmfu[jk] - pmfu[ik]
                      + (1.0 + zfac) * pmfude_rate[jk])
            zderate = (1.0 + zfac) * pmfude_rate[jk]
            zmfa = 1.0 / jnp.maximum(CMFCMIN, pmfu[jk])
            zuu = _mset(zuu, jk, mid,
                        (zuu[ik] * pmfu[ik] + zerate * puen[jk]
                         - zderate * zuu[ik]) * zmfa)
            zvu = _mset(zvu, jk, mid,
                        (zvu[ik] * pmfu[ik] + zerate * pven[jk]
                         - zderate * zvu[ik]) * zmfa)
        else:
            pgf_u = (-PGCOEF * 0.5
                     * (pmfu[ik] * (puen[ik] - puen[jk])
                        + pmfu[jk] * (puen[jk] - puen[jk - 1])))
            pgf_v = (-PGCOEF * 0.5
                     * (pmfu[ik] * (pven[ik] - pven[jk])
                        + pmfu[jk] * (pven[jk] - pven[jk - 1])))
            zerate = pmfu[jk] - pmfu[ik] + pmfude_rate[jk]
            zderate = pmfude_rate[jk]
            zmfa = 1.0 / jnp.maximum(CMFCMIN, pmfu[jk])
            zuu = _mset(zuu, jk, mid,
                        (zuu[ik] * pmfu[ik] + zerate * puen[jk]
                         - zderate * zuu[ik] + pgf_u) * zmfa)
            zvu = _mset(zvu, jk, mid,
                        (zvu[ik] * pmfu[ik] + zerate * pven[jk]
                         - zderate * zvu[ik] + pgf_v) * zmfa)
        return (zuu, zvu)

    zuu, zvu = jax.lax.fori_loop(0, klev - 2, mom_up_body, (zuu, zvu))

    # downdraft momentum (LMFDD statically True)
    def mom_dd_body(jk, carry):
        zud, zvd = carry
        ik = jk - 1
        md1 = ldcum & (jk == idtop)
        zud = _mset(zud, jk, md1, 0.5 * (zuu[jk] + puen[ik]))
        zvd = _mset(zvd, jk, md1, 0.5 * (zvu[jk] + pven[ik]))
        md2 = ldcum & (jk > idtop)
        zerate = -pmfd[jk] + pmfd[ik] + pmfdde_rate[jk]
        zmfa = 1.0 / jnp.minimum(-CMFCMIN, pmfd[jk])
        zud = _mset(zud, jk, md2,
                    (zud[ik] * pmfd[ik] - zerate * puen[ik]
                     + pmfdde_rate[jk] * zud[ik]) * zmfa)
        zvd = _mset(zvd, jk, md2,
                    (zvd[ik] * pmfd[ik] - zerate * pven[ik]
                     + pmfdde_rate[jk] * zvd[ik]) * zmfa)
        return (zud, zvd)

    zud, zvd = jax.lax.fori_loop(3, klev + 1, mom_dd_body, (zud, zvd))

    # rescale massfluxes for stability in momentum
    def mscale_body(jk, zmfs3):
        c = ldcum & (jk >= kctop - 1)
        zmfmax = (paph[jk] - paph[jk - 1]) * zcons
        hit = c & (pmfu[jk] > zmfmax) & (jk >= kctop)
        return jnp.where(hit, jnp.minimum(zmfs3, zmfmax / pmfu[jk]), zmfs3)

    zmfs3 = jax.lax.fori_loop(2, klev + 1, mscale_body, _f64(1.0))

    mm = lev_valid & ldcum & (lev_idx >= kctop - 1)
    zmfuus = jnp.where(mm, pmfu * zmfs3, pmfu)
    zmfdus = jnp.where(mm, pmfd * zmfs3, pmfd)

    # 9.1 update u and v in cududvn
    ztenu = pvom
    ztenv = pvol
    pvom, pvol = _cududvn_jax(klev, itopm2, ktype, kcbot, kctop, ldcum,
                              paph, puen, pven, zmfuus, zmfdus, zuu, zud,
                              zvu, zvd, pvom, pvol)

    # KE dissipation
    zuv2 = _zeros(klev)

    def ke_body(jk, carry):
        zuv2, zsum12, zsum22 = carry
        c = ldcum & (jk >= kctop - 1)
        zdz = paph[jk + 1] - paph[jk]
        zduten = pvom[jk] - ztenu[jk]
        zdvten = pvol[jk] - ztenv[jk]
        zuv2 = _mset(zuv2, jk, c, jnp.sqrt(zduten ** 2 + zdvten ** 2))
        zsum22 = jnp.where(c, zsum22 + zuv2[jk] * zdz, zsum22)
        zsum12 = jnp.where(c, zsum12 - (puen[jk] * zduten
                                        + pven[jk] * zdvten) * zdz, zsum12)
        return (zuv2, zsum12, zsum22)

    zuv2, zsum12, zsum22 = jax.lax.fori_loop(
        1, klev + 1, ke_body, (zuv2, _f64(0.0), _f64(0.0)))

    def ke_app_body(jk, ptte):
        c = ldcum & (jk >= kctop - 1)
        ztdis = RCPD * zsum12 * zuv2[jk] / jnp.maximum(_F32_1EM15, zsum22)
        return _mset(ptte, jk, c, ptte[jk] + ztdis)

    ptte = jax.lax.fori_loop(1, klev + 1, ke_app_body, ptte)

    return (prsfc, pssfc, ldcum, ktype, pcte, ptte, pqte, pvom, pvol, pqsen)


# --- cu_ntiedtke_run (scheme interface, single column) --------------------------
def _cu_ntiedtke_run_jax(km, dt, dx, evap, hfx, lndj,
                         pu, pv, pt, pqv, pqc, pqi, pqvf, ptf, poz, pzz,
                         pomg, pap, paph):
    """Functional cu_ntiedtke_run on 1-based TOP-DOWN arrays (length km+2).
    Returns (zprecc, pu, pv, pt, pqv, pqc, pqi)."""
    ztmst = dt
    lev_idx = jnp.arange(km + 2)
    lev_valid = (lev_idx >= 1) & (lev_idx <= km)

    # scale-dependency factor
    dxref = 15000.0
    small_dx = dx < dxref
    scale_fac_a = (_F32_1P06133 + jnp.log(dxref / dx)) ** 3
    scale_fac = jnp.where(small_dx, scale_fac_a, 1.0 + _F32_1P33EM5 * dx)
    scale_fac2 = jnp.where(small_dx, scale_fac_a ** 0.5, 1.0)

    pqhfl = evap
    phhfl = hfx
    pgeoh = jnp.where(lev_idx >= 1, G * pzz, 0.0)
    pgeo = jnp.where(lev_valid, G * poz, 0.0)

    pcte = _zeros(km)
    pvom = _zeros(km)
    pvol = _zeros(km)
    ztp1 = pt
    zqp1 = jnp.where(lev_valid, pqv / (1.0 + pqv), 0.0)
    pum1 = pu
    pvm1 = pv
    pverv = pomg
    zew = _foeewm(ztp1)
    zqs = zew / pap
    zqs = jnp.minimum(0.5, zqs)
    zcor = 1.0 / (1.0 - VTMPC1 * zqs)
    zqsat = jnp.where(lev_valid, zqs * zcor, 0.0)
    pqte = pqvf
    zqq = pqte
    ptte = ptf
    ztt = ptte

    # 2. cumastrn
    (prsfc, pssfc, ldcum, ktype, pcte, ptte, pqte, pvom, pvol, zqsat) = \
        _cumastrn_jax(km, ztp1, zqp1, pum1, pvm1, pverv, zqsat, pqhfl,
                      ztmst, pap, paph, pgeo, ptte, pqte, pvom, pvol,
                      phhfl, lndj, pgeoh, dx, scale_fac, scale_fac2)

    # include the cloud water and cloud ice detrained from convection
    det = lev_valid & (pcte > 0.0)
    fliq = _foealfa(ztp1)
    fice = 1.0 - fliq
    pqc = jnp.where(det, pqc + fliq * pcte * ztmst, pqc)
    pqi = jnp.where(det, pqi + fice * pcte * ztmst, pqi)

    pt_out = jnp.where(lev_valid, ztp1 + (ptte - ztt) * ztmst, pt)
    zqp1_n = jnp.where(lev_valid, zqp1 + (pqte - zqq) * ztmst, zqp1)
    pqv_out = jnp.where(lev_valid, zqp1_n / (1.0 - zqp1_n), pqv)

    # Fortran: zprecc = amax1(0.0, (prsfc+pssfc)*ztmst) -- AMAX1 is the
    # REAL*4-specific intrinsic: fp32 demotion of the max and result.
    zprecc = jnp.maximum(
        jnp.float32(0.0),
        ((prsfc + pssfc) * ztmst).astype(jnp.float32)).astype(jnp.float64)

    # LMFDUDV is statically True
    pu_out = jnp.where(lev_valid, pu + pvom * ztmst, pu)
    pv_out = jnp.where(lev_valid, pv + pvol * ztmst, pv)

    return zprecc, pu_out, pv_out, pt_out, pqv_out, pqc, pqi


# --- public column API (WRF bottom-up orientation, frozen interface) ------------
def ntiedtke_column_jax(T, QV, QC, QI, P, P8W, DZ, RHO, PI, U, V, W,
                        QVFTEN, THFTEN, QFX, HFX, XLAND, DX, DT,
                        stepcu: int = 1, itimestep: int = 2):
    """Faithful fp64 New-Tiedtke single-column step (jit/vmap-traceable).

    Bottom-up WRF columns exactly as the NumPy reference
    :func:`gpuwrf.physics.cumulus_ntiedtke.ntiedtke_column`: ``T``..``THFTEN``
    of shape (kx,), ``P8W``/``W`` of shape (kx+1,); QFX/HFX/XLAND/DX/DT traced
    scalars; ``stepcu``/``itimestep`` static Python ints.

    Returns a dict with RTHCUTEN, RQVCUTEN, RQCCUTEN, RQICUTEN, RUCUTEN,
    RVCUTEN (bottom-up, shape (kx,)) and RAINCV, PRATEC (0-d fp64).
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
    THFTEN = jnp.asarray(THFTEN, jnp.float64)
    QFX = jnp.asarray(QFX, jnp.float64)
    HFX = jnp.asarray(HFX, jnp.float64)
    XLAND = jnp.asarray(XLAND, jnp.float64)
    DX = jnp.asarray(DX, jnp.float64)
    DT = jnp.asarray(DT, jnp.float64)

    kx = T.shape[0]
    kx1 = kx + 1

    # --- cu_ntiedtke_pre_run ---------------------------------------------------
    delt = DT * stepcu
    # Fortran INTEGER truncation of |xland - 2|
    slimsk = jnp.trunc(jnp.abs(XLAND - 2.0)).astype(_I32)

    def zi_body(k, zi):
        return zi.at[k + 1].set(zi[k] + DZ[k])

    zi = jax.lax.fori_loop(0, kx, zi_body, jnp.zeros(kx1, jnp.float64))
    zl = 0.5 * (zi[0:kx] + zi[1:kx + 1])
    dot = -0.5 * G * RHO * (W[0:kx] + W[1:kx + 1])

    # flip bottom-up (0-based) -> top-down (1-based, dead 0 slot)
    z1 = jnp.zeros(1, jnp.float64)

    def flip_lev(arr):
        return jnp.concatenate([z1, arr[::-1], z1])

    def flip_int(arr):
        return jnp.concatenate([z1, arr[::-1]])

    ghti = flip_int(zi)
    prsi = flip_int(P8W)
    ghtl = flip_lev(zl)
    omg = flip_lev(dot)
    prsl = flip_lev(P)
    tf = flip_lev(T)
    qvf = flip_lev(QV)
    qcf = flip_lev(QC)
    qif = flip_lev(QI)
    uf = flip_lev(U)
    vf = flip_lev(V)
    if itimestep == 1:
        qvftenz = jnp.zeros(kx + 2, jnp.float64)
        thftenz = jnp.zeros(kx + 2, jnp.float64)
    else:
        qvftenz = flip_lev(QVFTEN)
        thftenz = flip_lev(THFTEN)

    # --- cu_ntiedtke_run ---------------------------------------------------------
    rn, uf, vf, tf, qvf, qcf, qif = _cu_ntiedtke_run_jax(
        kx, delt, DX, QFX, HFX, slimsk,
        uf, vf, tf, qvf, qcf, qif, qvftenz, thftenz, ghtl, ghti, omg,
        prsl, prsi)

    # --- cu_ntiedtke_post_run ----------------------------------------------------
    rdelt = 1.0 / delt
    raincv = rn / stepcu
    pratec = rn / (stepcu * DT)

    def unflip(arr):
        return arr[1:kx + 1][::-1]

    rthcuten = (unflip(tf) - T) / PI * rdelt
    rqvcuten = (unflip(qvf) - QV) * rdelt
    rqccuten = (unflip(qcf) - QC) * rdelt
    rqicuten = (unflip(qif) - QI) * rdelt
    rucuten = (unflip(uf) - U) * rdelt
    rvcuten = (unflip(vf) - V) * rdelt

    return {
        "RTHCUTEN": rthcuten,
        "RQVCUTEN": rqvcuten,
        "RQCCUTEN": rqccuten,
        "RQICUTEN": rqicuten,
        "RUCUTEN": rucuten,
        "RVCUTEN": rvcuten,
        "RAINCV": raincv,
        "PRATEC": pratec,
    }


__all__ = ["ntiedtke_column_jax"]
