"""Aerosol-aware Morrison (mp_physics=40) — aero constants, mdm_* subroutines,
and the copy-and-modify cold branch / sedimentation / finalize.

Companion to ``microphysics_morrison_aero.py`` (see its docstring for the full
delta inventory vs the base port).  This file holds everything the aero Fortran
``phys/module_mp_morr_two_moment_aero.F`` CHANGES relative to
``phys/module_mp_morr_two_moment.F``, plus faithful copies of the base blocks
whose *constants* change (DCS cascade):

* Aero init constants (``MORR_TWO_MOMENT_INIT_AERO``, aero.F l.469..688):
  DCS=350e-6 (base 125e-6) -> LAMMINI/CONS21/CONS22 recomputed; the ``*_pamdm``
  Abdul-Razzak & Ghan activation-constant block incl. the 10-mode DATA tables
  (aero.F l.273-324) and the CCNFACT table (l.669-686).
* ``DERF1`` (aero.F l.5334-5445): Ooura piecewise-minimax erf, ported exactly
  (both a host/scalar version for init-time CCNFACT and a traced JAX version
  for activation).
* ``mdm_prescribed_activate`` + ``mdm_prescribed_maxsat`` (aero.F l.5626-5900):
  AR&G 2000 activation over the 10 prescribed modes.
* ``mdm_prescribed_nucleati`` + ``mdm_prescribed_hf`` + ``mdm_prescribed_hetero``
  (aero.F l.5901-6120): Liu & Penner 2005 ice nucleation + Meyers mixed-phase.
* ``_cold_branch_aero``: copy of the base ``_cold_branch`` with (a) the
  Fortran's in-place number-concentration semantics (MAX(0,..) floors and slope
  clamp write-backs feed all downstream uses, exactly as NI3D/NS3D/NR3D/NG3D/
  NC3D are mutated in the Fortran — live now that NC is prognostic),
  (b) INUC=2 ice nucleation replacing the Cooper curve, (c) the DCS cascade.
* ``_sedimentation_aero``: copy with iinum=0 ``DUMFNC = MAX(0, NC+NCTEN*DT)``
  (aero.F l.4312-4327) and aero LAMMINI.
* ``_finalize_aero``: copy with aero DCS/LAMMINI, the RRTMG-consistent
  cloud-free effective radii EFFI=4.99 / EFFC=2.49 (aero.F l.4953-4974 vs base
  25.0), NO constant-droplet reset (iinum=0), and no NC aerosol bound (the
  ``(NANEW1+NANEW2)/RHO`` bound applies only to IACT=2, aero.F l.4993-4995).

Faithfulness notes:
- The Fortran STOPs on ``nuci>9999 .or. nuci<0`` (aero.F l.6070); a jitted port
  cannot stop, so it applies the same ``nuci=0`` reset without aborting (never
  triggered in the savepoint regimes).
- The homogeneous/heterogeneous interpolation ``(niimm/nihf)**((tc-regm)/5)``
  is guarded against 0**negative in lanes the Fortran would only reach with a
  NaN of its own (niimm=0 exactly); the guarded lane returns 0.
- ``real_tiny = TINY(1.0)`` is the fp64 value (the binding gate is the
  -fdefault-real-8 oracle build, where default REAL is fp64).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import morrison_constants as C
from gpuwrf.physics.microphysics_morrison import gamma_fn, polysvp
from gpuwrf.physics._morrison_sed import _slope_final

_EPS = 1.0e-30
_NSTEP_MAX = 64

# ===========================================================================
# Aero init constants (MORR_TWO_MOMENT_INIT_AERO).  DCS cascade.
# ===========================================================================
DCS = 350.0e-6                              # base: 125e-6 (aero.F l.469)
LAMMINI = 1.0 / (2.0 * DCS + 100.0e-6)      # cascades from DCS
CONS21 = 4.0 / (DCS * C.RHOI)
CONS22 = C.PI * C.RHOI * DCS ** 3 / 6.0

# WRF model constants used by the aero additions (share/module_model_constants.F)
T0 = 300.0
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
P0 = 1013.25e2                              # mdm_prescribed_activate data p0
REAL_TINY = float(np.finfo(np.float64).tiny)  # TINY(1.0) under -fdefault-real-8

NAER_CU = 10

# Aerosol property DATA tables (aero.F l.309-324), mode order:
# SULFATE, SEASALT2, DUST1..4, OCPHO, BCPHO, OCPHI, BCPHI
DRYRAD_AER = (0.695e-7, 0.200e-5, 0.151e-5, 0.151e-5, 0.151e-5, 0.151e-5,
              0.212e-7, 0.118e-7, 0.212e-7, 0.118e-7)   # (unused at runtime)
DENSITY_AER = (1770.0, 2200.0, 2600.0, 2600.0, 2600.0, 2600.0, 1800.0,
               1000.0, 2600.0, 1000.0)
HYGRO_AER = (0.507, 1.160, 0.140, 0.140, 0.140, 0.140, 0.100, 0.100,
             0.140, 0.100)
DISPERSION_AER = (2.030, 1.3732, 1.900, 1.900, 1.900, 1.900, 2.240, 2.000,
                  2.240, 2.000)
NUM_TO_MASS_AER = (42097098109277080.0, 8626504211623.0, 3484000000000000.0,
                   213800000000000.0, 22050000000000.0, 3165000000000.0,
                   0.745645e18, 0.167226e20, 0.516216e18, 0.167226e20)

# *_pamdm activation constants (aero.F l.631-688), evaluated in fp64.
SQ2_PAMDM = 1.41421356237
SURFTEN_PAMDM = 0.076
ATEN_PAMDM = 2.0 * C.MW * SURFTEN_PAMDM / (C.RR * T0 * C.RHOW)
ALOGATEN_PAMDM = math.log(ATEN_PAMDM)
ALOGSIG_PAMDM = tuple(math.log(d) for d in DISPERSION_AER)
EXP45LOGSIG_PAMDM = tuple(math.exp(4.5 * a * a) for a in ALOGSIG_PAMDM)
ARGFACTOR_PAMDM = tuple(2.0 / (3.0 * math.sqrt(2.0) * a) for a in ALOGSIG_PAMDM)
F1_PAMDM = tuple(0.5 * math.exp(2.5 * a * a) for a in ALOGSIG_PAMDM)
F2_PAMDM = tuple(1.0 + 0.25 * a for a in ALOGSIG_PAMDM)
AMCUBEFACTOR_PAMDM = tuple(
    3.0 / (4.0 * C.PI * e45 * rho_m)
    for e45, rho_m in zip(EXP45LOGSIG_PAMDM, DENSITY_AER))
SMCRITFACTOR_PAMDM = tuple(
    2.0 * ATEN_PAMDM * math.sqrt(ATEN_PAMDM / (27.0 * max(1.0e-10, h)))
    for h in HYGRO_AER)
AMCUBE_PAMDM = tuple(a / n for a, n in zip(AMCUBEFACTOR_PAMDM, NUM_TO_MASS_AER))
SMCRIT_PAMDM = tuple(
    (sf / math.sqrt(am)) if h > 1.0e-10 else 100.0
    for sf, am, h in zip(SMCRITFACTOR_PAMDM, AMCUBE_PAMDM, HYGRO_AER))
LNSM_PAMDM = tuple(math.log(s) for s in SMCRIT_PAMDM)

SUPERSAT_PAMDM = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)   # % supersaturation
PSAT_PAMDM = 7

# ===========================================================================
# DERF1 (Ooura erf, aero.F l.5334-5445).  Coefficients row-major: 5 rows of 13.
# ===========================================================================
_DERF_A = np.array([
    0.00000000005958930743e0, -0.00000000113739022964e0,
    0.00000001466005199839e0, -0.00000016350354461960e0,
    0.00000164610044809620e0, -0.00001492559551950604e0,
    0.00012055331122299265e0, -0.00085483269811296660e0,
    0.00522397762482322257e0, -0.02686617064507733420e0,
    0.11283791670954881569e0, -0.37612638903183748117e0,
    1.12837916709551257377e0,
    0.00000000002372510631e0, -0.00000000045493253732e0,
    0.00000000590362766598e0, -0.00000006642090827576e0,
    0.00000067595634268133e0, -0.00000621188515924000e0,
    0.00005103883009709690e0, -0.00037015410692956173e0,
    0.00233307631218880978e0, -0.01254988477182192210e0,
    0.05657061146827041994e0, -0.21379664776456006580e0,
    0.84270079294971486929e0,
    0.00000000000949905026e0, -0.00000000018310229805e0,
    0.00000000239463074000e0, -0.00000002721444369609e0,
    0.00000028045522331686e0, -0.00000261830022482897e0,
    0.00002195455056768781e0, -0.00016358986921372656e0,
    0.00107052153564110318e0, -0.00608284718113590151e0,
    0.02986978465246258244e0, -0.13055593046562267625e0,
    0.67493323603965504676e0,
    0.00000000000382722073e0, -0.00000000007421598602e0,
    0.00000000097930574080e0, -0.00000001126008898854e0,
    0.00000011775134830784e0, -0.00000111992758382650e0,
    0.00000962023443095201e0, -0.00007404402135070773e0,
    0.00050689993654144881e0, -0.00307553051439272889e0,
    0.01668977892553165586e0, -0.08548534594781312114e0,
    0.56909076642393639985e0,
    0.00000000000155296588e0, -0.00000000003032205868e0,
    0.00000000040424830707e0, -0.00000000471135111493e0,
    0.00000005011915876293e0, -0.00000048722516178974e0,
    0.00000430683284629395e0, -0.00003445026145385764e0,
    0.00024879276133931664e0, -0.00162940941748079288e0,
    0.00988786373932350462e0, -0.05962426839442303805e0,
    0.49766113250947636708e0], dtype=np.float64).reshape(5, 13)
_DERF_B = np.array([
    -0.00000000029734388465e0, 0.00000000269776334046e0,
    -0.00000000640788827665e0, -0.00000001667820132100e0,
    -0.00000021854388148686e0, 0.00000266246030457984e0,
    0.00001612722157047886e0, -0.00025616361025506629e0,
    0.00015380842432375365e0, 0.00815533022524927908e0,
    -0.01402283663896319337e0, -0.19746892495383021487e0,
    0.71511720328842845913e0,
    -0.00000000001951073787e0, -0.00000000032302692214e0,
    0.00000000522461866919e0, 0.00000000342940918551e0,
    -0.00000035772874310272e0, 0.00000019999935792654e0,
    0.00002687044575042908e0, -0.00011843240273775776e0,
    -0.00080991728956032271e0, 0.00661062970502241174e0,
    0.00909530922354827295e0, -0.20160072778491013140e0,
    0.51169696718727644908e0,
    0.00000000003147682272e0, -0.00000000048465972408e0,
    0.00000000063675740242e0, 0.00000003377623323271e0,
    -0.00000015451139637086e0, -0.00000203340624738438e0,
    0.00001947204525295057e0, 0.00002854147231653228e0,
    -0.00101565063152200272e0, 0.00271187003520095655e0,
    0.02328095035422810727e0, -0.16725021123116877197e0,
    0.32490054966649436974e0,
    0.00000000002319363370e0, -0.00000000006303206648e0,
    -0.00000000264888267434e0, 0.00000002050708040581e0,
    0.00000011371857327578e0, -0.00000211211337219663e0,
    0.00000368797328322935e0, 0.00009823686253424796e0,
    -0.00065860243990455368e0, -0.00075285814895230877e0,
    0.02585434424202960464e0, -0.11637092784486193258e0,
    0.18267336775296612024e0,
    -0.00000000000367789363e0, 0.00000000020876046746e0,
    -0.00000000193319027226e0, -0.00000000435953392472e0,
    0.00000018006992266137e0, -0.00000078441223763969e0,
    -0.00000675407647949153e0, 0.00008428418334440096e0,
    -0.00017604388937031815e0, -0.00239729611435071610e0,
    0.02064129023876022970e0, -0.06905562880005864105e0,
    0.09084526782065478489e0], dtype=np.float64).reshape(5, 13)


def derf1_host(x):
    """Host/scalar DERF1 (same arithmetic as the traced version), for init."""
    w = abs(float(x))
    if w < 2.2:
        t = w * w
        k = int(t)
        t = t - k
        a = _DERF_A[k]
        y = a[0]
        for i in range(1, 13):
            y = y * t + a[i]
        y = y * w
    elif w < 6.9:
        k = int(w)
        t = w - k
        b = _DERF_B[k - 2]
        y = b[0]
        for i in range(1, 13):
            y = y * t + b[i]
        y = y * y
        y = y * y
        y = y * y
        y = 1.0 - y * y
    else:
        y = 1.0
    return -y if x < 0 else y


# CCNFACT table (aero.F l.669-686): factor for diagnostic CCN at the 7
# supersaturations, per mode.  ccnfact[s][m].
def _build_ccnfact():
    tab = np.zeros((PSAT_PAMDM, NAER_CU), dtype=np.float64)
    for m in range(NAER_CU):
        for s in range(PSAT_PAMDM):
            super_s = 0.01 * SUPERSAT_PAMDM[s]
            arg = ARGFACTOR_PAMDM[m] * math.log(SMCRIT_PAMDM[m] / super_s)
            if arg < 2.0:
                if arg < -2.0:
                    tab[s, m] = 1.0e-6
                else:
                    tab[s, m] = 1.0e-6 * 0.5 * (1.0 - derf1_host(arg))
            else:
                tab[s, m] = 0.0
    return tab


CCNFACT_PAMDM = _build_ccnfact()


def derf1(x):
    """Traced (JAX) DERF1, vectorized; exact port of the Fortran algorithm."""
    x = jnp.asarray(x)
    w = jnp.abs(x)
    a_tab = jnp.asarray(_DERF_A, x.dtype)
    b_tab = jnp.asarray(_DERF_B, x.dtype)
    # |x| < 2.2 : T = w*w, K = INT(T) in 0..4
    t1 = w * w
    k1 = jnp.minimum(t1, 4.0).astype(jnp.int32)
    t1f = t1 - k1.astype(x.dtype)
    a = a_tab[k1]                       # (..., 13)
    y1 = a[..., 0]
    for i in range(1, 13):
        y1 = y1 * t1f + a[..., i]
    y1 = y1 * w
    # 2.2 <= |x| < 6.9 : K = INT(w) in 2..6
    k2 = jnp.clip(jnp.minimum(w, 6.0).astype(jnp.int32), 2, 6)
    t2 = w - k2.astype(x.dtype)
    b = b_tab[k2 - 2]
    y2 = b[..., 0]
    for i in range(1, 13):
        y2 = y2 * t2 + b[..., i]
    y2 = y2 * y2
    y2 = y2 * y2
    y2 = y2 * y2
    y2 = 1.0 - y2 * y2
    y = jnp.where(w < 2.2, y1, jnp.where(w < 6.9, y2, jnp.ones_like(w)))
    return jnp.where(x < 0.0, -y, y)


# ===========================================================================
# mdm_prescribed_activate + mdm_prescribed_maxsat (AR&G 2000, 10 modes).
# naero/maero: (..., 10) arrays in #/m3 and kg/m3.  Returns nact in #/kg.
# ===========================================================================
def mdm_prescribed_activate(wbar, tair, rhoair, naero, maero, latvap):
    one = jnp.ones_like(tair)
    wpos = wbar > 0.0
    wsafe = jnp.where(wpos, wbar, 1.0)

    pres = C.R * rhoair * tair
    diff0 = 0.211e-4 * (P0 / pres) * (tair / T0) ** 1.94
    conduct0 = (5.69 + 0.017 * (tair - T0)) * 4.186e2 * 1.0e-5
    es = 1000.0 * SVP1 * jnp.exp(SVP2 * (tair - T0) / (tair - SVP3))
    qs = C.EP_2 * es / (pres - es)
    dqsdt = latvap / (C.RV * tair * tair) * qs
    alpha = C.G * (latvap / (C.CP * C.RV * tair * tair) - 1.0 / (C.R * tair))
    gammaloc = (1.0 + latvap / C.CP * dqsdt) / (rhoair * qs)
    gloc = 1.0 / (C.RHOW / (diff0 * rhoair * qs)
                  + latvap * C.RHOW / (conduct0 * tair)
                  * (latvap / (C.RV * tair) - 1.0))
    sqrtg = jnp.sqrt(gloc)
    beta = 4.0 * C.PI * C.RHOW * gloc * gammaloc
    etafactor2max = 1.0e10 / (alpha * wsafe) ** 1.5

    alw = alpha * wsafe
    sqrtalw = jnp.sqrt(alw)
    zeta = 2.0 * sqrtalw * ATEN_PAMDM / (3.0 * sqrtg)
    etafactor1 = 2.0 * alw * sqrtalw

    smc = []
    eta = []
    for m in range(NAER_CU):
        na_m = naero[..., m]
        volc = maero[..., m] / DENSITY_AER[m]
        ok = (volc > REAL_TINY) & (na_m > REAL_TINY)
        na_safe = jnp.where(ok, na_m, 1.0)
        etaf2 = jnp.where(ok, 1.0 / (na_safe * beta * sqrtg), etafactor2max)
        amcube = 3.0 * volc / (4.0 * C.PI * EXP45LOGSIG_PAMDM[m] * na_safe)
        # hygro_aer(m) > 1e-10 for every mode in the DATA table (static branch)
        amcube_safe = jnp.where(ok, amcube, 1.0)
        smc_m = jnp.where(
            ok,
            2.0 * ATEN_PAMDM * jnp.sqrt(
                ATEN_PAMDM / (27.0 * HYGRO_AER[m] * amcube_safe)),
            1.0)
        smc.append(smc_m)
        eta.append(etafactor1 * etaf2)

    # ---- mdm_prescribed_maxsat ----
    weak_all = jnp.ones_like(tair, dtype=bool)
    for m in range(NAER_CU):
        weak_all = weak_all & ((zeta > 1.0e5 * eta[m])
                               | (smc[m] * smc[m] > 1.0e5 * eta[m]))
    ssum = jnp.zeros_like(tair)
    for m in range(NAER_CU):
        eta_ok = eta[m] > 1.0e-20
        eta_safe = jnp.where(eta_ok, eta[m], 1.0)
        g1 = jnp.sqrt(zeta / eta_safe)
        g1 = g1 * g1 * g1
        g2 = smc[m] / jnp.sqrt(eta_safe + 3.0 * zeta)
        g2 = jnp.sqrt(g2)
        g2 = g2 * g2 * g2
        term = (F1_PAMDM[m] * g1 + F2_PAMDM[m] * g2) / (smc[m] * smc[m])
        # Fortran: sum=sum+term when eta>1e-20 else sum=1.e20 (assignment)
        ssum = jnp.where(eta_ok, ssum + term, jnp.full_like(ssum, 1.0e20))
    smax = jnp.where(weak_all, jnp.full_like(ssum, 1.0e-20),
                     1.0 / jnp.sqrt(ssum))

    lnsmax = jnp.log(smax)
    nact = jnp.zeros_like(tair)
    for m in range(NAER_CU):
        xarg = 2.0 * (jnp.log(smc[m]) - lnsmax) / (3.0 * SQ2_PAMDM
                                                   * ALOGSIG_PAMDM[m])
        nact = nact + 0.5 * (1.0 - derf1(xarg)) * naero[..., m]
    nact = nact / rhoair          # #/m3 -> #/kg
    # if(wbar.le.0.)return  (nact stays 0)
    return jnp.where(wpos, nact, jnp.zeros_like(nact))


# ===========================================================================
# mdm_prescribed_nucleati (+hf/hetero): Liu & Penner 2005 + Meyers mixed-phase.
# Returns nuci (#/kg).
# ===========================================================================
def _mdm_hetero(t_c, lnw, ns_num):
    a11, a12 = 0.0263, -0.0185
    a21, a22 = 2.758, 1.3221
    b11, b12 = -0.008, -0.0468
    b21, b22 = -0.2667, -1.4588
    ns_safe = jnp.maximum(ns_num, REAL_TINY)
    ln_ns = jnp.log(ns_safe)
    bb = (a11 + b11 * ln_ns) * lnw + (a12 + b12 * ln_ns)
    cc = a21 + b21 * ln_ns
    nis = math.exp(a22) * ns_safe ** b22 * jnp.exp(bb * t_c) * jnp.exp(cc * lnw)
    nis = jnp.minimum(nis, ns_num)
    return nis  # Nid = 0 (deposition excluded for cirrus, per the Fortran)


def _mdm_hf(t_c, lnw, relhum, subgrid, na):
    a1_f, b1_f, c1_f, c2_f = 0.0231, -0.008, 0.0739, 1.2372
    a21_f, a22_f = -1.6387, -6.045
    b21_f, b22_f = -0.042, -0.112
    a1_s, a2_s = -0.3949, 1.282
    b1_s, b2_s, b3_s = -0.0156, 0.0111, 0.0217
    c1_s, c2_s = 0.120, 2.312

    aa = 6.0e-4 * lnw + 6.6e-3
    bb = 6.0e-2 * lnw + 1.052
    cc = 1.68 * lnw + 129.35
    rhw = (aa * t_c * t_c + bb * t_c + cc) * 0.01
    gate = (t_c <= -37.0) & ((relhum * subgrid) >= rhw)
    regm = 6.07 * lnw - 55.0

    na_safe = jnp.maximum(na, REAL_TINY)
    a2f = jnp.where(t_c > -64.0, a21_f, a22_f)
    b2f = jnp.where(t_c > -64.0, b21_f, b22_f)
    k1_fast = jnp.exp(a2f + b2f * t_c + c2_f * lnw)
    k2_fast = a1_f + b1_f * t_c + c1_f * lnw
    ni_fast = jnp.minimum(k1_fast * na_safe ** k2_fast, na)
    k1_slow = jnp.exp(a2_s + (b2_s + b3_s * lnw) * t_c + c2_s * lnw)
    k2_slow = a1_s + b1_s * t_c + c1_s * lnw
    ni_slow = jnp.minimum(k1_slow * na_safe ** k2_slow, na)
    ni = jnp.where(t_c >= regm, ni_fast, ni_slow)
    return jnp.where(gate, ni, jnp.zeros_like(ni))


def mdm_prescribed_nucleati(wbar, tair, relhum, relhumi, qc, rhoair, naero):
    """Liu-Penner 2005 aerosol ice nucleation + Meyers.  naero: (...,10) #/m3."""
    so4_num = naero[..., 0] * 1.0e-6      # mode 1 = sulfate, #/cm3
    soot_num = naero[..., 9] * 1.0e-6     # mode 10 = BCPHI
    dst_num = (naero[..., 2] + naero[..., 3] + naero[..., 4]
               + naero[..., 5]) * 1.0e-6  # modes 3-6 = dust
    tc = tair - 273.15
    subgrid = 1.2

    wsafe = jnp.maximum(wbar, REAL_TINY)
    lnw = jnp.log(wsafe)
    sd = soot_num + dst_num
    sd_safe = jnp.maximum(sd, REAL_TINY)
    ln_sd = jnp.log(sd_safe)

    main_gate = ((so4_num >= 1.0e-10) & (sd >= 1.0e-10)
                 & (tc <= -35.0) & ((relhumi * subgrid) >= 1.2))

    # <regm => T in Eq.10 of Liu et al. 2007>
    aa = -1.4938 * ln_sd + 12.884
    bb = -10.41 * ln_sd - 67.69
    regm = aa * lnw + bb

    nihf_tc = _mdm_hf(tc, lnw, relhum, subgrid, so4_num)
    nihf_r5 = _mdm_hf(regm - 5.0, lnw, relhum, subgrid, so4_num)
    niimm_tc = _mdm_hetero(tc, lnw, sd)
    niimm_rg = _mdm_hetero(regm, lnw, sd)

    excl = (tc < -40.0) & (wbar > 1.0)   # exclude T<-40 & W>1 m/s from hetero
    # transition interpolation (guard 0**negative when niimm_rg == 0 exactly)
    nihf_r5_safe = jnp.maximum(nihf_r5, REAL_TINY)
    base = niimm_rg / nihf_r5_safe
    base_safe = jnp.where(niimm_rg > 0.0, base, 1.0)
    n1_interp = jnp.where(nihf_r5 <= niimm_rg, nihf_r5,
                          niimm_rg * base_safe ** ((tc - regm) / 5.0))
    n1 = jnp.where(
        tc > regm,
        jnp.where(excl, nihf_tc, niimm_tc),
        jnp.where(tc < regm - 5.0, nihf_tc,
                  jnp.where(excl, nihf_tc, n1_interp)))
    ni = jnp.where(main_gate, n1, jnp.zeros_like(n1))

    # Meyers 1992 deposition/condensation in mixed clouds (TWG-fixed form)
    deles = jnp.where(relhumi > 1.5, jnp.full_like(relhumi, 1.5), relhumi)
    nimey = jnp.where((tc < 0.0) & (tc > -37.0) & (qc > 1.0e-12),
                      1.0e-3 * jnp.exp(12.96 * (deles - 1.0) - 0.639),
                      jnp.zeros_like(tc))

    nuci = ni + nimey
    # Fortran STOPs on out-of-range; the port applies the same reset only.
    nuci = jnp.where((nuci > 9999.0) | (nuci < 0.0), jnp.zeros_like(nuci), nuci)
    return nuci * 1.0e6 / rhoair          # #/cm3 -> #/kg


# ===========================================================================
# Slope helper (identical to base _morrison_cold._slope_generic).
# ===========================================================================
def _slope_generic(q, n, cmass, inv_pow, lammin, lammax, qsmall):
    active = q >= qsmall
    qs = jnp.where(active, q, 1.0)
    ns = jnp.maximum(n, 0.0)
    lam = (cmass * jnp.where(active, ns, 0.0) / qs) ** inv_pow
    lam = jnp.where(active, lam, lammin)
    too_small = lam < lammin
    too_big = lam > lammax
    lam_c = jnp.where(too_small, lammin, jnp.where(too_big, lammax, lam))
    n0_c = lam_c ** 4 * jnp.where(active, q, 0.0) / cmass
    n_c = jnp.where(too_small | too_big, n0_c / lam_c, ns)
    n0 = jnp.where(too_small | too_big, n0_c, jnp.where(active, ns * lam, 0.0))
    lam_out = jnp.where(active, lam_c, 0.0)
    n_out = jnp.where(active, n_c, n)
    n0_out = jnp.where(active, n0, 0.0)
    return lam_out, n0_out, n_out


# ===========================================================================
# COLD BRANCH (T < 273.15) — copy of base _cold_branch with the aero deltas.
# ===========================================================================
def _cold_branch_aero(cold, t, qv, qc, qr, qi, qs, qg, nc, ni, ns, nr, ng,
                      qvs, qvi, qvqvs, qvqvsi, ab, abi, rho, dv, mu, sc, kap,
                      ain, arn, asn, agn, acn, dum_dc, xxlv, xxls, xlf, cpm,
                      p, w, wvar, naer, dt):
    QSMALL = C.QSMALL
    PI = C.PI
    eps = _EPS
    one = jnp.ones_like(t)

    # Fortran cold-branch entry: NI3D..NG3D, NC3D = MAX(0., .) in place.
    ni = jnp.maximum(ni, 0.0)
    ns = jnp.maximum(ns, 0.0)
    nr = jnp.maximum(nr, 0.0)
    ng = jnp.maximum(ng, 0.0)
    nc = jnp.maximum(nc, 0.0)

    # slope params — clamp write-backs are in-place in the Fortran (NI3D etc.),
    # so the clamped values feed every downstream use.
    lami, n0i, ni = _slope_generic(qi, ni, C.CONS12, 1.0 / C.DI,
                                   LAMMINI, C.LAMMAXI, QSMALL)
    lamr, n0rr, nr = _slope_generic(qr, nr, PI * C.RHOW, 1.0 / 3.0,
                                    C.LAMMINR, C.LAMMAXR, QSMALL)
    lams, n0s, ns = _slope_generic(qs, ns, C.CONS1, 1.0 / C.DS,
                                   C.LAMMINS, C.LAMMAXS, QSMALL)
    lamg, n0g, ng = _slope_generic(qg, ng, C.CONS2, 1.0 / C.DG,
                                   C.LAMMING, C.LAMMAXG, QSMALL)

    # droplet PGAM/LAMC/CDIST1 (in-place NC3D clamp back-adjust)
    qc_act = qc >= QSMALL
    dum_rho = p / (287.15 * t)
    pgam = 0.0005714 * (nc / 1.0e6 * dum_rho) + 0.2714
    pgam = 1.0 / (pgam * pgam) - 1.0
    pgam = jnp.clip(pgam, 2.0, 10.0)
    g_p1 = gamma_fn(pgam + 1.0)
    g_p4 = gamma_fn(pgam + 4.0)
    g_p5 = gamma_fn(pgam + 5.0)
    g_p2 = gamma_fn(pgam + 2.0)
    g_p7 = gamma_fn(7.0 + pgam)
    qcs = jnp.where(qc_act, qc, 1.0)
    lamc = (C.CONS26 * nc * g_p4 / (qcs * g_p1)) ** (1.0 / 3.0)
    lammin = (pgam + 1.0) / 60.0e-6
    lammax = (pgam + 1.0) / 1.0e-6
    too_s = lamc < lammin
    too_b = lamc > lammax
    lamc = jnp.where(too_s, lammin, jnp.where(too_b, lammax, lamc))
    nc_adj = jnp.exp(3.0 * jnp.log(lamc) + jnp.log(qcs)
                     + jnp.log(g_p1) - jnp.log(g_p4)) / C.CONS26
    nc = jnp.where(qc_act & (too_s | too_b), nc_adj, nc)
    cdist1 = jnp.where(qc_act, nc / g_p1, 0.0)
    lamc = jnp.where(qc_act, lamc, 0.0)

    # ---- contact + immersion freezing of cloud droplets (T < 269.15) ----
    frz = qc_act & (t < 269.15)
    nacnt = jnp.exp(-2.80 + 0.262 * (273.15 - t)) * 1000.0
    dum_mfp = 7.37 * t / (288.0 * 10.0 * p) / 100.0
    dap = C.CONS37 * t * (1.0 + dum_mfp / C.RIN) / mu
    lamc_safe = jnp.where(lamc > 0, lamc, one)
    mnuccc = jnp.where(frz, C.CONS38 * dap * nacnt
                       * jnp.exp(jnp.log(jnp.where(cdist1 > 0, cdist1, one))
                                 + jnp.log(g_p5) - 4.0 * jnp.log(lamc_safe)), 0.0)
    nnuccc = jnp.where(frz, 2.0 * PI * dap * nacnt * cdist1 * g_p2 / lamc_safe, 0.0)
    mnuccc = jnp.where(frz, mnuccc + C.CONS39
                       * jnp.exp(jnp.log(jnp.where(cdist1 > 0, cdist1, one))
                                 + jnp.log(g_p7) - 6.0 * jnp.log(lamc_safe))
                       * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0), 0.0)
    nnuccc = jnp.where(frz, nnuccc + C.CONS40
                       * jnp.exp(jnp.log(jnp.where(cdist1 > 0, cdist1, one))
                                 + jnp.log(g_p4) - 3.0 * jnp.log(lamc_safe))
                       * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0), 0.0)
    nnuccc = jnp.minimum(nnuccc, nc / dt)
    nnuccc = jnp.where(frz, nnuccc, 0.0)
    mnuccc = jnp.where(frz, mnuccc, 0.0)

    # ---- autoconversion (KK2000) ----
    qc_ge6 = qc >= 1.0e-6
    prc = jnp.where(qc_ge6, 1350.0 * qc ** 2.47 * (nc / 1.0e6 * rho) ** (-1.79), 0.0)
    nprc1 = jnp.where(qc_ge6, prc / C.CONS29, 0.0)
    nprc = jnp.where(qc_ge6, prc / (qc / jnp.where(nc > 0, nc, one)), 0.0)
    nprc = jnp.minimum(nprc, nc / dt)
    nprc1 = jnp.minimum(nprc1, nprc)

    # ---- snow aggregation ----
    qs8 = qs >= 1.0e-8
    nsagg = jnp.where(qs8, C.CONS15 * asn * rho ** ((2.0 + C.BS) / 3.0)
                      * qs ** ((2.0 + C.BS) / 3.0)
                      * (ns * rho) ** ((4.0 - C.BS) / 3.0) / rho, 0.0)

    # ---- accretion of droplets onto snow (PSACWS) ----
    sn_qc = qs8 & qc_act
    psacws = jnp.where(sn_qc, C.CONS13 * asn * qc * rho * n0s
                       / (lams ** (C.BS + 3.0) + eps), 0.0)
    npsacws = jnp.where(sn_qc, C.CONS13 * asn * nc * rho * n0s
                        / (lams ** (C.BS + 3.0) + eps), 0.0)

    # ---- collection of droplets by graupel (PSACWG) ----
    qg8 = qg >= 1.0e-8
    g_qc = qg8 & qc_act
    psacwg = jnp.where(g_qc, C.CONS14 * agn * qc * rho * n0g
                       / (lamg ** (C.BG + 3.0) + eps), 0.0)
    npsacwg = jnp.where(g_qc, C.CONS14 * agn * nc * rho * n0g
                        / (lamg ** (C.BG + 3.0) + eps), 0.0)

    # ---- cloud ice collecting droplets (PSACWI), only if 1/lami >= 100 um ----
    qi8 = qi >= 1.0e-8
    lami_safe = jnp.where(lami > 0, lami, one)
    ice_rime = qi8 & qc_act & (1.0 / lami_safe >= 100.0e-6)
    psacwi = jnp.where(ice_rime, C.CONS16 * ain * qc * rho * n0i
                       / (lami ** (C.BI + 3.0) + eps), 0.0)
    npsacwi = jnp.where(ice_rime, C.CONS16 * ain * nc * rho * n0i
                        / (lami ** (C.BI + 3.0) + eps), 0.0)

    # ---- accretion of rain by snow (PRACS), and PSACR ----
    qr8 = qr >= 1.0e-8
    rs = qr8 & qs8
    ums = jnp.minimum(asn * C.CONS3 / (lams ** C.BS + eps), 1.2 * dum_dc)
    umr = jnp.minimum(arn * C.CONS4 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    uns = jnp.minimum(asn * C.CONS5 / (lams ** C.BS + eps), 1.2 * dum_dc)
    unr = jnp.minimum(arn * C.CONS6 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    pracs = jnp.where(
        rs, C.CONS41 * (((1.2 * umr - 0.95 * ums) ** 2 + 0.08 * ums * umr) ** 0.5
                        * rho * n0rr * n0s / (lamr ** 3 + eps)
                        * (5.0 / (lamr ** 3 * lams + eps)
                           + 2.0 / (lamr ** 2 * lams ** 2 + eps)
                           + 0.5 / (lamr * lams ** 3 + eps))), 0.0)
    npracs = jnp.where(
        rs, C.CONS32 * rho * (1.7 * (unr - uns) ** 2 + 0.3 * unr * uns) ** 0.5
        * n0rr * n0s * (1.0 / (lamr ** 3 * lams + eps)
                        + 1.0 / (lamr ** 2 * lams ** 2 + eps)
                        + 1.0 / (lamr * lams ** 3 + eps)), 0.0)
    pracs = jnp.where(rs, jnp.minimum(pracs, qr / dt), 0.0)
    npracs = jnp.where(rs, npracs, 0.0)
    psacr_cond = rs & (qs >= 0.1e-3) & (qr >= 0.1e-3)
    psacr = jnp.where(
        psacr_cond, C.CONS31 * (((1.2 * umr - 0.95 * ums) ** 2 + 0.08 * ums * umr) ** 0.5
                                * rho * n0rr * n0s / (lams ** 3 + eps)
                                * (5.0 / (lams ** 3 * lamr + eps)
                                   + 2.0 / (lams ** 2 * lamr ** 2 + eps)
                                   + 0.5 / (lams * lamr ** 3 + eps))), 0.0)

    # ---- collection of rain by graupel (PRACG) ----
    rg = qr8 & qg8
    umg = jnp.minimum(agn * C.CONS7 / (lamg ** C.BG + eps), 20.0 * dum_dc)
    umr2 = jnp.minimum(arn * C.CONS4 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    ung = jnp.minimum(agn * C.CONS8 / (lamg ** C.BG + eps), 20.0 * dum_dc)
    unr2 = jnp.minimum(arn * C.CONS6 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    pracg = jnp.where(
        rg, C.CONS41 * (((1.2 * umr2 - 0.95 * umg) ** 2 + 0.08 * umg * umr2) ** 0.5
                        * rho * n0rr * n0g / (lamr ** 3 + eps)
                        * (5.0 / (lamr ** 3 * lamg + eps)
                           + 2.0 / (lamr ** 2 * lamg ** 2 + eps)
                           + 0.5 / (lamr * lamg ** 3 + eps))), 0.0)
    npracg = jnp.where(
        rg, C.CONS32 * rho * (1.7 * (unr2 - ung) ** 2 + 0.3 * unr2 * ung) ** 0.5
        * n0rr * n0g * (1.0 / (lamr ** 3 * lamg + eps)
                        + 1.0 / (lamr ** 2 * lamg ** 2 + eps)
                        + 1.0 / (lamr * lamg ** 3 + eps)), 0.0)
    pracg = jnp.where(rg, jnp.minimum(pracg, qr / dt), 0.0)
    npracg = jnp.where(rg, npracg, 0.0)

    # ---- rime-splintering (Hallett-Mossop) for snow and graupel ----
    def _fmult(tt):
        f = jnp.where(tt > 270.16, 0.0,
                      jnp.where(tt > 268.16, (270.16 - tt) / 2.0,
                                jnp.where(tt >= 265.16, (tt - 265.16) / 3.0, 0.0)))
        return f
    in_band = (t < 270.16) & (t > 265.16)
    fmult = jnp.where(in_band, _fmult(t), 0.0)

    snow_split = (qs >= 0.1e-3) & ((qc >= 0.5e-3) | (qr >= 0.1e-3)) \
        & ((psacws > 0.0) | (pracs > 0.0)) & in_band
    nmults = jnp.where(snow_split & (psacws > 0.0), 35.0e4 * psacws * fmult * 1000.0, 0.0)
    qmults = nmults * C.MMULT
    qmults = jnp.where(snow_split & (psacws > 0.0), jnp.minimum(qmults, psacws), 0.0)
    psacws = jnp.where(snow_split & (psacws > 0.0), psacws - qmults, psacws)
    nmultr = jnp.where(snow_split & (pracs > 0.0), 35.0e4 * pracs * fmult * 1000.0, 0.0)
    qmultr = nmultr * C.MMULT
    qmultr = jnp.where(snow_split & (pracs > 0.0), jnp.minimum(qmultr, pracs), 0.0)
    pracs = jnp.where(snow_split & (pracs > 0.0), pracs - qmultr, pracs)

    g_split = (qg >= 0.1e-3) & ((qc >= 0.5e-3) | (qr >= 0.1e-3)) \
        & ((psacwg > 0.0) | (pracg > 0.0)) & in_band
    nmultg = jnp.where(g_split & (psacwg > 0.0), 35.0e4 * psacwg * fmult * 1000.0, 0.0)
    qmultg = nmultg * C.MMULT
    qmultg = jnp.where(g_split & (psacwg > 0.0), jnp.minimum(qmultg, psacwg), 0.0)
    psacwg = jnp.where(g_split & (psacwg > 0.0), psacwg - qmultg, psacwg)
    nmultrg = jnp.where(g_split & (pracg > 0.0), 35.0e4 * pracg * fmult * 1000.0, 0.0)
    qmultrg = nmultrg * C.MMULT
    qmultrg = jnp.where(g_split & (pracg > 0.0), jnp.minimum(qmultrg, pracg), 0.0)
    pracg = jnp.where(g_split & (pracg > 0.0), pracg - qmultrg, pracg)

    # ---- conversion of rimed cloud water on snow to graupel (PGSACW) ----
    pgsacw_cond = (psacws > 0.0) & (qs >= 0.1e-3) & (qc >= 0.5e-3)
    pgsacw = jnp.where(
        pgsacw_cond,
        jnp.minimum(psacws, C.CONS17 * dt * n0s * qc * qc * asn * asn
                    / (rho * lams ** (2.0 * C.BS + 2.0) + eps)), 0.0)
    dum_emb = jnp.maximum(C.RHOSN / (C.RHOG - C.RHOSN) * pgsacw, 0.0)
    nscng = jnp.where(pgsacw_cond, dum_emb / C.MG0 * rho, 0.0)
    nscng = jnp.where(pgsacw_cond, jnp.minimum(nscng, ns / dt), 0.0)
    psacws = jnp.where(pgsacw_cond, psacws - pgsacw, psacws)

    # ---- conversion of rimed rain on snow to graupel (PGRACS) ----
    pgracs_cond = (pracs > 0.0) & (qs >= 0.1e-3) & (qr >= 0.1e-3)
    dum_fr = (C.CONS18 * (4.0 / (lams + eps)) ** 3 * (4.0 / (lams + eps)) ** 3) \
        / (C.CONS18 * (4.0 / (lams + eps)) ** 3 * (4.0 / (lams + eps)) ** 3
           + C.CONS19 * (4.0 / (lamr + eps)) ** 3 * (4.0 / (lamr + eps)) ** 3 + eps)
    dum_fr = jnp.clip(dum_fr, 0.0, 1.0)
    pgracs = jnp.where(pgracs_cond, (1.0 - dum_fr) * pracs, 0.0)
    ngracs = jnp.where(pgracs_cond, (1.0 - dum_fr) * npracs, 0.0)
    ngracs = jnp.where(pgracs_cond, jnp.minimum(ngracs, nr / dt), ngracs)
    ngracs = jnp.where(pgracs_cond, jnp.minimum(ngracs, ns / dt), ngracs)
    pracs = jnp.where(pgracs_cond, pracs - pgracs, pracs)
    npracs = jnp.where(pgracs_cond, npracs - ngracs, npracs)
    psacr = jnp.where(pgracs_cond, psacr * (1.0 - dum_fr), psacr)

    # ---- freezing of rain (Bigg, T < 269.15) -> MNUCCR / NNUCCR ----
    rain_frz = (t < 269.15) & (qr >= QSMALL)
    lamr_safe = jnp.where(lamr > 0, lamr, one)
    mnuccr = jnp.where(rain_frz, C.CONS20 * nr * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0)
                       / lamr_safe ** 3 / lamr_safe ** 3, 0.0)
    nnuccr = jnp.where(rain_frz, PI * nr * C.BIMM * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0)
                       / lamr_safe ** 3, 0.0)
    nnuccr = jnp.where(rain_frz, jnp.minimum(nnuccr, nr / dt), 0.0)

    # ---- accretion of cloud water by rain (PRA) ----
    qrqc8 = (qr >= 1.0e-8) & (qc >= 1.0e-8)
    pra = jnp.where(qrqc8, 67.0 * (qc * qr) ** 1.15, 0.0)
    npra = jnp.where(qrqc8, pra / (qc / jnp.where(nc > 0, nc, one)), 0.0)

    # ---- self-collection of rain (NRAGG) ----
    inv_lamr = jnp.where(lamr > 0, 1.0 / lamr_safe, 0.0)
    br_dum = jnp.where(inv_lamr < 300.0e-6, 1.0,
                       2.0 - jnp.exp(2300.0 * (inv_lamr - 300.0e-6)))
    nragg = jnp.where(qr8, -5.78 * br_dum * nr * qr * rho, 0.0)

    # ---- autoconversion of cloud ice to snow (NPRCI, PRCI) — DCS=350e-6 ----
    ice_aut = qi8 & (qvqvsi >= 1.0)
    nprci = jnp.where(ice_aut, CONS21 * (qv - qvi) * rho * n0i
                      * jnp.exp(-lami * DCS) * dv / abi, 0.0)
    prci = jnp.where(ice_aut, CONS22 * nprci, 0.0)
    nprci = jnp.where(ice_aut, jnp.minimum(nprci, ni / dt), 0.0)

    # ---- accretion of cloud ice by snow (PRAI) ----
    sn_qi = qs8 & (qi >= QSMALL)
    prai = jnp.where(sn_qi, C.CONS23 * asn * qi * rho * n0s
                     / (lams ** (C.BS + 3.0) + eps), 0.0)
    nprai = jnp.where(sn_qi, C.CONS23 * asn * ni * rho * n0s
                      / (lams ** (C.BS + 3.0) + eps), 0.0)
    nprai = jnp.where(sn_qi, jnp.minimum(nprai, ni / dt), 0.0)

    # ---- collision of rain and ice -> snow or graupel (PIACR/PRACI[S]) ----
    ri = (qr >= 1.0e-8) & (qi >= 1.0e-8) & (t <= 273.15)
    big_rain = ri & (qr >= 0.1e-3)
    small_rain = ri & (qr < 0.1e-3)
    niacr = jnp.where(big_rain, C.CONS24 * ni * n0rr * arn
                      / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    piacr = jnp.where(big_rain, C.CONS25 * ni * n0rr * arn
                      / (lamr ** (C.BR + 3.0) + eps) / (lamr ** 3 + eps) * rho, 0.0)
    praci = jnp.where(big_rain, C.CONS24 * qi * n0rr * arn
                      / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    niacr = jnp.where(big_rain, jnp.minimum(jnp.minimum(niacr, nr / dt), ni / dt), niacr)
    niacrs = jnp.where(small_rain, C.CONS24 * ni * n0rr * arn
                       / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    piacrs = jnp.where(small_rain, C.CONS25 * ni * n0rr * arn
                       / (lamr ** (C.BR + 3.0) + eps) / (lamr ** 3 + eps) * rho, 0.0)
    pracis = jnp.where(small_rain, C.CONS24 * qi * n0rr * arn
                       / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    niacrs = jnp.where(small_rain, jnp.minimum(jnp.minimum(niacrs, nr / dt), ni / dt), niacrs)

    # ---- ice nucleation: INUC=2 (aero.F l.3528-3552), Liu-Penner 2005 ----
    # Same trigger as the base Cooper path; KC2 from mdm_prescribed_nucleati
    # with wbar = W + WVAR; NNUCCD floored at 0.
    nuc_cond = ((qvqvs >= 0.999) & (t <= 265.15)) | (qvqvsi >= 1.08)
    wbar = w + wvar
    kc2 = mdm_prescribed_nucleati(wbar, t, qvqvs, qvqvsi, qc, rho, naer)
    do_nuc = nuc_cond & (kc2 > (ni + ns + ng))
    nnuccd = jnp.where(do_nuc,
                       jnp.maximum((kc2 - ni - ns - ng) / dt, 0.0), 0.0)
    mnuccd = jnp.where(do_nuc, nnuccd * C.MI0, 0.0)

    # ---- evap/sub/dep terms for qi, qni, qg, qr ----
    epsi = jnp.where(qi >= QSMALL, 2.0 * PI * n0i * rho * dv / (lami * lami + eps), 0.0)
    epss = jnp.where(qs8 | (qs >= QSMALL),
                     2.0 * PI * n0s * rho * dv
                     * (C.F1S / (lams * lams + eps)
                        + C.F2S * (asn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                        * C.CONS10 / (lams ** C.CONS35 + eps)), 0.0)
    epss = jnp.where(qs >= QSMALL, epss, 0.0)
    epsg = jnp.where(qg >= QSMALL,
                     2.0 * PI * n0g * rho * dv
                     * (C.F1S / (lamg * lamg + eps)
                        + C.F2S * (agn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                        * C.CONS11 / (lamg ** C.CONS36 + eps)), 0.0)
    epsr = jnp.where(qr >= QSMALL,
                     2.0 * PI * n0rr * rho * dv
                     * (C.F1R / (lamr * lamr + eps)
                        + C.F2R * (arn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                        * C.CONS9 / (lamr ** C.CONS34 + eps)), 0.0)

    qi_ge = qi >= QSMALL
    dum_frac = jnp.where(qi_ge, (1.0 - jnp.exp(-lami * DCS) * (1.0 + lami * DCS)), 0.0)
    prd = jnp.where(qi_ge, epsi * (qv - qvi) / abi * dum_frac, 0.0)
    qs_ge = qs >= QSMALL
    prds = jnp.where(qs_ge, epss * (qv - qvi) / abi
                     + epsi * (qv - qvi) / abi * (1.0 - dum_frac), 0.0)
    prd = jnp.where(qi_ge & (~qs_ge), prd + epsi * (qv - qvi) / abi * (1.0 - dum_frac), prd)
    prdg = epsg * (qv - qvi) / abi
    pre = jnp.where(qv < qvs, jnp.minimum(epsr * (qv - qvs) / ab, 0.0), 0.0)

    # FUDGEF clamp on deposition
    dum_dep = (qv - qvi) / dt
    fudgef = 0.9999
    sum_dep = prd + prds + mnuccd + prdg
    clamp = ((dum_dep > 0.0) & (sum_dep > dum_dep * fudgef)) \
        | ((dum_dep < 0.0) & (sum_dep < dum_dep * fudgef))
    scale = jnp.where(clamp, fudgef * dum_dep / jnp.where(sum_dep != 0, sum_dep, one), 1.0)
    mnuccd = jnp.where(clamp, mnuccd * scale, mnuccd)
    prd = jnp.where(clamp, prd * scale, prd)
    prds = jnp.where(clamp, prds * scale, prds)
    prdg = jnp.where(clamp, prdg * scale, prdg)

    eprd = jnp.where(prd < 0.0, prd, 0.0)
    prd = jnp.where(prd < 0.0, 0.0, prd)
    eprds = jnp.where(prds < 0.0, prds, 0.0)
    prds = jnp.where(prds < 0.0, 0.0, prds)
    eprdg = jnp.where(prdg < 0.0, prdg, 0.0)
    prdg = jnp.where(prdg < 0.0, 0.0, prdg)

    # ---- conservation of water (negative-process adjustment) ----
    dum = (prc + pra + mnuccc + psacws + psacwi + qmults + psacwg + pgsacw + qmultg) * dt
    cqc = (dum > qc) & (qc >= QSMALL)
    rr = jnp.where(cqc, qc / jnp.where(dum != 0, dum, one), 1.0)
    prc = prc * rr; pra = pra * rr; mnuccc = mnuccc * rr; psacws = psacws * rr
    psacwi = psacwi * rr; qmults = qmults * rr; qmultg = qmultg * rr
    psacwg = psacwg * rr; pgsacw = pgsacw * rr

    dum = (-prd - mnuccc + prci + prai - qmults - qmultg - qmultr - qmultrg
           - mnuccd + praci + pracis - eprd - psacwi) * dt
    cqi = (dum > qi) & (qi >= QSMALL)
    denom_i = (prci + prai + praci + pracis - eprd)
    ri_ = jnp.where(cqi & (denom_i != 0),
                    (qi / dt + prd + mnuccc + qmults + qmultg + qmultr + qmultrg
                     + mnuccd + psacwi) / jnp.where(denom_i != 0, denom_i, one), 1.0)
    prci = jnp.where(cqi, prci * ri_, prci)
    prai = jnp.where(cqi, prai * ri_, prai)
    praci = jnp.where(cqi, praci * ri_, praci)
    pracis = jnp.where(cqi, pracis * ri_, pracis)
    eprd = jnp.where(cqi, eprd * ri_, eprd)

    dum = ((pracs - pre) + (qmultr + qmultrg - prc) + (mnuccr - pra)
           + piacr + piacrs + pgracs + pracg) * dt
    cqr = (dum > qr) & (qr >= QSMALL)
    denom_r = (-pre + qmultr + qmultrg + pracs + mnuccr + piacr + piacrs + pgracs + pracg)
    rr2 = jnp.where(cqr & (denom_r != 0),
                    (qr / dt + prc + pra) / jnp.where(denom_r != 0, denom_r, one), 1.0)
    pre = jnp.where(cqr, pre * rr2, pre)
    pracs = jnp.where(cqr, pracs * rr2, pracs)
    qmultr = jnp.where(cqr, qmultr * rr2, qmultr)
    qmultrg = jnp.where(cqr, qmultrg * rr2, qmultrg)
    mnuccr = jnp.where(cqr, mnuccr * rr2, mnuccr)
    piacr = jnp.where(cqr, piacr * rr2, piacr)
    piacrs = jnp.where(cqr, piacrs * rr2, piacrs)
    pgracs = jnp.where(cqr, pgracs * rr2, pgracs)
    pracg = jnp.where(cqr, pracg * rr2, pracg)

    dum = (-prds - psacws - prai - prci - pracs - eprds + psacr - piacrs - pracis) * dt
    cqs = (dum > qs) & (qs >= QSMALL)
    denom_s = (-eprds + psacr)
    rs2 = jnp.where(cqs & (denom_s != 0),
                    (qs / dt + prds + psacws + prai + prci + pracs + piacrs + pracis)
                    / jnp.where(denom_s != 0, denom_s, one), 1.0)
    eprds = jnp.where(cqs, eprds * rs2, eprds)
    psacr = jnp.where(cqs, psacr * rs2, psacr)

    dum = (-psacwg - pracg - pgsacw - pgracs - prdg - mnuccr - eprdg - piacr
           - praci - psacr) * dt
    cqg = (dum > qg) & (qg >= QSMALL)
    rg2 = jnp.where(cqg & (eprdg != 0),
                    (qg / dt + psacwg + pracg + pgsacw + pgracs + prdg + mnuccr
                     + psacr + piacr + praci) / jnp.where(eprdg != 0, -eprdg, one), 1.0)
    eprdg = jnp.where(cqg, eprdg * rg2, eprdg)

    # ---- tendencies ----
    qv_t = (-pre - prd - prds - mnuccd - eprd - eprds - prdg - eprdg)
    t_t = (pre * xxlv
           + (prd + prds + mnuccd + eprd + eprds + prdg + eprdg) * xxls
           + (psacws + psacwi + mnuccc + mnuccr + qmults + qmultg + qmultr + qmultrg
              + pracs + psacwg + pracg + pgsacw + pgracs + piacr + piacrs) * xlf) / cpm
    qc_t = (-pra - prc - mnuccc
            - psacws - psacwi - qmults - qmultg - psacwg - pgsacw)
    qi_t = (prd + eprd + psacwi + mnuccc - prci - prai
            + qmults + qmultg + qmultr + qmultrg + mnuccd - praci - pracis)
    qr_t = (pre + pra + prc - pracs - mnuccr - qmultr - qmultrg
            - piacr - piacrs - pracg - pgracs)
    qni_t = (prai + psacws + prds + pracs + prci + eprds - psacr + piacrs + pracis)
    ns_t = (nsagg + nprci - nscng - ngracs + niacrs)
    qg_t = (pracg + psacwg + pgsacw + pgracs + prdg + eprdg + mnuccr + piacr
            + praci + psacr)
    ng_t = (nscng + ngracs + nnuccr + niacr)
    nc_t = (-nnuccc - npsacws - npra - nprc - npsacwi - npsacwg)
    ni_t = (nnuccc - nprci - nprai + nmults + nmultg + nmultr + nmultrg
            + nnuccd - niacr - niacrs)
    nr_t = (nprc1 - npracs - nnuccr + nragg - niacr - niacrs - npracg - ngracs)

    # number sublimation (NSUBI/NSUBS/NSUBR/NSUBG)
    nsubi = jnp.where(eprd < 0.0, jnp.maximum(-1.0, eprd * dt / jnp.where(qi > 0, qi, one))
                      * ni / dt, 0.0)
    nsubs = jnp.where(eprds < 0.0, jnp.maximum(-1.0, eprds * dt / jnp.where(qs > 0, qs, one))
                      * ns / dt, 0.0)
    nsubr = jnp.where(pre < 0.0, jnp.maximum(-1.0, pre * dt / jnp.where(qr > 0, qr, one))
                      * nr / dt, 0.0)
    nsubg = jnp.where(eprdg < 0.0, jnp.maximum(-1.0, eprdg * dt / jnp.where(qg > 0, qg, one))
                      * ng / dt, 0.0)
    ni_t = ni_t + nsubi
    ns_t = ns_t + nsubs
    ng_t = ng_t + nsubg
    nr_t = nr_t + nsubr

    return (qv_t, t_t, qc_t, qr_t, qi_t, qni_t, qg_t,
            nc_t, ni_t, ns_t, nr_t, ng_t,
            ni, ns, nr, ng, nc)


# ===========================================================================
# SEDIMENTATION — copy of base _sedimentation with iinum=0 DUMFNC and aero
# LAMMINI.  (aero.F l.4300-4660)
# ===========================================================================
def _sedimentation_aero(qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
                        qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
                        nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
                        t, p, rho, dz, dt, do_cell):
    QSMALL = C.QSMALL
    eps = _EPS
    ncol, kx = t.shape

    dumi = qi + qi_ten * dt
    dumqs = qs + qni_ten * dt
    dumr = qr + qr_ten * dt
    dumfni = jnp.maximum(ni + ni_ten * dt, 0.0)
    dumfns = jnp.maximum(ns + ns_ten * dt, 0.0)
    dumfnr = jnp.maximum(nr + nr_ten * dt, 0.0)
    dumc = qc + qc_ten * dt
    # iinum=0: DUMFNC = NC3D + NC3DTEN*DT (aero.F l.4312/4316-4318 skipped)
    dumfnc = jnp.maximum(nc + nc_ten * dt, 0.0)
    dumg = qg + qg_ten * dt
    dumfng = jnp.maximum(ng + ng_ten * dt, 0.0)

    def dlam(q, n, cmass, inv_pow, lammin, lammax):
        act = q >= QSMALL
        qsf = jnp.where(act, q, 1.0)
        lam = (cmass * jnp.where(act, n, 0.0) / qsf) ** inv_pow
        lam = jnp.clip(jnp.where(act, lam, lammin), lammin, lammax)
        return jnp.where(act, lam, lammin)

    dlami = dlam(dumi, dumfni, C.CONS12, 1.0 / C.DI, LAMMINI, C.LAMMAXI)
    dlamr = dlam(dumr, dumfnr, C.PI * C.RHOW, 1.0 / 3.0, C.LAMMINR, C.LAMMAXR)
    dlams = dlam(dumqs, dumfns, C.CONS1, 1.0 / C.DS, C.LAMMINS, C.LAMMAXS)
    dlamg = dlam(dumg, dumfng, C.CONS2, 1.0 / C.DG, C.LAMMING, C.LAMMAXG)

    dum_rho = p / (287.15 * t)
    pgam = 0.0005714 * (nc / 1.0e6 * dum_rho) + 0.2714
    pgam = 1.0 / (pgam * pgam) - 1.0
    pgam = jnp.clip(pgam, 2.0, 10.0)
    g_p1 = gamma_fn(pgam + 1.0)
    g_p4 = gamma_fn(pgam + 4.0)
    actc = dumc >= QSMALL
    dumc_s = jnp.where(actc, dumc, 1.0)
    dlamc = (C.CONS26 * dumfnc * g_p4 / (dumc_s * g_p1)) ** (1.0 / 3.0)
    lcmin = (pgam + 1.0) / 60.0e-6
    lcmax = (pgam + 1.0) / 1.0e-6
    dlamc = jnp.clip(jnp.where(actc, dlamc, lcmin), lcmin, lcmax)

    dum_dc = (C.RHOSU / rho) ** 0.54
    mu = 1.496e-6 * t ** 1.5 / (t + 120.0)
    acn = C.G * C.RHOW / (18.0 * mu)
    ain = (C.RHOSU / rho) ** 0.35 * C.AI
    arn = dum_dc * C.AR
    asn = dum_dc * C.AS
    agn = dum_dc * C.AG

    g_bc_p1 = gamma_fn(1.0 + C.BC + pgam)
    g_bc_p4 = gamma_fn(4.0 + C.BC + pgam)
    unc = jnp.where(actc, acn * g_bc_p1 / (dlamc ** C.BC * g_p1 + eps), 0.0)
    umc = jnp.where(actc, acn * g_bc_p4 / (dlamc ** C.BC * g_p4 + eps), 0.0)
    uni = jnp.where(dumi >= QSMALL, ain * C.CONS27 / (dlami ** C.BI + eps), 0.0)
    umi = jnp.where(dumi >= QSMALL, ain * C.CONS28 / (dlami ** C.BI + eps), 0.0)
    unr = jnp.where(dumr >= QSMALL, arn * C.CONS6 / (dlamr ** C.BR + eps), 0.0)
    umr = jnp.where(dumr >= QSMALL, arn * C.CONS4 / (dlamr ** C.BR + eps), 0.0)
    ums = jnp.where(dumqs >= QSMALL, asn * C.CONS3 / (dlams ** C.BS + eps), 0.0)
    uns = jnp.where(dumqs >= QSMALL, asn * C.CONS5 / (dlams ** C.BS + eps), 0.0)
    umg = jnp.where(dumg >= QSMALL, agn * C.CONS7 / (dlamg ** C.BG + eps), 0.0)
    ung = jnp.where(dumg >= QSMALL, agn * C.CONS8 / (dlamg ** C.BG + eps), 0.0)

    ums = jnp.minimum(ums, 1.2 * dum_dc)
    uns = jnp.minimum(uns, 1.2 * dum_dc)
    umi = jnp.minimum(umi, 1.2 * (C.RHOSU / rho) ** 0.35)
    uni = jnp.minimum(uni, 1.2 * (C.RHOSU / rho) ** 0.35)
    umr = jnp.minimum(umr, 9.1 * dum_dc)
    unr = jnp.minimum(unr, 9.1 * dum_dc)
    umg = jnp.minimum(umg, 20.0 * dum_dc)
    ung = jnp.minimum(ung, 20.0 * dum_dc)

    fr = umr; fi = umi; fni = uni; fs = ums; fns = uns
    fnr = unr; fc = umc; fnc = unc; fg = umg; fng = ung

    def fill_down(fld):
        def body(kk, arr):
            k = kx - 2 - kk
            above = arr[:, k + 1]
            cur = arr[:, k]
            new = jnp.where(cur < 1.0e-10, above, cur)
            return arr.at[:, k].set(new)
        return jax.lax.fori_loop(0, kx - 1, body, fld)

    fr = fill_down(fr); fi = fill_down(fi); fni = fill_down(fni)
    fs = fill_down(fs); fns = fill_down(fns); fnr = fill_down(fnr)
    fc = fill_down(fc); fnc = fill_down(fnc); fg = fill_down(fg); fng = fill_down(fng)

    rgvm = fr
    for _fld in (fi, fs, fc, fni, fnr, fns, fnc, fg, fng):
        rgvm = jnp.maximum(rgvm, _fld)
    nstep_k = jnp.floor(rgvm * dt / dz + 1.0).astype(jnp.int32)
    nstep = jnp.maximum(jnp.max(nstep_k, axis=1), 1)
    nstep = jnp.minimum(nstep, _NSTEP_MAX)

    dumr = dumr * rho; dumi = dumi * rho; dumfni = dumfni * rho
    dumqs = dumqs * rho; dumfns = dumfns * rho; dumfnr = dumfnr * rho
    dumc = dumc * rho; dumfnc = dumfnc * rho; dumg = dumg * rho; dumfng = dumfng * rho

    qrsten = jnp.zeros_like(t); qisten = jnp.zeros_like(t); qnisten = jnp.zeros_like(t)
    qcsten = jnp.zeros_like(t); qgsten = jnp.zeros_like(t)
    ni_sed = jnp.zeros_like(t); ns_sed = jnp.zeros_like(t); nr_sed = jnp.zeros_like(t)
    nc_sed = jnp.zeros_like(t); ng_sed = jnp.zeros_like(t)
    precrt = jnp.zeros((ncol,), t.dtype)
    snowrt = jnp.zeros((ncol,), t.dtype)
    snowprt = jnp.zeros((ncol,), t.dtype)
    grplprt = jnp.zeros((ncol,), t.dtype)

    nstep_f = nstep.astype(t.dtype)[:, None]
    nstep_col = nstep_f[:, 0]

    def fall_one(fld, F):
        falout = F * fld
        ktop = kx - 1
        faltnd_top = falout[:, ktop] / dz[:, ktop]
        ten = jnp.zeros_like(fld)
        ten = ten.at[:, ktop].add(-faltnd_top / nstep_col / rho[:, ktop])
        fld = fld.at[:, ktop].add(-faltnd_top * dt / nstep_col)
        fo_kp1 = falout[:, 1:]
        fo_k = falout[:, :-1]
        faltnd = (fo_kp1 - fo_k) / dz[:, :-1]
        ten = ten.at[:, :-1].add(faltnd / nstep_f / rho[:, :-1])
        fld = fld.at[:, :-1].add(faltnd * dt / nstep_f)
        surf = falout[:, 0]
        return fld, ten, surf, falout

    def step_body(istep, carry):
        (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
         qrsten, qisten, qnisten, qcsten, qgsten,
         ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
         precrt, snowrt, snowprt, grplprt) = carry
        active = (istep < nstep)[:, None].astype(t.dtype)

        dumr2, tr, sr_r, _ = fall_one(dumr, fr)
        dumi2, ti, sr_i, _ = fall_one(dumi, fi)
        dumfni2, tni, _, _ = fall_one(dumfni, fni)
        dumqs2, ts, sr_s, _ = fall_one(dumqs, fs)
        dumfns2, tns, _, _ = fall_one(dumfns, fns)
        dumfnr2, tnr, _, _ = fall_one(dumfnr, fnr)
        dumc2, tc, sr_c, _ = fall_one(dumc, fc)
        dumfnc2, tnc, _, _ = fall_one(dumfnc, fnc)
        dumg2, tg, sr_g, _ = fall_one(dumg, fg)
        dumfng2, tng, _, _ = fall_one(dumfng, fng)

        a = active
        am = a
        dumr = jnp.where(a > 0, dumr2, dumr)
        dumi = jnp.where(a > 0, dumi2, dumi)
        dumfni = jnp.where(a > 0, dumfni2, dumfni)
        dumqs = jnp.where(a > 0, dumqs2, dumqs)
        dumfns = jnp.where(a > 0, dumfns2, dumfns)
        dumfnr = jnp.where(a > 0, dumfnr2, dumfnr)
        dumc = jnp.where(a > 0, dumc2, dumc)
        dumfnc = jnp.where(a > 0, dumfnc2, dumfnc)
        dumg = jnp.where(a > 0, dumg2, dumg)
        dumfng = jnp.where(a > 0, dumfng2, dumfng)

        qrsten = qrsten + tr * am
        qisten = qisten + ti * am
        qnisten = qnisten + ts * am
        qcsten = qcsten + tc * am
        qgsten = qgsten + tg * am
        ni_sed = ni_sed + tni * am
        ns_sed = ns_sed + tns * am
        nr_sed = nr_sed + tnr * am
        nc_sed = nc_sed + tnc * am
        ng_sed = ng_sed + tng * am

        amc = active[:, 0]
        precrt = precrt + (sr_r + sr_c + sr_s + sr_i + sr_g) * dt / nstep_f[:, 0] * amc
        snowrt = snowrt + (sr_s + sr_i + sr_g) * dt / nstep_f[:, 0] * amc
        snowprt = snowprt + (sr_i + sr_s) * dt / nstep_f[:, 0] * amc
        grplprt = grplprt + (sr_g) * dt / nstep_f[:, 0] * amc

        return (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
                qrsten, qisten, qnisten, qcsten, qgsten,
                ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
                precrt, snowrt, snowprt, grplprt)

    carry = (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
             qrsten, qisten, qnisten, qcsten, qgsten,
             ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
             precrt, snowrt, snowprt, grplprt)
    carry = jax.lax.fori_loop(0, _NSTEP_MAX, step_body, carry)
    (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
     qrsten, qisten, qnisten, qcsten, qgsten,
     ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
     precrt, snowrt, snowprt, grplprt) = carry

    return (qrsten, qisten, qnisten, qcsten, qgsten,
            ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
            precrt, snowrt, snowprt, grplprt)


# ===========================================================================
# FINALIZE — copy of base _finalize with the aero deltas:
#   * 2*DCS (350e-6) ice->snow threshold, aero LAMMINI in the ice slope
#   * cloud-free EFFI = 4.99 um, EFFC = 2.49 um (RRTMG consistency)
#   * NO constant-droplet reset (iinum=0) and NO (NANEW1+NANEW2)/RHO bound
#     (that bound is IACT=2-only; aero runs IACT=4).
# ===========================================================================
def _finalize_aero(t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
                   qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
                   t_ten, qv_ten,
                   nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
                   xxlv, xxls, xlf, cpm, p, rho, dt, do_cell):
    QSMALL = C.QSMALL
    EP_2 = C.EP_2
    one = jnp.ones_like(t)

    lami_pre = jnp.where(qi >= QSMALL,
                         (C.CONS12 * jnp.maximum(ni, 0.0)
                          / jnp.where(qi >= QSMALL, qi, one)) ** (1.0 / C.DI), 0.0)
    lami_pre = jnp.clip(jnp.where(qi >= QSMALL, lami_pre, 0.0), 0.0, None)
    big_ice = (qi >= QSMALL) & (t < 273.15) & (lami_pre >= 1.0e-10) \
        & (1.0 / jnp.where(lami_pre > 0, lami_pre, one) >= 2.0 * DCS)
    qni_ten = jnp.where(big_ice, qni_ten + qi / dt + qi_ten, qni_ten)
    ns_ten = jnp.where(big_ice, ns_ten + ni / dt + ni_ten, ns_ten)
    qi_ten = jnp.where(big_ice, -qi / dt, qi_ten)
    ni_ten = jnp.where(big_ice, -ni / dt, ni_ten)

    qc = qc + qc_ten * dt
    qi = qi + qi_ten * dt
    qs = qs + qni_ten * dt
    qr = qr + qr_ten * dt
    nc = nc + nc_ten * dt
    ni = ni + ni_ten * dt
    ns = ns + ns_ten * dt
    nr = nr + nr_ten * dt
    qg = qg + qg_ten * dt
    ng = ng + ng_ten * dt
    t = t + t_ten * dt
    qv = qv + qv_ten * dt

    evs = jnp.minimum(0.99 * p, polysvp(t, 0))
    eis = jnp.minimum(0.99 * p, polysvp(t, 1))
    eis = jnp.where(eis > evs, evs, eis)
    qvs = EP_2 * evs / (p - evs)
    qvi = EP_2 * eis / (p - eis)
    qvqvs = qv / qvs
    qvqvsi = qv / qvi

    def _rm_liq(qv_, t_, qx, thr, cpm_):
        rm = (qvqvs < 0.9) & (qx < thr)
        return (jnp.where(rm, qv_ + qx, qv_),
                jnp.where(rm, t_ - qx * xxlv / cpm_, t_),
                jnp.where(rm, 0.0, qx))

    def _rm_ice(qv_, t_, qx, thr, cpm_):
        rm = (qvqvsi < 0.9) & (qx < thr)
        return (jnp.where(rm, qv_ + qx, qv_),
                jnp.where(rm, t_ - qx * xxls / cpm_, t_),
                jnp.where(rm, 0.0, qx))

    qv, t, qr = _rm_liq(qv, t, qr, 1.0e-8, cpm)
    qv, t, qc = _rm_liq(qv, t, qc, 1.0e-8, cpm)
    qv, t, qi = _rm_ice(qv, t, qi, 1.0e-8, cpm)
    qv, t, qs = _rm_ice(qv, t, qs, 1.0e-8, cpm)
    qv, t, qg = _rm_ice(qv, t, qg, 1.0e-8, cpm)

    def _zero(qx, nx):
        m = qx < QSMALL
        return jnp.where(m, 0.0, qx), jnp.where(m, 0.0, nx)

    qc, nc = _zero(qc, nc)
    qr, nr = _zero(qr, nr)
    qi, ni = _zero(qi, ni)
    qs, ns = _zero(qs, ns)
    qg, ng = _zero(qg, ng)

    empty = ((qc < QSMALL) & (qi < QSMALL) & (qs < QSMALL)
             & (qr < QSMALL) & (qg < QSMALL))
    nonempty = ~empty

    melt_i = (qi >= QSMALL) & (t >= 273.15) & nonempty
    qr = jnp.where(melt_i, qr + qi, qr)
    t = jnp.where(melt_i, t - qi * xlf / cpm, t)
    nr = jnp.where(melt_i, nr + ni, nr)
    qi = jnp.where(melt_i, 0.0, qi)
    ni = jnp.where(melt_i, 0.0, ni)

    hf_c = (t <= 233.15) & (qc >= QSMALL) & nonempty
    qi = jnp.where(hf_c, qi + qc, qi)
    t = jnp.where(hf_c, t + qc * xlf / cpm, t)
    qc = jnp.where(hf_c, 0.0, qc)
    ni = jnp.where(hf_c, ni + nc, ni)
    nc = jnp.where(hf_c, 0.0, nc)

    hf_r = (t <= 233.15) & (qr >= QSMALL) & nonempty
    qg = jnp.where(hf_r, qg + qr, qg)
    t = jnp.where(hf_r, t + qr * xlf / cpm, t)
    qr = jnp.where(hf_r, 0.0, qr)
    ng = jnp.where(hf_r, ng + nr, ng)
    nr = jnp.where(hf_r, 0.0, nr)

    ni = jnp.maximum(ni, 0.0); ns = jnp.maximum(ns, 0.0)
    nc = jnp.maximum(nc, 0.0); nr = jnp.maximum(nr, 0.0); ng = jnp.maximum(ng, 0.0)

    lami, ni = _slope_final(qi, ni, C.CONS12, 1.0 / C.DI, LAMMINI, C.LAMMAXI)
    lamr, nr = _slope_final(qr, nr, C.PI * C.RHOW, 1.0 / 3.0, C.LAMMINR, C.LAMMAXR)
    lams, ns = _slope_final(qs, ns, C.CONS1, 1.0 / C.DS, C.LAMMINS, C.LAMMAXS)
    lamg, ng = _slope_final(qg, ng, C.CONS2, 1.0 / C.DG, C.LAMMING, C.LAMMAXG)

    actc = qc >= QSMALL
    dum_rho = p / (287.15 * t)
    pgam = 0.0005714 * (nc / 1.0e6 * dum_rho) + 0.2714
    pgam = 1.0 / (pgam * pgam) - 1.0
    pgam = jnp.clip(pgam, 2.0, 10.0)
    g_p1 = gamma_fn(pgam + 1.0)
    g_p3 = gamma_fn(pgam + 3.0)
    g_p4 = gamma_fn(pgam + 4.0)
    qcs = jnp.where(actc, qc, 1.0)
    lamc = (C.CONS26 * nc * g_p4 / (qcs * g_p1)) ** (1.0 / 3.0)
    lcmin = (pgam + 1.0) / 60.0e-6
    lcmax = (pgam + 1.0) / 1.0e-6
    too_s = lamc < lcmin
    too_b = lamc > lcmax
    lamc = jnp.where(too_s, lcmin, jnp.where(too_b, lcmax, lamc))
    nc_adj = jnp.exp(3.0 * jnp.log(lamc) + jnp.log(qcs) + jnp.log(g_p1)
                     - jnp.log(g_p4)) / C.CONS26
    nc = jnp.where(actc & (too_s | too_b), nc_adj, nc)
    lamc = jnp.where(actc, lamc, 0.0)

    def reff(active, lam, default):
        return jnp.where(active, 3.0 / jnp.where(lam > 0, lam, one) / 2.0 * 1.0e6,
                         default)

    effi = reff(qi >= QSMALL, lami, 4.99)     # aero.F l.4953-4954 (base: 25.)
    effs = reff(qs >= QSMALL, lams, 25.0)
    effr = reff(qr >= QSMALL, lamr, 25.0)
    effg = reff(qg >= QSMALL, lamg, 25.0)
    effc = jnp.where(qc >= QSMALL,
                     g_p4 / g_p3 / jnp.where(lamc > 0, lamc, one) / 2.0 * 1.0e6,
                     2.49)                    # aero.F l.4973-4974 (base: 25.)

    # ice number upper bound; NC stays prognostic (iinum=0, IACT=4: no
    # constant reset and no (NANEW1+NANEW2)/RHO bound — aero.F l.4985-5001)
    ni = jnp.minimum(ni, 0.3e6 / rho)

    return (t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
            effc, effi, effs, effr, effg)
