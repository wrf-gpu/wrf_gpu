"""WRF Shin-Hong scale-aware PBL column kernel (``bl_pbl_physics=11``).

This is a faithful host-NumPy fp64 port of the scalar-column path of WRF
``module_bl_shinhong.F`` (``shinhong`` -> ``shinhong2d``), including the
scale-aware partition functions (``pu``/``pq``/``pthnl``/``pthl``/``ptke``),
the prescribed nonlocal heat transport profile, and the optional TKE
diagnostic path (``mixlen``/``prodq2``/``vdifq``).

Shin-Hong (Shin and Hong 2015) is *not* a thin YSU delta: it reuses the YSU
nonlocal-K + countergradient skeleton for PBL height and free-atmosphere
diffusivity, but in the convective gray zone it:

* scales the nonlocal momentum/moisture countergradient transport by
  ``pu``/``pq`` (functions of ``dx/cslen``),
* replaces the YSU prescribed-flux nonlocal heat term with a Shin-Hong
  mass-flux profile (``amf1``/``amf2``/``amf3``) scaled by ``pthnl``,
* scales the *local* implicit diffusion (the tridiagonal off-diagonals) by
  ``pu``/``pq``/``pthl`` evaluated at a level-dependent effective ``dx``,
* and diagnoses SGS TKE/mixing-length from the post-mix profile with a
  ``ptke`` scale factor.

The port matches the unmodified pristine WRF module
``module_bl_shinhong.F`` (sha256
``99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56``) against
the single-column oracle savepoints under ``proofs/v090/savepoints``.

The driver/oracle uses ``ndiff=3`` (qv + qc + qi) with ``qc==qi==0`` and
``shinhong_tke_diag=1``; this kernel ports that configuration. The in-cloud
Richardson modification (active only with non-zero cloud water/ice) is ported
faithfully but is inert on the oracle cases.

Host-NumPy fp64 is used for the savepoint parity proof. A device/scan
adapter is intentionally NOT wired here; operational integration (vmap/scan,
fp32 downcast) is a separate manager-owned task gated on this proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants -- mirror the oracle driver exactly.
# ---------------------------------------------------------------------------
G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
XLV = 2.5e6
ROVCP = R_D / CP
EP1 = R_V / R_D - 1.0
EP2 = R_D / R_V
KARMAN = 0.4

# shinhong2d parameters (lines 321-346 of module_bl_shinhong.F).
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
H1 = 0.33333333
H2 = 0.6666667
CKZ = 0.001
ZFMIN = 1.0e-8
APHI5 = 5.0
APHI16 = 16.0
TMIN = 1.0e-2
GAMCRT = 3.0
GAMCRQ = 2.0e-3
XKA = 2.4e-5
IMVDIF = 1

# tunable parameters for TKE.
EPSQ2L = 0.01
C_1 = 1.0
GAMCRE = 0.224

# tunable parameters for prescribed nonlocal transport profile.
MLTOP = 1.0
SFCFRACN1 = 0.075
NLFRAC = 0.7
ENLFRAC = -0.4
A11 = 1.0
A12 = -1.15
EZFAC = 1.5
CPENT = -0.4
RIGSMAX = 100.0
ENTFMIN = 1.0
ENTFMAX = 5.0


@dataclass(frozen=True)
class ShinHongResult:
    """Per-column Shin-Hong tendencies and diagnostics (1:1 with WRF outputs)."""

    u_tend: np.ndarray
    v_tend: np.ndarray
    theta_tend: np.ndarray  # RTHBLTEN already divided by Exner (K/s)
    qv_tend: np.ndarray
    exch_h: np.ndarray
    pblh: float
    kpbl: int
    wstar: float
    delta: float
    tke: np.ndarray
    el_pbl: np.ndarray


# ---------------------------------------------------------------------------
# Scale-aware partition functions (module_bl_shinhong.F lines 2297-2426).
# ---------------------------------------------------------------------------
def pu(d: float, h: float) -> float:
    pmin, pmax = 0.0, 1.0
    a1, a2, a3, a4, a5 = 1.0, 0.070, 1.0, 0.142, 0.071
    b1, b2 = 2.0, 0.6666667
    if h != 0.0:
        doh = d / h
        num = a1 * doh**b1 + a2 * doh**b2
        den = a3 * doh**b1 + a4 * doh**b2 + a5
        val = num / den
    else:
        val = 1.0
    return min(max(val, pmin), pmax)


def pq(d: float, h: float) -> float:
    pmin, pmax = 0.0, 1.0
    a1, a2, a3, a4, a5 = 1.0, -0.098, 1.0, 0.106, 0.5
    b1 = 2.0
    if h != 0.0:
        doh = d / h
        num = a1 * doh**b1 + a2
        den = a3 * doh**b1 + a4
        val = a5 * num / den + (1.0 - a5)
    else:
        val = 1.0
    return min(max(val, pmin), pmax)


def pthnl(d: float, h: float) -> float:
    pmin, pmax = 0.0, 1.0
    a1, a2, a3 = 1.000, 0.936, -1.110
    a4, a5, a6, a7 = 1.000, 0.312, 0.329, 0.243
    b1, b2 = 2.0, 0.875
    if h != 0.0:
        doh = d / h
        num = a1 * doh**b1 + a2 * doh**b2 + a3
        den = a4 * doh**b1 + a5 * doh**b2 + a6
        val = a7 * num / den + (1.0 - a7)
    else:
        val = 1.0
    return min(max(val, pmin), pmax)


def pthl(d: float, h: float) -> float:
    pmin, pmax = 0.0, 1.0
    a1, a2, a3 = 1.000, 0.870, -0.913
    a4, a5, a6, a7 = 1.000, 0.153, 0.278, 0.280
    b1, b2 = 2.0, 0.5
    if h != 0.0:
        doh = d / h
        num = a1 * doh**b1 + a2 * doh**b2 + a3
        den = a4 * doh**b1 + a5 * doh**b2 + a6
        val = a7 * num / den + (1.0 - a7)
    else:
        val = 1.0
    return min(max(val, pmin), pmax)


def ptke(d: float, h: float) -> float:
    pmin, pmax = 0.0, 1.0
    a1, a2, a3, a4, a5 = 1.000, 0.070, 1.000, 0.142, 0.071
    b1, b2 = 2.0, 0.6666667
    if h != 0.0:
        doh = d / h
        num = a1 * doh**b1 + a2 * doh**b2
        den = a3 * doh**b1 + a4 * doh**b2 + a5
        val = num / den
    else:
        val = 1.0
    return min(max(val, pmin), pmax)


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas solve matching WRF ``tridin_ysu``/``tridi1n`` indexing."""

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


def _first_pbl_guess(thv, thermal, za, br, brcr, u, v):
    brup = br
    brdn = br
    stable = False
    kpbl = 1
    for k in range(1, thv.shape[0]):
        if not stable:
            brdn = brup
            spdk2 = max(u[k] * u[k] + v[k] * v[k], 1.0)
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
            kpbl = k + 1  # WRF 1-based kpbl
            stable = brup > brcr
    return kpbl, brdn, brup


def _interp_pblh(kpbl, brdn, brup, brcr, za, zq):
    if brdn >= brcr:
        brint = 0.0
    elif brup <= brcr:
        brint = 1.0
    else:
        brint = (brcr - brdn) / (brup - brdn)
    k0 = kpbl  # WRF 1-based: hpbl = za(k-1)+brint*(za(k)-za(k-1)) -> python za[k0-2],za[k0-1]
    hpbl = za[k0 - 2] + brint * (za[k0 - 1] - za[k0 - 2])
    if hpbl < zq[1]:
        kpbl = 1
    pblflg = kpbl > 1
    return hpbl, kpbl, pblflg


def _mixlen(u, v, t, the, q, cwm, q2, z, ustar, corf, epshol, hpbl, lpbl, pblflg,
            mf, ufxpbl, vfxpbl, qfxpbl, *, p608=EP1, vkarman=KARMAN, cp=CP):
    """Port of subroutine ``mixlen`` (module_bl_shinhong.F lines 1776-2012).

    All level arrays are 0-based length-nz here. WRF 1-based index ``k`` maps to
    Python ``k-1``. ``lmh=1`` (WRF) -> base level (Python 0). ``lpbl`` is the
    WRF 1-based kpbl. Returns ``(s2, gh, ri, el)`` length-nz, valid at python
    k>=1 (WRF kts+1..kte). The caller passes ``ufxpbl``/``vfxpbl``/``qfxpbl``
    with the ``hgam{u,v,q}/hpbl`` countergradient part already folded in (the
    WRF mixlen expression ``suk-hgamu/hpbl-ufxpbl(k)`` splits the two, and
    ufxpbl(k) there is the entrainment part only).
    """

    nz = u.shape[0]
    lmh = 1  # WRF
    blckdr, cn = 0.0063, 0.75
    eps1, epsl, epsru = 1.0e-12, 0.32, 1.0e-7
    el0max, el0min, elfc = 1000.0, 1.0, 0.23 * 0.5
    alph, beta, g = 0.30, 1.0 / 273.0, 9.81
    btg = beta * g
    a1, a2x = 0.659888514560862645, 0.6574209922667784586
    b1, b2 = 11.87799326209552761, 7.226971804046074028
    c1 = 0.000830955950095854396
    adnh = 9.0 * a1 * a2x * a2x * (12.0 * a1 + 3.0 * b2) * btg * btg
    adnm = 18.0 * a1 * a1 * a2x * (b2 - 3.0 * a2x) * btg
    bdnh = 3.0 * a2x * (7.0 * a1 + b2) * btg
    bdnm = 6.0 * a1 * a1
    aeqh = 9.0 * a1 * a2x * a2x * b1 * btg * btg + 9.0 * a1 * a2x * a2x * (12.0 * a1 + 3.0 * b2) * btg * btg
    aeqm = (3.0 * a1 * a2x * b1 * (3.0 * a2x + 3.0 * b2 * c1 + 18.0 * a1 * c1 - b2) * btg
            + 18.0 * a1 * a1 * a2x * (b2 - 3.0 * a2x) * btg)
    requ = -aeqh / aeqm
    epsgh = 1.0e-9
    epsgm = requ * epsgh
    epsrs = 1.0e-7
    ubryl = ((18.0 * requ * a1 * a1 * a2x * b2 * c1 * btg + 9.0 * a1 * a2x * a2x * b2 * btg * btg)
             / (requ * adnm + adnh))
    ubry = (1.0 + epsrs) * ubryl
    ubry3 = 3.0 * ubry
    aubh = 27.0 * a1 * a2x * a2x * b2 * btg * btg - adnh * ubry3
    aubm = 54.0 * a1 * a1 * a2x * b2 * c1 * btg - adnm * ubry3
    bubh = (9.0 * a1 * a2x + 3.0 * a2x * b2) * btg - bdnh * ubry3
    bubm = 18.0 * a1 * a1 * c1 - bdnm * ubry3
    cubr = 1.0 - ubry3
    rcubr = 1.0 / cubr
    elcbl = 0.77

    elocp = 2.72e6 / cp
    ct = 0.0

    s2 = np.zeros(nz, dtype=np.float64)
    gh = np.zeros(nz, dtype=np.float64)
    ri = np.zeros(nz, dtype=np.float64)
    el = np.zeros(nz, dtype=np.float64)
    elm = np.zeros(nz, dtype=np.float64)
    rel = np.zeros(nz, dtype=np.float64)
    en2 = np.zeros(nz, dtype=np.float64)
    q1 = np.zeros(nz, dtype=np.float64)

    dth = np.zeros(nz, dtype=np.float64)
    for k in range(1, nz):
        dth[k] = the[k] - the[k - 1]
    # ct adjustment (WRF kts+2..kte). ct=0 so dth unchanged on first hit.
    for k in range(2, nz):
        if dth[k] > 0.0 and dth[k - 1] <= 0.0:
            dth[k] = dth[k] + ct
            break

    # local gradient Richardson number, WRF k=kte..kts+1 -> python nz-1..1.
    for k in range(nz - 1, 0, -1):
        rdz = 2.0 / (z[k + 1] - z[k - 1])
        s2l = ((u[k] - u[k - 1]) ** 2 + (v[k] - v[k - 1]) ** 2) * rdz * rdz
        if pblflg and (k + 1) <= lpbl:
            suk = (u[k] - u[k - 1]) * rdz
            svk = (v[k] - v[k - 1]) * rdz
            s2l = (suk - ufxpbl[k]) * suk + (svk - vfxpbl[k]) * svk
        s2l = max(s2l, epsgm)
        s2[k] = s2l

        tem = (t[k] + t[k - 1]) * 0.5
        thm = (the[k] + the[k - 1]) * 0.5
        a = thm * p608
        b = (elocp / tem - 1.0 - p608) * thm
        ghl = (dth[k] * ((q[k] + q[k - 1] + cwm[k] + cwm[k - 1]) * (0.5 * p608) + 1.0)
               + (q[k] - q[k - 1] + cwm[k] - cwm[k - 1]) * a
               + (cwm[k] - cwm[k - 1]) * b) * rdz
        if pblflg and (k + 1) <= lpbl:
            ghl = ghl - mf[k] - qfxpbl[k] * a
        if abs(ghl) <= epsgh:
            ghl = epsgh
        en2[k] = ghl * g / thm
        gh[k] = ghl
        ri[k] = en2[k] / s2l

    # maximum mixing lengths.
    for k in range(nz - 1, 0, -1):
        s2l = s2[k]
        ghl = gh[k]
        if ghl >= epsgh:
            if s2l / ghl <= requ:
                elm[k] = epsl
            else:
                aubr = (aubm * s2l + aubh * ghl) * ghl
                bubr = bubm * s2l + bubh * ghl
                qol2st = (-0.5 * bubr + np.sqrt(bubr * bubr * 0.25 - aubr * cubr)) * rcubr
                eloq2x = 1.0 / qol2st
                elm[k] = max(np.sqrt(eloq2x * q2[k]), epsl)
        else:
            aden = (adnm * s2l + adnh * ghl) * ghl
            bden = bdnm * s2l + bdnh * ghl
            qol2un = -0.5 * bden + np.sqrt(bden * bden * 0.25 - aden)
            eloq2x = 1.0 / (qol2un + epsru)
            elm[k] = max(np.sqrt(eloq2x * q2[k]), epsl)

    # q1 for blackadar (WRF lpbl..lmh -> python lpbl-1..0).
    for k in range(lpbl - 1, lmh - 2, -1):
        if k < 0:
            break
        q1[k] = np.sqrt(q2[k])

    szq = 0.0
    sq = 0.0
    for k in range(nz - 1, 0, -1):
        qdzl = (q1[k] + q1[k - 1]) * (z[k] - z[k - 1])
        szq = (z[k] + z[k - 1] - z[lmh - 1] - z[lmh - 1]) * qdzl + szq
        sq = qdzl + sq

    el0 = min(alph * szq * 0.5 / sq, el0max)
    el0 = max(el0, el0min)

    # above the PBL top. lpblm = min(lpbl+1,kte) WRF (1-based level).
    lpblm = min(lpbl + 1, nz)
    for k in range(nz - 1, lpblm - 1 - 1, -1):  # WRF kte..lpblm -> python nz-1..lpblm-1
        if k <= 0:
            break
        el[k] = (z[k + 1] - z[k - 1]) * elfc
        rel[k] = el[k] / elm[k]

    # inside the PBL.
    epshol = min(epshol, 0.0)
    ckp = elcbl * ((1.0 - 8.0 * epshol) ** (1.0 / 3.0))
    if lpbl > lmh:
        for k in range(lpbl - 1, lmh - 1, -1):  # WRF lpbl..lmh+1 -> python lpbl-1..lmh(=1)
            vkrmz = (z[k] - z[lmh - 1]) * vkarman
            if pblflg:
                vkrmz = ckp * (z[k] - z[lmh - 1]) * vkarman
            el[k] = vkrmz / (vkrmz / el0 + 1.0)
            rel[k] = el[k] / elm[k]

    # smoothing WRF k=lpbl-1..lmh+2 -> python lpbl-2..lmh+1.
    for k in range(lpbl - 2, lmh, -1):
        if k <= 0:
            break
        srel = min(((rel[k - 1] + rel[k + 1]) * 0.5 + rel[k]) * 0.5, rel[k])
        el[k] = max(srel * elm[k], epsl)

    # QNSE mixing length in stable case.
    f = max(corf, eps1)
    rlambda = f / (blckdr * ustar)
    for k in range(nz - 1, 0, -1):
        if en2[k] >= 0.0:
            vkrmz = (z[k] - z[lmh - 1]) * vkarman
            rlb = rlambda + 1.0 / vkrmz
            rln = np.sqrt(2.0 * en2[k] / q2[k]) / cn
            el[k] = 1.0 / (rlb + rln)

    return s2, gh, ri, el


def _prodq2(dtturbl, ustar, s2, ri, q2, el, z, akm, akh, uxk, vxk, thxk, thvxk,
            hgamu, hgamv, hgamq, delxy, hpbl, pblflg, kpbl, mf, ufxpbl, vfxpbl, qfxpbl,
            *, p608=EP1):
    """Port of subroutine ``prodq2`` (lines 2016-2125). Updates q2 in place."""

    nz = uxk.shape[0]
    epsq2l, c0, ceps, g = 0.01, 0.55, 16.6, 9.81
    rc02 = 2.0 / (c0 * c0)

    for k in range(1, nz):  # WRF kts+1..kte -> python 1..nz-1
        deltaz = 0.5 * (z[k + 1] - z[k - 1])
        s2l = s2[k]
        q2l = q2[k]
        suk = (uxk[k] - uxk[k - 1]) / deltaz
        svk = (vxk[k] - vxk[k - 1]) / deltaz
        gthvk = (thvxk[k] - thvxk[k - 1]) / deltaz
        govrthvk = g / (0.5 * (thvxk[k] + thvxk[k - 1]))
        akml = akm[k]
        akhl = akh[k]
        thm = (thxk[k] + thxk[k - 1]) * 0.5

        if pblflg and (k + 1) <= kpbl:
            pru = (akml * (suk - hgamu / hpbl - ufxpbl[k])) * suk
            prv = (akml * (svk - hgamv / hpbl - vfxpbl[k])) * svk
        else:
            pru = akml * suk * suk
            prv = akml * svk * svk
        pr = pru + prv

        if pblflg and (k + 1) <= kpbl:
            bpr = (akhl * (gthvk - mf[k] - (hgamq / hpbl + qfxpbl[k]) * p608 * thm)) * govrthvk
        else:
            bpr = akhl * gthvk * govrthvk

        disel = min(delxy, ceps * el[k])
        dis = q2l ** 1.5 / disel

        q2l = q2l + 2.0 * (pr - bpr - dis) * dtturbl
        q2[k] = max(q2l, epsq2l)

    q2[0] = max(rc02 * ustar * ustar, epsq2l)
    return q2


def _vdifq(dtdif, q2, el, z, akhk, ptke1, hgame, hpbl, pblflg, kpbl, efxpbl):
    """Port of subroutine ``vdifq`` (lines 2129-2231). Updates q2 in place.

    WRF 1-based: lmh=1. Arrays length-nz (0-based). z length nz+1.
    """

    nz = q2.shape[0]
    lmh = 1  # WRF
    c_k, esq = 1.0, 5.0

    zfacentk = np.zeros(nz, dtype=np.float64)  # valid k>=1 (WRF kts+1..kte)
    akq = np.zeros(nz, dtype=np.float64)       # valid k>=2 (WRF kts+2..kte)
    cm = np.zeros(nz, dtype=np.float64)
    cr = np.zeros(nz, dtype=np.float64)
    dtoz = np.zeros(nz, dtype=np.float64)
    rsq2 = np.zeros(nz, dtype=np.float64)

    for k in range(1, nz):  # WRF kts+1..kte
        zak = 0.5 * (z[k] + z[k - 1])
        zfacentk[k] = (zak / hpbl) ** 3.0

    for k in range(nz - 1, 1, -1):  # WRF kte..kts+2 -> python nz-1..2
        dtoz[k] = (dtdif + dtdif) / (z[k + 1] - z[k - 1])
        akq[k] = c_k * (akhk[k] / (z[k + 1] - z[k - 1]) + akhk[k - 1] / (z[k] - z[k - 2]))
        akq[k] = akq[k] * ptke1[k]
        cr[k] = -dtoz[k] * akq[k]

    akqs = c_k * akhk[1] / (z[2] - z[0])  # WRF akhk(kts+1), z(kts+2)-z(kts)
    akqs = akqs * ptke1[1]
    cm[nz - 1] = dtoz[nz - 1] * akq[nz - 1] + 1.0
    rsq2[nz - 1] = q2[nz - 1]

    for k in range(nz - 2, 1, -1):  # WRF kte-1..kts+2 -> python nz-2..2
        cf = -dtoz[k] * akq[k + 1] / cm[k + 1]
        cm[k] = -cr[k + 1] * cf + (akq[k + 1] + akq[k]) * dtoz[k] + 1.0
        rsq2[k] = -rsq2[k + 1] * cf + q2[k]
        if pblflg and (k + 1) < kpbl:  # WRF k<kpbl
            rsq2[k] = (rsq2[k]
                       - dtoz[k] * (2.0 * hgame[k] / hpbl) * akq[k + 1] * (z[k + 1] - z[k])
                       + dtoz[k] * (2.0 * hgame[k - 1] / hpbl) * akq[k] * (z[k] - z[k - 1]))
            rsq2[k] = (rsq2[k]
                       - dtoz[k] * 2.0 * efxpbl * zfacentk[k + 1]
                       + dtoz[k] * 2.0 * efxpbl * zfacentk[k])

    # boundary at WRF lmh+1=2 -> python index 1.
    dtozs = (dtdif + dtdif) / (z[2] - z[0])  # z(kts+2)-z(kts) WRF
    cf = -dtozs * akq[2] / cm[2]  # akq(lmh+2)/cm(lmh+2), lmh+2 WRF=3 -> python 2

    if pblflg and ((lmh + 1) < kpbl):  # WRF (lmh+1)<kpbl ; lmh+1 WRF=2 -> python1
        val = (dtozs * akqs * q2[0] - rsq2[2] * cf + q2[1]
               - dtozs * (2.0 * hgame[1] / hpbl) * akq[2] * (z[2] - z[1])
               + dtozs * (2.0 * hgame[0] / hpbl) * akqs * (z[1] - z[0]))
        val = val - dtozs * 2.0 * efxpbl * zfacentk[2] + dtozs * 2.0 * efxpbl * zfacentk[1]
        val = val / ((akq[2] + akqs) * dtozs - cr[2] * cf + 1.0)
        q2[1] = val
    else:
        q2[1] = ((dtozs * akqs * q2[0] - rsq2[2] * cf + q2[1])
                 / ((akq[2] + akqs) * dtozs - cr[2] * cf + 1.0))

    for k in range(2, nz):  # WRF lmh+2..kte -> python 2..nz-1
        q2[k] = (-cr[k] * q2[k - 1] + rsq2[k]) / cm[k]

    return q2


def shinhong_column(
    u, v, t, qv, p, pdi, exner, dz, *,
    tke_in,
    psfc, znt, ust, hfx, qfx, wspd, br, psim, psih,
    dt, xland, u10, v10, dx, dy, corf=1.0e-4,
    ctopo=1.0, ctopo2=1.0, shinhong_tke_diag=1,
    qc=None, qi=None,
) -> ShinHongResult:
    """Faithful single-column Shin-Hong PBL step.

    Inputs are length-nz mass-level arrays (``pdi`` is length nz+1 interfaces),
    matching the oracle driver. ``t`` is temperature; ``exner`` is the Exner
    function on mass levels. ``tke_in`` is the input TKE (q2/2). Returns a
    :class:`ShinHongResult`. ``ndiff=3`` with ``qc``/``qi`` defaulting to zero.
    """

    u = np.asarray(u, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    tx = np.asarray(t, dtype=np.float64).copy()
    qx = np.asarray(qv, dtype=np.float64).copy()
    p2d = np.asarray(p, dtype=np.float64).copy()
    p2di = np.asarray(pdi, dtype=np.float64).copy()
    pi2d = np.asarray(exner, dtype=np.float64).copy()
    dz8w = np.asarray(dz, dtype=np.float64).copy()
    nz = u.shape[0]
    qc = np.zeros(nz, dtype=np.float64) if qc is None else np.asarray(qc, dtype=np.float64).copy()
    qi = np.zeros(nz, dtype=np.float64) if qi is None else np.asarray(qi, dtype=np.float64).copy()
    tke = np.asarray(tke_in, dtype=np.float64).copy()

    cont = CP / G
    conpr = BFAC * KARMAN * SFCFRAC
    dt2 = 2.0 * dt
    rdt = 1.0 / dt2

    thx = tx / pi2d
    thvx = thx * (1.0 + EP1 * qx)
    rhox = psfc / (R_D * tx[0] * (1.0 + EP1 * qx[0]))
    govrth = G / thx[0]

    zq = np.zeros(nz + 1, dtype=np.float64)
    for k in range(nz):
        zq[k + 1] = dz8w[k] + zq[k]
    za = 0.5 * (zq[:-1] + zq[1:])
    delp = p2di[:-1] - p2di[1:]
    dza = np.empty(nz, dtype=np.float64)
    dza[0] = za[0]
    dza[1:] = za[1:] - za[:-1]

    wspd1 = np.sqrt(u[0] * u[0] + v[0] * v[0]) + 1.0e-9

    sflux = hfx / rhox / CP + qfx / rhox * EP1 * thx[0]
    sfcflg = not (br > 0.0)
    thermal = float(thvx[0])
    zl1 = float(za[0])

    q2x = 2.0 * tke.copy()

    xkzom = np.full(nz, XKZMINM, dtype=np.float64)
    xkzoh = np.full(nz, XKZMINH, dtype=np.float64)
    xkzom[-1] = 0.0  # top entry unused by K-band loops (k<=kte-1)
    xkzoh[-1] = 0.0

    # First-guess PBL height (BRCR_UB).
    kpbl, brdn, brup = _first_pbl_guess(thvx, thermal, za, br, BRCR_UB, u, v)
    hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)

    fm = psim
    fh = psih
    zol1 = max(br * fm * fm / fh, RIMIN)
    zol1 = min(zol1, -ZFMIN) if sfcflg else max(zol1, ZFMIN)
    hol1 = zol1 * hpbl / zl1 * SFCFRAC
    epshol = hol1
    if sfcflg:
        phim = (1.0 - APHI16 * hol1) ** (-0.25)
        phih = (1.0 - APHI16 * hol1) ** (-0.5)
        bfx0 = max(sflux, 0.0)
        wstar3 = govrth * bfx0 * hpbl
        wstar = wstar3 ** H1
    else:
        phim = 1.0 + APHI5 * hol1
        phih = phim
        wstar = 0.0
        wstar3 = 0.0
    ust3 = ust ** 3
    wscale = (ust3 + PHIFAC * KARMAN * wstar3 * 0.5) ** H1
    wscale = min(wscale, ust * APHI16)
    wscale = max(wscale, ust / APHI5)

    hgamt = 0.0
    hgamq = 0.0
    hgamu = 0.0
    hgamv = 0.0
    if sfcflg and sflux > 0.0:
        gamfac = BFAC / rhox / wscale
        hgamt = min(gamfac * hfx / CP, GAMCRT)
        hgamq = min(gamfac * qfx, GAMCRQ)
        vpert = (hgamt + EP1 * thx[0] * hgamq) / BFAC * AFAC
        thermal = thermal + max(vpert, 0.0) * min(za[0] / (SFCFRAC * hpbl), 1.0)
        hgamt = max(hgamt, 0.0)
        hgamq = max(hgamq, 0.0)
        brint = -15.9 * ust * ust / wspd * wstar3 / (wscale ** 4)
        hgamu = brint * u[0]
        hgamv = brint * v[0]
    else:
        pblflg = False

    # enhance PBL height using thermal.
    if pblflg:
        kpbl = 1
        hpbl = zq[0]
        kpbl, brdn, brup = _first_pbl_guess(thvx, thermal, za, br, BRCR_UB, u, v)
        hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)

    # csfac / cslen for scale-aware functions.
    cslen = 0.0
    if pblflg:
        if wstar != 0.0:
            uwst = abs(ust / wstar - 0.5)
            uwstx = -80.0 * uwst + 14.0
            csfac = 0.5 * (np.tanh(uwstx) + 3.0)
        else:
            csfac = 1.0
        cslen = csfac * hpbl

    # stable boundary layer.
    if (not sfcflg) and hpbl < zq[1]:
        brup = br
        stable = False
    else:
        stable = True

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
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[0]) / spdk2
            kpbl = k + 1
            stable = brup > brcr

    if (not sfcflg) and hpbl < zq[1]:
        hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, brcr, za, zq)

    # scale dependency for nonlocal momentum and moisture transport.
    delxy = np.sqrt(dx * dy)
    pu1 = pu(delxy, cslen)
    pq1 = pq(delxy, cslen)
    if pblflg:
        hgamu = hgamu * pu1
        hgamv = hgamv * pu1
        hgamq = hgamq * pq1

    # entrainment parameters.
    we = 0.0
    wm2 = 0.0
    hfxpbl = 0.0
    qfxpbl = 0.0
    ufxpbl = 0.0
    vfxpbl = 0.0
    prpbl = 1.0
    delta = 0.0
    deltaoh = 0.0
    enlfrac2 = 0.0
    if pblflg:
        k = kpbl - 1 - 1  # WRF k=kpbl-1, 1-based -> python kpbl-2
        prpbl = 1.0
        wm3 = wstar3 + 5.0 * ust3
        wm2 = wm3 ** H2
        bfxpbl = -0.15 * thvx[0] / G * wm3 / hpbl
        dthvx = max(thvx[k + 1] - thvx[k], TMIN)
        dthx = max(thx[k + 1] - thx[k], TMIN)
        dqx = min(qx[k + 1] - qx[k], 0.0)
        we = max(bfxpbl / dthvx, -np.sqrt(wm2))
        hfxpbl = we * dthx
        qfxpbl = we * dqx * pq1
        dux = u[k + 1] - u[k]
        dvx = v[k + 1] - v[k]
        if dux > TMIN:
            ufxpbl = max(prpbl * we * dux * pu1, -ust * ust)
        elif dux < -TMIN:
            ufxpbl = min(prpbl * we * dux * pu1, ust * ust)
        else:
            ufxpbl = 0.0
        if dvx > TMIN:
            vfxpbl = max(prpbl * we * dvx * pu1, -ust * ust)
        elif dvx < -TMIN:
            vfxpbl = min(prpbl * we * dvx * pu1, ust * ust)
        else:
            vfxpbl = 0.0
        delb = govrth * D3 * hpbl
        delta = min(D1 * hpbl + D2 * wm2 / delb, 100.0)
        delb = govrth * dthvx
        deltaoh = D1 * hpbl + D2 * wm2 / delb
        deltaoh = max(EZFAC * deltaoh, hpbl - za[kpbl - 1 - 1] - 1.0)  # za(kpbl-1) WRF
        deltaoh = min(deltaoh, hpbl)
        if (dux != 0.0) or (dvx != 0.0):
            rigs = govrth * dthvx * deltaoh / (dux ** 2 + dvx ** 2)
        else:
            rigs = RIGSMAX
        rigs = max(min(rigs, RIGSMAX), RIMIN)
        if (rigs > 0.0) and (abs(rigs + CPENT) <= 1.0e-6):
            cenlfrac = ENTFMAX
        else:
            cenlfrac = rigs / (rigs + CPENT)
        cenlfrac = min(cenlfrac, ENTFMAX)
        enlfrac2 = max(wm3 / wstar3 * cenlfrac, ENTFMIN)
        enlfrac2 = enlfrac2 * ENLFRAC

    entfac = np.full(nz, 1.0e30, dtype=np.float64)
    entfacmf = np.zeros(nz, dtype=np.float64)
    for k in range(nz):
        if pblflg:
            entfacmf[k] = np.sqrt(((zq[k + 1] - hpbl) / deltaoh) ** 2)
        if pblflg and (k + 1) >= kpbl:
            entfac[k] = ((zq[k + 1] - hpbl) / deltaoh) ** 2
        else:
            entfac[k] = 1.0e30

    # diffusion coefficients below PBL.
    xkzm = np.zeros(nz, dtype=np.float64)
    xkzh = np.zeros(nz, dtype=np.float64)
    xkzq = np.zeros(nz, dtype=np.float64)
    xkzml = np.zeros(nz, dtype=np.float64)
    xkzhl = np.zeros(nz, dtype=np.float64)
    zfac = np.zeros(nz, dtype=np.float64)
    zfacent = np.zeros(nz, dtype=np.float64)
    wscalek = np.zeros(nz, dtype=np.float64)

    for k in range(nz):
        if (k + 1) < kpbl:
            zfac[k] = min(max(1.0 - (zq[k + 1] - zl1) / (hpbl - zl1), ZFMIN), 1.0)
            zfacent[k] = (1.0 - zfac[k]) ** 3
            wscalek[k] = (ust3 + PHIFAC * KARMAN * wstar3 * (1.0 - zfac[k])) ** H1
            if sfcflg:
                prfac = conpr
                prfac2 = 15.9 * wstar3 / ust3 / (1.0 + 4.0 * KARMAN * wstar3 / ust3)
                prnumfac = -3.0 * (max(zq[k + 1] - SFCFRAC * hpbl, 0.0)) ** 2 / hpbl ** 2
            else:
                prfac = 0.0
                prfac2 = 0.0
                prnumfac = 0.0
                phim8z = 1.0 + APHI5 * zol1 * zq[k + 1] / zl1
                wscalek[k] = max(ust / phim8z, 0.001)
            prnum0 = max(min(phih / phim + prfac, PRMAX), PRMIN)
            xkzm[k] = wscalek[k] * KARMAN * zq[k + 1] * zfac[k] ** PFAC
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            xkzq[k] = xkzm[k] / prnum * zfac[k] ** (PFAC_Q - PFAC)
            prnum0 = prnum0 / (1.0 + prfac2 * KARMAN * SFCFRAC)
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            xkzh[k] = xkzm[k] / prnum
            xkzm[k] = min(xkzm[k] + xkzom[k], XKZMAX)
            xkzh[k] = min(xkzh[k] + xkzoh[k], XKZMAX)
            xkzq[k] = min(xkzq[k] + xkzoh[k], XKZMAX)

    # diffusion coefficients over PBL (free atmosphere).
    for k in range(nz - 1):
        if (k + 1) >= kpbl:
            ss = (((u[k + 1] - u[k]) ** 2 + (v[k + 1] - v[k]) ** 2) / (dza[k + 1] ** 2)) + 1.0e-9
            govrthv = G / (0.5 * (thvx[k + 1] + thvx[k]))
            ri = govrthv * (thvx[k + 1] - thvx[k]) / (ss * dza[k + 1])
            # in-cloud Ri modification (inert when qc==qi==0).
            if IMVDIF == 1:
                if (qc[k] + qi[k]) > 0.01e-3 and (qc[k + 1] + qi[k + 1]) > 0.01e-3:
                    qmean = 0.5 * (qx[k] + qx[k + 1])
                    tmean = 0.5 * (tx[k] + tx[k + 1])
                    alpha = XLV * qmean / R_D / tmean
                    chi = XLV * XLV * qmean / CP / R_V / tmean / tmean
                    ri = (1.0 + alpha) * (ri - G * G / ss / tmean / CP * ((chi - alpha) / (1.0 + chi)))
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

    # prescribe nonlocal heat transport below PBL (Shin-Hong mass-flux profile).
    deltaoh_n = deltaoh / hpbl if hpbl != 0.0 else 0.0
    mf = np.zeros(nz, dtype=np.float64)
    zfacmf = np.zeros(nz, dtype=np.float64)
    delxy = np.sqrt(dx * dy)
    mlfrac = MLTOP - deltaoh_n
    zfacmf0 = min(max(zq[1] / hpbl, ZFMIN), 1.0) if hpbl != 0.0 else ZFMIN
    sfcfracn = max(SFCFRACN1, zfacmf0)
    sflux0 = (A11 + A12 * sfcfracn) * sflux
    snlflux0 = NLFRAC * sflux0
    amf1 = snlflux0 / sfcfracn
    amf2 = 0.0
    bmf2 = 0.0
    if pblflg:
        amf2 = -snlflux0 / (mlfrac - sfcfracn)
        bmf2 = -mlfrac * amf2
    if (deltaoh_n == 0.0) and (enlfrac2 == 0.0):
        amf3 = 0.0
    else:
        amf3 = snlflux0 * enlfrac2 / deltaoh_n
    bmf3 = -amf3 * mlfrac
    hfxpbl_prof = amf3 + bmf3
    pth1_nl = pthnl(delxy, cslen)
    hfxpbl_prof = hfxpbl_prof * pth1_nl
    for k in range(nz):
        zfacmf[k] = max(zq[k + 1] / hpbl, ZFMIN) if hpbl != 0.0 else ZFMIN
        if pblflg and (k + 1) < kpbl:
            if zfacmf[k] <= sfcfracn:
                mf[k] = amf1 * zfacmf[k]
            elif zfacmf[k] <= mlfrac:
                mf[k] = amf2 * zfacmf[k] + bmf2
            mf[k] = mf[k] + hfxpbl_prof * np.exp(-entfacmf[k])
            mf[k] = mf[k] * pth1_nl

    # ----- heat tridiagonal solve -----
    exch_h = np.zeros(nz, dtype=np.float64)
    rhs = np.zeros(nz, dtype=np.float64)
    lower = np.zeros(nz, dtype=np.float64)
    diag = np.zeros(nz, dtype=np.float64)
    upper = np.zeros(nz, dtype=np.float64)
    diag[0] = 1.0
    rhs[0] = thx[0] - 300.0 + hfx / cont / delp[0] * dt2
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p2d[k] - p2d[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzh[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzt = tem1 * (-mf[k] / xkzh[k])
            rhs[k] = rhs[k] + dtodsd * dsdzt
            rhs[k + 1] = thx[k + 1] - 300.0 - dtodsu * dsdzt
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzh[k] = -we * dza[kpbl - 1] * np.exp(-entfac[k])
            xkzh[k] = np.sqrt(xkzh[k] * xkzhl[k])
            xkzh[k] = min(max(xkzh[k], xkzoh[k]), XKZMAX)
            rhs[k + 1] = thx[k + 1] - 300.0
        else:
            rhs[k + 1] = thx[k + 1] - 300.0
        tem1 = dsig * xkzh[k] * rdz
        dsdz2 = tem1 * rdz
        upper[k] = -dtodsd * dsdz2
        lower[k + 1] = -dtodsu * dsdz2
        # scale dependency for local heat transport.
        zfacdx = 0.2 * hpbl / zq[k + 1]
        delxy_l = np.sqrt(dx * dy) * max(zfacdx, 1.0)
        pth1 = pthl(delxy_l, hpbl)
        if pblflg and (k + 1) < kpbl:
            upper[k] = upper[k] * pth1
            lower[k + 1] = lower[k + 1] * pth1
        diag[k] = diag[k] - upper[k]
        diag[k + 1] = 1.0 - lower[k + 1]
        exch_h[k + 1] = xkzh[k]
    theta_sol = _solve_tridiagonal(lower, diag, upper, rhs)
    # WRF: ttend=(f1-thx+300)*rdt*pi2d (RTHBLTEN), later divided by pi -> theta.
    theta_tend = (theta_sol - thx + 300.0) * rdt
    tflux_e = np.zeros(nz, dtype=np.float64)
    for k in range(nz - 1, -1, -1):
        ttend = theta_tend[k]  # WRF tflux_e uses ttend in theta-units (post /pi)
        if k == nz - 1:
            tflux_e[k] = ttend * dz8w[k]
        else:
            tflux_e[k] = tflux_e[k + 1] + ttend * dz8w[k]

    # ----- moisture (qv) tridiagonal solve -----
    for k in range(nz - 1):
        if (k + 1) >= kpbl:
            xkzq[k] = xkzh[k]
    rhs_q = np.zeros(nz, dtype=np.float64)
    lower_q = np.zeros(nz, dtype=np.float64)
    diag_q = np.zeros(nz, dtype=np.float64)
    upper_q = np.zeros(nz, dtype=np.float64)
    diag_q[0] = 1.0
    rhs_q[0] = qx[0] + qfx * G / delp[0] * dt2
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p2d[k] - p2d[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzq[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzq = tem1 * (-qfxpbl * zfacent[k] / xkzq[k])
            rhs_q[k] = rhs_q[k] + dtodsd * dsdzq
            rhs_q[k + 1] = qx[k + 1] - dtodsu * dsdzq
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzq[k] = -we * dza[kpbl - 1] * np.exp(-entfac[k])
            xkzq[k] = np.sqrt(xkzq[k] * xkzhl[k])
            xkzq[k] = min(max(xkzq[k], xkzoh[k]), XKZMAX)
            rhs_q[k + 1] = qx[k + 1]
        else:
            rhs_q[k + 1] = qx[k + 1]
        tem1 = dsig * xkzq[k] * rdz
        dsdz2 = tem1 * rdz
        upper_q[k] = -dtodsd * dsdz2
        lower_q[k + 1] = -dtodsu * dsdz2
        # scale dependency for local moisture transport.
        zfacdx = 0.2 * hpbl / zq[k + 1]
        delxy_l = np.sqrt(dx * dy) * max(zfacdx, 1.0)
        pq1l = pq(delxy_l, hpbl)
        if pblflg and (k + 1) < kpbl:
            upper_q[k] = upper_q[k] * pq1l
            lower_q[k + 1] = lower_q[k + 1] * pq1l
        diag_q[k] = diag_q[k] - upper_q[k]
        diag_q[k + 1] = 1.0 - lower_q[k + 1]
    qv_sol = _solve_tridiagonal(lower_q, diag_q, upper_q, rhs_q)
    qv_tend = (qv_sol - qx) * rdt
    qflux_e = np.zeros(nz, dtype=np.float64)
    tvflux_e = np.zeros(nz, dtype=np.float64)
    for k in range(nz - 1, -1, -1):
        qtend = qv_tend[k]
        if k == nz - 1:
            qflux_e[k] = qtend * dz8w[k]
        else:
            qflux_e[k] = qflux_e[k + 1] + qtend * dz8w[k]
        tvflux_e[k] = tflux_e[k] + qflux_e[k] * EP1 * thx[k]

    # hgame2d (used by vdifq).
    hgame2d = np.zeros(nz, dtype=np.float64)
    for k in range(nz):
        if pblflg and (k + 1) < kpbl:
            hgame_c = C_1 * 0.2 * 2.5 * (G / thvx[k]) * wstar / (0.25 * (q2x[k + 1] + q2x[k]))
            hgame_c = min(hgame_c, GAMCRE)
            if k == nz - 1:
                hgame2d[k] = max(hgame_c * 0.5 * tvflux_e[k] * hpbl, 0.0)
            else:
                hgame2d[k] = max(hgame_c * 0.5 * (tvflux_e[k] + tvflux_e[k + 1]) * hpbl, 0.0)

    # ----- momentum tridiagonal solve -----
    rhs_u = np.zeros(nz, dtype=np.float64)
    rhs_v = np.zeros(nz, dtype=np.float64)
    lower_m = np.zeros(nz, dtype=np.float64)
    diag_m = np.zeros(nz, dtype=np.float64)
    upper_m = np.zeros(nz, dtype=np.float64)
    diag_m[0] = 1.0 + ctopo * ust ** 2 / wspd1 * rhox * G / delp[0] * dt2 * (wspd1 / wspd) ** 2
    rhs_u[0] = u[0]
    rhs_v[0] = v[0]
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p2d[k] - p2d[k + 1]
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
        # scale dependency for local momentum transport.
        zfacdx = 0.2 * hpbl / zq[k + 1]
        delxy_l = np.sqrt(dx * dy) * max(zfacdx, 1.0)
        pu1l = pu(delxy_l, hpbl)
        if pblflg and (k + 1) < kpbl:
            upper_m[k] = upper_m[k] * pu1l
            lower_m[k + 1] = lower_m[k + 1] * pu1l
        diag_m[k] = diag_m[k] - upper_m[k]
        diag_m[k + 1] = 1.0 - lower_m[k + 1]
    u_sol = _solve_tridiagonal(lower_m, diag_m.copy(), upper_m, rhs_u)
    v_sol = _solve_tridiagonal(lower_m, diag_m.copy(), upper_m, rhs_v)
    u_tend = (u_sol - u) * rdt
    v_tend = (v_sol - v) * rdt

    # ----- SGS TKE diagnostic path (shinhong_tke_diag == 1) -----
    el_pbl = np.zeros(nz, dtype=np.float64)
    tke_out = tke.copy()
    if shinhong_tke_diag == 1:
        akmk = np.zeros(nz, dtype=np.float64)
        akhk = np.zeros(nz, dtype=np.float64)
        mfk = np.zeros(nz, dtype=np.float64)
        ufxpblk = np.zeros(nz, dtype=np.float64)
        vfxpblk = np.zeros(nz, dtype=np.float64)
        qfxpblk = np.zeros(nz, dtype=np.float64)
        for k in range(1, nz):  # WRF kts+1..kte: akmk(k)=xkzm(k-1) etc.
            akmk[k] = xkzm[k - 1]
            akhk[k] = xkzh[k - 1]
            mfk[k] = mf[k - 1] / xkzh[k - 1]
            ufxpblk[k] = ufxpbl * zfacent[k - 1] / xkzm[k - 1]
            vfxpblk[k] = vfxpbl * zfacent[k - 1] / xkzm[k - 1]
            qfxpblk[k] = qfxpbl * zfacent[k - 1] / xkzq[k - 1]

        ptke1 = np.ones(nz, dtype=np.float64)
        for k in range(nz - 1):  # WRF kts..kte-1
            if pblflg and (k + 1) <= kpbl:
                zfacdx = 0.2 * hpbl / za[k]
                delxy_t = np.sqrt(dx * dy) * max(zfacdx, 1.0)
                ptke1[k + 1] = ptke(delxy_t, hpbl)

        efxpbl = 0.0
        if pblflg:
            k = kpbl - 1 - 1  # WRF kpbl-1 -> python kpbl-2
            dex = 0.25 * (q2x[k + 2] - q2x[k])
            efxpbl = we * dex

        delxy_t = np.sqrt(dx * dy)

        # mixlen uses s2l=(suk-hgamu/hpbl-ufxpbl(k))*suk with ufxpbl(k)=entrain;
        # fold the hgam/hpbl countergradient part into the passed arrays.
        ufxpbl_fold = np.zeros(nz, dtype=np.float64)
        vfxpbl_fold = np.zeros(nz, dtype=np.float64)
        qfxpbl_fold = np.zeros(nz, dtype=np.float64)
        for k in range(1, nz):
            ufxpbl_fold[k] = (hgamu / hpbl + ufxpblk[k]) if hpbl != 0.0 else ufxpblk[k]
            vfxpbl_fold[k] = (hgamv / hpbl + vfxpblk[k]) if hpbl != 0.0 else vfxpblk[k]
            qfxpbl_fold[k] = (hgamq / hpbl + qfxpblk[k]) if hpbl != 0.0 else qfxpblk[k]

        cwm = np.zeros(nz, dtype=np.float64)  # cloud water = 0 in oracle cases
        s2k, ghk, rik, elk = _mixlen(
            u, v, tx, thx, qx, cwm, q2x, zq, ust, corf, epshol, hpbl, kpbl, pblflg,
            mfk, ufxpbl_fold, vfxpbl_fold, qfxpbl_fold,
        )

        q2_work = q2x.copy()
        q2_work = _prodq2(
            dt, ust, s2k, rik, q2_work, elk, zq, akmk, akhk, u, v, thx, thvx,
            hgamu, hgamv, hgamq, delxy_t, hpbl, pblflg, kpbl, mfk, ufxpblk, vfxpblk, qfxpblk,
        )
        q2_work = _vdifq(dt, q2_work, elk, zq, akhk, ptke1, hgame2d, hpbl, pblflg, kpbl, efxpbl)

        for k in range(nz):
            q2x[k] = max(q2_work[k], EPSQ2L)
            tke_out[k] = 0.5 * q2x[k]
            if k != 0:
                el_pbl[k] = elk[k]

    return ShinHongResult(
        u_tend=u_tend,
        v_tend=v_tend,
        theta_tend=theta_tend,
        qv_tend=qv_tend,
        exch_h=exch_h,
        pblh=float(hpbl),
        kpbl=int(kpbl),
        wstar=float(wstar),
        delta=float(delta),
        tke=tke_out,
        el_pbl=el_pbl,
    )
