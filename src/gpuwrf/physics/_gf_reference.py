"""Faithful single-column reference port of WRF Grell-Freitas cumulus.

This is a line-faithful NumPy translation of the WRF ``cu_physics=3`` call path
``GFDRV`` -> ``cup_gf`` (deep) + ``cup_gf_sh`` (shallow), from pristine
``module_cu_gf_deep.F`` / ``module_cu_gf_sh.F`` / ``module_cu_gf_wrfdrv.F``.

It is intentionally sequential (1-based-index emulation, level-by-level plume /
closure iterations) because the GF closure ensemble is inherently iterative; the
v0.6.0 savepoint gate runs a single column on CPU, so a faithful reference port
is the correct equivalence artifact. A vectorized/JAX hot-path is a separate
optimization sprint (recorded for the manager).

Indexing convention: WRF arrays are Fortran 1..KX. Here we use NumPy arrays of
length KX+1 with index 0 unused, so that all ``i,k`` references match the
Fortran source 1:1. ``itf=KX`` single column ``i=1``.

Constants exactly mirror the WRF modules.
"""

from __future__ import annotations

import math

import numpy as np

# --- deep-module parameters (module_cu_gf_deep.F head) ---
G = 9.81
CP = 1004.0
XLV = 2.5e6
R_V = 461.0
TCRIT = 258.0
C1 = 0.001
IRAINEVAP = 0
FRH_THRESH = 0.9
RH_THRESH = 0.97
BETAJB = 1.5
USE_EXCESS = 1
FLUXTUNE = 1.5
PGCD = 1.0
AUTOCONV = 1
AEROEVAP = 1
CCNCLEAN = 250.0
MAXENS3 = 16

# --- shallow-module parameters (module_cu_gf_sh.F head) ---
C1_SHAL = 0.0
C0_SHAL = 0.001

BDISPM = 0.366
BDISPC = 0.146


def _z(kx):
    """Return a fresh 1-based float array of length kx+1 (index 0 unused)."""
    return np.zeros(kx + 1, dtype=np.float64)


def satvap(temp2):
    """REAL FUNCTION satvap(temp2) from module_cu_gf_deep.F (Goff-Gratch)."""
    temp = temp2 - 273.155
    if temp < -20.0:  # ice saturation
        toot = 273.16 / temp2
        toto = 1.0 / toot
        eilog = (-9.09718 * (toot - 1.0) - 3.56654 * (math.log(toot) / math.log(10.0))
                 + 0.876793 * (1.0 - toto) + (math.log(6.1071) / math.log(10.0)))
        return 10.0 ** eilog
    tsot = 373.16 / temp2
    ewlog = -7.90298 * (tsot - 1.0) + 5.02808 * (math.log(tsot) / math.log(10.0))
    ewlog2 = ewlog - 1.3816e-07 * (10.0 ** (11.344 * (1.0 - (1.0 / tsot))) - 1.0)
    ewlog3 = ewlog2 + 0.0081328 * (10.0 ** (-3.49149 * (tsot - 1.0)) - 1.0)
    ewlog4 = ewlog3 + (math.log(1013.246) / math.log(10.0))
    return 10.0 ** ewlog4


def cup_env(z, t, q, p, z1, psur, ierr, itest, ktf):
    """cup_env: returns (z, qes, he, hes). itest=-1 keeps z, computes he/hes/qes."""
    kx = len(t) - 1
    qes = _z(kx)
    he = _z(kx)
    hes = _z(kx)
    tv = _z(kx)
    if ierr != 0:
        return z, qes, he, hes
    for k in range(1, ktf + 1):
        e = satvap(t[k])
        qes[k] = 0.622 * e / max(1.0e-8, (p[k] - e))
        if qes[k] <= 1.0e-16:
            qes[k] = 1.0e-16
        if qes[k] < q[k]:
            qes[k] = q[k]
        tv[k] = t[k] + 0.608 * q[k] * t[k]
    if itest in (1, 0):
        z[1] = max(0.0, z1) - (math.log(p[1]) - math.log(psur)) * 287.0 * tv[1] / 9.81
        for k in range(2, ktf + 1):
            tvbar = 0.5 * tv[k] + 0.5 * tv[k - 1]
            z[k] = z[k - 1] - (math.log(p[k]) - math.log(p[k - 1])) * 287.0 * tvbar / 9.81
    elif itest == 2:
        for k in range(1, ktf + 1):
            z[k] = (he[k] - 1004.0 * t[k] - 2.5e6 * q[k]) / 9.81
            z[k] = max(1.0e-3, z[k])
    for k in range(1, ktf + 1):
        if itest <= 0:
            he[k] = 9.81 * z[k] + 1004.0 * t[k] + 2.5e6 * q[k]
        hes[k] = 9.81 * z[k] + 1004.0 * t[k] + 2.5e6 * qes[k]
        if he[k] >= hes[k]:
            he[k] = hes[k]
    return z, qes, he, hes


def cup_env_clev(t, qes, q, he, hes, z, p, psur, z1, ierr, ktf):
    """cup_env_clev: returns the *_cup arrays."""
    kx = len(t) - 1
    qes_cup = _z(kx); q_cup = _z(kx); hes_cup = _z(kx); he_cup = _z(kx)
    z_cup = _z(kx); p_cup = _z(kx); gamma_cup = _z(kx); t_cup = _z(kx)
    if ierr != 0:
        return qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup
    for k in range(2, ktf + 1):
        qes_cup[k] = 0.5 * (qes[k - 1] + qes[k])
        q_cup[k] = 0.5 * (q[k - 1] + q[k])
        hes_cup[k] = 0.5 * (hes[k - 1] + hes[k])
        he_cup[k] = 0.5 * (he[k - 1] + he[k])
        if he_cup[k] > hes_cup[k]:
            he_cup[k] = hes_cup[k]
        z_cup[k] = 0.5 * (z[k - 1] + z[k])
        p_cup[k] = 0.5 * (p[k - 1] + p[k])
        t_cup[k] = 0.5 * (t[k - 1] + t[k])
        gamma_cup[k] = (XLV / CP) * (XLV / (R_V * t_cup[k] * t_cup[k])) * qes_cup[k]
    qes_cup[1] = qes[1]
    q_cup[1] = q[1]
    hes_cup[1] = 9.81 * z1 + 1004.0 * t[1] + 2.5e6 * qes[1]
    he_cup[1] = 9.81 * z1 + 1004.0 * t[1] + 2.5e6 * q[1]
    z_cup[1] = z1
    p_cup[1] = psur
    t_cup[1] = t[1]
    gamma_cup[1] = XLV / CP * (XLV / (R_V * t_cup[1] * t_cup[1])) * qes_cup[1]
    return qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup


def get_cloud_bc(array, k22, add=0.0):
    """get_cloud_bc: average of array over k22-2..k22 (order_aver=3)."""
    order_aver = 3
    local_order_aver = min(k22, order_aver)
    x_aver = 0.0
    for i in range(1, local_order_aver + 1):
        x_aver += array[k22 - i + 1]
    x_aver = x_aver / float(local_order_aver)
    return x_aver + add


def cup_minimi(array, ks, kend, ierr, ktf):
    """cup_minimi: index of min of array over ks..max(ks+1,kend)."""
    kt = ks
    if ierr != 0:
        return kt
    x = array[ks]
    kstop = max(ks + 1, kend)
    for k in range(ks + 1, kstop + 1):
        if array[k] < x:
            x = array[k]
            kt = k
    return kt


def cup_maximi(array, ks, ke, ierr, ktf):
    """cup_MAXIMI: index of max of array over ks..ke."""
    maxx = ks
    if ierr != 0:
        return maxx
    x = array[ks]
    for k in range(ks, ke + 1):
        xar = array[k]
        if xar >= x:
            x = xar
            maxx = k
    return maxx


def _maxloc1(arr_1based, ktf=None):
    """Fortran maxloc(arr(:),1) on a 1-based array (index 0 unused)."""
    n = len(arr_1based) - 1 if ktf is None else ktf
    best = 1
    bestv = arr_1based[1]
    for k in range(1, n + 1):
        if arr_1based[k] > bestv:
            bestv = arr_1based[k]
            best = k
    return best


def cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, ktf):
    """cup_up_aa0: cloud work function aa0."""
    aa0 = 0.0
    if ierr != 0:
        return 0.0
    for k in range(2, ktf + 1):
        if k < kbcon:
            continue
        if k > ktop:
            continue
        dz = z[k] - z[k - 1]
        da = zu[k] * dz * (9.81 / (1004.0 * (t_cup[k]))) * dby[k - 1] / (1.0 + gamma_cup[k])
        aa0 = aa0 + max(0.0, da)
        if aa0 < 0.0:
            aa0 = 0.0
    return aa0


def cup_up_aa1bl_full(z, t, tn, q, qo, dtime, kbcon, ierr, ktf):
    """cup_up_aa1bl faithful: dA = dz*g*(dtn + .608*dqo)/dtime, k<=kbcon."""
    aa0 = 0.0
    if ierr != 0:
        return 0.0
    for k in range(2, ktf + 1):
        if k > kbcon:
            continue
        dz = z[k] - z[k - 1]
        da = dz * 9.81 * (tn[k] - t[k] + 0.608 * (qo[k] - q[k])) / dtime
        aa0 = aa0 + da
    return aa0


def get_zu_zd_pdf_fim(kklev, p, zubeg, xland, draft, ierr, kb, kt, kts, kte, ktf,
                      max_mass, kpbli, csum, pmin_lev):
    """get_zu_zd_pdf_fim: beta-PDF normalized mass-flux profile.

    Returns zu (1-based length kte+1). Faithful to the UP/SH2/DOWN/MID branches.
    """
    zu = np.zeros(kte + 1, dtype=np.float64)
    kb_adj = max(kb, 2)
    if draft == "UP":
        lev_start = min(0.9, 0.4 + csum * 0.013)
        kb_adj = max(kb, 2)
        tunning = p[kt] + (p[kpbli] - p[kt]) * lev_start
        tunning = min(0.9, (tunning - p[kb_adj]) / (p[kt] - p[kb_adj]))
        tunning = max(0.2, tunning)
        beta = 1.3
        alpha = (tunning * (beta - 2.0) + 1.0) / (1.0 - tunning)
        fzu = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
        for k in range(kb_adj, min(kte, kt) + 1):
            kratio = (p[k] - p[kb_adj]) / (p[kt] - p[kb_adj])
            zu[k] = zubeg + fzu * kratio ** (alpha - 1.0) * (1.0 - kratio) ** (beta - 1.0)
        hi = min(ktf, kt + 1)
        mx = np.max(zu[kts:hi + 1])
        if mx > 0.0:
            zu[kts:hi + 1] = zu[kts:hi + 1] / mx
        ml = _maxloc1(zu, kte)
        for k in range(ml, 0, -1):
            if zu[k] < 1.0e-6:
                kb_adj = k + 1
                break
        kb_adj = max(2, kb_adj)
        for k in range(kts, kb_adj):
            zu[k] = 0.0
    elif draft == "SH2":
        tunning = min(0.8, (p[kpbli] - p[kb_adj]) / (p[kt] - p[kb_adj]))
        tunning = max(0.2, tunning)
        beta = 2.5
        alpha = (tunning * (beta - 2.0) + 1.0) / (1.0 - tunning)
        fzu = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
        for k in range(kb_adj, min(kte, kt) + 1):
            kratio = (p[k] - p[kb_adj]) / (p[kt] - p[kb_adj])
            zu[k] = zubeg + fzu * kratio ** (alpha - 1.0) * (1.0 - kratio) ** (beta - 1.0)
        hi = min(ktf, kt + 1)
        mx = np.max(zu[kts:hi + 1])
        if mx > 0.0:
            zu[kts:hi + 1] = zu[kts:hi + 1] / mx
        ml = _maxloc1(zu, kte)
        for k in range(ml, 0, -1):
            if zu[k] < 1.0e-6:
                kb_adj = k + 1
                break
    elif draft == "MID":
        kb_adj = max(kb, 2)
        tunning = p[kt] + (p[kb_adj] - p[kt]) * 0.9
        tunning = min(0.9, (tunning - p[kb_adj]) / (p[kt] - p[kb_adj]))
        tunning = max(0.2, tunning)
        beta = 1.3
        alpha = (tunning * (beta - 2.0) + 1.0) / (1.0 - tunning)
        fzu = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
        for k in range(kb_adj, min(kte, kt) + 1):
            kratio = (p[k] - p[kb_adj]) / (p[kt] - p[kb_adj])
            zu[k] = zubeg + fzu * kratio ** (alpha - 1.0) * (1.0 - kratio) ** (beta - 1.0)
        hi = min(ktf, kt + 1)
        mx = np.max(zu[kts:hi + 1])
        if mx > 0.0:
            zu[kts:hi + 1] = zu[kts:hi + 1] / mx
        ml = _maxloc1(zu, kte)
        for k in range(ml, 0, -1):
            if zu[k] < 1.0e-6:
                kb_adj = k + 1
                break
        kb_adj = max(2, kb_adj)
        for k in range(kts, kb_adj):
            zu[k] = 0.0
    elif draft in ("DOWN", "DOWNM"):
        tunning = p[kb]
        tunning = min(0.9, (tunning - p[1]) / (p[kt] - p[1]))
        tunning = max(0.2, tunning)
        beta = 4.0
        alpha = (tunning * (beta - 2.0) + 1.0) / (1.0 - tunning)
        fzu = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
        for k in range(2, min(kt, ktf) + 1):
            kratio = (p[k] - p[1]) / (p[kt] - p[1])
            zu[k] = fzu * kratio ** (alpha - 1.0) * (1.0 - kratio) ** (beta - 1.0)
        hi = min(ktf, kt + 1)
        fzu2 = np.max(zu[kts:hi + 1])
        if fzu2 > 0.0:
            zu[kts:hi + 1] = zu[kts:hi + 1] / fzu2
        zu[1] = 0.0
    return zu


def get_lateral_massflux(zo_cup, zuo, cd, entr_rate_2d, ktop, kbcon, k22,
                         lambau, ktf, with_u=True):
    """get_lateral_massflux: up_massentro/detro (+ u variants). Modifies cd, entr_rate_2d in place."""
    kte = len(zo_cup) - 1
    up_massentro = np.zeros(kte + 1); up_massdetro = np.zeros(kte + 1)
    up_massentr = np.zeros(kte + 1); up_massdetr = np.zeros(kte + 1)
    up_massentru = np.zeros(kte + 1); up_massdetru = np.zeros(kte + 1)
    mlz = _maxloc1(zuo, kte)
    for k in range(max(2, k22 + 1), mlz + 1):
        dz = zo_cup[k] - zo_cup[k - 1]
        up_massdetro[k - 1] = cd[k - 1] * dz * zuo[k - 1]
        up_massentro[k - 1] = zuo[k] - zuo[k - 1] + up_massdetro[k - 1]
        if up_massentro[k - 1] < 0.0:
            up_massentro[k - 1] = 0.0
            up_massdetro[k - 1] = zuo[k - 1] - zuo[k]
            if zuo[k - 1] > 0.0:
                cd[k - 1] = up_massdetro[k - 1] / (dz * zuo[k - 1])
        if zuo[k - 1] > 0.0:
            entr_rate_2d[k - 1] = up_massentro[k - 1] / (dz * zuo[k - 1])
    for k in range(mlz + 1, ktop + 1):
        dz = zo_cup[k] - zo_cup[k - 1]
        up_massentro[k - 1] = entr_rate_2d[k - 1] * dz * zuo[k - 1]
        up_massdetro[k - 1] = zuo[k - 1] + up_massentro[k - 1] - zuo[k]
        if up_massdetro[k - 1] < 0.0:
            up_massdetro[k - 1] = 0.0
            up_massentro[k - 1] = zuo[k] - zuo[k - 1]
            if zuo[k - 1] > 0.0:
                entr_rate_2d[k - 1] = up_massentro[k - 1] / (dz * zuo[k - 1])
        if zuo[k - 1] > 0.0:
            cd[k - 1] = up_massdetro[k - 1] / (dz * zuo[k - 1])
    up_massdetro[ktop] = zuo[ktop]
    up_massentro[ktop] = 0.0
    for k in range(ktop + 1, ktf + 1):
        cd[k] = 0.0
        entr_rate_2d[k] = 0.0
        up_massentro[k] = 0.0
        up_massdetro[k] = 0.0
    for k in range(2, ktf):
        up_massentr[k - 1] = up_massentro[k - 1]
        up_massdetr[k - 1] = up_massdetro[k - 1]
    if with_u:
        for k in range(2, ktf):
            up_massentru[k - 1] = up_massentro[k - 1] + lambau * up_massdetro[k - 1]
            up_massdetru[k - 1] = up_massdetro[k - 1] + lambau * up_massdetro[k - 1]
    return (up_massentro, up_massdetro, up_massentr, up_massdetr,
            up_massentru, up_massdetru)


def cup_kbcon(cap_inc, iloop_in, k22, kbcon, he_cup, hes_cup, hkb, ierr, kbmax,
              p_cup, cap_max, ztexec, zqexec, z_cup, entr_rate, heo, imid, ktf):
    """cup_kbcon faithful. Returns (k22, kbcon, hkb, ierr). Single column."""
    kte = len(he_cup) - 1
    iloop = iloop_in
    kbcon = 1
    if cap_max > 200 and imid == 1:
        iloop = 5
    if ierr != 0:
        return k22, kbcon, hkb, ierr
    start_level = k22
    kbcon = k22 + 1
    if iloop == 5:
        kbcon = k22
    hcot = _z(kte)
    for k in range(1, start_level + 1):
        hcot[k] = hkb
    for k in range(start_level + 1, kbmax + 3 + 1):
        dz = z_cup[k] - z_cup[k - 1]
        hcot[k] = ((1.0 - 0.5 * entr_rate * dz) * hcot[k - 1]
                   + entr_rate * dz * heo[k - 1]) / (1.0 + 0.5 * entr_rate * dz)
    # label 32 entry
    while True:
        # 32 CONTINUE
        hetest = hcot[kbcon]
        if hetest < hes_cup[kbcon]:
            # 31: increment kbcon, retest
            kbcon = kbcon + 1
            if kbcon > kbmax + 2:
                if iloop != 4:
                    ierr = 3
                return k22, kbcon, hkb, ierr  # GO TO 27
            continue
        # found hetest >= hes_cup
        if kbcon - k22 == 1:
            return k22, kbcon, hkb, ierr  # GO TO 27
        if iloop == 5 and (kbcon - k22) <= 2:
            return k22, kbcon, hkb, ierr
        pbcdif = -p_cup[kbcon] + p_cup[k22]
        plus = max(25.0, cap_max - float(iloop - 1) * cap_inc)
        if iloop == 4:
            plus = cap_max
        if iloop == 5:
            plus = 150.0
        if iloop == 5 and cap_max > 200:
            pbcdif = -p_cup[kbcon] + cap_max
        if pbcdif <= plus:
            return k22, kbcon, hkb, ierr  # GO TO 27
        # pbcdif > plus: raise k22
        k22 = k22 + 1
        kbcon = k22 + 1
        x_add = XLV * zqexec + CP * ztexec
        hkb = get_cloud_bc(he_cup, k22, x_add)
        start_level = k22
        hcot = _z(kte)
        for k in range(1, start_level + 1):
            hcot[k] = hkb
        for k in range(start_level + 1, kbmax + 3 + 1):
            dz = z_cup[k] - z_cup[k - 1]
            hcot[k] = ((1.0 - 0.5 * entr_rate * dz) * hcot[k - 1]
                       + entr_rate * dz * heo[k - 1]) / (1.0 + 0.5 * entr_rate * dz)
        if iloop == 5:
            kbcon = k22
        if kbcon > kbmax + 2:
            if iloop != 4:
                ierr = 3
            return k22, kbcon, hkb, ierr
        # GO TO 32 (loop)


def rates_up_pdf(name, ktop, ierr, p_cup, entr_rate_2d, hkbo, heo, heso_cup,
                 z_cup, xland, kstabi, k22, kbcon, zuo, kpbl, ktopdby, csum,
                 pmin_lev, kts, kte, ktf):
    """rates_up_pdf faithful. Modifies entr_rate_2d, zuo, kbcon, ktop, ktopdby, ierr.

    Returns (kbcon, ktop, ktopdby, ierr, zuo).
    """
    dbythresh = 1.0
    if name in ("shallow", "mid"):
        dbythresh = 1.0
    zustart = 0.1
    zux = np.zeros(kte + 1)
    beta_u = max(0.1, 0.2 - float(csum) * 0.01)
    zuo[:] = 0.0
    dby = np.zeros(kte + 1)
    dbm = np.zeros(kte + 1)
    kbcon = max(kbcon, 2)
    if ierr != 0:
        return kbcon, ktop, ktopdby, ierr, zuo
    start_level = k22
    zuo[start_level] = zustart
    zux[start_level] = zustart
    for k in range(start_level + 1, kbcon + 1):
        dz = z_cup[k] - z_cup[k - 1]
        massent = dz * entr_rate_2d[k - 1] * zuo[k - 1]
        massdetr = dz * 1.0e-9 * zuo[k - 1]
        zuo[k] = zuo[k - 1] + massent - massdetr
        zux[k] = zuo[k]
    zubeg = zustart
    kklev = 1
    if name == "deep":
        ktop = 0
        hcot = np.zeros(kte + 1)
        hcot[start_level] = hkbo
        for k in range(start_level + 1, ktf - 2 + 1):
            dz = z_cup[k] - z_cup[k - 1]
            hcot[k] = ((1.0 - 0.5 * entr_rate_2d[k - 1] * dz) * hcot[k - 1]
                       + entr_rate_2d[k - 1] * dz * heo[k - 1]) / (1.0 + 0.5 * entr_rate_2d[k - 1] * dz)
            if k >= kbcon:
                dby[k] = dby[k - 1] + (hcot[k] - heso_cup[k]) * dz
            if k >= kbcon:
                dbm[k] = hcot[k] - heso_cup[k]
        ktopdby = _maxloc1(dby, kte)
        kklev = _maxloc1(dbm, kte)
        dby_max = np.max(dby[1:kte + 1])
        kfinalzu = ktf - 2
        ktop = kfinalzu
        for k in range(_maxloc1(dby, kte) + 1, ktf - 2 + 1):
            if dby[k] < dbythresh * dby_max:
                kfinalzu = k - 1
                ktop = kfinalzu
                break
        if kfinalzu <= kbcon + 2:
            ierr = 41
            ktop = 0
        else:
            zu = get_zu_zd_pdf_fim(kklev, p_cup, zubeg, xland, "UP", ierr, k22,
                                   kfinalzu, kts, kte, ktf, beta_u, kstabi, csum, pmin_lev)
            zuo[:] = zu
    elif name == "mid":
        if ktop <= kbcon + 2:
            ierr = 41
            ktop = 0
        else:
            kfinalzu = ktop
            ktopdby = ktop + 1
            zu = get_zu_zd_pdf_fim(kklev, p_cup, zubeg, xland, "MID", ierr, k22,
                                   kfinalzu, kts, kte, ktf, beta_u, kbcon, csum, pmin_lev)
            zuo[:] = zu
    elif name == "shallow":
        if ktop <= kbcon + 2:
            ierr = 41
            ktop = 0
        else:
            kfinalzu = ktop
            ktopdby = ktop
            zu = get_zu_zd_pdf_fim(kklev, p_cup, zubeg, xland, "SH2", ierr, k22,
                                   kfinalzu, kts, kte, ktf, beta_u, kpbl, csum, pmin_lev)
            zuo[:] = zu
    return kbcon, ktop, ktopdby, ierr, zuo


def cup_dd_moisture(zd, hcd, hes_cup, qes_cup, q_cup, z_cup, dd_massentr,
                    dd_massdetr, jmin, ierr, gamma_cup, q, he, iloop, ktf):
    """cup_dd_moisture faithful. Returns (qcd, qrcd, pwd, pwev, bu, ierr)."""
    kte = len(zd) - 1
    qcd = _z(kte); qrcd = _z(kte); pwd = _z(kte)
    pwev = 0.0; bu = 0.0
    if ierr != 0:
        return qcd, qrcd, pwd, pwev, bu, ierr
    k = jmin
    dz = z_cup[k + 1] - z_cup[k]
    qcd[k] = q_cup[k]
    dh = hcd[k] - hes_cup[k]
    if dh < 0:
        qrcd[k] = qes_cup[k] + (1.0 / XLV) * (gamma_cup[k] / (1.0 + gamma_cup[k])) * dh
    else:
        qrcd[k] = qes_cup[k]
    pwd[jmin] = zd[jmin] * min(0.0, qcd[k] - qrcd[k])
    qcd[k] = qrcd[k]
    pwev = pwev + pwd[jmin]
    bu = dz * dh
    for ki in range(jmin - 1, 0, -1):
        dz = z_cup[ki + 1] - z_cup[ki]
        denom = zd[ki + 1] - 0.5 * dd_massdetr[ki] + dd_massentr[ki]
        if denom < 1.0e-8:
            ierr = 51
            break
        qcd[ki] = (qcd[ki + 1] * zd[ki + 1] - 0.5 * dd_massdetr[ki] * qcd[ki + 1]
                   + dd_massentr[ki] * q[ki]) / denom
        dh = hcd[ki] - hes_cup[ki]
        bu = bu + dz * dh
        qrcd[ki] = qes_cup[ki] + (1.0 / XLV) * (gamma_cup[ki] / (1.0 + gamma_cup[ki])) * dh
        dqeva = qcd[ki] - qrcd[ki]
        if dqeva > 0.0:
            dqeva = 0.0
            qrcd[ki] = qcd[ki]
        pwd[ki] = zd[ki] * dqeva
        qcd[ki] = qrcd[ki]
        pwev = pwev + pwd[ki]
    if pwev == 0.0 and iloop == 1:
        ierr = 7
    if bu >= 0.0 and iloop == 1:
        ierr = 7
    return qcd, qrcd, pwd, pwev, bu, ierr


def cup_up_moisture(name, ierr, z_cup, p_cup, kbcon, ktop, dby, xland1, q,
                    gamma_cup, zu, qes_cup, k22, qe_cup, zqexec, ccn, rho, c1d,
                    t, up_massentr, up_massdetr, ktf):
    """cup_up_moisture faithful (autoconv=1 branch). Returns qc,qrc,pw,pwav,clw_all,psum,psumh,ierr."""
    kte = len(z_cup) - 1
    qc = _z(kte); qrc = _z(kte); pw = _z(kte); clw_all = _z(kte)
    pwav = 0.0; psum = 0.0; psumh = 0.0
    iall = 0
    if ierr != 0:
        return qc, qrc, pw, pwav, clw_all, psum, psumh, ierr
    for k in range(1, ktf + 1):
        qc[k] = qe_cup[k]
    start_level = k22
    qaver = get_cloud_bc(qe_cup, k22)
    qc[start_level] = qaver
    for k in range(1, start_level):
        qc[k] = qe_cup[k]
    # below LFC
    for k in range(k22 + 1, kbcon + 1):
        c0 = 0.004
        if t[k] < 273.15:
            c0 = c0 * math.exp(0.07 * (t[k] - 273.15))
        qc[k] = (qc[k - 1] * zu[k - 1] - 0.5 * up_massdetr[k - 1] * qc[k - 1]
                 + up_massentr[k - 1] * q[k - 1]) / (zu[k - 1] - 0.5 * up_massdetr[k - 1] + up_massentr[k - 1])
        qrch = qes_cup[k] + (1.0 / XLV) * (gamma_cup[k] / (1.0 + gamma_cup[k])) * dby[k]
        if k < kbcon:
            qrch = qc[k]
        if qc[k] > qrch:
            dz = z_cup[k] - z_cup[k - 1]
            qrc[k] = (qc[k] - qrch) / (1.0 + c0 * dz)
            pw[k] = c0 * dz * qrc[k] * zu[k]
            qc[k] = qrch + qrc[k]
            clw_all[k] = qrc[k]
    # the rest
    for k in range(kbcon + 1, ktop + 1):
        c0 = 0.004
        if t[k] < 273.15:
            c0 = c0 * math.exp(0.07 * (t[k] - 273.15))
        denom = zu[k - 1] - 0.5 * up_massdetr[k - 1] + up_massentr[k - 1]
        if denom < 1.0e-8:
            ierr = 51
            break
        dz = z_cup[k] - z_cup[k - 1]
        qrch = qes_cup[k] + (1.0 / XLV) * (gamma_cup[k] / (1.0 + gamma_cup[k])) * dby[k]
        qc[k] = (qc[k - 1] * zu[k - 1] - 0.5 * up_massdetr[k - 1] * qc[k - 1]
                 + up_massentr[k - 1] * q[k - 1]) / denom
        if qc[k] <= qrch:
            qc[k] = qrch
        clw_all[k] = max(0.0, qc[k] - qrch)
        qrc[k] = max(0.0, (qc[k] - qrch))
        # autoconv==1 -> ELSE branch (not berry)
        if iall == 1:
            qrc[k] = 0.0
            pw[k] = (qc[k] - qrch) * zu[k]
            if pw[k] < 0.0:
                pw[k] = 0.0
        else:
            qrc[k] = (qc[k] - qrch) / (1.0 + (c1d[k] + c0) * dz)
            pw[k] = c0 * dz * qrc[k] * zu[k]
            if qrc[k] < 0:
                qrc[k] = 0.0
                pw[k] = 0.0
        qc[k] = qrc[k] + qrch
        pwav = pwav + pw[k]
        psum = psum + clw_all[k] * zu[k] * dz
    # do not include liquid/ice in qc
    for k in range(k22 + 1, ktop + 1):
        qc[k] = qc[k] - qrc[k]
    return qc, qrc, pw, pwav, clw_all, psum, psumh, ierr


def cup_dd_edt(us, vs, z, ktop, kbcon, p, pwav, pw, ccn, pwev, edtmax, edtmin,
               psum2, psumh, rho, ierr, ktf):
    """cup_dd_edt faithful (aeroevap=1 -> the aeroevap>1 branch is skipped)."""
    edt = 0.0
    edtc = 0.0
    if ierr != 0:
        return edt, edtc
    vws = 0.0; sdp = 0.0; vshear = 0.0
    for kk in range(1, ktf):
        if kk <= min(ktop, ktf) and kk >= kbcon:
            vws += (abs((us[kk + 1] - us[kk]) / (z[kk + 1] - z[kk]))
                    + abs((vs[kk + 1] - vs[kk]) / (z[kk + 1] - z[kk]))) * (p[kk] - p[kk + 1])
            sdp += p[kk] - p[kk + 1]
        if kk == ktf - 1:
            vshear = 1.0e3 * vws / sdp
    pef = (1.591 - 0.639 * vshear + 0.0953 * (vshear ** 2) - 0.00496 * (vshear ** 3))
    if pef > 0.9:
        pef = 0.9
    if pef < 0.1:
        pef = 0.1
    zkbc = z[kbcon] * 3.281e-3
    prezk = 0.02
    if zkbc > 3.0:
        prezk = (0.96729352 + zkbc * (-0.70034167 + zkbc * (0.162179896 + zkbc
                 * (-1.2569798e-2 + zkbc * (4.2772e-4 - zkbc * 5.44e-6)))))
    if zkbc > 25:
        prezk = 2.4
    pefb = 1.0 / (1.0 + prezk)
    if pefb > 0.9:
        pefb = 0.9
    if pefb < 0.1:
        pefb = 0.1
    edt = 1.0 - 0.5 * (pefb + pef)
    # aeroevap=1 -> if(aeroevap.gt.1) block NOT taken
    einc = 0.2 * edt
    edtc = edt - einc
    edtc = -edtc * pwav / pwev
    if edtc > edtmax:
        edtc = edtmax
    if edtc < edtmin:
        edtc = edtmin
    return edt, edtc


def cup_forcing_ens_3d(closure_n, xland, aa0, aa1, xaa0, mbdt, dtime, ierr,
                       ierr2, ierr3, mconv, p_cup, ktop, omeg, zd, k22, zu,
                       pr_ens, edt, kbcon, ichoice, imid, axx, tau_ecmwf,
                       aa1_bl, dicycle, ktf):
    """cup_forcing_ens_3d faithful (rand_clos=0). Returns (xf_ens, closure_n, xf_dicycle)."""
    kte = len(p_cup) - 1
    xf_ens = np.zeros(MAXENS3 + 1)  # 1-based 1..16
    xf_dicycle = 0.0
    ens_adj = 1.0
    xff_dicycle = 0.0
    if ierr != 0:
        if ierr != 20:
            return xf_ens, closure_n, 0.0
        return xf_ens, closure_n, 0.0
    kloc = _maxloc1(zu, kte)
    a_ave = axx
    a_ave = max(0.0, a_ave)
    a_ave = min(a_ave, aa1)
    a_ave = max(0.0, a_ave)
    xff_ens3 = np.zeros(MAXENS3 + 1)
    xff0 = (aa1 - aa0) / dtime
    xff_ens3[1] = max(0.0, (aa1 - aa0) / dtime)
    xff_ens3[2] = max(0.0, (aa1 - aa0) / dtime)
    xff_ens3[3] = max(0.0, (aa1 - aa0) / dtime)
    xff_ens3[16] = max(0.0, (aa1 - aa0) / dtime)
    # omega-based (4,5,6,14)
    xomg = 0.0
    kk = 0
    for k in range(kbcon - 1, kbcon + 1 + 1):
        if zu[k] > 0.0:
            xomg = xomg - omeg[k] / 9.81 / max(0.5, (1.0 - edt * zd[k] / zu[k]))
            kk = kk + 1
    if kk > 0:
        xff_ens3[4] = xomg / float(kk)
    xff_ens3[4] = BETAJB * xff_ens3[4]
    xff_ens3[5] = xff_ens3[4]
    xff_ens3[6] = xff_ens3[4]
    if xff_ens3[4] < 0.0:
        xff_ens3[4] = 0.0
    if xff_ens3[5] < 0.0:
        xff_ens3[5] = 0.0
    if xff_ens3[6] < 0.0:
        xff_ens3[6] = 0.0
    xff_ens3[14] = BETAJB * xff_ens3[4]
    # mconv-based (7,8,9,15)
    den = max(0.5, (1.0 - edt * zd[kbcon] / zu[kloc]))
    xff_ens3[7] = mconv / den
    xff_ens3[8] = mconv / den
    xff_ens3[9] = mconv / den
    xff_ens3[15] = mconv / den
    # tau-based (10,11,12,13)
    xff_ens3[10] = aa1 / tau_ecmwf
    xff_ens3[11] = aa1 / tau_ecmwf
    xff_ens3[12] = aa1 / tau_ecmwf
    xff_ens3[13] = aa1 / tau_ecmwf
    if dicycle == 1:
        xff_dicycle = max(0.0, aa1_bl / tau_ecmwf)
    if ichoice == 0:
        if xff0 < 0.0:
            xff_ens3[1] = 0.0
            xff_ens3[2] = 0.0
            xff_ens3[3] = 0.0
            xff_ens3[10] = 0.0
            xff_ens3[11] = 0.0
            xff_ens3[12] = 0.0
            xff_ens3[13] = 0.0
            xff_ens3[16] = 0.0
            closure_n = 12.0
    xk = (xaa0 - aa1) / mbdt
    if xk <= 0.0 and xk > -0.01 * mbdt:
        xk = -0.01 * mbdt
    if xk > 0.0 and xk < 1.0e-2:
        xk = 1.0e-2
    # over water adjustment (xland<0.1) skipped for land cases; replicate anyway
    if xland < 0.1:
        if ierr2 > 0 or ierr3 > 0:
            for n in range(1, MAXENS3 + 1):
                xff_ens3[n] = ens_adj * xff_ens3[n]
            xff_dicycle = ens_adj * xff_dicycle
    # stability closures 1,2,3,16
    if xk < 0.0:
        if xff_ens3[1] > 0:
            xf_ens[1] = max(0.0, -xff_ens3[1] / xk)
        if xff_ens3[2] > 0:
            xf_ens[2] = max(0.0, -xff_ens3[2] / xk)
        if xff_ens3[3] > 0:
            xf_ens[3] = max(0.0, -xff_ens3[3] / xk)
        if xff_ens3[16] > 0:
            xf_ens[16] = max(0.0, -xff_ens3[16] / xk)
    else:
        xff_ens3[1] = 0
        xff_ens3[2] = 0
        xff_ens3[3] = 0
        xff_ens3[16] = 0
    xf_ens[4] = max(0.0, xff_ens3[4])
    xf_ens[5] = max(0.0, xff_ens3[5])
    xf_ens[6] = max(0.0, xff_ens3[6])
    xf_ens[14] = max(0.0, xff_ens3[14])
    a1 = max(1.0e-5, pr_ens[7]); xf_ens[7] = max(0.0, xff_ens3[7] / a1)
    a1 = max(1.0e-5, pr_ens[8]); xf_ens[8] = max(0.0, xff_ens3[8] / a1)
    a1 = max(1.0e-5, pr_ens[9]); xf_ens[9] = max(0.0, xff_ens3[9] / a1)
    a1 = max(1.0e-3, pr_ens[15]); xf_ens[15] = max(0.0, xff_ens3[15] / a1)
    if xk < 0.0:
        xf_ens[10] = max(0.0, -xff_ens3[10] / xk)
        xf_ens[11] = max(0.0, -xff_ens3[11] / xk)
        xf_ens[12] = max(0.0, -xff_ens3[12] / xk)
        xf_ens[13] = max(0.0, -xff_ens3[13] / xk)
    else:
        xf_ens[10] = 0.0
        xf_ens[11] = 0.0
        xf_ens[12] = 0.0
        xf_ens[13] = 0.0
    if xk < 0.0:
        xf_dicycle = max(0.0, -xff_dicycle / xk)
    else:
        xf_dicycle = 0.0
    if ichoice >= 1:
        for n in range(1, MAXENS3 + 1):
            xf_ens[n] = xf_ens[ichoice]
    return xf_ens, closure_n, xf_dicycle


def cup_output_ens_3d(xff_mid, xf_ens, ierr, dellat, dellaq, dellaqc, zu, pw,
                      ktop, edt, pwd, name, ierr2, ierr3, p_cup, pr_ens, sig,
                      closure_n, xland1, xmbm_in, xmbs_in, ichoice, imid,
                      dicycle, xf_dicycle, ktf):
    """cup_output_ens_3d faithful (imid=0 deep branch). Returns (outtem,outq,outqc,pre,xmb,ierr)."""
    kte = len(p_cup) - 1
    outtem = _z(kte); outq = _z(kte); outqc = _z(kte)
    pre = 0.0; xmb = 0.0
    if ierr != 0:
        return outtem, outq, outqc, pre, xmb, ierr
    # zero xf_ens for non-precip ensembles
    for n in range(1, MAXENS3 + 1):
        if pr_ens[n] <= 0.0:
            xf_ens[n] = 0.0
    # deep (imid==0)
    xmb_ave = 0.0
    k = 0
    for n in range(1, MAXENS3 + 1):
        k = k + 1
        xmb_ave = xmb_ave + xf_ens[n]
    xmb_ave = xmb_ave / float(k)
    if dicycle == 2:
        xmb_ave = xmb_ave - max(0.0, xmbm_in, xmbs_in)
        xmb_ave = max(0.0, xmb_ave)
    elif dicycle == 1:
        xmb_ave = min(xmb_ave, xmb_ave - xf_dicycle)
        xmb_ave = max(0.0, xmb_ave)
    clos_wei = 16.0 / max(1.0, closure_n)
    xmb_ave = min(xmb_ave, 100.0)
    xmb = clos_wei * sig * xmb_ave
    if xmb < 1.0e-16:
        ierr = 19
        return outtem, outq, outqc, pre, xmb, ierr
    pwtot = 0.0
    for k in range(1, ktop + 1):
        pwtot = pwtot + pw[k]
    for k in range(1, ktop + 1):
        dp = 100.0 * (p_cup[k] - p_cup[k + 1]) / G
        dtt = dellat[k]
        dtq = dellaq[k]
        dtpwd = -pwd[k] * edt
        dtqc = dellaqc[k] * dp - dtpwd
        if dtqc < 0.0:
            dtpwd = dtpwd - dellaqc[k] * dp
            dtqc = 0.0
        else:
            dtpwd = 0.0
            dtqc = dtqc / dp
        outtem[k] = xmb * dtt
        outq[k] = xmb * dtq
        outqc[k] = xmb * dtqc
        pre = pre - xmb * dtpwd
    pre = -pre + xmb * pwtot
    return outtem, outq, outqc, pre, xmb, ierr


def neg_check(name, dt, q, outq, outt, outu, outv, outqc, pret, ktf):
    """neg_check faithful (only the first 'heating rate' block runs; routine RETURNs)."""
    thresh = 300.01
    names = 1.0
    if name == 'shallow':
        thresh = 148.01
        names = 2.0
    qmemf = 1.0
    for k in range(1, ktf + 1):
        qmem = outt[k] * 86400.0
        if qmem > thresh:
            qmem2 = thresh / qmem
            qmemf = min(qmemf, qmem2)
        if qmem < -0.5 * thresh * names:
            qmem2 = -0.5 * names * thresh / qmem
            qmemf = min(qmemf, qmem2)
    for k in range(1, ktf + 1):
        outq[k] = outq[k] * qmemf
        outt[k] = outt[k] * qmemf
        outu[k] = outu[k] * qmemf
        outv[k] = outv[k] * qmemf
        outqc[k] = outqc[k] * qmemf
    pret = pret * qmemf
    return pret


def cup_gf(dicycle, ichoice, ccn, dtime, imid, kpbl, dhdt, xland, zo, t, q, z1,
           tn, qo, po, psur, us, vs, rho, hfx, qfx, dx, mconv, omeg, csum, ktf):
    """Faithful single-column port of CUP_gf (deep). i=1 column.

    Returns dict with outt, outq, outqc, outu, outv, pre, kbcon, ktop, k22,
    cupclw, ierr, xmb_out.
    """
    kte = ktf  # single column: kte==ktf used as KX
    # local arrays 1-based
    out = {'outt': _z(kte), 'outq': _z(kte), 'outqc': _z(kte), 'outu': _z(kte),
           'outv': _z(kte), 'cupclw': _z(kte), 'pre': 0.0, 'kbcon': 0, 'ktop': 0,
           'k22': 0, 'ierr': 0, 'xmb_out': 0.0}
    ierr = 0
    flux_tun = FLUXTUNE
    pmin = 150.0
    elocp = XLV / CP
    pgcon = 0.0
    lambau = 2.0
    # zws/ztexec/zqexec
    buo_flux = (hfx / CP + 0.608 * t[1] * qfx / XLV) / rho[1]
    ztexec = 0.0; zqexec = 0.0; zws = 0.0
    zws_tmp = max(0.0, flux_tun * 0.41 * buo_flux * zo[2] * 9.81 / t[1])
    if zws_tmp > np.finfo(np.float64).tiny:
        zws_tmp = 1.2 * zws_tmp ** 0.3333
        ztexec = max(flux_tun * hfx / (rho[1] * zws_tmp * CP), 0.0)
        zqexec = max(flux_tun * qfx / XLV / (rho[1] * zws_tmp), 0.0)
    zws = max(0.0, flux_tun * 0.41 * buo_flux * zo[kpbl] * 9.81 / t[kpbl])
    zws = 1.2 * zws ** 0.3333
    zws = zws * rho[kpbl]
    cap_maxs = 75.0
    edto = 0.0
    closure_n = 16.0
    cap_max = cap_maxs
    cap_max_increment = 20.0
    xland1 = int(xland + 0.0001)
    if xland > 1.5 or xland < 0.5:
        xland1 = 0
        cap_max_increment = 20.0
    else:
        if ztexec > 0.0:
            cap_max = cap_max + 25.0
        if ztexec < 0.0:
            cap_max = cap_max - 25.0
    if USE_EXCESS == 0:
        ztexec = 0.0; zqexec = 0.0
    # entrainment / sig
    c1d = _z(kte)
    entr_rate = 7.0e-5 - min(20.0, float(csum)) * 3.0e-6
    if xland1 == 0:
        entr_rate = 7.0e-5
    radius = 0.2 / entr_rate
    frh = min(1.0, 3.14 * radius * radius / dx / dx)
    if frh > FRH_THRESH:
        frh = FRH_THRESH
        radius = math.sqrt(frh * dx * dx / 3.14)
        entr_rate = 0.2 / radius
    sig = (1.0 - frh) ** 2
    sig_thresh = (1.0 - FRH_THRESH) ** 2
    cnvwt = _z(kte); zuo = _z(kte); zdo = _z(kte)
    z = _z(kte); xz = _z(kte); cupclw = _z(kte); cd = _z(kte); cdd = _z(kte)
    hcdo = _z(kte); qrcdo = _z(kte); dellaqc = _z(kte)
    for k in range(1, ktf + 1):
        z[k] = zo[k]; xz[k] = zo[k]
        cd[k] = 1.0e-9
        cdd[k] = 1.0e-9
    edtmax = 1.0; edtmin = 0.1
    depth_min = 1000.0
    kbmax = 1; aa0 = 0.0; aa1 = 0.0; edt = 0.0
    kstabm = ktf - 1
    ierr2 = 0; ierr3 = 0
    zkbmax = 4000.0
    zcutdown = 4000.0
    z_detr = 1000.0
    xf_ens = np.zeros(MAXENS3 + 1)
    pr_ens = np.zeros(MAXENS3 + 1)
    # cup_env (non-forced and forced)
    z, qes, he, hes = cup_env(z, t, q, po, z1, psur, ierr, -1, ktf)
    zo, qeso, heo, heso = cup_env(zo, tn, qo, po, z1, psur, ierr, -1, ktf)
    # env clev
    (qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup) = \
        cup_env_clev(t, qes, q, he, hes, z, po, psur, z1, ierr, ktf)
    (qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup) = \
        cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, ierr, ktf)
    u_cup = _z(kte); v_cup = _z(kte)
    if ierr == 0:
        u_cup[1] = us[1]; v_cup[1] = vs[1]
        for k in range(2, ktf + 1):
            u_cup[k] = 0.5 * (us[k - 1] + us[k])
            v_cup[k] = 0.5 * (vs[k - 1] + vs[k])
    kdet = 0
    if ierr == 0:
        for k in range(1, ktf + 1):
            if zo_cup[k] > zkbmax + z1:
                kbmax = k
                break
        for k in range(1, ktf + 1):
            if zo_cup[k] > z_detr + z1:
                kdet = k
                break
    # K22 = level of highest moist static energy
    k22 = 0; ktop = 0; kbcon = 0
    start_k22 = 2
    if ierr == 0:
        k22 = _argmax_range(heo_cup, start_k22, kbmax + 2) 
        if k22 >= kbmax:
            ierr = 2; ktop = 0; k22 = 0; kbcon = 0
    # KBCON
    hkb = 0.0; hkbo = 0.0
    if ierr == 0:
        x_add = XLV * zqexec + CP * ztexec
        hkb = get_cloud_bc(he_cup, k22, x_add)
        hkbo = get_cloud_bc(heo_cup, k22, x_add)
    iloop = 1
    k22, kbcon, hkbo, ierr = cup_kbcon(cap_max_increment, iloop, k22, kbcon,
                                       heo_cup, heso_cup, hkbo, ierr, kbmax,
                                       po_cup, cap_max, ztexec, zqexec, z_cup,
                                       entr_rate, heo, imid, ktf)
    # cup_minimi for kstabi
    kstabi = cup_minimi(heso_cup, kbcon, kstabm, ierr, ktf) if ierr == 0 else kbcon
    pmin_lev = 0
    entr_rate_2d = _z(kte)
    if ierr == 0:
        frh = min(qo_cup[kbcon] / qeso_cup[kbcon], 1.0)
        if frh >= RH_THRESH and sig <= sig_thresh:
            ierr = 231
        else:
            x_add = 0.0
            for k in range(kbcon + 1, ktf + 1):
                if po[kbcon] - po[k] > pmin + x_add:
                    pmin_lev = k
                    break
            start_level = k22
            x_add = XLV * zqexec + CP * ztexec
            hkb = get_cloud_bc(he_cup, k22, x_add)
    if kstabi < kbcon:
        kbcon = 1; ierr = 42
    for k in range(1, ktf + 1):
        entr_rate_2d[k] = entr_rate
    ktopdby = 0
    start_level = k22 if k22 > 0 else kte
    if ierr == 0:
        kbcon = max(2, kbcon)
        for k in range(1, ktf + 1):
            frh = min(qo_cup[k] / qeso_cup[k], 1.0)
            entr_rate_2d[k] = entr_rate * (1.3 - frh)
        start_level = k22
    # rates_up_pdf (deep)
    kbcon, ktop, ktopdby, ierr, zuo = rates_up_pdf(
        'deep', ktop, ierr, po_cup, entr_rate_2d, hkbo, heo, heso_cup, zo_cup,
        xland1, kstabi, k22, kbcon, zuo, kbcon, ktopdby, csum, pmin_lev,
        1, kte, ktf)
    zu = _z(kte); xzu = _z(kte)
    if ierr == 0:
        if k22 > 1:
            for k in range(1, k22):
                zuo[k] = 0.0; zu[k] = 0.0; xzu[k] = 0.0
        for k in range(k22, ktop + 1):
            xzu[k] = zuo[k]; zu[k] = zuo[k]
        for k in range(ktop + 1, kte + 1):
            zuo[k] = 0.0; zu[k] = 0.0; xzu[k] = 0.0
    # lateral massflux
    (up_massentro, up_massdetro, up_massentr, up_massdetr,
     up_massentru, up_massdetru) = get_lateral_massflux(
        zo_cup, zuo, cd, entr_rate_2d, ktop if ierr == 0 else 0, kbcon, k22,
        lambau, ktf, with_u=True)
    # updraft hc/hco
    uc = _z(kte); vc = _z(kte); hc = _z(kte); dby = _z(kte); hco = _z(kte); dbyo = _z(kte)
    if ierr == 0:
        for k in range(1, start_level + 1):
            uc[k] = u_cup[k]; vc[k] = v_cup[k]
        for k in range(1, start_level):
            hc[k] = he_cup[k]; hco[k] = heo_cup[k]
        hc[start_level] = hkb
        hco[start_level] = hkbo
    dbyt = _z(kte); ktopkeep = 0
    if ierr == 0:
        ktopkeep = ktop
        for k in range(start_level + 1, ktop + 1):
            denom = zuo[k - 1] - 0.5 * up_massdetro[k - 1] + up_massentro[k - 1]
            if denom < 1.0e-8:
                ierr = 51
                break
            hc[k] = (hc[k - 1] * zu[k - 1] - 0.5 * up_massdetr[k - 1] * hc[k - 1]
                     + up_massentr[k - 1] * he[k - 1]) / (zu[k - 1] - 0.5 * up_massdetr[k - 1] + up_massentr[k - 1])
            uc[k] = (uc[k - 1] * zu[k - 1] - 0.5 * up_massdetru[k - 1] * uc[k - 1]
                     + up_massentru[k - 1] * us[k - 1]
                     - pgcon * 0.5 * (zu[k] + zu[k - 1]) * (u_cup[k] - u_cup[k - 1])) / \
                    (zu[k - 1] - 0.5 * up_massdetru[k - 1] + up_massentru[k - 1])
            vc[k] = (vc[k - 1] * zu[k - 1] - 0.5 * up_massdetru[k - 1] * vc[k - 1]
                     + up_massentru[k - 1] * vs[k - 1]
                     - pgcon * 0.5 * (zu[k] + zu[k - 1]) * (v_cup[k] - v_cup[k - 1])) / \
                    (zu[k - 1] - 0.5 * up_massdetru[k - 1] + up_massentru[k - 1])
            dby[k] = hc[k] - hes_cup[k]
            hco[k] = (hco[k - 1] * zuo[k - 1] - 0.5 * up_massdetro[k - 1] * hco[k - 1]
                      + up_massentro[k - 1] * heo[k - 1]) / (zuo[k - 1] - 0.5 * up_massdetro[k - 1] + up_massentro[k - 1])
            dbyo[k] = hco[k] - heso_cup[k]
            dz = zo_cup[k + 1] - zo_cup[k]
            dbyt[k] = dbyt[k - 1] + dbyo[k] * dz
        for k in range(ktop - 1, kbcon - 1, -1):
            if dbyo[k] > 0.0:
                ktopkeep = k + 1
                break
        ktop = ktopkeep
    if ierr == 0:
        for k in range(ktop + 1, ktf + 1):
            hc[k] = hes_cup[k]; uc[k] = u_cup[k]; vc[k] = v_cup[k]
            hco[k] = heso_cup[k]; dby[k] = 0.0; dbyo[k] = 0.0
            zu[k] = 0.0; zuo[k] = 0.0; cd[k] = 0.0
            entr_rate_2d[k] = 0.0
            up_massentr[k] = 0.0; up_massdetr[k] = 0.0
            up_massentro[k] = 0.0; up_massdetro[k] = 0.0
    if ierr == 0:
        if ktop < kbcon + 2:
            ierr = 5; ktop = 0
    # downdraft originating level
    kzdown = 0
    if ierr == 0:
        zktop = (zo_cup[ktop] - z1) * 0.6
        zktop = min(zktop + z1, zcutdown + z1)
        for k in range(1, ktf + 1):
            if zo_cup[k] > zktop:
                kzdown = k
                kzdown = min(kzdown, kstabi - 1)
                break
    jmin = cup_minimi(heso_cup, k22, kzdown, ierr, ktf) if ierr == 0 else 0
    if ierr == 0:
        jmini = jmin
        keep_going = True
        while keep_going:
            keep_going = False
            if jmini - 1 < kdet:
                kdet = jmini - 1
            if jmini >= ktop - 1:
                jmini = ktop - 2
            ki = jmini
            hcdo[ki] = heso_cup[ki]
            dz = zo_cup[ki + 1] - zo_cup[ki]
            dh = 0.0
            for k in range(ki - 1, 0, -1):
                hcdo[k] = heso_cup[jmini]
                dz = zo_cup[k + 1] - zo_cup[k]
                dh = dh + dz * (hcdo[k] - heso_cup[k])
                if dh > 0.0:
                    jmini = jmini - 1
                    if jmini > 5:
                        keep_going = True
                    else:
                        ierr = 9
                        break
        jmin = jmini
        if jmini <= 5:
            ierr = 4
    if ierr == 0:
        if jmin - 1 < kdet:
            kdet = jmin - 1
        if -zo_cup[kbcon] + zo_cup[ktop] < depth_min:
            ierr = 6
    # downdraft mass flux profile
    dd_massentro = _z(kte); dd_massdetro = _z(kte)
    dd_massentru = _z(kte); dd_massdetru = _z(kte)
    mentrd_rate_2d = _z(kte); ucd = _z(kte); vcd = _z(kte); dbydo = _z(kte)
    for k in range(1, ktf + 1):
        hcdo[k] = heso_cup[k]; ucd[k] = u_cup[k]; vcd[k] = v_cup[k]
        mentrd_rate_2d[k] = entr_rate
    bud = 0.0
    beta = max(0.02, 0.05 - float(csum) * 0.0015)
    if imid == 0 and xland1 == 0:
        edtmax = max(0.1, 0.4 - float(csum) * 0.015)
    if ierr == 0:
        for k in range(1, jmin + 1):
            cdd[k] = 1.0e-9
        cdd[jmin] = 0.0
        zdo = get_zu_zd_pdf_fim(0, po_cup, 0.0, xland1, "DOWN", ierr, kdet,
                                jmin, 1, kte, ktf, beta, kpbl, csum, pmin_lev)
        if zdo[jmin] < 1.0e-8:
            zdo[jmin] = 0.0
            jmin = jmin - 1
            if zdo[jmin] < 1.0e-8:
                ierr = 876
        if ierr == 0:
            mlzd = _maxloc1(zdo, kte)
            for ki in range(jmin, mlzd - 1, -1):
                dzo = zo_cup[ki + 1] - zo_cup[ki]
                dd_massdetro[ki] = cdd[ki] * dzo * zdo[ki + 1]
                dd_massentro[ki] = zdo[ki] - zdo[ki + 1] + dd_massdetro[ki]
                if dd_massentro[ki] < 0.0:
                    dd_massentro[ki] = 0.0
                    dd_massdetro[ki] = zdo[ki + 1] - zdo[ki]
                    if zdo[ki + 1] > 0.0:
                        cdd[ki] = dd_massdetro[ki] / (dzo * zdo[ki + 1])
                if zdo[ki + 1] > 0.0:
                    mentrd_rate_2d[ki] = dd_massentro[ki] / (dzo * zdo[ki + 1])
            mentrd_rate_2d[1] = 0.0
            for ki in range(mlzd - 1, 0, -1):
                dzo = zo_cup[ki + 1] - zo_cup[ki]
                dd_massentro[ki] = mentrd_rate_2d[ki] * dzo * zdo[ki + 1]
                dd_massdetro[ki] = zdo[ki + 1] + dd_massentro[ki] - zdo[ki]
                if dd_massdetro[ki] < 0.0:
                    dd_massdetro[ki] = 0.0
                    dd_massentro[ki] = zdo[ki] - zdo[ki + 1]
                    if zdo[ki + 1] > 0.0:
                        mentrd_rate_2d[ki] = dd_massentro[ki] / (dzo * zdo[ki + 1])
                if zdo[ki + 1] > 0.0:
                    cdd[ki] = dd_massdetro[ki] / (dzo * zdo[ki + 1])
            # c1d profile (overwritten to c1)
            for k in range(kbcon + 1, ktop):
                c1d[k] = C1
            for k in range(2, jmin + 1 + 1):
                dd_massentru[k - 1] = dd_massentro[k - 1] + lambau * dd_massdetro[k - 1]
                dd_massdetru[k - 1] = dd_massdetro[k - 1] + lambau * dd_massdetro[k - 1]
            dbydo[jmin] = hcdo[jmin] - heso_cup[jmin]
            bud = dbydo[jmin] * (zo_cup[jmin + 1] - zo_cup[jmin])
            for ki in range(jmin, 0, -1):
                dzo = zo_cup[ki + 1] - zo_cup[ki]
                h_entr = 0.5 * (heo[ki] + 0.5 * (hco[ki] + hco[ki + 1]))
                ucd[ki] = (ucd[ki + 1] * zdo[ki + 1] - 0.5 * dd_massdetru[ki] * ucd[ki + 1]
                           + dd_massentru[ki] * us[ki] - pgcon * zdo[ki + 1] * (us[ki + 1] - us[ki])) / \
                          (zdo[ki + 1] - 0.5 * dd_massdetru[ki] + dd_massentru[ki])
                vcd[ki] = (vcd[ki + 1] * zdo[ki + 1] - 0.5 * dd_massdetru[ki] * vcd[ki + 1]
                           + dd_massentru[ki] * vs[ki] - pgcon * zdo[ki + 1] * (vs[ki + 1] - vs[ki])) / \
                          (zdo[ki + 1] - 0.5 * dd_massdetru[ki] + dd_massentru[ki])
                hcdo[ki] = (hcdo[ki + 1] * zdo[ki + 1] - 0.5 * dd_massdetro[ki] * hcdo[ki + 1]
                            + dd_massentro[ki] * h_entr) / (zdo[ki + 1] - 0.5 * dd_massdetro[ki] + dd_massentro[ki])
                dbydo[ki] = hcdo[ki] - heso_cup[ki]
                bud = bud + dbydo[ki] * dzo
        if bud > 0:
            ierr = 7
    # downdraft moisture
    qcdo, qrcdo, pwdo, pwevo, bu, ierr = cup_dd_moisture(
        zdo, hcdo, heso_cup, qeso_cup, qo_cup, zo_cup, dd_massentro,
        dd_massdetro, jmin, ierr, gammao_cup, qo, heo, 1, ktf)
    # updraft moisture
    qco, qrco, pwo, pwavo, clw_all, psum, psumh, ierr = cup_up_moisture(
        'deep', ierr, zo_cup, p_cup, kbcon, ktop, dbyo, xland1, qo, gammao_cup,
        zuo, qeso_cup, k22, qo_cup, zqexec, ccn, rho, c1d, tn_cup,
        up_massentr, up_massdetr, ktf)
    if ierr == 0:
        for k in range(2, ktop + 1):
            dp = 100.0 * (po_cup[1] - po_cup[2])
            cupclw[k] = qrco[k]
            cnvwt[k] = zuo[k] * cupclw[k] * G / dp
    # work functions
    aa0 = cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, ktf)
    aa1 = cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, ktf)
    if ierr == 0 and aa1 == 0.0:
        ierr = 17
    # dicycle closure (iversion=1 ecmwf)
    aa1_bl = 0.0; xf_dicycle = 0.0; tau_ecmwf = 0.0
    wmean = 0.0; tau_bl = 0.0
    if ierr == 0:
        wmean = 7.0
        tau_ecmwf = (zo_cup[ktopdby] - zo_cup[kbcon]) / wmean
        tau_ecmwf = tau_ecmwf * (1.0061 + 1.23e-2 * (dx / 1000.0))
    t_star = 4.0
    if dicycle == 1 and ierr == 0:
        if xland1 == 0:
            umean = 2.0 + math.sqrt(2.0 * (us[1] ** 2 + vs[1] ** 2 + us[kbcon] ** 2 + vs[kbcon] ** 2))
            tau_bl = (zo_cup[kbcon] - z1) / umean
        else:
            tau_bl = (zo_cup[ktopdby] - zo_cup[kbcon]) / wmean
        aa1_bl = cup_up_aa1bl_full(zo, t, tn, q, qo, dtime, kbcon, ierr, ktf)
        if zo_cup[kbcon] - z1 > zo[min(kte, kpbl + 1)]:
            aa1_bl = 0.0
        else:
            aa1_bl = max(0.0, aa1_bl / t_star * tau_bl)
    axx = aa1
    # edt
    edt, edtc = cup_dd_edt(us, vs, zo, ktop, kbcon, po, pwavo, pwo, ccn, pwevo,
                           edtmax, edtmin, psum, psumh, rho, ierr, ktf)
    if ierr == 0:
        edto = edtc
    # dellah / dellaq / dellat budget
    dellah = _z(kte); dellaq = _z(kte); dellat = _z(kte)
    dellu = _z(kte); dellv = _z(kte)
    if ierr == 0:
        dp = 100.0 * (po_cup[1] - po_cup[2])
        dellu[1] = PGCD * (edto * zdo[2] * ucd[2] - edto * zdo[2] * u_cup[2]) * G / dp
        dellv[1] = PGCD * (edto * zdo[2] * vcd[2] - edto * zdo[2] * v_cup[2]) * G / dp
        for k in range(2, ktop + 1):
            dp = 100.0 * (po_cup[k] - po_cup[k + 1])
            dellu[k] = (-(zuo[k + 1] * (uc[k + 1] - u_cup[k + 1]) - zuo[k] * (uc[k] - u_cup[k])) * G / dp
                        + (zdo[k + 1] * (ucd[k + 1] - u_cup[k + 1]) - zdo[k] * (ucd[k] - u_cup[k])) * G / dp * edto * PGCD)
            dellv[k] = (-(zuo[k + 1] * (vc[k + 1] - v_cup[k + 1]) - zuo[k] * (vc[k] - v_cup[k])) * G / dp
                        + (zdo[k + 1] * (vcd[k + 1] - v_cup[k + 1]) - zdo[k] * (vcd[k] - v_cup[k])) * G / dp * edto * PGCD)
    if ierr == 0:
        dp = 100.0 * (po_cup[1] - po_cup[2])
        dellah[1] = (edto * zdo[2] * hcdo[2] - edto * zdo[2] * heo_cup[2]) * G / dp
        dellaq[1] = (edto * zdo[2] * qcdo[2] - edto * zdo[2] * qo_cup[2]) * G / dp
        g_rain = 0.5 * (pwo[1] + pwo[2]) * G / dp
        e_dn = -0.5 * (pwdo[1] + pwdo[2]) * G / dp * edto
        dellaq[1] = dellaq[1] + e_dn - g_rain
        for k in range(2, ktop + 1):
            dp = 100.0 * (po_cup[k] - po_cup[k + 1])
            dellah[k] = (-(zuo[k + 1] * (hco[k + 1] - heo_cup[k + 1]) - zuo[k] * (hco[k] - heo_cup[k])) * G / dp
                         + (zdo[k + 1] * (hcdo[k + 1] - heo_cup[k + 1]) - zdo[k] * (hcdo[k] - heo_cup[k])) * G / dp * edto)
            detup = up_massdetro[k]
            dz = zo_cup[k] - zo_cup[k - 1]
            if k < ktop:
                dellaqc[k] = zuo[k] * c1d[k] * qrco[k] * dz / dp * G
            if k == ktop:
                dellaqc[k] = detup * 0.5 * (qrco[k + 1] + qrco[k]) * G / dp
            g_rain = 0.5 * (pwo[k] + pwo[k + 1]) * G / dp
            e_dn = -0.5 * (pwdo[k] + pwdo[k + 1]) * G / dp * edto
            c_up = dellaqc[k] + (zuo[k + 1] * qrco[k + 1] - zuo[k] * qrco[k]) * G / dp + g_rain
            dellaq[k] = (-(zuo[k + 1] * (qco[k + 1] - qo_cup[k + 1]) - zuo[k] * (qco[k] - qo_cup[k])) * G / dp
                         + (zdo[k + 1] * (qcdo[k + 1] - qo_cup[k + 1]) - zdo[k] * (qcdo[k] - qo_cup[k])) * G / dp * edto
                         - c_up + e_dn)
    # x-profiles for static control
    mbdt = 0.1
    xaa0_ens = 0.0
    xhe = _z(kte); xq = _z(kte); xt = _z(kte)
    if ierr == 0:
        for k in range(1, ktf + 1):
            xhe[k] = dellah[k] * mbdt + heo[k]
            xq[k] = max(1.0e-16, dellaq[k] * mbdt + qo[k])
            dellat[k] = (1.0 / CP) * (dellah[k] - XLV * dellaq[k])
            xt[k] = dellat[k] * mbdt + tn[k]
            xt[k] = max(190.0, xt[k])
        xhe[ktf] = heo[ktf]; xq[ktf] = qo[ktf]; xt[ktf] = tn[ktf]
    xz, xqes, xhe2, xhes = cup_env(xz, xt, xq, po, z1, psur, ierr, -1, ktf)
    (xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, _xp_cup, _xg_cup, xt_cup) = \
        cup_env_clev(xt, xqes, xq, xhe2, xhes, xz, po, psur, z1, ierr, ktf)
    xhc = _z(kte); xdby = _z(kte); xhkb = 0.0
    if ierr == 0:
        x_add = XLV * zqexec + CP * ztexec
        xhkb = get_cloud_bc(xhe_cup, k22, x_add)
        for k in range(1, start_level):
            xhc[k] = xhe_cup[k]
        xhc[start_level] = xhkb
        for k in range(start_level + 1, ktop + 1):
            xhc[k] = (xhc[k - 1] * xzu[k - 1] - 0.5 * up_massdetro[k - 1] * xhc[k - 1]
                      + up_massentro[k - 1] * xhe[k - 1]) / (xzu[k - 1] - 0.5 * up_massdetro[k - 1] + up_massentro[k - 1])
            xdby[k] = xhc[k] - xhes_cup[k]
        for k in range(ktop + 1, ktf + 1):
            xhc[k] = xhes_cup[k]; xdby[k] = 0.0
    xaa0 = cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon, ktop, ierr, ktf)
    if ierr == 0:
        xaa0_ens = xaa0
        for k in range(1, ktop + 1):
            for nens3 in range(1, MAXENS3 + 1):
                pr_ens[nens3] = pr_ens[nens3] + pwo[k] + edto * pwdo[k]
        if pr_ens[7] < 1.0e-6:
            ierr = 18
            for nens3 in range(1, MAXENS3 + 1):
                pr_ens[nens3] = 0.0
        for nens3 in range(1, MAXENS3 + 1):
            if pr_ens[nens3] < 1.0e-5:
                pr_ens[nens3] = 0.0
    # large scale forcing - ierr2/ierr3 via cup_maximi + cup_kbcon
    ierr2 = ierr; ierr3 = ierr; k22x = k22
    if ierr == 0:
        k22x = cup_maximi(heo_cup, 2, kbmax, ierr, ktf)
        kbconx = 0
        _k22x2, kbconx, _hkbo2, ierr2 = cup_kbcon(cap_max_increment, 2, k22x, kbconx,
                                                  heo_cup, heso_cup, hkbo, ierr2, kbmax,
                                                  po_cup, cap_max, ztexec, zqexec, z_cup,
                                                  entr_rate, heo, imid, ktf)
        _k22x3, kbconx3, _hkbo3, ierr3 = cup_kbcon(cap_max_increment, 3, k22x, kbconx,
                                                   heo_cup, heso_cup, hkbo, ierr3, kbmax,
                                                   po_cup, cap_max, ztexec, zqexec, z_cup,
                                                   entr_rate, heo, imid, ktf)
    # mconv recompute
    mconv_d = 0.0
    if ierr == 0:
        for k in range(1, ktop + 1):
            dq = qo_cup[k + 1] - qo_cup[k]
            mconv_d = mconv_d + omeg[k] * dq / G
    else:
        mconv_d = 0.0
    xf_ens, closure_n, xf_dicycle = cup_forcing_ens_3d(
        closure_n, xland1, aa0, aa1, xaa0_ens, mbdt, dtime, ierr, ierr2, ierr3,
        mconv_d, po_cup, ktop, omeg, zdo, k22, zuo, pr_ens, edto, kbcon,
        ichoice, imid, axx, tau_ecmwf if ierr == 0 else 1.0, aa1_bl, dicycle, ktf)
    # output ensemble
    xff_mid = np.zeros(3)
    outt, outq, outqc, pre, xmb, ierr = cup_output_ens_3d(
        xff_mid, xf_ens, ierr, dellat, dellaq, dellaqc, zuo, pwo, ktop, edto,
        pwdo, 'deep', ierr2, ierr3, po_cup, pr_ens, sig, closure_n, xland1,
        0.0, 0.0, ichoice, imid, dicycle, xf_dicycle, ktf)
    outu = _z(kte); outv = _z(kte)
    xmb_out = 0.0
    if ierr == 0 and pre > 0.0:
        pre = max(pre, 0.0)
        xmb_out = xmb
        for k in range(1, ktop + 1):
            outu[k] = dellu[k] * xmb
            outv[k] = dellv[k] * xmb
    elif ierr != 0 or pre == 0.0:
        ktop = 0
        outt = _z(kte); outq = _z(kte); outqc = _z(kte)
        outu = _z(kte); outv = _z(kte)
    # KE dissipation heating (from ECMWF)
    if ierr == 0:
        dts = 0.0; fpi = 0.0
        for k in range(1, ktop + 1):
            dp = (po_cup[k] - po_cup[k + 1]) * 100.0
            dts = dts - (outu[k] * us[k] + outv[k] * vs[k]) * dp / G
            fpi = fpi + math.sqrt(outu[k] * outu[k] + outv[k] * outv[k]) * dp
        if fpi > 0.0:
            for k in range(1, ktop + 1):
                fp = math.sqrt((outu[k] * outu[k] + outv[k] * outv[k])) / fpi
                outt[k] = outt[k] + fp * dts * G / CP
    out.update({'outt': outt, 'outq': outq, 'outqc': outqc, 'outu': outu,
                'outv': outv, 'cupclw': cupclw, 'pre': pre, 'kbcon': kbcon,
                'ktop': ktop, 'k22': k22, 'ierr': ierr, 'xmb_out': xmb_out})
    return out


def _argmax_range(arr, lo, hi):
    """Fortran maxloc(arr(lo:hi),1)+lo-1: first index of max in lo..hi (1-based)."""
    hi = min(hi, len(arr) - 1)
    best = lo
    bestv = arr[lo]
    for k in range(lo, hi + 1):
        if arr[k] > bestv:
            bestv = arr[k]
            best = k
    return best


def get_inversion_layers(p_cup, t_cup, z_cup, qo_cup, qeso_cup, kstart, kend,
                         ierr, kts, kte, ktf):
    """get_inversion_layers faithful. Returns (k_inv_layers 1-based, dtempdz)."""
    l_mid = 300.0; l_shal = 100.0
    k_inv = np.ones(kte + 1, dtype=np.int64)  # 1-based; init to 1
    dtempdz = _z(kte)
    if ierr != 0:
        return k_inv, dtempdz
    first_deriv = _z(kte + 4); sec_deriv = _z(kte + 4)
    kend_p3 = kend + 3
    for k in range(2, min(kend_p3 + 4, kte - 1) + 1):
        first_deriv[k] = (t_cup[k + 1] - t_cup[k - 1]) / (z_cup[k + 1] - z_cup[k - 1])
        dtempdz[k] = first_deriv[k]
    for k in range(3, min(kend_p3 + 3, kte - 1) + 1):
        sec_deriv[k] = (first_deriv[k + 1] - first_deriv[k - 1]) / (z_cup[k + 1] - z_cup[k - 1])
        sec_deriv[k] = abs(sec_deriv[k])
    ilev = max(3, kstart + 1)
    ix = 1
    k = ilev
    while ilev < kend_p3:
        for kk in range(k, min(kend_p3 + 2, kte - 1) + 1):
            if sec_deriv[kk] < sec_deriv[kk + 1] and sec_deriv[kk] < sec_deriv[kk - 1]:
                k_inv[ix] = kk
                ix = min(5, ix + 1)
                ilev = kk + 1
                break
            ilev = kk + 1
        else:
            break
        k = ilev
    # 2nd criteria
    kadd = 0
    ken = _maxloc1_int(k_inv, kte)
    for kc in range(1, ken + 1):
        idx = kc + kadd
        if idx > kte:
            break
        kk = k_inv[idx]
        if kk == 1:
            break
        if dtempdz[kk] < dtempdz[kk - 1] and dtempdz[kk] < dtempdz[kk + 1]:
            kadd = kadd + 1
            for kj in range(kc, ken + 1):
                if kj + kadd <= kte:
                    if k_inv[kj + kadd] > 1:
                        k_inv[kj] = k_inv[kj + kadd]
                    if k_inv[kj + kadd] == 1:
                        k_inv[kj] = 1
    # find inversions near 800/550 hPa
    nmax = _maxloc1_int(k_inv, kte)
    big = 1.0e9
    sd = np.full(kte + 1, big)
    for kc in range(1, nmax + 1):
        dp = p_cup[k_inv[kc]] - p_cup[kstart]
        sd[kc] = abs(dp) - l_shal
    k800 = _minloc_abs(sd, 1, nmax)
    sd = np.full(kte + 1, big)
    for kc in range(1, nmax + 1):
        dp = p_cup[k_inv[kc]] - p_cup[kstart]
        sd[kc] = abs(dp) - l_mid
    k550 = _minloc_abs(sd, 1, nmax)
    shal_val = k_inv[k800]
    mid_val = k_inv[k550]
    k_inv2 = np.full(kte + 1, -1, dtype=np.int64)
    k_inv2[1] = shal_val
    k_inv2[2] = mid_val
    return k_inv2, dtempdz


def _maxloc1_int(arr, ktf):
    best = 1; bestv = arr[1]
    for k in range(1, ktf + 1):
        if arr[k] > bestv:
            bestv = arr[k]; best = k
    return best


def _minloc_abs(arr, lo, hi):
    if hi < lo:
        return lo
    best = lo; bestv = abs(arr[lo])
    for k in range(lo, hi + 1):
        if abs(arr[k]) < bestv:
            bestv = abs(arr[k]); best = k
    return best


def cup_gf_sh(zo, t, q, z1, tn, qo, po, psur, dhdt, kpbl, rho, hfx, qfx, xland,
              ichoice, tcrit, dtime, ktf):
    """Faithful single-column port of CUP_gf_sh (shallow). Returns dict."""
    kte = ktf
    res = {'outt': _z(kte), 'outq': _z(kte), 'outqc': _z(kte), 'pre': 0.0,
           'kbcon': 0, 'ktop': 0, 'k22': 0, 'ierr': 0, 'xmb_out': 0.0,
           'cnvwt': _z(kte), 'cupclw': _z(kte), 'zuo': _z(kte)}
    ierr = 0
    make_calc_for_xk = True
    flux_tun = FLUXTUNE
    xland1 = int(xland + 0.001)
    if xland > 1.5 or xland < 0.5:
        xland1 = 0
    entr_rate = 9.0e-5
    cap_max_increment = 25.0
    z = _z(kte); xz = _z(kte); qrco = _z(kte); pwo = _z(kte)
    cd = _z(kte); dellaqc = _z(kte); cupclw = _z(kte); cnvwt = _z(kte)
    up_massentro = _z(kte); up_massdetro = _z(kte)
    for k in range(1, ktf + 1):
        z[k] = zo[k]; xz[k] = zo[k]
        cd[k] = 1.0 * entr_rate
    cap_maxs = 125.0
    kbmax = 1; aa0 = 0.0; aa1 = 0.0
    cap_max = cap_maxs
    ztexec = 0.0; zqexec = 0.0; zws = 0.0
    buo_flux = (hfx / CP + 0.608 * t[1] * qfx / XLV) / rho[1]
    zws_t = max(0.0, flux_tun * 0.41 * buo_flux * zo[2] * 9.81 / t[1])
    if zws_t > np.finfo(np.float64).tiny:
        zws_t = 1.2 * zws_t ** 0.3333
        ztexec = max(flux_tun * hfx / (rho[1] * zws_t * CP), 0.0)
        zqexec = max(flux_tun * qfx / XLV / (rho[1] * zws_t), 0.0)
    zws = max(0.0, flux_tun * 0.41 * buo_flux * zo[kpbl] * 9.81 / t[kpbl])
    zws = 1.2 * zws ** 0.3333
    zws = zws * rho[kpbl]
    zkbmax = 3000.0
    z, qes, he, hes = cup_env(z, t, q, po, z1, psur, ierr, -1, ktf)
    zo, qeso, heo, heso = cup_env(zo, tn, qo, po, z1, psur, ierr, -1, ktf)
    (qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup) = \
        cup_env_clev(t, qes, q, he, hes, z, po, psur, z1, ierr, ktf)
    (qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup) = \
        cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, ierr, ktf)
    if ierr == 0:
        for k in range(1, ktf + 1):
            if zo_cup[k] > zkbmax + z1:
                kbmax = k
                break
        kbmax = min(kbmax, ktf // 2)
    # K22
    k22 = 0; kbcon = 0; ktop = 0
    if kpbl > 3:
        cap_max = po_cup[kpbl]
    if ierr == 0:
        k22 = _argmax_range(heo_cup, 2, kbmax)
        k22 = max(2, k22)
        if k22 > kbmax:
            ierr = 2; ktop = 0; k22 = 0; kbcon = 0
    hkb = 0.0; hkbo = 0.0
    if ierr == 0:
        x_add = XLV * zqexec + CP * ztexec
        hkb = get_cloud_bc(he_cup, k22, x_add)
        hkbo = get_cloud_bc(heo_cup, k22, x_add)
    dbyo = _z(kte)
    k22, kbcon, hkbo, ierr = cup_kbcon(cap_max_increment, 5, k22, kbcon, heo_cup,
                                       heso_cup, hkbo, ierr, kbmax, po_cup,
                                       cap_max, ztexec, zqexec, z_cup, entr_rate,
                                       heo, 0, ktf)
    kstabi = cup_minimi(heso_cup, kbcon, kbmax, ierr, ktf) if ierr == 0 else kbcon
    k_inv_layers, dtempdz = get_inversion_layers(p_cup, t_cup, z_cup, q_cup,
                                                 qes_cup, kbcon, kstabi, ierr,
                                                 1, kte, ktf)
    entr_rate_2d = _z(kte)
    start_level = 0
    for k in range(1, ktf + 1):
        entr_rate_2d[k] = entr_rate
    if ierr == 0:
        start_level = k22
        x_add = XLV * zqexec + CP * ztexec
        hkb = get_cloud_bc(he_cup, k22, x_add)
        if kbcon > ktf - 4:
            ierr = 231
        for k in range(1, ktf + 1):
            frh = 2.0 * min(qo_cup[k] / qeso_cup[k], 1.0)
            entr_rate_2d[k] = entr_rate * (2.3 - frh)
            cd[k] = entr_rate_2d[k]
        ktop = 1
        if k_inv_layers[1] > 0 and (po_cup[kbcon] - po_cup[k_inv_layers[1]]) < 200.0:
            ktop = k_inv_layers[1]
        else:
            for k in range(kbcon + 1, ktf + 1):
                if (po_cup[kbcon] - po_cup[k]) > 200.0:
                    ktop = k
                    break
    zuo = _z(kte); zu = _z(kte); xzu = _z(kte)
    ktopx = 0
    kbcon, ktop, ktopx, ierr, zuo = rates_up_pdf(
        'shallow', ktop, ierr, po_cup, entr_rate_2d, hkbo, heo, heso_cup,
        zo_cup, xland1, kstabi, k22, kbcon, zuo, kpbl, ktopx, kbcon, kbcon,
        1, kte, ktf)
    if ierr == 0:
        if k22 > 1:
            for k in range(1, k22):
                zuo[k] = 0.0; zu[k] = 0.0; xzu[k] = 0.0
        mlz = _maxloc1(zuo, kte)
        for k in range(mlz, ktop + 1):
            if zuo[k] < 1.0e-6:
                ktop = k - 1
                break
        for k in range(k22, ktop + 1):
            xzu[k] = zuo[k]; zu[k] = zuo[k]
        for k in range(ktop + 1, ktf + 1):
            zuo[k] = 0.0; zu[k] = 0.0; xzu[k] = 0.0
        k22 = max(2, k22)
    (up_massentro, up_massdetro, up_massentr, up_massdetr, _ue, _ud) = \
        get_lateral_massflux(zo_cup, zuo, cd, entr_rate_2d, ktop if ierr == 0 else 0,
                             kbcon, k22, 0.0, ktf, with_u=False)
    hc = _z(kte); qco = _z(kte); qrco = _z(kte); dby = _z(kte); hco = _z(kte); dbyo = _z(kte)
    if ierr == 0:
        for k in range(1, start_level):
            hc[k] = he_cup[k]; hco[k] = heo_cup[k]
        hc[start_level] = hkb
        hco[start_level] = hkbo
    dbyt = _z(kte)
    skip_42 = False
    if ierr == 0:
        for k in range(start_level + 1, ktop + 1):
            hc[k] = (hc[k - 1] * zu[k - 1] - 0.5 * up_massdetr[k - 1] * hc[k - 1]
                     + up_massentr[k - 1] * he[k - 1]) / (zu[k - 1] - 0.5 * up_massdetr[k - 1] + up_massentr[k - 1])
            dby[k] = max(0.0, hc[k] - hes_cup[k])
            hco[k] = (hco[k - 1] * zuo[k - 1] - 0.5 * up_massdetro[k - 1] * hco[k - 1]
                      + up_massentro[k - 1] * heo[k - 1]) / (zuo[k - 1] - 0.5 * up_massdetro[k - 1] + up_massentro[k - 1])
            dbyo[k] = hco[k] - heso_cup[k]
            dz = zo_cup[k + 1] - zo_cup[k]
            dbyt[k] = dbyt[k - 1] + dbyo[k] * dz
        ki = _maxloc1(dbyt, kte)
        if ktop > ki + 1:
            ktop = ki + 1
            for k in range(ktop + 1, ktf + 1):
                zuo[k] = 0.0; zu[k] = 0.0; cd[k] = 0.0
            up_massdetro[ktop] = zuo[ktop]
            for k in range(ktop, ktf + 1):
                up_massentro[k] = 0.0
            for k in range(ktop + 1, ktf + 1):
                up_massdetro[k] = 0.0
                entr_rate_2d[k] = 0.0
        if ktop < kbcon + 1:
            ierr = 5; skip_42 = True
        elif ktop > ktf - 2:
            ierr = 5; skip_42 = True
        if not skip_42:
            qaver = get_cloud_bc(qo_cup, k22)
            qaver = qaver + zqexec
            for k in range(1, start_level):
                qco[k] = qo_cup[k]
            qco[start_level] = qaver
            for k in range(start_level + 1, ktop + 1):
                trash = qeso_cup[k] + (1.0 / XLV) * (gammao_cup[k] / (1.0 + gammao_cup[k])) * dbyo[k]
                trash2 = qco[k - 1]
                qco[k] = (trash2 * (zuo[k - 1] - 0.5 * up_massdetr[k - 1])
                          + up_massentr[k - 1] * qo[k - 1]) / (zuo[k - 1] - 0.5 * up_massdetr[k - 1] + up_massentr[k - 1])
                if qco[k] >= trash:
                    dz = z_cup[k] - z_cup[k - 1]
                    qrco[k] = (qco[k] - trash) / (1.0 + (C0_SHAL + C1_SHAL) * dz)
                    pwo[k] = C0_SHAL * dz * qrco[k] * zuo[k]
                    qco[k] = trash + qrco[k]
                else:
                    qrco[k] = 0.0
                cupclw[k] = qrco[k]
            for k in range(k22 + 1, ktop + 1):
                dp = 100.0 * (po_cup[k] - po_cup[k + 1])
                cnvwt[k] = zuo[k] * cupclw[k] * G / dp
                qco[k] = qco[k] - qrco[k]
            for k in range(ktop + 1, ktf):
                hc[k] = hes_cup[k]; hco[k] = heso_cup[k]; qco[k] = qeso_cup[k]
                qrco[k] = 0.0; dby[k] = 0.0; dbyo[k] = 0.0
                zu[k] = 0.0; xzu[k] = 0.0; zuo[k] = 0.0
    # work functions
    if make_calc_for_xk:
        aa0 = cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, ktf)
        aa1 = cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, ktf)
        if ierr == 0 and aa1 <= 0.0:
            ierr = 17
    # dellah/dellaq
    dellah = _z(kte); dellaq = _z(kte); dellat = _z(kte)
    if ierr == 0:
        for k in range(k22, ktop + 1):
            entup = up_massentro[k]; detup = up_massdetro[k]
            dp = 100.0 * (po_cup[k] - po_cup[k + 1])
            dellah[k] = -(zuo[k + 1] * (hco[k + 1] - heo_cup[k + 1]) - zuo[k] * (hco[k] - heo_cup[k])) * G / dp
            dz = zo_cup[k + 1] - zo_cup[k]
            if k < ktop:
                dellaqc[k] = zuo[k] * C1_SHAL * qrco[k] * dz / dp * G
            else:
                dellaqc[k] = detup * qrco[k] * G / dp
            c_up = dellaqc[k] + (zuo[k + 1] * qrco[k + 1] - zuo[k] * qrco[k]) * G / dp
            dellaq[k] = (-(zuo[k + 1] * (qco[k + 1] - qo_cup[k + 1]) - zuo[k] * (qco[k] - qo_cup[k])) * G / dp
                         - c_up - 0.5 * (pwo[k] + pwo[k + 1]) * G / dp)
    mbdt = 0.5
    xhe = _z(kte); xq = _z(kte); xt = _z(kte)
    if ierr == 0:
        for k in range(1, ktf + 1):
            xhe[k] = dellah[k] * mbdt + heo[k]
            xq[k] = max(1.0e-16, (dellaq[k] + dellaqc[k]) * mbdt + qo[k])
            dellat[k] = (1.0 / CP) * (dellah[k] - XLV * dellaq[k])
            xt[k] = (-dellaqc[k] * XLV / CP + dellat[k]) * mbdt + tn[k]
            xt[k] = max(190.0, xt[k])
        xhe[ktf] = heo[ktf]; xq[ktf] = qo[ktf]; xt[ktf] = tn[ktf]
    xaa0 = 0.0
    if make_calc_for_xk:
        xz, xqes, xhe2, xhes = cup_env(xz, xt, xq, po, z1, psur, ierr, -1, ktf)
        (xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, _xp, _xg, xt_cup) = \
            cup_env_clev(xt, xqes, xq, xhe2, xhes, xz, po, psur, z1, ierr, ktf)
        xhc = _z(kte); xdby = _z(kte); xhkb = 0.0
        if ierr == 0:
            x_add = XLV * zqexec + CP * ztexec
            xhkb = get_cloud_bc(xhe_cup, k22, x_add)
            for k in range(1, start_level):
                xhc[k] = xhe_cup[k]
            xhc[start_level] = xhkb
            for k in range(1, ktf + 1):
                xzu[k] = zuo[k]
            for k in range(start_level + 1, ktop + 1):
                xhc[k] = (xhc[k - 1] * xzu[k - 1] - 0.5 * up_massdetro[k - 1] * xhc[k - 1]
                          + up_massentro[k - 1] * xhe[k - 1]) / (xzu[k - 1] - 0.5 * up_massdetro[k - 1] + up_massentro[k - 1])
                xdby[k] = xhc[k] - xhes_cup[k]
            for k in range(ktop + 1, ktf + 1):
                xhc[k] = xhes_cup[k]; xdby[k] = 0.0; xzu[k] = 0.0
        xaa0 = cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon, ktop, ierr, ktf)
    # shallow forcing
    xmb = 0.0
    outt = _z(kte); outq = _z(kte); outqc = _z(kte); pre = 0.0
    xmb_out = 0.0
    if ierr == 0:
        xmbmax = 1.0
        xkshal = (xaa0 - aa1) / mbdt
        if xkshal <= 0.0 and xkshal > -0.01 * mbdt:
            xkshal = -0.01 * mbdt
        if xkshal > 0.0 and xkshal < 1.0e-2:
            xkshal = 1.0e-2
        xff_shal = [0.0, 0.0, 0.0]
        xff_shal[0] = max(0.0, -(aa1 - aa0) / (xkshal * dtime))
        xff_shal[1] = 0.03 * zws
        blqe = 0.0
        for k in range(1, kpbl + 1):
            blqe = blqe + 100.0 * dhdt[k] * (po_cup[k] - po_cup[k + 1]) / G
        trash = max((hc[kbcon] - he_cup[kbcon]), 1.0e1)
        xff_shal[2] = max(0.0, blqe / trash)
        xff_shal[2] = min(xmbmax, xff_shal[2])
        xmb = (xff_shal[0] + xff_shal[1] + xff_shal[2]) / 3.0
        xmb = min(xmbmax, xmb)
        if ichoice > 0:
            xmb = min(xmbmax, xff_shal[ichoice - 1])
        if xmb <= 0.0:
            ierr = 21
    if ierr != 0:
        k22 = 0; kbcon = 0; ktop = 0; xmb = 0.0
        outt = _z(kte); outq = _z(kte); outqc = _z(kte)
    else:
        xmb_out = xmb
        pre = 0.0
        for k in range(2, ktop + 1):
            outt[k] = dellat[k] * xmb
            outq[k] = dellaq[k] * xmb
            outqc[k] = dellaqc[k] * xmb
            pre = pre + pwo[k] * xmb
    res.update({'outt': outt, 'outq': outq, 'outqc': outqc, 'pre': pre,
                'kbcon': kbcon, 'ktop': ktop, 'k22': k22, 'ierr': ierr,
                'xmb_out': xmb_out, 'cnvwt': cnvwt, 'cupclw': cupclw, 'zuo': zuo})
    return res


def gfdrv(t_col, qv_col, p_col, pi_col, dz8w_col, rho_col, u_col, v_col, w_col,
          rthblten_col, rqvblten_col, dt, dx, hfx, qfx, kpbl, xland, ht=0.0,
          ishallow_g3=1, ichoice=0):
    """Faithful single-column port of GFDRV (cu_physics=3).

    All *_col inputs are 0-based NumPy arrays of length KX (Fortran k=1..KX).
    Returns dict of WRF cumulus-driver outputs (0-based arrays length KX).
    """
    kx = len(t_col)
    ideep = 1
    imid_gf = 0
    ichoice_s = 0
    dicycle = 1
    tcrit = 258.0
    kts = 1; kte = kx; ktf = kx; ktf_drv = kx  # single column, ktf=MIN(kte,kde-1)=kx

    # 1-based working arrays (index 0 unused)
    def to1(arr):
        a = _z(kx)
        a[1:kx + 1] = np.asarray(arr, dtype=np.float64)
        return a
    t2d = to1(t_col); q2d = to1(qv_col); po_pa = to1(p_col); pi1 = to1(pi_col)
    dz8w = to1(dz8w_col); rhoi = to1(rho_col); us = to1(u_col); vs = to1(v_col)
    w1 = to1(w_col); rthbl = to1(rthblten_col); rqvbl = to1(rqvblten_col)
    # clamp q (GFDRV: IF(Q2d<1.e-8)Q2d=1.e-8)
    for k in range(1, ktf + 1):
        if q2d[k] < 1.0e-08:
            q2d[k] = 1.0e-08
    # pressure in mb
    po = _z(kx); p2d = _z(kx)
    for k in range(1, ktf + 1):
        po[k] = po_pa[k] * 0.01
        p2d[k] = po[k]
    # p8w = p (driver sets P8W=P), PSUR=p8w(1)*.01
    psur = po_pa[1] * 0.01
    ter11 = max(0.0, ht)
    # zo heights
    zo = _z(kx)
    zo[1] = ter11 + 0.5 * dz8w[1]
    for k in range(2, ktf + 1):
        zo[k] = zo[k - 1] + 0.5 * (dz8w[k - 1] + dz8w[k])
    # forced T/Q (RTHFTEN/RTHRATEN=0): TN=t2d+RTHBLTEN*pi*dt; QO=q2d+RQVBLTEN*dt
    tn = _z(kx); qo = _z(kx); tshall = _z(kx); qshall = _z(kx); dhdt = _z(kx)
    for k in range(1, ktf + 1):
        tn[k] = t2d[k] + (rthbl[k]) * pi1[k] * dt
        qo[k] = q2d[k] + (rqvbl[k]) * dt
        tshall[k] = t2d[k] + rthbl[k] * pi1[k] * dt
        dhdt[k] = CP * rthbl[k] * pi1[k] + XLV * rqvbl[k]
        qshall[k] = q2d[k] + rqvbl[k] * dt
        if tn[k] < 200.0:
            tn[k] = t2d[k]
        if qo[k] < 1.0e-08:
            qo[k] = 1.0e-08
    # omega, mconv
    omeg = _z(kx)
    for k in range(1, ktf + 1):
        omeg[k] = -G * rhoi[k] * w1[k]
    mconv = 0.0
    for k in range(1, ktf):
        dq = q2d[k + 1] - q2d[k]
        mconv = mconv + omeg[k] * dq / G
    if mconv < 0.0:
        mconv = 0.0
    csum = 0  # no memory carry in standalone driver
    ccn = 150.0

    # shallow convection
    cutens = 1.0 if ishallow_g3 == 1 else 0.0
    sh = None
    if ishallow_g3 == 1:
        sh = cup_gf_sh(zo.copy(), t2d.copy(), q2d.copy(), ter11, tshall.copy(),
                       qshall.copy(), po.copy(), psur, dhdt.copy(), kpbl,
                       rhoi.copy(), hfx, qfx, xland, ichoice_s, tcrit, dt, ktf)
        if sh['xmb_out'] <= 0.0:
            cutens = 0.0
        # neg_check shallow
        outus = _z(kx); outvs = _z(kx)
        pre_s = neg_check('shallow', dt, q2d, sh['outq'], sh['outt'], outus,
                          outvs, sh['outqc'], sh['pre'], ktf)
        sh['pre'] = pre_s
    # deep convection
    dp_res = cup_gf(dicycle, ichoice, ccn, dt, 0, kpbl, dhdt.copy(), xland,
                    zo.copy(), t2d.copy(), q2d.copy(), ter11, tn.copy(),
                    qo.copy(), po.copy(), psur, us.copy(), vs.copy(),
                    rhoi.copy(), hfx, qfx, dx, mconv, omeg.copy(), csum, ktf)
    # neg_check deep
    outu_d = dp_res['outu']; outv_d = dp_res['outv']
    pre_d = neg_check('deep', dt, q2d, dp_res['outq'], dp_res['outt'], outu_d,
                      outv_d, dp_res['outqc'], dp_res['pre'], ktf)
    dp_res['pre'] = pre_d

    # combine (GFDRV feedback). cuten=1 if pret>0 else 0
    cuten = 1.0 if (dp_res['ierr'] == 0 and dp_res['pre'] > 0.0) else 0.0
    kbcon_d = dp_res['kbcon']; ktop_d = dp_res['ktop']
    if cuten == 0.0:
        kbcon_d = 0; ktop_d = 0
    outts = sh['outt'] if sh else _z(kx)
    outqs = sh['outq'] if sh else _z(kx)
    outqcs = sh['outqc'] if sh else _z(kx)
    prets = sh['pre'] if sh else 0.0

    rthcuten = _z(kx); rqvcuten = _z(kx); rqccuten = _z(kx); rqicuten = _z(kx)
    for k in range(1, ktf + 1):
        rthcuten[k] = (cutens * outts[k] + cuten * dp_res['outt'][k]) / pi1[k]
        rqvcuten[k] = cuten * dp_res['outq'][k] + cutens * outqs[k]
    # pratec / raincv
    pratec = 0.0; raincv = 0.0
    if dp_res['pre'] > 0.0 or prets > 0.0:
        pratec = cuten * dp_res['pre'] + cutens * prets
        raincv = pratec * dt
    # RQCCUTEN / RQICUTEN with T<258 ice split
    cupclw_d = dp_res['cupclw']; cupclws = sh['cupclw'] if sh else _z(kx)
    for k in range(1, ktf + 1):
        rqccuten[k] = outqcs[k] + dp_res['outqc'][k] * cuten
    for k in range(1, ktf + 1):
        if t2d[k] < 258.0:
            rqicuten[k] = outqcs[k] + dp_res['outqc'][k] * cuten
            rqccuten[k] = 0.0
        else:
            rqicuten[k] = 0.0
            rqccuten[k] = outqcs[k] + dp_res['outqc'][k] * cuten

    def to0(a):
        return np.asarray(a[1:kx + 1], dtype=np.float64)
    return {
        'RTHCUTEN': to0(rthcuten),
        'RQVCUTEN': to0(rqvcuten),
        'RQCCUTEN': to0(rqccuten),
        'RQICUTEN': to0(rqicuten),
        'RAINCV': float(raincv),
        'PRATEC': float(pratec),
        'KTOP_DEEP': int(ktop_d),
        'XMB_SHALLOW': float(sh['xmb_out']) if sh else 0.0,
        'K22_SHALLOW': int(sh['k22']) if sh else 0,
        'KBCON_SHALLOW': int(sh['kbcon']) if sh else 0,
        'KTOP_SHALLOW': int(sh['ktop']) if sh else 0,
        'IERR_DEEP': int(dp_res['ierr']),
        'IERR_SHALLOW': int(sh['ierr']) if sh else -1,
        'SIG': None,
    }
