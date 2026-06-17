"""Goddard GCE (WRF mp_physics=97, ``gsfcgcescheme``) constants -- faithful port of consat_s.

These reproduce, in fp64, the CGS-unit constant block computed in
``phys/module_mp_gsfcgce.F`` SUBROUTINE ``consat_s(ihail=0, itaobraun=1)`` (the
operational WRF call: 3-ice graupel, Tao-et-al-2003 constants). Every value is
copied directly from the Fortran with the same formula and ordering so the JAX
port is byte-for-byte the WRF algorithm (not a re-derivation).

The Bergeron lookup tables ``aa1``/``aa2`` and the derived ``rn12a``/``rn12b``/
``rn25a`` arrays are temperature-indexed (1..31) exactly as in WRF.

Reference precision: the Fortran ``gammagce`` Lanczos series and ``ggamma``
polynomial (used by fall_flux) are reproduced identically.
"""

from __future__ import annotations

import math

import numpy as np

# ---- gammagce: WRF Lanczos series (consat_s gamma) ----
_GCOF = (76.18009173, -86.50532033, 24.01409822,
         -1.231739516, 0.120858003e-2, -0.536382e-5)
_GSTP = 2.50662827465


def gammagce(xx: float) -> float:
    """Faithful port of FUNCTION gammagce (consat_s gamma)."""
    x = xx - 1.0
    tmp = x + 5.5
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 1.0
    for j in range(6):
        x = x + 1.0
        ser = ser + _GCOF[j] / x
    gammln = tmp + math.log(_GSTP * ser)
    return math.exp(gammln)


def ggamma(x: float) -> float:
    """Faithful port of FUNCTION ggamma (fall_flux gamma, 8-term poly)."""
    B = (-.577191652, .988205891, -.897056937, .918206857,
         -.756704078, .482199394, -.193527818, .035868343)
    pf = 1.0
    temp = x
    while temp > 2.0:
        temp = temp - 1.0
        pf = pf * temp
    g1to2 = 1.0
    temp = temp - 1.0
    for k1 in range(1, 9):
        g1to2 = g1to2 + B[k1 - 1] * temp ** k1
    return pf * g1to2


# ---- Bergeron tables aa1/aa2 (module-level DATA) ----
AA1 = np.array([
    .7939e-7, .7841e-6, .3369e-5, .4336e-5, .5285e-5,
    .3728e-5, .1852e-5, .2991e-6, .4248e-6, .7434e-6,
    .1812e-5, .4394e-5, .9145e-5, .1725e-4, .3348e-4,
    .1725e-4, .9175e-5, .4412e-5, .2252e-5, .9115e-6,
    .4876e-6, .3473e-6, .4758e-6, .6306e-6, .8573e-6,
    .7868e-6, .7192e-6, .6513e-6, .5956e-6, .5333e-6,
    .4834e-6], dtype=np.float64)
AA2 = np.array([
    .4006, .4831, .5320, .5307, .5319,
    .5249, .4888, .3894, .4047, .4318,
    .4771, .5183, .5463, .5651, .5813,
    .5655, .5478, .5203, .4906, .4447,
    .4126, .3960, .4149, .4320, .4506,
    .4483, .4460, .4433, .4413, .4382,
    .4361], dtype=np.float64)


def _build_constants() -> dict:
    """Reproduce consat_s(ihail=0, itaobraun=1) exactly. Returns a dict of consts."""
    C: dict[str, float] = {}

    # JJS block
    al = 2.5e10
    cp = 1.004e7
    rd1 = 1.e-3
    rd2 = 2.2

    cpi = 4.0 * math.atan(1.0)
    cpi2 = cpi * cpi
    grvt = 980.0
    cd1 = 6.e-1
    cd2 = 4.0 * grvt / (3.0 * cd1)
    tca0 = 2.43e3   # base TCA constant (overwritten per-cell in saticel)
    dwv = .226
    dva = 1.718e-4
    amw = 18.016
    ars = 8.314e7
    scv0 = 2.2904487
    t0 = 273.16
    t00 = 238.16
    alv = 2.5e10
    alf = 3.336e9
    als = 2.8336e10
    avc = alv / cp
    afc = alf / cp
    asc = als / cp
    rw = 4.615e6
    cw = 4.187e7
    ci = 2.093e7
    c76 = 7.66
    c358 = 35.86
    c172 = 17.26939
    c409 = 4098.026
    c218 = 21.87456
    c580 = 5807.695
    c610 = 6.1078e3
    c149 = 1.496286e-5
    c879 = 8.794142
    c141 = 1.4144354e7

    # hail or graupel (ihail=0 -> graupel)
    roqg = .4
    ag = 351.2
    bg = .37
    tng = .04
    # snow
    tns = .16
    roqs = .1
    as_ = 78.63
    bs = .11
    # rain
    aw = 2115.0
    bw = .8
    roqr = 1.0
    tnw = .08

    bgh = .5 * bg
    bsh = .5 * bs
    bwh = .5 * bw
    bgq = .25 * bg
    bsq = .25 * bs
    bwq = .25 * bw

    ga3b = gammagce(3. + bw)
    ga4b = gammagce(4. + bw)
    ga6b = gammagce(6. + bw)
    ga5bh = gammagce((5. + bw) / 2.)
    ga3g = gammagce(3. + bg)
    ga4g = gammagce(4. + bg)
    ga5gh = gammagce((5. + bg) / 2.)
    ga3d = gammagce(3. + bs)
    ga4d = gammagce(4. + bs)
    ga5dh = gammagce((5. + bs) / 2.)

    ac1 = aw
    ac2 = ag
    ac3 = as_
    bc1 = bw
    cc1 = as_
    dc1 = bs
    zrc = (cpi * roqr * tnw) ** 0.25
    zsc = (cpi * roqs * tns) ** 0.25
    zgc = (cpi * roqg * tng) ** 0.25
    vrc = aw * ga4b / (6. * zrc ** bw)
    vsc = as_ * ga4d / (6. * zsc ** bs)
    vgc = ag * ga4g / (6. * zgc ** bg)

    rn1 = 9.4e-15
    bnd1 = 6.e-4
    rn2 = 1.e-3
    bnd2 = 2.0e-3
    rn3 = .25 * cpi * tns * cc1 * ga3d
    esw = 1.0
    rn4 = .25 * cpi * esw * tns * cc1 * ga3d
    eri = .1
    rn5 = .25 * cpi * eri * tnw * ac1 * ga3b
    ami = 1.0 / (24. * 6.e-9)
    rn6 = cpi2 * eri * tnw * ac1 * roqr * ga6b * ami
    esr = .5
    rn7 = cpi2 * esr * tnw * tns * roqs
    esr = 1.0
    rn8 = cpi2 * esr * tnw * tns * roqr
    rn9 = cpi2 * tns * tng * roqs
    rn10 = 2. * cpi * tns
    rn101 = .31 * ga5dh * math.sqrt(cc1)
    rn10a = als * als / rw
    rn10b = alv / tca0
    rn10c = ars / (dwv * amw)
    rn11 = 2. * cpi * tns / alf
    rn11a = cw / alf
    ami50 = 3.84e-6
    ami40 = 3.08e-8
    eiw = 1.0
    ui50 = 100.0
    ri50 = 2. * 5.e-3
    cmn = 1.05e-15
    rn12 = cpi * eiw * ui50 * ri50 ** 2

    # tables (k=1..31)
    rn12a = np.zeros(31, dtype=np.float64)
    rn12b = np.zeros(31, dtype=np.float64)
    rn13 = np.zeros(31, dtype=np.float64)
    rn25a = np.zeros(31, dtype=np.float64)
    for k in range(31):
        y1 = 1. - AA2[k]
        rn13[k] = AA1[k] * y1 / (ami50 ** y1 - ami40 ** y1)
        rn12a[k] = rn13[k] / ami50
        rn12b[k] = AA1[k] * ami50 ** AA2[k]
        rn25a[k] = AA1[k] * cmn ** AA2[k]

    egw = 1.0
    rn14 = .25 * cpi * egw * tng * ga3g * ag
    egi = .1
    rn15 = .25 * cpi * egi * tng * ga3g * ag
    egi = 1.0
    rn15a = .25 * cpi * egi * tng * ga3g * ag
    egr = 1.0
    rn16 = cpi2 * egr * tng * tnw * roqr
    rn17 = 2. * cpi * tng
    rn17a = .31 * ga5gh * math.sqrt(ag)
    rn17b = cw - ci
    rn17c = cw
    apri = .66
    bpri = 1.e-4
    bpri = 0.5 * bpri
    rn18 = 20. * cpi2 * bpri * tnw * roqr
    rn18a = apri
    rn19 = 2. * cpi * tng / alf
    rn19a = .31 * ga5gh * math.sqrt(ag)
    rn19b = cw / alf
    rn20 = 2. * cpi * tng
    rn20a = als * als / rw
    rn20b = .31 * ga5gh * math.sqrt(ag)
    bnd3 = 2.e-3
    rn21 = 1.e3 * 1.569e-12 / 0.15
    erw = 1.0
    rn22 = .25 * cpi * erw * ac1 * tnw * ga3b
    rn23 = 2. * cpi * tnw
    rn23a = .31 * ga5bh * math.sqrt(ac1)
    rn23b = alv * alv / rw

    # itaobraun=1
    cn0 = 1.e-6
    beta = -.46

    rn25 = cn0
    rn30a = alv * als * amw / (tca0 * ars)
    rn30b = alv / tca0
    rn30c = ars / (dwv * amw)
    rn31 = 1.e-17
    rn32 = 4. * 51.545e-4

    C.update(dict(
        al=al, cp=cp, rd1=rd1, rd2=rd2, cpi=cpi, cpi2=cpi2,
        dwv0=dwv, dva=dva, amw=amw, ars=ars, t0=t0, t00=t00,
        alv=alv, alf=alf, als=als, avc=avc, afc=afc, asc=asc,
        rw=rw, cw=cw, ci=ci, c76=c76, c358=c358, c172=c172,
        c409=c409, c218=c218, c580=c580, c610=c610, c149=c149,
        c879=c879, c141=c141,
        ag=ag, bg=bg, as_=as_, bs=bs, aw=aw, bw=bw,
        bgh=bgh, bsh=bsh, bwh=bwh, bgq=bgq, bsq=bsq, bwq=bwq,
        roqg=roqg, roqs=roqs, roqr=roqr, tng=tng, tns=tns, tnw=tnw,
        zrc=zrc, zsc=zsc, zgc=zgc, vrc=vrc, vsc=vsc, vgc=vgc,
        rn1=rn1, bnd1=bnd1, rn2=rn2, bnd2=bnd2, rn3=rn3, rn4=rn4,
        rn5=rn5, rn6=rn6, rn7=rn7, rn8=rn8, rn9=rn9, rn10=rn10,
        rn101=rn101, rn10a=rn10a, rn10b=rn10b, rn10c=rn10c,
        rn11=rn11, rn11a=rn11a, rn12=rn12,
        rn14=rn14, rn15=rn15, rn15a=rn15a, rn16=rn16, rn17=rn17,
        rn17a=rn17a, rn17b=rn17b, rn17c=rn17c, rn18=rn18, rn18a=rn18a,
        rn19=rn19, rn19a=rn19a, rn19b=rn19b, rn20=rn20, rn20a=rn20a,
        rn20b=rn20b, bnd3=bnd3, rn21=rn21, rn22=rn22, rn23=rn23,
        rn23a=rn23a, rn23b=rn23b, rn25=rn25, rn30a=rn30a, rn30b=rn30b,
        rn30c=rn30c, rn31=rn31, beta=beta, rn32=rn32,
    ))
    C["rn12a"] = rn12a
    C["rn12b"] = rn12b
    C["rn25a"] = rn25a
    return C


_C = _build_constants()


def __getattr__(name: str):
    if name in _C:
        return _C[name]
    raise AttributeError(name)


# ---- fall_flux PARAMETERs (these are NOT from consat_s; they are the
# fall_flux-local Lin/Hobbs/RH sedimentation constants) ----
XNOR = 8.0e6
XNOS = 1.6e7
XNOG = 4.0e6
RHOHAIL = 917.0
RHOGRAUL = 400.0
RHOWATER = 1000.0
RHOSNOW = 100.0
CONSTB = 0.8
CONSTD = 0.11
O6 = 1.0 / 6.0
CDRAG = 0.6
ABAR = 19.3
BBAR = 0.37
P0_FALL = 1.0e5
RHOE_S = 1.29
GRAV = 9.81

# fall_flux gamma values (computed at import in fp64)
CONSTA = 2115.0 * 0.01 ** (1 - CONSTB)
CONSTC = 78.63 * 0.01 ** (1 - CONSTD)
GAMBP4 = ggamma(CONSTB + 4.0)
GAMDP4 = ggamma(CONSTD + 4.0)
GAM4PT5 = ggamma(4.5)
GAM4BBAR = ggamma(4.0 + BBAR)

# itaobraun=1 (set inside saticel_s); kept as named consts for the process block
CN0 = 1.0e-6
BETA = -0.46
BETAH = 0.5 * BETA

# saticel_s overrides for ami values (re-set inside saticel before PINT)
AMI50_SAT = 3.76e-8
AMI100_SAT = 1.51e-7
AMI40_SAT = 2.41e-8
