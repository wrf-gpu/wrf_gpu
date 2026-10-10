"""nssl_2mom_gs PART B (module lines 16327-19695): collection, aggregation, rain self-collection,
autoconversion, Bigg/contact freezing, ventilation, melting and the deposition (qlimit) block.

Pristine source: WRF ``phys/module_mp_nssl_2mom.F`` (sha256 29f42e76...).  WRF-default mp=18
configuration only (ipconc=5, lzr=lzh=lzhl=0, ipelec=0, imurain=1, iacr=2, iacrsize=5,
ibiggopt=2, ibiggsnow=3, nsplinter=0, dmrauto=0, dmropt=0, irainbreak=0, icfn=2, ihrn=0,
iferwisventr=2, ihmlt=2, imltshddmr=1, ibinhmlr=ibinhlmlr=0, ivhmltsoak=1, mixedphase=.false.,
eqtset=1, iqvsopt=1, icond=1, irimdenopt=1, DoSublimationFix=.true.); ``_check_config`` asserts
these.  Branches dead in that configuration are omitted with a comment naming the killing flag.

Vectorised point-wise over the gathered points carried by ``G`` (see PORTING.md):
``G`` maps lower-case Fortran gs locals to per-point arrays (``(npts,)``), species-indexed locals
to dicts keyed by the Fortran index (``G['qx'][LR]``) and 2-extra-dim locals to tuple keys
(``G['xdia'][(LR, 3)]``).  Scalars may be Python/NumPy scalars.

(a) G1 keys CONSUMED (A -> B interface; all in ``gs_a.A_TO_B_KEYS``):
    alpha axx bxx ccmxd ccwresv cdxgs chmxd cimxd crmxd csmxd cx cxmxd da0lh da0lhl da0lr da1lc da1lr
    dab0lh dab1lh dtpinv ehi ehiclsn ehli ehliclsn ehlr ehls ehlsclsn ehlw ehr ehs ehsclsn ehw eiw eri
    erw esi esiclsn ess esw fadvisc fai fakvisc fav fbi fbv fci fcqv1 fcqv2 fcw felf felscp felv felvcp
    fschm ftka fwvdf g1x gf1 gf2 gf3 gf4 gf5 gf6 pi0 pii pk pqs pres qcmxd qcwresv qhmxd qimxd qis qrmxd
    qsmxd qss qss0 qv0 qvs qx qxmxd raindn rb rh rho0 rhoinv rhovt rimdn rzxs ssi ssw temcg temg theta0
    thetap ventrxn vtxbar vx xdia xdn xl2p xmas xmascw xv xvbiggsnow
    OPTIONAL (Fortran keeps the previous, stale value where part B does not assign; default zero):
    cilen dqci dqcw dqvcnd dqwv dqwvtmp fraci fracl qvtmp
    EXTRA global input (not a gs local): ``gz`` = dz (m) at each gathered point (Fortran
    gz(igs,jgs,kgs); in tests taken from the S0 dz2d column)
(b) keys PRODUCED for part C (B -> C interface): every per-point rate/coefficient assigned here, i.e.
    xplate xcolmn qracw qraci craci qracs qsacw csacw vsacw qsaci csaci csaci0 qsacr qsacrs csacr
    qhacw qhacwmlr vhacw vhsoak zhacw qhaci qhaci0 qhacs qhacs0 qhacr qhacrmlr vhacr chacr zhacr
    qhlacw qhlacwmlr vhlacw vhlsoak qhlaci qhlaci0 qhlacis qhlacis0 qhlacs qhlacs0 qhlacr qhlacrmlr
    chlacr vhlacr qiacw qiacr qiacrf qiacrs ciacrs ciacr ciacrf viacrf csplinter qsplinter csplinter2
    qsplinter2 csacs ciacw cracw cracr chacw chaci chaci0 chacs chacs0 chlacw chlaci chlaci0 chlacs
    chlacs0 zrcnw qrcnw crcnw cautn qrfrz qrfrzs qrfrzf vrfrzf crfrz crfrzs crfrzf zrfrz zrfrzs
    zrfrzf qwcnr qwfrz cwfrz qwfrzc cwfrzc qwfrzp cwfrzp ccia cwctfz qwctfz cwctfzc qwctfzc cwctfzp
    qwctfzp ciihr qiihr cicichr qicichr cipiphr qipiphr qscnvi cscnvi cscnvis fvent civent rwvent
    swvent hwvent hwventy hlvent hlventy fwet1 fwet2 fmlt1 fmlt2 fmlt1e fvds fvce qsmlr qimlr qhmlr
    qhlmlr qhfzh qffzf qhlfzhl qhfzhlg qhlfzhllg vhfzh vffzf vhlfzhl qsfzs zhmlr zhmlrr zsmlrr zhshr
    zhlmlr zhlshr zhshrr zhlmlrr zhlshrr csmlr csmlrr chmlr chmlrr chlmlr chlfmlr chlmlrr qhmlh cimlr
    rwcap swcap hwcap hlcap cilen cicap qhdsv qhldsv qidsv qsdsv qitmp qrtmp qctmp qsimxdep qsimxsub
    dqcitmp qvtmp dqwvtmp dqcw dqci dqwv fracl fraci dqvcnd qss ec0 rarx
    and the updated ``vtxbar`` (hail rows capped at dz/dt), ``rimdn``, ``raindn``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .indices import LC, LH, LHL, LI, LR, LS, LV
from .mathfun import gaminterp, iacr_table
from .satfun import tabqis, tabqvs

F64 = jnp.float64

# gs-local PARAMETERs (module lines 12922-13027, 12866-12950)
_AA1 = 9.44e15
_AA2 = 5.78e3
_RVT = 0.104
_RWRADMN = 50.e-6
_THNUC = 235.15
_RAERO, _KAERO = 3.e-7, 5.39e-3  # parameter ( raero = 3.e-7, kaero = 5.39e-3 ) (line 12993)
_KB = 1.3807e-23  # parameter (kb = 1.3807e-23) (line 12995)
_VR3MM_BASE = 5.23599e-10  # vr3mm = 5.23599e-10*(3.0/1.)**3 (REAL parameter)


def _check_config(C):
    want = dict(ipconc=5, lzh=0, lzr=0, lzhl=0, lhl=8, ipelec=0, imurain=1, iacr=2, iacrsize=5,
                ibiggsnow=3, nsplinter=0, iessec0flag=0, dmrauto=0, dmropt=0, icracrthresh=1,
                irainbreak=0, icracr=1, ibincracr=0, ibiggopt=2, ibiggsmallrain=0, ibfr=2, ibfc=1,
                icfn=2, ihrn=0, iferwisventr=2, ihmlt=2, imltshddmr=1, ibinhmlr=0, ibinhlmlr=0,
                ivhmltsoak=1, lhlw=0, lhwlg=0, eqtset=1, iqvsopt=1, icond=1, irimdenopt=1,
                iehr0c=0, iehlr0c=0, iqhacrmlr=1, iqhlacrmlr=1, iqhacwshr=1, iqhlacwshr=1, lis=0)
    bad = {k: getattr(C, k) for k, v in want.items() if getattr(C, k) != v}
    if bad or C.mixedphase or C.lwsm6 or float(C.rimdenvwgt) != 0.0 or float(C.snowmeltdia) > 0.0:
        raise NotImplementedError(f"gs_part_b ported for the WRF-default mp=18 configuration only: {bad}")
    if C.lvol[LS] > 1 or not (C.lvol[LH] > 1 and C.lvol[LHL] > 1):
        raise NotImplementedError("gs_part_b expects graupel+hail volume, no snow volume")


def gs_part_b(G, C, prec, dtp):
    """Run gs lines 16327-19695 on the gathered points in ``G``; returns a new dict."""
    _check_config(C)
    R = prec.R
    G = dict(G)

    def r(x):
        return jnp.asarray(x, R)

    W = jnp.where
    mn = jnp.minimum
    mx = jnp.maximum

    qx, cx, xv, xmas, alpha, xdn = G["qx"], G["cx"], G["xv"], G["xmas"], G["alpha"], G["xdn"]
    xdia = G["xdia"]
    vtxbar = dict(G["vtxbar"])
    rimdn = dict(G["rimdn"])
    raindn = dict(G["raindn"])
    qxmxd, cxmxd = G["qxmxd"], G["cxmxd"]
    rho0, rhoinv, temg, temcg = G["rho0"], G["rhoinv"], G["temg"], G["temcg"]
    qxmin = {il: r(C.qxmin[il]) for il in range(LC, LHL + 1)}
    zero = jnp.zeros_like(rho0)
    pi = r(C.pi)
    dtp = r(dtp)
    dtpinv = r(G["dtpinv"])
    tfr = r(C.tfr)
    aa1, aa2, rvt, rwradmn = r(_AA1), r(_AA2), r(_RVT), r(_RWRADMN)
    gf1, gf2, gf3 = r(G["gf1"]), r(G["gf2"]), r(G["gf3"])
    cinu = r(C.cinu)
    da0 = {il: r(C.da0[il]) for il in range(LC, LHL + 1)}
    da1 = {il: r(C.da1[il]) for il in range(LC, LHL + 1)}
    dab0 = lambda a, b: r(C.dab0[a, b])  # noqa: E731
    dab1 = lambda a, b: r(C.dab1[a, b])  # noqa: E731
    rimc1, rimc2, rimc3, rimc4 = r(C.rimc1), r(C.rimc2), r(C.rimc3), r(C.rimc4)
    qcwresv, ccwresv = G["qcwresv"], G["ccwresv"]
    vt_hl = lambda a, b: jnp.sqrt((vtxbar[(a, 1)] - vtxbar[(b, 1)]) ** 2 + 0.04 * vtxbar[(a, 1)] * vtxbar[(b, 1)])  # noqa: E731
    out = {}

    # ---------------------------------------------------------------- plates vs. columns (16334)
    out["xplate"] = zero
    out["xcolmn"] = zero + 1.0

    # ---------------------------------------------------------------- rain collects cloud (16371)
    # ipconc >= 3 branch; dmrauto = 0 makes the (dmrauto <= 0 .or. ...) test always true
    rwrad = 0.5 * xdia[(LR, 3)]
    q_big = G["erw"] * aa2 * cx[LR] * cx[LC] * xmas[LC] * (
        (alpha[LC] + 2.) * xv[LC] / (alpha[LC] + 1.) + xv[LR]) / rho0
    q_small = aa1 * cx[LR] * (qx[LC] - qcwresv) * (
        (alpha[LC] + 3.) * (alpha[LC] + 2.) * xv[LC] ** 2 / (alpha[LC] + 1.) ** 2
        + (alpha[LR] + 6.) * (alpha[LR] + 5.) * (alpha[LR] + 4.) * xv[LR] ** 2
        / ((alpha[LR] + 3.) * (alpha[LR] + 2.) * (alpha[LR] + 1.)))  # imurain == 1
    qracw = W(rwrad.astype(F64) > G["rh"], W(rwrad > rwradmn, q_big, q_small), zero)
    on_racw = (qx[LR] > qxmin[LR]) & (G["erw"] > 0.0)
    qracw = W(on_racw, mn(qracw, G["qcmxd"]), zero)
    out["qracw"] = qracw

    # ---------------------------------------------------------------- rain collects ice (16433)
    on = (G["eri"] > 0.0) & (xdia[(LR, 3)] > 2. * rwradmn)  # iacr = 2 >= 1
    tmp = G["eri"] * aa2 * cx[LR] * cx[LI] * ((cinu + 2.) * xv[LI] / (cinu + 1.) + xv[LR])
    qraci = W(on, mn(qxmxd[LI], tmp * xmas[LI] * rhoinv), zero)
    craci = W(on, mn(cxmxd[LI], tmp), zero)
    qraci = W(on & (temg > 268.15), zero, qraci)
    out["qraci"], out["craci"] = qraci, craci
    out["qracs"] = zero  # ipconc >= 3: the qracs (ipconc < 3) block is dead

    # ---------------------------------------------------------------- snow collects cloud (16507)
    # ipconc >= 4; lvol(ls) = 0 -> no snow rime volume (vsacw stays 0)
    on = G["esw"] > 0.0
    tmp = 1.0 * rvt * aa2 * cx[LS] * cx[LC] * ((alpha[LC] + 2.) * xv[LC] / (alpha[LC] + 1.) + xv[LS])
    out["qsacw"] = W(on, mn(qxmxd[LC], tmp * xmas[LC] * rhoinv), zero)
    out["csacw"] = W(on, mn(cxmxd[LC], tmp), zero)
    out["vsacw"] = zero

    # ---------------------------------------------------------------- snow collects ice (16565)
    on = G["esi"] > 0.0  # ipelec = 0
    tmp = G["esiclsn"] * rvt * aa2 * cx[LS] * cx[LI] * ((cinu + 2.) * xv[LI] / (cinu + 1.) + xv[LS])
    out["qsaci"] = W(on, mn(qxmxd[LI], G["esi"] * tmp * xmas[LI] * rhoinv), zero)
    out["csaci0"] = W(on, tmp, zero)
    out["csaci"] = W(on, mn(cxmxd[LI], G["esi"] * tmp), zero)

    # ---------------------------------------------------------------- snow collects rain (16606)
    # ipconc >= 3: the IF(ipconc >= 3) branch is empty -> qsacr = qsacrs = csacr = 0
    out["qsacr"] = zero
    out["qsacrs"] = zero
    out["csacr"] = zero

    # ---------------------------------------------------------------- graupel collects cloud (16646)
    vt = jnp.abs(vtxbar[(LH, 1)] - vtxbar[(LC, 1)])
    qhacw = 0.25 * pi * G["ehw"] * cx[LH] * (qx[LC] - qcwresv) * vt * (
        G["da0lh"] * xdia[(LH, 3)] ** 2
        + G["dab1lh"][(LH, LC)] * xdia[(LH, 3)] * xdia[(LC, 3)]
        + G["da1lc"] * xdia[(LC, 3)] ** 2)
    qhacw = mn(qhacw, 0.5 * qx[LC] * dtpinv)
    on_hw = G["ehw"] > 0.0
    qhacw = W(on_hw, qhacw, zero)
    qhacwmlr = qhacw  # iqhacwshr = 1: qhacw kept for T > 0C
    # rime density (lvol(lh) > 1), irimdenopt = 1, rimdenvwgt = 0
    vt = (1.0 - r(C.rimdenvwgt)) * vtxbar[(LH, 1)] + r(C.rimdenvwgt) * vtxbar[(LH, 2)]
    cold = temg < 273.15
    rd = rimc1 * jnp.power(-((0.5) * (1.e+06) * xdia[(LC, 1)]) * ((0.60) * vt) / W(cold, temg - 273.15, -1.0), rimc2)
    rd = mn(mx(rimc3, rd), rimc4)
    rimdn[LH] = W(on_hw, W(cold, rd, r(1000.)), rimdn[LH])
    vhacw = W(on_hw, rho0 * qhacw / rimdn[LH], zero)
    out["qhacw"], out["qhacwmlr"], out["vhacw"] = qhacw, qhacwmlr, vhacw
    out["vhsoak"] = zero
    out["zhacw"] = zero
    rarx = {LH: zero}  # ipelec = 0

    # ---------------------------------------------------------------- graupel collects ice (16780)
    vt = vt_hl(LH, LI)
    qhaci0 = 0.25 * pi * G["ehiclsn"] * cx[LH] * qx[LI] * vt * (
        G["da0lh"] * xdia[(LH, 3)] ** 2 + G["dab1lh"][(LH, LI)] * xdia[(LH, 3)] * xdia[(LI, 3)]
        + da1[LI] * xdia[(LI, 3)] ** 2)
    on = G["ehi"] > 0.0
    out["qhaci0"] = W(on, qhaci0, zero)
    out["qhaci"] = W(on, mn(G["ehi"] * qhaci0, G["qimxd"]), zero)

    # ---------------------------------------------------------------- graupel collects snow (16810)
    vt = vt_hl(LH, LS)
    qhacs0 = 0.25 * pi * G["ehsclsn"] * cx[LH] * qx[LS] * vt * (
        G["da0lh"] * xdia[(LH, 3)] ** 2 + G["dab1lh"][(LH, LS)] * xdia[(LH, 3)] * xdia[(LS, 3)]
        + da1[LS] * xdia[(LS, 3)] ** 2)
    on = G["ehs"] > 0.0
    out["qhacs0"] = W(on, qhacs0, zero)
    out["qhacs"] = W(on, mn(G["ehs"] * qhacs0, G["qsmxd"]), zero)

    # ---------------------------------------------------------------- graupel collects rain (16839)
    raindn[LH] = W(temg > tfr, r(1000.0), raindn[LH])
    on = G["ehr"] > 0.0
    vt = vt_hl(LH, LR)
    qhacr = 0.25 * pi * G["ehr"] * cx[LH] * qx[LR] * vt * (
        G["da0lh"] * xdia[(LH, 3)] ** 2 + G["dab1lh"][(LH, LR)] * xdia[(LH, 3)] * xdia[(LR, 3)]
        + G["da1lr"] * xdia[(LR, 3)] ** 2)
    qhacr = mn(qhacr, qxmxd[LR])
    qhacrmlr = qhacr  # iqhacrmlr >= 1
    warm = temg > tfr  # iehr0c = 0
    chacr = 0.25 * pi * G["ehr"] * cx[LH] * cx[LR] * vt * (
        G["da0lh"] * xdia[(LH, 3)] ** 2 + G["dab0lh"][(LH, LR)] * xdia[(LH, 3)] * xdia[(LR, 3)]
        + G["da0lr"] * xdia[(LR, 3)] ** 2)
    chacr = mn(chacr, G["crmxd"])
    qhacr = W(on, W(warm, zero, qhacr), zero)
    qhacrmlr = W(on, qhacrmlr, zero)
    chacr = W(on & ~warm, chacr, zero)
    # rime density of collected rain: WRF line 16939 clamps rimdn(lh) (not the raindn value computed
    # on line 16935, which is dead) -- ported literally
    raindn[LH] = W(on, W(cold, mn(mx(rimc3, rimdn[LH]), rimc4), r(1000.)), raindn[LH])
    vhacr = W(on, rho0 * qhacr / raindn[LH], zero)
    out["qhacr"], out["qhacrmlr"], out["vhacr"], out["chacr"] = qhacr, qhacrmlr, vhacr, chacr
    out["zhacr"] = zero

    # ---------------------------------------------------------------- hail collects cloud (16954)
    vtmax = G["gz"] * dtpinv
    for m in (1, 2, 3):
        vtxbar[(LHL, m)] = mn(vtmax, vtxbar[(LHL, m)])
    rarx[LHL] = zero
    on_hlw = G["ehlw"] > 0.0
    vt = jnp.abs(vtxbar[(LHL, 1)] - vtxbar[(LC, 1)])
    qhlacw = 0.25 * pi * G["ehlw"] * cx[LHL] * (qx[LC] - qcwresv) * vt * (
        G["da0lhl"] * xdia[(LHL, 3)] ** 2 + G["dab1lh"][(LHL, LC)] * xdia[(LHL, 3)] * xdia[(LC, 3)]
        + G["da1lc"] * xdia[(LC, 3)] ** 2)
    qhlacw = W(on_hlw, mn(qhlacw, 0.5 * qx[LC] * dtpinv), zero)
    qhlacwmlr = qhlacw  # iqhlacwshr = 1
    vt = (1.0 - r(C.rimdenvwgt)) * vtxbar[(LHL, 1)] + r(C.rimdenvwgt) * vtxbar[(LHL, 2)]
    rd = rimc1 * jnp.power(-((0.5) * (1.e+06) * xdia[(LC, 1)]) * ((0.60) * vt) / W(cold, temg - 273.15, -1.0), rimc2)
    rd = mn(mx(mx(r(C.hldnmn), rimc3), rd), rimc4)  # Max(hldnmn, rimc3, rimdn) left to right
    rimdn[LHL] = W(on_hlw, W(cold, rd, r(1000.)), rimdn[LHL])
    vhlacw = W(on_hlw, rho0 * qhlacw / rimdn[LHL], zero)
    out["qhlacw"], out["qhlacwmlr"], out["vhlacw"] = qhlacw, qhlacwmlr, vhlacw
    out["vhlsoak"] = zero

    # ---------------------------------------------------------------- hail collects ice (17037)
    vt = vt_hl(LHL, LI)
    qhlaci0 = 0.25 * pi * G["ehliclsn"] * cx[LHL] * qx[LI] * vt * (
        G["da0lhl"] * xdia[(LHL, 3)] ** 2 + G["dab1lh"][(LHL, LI)] * xdia[(LHL, 3)] * xdia[(LI, 3)]
        + da1[LI] * xdia[(LI, 3)] ** 2)
    on = G["ehli"] > 0.0
    out["qhlaci0"] = W(on, qhlaci0, zero)
    out["qhlaci"] = W(on, mn(G["ehli"] * qhlaci0, G["qimxd"]), zero)

    # ---------------------------------------------------------------- hail collects snow (17058)
    out["qhlacis"] = zero
    out["qhlacis0"] = zero
    vt = vt_hl(LHL, LS)
    qhlacs0 = 0.25 * pi * G["ehlsclsn"] * cx[LHL] * qx[LS] * vt * (
        G["da0lhl"] * xdia[(LHL, 3)] ** 2 + G["dab1lh"][(LHL, LS)] * xdia[(LHL, 3)] * xdia[(LS, 3)]
        + da1[LS] * xdia[(LS, 3)] ** 2)
    on = G["ehls"] > 0.0
    out["qhlacs0"] = W(on, qhlacs0, zero)
    out["qhlacs"] = W(on, mn(G["ehls"] * qhlacs0, G["qsmxd"]), zero)

    # ---------------------------------------------------------------- hail collects rain (17083)
    raindn[LHL] = W(temg > tfr, r(1000.0), raindn[LHL])
    on = G["ehlr"] > 0.0
    vt = vt_hl(LHL, LR)
    qhlacr = 0.25 * pi * G["ehlr"] * cx[LHL] * qx[LR] * vt * (
        G["da0lhl"] * xdia[(LHL, 3)] ** 2 + G["dab1lh"][(LHL, LR)] * xdia[(LHL, 3)] * xdia[(LR, 3)]
        + G["da1lr"] * xdia[(LR, 3)] ** 2)
    qhlacr = mn(qhlacr, qxmxd[LR])
    qhlacrmlr = qhlacr  # iqhlacrmlr >= 1
    warm = temg > tfr  # iehlr0c = 0
    chlacr = 0.25 * pi * G["ehlr"] * cx[LHL] * cx[LR] * vt * (
        G["da0lhl"] * xdia[(LHL, 3)] ** 2 + G["dab0lh"][(LHL, LR)] * xdia[(LHL, 3)] * xdia[(LR, 3)]
        + G["da0lr"] * xdia[(LR, 3)] ** 2)
    chlacr = mn(chlacr, G["crmxd"])
    vhlacr = rho0 * qhlacr / raindn[LHL]
    cold_on = on & ~warm
    out["qhlacr"] = W(on & ~warm, qhlacr, zero)
    out["qhlacrmlr"] = W(on, qhlacrmlr, zero)
    out["chlacr"] = W(cold_on, chlacr, zero)
    out["vhlacr"] = W(cold_on, vhlacr, zero)

    # ---------------------------------------------------------------- ice collects cloud (17141)
    vt = vt_hl(LI, LC)
    qiacw = 0.25 * pi * G["eiw"] * cx[LI] * qx[LC] * vt * (
        da0[LI] * xdia[(LI, 3)] ** 2 + dab1(LI, LC) * xdia[(LI, 3)] * xdia[(LC, 3)]
        + G["da1lc"] * xdia[(LC, 3)] ** 2)
    qiacw = W(G["eiw"] > 0.0, mn(qiacw, qxmxd[LC]), zero)
    out["qiacw"] = qiacw

    # ---------------------------------------------------------------- ice collects rain (17162)
    onA = (G["eri"] > 0.0) & (temg <= 270.15)  # iacr = 2 >= 1
    ni = W(xdia[(LI, 1)] >= 10.e-6, cx[LI] * jnp.exp(-(40.e-6 / xdia[(LI, 1)]) ** 3), zero)
    ratio = 150.e-6 / W(onA, xdia[(LR, 1)], 1.0)  # iacrsize = 5 (imurain = 1)
    nr, qr = _iacr_interp(ratio, alpha[LR], C, R, 0.0, 15., ("ciacrratio", "qiacrratio"))
    nr = nr * cx[LR]
    qr = qr * qx[LR]
    vt = vt_hl(LR, LI)
    qiacr = 0.25 * pi * G["eri"] * ni * qr * vt * (
        da0[LI] * xdia[(LI, 3)] ** 2 + G["dab1lh"][(LI, LR)] * xdia[(LH, 3)] * xdia[(LI, 3)]
        + da1[LR] * xdia[(LR, 3)] ** 2)
    qiacr = mn(G["qrmxd"], qiacr)
    ciacr = 0.25 * pi * G["eri"] * ni * nr * vt * (
        da0[LI] * xdia[(LI, 3)] ** 2 + G["dab0lh"][(LI, LR)] * xdia[(LR, 3)] * xdia[(LI, 3)]
        + da0[LR] * xdia[(LR, 3)] ** 2)
    ciacr = mn(G["crmxd"], ciacr)
    # literal WRF nesting: the "single-moment rain" ELSE belongs to the (iacr, eri, temg) test,
    # so points failing it get the single-moment qiacr formula (lines 17297-17305)
    gf4, gf5, gf6 = r(G["gf4"]), r(G["gf5"]), r(G["gf6"])
    qiacr_sm = mn(((0.25 / gf4) * pi) * G["eri"] * cx[LI] * qx[LR]
                  * jnp.abs(vtxbar[(LR, 1)] - vtxbar[(LI, 1)])
                  * (gf6 * gf1 * xdia[(LR, 2)] + 2.0 * gf5 * gf2 * xdia[(LR, 1)] * xdia[(LI, 1)]
                     + gf4 * gf3 * xdia[(LI, 2)]), G["qrmxd"])
    qiacr = W(onA, qiacr, qiacr_sm)
    ciacr = W(onA, ciacr, zero)
    ciacrf = ciacr  # iacr == 2
    # nsplinter = 0 >= 0
    csplinter = r(float(C.nsplinter)) * ciacr
    qsplinter = mn(0.1 * qiacr, csplinter * r(C.splintermass) / rho0)
    # ibiggsnow = 3: small frozen drops go to snow (frach in DOUBLE PRECISION)
    hasc = ciacr > qxmin[LH]
    xvfrz = (rho0 * qiacr / W(hasc, ciacr * 900., 1.0)).astype(F64)
    frach = 0.5 * (1. + jnp.tanh(r(0.2e12).astype(F64) * (xvfrz - (r(1.15).astype(F64) * jnp.asarray(G["xvbiggsnow"], F64)))))
    frach = W(hasc, frach, jnp.ones_like(frach))
    qiacrs = W(hasc, ((1. - frach) * qiacr.astype(F64)).astype(R), zero)
    ciacrs = W(hasc, ((1. - frach) * ciacrf.astype(F64)).astype(R), zero)
    qiacrf = (frach * qiacr.astype(F64)).astype(R)
    ciacrf = (frach * ciacrf.astype(F64)).astype(R)
    viacrf = rho0 * qiacrf / r(C.rhofrz)  # lvol(lh) > 1
    out.update(qiacr=qiacr, qiacrf=qiacrf, qiacrs=qiacrs, ciacrs=ciacrs, ciacr=ciacr, ciacrf=ciacrf,
               viacrf=viacrf, csplinter=csplinter, qsplinter=qsplinter, csplinter2=zero, qsplinter2=zero)

    # ---------------------------------------------------------------- snow aggregation (17360)
    on = (qx[LS] > qxmin[LS]) & (G["ess"] > 0.0)
    ec0 = zero + 1.0  # iessec0flag = 0
    csacs = ec0 * rvt * aa2 * G["ess"] * cx[LS] ** 2 * mn(xv[LS], 4. * r(G["pii"]) / 3. * r(C.essrmax) ** 3)
    out["csacs"] = W(on, mn(csacs, G["csmxd"]), zero)

    # ---------------------------------------------------------------- ice-cloud number (17388)
    on = (G["eiw"] > 0.0) & (xmas[LC] > 0.0)
    ciacw = qiacw * rho0 / W(on, xmas[LC], 1.0)
    out["ciacw"] = W(on, mn(ciacw, G["ccmxd"]), zero)

    # ---------------------------------------------------------------- rain-cloud number, rain self-collection (17400)
    on = (qx[LC] > qxmin[LC]) & (qx[LR] > qxmin[LR]) & (qracw > 0.0)
    halfd = 0.5 * xdia[(LR, 3)]
    c_big = aa2 * cx[LR] * (cx[LC] - ccwresv) * (xv[LC] + xv[LR])
    c_small = aa1 * cx[LR] * (cx[LC] - ccwresv) * (
        (alpha[LC] + 2.) * xv[LC] ** 2 / (alpha[LC] + 1.)
        + (alpha[LR] + 6.) * (alpha[LR] + 5.) * (alpha[LR] + 4.) * xv[LR] ** 2
        / ((alpha[LR] + 3.) * (alpha[LR] + 2.) * (alpha[LR] + 1.)))
    cracw = W(on & (halfd.astype(F64) > G["rh"]), W(halfd > rwradmn, c_big, c_small), zero)
    # self-collection (icracrthresh = 1, irainbreak = 0, icracr = 1)
    rr = qx[LR] > qxmin[LR]
    rwrad = 0.5 * xdia[(LR, 3)]
    tmp = xdia[(LR, 3)] - 0.1e-3
    brk = tmp > 1.9e-3
    ec0 = W(xdia[(LR, 3)] < 6.1e-4, r(1.0), jnp.exp(-2500.0 * (xdia[(LR, 3)] - 6.0e-4)))
    tmp1 = aa2 * cx[LR] ** 2 * xv[LR]
    cr_big = ec0 * tmp1
    tmp1s = aa1 * (cx[LR] * xv[LR]) ** 2 * (alpha[LR] + 6.) * (alpha[LR] + 5.) * (alpha[LR] + 4.) / (
        (alpha[LR] + 3.) * (alpha[LR] + 2.) * (alpha[LR] + 1.))
    cr_small = ec0 * tmp1s
    cracr = W(rwrad >= 50.e-6, cr_big, cr_small)
    ec0 = W(brk, zero, ec0)
    cracr = W(brk, zero, cracr)
    ec0 = W(rr, ec0, r(1.0))  # ec0 = 1.0 before the qr test (line 17446)
    cracr = W(rr, cracr, zero)
    out["cracw"], out["cracr"], out["ec0"] = cracw, cracr, ec0

    # ---------------------------------------------------------------- graupel/hail-cloud number (17585)
    on = (qhacw > 0.0) & (xmas[LC] > 0.0)
    chacw = qhacw * rho0 / W(on, G["xmascw"], 1.0)
    chacw = mn(chacw, 0.5 * (cx[LC] - ccwresv) * dtpinv)
    out["chacw"] = W(on, chacw, zero)
    out["qhacw"] = W(on, qhacw, zero)

    vt = vt_hl(LH, LI)
    chaci0 = 0.25 * pi * G["ehiclsn"] * cx[LH] * cx[LI] * vt * (
        G["da0lh"] * xdia[(LH, 3)] ** 2 + G["dab0lh"][(LH, LI)] * xdia[(LH, 3)] * xdia[(LI, 3)]
        + da0[LI] * xdia[(LI, 3)] ** 2)
    on = G["ehi"] > 0.0
    out["chaci0"] = W(on, chaci0, zero)
    out["chaci"] = W(on, mn(G["ehi"] * chaci0, G["cimxd"]), zero)

    vt = vt_hl(LH, LS)
    chacs0 = 0.25 * pi * G["ehsclsn"] * cx[LH] * cx[LS] * vt * (
        G["da0lh"] * xdia[(LH, 3)] ** 2 + G["dab0lh"][(LH, LS)] * xdia[(LH, 3)] * xdia[(LS, 3)]
        + da0[LS] * xdia[(LS, 3)] ** 2)
    on = G["ehs"] > 0
    out["chacs0"] = W(on, chacs0, zero)
    out["chacs"] = W(on, mn(G["ehs"] * chacs0, G["csmxd"]), zero)

    on = (qhlacw > 0.0) & (xmas[LC] > 0.0)
    chlacw = qhlacw * rho0 / W(on, G["xmascw"], 1.0)
    chlacw = mn(chlacw, 0.5 * cx[LC] * dtpinv)
    out["chlacw"] = W(on, chlacw, zero)
    out["qhlacw"] = W(on, qhlacw, zero)
    qhlacw = out["qhlacw"]

    vt = vt_hl(LHL, LI)
    chlaci0 = 0.25 * pi * G["ehliclsn"] * cx[LHL] * cx[LI] * vt * (
        G["da0lhl"] * xdia[(LHL, 3)] ** 2 + dab0(LHL, LI) * xdia[(LHL, 3)] * xdia[(LI, 3)]
        + da0[LI] * xdia[(LI, 3)] ** 2)
    on = G["ehli"] > 0.0
    out["chlaci0"] = W(on, chlaci0, zero)
    out["chlaci"] = W(on, mn(G["ehli"] * chlaci0, G["cimxd"]), zero)

    vt = vt_hl(LHL, LS)
    chlacs0 = 0.25 * pi * G["ehlsclsn"] * cx[LHL] * cx[LS] * vt * (
        G["da0lhl"] * xdia[(LHL, 3)] ** 2 + dab0(LHL, LS) * xdia[(LHL, 3)] * xdia[(LS, 3)]
        + da0[LS] * xdia[(LS, 3)] ** 2)
    on = G["ehls"] > 0.0
    out["chlacs0"] = W(on, chlacs0, zero)
    out["chlacs"] = W(on, mn(G["ehls"] * chlacs0, G["csmxd"]), zero)

    # ---------------------------------------------------------------- autoconversion (17798)
    # dmrauto = 0 (>= -1), dmropt = 0, ipconc < 6
    on = (qx[LC] > qxmin[LC]) & (cx[LC] > 1000.) & (temg > r(C.tfrh) + 4.)
    cautn = mn(G["ccmxd"], ((alpha[LC] + 2.) / (alpha[LC] + 1.)) * aa1 * cx[LC] ** 2 * xv[LC] ** 2).astype(F64)
    cautn = mx(0.0, cautn)
    rb = G["rb"]
    big = rb > 7.51e-6
    t2s = (r(3.72).astype(F64) / (1.e6 * (rb - 7.500e-6) * rho0.astype(F64) * qx[LC].astype(F64)))
    t2s = W(big, t2s, 1.0)
    xl2p = G["xl2p"]
    qrcnw = mx(0.0, xl2p / (t2s * rho0.astype(F64))).astype(R)
    crcnw = mx(0.0, mn((r(3.5e9).astype(F64) * xl2p / t2s), (0.5 * cautn))).astype(R)
    sw = ((qx[LR] * rho0).astype(F64) > r(1.2).astype(F64) * xl2p) & (cx[LR] > r(C.cxmin))
    crcnw = W(sw, cx[LR] / qx[LR] * qrcnw, crcnw)
    qrcnw = W(crcnw < 1.e-30, zero, qrcnw)
    act = on & big
    out["qrcnw"] = W(act, qrcnw, zero)
    out["crcnw"] = W(act, crcnw, zero)
    out["cautn"] = W(on, cautn, jnp.zeros_like(cautn))
    out["zrcnw"] = zero

    # ---------------------------------------------------------------- Bigg freezing of rain (18049)
    out.update(_bigg_rain(G, C, R, out, dtp, dtpinv, qxmin))

    # ---------------------------------------------------------------- Bigg freezing of droplets (18437)
    out.update(_bigg_droplets(G, C, R, dtpinv, qxmin))

    # ---------------------------------------------------------------- contact freezing (18529)
    out.update(_contact_freezing(G, C, R, dtpinv, qxmin))

    # ---------------------------------------------------------------- Hallett-Mossop / ice -> snow (18631, 18705)
    for k in ("ciihr", "qiihr", "cicichr", "qicichr", "cipiphr", "qipiphr"):  # ihrn = 0
        out[k] = zero
    for k in ("qscnvi", "cscnvi", "cscnvis"):  # ipconc >= 4 with the .false. guard -> nothing
        out[k] = zero

    # ---------------------------------------------------------------- ventilation (18748)
    out.update(_ventilation(G, C, R, qxmin, r))

    # ---------------------------------------------------------------- wet growth / melting / deposition coefficients (19030)
    felv, felf, fcw, fci, ftka, fwvdf = G["felv"], G["felf"], G["fcw"], G["fci"], G["ftka"], G["fwvdf"]
    qss0 = G["qss0"]
    out["fwet1"] = (2.0 * pi) * (felv * fwvdf * rho0 * (qss0 - qx[LV]) - ftka * temcg) / (rho0 * (felf + fcw * temcg))
    out["fwet2"] = (1.0) - fci * temcg / (felf + fcw * temcg)
    fmlt1 = (2.0 * pi) * (felv * fwvdf * (qss0 - qx[LV]) - ftka * temcg / rho0) / (felf)
    out["fmlt1"] = fmlt1
    out["fmlt2"] = -fcw * temcg / felf
    out["fmlt1e"] = (2.0 * pi) * (felv * fwvdf * (qss0 - qx[LV])) / (felf)
    fvds = (4.0 * pi / rho0) * (G["ssi"] - 1.0) * (1.0 / (G["fai"] + G["fbi"]))
    out["fvds"] = fvds
    out["fvce"] = (4.0 * pi / rho0) * (G["ssw"] - 1.0) * (1.0 / (G["fav"] + G["fbv"]))

    # ---------------------------------------------------------------- melting (19069)
    out.update(_melting(G, C, R, out, dtp, dtpinv, qxmin, r))

    # ---------------------------------------------------------------- capacitances and deposition (19422)
    out.update(_deposition(G, C, R, out, dtpinv, qxmin, r))

    for k, v in out.items():
        G[k] = v
    G["vtxbar"] = vtxbar
    G["rimdn"] = rimdn
    G["raindn"] = raindn
    G["rarx"] = rarx
    return G


def _iacr_interp(ratio, alp, C, R, alp_lo, alp_hi, names):
    """Bilinear ciacrratio/qiacrratio/ziacrratio lookups (lines 17192-17214, 18097-18122):
    i = Min(nqiacrratio, Int(ratio*dqiacrratioinv)), j = Int(Max(alp_lo, Min(alp_hi, alp))*dqiacralphainv);
    table elements from mathfun.iacr_table (init tables rebuilt on the fly)."""
    ratio = jnp.asarray(ratio, R)
    alp = jnp.asarray(alp, R)
    dqr, dqa = jnp.asarray(C.dqiacrratio, R), jnp.asarray(C.dqiacralpha, R)
    dqri, dqai = jnp.asarray(C.dqiacrratioinv, R), jnp.asarray(C.dqiacralphainv, R)
    i = jnp.minimum(C.nqiacrratio, jnp.trunc(ratio * dqri).astype(jnp.int32))
    j = jnp.trunc(jnp.maximum(jnp.asarray(alp_lo, R), jnp.minimum(jnp.asarray(alp_hi, R), alp)) * dqai).astype(jnp.int32)
    delx = ratio - i.astype(R) * dqr
    dely = alp - j.astype(R) * dqa
    ip1 = jnp.minimum(i + 1, C.nqiacrratio)
    jp1 = jnp.minimum(j + 1, C.nqiacralpha)
    out = []
    for name in names:
        t = lambda ii, jj: iacr_table(name, ii, jj, R)  # noqa: E731
        tmp1 = t(i, j) + delx * dqri * (t(ip1, j) - t(i, j))
        tmp2 = t(i, jp1) + delx * dqri * (t(ip1, jp1) - t(i, jp1))
        out.append(tmp1 + dely * dqai * (tmp2 - tmp1))
    return out


def _bigg_rain(G, C, R, out, dtp, dtpinv, qxmin):
    """Bigg freezing of rain, ipconc >= 3 with ibiggopt = 2, imurain = 1 (lines 18049-18430)."""
    W, mn = jnp.where, jnp.minimum
    r = lambda x: jnp.asarray(x, R)  # noqa: E731
    qx, cx, alpha, xdia, temcg, rho0 = G["qx"], G["cx"], G["alpha"], G["xdia"], G["temcg"], G["rho0"]
    zero = jnp.zeros_like(rho0)
    res = {k: zero for k in ("qrfrz", "qrfrzs", "qrfrzf", "vrfrzf", "crfrz", "crfrzs", "crfrzf",
                             "zrfrz", "zrfrzs", "zrfrzf", "qwcnr")}
    on = (qx[LR] > qxmin[LR]) & (temcg < -5.)  # ibiggopt = 2 > 0
    volt = jnp.exp(16.2 + 1.0 * temcg) * 1.0e-6
    dbigg = jnp.power(6. / r(C.pi) * volt, r(1.) / r(3.))
    dig = on & (dbigg < 8.e-3)
    ratio = mn(r(C.maxratiolu), dbigg / xdia[(LR, 1)])
    # alp0flag = .false.: j uses Max(minalphalu, Min(maxalphalu, alpha))
    nfrac, qfrac = _iacr_interp(ratio, alpha[LR], C, R, C.minalphalu, C.maxalphalu, ('ciacrratio', 'qiacrratio'))
    crfrz0 = nfrac * cx[LR] * dtpinv
    qrfrz0 = qfrac * qx[LR] * dtpinv
    crfrzf = crfrz0  # line 18115: set BEFORE the threshold test and not reset by it (WRF quirk)
    keep = ~((qrfrz0 * dtp < qxmin[LH]) | (crfrz0 * dtp < r(C.cxmin)))
    crfrz = W(keep, crfrz0, zero)
    qrfrz = W(keep, qrfrz0, zero)
    qrfrzf = qrfrz
    # ipconc >= 5, lzr = 0: rescale FD number to keep the converted reflectivity (lines 18134-18169)
    cxd1 = crfrz * dtp
    qxd1 = qrfrz * dtp
    (zfrac,) = _iacr_interp(ratio, alpha[LR], C, R, C.minalphalu, C.maxalphalu, ('ziacrratio',))
    g1xr = G["g1x"][LR]
    xdnr = G["xdn"][LR]
    pi = r(C.pi)
    tmp5 = g1xr * (rho0 * qx[LR]) ** 2 / ((pi * xdnr / 6.) ** 2 * cx[LR])
    zxd1 = zfrac * tmp5
    tmp3 = g1xr * (rho0 * qxd1) ** 2 / ((pi * xdnr / 6.0) ** 2)
    tmp4 = tmp3 / W(keep, cxd1, 1.0)
    crfrzf = W(keep & (tmp4 > zxd1), tmp3 / zxd1 * dtpinv, crfrzf)
    # ibiggsnow = 3: small Bigg-frozen drops to snow (ibiggsmallrain = 0)
    small = dbigg < mx_(r(C.biggsnowdiam), mx_(r(C.dfrz), r(C.dhmn)))
    crfrzs = W(keep & small, crfrz, zero)
    qrfrzs = W(keep & small, qrfrz, zero)
    ratio2 = mn(r(C.maxratiolu), mx_(r(C.dfrz), r(C.dhmn)) / xdia[(LR, 1)])
    nf2, qf2 = _iacr_interp(ratio2, alpha[LR], C, R, C.minalphalu, C.maxalphalu, ('ciacrratio', 'qiacrratio'))
    crfrzf = W(keep & small, nf2 * cx[LR] * dtpinv, crfrzf)
    qrfrzf = W(keep & small, qf2 * qx[LR] * dtpinv, qrfrzf)
    crfrzs = W(keep & small, crfrzs - crfrzf, crfrzs)
    qrfrzs = W(keep & small, qrfrzs - qrfrzf, qrfrzs)
    # limit to available rain (lines 18260-18272)
    lim = keep & (qrfrz * dtp > qx[LR])
    fac = qrfrz * dtp / qx[LR]
    res["qrfrz"] = W(dig, W(lim, fac * qrfrz, qrfrz), zero)
    res["qrfrzs"] = W(dig, W(lim, fac * qrfrzs, qrfrzs), zero)
    res["qrfrzf"] = W(dig, W(lim, fac * qrfrzf, qrfrzf), zero)
    res["crfrz"] = W(dig, W(lim, fac * crfrz, crfrz), zero)
    res["crfrzs"] = W(dig, W(lim, fac * crfrzs, crfrzs), zero)
    res["crfrzf"] = W(dig, W(lim, fac * crfrzf, crfrzf), zero)
    res["vrfrzf"] = rho0 * res["qrfrzf"] / r(C.rhofrz)  # lvol(lh) > 1
    # nsplinter = 0: no splinters from freezing rain
    return res


def mx_(a, b):
    return jnp.maximum(a, b)


def _bigg_droplets(G, C, R, dtpinv, qxmin):
    """Bigg freezing of cloud droplets, ibfc = 1, ipconc >= 2 (lines 18437-18522)."""
    W = jnp.where
    r = lambda x: jnp.asarray(x, R)  # noqa: E731
    qx, cx, alpha, xv, xdia, temg, temcg = G["qx"], G["cx"], G["alpha"], G["xv"], G["xdia"], G["temg"], G["temcg"]
    rhoinv = G["rhoinv"]
    zero = jnp.zeros_like(temg)
    on = (temg < 268.15) & (qx[LC] > qxmin[LC]) & (cx[LC] > r(C.cxmin)) & (xdia[(LC, 3)] > 0.e-6)
    volt = jnp.exp(16.2 + 1.0 * temcg) * 1.0e-6
    xvs = W(on, xv[LC], 1.0)
    a0 = alpha[LC] == 0.0
    cw_a0 = cx[LC] * jnp.exp(-volt / xvs) * dtpinv
    qw_a0 = cw_a0 * r(C.xdn0[LC]) * rhoinv * (volt + xv[LC])
    ratio = (1. + alpha[LC]) * volt / xvs
    ratio = jnp.minimum(r(C.maxratiolu), ratio)
    t1 = gaminterp(ratio, alpha[LC], 1, 1, R, C.bx[LH], C.bx[LHL])
    cw_g = cx[LC] * t1 * dtpinv
    t12 = gaminterp(ratio, alpha[LC], 12, 1, R, C.bx[LH], C.bx[LHL])
    qw_g = cx[LC] * r(C.xdn0[LC]) * xv[LC] * rhoinv * dtpinv * t12
    cwfrz = W(on, W(a0, cw_a0, cw_g), zero)
    qwfrz = W(on, W(a0, qw_a0, qw_g), zero)
    # the 'temg > 268.15' reset (line 18501) is unreachable inside temg < 268.15
    return dict(qwfrz=qwfrz, cwfrz=cwfrz, qwfrzc=qwfrz, cwfrzc=cwfrz, qwfrzp=zero, cwfrzp=zero)


def _contact_freezing(G, C, R, dtpinv, qxmin):
    """Contact freezing, icfn = 2 (Cotton/Meyers), lines 18529-18624."""
    W = jnp.where
    r = lambda x: jnp.asarray(x, R)  # noqa: E731
    qx, cx, xdia, xmas, temg, temcg, rho0, pres = G["qx"], G["cx"], G["xdia"], G["xmas"], G["temg"], G["temcg"], G["rho0"], G["pres"]
    pi = r(C.pi)
    zero = jnp.zeros_like(temg)
    on = (temg < 271.15) & (qx[LC] > qxmin[LC])
    raero, kaero, kb = r(_RAERO), r(_KAERO), r(_KB)
    ccia = jnp.exp(4.11 - (0.262) * temcg)
    knud = 2.28e-5 * temg / (pres * raero)
    knuda = 1.257 + 0.4 * jnp.exp(-1.1 / knud)
    gtp = 1. / (G["fai"] + G["fbi"])
    dfar = kb * temg * (1. + knuda * knud) / (6. * pi * G["fadvisc"] * raero)
    fn1 = 2. * pi * xdia[(LC, 1)] * cx[LC] * ccia
    fn2 = -gtp * (G["ssw"] - 1.) * G["felv"] / pres
    ftka = G["ftka"]
    fnft = 0.4 * (1. + 1.45 * knud + 0.4 * knud * jnp.exp(-1. / knud)) * (ftka + 2.5 * knud * kaero) \
        / ((1. + 3. * knud) * (2 * ftka + 5. * knud * kaero + kaero))
    ctfzbd = fn1 * dfar
    ctfzth = fn1 * fn2 * fnft / rho0
    ctfzdi = fn1 * fn2 * r(C.rw) * temg / (G["felv"] * rho0)
    cwctfz = jnp.maximum(ctfzbd + ctfzth + ctfzdi, 0.)
    cwctfz = jnp.minimum(cwctfz * dtpinv, G["ccmxd"])  # ipconc >= 2
    qwctfz = xmas[LC] * cwctfz / rho0
    cwctfz = W(on, cwctfz, zero)
    qwctfz = W(on, qwctfz, zero)
    return dict(ccia=W(on, ccia, zero), cwctfz=cwctfz, qwctfz=qwctfz, cwctfzc=cwctfz, qwctfzc=qwctfz,
                cwctfzp=zero, qwctfzp=zero)


def _ventilation(G, C, R, qxmin, r):
    """Ventilation coefficients (lines 18748-19023)."""
    W = jnp.where
    qx, xdia, vtxbar, alpha, xdn, rho0 = G["qx"], G["xdia"], G["vtxbar"], G["alpha"], G["xdn"], G["rho0"]
    zero = jnp.zeros_like(rho0)
    fschm, fakvisc = G["fschm"], G["fakvisc"]
    fvent = (fschm ** (r(1.) / r(3.))) * (fakvisc ** (-0.5))
    # cloud ice (icond = 1)
    civenta, civentb, civentc, civentd = r(1.258e4), r(2.331), r(5.662e4), r(2.373)
    civente, civentf, civentg = r(0.8241), r(-0.042), r(1.70)
    d = W(qx[LI] > qxmin[LI], xdia[(LI, 1)], 1.0)
    cireyn = (civenta * d ** civentb + civentc * d ** civentd) / (civente * d ** civentf + civentg)
    xcivent = (fschm ** (r(1.) / r(3.))) * ((cireyn / fakvisc) ** 0.5)
    civent = W(xcivent < 1.0, 1.0 + 0.14 * xcivent ** 2, 0.86 + 0.28 * xcivent)
    civent = W(qx[LI] > qxmin[LI], civent, zero)
    # rain: ipconc >= 3, imurain = 1, iferwisventr = 2 (ipconc < 7: no rwventz)
    ar, br = r(C.ar), r(C.br)
    x = 1. + alpha[LR]
    rwvent = (0.78 * x + 0.308 * G["ventrxn"] * fvent * jnp.sqrt((ar * G["rhovt"])) * (xdia[(LR, 1)] ** ((1.0 + br) / 2.0)))
    rwvent = W(qx[LR] > qxmin[LR], rwvent, zero)
    # snow: ipconc >= 4
    swvent = 0.65 + 0.44 * fvent * jnp.sqrt(vtxbar[(LS, 1)] * xdia[(LS, 1)])
    swvent = W(qx[LS] > qxmin[LS], swvent, zero)
    # graupel / hail (Ferrier 1994 B.36 unless alpha == 0)
    gmoi = jnp.asarray(C.gmoi, F64)
    gr = r(C.gr)
    hwventa = (r(0.78).astype(F64) * gmoi[200]).astype(R)   # (0.78)*gmoi(igmhwa), igmhwa = 200
    hwventb = (r(0.308).astype(F64) * gmoi[275]).astype(R)  # (0.308)*gmoi(igmhwb), igmhwb = 275
    res = dict(fvent=fvent, civent=civent, rwvent=rwvent, swvent=swvent)
    for il, kv, ky in ((LH, "hwvent", "hwventy"), (LHL, "hlvent", "hlventy")):
        on = qx[il] > qxmin[il]
        hwventc = (4.0 * gr / (3.0 * G["cdxgs"][il])) ** (0.25)
        v0 = (hwventa + hwventb * hwventc * fvent * ((xdn[il] / rho0) ** (0.25)) * (xdia[(il, 1)] ** (0.75)))
        x = 1. + alpha[il]
        g1palp = _gmoi_interp(gmoi, 1 + alpha[il], C)
        y = (_gmoi_interp(gmoi, 2.5 + alpha[il] + 0.5 * G["bxx"][il], C) / g1palp).astype(R)
        venty = 0.308 * fvent * (xdia[(il, 1)] ** (0.5 + 0.5 * G["bxx"][il])) * jnp.sqrt(G["axx"][il] * G["rhovt"])
        v1 = (0.78 * x + y * venty)
        a0 = alpha[il] == 0.0
        res[kv] = W(on, W(a0, v0, v1), zero)
        res[ky] = W(on & ~a0, venty, zero)
    return res


def _gmoi_interp(gmoi, tmp, C):
    """gmoi(i) + (gmoi(i+1)-gmoi(i))*del*dgami with i = Int(dgami*tmp), del = tmp - dgam*i (DOUBLE)."""
    tmp_r = tmp
    dgami = jnp.asarray(C.dgami, F64)
    dgam = jnp.asarray(C.dgam, F64)
    i = jnp.trunc(dgami * tmp_r.astype(F64)).astype(jnp.int32)
    dl = tmp_r.astype(F64) - dgam * i
    # del is a REAL local: rounded to REAL before use
    dl = dl.astype(tmp_r.dtype).astype(F64)
    gi = gmoi[i]
    gi1 = gmoi[i + 1]
    return gi + (gi1 - gi) * dl * dgami


def _melting(G, C, R, out, dtp, dtpinv, qxmin, r):
    """Snow/graupel/hail melting and melt number tendencies (lines 19069-19417), mixedphase false."""
    W, mn, mx = jnp.where, jnp.minimum, jnp.maximum
    qx, cx, xv, xdn, xdia, temg, rho0 = G["qx"], G["cx"], G["xv"], G["xdn"], G["xdia"], G["temg"], G["rho0"]
    zero = jnp.zeros_like(rho0)
    fmlt1, fmlt2 = out["fmlt1"], out["fmlt2"]
    names0 = ("qsmlr", "qimlr", "qhmlr", "qhlmlr", "qhfzh", "qffzf", "qhlfzhl", "qhfzhlg", "qhlfzhllg", "vhfzh",
              "vffzf", "vhlfzhl", "qsfzs", "zhmlr", "zhmlrr", "zsmlrr", "zhshr", "zhlmlr", "zhlshr", "zhshrr",
              "zhlmlrr", "zhlshrr", "csmlr", "csmlrr", "chmlr", "chmlrr", "chlmlr", "chlfmlr", "chlmlrr", "qhmlh")
    res = {k: zero for k in names0}
    warm = temg > r(C.tfr)
    meltfac = r(C.meltfac)
    xdnmx = {LH: r(C.xdnmx[LH]), LHL: r(C.xdnmx[LHL])}
    # snow
    qsmlr = mn(r(C.c1sw) * fmlt1 * cx[LS] * out["swvent"] * xdia[(LS, 1)], 0.0)
    qsmlr = W(warm & (qx[LS] > qxmin[LS]), qsmlr, zero)
    # graupel
    onh = warm & (qx[LH] > qxmin[LH])
    qhmlr = meltfac * mn(fmlt1 * cx[LH] * out["hwvent"] * xdia[(LH, 1)] + fmlt2 * (out["qhacrmlr"] + out["qhacwmlr"]), 0.0)
    qhmlr = W(onh, qhmlr, zero)
    vx = G["vx"]
    soak = onh & (qhmlr < 0.0) & (xdn[LH] < xdnmx[LH])
    v1 = (1. - xdn[LH] / xdnmx[LH]) * (vx[LH] + rho0 * qhmlr / xdn[LH]) / (dtp)
    v2 = -1.0 * rho0 * qhmlr / xdnmx[LH]
    vhsoak = W(soak, mn(v1, v2), out["vhsoak"])
    # hail (lhlw < 1)
    onl = warm & (qx[LHL] > qxmin[LHL])
    qhlmlr = meltfac * mn(fmlt1 * cx[LHL] * out["hlvent"] * xdia[(LHL, 1)] + fmlt2 * (out["qhlacrmlr"] + out["qhlacwmlr"]), 0.0)
    qhlmlr = W(onl, qhlmlr, zero)
    soak = onl & (qhlmlr < 0.0) & (xdn[LHL] < xdnmx[LHL])
    v1 = (1. - xdn[LHL] / xdnmx[LHL]) * (vx[LHL] + rho0 * qhlmlr / xdn[LHL]) / (dtp)
    v2 = -1.0 * rho0 * qhlmlr / xdnmx[LHL]
    vhlsoak = W(soak, mn(v1, v2), out["vhlsoak"])
    # limiters (applied to every point, lines 19201-19215)
    qsmlr = mx(qsmlr, mn(-G["qsmxd"], -0.7 * qx[LS] * dtpinv))
    qhmlr = mx(qhmlr, mn(-G["qhmxd"], -0.95 * qx[LH] * dtpinv))
    chmlr = mx(zero, mn(-G["chmxd"], -0.95 * cx[LH] * dtpinv))
    qhlmlr = mx(qhlmlr, mn(-G["qxmxd"][LHL], -0.95 * qx[LHL] * dtpinv))
    chlmlr = mx(zero, mn(-G["cxmxd"][LHL], -0.95 * cx[LHL] * dtpinv))
    # number tendencies (ipconc >= 1)
    cimlr = (cx[LI] / (qx[LI] + 1.e-20)) * zero  # qimlr = 0
    c1 = (xdia[(LS, 1)] > 1.e-6) & (-qsmlr >= 0.5 * qxmin[LS])  # ipconc >= 4
    c2 = qx[LS] > qxmin[LS]
    csm = (cx[LS] / W(c1 | c2, qx[LS], 1.0)) * qsmlr
    csmlr = W(c1 | c2, csm, zero)
    csmlrr = csmlr / G["rzxs"]
    # graupel: ibinhmlr = 0, imltshddmr = 1 (no imltshddmr == 3 correction)
    chmlr = (cx[LH] / (qx[LH] + 1.e-20)) * qhmlr
    vr3mm = r(_VR3MM_BASE) * (r(3.0) / r(1.)) ** 3
    sheddiam, sheddiam0 = r(C.sheddiam), r(C.sheddiam0)
    xvmxr = r(C.xvmx[LR])
    tmp = -rho0 * qhmlr / (mn(xdn[LR] * xvmxr, xdn[LH] * xv[LH]))
    tmp2 = -rho0 * qhmlr / (xdn[LR] * vr3mm)
    cr = tmp * (sheddiam0 - xdia[(LH, 3)]) / (sheddiam0 - sheddiam) + tmp2 * (xdia[(LH, 3)] - sheddiam) / (sheddiam0 - sheddiam)
    cr = -mx(tmp, mn(tmp2, cr))
    neg = chmlr < 0.0
    chmlrr = W(neg, W(xv[LH] > 0.0, cr, chmlr), zero)  # ihmlt = 2 (else-branch: chmlrr = chmlr)
    # hail
    onl = qhlmlr < 0.0
    chlm = (cx[LHL] / (qx[LHL] + 1.e-20)) * qhlmlr
    chlmlr = W(onl, chlm, chlmlr)
    tmp = -rho0 * qhlmlr / (mn(xdn[LR] * xvmxr, xdn[LHL] * xv[LHL]))
    tmp2 = -rho0 * qhlmlr / (xdn[LR] * vr3mm)
    cr = tmp * (20.e-3 - xdia[(LHL, 3)]) / (20.e-3 - sheddiam) + tmp2 * (xdia[(LHL, 3)] - sheddiam) / (20.e-3 - sheddiam)
    cr = -mx(tmp, mn(tmp2, cr))
    chlmlrr = W(onl, W((xv[LHL] > 0.0) & (chlmlr < 0.0), cr, chlmlr), zero)
    res.update(qsmlr=qsmlr, qhmlr=qhmlr, qhlmlr=qhlmlr, chmlr=chmlr, chlmlr=chlmlr, cimlr=cimlr,
               csmlr=csmlr, csmlrr=csmlrr, chmlrr=chmlrr, chlmlrr=chlmlrr, vhsoak=vhsoak, vhlsoak=vhlsoak)
    return res


def _deposition(G, C, R, out, dtpinv, qxmin, r):
    """Capacitances, depositional growth coefficients and the DoSublimationFix qlimit loop (19422-19695)."""
    W, mn, mx = jnp.where, jnp.minimum, jnp.maximum
    qx, cx, xdia, temg, rho0 = G["qx"], G["cx"], G["xdia"], G["temg"], G["rho0"]
    zero = jnp.zeros_like(rho0)
    rwcap = (0.5) * xdia[(LR, 1)]
    swcap = (0.5) * xdia[(LS, 1)]
    hwcap = (0.5) * xdia[(LH, 1)]
    hlcap = (0.5) * xdia[(LHL, 1)]
    oni = (qx[LI] > qxmin[LI]) & (xdia[(LI, 1)] > 0.0)
    cval = W(oni, xdia[(LI, 1)], 1.0)
    cilen = 0.4764 * (cval) ** (0.958)
    aval = cilen
    eval_ = jnp.sqrt(1.0 - (aval ** 2) / (cval ** 2))
    fval = mn(0.99, eval_)
    gval = jnp.log(jnp.abs((1. + fval) / (1. - fval)))
    cicap = W(oni, cval * fval / gval, zero)
    cilen = W(oni, cilen, G["cilen"] if "cilen" in G else zero)
    fvds = out["fvds"]
    depfac = r(C.depfac)
    qidsv = fvds * cx[LI] * out["civent"] * cicap * depfac  # icond = 1
    qsdsv = fvds * cx[LS] * out["swvent"] * swcap * depfac
    qhdsv = fvds * cx[LH] * out["hwvent"] * hwcap * depfac
    qhldsv = fvds * cx[LHL] * out["hlvent"] * hlcap * depfac
    res = dict(rwcap=rwcap, swcap=swcap, hwcap=hwcap, hlcap=hlcap, cilen=cilen, cicap=cicap,
               qidsv=qidsv, qsdsv=qsdsv, qhdsv=qhdsv, qhldsv=qhldsv)

    # ---- DoSublimationFix = .true. (lines 19485-19686)
    qitmp = qx[LI] + qx[LS] + qx[LH]
    qitmp = qitmp + qx[LHL]  # lis = 0, lhl > 1
    qrtmp = qx[LR]
    qctmp = qx[LC]
    qsimxdep = zero
    qsimxsub = zero
    dqcitmp = zero
    act = qitmp > qxmin[LI]
    qitmp1 = qitmp
    felvcp, felscp, pi0 = G["felvcp"], G["felscp"], G["pi0"]
    qvtmp = qx[LV]
    qss = G["qvs"]
    qisstmp = G["qis"]
    thetaptmp = G["thetap"]
    qvptmp = zero
    qsstmp = qisstmp
    dqwvtmp = (qvtmp - qsstmp)
    tfr = r(C.tfr)
    cbi, cbw = r(C.cbi), r(C.cbw)
    dqcw = zero
    dqci = zero
    dqwv = zero
    dqvcnd = G["dqvcnd"] if "dqvcnd" in G else zero
    fracl = G["fracl"] if "fracl" in G else zero
    fraci = G["fraci"] if "fraci" in G else zero
    for itertd in (1, 2):
        if itertd == 2:
            dqcitmp = dqci
        dqcw = zero
        dqci = zero
        dqwv = (qvtmp - qsstmp)
        sub = dqwv < 0.
        enough = qitmp > -dqwv
        dqci_s = W(enough, dqwv, -qitmp)
        dqwv_s = W(enough, zero, dqwv + qitmp)
        qvptmp_s = qvptmp - (dqcw + dqci_s)
        qctmp_s = qctmp + dqcw
        qitmp_s = qitmp + dqci_s
        thetaptmp_s = thetaptmp + 1. / pi0 * (felvcp * dqcw + felscp * dqci_s)
        dqci = W(sub, dqci_s, dqci)
        dqwv = W(sub, dqwv_s, dqwv)
        qvptmp = W(sub, qvptmp_s, qvptmp)
        qctmp = W(sub, qctmp_s, qctmp)
        qitmp = W(sub, qitmp_s, qitmp)
        thetaptmp = W(sub, thetaptmp_s, thetaptmp)
        # supersaturated (or exactly zero after the subsaturated update)
        dep = dqwv >= 0.
        fracl_d = zero
        fraci_d = zero + 1.0
        # 'temg < tfr .and. temg > thnuc' branch body is empty; temg <= thnuc sets the same values
        dqvcnd_d = dqwv / (1. + G["fcqv2"] * qsstmp / ((temg - cbi) ** 2))
        dqvcnd_d = W(temg >= tfr, dqwv / (1. + G["fcqv1"] * qsstmp / ((temg - cbw) ** 2)), dqvcnd_d)
        dqcw_d = dqvcnd_d * fracl_d
        dqci_d = dqvcnd_d * fraci_d
        thetaptmp_d = thetaptmp + (felvcp * dqcw_d + felscp * dqci_d) / (pi0)
        qvptmp_d = qvptmp - (dqvcnd_d)
        qctmp_d = qctmp + dqcw_d
        qitmp_d = qitmp + dqci_d
        fracl = W(dep, fracl_d, fracl)
        fraci = W(dep, fraci_d, fraci)
        dqvcnd = W(dep, dqvcnd_d, dqvcnd)
        dqcw = W(dep, dqcw_d, dqcw)
        dqci = W(dep, dqci_d, dqci)
        thetaptmp = W(dep, thetaptmp_d, thetaptmp)
        qvptmp = W(dep, qvptmp_d, qvptmp)
        qctmp = W(dep, qctmp_d, qctmp)
        qitmp = W(dep, qitmp_d, qitmp)
        if itertd == 1:
            thetatmp = thetaptmp + G["theta0"]
            temgtmp = thetatmp * G["pk"]
            qvaptmp = mx((qvptmp + G["qv0"]), 0.0)
            ltemq = jnp.trunc((temgtmp - 163.15) / r(C.fqsat) + 1.5).astype(jnp.int32)
            ltemq = mn(C.nqsat, mx(1, ltemq))
            qisstmp = G["pqs"] * tabqis(ltemq, C, R)
            qctmp = mx(0.0, qctmp)
            qitmp = mx(0.0, qitmp)
            qvtmp = mx(0.0, qvaptmp)
            qsstmp = qisstmp
        else:
            qctmp = mx(0.0, qctmp)
            qitmp = mx(0.0, qitmp)
            qsimxsub = W(qitmp < qitmp1, (qitmp1 - qitmp) * dtpinv, qsimxsub)
            qsimxdep = W(qitmp > qitmp1, (qitmp - qitmp1) * dtpinv, qsimxdep)
    keep = lambda new, old: W(act, new, old)  # noqa: E731
    res.update(
        qitmp=W(act, qitmp, qitmp1), qrtmp=qrtmp, qctmp=W(act, qctmp, qx[LC]),
        qsimxdep=W(act, qsimxdep, zero), qsimxsub=W(act, qsimxsub, zero), dqcitmp=W(act, dqcitmp, zero),
        qvtmp=keep(qvtmp, G["qvtmp"] if "qvtmp" in G else zero), dqwvtmp=keep(dqwvtmp, G["dqwvtmp"] if "dqwvtmp" in G else zero),
        dqcw=keep(dqcw, G["dqcw"] if "dqcw" in G else zero), dqci=keep(dqci, G["dqci"] if "dqci" in G else zero),
        dqwv=keep(dqwv, G["dqwv"] if "dqwv" in G else zero), fracl=keep(fracl, G["fracl"] if "fracl" in G else zero),
        fraci=keep(fraci, G["fraci"] if "fraci" in G else zero), dqvcnd=keep(dqvcnd, G["dqvcnd"] if "dqvcnd" in G else zero),
        qss=keep(qss, G["qss"]),
    )
    return res
