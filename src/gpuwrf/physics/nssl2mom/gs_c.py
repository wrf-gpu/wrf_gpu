"""nssl_2mom_gs PART C (WRF ``module_mp_nssl_2mom.F`` lines 19696-25151), default mp=18.

Evaporation/sublimation limits, ice->snow conversion, wet growth and shedding, soaking,
ice/snow -> graupel and graupel -> hail conversion, rain-freezing heat budget, rain
evaporation, Hallett-Mossop, primary ice nucleation, number / mass / volume tendencies with
their depletion limiters, latent heating, state update, melting of cloud ice above 0C, the
2-moment size limiter and the scatter back to ``an`` / ``t0``.

Point-wise and vectorised: every per-point quantity in ``G`` has the shape of the point set
(the full grid in the product, with ``G['gathered']`` selecting the points nssl_2mom_gs gathers;
in the oracle tests the full column).  The scatter writes ONLY where ``G['gathered']`` is true;
every other grid point keeps its input value.

Branches dead for the default mp=18 configuration are omitted (flag named at each site):
icond=1, iscni=4, incwet=0, ihlcnh=3, iusedw=0, icvhl2h=0, iglcnvi=1, iglcnvs=2, irwfrz=1,
rcond=2, icrcev=1, lhwlg=lhlwlg=0, itype1=0, itype2=2, isnwfrac=0, icenucopt=1, imeyers5=F,
lcin=lcina=0, ffrzs=0, warmonly=0, cwfrz2snowfrac=0, imixedphase=0, mixedphase=F,
lzr=lzs=lzh=lzhl=0 (2-moment), eqtset=1, ibfc=1, ipconc=5, imaxdiaopt=3, imurain=1, imusnow=3,
iwetsoak=T, lnhf=lnhlf=lss=0, has_wetscav=F, ndebug<0.  ``_check_config`` asserts them.

G keys consumed (the B->C interface, i.e. gs locals read here before being set in part C) are
listed in ``G2_KEYS_CONSUMED`` (verified by the oracle test, which records every first read).
"""

from __future__ import annotations

import jax.numpy as jnp

from .indices import LC, LH, LHL, LI, LR, LS, LT, LV, LCCN, LN
from .mathfun import gaminterp
from .satfun import tabqvs

F64 = jnp.float64

# gs-local PARAMETERs used in part C (module lines 12725, 12884, 12910)
TFRDRY = 243.15
THNUC = 235.15


# gs locals read by part C before being set in part C (B->C interface; read directly from G
# in addition: 'gathered' and 'gz').  Verified by tests/v034/nssl2mom/test_gs_c.py.
G2_KEYS_CONSUMED = (
    'alpha', 'axx', 'bxx', 'cautn', 'ccnc', 'chaci', 'chaci0', 'chacr', 'chacs', 'chacs0', 'chacw',
    'chlaci', 'chlaci0', 'chlacr', 'chlacs', 'chlacs0', 'chlacw', 'chlmlr', 'chlmlrr', 'chmlr',
    'chmlrr', 'ciacr', 'ciacrf', 'ciacrs', 'ciacw', 'cicichr', 'cimasn', 'cimlr', 'cimxd', 'cina',
    'cninm', 'cninp', 'craci', 'cracr', 'cracw', 'crcnw', 'crfrz', 'crfrzf', 'crfrzs', 'csaci',
    'csacs', 'csacw', 'cscnvi', 'cscnvis', 'csmlr', 'csmlrr', 'csmxd', 'csplinter', 'csplinter2',
    'cwctfzc', 'cwctfzp', 'cwfrz', 'cwfrzc', 'cx', 'ehi', 'ehli', 'ehls', 'ehr', 'ehs', 'ehw',
    'fakvisc', 'fav', 'fbv', 'fcc3', 'fci', 'fcw', 'felf', 'felfcp', 'felscp', 'felv', 'felvcp',
    'ffrzh', 'fpndl', 'ftka', 'fvce', 'fwet1', 'fwet2', 'fwvdf', 'g1x', 'hlvent', 'hwvent', 'il2',
    'il3', 'il5', 'kgsm', 'kgsp', 'pi0', 'pk', 'pqs', 'qhaci', 'qhaci0', 'qhacr', 'qhacs', 'qhacs0',
    'qhacw', 'qhdsv', 'qhlaci', 'qhlaci0', 'qhlacr', 'qhlacs', 'qhlacs0', 'qhlacw', 'qhldsv', 'qhlmlr',
    'qhlmxd', 'qhmlr', 'qhmxd', 'qiacr', 'qiacrf', 'qiacrs', 'qiacw', 'qicichr', 'qidsv', 'qiihr',
    'qimlr', 'qimxd', 'qis', 'qraci', 'qracs', 'qracw', 'qrcnw', 'qrfrz', 'qrfrzf', 'qrfrzs', 'qrmxd',
    'qsaci', 'qsacr', 'qsacw', 'qscnvi', 'qsdsv', 'qsimxdep', 'qsimxsub', 'qsmlr', 'qsmxd',
    'qsplinter', 'qsplinter2', 'qss0', 'qv0', 'qwcnr', 'qwctfz', 'qwctfzc', 'qwfrz', 'qwfrzc', 'qwvp',
    'qx', 'qxmxd', 'raindn', 'rho0', 'rhoinv', 'rhovt', 'rimdn', 'rwcap', 'rwvent', 'rzxh', 'rzxhl',
    'rzxhlh', 'ssi', 'swcap', 'swvent', 'temcg', 'temg', 'theta0', 'thetap', 'tsqr', 'vhacr', 'vhacw',
    'vhlacr', 'vhlacw', 'vhlsoak', 'vhsoak', 'viacrf', 'vrfrzf', 'vshdgs', 'vtxbar', 'vx', 'wvel',
    'xcolmn', 'xdia', 'xdn', 'xmas', 'xv',
)


def _check_config(C):
    exp = dict(icond=1, iscni=4, incwet=0, ihlcnh=3, iusedw=0, icvhl2h=0, iglcnvi=1, iglcnvs=2,
               irwfrz=1, rcond=2, icrcev=1, lhwlg=0, lhlwlg=0, itype1=0, itype2=2, isnwfrac=0,
               icenucopt=1, imeyers5=False, lcin=0, lcina=0, imixedphase=0, mixedphase=False,
               lzr=0, lzs=0, lzh=0, lzhl=0, eqtset=1, ibfc=1, ipconc=5, imaxdiaopt=3, imurain=1,
               imusnow=3, iwetsoak=True, lnhf=0, lnhlf=0, lss=0, lhl=8, lh=7, ls=6, lccn=9,
               lwsm6=False, ldovol=True, ibinhmlr=0, ibinhlmlr=0, dmhlopt=0)
    for k, v in exp.items():
        got = getattr(C, k)
        if got != v:
            raise NotImplementedError(f"nssl gs part C ported for default mp=18 only: {k}={got} (expected {v})")
    for k in ("ffrzs", "warmonly", "cwfrz2snowfrac"):
        if float(getattr(C, k)) != 0.0:
            raise NotImplementedError(f"{k}={getattr(C, k)}")


class _State:
    """Mutable view on the gs locals; records the keys read before being written (consumed)."""

    def __init__(self, G):
        self._G = G
        self._d = {}
        self.consumed = set()

    def __getitem__(self, k):
        if k in self._d:
            return self._d[k]
        if k not in self._G:
            raise KeyError(f"gs part C needs gs local '{k}' from part B (G)")
        self.consumed.add(k)
        v = self._G[k]
        if isinstance(v, dict):
            v = dict(v)
        self._d[k] = v
        return v

    def __setitem__(self, k, v):
        self._d[k] = v


def gs_part_c(G, an, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, C, prec, dtp, gz=None, return_state=False):
    """Run nssl_2mom_gs lines 19696-25151.

    G      : dict of gs locals after line 19695 (``oracle_io.load_gs_dump`` layout); per-point arrays
             have the grid shape; ``G['gathered']`` (bool, grid shape) selects the gathered points.
    an     : (NA+1, *grid) state stack (density-scaled numbers), Fortran species index on axis 0.
    t0..t9 : gs work arrays (grid shape); only t0 is written by part C (temperature).
    gz     : dz (grid shape) -- the gs argument ``gz`` (driver dz2d); defaults to ``G['gz']``.
    Returns (an_new, (t0, ..., t9)); with ``return_state`` also the internal state object.
    """
    _check_config(C)
    R = prec.R
    g = _State(G)
    W = jnp.where
    mn = jnp.minimum
    mx = jnp.maximum

    def r(x):
        return jnp.asarray(x, R)

    dtp = r(dtp)
    dtpinv = r(1.0) / dtp
    pi = r(C.pi)
    tfr = r(C.tfr)
    evapfac = r(C.evapfac)
    qxmin = {il: r(C.qxmin[il]) for il in (LC, LR, LI, LS, LH, LHL)}
    xdnmx = {il: r(C.xdnmx[il]) for il in (LC, LR, LI, LS, LH, LHL)}
    xdnmn = {il: r(C.xdnmn[il]) for il in (LC, LR, LI, LS, LH, LHL)}
    xvmn = {il: r(C.xvmn[il]) for il in (LC, LR, LI, LS, LH, LHL)}
    xvmx = {il: r(C.xvmx[il]) for il in (LC, LR, LI, LS, LH, LHL)}
    gathered = jnp.asarray(G["gathered"], bool)
    zero = jnp.zeros(gathered.shape, R)
    false = jnp.zeros(gathered.shape, bool)

    qx = g["qx"]
    cx = g["cx"]
    temg = g["temg"]
    rho0 = g["rho0"]
    il5 = g["il5"]
    il5r = jnp.asarray(il5, R)

    # ------------------------------------------------------------------ 19699-19828
    # evaporation / sublimation / deposition rates and their limits (icond = 1)
    qhcev = zero
    chcev = zero
    qhlcev = zero
    chlcev = zero
    qfcev = zero
    qsmlr, qhmlr, qhlmlr = g["qsmlr"], g["qhmlr"], g["qhlmlr"]
    qss0, fav, fbv = g["qss0"], g["fav"], g["fbv"]
    qisbv = mx(mn(g["qidsv"], 0.0), mn(-g["qimxd"], -0.5 * qx[LI] * dtpinv))
    qssbv = W((temg < tfr) | ~(qsmlr < 0.0),
              mx(mn(g["qsdsv"], 0.0), mn(-g["qsmxd"], -0.5 * qx[LS] * dtpinv)), zero)
    qidpv = mx(g["qidsv"], 0.0)
    qsdpv = mx(g["qsdsv"], 0.0)
    tmp = evapfac * 4.0 * pi * (qx[LV] - qss0) * cx[LS] * g["swcap"] * g["swvent"] / (qss0 * (fav + fbv))
    tmp = mx(mn(0.0, tmp), mn(-g["qsmxd"], -0.5 * qx[LS] * dtpinv))
    qscev = W(qsmlr < 0.0, tmp, zero)
    cscev = zero

    xdia = g["xdia"]
    hasg = qx[LH] > qxmin[LH]
    c1 = hasg & ((temg < tfr) | ~(qhmlr < 0.0))
    qhsbv = W(c1, mx(mn(g["qhdsv"], 0.0), -g["qhmxd"]), zero)
    qhdpv = W(c1, mx(g["qhdsv"], 0.0), zero)
    tmp = evapfac * 2.0 * pi * (qx[LV] - qss0) * cx[LH] * xdia[(LH, 1)] * g["hwvent"] / (qss0 * (fav + fbv))
    tmp = mx(tmp, -g["qhmxd"])
    tmp = W(temg > tfr, mn(0.0, tmp), tmp)
    qhcev = W(hasg & (qhmlr < 0.0), tmp, qhcev)

    hashl = qx[LHL] > qxmin[LHL]
    c1 = hashl & ((temg < tfr) | ~(qhlmlr < 0.0))
    qxmxd = g["qxmxd"]
    qhlsbv = W(c1, mx(mn(g["qhldsv"], 0.0), -qxmxd[LHL]), zero)
    qhldpv = W(c1, mx(g["qhldsv"], 0.0), zero)
    tmp = evapfac * 2.0 * pi * (qx[LV] - qss0) * cx[LHL] * xdia[(LHL, 1)] * g["hlvent"] / (qss0 * (fav + fbv))
    tmp = mx(tmp, -g["qhlmxd"])
    tmp = W(temg > tfr, mn(0.0, tmp), tmp)
    qhlcev = W(hashl & (qhlmlr < 0.0), tmp, qhlcev)

    # deposition limiter (frac is DOUBLE PRECISION; frac*x of two REALs is exact in double)
    temp1 = qidpv + qsdpv + qhdpv + qhldpv
    qsimxdep = g["qsimxdep"]
    lim = temp1 > qsimxdep
    frac = qsimxdep / W(lim, temp1, r(1.0))
    qidpv = W(lim, frac * qidpv, qidpv)
    qsdpv = W(lim, frac * qsdpv, qsdpv)
    qhdpv = W(lim, frac * qhdpv, qhdpv)
    qhldpv = W(lim, frac * qhldpv, qhldpv)
    temp1 = qisbv + qssbv + qhsbv + qhlsbv
    qsimxsub = g["qsimxsub"]
    lim = temp1 < -qsimxsub
    frac = -qsimxsub / W(lim, temp1, r(1.0))
    qisbv = W(lim, frac * qisbv, qisbv)
    qssbv = W(lim, frac * qssbv, qssbv)
    qhsbv = W(lim, frac * qhsbv, qhsbv)
    qhlsbv = W(lim, frac * qhlsbv, qhlsbv)

    # 19831-19843 (ipconc >= 1)
    cssbv = (cx[LS] / (qx[LS] + 1.e-20)) * qssbv
    cisbv = (cx[LI] / (qx[LI] + 1.e-20)) * qisbv
    chsbv = (cx[LH] / (qx[LH] + 1.e-20)) * qhsbv
    chlsbv = (cx[LHL] / (qx[LHL] + 1.e-20)) * qhlsbv
    csdpv = zero
    cidpv = zero
    chdpv = zero
    chldpv = zero

    # ------------------------------------------------------------------ 19849-19919 ice -> snow (iscni = 4)
    rhoinv = g["rhoinv"]
    xmas = g["xmas"]
    c1 = (qx[LI] > qxmin[LI]) & (qidpv > 0.0) & (xdia[(LI, 3)] >= 100.e-6)
    qscni = W(c1, mn(0.5, xdia[(LI, 3)] / 200.e-6) * qidpv, zero)
    cscni = W(c1, r(C.fscni) * qscni * rho0 / mx(r(C.rho_qs) * xvmn[LS], xmas[LI]), zero)
    cscnis = cscni

    # 19921-20047: incwet = 0 -> dhwet = dhlwet = dfwet = d1t (not used further in part C)

    # ------------------------------------------------------------------ 20055-20270 dry / wet growth
    qsacr, qsacw, qsaci = g["qsacr"], g["qsacw"], g["qsaci"]
    qhaci, qhacs, qhacr, qhacw = g["qhaci"], g["qhacs"], g["qhacr"], g["qhacw"]
    qhlaci, qhlacs, qhlacr, qhlacw = g["qhlaci"], g["qhlacs"], g["qhlacr"], g["qhlacw"]
    qsdry = qsacr + qsacw + qsaci
    qhdry = qhaci + qhacs + qhacr + qhacw
    qhldry = qhlaci + qhlacs + qhlacr + qhlacw
    fwet1, fwet2 = g["fwet1"], g["fwet2"]
    inwet = (TFRDRY < temg) & (temg < tfr)
    qhwet = mx(0.0, xdia[(LH, 1)] * g["hwvent"] * cx[LH] * fwet1 + fwet2 * (qhaci + qhacs))
    qhlwet = mx(0.0, xdia[(LHL, 1)] * g["hlvent"] * cx[LHL] * fwet1 + fwet2 * (qhlaci + qhlacs))
    qhwet = W(inwet, qhwet, qhdry)
    qhlwet = W(inwet, qhlwet, qhldry)

    # ------------------------------------------------------------------ 20275-20353 shedding
    vhacw, vhacr = g["vhacw"], g["vhacr"]
    vhlacw, vhlacr = g["vhlacw"], g["vhlacr"]
    qhshr = mn(0.0, qhwet - qhdry)
    qhlshr = mn(0.0, qhlwet - qhldry)
    qsshr = zero
    vhshdr = zero
    vhlshdr = zero
    cold = temg < 243.15
    qhshr = W(cold, zero, qhshr)
    qhlshr = W(cold, zero, qhlshr)
    warm = temg > tfr
    qsshr = W(warm, -qsacr - qsacw, qsshr)
    qhlshr = W(warm, -qhlacw - qhlacr, qhlshr)
    qhshr = W(warm, -qhacw - qhacr, qhshr)
    vhshdr = W(warm, -vhacw - vhacr, vhshdr)
    vhlshdr = W(warm, -vhlacw - vhlacr, vhlshdr)
    qhwet = W(warm, zero, qhwet)
    qhlwet = W(warm, zero, qhlwet)
    wetsfc = ((qhshr < 0.0) & (temg < tfr)) | ((qhmlr < -qxmin[LH]) & (temg > tfr))
    wetgrowth = (qhshr < 0.0) & (temg < tfr)
    c1 = (qhlshr < 0.0) & (temg < tfr)
    wetsfchl = c1  # inside the IF the assigned expression equals .true.; else stays .false.
    wetgrowthhl = c1

    # 20355-20387 (ipconc >= 1)
    csshr = zero
    chshr = zero
    xdn = g["xdn"]
    vshdgs = g["vshdgs"]
    chshrr = rho0 * qhshr / (xdn[LR] * vshdgs[LH])
    chlshr = zero
    chlshrr = rho0 * qhlshr / (xdn[LR] * vshdgs[LHL])

    # ------------------------------------------------------------------ 20394-20560 final decisions
    sn = qsshr < 0.0
    qsdpv = W(sn, zero, qsdpv)
    qssbv = W(sn, zero, qssbv)
    qsshr = W(sn, qsshr, zero)

    # graupel wet growth (lvol(lh) > 1, iwetsoak, not mixedphase)
    vx = g["vx"]
    rimdn = g["rimdn"]
    raindn = g["raindn"]
    vhsoak = g["vhsoak"]
    vhlsoak = g["vhlsoak"]
    wg = wetgrowth
    rimdn[LH] = W(wg, xdnmx[LH], rimdn[LH])
    raindn[LH] = W(wg, xdnmx[LH], raindn[LH])
    vhacw = W(wg, qhacw * rho0 / rimdn[LH], vhacw)
    vhacr = W(wg, qhacr * rho0 / raindn[LH], vhacr)
    soak = wg & (xdn[LH] < xdnmx[LH])
    v1 = (1. - xdn[LH] / xdnmx[LH]) * vx[LH] / (dtp)
    v2 = rho0 * qhwet / xdnmx[LH]
    vhsoak = W(soak, mn(v1, v2), vhsoak)
    vhshdr = W(wg, mn(0.0, rho0 * qhwet / xdnmx[LH] - vhacw - vhacr), vhshdr)
    qhdpv = W(wg, zero, qhdpv)
    chdpv = W(wg, zero, chdpv)
    ehi, ehs = g["ehi"], g["ehs"]
    qimxd, cimxd, qsmxd, csmxd = g["qimxd"], g["cimxd"], g["qsmxd"], g["csmxd"]
    chaci, chacs = g["chaci"], g["chacs"]
    c1 = wg & (ehi > 0.0)
    qhaci = W(c1, mn(qimxd, g["qhaci0"]), qhaci)
    chaci = W(c1, mn(cimxd, g["chaci0"]), chaci)
    c1 = wg & (ehs > 0.0)
    qhacs = W(c1, mn(qsmxd, g["qhacs0"]), qhacs)
    chacs = W(c1, mn(csmxd, g["chacs0"]), chacs)
    ehs = W(c1, r(C.ehsmax), ehs)
    qhacs = W(c1, mn(qsmxd, qhacs), qhacs)
    wetsfc = wetsfc | wg

    # hail wet growth
    wg = wetgrowthhl
    qhldpv = W(wg, zero, qhldpv)
    chldpv = W(wg, zero, chldpv)
    rimdn[LHL] = W(wg, xdnmx[LHL], rimdn[LHL])
    raindn[LHL] = W(wg, xdnmx[LHL], raindn[LHL])
    vhlacw = W(wg, qhlacw * rho0 / rimdn[LHL], vhlacw)
    vhlacr = W(wg, qhlacr * rho0 / raindn[LHL], vhlacr)
    soak = xdn[LHL] < xdnmx[LHL]
    v1 = (1. - xdn[LHL] / xdnmx[LHL]) * vx[LHL] / (dtp)
    v2 = rho0 * qhlwet / xdnmx[LHL]
    vhlsoak = W(wg, W(soak, W(v1 > v2, v2, v1), zero), vhlsoak)
    vhlshdr = W(wg, mn(0.0, rho0 * qhlwet / xdnmx[LHL] - vhlacw - vhlacr), vhlshdr)
    ehli, ehls = g["ehli"], g["ehls"]
    chlaci, chlacs = g["chlaci"], g["chlacs"]
    c1 = wg & (ehli > 0.0)
    qhlaci = W(c1, mn(qimxd, g["qhlaci0"]), qhlaci)
    chlaci = W(c1, mn(cimxd, g["chlaci0"]), chlaci)
    c1 = wg & (ehls > 0.0)
    qhlacs = W(c1, mn(qsmxd, g["qhlacs0"]), qhlacs)
    chlacs = W(c1, mn(csmxd, g["chlacs0"]), chlacs)
    ehls = W(c1, r(C.ehsmax), ehls)
    wetsfchl = wetsfchl | wg

    # ------------------------------------------------------------------ 20564-20632 ice -> graupel (iglcnvi = 1)
    vtxbar = g["vtxbar"]
    rimc1, rimc2, rimc3 = r(C.rimc1), r(C.rimc2), r(C.rimc3)
    qiacw = g["qiacw"]
    c1 = (temg < 273.0) & (qiacw - qidpv > 0.0)
    den = W(c1, temg - 273.15, r(-1.0))
    base = -((0.5) * (1.e+06) * xdia[(LC, 1)]) * ((0.60) * vtxbar[(LI, 1)]) / den
    tmp = rimc1 * jnp.power(W(c1, base, r(1.0)), rimc2)
    tmp = mn(mx(rimc3, tmp), 900.0)
    c1 = c1 & (tmp >= 200.0)
    rr = mx(0.5 * (xdn[LI] + tmp), xdnmn[LH])
    qhcni = W(c1, qiacw - qidpv, zero)
    chcni = W(c1, cx[LI] * qhcni / W(c1, qx[LI], r(1.0)), zero)
    chcnih = W(c1, mn(chcni, rho0 * qhcni / (rr * xvmn[LH])), zero)
    vhcni = W(c1, rho0 * qhcni / rr, zero)

    # ------------------------------------------------------------------ 20635-21010 graupel -> hail (ihlcnh = 3)
    qhlcnh = zero
    chlcnh = zero
    chlcnhhl = zero
    vhlcnh = zero
    vhlcnhl = zero
    qhcnhl = zero
    chcnhl = zero
    vhcnhl = zero
    hailcnvtoffset = r(C.hailcnvtoffset)
    hlcnhqmin = r(C.hlcnhqmin)
    dg0thresh = r(C.dg0thresh)
    temcg = g["temcg"]
    ehw, ehr = g["ehw"], g["ehr"]
    rimes = (qhacw + qhacr) * dtp > qxmin[LH]
    cnd = (temg <= tfr + hailcnvtoffset) & (
        (rimes & (qx[LH] > hlcnhqmin) & (temg > r(C.dwtempmin))) | (wetgrowth & (qx[LH] > hlcnhqmin)))
    x = 1.1e4 * rho0 * (ehw * qx[LC] + ehr * qx[LR]) - 1.3e3 * rho0 * qx[LI] + 1.0
    xpos = x > 1.e-20
    arg = mn(70.0, (-temcg / W(xpos, x, r(1.0))))
    dwr = W(xpos, 0.01 * (jnp.exp(arg) - 1.0), r(1.e30))
    d = mn(dwr, dg0thresh + 0.0001)
    it = cnd & (dwr < 0.2) & (dwr > 0.0) & (rho0 * (qx[LC] + qx[LR]) > 1.e-4)
    sqrtrhovt = jnp.sqrt(g["rhovt"])
    fventh = sqrtrhovt * (g["fpndl"] ** (1. / 3.)) * (g["fakvisc"]) ** (-0.5)
    ltq = jnp.trunc((tfr - 163.15) / r(C.fqsat) + 1.5).astype(jnp.int32)
    qvs0 = g["pqs"] * tabqvs(ltq, C, R)
    felf, fcw = g["felf"], g["fcw"]
    denomdp = (felf + fcw * temcg).astype(F64)
    h1 = (-g["ftka"] * temcg - g["felv"] * g["fwvdf"] * rho0 * (qx[LV] - qvs0)).astype(F64)
    h2 = (ehi * qx[LI] * rho0 * g["fci"] * temcg).astype(F64)
    h3 = (mx(r(C.dwehwmin), ehw) * qx[LC]).astype(F64)
    h4 = (ehr * qx[LR]).astype(F64)
    axx, bxx = g["axx"], g["bxx"]
    done = ~it
    for n in range(1, 11):
        dn = mx(d, 1.e-4)
        dold = dn
        vth = axx[LH] * dn ** bxx[LH]
        x2 = fventh * sqrtrhovt * jnp.sqrt(dn * vth)
        ah = W(x2 > 1.4, 0.78 + 0.308 * x2, 1.0 + 0.108 * x2 ** 2)
        num = (8. * ah).astype(F64) * h1
        den = ((mx(0.001, vth - vtxbar[(LC, 1)]).astype(F64) * h3
                + mx(0.001, vth - vtxbar[(LR, 1)]).astype(F64) * h4) * rho0.astype(F64) * denomdp
               + mx(0.001, vth - vtxbar[(LI, 1)]).astype(F64) * h2)
        dnew = (num / W(done, 1.0, den)).astype(R)
        conv = (jnp.abs(dold - dnew) / dold < 0.05) | ((n > 3) & (dnew > dg0thresh))
        d = W(done, d, dnew)
        done = done | conv
    d = W(it, mn(d, dg0thresh + 0.0001), d)
    dg0 = W(cnd, mx(d, r(C.dwmin)), dg0thresh + 0.0001)
    c2 = rimes & (qx[LH] > hlcnhqmin) & (temg <= tfr + hailcnvtoffset) & (temg > 238.0)
    dg0 = W(c2, mn(dg0, r(C.dwmax)), dg0)
    wtest = (dg0 > 0.0) & (dg0 < dg0thresh)
    conv = wtest & ((qhacw * dtp > qxmin[LH]) & (temg < tfr + hailcnvtoffset) & (qx[LH] > hlcnhqmin))
    alpha = g["alpha"]
    ratio = mn(r(100.0), dg0 / xdia[(LH, 1)])
    ratio = W(conv, ratio, zero)
    alph = W(conv, alpha[LH], zero)
    tmp2 = gaminterp(ratio, alph, 4, 1, R)
    qxd1 = qx[LH] * (tmp2)
    qhlcnh_c = dtpinv * qxd1
    big = qxd1 > 10. * qxmin[LHL]
    tmpn = gaminterp(ratio, alph, 1, 1, R)
    cxd1 = r(1.0) * cx[LH] * (tmpn)
    chlcnh_c = dtpinv * cxd1
    chlcnhhl_c = chlcnh_c
    # dmhlopt = 0 ; ipconc == 5: adjust number by the reflectivity removed from graupel
    tmp3 = gaminterp(ratio, alph, 11, 1, R)
    g1x = g["g1x"]
    tmp5 = g1x[LH] * (rho0 * qx[LH]) ** 2 / ((pi * xdn[LH] / 6.) ** 2 * cx[LH])
    zxd1 = r(1.0) * (tmp3) * tmp5
    tmp3 = g1x[LH] * (rho0 * qxd1) ** 2 / ((pi * xdn[LH] / 6.0) ** 2)
    tmp4 = tmp3 / cxd1
    adj = tmp4 > zxd1
    chlcnhhl_c = W(adj, dtpinv * (tmp3 / W(adj, zxd1, r(1.0))), chlcnhhl_c)
    qhlcnh = W(conv, W(big, qhlcnh_c, zero), qhlcnh)
    chlcnh = W(conv & big, chlcnh_c, chlcnh)
    chlcnhhl = W(conv & big, chlcnhhl_c, chlcnhhl)
    vhlcnh = W(conv, rho0 * qhlcnh / xdn[LH], vhlcnh)
    vhlcnhl = W(conv, rho0 * qhlcnh / mx(xdnmn[LHL], xdn[LH]), vhlcnhl)
    # icvhl2h = 0: no hail -> graupel conversion

    # ------------------------------------------------------------------ 21018-21142 snow -> graupel (iglcnvs = 2)
    qscnh = zero
    cscnh = zero
    vscnh = zero
    c1 = (qx[LS] > qxmin[LS]) & (qsacw > 0.0) & (temg < 273.0) & (qsacw - qsdpv > 0.0)
    den = W(c1, temg - 273.15, r(-1.0))
    base = -((0.5) * (1.e+06) * xdia[(LC, 1)]) * ((0.60) * vtxbar[(LS, 1)]) / den
    tmp = rimc1 * jnp.power(W(c1, base, r(1.0)), rimc2)
    tmp = mn(tmp, 900.0)
    c1 = c1 & (tmp >= 200.0)
    rr = mx(0.5 * (xdn[LS] + tmp), xdnmn[LH])
    qhcns = W(c1, qsacw - qsdpv, zero)
    chcns = W(c1, cx[LS] * qhcns / W(c1, qx[LS], r(1.0)), zero)
    chcnsh = W(c1, mn(chcns, rho0 * qhcns / (rr * xvmn[LH])), zero)
    vhcns = W(c1, rho0 * qhcns / rr, zero)

    # ------------------------------------------------------------------ 21149-21215 rain freezing heat budget
    qrfrz, qrfrzs, qrfrzf = g["qrfrz"], g["qrfrzs"], g["qrfrzf"]
    qiacr, qiacrf, qiacrs = g["qiacr"], g["qiacrf"], g["qiacrs"]
    crfrz, crfrzf, crfrzs = g["crfrz"], g["crfrzf"], g["crfrzs"]
    ciacr, ciacrf, ciacrs = g["ciacr"], g["ciacrf"], g["ciacrs"]
    vrfrzf, viacrf = g["vrfrzf"], g["viacrf"]
    rwvent = g["rwvent"]
    qrztot = qrfrz + qiacr + qsacr
    qrzmax = (xdia[(LR, 1)] * rwvent * cx[LR] * fwet1)
    qrzmax = mx(qrzmax, 0.0)
    qrzmax = mn(qrztot, qrzmax)
    qrzmax = mn(qx[LR] * dtpinv, qrzmax)
    qrzmax = W(temcg < -30., qx[LR] * dtpinv, qrzmax)
    c1 = (qrztot > qrzmax) & (qrztot > qxmin[LR])
    qrzfac = W(c1, qrzmax / W(c1, qrztot, r(1.0)), r(1.0))
    qrzfac = mn(1.0, qrzfac)
    c1 = (temg <= 273.15) & (qrzfac < 1.0)

    def sc(v):
        return W(c1, qrzfac * v, v)

    qrfrz, qrfrzs, qrfrzf = sc(qrfrz), sc(qrfrzs), sc(qrfrzf)
    qiacr, qsacr, qiacrf, qiacrs = sc(qiacr), sc(qsacr), sc(qiacrf), sc(qiacrs)
    crfrz, crfrzf, crfrzs = sc(crfrz), sc(crfrzf), sc(crfrzs)
    ciacr, ciacrf, ciacrs = sc(ciacr), sc(ciacrf), sc(ciacrs)
    vrfrzf, viacrf = sc(vrfrzf), sc(viacrf)

    # ------------------------------------------------------------------ 21223-21260 rain evaporation (rcond = 2)
    hasr = qx[LR] > qxmin[LR]
    qrcev = g["fvce"] * cx[LR] * rwvent * g["rwcap"] * evapfac
    qrcev = mn(qrcev, 0.0)
    qrcev = mx(qrcev, -g["qrmxd"])
    crcev = W(qrcev < 0.0, (cx[LR] / W(hasr, qx[LR], r(1.0))) * qrcev, zero)
    qrcev = W(hasr, qrcev, zero)
    crcev = W(hasr, crcev, zero)

    # ------------------------------------------------------------------ 21281-21471 Hallett-Mossop (itype2 = 2)
    chmul1 = zero
    chlmul1 = zero
    qhmul1 = zero
    qhlmul1 = zero
    xv = g["xv"]
    ltest = (qx[LH] > qxmin[LH]) | (qx[LHL] > qxmin[LHL])
    c1 = (qx[LC] > qxmin[LC]) & (temg >= 265.15) & (temg <= 271.15) & (xv[LC] > 0.0) & ltest
    xvc = W(c1, xv[LC], r(1.0))
    a0 = alpha[LC] == 0.0
    ex1a = (1. / 250.) * jnp.exp(-7.23e-15 / xvc)
    ratio = mn(r(100.0), (1. + alpha[LC]) * (7.23e-15) / xvc)
    ex1b = (1. / 250.) * gaminterp(W(c1 & ~a0, ratio, zero), W(c1 & ~a0, alpha[LC], zero), 1, 1, R)
    ex1 = W(a0, ex1a, ex1b)
    ft = mx(0.0, mn(1.0, -0.11 * temcg ** 2 - 1.1 * temcg - 1.7))
    c1 = c1 & (ft > 0.0)
    chacw, chlacw = g["chacw"], g["chlacw"]
    cimas0 = r(C.cimas0)
    c2 = c1 & (qx[LH] > qxmin[LH]) & ~wetsfc
    chmul1 = W(c2, ft * ex1 * chacw, chmul1)
    qhmul1 = W(c2, cimas0 * chmul1 * rhoinv, qhmul1)
    c2 = c1 & (qx[LHL] > qxmin[LHL]) & ~wetsfchl
    chlmul1 = W(c2, (ft * ex1 * chlacw), chlmul1)
    qhlmul1 = W(c2, cimas0 * chlmul1 * rhoinv, qhlmul1)
    # isnwfrac = 0: no snow fragmentation
    csmul = zero
    qsmul = zero
    qraci, craci = g["qraci"], g["craci"]
    qracif = qraci
    cracif = craci

    # ------------------------------------------------------------------ 21540-21654 primary ice nucleation (icenucopt = 1)
    cmassin = r(g["cimasn"])
    ciintmx = r(C.ciintmx)
    c1 = (temg < 268.15) & (ciintmx > (cx[LI] + 0.0))
    felv = g["felv"]
    fiinit = (felv ** 2) / (r(C.cp) * r(C.rw))
    qis = g["qis"]
    dqisdt = (qx[LV] - qis) / (1.0 + fiinit * qis / g["tsqr"])
    c1 = c1 & (g["ssi"] > 1.0)
    # kgs(mgs): Fortran level of the point; full-grid convention -> last axis is the level (k-1)
    kgs = jnp.broadcast_to(jnp.arange(1, gathered.shape[-1] + 1), gathered.shape)
    dzfacp = mx((jnp.asarray(g["kgsp"]) - kgs).astype(R), 0.0)
    dzfacm = mx((kgs - jnp.asarray(g["kgsm"])).astype(R), 0.0)
    gzp = jnp.asarray(G["gz"] if gz is None else gz, R)
    qiint = ((jnp.asarray(il5, R) * (cmassin / rho0)) * mx(0.0, g["wvel"]) * mx((g["cninp"] - g["cninm"]), 0.0)
             / gzp / ((dzfacp + dzfacm)))
    qiint = mn(qiint, mx(0.25 * dqisdt, 0.0))
    ciint = qiint * rho0 / cmassin
    c2 = ciint > mx(0.0, ciintmx - cx[LI] - 0.0) * dtpinv
    ciint = W(c2, mx(0.0, ciintmx - (cx[LI])) * dtpinv, ciint)
    qiint = W(c2, ciint * cmassin / rho0, qiint)
    qiint = W(c1, qiint, zero)
    ciint = W(c1, ciint, zero)
    qicicnt = W(g["xcolmn"] == 1, qiint, zero)
    cicint = W(g["xcolmn"] == 1, ciint, zero)

    # ------------------------------------------------------------------ 21683-21703 shed rain
    qwshw = zero
    cwshw = zero
    qrshr = qsshr + qhshr + qhlshr
    crshr = chshrr / g["rzxh"] + chlshrr / g["rzxhl"]

    # ------------------------------------------------------------------ 21713-22189 number tendencies
    cwfrzc, cwctfzc, cicichr = g["cwfrzc"], g["cwctfzc"], g["cicichr"]
    csplinter, csplinter2 = g["csplinter"], g["csplinter2"]
    pccii = (il5r * cicint
             + il5r * ((1.0 - r(C.cwfrz2snowfrac)) * cwfrzc + cwctfzc + cicichr)
             + chmul1 + chlmul1 + csplinter + csplinter2 + csmul)
    pccii = pccii * (1.0 - r(C.ffrzs))
    cimlr = g["cimlr"]
    pccid = (il5r * (-cscni - g["cscnvi"] - craci - g["csaci"] - chaci - chlaci - chcni)
             + il5r * cisbv - (1. - il5r) * cimlr)
    pccin = ciint

    ciacw, cwfrz, cwctfzp = g["ciacw"], g["cwfrz"], g["cwctfzp"]
    cracw, csacw = g["cracw"], g["csacw"]
    pccwi = (0.0) - cwshw
    cautn = jnp.asarray(g["cautn"], F64)
    pccwd = ((-cautn + (il5r * (-ciacw - cwfrz - cwctfzp - cwctfzc)).astype(F64))
             - cracw.astype(F64) - csacw.astype(F64) - chacw.astype(F64) - chlacw.astype(F64)).astype(R)
    lim = -pccwd * dtp > cx[LC]
    frac = -cx[LC] / W(lim, pccwd * dtp, r(1.0))
    pccwd = W(lim, -cx[LC] * dtpinv, pccwd)
    ciacw = W(lim, frac * ciacw, ciacw)
    cwfrz = W(lim, frac * cwfrz, cwfrz)
    cracw = W(lim, frac * cracw, cracw)
    csacw = W(lim, frac * csacw, csacw)
    chacw = W(lim, frac * chacw, chacw)
    cwfrzc = W(lim, frac * cwfrzc, cwfrzc)
    cwctfzc = W(lim, frac * cwctfzc, cwctfzc)
    # frac is DOUBLE PRECISION: (1.-frac)*... is evaluated in double (line 21907)
    pccii = W(lim, (pccii.astype(F64) - (1. - frac.astype(F64)) * il5r.astype(F64)
                    * (cwfrzc + cwctfzc).astype(F64) * (1. - r(C.ffrzs)).astype(F64)).astype(R), pccii)
    chlacw = W(lim, frac * chlacw, chlacw)

    crcnw = g["crcnw"]
    chmlrr, chlmlrr, csmlrr = g["chmlrr"], g["chlmlrr"], g["csmlrr"]
    rzxh, rzxhl = g["rzxh"], g["rzxhl"]
    cracr, chacr, chlacr = g["cracr"], g["chacr"], g["chlacr"]
    oml5 = (1 - jnp.asarray(il5)).astype(R)
    pcrwi = (crcnw + oml5 * (-chmlrr / rzxh - chlmlrr / rzxhl - csmlrr - cimlr)
             - mn(0.0, cracr) - crshr)
    pcrwd = (il5r * (-ciacr - crfrz) - chacr - chlacr + crcev - mx(0.0, cracr))
    lim = -pcrwd * dtp > cx[LR]
    frac = -cx[LR] / W(lim, pcrwd * dtp, r(1.0))
    pcrwd = W(lim, -cx[LR] * dtpinv, pcrwd)

    def scr(v):
        return W(lim, frac * v, v)

    ciacr, ciacrf, ciacrs = scr(ciacr), scr(ciacrf), scr(ciacrs)
    crfrz, crfrzf, crfrzs = scr(crfrz), scr(crfrzf), scr(crfrzs)
    chacr, chlacr, crcev, cracr = scr(chacr), scr(chlacr), scr(crcev), scr(cracr)

    cscnvis, csacs = g["cscnvis"], g["csacs"]
    csmlr = g["csmlr"]
    ifrzs = r(C.ifrzs)
    pcswi = (il5r * (cscnis + cscnvis) + r(C.cwfrz2snowfrac) * cwfrz / r(C.cwfrz2snowratio) + cscnh)
    pcswd = (-chacs - chlacs - chcns + oml5 * csmlr + csshr + cssbv - csacs)
    lim = cx[LS] + dtp * (pcswi + pcswd) < 0.0
    frac = (-cx[LS] + pcswi * dtp) / W(lim, pcswd * dtp, r(1.0))

    def scs(v):
        return W(lim, frac * v, v)

    pcswd = scs(pcswd)
    chacs, chlacs, chcns, csmlr = scs(chacs), scs(chlacs), scs(chcns), scs(csmlr)
    csshr, cssbv, csacs = scs(csshr), scs(cssbv), scs(csacs)
    pccii = pccii + (1. - ifrzs) * crfrzs + (1. - ifrzs) * ciacrs
    pcswi = pcswi + (ifrzs) * crfrzs + (ifrzs) * ciacrs

    ffrzh = r(g["ffrzh"])
    ifrzg, ifiacrg, f2h = r(C.ifrzg), r(C.ifiacrg), r(C.f2h)
    chmlr, chlmlr = g["chmlr"], g["chlmlr"]
    pchwi = ((ffrzh * ifrzg * crfrzf + il5r * ffrzh * ifiacrg * (ciacrf))
             + f2h * chcnsh + f2h * chcnih + chcnhl)
    pchwd = (oml5 * chmlr + chsbv - il5r * chlcnh - cscnh)
    pchli = ((ffrzh * (1.0 - ifrzg) * crfrzf + il5r * ffrzh * (1.0 - ifiacrg) * (ciacrf))
             + chlcnhhl * g["rzxhlh"])
    pchld = (oml5 * chlmlr + chlsbv - chcnhl)
    lim = cx[LHL] + dtp * (pchli + pchld) < 0.0
    frac = (-cx[LHL] + pchli * dtp) / W(lim, pchld * dtp, r(1.0))
    chlmlr = W(lim, frac * chlmlr, chlmlr)
    chlsbv = W(lim, frac * chlsbv, chlsbv)
    chcnhl = W(lim, frac * chcnhl, chcnhl)
    pchld = W(lim, frac * pchld, pchld)

    # ------------------------------------------------------------------ 22199-22685 mass tendencies
    qhmlr_, qsmlr_, qhlmlr_, qimlr = qhmlr, qsmlr, qhlmlr, g["qimlr"]
    qiint_ = qiint

    def vap_i():
        return (-mn(0.0, qrcev) - mn(0.0, qhcev) - mn(0.0, qhlcev) - mn(0.0, qscev)
                - qhsbv - qhlsbv - qssbv - il5r * qisbv)

    def vap_d():
        return (-mx(0.0, qrcev) - mx(0.0, qhcev) - mx(0.0, qhlcev) - mx(0.0, qscev)
                + il5r * (-qiint_ - qhdpv - qsdpv - qhldpv) - il5r * qidpv)

    pqwvi = vap_i()
    pqwvd = vap_d()

    qwcnr = g["qwcnr"]
    qwfrz, qwctfz, qiihr = g["qwfrz"], g["qwctfz"], g["qiihr"]
    qracw, qrcnw = g["qracw"], g["qrcnw"]
    qwfrzc, qwctfzc = g["qwfrzc"], g["qwctfzc"]
    pqcwi = (0.0) + qwcnr - qwshw
    pqcwd = (il5r * (-qiacw - qwfrz - qwctfz) - il5r * (qiihr)
             - qracw - qsacw - qrcnw - qhacw - qhlacw)
    lim = (pqcwd < 0.0) & (-pqcwd * dtp > qx[LC])
    frac = -mx(0.0, qx[LC]) / W(lim, pqcwd * dtp, r(1.0))
    pqcwd = W(lim, -qx[LC] * dtpinv, pqcwd)

    def scw(v):
        return W(lim, frac * v, v)

    qiacw, qwfrzc, qwfrz, qwctfzc, qwctfz = scw(qiacw), scw(qwfrzc), scw(qwfrz), scw(qwctfzc), scw(qwctfz)
    qracw, qsacw, qhacw, vhacw, qrcnw = scw(qracw), scw(qsacw), scw(qhacw), scw(vhacw), scw(qrcnw)
    qhlacw, vhlacw = scw(qhlacw), scw(vhlacw)

    qicichr = g["qicichr"]
    qsplinter, qsplinter2 = g["qsplinter"], g["qsplinter2"]
    pqcii = (il5r * qicicnt
             + il5r * ((1.0 - r(C.cwfrz2snowfrac)) * qwfrzc + qwctfzc)
             + il5r * (qicichr)
             + qsmul + qhmul1 + qhlmul1 + qsplinter + qsplinter2)
    pqcii = pqcii * (1.0 - r(C.ffrzs)) + il5r * qidpv + il5r * qiacw
    qscnvi = g["qscnvi"]
    pqcid = (il5r * (-qscni - qscnvi - qraci - qsaci) - qhaci - qhlaci
             + il5r * qisbv + (1. - il5r) * qimlr - qhcni)

    pqrwi = (qracw + qrcnw + mx(0.0, qrcev)
             + oml5 * (-qhmlr_ - qsmlr_ - qhlmlr_ - qimlr)
             - qrshr)
    pqrwd = (il5r * (-qiacr - qrfrz) - qsacr - qhacr - qhlacr - qwcnr + mn(0.0, qrcev))
    lim = (pqrwd < 0.0) & (-(pqrwd + pqrwi) * dtp > qx[LR])
    frac = (-qx[LR] + pqrwi * dtp) / W(lim, pqrwd * dtp, r(1.0))
    f64 = frac.astype(F64)
    m0 = mn(0.0, qrcev)
    m1 = mx(0.0, qrcev)
    pqwvi = W(lim, ((pqwvi + m0).astype(F64) - f64 * m0.astype(F64)).astype(R), pqwvi)
    pqwvd = W(lim, ((pqwvd + m1).astype(F64) - f64 * m1.astype(F64)).astype(R), pqwvd)

    def srr(v):
        return W(lim, frac * v, v)

    qiacr, qiacrf, qiacrs, viacrf = srr(qiacr), srr(qiacrf), srr(qiacrs), srr(viacrf)
    qrfrz, qrfrzs, qrfrzf, vrfrzf = srr(qrfrz), srr(qrfrzs), srr(qrfrzf), srr(vrfrzf)
    qsacr, qhacr, vhacr, qrcev = srr(qsacr), srr(qhacr), srr(vhacr), srr(qrcev)
    qhlacr, vhlacr, qhcev, qhlcev = srr(qhlacr), srr(vhlacr), srr(qhcev), srr(qhlcev)
    pqrwd = W(lim, il5r * (-qiacr - qrfrz - qsacr) - qhacr - qhlacr - qwcnr + mn(0.0, qrcev), pqrwd)
    rs = lim & (qrcev != 0.0)
    pqwvi = W(rs, vap_i(), pqwvi)
    pqwvd = W(rs, vap_d(), pqwvd)

    il2, il3 = jnp.asarray(g["il2"], R), jnp.asarray(g["il3"], R)
    qracs = g["qracs"]
    pqswi = (il5r * (qscni + qsaci + qsdpv
                     + qscnvi
                     + ifrzs * (qiacrs + qrfrzs)
                     + il5r * ((qwfrzc + qwctfzc + qicichr) * r(C.ffrzs)
                               + (1.0 - r(C.ffrzs)) * r(C.cwfrz2snowfrac) * qwfrz)
                     + il2 * qsacr)
             + il5r * qicicnt * r(C.ffrzs)
             + il3 * (qiacrf + qracif)
             + mx(0.0, qscev)
             + qsacw + qscnh
             + r(C.ffrzs) * (qsmul + qhmul1 + qhlmul1 + qsplinter + qsplinter2))
    pqswd = (-qracs * (1 - il2) - qhacs - qhlacs
             - qhcns
             + oml5 * qsmlr_ + qsshr
             + qssbv
             + mn(0.0, qscev)
             - qsmul)
    lim = (pqswd < 0.0) & (qx[LS] + dtp * (pqswi + pqswd) < 0.0)
    frac = (-qx[LS] + pqswi * dtp) / W(lim, pqswd * dtp, r(1.0))

    def ssw(v):
        return W(lim, frac * v, v)

    pqswd = ssw(pqswd)
    qracs, qhacs, qhlacs, qhcns = ssw(qracs), ssw(qhacs), ssw(qhlacs), ssw(qhcns)
    qsmlr_, qsshr, qssbv, qsmul = ssw(qsmlr_), ssw(qsshr), ssw(qssbv), ssw(qsmul)
    qscev = W(lim & (qscev < 0.0), frac * qscev, qscev)
    pqcii = pqcii + (1. - ifrzs) * qrfrzs + (1. - ifrzs) * qiacrs

    pqhwi = (il5r * (ffrzh * ifrzg * qrfrzf + (1 - il3) * ffrzh * ifiacrg * (qiacrf + qracif))
             + (1 - il2) * (qracs + qsacr)
             + il5r * (qhdpv)
             + mx(0.0, qhcev)
             + qhacr + qhacw
             + qhacs + qhaci
             + f2h * qhcns + f2h * qhcni + qhcnhl)
    pqhwd = (qhshr
             + oml5 * qhmlr_
             + qhsbv
             + mn(0.0, qhcev)
             - qhmul1 - qhlcnh - qscnh
             - ffrzh * (qsplinter + qsplinter2))

    pqhli = (il5r * (qhldpv + ((1.0 - ifrzg) * qrfrzf + (1.0 - ifiacrg) * (qiacrf + qracif)))
             + mx(0.0, qhlcev)
             + qhlacr + qhlacw
             + qhlacs + qhlaci
             + qhlcnh)
    pqhld = (qhlshr
             + oml5 * qhlmlr_
             + qhlsbv
             + mn(0.0, qhlcev)
             - qhlmul1 - qhcnhl)
    lim = qx[LHL] + dtp * (pqhli + pqhld) < 0.0
    frac = (-qx[LHL] + pqhli * dtp) / W(lim, pqhld * dtp, r(1.0))
    qhlmlr_ = W(lim, frac * qhlmlr_, qhlmlr_)
    qhlsbv = W(lim, frac * qhlsbv, qhlsbv)
    qhcnhl = W(lim, frac * qhcnhl, qhcnhl)
    qhlmul1 = W(lim, frac * qhlmul1, qhlmul1)
    qhlcev = W(lim & (qhlcev < 0.0), frac * qhlcev, qhlcev)
    pqhld = W(lim, frac * pqhld, pqhld)

    # 22691-22708 (not mixedphase)
    vhmlr = qhmlr_
    vhlmlr = qhlmlr_
    vhfzh = zero
    vhlfzhl = zero

    # ------------------------------------------------------------------ 23308-23486 graupel / hail volume
    rhofrz = r(C.rhofrz)
    pvhwi = (rho0 * (il5r * (ifiacrg * ffrzh * qracif) / rhofrz
                     + (il5r * qhdpv / r(C.qhdpvdn)
                        + (qhacs + qhaci) / r(C.qhacidn)))
             + rho0 * mx(0.0, qhcev) / 1000.
             + f2h * vhcns
             + vhacr + vhacw + vhfzh
             + f2h * vhcni + (ifiacrg * viacrf + ifrzg * vrfrzf) * ffrzh)
    pvhwd = (rho0 * ((oml5 * vhmlr
                      + qhsbv
                      + mn(0.0, qhcev)
                      - qhmul1) / xdn[LH])
             - vhlcnh + vhshdr - vhsoak - vscnh)
    pvhli = (rho0 * ((il5r * (((1.0 - ifiacrg) * ffrzh * qracif) / rhofrz + qhldpv)
                      + qhlacs + qhlaci) / 500.)
             + rho0 * mx(0.0, qhlcev) / 1000.
             + vhlcnhl + ((1.0 - ifiacrg) * ffrzh * viacrf + (1.0 - ifrzg) * ffrzh * vrfrzf)
             + vhlacr + vhlacw + vhlfzhl)
    pvhld = (rho0 * ((qhlsbv
                      + mn(0.0, qhlcev)
                      - qhlmul1) / xdn[LHL])
             + rho0 * oml5 * vhlmlr / xdn[LHL]
             + vhlshdr - vhlsoak)

    # ------------------------------------------------------------------ 23784-23864 latent heating
    imix = 1 - int(C.imixedphase)
    pfrz = (oml5 * (qhmlr_ + qsmlr_ + qhlmlr_)
            + (jnp.asarray(il5) * imix).astype(R) * (
                qsacw + qhacw + qhlacw + qsacr + qhacr + qhlacr + qsshr + qhshr + qhlshr + qrfrz + qiacr)
            + il5r * (qwfrz + qwctfz + qiihr + qiacw))
    pmlt = oml5 * (qhmlr_ + qsmlr_ + qhlmlr_)
    psub = (il5r * (qsdpv + qhdpv + qhldpv + qidpv + qisbv)
            + qssbv + qhsbv + qhlsbv
            + il5r * (qiint_))
    pvap = qrcev + qhcev + qscev + qhlcev + qfcev
    pi0 = g["pi0"]
    felfcp, felscp, felvcp = g["felfcp"], g["felscp"], g["felvcp"]
    ptem = (1. / pi0) * (felfcp * pfrz + felscp * psub + felvcp * pvap)
    thetap = g["thetap"] + dtp * ptem

    # ------------------------------------------------------------------ 23873-23983 state update
    qwvp = g["qwvp"] + dtp * (pqwvi + pqwvd)
    qx = dict(qx)
    qx[LC] = qx[LC] + dtp * (pqcwi + pqcwd)
    qx[LR] = qx[LR] + dtp * (pqrwi + pqrwd)
    qx[LI] = qx[LI] + dtp * (pqcii + pqcid)
    qx[LS] = qx[LS] + dtp * (pqswi + pqswd)
    qx[LH] = qx[LH] + dtp * (pqhwi + pqhwd)
    qx[LHL] = qx[LHL] + dtp * (pqhli + pqhld)
    vx = dict(vx)
    vx[LH] = vx[LH] + dtp * (pvhwi + pvhwd)
    vx[LHL] = vx[LHL] + dtp * (pvhli + pvhld)
    cx = dict(cx)
    cx[LI] = cx[LI] + dtp * (pccii + pccid)
    cina = g["cina"] + pccin * dtp
    cx[LC] = cx[LC] + dtp * (pccwi + pccwd)
    cx[LR] = cx[LR] + dtp * (pcrwi + pcrwd)
    cx[LS] = cx[LS] + dtp * (pcswi + pcswd)
    cx[LH] = cx[LH] + dtp * (pchwi + pchwd)
    cx[LHL] = cx[LHL] + dtp * (pchli + pchld)

    # ------------------------------------------------------------------ 24008-24046 melt cloud ice above 0C
    theta0, qv0, pk = g["theta0"], g["qv0"], g["pk"]
    theta = thetap + theta0
    temg = theta * pk
    qitmp = qx[LI]
    melt = (temg > tfr) & (qitmp > 0.0)
    qx[LC] = W(melt, qx[LC] + qitmp, qx[LC])
    ptem = W(melt, ptem + (1. / pi0) * felfcp * (-qitmp * dtpinv), ptem)
    pmlt = W(melt, pmlt - qitmp * dtpinv, pmlt)
    thetap = W(melt, thetap - g["fcc3"] * qitmp, thetap)
    cx[LC] = W(melt, cx[LC] + cx[LI], cx[LC])
    qx[LI] = W(melt, zero, qx[LI])
    cx[LI] = W(melt, zero, cx[LI])
    vx[LI] = W(melt, zero, vx[LI]) if LI in vx else zero
    # 24060-24157 homogeneous freezing: ibfc = 1 with ipconc = 5 -> branch never taken
    # 24167-24438 ipconc <= 1 saturation adjustment: dead

    # ------------------------------------------------------------------ 24455-25112 scatter + size limiter
    an = jnp.asarray(an)
    t0n = W(gathered, temg, jnp.asarray(t0))
    an_new = an
    an_new = an_new.at[LT].set(W(gathered, theta0 + thetap, an[LT]))
    an_new = an_new.at[LV].set(W(gathered, qwvp + qv0, an[LV]))
    for il in (LC, LR, LI, LS, LH, LHL):
        val = qx[il] + mn(an[il], 0.0)
        an_new = an_new.at[il].set(W(gathered, val, an[il]))
        qx[il] = val
    xvbar = {}
    for il in (LC, LR, LI, LS, LH, LHL):
        q = qx[il]
        c = cx[il]
        nonpos = q <= 0.0
        act = ~nonpos & (c > r(C.cxmin)) & (q > qxmin[il])
        xvi = rho0 * q / (xdn[il] * W(act, c, r(1.0)))
        if il == LR:
            a = alpha[LR]
            xvbarmax = xvmx[il] / ((4. + a) ** 3 / ((3. + a) * (2. + a) * (1. + a)))
        else:
            xvbarmax = xvmx[il] + zero
        if il == LS:
            xvbarmax = xvbarmax * mx(1., 100. / mn(100., xdn[LS]))
        clip = act & ((xvi < xvmn[il]) | (xvi > xvbarmax))
        xvc = mx(xvmn[il], mn(xvbarmax, xvi))
        c = W(clip, rho0 * q / (xvc * xdn[il]), c)
        c = W(nonpos, zero, c)
        cx[il] = c
        xvbar[il] = xvbarmax
        an_new = an_new.at[LN[il]].set(W(gathered, mx(c, 0.0), an[LN[il]]))
    an_new = an_new.at[LCCN].set(W(gathered, mx(0.0, g["ccnc"]), an[LCCN]))
    for il in (LI, LS, LH, LHL):
        lv_ = int(C.lvol[il])
        if lv_ >= 1:
            an_new = an_new.at[lv_].set(W(gathered, mx(0.0, vx[il]), an[lv_]))

    if return_state:
        state = dict(qhlcnh=qhlcnh, chlcnh=chlcnh, chlcnhhl=chlcnhhl, dg0=dg0, qhcni=qhcni, qhcns=qhcns,
                     qrcev=qrcev, crcev=crcev, qiint=qiint, ciint=ciint, pqwvi=pqwvi, pqwvd=pqwvd,
                     pqcwd=pqcwd, pqrwi=pqrwi, pqrwd=pqrwd, pqcii=pqcii, pqcid=pqcid, pqswi=pqswi,
                     pqswd=pqswd, pqhwi=pqhwi, pqhwd=pqhwd, pqhli=pqhli, pqhld=pqhld, pccii=pccii,
                     pccid=pccid, pccwd=pccwd, pcrwi=pcrwi, pcrwd=pcrwd, pcswi=pcswi, pcswd=pcswd,
                     pchwi=pchwi, pchwd=pchwd, pchli=pchli, pchld=pchld, pvhwi=pvhwi, pvhwd=pvhwd,
                     pvhli=pvhli, pvhld=pvhld, ptem=ptem, thetap=thetap, qwvp=qwvp, chmul1=chmul1,
                     qhmul1=qhmul1, vhsoak=vhsoak, vhlsoak=vhlsoak, qhwet=qhwet, qhshr=qhshr,
                     qhlshr=qhlshr, qisbv=qisbv, qssbv=qssbv, qhsbv=qhsbv, qhdpv=qhdpv, qidpv=qidpv,
                     qsdpv=qsdpv, qscev=qscev, qhcev=qhcev, qhlcev=qhlcev, qscni=qscni, cscni=cscni,
                     wetgrowth=wetgrowth, wetgrowthhl=wetgrowthhl, qx=qx, cx=cx, vx=vx, consumed=g.consumed)
        return an_new, (t0n, t1, t2, t3, t4, t5, t6, t7, t8, t9), state
    return an_new, (t0n, t1, t2, t3, t4, t5, t6, t7, t8, t9)
