"""NSSL 2-moment ``nssl_2mom_gs`` PART A (module lines 12620-16326), WRF default mp=18.

Covers: gs-local constants (13729-13968), the gather condition (14031-14071), state temporaries
(14106-14340), densities / shape parameters (14400-14807), thermodynamic factors (14492-14604),
the ``setvtz`` call (15562), number/size diagnostics and depletion limits (15626-15785) and the
collection efficiencies (15797-16326).

Vectorisation: every per-point quantity is computed at EVERY point (shape = the input field shape,
``(..., nz)``; for one column the point index is k-1); ``G['gathered']`` is the gather mask of
lines 14066-14071 (Fortran processes only those points; later parts must scatter with it).
Neighbour-level inputs (``kgsm/kgsp/kgsm2``) are resolved here from the column fields.

Dead branches for the default configuration (ipconc=5, imurain=1, imusnow=3, lzr=lzh=lzhl=0,
eqtset=1, iqvsopt=1, iqis0=2, ilimit=0, imydiagalpha=0, lwsm6=.false., mixedphase=.false.,
lis=0, lss=0, lcina=0, lcin=0, lvol(li)=lvol(ls)=0, exwmindiam=0, iehw=iehlw=1, ibfc=1,
iessopt=1, ibinnum=2, numshedregimes=3) are omitted with the killing flag named inline.

A -> B interface (locals assigned here AND read in lines 16327-25151; scratch temporaries such as
tmp/x/y/i/j/alp/fac, dead-configuration locals, and ``dnmx``/``tqvcon`` (reassigned before use)
are not listed): see ``A_TO_B_KEYS`` (166 names); all are checked against the oracle G1 dump in
``tests/v034/nssl2mom/test_gs_a.py`` (cdxgs/axx/bxx only for graupel/hail where present: Fortran
leaves them uninitialised elsewhere).  ``G`` additionally carries every other part-A local
computed here (gs-local constants such as ``vshd``, ``mltmass*``, ``massfacmlr``, ``maxmassfac``,
``qeps``, ``aradcw..dradcw``), the gather mask ``gathered`` and the gs dummy arguments later parts
read (``gz``(=dz), ``dn``, ``p2``(=t77), ``w``, ``t0``, ``t7``, ``t00``, ``dtp``).

Per-point arrays keep the input field shape (one column: index k-1, i.e. the flattened point
order of the Fortran k-major gather).  Integer level locals (kgsm/kgsp/kgsm2) are Fortran
1-based levels.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .indices import LT, LV, LC, LR, LI, LS, LH, LHL, LCCN, LNC, LNR, LNI, LNS, LNH, LNHL, LVH, LVHL, LCCNA, LHAB
from . import mathfun, satfun

F64 = jnp.float64

# species ranges (Fortran indices)
_LC_LHAB = tuple(range(LC, LHAB + 1))
_LR_LHAB = tuple(range(LR, LHAB + 1))
_LI_LHAB = tuple(range(LI, LHAB + 1))
_LS_LHAB = tuple(range(LS, LHAB + 1))
_LH_LHAB = tuple(range(LH, LHAB + 1))
_LV_LHAB = tuple(range(LV, LHAB + 1))

NDIAM = 10  # gs parameter ndiam
SHEDALP = 3.0  # gs parameter shedalp
RIMEDENS = 500.0  # gs parameter rimedens
EHS_COLLSN, EHI_COLLSN = 0.5, 1.0  # gs local initialisers (line 13454)
EHLS_COLLSN, EHLI_COLLSN = 1.0, 1.0
ESI_COLLSN = 1.0

# Collection-efficiency tables (gs DATA, lines 13459-13475), Fortran (row, col) 1-based
_CWR = ((2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0), (1.0, 1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 1.0))
_GRAD = ((100., 200., 300., 400., 600., 1000.), (1.e-2, 1.e-2, 1.e-2, 5.e-3, 2.5e-3, 1.))

A_TO_B_KEYS = (
    "alpha", "arz", "axx", "bfnu1", "brz", "bxx", "ccimx", "ccmxd", "ccnc", "ccwresv", "cdxgs",
    "chmxd", "cimasn", "cimxd", "cina", "cnina", "cninm", "cninp", "crmxd", "cs", "csmxd", "cx",
    "cxmxd", "da0lh", "da0lhl", "da0lr", "da1lc", "da1lr", "dab0lh", "dab1lh", "ds", "dtpinv",
    "ehi", "ehiclsn", "ehli", "ehliclsn", "ehlr", "ehls", "ehlsclsn", "ehlw", "ehr", "ehs",
    "ehsclsn", "ehscnv", "ehw", "eii", "eiw", "eri", "ers", "erw", "esi", "esiclsn", "esr", "ess",
    "esw", "fadvisc", "fai", "fakvisc", "fav", "fbi", "fbv", "fcc3", "fci", "fcqv1", "fcqv2", "fcw",
    "felf", "felfcp", "fels", "felscp", "felv", "felvcp", "ffrzh", "fhlw", "fhw", "fpndl",
    "frac", "fschm", "ftka", "fwvdf", "g1shr", "g1smlr", "g1x", "galpharaut", "gamice73fac",
    "gamsnow73fac", "gf1", "gf1palp", "gf2", "gf3", "gf4", "gf5", "gf6", "gf73rds", "gf83rds",
    "il2", "il3", "il5", "kgsm", "kgsp", "ldovol", "lrescalelow", "pi0", "pii", "pinit",
    "pipert", "pk", "pqs", "pres", "qcmxd", "qcwresv", "qhlmxd", "qhmxd", "qhshr", "qimxd", "qis",
    "qrmxd", "qsmxd", "qss", "qss0", "qv0", "qvimxd", "qvs", "qwvp", "qx", "qxmxd", "qxw",
    "raindn", "rb", "rh", "rho0", "rhoinv", "rhovt", "rimdn", "rzxh", "rzxhl", "rzxhlh", "rzxs",
    "scx", "snowmeltmass", "ssi", "ssw", "temcg", "temg", "temgkm2", "tfrcbi", "tfrcbw",
    "theta", "theta0", "thetap", "tsqr", "ventrx", "ventrxn", "vfrz", "vmlt",
    "vshdgs", "vtxbar", "vx", "wvel", "xdia", "xdn", "xl2p", "xmas", "xmascw", "xv",
    "xvbiggsnow",
)


def _w(cond, a, b):
    return jnp.where(cond, a, b)


def gs_part_a(an, t0, t7, t00, t77, pn, wn, dn, dz, dtp, C, prec, setvtz_fn=None):
    """Port of nssl_2mom_gs lines 12620-16326.

    ``an``: (NA+1, ..., nz) density-scaled state at gs entry (driver S2 stage); ``t0..wn``: (..., nz)
    columns (t0 temperature, t7 primary ice nucleation rate, t00 = 380/p, t77 = pii (passed as p2),
    pn pressure, wn w, dn air density, dz layer depth). Returns the gs-local dict ``G``.

    ``setvtz_fn(G, C, prec) -> dict`` overrides the setvtz call (testing only); default
    ``fallspeed.setvtz``.
    """
    R = prec.R
    r = C.R  # numpy scalar type of REAL literals
    an = jnp.asarray(an, R)
    t0 = jnp.asarray(t0, R)
    t7 = jnp.asarray(t7, R)
    t00 = jnp.asarray(t00, R)
    t77 = jnp.asarray(t77, R)
    pn = jnp.asarray(pn, R)
    wn = jnp.asarray(wn, R)
    dn = jnp.asarray(dn, R)
    shape = t0.shape
    nz = shape[-1]
    zero = jnp.zeros(shape, R)
    one = jnp.ones(shape, R)
    full = lambda v: jnp.full(shape, v, R)
    G: dict = {}
    # gs dummy arguments later parts read directly (gz = dz, dn, p2 = t77, w, t0, t7, t00)
    G["gz"], G["dn"], G["p2"], G["w"] = jnp.asarray(dz, R), dn, t77, wn
    G["t0"], G["t7"], G["t00"], G["dtp"] = t0, t7, t00, dtp

    # ------------------------------------------------------------------ constants 13729-13968
    dtpinv = r(1.0 / np.float64(dtp))  # dtpinv = 1.d0/dtp (double, stored REAL); dtp is a static Python/NumPy scalar
    G["dtpinv"] = dtpinv
    G["pinit"] = r(0.0)
    G["ldovol"] = any(int(C.lvol[il]) > 1 for il in _LC_LHAB)
    G["lrescalelow"] = {il: bool(C.rescale_low_alpha) for il in _LC_LHAB}
    G["lrescalelow"][LR] = bool(C.rescale_low_alphar and C.rescale_low_alpha)
    G["lrescalelow"][LH] = bool(C.rescale_low_alphah and C.rescale_low_alpha)
    G["lrescalelow"][LHL] = bool(C.rescale_low_alphahl and C.rescale_low_alpha)
    G["ffrzh"] = r(1.0)
    G["qeps"] = r(1.0e-20)
    G["aradcw"], G["bradcw"] = r(-0.27544), r(0.26249e+06)
    G["cradcw"], G["dradcw"] = r(-1.8896e+10), r(4.4626e+14)
    G["bta1"], G["cnit"], G["dragh"], G["dnz00"] = r(0.6), r(1.0e-02), r(0.60), r(1.225)
    G["cs"], G["ds"] = r(12.42), r(0.42)
    G["pii"] = C.piinv
    G["pid4"] = C.pi / r(4.0)
    G["gf1"], G["gf1p5"], G["gf2"], G["gf3"] = r(1.0), r(0.8862269255), r(1.0), r(2.0)
    G["gf3p5"], G["gf4"], G["gf5"], G["gf6"], G["gf7"] = r(3.32335097), r(6.00), r(24.0), r(120.0), r(720.0)
    G["gf4br"], G["gf4ds"], G["gf4p5"] = r(17.837861981813607), r(10.41688578110938), r(11.63172839656745)
    G["gf3ds"], G["gf1ds"] = r(3.0458730354120997), r(0.8863557896089221)
    G["gf43rds"], G["gf53rds"], G["gf73rds"], G["gf83rds"] = r(0.8929795116), r(0.9027452930), r(1.190639349), r(1.504575488)
    g73i = mathfun.gamma_sp(r(7.) / r(3.) + C.cinu, R)
    g1i = mathfun.gamma_sp(r(1.) + C.cinu, R)
    G["gamice73fac"] = g73i ** 3 / (g1i ** 3 * (r(1.) + C.cinu) ** 4)
    g73s = mathfun.gamma_sp(r(7.) / r(3.) + C.snu, R)
    g1s = mathfun.gamma_sp(r(1.) + C.snu, R)
    G["gamsnow73fac"] = g73s ** 3 / (g1s ** 3 * (r(1.) + C.snu) ** 4)
    G["brz"], G["arz"] = r(100.0), r(0.66)
    ar = C.alphar
    G["bfnu1"] = (r(4.) + ar) * (r(5.) + ar) * (r(6.) + ar) / ((r(1.) + ar) * (r(2.) + ar) * (r(3.) + ar))
    au = r(C.alpharaut)
    G["galpharaut"] = (r(6.) + au) * (r(5.) + au) * (r(4.) + au) / ((r(3.) + au) * (r(2.) + au) * (r(1.) + au))
    G["vfrz"] = r(0.523599) * C.dfrz ** 3
    G["vmlt"] = min(C.xvmx[LR], r(0.523599) * C.dmlt ** 3)
    vshd = min(C.xvmx[LR], r(0.523599) * C.dshd ** 3)
    G["vshd"] = vshd
    G["snowmeltmass"] = r(0.0)  # snowmeltdia = 0: keeps its initialiser
    G["tdtol"] = r(1.0e-05)
    G["tfrcbw"] = C.tfr - C.cbw
    G["tfrcbi"] = C.tfr - C.cbi
    G["mltmass0inv"] = r(1.0) / (r(1000.0) * C.xvmx[LR])
    pi = C.pi
    for n, ts in ((1, C.takshedsize1), (2, C.takshedsize2), (3, C.takshedsize3)):
        G[f"mltmass{n}inv"] = r(1.0) / (r(1000.0) * (r(4.0) * pi / r(3.0)) * ((r(0.01) * r(0.5) * ts) ** 3))
        G[f"mltmass{n}cgs"] = r(1.0) * (r(4.0) * pi / r(3.0)) * ((r(0.5) * ts) ** 3)
    # ibinnum == 2, numshedregimes == 3
    G["numdiam"] = 2
    mltdiam = {1: C.mltdiam1 / r(6.), 2: C.mltdiam1 / r(2.), NDIAM + 1: C.mltdiam1, NDIAM + 2: C.mltdiam2,
               NDIAM + 3: C.mltdiam3, NDIAM + 4: C.mltdiam4}
    G["mltdiam"] = mltdiam
    G["mwfac"] = r(6.0) ** (r(1.) / r(3.))
    G["rwmasn"] = C.xvmn[LR] * r(1000.)
    G["rwmasx"] = C.xvmx[LR] * r(1000.)
    G["xvbiggsnow"] = F64(C.xvmn[LH])  # biggsnowdiam <= 0; DOUBLE local
    G["cimasn"] = min(C.cimas0, C.cimas1)
    G["cimasx"] = r(1.0e-8)
    G["ccimx"] = r(5000.0e3)

    # ------------------------------------------------------------------ gather condition 14031-14071
    pres1 = pn + r(0.0)  # pb(kz) = 0
    lt1 = satfun.ltemq_index(t0, C, R)
    qvs1 = C.rdorv * C.esbolton * satfun.tabqvs(lt1, C, R) / (pres1 - C.esbolton * satfun.tabqvs(lt1, C, R))
    ltfr = satfun.ltemq_index(C.tfr, C, R)  # (tfr - 163.15)/fqsat + 1.5
    qis1 = _w(t0 <= C.tfr + r(0.5), t00 * satfun.tabqis(lt1, C, R), t00 * satfun.tabqis(ltfr, C, R))
    qss1 = _w(t0 < C.tfr, qis1, qvs1)
    ishail = an[LHL] > C.qxmin[LHL]
    gathered = ((an[LV] > qss1) | (an[LC] > C.qxmin[LC]) | (an[LI] > C.qxmin[LI]) | (an[LR] > C.qxmin[LR])
                | (an[LS] > C.qxmin[LS]) | (an[LH] > C.qxmin[LH]) | ishail)
    G["gathered"] = gathered

    # ------------------------------------------------------------------ temporaries 14089-14154
    k1 = jnp.arange(1, nz + 1)  # Fortran level index
    kgsm = jnp.maximum(k1 - 1, 1)
    kgsp = jnp.minimum(k1 + 1, nz - 1)
    kgsm2 = jnp.maximum(k1 - 2, 1)
    G["kgsm"], G["kgsp"], G["kgsm2"] = jnp.broadcast_to(kgsm, shape), jnp.broadcast_to(kgsp, shape), jnp.broadcast_to(kgsm2, shape)
    take = lambda f, k: jnp.take(f, k - 1, axis=-1)

    G["raindn"] = {il: full(900.) for il in _LI_LHAB}
    G["rimdn"] = {il: full(RIMEDENS) for il in _LI_LHAB}
    G["theta0"] = an[LT]
    G["thetap"] = an[LT] - an[LT]
    G["theta"] = an[LT]
    G["qv0"] = an[LV]
    G["qwvp"] = an[LV] - an[LV]
    pres = pn + r(0.0)
    G["pres"] = pres
    G["pipert"] = t77
    rho0 = dn
    G["rho0"] = rho0
    rhoinv = r(1.0) / rho0
    G["rhoinv"] = rhoinv
    G["rhovt"] = jnp.sqrt(C.rho00 / jnp.maximum(r(0.05), rho0))
    pi0 = t77 + r(0.0)  # p2 + pinit
    G["pi0"] = pi0
    temg = t0
    G["temg"] = temg
    G["temgkm1"] = take(t0, kgsm)
    G["temgkm2"] = take(t0, kgsm2)
    G["pk"] = t77 + r(0.0)
    temcg = temg - C.tfr
    G["temcg"] = temcg
    G["tqvcon"] = temg - C.cbw  # gather-loop scalar, value per point
    G["qss0"] = r(380.0) / pres
    pqs = r(380.0) / pres
    G["pqs"] = pqs
    ltemq = satfun.ltemq_index(temg, C, R)
    qvs = C.rdorv * C.esbolton * satfun.tabqvs(ltemq, C, R) / (pres - C.esbolton * satfun.tabqvs(ltemq, C, R))
    qis = _w(temg <= C.tfr + r(0.5), pqs * satfun.tabqis(ltemq, C, R), pqs * satfun.tabqis(ltfr, C, R))
    G["qvs"], G["qis"] = qvs, qis
    G["qss"] = qvs
    G["cnostmp"] = full(C.cno[LS])
    G["il5"] = (temg < C.tfr).astype(jnp.int32)
    G["qhshr"] = zero  # (lwsm6 dead)

    qx = {il: jnp.maximum(an[il], r(0.0)) for il in _LV_LHAB}
    G["qxw"] = {il: zero for il in _LS_LHAB}
    G["qxwlg"] = {il: zero for il in _LS_LHAB}

    cx = {il: zero for il in _LC_LHAB}
    # ipconc >= 1
    cx[LI] = jnp.maximum(an[LNI], r(0.0))
    cx[LI] = _w(qx[LI] <= C.qxmin[LI], r(0.0), cx[LI])
    G["cina"] = cx[LI]  # lcina = 0
    # ipconc >= 2
    cx[LC] = jnp.maximum(an[LNC], r(0.0))
    cx[LC] = _w(qx[LC] <= C.qxmin[LC], r(0.0), cx[LC])
    G["ccnc"] = an[LCCN]
    G["ccna"] = an[LCCNA]

    def _num(il, ln, zero_below):
        c = jnp.maximum(an[ln], r(0.0))
        below = qx[il] <= C.qxmin[il]
        fold = (~below) & (c == 0.0) & (qx[il] < r(3.0) * C.qxmin[il])
        rest = (~below) & (~fold)
        qx[LV] = _w(fold, qx[LV] + qx[il], qx[LV])
        qx[il] = _w(fold, r(0.0), qx[il])
        c = _w(rest, jnp.maximum(r(1.e-9), c), c)
        if zero_below:
            c = _w(below, r(0.0), c)
        # ilimit (=0) >= ipc(il) is false for ls/lh/lhl: no intercept limiter
        cx[il] = c

    _num(LR, LNR, False)  # ipconc >= 3
    _num(LS, LNS, False)  # ipconc >= 4
    _num(LH, LNH, False)  # ipconc >= 5
    _num(LHL, LNHL, True)  # lhl > 1

    # volumes (ldovol): lvol(lh) = lvh, lvol(lhl) = lvhl; lvol(li) = lvol(ls) = 0
    vx = {il: zero for il in _LI_LHAB}
    vx[LH] = jnp.maximum(an[LVH], r(0.0))
    vx[LHL] = jnp.maximum(an[LVHL], r(0.0))
    G["fhw"], G["fsw"], G["fhlw"] = zero, zero, zero

    # ipconc == 5: g1x factors (lines 14400-14410)
    def _g1(a):
        return (r(6.0) + a) * (r(5.0) + a) * (r(4.0) + a) / ((r(3.0) + a) * (r(2.0) + a) * (r(1.0) + a))
    G["g1x"] = {LR: full(_g1(C.alphar)), LH: full(_g1(C.alphah)), LHL: full(_g1(C.alphahl))}
    G["scx"] = {il: zero for il in _LC_LHAB}

    # shape parameters (14417-14444)
    alpha = {il: zero for il in _LC_LHAB}
    alpha[LR] = full(C.alphar)  # imurain == 1
    alpha[LI] = full(C.xnu[LI])
    alpha[LC] = full(C.xnu[LC])
    alpha[LS] = full(C.xnu[LS])  # imusnow == 3
    for il in _LR_LHAB:
        if il >= C.lg:
            alpha[il] = full(C.dnu[il])
    G["dab0lh"] = {(il, ic): full(C.dab0[il, ic]) for il in _LR_LHAB for ic in _LC_LHAB}
    G["dab1lh"] = {(il, ic): full(C.dab1[il, ic]) for il in _LR_LHAB for ic in _LC_LHAB}
    G["da0lx"] = {il: full(C.da0[il]) for il in _LR_LHAB}
    G["da0lh"] = full(C.da0[LH])
    G["da0lr"] = full(C.da0[LR])
    G["da1lr"] = full(C.da1[LR])
    G["da0lc"] = full(C.da0[LC])
    G["da1lc"] = full(C.da1[LC])
    G["rzxhlh"] = full(C.rzhl / C.rz)  # lzh < 1
    G["rzxh"] = full(C.rz)  # lzr <= 1
    G["rzxhl"] = full(C.rzhl)
    G["rzxs"] = full(C.rzs)  # imurain == 1, imusnow == 3, lzr < 1
    G["da0lhl"] = full(C.da0[LHL])
    G["ventrx"] = full(C.ventr)
    G["ventrxn"] = full(C.ventrn)
    G["gf1palp"] = jnp.broadcast_to(mathfun.gamma_sp(r(1.0) + C.alphar, R), shape)

    # thermodynamic factors (14492-14604)
    G["ssi"] = qx[LV] / qis
    G["ssw"] = qx[LV] / qvs
    G["tsqr"] = temg ** 2
    temgx = jnp.minimum(temg, r(313.15))
    temgx = jnp.maximum(temgx, r(233.15))
    felv = r(2500837.367) * (r(273.15) / temgx) ** ((r(0.167)) + (r(3.67e-4)) * temgx)
    temcgx = jnp.minimum(temg, r(273.15))
    temcgx = jnp.maximum(temcgx, r(223.15))
    temcgx = temcgx - r(273.15)
    felf = r(333690.6098) + (r(2030.61425)) * temcgx - (r(10.46708312)) * temcgx ** 2
    fels = felv + felf
    G["felv"], G["felf"], G["fels"] = felv, felf, fels
    G["felvs"] = felv * felv
    G["felss"] = fels * fels
    felvcp = felv * C.cpi  # eqtset <= 1
    felscp = fels * C.cpi
    felfcp = felf * C.cpi
    G["felvcp"], G["felscp"], G["felfcp"] = felvcp, felscp, felfcp
    fgamw = felvcp / pi0
    fgams = felscp / pi0
    G["fgamw"], G["fgams"] = fgamw, fgams
    G["fcqv1"] = r(4098.0258) * pi0 * fgamw
    G["fcqv2"] = r(5807.6953) * pi0 * fgams
    G["fcc3"] = felfcp / pi0
    fwvdf = (r(2.11e-05)) * ((temg / C.tfr) ** r(1.94)) * (r(101325.0) / (pres))
    fadvisc = C.advisc0 * (r(416.16) / (temg + r(120.0))) * (temg / r(296.0)) ** (r(1.5))
    fakvisc = fadvisc * rhoinv
    G["fwvdf"], G["fadvisc"], G["fakvisc"] = fwvdf, fadvisc, fakvisc
    temcgx = jnp.minimum(temg, r(273.15))
    temcgx = jnp.maximum(temcgx, r(233.15))
    temcgx = temcgx - r(273.15)
    G["fci"] = (r(2.118636) + r(0.007371) * (temcgx)) * (r(1.0e+03))
    cold = temg < r(273.15)
    tcc = jnp.maximum(jnp.minimum(temg, r(273.15)), r(233.15)) - r(273.15)
    fcw_c = r(4203.1548) + (r(1.30572e-2)) * ((tcc - r(35.)) ** 2) + (r(1.60056e-5)) * ((tcc - r(35.)) ** 4)
    tcw = jnp.maximum(jnp.minimum(temg, r(308.15)), r(273.15)) - r(273.15)
    fcw_w = r(4243.1688) + (r(3.47104e-1)) * (tcw ** 2)
    G["fcw"] = _w(cold, fcw_c, fcw_w)
    G["temcgx"] = _w(cold, tcc, tcw)
    ftka = C.tka0 * fadvisc / C.advisc1
    fthdf = ftka * C.cpi * rhoinv
    G["ftka"], G["fthdf"] = ftka, fthdf
    G["fschm"] = (fakvisc / fwvdf)
    G["fpndl"] = (fakvisc / fthdf)
    G["fai"] = (fels ** 2) / (ftka * C.rw * temg ** 2)
    G["fbi"] = (r(1.0) / (rho0 * fwvdf * qis))
    G["fav"] = (felv ** 2) / (ftka * C.rw * temg ** 2)
    G["fbv"] = (r(1.0) / (rho0 * fwvdf * qvs))
    kp1 = jnp.minimum(nz, k1 + 1)
    G["wvel"] = (r(0.5)) * (take(wn, kp1) + wn)

    # densities (14616-14672)
    xdn = {il: zero for il in _LC_LHAB}
    for il in (LI, LC, LR, LS, LH):
        xdn[il] = full(C.xdn0[il])
    # lvol(ls) = 0: no snow density
    dnmx = C.xdnmx[LH]  # mixedphase = .false.
    c1 = (vx[LH] > 0.0) & (qx[LH] > C.qxmin[LH])
    xdn[LH] = _w(c1, jnp.minimum(dnmx, jnp.maximum(C.xdnmn[LH], rho0 * qx[LH] / _w(c1, vx[LH], one))), xdn[LH])
    vx[LH] = _w(c1, rho0 * qx[LH] / xdn[LH], vx[LH])
    c2 = (~c1) & (vx[LH] == 0.0) & (qx[LH] > C.qxmin[LH])
    vx[LH] = _w(c2, rho0 * qx[LH] / xdn[LH], vx[LH])
    xdn[LHL] = full(C.xdn0[LHL])
    xdntmp = {LHL: full(C.xdn0[LHL])}
    dnmx = C.xdnmx[LHL]
    c1 = (vx[LHL] > 0.0) & (qx[LHL] > C.qxmin[LHL])
    xdn[LHL] = _w(c1, jnp.minimum(dnmx, jnp.maximum(C.xdnmn[LHL], rho0 * qx[LHL] / _w(c1, vx[LHL], one))), xdn[LHL])
    vx[LHL] = _w(c1, rho0 * qx[LHL] / xdn[LHL], vx[LHL])
    xdntmp[LHL] = _w(c1, xdn[LHL], xdntmp[LHL])
    c2 = (~c1) & (vx[LHL] == 0.0) & (qx[LHL] > C.qxmin[LHL])
    vx[LHL] = _w(c2, rho0 * qx[LHL] / xdn[LHL], vx[LHL])
    G["dnmx"] = dnmx
    G["xdntmp"] = xdntmp
    # (imydiagalpha == 2 block 14674-14776 dead)

    # rain shape factors for melting/shedding (imurain == 1, lzr <= 1)
    alphashr = C.alphar
    alphamlr = C.alphar
    alphasmlr = C.alphar
    G["alphashr"], G["alphamlr"], G["alphasmlr"] = alphashr, alphamlr, alphasmlr
    massfacshr = (r(3.0) + alphashr) ** 3 / ((r(3.) + alphashr) * (r(2.) + alphashr) * (r(1.) + alphashr))
    G["massfacshr"] = massfacshr
    G["massfacmlr"] = (r(3.0) + alphamlr) ** 3 / ((r(3.) + alphamlr) * (r(2.) + alphamlr) * (r(1.) + alphamlr))
    G["g1shr"], G["g1mlr"], G["g1smlr"] = r(1.0), r(1.0), r(1.0)
    # (ipconc >= 6 blocks 14817-15525 dead)

    G["wvelkm1"] = (r(0.5)) * (wn + take(wn, kgsm))
    G["cninm"] = take(t7, kgsm)
    cnina = t7
    G["cnina"] = cnina
    G["cninp"] = take(t7, kgsp)

    # state before setvtz
    G["qx"], G["cx"], G["vx"], G["xdn"], G["alpha"] = qx, cx, vx, xdn, alpha
    G["xv"] = {il: zero for il in _LC_LHAB}
    G["xmas"] = {il: zero for il in _LC_LHAB}
    G["vtxbar"] = {(il, m): zero for il in _LC_LHAB for m in (1, 2, 3)}
    G["xdia"] = {(il, m): zero for il in _LC_LHAB for m in (1, 2, 3)}

    # ------------------------------------------------------------------ setvtz (15559-15566)
    G["infdo"] = 1
    if setvtz_fn is None:
        from .fallspeed import setvtz as _setvtz
        out = _setvtz_call(_setvtz, G, C, prec)
    else:
        out = setvtz_fn(G, C, prec)
    G.update(out)
    # setvtz rain branch (module lines 7075-7079): with infdo < 2 Fortran sets the Z-weighted rain
    # fall speed to the mass-weighted one where qr > qxmin. Idempotent guard (fallspeed.setvtz
    # omitted this ELSE branch when first ported).
    if G["infdo"] < 2:
        vt = dict(G["vtxbar"])
        vt[(LR, 3)] = jnp.where(G["qx"][LR] > C.qxmin[LR], vt[(LR, 1)], vt[(LR, 3)])
        G["vtxbar"] = vt

    _post_setvtz(G, C, prec, shape)
    return G


def _setvtz_call(setvtz, G, C, prec):
    """Bind the gs call of setvtz (line 15562) to fallspeed.setvtz (ildo = 0, infdo = 1).

    cdxgs/axx/bxx are uninitialised gs locals at this point in Fortran; they enter as zeros.
    """
    shape = G["temg"].shape
    z = jnp.zeros(shape, prec.R)
    cdxgs = {il: z for il in _LC_LHAB}
    axx = {il: z for il in _LH_LHAB}
    bxx = {il: z for il in _LH_LHAB}
    return setvtz(G["qx"], G["cx"], G["rho0"], G["rhovt"], G["xdia"], G["xmas"], G["vtxbar"], G["xdn"],
                  G["xv"], cdxgs, G["fadvisc"], G["temcg"], G["alpha"], axx, bxx, C=C, prec=prec,
                  infdo=G["infdo"], ildo=0, cnostmp=G["cnostmp"])


def _post_setvtz(G, C, prec, shape):
    """Lines 15570-16326 (after setvtz)."""
    R = prec.R
    r = C.R
    zero = jnp.zeros(shape, R)
    one = jnp.ones(shape, R)
    qx, cx, xv, xmas, xdn, xdia, alpha = G["qx"], G["cx"], G["xv"], G["xmas"], G["xdn"], G["xdia"], G["alpha"]
    vtxbar = G["vtxbar"]
    temg, temcg, rho0, rhoinv = G["temg"], G["temcg"], G["rho0"], G["rhoinv"]
    dtpinv = G["dtpinv"]
    D = F64
    # (lwsm6/ipconc<5 blocks 15570-15624 dead)

    # droplet autoconversion helpers (ipconc >= 2), DOUBLE rb/xl2p/rh/nh
    dl = lambda v: jnp.asarray(r(v), D)  # REAL literal promoted to DOUBLE
    rb = (r(0.5) * xdia[(LC, 1)] * (r(1.) / (r(1.) + alpha[LC])) ** (r(1.) / r(6.))).astype(D)
    xl2p = jnp.maximum(D(0.0), (r(2.7e-2) * xdn[LC] * cx[LC] * xv[LC]).astype(D)
                       * ((dl(0.5e20) * rb ** 3 * xdia[(LC, 1)].astype(D)) - dl(0.4)))
    big = rb > dl(3.51e-6)
    rh = jnp.where(big, jnp.maximum(D(41.e-6), D(6.3e-4) / (D(1.e6) * (jnp.where(big, rb, D(1.0)) - D(3.5e-6)))), D(41.e-6))
    nh = jnp.where(xl2p > 0.0, D(4.2e9) * xl2p, dl(1.e30))
    G["rb"], G["xl2p"], G["rh"], G["nh"] = rb, xl2p, rh, nh

    # depletion limits (15656-15716)
    qvimxd = r(0.70) * (qx[LV] - G["qis"]) * dtpinv
    qvimxd = jnp.where(qx[LC] < C.qxmin[LC], r(0.99) * (qx[LV] - G["qis"]) * dtpinv, qvimxd)
    G["qvimxd"] = jnp.maximum(qvimxd, r(0.0))
    frac = D(0.1)
    G["frac"] = frac
    fd = lambda x: (frac * x.astype(D) * D(dtpinv)).astype(R)  # frac*x*dtpinv (DOUBLE) -> REAL
    G["qimxd"] = fd(qx[LI])
    G["qcmxd"] = fd(qx[LC])
    G["qrmxd"] = fd(qx[LR])
    G["qsmxd"] = fd(qx[LS])
    G["qhmxd"] = fd(qx[LH])
    G["qhlmxd"] = fd(qx[LHL])
    G["ccmxd"] = fd(cx[LC])
    G["cimxd"] = fd(cx[LI])
    G["crmxd"] = fd(cx[LR])
    G["csmxd"] = fd(cx[LS])
    G["chmxd"] = fd(cx[LH])
    G["qxmxd"] = {LV: jnp.maximum(r(0.0), r(0.1) * (qx[LV] - G["qvs"]) * dtpinv)}
    G["cxmxd"] = {}
    for il in _LC_LHAB:
        G["qxmxd"][il] = fd(qx[il])
        G["cxmxd"][il] = fd(cx[il])

    # maximum-mass factors (15737-15760)
    def _mmf_v(nu):
        return (r(2.) + r(3.) * (r(1.) + nu)) ** 3 / (r(3.) * (r(1.) + nu))

    def _mmf_d(a):
        return (r(3.0) + a) ** 3 / ((r(3.) + a) * (r(2.) + a) * (r(1.) + a))

    G["maxmassfac"] = {LC: _mmf_v(C.xnu[LC]), LI: _mmf_v(C.xnu[LI]), LR: _mmf_d(C.alphar),
                       LS: _mmf_v(C.alphas), LH: _mmf_d(C.alphah), LHL: _mmf_d(C.alphahl)}

    # shed-drop volumes (15764-15785), ivshdgs = 1
    massfacshr = G["massfacshr"]
    vshdgs = {}
    for il in _LH_LHAB:
        v = jnp.full(shape, G["vshd"], R)
        tmpdiam = (r(SHEDALP) + r(0.0)) * xdia[(il, 1)]
        v_big = jnp.full(shape, r(0.523599) * r(1.5e-3) ** 3 / massfacshr, R)
        v_mid = jnp.full(shape, r(0.523599) * r(3.0e-3) ** 3 / massfacshr, R)
        v_sm = jnp.minimum(C.xvmx[LR], r(6.) / C.pi * xdn[il] * r(0.001) * tmpdiam ** 3) / massfacshr
        vv = jnp.where(tmpdiam > C.sheddiam0, v_big, jnp.where(tmpdiam > C.sheddiam, v_mid, v_sm))
        vshdgs[il] = jnp.where(qx[il] > C.qxmin[il], vv, v)
    G["vshdgs"] = vshdgs

    # ------------------------------------------------------------------ efficiencies 15797-16326
    qmin = C.qxmin
    has = {il: qx[il] > qmin[il] for il in _LC_LHAB}
    G["qcwresv"] = zero
    G["ccwresv"] = zero  # exwmindiam = 0
    # icwr / irwr / igwr / ihlr
    def _bin(cond, rad, tab, n):
        idx = jnp.ones(shape, jnp.int32)
        for il in range(1, n + 1):
            idx = jnp.where(cond & (rad >= r(1.e-6) * r(tab[il - 1])), il, idx)
        return idx
    G["icwr"] = _bin(has[LC], r(0.5) * xdia[(LC, 1)], _CWR[0], 8)
    G["irwr"] = _bin(has[LR], r(0.5) * xdia[(LR, 3)], _GRAD[0], 6)
    G["igwr"] = _bin(has[LH], r(0.5) * xdia[(LH, 3)], _GRAD[0], 6)
    G["ihlr"] = _bin(has[LHL], r(0.5) * xdia[(LHL, 3)], _GRAD[0], 6)

    # ice-ice
    eii = jnp.where(has[LI], jnp.exp(r(0.025) * jnp.minimum(temcg, r(0.0))), zero)
    eii = jnp.where(has[LI] & (temg > r(273.15)), r(1.0), eii)
    G["eii"] = eii
    # ice-cloud water
    eiw = jnp.where(has[LI] & has[LC] & (xdia[(LC, 1)] > C.ewi_dcmin) & (xdia[(LI, 1)] > C.ewi_dimin), C.eiw0, zero)
    eiw = jnp.where(has[LI] & has[LC] & (temg >= r(273.15)), r(0.0), eiw)
    G["eiw"] = eiw
    # rain-cloud (lnr > 1)
    erw = jnp.where(has[LR] & has[LC], r(1.0), zero)
    G["erw"] = jnp.where(cx[LC] <= 0.0, r(0.0), erw)
    G["err"] = jnp.where(has[LR], r(1.0), zero)
    G["ers"] = jnp.where(has[LR] & has[LS], r(1.0), zero)
    eri = jnp.where(has[LR] & has[LI], C.eri0, zero)
    G["eri"] = jnp.where(has[LR] & has[LI] & (xdia[(LI, 3)] < C.eri_cimin), r(0.0), eri)
    # snow-cloud
    c = has[LS] & has[LC]
    esw = jnp.where(c, r(0.5), zero)
    big = (xdia[(LC, 1)] > r(15.e-6)) & (xdia[(LS, 1)] > r(100.e-6))
    esw = jnp.where(c & big, r(0.5), esw)
    esw = jnp.where(c & (~big) & (xdia[(LS, 1)] >= r(500.e-6)),
                    jnp.minimum(r(0.5), r(0.05) + (r(0.8) - r(0.05)) / (r(40.e-6)) * xdia[(LC, 1)]), esw)
    G["esw"] = esw
    # snow-rain
    c = has[LS] & has[LR] & (temg < C.tfr - r(1.))
    xv_lr = jnp.where(c, xv[LR], one)
    xd_ls = jnp.where(c, xdia[(LS, 1)], one)
    G["esr"] = jnp.where(c, jnp.exp(-(r(40.e-6) ** 3) / xv_lr) * jnp.exp(-r(40.e-6) / xd_ls), zero)
    G["il2"] = jnp.where(c & (qx[LS] < r(1.e-4)) & (qx[LR] < r(1.e-4)), 1, 0).astype(jnp.int32)
    G["il3"] = jnp.zeros(shape, jnp.int32)  # ipconc < 3 dead
    # snow aggregation (iessopt = 1)
    fac = abs(C.ess0)
    ramp = fac * jnp.exp(C.ess1 * (C.esstem2)) * (temcg - C.esstem1) / (C.esstem2 - C.esstem1)
    hi = fac * jnp.exp(C.ess1 * jnp.minimum(temcg, r(0.0)))
    ess = jnp.where((temcg > C.esstem1) & (temcg < C.esstem2), ramp, jnp.where(temcg >= C.esstem2, hi, zero))
    G["ess"] = jnp.where((temcg < 0.0) & ~(temcg < C.esstem1), ess, zero)  # ipconc >= 4
    G["ehsfac"] = one
    # snow-ice
    c = has[LS] & has[LI]
    G["esiclsn"] = jnp.where(c, r(ESI_COLLSN), zero)
    esi = jnp.minimum(r(0.1), C.esi0 * jnp.exp(r(0.1) * jnp.minimum(temcg, r(0.0))))
    esi = jnp.where(temg > r(273.15), r(0.0), esi)
    G["esi"] = jnp.where(c, esi, zero)
    # graupel
    G["xmascw"] = xmas[LC]
    c = has[LH] & has[LC]
    cwrad = r(0.5) * xdia[(LC, 1)]
    poly = jnp.minimum((G["aradcw"] + cwrad * (G["bradcw"] + cwrad * (G["cradcw"] + cwrad * (G["dradcw"])))), r(1.0))
    ehw = jnp.minimum(C.ehw0, C.ewfac * poly)  # iehw == 1
    ehw = jnp.where(xdia[(LC, 1)] < r(2.4e-06), r(0.0), ehw)
    ehw = jnp.minimum(C.ehw0, ehw)
    G["ehw"] = jnp.where(c, ehw, zero)
    c = has[LH] & has[LR]
    xd_r = jnp.where(c, xdia[(LR, 3)], one)
    xd_h = jnp.where(c, xdia[(LH, 3)], one)
    ehr = jnp.exp(-(r(40.e-6)) / xd_r) * jnp.exp(-r(40.e-6) / xd_h)
    G["ehr"] = jnp.where(c, jnp.minimum(C.ehr0, ehr), zero)
    ehscnv = jnp.where(has[LS], C.ehs0 * jnp.exp(C.ehs1 * jnp.minimum(temcg, r(0.0))), zero)
    G["ehscnv"] = ehscnv
    c = has[LS] & has[LH] & (qx[LC] >= qmin[LC])
    xs3 = xdia[(LS, 3)]
    clsn = jnp.where(xs3 < r(40.e-6), r(0.0),
                     jnp.where(xs3 < r(150.e-6), r(EHS_COLLSN) * (xs3 - r(40.e-6)) / (r(150.e-6) - r(40.e-6)), r(EHS_COLLSN)))
    G["ehsclsn"] = jnp.where(c, clsn, zero)
    ehs = ehscnv * jnp.minimum(r(1.0), jnp.maximum(r(0.0), xdn[LH] - r(300.)) / r(300.))
    G["ehs"] = jnp.where(c, jnp.minimum(ehs, C.ehsmax), zero)
    c = has[LH] & has[LI]
    G["ehiclsn"] = jnp.where(c, r(EHI_COLLSN), zero)
    ehi = C.eii0 * jnp.exp(C.eii1 * jnp.minimum(temcg, r(0.0)))
    G["ehi"] = jnp.where(c, jnp.minimum(C.ehimax, jnp.maximum(ehi, C.ehimin)), zero)
    G["ehis"] = zero  # lis = 0
    # hail (lhl > 1)
    c = has[LHL] & has[LC]
    ehlw = jnp.minimum(C.ehlw0, C.ewfac * poly)  # iehlw == 1
    ehlw = jnp.where(xdia[(LC, 1)] < r(2.4e-06), r(0.0), ehlw)
    ehlw = jnp.minimum(C.ehlw0, ehlw)
    G["ehlw"] = jnp.where(c, ehlw, zero)
    c = has[LHL] & has[LR]
    G["ehlr"] = jnp.where(c, jnp.minimum(C.ehlr0, r(1.0)), zero)
    c = has[LS] & has[LHL]
    G["ehlsclsn"] = jnp.where(c, r(EHLS_COLLSN), zero)
    G["ehls"] = jnp.where(c, jnp.minimum(ehscnv, C.ehsmax), zero)
    c = has[LHL] & has[LI]
    G["ehliclsn"] = jnp.where(c, r(EHLI_COLLSN), zero)
    ehli = C.eii0hl * jnp.exp(C.eii1hl * jnp.minimum(temcg, r(0.0)))
    ehli = jnp.minimum(C.ehimax, jnp.maximum(ehli, C.ehimin))
    ehli = jnp.where((temg > r(273.15)) | (qx[LC] < qmin[LC]), r(0.0), ehli)
    G["ehli"] = jnp.where(c, ehli, zero)
    G["ehlis"] = zero  # lis = 0
    G["efw"] = zero  # initialised in the loop, no frozen drops (lf = 0)
    return G
