"""NSSL 2-moment fall speeds: ``setvtz`` (module lines 6347-7721) and ``ziegfall1d`` (7731-8716).

Default mp=18 configuration only (ipconc=5, imurain=1, isnowfall=2, isnowdens=1, icefallopt=3,
ixtaltype=1, icdx=icdxhl=6, dmuh=dmuhl=1, rssflg=sssflg=hssflg=hlssflg=1, imaxdiaopt=3, all
``*fallfac`` = 1, mixedphase off).  Branches killed by that configuration are omitted (noted inline).

``setvtz`` -- vectorised point function
=======================================

    setvtz(qx, cx, rho0, rhovt, xdia, xmas, vtxbar, xdn, xv, cdxgs, fadvisc, temcg, alpha,
           axx, bxx, *, C, prec, infdo, ildo=0, qxmin=None, cdx=None, cno=None, cnostmp=None,
           cwmasn=None, cwmasx=None, cwradn=None) -> dict

Mirrors ``SUBROUTINE setvtz(ngscnt,qx,qxmin,qxw,cx,rho0,rhovt,xdia,cno,cnostmp,xmas,vtxbar,xdn,
xvmn0,xvmx0,xv,cdx,cdxgs,ipconc1,ndebug1,ngs,nz,igs,kgs,fadvisc,cwmasn,cwmasx,cwradn,cnina,cimna,
cimxa,itype1a,itype2a,temcg,infdo,alpha,axx,bxx,ildo)``.  The Fortran ``mgs`` axis is any array
shape ``S`` (all per-point arrays broadcast to ``S``).  Species-indexed arrays are dicts keyed by
the FORTRAN index (``indices.LC`` ... ``LHL``):

* ``qx``      {il: S}  mixing ratios (kg/kg) for il in lc..lhab (lv not used)
* ``cx``      {il: S}  number concentrations (#/m3)                    -- INOUT
* ``xdia``    {(il, m): S}, m=1..3 diameters                           -- INOUT
* ``xmas, xdn, xv, cdxgs`` {il: S}                                     -- INOUT
* ``vtxbar``  {(il, m): S}, m=1 mass-, 2 number-, 3 Z-weighted          -- INOUT
* ``alpha``   {il: S}  shape parameters (read for lr, lh, lhl)
* ``axx, bxx``{il: S}  for il in (lh, lhl)                              -- INOUT
* ``rho0, rhovt, fadvisc, temcg``: S

Entries Fortran does not assign keep their input values (e.g. ``vtxbar[(LR, 3)]`` when
``qx[LR] <= qxmin``), so pass the caller's current arrays.  Returns a dict with keys ``cx, xdia,
xmas, xdn, xv, vtxbar, cdxgs, axx, bxx`` (full dicts, untouched entries copied through).
``qxmin/cdx/cno`` default to the module arrays ``C.qxmin/C.cdx/C.cno``; ``cwmasn/cwmasx/cwradn``
to ``C.cwmasn/...``.  Dead for this configuration (accepted by the Fortran signature, unused
here): qxw, xvmn0, xvmx0 (module xvmn/xvmx are used), ipconc1, ndebug1, igs, kgs, cnina, cimna,
cimxa, itype1a, itype2a (all ipconc=0 / mixedphase / debug only).

``ziegfall1d``
==============

    ziegfall1d(an, t0, dn, rhovtzx, *, C, prec, infdo) -> (an_new, xvt)

Column-wise (ildo = 0 call used by sediment1d): ``an`` (NA+1, ..., nz) density-scaled stack,
``t0`` temperature (K), ``dn`` air density, ``rhovtzx`` = sqrt(rho00*min(1/0.05, 1/dn)).
Returns the stack after the "vaporize tiny values" repair and ``xvt`` = {(il, m): (..., nz)}
fall speeds (zero at points that are not gathered).
"""

from __future__ import annotations

import jax.numpy as jnp

from .indices import LC, LHAB, LHL, LH, LI, LR, LS, LV, LN, LVOL

F64 = jnp.float64

# setvtz local REAL constants (module lines 6460-6475)
_ARX = 10.0
_FRX = 516.575
_GR = 9.8
_RHO00_LOCAL = 1.225


def gmoi_interp(tmp, C, R):
    """``i = Int(dgami*tmp); del = tmp - dgam*i; x = gmoi(i) + (gmoi(i+1)-gmoi(i))*del*dgami``.

    tmp REAL; dgam/dgami/gmoi DOUBLE; del and the result are REAL locals in setvtz.
    """
    tmp = jnp.asarray(tmp, R)
    dgami = jnp.asarray(C.dgami, F64)
    dgam = jnp.asarray(C.dgam, F64)
    t64 = tmp.astype(F64)
    i = jnp.trunc(dgami * t64).astype(jnp.int32)
    dl = (t64 - dgam * i.astype(F64)).astype(R)
    g = jnp.asarray(C.gmoi, F64)
    gi = jnp.take(g, i, mode="clip")
    gi1 = jnp.take(g, i + 1, mode="clip")
    return (gi + (gi1 - gi) * dl.astype(F64) * dgami).astype(R)


def _mm_coeffs(xdn, C, R):
    """Milbrandt & Morrison (2013) density lookup (icdx = 6), returns (axx, bxx)."""
    ngdnmm = int(C.ngdnmm)
    mm = jnp.asarray(C.mmgraupvt, R)  # Fortran (1:ngdnmm, 1:3) padded
    indxr = jnp.trunc((xdn - 50.0) / 100.0).astype(jnp.int32) + 1
    indxr = jnp.minimum(ngdnmm, jnp.maximum(1, indxr))
    m1 = mm[indxr, 1]
    m2 = mm[indxr, 2]
    m3 = mm[indxr, 3]
    nxt = jnp.minimum(indxr + 1, ngdnmm)
    m2n = mm[nxt, 2]
    m3n = mm[nxt, 3]
    delrho = jnp.maximum(jnp.asarray(0.0, R), 0.01 * (xdn - m1))
    lt = indxr < ngdnmm
    axx = jnp.where(lt, m2 + delrho * (m2n - m2), m2)
    bxx = jnp.where(lt, m3 + delrho * (m3n - m3), m3)
    return axx, bxx


def setvtz(qx, cx, rho0, rhovt, xdia, xmas, vtxbar, xdn, xv, cdxgs, fadvisc, temcg, alpha,
           axx, bxx, *, C, prec, infdo, ildo=0, qxmin=None, cdx=None, cno=None, cnostmp=None,
           cwmasn=None, cwmasx=None, cwradn=None):
    """Port of SUBROUTINE setvtz (see module docstring for the argument contract)."""
    R = prec.R
    f = lambda v: jnp.asarray(v, R)  # noqa: E731
    qxmin = C.qxmin if qxmin is None else qxmin
    cwmasn = C.cwmasn if cwmasn is None else cwmasn
    cwmasx = C.cwmasx if cwmasx is None else cwmasx
    cwradn = C.cwradn if cwradn is None else cwradn
    del cdx, cno, cnostmp, temcg  # dead for ipconc=5 / sssflg>0 / icdx=6 (kept for signature)

    cx = dict(cx)
    xdia = dict(xdia)
    xmas = dict(xmas)
    vtxbar = dict(vtxbar)
    xdn = dict(xdn)
    xv = dict(xv)
    cdxgs = dict(cdxgs)
    axx = dict(axx)
    bxx = dict(bxx)
    rho0 = jnp.asarray(rho0, R)
    rhovt = jnp.asarray(rhovt, R)

    def do(il):
        return ildo == 0 or ildo == il

    pi = f(C.pi)
    pii = f(C.piinv)
    gr = f(_GR)
    arx = f(_ARX)
    frx = f(_FRX)
    zero = f(0.0)

    # cwch / cwchl: dmuh == dmuhl == 1.0 and ipconc <= 5 -> constants (lines 6492-6520)
    dnu_h = f(C.dnu[LH])
    cwchtmp = ((3. + dnu_h) * (2. + dnu_h) * (1.0 + dnu_h)) ** (-1. / 3.)
    dnu_hl = f(C.dnu[LHL])
    cwchltmp = ((3. + dnu_hl) * (2. + dnu_hl) * (1.0 + dnu_hl)) ** (-1. / 3.)

    cimasn = jnp.minimum(f(C.cimas0), f(6.88e-13))
    cimasx = f(1.0e-8)
    cwc1 = f(6.0) / (pi * 1000.)
    cwc0 = pii

    # ------------------------------------------------------------------ DROPLETS (ipconc >= 2)
    if do(LC):
        q = qx[LC]
        has = q > f(qxmin[LC])
        xdn_c = xdn[LC]
        cxc = cx[LC]
        big = cxc > f(C.cxmin)
        cx_alt = jnp.maximum(f(C.cxmin), rho0 * q / f(cwmasx))
        cxc_h = jnp.where(big, cxc, cx_alt)
        xm_h = jnp.minimum(jnp.maximum(q * rho0 / cxc_h, f(cwmasn)), f(cwmasx))
        xv_h = xm_h / xdn_c
        d1_h = (xm_h * cwc1) ** (1. / 3.)
        cwrad = 0.5 * d1_h
        fad = jnp.asarray(fadvisc, R)
        vt_h = jnp.where(fad > 0.0,
                         (2.0 * gr * xdn_c * (cwrad ** 2)) / (9.0 * jnp.where(fad > 0.0, fad, 1.0)),
                         zero)
        # qx <= qxmin branch
        xm_n = f(cwmasn)
        cx_n = jnp.where(q <= 0.0, zero, cxc)
        cx[LC] = jnp.where(has, cxc_h, cx_n)
        xmas[LC] = jnp.where(has, xm_h, xm_n)
        xv[LC] = jnp.where(has, xv_h, xm_n / xdn_c)
        d1 = jnp.where(has, d1_h, 2. * f(cwradn))
        xdia[(LC, 1)] = d1
        xdia[(LC, 2)] = jnp.where(has, d1_h ** 2, 4. * f(cwradn) ** 2)
        xdia[(LC, 3)] = d1
        vtxbar[(LC, 1)] = jnp.where(has, vt_h, zero)

    # ------------------------------------------------------------------ CLOUD ICE (ipconc >= 1)
    if do(LI):
        q = qx[LI]
        has = q > f(qxmin[LI])
        xdn_i = jnp.broadcast_to(f(900.0), jnp.shape(q))
        cxi = cx[LI]
        cxi_h = jnp.maximum(cxi, q * rho0 / cimasx)
        cxi_h = jnp.minimum(cxi_h, q * rho0 / cimasn)
        xm_h = jnp.maximum(q * rho0 / jnp.where(has, cxi_h, 1.0), cimasn)
        # ixtaltype == 1 (column); xmas > 0 always true here (max with cimasn)
        d1_h = 0.1871 * (xm_h ** 0.3429)
        xv_h = xm_h / xdn_i
        # icefallopt == 3 (adjusted Ferrier)
        tmp = (47.6273 * rhovt) / (((1.0 + f(C.cinu)) / xv_h) ** 0.18333 * f(C.gfcinu1))
        vt2_h = tmp * f(C.gfcinu1p18)
        vt1_h = tmp * f(C.gfcinu2p18) / (1. + f(C.cinu))
        cx[LI] = jnp.where(has, cxi_h, jnp.where(q <= 0.0, zero, cxi))
        xmas[LI] = jnp.where(has, xm_h, f(1.e-13))
        xdn[LI] = xdn_i
        xv[LI] = jnp.where(has, xv_h, xv[LI])
        xdia[(LI, 1)] = jnp.where(has, d1_h, f(1.e-7))
        xdia[(LI, 2)] = jnp.where(has, d1_h ** 2, f(1.e-14))
        xdia[(LI, 3)] = jnp.where(has, d1_h, f(1.e-7))
        vtxbar[(LI, 1)] = jnp.where(has, vt1_h, zero)
        vtxbar[(LI, 2)] = jnp.where(has, vt2_h, vtxbar[(LI, 2)])
        vtxbar[(LI, 3)] = jnp.where(has, vt1_h, vtxbar[(LI, 3)])

    # ------------------------------------------------------------------ RAIN (ipconc >= 3)
    if do(LR):
        q = qx[LR]
        has = q > f(qxmin[LR])
        xdn_r = xdn[LR]
        a = alpha[LR]
        xv_h = rho0 * q / (xdn_r * jnp.maximum(f(1.0e-11), cx[LR]))
        # imaxdiaopt == 3, imurain == 1
        xvbarmax = f(C.xvmx[LR]) / ((4. + a) ** 3 / ((3. + a) * (2. + a) * (1. + a)))
        gt = xv_h > xvbarmax
        lt = (~gt) & (xv_h < f(C.xvmn[LR]))
        cx_h = jnp.where(gt, rho0 * q / (xvbarmax * xdn_r),
                         jnp.where(lt, rho0 * q / (f(C.xvmn[LR]) * xdn_r), cx[LR]))
        xv_h = jnp.where(gt, xvbarmax, jnp.where(lt, f(C.xvmn[LR]), xv_h))
        xm_h = xv_h * xdn_r
        d3_h = (xm_h * cwc1) ** (1. / 3.)
        d1_h = (6. * pii * xv_h / ((a + 3.) * (a + 2.) * (a + 1.))) ** (1. / 3.)
        d1_n = f(1.e-9)
        xm_n = xdn_r * (pi / 6.) * d1_n ** 3
        cx[LR] = jnp.where(has, cx_h, cx[LR])
        xv[LR] = jnp.where(has, xv_h, xv[LR])
        xmas[LR] = jnp.where(has, xm_h, xm_n)
        d1 = jnp.where(has, d1_h, d1_n)
        xdia[(LR, 1)] = d1
        xdia[(LR, 3)] = jnp.where(has, d3_h, d1_n)
        xdia[(LR, 2)] = d1 ** 2

    # ------------------------------------------------------------------ SNOW (ipconc >= 4)
    if do(LS):
        q = qx[LS]
        has = q > f(qxmin[LS])
        xdn_s = xdn[LS]
        cxs = cx[LS]
        xm = rho0 * q / jnp.maximum(f(1.0e-9), cxs)
        # isnowdens == 1: leave xdn at its default value
        xvs = rho0 * q / (xdn_s * jnp.maximum(f(1.0e-9), cxs))
        d1 = (xvs * cwc0 * 6.0) ** (1. / 3.)
        c1 = xvs < f(C.xvmn[LS])
        xvs1 = jnp.maximum(f(C.xvmn[LS]), xvs)
        xm1 = xvs1 * xdn_s
        xvs = jnp.where(c1, xvs1, xvs)
        xm = jnp.where(c1, xm1, xm)
        cxs = jnp.where(c1, rho0 * q / xm1, cxs)
        d1 = jnp.where(c1, (xvs1 * cwc0 * 6.0) ** (1. / 3.), d1)
        c2 = xvs > f(C.xvmx[LS]) * jnp.maximum(f(1.), 100. / jnp.minimum(f(100.), xdn_s))
        xvs2 = jnp.minimum(f(C.xvmx[LS]), jnp.maximum(f(C.xvmn[LS]), xvs))
        xm2 = 0.106214 * xvs2 ** (2. / 3.)
        cx2 = rho0 * q / xm2
        xdn2 = 0.0346159 * jnp.sqrt(cx2 / jnp.where(has, q * rho0, 1.0))
        xvs = jnp.where(c2, xvs2, xvs)
        xm = jnp.where(c2, xm2, xm)
        cxs = jnp.where(c2, cx2, cxs)
        xdn_h = jnp.where(c2, xdn2, xdn_s)
        d1 = jnp.where(c2, jnp.sqrt(xm2 / 0.069), d1)
        d1 = jnp.where(has, d1, f(1.e-9))
        xdia[(LS, 1)] = d1
        xdia[(LS, 3)] = d1
        xdia[(LS, 2)] = d1 ** 2
        cx[LS] = jnp.where(has, cxs, zero)
        xv[LS] = jnp.where(has, xvs, xv[LS])
        xmas[LS] = jnp.where(has, xm, xmas[LS])
        xdn[LS] = jnp.where(has, xdn_h, xdn_s)

    # ------------------------------------------------------------------ GRAUPEL / HAIL (ipconc >= 5)
    for il, cwchx, form in ((LH, cwchtmp, "pii"), (LHL, cwchltmp, "div")):
        if not do(il):
            continue
        q = qx[il]
        has = q > f(qxmin[il])
        xdnx = xdn[il]
        xvx = rho0 * q / (xdnx * jnp.maximum(f(1.0e-9), cx[il]))
        xmx = xvx * xdnx
        oob = (xvx < f(C.xvmn[il])) | (xvx > f(C.xvmx[il]))
        xvc = jnp.minimum(f(C.xvmx[il]), jnp.maximum(f(C.xvmn[il]), xvx))
        xmc = xvc * xdnx
        xvx = jnp.where(oob, xvc, xvx)
        xmx = jnp.where(oob, xmc, xmx)
        cxx = jnp.where(oob, rho0 * q / xmc, cx[il])
        if form == "pii":
            d3 = (xvx * 6. * pii) ** (1. / 3.)
        else:
            d3 = (xvx * 6. / pi) ** (1. / 3.)
        d1 = cwchx * d3  # dmuh == 1
        d1 = jnp.where(has, d1, f(1.e-9))
        xdia[(il, 1)] = d1
        xdia[(il, 3)] = jnp.where(has, d3, f(1.e-9))
        xdia[(il, 2)] = d1 ** 2
        xv[il] = jnp.where(has, xvx, xv[il])
        xmas[il] = jnp.where(has, xmx, xmas[il])
        cx[il] = jnp.where(has, cxx, cx[il])

    # ------------------------------------------------------------------ RAIN fall speeds (imurain=1)
    if do(LR):
        q = qx[LR]
        has = q > f(qxmin[LR])
        alp = alpha[LR]
        base = 1.0 + frx * xdia[(LR, 1)]
        vt1 = rhovt * arx * (1.0 - base ** (-alp - 4.0))
        vt2 = rhovt * arx * (1.0 - base ** (-alp - 1.0)) if infdo >= 1 else vt1
        # lines 7069-7078: rssflg == 1; the ELSE branches copy the mass-weighted speed
        vt3 = rhovt * arx * (1.0 - base ** (-alp - 7.0)) if infdo >= 2 else vt1
        vtxbar[(LR, 1)] = jnp.where(has, vt1, zero)
        vtxbar[(LR, 2)] = jnp.where(has, vt2, zero)
        vtxbar[(LR, 3)] = jnp.where(has, vt3, vtxbar[(LR, 3)])

    # ------------------------------------------------------------------ SNOW fall speeds (isnowfall=2)
    if do(LS):
        q = qx[LS]
        has = q > f(qxmin[LS])
        p = xv[LS] ** 0.14
        vtxbar[(LS, 1)] = jnp.where(has, 11.9495 * rhovt * p, zero)
        vtxbar[(LS, 2)] = jnp.where(has, 7.02909 * rhovt * p, vtxbar[(LS, 2)])
        if infdo >= 2:
            vtxbar[(LS, 3)] = jnp.where(has, 13.3436 * rhovt * p, vtxbar[(LS, 3)])

    # ------------------------------------------------------------------ GRAUPEL / HAIL fall speeds (icdx=6)
    aaxd = {}
    bbxd = {}
    for il, dnmn in ((LH, C.hdnmn), (LHL, C.hldnmn)):
        if not do(il):
            continue
        q = qx[il]
        has = q > f(qxmin[il])
        a_mm, b_mm = _mm_coeffs(xdn[il], C, R)
        axx[il] = jnp.where(has, a_mm, axx[il])
        bxx[il] = jnp.where(has, b_mm, bxx[il])
        aax, bbx = a_mm, b_mm
        aaxd[il], bbxd[il] = aax, bbx
        cd = jnp.maximum(f(0.45), jnp.minimum(f(1.2), 0.45 + 0.55 * (800.0 - jnp.maximum(f(dnmn), jnp.minimum(f(800.0), xdn[il]))) / (f(800.) - 170.0)))
        cdxgs[il] = jnp.where(has, cd, cdxgs[il])
        al = alpha[il]
        x = gmoi_interp(4. + al + bbx, C, R)
        y = gmoi_interp(4. + al, C, R)
        vt1 = rhovt * aax * xdia[(il, 1)] ** bbx * x / y
        vtxbar[(il, 1)] = jnp.where(has, vt1, zero)

    if infdo >= 1:
        if do(LC):
            vtxbar[(LC, 2)] = vtxbar[(LC, 1)]
        # lg (= lh) > lr: number/Z-weighted speeds for graupel and hail
        for il in (LH, LHL):
            if not do(il):
                continue
            q = qx[il]
            has = q > f(qxmin[il])
            aax, bbx = aaxd[il], bbxd[il]  # == axx/bxx where has (icdx == 6)
            al = alpha[il]
            x = gmoi_interp(1. + al + bbx, C, R)
            y = gmoi_interp(1. + al, C, R)
            p = xdia[(il, 1)] ** bbx
            vt2 = rhovt * aax * p * x / y
            vtxbar[(il, 2)] = jnp.where(has, vt2, zero)
            if infdo >= 2:
                x3 = gmoi_interp(7. + al + bbx, C, R)
                y3 = gmoi_interp(7. + al, C, R)
                vt3 = rhovt * (aax * p * x3) / y3
                vtxbar[(il, 3)] = jnp.where(has, vt3, vtxbar[(il, 3)])

    return dict(cx=cx, xdia=xdia, xmas=xmas, xdn=xdn, xv=xv, vtxbar=vtxbar, cdxgs=cdxgs,
                axx=axx, bxx=bxx)


def ziegfall1d(an, t0, dn, rhovtzx, *, C, prec, infdo):
    """ziegfall1d with ildo = 0 (all species), vectorised over columns and levels.

    Returns (an_new, xvt) where xvt[(il, m)] has the shape of ``t0`` and is zero at points not
    gathered (no hydrometeor mass > 0).  ``an_new`` carries the "vaporize tiny values" repair
    (module lines 8051-8064) applied at gathered points.
    """
    R = prec.R
    f = lambda v: jnp.asarray(v, R)  # noqa: E731
    an = jnp.asarray(an, R)
    t0 = jnp.asarray(t0, R)
    rho0 = jnp.asarray(dn, R)
    rhovt = jnp.asarray(rhovtzx, R)
    species = (LC, LR, LI, LS, LH, LHL)

    flag = jnp.zeros(t0.shape, bool)
    for il in range(LC, LHAB + 1):
        flag = flag | (an[il] > 0.0)

    temg = t0
    temcg = temg - f(C.tfr)
    fadvisc = f(C.advisc0) * (416.16 / (temg + 120.0)) * (temg / 296.0) ** (1.5)

    qx = {il: jnp.maximum(an[il], f(0.0)) for il in range(LC, LHAB + 1)}
    cx = {il: jnp.maximum(an[LN[il]], f(0.0)) for il in species}

    # vaporize tiny values (lz < 1, ln > 1 for every hydrometeor), sequential over il
    for il in range(LC, LHAB + 1):
        vap = flag & ((cx[il] <= f(C.cxmin)) | (qx[il] < f(C.qxmin[il])))
        cx[il] = jnp.where(vap, f(0.0), cx[il])
        an = an.at[LV].set(jnp.where(vap, an[LV] + an[il], an[LV]))
        qx[il] = jnp.where(vap, f(0.0), qx[il])
        an = an.at[il].set(jnp.where(vap, qx[il], an[il]))
        an = an.at[LN[il]].set(jnp.where(vap, cx[il], an[LN[il]]))

    shape = t0.shape
    xdn = {il: jnp.broadcast_to(f(C.xdn0[il]), shape) for il in species}
    # ldovol: graupel/hail density from the predicted volume
    for il in (LH, LHL):
        vx = jnp.maximum(an[LVOL[il]], f(0.0))
        ok = (vx > rho0 * f(C.qxmin[il]) * 1.e-3) & (qx[il] > f(C.qxmin[il]))
        dens = jnp.minimum(f(C.xdnmx[il]), jnp.maximum(f(C.xdnmn[il]), rho0 * qx[il] / jnp.where(ok, vx, 1.0)))
        xdn[il] = jnp.where(ok, dens, xdn[il])

    zero = jnp.zeros(shape, R)
    alpha = {il: zero for il in species}
    alpha[LH] = jnp.broadcast_to(f(C.dnu[LH]), shape)
    alpha[LHL] = jnp.broadcast_to(f(C.dnu[LHL]), shape)
    alpha[LR] = jnp.broadcast_to(f(C.alphar), shape)  # imurain == 1

    out = setvtz(
        qx, cx, rho0, rhovt,
        xdia={(il, m): zero for il in species for m in (1, 2, 3)},
        xmas={il: zero for il in species}, vtxbar={(il, m): zero for il in species for m in (1, 2, 3)},
        xdn=xdn, xv={il: zero for il in species}, cdxgs={il: zero for il in species},
        fadvisc=fadvisc, temcg=temcg, alpha=alpha,
        axx={LH: zero, LHL: zero}, bxx={LH: zero, LHL: zero},
        C=C, prec=prec, infdo=infdo, ildo=0)
    vt = out["vtxbar"]

    vtmax = f(150.0)
    xvt = {}
    for il in species:
        v1, v2, v3 = vt[(il, 1)], vt[(il, 2)], vt[(il, 3)]
        fix = (v2 > v1) | ((v1 > v3) & (v3 > 0.0))
        v1n = jnp.maximum(v1, v2)
        v3n = jnp.maximum(v3, v1n)
        v1 = jnp.where(fix, v1n, v1)
        v3 = jnp.where(fix, v3n, v3)
        big = (v1 > vtmax) | (v2 > vtmax) | (v3 > vtmax)
        v1 = jnp.where(big, jnp.minimum(vtmax, v1), v1)
        v2 = jnp.where(big, jnp.minimum(vtmax, v2), v2)
        v3 = jnp.where(big, jnp.minimum(vtmax, v3), v3)
        if infdo < 2:
            v3 = zero
        xvt[(il, 1)] = jnp.where(flag, v1, zero)
        xvt[(il, 2)] = jnp.where(flag, v2, zero)
        xvt[(il, 3)] = jnp.where(flag, v3, zero)
    return an, xvt
