"""JAX Goddard GCE single-moment microphysics (WRF mp_physics=97, ``gsfcgcescheme``).

This is ``gsfcgcescheme`` = WRF v4 ``mp_physics=97``. (NOTE: WRF v4
``mp_physics=7`` is the SEPARATE ``nuwrf4icescheme`` / ``module_mp_gsfcgce_4ice_nuwrf.F``
4-ice scheme with a precipitating hail class + a large diagnostic state array --
NOT this scheme, and NOT ported here.)

Faithful fp64 port of WRF ``phys/module_mp_gsfcgce.F`` for the operational call
(ihail=0 graupel, ice2=0 three-ice, itaobraun=1, new_ice_sat=2):

  gsfcgce  ->  fall_flux (adaptive-substep sedimentation of qr/qs/qg/qi)
           ->  consat_s  (constant setup; ported in goddard_constants.py)
           ->  saticel_s (per-level bulk microphysics + ice/water sat. adj.)

WRF call order is preserved exactly:

  1. fall_flux: Courant-limited terminal-velocity sedimentation, in order
       rain, snow, graupel, cloud-ice, accumulating surface precip.
  2. saticel_s, per level k (each k independent; the only cross-level coupling
     is ``fv(k)=sqrt(rho(2)/rho(k))`` using the GLOBAL k=2 reference density):
       - convert MKS->CGS (rho*1e-3, p*10)
       - process block (~40 terms): psaut/psaci/psacw/praci/piacr/pracw/praut/
         psfw/psfi/pidep, graupel accretion (pgacs/dgacw/dgacr/dgaci/wgaci/
         pgwet/shed), qc/qi/qr/qs negative-handling clamps, pracs/psacr,
         pgaut/pgfr, melting (psmlt/pgmlt), homog freeze / ice melt / Bergeron
         (pihom/pimlt/pidw), ice initiation+deposition (pint/pidep, itaobraun=1),
         the new_ice_sat=2 condensation/deposition saturation adjustment,
         psdep/pssub/pgsub deposition-sublimation, and ern rain evaporation.

The scheme works on POTENTIAL TEMPERATURE (``pt`` = dpt) internally; tendencies
are added in theta units via ``pt += (L/cp/pi) * dq``. Units inside saticel are
CGS, but mixing ratios are dimensionless (kg/kg == g/g) so the q updates need no
unit conversion; only rho (g/cm^3) and p (dyne/cm^2) are converted.

Validation: per-column WRF savepoint parity against the real Fortran scheme
(proofs/v090/savepoints_goddard, run via proofs/v090/run_goddard_parity.py). The
port defaults to fp64; the binding oracle is default-REAL (fp32), so parity is to
a predeclared physical tolerance (and a transparency fp64 oracle proves residuals
are the reference's own single-precision roundoff), never bitwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import goddard_constants as G

# Pull the scalar consat constants into module locals (fp64 python floats).
_AL = G.al
_CP = G.cp
_RD1 = G.rd1
_RD2 = G.rd2
_T0 = G.t0
_T00 = G.t00
_ALV = G.alv
_ALF = G.alf
_ALS = G.als
_AVC = G.avc
_AFC = G.afc
_ASC = G.asc
_RW = G.rw
_CW = G.cw
_CI = G.ci
_C76 = G.c76
_C358 = G.c358
_C172 = G.c172
_C409 = G.c409
_C218 = G.c218
_C580 = G.c580
_C610 = G.c610
_C149 = G.c149
_C879 = G.c879
_C141 = G.c141
_AG = G.ag
_BG = G.bg
_AS = G.as_
_BS = G.bs
_AW = G.aw
_BW = G.bw
_ROQG = G.roqg
_ROQS = G.roqs
_ROQR = G.roqr
_TNG = G.tng
_TNS = G.tns
_TNW = G.tnw
_ZRC = G.zrc
_ZSC = G.zsc
_ZGC = G.zgc
_VRC = G.vrc
_VSC = G.vsc
_VGC = G.vgc

_RN1 = G.rn1
_BND1 = G.bnd1
_RN2 = G.rn2
_BND2 = G.bnd2
_RN3 = G.rn3
_RN4 = G.rn4
_RN5 = G.rn5
_RN6 = G.rn6
_RN7 = G.rn7
_RN8 = G.rn8
_RN9 = G.rn9
_RN10 = G.rn10
_RN101 = G.rn101
_RN10A = G.rn10a
_RN10B = G.rn10b
_RN10C = G.rn10c
_RN11 = G.rn11
_RN11A = G.rn11a
_RN12 = G.rn12
_RN14 = G.rn14
_RN15 = G.rn15
_RN15A = G.rn15a
_RN16 = G.rn16
_RN17 = G.rn17
_RN17A = G.rn17a
_RN17B = G.rn17b
_RN17C = G.rn17c
_RN18 = G.rn18
_RN18A = G.rn18a
_RN19 = G.rn19
_RN19A = G.rn19a
_RN19B = G.rn19b
_RN20 = G.rn20
_RN20A = G.rn20a
_RN20B = G.rn20b
_BND3 = G.bnd3
_RN21 = G.rn21
_RN22 = G.rn22
_RN23 = G.rn23
_RN23A = G.rn23a
_RN23B = G.rn23b
_RN25 = G.rn25
_RN30A = G.rn30a
_RN30B = G.rn30b
_RN30C = G.rn30c
_RN31 = G.rn31
_BETA = G.beta
_RN32 = G.rn32

_BGH = G.bgh
_BSH = G.bsh
_BWH = G.bwh
_BGQ = G.bgq
_BSQ = G.bsq
_BWQ = G.bwq

# Bergeron tables -> jnp (fp64). 0-based index = WRF it - 1.
_RN12A = jnp.asarray(G.rn12a)
_RN12B = jnp.asarray(G.rn12b)
_RN25A = jnp.asarray(G.rn25a)
_AA1 = jnp.asarray(G.AA1)
_AA2 = jnp.asarray(G.AA2)

# saticel-local re-set values
_BETAH = G.BETAH      # 0.5*beta
_CN0 = G.CN0          # 1e-6 (itaobraun=1)
_AMI50_SAT = G.AMI50_SAT
_AMI100_SAT = G.AMI100_SAT
_AMI40_SAT = G.AMI40_SAT

# r2is/r2ig for ice2=0 (3-ice graupel)
_R2IS = 1.0
_R2IG = 1.0

_CMIN = 1.e-19
_CMIN1 = 1.e-20
_CMIN2 = 1.e-12


# ==========================================================================
# fall_flux: adaptive-substep terminal-velocity sedimentation. Mirrors the
# WRF Courant while-loop EXACTLY (same threshold 1e-8, top cell k=kte excluded,
# fluxin top-down sweep, surface vs deposit-into-(min_q-1)).
# ==========================================================================
def _sediment_species(q, rho, dzw, zz, zsfc, dtb, vt_of_q):
    km = q.shape[0]
    kidx = jnp.arange(km)
    thresh = 1.0e-8
    zprev = jnp.concatenate([jnp.asarray([zsfc], q.dtype), zz[:-1]])
    dz_courant = zz - zprev  # (zz(k)-zz(k-1)) with zz(-1)=zsfc=topo

    def vtold_all(qz):
        vt = jnp.where(qz > thresh, vt_of_q(jnp.maximum(qz, thresh)), 0.0)
        # WRF computes vt only for k=kts..kte-1 (0-based: k<km-1)
        vt = jnp.where(kidx < km - 1, vt, 0.0)
        return vt

    def cond(state):
        _, _, _, notlast, _ = state
        return notlast

    def body(state):
        qz, ppt, t_del_tv, notlast, _ = state
        vt = vtold_all(qz)
        active = (qz > thresh) & (kidx < km - 1)
        any_active = jnp.any(active)
        big = km
        small = -1
        min_q = jnp.min(jnp.where(active, kidx, big))
        max_q = jnp.max(jnp.where(active, kidx, small))
        cour = jnp.where(active & (vt > 0.0),
                         0.9 * dz_courant / jnp.maximum(vt, 1e-30), dtb)
        del_tv0 = jnp.minimum(dtb, jnp.min(jnp.where(active, cour, dtb)))

        proceed = any_active & (max_q >= min_q)

        t_new = t_del_tv + del_tv0
        over = t_new >= dtb
        del_tv = jnp.where(over, dtb + del_tv0 - t_new, del_tv0)
        notlast_next = jnp.where(proceed, jnp.logical_not(over), False)

        def scan_step(fluxin, k_from_top):
            k = max_q - k_from_top
            in_span = (k >= min_q) & (k <= max_q)
            fluxout = rho[k] * vt[k] * qz[k]
            flux = (fluxin - fluxout) / rho[k] / dzw[k]
            dq = jnp.where(in_span & proceed, del_tv * flux, 0.0)
            fluxin_next = jnp.where(in_span, fluxout, fluxin)
            return fluxin_next, (k, dq)

        fluxin_final, (ks, dqs) = jax.lax.scan(
            scan_step, jnp.asarray(0.0, q.dtype), jnp.arange(km))
        dq_by_k = jnp.zeros(km, q.dtype).at[ks].add(dqs)
        qz_upd = qz + dq_by_k
        qz_upd = jnp.maximum(qz_upd, 0.0)  # WRF amax1(0.,qz)
        fluxin_at_min = fluxin_final
        at_surface = (min_q == 0)
        ppt_inc = jnp.where(proceed & at_surface, fluxin_at_min * del_tv, 0.0)
        idx_below = jnp.maximum(min_q - 1, 0)
        deposit = jnp.where(
            proceed & jnp.logical_not(at_surface),
            del_tv * fluxin_at_min / rho[idx_below] / dzw[idx_below], 0.0)
        qz_upd = qz_upd.at[idx_below].add(deposit)

        ppt_next = ppt + ppt_inc
        qz_out = jnp.where(proceed, qz_upd, qz)
        return (qz_out, ppt_next, jnp.where(proceed, t_new, t_del_tv),
                notlast_next, vt)

    init = (q, jnp.asarray(0.0, q.dtype), jnp.asarray(0.0, q.dtype),
            jnp.asarray(True), jnp.zeros(km, q.dtype))
    qz, ppt, _, _, _ = jax.lax.while_loop(cond, body, init)
    return qz, ppt


def _fall_flux(qr, qi, qs, qg, p, rho, z, dzw, zsfc, dt):
    """fall_flux for one column. rho/p/z/dzw in MKS. Returns updated
    (qr,qi,qs,qg) and surface precip (mm): pptrain/snow/graul/ice."""
    sqrhoz = jnp.sqrt(G.RHOE_S / rho)
    pi = jnp.pi

    def vt_rain(q):
        tmp1 = jnp.sqrt(pi * G.RHOWATER * G.XNOR / rho / q)
        tmp1 = jnp.sqrt(tmp1)
        return (G.CONSTA * G.GAMBP4 * sqrhoz / tmp1 ** G.CONSTB) / 6.0

    def vt_snow(q):
        tmp1 = jnp.sqrt(pi * G.RHOSNOW * G.XNOS / rho / q)
        tmp1 = jnp.sqrt(tmp1)
        return (G.CONSTC * G.GAMDP4 * sqrhoz / tmp1 ** G.CONSTD) / 6.0

    def vt_graupel(q):
        # ihail=0 -> graupel, RH(1984): vtg=term0*tmp1*(p0/prez)**0.4, term0=abar*gam4bbar/6
        tmp1 = jnp.sqrt(pi * G.RHOGRAUL * G.XNOG / rho / q)
        tmp1 = jnp.sqrt(tmp1)
        tmp1 = tmp1 ** G.BBAR
        tmp1 = 1.0 / tmp1
        term0 = G.ABAR * G.GAM4BBAR / 6.0
        return term0 * tmp1 * (G.P0_FALL / p) ** 0.4

    def vt_ice(q):
        return 3.29 * (rho * q) ** 0.16

    qr2, pptrain = _sediment_species(qr, rho, dzw, z, zsfc, dt, vt_rain)
    qs2, pptsnow = _sediment_species(qs, rho, dzw, z, zsfc, dt, vt_snow)
    qg2, pptgraul = _sediment_species(qg, rho, dzw, z, zsfc, dt, vt_graupel)
    qi2, pptice = _sediment_species(qi, rho, dzw, z, zsfc, dt, vt_ice)
    return qr2, qi2, qs2, qg2, pptrain, pptsnow, pptgraul, pptice


# ==========================================================================
# saticel_s, ONE level. All inputs/outputs are per-cell scalars. CGS internally.
#   pt,qv,qc,qr,qi,qs,qg : after fall_flux (MKS mixing ratios, theta for pt)
#   rho_mks,pi_mks,p0_mks: MKS
#   fv0                  : sqrt(rho(2)/rho(k))  (the GLOBAL k=2 coupling)
# Returns updated (pt,qv,qc,qr,qi,qs,qg).
# ==========================================================================
def _saticel_cell(pt, qv, qc, qr, qi, qs, qg, rho_mks, pi_mks, p0_mks, fv0, dt):
    d2t = dt
    pi0 = pi_mks
    p0 = p0_mks * 10.0          # dyne/cm^2
    rho = rho_mks * 1.0e-3      # g/cm^3

    # cmin-clamp inputs (cmin1)
    qc = jnp.where(qc <= _CMIN1, 0.0, qc)
    qr = jnp.where(qr <= _CMIN1, 0.0, qr)
    qi = jnp.where(qi <= _CMIN1, 0.0, qi)
    qs = jnp.where(qs <= _CMIN1, 0.0, qs)
    qg = jnp.where(qg <= _CMIN1, 0.0, qg)

    rp0 = 3.799052e3 / p0
    pir = 1.0 / pi0
    pr0 = 1.0 / p0
    r00 = rho
    rr0 = 1.0 / rho
    rrs = jnp.sqrt(rr0)
    rrq = jnp.sqrt(rrs)
    f00 = _AL / _CP / pi0
    fvs = jnp.sqrt(fv0)
    zrr = 1.e5 * _ZRC * rrq
    zsr = 1.e5 * _ZSC * rrq
    zgr = 1.e5 * _ZGC * rrq
    cp409 = _C409 * pi0
    cv409 = _C409 * _AVC
    cp580 = _C580 * pi0
    cs580 = _C580 * _ASC
    alvr = r00 * _ALV
    afcp = _AFC * pir
    avcp = _AVC * pir
    ascp = _ASC * pir
    vrcf = _VRC * fv0
    vscf = _VSC * fv0
    vgcf = _VGC * fv0
    dwvp = _C879 * pr0
    r3f = _RN3 * fv0
    r4f = _RN4 * fv0
    r5f = _RN5 * fv0
    r6f = _RN6 * fv0
    r7r = _RN7 * rr0
    r8r = _RN8 * rr0
    r9r = _RN9 * rr0
    r101f = _RN101 * fvs
    r10ar = _RN10A * r00
    r11rt = _RN11 * rr0 * d2t
    r12r = _RN12 * r00
    r14f = _RN14 * fv0
    r15f = _RN15 * fv0
    r15af = _RN15A * fv0
    r16r = _RN16 * rr0
    r17r = _RN17 * rr0
    r17as = _RN17A * fvs
    r18r = _RN18 * rr0
    r19rt = _RN19 * rr0 * d2t
    r19as = _RN19A * fvs
    r20bs = _RN20B * fvs
    r22f = _RN22 * fv0
    r23af = _RN23A * fvs
    r23br = _RN23B * r00
    r25rt = _RN25 * rr0 * d2t
    r31r = _RN31 * rr0
    r32rt = _RN32 * d2t * rrs

    bw3 = _BW + 3.
    bs3 = _BS + 3.
    bg3 = _BG + 3.
    bsh5 = 2.5 + _BSH
    bgh5 = 2.5 + _BGH
    bwh5 = 2.5 + _BWH
    bw6 = _BW + 6.
    r10t = _RN10 * d2t
    r11at = _RN11A * d2t
    r19bt = _RN19B * d2t
    r20t = -_RN20 * d2t
    r23t = -_RN23 * d2t
    r25a = _RN25

    rt0 = 1.0 / (_T0 - _T00)

    tair = pt * pi0
    tairc = tair - _T0

    zr = zrr
    zs = zsr
    zg = zgr
    vr = jnp.asarray(0.0)
    vs = jnp.asarray(0.0)
    vg = jnp.asarray(0.0)

    # ---- COMPUTE ZR,ZS,ZG,VR,VS,VG ----
    dd = r00 * qr
    y1 = dd ** 0.25
    zr = jnp.where(qr > _CMIN1, _ZRC / y1, zr)
    vr = jnp.where(qr > _CMIN1, jnp.maximum(vrcf * dd ** _BWQ, 0.0), vr)

    dd = r00 * qs
    y1 = dd ** 0.25
    zs = jnp.where(qs > _CMIN1, _ZSC / y1, zs)
    vs = jnp.where(qs > _CMIN1, jnp.maximum(vscf * dd ** _BSQ, 0.0), vs)

    dd = r00 * qg
    y1 = dd ** 0.25
    zg = jnp.where(qg > _CMIN1, _ZGC / y1, zg)
    vg = jnp.where(qg > _CMIN1, jnp.maximum(vgcf * dd ** _BGQ, 0.0), vg)  # ihail=0

    vr = jnp.where(qr <= _CMIN2, 0.0, vr)
    vs = jnp.where(qs <= _CMIN2, 0.0, vs)
    vg = jnp.where(qg <= _CMIN2, 0.0, vg)

    # ---- viscosity / diffusivity / conductivity (per-cell) ----
    y1v = _C149 * tair ** 1.5 / (tair + 120.)
    dwv = dwvp * tair ** 1.81
    tca = _C141 * y1v
    scv = 1.0 / ((rr0 * y1v) ** .1666667 * dwv ** .3333333)

    cold = tair < _T0

    # ---- PSAUT/PSACI/PSACW/PRACI/PIACR (snow) ; QSACW (T>=T0) ----
    dds = 1.0 / zs ** bs3
    esi = jnp.exp(.025 * tairc)
    psaut = jnp.where(cold, _R2IS * jnp.maximum(_RN1 * esi * (qi - _BND1), 0.0), 0.0)
    psaci = jnp.where(cold, _R2IS * r3f * esi * qi * dds, 0.0)
    psacw = jnp.where(cold, _R2IS * 0.5 * r4f * qc * dds, 0.0)
    praci = jnp.where(cold, _R2IS * r5f * qi / zr ** bw3, 0.0)
    piacr = jnp.where(cold, _R2IS * r6f * qi * (zr ** (-bw6)), 0.0)
    qsacw = jnp.where(cold, 0.0, _R2IS * r4f * qc * dds)

    # ---- PRAUT/PRACW ----
    pracw = r22f * qc / zr ** bw3
    y1 = qc - _BND3
    praut = jnp.where(y1 > 0.0, r00 * y1 * y1 / (1.2e-4 + _RN21 / y1), 0.0)

    # ---- PSFW/PSFI/PIDEP (Bergeron) ----
    psfw = jnp.zeros_like(pt)
    psfi = jnp.zeros_like(pt)
    pidep = jnp.zeros_like(pt)
    bergeron = cold & (qi > _CMIN)
    y1b = jnp.maximum(jnp.minimum(tairc, -1.), -31.)
    it_idx = jnp.minimum(jnp.maximum((jnp.abs(y1b)).astype(jnp.int32), 1), 31) - 1
    y1t = _RN12A[it_idx]
    y2t = _RN12B[it_idx]
    psfw_b = _R2IS * jnp.maximum(d2t * y1t * (y2t + r12r * qc) * qi, 0.0)
    rtair_b = 1.0 / (tair - _C76)
    y2e = jnp.exp(_C218 - _C580 * rtair_b)
    qsi_b = rp0 * y2e
    esi_b = _C610 * y2e
    ssi_b = qv / qsi_b - 1.
    dm_b = jnp.maximum(qv - qsi_b, 0.)
    rsub1_b = cs580 * qsi_b * rtair_b * rtair_b
    y3_b = 1.0 / tair
    dd_b = y3_b * (_RN30A * y3_b - _RN30B) + _RN30C * tair / esi_b
    y1d = 206.18 * ssi_b / dd_b
    # WRF: r_nci=min(1e-6*exp(-.46*tairc),1) ; pidep=y1*sqrt(r_nci*qi/r00)
    r_nci_b = jnp.minimum(1.e-6 * jnp.exp(-.46 * tairc), 1.)
    pidep_b = y1d * jnp.sqrt(jnp.maximum(r_nci_b * qi / r00, 0.0))
    dep_b = dm_b / (1. + rsub1_b) / d2t
    big_dm = dm_b > _CMIN2
    a2 = jnp.where(big_dm & (pidep_b > dep_b) & (pidep_b > _CMIN2), dep_b / pidep_b, 1.0)
    pidep_bf = jnp.where(big_dm & (pidep_b > dep_b) & (pidep_b > _CMIN2), dep_b, pidep_b)
    psfi_b = jnp.where(big_dm, _R2IS * a2 * .5 * qi * y1d / (jnp.sqrt(_AMI100_SAT) - jnp.sqrt(_AMI40_SAT)), 0.0)
    pidep_bf = jnp.where(big_dm, pidep_bf, 0.0)
    psfw = jnp.where(bergeron, psfw_b, psfw)
    psfi = jnp.where(bergeron, psfi_b, psfi)
    pidep = jnp.where(bergeron, pidep_bf, pidep)

    # ---- PGACS/DGACS/WGACS, DGACW, DGACR (graupel accretion) ----
    ee1 = jnp.where(qc + qr < 1.e-4, .01, 1.)
    ee2 = 0.09
    egs = ee1 * jnp.exp(ee2 * tairc)
    egs = jnp.where(tair >= _T0, 1.0, egs)
    y1 = jnp.abs(vg - vs)
    y2 = zs * zg
    y3 = 5. / y2
    y4 = .08 * y3 * y3
    y5 = .05 * y3 * y4
    dd = y1 * (y3 / zs ** 5 + y4 / zs ** 3 + y5 / zs)
    pgacs = _R2IG * _R2IS * r9r * egs * dd
    dgacs = jnp.zeros_like(pt)  # ihail=0
    wgacs = _R2IG * _R2IS * r9r * dd
    y1 = 1.0 / zg ** bg3
    dgacw = _R2IG * jnp.maximum(r14f * qc * y1, 0.0)  # ihail=0
    qgacw = dgacw
    y1 = jnp.abs(vg - vr)
    y2 = zr * zg
    y3 = 5. / y2
    y4 = .08 * y3 * y3
    y5 = .05 * y3 * y4
    dd = r16r * y1 * (y3 / zr ** 5 + y4 / zr ** 3 + y5 / zr)
    dgacr = _R2IG * jnp.maximum(dd, 0.0)
    qgacr = dgacr

    # WRF lines 2060-2069: WARM (tair>=t0) zeroes dgacs/wgacs/dgacw/dgacr;
    # COLD (else) zeroes pgacs/qgacw/qgacr.
    warm_g = tair >= _T0
    dgacs = jnp.where(warm_g, 0.0, dgacs)
    wgacs = jnp.where(warm_g, 0.0, wgacs)
    dgacw = jnp.where(warm_g, 0.0, dgacw)
    dgacr = jnp.where(warm_g, 0.0, dgacr)
    pgacs = jnp.where(warm_g, pgacs, 0.0)
    qgacw = jnp.where(warm_g, qgacw, 0.0)
    qgacr = jnp.where(warm_g, qgacr, 0.0)

    # ---- DGACI/WGACI/PGWET ----
    dgaci = jnp.zeros_like(pt)  # ihail=0 -> dgaci=0
    y1g = qi / zg ** bg3
    wgaci = jnp.where(cold, _R2IG * r15af * y1g, 0.0)
    pgwet = jnp.zeros_like(pt)
    pgwet_cond = cold & (tairc >= -50.)
    y1w = 1.0 / (_ALF + _RN17C * tairc)
    y3w = .78 / zg ** 2 + r17as * scv / zg ** bgh5
    y4w = alvr * dwv * (rp0 - qv) - tca * tairc
    ddw = y1w * (r17r * y4w * y3w + (wgaci + wgacs) * (_ALF + _RN17B * tairc))
    pgwet = jnp.where(pgwet_cond, _R2IG * jnp.maximum(ddw, 0.0), pgwet)

    # ---- negative cloud water (qc) handling ----
    y1 = qc / d2t
    psacw = jnp.minimum(y1, psacw)
    praut = jnp.minimum(y1, praut)
    pracw = jnp.minimum(y1, pracw)
    psfw = jnp.minimum(y1, psfw)
    dgacw = jnp.minimum(y1, dgacw)
    qsacw = jnp.minimum(y1, qsacw)
    qgacw = jnp.minimum(y1, qgacw)
    y1 = (psacw + praut + pracw + psfw + dgacw + qsacw + qgacw) * d2t
    qc = qc - y1
    neg_qc = qc < 0.0
    a1 = jnp.where(neg_qc & (y1 != 0.0), qc / y1 + 1., 1.0)
    a1 = jnp.where(neg_qc, a1, 1.0)
    psacw = psacw * a1
    praut = praut * a1
    pracw = pracw * a1
    psfw = psfw * a1
    dgacw = dgacw * a1
    qsacw = qsacw * a1
    qgacw = qgacw * a1
    qc = jnp.where(neg_qc, 0.0, qc)

    # ---- shed process ----
    wgacr = pgwet - dgacw - wgaci - wgacs
    y2 = dgacw + dgaci + dgacr + dgacs
    pgwet_ge = pgwet >= y2
    wgacr = jnp.where(pgwet_ge, 0.0, wgacr)
    wgaci = jnp.where(pgwet_ge, 0.0, wgaci)
    wgacs = jnp.where(pgwet_ge, 0.0, wgacs)
    dgacr = jnp.where(pgwet_ge, dgacr, 0.0)
    dgaci = jnp.where(pgwet_ge, dgaci, 0.0)
    dgacs = jnp.where(pgwet_ge, dgacs, 0.0)

    # ---- negative cloud ice (qi) handling ----
    y1 = qi / d2t
    psaut = jnp.minimum(y1, psaut)
    psaci = jnp.minimum(y1, psaci)
    praci = jnp.minimum(y1, praci)
    psfi = jnp.minimum(y1, psfi)
    dgaci = jnp.minimum(y1, dgaci)
    wgaci = jnp.minimum(y1, wgaci)
    y2 = (psaut + psaci + praci + psfi + dgaci + wgaci) * d2t
    qi = qi - y2 + pidep * d2t
    neg_qi = qi < 0.0
    a2 = jnp.where(neg_qi & (y2 != 0.0), qi / y2 + 1., 1.0)
    a2 = jnp.where(neg_qi, a2, 1.0)
    psaut = psaut * a2
    psaci = psaci * a2
    praci = praci * a2
    psfi = psfi * a2
    dgaci = dgaci * a2
    wgaci = wgaci * a2
    qi = jnp.where(neg_qi, 0.0, qi)

    # ---- dlt2/dlt3 ----
    dlt3 = jnp.zeros_like(pt)
    dlt2 = jnp.zeros_like(pt)
    cold_dlt = tair < _T0
    qr_small = qr < 1.e-4
    dlt3 = jnp.where(cold_dlt & qr_small, 1.0, dlt3)
    dlt2 = jnp.where(cold_dlt & qr_small, 1.0, dlt2)
    dlt2 = jnp.where(cold_dlt & (qs >= 1.e-4), 0.0, dlt2)
    # ice2=0 so no override of dlt to 1

    pr = (qsacw + praut + pracw + qgacw) * d2t
    ps = (psaut + psaci + psacw + psfw + psfi + dlt3 * praci) * d2t
    pg = ((1. - dlt3) * praci + dgaci + wgaci + dgacw) * d2t

    # ---- PRACS/PSACR ----
    y1 = jnp.abs(vr - vs)
    y2 = zr * zs
    y3 = 5. / y2
    y4 = .08 * y3 * y3
    y5 = .05 * y3 * y4
    pracs = _R2IG * _R2IS * r7r * y1 * (y3 / zs ** 5 + y4 / zs ** 3 + y5 / zs)
    psacr = _R2IS * r8r * y1 * (y3 / zr ** 5 + y4 / zr ** 3 + y5 / zr)
    qsacr = psacr
    warm2 = tair >= _T0
    pracs = jnp.where(warm2, 0.0, pracs)
    psacr = jnp.where(warm2, 0.0, psacr)
    qsacr = jnp.where(warm2, 0.0, qsacr)

    # ---- PGAUT/PGFR ----
    pgaut = jnp.zeros_like(pt)
    pgfr = jnp.zeros_like(pt)
    y2f = jnp.exp(_RN18A * (_T0 - tair))
    temp = 1.0 / zr
    temp = temp * temp * temp * temp * temp * temp * temp
    pgfr = jnp.where(cold_dlt, _R2IG * jnp.maximum(r18r * (y2f - 1.) * temp, 0.0), pgfr)

    # ---- negative rain (qr) / snow (qs) handling ----
    y1 = qr / d2t
    y2 = -qg / d2t
    piacr = jnp.minimum(y1, piacr)
    dgacr = jnp.minimum(y1, dgacr)
    wgacr = jnp.minimum(y1, wgacr)
    wgacr = jnp.maximum(y2, wgacr)
    psacr = jnp.minimum(y1, psacr)
    pgfr = jnp.minimum(y1, pgfr)
    delv = jnp.where(wgacr < 0., 1.0, 0.0)
    y1 = (piacr + dgacr + (1. - delv) * wgacr + psacr + pgfr) * d2t
    qr = qr + pr - y1 - delv * wgacr * d2t
    neg_qr = qr < 0.0
    a1 = jnp.where(neg_qr & (y1 != 0.), qr / y1 + 1., 1.0)
    a1 = jnp.where(neg_qr, a1, 1.0)
    piacr = piacr * a1
    dgacr = dgacr * a1
    wgacr = jnp.where(neg_qr & (wgacr > 0.), wgacr * a1, wgacr)
    pgfr = pgfr * a1
    psacr = psacr * a1
    qr = jnp.where(neg_qr, 0.0, qr)

    prn = d2t * ((1. - dlt3) * piacr + dgacr + wgacr + (1. - dlt2) * psacr + pgfr)
    ps = ps + d2t * (dlt3 * piacr + dlt2 * psacr)
    pracs = (1. - dlt2) * pracs
    y1 = qs / d2t
    pgacs = jnp.minimum(y1, pgacs)
    dgacs = jnp.minimum(y1, dgacs)
    wgacs = jnp.minimum(y1, wgacs)
    pgaut = jnp.minimum(y1, pgaut)
    pracs = jnp.minimum(y1, pracs)
    psn = d2t * (pgacs + dgacs + wgacs + pgaut + pracs)
    qs = qs + ps - psn
    neg_qs = qs < 0.0
    a2 = jnp.where(neg_qs & (psn != 0.0), qs / psn + 1., 1.0)
    a2 = jnp.where(neg_qs, a2, 1.0)
    pgacs = pgacs * a2
    dgacs = dgacs * a2
    wgacs = wgacs * a2
    pgaut = pgaut * a2
    pracs = pracs * a2
    psn = psn * a2
    qs = jnp.where(neg_qs, 0.0, qs)

    y2 = d2t * (psacw + psfw + dgacw + piacr + dgacr + wgacr + psacr + pgfr)
    pt = pt + afcp * y2
    qg = qg + pg + prn + psn

    # ---- PSMLT/PGMLT (melting) ----
    psmlt = jnp.zeros_like(pt)
    pgmlt = jnp.zeros_like(pt)
    tair = pt * pi0
    melt = tair >= _T0
    tairc_m = tair - _T0
    y1m = tca * tairc_m - alvr * dwv * (rp0 - qv)
    y2m = .78 / zs ** 2 + r101f * scv / zs ** bsh5
    ddm = r11rt * y1m * y2m + r11at * tairc_m * (qsacw + qsacr)
    psmlt_m = _R2IS * jnp.maximum(0.0, jnp.minimum(ddm, qs))
    y3m = .78 / zg ** 2 + r19as * scv / zg ** bgh5
    dd1m = r19rt * y1m * y3m + r19bt * tairc_m * (qgacw + qgacr)
    pgmlt_m = _R2IG * jnp.maximum(0.0, jnp.minimum(dd1m, qg))
    psmlt = jnp.where(melt, psmlt_m, psmlt)
    pgmlt = jnp.where(melt, pgmlt_m, pgmlt)
    pt = jnp.where(melt, pt - afcp * (psmlt + pgmlt), pt)
    qr = jnp.where(melt, qr + psmlt + pgmlt, qr)
    qs = jnp.where(melt, qs - psmlt, qs)
    qg = jnp.where(melt, qg - pgmlt, qg)

    # ---- PIHOM/PIDW/PIMLT (homog freeze / Bergeron dep / ice melt) ----
    qc = jnp.where(qc <= _CMIN1, 0.0, qc)
    qi = jnp.where(qi <= _CMIN1, 0.0, qi)
    tair = pt * pi0
    pihom = jnp.where(tair <= _T00, qc, 0.0)
    pimlt = jnp.where(tair >= _T0, qi, 0.0)
    pidw = jnp.zeros_like(pt)
    dw_cond = (tair < _T0) & (tair > _T00)
    tairc_d = tair - _T0
    y1d2 = jnp.maximum(jnp.minimum(tairc_d, -1.), -31.)
    it_d = jnp.minimum(jnp.maximum((jnp.abs(y1d2)).astype(jnp.int32), 1), 31) - 1
    y2d = _AA1[it_d]
    y3d = _AA2[it_d]
    y4d = jnp.exp(jnp.abs(_BETA * tairc_d))
    y5d = (r00 * qi / (r25a * y4d)) ** y3d
    pidw = jnp.where(dw_cond, jnp.minimum(r25rt * y2d * y4d * y5d, qc), pidw)

    y1 = pihom - pimlt + pidw
    pt = pt + afcp * y1 + ascp * pidep * d2t
    qv = qv - pidep * d2t
    qc = qc - y1
    qi = qi + y1

    # ---- PINT/PIDEP (ice initiation+deposition, itaobraun=1) ----
    pint = jnp.zeros_like(pt)
    tair = pt * pi0
    init_cold = tair < _T0
    qi = jnp.where(init_cold & (qi <= _CMIN2), 0.0, qi)
    tairc_i = tair - _T0
    rtair_i = 1.0 / (tair - _C76)
    y2i = jnp.exp(_C218 - _C580 * rtair_i)
    qsi_i = rp0 * y2i
    esi_i = _C610 * y2i
    ssi_i = qv / qsi_i - 1.
    y1i = 1.0 / tair
    tairccri = jnp.where(tairc_i <= -30., -30., tairc_i)
    y2i2 = jnp.exp(_BETAH * tairccri)
    y3i = jnp.sqrt(jnp.maximum(qi, 0.0))
    ddi = y1i * (_RN10A * y1i - _RN10B) + _RN10C * tair / esi_i
    pidep_i = jnp.maximum(r32rt * ssi_i * y2i2 * y3i / ddi, 0.0)
    r_nci_i = jnp.minimum(_CN0 * jnp.exp(_BETA * tairc_i), 1.)
    ddi2 = jnp.maximum(1.e-9 * r_nci_i / r00 - qi * 1.e-9 / _AMI50_SAT, 0.)
    dm_i = jnp.maximum(qv - qsi_i, 0.0)
    rsub1_i = cs580 * qsi_i * rtair_i * rtair_i
    dep_i = dm_i / (1. + rsub1_i)
    pint_i = jnp.maximum(jnp.minimum(ddi2, dm_i), 0.)
    pint_i = jnp.minimum(pint_i + pidep_i, dep_i)
    pint_i = jnp.where(pint_i <= _CMIN2, 0.0, pint_i)
    pt = jnp.where(init_cold, pt + ascp * pint_i, pt)
    qv = jnp.where(init_cold, qv - pint_i, qv)
    qi = jnp.where(init_cold, qi + pint_i, qi)
    pint = jnp.where(init_cold, pint_i, pint)

    # ---- new_ice_sat=2 saturation adjustment ----
    dep = jnp.zeros_like(pt)
    cnd = jnp.zeros_like(pt)
    tair = pt * pi0
    sat_liq = tair >= 253.16
    y1s = 1.0 / (tair - _C358)
    qsw_s = rp0 * jnp.exp(_C172 - _C409 * y1s)
    dds = cp409 * y1s * y1s
    dm_s = qv - qsw_s
    cnd_s = dm_s / (1. + avcp * dds * qsw_s)
    cnd_s = jnp.maximum(-qc, cnd_s)
    pt = jnp.where(sat_liq, pt + avcp * cnd_s, pt)
    qv = jnp.where(sat_liq, qv - cnd_s, qv)
    qc = jnp.where(sat_liq, qc + cnd_s, qc)
    cnd = jnp.where(sat_liq, cnd_s, cnd)

    sat_ice = tair <= 258.16
    y2s = 1.0 / (tair - _C76)
    qsi_s = rp0 * jnp.exp(_C218 - _C580 * y2s)
    dd1s = cp580 * y2s * y2s
    dep_s = (qv - qsi_s) / (1. + ascp * dd1s * qsi_s)
    dep_s = jnp.maximum(-qi, dep_s)
    pt = jnp.where(sat_ice, pt + ascp * dep_s, pt)
    qv = jnp.where(sat_ice, qv - dep_s, qv)
    qi = jnp.where(sat_ice, qi + dep_s, qi)
    dep = jnp.where(sat_ice, dep_s, dep)

    # ---- PSDEP/PSSUB/PGSUB (deposition/sublimation of qs, qg) ----
    tair = pt * pi0
    dsub_cold = tair < _T0
    qs = jnp.where(dsub_cold & (qs < _CMIN1), 0.0, qs)
    qg = jnp.where(dsub_cold & (qg < _CMIN1), 0.0, qg)
    rtair_d = 1.0 / (tair - _C76)
    qsi_d = rp0 * jnp.exp(_C218 - _C580 * rtair_d)
    ssi_d = qv / qsi_d - 1.
    y1ds = r10ar / (tca * tair ** 2) + 1. / (dwv * qsi_d)
    y2ds = .78 / zs ** 2 + r101f * scv / zs ** bsh5
    psdep_d = r10t * ssi_d * y2ds / y1ds
    pssub_d = psdep_d
    psdep_d = _R2IS * jnp.maximum(psdep_d, 0.)
    pssub_d = _R2IS * jnp.maximum(-qs, jnp.minimum(pssub_d, 0.))
    y2g = .78 / zg ** 2 + r20bs * scv / zg ** bgh5
    pgsub_d = _R2IG * r20t * ssi_d * y2g / y1ds
    dm_d = qv - qsi_d
    rsub1_d = cs580 * qsi_d * rtair_d * rtair_d
    y1ds2 = dm_d / (1. + rsub1_d)
    psdep_d = _R2IS * jnp.minimum(psdep_d, jnp.maximum(y1ds2, 0.))
    y2ds2 = jnp.minimum(y1ds2, 0.)
    pssub_d = _R2IS * jnp.maximum(pssub_d, y2ds2)
    ddg = jnp.maximum(-y2ds2 - qs, 0.)
    pgsub_d = _R2IG * jnp.minimum(jnp.minimum(ddg, qg), jnp.maximum(pgsub_d, 0.))
    dlt1 = jnp.where(qc + qi > 1.e-5, 1.0, 0.0)
    psdep_d = dlt1 * psdep_d
    pssub_d = (1. - dlt1) * pssub_d
    pgsub_d = (1. - dlt1) * pgsub_d
    pt = jnp.where(dsub_cold, pt + ascp * (psdep_d + pssub_d - pgsub_d), pt)
    qv = jnp.where(dsub_cold, qv + pgsub_d - pssub_d - psdep_d, qv)
    qs = jnp.where(dsub_cold, qs + psdep_d + pssub_d, qs)
    qg = jnp.where(dsub_cold, qg - pgsub_d, qg)

    # ---- ERN: evaporation of qr ----
    tair = pt * pi0
    rtair_e = 1.0 / (tair - _C358)
    qsw_e = rp0 * jnp.exp(_C172 - _C409 * rtair_e)
    ssw_e = qv / qsw_e - 1.0
    dm_e = qv - qsw_e
    rsub1_e = cv409 * qsw_e * rtair_e * rtair_e
    dd1e = jnp.maximum(-dm_e / (1. + rsub1_e), 0.0)
    y1e = .78 / zr ** 2 + r23af * scv / zr ** bwh5
    y2e2 = r23br / (tca * tair ** 2) + 1. / (dwv * qsw_e)
    ern_e = r23t * ssw_e * y1e / y2e2
    ern_e = jnp.minimum(jnp.minimum(dd1e, qr), jnp.maximum(ern_e, 0.))
    has_qr = qr > 0.0
    pt = jnp.where(has_qr, pt - avcp * ern_e, pt)
    qv = jnp.where(has_qr, qv + ern_e, qv)
    qr = jnp.where(has_qr, qr - ern_e, qr)

    # ---- final cmin1 clamp ----
    qc = jnp.where(qc <= _CMIN1, 0.0, qc)
    qr = jnp.where(qr <= _CMIN1, 0.0, qr)
    qi = jnp.where(qi <= _CMIN1, 0.0, qi)
    qs = jnp.where(qs <= _CMIN1, 0.0, qs)
    qg = jnp.where(qg <= _CMIN1, 0.0, qg)

    return pt, qv, qc, qr, qi, qs, qg


_saticel_col = jax.vmap(
    _saticel_cell,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))


# ==========================================================================
# Column driver: gsfcgce for one column.
# ==========================================================================
def _goddard_column(th, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz, zsfc, dt):
    # fall_flux (MKS) -- updates qr/qi/qs/qg, returns surface precip
    qr2, qi2, qs2, qg2, pptrain, pptsnow, pptgraul, pptice = _fall_flux(
        qr, qi, qs, qg, p, rho, z, dz, zsfc, dt)

    # fv(k) = sqrt(rho(2)/rho(k))  -- GLOBAL k=2 reference (0-based index 1)
    fv = jnp.sqrt(rho[1] / rho)

    pt, qv2, qc2, qr3, qi3, qs3, qg3 = _saticel_col(
        th, qv, qc, qr2, qi2, qs2, qg2, rho, pii, p, fv, dt)

    # surface precip in mm: WRF pptX are in m of water-equivalent... fluxin has
    # units (g/cm^3)*(cm/s)*(g/g) = g/cm^2/s; *del_tv -> g/cm^2 == mm? Actually
    # fall_flux runs in MKS (rho kg/m^3, vt m/s, q kg/kg, dz m) so flux is
    # kg/m^2/s; fluxin*del_tv is kg/m^2 == mm. RAINNCV = sum of all four (mm).
    rainncv = pptrain + pptsnow + pptgraul + pptice
    snowncv = pptsnow
    graupelncv = pptgraul
    sr = jnp.where(rainncv > 0., (pptsnow + pptgraul + pptice) / rainncv, 0.0)

    return {
        "th": pt, "qv": qv2, "qc": qc2, "qr": qr3,
        "qi": qi3, "qs": qs3, "qg": qg3,
        "rainncv": rainncv, "snowncv": snowncv,
        "graupelncv": graupelncv, "sr": sr,
    }


# batch over (ncol) of columns; arrays shaped (ncol, nlev)
_goddard_batch = jax.vmap(
    _goddard_column,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))


def goddard_run(theta, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz, delt):
    """Run Goddard GCE over a batch of columns (ncol, nlev). Returns dict of
    (ncol, nlev) species arrays + (ncol,) surface precip accumulators (mm)."""
    th = jnp.asarray(theta)
    zsfc = jnp.zeros(th.shape[0], dtype=th.dtype)  # topo=0 in the savepoint cases
    out = _goddard_batch(
        th, jnp.asarray(qv), jnp.asarray(qc), jnp.asarray(qr),
        jnp.asarray(qi), jnp.asarray(qs), jnp.asarray(qg),
        jnp.asarray(rho), jnp.asarray(pii), jnp.asarray(p),
        jnp.asarray(z), jnp.asarray(dz), zsfc, float(delt))
    return out


def goddard_physics_tendency(theta, qv, qc, qr, qi, qs, qg, pii, rho, p, z, dz, delt):
    """Goddard GCE adapter returning a frozen PhysicsTendency (in-place repl.)."""
    out = goddard_run(theta, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz, delt)
    return PhysicsTendency(
        state_replacements={
            "theta": out["th"], "qv": out["qv"], "qc": out["qc"],
            "qr": out["qr"], "qi": out["qi"], "qs": out["qs"], "qg": out["qg"],
        },
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["snowncv"],
            "graupel_acc": out["graupelncv"],
        },
        diagnostics={"sr": out["sr"]},
    )


__all__ = ["goddard_run", "goddard_physics_tendency"]
