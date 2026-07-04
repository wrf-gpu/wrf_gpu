"""JAX aerosol-aware Morrison 2-moment microphysics (WRF mp_physics=40).

Faithful fp64 port of WRF ``phys/module_mp_morr_two_moment_aero.F`` as a DELTA
on top of the savepoint-proven base-Morrison port
(``microphysics_morrison.py``), for the aerosol-aware operating point the
oracle uses (``proofs/v023/oracle/morraero``): ``aercu_opt=2`` ->
``INUM=0/iinum=0`` (prognostic droplet number NC), ``IACT=4`` (Abdul-Razzak &
Ghan 2000 activation from the 10 prescribed CESM aerosol modes), ``INUC=2``
(Liu & Penner 2005 aerosol ice nucleation), ``IBASE=2``, ``ISUB=0``,
``morr_rimed_ice=0`` (graupel), WRF_CHEM compiled out, radar diagnostics off.

Enumerated Fortran deltas ported (aero.F line refs; diff vs
``module_mp_morr_two_moment.F``):

 1. Init (``MORR_TWO_MOMENT_INIT_AERO`` l.469): DCS=350e-6 (base 125e-6)
    cascading into LAMMINI, CONS21, CONS22 and the ice->snow 2*DCS threshold;
    plus the ``*_pamdm`` activation-constant block and CCNFACT table
    (l.631-688) and the 10-mode aerosol DATA tables (l.273-324).
    -> ``_morrison_aero_cold.py`` module constants.
 2. Wrapper (``MP_MORR_TWO_MOMENT_AERO`` l.705+):
    * WVAR = KZH(I,K+1,J)/20 clamped [0.10, 50] (l.923-933; base: 0.5 const);
      KZH is read one staggered level up (kme=kte+1 layout).
    * NC as 12th prognostic: nc1d(k)=NC(i,k,j) in, NC(i,k,j)=nc1d(k) out
      (l.1022, l.1145; non-chem branch, F_QNDROP=.FALSE.).
    * AEROCU -> internal-mode shuffle + mass multipliers (l.1031-1059):
      mode1=SULFATE(idx6) with naer1=5.64259e13*m1**0.58, mode2=SEASALT(idx5),
      modes3-6=DUST1..4(idx1..4, x1.44), mode7=OCPHO(idx9, x1.54),
      mode8=BCPHO(idx7, x1.37), mode9=OCPHI(idx10, x1.25),
      mode10=BCPHI(idx8, x1.37); all x aercu_fct x 1e-9 (ug/m3 -> kg/m3).
    * CCN1..7_GS diagnostics = sum_L naer(k,L)*ccnfact(x,L) (l.1061-1069).
    * EFCG/EFIG/EFSG clamps for RRTMG + WACT=WVAR+W (l.1130-1136).
 3. Micro (``MORR_TWO_MOMENT_MICRO``):
    * iinum=0: constant-droplet sets skipped (l.1797, l.2711, l.4997);
      QSMALL zeroing zeroes the prognostic NC (l.1690); cold/warm-branch
      MAX(0.,NC3D) floors and slope-clamp write-backs now feed live NC.
    * Droplet activation blocks (INUM=0) after the saturation adjustment in
      BOTH temperature branches (l.2388-2683 warm, l.3903-4215 cold); with
      IBASE=2/ISUB=0/IACT=4 the executed path is identical in both:
      gate QC3D+QC3DTEN*DT>=QSMALL (the DUM>=0.001 gate is always true since
      DUM=MAX(W+WVAR,0.10)), wbar=W3D+WVAR (unfloored, per the IACT=4
      reassignment), nact from ``mdm_prescribed_activate``,
      NC3DTEN += MAX(0,(nact-NC3D)/DT).
    * INUC=2 ice nucleation (l.3528-3552) replacing the Cooper curve, same
      ``(QVQVS>=0.999 & T<=265.15) or QVQVSI>=1.08`` trigger, KC2 from
      ``mdm_prescribed_nucleati`` (wbar=W+WVAR), NNUCCD floored at 0.
    * Sedimentation DUMFNC = MAX(0, NC3D+NC3DTEN*DT) (l.4312; iinum=0).
    * Cloud-free effective radii EFFI=4.99 / EFFC=2.49 (l.4953-4974).
    * NO NC aerosol bound at the end (the (NANEW1+NANEW2)/RHO bound is
      iinum=0 & IACT=2 only, l.4993-4995) and NO constant reset (l.4997).
 4. ``mdm_prescribed_activate/maxsat/nucleati/hetero/hf`` + ``DERF1``
    (l.5334-5445, l.5626-6177) -> ``_morrison_aero_cold.py``.

Deltas NOT ported (unexercised by the savepoints, all diagnostic-only):
 * The ``diagflag`` radar branch incl. the second refl10cm_hm call on
   NR/QR/NS/QS+cumulus(CU_UAF) -> mskf_refl_10cm (l.1185-1201): the oracle
   runs DIAGFLAG=.FALSE., DO_RADAR_REF=0 exactly like the base oracle; the
   NR_CU/QR_CU/NS_CU/QS_CU/CU_UAF inputs are inert zeros here.
 * IACT=1/2/3 and IBASE=1 activation paths, INUC=0/1 (dead under aercu_opt=2).
 * WRF-CHEM-only wetscav/rainprod/evapprod/QLSINK/PREC* optionals (compiled
   out in the oracle build, PRESENT()=false in WRF's mp_physics=40 call).
 * ``mdm_prescribed_polysvp`` (defined but never called in the aero module).

Reused base-port helpers by IMPORT (Fortran unchanged): ``polysvp``,
``gamma_fn``, ``_slope_generic``, ``_slope_droplet`` (this file) and
``_slope_final`` (in ``_morrison_aero_cold``).  Copy-and-modify (Fortran or
its constants changed): the micro driver body below (prognostic-NC plumbing +
activation), and in ``_morrison_aero_cold.py`` the cold branch (INUC=2 + DCS
cascade + live in-place number semantics), sedimentation (DUMFNC, LAMMINI)
and finalize (DCS, LAMMINI, Reff defaults, prognostic NC).

WRF unsuffixed-REAL literal handling: identical to the base port — plain
Python fp64 literals, matching the binding fp64 oracle built with
``-fdefault-real-8`` (default-REAL literals ARE fp64 there); the base port
validated this convention at 1e-9.

Validation: per-column WRF savepoint parity against the unmodified Fortran
scheme (``proofs/v022/f2_oracles/morrison_aero``), fp64 machine band binding
(tests/savepoint/test_morrison_aero_parity.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.physics import morrison_constants as C
from gpuwrf.physics.microphysics_morrison import (
    polysvp, _slope_generic, _slope_droplet)
from gpuwrf.physics import _morrison_aero_cold as A
from gpuwrf.physics._morrison_aero_cold import (
    _cold_branch_aero, _sedimentation_aero, _finalize_aero,
    mdm_prescribed_activate)


@jax.jit
def morrison_aero_run(th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, nc,
                      pii, p, dz, w, kzh,
                      aero_dust1, aero_dust2, aero_dust3, aero_dust4,
                      aero_seasalt, aero_sulfate, aero_bcpho, aero_bcphi,
                      aero_ocpho, aero_ocphi,
                      dt, aercu_fct=1.0):
    """Run one aerosol-aware Morrison step on a batch of columns.

    All 3-D-in-WRF args have shape (ncol, kx), bottom-up in k, except ``kzh``
    which carries the staggered padding level: (ncol, kx+1) — the wrapper
    reads KZH(I,K+1,J).  The 10 ``aero_*`` args are the AEROCU species in
    WRF Registry order (ug/m3).  Returns the base outputs + ``nc`` and the
    aero diagnostics (efcg/efig/efsg/wact/ccn1..7).
    """
    th = jnp.asarray(th)
    f = th.dtype
    qv = jnp.asarray(qv, f); qc = jnp.asarray(qc, f); qr = jnp.asarray(qr, f)
    qi = jnp.asarray(qi, f); qs = jnp.asarray(qs, f); qg = jnp.asarray(qg, f)
    ni = jnp.asarray(ni, f); ns = jnp.asarray(ns, f); nr = jnp.asarray(nr, f)
    ng = jnp.asarray(ng, f); nc = jnp.asarray(nc, f)
    pii = jnp.asarray(pii, f); p = jnp.asarray(p, f)
    dz = jnp.asarray(dz, f); w = jnp.asarray(w, f)
    kzh = jnp.asarray(kzh, f)
    dt = jnp.asarray(dt, f)
    aercu_fct = jnp.asarray(aercu_fct, f)

    QSMALL = C.QSMALL
    R = C.R; RV = C.RV; CP = C.CP; EP_2 = C.EP_2; PI = C.PI

    # ----------------------------------------------------------------------
    # Wrapper: WVAR from KZH one staggered level up (aero.F l.923-933).
    # ----------------------------------------------------------------------
    wvar = kzh[:, 1:] / 20.0
    wvar = jnp.maximum(0.10, wvar)
    wvar = jnp.minimum(50.0, wvar)

    # Wrapper: AEROCU -> internal modes (shuffle + multipliers, l.1031-1059).
    maero = [
        aercu_fct * aero_sulfate * 1.0e-9,
        aercu_fct * aero_seasalt * 1.0e-9,
        aercu_fct * 1.44 * aero_dust1 * 1.0e-9,
        aercu_fct * 1.44 * aero_dust2 * 1.0e-9,
        aercu_fct * 1.44 * aero_dust3 * 1.0e-9,
        aercu_fct * 1.44 * aero_dust4 * 1.0e-9,
        aercu_fct * 1.54 * aero_ocpho * 1.0e-9,
        aercu_fct * 1.37 * aero_bcpho * 1.0e-9,
        aercu_fct * 1.25 * aero_ocphi * 1.0e-9,
        aercu_fct * 1.37 * aero_bcphi * 1.0e-9,
    ]
    naer = [5.64259e13 * maero[0] ** 0.58]
    for m in range(1, A.NAER_CU):
        naer.append(maero[m] * A.NUM_TO_MASS_AER[m])
    maero = jnp.stack(maero, axis=-1)      # (ncol, kx, 10)
    naer = jnp.stack(naer, axis=-1)        # (ncol, kx, 10)

    # Wrapper: CCN diagnostics (pure functions of naer, l.1061-1069);
    # sequential per-mode accumulation matches the Fortran DO L order.
    ccn = []
    for s in range(A.PSAT_PAMDM):
        acc = jnp.zeros_like(th)
        for m in range(A.NAER_CU):
            acc = acc + naer[..., m] * A.CCNFACT_PAMDM[s, m]
        ccn.append(acc)

    # wrapper: T = TH*PII
    t = th * pii

    # ----------------------------------------------------------------------
    # Setup: latent heats, CPM, saturation, RHO. (cumulus tendencies zero.)
    # ----------------------------------------------------------------------
    xxlv = 3.1484e6 - 2370.0 * t
    xxls = 3.15e6 - 2370.0 * t + 0.3337e6
    cpm = CP * (1.0 + 0.887 * qv)

    evs = jnp.minimum(0.99 * p, polysvp(t, 0))
    eis = jnp.minimum(0.99 * p, polysvp(t, 1))
    eis = jnp.where(eis > evs, evs, eis)
    qvs = EP_2 * evs / (p - evs)
    qvi = EP_2 * eis / (p - eis)
    qvqvs = qv / qvs
    qvqvsi = qv / qvi
    rho = p / (R * t)

    def _sub_remove_liq(qv_, t_, qx, thr):
        rm = (qvqvs < 0.9) & (qx < thr)
        qv2 = jnp.where(rm, qv_ + qx, qv_)
        t2 = jnp.where(rm, t_ - qx * xxlv / cpm, t_)
        qx2 = jnp.where(rm, 0.0, qx)
        return qv2, t2, qx2

    def _sub_remove_ice(qv_, t_, qx, thr):
        rm = (qvqvsi < 0.9) & (qx < thr)
        qv2 = jnp.where(rm, qv_ + qx, qv_)
        t2 = jnp.where(rm, t_ - qx * xxls / cpm, t_)
        qx2 = jnp.where(rm, 0.0, qx)
        return qv2, t2, qx2

    qv, t, qr = _sub_remove_liq(qv, t, qr, 1.0e-8)
    qv, t, qc = _sub_remove_liq(qv, t, qc, 1.0e-8)
    qv, t, qi = _sub_remove_ice(qv, t, qi, 1.0e-8)
    qv, t, qs = _sub_remove_ice(qv, t, qs, 1.0e-8)
    qv, t, qg = _sub_remove_ice(qv, t, qg, 1.0e-8)

    xlf = xxls - xxlv

    # QSMALL zeroing of mass+number (NC is prognostic here: zeroed with QC)
    def _qsmall_zero(qx, nx):
        m = qx < QSMALL
        return jnp.where(m, 0.0, qx), jnp.where(m, 0.0, nx)

    qc, nc = _qsmall_zero(qc, nc)
    qr, nr = _qsmall_zero(qr, nr)
    qi, ni = _qsmall_zero(qi, ni)
    qs, ns = _qsmall_zero(qs, ns)
    qg, ng = _qsmall_zero(qg, ng)

    mu = 1.496e-6 * t ** 1.5 / (t + 120.0)
    dum_dc = (C.RHOSU / rho) ** 0.54
    ain = (C.RHOSU / rho) ** 0.35 * C.AI
    arn = dum_dc * C.AR
    asn = dum_dc * C.AS
    acn = C.G * C.RHOW / (18.0 * mu)
    agn = dum_dc * C.AG
    kap = 1.414e3 * mu
    dv = 8.794e-5 * t ** 1.81 / p
    sc = mu / (rho * dv)

    dqsdt = xxlv * qvs / (RV * t * t)
    dqsidt = xxls * qvi / (RV * t * t)
    abi = 1.0 + dqsidt * xxls / cpm
    ab = 1.0 + dqsdt * xxlv / cpm

    # GOTO 200 mask
    empty = ((qc < QSMALL) & (qi < QSMALL) & (qs < QSMALL)
             & (qr < QSMALL) & (qg < QSMALL))
    skip200 = empty & (((t < 273.15) & (qvqvsi < 0.999))
                       | ((t >= 273.15) & (qvqvs < 0.999)))
    do_cell = ~skip200

    # iinum=0: NC stays prognostic — the base port's constant-droplet
    # assignment (NDCNST*1e6/rho) is skipped.

    warm = t >= 273.15

    z = jnp.zeros_like(t)
    qv_ten = z; t_ten = z; qc_ten = z; qr_ten = z
    qi_ten = z; qni_ten = z; qg_ten = z
    nc_ten = z; ni_ten = z; ns_ten = z; nr_ten = z; ng_ten = z

    # =====================================================================
    # WARM BRANCH (T >= 273.15) — identical to the base port; nc prognostic.
    # =====================================================================
    melt_sn = warm & (qs < 1.0e-6)
    qr = jnp.where(melt_sn, qr + qs, qr)
    nr = jnp.where(melt_sn, nr + ns, nr)
    t = jnp.where(melt_sn, t - qs * xlf / cpm, t)
    qs = jnp.where(melt_sn, 0.0, qs)
    ns = jnp.where(melt_sn, 0.0, ns)
    melt_g = warm & (qg < 1.0e-6)
    qr = jnp.where(melt_g, qr + qg, qr)
    nr = jnp.where(melt_g, nr + ng, nr)
    t = jnp.where(melt_g, t - qg * xlf / cpm, t)
    qg = jnp.where(melt_g, 0.0, qg)
    ng = jnp.where(melt_g, 0.0, ng)

    warm_active = warm & do_cell & ~(
        (qc < QSMALL) & (qs < 1.0e-8) & (qr < QSMALL) & (qg < 1.0e-8))

    nsw = jnp.maximum(ns, 0.0); ncw = jnp.maximum(nc, 0.0)
    nrw = jnp.maximum(nr, 0.0); ngw = jnp.maximum(ng, 0.0)

    lamr_w, n0rr_w, nr_w = _slope_generic(qr, nrw, PI * C.RHOW, 1.0 / 3.0,
                                          C.LAMMINR, C.LAMMAXR, QSMALL)
    lams_w, n0s_w, ns_w = _slope_generic(qs, nsw, C.CONS1, 1.0 / C.DS,
                                         C.LAMMINS, C.LAMMAXS, QSMALL)
    lamg_w, n0g_w, ng_w = _slope_generic(qg, ngw, C.CONS2, 1.0 / C.DG,
                                         C.LAMMING, C.LAMMAXG, QSMALL)
    lamc_w, pgam_w, _cd_w, ncw2, _gp1w, _gp4w = _slope_droplet(qc, ncw, t, p, QSMALL)

    nr = jnp.where(warm_active, nr_w, nr)
    ns = jnp.where(warm_active, ns_w, ns)
    ng = jnp.where(warm_active, ng_w, ng)
    nc = jnp.where(warm_active, ncw2, nc)

    one = jnp.ones_like(t)
    eps = 1.0e-30

    qc_ge6 = qc >= 1.0e-6
    prc_w = jnp.where(qc_ge6, 1350.0 * qc ** 2.47 * (nc / 1.0e6 * rho) ** (-1.79), 0.0)
    nprc1_w = jnp.where(qc_ge6, prc_w / C.CONS29, 0.0)
    nprc_w = jnp.where(qc_ge6, prc_w / (qc / jnp.where(nc > 0, nc, one)), 0.0)
    nprc_w = jnp.minimum(nprc_w, nc / dt)
    nprc1_w = jnp.minimum(nprc1_w, nprc_w)
    prc_w = jnp.where(qc_ge6, prc_w, 0.0)

    qrqc8 = (qr >= 1.0e-8) & (qc >= 1.0e-8)
    pra_w = jnp.where(qrqc8, 67.0 * (qc * qr) ** 1.15, 0.0)
    npra_w = jnp.where(qrqc8, pra_w / (qc / jnp.where(nc > 0, nc, one)), 0.0)

    qr8 = qr >= 1.0e-8
    inv_lamr = jnp.where(lamr_w > 0, 1.0 / jnp.where(lamr_w > 0, lamr_w, one), 0.0)
    br_dum = jnp.where(inv_lamr < 300.0e-6, 1.0,
                       2.0 - jnp.exp(2300.0 * (inv_lamr - 300.0e-6)))
    nragg_w = jnp.where(qr8, -5.78 * br_dum * nr * qr * rho, 0.0)

    qr_qs = qr >= QSMALL
    epsr_w = jnp.where(qr_qs,
                       2.0 * PI * n0rr_w * rho * dv
                       * (C.F1R / (lamr_w * lamr_w + eps)
                          + C.F2R * (arn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                          * C.CONS9 / (lamr_w ** C.CONS34 + eps)), 0.0)
    pre_w = jnp.where(qv < qvs, jnp.minimum(epsr_w * (qv - qvs) / ab, 0.0), 0.0)

    qr_qs8 = (qr >= 1.0e-8) & (qs >= 1.0e-8)
    ums_s = jnp.minimum(asn * C.CONS3 / (lams_w ** C.BS + eps), 1.2 * dum_dc)
    umr_s = jnp.minimum(arn * C.CONS4 / (lamr_w ** C.BR + eps), 9.1 * dum_dc)
    pracs_w = jnp.where(
        qr_qs8,
        C.CONS41 * (((1.2 * umr_s - 0.95 * ums_s) ** 2 + 0.08 * ums_s * umr_s) ** 0.5
                    * rho * n0rr_w * n0s_w / (lamr_w ** 3 + eps)
                    * (5.0 / (lamr_w ** 3 * lams_w + eps)
                       + 2.0 / (lamr_w ** 2 * lams_w ** 2 + eps)
                       + 0.5 / (lamr_w * lams_w ** 3 + eps))), 0.0)

    qr_qg8 = (qr >= 1.0e-8) & (qg >= 1.0e-8)
    umg_g = jnp.minimum(agn * C.CONS7 / (lamg_w ** C.BG + eps), 20.0 * dum_dc)
    umr_g = jnp.minimum(arn * C.CONS4 / (lamr_w ** C.BR + eps), 9.1 * dum_dc)
    ung_g = jnp.minimum(agn * C.CONS8 / (lamg_w ** C.BG + eps), 20.0 * dum_dc)
    unr_g = jnp.minimum(arn * C.CONS6 / (lamr_w ** C.BR + eps), 9.1 * dum_dc)
    pracg_w = jnp.where(
        qr_qg8,
        C.CONS41 * (((1.2 * umr_g - 0.95 * umg_g) ** 2 + 0.08 * umg_g * umr_g) ** 0.5
                    * rho * n0rr_w * n0g_w / (lamr_w ** 3 + eps)
                    * (5.0 / (lamr_w ** 3 * lamg_w + eps)
                       + 2.0 / (lamr_w ** 2 * lamg_w ** 2 + eps)
                       + 0.5 / (lamr_w * lamg_w ** 3 + eps))), 0.0)
    npracg_w = jnp.where(
        qr_qg8,
        C.CONS32 * rho * (1.7 * (unr_g - ung_g) ** 2 + 0.3 * unr_g * ung_g) ** 0.5
        * n0rr_w * n0g_w * (1.0 / (lamr_w ** 3 * lamg_w + eps)
                            + 1.0 / (lamr_w ** 2 * lamg_w ** 2 + eps)
                            + 1.0 / (lamr_w * lamg_w ** 3 + eps)), 0.0)
    npracg_w = jnp.where(qr_qg8, npracg_w - pracg_w / 5.2e-7, 0.0)

    qs8 = qs >= 1.0e-8
    dum_psmlt = -C.CPW / xlf * (t - 273.15) * pracs_w
    psmlt_w = jnp.where(
        qs8,
        2.0 * PI * n0s_w * kap * (273.15 - t) / xlf
        * (C.F1S / (lams_w * lams_w + eps)
           + C.F2S * (asn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS10 / (lams_w ** C.CONS35 + eps)) + dum_psmlt, 0.0)
    sub_s = qs8 & (qvqvs < 1.0)
    epss_w = jnp.where(
        sub_s,
        2.0 * PI * n0s_w * rho * dv
        * (C.F1S / (lams_w * lams_w + eps)
           + C.F2S * (asn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS10 / (lams_w ** C.CONS35 + eps)), 0.0)
    evpms_w = jnp.where(sub_s, (qv - qvs) * epss_w / ab, 0.0)
    evpms_w = jnp.where(sub_s, jnp.maximum(evpms_w, psmlt_w), 0.0)
    psmlt_w = jnp.where(sub_s, psmlt_w - evpms_w, psmlt_w)

    qg8 = qg >= 1.0e-8
    dum_pgmlt = -C.CPW / xlf * (t - 273.15) * pracg_w
    pgmlt_w = jnp.where(
        qg8,
        2.0 * PI * n0g_w * kap * (273.15 - t) / xlf
        * (C.F1S / (lamg_w * lamg_w + eps)
           + C.F2S * (agn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS11 / (lamg_w ** C.CONS36 + eps)) + dum_pgmlt, 0.0)
    sub_g = qg8 & (qvqvs < 1.0)
    epsg_w = jnp.where(
        sub_g,
        2.0 * PI * n0g_w * rho * dv
        * (C.F1S / (lamg_w * lamg_w + eps)
           + C.F2S * (agn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS11 / (lamg_w ** C.CONS36 + eps)), 0.0)
    evpmg_w = jnp.where(sub_g, (qv - qvs) * epsg_w / ab, 0.0)
    evpmg_w = jnp.where(sub_g, jnp.maximum(evpmg_w, pgmlt_w), 0.0)
    pgmlt_w = jnp.where(sub_g, pgmlt_w - evpmg_w, pgmlt_w)

    pracg_w = jnp.zeros_like(t)
    pracs_w = jnp.zeros_like(t)

    dum_qc = (prc_w + pra_w) * dt
    cons_qc = (dum_qc > qc) & (qc >= QSMALL)
    ratio = jnp.where(cons_qc, qc / jnp.where(dum_qc != 0, dum_qc, one), 1.0)
    prc_w = prc_w * ratio
    pra_w = pra_w * ratio
    dum_sn = (-psmlt_w - evpms_w + pracs_w) * dt
    cons_sn = (dum_sn > qs) & (qs >= QSMALL)
    rsn = jnp.where(cons_sn, qs / jnp.where(dum_sn != 0, dum_sn, one), 1.0)
    psmlt_w = psmlt_w * rsn; evpms_w = evpms_w * rsn; pracs_w = pracs_w * rsn
    dum_g = (-pgmlt_w - evpmg_w + pracg_w) * dt
    cons_g = (dum_g > qg) & (qg >= QSMALL)
    rg = jnp.where(cons_g, qg / jnp.where(dum_g != 0, dum_g, one), 1.0)
    pgmlt_w = pgmlt_w * rg; evpmg_w = evpmg_w * rg; pracg_w = pracg_w * rg
    dum_qr = (-pracs_w - pracg_w - pre_w - pra_w - prc_w + psmlt_w + pgmlt_w) * dt
    cons_qr = (dum_qr > qr) & (qr >= QSMALL)
    rqr = jnp.where(cons_qr & (pre_w != 0),
                    (qr / dt + pracs_w + pracg_w + pra_w + prc_w - psmlt_w - pgmlt_w)
                    / jnp.where(pre_w != 0, -pre_w, one), 1.0)
    pre_w = jnp.where(cons_qr, pre_w * rqr, pre_w)

    nsubr_w = jnp.where(pre_w < 0.0,
                        jnp.maximum(-1.0, pre_w * dt / jnp.where(qr > 0, qr, one))
                        * nr / dt, 0.0)
    nsmlts_w = jnp.where((evpms_w + psmlt_w) < 0.0,
                         jnp.maximum(-1.0, (evpms_w + psmlt_w) * dt / jnp.where(qs > 0, qs, one))
                         * ns / dt, 0.0)
    nsmltr_w = jnp.where(psmlt_w < 0.0,
                         jnp.maximum(-1.0, psmlt_w * dt / jnp.where(qs > 0, qs, one))
                         * ns / dt, 0.0)
    ngmltg_w = jnp.where((evpmg_w + pgmlt_w) < 0.0,
                         jnp.maximum(-1.0, (evpmg_w + pgmlt_w) * dt / jnp.where(qg > 0, qg, one))
                         * ng / dt, 0.0)
    ngmltr_w = jnp.where(pgmlt_w < 0.0,
                         jnp.maximum(-1.0, pgmlt_w * dt / jnp.where(qg > 0, qg, one))
                         * ng / dt, 0.0)

    qv_w = -pre_w - evpms_w - evpmg_w
    t_w = (pre_w * xxlv + (evpms_w + evpmg_w) * xxls
           + (psmlt_w + pgmlt_w - pracs_w - pracg_w) * xlf) / cpm
    qc_w = -pra_w - prc_w
    qr_w = pre_w + pra_w + prc_w - psmlt_w - pgmlt_w + pracs_w + pracg_w
    qni_w = psmlt_w + evpms_w - pracs_w
    qg_w = pgmlt_w + evpmg_w - pracg_w
    nc_w = -npra_w - nprc_w
    nr_w_t = (nprc1_w + nragg_w - npracg_w) + (nsubr_w - nsmltr_w - ngmltr_w)
    ns_w_t = nsmlts_w
    ng_w_t = ngmltg_w

    wa = warm_active
    qv_ten = jnp.where(wa, qv_ten + qv_w, qv_ten)
    t_ten = jnp.where(wa, t_ten + t_w, t_ten)
    qc_ten = jnp.where(wa, qc_ten + qc_w, qc_ten)
    qr_ten = jnp.where(wa, qr_ten + qr_w, qr_ten)
    qni_ten = jnp.where(wa, qni_ten + qni_w, qni_ten)
    qg_ten = jnp.where(wa, qg_ten + qg_w, qg_ten)
    nc_ten = jnp.where(wa, nc_ten + nc_w, nc_ten)
    nr_ten = jnp.where(wa, nr_ten + nr_w_t, nr_ten)
    ns_ten = jnp.where(wa, ns_ten + ns_w_t, ns_ten)
    ng_ten = jnp.where(wa, ng_ten + ng_w_t, ng_ten)

    # =====================================================================
    # COLD BRANCH (T < 273.15) — aero variant (INUC=2, DCS cascade, live NC).
    # =====================================================================
    cold = (~warm) & do_cell
    out = _cold_branch_aero(
        cold, t, qv, qc, qr, qi, qs, qg, nc, ni, ns, nr, ng,
        qvs, qvi, qvqvs, qvqvsi, ab, abi, rho, dv, mu, sc, kap,
        ain, arn, asn, agn, acn, dum_dc, xxlv, xxls, xlf, cpm, p,
        w, wvar, naer, dt)
    (qv_c, t_c, qc_c, qr_c, qi_c, qni_c, qg_c,
     nc_c, ni_c, ns_c, nr_c, ng_c,
     ni_clamp, ns_clamp, nr_clamp, ng_clamp, nc_clamp) = out

    qv_ten = jnp.where(cold, qv_ten + qv_c, qv_ten)
    t_ten = jnp.where(cold, t_ten + t_c, t_ten)
    qc_ten = jnp.where(cold, qc_ten + qc_c, qc_ten)
    qr_ten = jnp.where(cold, qr_ten + qr_c, qr_ten)
    qi_ten = jnp.where(cold, qi_ten + qi_c, qi_ten)
    qni_ten = jnp.where(cold, qni_ten + qni_c, qni_ten)
    qg_ten = jnp.where(cold, qg_ten + qg_c, qg_ten)
    nc_ten = jnp.where(cold, nc_ten + nc_c, nc_ten)
    ni_ten = jnp.where(cold, ni_ten + ni_c, ni_ten)
    ns_ten = jnp.where(cold, ns_ten + ns_c, ns_ten)
    nr_ten = jnp.where(cold, nr_ten + nr_c, nr_ten)
    ng_ten = jnp.where(cold, ng_ten + ng_c, ng_ten)
    # in-place number semantics (Fortran mutates NI3D..NC3D on cold cells)
    ni = jnp.where(cold, ni_clamp, ni)
    ns = jnp.where(cold, ns_clamp, ns)
    nr = jnp.where(cold, nr_clamp, nr)
    ng = jnp.where(cold, ng_clamp, ng)
    nc = jnp.where(cold, nc_clamp, nc)

    # =====================================================================
    # SATURATION ADJUSTMENT (liquid) PCC, all do_cell cells
    # =====================================================================
    dumt = t + dt * t_ten
    dumqv = qv + dt * qv_ten
    dum_svp = jnp.minimum(0.99 * p, polysvp(dumt, 0))
    dumqss = EP_2 * dum_svp / (p - dum_svp)
    dumqc = jnp.maximum(qc + dt * qc_ten, 0.0)
    dums = dumqv - dumqss
    pcc = dums / (1.0 + xxlv ** 2 * dumqss / (cpm * RV * dumt ** 2)) / dt
    pcc = jnp.where(pcc * dt + dumqc < 0.0, -dumqc / dt, pcc)
    pcc = jnp.where(do_cell, pcc, 0.0)
    qv_ten = qv_ten - pcc
    t_ten = t_ten + pcc * xxlv / cpm
    qc_ten = qc_ten + pcc

    # =====================================================================
    # DROPLET ACTIVATION (INUM=0, IACT=4, IBASE=2, ISUB=0) — aero.F
    # l.2388-2683 (warm) / l.3903-4215 (cold); the executed IACT=4 path is
    # identical in both branches and sits right after the saturation
    # adjustment, so it is applied once for all do_cell cells here.
    # Gate: QC3D+QC3DTEN*DT >= QSMALL.  (The ISUB=0 "DUM>=0.001" gate is
    # always true because DUM=MAX(W+WVAR,0.10)>=0.10.)  wbar = W3D+WVAR
    # (the IACT=4 reassignment drops the 0.10 floor); activate returns 0
    # for wbar<=0.
    # =====================================================================
    act = do_cell & ((qc + qc_ten * dt) >= QSMALL)
    wbar = w + wvar
    nact = mdm_prescribed_activate(wbar, t, rho, naer, maero, xxlv)
    dum2 = jnp.maximum((nact - nc) / dt, 0.0)
    nc_ten = nc_ten + jnp.where(act, dum2, 0.0)

    # =====================================================================
    # SEDIMENTATION (aero: DUMFNC = MAX(0, NC+NCTEN*DT), aero LAMMINI)
    # =====================================================================
    (qr_st, qi_st, qni_st, qc_st, qg_st,
     ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
     precrt, snowrt, snowprt, grplprt) = _sedimentation_aero(
        qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
        qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
        nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
        t, p, rho, dz, dt, do_cell)

    ni_ten = ni_ten + ni_sed
    ns_ten = ns_ten + ns_sed
    nr_ten = nr_ten + nr_sed
    nc_ten = nc_ten + nc_sed
    ng_ten = ng_ten + ng_sed
    qr_ten = qr_ten + qr_st
    qi_ten = qi_ten + qi_st
    qc_ten = qc_ten + qc_st
    qg_ten = qg_ten + qg_st
    qni_ten = qni_ten + qni_st

    # =====================================================================
    # FINAL STATE UPDATE (aero variant: prognostic NC, Reff defaults, DCS)
    # =====================================================================
    res = _finalize_aero(t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
                         qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
                         t_ten, qv_ten,
                         nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
                         xxlv, xxls, xlf, cpm, p, rho, dt, do_cell)
    (t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
     effc, effi, effs, effr, effg) = res

    th_out = t / pii

    rainncv = precrt
    snowncv = snowprt
    graupelncv = grplprt
    sr = snowrt / (precrt + 1.0e-12)

    # Wrapper aero outputs (aercu_opt>0, aero.F l.1130-1136)
    efcg = jnp.maximum(2.49, jnp.minimum(effc, 50.0))
    efig = jnp.maximum(4.99, jnp.minimum(effi, 120.0))
    efsg = jnp.maximum(9.99, jnp.minimum(effs, 999.0))
    wact = wvar + w

    return {
        "th": th_out, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
        "ni": ni, "ns": ns, "nr": nr, "ng": ng, "nc": nc,
        "effc": effc, "effi": effi, "effs": effs, "effr": effr, "effg": effg,
        "efcg": efcg, "efig": efig, "efsg": efsg, "wact": wact,
        "ccn1": ccn[0], "ccn2": ccn[1], "ccn3": ccn[2], "ccn4": ccn[3],
        "ccn5": ccn[4], "ccn6": ccn[5], "ccn7": ccn[6],
        "rainncv": rainncv, "snowncv": snowncv, "graupelncv": graupelncv, "sr": sr,
    }
