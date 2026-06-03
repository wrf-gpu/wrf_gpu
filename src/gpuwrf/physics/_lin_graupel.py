"""Purdue-Lin graupel process block (T<0C and T>0C), per-cell vectorized.

Split out of ``microphysics_lin`` for readability. Mirrors the graupel section
of WRF ``clphy1d`` (module_mp_lin.F lines ~1573-1883) exactly, including the
dry/wet growth comparison (``delta4``), the ``qgz==0`` short-circuit (go to
4000), and the Bigg freezing / deposition / melting / melt-evaporation terms.
"""

from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.physics import lin_constants as C


def _graupel_block(cold, qlz, qiz, qrz, qsz, qgz, rho, orho, sqrho, visc,
                   schmidt, xka, diffwv, tem, temcc, qvz, qvoqsiz, qvoqswz,
                   qsiz, qswz, rs0, olambdar, olambdas, olambdag, vtr, vts, vtg,
                   qlzodt, qizodt, qrzodt, qszodt, qgzodt, odtb, dtb):
    pi = C.PI
    pio4 = C.PIO4
    have_g = qgz > 0.0

    # ================= T < 0 C graupel =================
    # (1) pgaut
    alpha2 = 1.0e-3 * jnp.exp(0.09 * temcc)
    pgaut_cold = jnp.maximum(0.0, odtb * (qsz - C.QS0) * (1.0 - jnp.exp(-alpha2 * dtb)))
    # (2) pgfr (Bigg freezing of rain), qrz>1e-8
    Bp = 100.0
    Ap = 0.66
    olr = olambdar
    tmp1_gf = olr * olr * olr
    tmp2_gf = (20.0 * pi * pi * Bp * C.XNOR * C.RHOWATER * orho
               * (jnp.exp(-Ap * temcc) - 1.0) * tmp1_gf * tmp1_gf * olr)
    pgfr_cold = jnp.where(qrz > 1.0e-8, jnp.minimum(tmp2_gf, qrzodt), 0.0)

    # dry processes (only meaningful where qgz>0; else go-to-4000 zeros them)
    egw = 1.0
    constg = jnp.sqrt(4.0 * C.G * C.RHOGRAUL * 0.33334 * orho * C.OCDRAG)
    tmp1_dry = pio4 * C.XNOG * C.GAM3PT5 * constg * olambdag ** 3.5
    pgacw_dry = jnp.minimum(qlz * egw * tmp1_dry, qlzodt)
    egi = 0.1
    pgaci_dry = jnp.minimum(qiz * egi * tmp1_dry, qizodt)
    egs_cold = jnp.exp(0.09 * temcc)
    tmpa_gs = olambdas * olambdas
    tmpb_gs = olambdag * olambdag
    tmpc_gs = olambdas * olambdag
    tmp1_gs = pi * pi * C.XNOS * C.XNOG * jnp.abs(vts - vtg) * orho
    tmp2_gs = tmpa_gs * tmpa_gs * olambdag * (5.0 * tmpa_gs + 2.0 * tmpc_gs + 0.5 * tmpb_gs)
    pgacs_dry = jnp.minimum(tmp1_gs * egs_cold * C.RHOSNOW * tmp2_gs, qszodt)
    egr = 1.0
    tmpa_gr = olambdar * olambdar
    tmpb_gr = olambdag * olambdag
    tmpc_gr = olambdar * olambdag
    tmp1_gr = pi * pi * C.XNOR * C.XNOG * jnp.abs(vtr - vtg) * orho
    tmp2_gr = tmpa_gr * tmpa_gr * olambdag * (5.0 * tmpa_gr + 2.0 * tmpc_gr + 0.5 * tmpb_gr)
    pgacr_dry = jnp.minimum(tmp1_gr * egr * C.RHOWATER * tmp2_gr, qrzodt)
    pdry = pgacw_dry + pgaci_dry + pgacs_dry + pgacr_dry

    # wet processes
    pgacip_raw = jnp.minimum(10.0 * pgaci_dry, qizodt)
    pgacsp_raw = jnp.minimum(pgacs_dry * 1.0 / egs_cold, qszodt)
    # pgwet (only if temcc>-40)
    term0_w = constg * olambdag ** 5.5 / visc
    delrs_w = rs0 - qvz
    tmp0_w = 1.0 / (C.XLF + C.CW * temcc)
    tmp1_w = (2.0 * pi * C.XNOG * (rho * C.XLV * diffwv * delrs_w - xka * temcc)
              * orho * tmp0_w)
    constg2_w = (C.VF1S * olambdag * olambdag
                 + C.VF2S * schmidt ** 0.33334 * C.GAM2PT75 * jnp.sqrt(term0_w))
    tmp3_w = tmp1_w * constg2_w + (pgacip_raw + pgacsp_raw) * (1.0 - C.CI * temcc * tmp0_w)
    tmp3_w = jnp.maximum(0.0, tmp3_w)
    pgwet_raw = jnp.minimum(tmp3_w, qlzodt + qszodt + qizodt)
    warm_enough = temcc > -40.0
    # delta4 decision: if temcc>-40: delta4 = (pdry<pgwet ? 1 : 0); else delta4=1
    delta4 = jnp.where(warm_enough, jnp.where(pdry < pgwet_raw, 1.0, 0.0), 1.0)
    pgwet = jnp.where(warm_enough, pgwet_raw, 0.0)
    pgacrp_raw = pgwet - pgacw_dry - pgacip_raw - pgacsp_raw

    # (8) pgdep/pgsub
    tmpa_gd = C.RVAPOR * xka * tem * tem
    tmpb_gd = C.XLS * C.XLS * rho * qsiz * diffwv
    tmpc_gd = tmpa_gd * qsiz * diffwv
    abg = 2.0 * pi * (qvoqsiz - 1.0) * tmpc_gd / (tmpa_gd + tmpb_gd)
    term0_gd = constg * olambdag ** 5.5 / visc
    constg2_gd = (C.VF1S * olambdag * olambdag
                  + C.VF2S * schmidt ** 0.33334 * C.GAM2PT75 * jnp.sqrt(term0_gd))
    tmp2_gd = abg * C.XNOG * constg2_gd
    pgdep_cold = jnp.maximum(0.0, tmp2_gd)
    pgsub_cold = jnp.maximum(jnp.minimum(0.0, tmp2_gd), -qgzodt)

    # apply qgz==0 short-circuit (go to 4000 zeros dry/wet/dep/sub but KEEPS
    # pgaut, pgfr which are computed before the jump). delta4 stays at its
    # default (0.) when skipped -- WRF leaves delta4 unchanged from prior k...
    # but delta4 is a scalar reset per-process-use; to be faithful, where qgz==0
    # the graupel-dry/wet/dep terms are zero and delta4 is irrelevant (all the
    # terms it gates are zero). We set delta4=0 there (its value only multiplies
    # zeroed terms in the conservation update for qgz==0 cells).
    have_g_cold = cold & have_g
    pgacw_c = jnp.where(have_g_cold, pgacw_dry, 0.0)
    pgaci_c = jnp.where(have_g_cold, pgaci_dry, 0.0)
    pgacs_c = jnp.where(have_g_cold, pgacs_dry, 0.0)
    pgacr_c = jnp.where(have_g_cold, pgacr_dry, 0.0)
    pgacip_c = jnp.where(have_g_cold, pgacip_raw, 0.0)
    pgacsp_c = jnp.where(have_g_cold, pgacsp_raw, 0.0)
    pgwet_c = jnp.where(have_g_cold, pgwet, 0.0)
    pgacrp_c = jnp.where(have_g_cold, pgacrp_raw, 0.0)
    pgdep_c = jnp.where(have_g_cold, pgdep_cold, 0.0)
    pgsub_c = jnp.where(have_g_cold, pgsub_cold, 0.0)
    delta4_cold = jnp.where(have_g_cold, delta4, 0.0)

    # ================= T > 0 C graupel =================
    # (1) pgacw
    constg_w = jnp.sqrt(4.0 * C.G * C.RHOGRAUL * 0.33334 * orho * C.OCDRAG)
    tmp1_wgw = pio4 * C.XNOG * C.GAM3PT5 * constg_w * olambdag ** 3.5
    pgacw_warm = jnp.minimum(qlz * 1.0 * tmp1_wgw, qlzodt)
    # (2) pgacr
    tmp2_wgr = (olambdar * olambdar) ** 2 * olambdag * (
        5.0 * (olambdar * olambdar) + 2.0 * (olambdar * olambdag)
        + 0.5 * (olambdag * olambdag))
    tmp1_wgr = pi * pi * C.XNOR * C.XNOG * jnp.abs(vtr - vtg) * orho
    pgacr_warm = jnp.minimum(tmp1_wgr * 1.0 * C.RHOWATER * tmp2_wgr, qrzodt)
    # (3) pgmlt
    delrs_wm = rs0 - qvz
    term1_wm = 2.0 * pi * orho * (C.XLV * diffwv * rho * delrs_wm - xka * temcc)
    term0_wm = (jnp.sqrt(4.0 * C.G * C.RHOGRAUL * 0.33334 * orho * C.OCDRAG)
                * olambdag ** 5.5 / visc)
    constg2_wm = (C.VF1S * olambdag * olambdag
                  + C.VF2S * schmidt ** 0.33334 * C.GAM2PT75 * jnp.sqrt(term0_wm))
    tmp2_wm = C.XNOG * constg2_wm
    tmp3_wm = term1_wm * C.OXLF * tmp2_wm - C.CWOXLF * temcc * (pgacw_warm + pgacr_warm)
    pgmlt_warm = jnp.maximum(jnp.minimum(0.0, tmp3_wm), -qgzodt)
    # (4) pgmltevp
    tmpa_we = C.RVAPOR * xka * tem * tem
    tmpb_we = C.XLV * C.XLV * rho * qswz * diffwv
    tmpc_we = tmpa_we * qswz * diffwv
    tmpd_we = jnp.minimum(0.0, (qvoqswz - 0.90) * qswz * odtb)
    abg_we = 2.0 * pi * (qvoqswz - 0.90) * tmpc_we / (tmpa_we + tmpb_we)
    tmp2_wev = abg_we * C.XNOG * constg2_wm
    tmp3_wev = jnp.maximum(jnp.minimum(0.0, tmp2_wev), tmpd_we)
    pgmltevp_warm = jnp.maximum(tmp3_wev, -qgzodt)
    # (5) pgacs (egs=1)
    tmpa_wgs = olambdas * olambdas
    tmpb_wgs = olambdag * olambdag
    tmpc_wgs = olambdas * olambdag
    tmp1_wgs = pi * pi * C.XNOS * C.XNOG * jnp.abs(vts - vtg) * orho
    tmp2_wgs = tmpa_wgs * tmpa_wgs * olambdag * (5.0 * tmpa_wgs + 2.0 * tmpc_wgs + 0.5 * tmpb_wgs)
    pgacs_warm = jnp.minimum(tmp1_wgs * 1.0 * C.RHOSNOW * tmp2_wgs, qszodt)

    warm = jnp.logical_not(cold)
    # combine cold/warm. The T>0C path does not compute pgaut/pgfr/pgacip/...
    pgaut = jnp.where(cold, pgaut_cold, 0.0)
    pgfr = jnp.where(cold, pgfr_cold, 0.0)
    pgacw = jnp.where(cold, pgacw_c, pgacw_warm)
    pgaci = jnp.where(cold, pgaci_c, 0.0)
    pgacr = jnp.where(cold, pgacr_c, pgacr_warm)
    pgacs = jnp.where(cold, pgacs_c, pgacs_warm)
    pgacip = jnp.where(cold, pgacip_c, 0.0)
    pgacrp = jnp.where(cold, pgacrp_c, 0.0)
    pgacsp = jnp.where(cold, pgacsp_c, 0.0)
    pgwet = jnp.where(cold, pgwet_c, 0.0)
    pgsub = jnp.where(cold, pgsub_c, 0.0)
    pgdep = jnp.where(cold, pgdep_c, 0.0)
    pgmlt = jnp.where(cold, 0.0, pgmlt_warm)
    pgmltevp = jnp.where(cold, 0.0, pgmltevp_warm)
    delta4 = jnp.where(cold, delta4_cold, 0.0)

    return (pgaut, pgfr, pgacw, pgaci, pgacr, pgacs, pgacip, pgacrp, pgacsp,
            pgwet, pgsub, pgdep, pgmlt, pgmltevp, delta4)
