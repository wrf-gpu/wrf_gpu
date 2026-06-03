"""Purdue-Lin conservation feedback + state update + satadj + melt/freeze.

Mirrors module_mp_lin.F clphy1d lines ~1887-2382 exactly:

  * temcc<0 branch: depletion clamps for vapor/cloud-water/cloud-ice/rain/snow/
    graupel, delta2/delta3 (rain<1e-4 / snow<1e-4 partitioning), then the
    new-substance updates (pvapor, pclw, pcli, prain, psnow, pgraupel) and the
    theiz latent-heat update.
  * temcc>=0 branch: cloud-water / snow / graupel / rain depletion clamps and
    melting-driven updates.
  * common tail: supersaturation check -> satadj (Newton) if saturated; then
    ice/water melt/freeze (pihom, pimlt, pidw) + a second satadj.

Returns the updated (qvz, qlz, qiz, qrz, qsz, qgz, theiz, pclw, pvapor) for the
cell. ``thz`` is derived from ``theiz`` by the caller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.physics import lin_constants as C


def _es_liquid(tem):
    temcc = tem - 273.15
    return 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * temcc / (tem - C.SVP3))


def _es_ice(tem):
    return 1000.0 * C.SVP1 * jnp.exp(21.8745584 * (tem - 273.16) / (tem - 7.66))


def _qfromes(es, prez):
    return C.EP2 * es / (prez - es)


def _parama(temp, table):
    mtemp = -temp
    i0 = jnp.floor(mtemp).astype(jnp.int32)
    ratio = mtemp - i0.astype(temp.dtype)
    i0 = jnp.clip(i0, 0, table.shape[0] - 2)
    return table[i0] + ratio * (table[i0 + 1] - table[i0])


_PA1 = jnp.asarray(C.PARAMA1, dtype=jnp.float64)
_PA2 = jnp.asarray(C.PARAMA2, dtype=jnp.float64)


def _conservation_and_update(
        cold, gindex, dtb, odtb, tothz, prez, orho, qvz, qlz, qiz, qrz, qsz, qgz, theiz,
        qvzodt, qlzodt, qizodt, qrzodt, qszodt, qgzodt,
        psaut, psfw, psfi, praci, piacr, psaci, psacw, psdep, pssub, pracs, psacr,
        psmlt, psmltevp, praut, pracw, prevp,
        pgaut, pgfr, pgacw, pgaci, pgacr, pgacs, pgacip, pgacrp, pgacsp, pgwet,
        pgsub, pgdep, pgmlt, pgmltevp, delta4):
    xlvocp = C.XLVOCP
    xlfocp = C.XLFOCP
    qvmin = C.QVMIN

    # ============================================================
    # COLD branch (temcc < 0)
    # ============================================================
    gdelta4 = gindex * delta4
    g1sdelt4 = gindex * (1.0 - delta4)

    psdep_c = psdep
    pgdep_c = pgdep
    # combined water-vapor depletions
    tmp = psdep_c + pgdep_c * gindex
    f = jnp.where(tmp > qvzodt, qvzodt / tmp, 1.0)
    psdep_c = psdep_c * f
    pgdep_c = pgdep_c * f * gindex
    # combined cloud-water depletions
    praut_c = praut
    psacw_c = psacw
    psfw_c = psfw
    pracw_c = pracw
    pgacw_c = pgacw
    tmp = praut_c + psacw_c + psfw_c + pracw_c + gindex * pgacw_c
    f = jnp.where(tmp > qlzodt, qlzodt / tmp, 1.0)
    praut_c = praut_c * f
    psacw_c = psacw_c * f
    psfw_c = psfw_c * f
    pracw_c = pracw_c * f
    pgacw_c = pgacw_c * f * gindex
    # combined cloud-ice depletions
    psaut_c = psaut
    psaci_c = psaci
    praci_c = praci
    psfi_c = psfi
    pgaci_c = pgaci
    pgacip_c = pgacip
    tmp = psaut_c + psaci_c + praci_c + psfi_c + pgaci_c * gdelta4 + pgacip_c * g1sdelt4
    f = jnp.where(tmp > qizodt, qizodt / tmp, 1.0)
    psaut_c = psaut_c * f
    psaci_c = psaci_c * f
    praci_c = praci_c * f
    psfi_c = psfi_c * f
    pgaci_c = pgaci_c * f * gdelta4
    pgacip_c = pgacip_c * f * g1sdelt4
    # combined all rain processes
    piacr_c = piacr
    psacr_c = psacr
    prevp_c = prevp
    pgfr_c = pgfr
    pgacr_c = pgacr
    pgacrp_c = pgacrp
    tmp_r = (piacr_c + psacr_c - prevp_c - praut_c - pracw_c
             + pgfr_c * gindex + pgacr_c * gdelta4 + pgacrp_c * g1sdelt4)
    f = jnp.where(tmp_r > qrzodt, qrzodt / tmp_r, 1.0)
    piacr_c = piacr_c * f
    psacr_c = psacr_c * f
    prevp_c = prevp_c * f
    pgfr_c = pgfr_c * f * gindex
    pgacr_c = pgacr_c * f * gdelta4
    pgacrp_c = pgacrp_c * f * g1sdelt4
    # delta2 / delta3
    delta2 = jnp.where((qrz < 1.0e-4) & (qsz < 1.0e-4), 1.0, 0.0)
    delta3 = jnp.where(qrz < 1.0e-4, 1.0, 0.0)
    # gindex=1 here so the gindex==0 override does not apply.
    # combined all snow processes
    pssub_c = pssub
    pracs_c = pracs
    pgaut_c = pgaut
    pgacs_c = pgacs
    pgacsp_c = pgacsp
    tmp_s = (-pssub_c - (psaut_c + psaci_c + psacw_c + psfw_c + psfi_c
                         + praci_c * delta3 + piacr_c * delta3 + psdep_c)
             + pgaut_c * gindex + pgacs_c * gdelta4 + pgacsp_c * g1sdelt4
             + pracs_c * (1.0 - delta2) - psacr_c * delta2)
    f = jnp.where(tmp_s > qszodt, qszodt / tmp_s, 1.0)
    pssub_c = pssub_c * f
    pracs_c = pracs_c * f
    pgaut_c = pgaut_c * f * gindex
    pgacs_c = pgacs_c * f * gdelta4
    pgacsp_c = pgacsp_c * f * g1sdelt4
    # combined all graupel processes
    # if delta4<0.5: redefine pgwet
    pgwet_c = jnp.where(delta4 < 0.5,
                        pgacrp_c + pgacw_c + pgacip_c + pgacsp_c, pgwet)
    pgsub_c = pgsub
    tmp_g = (-pgaut_c - pgfr_c - pgacw_c * delta4 - pgaci_c * delta4
             - pgacr_c * delta4 - pgacs_c * delta4
             - pgwet_c * (1.0 - delta4) - pgsub_c - pgdep_c
             - psacr_c * (1.0 - delta2) - pracs_c * (1.0 - delta2)
             - praci_c * (1.0 - delta3) - piacr_c * (1.0 - delta3))
    f = jnp.where(tmp_g > qgzodt, qgzodt / tmp_g, 1.0)
    pgsub_c = pgsub_c * f

    # new water substances (cold)
    pvapor_cold = -pssub_c - psdep_c - prevp_c - pgsub_c * gindex - pgdep_c * gindex
    qvz_cold = jnp.maximum(qvmin, qvz + dtb * pvapor_cold)
    pclw_cold = -praut_c - pracw_c - psacw_c - psfw_c - pgacw_c * gindex
    qlz_cold = jnp.maximum(0.0, qlz + dtb * pclw_cold)
    pcli_cold = -psaut_c - psfi_c - psaci_c - praci_c - pgaci_c * gdelta4 - pgacip_c * g1sdelt4
    qiz_cold = jnp.maximum(0.0, qiz + dtb * pcli_cold)
    tmp_r2 = (piacr_c + psacr_c - prevp_c - praut_c - pracw_c
              + pgfr_c * gindex + pgacr_c * gdelta4 + pgacrp_c * g1sdelt4)
    prain_cold = -tmp_r2
    qrz_cold = jnp.maximum(0.0, qrz + dtb * prain_cold)
    tmp_s2 = (-pssub_c - (psaut_c + psaci_c + psacw_c + psfw_c + psfi_c
                          + praci_c * delta3 + piacr_c * delta3 + psdep_c)
              + pgaut_c * gindex + pgacs_c * gdelta4 + pgacsp_c * g1sdelt4
              + pracs_c * (1.0 - delta2) - psacr_c * delta2)
    psnow_cold = -tmp_s2
    qsz_cold = jnp.maximum(0.0, qsz + dtb * psnow_cold)
    qschg_cold = psnow_cold
    tmp_g2 = (-pgaut_c - pgfr_c - pgacw_c * delta4 - pgaci_c * delta4
              - pgacr_c * delta4 - pgacs_c * delta4
              - pgwet_c * (1.0 - delta4) - pgsub_c - pgdep_c
              - psacr_c * (1.0 - delta2) - pracs_c * (1.0 - delta2)
              - praci_c * (1.0 - delta3) - piacr_c * (1.0 - delta3))
    pgraupel_cold = -tmp_g2 * gindex
    qgz_cold = jnp.maximum(0.0, qgz + dtb * pgraupel_cold) * gindex
    qgchg_cold = pgraupel_cold
    tmp_theiz_cold = C.OCP / tothz * C.XLF * (qschg_cold + qgchg_cold)
    theiz_cold = theiz + dtb * tmp_theiz_cold

    # ============================================================
    # WARM branch (temcc >= 0)
    # ============================================================
    praut_w = praut
    psacw_w = psacw
    pracw_w = pracw
    pgacw_w = pgacw
    tmp = praut_w + psacw_w + pracw_w + pgacw_w * gindex
    f = jnp.where(tmp > qlzodt, qlzodt / tmp, 1.0)
    praut_w = praut_w * f
    psacw_w = psacw_w * f
    pracw_w = pracw_w * f
    pgacw_w = pgacw_w * f * gindex
    # combined all snow processes (warm): tmp_s = -(psmlt+psmltevp)+pgacs*gindex
    psmlt_w = psmlt
    psmltevp_w = psmltevp
    pgacs_w = pgacs
    tmp_s = -(psmlt_w + psmltevp_w) + pgacs_w * gindex
    f = jnp.where(tmp_s > qszodt, qszodt / tmp_s, 1.0)
    psmlt_w = psmlt_w * f
    psmltevp_w = psmltevp_w * f
    pgacs_w = pgacs_w * f * gindex
    # combined all graupel processes (warm): tmp_g=-pgmlt-pgacs-pgmltevp
    pgmlt_w = pgmlt
    pgmltevp_w = pgmltevp
    tmp_g = -pgmlt_w - pgacs_w - pgmltevp_w
    f = jnp.where(tmp_g > qgzodt, qgzodt / tmp_g, 1.0)
    pgmltevp_w = pgmltevp_w * f
    pgmlt_w = pgmlt_w * f
    # combined all rain processes (warm)
    prevp_w = prevp
    tmp_r = (-prevp_w - (praut_w + pracw_w + psacw_w - psmlt_w)
             + pgmlt_w * gindex - pgacw_w * gindex)
    f = jnp.where(tmp_r > qrzodt, qrzodt / tmp_r, 1.0)
    prevp_w = prevp_w * f

    pvapor_warm = -psmltevp_w - prevp_w - pgmltevp_w * gindex
    qvz_warm = jnp.maximum(qvmin, qvz + dtb * pvapor_warm)
    pclw_warm = -praut_w - pracw_w - psacw_w - pgacw_w * gindex
    qlz_warm = jnp.maximum(0.0, qlz + dtb * pclw_warm)
    pcli_warm = jnp.zeros_like(qvz)
    qiz_warm = jnp.maximum(0.0, qiz + dtb * pcli_warm)
    tmp_r2w = (-prevp_w - (praut_w + pracw_w + psacw_w - psmlt_w)
               + pgmlt_w * gindex - pgacw_w * gindex)
    prain_warm = -tmp_r2w
    qrz_warm = jnp.maximum(0.0, qrz + dtb * prain_warm)
    tmp_s2w = -(psmlt_w + psmltevp_w) + pgacs_w * gindex
    psnow_warm = -tmp_s2w
    qsz_warm = jnp.maximum(0.0, qsz + dtb * psnow_warm)
    qschg_warm = psnow_warm
    tmp_g2w = -pgmlt_w - pgacs_w - pgmltevp_w
    pgraupel_warm = -tmp_g2w * gindex
    qgz_warm = jnp.maximum(0.0, qgz + dtb * pgraupel_warm) * gindex
    qgchg_warm = pgraupel_warm
    tmp_theiz_warm = C.OCP / tothz * C.XLF * (qschg_warm + qgchg_warm)
    theiz_warm = theiz + dtb * tmp_theiz_warm

    # ---- select branch ----
    qvz1 = jnp.where(cold, qvz_cold, qvz_warm)
    qlz1 = jnp.where(cold, qlz_cold, qlz_warm)
    qiz1 = jnp.where(cold, qiz_cold, qiz_warm)
    qrz1 = jnp.where(cold, qrz_cold, qrz_warm)
    qsz1 = jnp.where(cold, qsz_cold, qsz_warm)
    qgz1 = jnp.where(cold, qgz_cold, qgz_warm)
    theiz1 = jnp.where(cold, theiz_cold, theiz_warm)
    pclw1 = jnp.where(cold, pclw_cold, pclw_warm)
    pvapor1 = jnp.where(cold, pvapor_cold, pvapor_warm)
    pcli1 = jnp.where(cold, pcli_cold, pcli_warm)

    thz1 = theiz1 - (xlvocp * qvz1 - xlfocp * qiz1) / tothz
    tem1 = thz1 * tothz

    # ============================================================
    # saturation adjustment (common tail)
    # ============================================================
    # qvsbar lower bound (bloss)
    tmp_tem = jnp.where(qlz1 + qiz1 > 0.0,
                        tem1 - xlvocp * qlz1 - (xlvocp + xlfocp) * qiz1, tem1)
    tmp_temcc = tmp_tem - 273.15
    es_qb = jnp.where(tmp_temcc < 0.0, _es_ice(tmp_tem), _es_liquid(tmp_tem))
    qvsbar = _qfromes(es_qb, prez)
    rsat = 1.0  # WRF overrides rsat=1.0 unconditionally

    saturated = (qvz1 + qlz1 + qiz1) >= rsat * qvsbar

    # --- unsaturated path ---
    qvz_un = qvz1 + qlz1 + qiz1
    qlz_un = jnp.zeros_like(qvz)
    qiz_un = jnp.zeros_like(qvz)

    # --- saturated path: satadj ---
    qvz_sa, qlz_sa, qiz_sa = _satadj_inline(qvz1, qlz1, qiz1, prez, theiz1, tothz)

    qvz2 = jnp.where(saturated, qvz_sa, qvz_un)
    qlz2 = jnp.where(saturated, qlz_sa, qlz_un)
    qiz2 = jnp.where(saturated, qiz_sa, qiz_un)

    # ============================================================
    # melt/freeze of cloud ice and cloud water (only if qlz+qiz>0)
    # ============================================================
    thz2 = theiz1 - (xlvocp * qvz2 - xlfocp * qiz2) / tothz
    tem2 = thz2 * tothz
    temcc2 = tem2 - 273.15

    qlpqi = qlz2 + qiz2
    do_mf = qlpqi > 0.0

    pihom = jnp.where(temcc2 < -40.0, qlz2 * odtb, 0.0)
    pimlt = jnp.where(temcc2 > 0.0, qiz2 * odtb, 0.0)
    in_berg = (temcc2 < 0.0) & (temcc2 > -31.0)
    a1 = _parama(temcc2, _PA1)
    a2 = _parama(temcc2, _PA2)
    a1m = a1 * 0.001 ** (1.0 - a2)
    xnin = C.XNI0 * jnp.exp(-C.BNI * temcc2)
    # WRF: pidw = xnin*orho*(a1*xmnin**a2)
    pidw = jnp.where(in_berg, xnin * orho * (a1m * C.XMNIN ** a2), 0.0)

    qlz3 = jnp.where(do_mf, jnp.maximum(0.0, qlz2 + dtb * (-pihom + pimlt - pidw)), qlz2)
    qiz3 = jnp.where(do_mf, jnp.maximum(0.0, qiz2 + dtb * (pihom - pimlt + pidw)), qiz2)

    qvz4, qlz4, qiz4 = _satadj_inline(qvz2, qlz3, qiz3, prez, theiz1, tothz)
    qvz4 = jnp.where(do_mf, qvz4, qvz2)
    qlz4 = jnp.where(do_mf, qlz4, qlz2)
    qiz4 = jnp.where(do_mf, qiz4, qiz2)

    return (qvz4, qlz4, qiz4, qrz1, qsz1, qgz1, theiz1, pclw1, pvapor1)


# Inline satadj (same as microphysics_lin._satadj_cell) so this module is
# self-contained per cell (vmap applied at the column level by the caller).
def _satadj_inline(qvz, qlz, qiz, prez, theiz, tothz):
    xlvocp = C.XLVOCP
    xlfocp = C.XLFOCP
    tem = tothz * (theiz - (xlvocp * qvz - xlfocp * qiz) / tothz)
    tem_noliqice = tem - xlvocp * qlz - (xlvocp + xlfocp) * qiz
    es_liq = _es_liquid(tem_noliqice)
    qsat_nl_liq = _qfromes(es_liq, prez)
    qsat_nl_ice = (C.EPISP0K / prez
                   * jnp.exp(21.8745584 * (tem_noliqice - 273.15) / (tem_noliqice - 7.66)))
    qsat_noliqice = jnp.where(tem_noliqice > 273.15, qsat_nl_liq, qsat_nl_ice)
    qpz = qvz + qlz + qiz
    qlpqi = qlz + qiz
    big = qlpqi >= 1.0e-5
    denom = jnp.where(qlpqi > 0.0, qlpqi, 1.0)
    ratql_big = qlz / denom
    ratqi_big = qiz / denom
    t0 = 273.15
    t1 = 248.15
    tmp1 = jnp.clip((t0 - tem) / (t0 - t1), 0.0, 1.0)
    ratql = jnp.where(big, ratql_big, 1.0 - tmp1)
    ratqi = jnp.where(big, ratqi_big, tmp1)

    def newton_body(n, carry):
        tsat, absft, qvsbar = carry
        denom1 = 1.0 / (tsat - C.SVP3)
        denom2 = 1.0 / (tsat - 7.66)
        es1 = 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * denom1 * (tsat - C.SVPT0))
        qswz = _qfromes(es1, prez)
        es2 = 1000.0 * C.SVP1 * jnp.exp(21.8745584 * denom2 * (tsat - 273.15))
        qsiz_cold = _qfromes(es2, prez)
        qsiz = jnp.where(tem < 273.15, qsiz_cold, qswz)
        qswz = jnp.where(tem < 233.15, qsiz_cold, qswz)
        qvsbar_new = ratql * qswz + ratqi * qsiz
        converged = absft < 0.01
        dqvsbar = (ratql * qswz * C.SVP2 * 243.5 * denom1 * denom1
                   + ratqi * qsiz * 21.8745584 * 265.5 * denom2 * denom2)
        ftsat = (tsat + (xlvocp + ratqi * xlfocp) * qvsbar_new
                 - tothz * theiz - xlfocp * ratqi * (qvz + qlz + qiz))
        dftsat = 1.0 + (xlvocp + ratqi * xlfocp) * dqvsbar
        tsat_new = tsat - ftsat / dftsat
        absft_new = jnp.abs(ftsat)
        # WRF exits (go to 300) the FIRST time absft<0.01 with tsat/qvsbar at
        # their just-computed (converged) values. We freeze tsat/absft once
        # converged; qvsbar is recomputed from the (frozen) tsat each step so it
        # equals the converged value -- always adopt qvsbar_new, never the stale
        # carry (this is the per-cell condensation-amount fix).
        return (jnp.where(converged, tsat, tsat_new),
                jnp.where(converged, absft, absft_new),
                qvsbar_new)

    tsat_f, absft_f, qvsbar_f = jax.lax.fori_loop(
        0, 20, newton_body,
        (tem, jnp.ones_like(tem), jnp.zeros_like(tem)))
    sat = qpz > qvsbar_f
    qvz_B = jnp.where(sat, qvsbar_f, qpz)
    qiz_B = jnp.where(sat, ratqi * (qpz - qvsbar_f), 0.0)
    qlz_B = jnp.where(sat, ratql * (qpz - qvsbar_f), 0.0)
    sub = qpz < qsat_noliqice
    return (jnp.where(sub, qpz, qvz_B),
            jnp.where(sub, 0.0, qlz_B),
            jnp.where(sub, 0.0, qiz_B))
