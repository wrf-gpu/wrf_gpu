"""Sprint S2 oracle-parity harness for ``noahmp.soil_thermo``.

S0b WRF-savepoint fixtures are not yet present under ``proofs/noahmp/``; this
harness stands in by validating the JAX port against an INDEPENDENT NumPy
re-implementation transcribed 1:1 from pristine WRF
``/home/enric/src/wrf_pristine/WRF/phys/noahmp/src/module_sf_noahmplsm.F``:

  THERMOPROP (:2400-2510) + CSNOW (:2514-2569) + TDFCND (:2573-2680)
  TSNOSOI (:5258-5371) = HRT (:5375-5473) + HSTEP (:5477-5530) + ROSR12 (:5534-5591)
  PHASECHANGE (:5595-5810), opt_frz=1 / opt_stc=1 / opt_tbot=2

This is NOT a JAX-vs-JAX self-compare: the oracle is a separate scalar-loop NumPy
implementation that mirrors the Fortran array indexing (-NSNOW+1 .. NSOIL) exactly.
The JAX port is run on the layer-axis-0 ``(NLAY, ny, nx)`` tile and compared per
field. A realistic snow-free Canary soil column (the operational d02/d03 land case,
ISNOW=0) and a 2-snow-layer column are both exercised.

Run (CPU, to respect one-job-at-a-time GPU discipline):
  JAX_PLATFORM_NAME=cpu python proofs/noahmp/soil_thermo_oracle_parity.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# CPU-only + ≤4 cores (CLAUDE core budget; avoid contending for the busy GPU).
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import jax  # noqa: E402
from jax import config  # noqa: E402

config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics.noahmp import soil_thermo as st  # noqa: E402

# ---------------------------------------------------------------------------
# WRF constants (module_sf_noahmplsm.F:204-220)
# ---------------------------------------------------------------------------
GRAV = 9.80616
TFRZ = 273.16
HFUS = 0.3336e06
CWAT = 4.188e06
CICE = 2.094e06
CPAIR = 1004.64
TKWAT = 0.6
TKICE = 2.2
DENH2O = 1000.0
DENICE = 917.0
ZBOT = 8.0
NSOIL = 4
NSNOW = 3
NLAY = NSNOW + NSOIL


# ===========================================================================
# INDEPENDENT NumPy ORACLE (scalar loops, WRF -NSNOW+1..NSOIL indexing)
# Fortran index K in [-2..4] maps to python index K+NSNOW in [1..6]; we use a
# dict keyed by the Fortran index to keep the transcription literal.
# ===========================================================================
def _tdfcnd_ref(smc, sh2o, smcmax, quartz):
    thkw, thko, thkqtz = 0.57, 2.0, 7.7
    satratio = smc / smcmax
    thks = (thkqtz ** quartz) * (thko ** (1.0 - quartz))
    xunfroz = 1.0 if smc <= 0.0 else sh2o / smc
    xu = xunfroz * smcmax
    thksat = thks ** (1.0 - smcmax) * TKICE ** (smcmax - xu) * thkw ** xu
    gammd = (1.0 - smcmax) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    if (sh2o + 0.0005) < smc:
        ake = satratio
    else:
        ake = (np.log10(satratio) + 1.0) if satratio > 0.1 else 0.0
    return ake * (thksat - thkdry) + thkdry


def thermoprop_ref(isnow, dzsnso, snice, snliq, smc, sh2o, snowh, smcmax, quartz, csoil):
    """THERMOPROP land branch. dict-indexed by Fortran layer index (-2..4)."""
    df = {k: 0.0 for k in range(-NSNOW + 1, NSOIL + 1)}
    hcpct = {k: 0.0 for k in range(-NSNOW + 1, NSOIL + 1)}
    # CSNOW for active snow layers ISNOW+1..0
    for iz in range(isnow + 1, 1):
        dz = dzsnso[iz]
        snicev = min(1.0, snice[iz] / (dz * DENICE))
        epore = 1.0 - snicev
        snliqv = min(epore, snliq[iz] / (dz * DENH2O))
        bdsnoi = (snice[iz] + snliq[iz]) / dz
        cvsno = CICE * snicev + CWAT * snliqv
        tksno = 3.2217e-6 * bdsnoi ** 2.0
        df[iz] = tksno
        hcpct[iz] = cvsno
    # soil layers 1..NSOIL
    for iz in range(1, NSOIL + 1):
        sice = smc[iz] - sh2o[iz]
        hcpct[iz] = (sh2o[iz] * CWAT + (1.0 - smcmax[iz]) * csoil
                     + (smcmax[iz] - smc[iz]) * CPAIR + sice * CICE)
        df[iz] = _tdfcnd_ref(smc[iz], sh2o[iz], smcmax[iz], quartz[iz])
    # snow/soil interface DF blend (:2503-2507)
    if isnow == 0:
        df[1] = (df[1] * dzsnso[1] + 0.35 * snowh) / (snowh + dzsnso[1])
    else:
        df[1] = (df[1] * dzsnso[1] + df[0] * dzsnso[0]) / (dzsnso[0] + dzsnso[1])
    return df, hcpct


def hrt_ref(isnow, zsnso, stc, tbot, zbot, df, hcpct, ssoil):
    """HRT (:5375-5473), opt_stc=1 / opt_tbot=2, PHI=0."""
    ai = {}; bi = {}; ci = {}; rhsts = {}
    ddz = {}; dtsdz = {}; denom = {}; eflux = {}
    for k in range(isnow + 1, NSOIL + 1):
        if k == isnow + 1:
            denom[k] = -zsnso[k] * hcpct[k]
            temp1 = -zsnso[k + 1]
            ddz[k] = 2.0 / temp1
            dtsdz[k] = 2.0 * (stc[k] - stc[k + 1]) / temp1
            eflux[k] = df[k] * dtsdz[k] - ssoil
        elif k < NSOIL:
            denom[k] = (zsnso[k - 1] - zsnso[k]) * hcpct[k]
            temp1 = zsnso[k - 1] - zsnso[k + 1]
            ddz[k] = 2.0 / temp1
            dtsdz[k] = 2.0 * (stc[k] - stc[k + 1]) / temp1
            eflux[k] = df[k] * dtsdz[k] - df[k - 1] * dtsdz[k - 1]
        else:  # k == NSOIL, opt_tbot=2
            denom[k] = (zsnso[k - 1] - zsnso[k]) * hcpct[k]
            dtsdz[k] = (stc[k] - tbot) / (0.5 * (zsnso[k - 1] + zsnso[k]) - zbot)
            botflx = -df[k] * dtsdz[k]
            eflux[k] = (-botflx - df[k - 1] * dtsdz[k - 1])
    for k in range(isnow + 1, NSOIL + 1):
        if k == isnow + 1:
            ai[k] = 0.0
            ci[k] = -df[k] * ddz[k] / denom[k]
            bi[k] = -ci[k]  # opt_stc=1
        elif k < NSOIL:
            ai[k] = -df[k - 1] * ddz[k - 1] / denom[k]
            ci[k] = -df[k] * ddz[k] / denom[k]
            bi[k] = -(ai[k] + ci[k])
        else:
            ai[k] = -df[k - 1] * ddz[k - 1] / denom[k]
            ci[k] = 0.0
            bi[k] = -(ai[k] + ci[k])
        rhsts[k] = eflux[k] / (-denom[k])
    return ai, bi, ci, rhsts


def rosr12_ref(a, b, c, d, ntop, nsoil):
    """ROSR12 (:5534-5591)."""
    p = {}; delta = {}
    c = dict(c)
    c[nsoil] = 0.0
    p[ntop] = -c[ntop] / b[ntop]
    delta[ntop] = d[ntop] / b[ntop]
    for k in range(ntop + 1, nsoil + 1):
        p[k] = -c[k] * (1.0 / (b[k] + a[k] * p[k - 1]))
        delta[k] = (d[k] - a[k] * delta[k - 1]) * (1.0 / (b[k] + a[k] * p[k - 1]))
    p[nsoil] = delta[nsoil]
    for k in range(ntop + 1, nsoil + 1):
        kk = nsoil - k + (ntop - 1) + 1
        p[kk] = p[kk] * p[kk + 1] + delta[kk]
    return p


def tsnosoi_ref(isnow, zsnso, ssoil, df, hcpct, snowh, tbot, stc, dt):
    """TSNOSOI (:5258-5371) opt_stc=1 / opt_tbot=2."""
    zbotsno = ZBOT - snowh
    ai, bi, ci, rhsts = hrt_ref(isnow, zsnso, stc, tbot, zbotsno, df, hcpct, ssoil)
    # HSTEP scaling
    for k in range(isnow + 1, NSOIL + 1):
        rhsts[k] *= dt
        ai[k] *= dt
        bi[k] = 1.0 + bi[k] * dt
        ci[k] *= dt
    p = rosr12_ref(ai, bi, ci, rhsts, isnow + 1, NSOIL)
    stc_new = dict(stc)
    for k in range(isnow + 1, NSOIL + 1):
        stc_new[k] = stc[k] + p[k]
    return stc_new


def phasechange_ref(isnow, dt, fact, dzsnso, hcpct, stc, snice, snliq, sneqv,
                    snowh, smc, sh2o, smcmax, psisat, bexp):
    """PHASECHANGE (:5595-5810), opt_frz=1 land (IST=1)."""
    qmelt = 0.0; ponding = 0.0; xmf = 0.0
    supercool = {j: 0.0 for j in range(-NSNOW + 1, NSOIL + 1)}
    mice = {}; mliq = {}
    for j in range(isnow + 1, 1):
        mice[j] = snice[j]; mliq[j] = snliq[j]
    for j in range(1, NSOIL + 1):
        mliq[j] = sh2o[j] * dzsnso[j] * 1000.0
        mice[j] = (smc[j] - sh2o[j]) * dzsnso[j] * 1000.0
    imelt = {}; hm = {}; xm = {}; wice0 = {}; wliq0 = {}; wmass0 = {}
    for j in range(isnow + 1, NSOIL + 1):
        imelt[j] = 0; hm[j] = 0.0; xm[j] = 0.0
        wice0[j] = mice[j]; wliq0[j] = mliq[j]; wmass0[j] = mice[j] + mliq[j]
    # opt_frz=1 supercool
    for j in range(1, NSOIL + 1):
        if stc[j] < TFRZ:
            smp = HFUS * (TFRZ - stc[j]) / (GRAV * stc[j])
            supercool[j] = smcmax[j] * (smp / psisat[j]) ** (-1.0 / bexp[j])
            supercool[j] = supercool[j] * dzsnso[j] * 1000.0
    for j in range(isnow + 1, NSOIL + 1):
        if mice[j] > 0.0 and stc[j] >= TFRZ:
            imelt[j] = 1
        if mliq[j] > supercool[j] and stc[j] < TFRZ:
            imelt[j] = 2
        if isnow == 0 and sneqv > 0.0 and j == 1:
            if stc[j] >= TFRZ:
                imelt[j] = 1
    for j in range(isnow + 1, NSOIL + 1):
        if imelt[j] > 0:
            hm[j] = (stc[j] - TFRZ) / fact[j]
            stc[j] = TFRZ
        if imelt[j] == 1 and hm[j] < 0.0:
            hm[j] = 0.0; imelt[j] = 0
        if imelt[j] == 2 and hm[j] > 0.0:
            hm[j] = 0.0; imelt[j] = 0
        xm[j] = hm[j] * dt / HFUS
    # no-layer snow melt (:5736-5753)
    if isnow == 0 and sneqv > 0.0 and xm[1] > 0.0:
        temp1 = sneqv
        sneqv = max(0.0, temp1 - xm[1])
        propor = sneqv / temp1
        snowh = max(0.0, propor * snowh)
        snowh = min(max(snowh, sneqv / 500.0), sneqv / 50.0)
        heatr = hm[1] - HFUS * (temp1 - sneqv) / dt
        if heatr > 0.0:
            xm[1] = heatr * dt / HFUS; hm[1] = heatr
        else:
            xm[1] = 0.0; hm[1] = 0.0
        qmelt = max(0.0, (temp1 - sneqv)) / dt
        xmf = HFUS * qmelt
        ponding = temp1 - sneqv
    for j in range(isnow + 1, NSOIL + 1):
        if imelt[j] > 0 and abs(hm[j]) > 0.0:
            heatr = 0.0
            if xm[j] > 0.0:
                mice[j] = max(0.0, wice0[j] - xm[j])
                heatr = hm[j] - HFUS * (wice0[j] - mice[j]) / dt
            elif xm[j] < 0.0:
                if j <= 0:
                    mice[j] = min(wmass0[j], wice0[j] - xm[j])
                else:
                    if wmass0[j] < supercool[j]:
                        mice[j] = 0.0
                    else:
                        mice[j] = min(wmass0[j] - supercool[j], wice0[j] - xm[j])
                        mice[j] = max(mice[j], 0.0)
                heatr = hm[j] - HFUS * (wice0[j] - mice[j]) / dt
            mliq[j] = max(0.0, wmass0[j] - mice[j])
            if abs(heatr) > 0.0:
                stc[j] = stc[j] + fact[j] * heatr
                if j <= 0:
                    if mliq[j] * mice[j] > 0.0:
                        stc[j] = TFRZ
                    if mice[j] == 0.0:
                        stc[j] = TFRZ
                        hm[j + 1] = hm[j + 1] + heatr
                        xm[j + 1] = hm[j + 1] * dt / HFUS
            xmf += HFUS * (wice0[j] - mice[j]) / dt
            if j < 1:
                qmelt += max(0.0, (wice0[j] - mice[j])) / dt
    for j in range(isnow + 1, 1):
        snliq[j] = mliq[j]; snice[j] = mice[j]
    for j in range(1, NSOIL + 1):
        sh2o[j] = mliq[j] / (1000.0 * dzsnso[j])
        smc[j] = (mliq[j] + mice[j]) / (1000.0 * dzsnso[j])
    return stc, snice, snliq, smc, sh2o, sneqv, snowh, qmelt, imelt, ponding


# ===========================================================================
# Column fixtures -> dict (Fortran index) and JAX tile (layer-axis-0)
# ===========================================================================
def make_dicts(isnow, zsnso_soil, dzsnso_all, stc_all, snice_a, snliq_a,
               smc_s, sh2o_s):
    """Build Fortran-index dicts from per-layer python lists.

    zsnso_soil: soil interface depths (4) negative; snow interfaces derived from dz.
    dzsnso_all: thickness for layers -2..4 (len 7), only active used.
    stc_all   : temperatures -2..4 (len 7).
    """
    # ZSNSO (zsnso_d) is rebuilt by the caller from dz; here just map the lists to
    # Fortran-index dicts. dzsnso_all is axis-0 order (snow[-2..0] then soil[1..4]).
    zsnso = None
    # axis-0 index = Fortran_k + NSNOW - 1  (k=-2 -> axis 0, k=1 -> axis NSNOW=3).
    dz = {k: dzsnso_all[k + NSNOW - 1] for k in range(-NSNOW + 1, NSOIL + 1)}
    stc = {k: stc_all[k + NSNOW - 1] for k in range(-NSNOW + 1, NSOIL + 1)}
    snice = {k: snice_a[k + NSNOW - 1] for k in range(-NSNOW + 1, 1)}
    snliq = {k: snliq_a[k + NSNOW - 1] for k in range(-NSNOW + 1, 1)}
    smc = {k: smc_s[k - 1] for k in range(1, NSOIL + 1)}
    sh2o = {k: sh2o_s[k - 1] for k in range(1, NSOIL + 1)}
    return zsnso, dz, stc, snice, snliq, smc, sh2o


def col_to_tile(values_by_layer):
    """list len NLAY (axis0 order: snow[-2..0] then soil[1..4]) -> (NLAY,1,1)."""
    return jnp.asarray(np.array(values_by_layer, dtype=np.float64)).reshape(NLAY, 1, 1)


def _soil_dict(lst):
    """list (0-indexed, len NSOIL) -> dict keyed by Fortran soil index 1..NSOIL."""
    return {k: lst[k - 1] for k in range(1, NSOIL + 1)}


def run_case(name, isnow, dz_all, stc_all, snice_a, snliq_a, smc_s, sh2o_s,
             ssoil, tbot, snowh, sneqv, smcmax, quartz, csoil, psisat, bexp,
             dt, results):
    zsnso, dzd, stcd, snid, snld, smcd, sh2od = make_dicts(
        isnow, None, dz_all, stc_all, snice_a, snliq_a, smc_s, sh2o_s)

    smcmax_d = _soil_dict(smcmax)
    quartz_d = _soil_dict(quartz)
    psisat_d = _soil_dict(psisat)
    bexp_d = _soil_dict(bexp)

    # ---- ORACLE ----
    df_o, hc_o = thermoprop_ref(isnow, dzd, snid, snld, smcd, sh2od, snowh,
                                smcmax_d, quartz_d, csoil)
    # build zsnso dict from dz (cumulative, snow surface = 0)
    zsnso_d = {}
    acc = 0.0
    for k in range(isnow + 1, NSOIL + 1):
        acc -= dzd[k]
        zsnso_d[k] = acc
    stc_in = dict(stcd)
    stc_after_solve = tsnosoi_ref(isnow, zsnso_d, ssoil, df_o, hc_o, snowh,
                                  tbot, dict(stc_in), dt)
    fact_o = {k: dt / (hc_o[k] * dzd[k]) for k in range(isnow + 1, NSOIL + 1)}
    (stc_pc_o, snice_o, snliq_o, smc_o, sh2o_o, sneqv_o, snowh_o,
     qmelt_o, imelt_o, pond_o) = phasechange_ref(
        isnow, dt, fact_o, dzd, hc_o, dict(stc_after_solve),
        dict(snid), dict(snld), sneqv, snowh, dict(smcd), dict(sh2od),
        smcmax_d, psisat_d, bexp_d)

    # ---- JAX PORT ----
    class P:
        pass
    p = P()
    p.smcmax = jnp.asarray(np.array(smcmax, dtype=np.float64)).reshape(NSOIL, 1, 1)
    p.quartz = jnp.asarray(np.array(quartz, dtype=np.float64)).reshape(NSOIL, 1, 1)
    p.csoil = jnp.float64(csoil)

    class LS:
        pass
    ls = LS()
    ls.smois = jnp.asarray(np.array(smc_s, dtype=np.float64)).reshape(NSOIL, 1, 1)
    ls.sh2o = jnp.asarray(np.array(sh2o_s, dtype=np.float64)).reshape(NSOIL, 1, 1)
    ls.snice = jnp.asarray(np.array(snice_a[:NSNOW], dtype=np.float64)).reshape(NSNOW, 1, 1)
    ls.snliq = jnp.asarray(np.array(snliq_a[:NSNOW], dtype=np.float64)).reshape(NSNOW, 1, 1)
    ls.snowh = jnp.float64(snowh).reshape(1, 1)
    ls.isnow = jnp.int32(isnow).reshape(1, 1)
    # zsnso tile (NLAY,1,1) — cumulative from snow surface, inactive snow above filled
    zsnso_tile = build_zsnso_tile(isnow, dz_all)
    ls.zsnso = zsnso_tile

    class STAT:
        pass
    stat = STAT()
    stat.parameters = p

    df_j, hc_j = st.noahmp_thermoprop(ls, stat, jnp.float64(0.0).reshape(1, 1))
    dzsnso_tile = st._dzsnso_from_zsnso(zsnso_tile)
    stc_tile = col_to_tile(stc_all)
    stc_solve_j = st.noahmp_soil_thermo(
        stc_tile, df_j, hc_j,
        jnp.float64(ssoil).reshape(1, 1), jnp.float64(tbot).reshape(1, 1),
        zsnso_tile, dzsnso_tile, jnp.int32(isnow).reshape(1, 1), float(dt))

    (stc_pc_j, snice_j, snliq_j, smc_j, sh2o_j, sneqv_j, snowh_j,
     qmelt_j, imelt_j, pond_j) = st.noahmp_phasechange(
        stc_solve_j, ls.snice, ls.snliq, ls.smois, ls.sh2o,
        jnp.float64(sneqv).reshape(1, 1), jnp.float64(snowh).reshape(1, 1),
        hc_j, dzsnso_tile, jnp.int32(isnow).reshape(1, 1), float(dt),
        smcmax=p.smcmax,
        psisat=jnp.asarray(np.array(psisat, dtype=np.float64)).reshape(NSOIL, 1, 1),
        bexp=jnp.asarray(np.array(bexp, dtype=np.float64)).reshape(NSOIL, 1, 1))

    # ---- COMPARE (only active layers) ----
    rec = {"case": name, "isnow": isnow}

    # DF / HCPCT (active layers)
    df_o_vec = np.array([df_o[k] for k in range(isnow + 1, NSOIL + 1)])
    df_j_vec = np.array(df_j[NSNOW + isnow:, 0, 0])
    hc_o_vec = np.array([hc_o[k] for k in range(isnow + 1, NSOIL + 1)])
    hc_j_vec = np.array(hc_j[NSNOW + isnow:, 0, 0])
    rec["df_max_abs_err"] = float(np.max(np.abs(df_o_vec - df_j_vec)))
    rec["hcpct_max_rel_err"] = float(np.max(np.abs(hc_o_vec - hc_j_vec) / np.abs(hc_o_vec)))

    # STC after solve
    stc_o_vec = np.array([stc_after_solve[k] for k in range(isnow + 1, NSOIL + 1)])
    stc_j_vec = np.array(stc_solve_j[NSNOW + isnow:, 0, 0])
    rec["stc_solve_max_abs_err_K"] = float(np.max(np.abs(stc_o_vec - stc_j_vec)))
    rec["stc_solve_oracle"] = stc_o_vec.tolist()
    rec["stc_solve_jax"] = stc_j_vec.tolist()

    # GRDFLX (= SSOIL top tendency proxy: report the implied bottom flux EFLXB-like
    # check via energy: dStC consistency). We compare STC evolution which embeds it.

    # PHASECHANGE STC
    stc_pc_o_vec = np.array([stc_pc_o[k] for k in range(isnow + 1, NSOIL + 1)])
    stc_pc_j_vec = np.array(stc_pc_j[NSNOW + isnow:, 0, 0])
    rec["stc_phasechange_max_abs_err_K"] = float(np.max(np.abs(stc_pc_o_vec - stc_pc_j_vec)))

    # soil moisture after phase change
    smc_o_vec = np.array([smc_o[k] for k in range(1, NSOIL + 1)])
    smc_j_vec = np.array(smc_j[:, 0, 0])
    sh2o_o_vec = np.array([sh2o_o[k] for k in range(1, NSOIL + 1)])
    sh2o_j_vec = np.array(sh2o_j[:, 0, 0])
    rec["smc_max_abs_err"] = float(np.max(np.abs(smc_o_vec - smc_j_vec)))
    rec["sh2o_max_abs_err"] = float(np.max(np.abs(sh2o_o_vec - sh2o_j_vec)))
    rec["qmelt_oracle"] = float(qmelt_o)
    rec["qmelt_jax"] = float(qmelt_j[0, 0])
    rec["qmelt_abs_err"] = float(abs(qmelt_o - float(qmelt_j[0, 0])))

    # Energy-balance diagnostic for the opt_stc=1 semi-implicit solve (TSNOSOI:5350-5356):
    #   sum (STC_new - STC_old) * DZ * HCPCT / DT  ==  SSOIL + EFLXB
    # WRF itself SKIPS this check at small DT (TSNOSOI:5344-5346 early return) and only
    # WARNS when |ERR_EST| > 1.0 W/m2 (:5362). A small nonzero residual is therefore an
    # intrinsic property of the semi-implicit discretisation, NOT a port defect. We
    # report it AND verify the JAX storage term reproduces the oracle storage term
    # bit-for-bit (the load-bearing parity check).
    zsnso_d2 = {}
    acc = 0.0
    for k in range(isnow + 1, NSOIL + 1):
        acc -= dzd[k]
        zsnso_d2[k] = acc
    zbotsno = ZBOT - snowh
    eflxb = df_o[NSOIL] * (tbot - stc_after_solve[NSOIL]) / (
        0.5 * (zsnso_d2[NSOIL - 1] + zsnso_d2[NSOIL]) - zbotsno)
    storage_o = sum((stc_after_solve[k] - stc_in[k]) * dzd[k] * hc_o[k] / dt
                    for k in range(isnow + 1, NSOIL + 1))
    # JAX storage term, from the JAX-solved STC and JAX HCPCT.
    stc_in_vec = np.array([stc_in[k] for k in range(isnow + 1, NSOIL + 1)])
    dz_vec = np.array([dzd[k] for k in range(isnow + 1, NSOIL + 1)])
    storage_j = float(np.sum((stc_j_vec - stc_in_vec) * dz_vec * hc_j_vec / dt))
    rec["energy_residual_Wm2"] = float(storage_o - (ssoil + eflxb))   # intrinsic; |.|<1
    rec["energy_residual_abs_lt_1Wm2"] = bool(abs(rec["energy_residual_Wm2"]) < 1.0)
    rec["storage_oracle_vs_jax_abs_err_Wm2"] = float(abs(storage_o - storage_j))

    results.append(rec)
    return rec


def build_zsnso_tile(isnow, dz_all):
    """Build a (NLAY,1,1) ZSNSO tile from dz_all (len NLAY, snow..soil).

    Active layers ISNOW+1..NSOIL accumulate from the snow surface (=0). Inactive
    snow layers above the active top are set to 0 (WRF leaves them unused). The
    JAX path's _dzsnso_from_zsnso reproduces dz for active layers from these.
    """
    zsnso = np.zeros(NLAY, dtype=np.float64)
    acc = 0.0
    # active top axis index
    top_axis = NSNOW + isnow
    for axis in range(top_axis, NLAY):
        acc -= dz_all[axis]
        zsnso[axis] = acc
    # inactive snow layers: keep their own zero-level so dz>0 not required (masked)
    return jnp.asarray(zsnso).reshape(NLAY, 1, 1)


# ===========================================================================
# Cases
# ===========================================================================
def main():
    results = []
    # Canary loam-ish soil parameters (SOILPARM cat ~ 6 'loam'); ZSOIL 0.1/0.3/0.6/1.0
    smcmax = [0.434, 0.434, 0.434, 0.434]
    quartz = [0.25, 0.25, 0.25, 0.25]
    psisat = [0.3548, 0.3548, 0.3548, 0.3548]   # |satpsi| (m)
    bexp = [5.25, 5.25, 5.25, 5.25]
    csoil = 2.00e06
    dt = 1800.0  # 30-min physics step

    # soil layer thicknesses (Noah 4-layer): 0.10, 0.30, 0.60, 1.00 m
    dz_soil = [0.10, 0.30, 0.60, 1.00]

    # ---- CASE 1: snow-free Canary daytime column (the operational d02/d03 case) ----
    isnow = 0
    dz_all = [0.0, 0.0, 0.0] + dz_soil           # snow dz unused
    stc_all = [0.0, 0.0, 0.0, 295.0, 293.0, 290.0, 288.0]  # warm surface, cooler depth
    snice_a = [0.0, 0.0, 0.0]
    snliq_a = [0.0, 0.0, 0.0]
    smc_s = [0.30, 0.30, 0.30, 0.30]
    sh2o_s = [0.30, 0.30, 0.30, 0.30]
    ssoil = 60.0      # downward ground heat flux (W/m2), daytime
    tbot = 287.0
    run_case("canary_snowfree_daytime", isnow, dz_all, stc_all, snice_a, snliq_a,
             smc_s, sh2o_s, ssoil, tbot, snowh=0.0, sneqv=0.0,
             smcmax=smcmax, quartz=quartz, csoil=csoil, psisat=psisat, bexp=bexp,
             dt=dt, results=results)

    # ---- CASE 2: snow-free nocturnal cooling (negative ssoil) ----
    stc_all2 = [0.0, 0.0, 0.0, 283.0, 285.0, 287.0, 288.0]
    run_case("canary_snowfree_night", 0, dz_all, stc_all2, snice_a, snliq_a,
             smc_s, sh2o_s, -40.0, 289.0, snowh=0.0, sneqv=0.0,
             smcmax=smcmax, quartz=quartz, csoil=csoil, psisat=psisat, bexp=bexp,
             dt=dt, results=results)

    # ---- CASE 3: frozen surface soil -> phase change (refreeze + supercool) ----
    stc_all3 = [0.0, 0.0, 0.0, 271.0, 272.5, 274.0, 276.0]   # top 2 layers below TFRZ
    smc_s3 = [0.35, 0.35, 0.30, 0.30]
    sh2o_s3 = [0.35, 0.35, 0.30, 0.30]   # all liquid -> refreeze candidate
    run_case("frozen_surface_refreeze", 0, dz_all, stc_all3, snice_a, snliq_a,
             smc_s3, sh2o_s3, -50.0, 277.0, snowh=0.0, sneqv=0.0,
             smcmax=smcmax, quartz=quartz, csoil=csoil, psisat=psisat, bexp=bexp,
             dt=dt, results=results)

    # ---- CASE 4: 2 active snow layers (isnow=-2) + cold soil, melt at surface ----
    isnow4 = -2
    dz4 = [0.0, 0.05, 0.10] + dz_soil   # 2 active snow layers (axis 1,2)
    stc4 = [0.0, 274.0, 273.5, 272.0, 272.5, 274.0, 276.0]  # snow above TFRZ -> melt
    snice4 = [0.0, 8.0, 15.0]
    snliq4 = [0.0, 1.0, 2.0]
    snowh4 = 0.15
    sneqv4 = 26.0
    run_case("two_snow_layers_melt", isnow4, dz4, stc4, snice4, snliq4,
             smc_s, sh2o_s, 30.0, 277.0, snowh=snowh4, sneqv=sneqv4,
             smcmax=smcmax, quartz=quartz, csoil=csoil, psisat=psisat, bexp=bexp,
             dt=dt, results=results)

    # ---- verdict ----
    # Parity tolerances are bit-near (fp64): the JAX port must reproduce the
    # independent NumPy WRF transcription to round-off on every PHYSICAL field.
    # The semi-implicit energy residual is NOT a parity field (intrinsic to the
    # scheme; WRF itself skips it and only warns above 1.0 W/m2) — instead we
    # gate that the JAX storage term matches the oracle storage term bit-for-bit,
    # and (informationally) that the intrinsic residual stays under WRF's 1.0 W/m2.
    TOL = {
        "df_max_abs_err": 1e-10,
        "hcpct_max_rel_err": 1e-12,
        "stc_solve_max_abs_err_K": 1e-9,
        "stc_phasechange_max_abs_err_K": 1e-9,
        "smc_max_abs_err": 1e-12,
        "sh2o_max_abs_err": 1e-12,
        "qmelt_abs_err": 1e-12,
        "storage_oracle_vs_jax_abs_err_Wm2": 1e-6,
    }
    all_pass = True
    for rec in results:
        rec["pass"] = {}
        for k, tol in TOL.items():
            ok = abs(rec[k]) <= tol
            rec["pass"][k] = bool(ok)
            all_pass = all_pass and ok
        # WRF's own semi-implicit balance threshold (warn-only in WRF, :5362).
        rec["pass"]["energy_residual_abs_lt_1Wm2"] = rec["energy_residual_abs_lt_1Wm2"]
        all_pass = all_pass and rec["energy_residual_abs_lt_1Wm2"]

    out = {
        "harness": "soil_thermo_oracle_parity",
        "oracle": "independent NumPy transcription of WRF module_sf_noahmplsm.F "
                  "(THERMOPROP/CSNOW/TDFCND/TSNOSOI/HRT/HSTEP/ROSR12/PHASECHANGE)",
        "scope": "opt_stc=1 semi-implicit, opt_tbot=2 Noah deep BC, opt_frz=1 NY06",
        "nsoil": NSOIL, "nsnow": NSNOW, "dt_s": dt,
        "tolerances": TOL,
        "cases": results,
        "S0b_savepoint_fixtures_present": os.path.isdir(
            os.path.join(REPO, "proofs", "noahmp", "fixtures")),
        "verdict": "PASS" if all_pass else "FAIL",
    }
    path = os.path.join(REPO, "proofs", "noahmp", "soil_thermo_parity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", out["verdict"], "->", path)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
