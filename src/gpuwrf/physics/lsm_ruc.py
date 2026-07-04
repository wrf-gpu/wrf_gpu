"""WRF RUC multi-layer soil land-surface model (``sf_surface_physics=3``).

This module carries a traceable JAX/fp64 port of the no-snow land-column path
exercised by the staged WRF oracle:

``LSMRUC -> SOILVEGIN -> SFCTMP(no snow) -> SOIL -> SOILTEMP/SOILMOIST``.

The oracle is a compiled, unmodified WRF Fortran driver at
``proofs/v017/oracle/ruclsm`` and the committed savepoint is
``proofs/v017/savepoints/ruclsm/fp64/ruclsm_fp64.json``.  The operational scheme
registry still marks RUC as reference-only/fail-closed until the integration
owner wires the land carry into the scan path.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

RUC_ORACLE_DIR = "proofs/v017/oracle/ruclsm"
RUC_SAVEPOINT = "proofs/v017/savepoints/ruclsm/fp64/ruclsm_fp64.json"

RUC_NUM_SOIL_LAYERS = 6

_CP = 1004.5
_ROVCP = 0.2857
_G = 9.81
_LV = 2.5e6
_STBOLT = 5.67051e-8
_RHOWATER = 1000.0
_P1000MB = 100000.0
_PI = 3.141592653589793
_R_V = 461.525

_ZS = jnp.asarray([0.0, 0.05, 0.20, 0.40, 1.00, 2.00], dtype=jnp.float64)
_ZSHALF = jnp.asarray([0.0, 0.025, 0.125, 0.30, 0.70, 1.50], dtype=jnp.float64)

# VEGPARM.TBL, USGS-RUC block.  Only the classes exercised by the oracle are
# needed here; unsupported classes fail closed through _veg_params.
_VEG = {
    # ivgtyp: (albedo, z0, emissivity, pc, shdfac, iforest, rs, rgl, hs, snup, lai, maxalb)
    2: (0.17, 0.20, 0.92, 0.30, 0.80, 7, 40.0, 100.0, 36.25, 0.04, 5.68, 64.0),
    7: (0.19, 0.075, 0.92, 0.40, 0.80, 5, 40.0, 100.0, 36.35, 0.04, 2.90, 64.0),
    8: (0.22, 0.10, 0.88, 0.40, 0.70, 4, 300.0, 100.0, 42.00, 0.03, 3.66, 69.0),
}

# SOILPARM.TBL, STAS-RUC block.  Only sandy-loam (3) and loam (6) are exercised.
_SOIL = {
    # isltyp: (bb, drysmc, hc, maxsmc, refsmc, satpsi, satdk, satdw, wltsmc, qtz)
    3: (4.90, 0.041, 1.34, 0.435, 0.249, 0.218, 3.47e-5, 0.805e-5, 0.095, 0.60),
    6: (5.39, 0.050, 1.21, 0.451, 0.314, 0.478, 6.95e-6, 0.143e-4, 0.137, 0.40),
}

_CFACTR_DATA = 0.5
_RSMAX_DATA = 5000.0


class RucLandState(NamedTuple):
    """RUC multi-layer soil/snow land carry.

    The current column port validates the warm no-snow land path in the staged
    WRF oracle.  Snow carry fields are preserved in the interface for the later
    operational hook.
    """

    soilt: "object"
    tso: "object"
    soilmois: "object"
    sh2o: "object"
    smfr3d: "object"
    keepfr3dflag: "object"
    snow: "object"
    snowh: "object"
    qsfc: "object"

    def replace(self, **updates) -> "RucLandState":
        return self._replace(**updates)


def _as_f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _soil_depth_coefficients(dt: float):
    dtdzs = jnp.zeros((2 * (RUC_NUM_SOIL_LAYERS - 2),), dtype=jnp.float64)
    dtdzs2 = jnp.zeros((RUC_NUM_SOIL_LAYERS,), dtype=jnp.float64)
    for k_fortran in range(2, RUC_NUM_SOIL_LAYERS):
        k1 = 2 * k_fortran - 3
        x = dt / 2.0 / (_ZSHALF[k_fortran] - _ZSHALF[k_fortran - 1])
        dtdzs = dtdzs.at[k1 - 1].set(x / (_ZS[k_fortran - 1] - _ZS[k_fortran - 2]))
        dtdzs2 = dtdzs2.at[k_fortran - 2].set(x)
        dtdzs = dtdzs.at[k1].set(x / (_ZS[k_fortran] - _ZS[k_fortran - 1]))
    return dtdzs, dtdzs2


def _qsn_table():
    t = 173.15 + 0.05 * jnp.arange(5001, dtype=jnp.float64)
    evs = jnp.exp(17.67 * (t - 273.15) / (t - 29.65))
    eis = jnp.exp(22.514 - 6.15e3 / t)
    return 6.1153 * 0.62198 * jnp.where(t >= 273.15, evs, eis)


_TBQ = _qsn_table()


def _qsn(tn, tbq=_TBQ):
    r = (tn - 173.15) / 0.05 + 1.0
    idx_fortran = jnp.trunc(r).astype(jnp.int32)
    idx_fortran = jnp.clip(idx_fortran, 1, 5000)
    r1 = jnp.take(tbq, idx_fortran - 1)
    r2 = r - idx_fortran.astype(jnp.float64)
    return (jnp.take(tbq, idx_fortran) - r1) * r2 + r1


def _vilka(tn, d1, d2, pp, tbq=_TBQ):
    """Vector JAX port of WRF ``VILKA`` table solve.

    WRF iterates until the integer table index is stable.  The oracle regimes
    converge in a handful of iterations; fixed fp64 iterations avoid a
    data-dependent Python loop while preserving the Fortran update.
    """

    idx = jnp.trunc((tn - 173.15) / 0.05 + 1.0).astype(jnp.int32)
    idx = jnp.clip(idx, 1, 5000)
    t_i = jnp.take(tbq, idx - 1)
    t_ip1 = jnp.take(tbq, idx)
    t1 = 173.1 + idx.astype(jnp.float64) * 0.05
    f1 = t1 + d1 * t_i - d2
    denom = 0.05 + d1 * (t_ip1 - t_i)
    idx = idx - jnp.trunc(f1 / denom).astype(jnp.int32)
    idx = jnp.clip(idx, 1, 5000)

    rn = jnp.zeros_like(tn, dtype=jnp.float64)
    for _ in range(24):
        t_i = jnp.take(tbq, idx - 1)
        t_ip1 = jnp.take(tbq, idx)
        t1 = 173.1 + idx.astype(jnp.float64) * 0.05
        f1 = t1 + d1 * t_i - d2
        rn = f1 / (0.05 + d1 * (t_ip1 - t_i))
        idx = idx - jnp.trunc(rn).astype(jnp.int32)
        idx = jnp.clip(idx, 1, 5000)

    t_i = jnp.take(tbq, idx - 1)
    t_ip1 = jnp.take(tbq, idx)
    t1 = 173.1 + idx.astype(jnp.float64) * 0.05
    f1 = t1 + d1 * t_i - d2
    rn = f1 / (0.05 + d1 * (t_ip1 - t_i))
    ts = t1 - 0.05 * rn
    qs = (t_i + (t_i - t_ip1) * rn) / pp
    return qs, ts


def _gather_table(table: dict[int, tuple[float, ...]], idx, width: int):
    rows = jnp.asarray([table[k] for k in sorted(table)], dtype=jnp.float64)
    keys = jnp.asarray(sorted(table), dtype=jnp.int32)
    pos = jnp.argmax(keys[None, :] == idx.astype(jnp.int32)[:, None], axis=1)
    valid = jnp.any(keys[None, :] == idx.astype(jnp.int32)[:, None], axis=1)
    vals = rows[pos]
    if not bool(jnp.all(valid)):
        raise ValueError("RUC JAX port currently supports only the oracle table classes")
    return tuple(vals[:, n] for n in range(width))


def _veg_params(ivgtyp, vegfra, *, shdmin=1.0, shdmax=80.0):
    (
        _albedo,
        z0tbl,
        lemitbl,
        pctbl,
        _shdfac,
        ifortbl,
        rstbl,
        rgltbl,
        _hstbl,
        _snuptbl,
        laitbl,
        _maxalb,
    ) = _gather_table(_VEG, ivgtyp, 12)
    factor = jnp.where(
        (shdmax - shdmin) < 1.0,
        1.0,
        1.0 - jnp.maximum(0.0, jnp.minimum(1.0, (vegfra - shdmin) / jnp.maximum(1.0, shdmax - shdmin))),
    )
    deltalai = jnp.zeros_like(laitbl)
    deltalai = jnp.where(ifortbl == 1, jnp.minimum(0.2, 0.8 * laitbl), deltalai)
    deltalai = jnp.where((ifortbl == 2) | (ifortbl == 7), jnp.minimum(0.5, 0.8 * laitbl), deltalai)
    deltalai = jnp.where(ifortbl == 3, jnp.minimum(0.45, 0.8 * laitbl), deltalai)
    deltalai = jnp.where(ifortbl == 4, jnp.minimum(0.75, 0.8 * laitbl), deltalai)
    deltalai = jnp.where(ifortbl == 5, jnp.minimum(0.86, 0.8 * laitbl), deltalai)
    lai = laitbl - deltalai * factor
    znt = jnp.where(ifortbl == 7, z0tbl - 0.125 * factor, z0tbl)
    return {
        "iforest": ifortbl,
        "emiss": lemitbl,
        "pc": pctbl,
        "znt": znt,
        "lai": lai,
        "rst": rstbl,
        "rgl": rgltbl,
    }


def _soil_params(isltyp):
    bb, drysmc, hc, maxsmc, refsmc, satpsi, satdk, _satdw, wltsmc, qtz = _gather_table(
        _SOIL, isltyp, 10
    )
    return {
        "rhocs": hc * 1.0e6,
        "bclh": bb,
        "dqm": maxsmc - drysmc,
        "ksat": satdk,
        "psis": -satpsi,
        "qmin": drysmc,
        "ref": refsmc,
        "wilt": wltsmc,
        "qwrtz": qtz,
    }


def _soilprop_from_tav(soilmois, soiliqw, soilice, soilmoism, soiliqwm, soilicem, tav, keepfr, p):
    dqm = p["dqm"]
    qmin = p["qmin"]
    psis = p["psis"]
    bclh = p["bclh"]
    ksat = p["ksat"]
    qwrtz = p["qwrtz"]
    rhocs = p["rhocs"]

    riw = 0.9
    xlmelt = 3.35e5
    cvw = 4.183e6
    ci = 900.0 * 2100.0
    kqwrtz = 7.7
    kice = 2.2
    kwt = 0.57

    ws = dqm + qmin
    x1 = xlmelt / (_G * psis)
    x2 = x1 / bclh * ws
    x4 = (bclh + 1.0) / bclh
    gamd = (1.0 - ws) * 2700.0
    kdry = (0.135 * gamd + 64.7) / (2700.0 - 0.947 * gamd)
    kas = jnp.where(qwrtz > 0.2, kqwrtz**qwrtz * 2.0 ** (1.0 - qwrtz), kqwrtz**qwrtz * 3.0 ** (1.0 - qwrtz))

    ncol = soilmois.shape[0]
    thdif = jnp.zeros_like(soilmois)
    diffu = jnp.zeros_like(soilmois)
    cap = jnp.zeros_like(soilmois)
    fwsat = jnp.zeros((ncol, RUC_NUM_SOIL_LAYERS - 1), dtype=jnp.float64)
    lwsat = jnp.zeros((ncol, RUC_NUM_SOIL_LAYERS - 1), dtype=jnp.float64)

    tavln = jnp.log(tav / 273.15)
    raw_soiliqwm = (dqm[:, None] + qmin[:, None]) * (
        3.35e5 * (tav - 273.15) / tav / _G / psis[:, None]
    ) ** (-1.0 / bclh[:, None]) - qmin[:, None]
    cold = tavln < 0.0
    fwsat = jnp.where(cold, dqm[:, None] - raw_soiliqwm, 0.0)
    lwsat = jnp.where(cold, raw_soiliqwm + qmin[:, None], dqm[:, None] + qmin[:, None])

    tn = tav - 273.15
    wd = ws[:, None] - riw * soilicem
    psif = psis[:, None] * 100.0 * (wd / (soiliqwm + qmin[:, None])) ** bclh[:, None] * (
        ws[:, None] / wd
    ) ** 3.0
    pf = jnp.log10(jnp.abs(psif))
    fact = 1.0 + riw * soilicem
    hk = jnp.where(pf <= 5.2, 420.0 * jnp.exp(-(pf + 2.7)) * fact, 0.1744 * fact)

    detal = jnp.where(
        (soilicem != 0.0) & (tn < 0.0) & (keepfr[:, :-1] != 1.0),
        273.15 * x2[:, None] / (tav * tav) * (tav / (x1[:, None] * tn)) ** x4[:, None],
        0.0,
    )

    kasat = kas[:, None] ** (1.0 - ws[:, None]) * kice ** fwsat * kwt ** lwsat
    x5 = (soilmoism + qmin[:, None]) / ws[:, None]
    sr = jnp.maximum(0.101, x5)
    ke = jnp.where(soilicem == 0.0, jnp.log10(sr) + 1.0, x5)
    kjpl = ke * (kasat - kdry[:, None]) + kdry[:, None]

    cap_mid = (
        (1.0 - ws[:, None]) * rhocs[:, None]
        + (soiliqwm + qmin[:, None]) * cvw
        + soilicem * ci
        + (dqm[:, None] - soilmoism) * _CP * 1.2
        - detal * 1.0e3 * xlmelt
    )
    thdif_mid = kjpl / cap_mid

    a = riw * soilicem
    h = jnp.maximum(0.0, (soilmoism + qmin[:, None] - a) / jnp.maximum(1.0e-8, ws[:, None] - a))
    facd = jnp.where(a != 0.0, 1.0 - a / jnp.maximum(1.0e-8, soilmoism), 1.0)
    ame = jnp.maximum(1.0e-8, ws[:, None] - riw * soilicem)
    diffu_mid = -bclh[:, None] * ksat[:, None] * psis[:, None] / ame * (ws[:, None] / ame) ** 3.0
    diffu_mid = diffu_mid * h ** (bclh[:, None] + 2.0) * facd
    diffu_mid = jnp.where((ws[:, None] - a) < 0.12, 0.0, diffu_mid)

    thdif = thdif.at[:, : RUC_NUM_SOIL_LAYERS - 1].set(thdif_mid)
    diffu = diffu.at[:, : RUC_NUM_SOIL_LAYERS - 1].set(diffu_mid)
    cap = cap.at[:, : RUC_NUM_SOIL_LAYERS - 1].set(cap_mid)

    fach = jnp.where(soilice != 0.0, 1.0 - riw * soilice / jnp.maximum(1.0e-8, soilmois), 1.0)
    am = jnp.maximum(1.0e-8, ws[:, None] - riw * soilice)
    hydro = jnp.minimum(
        ksat[:, None],
        ksat[:, None] / am * (soiliqw / am) ** (2.0 * bclh[:, None] + 2.0) * fach,
    )
    hydro = jnp.where((ws[:, None] - riw * soilice) < 0.12, 0.0, hydro)
    hydro = jnp.where(hydro < 1.0e-10, 0.0, hydro)
    return thdif, diffu, hydro, cap


def _transf(soiliqw, tabs, lai, gswin, veg, p):
    dqm = p["dqm"]
    qmin = p["qmin"]
    ref = p["ref"]
    wilt = p["wilt"]
    pc = veg["pc"]
    rst = veg["rst"]
    rgl = veg["rgl"]
    iland_rst = rst

    ncol = soiliqw.shape[0]
    nroot = 4
    tranf = jnp.zeros_like(soiliqw)
    ap0, ap1, ap2, ap3, ap4 = 0.299, -8.152, 61.653, -115.876, 59.656
    for k in range(nroot):
        totliq = soiliqw[:, k] + qmin
        sm1 = totliq
        gx = ap0 + ap1 * sm1 + ap2 * sm1**2 + ap3 * sm1**3 + ap4 * sm1**4
        gx = jnp.where(totliq >= ref, 1.0, gx)
        gx = jnp.where(totliq <= 0.0, 0.0, gx)
        gx = jnp.clip(gx, 0.0, 1.0)
        did = _ZSHALF[k + 1] - _ZSHALF[k] if k > 0 else _ZSHALF[1]
        linear = jnp.where(
            totliq >= ref,
            did,
            jnp.where(totliq <= wilt, 0.0, (totliq - wilt) / (ref - wilt) * did),
        )
        tranf = tranf.at[:, k].set(linear)

    pctot = jnp.where(lai > 4.0, 0.8, pc)
    ftem = jnp.where(
        tabs <= 302.15,
        1.0 / (1.0 + jnp.exp(-0.41 * (tabs - 282.05))),
        1.0 / (1.0 + jnp.exp(0.5 * (tabs - 314.0))),
    )
    cmin = 1.0 / _RSMAX_DATA
    cmax = jnp.where(lai > 1.0, lai / iland_rst, 1.0 / iland_rst)
    fsol = jnp.where(gswin < rgl, 1.0 / (1.0 + jnp.exp(-0.034 * (gswin - 3.5))), 1.0)
    totcnd = (cmin + (cmax - cmin) * pctot * ftem * fsol) / cmax
    tranf = tranf.at[:, :nroot].set(jnp.maximum(cmin, tranf[:, :nroot] * totcnd[:, None]))
    transum = jnp.sum(tranf[:, :nroot], axis=1)
    return tranf, transum


def _soiltemp(
    tso,
    soilt,
    qvg,
    qsg,
    qcg,
    *,
    dt,
    ktau,
    conflx,
    prcpms,
    rainf,
    patm,
    tabs,
    qvatm,
    qcatm,
    emiss,
    rnet,
    qkms,
    tkms,
    rho,
    vegfrac,
    lai,
    thdif,
    cap,
    drycan,
    wetcan,
    transum,
    dew,
    mavail,
    soilres,
    dqm,
    qmin,
    dtdzs,
    tbq=_TBQ,
):
    del ktau, qcatm, lai, dqm, qmin
    nzs1 = RUC_NUM_SOIL_LAYERS - 1
    nzs2 = RUC_NUM_SOIL_LAYERS - 2
    dzstop = 1.0 / (_ZS[1] - _ZS[0])
    qgold = qvg
    ncol = tso.shape[0]
    cotso = jnp.zeros((ncol, RUC_NUM_SOIL_LAYERS), dtype=jnp.float64)
    rhtso = jnp.zeros((ncol, RUC_NUM_SOIL_LAYERS), dtype=jnp.float64)
    rhtso = rhtso.at[:, 0].set(tso[:, -1])

    for kk in range(1, nzs2 + 1):
        kn = RUC_NUM_SOIL_LAYERS - kk
        k1 = 2 * kn - 3
        x1 = dtdzs[k1 - 1] * thdif[:, kn - 2]
        x2 = dtdzs[k1] * thdif[:, kn - 1]
        ft = tso[:, kn - 1] + x1 * (tso[:, kn - 2] - tso[:, kn - 1]) - x2 * (
            tso[:, kn - 1] - tso[:, kn]
        )
        denom = 1.0 + x1 + x2 - x2 * cotso[:, kk - 1]
        cotso = cotso.at[:, kk].set(x1 / denom)
        rhtso = rhtso.at[:, kk].set((ft + x2 * rhtso[:, kk - 1]) / denom)

    rhcs = cap[:, 0]
    h = mavail
    trans = transum * drycan / _ZSHALF[4]
    can = wetcan + trans
    umveg = (1.0 - vegfrac) * soilres
    tn = soilt
    d1 = cotso[:, nzs1 - 1]
    d2 = rhtso[:, nzs1 - 1]
    d9 = thdif[:, 0] * rhcs * dzstop
    d10 = tkms * _CP * rho
    r211 = 0.5 * conflx / dt
    r21 = r211 * _CP * rho
    r22 = 0.5 / (thdif[:, 0] * dt * dzstop**2)
    r6 = emiss * _STBOLT * 0.5 * tn**4
    r7 = r6 / tn
    d11 = rnet + r6
    tdenom = d9 * (1.0 - d1 + r22) + d10 + r21 + r7 + rainf * 4.183e6 * prcpms
    fkq = qkms * rho
    r210 = r211 * rho
    c = vegfrac * fkq * can
    cc = c * _LV / tdenom
    aa = _LV * (fkq * umveg + r210) / tdenom
    bb = (
        d10 * tabs
        + r21 * tn
        + _LV * (qvatm * (fkq * umveg + c) + r210 * qvg)
        + d11
        + d9 * (d2 + r22 * tn)
        + rainf * 4.183e6 * prcpms * jnp.maximum(273.15, tabs)
    ) / tdenom

    pp = patm * 1.0e3
    qs1a, ts1a = _vilka(tn, (aa + cc) / pp, bb, pp, tbq)
    tx2 = qvatm * (1.0 - h)
    q1a = tx2 + h * qs1a
    saturated = q1a >= qs1a

    bb2 = bb - aa * tx2
    aa2 = (aa * h + cc) / pp
    qs1b, ts1b = _vilka(tn, aa2, bb2, pp, tbq)
    q1b = tx2 + h * qs1b
    saturated_b = q1b >= qs1b

    use_sat = saturated | saturated_b
    qs1 = jnp.where(saturated, qs1a, qs1b)
    ts1 = jnp.where(saturated, ts1a, ts1b)
    q1 = jnp.where(saturated, q1a, q1b)
    qvg_new = jnp.where(use_sat, qs1, q1)
    qsg_new = qs1
    qcg_new = jnp.where(use_sat, jnp.maximum(0.0, q1 - qs1), 0.0)

    tso_new = tso.at[:, 0].set(ts1)
    for k in range(2, RUC_NUM_SOIL_LAYERS + 1):
        kk = RUC_NUM_SOIL_LAYERS - k + 1
        tso_new = tso_new.at[:, k - 1].set(rhtso[:, kk - 1] + cotso[:, kk - 1] * tso_new[:, k - 2])

    storage = (_CP * rho * r211 + rhcs * _ZS[1] * 0.5 / dt) * (ts1 - tn) + _LV * rho * r211 * (
        qvg_new - qgold
    )
    storage = storage - rainf * 4.183e6 * prcpms * (jnp.maximum(273.15, tabs) - ts1)
    return tso_new, ts1, qvg_new, qsg_new, qcg_new, storage


def _soilmoist_full(
    soilmois,
    soiliqw,
    *,
    dt,
    dtdzs,
    dtdzs2,
    diffu,
    hydro,
    qsg,
    qvg,
    qcg,
    qcatm,
    qvatm,
    prcp,
    qkms,
    transp,
    drip,
    dew,
    soilice,
    vegfrac,
    soilres,
    dqm,
    qmin,
    ref,
    ksat,
    ras,
):
    ncol = soilmois.shape[0]
    nzs1 = RUC_NUM_SOIL_LAYERS - 1
    nzs2 = RUC_NUM_SOIL_LAYERS - 2
    cosmc = jnp.zeros_like(soilmois)
    rhsmc = jnp.zeros_like(soilmois)
    rhsmc = rhsmc.at[:, 0].set(soilmois[:, -1])

    for kk in range(1, nzs2 + 1):
        kn = RUC_NUM_SOIL_LAYERS - kk
        k1 = 2 * kn - 3
        x4 = 2.0 * dtdzs[k1 - 1] * diffu[:, kn - 2]
        x2 = 2.0 * dtdzs[k1] * diffu[:, kn - 1]
        q4 = x4 + hydro[:, kn - 2] * dtdzs2[kn - 2]
        q2 = x2 - hydro[:, kn] * dtdzs2[kn - 2]
        denom = 1.0 + x2 + x4 - q2 * cosmc[:, kk - 1]
        cosmc = cosmc.at[:, kk].set(q4 / denom)
        rhsmc = rhsmc.at[:, kk].set(
            (
                soilmois[:, kn - 1]
                + q2 * rhsmc[:, kk - 1]
                + transp[:, kn - 1] / (_ZSHALF[kn] - _ZSHALF[kn - 1]) * dt
            )
            / denom
        )

    trans = transp[:, 0]
    umveg = (1.0 - vegfrac) * soilres
    runoff = jnp.zeros((ncol,), dtype=jnp.float64)
    runoff2 = jnp.zeros((ncol,), dtype=jnp.float64)
    dzs = _ZS[1]
    r1 = cosmc[:, nzs1 - 1]
    r2 = rhsmc[:, nzs1 - 1]
    r3 = diffu[:, 0] / dzs
    r4 = r3 + hydro[:, 0] * 0.5
    r5 = r3 - hydro[:, 1] * 0.5
    r6 = qkms * ras
    totliq = prcp - drip / dt - umveg * dew * ras
    flx = totliq
    infiltrp = totliq

    cvfrz = 3.0
    refkdt = 3.0
    refdk = 3.4341e-6
    delt1 = dt / 86400.0
    f1max = dqm * _ZSHALF[1]
    fd = f1max * (1.0 - soilmois[:, 0] / dqm)
    dice = soilice[:, 0] * _ZSHALF[1]
    for k in range(1, nzs1):
        dice = dice + (_ZSHALF[k + 1] - _ZSHALF[k]) * soilice[:, k]
        fkmax = dqm * (_ZSHALF[k + 1] - _ZSHALF[k])
        fd = fd + fkmax * (1.0 - soilmois[:, k] / dqm)
    kdt = refkdt * ksat / refdk
    ddt = fd * (1.0 - jnp.exp(-kdt * delt1))
    px = jnp.maximum(0.0, -totliq * dt)
    infmax1 = jnp.where(px > 0.0, (px * (ddt / (px + ddt))) / dt, 0.0)
    frzx = 0.15 * ((dqm + qmin) / ref) * (0.412 / 0.468)
    acrt = cvfrz * frzx / dice
    frozen_sum = 1.0 + acrt**2 / 2.0 + acrt
    fcr = jnp.where(dice > 1.0e-2, 1.0 - jnp.exp(-acrt) * frozen_sum, 1.0)
    infmax1 = infmax1 * fcr
    infmax = jnp.maximum(infmax1, hydro[:, 0] * soilmois[:, 0])
    infmax = jnp.minimum(infmax, -totliq)
    runoff = jnp.where(-totliq > infmax, -totliq - infmax, runoff)
    flx = jnp.where(-totliq > infmax, -infmax, flx)
    infiltrp = flx

    r7 = 0.5 * dzs / dt
    r4 = r4 + r7
    flx = flx - soilmois[:, 0] * r7
    r8 = umveg * r6
    qtot = qvatm + qcatm
    r9 = trans
    r10 = qtot - qsg
    evap_regime = r10 <= 0.0
    qq_evap = (r5 * r2 - flx + r9) / (r4 - r5 * r1 - r10 * r8 / (ref - qmin))
    flxsat_evap = -dqm * (r4 - r5 * r1 - r10 * r8 / (ref - qmin)) + r5 * r2 + r9
    qq_dew = (r2 * r5 - flx + r8 * (qtot - qcg - qvg) + r9) / (r4 - r1 * r5)
    flxsat_dew = -dqm * (r4 - r1 * r5) + r2 * r5 + r8 * (qtot - qvg - qcg) + r9
    qq = jnp.where(evap_regime, qq_evap, qq_dew)
    flxsat = jnp.where(evap_regime, flxsat_evap, flxsat_dew)

    top_sat = qq > dqm
    runoff = jnp.where(top_sat, runoff + (flxsat - flx), runoff)
    top = jnp.where(qq < 0.0, 1.0e-8, jnp.where(top_sat, dqm, jnp.minimum(dqm, jnp.maximum(1.0e-8, qq))))
    soilmois = soilmois.at[:, 0].set(top)

    for k in range(2, RUC_NUM_SOIL_LAYERS + 1):
        kk = RUC_NUM_SOIL_LAYERS - k + 1
        qq = cosmc[:, kk - 1] * soilmois[:, k - 2] + rhsmc[:, kk - 1]
        sat = qq > dqm
        thickness = jnp.where(k == RUC_NUM_SOIL_LAYERS, _ZS[k - 1] - _ZSHALF[k - 1], _ZSHALF[k] - _ZSHALF[k - 1])
        runoff2 = jnp.where(sat, runoff2 + ((qq - dqm) * thickness) / dt, runoff2)
        val = jnp.where(qq < 0.0, 1.0e-8, jnp.where(sat, dqm, jnp.minimum(dqm, jnp.maximum(1.0e-8, qq))))
        soilmois = soilmois.at[:, k - 1].set(val)

    mavail = jnp.maximum(0.00001, jnp.minimum(1.0, soilmois[:, 0] / (ref - qmin)))
    return soilmois, mavail, runoff, runoff2, infiltrp


def _soil_step(
    state: dict,
    forcing: dict,
    p: dict,
    veg: dict,
    *,
    dt: float,
    ktau: int,
    dtdzs,
    dtdzs2,
):
    soilmois_total = state["soilmois"]
    sh2o_total = state["sh2o"]
    tso = state["tso"]
    soilt = state["soilt"]
    qvg = state["qvg"]
    qsg = state["qsg"]
    qcg = state["qcg"]
    canwatr = state["canwatr"]

    dqm = p["dqm"]
    qmin = p["qmin"]
    ref = p["ref"]
    bclh = p["bclh"]
    psis = p["psis"]

    soilm1d = jnp.minimum(jnp.maximum(0.0, soilmois_total - qmin[:, None]), dqm[:, None])
    soiliqw = jnp.minimum(jnp.maximum(0.0, sh2o_total - qmin[:, None]), soilm1d)
    soilice = (soilm1d - soiliqw) / 0.9
    smfrkeep = state["smfr3d"]
    keepfr = state["keepfr3dflag"]
    lmavail = jnp.maximum(0.00001, jnp.minimum(1.0, soilm1d[:, 0] / (ref - qmin)))

    patm = forcing["p8w"] * 1.0e-5
    patmb = forcing["p8w"] * 1.0e-2
    tabs = forcing["tabs"]
    qvatm = forcing["qv"]
    qcatm = forcing["qc"]
    rho = forcing["rho"]
    glw = forcing["glw"]
    gsw = forcing["gsw"]
    alb = state["alb"]
    emiss = veg["emiss"]
    znt = veg["znt"]
    lai = veg["lai"]

    if ktau == 1:
        qsg0 = _qsn(soilt) / patmb
        qsg = qsg0
        qvg = qsg0 * state["mavail_for_qkms"]
        qcg = forcing["qc"]
        state["sfcrunoff"] = jnp.zeros_like(soilt)
        state["udrunoff"] = jnp.zeros_like(soilt)
        state["sfcevp"] = jnp.zeros_like(soilt)

    rainbl = forcing["rainbl"]
    prcpms = jnp.where(tabs <= 273.15, 0.0, rainbl / dt * 1.0e-3)
    newsnms = jnp.where(tabs <= 273.15, rainbl / dt * 1.0e-3, 0.0)
    if bool(jnp.any(newsnms != 0.0)):
        raise NotImplementedError("RUC JAX port currently validates the no-snow oracle path only")

    qkms = forcing["flqc"] / rho / state["mavail_for_qkms"]
    tkms = forcing["flhc"] / rho / (_CP * (1.0 + 0.84 * qvatm))
    conflx = forcing["z3d"] * 0.5
    gswin = gsw / (1.0 - alb)
    rnet = gsw + emiss * (glw - _STBOLT * soilt**4)

    vegfrac = 0.01 * forcing["vegfra"]
    interw = jnp.where(vegfrac > 0.01, 0.25 * dt * prcpms * (1.0 - jnp.exp(-0.5 * lai)) * vegfrac, 0.0)
    infwater = jnp.where(vegfrac > 0.01, prcpms - interw / dt, prcpms)
    dd1 = canwatr + interw
    sat = 5.0e-4
    drip = jnp.where((vegfrac > 0.01) & (dd1 > sat), dd1 - sat, 0.0)
    canwatr = jnp.where(vegfrac > 0.01, jnp.minimum(dd1, sat), 0.0)

    tav = 0.5 * (tso[:, :-1] + tso[:, 1:])
    soilmoism = 0.5 * (soilm1d[:, :-1] + soilm1d[:, 1:])
    tavln = jnp.log(tav / 273.15)
    raw_soiliqwm = (dqm[:, None] + qmin[:, None]) * (
        3.35e5 * (tav - 273.15) / tav / _G / psis[:, None]
    ) ** (-1.0 / bclh[:, None]) - qmin[:, None]
    soiliqwm = jnp.where(tavln < 0.0, jnp.minimum(jnp.maximum(0.0, raw_soiliqwm), soilmoism), soilmoism)
    soilicem = jnp.where(tavln < 0.0, (soilmoism - soiliqwm) / 0.9, 0.0)

    thdif, diffu, hydro, cap = _soilprop_from_tav(
        soilm1d, soiliqw, soilice, soilmoism, soiliqwm, soilicem, tav, keepfr, p
    )

    fq = qkms
    dew = jnp.where(qvatm >= qsg, fq * (qvatm - qsg), 0.0)
    wetcan = jnp.minimum(0.25, jnp.maximum(0.0, canwatr / sat) ** _CFACTR_DATA)
    drycan = 1.0 - wetcan
    tranf, transum = _transf(soiliqw, tabs, lai, gswin, veg, p)

    fc = jnp.maximum(qmin, ref * 0.5)
    soilres = jnp.where(
        ((soilm1d[:, 0] + qmin) > fc) | ((qvatm - qvg) > 0.0),
        1.0,
        0.25
        * (
            1.0
            - jnp.cos(
                _PI
                * jnp.maximum(0.01, jnp.minimum(1.0, (soilm1d[:, 0] + qmin) / fc))
            )
        )
        ** 2.0,
    )
    tso_new, soilt_new, qvg_new, qsg_new, qcg_new, storage_x = _soiltemp(
        tso,
        soilt,
        qvg,
        qsg,
        qcg,
        dt=dt,
        ktau=ktau,
        conflx=conflx,
        prcpms=prcpms,
        rainf=jnp.where(prcpms != 0.0, 1.0, 0.0),
        patm=patm,
        tabs=tabs,
        qvatm=qvatm,
        qcatm=qcatm,
        emiss=emiss,
        rnet=rnet,
        qkms=qkms,
        tkms=tkms,
        rho=rho,
        vegfrac=vegfrac,
        lai=lai,
        thdif=thdif,
        cap=cap,
        drycan=drycan,
        wetcan=wetcan,
        transum=transum,
        dew=dew,
        mavail=lmavail,
        soilres=soilres,
        dqm=dqm,
        qmin=qmin,
        dtdzs=dtdzs,
    )

    # Recompute flux partition after the surface solve, following SOIL.
    dew2 = jnp.where(qvatm >= qsg_new, qkms * (qvatm - qsg_new), 0.0)
    transp = jnp.zeros_like(soilm1d)
    ett1 = jnp.zeros_like(soilt)
    evap = qvatm < qsg_new
    for k in range(4):
        tr = vegfrac * (rho * 1.0e-3) * qkms * (qvatm - qsg_new) * tranf[:, k] * drycan / _ZSHALF[4]
        tr = jnp.where(tr > 0.0, 0.0, tr)
        transp = transp.at[:, k].set(jnp.where(evap, tr, 0.0))
        ett1 = jnp.where(evap, ett1 - tr, ett1)

    # Warm oracle path: no frozen soil after the heat solve.
    soiliqw = soilm1d
    soilice = jnp.zeros_like(soilm1d)

    soilm1d_new, lmavail_new, runoff1, runoff2, infiltrp = _soilmoist_full(
        soilm1d,
        soiliqw,
        dt=dt,
        dtdzs=dtdzs,
        dtdzs2=dtdzs2,
        diffu=diffu,
        hydro=hydro,
        qsg=qsg_new,
        qvg=qvg_new,
        qcg=qcg_new,
        qcatm=qcatm,
        qvatm=qvatm,
        prcp=-infwater,
        qkms=qkms,
        transp=transp,
        drip=drip,
        dew=dew2,
        soilice=soilice,
        vegfrac=vegfrac,
        soilres=soilres,
        dqm=dqm,
        qmin=qmin,
        ref=ref,
        ksat=p["ksat"],
        ras=rho * 1.0e-3,
    )

    hft = -tkms * _CP * rho * (tabs - soilt_new)
    hfx = hft * (_P1000MB * 0.00001 / patm) ** _ROVCP
    cond = -qkms * (rho * 1.0e-3) * (qvatm - qsg_new) <= 0.0
    eeta_cond = -rho * dew2
    edir1 = -soilres * (1.0 - vegfrac) * qkms * (rho * 1.0e-3) * (qvatm - qvg_new)
    ec1 = (-qkms * (rho * 1.0e-3) * (qvatm - qsg_new)) * wetcan * vegfrac
    canwatr = jnp.where(cond, canwatr + dt * dew2 * (rho * 1.0e-3) * vegfrac, jnp.maximum(0.0, canwatr - ec1 * dt))
    eeta_evap = (edir1 + ec1 + ett1) * 1.0e3
    eeta = jnp.where(cond, eeta_cond, eeta_evap)
    lh = _LV * eeta
    s = thdif[:, 0] * cap[:, 0] * (1.0 / (_ZS[1] - _ZS[0])) * (tso_new[:, 0] - tso_new[:, 1])
    _fltot = rnet - hft - _LV * eeta - s - storage_x
    del _fltot, infiltrp

    soilmois_total_new = soilm1d_new + qmin[:, None]
    # SOILMOIST mutates total soil moisture, but leaves SOILIQW as the
    # post-temperature, pre-moisture-solve liquid-water profile.  LSMRUC exports
    # SH2O from that liquid carry, capped by the updated total moisture.
    sh2o_total_new = jnp.minimum(soiliqw + qmin[:, None], soilmois_total_new)
    tso_total_new = tso_new.at[:, -1].set(forcing["tbot"])
    smavail = _soil_water_integral(soilmois_total_new)
    smmax = _soil_water_integral((qmin + dqm)[:, None] + jnp.zeros_like(soilmois_total_new))
    sfcrunoff = state["sfcrunoff"] + runoff1 * dt * 1000.0
    udrunoff = state["udrunoff"] + runoff2 * dt * 1000.0
    sfcevp = state["sfcevp"] + eeta * dt
    sfcevp = sfcevp + eeta * dt

    patmb = forcing["p8w"] * 1.0e-2
    q2sat = _qsn(tabs) / patmb
    qsfc = qvg_new / (1.0 + qvg_new)
    chklowq = jnp.where((qvatm >= q2sat * 0.95) & (qvatm < qvg_new), 0.0, 1.0)
    del chklowq

    return {
        **state,
        "soilt": soilt_new,
        "tso": tso_total_new,
        "soilmois": soilmois_total_new,
        "sh2o": sh2o_total_new,
        "smfr3d": smfrkeep,
        "keepfr3dflag": keepfr,
        "qvg": qvg_new,
        "qsg": qsg_new,
        "qcg": qcg_new,
        "dew": dew2,
        "qsfc": qsfc,
        "hfx": hfx,
        "qfx": eeta,
        "lh": lh,
        "grdflx": -s,
        "sfcexc": tkms,
        "sfcrunoff": sfcrunoff,
        "udrunoff": udrunoff,
        "sfcevp": sfcevp,
        "smavail": smavail * 1000.0,
        "smmax": smmax * 1000.0,
        "mavail": lmavail_new,
        "mavail_for_qkms": lmavail_new,
        "canwatr": canwatr,
        "emiss": emiss,
        "znt": znt,
        "lai": lai,
        "alb": alb,
        "tsnav": state["tsnav"],
        "soilt1": state["soilt1"],
    }


def _soil_water_integral(profile):
    out = jnp.zeros((profile.shape[0],), dtype=jnp.float64)
    for k in range(RUC_NUM_SOIL_LAYERS - 1):
        out = out + profile[:, k] * (_ZSHALF[k + 1] - _ZSHALF[k])
    out = out + profile[:, -1] * (_ZS[-1] - _ZSHALF[-1])
    return out


def ruc_columns(
    *,
    gsw,
    glw,
    emiss,
    tabs,
    qv,
    qc,
    rho,
    p8w,
    z3d,
    rainbl,
    vegfra,
    flhc,
    flqc,
    tbot,
    xland,
    mavail,
    ivgtyp,
    isltyp,
    soilt,
    tso,
    soilmois,
    sh2o,
    snow=None,
    snowh=None,
    alb=None,
    dt: float = 180.0,
    nsteps: int = 6,
):
    """Run the RUC fp64 land-column path over a batch of columns.

    This port intentionally validates the WRF-oracle no-snow land path.  Water,
    sea-ice, and snow columns still fail closed until corresponding WRF oracle
    regimes are added.
    """

    gsw = _as_f64(gsw)
    ncol = gsw.shape[0]
    glw = _as_f64(glw)
    tabs = _as_f64(tabs)
    qv = _as_f64(qv)
    qc = _as_f64(qc)
    rho = _as_f64(rho)
    p8w = _as_f64(p8w)
    z3d = _as_f64(z3d)
    rainbl = _as_f64(rainbl)
    vegfra = _as_f64(vegfra)
    flhc = _as_f64(flhc)
    flqc = _as_f64(flqc)
    tbot = _as_f64(tbot)
    xland = _as_f64(xland)
    mavail = _as_f64(mavail)
    soilt = _as_f64(soilt)
    tso = _as_f64(tso)
    soilmois = _as_f64(soilmois)
    sh2o = _as_f64(sh2o)
    ivgtyp = jnp.asarray(ivgtyp, dtype=jnp.int32)
    isltyp = jnp.asarray(isltyp, dtype=jnp.int32)
    snow = jnp.zeros((ncol,), dtype=jnp.float64) if snow is None else _as_f64(snow)
    snowh = jnp.zeros((ncol,), dtype=jnp.float64) if snowh is None else _as_f64(snowh)
    alb = jnp.full((ncol,), 0.18, dtype=jnp.float64) if alb is None else _as_f64(alb)

    if bool(jnp.any(xland >= 1.5)):
        raise NotImplementedError("RUC JAX port currently validates land columns only")
    if bool(jnp.any(snow != 0.0)) or bool(jnp.any(snowh != 0.0)):
        raise NotImplementedError("RUC JAX port currently validates no-snow columns only")

    veg = _veg_params(ivgtyp, vegfra)
    p = _soil_params(isltyp)
    dtdzs, dtdzs2 = _soil_depth_coefficients(float(dt))
    forcing = {
        "gsw": gsw,
        "glw": glw,
        "emiss_in": _as_f64(emiss),
        "tabs": tabs,
        "qv": qv,
        "qc": qc,
        "rho": rho,
        "p8w": p8w,
        "z3d": z3d,
        "rainbl": rainbl,
        "vegfra": vegfra,
        "flhc": flhc,
        "flqc": flqc,
        "tbot": tbot,
    }
    state = {
        "soilt": soilt,
        "tso": tso,
        "soilmois": soilmois,
        "sh2o": sh2o,
        "smfr3d": jnp.zeros_like(soilmois),
        "keepfr3dflag": jnp.zeros_like(soilmois),
        "snow": snow,
        "snowh": snowh,
        "qvg": jnp.zeros((ncol,), dtype=jnp.float64),
        "qsg": jnp.zeros((ncol,), dtype=jnp.float64),
        "qcg": jnp.zeros((ncol,), dtype=jnp.float64),
        "dew": jnp.zeros((ncol,), dtype=jnp.float64),
        "qsfc": jnp.zeros((ncol,), dtype=jnp.float64),
        "canwatr": jnp.zeros((ncol,), dtype=jnp.float64),
        "sfcrunoff": jnp.zeros((ncol,), dtype=jnp.float64),
        "udrunoff": jnp.zeros((ncol,), dtype=jnp.float64),
        "sfcevp": jnp.zeros((ncol,), dtype=jnp.float64),
        "alb": alb,
        "mavail": mavail,
        "mavail_for_qkms": mavail,
        "tsnav": 0.5 * (soilt + tso[:, 0]) - 273.15,
        "soilt1": soilt,
    }
    for ktau in range(1, int(nsteps) + 1):
        state = _soil_step(state, forcing, p, veg, dt=float(dt), ktau=ktau, dtdzs=dtdzs, dtdzs2=dtdzs2)

    return {
        "soilt": state["soilt"],
        "tso": state["tso"],
        "soilmois": state["soilmois"],
        "sh2o": state["sh2o"],
        "smfr3d": state["smfr3d"],
        "keepfr3dflag": state["keepfr3dflag"],
        "snow": state["snow"],
        "snowh": state["snowh"],
        "hfx": state["hfx"],
        "qfx": state["qfx"],
        "lh": state["lh"],
        "grdflx": state["grdflx"],
        "smavail": state["smavail"],
        "smmax": state["smmax"],
        "sfcrunoff": state["sfcrunoff"],
        "udrunoff": state["udrunoff"],
        "qsfc": state["qsfc"],
        "qsg": state["qsg"],
        "qvg": state["qvg"],
        "qcg": state["qcg"],
        "dew": state["dew"],
        "alb": state["alb"],
        "emiss": state["emiss"],
        "znt": state["znt"],
        "lai": state["lai"],
        "mavail": state["mavail"],
        "sfcevp": state["sfcevp"],
        "sfcexc": state["sfcexc"],
        "tsnav": state["tsnav"],
        "soilt1": state["soilt1"],
    }


def ruc_column(**kwargs):
    """Single-column convenience wrapper around :func:`ruc_columns`."""

    batched = {}
    for key, value in kwargs.items():
        if key in {"dt", "nsteps"}:
            batched[key] = value
        else:
            arr = _as_f64(value)
            batched[key] = arr[None, ...]
    out = ruc_columns(**batched)
    return {key: value[0] for key, value in out.items()}


__all__ = [
    "RUC_ORACLE_DIR",
    "RUC_SAVEPOINT",
    "RUC_NUM_SOIL_LAYERS",
    "RucLandState",
    "ruc_column",
    "ruc_columns",
]
