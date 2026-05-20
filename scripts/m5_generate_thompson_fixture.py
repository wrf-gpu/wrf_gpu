#!/usr/bin/env python3
"""Generate the M5 Thompson analytic column fixture from transcribed WRF formulas."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.physics.thompson_constants import (  # noqa: E402
    AM_G_MP8,
    AM_I,
    AM_R,
    C_CUBE,
    C_SQRD,
    CCG2_NU12,
    CCG3_NU12,
    CP,
    CRE10,
    CRE11,
    CRE2,
    CRE9,
    CRG2,
    CRG3,
    D0C,
    D0I,
    D0R,
    D0S,
    EPS,
    FV_R,
    HGFR,
    LFUS,
    LSUB,
    NT_C,
    NU_C_MP8,
    OBMI,
    OBMR,
    OCG1_NU12,
    OCG2_NU12,
    OIG1,
    OIG2,
    ORG2,
    ORG3,
    PI,
    R1,
    R2,
    R_D,
    RHO_NOT,
    RV,
    T1_MELT_QG,
    T1_MELT_QS,
    T1_QR_EV,
    T1_QR_QC,
    T1_SUBL_QG,
    T1_SUBL_QS,
    T2_MELT_QG,
    T2_MELT_QS,
    T2_QR_EV,
    T2_SUBL_QG,
    T2_SUBL_QS,
    T_0,
    XM0I,
)


FIXTURE_ID = "analytic-thompson-column-v1"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
WRF_SOURCE = ROOT.parent / "wrf_gpu" / "sidecar_reports" / "post13_thompson_first_divergence_20260508T224837Z" / "source_snapshots_pre" / "module_mp_thompson.F.pre"


def _rslf(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """NumPy RSLF transcription from module_mp_thompson.F.pre lines 5444-5468."""

    x = np.maximum(-80.0, T - 273.16)
    esl = (
        0.611583699e03
        + x
        * (
            0.444606896e02
            + x
            * (
                0.143177157e01
                + x
                * (
                    0.264224321e-1
                    + x
                    * (
                        0.299291081e-3
                        + x
                        * (
                            0.203154182e-5
                            + x * (0.702620698e-8 + x * (0.379534310e-11 + x * -0.321582393e-13))
                        )
                    )
                )
            )
        )
    )
    esl = np.minimum(esl, p * 0.15)
    return 0.622 * esl / (p - esl)


def _rsif(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """NumPy RSIF transcription from module_mp_thompson.F.pre lines 5473-5490."""

    x = np.maximum(-80.0, T - 273.16)
    esi = (
        0.609868993e03
        + x
        * (
            0.499320233e02
            + x
            * (
                0.184672631e01
                + x
                * (
                    0.402737184e-1
                    + x
                    * (
                        0.565392987e-3
                        + x
                        * (
                            0.521693933e-5
                            + x * (0.307839583e-7 + x * (0.105785160e-9 + x * 0.161444444e-12))
                        )
                    )
                )
            )
        )
    )
    esi = np.minimum(esi, p * 0.15)
    return 0.622 * esi / np.maximum(1.0e-4, p - esi)


def _lvap(T: np.ndarray) -> np.ndarray:
    """NumPy lvap formula from module_mp_thompson.F.pre lines 2063 and 3270."""

    return 2.5e6 + (2106.0 - 4218.0) * (T - 273.15)


def _ocp(qv: np.ndarray) -> np.ndarray:
    """NumPy ocp formula from module_mp_thompson.F.pre lines 2061 and 3272."""

    return 1.0 / (CP * (1.0 + 0.887 * qv))


def _rho(p: np.ndarray, T: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """NumPy rho formula from mp_gt_driver line 1270."""

    return 0.622 * p / (R_D * T * (qv + 0.622))


def _air_props_np(qv: np.ndarray, T: np.ndarray, p: np.ndarray, rho: np.ndarray):
    """NumPy transcription of WRF thermodynamic scalars at lines 2055-2064."""

    tempc = T - 273.15
    diffu = 2.11e-5 * (T / 273.15) ** 1.94 * (101325.0 / p)
    visco = np.where(tempc >= 0.0, 1.718 + 0.0049 * tempc, 1.718 + 0.0049 * tempc - 1.2e-5 * tempc * tempc) * 1.0e-5
    tcond = (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936
    lvap = _lvap(T)
    ocp = _ocp(qv)
    rhof = np.sqrt(RHO_NOT / np.maximum(rho, R1))
    rhof2 = np.sqrt(rhof)
    vsc2 = np.sqrt(np.maximum(rho, R1) / np.maximum(visco, R1))
    return tempc, diffu, tcond, lvap, ocp, rhof, rhof2, vsc2


def _rain_distribution_np(qr: np.ndarray, Nr: np.ndarray, rho: np.ndarray):
    """NumPy rain distribution using WRF lines 2210-2215."""

    rr = np.maximum(qr * rho, R1)
    nr = np.maximum(Nr * rho, R2)
    lamr = (AM_R * CRG3 * ORG2 * nr / rr) ** OBMR
    ilamr = 1.0 / lamr
    mvd_r = (3.0 + 0.672) / lamr
    n0_r = nr * ORG2 * lamr**CRE2
    active = (qr > R1) & (Nr > 0.0)
    return rr, nr, lamr, ilamr, mvd_r, n0_r, active


def _cloud_distribution_np(qc: np.ndarray, rho: np.ndarray):
    """NumPy mp=8 cloud distribution for WRF lines 2233-2239."""

    rc = np.maximum(qc * rho, R1)
    lamc = (NT_C * AM_R * CCG2_NU12 * OCG1_NU12 / rc) ** OBMR
    xdc = np.maximum(D0C * 1.0e6, (rc / (AM_R * NT_C)) ** OBMR * 1.0e6)
    mvd_c = (3.0 + NU_C_MP8 + 0.672) / lamc
    mvd_c = np.maximum(D0C, np.minimum(mvd_c, D0R))
    return rc, lamc, xdc, mvd_c, qc > R1


def _ice_distribution_np(qi: np.ndarray, Ni: np.ndarray, rho: np.ndarray):
    """NumPy ice particle terms for WRF lines 2711-2715."""

    ri = np.maximum(qi * rho, R1)
    ni = np.maximum(Ni * rho, R2)
    lami = (AM_I * 6.0 * OIG1 * ni / ri) ** OBMI
    ilami = 1.0 / lami
    xdi = np.maximum(D0I, 4.0 * ilami)
    xmi = AM_I * xdi**3.0
    return ri, ni, ilami, xmi, qi > R1


def _snow_moments_np(qs: np.ndarray, rho: np.ndarray, tempc: np.ndarray):
    """NumPy snow moment proxy for WRF lines 2745-2752 and 2845-2857."""

    rs = np.maximum(qs * rho, R1)
    xds = np.maximum(D0S, (rs / 0.069) ** 0.5)
    smo0 = rs / np.maximum(0.069 * xds * xds, R1)
    smo1 = smo0 * xds
    smof = smo0 * np.sqrt(xds)
    c_snow = C_SQRD + (tempc + 1.5) * (C_CUBE - C_SQRD) / (-30.0 + 1.5)
    c_snow = np.maximum(C_SQRD, np.minimum(c_snow, C_CUBE))
    return rs, smo0, smo1, smof, c_snow, qs > R1


def _graupel_distribution_np(qg: np.ndarray, rho: np.ndarray):
    """NumPy mp=8 graupel terms for WRF lines 2200-2203."""

    rg = np.maximum(qg * rho, R1)
    ng = np.maximum(4.0e5 * rho, R2)
    lamg = (AM_G_MP8 * CRG3 * ORG2 * ng / rg) ** (1.0 / 3.0)
    ilamg = 1.0 / lamg
    n0_g = ng * ORG2 * lamg
    return rg, ng, ilamg, n0_g, qg > R1


def _sublimation_prefactor_np(qv: np.ndarray, T: np.ndarray, p: np.ndarray, rho: np.ndarray, ssati: np.ndarray, diffu: np.ndarray, tcond: np.ndarray):
    """NumPy Srivastava-Coen ice prefactor from WRF lines 2450-2464."""

    otemp = 1.0 / T
    qvsi = _rsif(p, T)
    rvs = rho * qvsi
    rvs_p = rvs * otemp * (LSUB * otemp / RV - 1.0)
    rvs_pp = rvs * (otemp * (LSUB * otemp / RV - 1.0) * otemp * (LSUB * otemp / RV - 1.0) + (-2.0 * LSUB * otemp**3 / RV) + otemp * otemp)
    gamsc = LSUB * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) ** 2 * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = np.maximum(1.0e-9, alphsc)
    xsat = np.where(np.abs(ssati) < 1.0e-9, 0.0, ssati)
    t1_subl = 4.0 * PI * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat - 5.0 * alphsc**3 * xsat**3) / (1.0 + gamsc)
    return t1_subl, rvs


def reference_step_numpy(fields: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Path-B-strict oracle: WRF-style tendency ledger, not the JAX helper sequence."""

    qv = np.maximum(fields["qv"].copy(), 1.0e-10)
    qc = np.maximum(fields["qc"].copy(), 0.0)
    qr = np.maximum(fields["qr"].copy(), 0.0)
    qi = np.maximum(fields["qi"].copy(), 0.0)
    qs = np.maximum(fields["qs"].copy(), 0.0)
    qg = np.maximum(fields["qg"].copy(), 0.0)
    Ni = np.maximum(fields["Ni"].copy(), 0.0)
    Nr = np.maximum(fields["Nr"].copy(), 0.0)
    T = fields["T"].copy()
    p = fields["p"].copy()
    rho = _rho(p, T, qv)

    qvs = _rslf(p, T)
    lvap = _lvap(T)
    ocp = _ocp(qv)
    lvt2 = lvap * lvap * ocp / RV / (T * T)
    clap = (qv - qvs) / (1.0 + lvt2 * qvs)
    for _ in range(3):
        expo = np.exp(lvt2 * clap)
        clap = clap - (qvs * expo - qv + clap) / (qvs * lvt2 * expo + 1.0)
    ssatw = qv / qvs - 1.0
    active = (ssatw > EPS) | ((ssatw < -EPS) & (qc > 0.0))
    clap = np.where(active, clap, 0.0)
    clap = np.where(clap < 0.0, np.maximum(clap, -qc), np.minimum(clap, qv - 1.0e-10))
    qvten = -clap / dt
    qcten = clap / dt
    tten = lvap * ocp * clap / dt
    qv = qv + dt * qvten
    qc = qc + dt * qcten
    T = T + dt * tten
    rho = _rho(p, T, qv)

    ocp = _ocp(qv)
    lvap = _lvap(T)
    lfus2 = LSUB - lvap
    qi_melt = np.where(T > T_0, qi, 0.0)
    qc = qc + qi_melt
    qi = qi - qi_melt
    Ni = np.where(qi_melt > 0.0, 0.0, Ni)
    T = T - LFUS * ocp * qi_melt

    qc_freeze = np.where(T < HGFR, qc, 0.0)
    qc = qc - qc_freeze
    qi = qi + qc_freeze
    Ni = Ni + qc_freeze / XM0I
    T = T + lfus2 * ocp * qc_freeze

    rain_freeze = np.where(T < HGFR, qr, 0.0)
    qr = qr - rain_freeze
    qi = qi + rain_freeze
    Ni = Ni + np.where(rain_freeze > 0.0, Nr, 0.0)
    Nr = np.where(rain_freeze > 0.0, 0.0, Nr)
    T = T + lfus2 * ocp * rain_freeze
    rho = _rho(p, T, qv)

    tempc, diffu, tcond, _lvap_unused, ocp, _rhof, rhof2, vsc2 = _air_props_np(qv, T, p, rho)
    qvs0 = _rslf(p, T_0)
    del_qvs = np.maximum(0.0, qvs0 - qv)
    twet = np.minimum(T, T_0)
    rs, smo0, smo1, smof, _c_snow, active_snow = _snow_moments_np(qs, rho, tempc)
    rg, ng, ilamg, n0_g, active_graupel = _graupel_distribution_np(qg, rho)
    prr_sml = (tempc * tcond - 2.5e6 * diffu * del_qvs) * (T1_MELT_QS * smo1 + T2_MELT_QS * rhof2 * vsc2 * smof)
    prr_sml = np.minimum(rs / dt, np.maximum(0.0, prr_sml)) / rho
    snow_melt = np.where((T > T_0) & active_snow, prr_sml * dt, 0.0)
    pnr_sml = np.where(rs > R1, smo0 / rs * snow_melt * rho * 10.0 ** (-0.25 * (twet - T_0)), 0.0)
    prr_gml = (tempc * tcond - 2.5e6 * diffu * del_qvs) * n0_g * (T1_MELT_QG * ilamg**CRE10 + T2_MELT_QG * rhof2 * vsc2 * ilamg**CRE11)
    prr_gml = np.minimum(rg / dt, np.maximum(0.0, prr_gml)) / rho
    graupel_melt = np.where((T > T_0) & active_graupel, prr_gml * dt, 0.0)
    pnr_gml = np.where(rg > R1, graupel_melt * ng / rg * 10.0 ** (-0.33 * (twet - T_0)), 0.0)
    qs = qs - snow_melt
    qg = qg - graupel_melt
    qr = qr + snow_melt + graupel_melt
    Nr = Nr + pnr_sml + pnr_gml
    T = T - LFUS * ocp * (snow_melt + graupel_melt)
    rho = _rho(p, T, qv)

    tempc, diffu, tcond, _lvap_unused, ocp, _rhof, rhof2, vsc2 = _air_props_np(qv, T, p, rho)
    qvsi = _rsif(p, T)
    ssati = qv / qvsi - 1.0
    t1_subl, rvs = _sublimation_prefactor_np(qv, T, p, rho, ssati, diffu, tcond)
    ri, ni, ilami, xmi, active_ice = _ice_distribution_np(qi, Ni, rho)
    rs, _smo0, smo1, smof, c_snow, active_snow = _snow_moments_np(qs, rho, T - 273.15)
    rg, ng, ilamg, n0_g, active_graupel = _graupel_distribution_np(qg, rho)
    pri_ide = C_CUBE * t1_subl * diffu * ssati * rvs * OIG1 * ni * ilami
    pri_ide = np.where(active_ice, pri_ide, 0.0)
    pri_ide = np.where(pri_ide < 0.0, np.maximum(-ri / dt, pri_ide), np.minimum(pri_ide, np.maximum(qv - qvsi, 0.0) * rho / dt * 0.999))
    prs_sde = c_snow * t1_subl * diffu * ssati * rvs * (T1_SUBL_QS * smo1 + T2_SUBL_QS * rhof2 * vsc2 * smof)
    prs_sde = np.where(active_snow, np.where(prs_sde < 0.0, np.maximum(-rs / dt, prs_sde), np.minimum(prs_sde, np.maximum(qv - qvsi, 0.0) * rho / dt * 0.999)), 0.0)
    prg_gde = C_CUBE * t1_subl * diffu * ssati * rvs * n0_g * (T1_SUBL_QG * ilamg**CRE10 + T2_SUBL_QG * vsc2 * rhof2 * ilamg**CRE11)
    prg_gde = np.where(active_graupel, np.where(prg_gde < 0.0, np.maximum(-rg / dt, prg_gde), np.minimum(prg_gde, np.maximum(qv - qvsi, 0.0) * rho / dt * 0.999)), 0.0)
    qi_delta = pri_ide * dt / rho
    qs_delta = prs_sde * dt / rho
    qg_delta = prg_gde * dt / rho
    vapor_sink = np.maximum(0.0, qi_delta) + np.maximum(0.0, qs_delta) + np.maximum(0.0, qg_delta)
    vapor_source = np.maximum(0.0, -qi_delta) + np.maximum(0.0, -qs_delta) + np.maximum(0.0, -qg_delta)
    qv = qv - vapor_sink + vapor_source
    qi = qi + qi_delta
    qs = qs + qs_delta
    qg = qg + qg_delta
    Ni = np.maximum(0.0, Ni + np.where(qi_delta < 0.0, qi_delta / np.maximum(xmi, XM0I), qi_delta / XM0I))
    T = T + LSUB * ocp * (vapor_sink - vapor_source)
    rho = _rho(p, T, qv)

    tempc, diffu, tcond, lvap, ocp, rhof, rhof2, vsc2 = _air_props_np(qv, T, p, rho)
    rc, lamc, xdc, mvd_c, active_cloud = _cloud_distribution_np(qc, rho)
    rr, nr, lamr, ilamr, mvd_r, n0_r, active_rain = _rain_distribution_np(qr, Nr, rho)
    dc_g = ((CCG3_NU12 * OCG2_NU12) ** OBMR / lamc) * 1.0e6
    dc_b = np.maximum(xdc**3 * dc_g**3 - xdc**6, 0.0) ** (1.0 / 6.0)
    zeta1_raw = 6.25e-6 * xdc * dc_b**3 - 0.4
    zeta1 = 0.5 * (zeta1_raw + np.abs(zeta1_raw))
    zeta = 0.027 * rc * zeta1
    taud_raw = 0.5 * dc_b - 7.5
    taud = 0.5 * (taud_raw + np.abs(taud_raw)) + R1
    tau = 3.72 / np.maximum(rc * taud, R1)
    prr_wau = np.where((rc > 0.01e-3) & active_cloud, np.minimum(rc / dt, zeta / tau), 0.0)
    pnr_wau = prr_wau / np.maximum(AM_R * NU_C_MP8 * 10.0 * D0R**3, R2)
    ef_rw = np.clip(0.55 + 0.45 * (mvd_r - D0R) / np.maximum(2.5e-3 - D0R, R1), 0.0, 1.0)
    prr_rcw = rhof * T1_QR_QC * ef_rw * rc * n0_r * ((lamr + FV_R) ** (-CRE9))
    prr_rcw = np.where(active_rain & (mvd_r > D0R) & (mvd_c > D0C), prr_rcw, 0.0)
    prr_rcw = np.minimum(np.maximum(rc - prr_wau * dt, 0.0) / dt, prr_rcw)
    autoconv = prr_wau * dt / rho
    accretion = prr_rcw * dt / rho
    transfer = np.minimum(qc, autoconv + accretion)
    Nr = Nr + pnr_wau * dt / rho
    qc = qc - transfer
    qr = qr + transfer

    qvs = _rslf(p, T)
    ssatw = qv / qvs - 1.0
    rvs = rho * qvs
    otemp = 1.0 / T
    rvs_p = rvs * otemp * (lvap * otemp / RV - 1.0)
    rvs_pp = rvs * (otemp * (lvap * otemp / RV - 1.0) * otemp * (lvap * otemp / RV - 1.0) + (-2.0 * lvap * otemp**3 / RV) + otemp * otemp)
    gamsc = lvap * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) ** 2 * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = np.maximum(1.0e-9, alphsc)
    xsat = np.minimum(-1.0e-9, ssatw)
    t1_evap = 2.0 * PI * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat - 5.0 * alphsc**3 * xsat**3) / (1.0 + gamsc)
    rr, nr, lamr, ilamr, _mvd_r, n0_r, active_rain = _rain_distribution_np(qr, Nr, rho)
    evap_raw = t1_evap * diffu * (-ssatw) * n0_r * rvs * (T1_QR_EV * ilamr**CRE10 + T2_QR_EV * vsc2 * rhof2 * ((lamr + 0.5 * FV_R) ** (-CRE11))) / rho
    fast_clear = (qv / qvs < 0.95) & (rr / rho <= 1.0e-8)
    evap_rate = np.where(fast_clear, qr / dt, evap_raw)
    rate_max = np.minimum(qr / dt, np.maximum(qvs - qv, 0.0) / dt)
    evap = np.where((ssatw < -EPS) & active_rain, np.minimum(rate_max, evap_rate) * dt, 0.0)
    nr_loss = np.where(qr > 0.0, np.minimum(Nr * 0.99, Nr * evap / np.maximum(qr, R1)), 0.0)
    qv = qv + evap
    qr = qr - evap
    Nr = np.maximum(0.0, Nr - nr_loss)
    T = T - lvap * ocp * evap

    qv = np.maximum(qv, 1.0e-10)
    qc = np.where(qc <= R1, 0.0, np.maximum(qc, 0.0))
    qr = np.where(qr <= R1, 0.0, np.maximum(qr, 0.0))
    qi = np.where(qi <= R1, 0.0, np.maximum(qi, 0.0))
    qs = np.where(qs <= R1, 0.0, np.maximum(qs, 0.0))
    qg = np.where(qg <= R1, 0.0, np.maximum(qg, 0.0))
    T = np.maximum(T, 50.0)
    rho = _rho(p, T, qv)

    ni_raw = np.maximum(R2 / rho, Ni)
    ri = np.maximum(qi * rho, R1)
    xni = np.maximum(R2, ni_raw * rho)
    lami = (AM_I * 6.0 * OIG1 * xni / ri) ** OBMI
    xdi = 4.0 / lami
    lami = np.where(xdi < 5.0e-6, 6.0 / 5.0e-6, lami)
    lami = np.where(xdi > 300.0e-6, 6.0 / 300.0e-6, lami)
    Ni = np.where(qi <= R1, 0.0, np.minimum((ri / AM_I * lami**3.0 * OIG2) / rho, 999.0e3 / rho))
    nr_raw = np.maximum(R2 / rho, Nr)
    rr = np.maximum(qr * rho, R1)
    xnr = np.maximum(R2, nr_raw * rho)
    lamr = (AM_R * CRG3 * ORG2 * xnr / rr) ** OBMR
    mvd_r = (3.0 + 0.672) / lamr
    mvd_r = np.minimum(2.5e-3, np.maximum(D0R * 0.75, mvd_r))
    lamr = (3.0 + 0.672) / mvd_r
    Nr = np.where(qr <= R1, 0.0, CRG2 * ORG3 * rr * lamr**3.0 / AM_R / rho)
    return {"qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "Ni": Ni, "Nr": Nr, "T": T, "rho": rho}


def make_scenarios() -> tuple[dict[str, np.ndarray], float]:
    """Constructs the three required maritime, mixed-phase, and precipitating columns."""

    nz = 12
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    dt = 60.0
    p = np.stack(
        [
            96000.0 - 47000.0 * z,
            90000.0 - 52000.0 * z,
            94000.0 - 50000.0 * z,
        ]
    )
    T = np.stack(
        [
            290.0 - 18.0 * z,
            268.0 - 42.0 * z,
            282.0 - 32.0 * z,
        ]
    )
    qvs = _rslf(p, T)
    qv = np.stack([0.995 * qvs[0], 0.92 * qvs[1], 0.88 * qvs[2]])
    qc = np.stack([2.5e-4 * np.exp(-((z - 0.24) / 0.18) ** 2), 9.0e-5 * np.exp(-((z - 0.42) / 0.16) ** 2), 1.6e-4 * np.exp(-((z - 0.34) / 0.20) ** 2)])
    qr = np.stack([2.0e-6 + z * 0.0, 1.0e-6 + z * 0.0, 5.0e-4 * np.exp(-((z - 0.22) / 0.20) ** 2)])
    qi = np.stack([z * 0.0, 7.0e-5 * np.exp(-((z - 0.58) / 0.18) ** 2), 5.0e-5 * np.exp(-((z - 0.72) / 0.16) ** 2)])
    qs = np.stack([z * 0.0, 1.5e-4 * np.exp(-((z - 0.66) / 0.20) ** 2), 2.0e-4 * np.exp(-((z - 0.64) / 0.18) ** 2)])
    qg = np.stack([z * 0.0, 2.0e-5 * np.exp(-((z - 0.52) / 0.14) ** 2), 2.8e-4 * np.exp(-((z - 0.38) / 0.18) ** 2)])
    Ni = np.where(qi > 0.0, 2.0e5, 0.0).astype(np.float64)
    Nr = np.where(qr > 1.0e-8, 8.0e4, 0.0).astype(np.float64)
    fields = {"T": T, "p": p, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "Ni": Ni, "Nr": Nr}
    return fields, dt


def _sha256(path: Path) -> str:
    """Computes a lowercase SHA-256 digest for manifest file entries."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_rev() -> str:
    """Returns the current git revision or branch name for manifest provenance."""

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "worker/gpt/m5-s1-thompson-microphysics-column"


def _variable(name: str, units: str, shape: tuple[int, ...], abs_tol: float, rel_tol: float, rationale: str) -> dict[str, Any]:
    """Builds one manifest variable record without schema drift."""

    return {
        "name": name,
        "units": units,
        "shape": list(shape),
        "staggering": "mass",
        "dtype": "float64",
        "tolerance_abs": float(abs_tol),
        "tolerance_rel": float(rel_tol),
        "tolerance_rationale": rationale,
        "tier_overrides": None,
    }


def write_fixture() -> dict[str, Any]:
    """Writes the sample NPZ and manifest in the M1 schema."""

    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields, dt = make_scenarios()
    output = reference_step_numpy(fields, dt)
    payload = {
        "input_T": fields["T"],
        "input_p": fields["p"],
        "input_rho": _rho(fields["p"], fields["T"], fields["qv"]),
        "input_dt": np.asarray([dt], dtype=np.float64),
    }
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr"):
        payload[f"input_{name}"] = fields[name]
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T"):
        payload[f"output_{name}"] = output[name]
    np.savez_compressed(SAMPLE, **payload)
    sample_bytes = SAMPLE.stat().st_size
    if sample_bytes > 100_000:
        raise RuntimeError(f"{SAMPLE} is {sample_bytes} bytes, over the schema limit")

    shape = tuple(fields["T"].shape)
    variables = [
        _variable("input_T", "K", shape, 1.0e-10, 1.0e-12, "fp64 synthetic thermodynamic input"),
        _variable("input_p", "Pa", shape, 1.0e-8, 1.0e-12, "fp64 synthetic pressure input"),
        _variable("input_rho", "kg m-3", shape, 1.0e-12, 1.0e-12, "rho computed with WRF mp_gt_driver formula"),
        _variable("input_dt", "s", (1,), 0.0, 0.0, "static Thompson timestep"),
    ]
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        variables.append(_variable(f"input_{name}", "kg kg-1", shape, 1.0e-12, 1.0e-12, "fp64 hydrometeor input"))
    for name in ("Ni", "Nr"):
        variables.append(_variable(f"input_{name}", "kg-1", shape, 1.0e-3, 1.0e-6, "fp64 number concentration input"))
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        variables.append(_variable(f"output_{name}", "kg kg-1", shape, 1.0e-10, 1.0e-8, "ADR-005 hydrometeor tolerance"))
    for name in ("Ni", "Nr"):
        variables.append(_variable(f"output_{name}", "kg-1", shape, 1.0e-3, 1.0e-6, "ADR-005 number concentration tolerance"))
    variables.append(_variable("output_T", "K", shape, 1.0e-8, 1.0e-10, "fp64 latent heating tolerance"))

    manifest = {
        "fixture_id": FIXTURE_ID,
        "source": "analytic",
        "source_commit": "module_mp_thompson.F.pre Path-B-strict formulas lines 2040-2064, 2207-2268, 2450-2464, 2623-2675, 2709-2770, 2845-2889, 2967-3260, 3456-3636, 4007-4142, 5444-5495",
        "wrf_version": "v4.7.1",
        "scenario": "three Thompson source/sink columns: maritime shallow cloud, cold mixed-phase, precipitating column; sedimentation disabled by construction",
        "created_utc": "2026-05-20T02:42:06Z",
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": "python scripts/m5_generate_thompson_fixture.py",
        "external_uri": None,
        "sample_slice_path": "fixtures/samples/analytic-thompson-column-v1.npz",
        "git_commit": _git_rev(),
        "license_notes": "Synthetic fixture generated in-repository from WRF Thompson source formulas; WRF source itself is not redistributed here.",
        "variables": variables,
        "files": [
            {
                "path": "fixtures/samples/analytic-thompson-column-v1.npz",
                "checksum_sha256": _sha256(SAMPLE),
                "bytes": sample_bytes,
                "external": False,
            }
        ],
    }
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {
        "sample": str(SAMPLE.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "bytes": sample_bytes,
        "sha256": _sha256(SAMPLE),
        "path": "B-strict",
        "path_a_investigation": "no reusable module_mp_thompson object/module found under ../wrf_gpu or /mnt/data/wrf_gpu2; direct wrapper compile would require WRF module dependencies outside this worker scope",
        "wrf_source_exists": WRF_SOURCE.exists(),
    }


def main() -> int:
    """CLI entry point used by the sprint validation command."""

    os.environ.setdefault("JAX_ENABLE_X64", "true")
    record = write_fixture()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
