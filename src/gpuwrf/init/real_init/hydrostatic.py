"""S1 (Opus) — hydrostatic balance: perturbation P / PH / AL / ALT.

FROZEN ENTRY SIGNATURE. Implements the real.exe final hydrostatic-balance block
(module_initialize_real.F:3876-4044): given the interpolated dynamics + the dry
base state, produce the perturbation pressure, geopotential, and inverse density
that make the column hydrostatically consistent with the WRF model equations.

Algorithm (faithful spec):
  t_2 := theta - t0 already (perturbation theta).
  MU_2 = MU0 - MUB                                                 (:3881)
  Top-down integration of perturbation dry pressure p (real.exe:3902-3958):
    at k=top:  qtot=sum(moist); qvf2=1/(1+qtot); qvf1=qtot*qvf2;
       p[top] = -0.5*((c1f*MU_2)+qvf1*(c1f*MUB+c2f))/rdnw/qvf2     (:3935)
    downward: p[k]=p[k+1] - ((c1f*MU_2)+qvf1*(c1f*MUB+c2f))/qvf2/rdn[k+1] (:3953)
    alt[k]=(r_d/p1000mb)*(t_2[k]+t0)*qvf*((p+pb)/p1000mb)^cvpm     (:3937/:3955)
    al[k]=alt[k]-alb[k]; p_hyd[k]=p[k]+pb[k]                       (:3939-3940)
  Then geopotential ph_2 from hydrostatic eq (real.exe:3965-3994), the
  hybrid_opt branch using c1h/c2h/c3h/c4h + the dZ=-al*p*dlog(p) form (:3974-3987).
  (Optionally re-diagnose al from ph_2 via :4001-4013 and recompute p :4031-4036
  to match the model's exact post-substep relation; reproduce whichever branch
  real.exe takes for this config — confirm at impl by reading :3960-4040.)

Oracle: wrfinput P / PH / AL(diag) / T / MU(=MU_2); tols ``WRFINPUT_TOLS``. This
is THE hour-0-critical check: a small PH/P error here propagates into every
downstream forecast comparison.

FILE OWNERSHIP: S1 exclusive.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gpuwrf.init.real_init.types import (
    BaseStateColumns,
    CP,
    CPOVCV,
    CVPM,
    DynamicsInit,
    P1000MB,
    R_D,
    RVOVRD,
    T0,
    RealInitConfig,
    VerticalCoord1D,
)


def balance(
    config: RealInitConfig,
    vcoord: VerticalCoord1D,
    base: BaseStateColumns,
    dyn_seed: DynamicsInit,
) -> DynamicsInit:
    """Returns ``dyn_seed`` with hydrostatically-balanced p/ph/al/alt/p_hyd/mu.

    Takes the pre-balance interpolated dynamics from :func:`vinterp.vertical_interpolate`
    plus the dry base state from :func:`base_state.compute_base_state`, and fills
    the final perturbation pressure, geopotential, and inverse densities.
    """

    qv = np.asarray(dyn_seed.qv, dtype=np.float64)
    theta = np.asarray(dyn_seed.theta, dtype=np.float64)
    mu0 = np.asarray(dyn_seed.mu0, dtype=np.float64)
    mub = np.asarray(base.mub, dtype=np.float64)
    mu = mu0 - mub
    p_top = float(config.p_top_pa)
    nz = config.nz

    p = np.empty_like(base.pb, dtype=np.float64)
    alt = np.empty_like(base.pb, dtype=np.float64)
    al = np.empty_like(base.pb, dtype=np.float64)
    p_hyd = np.empty_like(base.pb, dtype=np.float64)

    top = nz - 1
    qtot = qv[top]
    qvf2 = 1.0 / (1.0 + qtot)
    qvf1 = qtot * qvf2
    p[top] = (
        -0.5
        * ((vcoord.c1f[nz] * mu) + qvf1 * (vcoord.c1f[nz] * mub + vcoord.c2f[nz]))
        / vcoord.rdnw[top]
        / qvf2
    )
    qvf = 1.0 + RVOVRD * qv[top]
    alt[top] = (
        (R_D / P1000MB)
        * (theta[top] + T0)
        * qvf
        * (((p[top] + base.pb[top]) / P1000MB) ** CVPM)
    )
    al[top] = alt[top] - base.alb[top]
    p_hyd[top] = p[top] + base.pb[top]

    for kk in range(nz - 2, -1, -1):
        full_k = kk + 1
        qtot = 0.5 * (qv[kk] + qv[kk + 1])
        qvf2 = 1.0 / (1.0 + qtot)
        qvf1 = qtot * qvf2
        p[kk] = p[kk + 1] - (
            (vcoord.c1f[full_k] * mu)
            + qvf1 * (vcoord.c1f[full_k] * mub + vcoord.c2f[full_k])
        ) / qvf2 / vcoord.rdn[kk + 1]
        qvf = 1.0 + RVOVRD * qv[kk]
        alt[kk] = (
            (R_D / P1000MB)
            * (theta[kk] + T0)
            * qvf
            * (((p[kk] + base.pb[kk]) / P1000MB) ** CVPM)
        )
        al[kk] = alt[kk] - base.alb[kk]
        p_hyd[kk] = p[kk] + base.pb[kk]

    ph = _compute_ph(config, vcoord, base, mu0, mu, al, alt)

    if config.hybrid_opt == 0:
        for k in range(nz):
            denom = (vcoord.c1h[k] * mub + vcoord.c2h[k]) + vcoord.c1h[k] * mu
            al[k] = -(
                base.alb[k] * (vcoord.c1h[k] * mu)
                + vcoord.rdnw[k] * (ph[k + 1] - ph[k])
            ) / denom
    else:
        for k in range(nz):
            pfu = vcoord.c3f[k + 1] * (mub + mu) + vcoord.c4f[k + 1] + p_top
            pfd = vcoord.c3f[k] * (mub + mu) + vcoord.c4f[k] + p_top
            phm = vcoord.c3h[k] * (mub + mu) + vcoord.c4h[k] + p_top
            al[k] = (
                (ph[k + 1] - ph[k] + base.phb[k + 1] - base.phb[k])
                / phm
                / np.log(pfd / pfu)
                - base.alb[k]
            )

    for k in range(nz):
        qvf = 1.0 + RVOVRD * qv[k]
        p[k] = (
            P1000MB
            * (
                (R_D * (T0 + theta[k]) * qvf)
                / (P1000MB * (al[k] + base.alb[k]))
            )
            ** CPOVCV
            - base.pb[k]
        )
        p_hyd[k] = p[k] + base.pb[k]
        alt[k] = al[k] + base.alb[k]

    qv_final = _final_qv_from_seed_rh(theta, qv, dyn_seed.p, p_hyd)
    for k in range(nz):
        qvf = 1.0 + RVOVRD * qv_final[k]
        p[k] = (
            P1000MB
            * (
                (R_D * (T0 + theta[k]) * qvf)
                / (P1000MB * (al[k] + base.alb[k]))
            )
            ** CPOVCV
            - base.pb[k]
        )
        p_hyd[k] = p[k] + base.pb[k]

    return replace(
        dyn_seed,
        qv=qv_final,
        mu=mu,
        p=p,
        ph=ph,
        al=al,
        alt=alt,
        p_hyd=p_hyd,
    )


def _compute_ph(
    config: RealInitConfig,
    vcoord: VerticalCoord1D,
    base: BaseStateColumns,
    mu0: np.ndarray,
    mu: np.ndarray,
    al: np.ndarray,
    alt: np.ndarray,
) -> np.ndarray:
    nz = config.nz
    p_top = float(config.p_top_pa)
    ph_total = np.empty_like(base.phb, dtype=np.float64)
    if config.hybrid_opt == 0:
        ph = np.empty_like(base.phb, dtype=np.float64)
        ph[0] = 0.0
        for k in range(1, nz + 1):
            h = k - 1
            ph[k] = ph[k - 1] - vcoord.dnw[h] * (
                (
                    (vcoord.c1h[h] * base.mub + vcoord.c2h[h])
                    + vcoord.c1h[h] * mu
                )
                * al[h]
                + (vcoord.c1h[h] * mu) * base.alb[h]
            )
        return ph

    ph_total[0] = base.phb[0]
    for k in range(1, nz + 1):
        h = k - 1
        pfu = vcoord.c3f[k] * mu0 + vcoord.c4f[k] + p_top
        pfd = vcoord.c3f[k - 1] * mu0 + vcoord.c4f[k - 1] + p_top
        phm = vcoord.c3h[h] * mu0 + vcoord.c4h[h] + p_top
        ph_total[k] = ph_total[k - 1] + alt[h] * phm * np.log(pfd / pfu)
    return ph_total - base.phb


def _final_qv_from_seed_rh(
    theta: np.ndarray,
    qv_seed: np.ndarray,
    seed_pressure: np.ndarray,
    hyd_pressure: np.ndarray,
) -> np.ndarray:
    seed_temperature = _temperature_from_theta(theta, seed_pressure)
    rh = _relative_humidity_from_mixing_ratio(qv_seed, seed_temperature, seed_pressure)
    hyd_temperature = _temperature_from_theta(theta, hyd_pressure)
    return _mixing_ratio_from_rh_liquid(rh, hyd_temperature, hyd_pressure)


def _temperature_from_theta(theta: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    return (theta + T0) / ((P1000MB / pressure) ** (R_D / CP))


def _relative_humidity_from_mixing_ratio(
    qv: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    sat_vap_pres_mb = 0.6112 * 10.0 * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )
    vap_pres_mb = qv * pressure / 100.0 / (qv + 0.622)
    return np.where(sat_vap_pres_mb > 0.0, (vap_pres_mb / sat_vap_pres_mb) * 100.0, 0.0)


def _mixing_ratio_from_rh_liquid(
    rh: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    rh_bounded = np.minimum(np.maximum(rh, 0.0), 100.0)
    es = (
        0.01
        * rh_bounded
        * 0.6112
        * 10.0
        * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
    )
    q = np.where(
        es >= pressure / 100.0,
        1.0e-6,
        np.maximum(0.622 * es / (pressure / 100.0 - es), 1.0e-6),
    )
    q = np.where((pressure < 10000.0) & (q > 1.0e-5), 3.0e-6, q)
    q = np.where((pressure < 110000.0) & (q < 1.0e-6), 1.0e-6, q)
    return q
