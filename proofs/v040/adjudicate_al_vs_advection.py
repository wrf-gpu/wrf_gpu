"""Opus MAX round adjudication: agy `al` mass-swap vs GPT momentum-advection BC gap.

Two candidate drivers of the standalone U10+ / PSFC- bias:
  (A) agy: `_absolute_diagnostics` `al` uses base column mass (mub) in the
      denominator and total inverse density (alt) on the mu' term, where WRF
      `calc_p_rho_phi` uses TOTAL column mass (muts) and BASE inverse density
      (alb).  We quantify the al error and its downstream PGF impact.
  (B) GPT: momentum advect_u/v use the PERIODIC flux5 path; WRF degrades to
      2nd/3rd order with upstream boundary fluxes for specified/nested d01.

We compute, on the identical real-d01 savepoint state both prior rounds used:
  * al_jax vs al_wrf (agy fix) -> relative error, PGF dpx/dpy delta.
  * JAX-periodic momentum advection vs WRF-specified NumPy oracle (GPT), with
    the residual split into interior vs LBC ring so we see where it lives.
  * The KEY discriminator: the *implied acceleration* (tendency / mass) and its
    DOMAIN-MEAN signed value per term, because the forecast bias is a slow
    systematic mean drift, not a local spike.  A local boundary-ring spike that
    averages to ~0 over the interior cannot be the mean-bias driver; a term with
    a non-zero interior signed mean can.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT / "proofs" / "v040", ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import config as jax_config  # noqa: E402

jax_config.update("jax_enable_x64", True)

import netCDF4  # noqa: E402

from gpuwrf.dynamics.metrics import load_wrfinput_metrics  # noqa: E402
from gpuwrf.dynamics.acoustic_wrf import (  # noqa: E402
    _inverse_density_from_theta_pressure,
    moisture_coupling_factors,
)
from gpuwrf.dynamics.core.rk_addtend_dry import large_step_horizontal_pgf  # noqa: E402
from gpuwrf.dynamics.flux_advection import (  # noqa: E402
    advect_u_flux,
    advect_v_flux,
    couple_velocities_periodic,
)

from lbc_budget_trace import _state_from_wrfout  # noqa: E402
from pgf_inloop_isolation import (  # noqa: E402
    _metrics_np,
    _state_np,
    wrf_advect_u_components_np,
    wrf_advect_v_components_np,
    wrf_transport_np,
)

TARGET_METADATA = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260429_18z_l2_72h_20260524T204451Z"
)
TARGET_WRFOUT = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/"
    "20260429_18z_l2_72h_20260524T204451Z/wrfout_d01_2026-04-29_18:00:00"
)


def _np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "absmean": float(np.mean(np.abs(a))),
        "rmse": float(np.sqrt(np.mean(a * a))),
        "max_abs": float(np.max(np.abs(a))),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _interior_mask(shape_2d: tuple[int, int], ring: int) -> np.ndarray:
    m = np.ones(shape_2d, dtype=bool)
    m[:ring, :] = False
    m[-ring:, :] = False
    m[:, :ring] = False
    m[:, -ring:] = False
    return m


def al_jax(state, metrics):
    """Reproduce the current JAX `_absolute_diagnostics` al (rk_addtend_dry:116-128)."""
    ph_pert = _np(state.ph_perturbation)
    mu_pert = _np(state.mu_perturbation)
    mut = _np(state.mu_total) - _np(state.mu_perturbation)  # = mub (base)
    alt = _np(_inverse_density_from_theta_pressure(
        jnp.asarray(state.theta, dtype=jnp.float64),
        jnp.asarray(state.p_total, dtype=jnp.float64),
    ))
    c1h = _np(metrics.c1h)[:, None, None]
    c2h = _np(metrics.c2h)[:, None, None]
    rdnw = _np(metrics.rdnw)[:, None, None]
    mass_h = c1h * mut[None, :, :] + c2h
    safe = np.where(np.abs(mass_h) > 1.0e-12, mass_h, 1.0e-12)
    mu_term = c1h * mu_pert[None, :, :]
    al = -(alt * mu_term + rdnw * (ph_pert[1:] - ph_pert[:-1])) / safe
    return al, alt, mut


def al_wrf(state, metrics):
    """WRF calc_p_rho_phi al (module_big_step_utilities_em.F:1029).

    al = -1/(c1*muts+c2) * ( alb*(c1*mu_pert) + rdnw*(ph(k+1)-ph(k)) )
    muts = mut + mu_2 (solve_em.F:3610) = (mub+mu_2) + mu_2 = mub + 2*mu_2.
    BUT note: at rk_step_prep entry WRF uses grid%muts which is the *total moist*
    surface mass.  The faithful relation in the dry-mass diagnostic context that
    matters for `al`: WRF passes grid%muts = grid%mut + grid%mu_2.  We compute it
    that way to match the binary; we also report a variant with muts=mut (the
    commonly-stated total dry mass) to bound the sensitivity.
    alb = base inverse density = alt - al (al is perturbation inverse density), but
    for a faithful oracle alb is the hydrostatic base-state value:
        alb = -1/(c1*mub+c2) * rdnw*(phb(k+1)-phb(k))   (WRF init, calc of alb).
    """
    ph_pert = _np(state.ph_perturbation)
    ph_total = _np(state.ph_total)
    phb = ph_total - ph_pert
    mu_pert = _np(state.mu_perturbation)
    mu_total = _np(state.mu_total)  # = grid%mut = mub + mu_2
    mub = mu_total - mu_pert
    c1h = _np(metrics.c1h)[:, None, None]
    c2h = _np(metrics.c2h)[:, None, None]
    rdnw = _np(metrics.rdnw)[:, None, None]
    dnw = _np(metrics.dnw)[:, None, None]

    # base-state inverse density alb from the hydrostatic base column:
    #   d(phb)/dnu = -alb * (c1*mub + c2)   ->  alb = -rdnw*dphb / (c1*mub+c2)
    mass_b = c1h * mub[None, :, :] + c2h
    safe_b = np.where(np.abs(mass_b) > 1.0e-12, mass_b, 1.0e-12)
    alb = -rdnw * (phb[1:] - phb[:-1]) / safe_b

    # muts = grid%mut + grid%mu_2  (solve_em.F:3610)
    muts = mu_total + mu_pert
    mass_t = c1h * muts[None, :, :] + c2h
    safe_t = np.where(np.abs(mass_t) > 1.0e-12, mass_t, 1.0e-12)
    mu_term = c1h * mu_pert[None, :, :]
    al = -(alb * mu_term + rdnw * (ph_pert[1:] - ph_pert[:-1])) / safe_t

    # sensitivity variant: muts = mut (total dry mass only)
    mass_t2 = c1h * mu_total[None, :, :] + c2h
    safe_t2 = np.where(np.abs(mass_t2) > 1.0e-12, mass_t2, 1.0e-12)
    al_v2 = -(alb * mu_term + rdnw * (ph_pert[1:] - ph_pert[:-1])) / safe_t2
    return al, alb, muts, al_v2


def main() -> int:
    metrics = load_wrfinput_metrics(TARGET_METADATA / "wrfinput_d01")
    state = _state_from_wrfout(TARGET_WRFOUT)
    metrics_n = _metrics_np(metrics)
    state_n = _state_np(state)
    with netCDF4.Dataset(TARGET_METADATA / "wrfinput_d01") as ds:
        dx_m = float(getattr(ds, "DX"))
        dy_m = float(getattr(ds, "DY"))

    report: dict[str, Any] = {
        "schema": "v040_adjudicate_al_vs_advection.v1",
        "created_by": "Opus 4.8 MAX",
        "savepoint": str(TARGET_WRFOUT),
        "dx_m": dx_m,
        "dy_m": dy_m,
    }

    # ---------- (A) agy al adjudication ----------
    alj, altj, mutj = al_jax(state, metrics)
    alw, albw, muts, alw_v2 = al_wrf(state, metrics)
    al_err = alj - alw
    al_err_v2 = alj - alw_v2
    # relative error normalised by alt (full inverse density ~ 0.8-1.2 m3/kg)
    rel = al_err / np.maximum(np.abs(altj), 1e-12)
    report["agy_al"] = {
        "al_jax_stats": _stats(alj),
        "al_wrf_muts=mut+mu2_stats": _stats(alw),
        "al_wrf_muts=mut_stats": _stats(alw_v2),
        "al_error_jax_minus_wrf(muts=mut+mu2)": _stats(al_err),
        "al_error_jax_minus_wrf(muts=mut)": _stats(al_err_v2),
        "al_relative_error_over_alt": _stats(rel),
        "alt_stats": _stats(altj),
        "alb_wrf_stats": _stats(albw),
        "note": (
            "al error = (alt-alb)*c1*mu'/mass + al*c1*mu'/mass-type 2nd-order terms; "
            "expected sub-percent of alt."
        ),
    }

    # PGF impact of the al fix: build a corrected-al state and re-run PGF by
    # monkeypatching the perturbation-pressure path is intrusive; instead we
    # quantify the dpx/dpy sensitivity analytically.  The al term enters
    # pb_term = (al_l+al_r)*(pb_r-pb_l).  We recompute that single sub-term with
    # both al fields and report the dpx/dpy delta it implies.
    pb = _np(state.p_total) - _np(state.p_perturbation)
    c1h = _np(metrics.c1h)[:, None, None]
    c2h = _np(metrics.c2h)[:, None, None]
    mu_total = _np(state.mu_total)
    rdx = 1.0 / dx_m
    rdy = 1.0 / dy_m
    msf_u = _np(metrics.msfux / metrics.msfuy)[None, :, :]
    msf_v = _np(metrics.msfvy / metrics.msfvx)[None, :, :]
    cqu, cqv = (_np(c) for c in moisture_coupling_factors(state))

    def xpair3(f):  # mass cell -> u-faces (nx+1); edge pad
        p = np.pad(f, ((0, 0), (0, 0), (1, 1)), mode="edge")
        return p[:, :, :-1], p[:, :, 1:]

    def ypair3(f):
        p = np.pad(f, ((0, 0), (1, 1), (0, 0)), mode="edge")
        return p[:, :-1, :], p[:, 1:, :]

    def xpair2(f):
        p = np.pad(f, ((0, 0), (1, 1)), mode="edge")
        return p[:, :-1], p[:, 1:]

    def ypair2(f):
        p = np.pad(f, ((1, 1), (0, 0)), mode="edge")
        return p[:-1, :], p[1:, :]

    pb_l, pb_r = xpair3(pb)
    pb_s, pb_n = ypair3(pb)
    muu = 0.5 * sum(xpair2(mu_total))
    muv = 0.5 * sum(ypair2(mu_total))
    mass_u = c1h * muu[None, :, :] + c2h
    mass_v = c1h * muv[None, :, :] + c2h

    def pb_term_x(al):
        al_l, al_r = xpair3(al)
        return msf_u * 0.5 * rdx * mass_u * ((al_l + al_r) * (pb_r - pb_l))

    def pb_term_y(al):
        al_s, al_n = ypair3(al)
        return msf_v * 0.5 * rdy * mass_v * ((al_s + al_n) * (pb_n - pb_s))

    # dpx is the coupled tendency; -cqu*dpx is the ru_pgf contribution.
    d_rux = -cqu * (pb_term_x(alj) - pb_term_x(alw))
    d_rvy = -cqv * (pb_term_y(alj) - pb_term_y(alw))
    # convert to an implied acceleration: ru_pgf is coupled (mass*accel); divide
    # by mass_u/mass_v to get m/s^2 (the physically comparable signal).
    accel_x = d_rux / np.maximum(np.abs(mass_u), 1e-12)
    accel_y = d_rvy / np.maximum(np.abs(mass_v), 1e-12)
    report["agy_al_pgf_impact"] = {
        "delta_ru_pgf_coupled_stats": _stats(d_rux),
        "delta_rv_pgf_coupled_stats": _stats(d_rvy),
        "delta_implied_u_accel_m_s2_stats": _stats(accel_x),
        "delta_implied_v_accel_m_s2_stats": _stats(accel_y),
        "delta_u_accel_x_24h_m_s": _stats(accel_x * 86400.0),
        "note": (
            "pb_term al-fix delta in the u/v PGF; implied accel * 86400 s bounds "
            "the 24h velocity bias the al-swap alone could explain."
        ),
    }

    # ---------- (B) GPT momentum-advection BC gap ----------
    vel = couple_velocities_periodic(
        state.u, state.v, state.mu_total,
        c1h=metrics.c1h, c2h=metrics.c2h, dnw=metrics.dnw,
        rdx=rdx, rdy=rdy,
        msfuy=metrics.msfuy, msfvx=metrics.msfvx, msftx=metrics.msftx,
        msfux=metrics.msfux, msfvy=metrics.msfvy,
    )
    wrf_transport = wrf_transport_np(state_n, metrics_n, dx_m=dx_m, dy_m=dy_m)
    wrf_u = wrf_advect_u_components_np(state_n, wrf_transport, metrics_n, dx_m=dx_m, dy_m=dy_m)
    wrf_v = wrf_advect_v_components_np(state_n, wrf_transport, metrics_n, dx_m=dx_m, dy_m=dy_m)
    ju = _np(advect_u_flux(state.u, vel, rdx=rdx, rdy=rdy, rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp))
    jv = _np(advect_v_flux(state.v, vel, rdx=rdx, rdy=rdy, rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp))

    # mass to convert coupled adv tendency -> implied accel (m/s^2).
    mass_u_full = c1h * (0.5 * sum(xpair2(mu_total)))[None, :, :] + c2h  # (nz,ny,nx+1)
    mass_v_full = c1h * (0.5 * sum(ypair2(mu_total)))[None, :, :] + c2h  # (nz,ny+1,nx)

    du = ju - wrf_u["total"]
    dv = jv - wrf_v["total"]
    du_acc = du / np.maximum(np.abs(mass_u_full), 1e-12)
    dv_acc = dv / np.maximum(np.abs(mass_v_full), 1e-12)

    # interior vs LBC-ring split (ring width 5, matching prior rounds), at all k.
    ny_u, nx_u = du.shape[1], du.shape[2]
    ny_v, nx_v = dv.shape[1], dv.shape[2]
    int_u = _interior_mask((ny_u, nx_u), ring=5)[None, :, :]
    int_v = _interior_mask((ny_v, nx_v), ring=5)[None, :, :]
    int_u_full = np.broadcast_to(int_u, du.shape)
    int_v_full = np.broadcast_to(int_v, dv.shape)

    report["gpt_advection_bc_gap"] = {
        "u_total_coupled_resid_full": _stats(du),
        "v_total_coupled_resid_full": _stats(dv),
        "u_implied_accel_resid_full_m_s2": _stats(du_acc),
        "v_implied_accel_resid_full_m_s2": _stats(dv_acc),
        "u_implied_accel_resid_INTERIOR(ring5)_m_s2": _stats(du_acc[int_u_full]),
        "v_implied_accel_resid_INTERIOR(ring5)_m_s2": _stats(dv_acc[int_v_full]),
        "u_implied_accel_resid_LBCRING(width5)_m_s2": _stats(du_acc[~int_u_full]),
        "v_implied_accel_resid_LBCRING(width5)_m_s2": _stats(dv_acc[~int_v_full]),
        "u_resid_signed_mean_INTERIOR": float(np.mean(du[int_u_full])),
        "v_resid_signed_mean_INTERIOR": float(np.mean(dv[int_v_full])),
        "u_resid_signed_mean_LBCRING": float(np.mean(du[~int_u_full])),
        "v_resid_signed_mean_LBCRING": float(np.mean(dv[~int_v_full])),
        "note": (
            "Implied-accel residual = (JAX-periodic - WRF-specified) advection / face mass. "
            "INTERIOR signed mean drives slow mean bias; LBC-ring spikes are relaxed away by the "
            "lateral BC nudge each step and average out."
        ),
    }

    # 24h-scale bound on the interior advection accel residual.
    report["gpt_advection_24h_bound"] = {
        "u_interior_signed_mean_accel_x_24h_m_s": float(np.mean(du_acc[int_u_full]) * 86400.0),
        "v_interior_signed_mean_accel_x_24h_m_s": float(np.mean(dv_acc[int_v_full]) * 86400.0),
        "u_lbcring_signed_mean_accel_x_24h_m_s": float(np.mean(du_acc[~int_u_full]) * 86400.0),
    }

    # ---------- (C) FULLY WRF-FAITHFUL PGF vs JAX PGF -- interior signed mean ----------
    # GPT round-4 compared JAX PGF vs a NumPy oracle that SHARED the al-mass-swap and
    # the dpn cf1/cf2/cf3 top-face transcription, so its "PGF source-equivalent" verdict
    # could not see either bug (a self-compare for those terms).  Here we build a PGF
    # with the WRF-correct al (alb,muts) and dpn (cfn/cfn1 top) and measure the INTERIOR
    # signed-mean PGF residual -- the quantity that can drive a slow surface bias.
    jax_ru_pgf, jax_rv_pgf = large_step_horizontal_pgf(state, metrics, dx_m=dx_m, dy_m=dy_m, top_lid=True)
    jax_ru_pgf = _np(jax_ru_pgf)
    jax_rv_pgf = _np(jax_rv_pgf)

    # cfn/cfn1 from dn (mass-level spacing) and dnw: cfn=(0.5*dnw[nz-2]+dn[nz-2])/dn[nz-2];
    # cfn1=-0.5*dnw[nz-2]/dn[nz-2].  dn = 1/rdn; here metrics carry rdn.
    dnw_v = _np(metrics.dnw)
    rdn_v = _np(metrics.rdn)
    dn_v = np.where(np.abs(rdn_v) > 1e-30, 1.0 / np.where(rdn_v == 0, 1.0, rdn_v), 0.0)
    nzc = dnw_v.shape[0]
    # WRF kde-1 (top mass level) is JAX index nz-1 -> dnw[nz-1]; dn[kde-1] is dn at the
    # interface below the top.  WRF uses dn(kde-1) which is the mass spacing dn[nz-1].
    cfn = (0.5 * dnw_v[nzc - 1] + dn_v[nzc - 1]) / dn_v[nzc - 1] if dn_v[nzc - 1] != 0 else 0.0
    cfn1 = -0.5 * dnw_v[nzc - 1] / dn_v[nzc - 1] if dn_v[nzc - 1] != 0 else 0.0

    def pgf_faithful(use_al_fix: bool, use_dpn_fix: bool):
        c1hh = c1h
        c2hh = c2h
        rdnw = _np(metrics.rdnw)[:, None, None]
        ph = _np(state.ph_perturbation)
        p_abs = _np(state.p_perturbation)
        altf = _np(_inverse_density_from_theta_pressure(
            jnp.asarray(state.theta, dtype=jnp.float64),
            jnp.asarray(state.p_total, dtype=jnp.float64)))
        alf = alw if use_al_fix else alj  # WRF-correct vs JAX al
        phb_loc = _np(state.ph_total) - ph
        ph_total = phb_loc + ph
        php = 0.5 * (ph_total[:-1] + ph_total[1:])

        def _dpn(pair_sum):
            nz = pair_sum.shape[0]
            dpn = np.zeros((nz + 1,) + pair_sum.shape[1:])
            dpn[0] = 0.5 * (_np(metrics.cf1) * pair_sum[0] + _np(metrics.cf2) * pair_sum[1] + _np(metrics.cf3) * pair_sum[2])
            dpn[1:nz] = 0.5 * (_np(metrics.fnm)[1:, None, None] * pair_sum[1:] + _np(metrics.fnp)[1:, None, None] * pair_sum[:-1])
            # top_lid=True: WRF uses cfn/cfn1 (fix) vs JAX cf1/cf2/cf3 (current)
            if use_dpn_fix:
                dpn[nz] = 0.5 * (cfn * pair_sum[-1] + cfn1 * pair_sum[-2])
            else:
                dpn[nz] = 0.5 * (_np(metrics.cf1) * pair_sum[-1] + _np(metrics.cf2) * pair_sum[-2] + _np(metrics.cf3) * pair_sum[-3])
            return dpn

        # x
        ph_l, ph_r = xpair3(ph); p_l, p_r = xpair3(p_abs); pb_l, pb_r = xpair3(pb)
        al_l, al_r = xpair3(alf); alt_l, alt_r = xpair3(altf)
        ph_term = (ph_r[1:] - ph_l[1:]) + (ph_r[:-1] - ph_l[:-1])
        dpx = msf_u * 0.5 * rdx * mass_u * (ph_term + (alt_l + alt_r) * (p_r - p_l) + (al_l + al_r) * (pb_r - pb_l))
        php_l, php_r = xpair3(php); ps_l, ps_r = xpair3(p_abs)
        dpn = _dpn(ps_l + ps_r); mu_l, mu_r = xpair2(_np(state.mu_perturbation))
        br = rdnw * (dpn[1:] - dpn[:-1]) - 0.5 * (c1h * (mu_l + mu_r)[None, :, :])
        dpx = dpx + msf_u * rdx * (php_r - php_l) * br
        # y
        ph_s, ph_n = ypair3(ph); p_s, p_n = ypair3(p_abs); pb_s, pb_n = ypair3(pb)
        al_s, al_n = ypair3(alf); alt_s, alt_n = ypair3(altf)
        ph_term_y = (ph_n[1:] - ph_s[1:]) + (ph_n[:-1] - ph_s[:-1])
        dpy = msf_v * 0.5 * rdy * mass_v * (ph_term_y + (alt_s + alt_n) * (p_n - p_s) + (al_s + al_n) * (pb_n - pb_s))
        php_s, php_n = ypair3(php); ps_s, ps_n = ypair3(p_abs)
        dpn_y = _dpn(ps_s + ps_n); mu_s, mu_n = ypair2(_np(state.mu_perturbation))
        br_y = rdnw * (dpn_y[1:] - dpn_y[:-1]) - 0.5 * (c1h * (mu_s + mu_n)[None, :, :])
        dpy = dpy + msf_v * rdy * (php_n - php_s) * br_y
        return -cqu * dpx, -cqv * dpy

    ru_wrf, rv_wrf = pgf_faithful(use_al_fix=True, use_dpn_fix=True)
    ru_aonly, rv_aonly = pgf_faithful(use_al_fix=True, use_dpn_fix=False)
    ru_donly, rv_donly = pgf_faithful(use_al_fix=False, use_dpn_fix=True)

    int_pu = _interior_mask((jax_ru_pgf.shape[1], jax_ru_pgf.shape[2]), ring=5)[None, :, :]
    int_pv = _interior_mask((jax_rv_pgf.shape[1], jax_rv_pgf.shape[2]), ring=5)[None, :, :]
    int_pu_f = np.broadcast_to(int_pu, jax_ru_pgf.shape)
    int_pv_f = np.broadcast_to(int_pv, jax_rv_pgf.shape)
    mass_u_full = c1h * (0.5 * sum(xpair2(mu_total)))[None, :, :] + c2h
    mass_v_full = c1h * (0.5 * sum(ypair2(mu_total)))[None, :, :] + c2h

    def pgf_resid_block(ru_ref, rv_ref, tag):
        dru = jax_ru_pgf - ru_ref
        drv = jax_rv_pgf - rv_ref
        dru_acc = dru / np.maximum(np.abs(mass_u_full), 1e-12)
        drv_acc = drv / np.maximum(np.abs(mass_v_full), 1e-12)
        return {
            f"{tag}_du_accel_interior_signed_mean_m_s2": float(np.mean(dru_acc[int_pu_f])),
            f"{tag}_dv_accel_interior_signed_mean_m_s2": float(np.mean(drv_acc[int_pv_f])),
            f"{tag}_du_accel_interior_x24h_m_s": float(np.mean(dru_acc[int_pu_f]) * 86400.0),
            f"{tag}_dv_accel_interior_x24h_m_s": float(np.mean(drv_acc[int_pv_f]) * 86400.0),
            f"{tag}_du_accel_interior_rmse_m_s2": float(np.sqrt(np.mean(dru_acc[int_pu_f] ** 2))),
            f"{tag}_du_accel_LOWLEVEL_k0_2_interior_signed_mean": float(np.mean(dru_acc[:3][int_pu_f[:3]])),
        }

    pgf_block: dict[str, Any] = {
        "cfn": float(cfn), "cfn1": float(cfn1),
        "note": (
            "JAX PGF minus a FULLY WRF-faithful PGF (correct al=alb/muts + dpn cfn/cfn1 top). "
            "GPT round-4 oracle shared both transcription bugs, so its 'source-equivalent' verdict "
            "was a self-compare for these terms.  Interior signed-mean accel x 24h bounds the surface "
            "bias each correction can explain."
        ),
    }
    pgf_block.update(pgf_resid_block(ru_wrf, rv_wrf, "both_fixes"))
    pgf_block.update(pgf_resid_block(ru_aonly, rv_aonly, "al_fix_only"))
    pgf_block.update(pgf_resid_block(ru_donly, rv_donly, "dpn_fix_only"))
    report["pgf_faithful_interior"] = pgf_block

    out = Path(__file__).resolve().parent / "adjudicate_al_vs_advection.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["pgf_faithful_interior"], indent=2, sort_keys=True))
    print("--- agy_al_pgf_impact (signed means) ---")
    print("delta_u_accel mean m/s2:", report["agy_al_pgf_impact"]["delta_implied_u_accel_m_s2_stats"]["mean"])
    print("--- gpt_advection_24h_bound ---")
    print(json.dumps(report["gpt_advection_24h_bound"], indent=2))
    print(f"\nWROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
