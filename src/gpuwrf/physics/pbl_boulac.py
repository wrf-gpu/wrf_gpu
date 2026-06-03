"""WRF Bougeault-Lacarrere PBL column kernel (``bl_pbl_physics=8``).

This module transcribes the no-BEP, no-urban single-column path in pristine WRF
``phys/module_bl_boulac.F``. Inputs are bottom-up mass-level columns. The public
batched entry is JAX-transformable and uses ``jax.vmap`` over columns with
``jax.lax`` loops over vertical levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsCarry, PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency
from gpuwrf.physics.tridiagonal_solver import solve_tridiagonal


config.update("jax_enable_x64", True)


CK_B = 0.4
CEPS_B = 1.0 / 1.4
TEMIN = 1.0e-4
TH0_REF = 300.0
CP_DEFAULT = 7.0 * 287.0 / 2.0
G_DEFAULT = 9.81


@dataclass(frozen=True)
class BouLacDiagnostics:
    pblh: jax.Array
    tke: jax.Array
    dlk: jax.Array
    exch_h: jax.Array
    exch_m: jax.Array


def _edge_heights(dz):
    return jnp.concatenate((jnp.zeros((1,), dtype=dz.dtype), jnp.cumsum(dz)))


def _rhoz(rho, dz):
    interior = (rho[1:] * dz[:-1] + rho[:-1] * dz[1:]) / (dz[:-1] + dz[1:])
    return jnp.concatenate((rho[:1], interior, rho[-1:]))


def _pbl_height(dz, z, theta, qv):
    zc = z[:-1] + 0.5 * dz
    thv = theta * (1.0 + 0.61 * qv)
    thsfc = thv[0] + 0.5

    def body(pblh, k):
        denom = jnp.maximum(0.01, thv[k] - thv[k - 1])
        candidate = zc[k - 1] + (thsfc - thv[k - 1]) / denom * (zc[k] - zc[k - 1])
        hit = jnp.logical_and(pblh == 0.0, thv[k] > thsfc)
        return jnp.where(hit, candidate, pblh), None

    pblh, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype=theta.dtype), jnp.arange(1, theta.shape[0]))
    return pblh


def _dissip_level(k, g, z, dz, tke, theta):
    beta = g / TH0_REF
    center = z[k] + 0.5 * dz[k]
    top = z[-1]
    dlu0 = top - z[k] - 0.5 * dz[k]
    dld0 = center
    te_k = tke[k]
    th_k = theta[k]

    def up_body(j, carry):
        dlu, zup, zup_inf, zzz = carry
        dzt = 0.5 * (dz[j + 1] + dz[j])
        zup_new = zup - beta * th_k * dzt + beta * (theta[j + 1] + theta[j]) * dzt * 0.5
        zzz_new = zzz + dzt
        bbb = (theta[j + 1] - theta[j]) / dzt
        discr = jnp.maximum(0.0, (beta * (theta[j] - th_k)) ** 2 + 2.0 * bbb * beta * (te_k - zup_inf))
        tl_grad = (-beta * (theta[j] - th_k) + jnp.sqrt(discr)) / (bbb * beta)
        tl_flat = jnp.where(theta[j] != th_k, (te_k - zup_inf) / (beta * (theta[j] - th_k)), 0.0)
        tl = jnp.where(bbb != 0.0, tl_grad, tl_flat)
        hit = jnp.logical_and(te_k < zup_new, te_k >= zup_inf)
        dlu_new = jnp.where(hit, zzz_new - dzt + tl, dlu)
        return dlu_new, zup_new, zup_new, zzz_new

    dlu, _, _, _ = jax.lax.fori_loop(
        k,
        theta.shape[0] - 1,
        up_body,
        (dlu0, jnp.asarray(0.0, theta.dtype), jnp.asarray(0.0, theta.dtype), jnp.asarray(0.0, theta.dtype)),
    )

    def down_body(n, carry):
        dld, zdo, zdo_sup, zzz = carry
        j = k - n
        dzt = 0.5 * (dz[j - 1] + dz[j])
        zdo_new = zdo + beta * th_k * dzt - beta * (theta[j - 1] + theta[j]) * dzt * 0.5
        zzz_new = zzz + dzt
        bbb = (theta[j] - theta[j - 1]) / dzt
        discr = jnp.maximum(0.0, (beta * (theta[j] - th_k)) ** 2 + 2.0 * bbb * beta * (te_k - zdo_sup))
        tl_grad = (beta * (theta[j] - th_k) + jnp.sqrt(discr)) / (bbb * beta)
        tl_flat = jnp.where(theta[j] != th_k, (te_k - zdo_sup) / (beta * (theta[j] - th_k)), 0.0)
        tl = jnp.where(bbb != 0.0, tl_grad, tl_flat)
        hit = jnp.logical_and(te_k < zdo_new, te_k >= zdo_sup)
        dld_new = jnp.where(hit, zzz_new - dzt + tl, dld)
        return dld_new, zdo_new, zdo_new, zzz_new

    dld, _, _, _ = jax.lax.fori_loop(
        0,
        k,
        down_body,
        (dld0, jnp.asarray(0.0, theta.dtype), jnp.asarray(0.0, theta.dtype), jnp.asarray(0.0, theta.dtype)),
    )
    return dlu, dld


def _dissip_bougeault(g, z, dz, tke, theta):
    levels = jnp.arange(theta.shape[0])
    dlu, dld = jax.vmap(lambda k: _dissip_level(k, g, z, dz, tke, theta))(levels)
    return dlu, dld


def _length_bougeault(dlu, dld, dlg):
    dld_limited = jnp.minimum(dld, dlg)
    dls = jnp.sqrt(dlu * dld_limited)
    dlk = jnp.minimum(dlu, dld_limited)
    return dls, dlk


def _cdtur_bougeault(tke, dz, dlk):
    te_m = (tke[:-1] * dz[1:] + tke[1:] * dz[:-1]) / (dz[1:] + dz[:-1])
    dlk_m = (dlk[:-1] * dz[1:] + dlk[1:] * dz[:-1]) / (dz[1:] + dz[:-1])
    interior = jnp.maximum(CK_B * dlk_m * jnp.sqrt(te_m), 0.1)
    return jnp.concatenate((jnp.zeros((1,), dtype=tke.dtype), interior, jnp.asarray([0.1], dtype=tke.dtype)))


def _shear_bougeault(u, v, dz, exch):
    u2 = (dz[2:] * u[1:-1] + dz[1:-1] * u[2:]) / (dz[1:-1] + dz[2:])
    u1 = (dz[1:-1] * u[:-2] + dz[:-2] * u[1:-1]) / (dz[:-2] + dz[1:-1])
    v2 = (dz[2:] * v[1:-1] + dz[1:-1] * v[2:]) / (dz[1:-1] + dz[2:])
    v1 = (dz[1:-1] * v[:-2] + dz[:-2] * v[1:-1]) / (dz[:-2] + dz[1:-1])
    cdm = 0.5 * (exch[1:-2] + exch[2:-1])
    dumdz = ((u2 - u1) / dz[1:-1]) ** 2 + ((v2 - v1) / dz[1:-1]) ** 2
    inner = cdm * dumdz
    return jnp.zeros_like(u).at[1:-1].set(inner)


def _buoy_bougeault(theta, dz, exch, gamma, g):
    dtdz1 = 2.0 * (theta[1:-1] - theta[:-2]) / (dz[:-2] + dz[1:-1])
    dtdz2 = 2.0 * (theta[2:] - theta[1:-1]) / (dz[2:] + dz[1:-1])
    dtmdz = 0.5 * (dtdz1 + dtdz2)
    cdm = 0.5 * (exch[1:-2] + exch[2:-1])
    gammam = 0.5 * (gamma[1:-2] + gamma[2:-1])
    inner = -cdm * (dtmdz - gammam) * g / TH0_REF
    return jnp.zeros_like(theta).at[1:-1].set(inner)


def _diff_boulac(co, rho, rhoz, cd, aa, bb, dz, dt):
    cddz_surface = rhoz[0] * cd[0] / dz[0]
    cddz_interior = 2.0 * rhoz[1:-1] * cd[1:-1] / (dz[1:] + dz[:-1])
    cddz_top = rhoz[-1] * cd[-1] / dz[-1]
    cddz = jnp.concatenate((cddz_surface[None], cddz_interior, cddz_top[None]))

    nz = co.shape[0]
    coef = dt / (dz[:-1] * rho[:-1])
    lower_vals = -cddz[:-2] * coef
    upper_vals = -cddz[1:-1] * coef
    diag_vals = 1.0 + (cddz[:-2] + cddz[1:-1]) * coef - aa[:-1] * dt
    rhs_vals = co[:-1] + bb[:-1] * dt

    lower = jnp.zeros_like(co).at[:-1].set(lower_vals)
    diag = jnp.ones_like(co).at[:-1].set(diag_vals)
    upper = jnp.zeros_like(co).at[:-1].set(upper_vals)
    rhs = co.at[:-1].set(rhs_vals)

    new = solve_tridiagonal(lower, diag, upper, rhs)
    flux_tail = -(cddz[1:-1] * (new[1:] - new[:-1])) / rho[1:]
    flux = jnp.concatenate((jnp.zeros((1,), dtype=co.dtype), flux_tail))
    return new, flux


def _tke_source_terms(u, v, theta, tke, dz, exch, dls, gamma, g):
    sh = _shear_bougeault(u, v, dz, exch)
    bu = _buoy_bougeault(theta, dz, exch, gamma, g)
    td = -CEPS_B * jnp.sqrt(jnp.maximum(tke, TEMIN)) / jnp.maximum(dls, 0.1)
    return td, sh + bu, sh, bu


def _boulac_column(u, v, theta, qv, qc, rho, dz, tke, hfx, qfx, ust, dt, cp, g):
    z = _edge_heights(dz)
    rhoz = _rhoz(rho, dz)
    dlg = 0.5 * (z[:-1] + z[1:])

    pblh = _pbl_height(dz, z, theta, qv)
    wts = jnp.maximum(0.0, hfx / rho[0] / cp)
    wstar = (g * wts * pblh / TH0_REF) ** (1.0 / 3.0)
    tstar_w = jnp.where(wts != 0.0, wts / wstar, 0.0)
    del tstar_w  # WRF computes this for disabled variance diagnostics in this path.

    face_idx = jnp.arange(dz.shape[0] + 1)
    gamma_value = 10.0 * wts / wstar / pblh
    gamma_active = (face_idx >= 1) & (face_idx <= dz.shape[0] - 1) & (z <= pblh) & (wts > 0.0)
    gamma = jnp.where(gamma_active, gamma_value, 0.0)

    dlu, dld = _dissip_bougeault(g, z, dz, tke, theta)
    dls, dlk = _length_bougeault(dlu, dld, dlg)
    exch = _cdtur_bougeault(tke, dz, dlk)
    a_e, b_e, sh, bu = _tke_source_terms(u, v, theta, tke, dz, exch, dls, gamma, g)
    del sh, bu
    tke_new, _we = _diff_boulac(tke, rho, rhoz, exch, a_e, b_e, dz, dt)
    tke_new = jnp.maximum(tke_new, TEMIN)

    zeros = jnp.zeros_like(theta)
    wind0 = jnp.sqrt(u[0] * u[0] + v[0] * v[0])
    a_uv0 = -ust * ust / dz[0] / wind0
    a_u = zeros.at[0].set(a_uv0)
    a_v = zeros.at[0].set(a_uv0)
    b_t = zeros.at[0].set(hfx / dz[0] / rho[0] / cp)
    b_q = zeros.at[0].set(qfx / dz[0] / rho[0])

    gamma_flux_div = (exch[1:] * gamma[1:] - exch[:-1] * gamma[:-1]) / dz
    gamma_levels = (jnp.arange(theta.shape[0]) >= 1) & (z[:-1] <= pblh) & (wts > 0.0)
    b_t = b_t - jnp.where(gamma_levels, gamma_flux_div, 0.0)

    u_new, wu = _diff_boulac(u, rho, rhoz, exch, a_u, zeros, dz, dt)
    v_new, wv = _diff_boulac(v, rho, rhoz, exch, a_v, zeros, dz, dt)
    theta_new, wt = _diff_boulac(theta, rho, rhoz, exch, zeros, b_t, dz, dt)
    qv_new, wq = _diff_boulac(qv, rho, rhoz, exch, zeros, b_q, dz, dt)
    qc_new, _wqc = _diff_boulac(qc, rho, rhoz, exch, zeros, zeros, dz, dt)

    return (
        (u_new - u) / dt,
        (v_new - v) / dt,
        (theta_new - theta) / dt,
        (qv_new - qv) / dt,
        (qc_new - qc) / dt,
        tke_new,
        dlk,
        exch[:-1],
        exch[:-1],
        pblh,
        wu,
        wv,
        wt,
        wq,
    )


def boulac_columns(u, v, theta, qv, qc, rho, dz, tke, *, hfx, qfx, ust, dt, cp=CP_DEFAULT, g=G_DEFAULT):
    """Batched, jit/vmap-traceable BouLac over ``(ncol, nz)`` columns."""

    ncol = u.shape[0]
    dt_b = jnp.broadcast_to(jnp.asarray(dt, jnp.float64), (ncol,))
    cp_b = jnp.broadcast_to(jnp.asarray(cp, jnp.float64), (ncol,))
    g_b = jnp.broadcast_to(jnp.asarray(g, jnp.float64), (ncol,))
    out = jax.vmap(
        lambda u0, v0, th0, qv0, qc0, rho0, dz0, tke0, hfx0, qfx0, ust0, dt0, cp0, g0: _boulac_column(
            u0, v0, th0, qv0, qc0, rho0, dz0, tke0, hfx0, qfx0, ust0, dt0, cp0, g0
        )
    )(u, v, theta, qv, qc, rho, dz, tke, hfx, qfx, ust, dt_b, cp_b, g_b)
    (
        u_t,
        v_t,
        theta_t,
        qv_t,
        qc_t,
        tke_new,
        dlk,
        exch_h,
        exch_m,
        pblh,
        wu,
        wv,
        wt,
        wq,
    ) = out
    return {
        "u": u_t,
        "v": v_t,
        "theta": theta_t,
        "qv": qv_t,
        "qc": qc_t,
        "tke": tke_new,
        "dlk": dlk,
        "exch_h": exch_h,
        "exch_m": exch_m,
        "pblh": pblh,
        "wu": wu,
        "wv": wv,
        "wt": wt,
        "wq": wq,
    }


def step_boulac_column(
    u,
    v,
    theta,
    qv,
    qc,
    rho,
    dz,
    tke,
    *,
    hfx,
    qfx,
    ust,
    dt,
    cp=CP_DEFAULT,
    g=G_DEFAULT,
) -> PhysicsStepResult:
    """Run one WRF BouLac column and return frozen S0 tendency/diagnostics."""

    out = _boulac_column(
        jnp.asarray(u, jnp.float64),
        jnp.asarray(v, jnp.float64),
        jnp.asarray(theta, jnp.float64),
        jnp.asarray(qv, jnp.float64),
        jnp.asarray(qc, jnp.float64),
        jnp.asarray(rho, jnp.float64),
        jnp.asarray(dz, jnp.float64),
        jnp.asarray(tke, jnp.float64),
        jnp.asarray(hfx, jnp.float64),
        jnp.asarray(qfx, jnp.float64),
        jnp.asarray(ust, jnp.float64),
        jnp.asarray(dt, jnp.float64),
        jnp.asarray(cp, jnp.float64),
        jnp.asarray(g, jnp.float64),
    )
    (u_t, v_t, theta_t, qv_t, qc_t, tke_new, dlk, exch_h, exch_m, pblh, wu, wv, wt, wq) = out
    tendency = PhysicsTendency(
        state_tendencies={
            "u": u_t,
            "v": v_t,
            "theta": theta_t,
            "qv": qv_t,
            "qc": qc_t,
        }
    )
    tendency.validate_keys()
    carry = PhysicsCarry(pbl={"tke_pbl": tke_new})
    diagnostics = PhysicsDiagnostics(
        pbl={
            "pblh": pblh,
            "tke_pbl": tke_new,
            "dlk": dlk,
            "exch_h": exch_h,
            "exch_m": exch_m,
            "wu": wu,
            "wv": wv,
            "wt": wt,
            "wq": wq,
        }
    )
    return PhysicsStepResult(tendency=tendency, carry=carry, diagnostics=diagnostics)


__all__ = ["BouLacDiagnostics", "boulac_columns", "step_boulac_column"]
