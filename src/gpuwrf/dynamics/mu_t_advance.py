"""WRF-shaped ``advance_mu_t`` helper for savepoint parity comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class AdvanceMuTInputs:
    """Arrays consumed by WRF ``module_small_step_em.F:969-1175``."""

    ww: jnp.ndarray
    ww_1: jnp.ndarray
    u: jnp.ndarray
    u_1: jnp.ndarray
    v: jnp.ndarray
    v_1: jnp.ndarray
    mu: jnp.ndarray
    mut: jnp.ndarray
    muave: jnp.ndarray
    muts: jnp.ndarray
    muu: jnp.ndarray
    muv: jnp.ndarray
    mudf: jnp.ndarray
    theta: jnp.ndarray
    theta_1: jnp.ndarray
    theta_ave: jnp.ndarray
    theta_tend: jnp.ndarray
    mu_tend: jnp.ndarray
    dnw: jnp.ndarray
    fnm: jnp.ndarray
    fnp: jnp.ndarray
    rdnw: jnp.ndarray
    c1h: jnp.ndarray
    c2h: jnp.ndarray
    msfuy: jnp.ndarray
    msfvx_inv: jnp.ndarray
    msftx: jnp.ndarray
    msfty: jnp.ndarray
    rdx: float
    rdy: float
    dts: float
    epssm: float


def _interior_2d(array: jnp.ndarray) -> tuple[slice, slice]:
    if array.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {array.shape}")
    if array.shape[0] < 3 or array.shape[1] < 3:
        return slice(0, array.shape[0]), slice(0, array.shape[1])
    return slice(1, -1), slice(1, -1)


def _update_2d(base: jnp.ndarray, values: jnp.ndarray, ys: slice, xs: slice) -> jnp.ndarray:
    return base.at[ys, xs].set(values)


def _update_3d(base: jnp.ndarray, values: jnp.ndarray, ys: slice, xs: slice) -> jnp.ndarray:
    return base.at[:, ys, xs].set(values)


def advance_mu_t_wrf(inputs: AdvanceMuTInputs) -> dict[str, jnp.ndarray]:
    """Advance MU, MUTS, MUAVE, ``ww`` and theta like WRF ``advance_mu_t``.

    This is a comparator helper, not the production dycore path. It follows WRF
    ``dyn_em/module_small_step_em.F:1066-1175`` with arrays in Python order:
    ``(k, y, x)`` for 3-D fields, ``(y, x)`` for mass fields, u staggered on x,
    and v staggered on y.
    """

    ys, xs = _interior_2d(inputs.mu)
    y0, y1 = ys.indices(inputs.mu.shape[0])[:2]
    x0, x1 = xs.indices(inputs.mu.shape[1])[:2]
    yy = slice(y0, y1)
    xx = slice(x0, x1)
    yp = slice(y0 + 1, y1 + 1)
    xp = slice(x0 + 1, x1 + 1)
    ym = slice(y0 - 1, y1 - 1)
    xm = slice(x0 - 1, x1 - 1)

    nz = int(inputs.theta.shape[0])
    dvdxi_levels = []
    for k in range(nz):
        c1 = inputs.c1h[k]
        c2 = inputs.c2h[k]
        v_north = inputs.v[k, yp, xx] + (c1 * inputs.muv[yp, xx] + c2) * inputs.v_1[k, yp, xx] * inputs.msfvx_inv[yp, xx]
        v_south = inputs.v[k, yy, xx] + (c1 * inputs.muv[yy, xx] + c2) * inputs.v_1[k, yy, xx] * inputs.msfvx_inv[yy, xx]
        u_east = inputs.u[k, yy, xp] + (c1 * inputs.muu[yy, xp] + c2) * inputs.u_1[k, yy, xp] / inputs.msfuy[yy, xp]
        u_west = inputs.u[k, yy, xx] + (c1 * inputs.muu[yy, xx] + c2) * inputs.u_1[k, yy, xx] / inputs.msfuy[yy, xx]
        dvdxi = inputs.msftx[yy, xx] * inputs.msfty[yy, xx] * (
            float(inputs.rdy) * (v_north - v_south) + float(inputs.rdx) * (u_east - u_west)
        )
        dvdxi_levels.append(dvdxi)
    dvdxi_stack = jnp.stack(dvdxi_levels, axis=0)
    dmdt = jnp.sum(inputs.dnw[:nz, None, None] * dvdxi_stack, axis=0)

    mu_old = inputs.mu[yy, xx]
    mu_tendency = dmdt + inputs.mu_tend[yy, xx]
    mu_new_i = mu_old + float(inputs.dts) * mu_tendency
    mudf_i = mu_tendency
    muts_i = inputs.mut[yy, xx] + mu_new_i
    muave_i = 0.5 * ((1.0 + float(inputs.epssm)) * mu_new_i + (1.0 - float(inputs.epssm)) * mu_old)

    mu_new = _update_2d(inputs.mu, mu_new_i, yy, xx)
    mudf_new = _update_2d(inputs.mudf, mudf_i, yy, xx)
    muts_new = _update_2d(inputs.muts, muts_i, yy, xx)
    muave_new = _update_2d(inputs.muave, muave_i, yy, xx)

    ww_i = inputs.ww[:, yy, xx]
    ww_rows = [ww_i[0]]
    for kk in range(1, nz):
        k = kk - 1
        increment = inputs.dnw[kk - 1] * (
            inputs.c1h[k] * dmdt + dvdxi_stack[kk - 1] + inputs.c1h[k] * inputs.mu_tend[yy, xx]
        ) / inputs.msfty[yy, xx]
        ww_rows.append(ww_rows[-1] - increment)
    ww_rows.append(ww_i[nz])
    ww_updated = jnp.stack(ww_rows, axis=0)
    ww_updated = ww_updated.at[:nz].set(ww_updated[:nz] - inputs.ww_1[:nz, yy, xx])
    ww_new = _update_3d(inputs.ww, ww_updated, yy, xx)

    theta_i = inputs.theta[:, yy, xx] + inputs.msfty[yy, xx][None, :, :] * float(inputs.dts) * inputs.theta_tend[:, yy, xx]
    theta_ave_new = _update_3d(inputs.theta_ave, inputs.theta[:, yy, xx], yy, xx)

    theta_flux_source = inputs.theta_1

    wdtn_rows = [jnp.zeros_like(mu_tendency)]
    for k in range(1, nz):
        face_theta = inputs.fnm[k] * theta_flux_source[k, yy, xx] + inputs.fnp[k] * theta_flux_source[k - 1, yy, xx]
        wdtn_rows.append(ww_updated[k] * face_theta)
    wdtn_rows.append(jnp.zeros_like(mu_tendency))
    wdtn = jnp.stack(wdtn_rows, axis=0)

    theta_levels = []
    for k in range(nz):
        v_flux = inputs.v[k, yp, xx] * (theta_flux_source[k, yp, xx] + theta_flux_source[k, yy, xx]) - inputs.v[k, yy, xx] * (
            theta_flux_source[k, yy, xx] + theta_flux_source[k, ym, xx]
        )
        u_flux = inputs.u[k, yy, xp] * (theta_flux_source[k, yy, xp] + theta_flux_source[k, yy, xx]) - inputs.u[k, yy, xx] * (
            theta_flux_source[k, yy, xx] + theta_flux_source[k, yy, xm]
        )
        tendency = inputs.msftx[yy, xx] * (0.5 * float(inputs.rdy) * v_flux + 0.5 * float(inputs.rdx) * u_flux) + inputs.rdnw[k] * (
            wdtn[k + 1] - wdtn[k]
        )
        theta_levels.append(theta_i[k] - float(inputs.dts) * inputs.msfty[yy, xx] * tendency)
    theta_new = _update_3d(inputs.theta, jnp.stack(theta_levels, axis=0), yy, xx)

    return {
        "mu": mu_new,
        "mudf": mudf_new,
        "muts": muts_new,
        "muave": muave_new,
        "ww": ww_new,
        "theta": theta_new,
    }
