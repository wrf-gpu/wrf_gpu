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

    Neighbour access is periodic (every dycore halo in this codebase is
    ``edge_type="periodic"``).  ``xx``/``yy`` index the full mass domain; the
    ``+1`` neighbour wraps via ``jnp.roll``.  For the staggered momentum fields
    (u has ``nx+1`` x-faces, v has ``ny+1`` y-faces) the "east"/"north" face of
    a mass cell is the next staggered index, which under periodicity equals the
    first face, so the function uses the staggered faces directly.
    """

    nz = int(inputs.theta.shape[0])
    ny = int(inputs.mu.shape[0])
    nx = int(inputs.mu.shape[1])
    yy = slice(0, ny)
    xx = slice(0, nx)
    # Periodic neighbour helpers for mass-point (and theta) fields.
    def _xp(field):  # i+1 (east), periodic
        return jnp.roll(field, -1, axis=-1)

    def _xm(field):  # i-1 (west), periodic
        return jnp.roll(field, 1, axis=-1)

    def _yp(field):  # j+1 (north), periodic
        return jnp.roll(field, -1, axis=-2)

    def _ym(field):  # j-1 (south), periodic
        return jnp.roll(field, 1, axis=-2)

    # Legacy slice names retained below were a haloed-interior assumption; the
    # periodic rewrite replaces every ``[..., xp]`` neighbour read with a roll.
    # Staggered momentum faces for a mass cell (periodic): u has nx+1 x-faces,
    # so west = u[:, :, :nx] (face i) and east = u[:, :, 1:nx+1] (face i+1);
    # v has ny+1 y-faces, so south = v[:, :ny, :] and north = v[:, 1:ny+1, :].
    u_west_faces = inputs.u[:, :, :nx]
    u_east_faces = inputs.u[:, :, 1 : nx + 1]
    u1_west_faces = inputs.u_1[:, :, :nx]
    u1_east_faces = inputs.u_1[:, :, 1 : nx + 1]
    muu_west = inputs.muu[:, :nx]
    muu_east = inputs.muu[:, 1 : nx + 1]
    msfuy_west = inputs.msfuy[:, :nx]
    msfuy_east = inputs.msfuy[:, 1 : nx + 1]
    v_south_faces = inputs.v[:, :ny, :]
    v_north_faces = inputs.v[:, 1 : ny + 1, :]
    v1_south_faces = inputs.v_1[:, :ny, :]
    v1_north_faces = inputs.v_1[:, 1 : ny + 1, :]
    muv_south = inputs.muv[:ny, :]
    muv_north = inputs.muv[1 : ny + 1, :]
    msfvxi_south = inputs.msfvx_inv[:ny, :]
    msfvxi_north = inputs.msfvx_inv[1 : ny + 1, :]

    dvdxi_levels = []
    for k in range(nz):
        c1 = inputs.c1h[k]
        c2 = inputs.c2h[k]
        v_north = v_north_faces[k] + (c1 * muv_north + c2) * v1_north_faces[k] * msfvxi_north
        v_south = v_south_faces[k] + (c1 * muv_south + c2) * v1_south_faces[k] * msfvxi_south
        u_east = u_east_faces[k] + (c1 * muu_east + c2) * u1_east_faces[k] / msfuy_east
        u_west = u_west_faces[k] + (c1 * muu_west + c2) * u1_west_faces[k] / msfuy_west
        dvdxi = inputs.msftx * inputs.msfty * (
            float(inputs.rdy) * (v_north - v_south) + float(inputs.rdx) * (u_east - u_west)
        )
        dvdxi_levels.append(dvdxi)
    dvdxi_stack = jnp.stack(dvdxi_levels, axis=0)
    dmdt = jnp.sum(inputs.dnw[:nz, None, None] * dvdxi_stack, axis=0)

    # WRF ``small_step_prep`` saves the RK-stage ``MU`` and advances a
    # small-step delta.  The operational carry encodes that delta as
    # ``muts - mut`` so callers can keep the physical perturbation in ``mu``.
    mu_work_old = inputs.muts[yy, xx] - inputs.mut[yy, xx]
    mu_save = inputs.mu[yy, xx] - mu_work_old
    # WRF advance_mu_t (module_small_step_em.F:1099-1104): DMDT = sum_k dnw(k)*dvdxi(k)
    # with WRF-SIGNED dnw (negative for normal eta), then MU += dts*(DMDT+MU_TEND).
    # F7G adopts the WRF-signed metric throughout, so ``dmdt`` here already carries
    # the WRF sign and the mass tendency is the literal WRF ``DMDT + MU_TEND`` --
    # NOT the previous positive-|dnw| ``-dmdt`` compensation.
    mu_tendency = dmdt + inputs.mu_tend[yy, xx]
    mu_work_new = mu_work_old + float(inputs.dts) * mu_tendency
    mu_new_i = mu_save + mu_work_new
    mudf_i = mu_tendency
    muts_i = inputs.mut[yy, xx] + mu_work_new
    muave_i = 0.5 * ((1.0 + float(inputs.epssm)) * mu_work_new + (1.0 - float(inputs.epssm)) * mu_work_old)

    mu_new = _update_2d(inputs.mu, mu_new_i, yy, xx)
    mudf_new = _update_2d(inputs.mudf, mudf_i, yy, xx)
    muts_new = _update_2d(inputs.muts, muts_i, yy, xx)
    muave_new = _update_2d(inputs.muave, muave_i, yy, xx)

    ww_i = inputs.ww
    ww_rows = [ww_i[0]]
    for kk in range(1, nz):
        k = kk - 1
        increment = inputs.dnw[kk - 1] * (
            inputs.c1h[k] * dmdt + dvdxi_stack[kk - 1] + inputs.c1h[k] * inputs.mu_tend
        ) / inputs.msfty
        ww_rows.append(ww_rows[-1] - increment)
    ww_rows.append(ww_i[nz])
    ww_updated = jnp.stack(ww_rows, axis=0)
    ww_updated = ww_updated.at[:nz].set(ww_updated[:nz] - inputs.ww_1[:nz])
    ww_new = ww_updated

    theta_i = inputs.theta + inputs.msfty[None, :, :] * float(inputs.dts) * inputs.theta_tend
    theta_ave_new = inputs.theta

    theta_flux_source = inputs.theta_1

    wdtn_rows = [jnp.zeros_like(mu_tendency)]
    for k in range(1, nz):
        face_theta = inputs.fnm[k] * theta_flux_source[k] + inputs.fnp[k] * theta_flux_source[k - 1]
        wdtn_rows.append(ww_updated[k] * face_theta)
    wdtn_rows.append(jnp.zeros_like(mu_tendency))
    wdtn = jnp.stack(wdtn_rows, axis=0)

    # u/v staggered faces (periodic, see above): east/north use the next face.
    u_west_t = inputs.u[:, :, :nx]
    u_east_t = inputs.u[:, :, 1 : nx + 1]
    v_south_t = inputs.v[:, :ny, :]
    v_north_t = inputs.v[:, 1 : ny + 1, :]

    theta_levels = []
    for k in range(nz):
        th = theta_flux_source[k]  # (ny, nx)
        th_e = _xp(th)  # i+1, periodic
        th_w = _xm(th)  # i-1, periodic
        th_n = _yp(th)  # j+1, periodic
        th_s = _ym(th)  # j-1, periodic
        v_flux = v_north_t[k] * (th_n + th) - v_south_t[k] * (th + th_s)
        u_flux = u_east_t[k] * (th_e + th) - u_west_t[k] * (th + th_w)
        tendency = inputs.msftx * (0.5 * float(inputs.rdy) * v_flux + 0.5 * float(inputs.rdx) * u_flux) + inputs.rdnw[k] * (
            wdtn[k + 1] - wdtn[k]
        )
        theta_levels.append(theta_i[k] - float(inputs.dts) * inputs.msfty * tendency)
    theta_new = jnp.stack(theta_levels, axis=0)

    return {
        "mu": mu_new,
        "mudf": mudf_new,
        "muts": muts_new,
        "muave": muave_new,
        "ww": ww_new,
        "theta": theta_new,
    }
