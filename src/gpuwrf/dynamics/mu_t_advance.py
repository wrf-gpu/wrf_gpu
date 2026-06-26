"""WRF-shaped ``advance_mu_t`` helper for savepoint parity comparisons."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass

from jax import config
import jax.numpy as jnp


configure_jax_x64()


# Acoustic dry-mass positivity guard.  WRF advance_mu_t (module_small_step_em.F:
# 1103-1107) forms the total small-step dry mass MUTS = MUT + MU_work directly and
# every downstream routine divides by (c1h*MUTS+c2h) / uses alt(MUTS), ASSUMING
# MUTS stays physically positive.  Over the steepest d01 boundary-zone terrain the
# acoustic continuity loop pumps a pathological NEGATIVE mass tendency that drains
# MU_work until MUTS crosses 0 -> the mass denominators and the inverse density
# `alt` collapse -> c2a = cpovcv*(pb+p)/alt -> singular -> the implicit-w solve
# detonates (the Mont Blanc d01 blowup; downstream symptom is the Thompson Ni NaN).
# Bound the per-substep drain so MUTS cannot fall below MIN_MUTS_FRACTION of the
# stage MUT.  IDENTITY for every physical substep (an 18s/n_sound acoustic step
# changes the ~9e4 Pa column mass by O(1-100) Pa, never tens of percent); engages
# only in the pathological drain.
MIN_MUTS_FRACTION = 0.5
MIN_MUTS_PA = 1.0


def _limit_mu_drain_scale(mut, mu_work_old, mu_tendency, dts):
    """Scale in [0,1] bounding the per-substep MUTS drain (1.0 == identity).

    Returns the fraction of the raw continuity tendency that keeps
    MUTS = MUT + MU_work_new >= max(MIN_MUTS_PA, MIN_MUTS_FRACTION*MUT).  Only a
    negative (draining) tendency is limited; a positive/zero tendency is identity.
    A non-finite tendency is zeroed (scale 0) so a single bad cell cannot inject
    Inf/NaN into the conserved continuity update.  Apply the SAME scale to every
    continuity term (dmdt, dvdxi, mu_tend, the ww increments) so the limited step
    stays self-consistent.
    """

    mut = jnp.asarray(mut)
    mu_work_old = jnp.asarray(mu_work_old, dtype=mut.dtype)
    mu_tendency = jnp.asarray(mu_tendency, dtype=mut.dtype)
    floor = jnp.maximum(
        jnp.asarray(MIN_MUTS_PA, mut.dtype),
        jnp.asarray(MIN_MUTS_FRACTION, mut.dtype) * mut,
    )
    muts_old = mut + mu_work_old
    raw_delta = jnp.asarray(float(dts), mut.dtype) * mu_tendency
    allowed_loss = jnp.maximum(muts_old - floor, jnp.asarray(0.0, mut.dtype))
    loss = -raw_delta
    finite = jnp.isfinite(raw_delta) & jnp.isfinite(muts_old)
    scale = jnp.where(
        finite & (loss > allowed_loss),
        allowed_loss / jnp.maximum(loss, jnp.asarray(1.0, mut.dtype)),
        jnp.where(finite, jnp.asarray(1.0, mut.dtype), jnp.asarray(0.0, mut.dtype)),
    )
    return jnp.clip(scale, jnp.asarray(0.0, mut.dtype), jnp.asarray(1.0, mut.dtype))


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
    periodic_x: bool = True
    specified: bool = False
    nested: bool = False


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


def _empty_diagnostics(inputs: AdvanceMuTInputs) -> dict[str, jnp.ndarray]:
    nz = int(inputs.theta.shape[0])
    return {
        "dmdt": jnp.zeros_like(inputs.mu),
        "dvdxi": jnp.zeros_like(inputs.theta[:nz]),
        "wdtn": jnp.zeros_like(inputs.ww),
        "theta_tendency": jnp.zeros_like(inputs.theta),
    }


def advance_mu_t_wrf(inputs: AdvanceMuTInputs) -> dict[str, jnp.ndarray]:
    """Advance MU/theta through WRF ``advance_mu_t`` with BC-conditional bounds."""

    if bool(inputs.specified) or bool(inputs.nested):
        return _advance_mu_t_specified_or_nested(inputs)
    return _advance_mu_t_periodic(inputs)


def _advance_mu_t_periodic(inputs: AdvanceMuTInputs) -> dict[str, jnp.ndarray]:
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
    theta_tendency_levels = []
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
        theta_tendency_levels.append(tendency)
        theta_levels.append(theta_i[k] - float(inputs.dts) * inputs.msfty * tendency)
    theta_new = jnp.stack(theta_levels, axis=0)
    theta_tendency = jnp.stack(theta_tendency_levels, axis=0)

    return {
        "mu": mu_new,
        "mudf": mudf_new,
        "muts": muts_new,
        "muave": muave_new,
        "ww": ww_new,
        "theta": theta_new,
        "dmdt": dmdt,
        "dvdxi": dvdxi_stack,
        "wdtn": wdtn,
        "theta_tendency": theta_tendency,
    }


def _advance_mu_t_specified_or_nested(inputs: AdvanceMuTInputs) -> dict[str, jnp.ndarray]:
    """WRF specified/nested ``advance_mu_t`` path with non-periodic lateral bounds.

    Source bounds are WRF ``module_small_step_em.F:1048-1063``.  For
    specified/nested real domains WRF advances only interior mass cells, leaving the
    outer lateral rows/columns to the boundary-condition machinery.  If a
    specified/nested domain is periodic in x, WRF keeps the full x range and only
    restricts y.
    """

    nz = int(inputs.theta.shape[0])
    ny = int(inputs.mu.shape[0])
    nx = int(inputs.mu.shape[1])

    y0, y1 = (1, ny - 1)
    if bool(inputs.periodic_x):
        x0, x1 = 0, nx
    else:
        x0, x1 = 1, nx - 1
    if y1 <= y0 or x1 <= x0:
        return {
            "mu": inputs.mu,
            "mudf": inputs.mudf,
            "muts": inputs.muts,
            "muave": inputs.muave,
            "ww": inputs.ww,
            "theta": inputs.theta,
        } | _empty_diagnostics(inputs)

    ys = slice(y0, y1)
    xs = slice(x0, x1)
    xs_e = slice(x0 + 1, x1 + 1)
    ys_n = slice(y0 + 1, y1 + 1)

    dvdxi_levels = []
    for k in range(nz):
        c1 = inputs.c1h[k]
        c2 = inputs.c2h[k]
        v_north = (
            inputs.v[k, ys_n, xs]
            + (c1 * inputs.muv[ys_n, xs] + c2)
            * inputs.v_1[k, ys_n, xs]
            * inputs.msfvx_inv[ys_n, xs]
        )
        v_south = (
            inputs.v[k, ys, xs]
            + (c1 * inputs.muv[ys, xs] + c2)
            * inputs.v_1[k, ys, xs]
            * inputs.msfvx_inv[ys, xs]
        )
        u_east = (
            inputs.u[k, ys, xs_e]
            + (c1 * inputs.muu[ys, xs_e] + c2)
            * inputs.u_1[k, ys, xs_e]
            / inputs.msfuy[ys, xs_e]
        )
        u_west = (
            inputs.u[k, ys, xs]
            + (c1 * inputs.muu[ys, xs] + c2)
            * inputs.u_1[k, ys, xs]
            / inputs.msfuy[ys, xs]
        )
        dvdxi = inputs.msftx[ys, xs] * inputs.msfty[ys, xs] * (
            float(inputs.rdy) * (v_north - v_south) + float(inputs.rdx) * (u_east - u_west)
        )
        dvdxi_levels.append(dvdxi)
    dvdxi_active = jnp.stack(dvdxi_levels, axis=0)
    dmdt_active = jnp.sum(inputs.dnw[:nz, None, None] * dvdxi_active, axis=0)

    mu_work_old = inputs.muts[ys, xs] - inputs.mut[ys, xs]
    mu_save = inputs.mu[ys, xs] - mu_work_old
    mu_tendency = dmdt_active + inputs.mu_tend[ys, xs]
    # Dry-mass positivity guard: scale the continuity tendency so MUTS stays >=
    # MIN_MUTS_FRACTION of the stage MUT (identity for physical substeps).  The
    # SAME scale multiplies every continuity term below (dmdt, dvdxi, mu_tend, the
    # ww increments, and the returned dmdt/dvdxi diagnostics) so the limited
    # acoustic mass update stays self-consistent.
    mu_scale = _limit_mu_drain_scale(
        inputs.mut[ys, xs], mu_work_old, mu_tendency, float(inputs.dts)
    )
    mu_tendency = mu_tendency * mu_scale
    dmdt_active = dmdt_active * mu_scale
    dvdxi_active = dvdxi_active * mu_scale[None, :, :]
    mu_tend_active = inputs.mu_tend[ys, xs] * mu_scale
    dvdxi_full = inputs.theta[:nz] * 0.0
    dvdxi_full = dvdxi_full.at[:, ys, xs].set(dvdxi_active)
    dmdt_full = jnp.zeros_like(inputs.mu).at[ys, xs].set(dmdt_active)
    mu_work_new = mu_work_old + float(inputs.dts) * mu_tendency
    mu_new_i = mu_save + mu_work_new
    mudf_i = mu_tendency
    muts_i = inputs.mut[ys, xs] + mu_work_new
    muave_i = 0.5 * ((1.0 + float(inputs.epssm)) * mu_work_new + (1.0 - float(inputs.epssm)) * mu_work_old)

    mu_new = _update_2d(inputs.mu, mu_new_i, ys, xs)
    mudf_new = _update_2d(inputs.mudf, mudf_i, ys, xs)
    muts_new = _update_2d(inputs.muts, muts_i, ys, xs)
    muave_new = _update_2d(inputs.muave, muave_i, ys, xs)

    ww_rows = [inputs.ww[0, ys, xs]]
    for kk in range(1, nz):
        k = kk - 1
        increment = inputs.dnw[kk - 1] * (
            inputs.c1h[k] * dmdt_active + dvdxi_active[kk - 1] + inputs.c1h[k] * mu_tend_active
        ) / inputs.msfty[ys, xs]
        ww_rows.append(ww_rows[-1] - increment)
    ww_rows.append(inputs.ww[nz, ys, xs])
    ww_active = jnp.stack(ww_rows, axis=0)
    ww_active = ww_active.at[:nz].set(ww_active[:nz] - inputs.ww_1[:nz, ys, xs])
    ww_new = inputs.ww.at[:, ys, xs].set(ww_active)

    theta_i = inputs.theta
    theta_i_active = inputs.theta[:, ys, xs] + inputs.msfty[None, ys, xs] * float(inputs.dts) * inputs.theta_tend[:, ys, xs]
    theta_i = theta_i.at[:, ys, xs].set(theta_i_active)
    theta_flux_source = inputs.theta_1

    wdtn_rows = [jnp.zeros_like(mu_tendency)]
    for k in range(1, nz):
        face_theta = inputs.fnm[k] * theta_flux_source[k, ys, xs] + inputs.fnp[k] * theta_flux_source[k - 1, ys, xs]
        wdtn_rows.append(ww_active[k] * face_theta)
    wdtn_rows.append(jnp.zeros_like(mu_tendency))
    wdtn = jnp.stack(wdtn_rows, axis=0)
    wdtn_full = jnp.zeros_like(inputs.ww).at[:, ys, xs].set(wdtn)

    theta_new = inputs.theta
    theta_tendency_full = jnp.zeros_like(inputs.theta)
    for k in range(nz):
        th = theta_flux_source[k]
        if bool(inputs.periodic_x):
            th_e = jnp.roll(th, -1, axis=-1)[ys, xs]
            th_w = jnp.roll(th, 1, axis=-1)[ys, xs]
        else:
            th_e = th[ys, slice(x0 + 1, x1 + 1)]
            th_w = th[ys, slice(x0 - 1, x1 - 1)]
        th_n = th[slice(y0 + 1, y1 + 1), xs]
        th_s = th[slice(y0 - 1, y1 - 1), xs]

        v_flux = inputs.v[k, ys_n, xs] * (th_n + th[ys, xs]) - inputs.v[k, ys, xs] * (th[ys, xs] + th_s)
        u_flux = inputs.u[k, ys, xs_e] * (th_e + th[ys, xs]) - inputs.u[k, ys, xs] * (th[ys, xs] + th_w)
        tendency = inputs.msftx[ys, xs] * (0.5 * float(inputs.rdy) * v_flux + 0.5 * float(inputs.rdx) * u_flux) + inputs.rdnw[k] * (
            wdtn[k + 1] - wdtn[k]
        )
        theta_k = theta_i[k, ys, xs] - float(inputs.dts) * inputs.msfty[ys, xs] * tendency
        theta_new = theta_new.at[k, ys, xs].set(theta_k)
        theta_tendency_full = theta_tendency_full.at[k, ys, xs].set(tendency)

    return {
        "mu": mu_new,
        "mudf": mudf_new,
        "muts": muts_new,
        "muave": muave_new,
        "ww": ww_new,
        "theta": theta_new,
        "dmdt": dmdt_full,
        "dvdxi": dvdxi_full,
        "wdtn": wdtn_full,
        "theta_tendency": theta_tendency_full,
    }
