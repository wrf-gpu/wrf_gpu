"""CAM-UW moist-turbulence PBL column endpoint (``bl_pbl_physics=9``).

WRF's CAM-UW driver (``phys/module_bl_camuwpbl_driver.F``) is a wrapper around
CAM5 vertical diffusion:

* diagnose UW/Park-Bretherton TKE eddy diffusivities with
  ``module_cam_bl_eddy_diff.F:compute_eddy_diff``;
* solve implicit vertical diffusion with
  ``module_cam_bl_diffusion_solver.F:compute_vdiff``;
* diffuse momentum, dry static energy, water vapor, cloud liquid, and cloud ice;
* carry residual stresses plus previous-step ``kvm``/``kvh`` and TKE diagnostics.

This module implements the same operational shape in a traceable JAX column
kernel: CAM-style diagnostic TKE/K profiles, implicit diffusion of the driving
fields, and the CAM bottom-flux conventions.  It is intentionally not advertised
as bitwise/source-parity with the full CAM stack until a pristine-WRF CAM-UW
savepoint fixture exists for the CAM cloud-number/sedimentation/residual-stress
state.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
import jax.numpy as jnp

configure_jax_x64()

G = 9.80665
R_D = 287.0
CP_D = 1004.0
KARMAN = 0.4
RICRIT = 0.25
K_MAX = 1000.0
K_BACKGROUND = 0.01
TKE_MIN = 1.0e-4
SMAW_MAX = 4.964


def _solve_tridiagonal_1d(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Thomas solve for one column."""

    nlev = rhs.shape[0]
    if nlev == 1:
        return rhs / diag

    c0 = upper[0] / diag[0]
    d0 = rhs[0] / diag[0]

    def fwd(carry, i):
        c_prev, d_prev = carry
        denom = diag[i] - lower[i] * c_prev
        c_i = jnp.where(i < nlev - 1, upper[i] / denom, 0.0)
        d_i = (rhs[i] - lower[i] * d_prev) / denom
        return (c_i, d_i), (c_i, d_i)

    (_, _), (c_tail, d_tail) = jax.lax.scan(
        fwd, (c0, d0), jnp.arange(1, nlev, dtype=jnp.int32)
    )
    c = jnp.concatenate([c0[None], c_tail])
    d = jnp.concatenate([d0[None], d_tail])

    def bwd(x_next, jj):
        i = nlev - 2 - jj
        x_i = d[i] - c[i] * x_next
        return x_i, x_i

    _, x_rev = jax.lax.scan(bwd, d[-1], jnp.arange(nlev - 1, dtype=jnp.int32))
    return jnp.concatenate([x_rev[::-1], d[-1:]])


def _implicit_diffuse(
    field: jax.Array,
    k_face: jax.Array,
    dz: jax.Array,
    dt: float,
    *,
    bottom_flux: jax.Array,
    top_flux: jax.Array | None = None,
    explicit_flux: jax.Array | None = None,
) -> jax.Array:
    """Backward-Euler vertical diffusion with prescribed boundary fluxes.

    ``field`` is bottom-up ``(ncol, nlev)``. ``k_face`` is an interface array
    ``(ncol, nlev+1)``; only the interior interfaces enter the implicit matrix.
    ``bottom_flux`` and ``top_flux`` are positive upward turbulent fluxes in the
    units of ``field * m s-1``.
    """

    ncol, nlev = field.shape
    dtype = field.dtype
    dt_arr = jnp.asarray(dt, dtype)
    lower = jnp.zeros_like(field)
    upper = jnp.zeros_like(field)

    if nlev > 1:
        dz_between = jnp.maximum(0.5 * (dz[:, :-1] + dz[:, 1:]), 1.0)
        coef = jnp.maximum(k_face[:, 1:nlev], 0.0) / dz_between
        upper = upper.at[:, :-1].set(-dt_arr * coef / jnp.maximum(dz[:, :-1], 1.0))
        lower = lower.at[:, 1:].set(-dt_arr * coef / jnp.maximum(dz[:, 1:], 1.0))

    diag = 1.0 - lower - upper
    rhs = field.at[:, 0].add(dt_arr * bottom_flux / jnp.maximum(dz[:, 0], 1.0))
    if top_flux is not None:
        rhs = rhs.at[:, -1].add(-dt_arr * top_flux / jnp.maximum(dz[:, -1], 1.0))
    if explicit_flux is not None:
        flux_div = (explicit_flux[:, 1:] - explicit_flux[:, :-1]) / jnp.maximum(dz, 1.0)
        rhs = rhs - dt_arr * flux_div

    return jax.vmap(_solve_tridiagonal_1d)(lower, diag, upper, rhs)


def _interface_height(dz: jax.Array) -> jax.Array:
    zero = jnp.zeros((dz.shape[0], 1), dtype=dz.dtype)
    return jnp.concatenate([zero, jnp.cumsum(jnp.maximum(dz, 1.0), axis=1)], axis=1)


def _mass_to_face(field: jax.Array) -> jax.Array:
    if field.shape[1] == 1:
        return jnp.concatenate([field, field], axis=1)
    return jnp.concatenate(
        [field[:, :1], 0.5 * (field[:, :-1] + field[:, 1:]), field[:, -1:]],
        axis=1,
    )


def _face_gradients(
    u: jax.Array,
    v: jax.Array,
    theta_v: jax.Array,
    dz: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return face shear squared and Brunt-Vaisala frequency squared."""

    ncol, nlev = u.shape
    shear2 = jnp.zeros((ncol, nlev + 1), dtype=u.dtype)
    n2 = jnp.zeros((ncol, nlev + 1), dtype=u.dtype)
    if nlev == 1:
        return shear2, n2

    dz_between = jnp.maximum(0.5 * (dz[:, :-1] + dz[:, 1:]), 1.0)
    du_dz = (u[:, 1:] - u[:, :-1]) / dz_between
    dv_dz = (v[:, 1:] - v[:, :-1]) / dz_between
    dthv_dz = (theta_v[:, 1:] - theta_v[:, :-1]) / dz_between
    thv_face = jnp.maximum(0.5 * (theta_v[:, :-1] + theta_v[:, 1:]), 150.0)
    shear_int = du_dz * du_dz + dv_dz * dv_dz
    n2_int = G * dthv_dz / thv_face
    shear2 = shear2.at[:, 1:nlev].set(shear_int)
    n2 = n2.at[:, 1:nlev].set(n2_int)
    shear2 = shear2.at[:, 0].set(shear_int[:, 0])
    shear2 = shear2.at[:, -1].set(shear_int[:, -1])
    n2 = n2.at[:, 0].set(n2_int[:, 0])
    n2 = n2.at[:, -1].set(n2_int[:, -1])
    return shear2, n2


def _diagnose_pbl_height(
    z_mid: jax.Array,
    theta_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    wtheta_v_sfc: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    z_agl = jnp.maximum(z_mid - z_mid[:, :1] + jnp.maximum(z_mid[:, :1], 1.0), 1.0)
    du = u - u[:, :1]
    dv = v - v[:, :1]
    shear = jnp.maximum(du * du + dv * dv, 0.25)
    dthv = theta_v - theta_v[:, :1]
    rib = G * z_agl * dthv / jnp.maximum(theta_v[:, :1] * shear, 1.0e-6)
    threshold = jnp.where(wtheta_v_sfc > 0.0, 0.0, RICRIT)
    above = rib > threshold[:, None]
    above = above.at[:, 0].set(False)
    has_top = jnp.any(above, axis=1)
    first_top = jnp.argmax(above, axis=1)
    top_idx = jnp.where(has_top, first_top, z_mid.shape[1] - 1)
    pblh = jnp.take_along_axis(z_agl, top_idx[:, None], axis=1)[:, 0]
    pblh = jnp.maximum(pblh, jnp.maximum(2.0 * z_agl[:, 0], 50.0))
    convective_floor = jnp.minimum(z_agl[:, -1], jnp.maximum(800.0, 2.0 * z_agl[:, 0]))
    pblh = jnp.where(wtheta_v_sfc > 0.0, jnp.maximum(pblh, convective_floor), pblh)
    return pblh, top_idx.astype(jnp.int32) + 1


def _diagnose_camuw_diffusivity(
    u: jax.Array,
    v: jax.Array,
    theta: jax.Array,
    t: jax.Array,
    qv: jax.Array,
    qc: jax.Array,
    qi: jax.Array,
    p: jax.Array,
    dz: jax.Array,
    z_mid: jax.Array,
    tke_initial: jax.Array,
    hfx: jax.Array,
    qfx: jax.Array,
    ust: jax.Array,
    wspd: jax.Array,
    dt: float,
) -> dict[str, jax.Array]:
    rho0 = jnp.maximum(
        p[:, 0] / (R_D * jnp.maximum(t[:, 0], 180.0) * (1.0 + 0.61 * jnp.maximum(qv[:, 0], 0.0))),
        0.2,
    )
    theta_v = theta * (1.0 + 0.61 * qv - qc - qi)
    wtheta = hfx / (rho0 * CP_D)
    wqv = qfx / rho0
    wtheta_v = wtheta + 0.61 * theta[:, 0] * wqv
    pblh, kpbl = _diagnose_pbl_height(z_mid, theta_v, u, v, wtheta_v)

    z_face = _interface_height(dz)
    eta = jnp.clip(z_face / jnp.maximum(pblh[:, None], 1.0), 0.0, 1.0)
    inside = z_face <= pblh[:, None]
    shear2, n2 = _face_gradients(u, v, theta_v, dz)

    wstar = jnp.maximum(G * wtheta_v * pblh / jnp.maximum(theta_v[:, 0], 150.0), 0.0) ** (1.0 / 3.0)
    surface_tke = 3.75 * ust * ust + 0.5 * wstar * wstar
    mixlen = KARMAN * jnp.maximum(z_face, 1.0) * (1.0 - eta) ** 1.5
    mixlen = jnp.where(inside, mixlen, 0.0)
    shear_target = mixlen * mixlen * jnp.maximum(shear2 - jnp.minimum(n2, 0.0), 0.0)
    tke_target = TKE_MIN + (surface_tke[:, None] * (1.0 - eta) ** 2 + shear_target) * inside

    tke_old_face = _mass_to_face(jnp.maximum(tke_initial, TKE_MIN))
    tau = jnp.maximum(pblh / jnp.maximum(ust + wstar, 0.1), 60.0)
    alpha = 1.0 - jnp.exp(-jnp.asarray(dt, u.dtype) / tau)
    tke_face = (1.0 - alpha[:, None]) * tke_old_face + alpha[:, None] * tke_target
    tke_face = jnp.maximum(tke_face, TKE_MIN)

    ri = n2 / jnp.maximum(shear2, 1.0e-8)
    smaw = jnp.clip((1.0 - 5.0 * ri) / (1.0 + 10.0 * jnp.maximum(ri, 0.0)), 0.0, SMAW_MAX)
    smaw = jnp.where(jnp.logical_and(wtheta_v[:, None] > 0.0, inside), jnp.maximum(smaw, 1.0), smaw)
    prandtl = jnp.where(ri < 0.0, 0.74, jnp.clip(1.0 + 5.0 * ri, 0.74, 4.0))
    velocity_scale = jnp.sqrt(jnp.maximum(2.0 * tke_face, 0.0))
    kvm = jnp.minimum(K_MAX, mixlen * velocity_scale * smaw + K_BACKGROUND)
    kvh = jnp.minimum(K_MAX, kvm / prandtl)
    kvm = kvm.at[:, 0].set(0.0).at[:, -1].set(0.0)
    kvh = kvh.at[:, 0].set(0.0).at[:, -1].set(0.0)
    tke_mass = 0.5 * (tke_face[:, :-1] + tke_face[:, 1:])

    turbtype = jnp.where(inside, jnp.where(wtheta_v[:, None] > 0.0, 1.0, 2.0), 0.0)
    tpert = jnp.maximum(wtheta / jnp.maximum(wstar, 0.2), 0.0)
    qpert = jnp.maximum(wqv / jnp.maximum(wstar, 0.2), 0.0)
    return {
        "kvh": kvh,
        "kvm": kvm,
        "tke": tke_mass,
        "pblh": pblh,
        "kpbl": kpbl,
        "wstar": wstar,
        "wtheta": wtheta,
        "wqv": wqv,
        "rho0": rho0,
        "smaw": smaw,
        "turbtype": turbtype,
        "tpert": tpert,
        "qpert": qpert,
        "wpert": wstar,
        "inside": inside,
        "eta": eta,
        "wspd": jnp.maximum(wspd, 0.1),
    }


def camuw_columns(
    u: jax.Array,
    v: jax.Array,
    t: jax.Array,
    theta: jax.Array,
    qv: jax.Array,
    qc: jax.Array,
    qi: jax.Array,
    p: jax.Array,
    pii: jax.Array,
    dz: jax.Array,
    z_mid: jax.Array,
    tke_initial: jax.Array,
    *,
    hfx: jax.Array,
    qfx: jax.Array,
    ust: jax.Array,
    wspd: jax.Array,
    dt: float,
) -> dict[str, jax.Array]:
    """Run one CAM-UW-style PBL column batch.

    Inputs and outputs use the project PBL convention: bottom-up arrays shaped
    ``(ncol, nlev)`` and tendencies in per-second units. ``u``/``v`` are mass-grid
    winds; the scan adapter maps increments back to C-grid faces.
    """

    diag = _diagnose_camuw_diffusivity(
        u, v, theta, t, qv, qc, qi, p, dz, z_mid, tke_initial, hfx, qfx, ust, wspd, dt
    )
    kvh = diag["kvh"]
    kvm = diag["kvm"]
    rho0 = diag["rho0"]

    s = CP_D * t + G * z_mid
    s_flux = hfx / rho0
    theta_flux = hfx / (rho0 * CP_D * jnp.maximum(pii[:, 0], 0.2))
    qv_flux = qfx / rho0
    profile = diag["inside"] * diag["eta"] * (1.0 - diag["eta"]) ** 2
    gamma_s = jnp.where(
        diag["wtheta"] > 0.0,
        8.5 * s_flux / jnp.maximum(diag["wstar"] * diag["pblh"], 1.0),
        0.0,
    )
    gamma_q = jnp.where(
        diag["wqv"] > 0.0,
        8.5 * qv_flux / jnp.maximum(diag["wstar"] * diag["pblh"], 1.0),
        0.0,
    )
    s_nonlocal_flux = kvh * gamma_s[:, None] * profile
    q_nonlocal_flux = kvh * gamma_q[:, None] * profile

    u_flux = -ust * ust * u[:, 0] / diag["wspd"]
    v_flux = -ust * ust * v[:, 0] / diag["wspd"]
    zeros = jnp.zeros_like(hfx)
    u_new = _implicit_diffuse(u, kvm, dz, dt, bottom_flux=u_flux)
    v_new = _implicit_diffuse(v, kvm, dz, dt, bottom_flux=v_flux)
    s_new = _implicit_diffuse(
        s, kvh, dz, dt, bottom_flux=s_flux, explicit_flux=s_nonlocal_flux
    )
    qv_new = _implicit_diffuse(
        qv, kvh, dz, dt, bottom_flux=qv_flux, explicit_flux=q_nonlocal_flux
    )
    qc_new = _implicit_diffuse(qc, kvh, dz, dt, bottom_flux=zeros)
    qi_new = _implicit_diffuse(qi, kvh, dz, dt, bottom_flux=zeros)
    theta_new = (s_new - G * z_mid) / (CP_D * jnp.maximum(pii, 0.2))

    dt_arr = jnp.asarray(dt, u.dtype)
    return {
        "u": (u_new - u) / dt_arr,
        "v": (v_new - v) / dt_arr,
        "theta": (theta_new - theta) / dt_arr,
        "qv": (qv_new - qv) / dt_arr,
        "qc": (qc_new - qc) / dt_arr,
        "qi": (qi_new - qi) / dt_arr,
        "tke": diag["tke"],
        "kvh": kvh,
        "kvm": kvm,
        "pblh": diag["pblh"],
        "kpbl": diag["kpbl"],
        "smaw": diag["smaw"],
        "turbtype": diag["turbtype"],
        "tpert": diag["tpert"],
        "qpert": diag["qpert"],
        "wpert": diag["wpert"],
        # Alias matching the CAM wrapper name used by downstream proof checks.
        "wstarPBL": diag["wstar"],
        "theta_flux_bottom": theta_flux,
    }


__all__ = ["TKE_MIN", "camuw_columns"]
