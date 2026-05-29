"""WRF flux-form, mass-coupled large-step advection (Block 2).

Source: WRF ``dyn_em/module_advect_em.F`` subroutines ``advect_scalar``
(``:3029-4359``), ``advect_u`` (``:126-1526``), ``advect_v`` (``:1530-3024``),
and ``advect_w`` (``:4364-6064``); the mass-coupled velocities ``ru``/``rv``/
``rom`` come from ``rk_step_prep`` -> ``couple_momentum``
(``module_em.F:195-201``) and ``calc_ww_cp`` (``module_big_step_utilities_em.F:640-782``).

Scope (a documented restriction, **not** a WRF fact): this module freezes the
advection orders to ``h_sca_adv_order = 5`` and ``v_sca_adv_order = 3`` and the
periodic-x/periodic-y boundary path (no boundary degradation), which is the
configuration the F7-B idealized gates use.  WRF selects the order from
``config_flags`` and degrades flux order near specified/nested boundaries
(``module_advect_em.F:3137-3392``); those paths are out of scope here and are
documented in ``proofs/f7b/advection_order_proof.md``.

The flux operators are the exact WRF definitions
(``module_advect_em.F:3105-3119``):

* ``flux6(q_im3..q_ip2)   = (37*(q_i+q_im1) - 8*(q_ip1+q_im2) + (q_ip2+q_im3))/60``
* ``flux5 = flux6 - sign(time_step)*sign(vel)*((q_ip2-q_im3) - 5*(q_ip1-q_im2) + 10*(q_i-q_im1))/60``
* ``flux4(q_im2..q_ip1)   = (7*(q_i+q_im1) - (q_ip1+q_im2))/12``
* ``flux3 = flux4 + sign(time_step)*sign(vel)*((q_ip1-q_im2) - 3*(q_i-q_im1))/12``

with ``time_step > 0`` so ``sign(time_step)=+1``.  Tendencies are flux
divergences: ``tendency -= mrdx*(fqx(i+1)-fqx(i))`` etc., with
``mrdx = msftx*rdx`` (``module_advect_em.F:3387-3388``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies


config.update("jax_enable_x64", True)


# --- WRF flux operators (module_advect_em.F:3105-3119); time_step>0 -> sign=+1 ---


def _flux6_face(field: jax.Array, axis: int) -> jax.Array:
    """6th-order centered face value at the face between cell i-1 and i (roll axis).

    Returns the value located at the *left* face of cell ``i`` using the WRF
    ``flux6`` stencil ``q_im3..q_ip2`` centered on that face.
    """

    q_im3 = jnp.roll(field, 3, axis=axis)
    q_im2 = jnp.roll(field, 2, axis=axis)
    q_im1 = jnp.roll(field, 1, axis=axis)
    q_i = field
    q_ip1 = jnp.roll(field, -1, axis=axis)
    q_ip2 = jnp.roll(field, -2, axis=axis)
    return (37.0 * (q_i + q_im1) - 8.0 * (q_ip1 + q_im2) + (q_ip2 + q_im3)) / 60.0


def _flux5_correction(field: jax.Array, axis: int) -> jax.Array:
    """Upwind dissipative correction term (the part multiplied by sign(vel))."""

    q_im3 = jnp.roll(field, 3, axis=axis)
    q_im2 = jnp.roll(field, 2, axis=axis)
    q_im1 = jnp.roll(field, 1, axis=axis)
    q_i = field
    q_ip1 = jnp.roll(field, -1, axis=axis)
    q_ip2 = jnp.roll(field, -2, axis=axis)
    return ((q_ip2 - q_im3) - 5.0 * (q_ip1 - q_im2) + 10.0 * (q_i - q_im1)) / 60.0


def flux5_face_periodic(field: jax.Array, vel: jax.Array, axis: int) -> jax.Array:
    """WRF 5th-order upwind face flux value at the left face of cell ``i``.

    ``flux5 = flux6 - sign(vel) * correction`` (time_step>0).  ``vel`` is the
    mass-coupled face velocity collocated at the same face as the flux.
    """

    return _flux6_face(field, axis) - jnp.sign(vel) * _flux5_correction(field, axis)


# --- mass-coupled velocities (rk_step_prep -> couple_momentum, calc_ww_cp) ---


@dataclass(frozen=True)
class CoupledVelocities:
    """Mass-coupled velocities for flux-form advection (dry, map factor=1)."""

    ru: jax.Array  # (nz, ny, nx+1) coupled u at x faces = (c1h*muu+c2h)*u/msfuy
    rv: jax.Array  # (nz, ny+1, nx) coupled v at y faces = (c1h*muv+c2h)*v*msfvx_inv
    rom: jax.Array  # (nz+1, ny, nx) coupled omega ww at w faces (calc_ww_cp)


def _muu_face(mu_total: jax.Array) -> jax.Array:
    """muu = 0.5*(mu(i)+mu(i-1)) at x faces, periodic (calc_mu_uv / calc_ww_cp:700)."""

    return 0.5 * (mu_total + jnp.roll(mu_total, 1, axis=-1))


def _muv_face(mu_total: jax.Array) -> jax.Array:
    """muv = 0.5*(mu(j)+mu(j-1)) at y faces, periodic."""

    return 0.5 * (mu_total + jnp.roll(mu_total, 1, axis=-2))


def couple_velocities_periodic(
    u: jax.Array,
    v: jax.Array,
    mu_total: jax.Array,
    *,
    c1h: jax.Array,
    c2h: jax.Array,
    dnw: jax.Array,
    rdx: float,
    rdy: float,
) -> CoupledVelocities:
    """Build ``ru``/``rv``/``rom`` for the periodic dry case (msf=1).

    Source: ``couple_momentum`` (``module_em.F:195``) and ``calc_ww_cp``
    (``module_big_step_utilities_em.F:640-782``).  ``u`` is on x faces
    ``(nz, ny, nx+1)`` collapsed to ``(nz, ny, nx)`` periodically (the project's
    one-row C-grid keeps ``nx+1`` u faces; the last equals the first under
    periodicity).  For the idealized slab the map factors are unity.
    """

    nz = int(u.shape[0])
    # u has nx+1 faces; in the periodic project layout face nx == face 0.  Use the
    # first nx faces as the staggered u at x-faces 0..nx-1 (WRF u(i) at face i).
    u_faces = u[:, :, :-1] if u.shape[-1] == mu_total.shape[-1] + 1 else u
    v_faces = v[:, :-1, :] if v.shape[-2] == mu_total.shape[-2] + 1 else v

    muu = _muu_face(mu_total)  # (ny, nx)
    muv = _muv_face(mu_total)  # (ny, nx)
    mass_u = c1h[:, None, None] * muu[None, :, :] + c2h[:, None, None]  # (nz, ny, nx)
    mass_v = c1h[:, None, None] * muv[None, :, :] + c2h[:, None, None]
    ru = mass_u * u_faces  # msfuy = 1
    rv = mass_v * v_faces  # msfvx_inv = 1

    # calc_ww_cp: divv(k) = dnw(k)*(rdx*(ru(i+1)-ru(i)) + rdy*(rv(j+1)-rv(j)))  (msftx=1)
    ru_ip1 = jnp.roll(ru, -1, axis=-1)
    rv_jp1 = jnp.roll(rv, -1, axis=-2)
    divv = dnw[:, None, None] * (rdx * (ru_ip1 - ru) + rdy * (rv_jp1 - rv))  # (nz, ny, nx)
    dmdt = jnp.sum(divv, axis=0, keepdims=True)  # (1, ny, nx) integral over levels
    # ww(k) = ww(k-1) - dnw(k-1)*c1h(k-1)*dmdt - divv(k-1), with ww(0)=ww(nz)=0.
    increments = -(dnw[:, None, None] * c1h[:, None, None] * dmdt) - divv  # (nz, ny, nx) per k-1
    # ww at faces 1..nz-1 = cumulative sum of increments[0..k-1]; ww(0)=0, ww(nz)=0 (rigid).
    cum = jnp.cumsum(increments, axis=0)  # (nz, ny, nx); cum[k-1] is ww at face k
    rom = jnp.zeros((nz + 1,) + tuple(mu_total.shape), dtype=u.dtype)
    rom = rom.at[1:nz, :, :].set(cum[: nz - 1, :, :])
    # face nz (top) stays 0 (rigid lid); face 0 (surface) stays 0.
    return CoupledVelocities(ru=ru, rv=rv, rom=rom)


# --- scalar flux-form advection (advect_scalar, h=5/v=3) ---


def _u_face_to_mass(field_face: jax.Array) -> jax.Array:
    """Collapse u (nz,ny,nx+1) to nx periodic faces (face i)."""

    return field_face[:, :, :-1] if field_face.shape[-1] > field_face.shape[-1] - 1 else field_face


def advect_scalar_flux(
    field: jax.Array,
    vel: CoupledVelocities,
    *,
    mut: jax.Array,
    c1: jax.Array,
    rdx: float,
    rdy: float,
    rdzw: jax.Array,
    fzm: jax.Array,
    fzp: jax.Array,
) -> jax.Array:
    """WRF flux-form mass-coupled scalar advection tendency (h=5, v=3).

    Source: ``advect_scalar`` (``module_advect_em.F:3029-4359``).  Returns the
    coupled tendency ``d(mu*phi)/dt`` so it is added to the coupled prognostic;
    callers that hold uncoupled state must divide by mass.  For the idealized
    periodic slab ``msftx = msfty = 1``.

    Horizontal (order 5, periodic, no degradation): for each x face i,
    ``fqx(i) = ru(i) * flux5(field[i-3..i+2], ru(i))`` and
    ``tend -= rdx*(fqx(i+1)-fqx(i))``.  Same in y with ``rv``.

    Vertical (order 3): interior faces k=2..ktf-1 use ``flux3``; the two faces
    adjacent to the rigid boundaries use 2nd-order ``fzm/fzp`` interpolation;
    the surface (k=0) and top (k=nz) faces carry zero flux.
    """

    # ---- x flux divergence ----
    # fqx located at left face of cell i; ru is the coupled u at face i.
    fqx = vel.ru * flux5_face_periodic(field, vel.ru, axis=2)  # (nz, ny, nx)
    fqx_ip1 = jnp.roll(fqx, -1, axis=2)
    tend = -rdx * (fqx_ip1 - fqx)

    # ---- y flux divergence ----
    fqy = vel.rv * flux5_face_periodic(field, vel.rv, axis=1)  # (nz, ny, nx)
    fqy_jp1 = jnp.roll(fqy, -1, axis=1)
    tend = tend - rdy * (fqy_jp1 - fqy)

    # ---- z flux divergence (order 3, rigid top/bottom) ----
    nz = int(field.shape[0])
    rom = vel.rom  # (nz+1, ny, nx) at w faces; rom[0]=rom[nz]=0
    vflux = jnp.zeros((nz + 1,) + tuple(field.shape[1:]), dtype=field.dtype)
    # interior faces k=2..nz-2 (WRF kts+2..ktf-1, 0-based face index 2..nz-2): flux3
    if nz >= 4:
        # flux3 face value at face k uses field[k-2,k-1,k,k+1] with vel=-rom(k).
        q_km2 = field[: nz - 3, :, :]  # k-2 for faces 2..nz-2 -> field idx 0..nz-4
        q_km1 = field[1 : nz - 2, :, :]
        q_k = field[2 : nz - 1, :, :]
        q_kp1 = field[3:nz, :, :]
        velz = -rom[2 : nz - 1, :, :]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 + 0.0) - q_km2 - 3.0 * (q_k - q_km1)) / 12.0
        # WRF flux3 = flux4 + sign(time_step)*sign(vel)*((q_ip1-q_im2)-3*(q_i-q_im1))/12
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        flux3 = flux4 + jnp.sign(velz) * corr
        vflux = vflux.at[2 : nz - 1, :, :].set(rom[2 : nz - 1, :, :] * flux3)
    # faces adjacent to rigid boundaries: k=1 (kts+1) and k=nz-1 (ktf): 2nd order
    if nz >= 2:
        # face 1: rom(1)*(fzm(1)*field(1)+fzp(1)*field(0))
        vflux = vflux.at[1, :, :].set(
            rom[1, :, :] * (fzm[1] * field[1, :, :] + fzp[1] * field[0, :, :])
        )
        # face nz-1 (WRF ktf): rom(nz-1)*(fzm(nz-1)*field(nz-1)+fzp(nz-1)*field(nz-2))
        vflux = vflux.at[nz - 1, :, :].set(
            rom[nz - 1, :, :] * (fzm[nz - 1] * field[nz - 1, :, :] + fzp[nz - 1] * field[nz - 2, :, :])
        )
    # surface (face 0) and top (face nz) carry zero flux (rigid).
    vflux_kp1 = vflux[1:, :, :]  # faces 1..nz
    vflux_k = vflux[:nz, :, :]  # faces 0..nz-1
    tend = tend - rdzw[:, None, None] * (vflux_kp1 - vflux_k)
    return tend


# --- flux-form momentum advection (advect_u / advect_v / advect_w, h=5/v=3) ---
#
# WRF advances momentum with the *conservative* flux-divergence form
# ``d(coupled u)/dt = -[ rdx d/dx (vel_x u) + rdy d/dy (vel_y u)
#                        + rdzw d/dz (vel_z u) ]`` where the transporting
# velocities are the mass-coupled fluxes ``ru/rv/rom`` averaged onto the u
# stagger (``vel = 0.5*(r(i)+r(i-1))``):
#   - advect_u  : module_advect_em.F:126-1526 (x-flux :420, y-flux :298,
#                 z-flux :1366/1406, assembly :480/:1395).
#   - advect_w  : module_advect_em.F:4364-6064 (vel averaged onto the w/full
#                 level via fzm/fzp; z-flux uses the mass-level rom average).
# The previous JAX path used the *advective* (non-conservative) primitive form
# ``u du/dx`` (advection.py advect_u_face), which does not conserve momentum and
# lets the Straka cold-front outflow pile up instead of propagating (the front
# crawls while |w| at the head runs away).  ``ru/rv`` differ from the advective
# form by ``u (div . vel)`` which is non-zero for compressible mass flux.


def _avg_to_u_face_x(field_mass: jax.Array) -> jax.Array:
    """Average a mass-located (nz,ny,nx) field onto u x-faces: 0.5*(r(i)+r(i-1))."""

    return 0.5 * (field_mass + jnp.roll(field_mass, 1, axis=-1))


def _avg_to_v_face_y(field_mass: jax.Array) -> jax.Array:
    return 0.5 * (field_mass + jnp.roll(field_mass, 1, axis=-2))


def advect_u_flux(
    u: jax.Array,
    vel: CoupledVelocities,
    *,
    rdx: float,
    rdy: float,
    rdzw: jax.Array,
    fzm: jax.Array,
    fzp: jax.Array,
) -> jax.Array:
    """WRF flux-form coupled u advection tendency (h=5, v=3), periodic dry slab.

    ``u`` is on x-faces ``(nz, ny, nx+1)`` (periodic: face nx == face 0).  The
    transporting velocities are ``ru/rv/rom`` averaged onto the u faces.
    Returns the COUPLED tendency ``d(mu*u)/dt`` (so callers add it directly to
    the coupled momentum, exactly like ``advect_scalar_flux``).
    """

    nx = vel.ru.shape[-1]
    u_f = u[:, :, :nx] if u.shape[-1] == nx + 1 else u  # collapse to nx periodic faces
    # x-flux: at u-face i the velocity is 0.5*(ru(i)+ru(i-1)); flux5 stencil on u_f.
    velx = _avg_to_u_face_x(vel.ru)
    fqx = velx * flux5_face_periodic(u_f, velx, axis=2)
    tend = -rdx * (jnp.roll(fqx, -1, axis=2) - fqx)
    # y-flux: velocity 0.5*(rv(i)+rv(i-1)) averaged in x onto the u stagger.
    if u_f.shape[1] > 1:
        rv_collapsed = vel.rv[:, : u_f.shape[1], :]
        vely = _avg_to_u_face_x(rv_collapsed)
        fqy = vely * flux5_face_periodic(u_f, vely, axis=1)
        tend = tend - rdy * (jnp.roll(fqy, -1, axis=1) - fqy)
    # z-flux (order 3): vel = 0.5*(rom(i)+rom(i-1)) averaged in x onto u faces.
    tend = tend + _vertical_flux_div_3(u_f, _avg_to_u_face_x_w(vel.rom), rdzw, fzm, fzp)
    return _restore_periodic_u_face(tend, u.shape[-1] == nx + 1)


def advect_v_flux(
    v: jax.Array,
    vel: CoupledVelocities,
    *,
    rdx: float,
    rdy: float,
    rdzw: jax.Array,
    fzm: jax.Array,
    fzp: jax.Array,
) -> jax.Array:
    """WRF flux-form coupled v advection tendency (h=5, v=3).  v on y-faces."""

    ny = vel.rv.shape[-2]
    v_f = v[:, :ny, :] if v.shape[-2] == ny + 1 else v
    # x-flux: velocity 0.5*(ru(j)+ru(j-1)) averaged in y onto the v stagger.
    velx = _avg_to_v_face_y(vel.ru[:, : v_f.shape[1], :])
    fqx = velx * flux5_face_periodic(v_f, velx, axis=2)
    tend = -rdx * (jnp.roll(fqx, -1, axis=2) - fqx)
    # y-flux: velocity 0.5*(rv(j)+rv(j-1)).
    vely = _avg_to_v_face_y(vel.rv)
    fqy = vely * flux5_face_periodic(v_f, vely, axis=1)
    tend = tend - rdy * (jnp.roll(fqy, -1, axis=1) - fqy)
    # z-flux (order 3).
    tend = tend + _vertical_flux_div_3(v_f, _avg_to_v_face_y_w(vel.rom), rdzw, fzm, fzp)
    if v.shape[-2] == ny + 1:
        return jnp.concatenate((tend, tend[:, :1, :]), axis=1)
    return tend


def advect_w_flux(
    w: jax.Array,
    vel: CoupledVelocities,
    *,
    rdx: float,
    rdy: float,
    rdn: jax.Array,
    fzm: jax.Array,
    fzp: jax.Array,
    top_lid: bool = True,
) -> jax.Array:
    """WRF flux-form coupled w advection tendency (h=5, v=3).

    ``w`` is on z-faces ``(nz+1, ny, nx)``.  The horizontal transporting
    velocities ``ru/rv`` are interpolated to w (full) levels with ``fzm/fzp``;
    the vertical flux uses the mass-level coupled velocity built from rom.
    Source: ``advect_w`` (``module_advect_em.F:4364-6064``).  Returns the
    coupled tendency ``d(mu*w)/dt`` on the w faces (surface/top faces zeroed by
    the lower-BC / rigid-lid handling downstream).

    ``top_lid`` selects the WRF top-face (lid) handling.  When ``top_lid`` is
    True (the idealized rigid-lid configuration) the top w-face carries zero
    vertical flux and no lid pickup, matching the closed F7 idealized path
    exactly.  When ``top_lid`` is False (open/top-damped real configs) the WRF
    ``advect_w`` top-face branch (``module_advect_em.F:6014-6028``) is applied:
    the top face gets the 2nd-order flux ``0.25*(rom(kde)+rom(kde-1))*(w(kde)+
    w(kde-1))`` and the lid pickup ``tend(kde) += 2*rdn(ktf)*vflux(kde)``.
    """

    nz_mass = vel.ru.shape[0]  # = nz
    nzp1 = w.shape[0]  # nz+1
    # ru/rv on mass levels -> interpolate to w (full) levels: rw(k)=fzm(k)*r(k)+fzp(k)*r(k-1).
    ru_w = _mass_to_full_levels(vel.ru, fzm, fzp)  # (nz+1, ny, nx)
    rv_w = _mass_to_full_levels(vel.rv, fzm, fzp)
    # x-flux of w by ru_w (w and ru_w both on full levels, mass-located in x).
    fqx = ru_w * flux5_face_periodic(w, ru_w, axis=2)
    tend = -rdx * (jnp.roll(fqx, -1, axis=2) - fqx)
    if w.shape[1] > 1:
        fqy = rv_w * flux5_face_periodic(w, rv_w, axis=1)
        tend = tend - rdy * (jnp.roll(fqy, -1, axis=1) - fqy)
    # vertical flux of w lives on MASS levels (between w faces): vel = mass-level rom
    # average 0.5*(rom(k)+rom(k+1)); flux3 of the w faces; tend on faces via rdn.
    tend = tend + _vertical_flux_div_w(w, vel.rom, rdn, top_lid=top_lid)
    return tend


def _mass_to_full_levels(field_mass: jax.Array, fzm: jax.Array, fzp: jax.Array) -> jax.Array:
    """Interpolate a mass-level (nz,..) field to full/w levels (nz+1,..).

    Interior face k: fzm(k)*field(k)+fzp(k)*field(k-1).  Bottom/top faces use the
    nearest mass level (rigid extrapolation); they carry no net flux divergence
    contribution for the rigid-boundary w advection.
    """

    nz = int(field_mass.shape[0])
    out = jnp.zeros((nz + 1,) + tuple(field_mass.shape[1:]), dtype=field_mass.dtype)
    interior = fzm[1:nz, None, None] * field_mass[1:nz, :, :] + fzp[1:nz, None, None] * field_mass[: nz - 1, :, :]
    out = out.at[1:nz, :, :].set(interior)
    out = out.at[0, :, :].set(field_mass[0, :, :])
    out = out.at[nz, :, :].set(field_mass[nz - 1, :, :])
    return out


def _avg_to_u_face_x_w(rom: jax.Array) -> jax.Array:
    """rom (nz+1,ny,nx) averaged in x onto u faces: 0.5*(rom(i)+rom(i-1))."""

    return 0.5 * (rom + jnp.roll(rom, 1, axis=-1))


def _avg_to_v_face_y_w(rom: jax.Array) -> jax.Array:
    return 0.5 * (rom + jnp.roll(rom, 1, axis=-2))


def _vertical_flux_div_3(field_mass: jax.Array, romq: jax.Array, rdzw: jax.Array, fzm: jax.Array, fzp: jax.Array) -> jax.Array:
    """Vertical flux divergence (order 3) of a MASS-level field (u or v stagger).

    ``romq`` is the transporting vertical velocity at w faces (nz+1).  Interior
    w faces k=2..nz-1 use flux3 on the mass field; faces 1 and nz-1 use the
    2nd-order fzm/fzp interpolation; surface/top faces carry zero flux.  The
    tendency is on mass levels: ``-rdzw(k)*(vflux(k+1)-vflux(k))``.
    """

    nz = int(field_mass.shape[0])
    vflux = jnp.zeros((nz + 1,) + tuple(field_mass.shape[1:]), dtype=field_mass.dtype)
    if nz >= 4:
        q_km2 = field_mass[: nz - 3, :, :]
        q_km1 = field_mass[1 : nz - 2, :, :]
        q_k = field_mass[2 : nz - 1, :, :]
        q_kp1 = field_mass[3:nz, :, :]
        rom_k = romq[2 : nz - 1, :, :]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        # WRF advect_u/v vertical flux (module_advect_em.F:1474-1480): the flux at
        # face k is ``vel*flux3(u(k-2..k+1), -vel)`` with ``vel=0.5*(rom(i-1,k)+
        # rom(i,k))`` (here ``romq`` is that x-averaged rom).  WRF's flux3
        # (:202-204) applies ``sign(1.,ua)`` to the correction with ``ua = -vel``,
        # so the upwind correction enters as ``-sign(vel)*|...| = -|vel|*corr``
        # (DISSIPATIVE).  The previous JAX code used ``sign(+romq)``, i.e. it
        # ADDED ``+|vel|*corr`` -- the opposite sign, an ANTI-dissipative term that
        # excited a growing 2Δz vertical mode in u/v in the Straka cold-pool
        # descent column (F7N: novadv test removed the mode; the scalar path
        # advect_scalar_flux already negates the velocity correctly).  Use the WRF
        # sign: subtract |vel|*corr.
        flux3 = flux4 + jnp.sign(-rom_k) * corr
        vflux = vflux.at[2 : nz - 1, :, :].set(rom_k * flux3)
    if nz >= 2:
        vflux = vflux.at[1, :, :].set(
            romq[1, :, :] * (fzm[1] * field_mass[1, :, :] + fzp[1] * field_mass[0, :, :])
        )
        vflux = vflux.at[nz - 1, :, :].set(
            romq[nz - 1, :, :] * (fzm[nz - 1] * field_mass[nz - 1, :, :] + fzp[nz - 1] * field_mass[nz - 2, :, :])
        )
    return -rdzw[:, None, None] * (vflux[1:, :, :] - vflux[:nz, :, :])


def _vertical_flux_div_w(w: jax.Array, rom: jax.Array, rdn: jax.Array, *, top_lid: bool = True) -> jax.Array:
    """Vertical flux divergence (order 3) of the w field (on z-faces, nz+1).

    Source: ``advect_w`` vertical block (``module_advect_em.F:5996-6029``,
    ``vert_order==3``).  WRF indexes ``w`` and ``rom`` at full levels k; the
    vertical flux at face k uses ``vel = 0.5*(rom(k)+rom(k-1))`` and
    ``flux3(w[k-2..k+1], -vel)`` (note the flipped sign on ``vel``).  Interior w
    faces ``k=kts+2..ktf`` (JAX 0-based ``2..nz-1``) use flux3; ``kts+1`` (JAX 1)
    uses 2nd order.  The tendency on w faces ``k=kts+1..ktf`` (JAX ``1..nz-1``) is
    ``-rdn(k)*(vflux(k+1)-vflux(k))``.

    Top face (lid), WRF Python 0-based index ``nz`` = WRF ``ktf+1=kde``:

    * ``top_lid=True`` (rigid-lid idealized config): the top-face flux stays 0
      and there is no lid pickup, matching the closed F7 idealized path.
    * ``top_lid=False`` (open/top-damped real config): the WRF
      ``advect_w`` top branch is applied --
      ``vflux(kde)=0.25*(rom(kde)+rom(kde-1))*(w(kde)+w(kde-1))``
      (``module_advect_em.F:6014-6015``) is added into the interior face-(nz-1)
      tendency via ``vflux(k+1)`` and the lid pickup
      ``tend(kde) += 2*rdn(ktf)*vflux(kde)`` (``:6025-6028``) closes the open top.
    """

    nzp1 = int(w.shape[0])
    nz = nzp1 - 1  # mass levels; w faces 0..nz
    # vel at face k = 0.5*(rom(k)+rom(k-1)); valid for faces 1..nz.
    vel_face = 0.5 * (rom + jnp.roll(rom, 1, axis=0))  # (nz+1,..); index 0 invalid
    vflux = jnp.zeros_like(w)  # (nz+1, ny, nx); vflux at face k
    # flux3 at interior faces k=3..nz-1 (WRF kts+3..ktf-1): stencil w[k-2,k-1,k,k+1], vel=-vel_face(k)
    if nz >= 4:
        q_km2 = w[1 : nz - 2, :, :]  # w(k-2) for k=3..nz-1 -> idx 1..nz-3
        q_km1 = w[2 : nz - 1, :, :]
        q_k = w[3:nz, :, :]
        q_kp1 = w[4 : nz + 1, :, :]
        velz = -vel_face[3:nz, :, :]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        flux3 = flux4 + jnp.sign(velz) * corr
        vflux = vflux.at[3:nz, :, :].set(vel_face[3:nz, :, :] * flux3)
    # k=1 (kts+1): 2nd order 0.25*(rom(1)+rom(0))*(w(1)+w(0)) = vel_face(1)*0.5*(w1+w0)
    if nz >= 2:
        vflux = vflux.at[1, :, :].set(vel_face[1, :, :] * 0.5 * (w[1, :, :] + w[0, :, :]))
    # k=2 (kts+2) and k=nz-1 (ktf): flux3 with stencil w[k-2..k+1], vel=-vel_face(k)
    def _flux3_at(k: int) -> jax.Array:
        velz = -vel_face[k, :, :]
        q_km2 = w[k - 2, :, :]
        q_km1 = w[k - 1, :, :]
        q_k = w[k, :, :]
        q_kp1 = w[k + 1, :, :]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        return vel_face[k, :, :] * (flux4 + jnp.sign(velz) * corr)
    if nz >= 4:
        vflux = vflux.at[2, :, :].set(_flux3_at(2))
        vflux = vflux.at[nz - 1, :, :].set(_flux3_at(nz - 1))
    # Top-face (lid) flux.  WRF advect_w (module_advect_em.F:6014-6015) overwrites
    # vflux(kde) with the 2nd-order form for the open top; the rigid lid keeps it 0.
    if (not top_lid) and nz >= 1:
        vflux = vflux.at[nz, :, :].set(vel_face[nz, :, :] * 0.5 * (w[nz, :, :] + w[nz - 1, :, :]))
    # tendency on interior w faces k=1..nz-1: -rdn(k)*(vflux(k+1)-vflux(k)).
    tend = jnp.zeros_like(w)
    if nz >= 2:
        interior = -rdn[1:nz, None, None] * (vflux[2 : nz + 1, :, :] - vflux[1:nz, :, :])
        tend = tend.at[1:nz, :, :].set(interior)
    # Lid pickup (WRF module_advect_em.F:6025-6028): tend(kde) += 2*rdn(ktf)*vflux(kde).
    # ktf = kde-1 (JAX nz-1), so rdn(ktf) = rdn[nz-1].  Only for the open top.
    if (not top_lid) and nz >= 1:
        tend = tend.at[nz, :, :].add(2.0 * rdn[nz - 1] * vflux[nz, :, :])
    return tend


def _restore_periodic_u_face(tend: jax.Array, was_face: bool) -> jax.Array:
    """If u was passed as nx+1 faces, append the periodic wrap face."""

    if was_face:
        return jnp.concatenate((tend, tend[:, :, :1]), axis=2)
    return tend


__all__ = [
    "CoupledVelocities",
    "couple_velocities_periodic",
    "flux5_face_periodic",
    "advect_scalar_flux",
    "advect_u_flux",
    "advect_v_flux",
    "advect_w_flux",
]
