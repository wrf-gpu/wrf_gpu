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


__all__ = [
    "CoupledVelocities",
    "couple_velocities_periodic",
    "flux5_face_periodic",
    "advect_scalar_flux",
]
