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
``mrdx = msftx*rdx`` (``module_advect_em.F:3387-3388``).  Real-grid map
factor terms are threaded for the periodic high-order path; specified/nested
boundary order degradation remains out of scope for this module.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies


configure_jax_x64()
_SHARDED_HALO_CONTEXT: tuple[object, int] | None = None


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
    """Mass-coupled velocities and map factors for flux-form advection."""

    ru: jax.Array  # (nz, ny, nx+1) coupled u at x faces = (c1h*muu+c2h)*u/msfuy
    rv: jax.Array  # (nz, ny+1, nx) coupled v at y faces = (c1h*muv+c2h)*v*msfvx_inv
    rom: jax.Array  # (nz+1, ny, nx) coupled omega ww at w faces (calc_ww_cp)
    msftx: jax.Array | None = None  # (ny, nx) mass-point x map factor
    msfux: jax.Array | None = None  # (ny, nx) periodic-collapsed u-point x map factor
    msfvy: jax.Array | None = None  # (ny, nx) periodic-collapsed v-point y map factor
    msfvx: jax.Array | None = None  # (ny, nx) periodic-collapsed v-point x map factor
    # v0.14 SPECIFIED-boundary advection degradation: static switch + the
    # edge-faithful FULL-FACE coupled velocities (couple_uv_specified; the
    # periodic-collapsed ru/rv drop the last staggered face and wrap face 0's
    # mass).  None/False on every existing path (byte-identical).
    specified: bool = False
    ru_full: jax.Array | None = None  # (nz, ny, nx+1) edge-faithful coupled u
    rv_full: jax.Array | None = None  # (nz, ny+1, nx) edge-faithful coupled v


def _muu_face(mu_total: jax.Array) -> jax.Array:
    """muu = 0.5*(mu(i)+mu(i-1)) at x faces, periodic (calc_mu_uv / calc_ww_cp:700)."""

    return 0.5 * (mu_total + jnp.roll(mu_total, 1, axis=-1))


def _muv_face(mu_total: jax.Array) -> jax.Array:
    """muv = 0.5*(mu(j)+mu(j-1)) at y faces, periodic."""

    return 0.5 * (mu_total + jnp.roll(mu_total, 1, axis=-2))


def _mass_factor_or_one(factor: jax.Array | None, reference: jax.Array) -> jax.Array:
    """Return a mass-point map factor or shaped unity for the unit-map path."""

    if factor is None:
        return jnp.ones(tuple(reference.shape[-2:]), dtype=reference.dtype)
    return jnp.asarray(factor, dtype=reference.dtype)


def _x_face_factor_or_one(factor: jax.Array | None, *, nx: int, reference: jax.Array) -> jax.Array:
    """Return an x-face map factor collapsed to the periodic ``nx`` faces."""

    if factor is None:
        return jnp.ones((int(reference.shape[-2]), nx), dtype=reference.dtype)
    arr = jnp.asarray(factor, dtype=reference.dtype)
    return _collapse_x_face_periodic(arr, nx=nx) if arr.shape[-1] == nx + 1 else arr


def _collapse_x_face_periodic(face: jax.Array, *, nx: int) -> jax.Array:
    """Collapse an ``nx+1`` x-face field to the periodic ``nx`` faces."""

    if int(face.shape[-1]) != int(nx) + 1:
        return face
    collapsed = face[..., :nx]
    context = _SHARDED_HALO_CONTEXT
    if context is None:
        return collapsed
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return collapsed
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("flux-form sharded x-face collapse supports x-axis decomposition only")
    h = int(width)
    owned = int(nx) - 2 * h
    if owned < 1:
        raise ValueError("haloed x-face field has no owned cells")
    rank = jax.lax.axis_index(str(sharding.axis_name))
    start = rank * owned
    global_nx = owned * int(sharding.resolved_partitions())
    is_last = start + owned == global_nx
    right_start = h + owned
    replacement = face[..., right_start + 1 : right_start + 1 + h]
    corrected = collapsed.at[..., right_start:].set(replacement)
    return jnp.where(is_last, corrected, collapsed)


def _y_face_factor_or_one(factor: jax.Array | None, *, ny: int, reference: jax.Array) -> jax.Array:
    """Return a y-face map factor collapsed to the periodic ``ny`` faces."""

    if factor is None:
        return jnp.ones((ny, int(reference.shape[-1])), dtype=reference.dtype)
    arr = jnp.asarray(factor, dtype=reference.dtype)
    return arr[:ny, :] if arr.shape[-2] == ny + 1 else arr


def stage_omega_specified(
    u: jax.Array,
    v: jax.Array,
    mu_total: jax.Array,
    *,
    c1h: jax.Array,
    c2h: jax.Array,
    dnw: jax.Array,
    rdx: float,
    rdy: float,
    msfuy: jax.Array,
    msfvx: jax.Array,
    msftx: jax.Array,
) -> jax.Array:
    """WRF ``calc_ww_cp`` stage omega for SPECIFIED-boundary real domains.

    Source: ``module_big_step_utilities_em.F:640-782``.  Unlike
    :func:`couple_velocities_periodic` (whose ``rom`` wraps the x/y edges
    periodically -- exact in the interior but up to ~5x the physical omega on
    the outermost row/column of a real specified domain, v0.14 continuation
    proof D1: band rmse 6.99 / max 116 vs the oracle 2.48 / 24), this uses the
    domain's actual staggered ``u``/``v`` faces (``(nz, ny, nx+1)`` /
    ``(nz, ny+1, nx)``) with NO wrap, and edge-pads ``mu`` for the boundary
    face masses (the WRF halo carries the spec-zone value there).  Matches the
    fp64 numpy oracle to machine precision on every interior ring and to the
    halo-pad convention on ring 0.

    ``msfuy``/``msfvx`` are the STAGGERED map factors ``(ny, nx+1)`` /
    ``(ny+1, nx)``; ``msftx`` is the mass-point factor ``(ny, nx)``.
    """

    nz = int(u.shape[0])
    ny, nx = int(mu_total.shape[-2]), int(mu_total.shape[-1])
    mu_pad_x = jnp.pad(mu_total, ((0, 0), (1, 0)), mode="edge")
    muu = 0.5 * (mu_pad_x[:, 1:] + mu_pad_x[:, :-1])  # faces 0..nx-1
    muu = jnp.concatenate([muu, mu_total[:, -1:]], axis=1)  # face nx (edge pad)
    mu_pad_y = jnp.pad(mu_total, ((1, 0), (0, 0)), mode="edge")
    muv = 0.5 * (mu_pad_y[1:, :] + mu_pad_y[:-1, :])
    muv = jnp.concatenate([muv, mu_total[-1:, :]], axis=0)

    c1 = c1h[:, None, None]
    c2 = c2h[:, None, None]
    ru = (c1 * muu[None, :, :] + c2) * u / msfuy[None, :, :]  # (nz, ny, nx+1)
    rv = (c1 * muv[None, :, :] + c2) * v / msfvx[None, :, :]  # (nz, ny+1, nx)
    divv = msftx[None, :, :] * dnw[:, None, None] * (
        float(rdx) * (ru[:, :, 1:] - ru[:, :, :-1]) + float(rdy) * (rv[:, 1:, :] - rv[:, :-1, :])
    )  # (nz, ny, nx)
    dmdt = jnp.sum(divv, axis=0, keepdims=True)
    increments = -(dnw[:, None, None] * c1h[:, None, None] * dmdt) - divv
    cum = jnp.cumsum(increments, axis=0)
    rom = jnp.zeros((nz + 1, ny, nx), dtype=cum.dtype)
    rom = rom.at[1:nz, :, :].set(cum[: nz - 1, :, :])
    return rom


# ---------------------------------------------------------------------------
# v0.14 SPECIFIED-boundary horizontal flux degradation (stage3/wrapper sprint).
#
# WRF degrades the horizontal flux order near every non-periodic/non-symmetric
# boundary (module_advect_em.F, the ``degrade_*`` blocks of advect_scalar :3392+,
# advect_u :126-1526, advect_v :1530-3024, advect_w :4364+, all horz_order==5):
#   * the outermost cells/faces get NO horizontal advection at all (the spec
#     zone is boundary-driven),
#   * the flux face next to the boundary is 2nd-order (for the NORMAL momentum
#     component with the SPECIFIED upstream rule: the boundary-face value is
#     replaced by the interior face under inflow),
#   * the next face in is 3rd-order (flux3 = flux4 - upwind correction),
#   * faces >= 3 cells in use the full 5th-order flux.
# The periodic JAX implementation wraps every stencil with jnp.roll, so rings
# 0-2 of a real specified domain advect with OPPOSITE-EDGE values and no order
# degradation -- the documented out-of-scope restriction in this module's
# docstring, and the v0.14 ring-1 mass-drift fingerprint (stage-compare ring-1
# mu increment error 11.7 Pa/stage vs WRF 0.94, immune to the boundary-cadence
# fix because it enters through the large-step tendencies).
# ---------------------------------------------------------------------------


def _shift0(field: jax.Array, shift: int, axis: int) -> jax.Array:
    """Shift ``field`` by ``shift`` along ``axis`` with ZERO fill (no wrap).

    ``shift=+1`` brings cell ``i-1`` to position ``i`` (like ``jnp.roll(+1)``)
    but fills the vacated edge with zeros; the specified tier masks never read
    the filled cells.
    """

    n = int(field.shape[axis])
    s = int(shift)
    if s == 0:
        return field
    pad = [(0, 0)] * field.ndim
    if s > 0:
        pad[axis] = (s, 0)
        sl = [slice(None)] * field.ndim
        sl[axis] = slice(0, n)
    else:
        pad[axis] = (0, -s)
        sl = [slice(None)] * field.ndim
        sl[axis] = slice(-s, n - s)
    return jnp.pad(field, pad)[tuple(sl)]


def _axis_index(field: jax.Array, axis: int) -> jax.Array:
    n = int(field.shape[axis])
    shape = [1] * field.ndim
    shape[axis] = n
    return jnp.arange(n).reshape(shape)


def specified_flux_faces(
    field: jax.Array,
    vel: jax.Array,
    axis: int,
    *,
    upstream: bool = False,
) -> jax.Array:
    """WRF order-5 SPECIFIED-boundary tiered flux faces along ``axis``.

    Face index ``m`` sits between cells ``m-1`` and ``m`` (the same convention
    as :func:`flux5_face_periodic`); ``vel`` is the transporting velocity at
    the SAME face locations.  Tier map for axis extent ``n`` (spec_bdy width 1,
    WRF advect_* horz_order==5 degrade blocks):

      m == 1        2nd order (with the specified upstream rule when
                    ``upstream`` -- the NORMAL momentum component: the
                    boundary-side value is replaced by the interior one under
                    inflow/outflow, module_advect_em.F "specified uses upstream
                    normal wind at boundaries")
      m == 2        flux3 (flux4 - sign(vel)*correction)
      3..n-3        flux5 (flux6 - sign(vel)*correction)
      m == n-2      flux3
      m == n-1      2nd order (mirrored upstream rule when ``upstream``)
      m == 0        ZERO (never consumed: the outermost cells get no advection)

    Every stencil neighbour is fetched with the zero-fill shift, so no value
    ever wraps across the domain.  Returns ``vel * face_value`` (the flux).
    """

    q_im3 = _shift0(field, 3, axis)
    q_im2 = _shift0(field, 2, axis)
    q_im1 = _shift0(field, 1, axis)
    q_i = field
    q_ip1 = _shift0(field, -1, axis)
    q_ip2 = _shift0(field, -2, axis)

    face6 = (37.0 * (q_i + q_im1) - 8.0 * (q_ip1 + q_im2) + (q_ip2 + q_im3)) / 60.0
    corr6 = ((q_ip2 - q_im3) - 5.0 * (q_ip1 - q_im2) + 10.0 * (q_i - q_im1)) / 60.0
    face5 = face6 - jnp.sign(vel) * corr6
    face4 = (7.0 * (q_i + q_im1) - (q_ip1 + q_im2)) / 12.0
    corr4 = ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1)) / 12.0
    face3 = face4 + jnp.sign(vel) * corr4
    face2 = 0.5 * (q_i + q_im1)
    if upstream:
        # WRF: at the low face (m==1) the boundary-side value is q_im1 (the
        # spec face); under inflow (q_i < 0, flow INTO the domain edge) it is
        # replaced by the interior value q_i.  Mirrored at the high face.
        low = 0.5 * (q_i + jnp.where(q_i < 0.0, q_i, q_im1))
        high = 0.5 * (q_im1 + jnp.where(q_im1 > 0.0, q_im1, q_i))
        n = int(field.shape[axis])
        idx = _axis_index(field, axis)
        face2 = jnp.where(idx == 1, low, jnp.where(idx == n - 1, high, face2))

    n = int(field.shape[axis])
    idx = _axis_index(field, axis)
    five = (idx >= 3) & (idx <= n - 3)
    three = (idx == 2) | (idx == n - 2)
    two = (idx == 1) | (idx == n - 1)
    face = jnp.where(five, face5, jnp.where(three, face3, jnp.where(two, face2, 0.0)))
    return vel * face


def _specified_div(fq: jax.Array, axis: int) -> jax.Array:
    """Masked flux divergence ``fq(m+1) - fq(m)`` for cells ``1..n-2``.

    The outermost cells (the WRF spec zone) receive NO horizontal advection
    (the divergence loops run ``[ids+1, ide-2]`` / ``[jds+1, jde-2]``).
    """

    div = _shift0(fq, -1, axis) - fq
    n = int(fq.shape[axis])
    idx = _axis_index(fq, axis)
    interior = (idx >= 1) & (idx <= n - 2)
    return jnp.where(interior, div, 0.0)


def couple_uv_specified(
    u: jax.Array,
    v: jax.Array,
    mu_total: jax.Array,
    *,
    c1h: jax.Array,
    c2h: jax.Array,
    msfuy: jax.Array,
    msfvx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Edge-faithful FULL-FACE coupled velocities for the specified path.

    Same construction as :func:`stage_omega_specified` (edge-padded face
    masses, the actual staggered faces, NO wrap): ``ru`` on ``(nz, ny, nx+1)``,
    ``rv`` on ``(nz, ny+1, nx)``.
    """

    mu_pad_x = jnp.pad(mu_total, ((0, 0), (1, 0)), mode="edge")
    muu = 0.5 * (mu_pad_x[:, 1:] + mu_pad_x[:, :-1])
    muu = jnp.concatenate([muu, mu_total[:, -1:]], axis=1)
    mu_pad_y = jnp.pad(mu_total, ((1, 0), (0, 0)), mode="edge")
    muv = 0.5 * (mu_pad_y[1:, :] + mu_pad_y[:-1, :])
    muv = jnp.concatenate([muv, mu_total[-1:, :]], axis=0)
    c1 = c1h[:, None, None]
    c2 = c2h[:, None, None]
    ru = (c1 * muu[None, :, :] + c2) * u / msfuy[None, :, :]
    rv = (c1 * muv[None, :, :] + c2) * v / msfvx[None, :, :]
    return ru, rv


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
    msfuy: jax.Array | None = None,
    msfvx: jax.Array | None = None,
    msftx: jax.Array | None = None,
    msfux: jax.Array | None = None,
    msfvy: jax.Array | None = None,
) -> CoupledVelocities:
    """Build ``ru``/``rv``/``rom`` for the periodic dry case.

    Source: ``couple_momentum`` (``module_em.F:195``) and ``calc_ww_cp``
    (``module_big_step_utilities_em.F:640-782``).  ``u`` is on x faces
    ``(nz, ny, nx+1)`` collapsed to ``(nz, ny, nx)`` periodically (the project's
    one-row C-grid keeps ``nx+1`` u faces; the last equals the first under
    periodicity).  WRF map factors enter the mass-coupled velocities and the
    diagnosed ``ww`` recurrence; when omitted they reduce exactly to unity for
    idealized slabs.
    """

    nz = int(u.shape[0])
    ny = int(mu_total.shape[-2])
    nx = int(mu_total.shape[-1])
    # u has nx+1 faces; in the periodic project layout face nx == face 0.  Use the
    # first nx faces as the staggered u at x-faces 0..nx-1 (WRF u(i) at face i).
    u_faces = _collapse_x_face_periodic(u, nx=nx) if u.shape[-1] == nx + 1 else u
    v_faces = v[:, :-1, :] if v.shape[-2] == ny + 1 else v
    msfuy_u = _x_face_factor_or_one(msfuy, nx=nx, reference=mu_total)
    msfvx_v = _y_face_factor_or_one(msfvx, ny=ny, reference=mu_total)
    msftx_m = _mass_factor_or_one(msftx, mu_total)
    msfux_u = _x_face_factor_or_one(msfux, nx=nx, reference=mu_total)
    msfvy_v = _y_face_factor_or_one(msfvy, ny=ny, reference=mu_total)

    muu = _muu_face(mu_total)  # (ny, nx)
    muv = _muv_face(mu_total)  # (ny, nx)
    mass_u = c1h[:, None, None] * muu[None, :, :] + c2h[:, None, None]  # (nz, ny, nx)
    mass_v = c1h[:, None, None] * muv[None, :, :] + c2h[:, None, None]
    ru = mass_u * u_faces / msfuy_u[None, :, :]
    rv = mass_v * v_faces / msfvx_v[None, :, :]

    # calc_ww_cp: divv(k)=msftx*dnw(k)*(rdx*d_i(mu*u/msfuy)+rdy*d_j(mu*v/msfvx)).
    ru_ip1 = jnp.roll(ru, -1, axis=-1)
    rv_jp1 = jnp.roll(rv, -1, axis=-2)
    divv = dnw[:, None, None] * msftx_m[None, :, :] * (
        rdx * (ru_ip1 - ru) + rdy * (rv_jp1 - rv)
    )  # (nz, ny, nx)
    dmdt = jnp.sum(divv, axis=0, keepdims=True)  # (1, ny, nx) integral over levels
    # ww(k) = ww(k-1) - dnw(k-1)*c1h(k-1)*dmdt - divv(k-1), with ww(0)=ww(nz)=0.
    increments = -(dnw[:, None, None] * c1h[:, None, None] * dmdt) - divv  # (nz, ny, nx) per k-1
    # ww at faces 1..nz-1 = cumulative sum of increments[0..k-1]; ww(0)=0, ww(nz)=0 (rigid).
    cum = jnp.cumsum(increments, axis=0)  # (nz, ny, nx); cum[k-1] is ww at face k
    # Sprint U P0-1: ``rom`` must carry the result dtype of the increment chain
    # (which promotes to fp64 whenever the c1h/c2h/dnw metrics are fp64), not
    # ``u.dtype``.  Allocating at ``u.dtype`` silently down-cast the coupled
    # vertical velocity to fp32 when ``u`` was fp32, which both (a) tripped the
    # fp64->fp32 scatter warning and (b) ran the whole flux-form advection in
    # fp32 despite force_fp64.  result_type keeps it fp64 on the operational
    # force_fp64 path and is the identity once ``u`` is fp64.
    rom = jnp.zeros((nz + 1,) + tuple(mu_total.shape), dtype=cum.dtype)
    rom = rom.at[1:nz, :, :].set(cum[: nz - 1, :, :])
    # face nz (top) stays 0 (rigid lid); face 0 (surface) stays 0.
    return CoupledVelocities(
        ru=ru,
        rv=rv,
        rom=rom,
        msftx=msftx_m,
        msfux=msfux_u,
        msfvy=msfvy_v,
        msfvx=msfvx_v,
    )


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
    callers that hold uncoupled state must divide by mass.  Horizontal
    divergence uses WRF ``mrdx=msftx*rdx`` / ``mrdy=msftx*rdy``; vertical
    divergence has no additional horizontal map factor in this high-order path.

    Horizontal (order 5, periodic, no degradation): for each x face i,
    ``fqx(i) = ru(i) * flux5(field[i-3..i+2], ru(i))`` and
    ``tend -= rdx*(fqx(i+1)-fqx(i))``.  Same in y with ``rv``.

    Vertical (order 3): interior faces k=2..ktf-1 use ``flux3``; the two faces
    adjacent to the rigid boundaries use 2nd-order ``fzm/fzp`` interpolation;
    the surface (k=0) and top (k=nz) faces carry zero flux.
    """

    # ---- x flux divergence ----
    msftx = _mass_factor_or_one(vel.msftx, field)
    if bool(getattr(vel, "specified", False)):
        # v0.14 SPECIFIED-boundary degraded fluxes (advect_scalar degrade_*
        # blocks): tiered faces, zero-fill stencils, NO advection for the
        # outermost cells in each direction.
        fqx = specified_flux_faces(field, vel.ru, axis=2)
        tend = -msftx[None, :, :] * rdx * _specified_div(fqx, axis=2)
        fqy = specified_flux_faces(field, vel.rv, axis=1)
        tend = tend - msftx[None, :, :] * rdy * _specified_div(fqy, axis=1)
    else:
        # fqx located at left face of cell i; ru is the coupled u at face i.
        fqx = vel.ru * flux5_face_periodic(field, vel.ru, axis=2)  # (nz, ny, nx)
        fqx_ip1 = jnp.roll(fqx, -1, axis=2)
        tend = -msftx[None, :, :] * rdx * (fqx_ip1 - fqx)

        # ---- y flux divergence ----
        fqy = vel.rv * flux5_face_periodic(field, vel.rv, axis=1)  # (nz, ny, nx)
        fqy_jp1 = jnp.roll(fqy, -1, axis=1)
        tend = tend - msftx[None, :, :] * rdy * (fqy_jp1 - fqy)

    # ---- z flux divergence (order 3, rigid top/bottom) ----
    nz = int(field.shape[0])
    rom = vel.rom  # (nz+1, ny, nx) at w faces; rom[0]=rom[nz]=0
    # Sprint U P0-1: promote the scatter buffer to result_type(field, rom) so an
    # fp64 transporting velocity against an fp32 scalar field is carried in fp64
    # rather than silently down-cast on the .at[].set (identity when both fp64).
    out_dtype = jnp.result_type(field.dtype, rom.dtype)
    vflux = jnp.zeros((nz + 1,) + tuple(field.shape[1:]), dtype=out_dtype)
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


# --- positive-definite / monotonic scalar transport (advect_scalar_pd/mono) ---
#
# WRF ``module_advect_em.F`` offers a flux-renormalization limiter on TOP of the
# high-order flux-form scalar advection above, selected by the namelist option
# ``moist_adv_opt`` / ``scalar_adv_opt`` (WRF canonical values, NOT the order in
# which the sprint brief listed them):
#
#   * 0 -> none (plain h5/v3 -- the bit-for-bit DEFAULT path above).
#   * 1 -> positive-definite (``advect_scalar_pd``, :6069).  Smolarkiewicz
#          MWR-1989 FCT: blend the high-order flux toward a first-order monotone
#          (donor-cell) flux just enough that the low-order-updated value stays
#          NON-NEGATIVE.
#   * 2 -> monotonic (``advect_scalar_mono``, :9495).  Same renormalization but
#          the bound is the local field_old min/max of the cell + its 6
#          neighbours (no new extrema), enforced with separate inflow/outflow
#          scale factors.
#   * 3 -> WENO; 4 -> WENO-PD.  OUT OF SCOPE for this sprint (noted in the
#          handoff -- WENO is a separate ~1000-LOC reconstruction).
#
# This module implements 1 (PD) and 2 (monotonic) for the project's PERIODIC,
# unit-map, h=5/v=3 configuration -- the same scope restriction the plain path
# documents.  Both return the COUPLED tendency ``d(mu*phi)/dt`` so they are a
# drop-in replacement for ``advect_scalar_flux`` (selected by the option value).
#
# The limiter is a Flux-Corrected-Transport (FCT) construction and therefore
# needs three things the plain tendency does NOT:
#   * ``field_old`` -- the scalar value at the START of the RK3 timestep (WRF
#     applies the limiter only on the final, full-``dt`` RK3 stage; the low-order
#     monotone update and the qmin/qmax bounds are built from ``field_old``).
#   * ``mu_old`` and ``mut`` -- the start-of-step and current total dry-air mass
#     (the low-order update is on the coupled scalar ``(c1*mu+c2)*phi``).
#   * ``dt`` -- the full RK3 step (the antidiffusive flux is scaled so the
#     positivity/monotonicity bound holds after the EXPLICIT update by ``dt``).
#
# The high-order flux ``fqx`` here is the SAME as the plain path: in the periodic
# unit-map h5 case WRF's ``vel*flux6 - sign*corr`` with ``vel = ru`` is exactly
# ``ru * flux5_face_periodic(field, ru)``.  The antidiffusive flux is
# ``A = high_order - low_order``; the limiter scales only ``A``.


_PD_EPS = 1.0e-20


def _flux_upwind_face(
    field_old: jax.Array, vel: jax.Array, mu_face: jax.Array, dxy: float, dt: float, axis: int
) -> jax.Array:
    """WRF first-order monotone (donor-cell) flux at the left face of cell ``i``.

    Source: ``advect_scalar_pd`` ``flux_upwind`` statement-function
    (``module_advect_em.F:6190``) and its use at each face.  WRF defines

        cr   = vel*dt/dx/mu                      (cell Courant number)
        fqxl = mu*(dx/dt)*flux_upwind(q_im1, q_i, cr)
        flux_upwind(q_im1,q_i,cr) = 0.5*min(1, cr+|cr|)*q_im1
                                  + 0.5*max(-1, cr-|cr|)*q_i

    For ``0 <= cr <= 1`` this reduces to ``vel*q_im1`` (pure donor cell); the
    ``min/max`` clamps bound the scheme at CFL>=1.  ``vel`` is the mass-coupled
    face velocity (project ``ru``/``rv``/``rom``) and ``mu_face`` the face dry-air
    mass, so ``vel/mu_face`` is the physical face velocity and ``cr`` the true
    Courant number on the unit-map periodic grid (``dx = 1/rdx``).
    """

    q_im1 = jnp.roll(field_old, 1, axis=axis)
    q_i = field_old
    cr = vel * dt / dxy / mu_face
    abs_cr = jnp.abs(cr)
    w_im1 = 0.5 * jnp.minimum(1.0, cr + abs_cr)
    w_i = 0.5 * jnp.maximum(-1.0, cr - abs_cr)
    return mu_face * (dxy / dt) * (w_im1 * q_im1 + w_i * q_i)


def _flux_upwind_face_z(
    field_old: jax.Array, rom: jax.Array, mu_h: jax.Array, rdzw: jax.Array, dt: float
) -> jax.Array:
    """WRF first-order monotone vertical flux at w-faces ``k`` (donor cell).

    Source: ``advect_scalar_pd`` vertical block (``module_advect_em.F:7478-7541``).
    WRF builds ``dz = 2/(rdzw(k)+rdzw(k-1))``, ``mu = c1*mut+c2`` (mass level),
    ``cr = rom*dt/dz/mu`` and ``fqzl(k) = mu*(dz/dt)*flux_upwind(q(k-1), q(k), cr)``.
    Returns the (nz+1) low-order flux at every w face; faces 0 and nz carry no
    flux (rigid boundaries) consistent with the high-order path.
    """

    nz = int(field_old.shape[0])
    out_dtype = jnp.result_type(field_old.dtype, rom.dtype, rdzw.dtype)
    fqzl = jnp.zeros((nz + 1,) + tuple(field_old.shape[1:]), dtype=out_dtype)
    if nz < 2:
        return fqzl
    # interior faces k = 1..nz-1 (WRF kts+1..ktf): donor cell between k-1 and k.
    q_km1 = field_old[: nz - 1, :, :]  # field at mass level k-1, for faces 1..nz-1
    q_k = field_old[1:nz, :, :]
    rom_f = rom[1:nz, :, :]
    # dz at face k = 2/(rdzw(k)+rdzw(k-1)); mu at the face uses the mass-level mu.
    rdzw_k = rdzw[1:nz, None, None]
    rdzw_km1 = rdzw[: nz - 1, None, None]
    dz = 2.0 / (rdzw_k + rdzw_km1)
    mu_face = 0.5 * (mu_h[1:nz, :, :] + mu_h[: nz - 1, :, :])
    cr = rom_f * dt / dz / mu_face
    abs_cr = jnp.abs(cr)
    w_km1 = 0.5 * jnp.minimum(1.0, cr + abs_cr)
    w_k = 0.5 * jnp.maximum(-1.0, cr - abs_cr)
    flux = mu_face * (dz / dt) * (w_km1 * q_km1 + w_k * q_k)
    fqzl = fqzl.at[1:nz, :, :].set(flux)
    return fqzl


def _high_order_flux_x(field: jax.Array, ru: jax.Array) -> jax.Array:
    """High-order x flux at the left face of cell i (= plain-path fqx). h5."""

    return ru * flux5_face_periodic(field, ru, axis=2)


def _high_order_flux_y(field: jax.Array, rv: jax.Array) -> jax.Array:
    return rv * flux5_face_periodic(field, rv, axis=1)


def _high_order_flux_z(field: jax.Array, rom: jax.Array, fzm: jax.Array, fzp: jax.Array) -> jax.Array:
    """High-order vertical flux at w-faces (= the plain-path vflux). v3."""

    nz = int(field.shape[0])
    out_dtype = jnp.result_type(field.dtype, rom.dtype)
    vflux = jnp.zeros((nz + 1,) + tuple(field.shape[1:]), dtype=out_dtype)
    if nz >= 4:
        q_km2 = field[: nz - 3, :, :]
        q_km1 = field[1 : nz - 2, :, :]
        q_k = field[2 : nz - 1, :, :]
        q_kp1 = field[3:nz, :, :]
        velz = -rom[2 : nz - 1, :, :]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        flux3 = flux4 + jnp.sign(velz) * corr
        vflux = vflux.at[2 : nz - 1, :, :].set(rom[2 : nz - 1, :, :] * flux3)
    if nz >= 2:
        vflux = vflux.at[1, :, :].set(
            rom[1, :, :] * (fzm[1] * field[1, :, :] + fzp[1] * field[0, :, :])
        )
        vflux = vflux.at[nz - 1, :, :].set(
            rom[nz - 1, :, :] * (fzm[nz - 1] * field[nz - 1, :, :] + fzp[nz - 1] * field[nz - 2, :, :])
        )
    return vflux


def _neighbour_min_max(field_old: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Local max/min of ``field_old`` over the cell and its 6 face neighbours.

    Source: ``advect_scalar_mono`` qmax/qmin accumulation
    (``module_advect_em.F:9625-10001``), periodic in x/y and clamped (no wrap) in
    z.  Returns ``(qmax, qmin)`` with the same shape as ``field_old``.
    """

    qmax = field_old
    qmin = field_old
    for axis in (2, 1):  # x, y: periodic neighbours
        qmax = jnp.maximum(qmax, jnp.maximum(jnp.roll(field_old, 1, axis=axis), jnp.roll(field_old, -1, axis=axis)))
        qmin = jnp.minimum(qmin, jnp.minimum(jnp.roll(field_old, 1, axis=axis), jnp.roll(field_old, -1, axis=axis)))
    # z: clamp at top/bottom (no vertical periodicity); neighbour-less faces keep
    # the cell value.
    up = jnp.concatenate((field_old[:1, :, :], field_old[:-1, :, :]), axis=0)  # field_old(k-1)
    dn = jnp.concatenate((field_old[1:, :, :], field_old[-1:, :, :]), axis=0)  # field_old(k+1)
    qmax = jnp.maximum(qmax, jnp.maximum(up, dn))
    qmin = jnp.minimum(qmin, jnp.minimum(up, dn))
    return qmax, qmin


def advect_scalar_flux_limited(
    field: jax.Array,
    field_old: jax.Array,
    vel: CoupledVelocities,
    *,
    scalar_adv_opt: int,
    mut: jax.Array,
    mu_old: jax.Array,
    c1: jax.Array,
    c2: jax.Array,
    rdx: float,
    rdy: float,
    rdzw: jax.Array,
    fzm: jax.Array,
    fzp: jax.Array,
    dt: float,
) -> jax.Array:
    """WRF positive-definite (opt=1) / monotonic (opt=2) limited scalar advection.

    Source: ``advect_scalar_pd`` (``module_advect_em.F:6069-7885``) and
    ``advect_scalar_mono`` (``:9495-10560``), periodic / unit-map / h5-v3 scope.

    Returns the COUPLED tendency ``d(mu*phi)/dt`` (identical return contract to
    ``advect_scalar_flux``), so callers select the limiter purely by the option
    value and add the result to the coupled prognostic exactly as before.

    ``scalar_adv_opt``:
      * 1 -> positive-definite (PD) -- the limited field stays >= 0 after a full
        explicit step by ``dt``.
      * 2 -> monotonic -- no new extrema beyond the local ``field_old`` min/max.

    For ``scalar_adv_opt`` 0 (or anything else) the caller must use the plain
    ``advect_scalar_flux``; this function is the OPT-IN limiter only and never
    touches the default path.

    Mass contract: ``mut`` is the CURRENT total dry-air mass column ``(ny, nx)`` and
    ``mu_old`` is the TOTAL dry-air mass at the START of the RK3 step.  WRF builds
    the limiter's start-of-step coupled mass as ``(c1*mub+c2)+(c1*mu_old)`` from a
    base/perturbation split; this project folds the base into the total, so a
    caller must pass the FULL old total mass as ``mu_old`` (NOT a perturbation) and
    the FULL current total as ``mut`` -- the function then forms ``c1*mu_old+c2``,
    which equals WRF's start mass when ``mub`` is folded in.

    The FCT construction (Smolarkiewicz / Zalesak):
      1. low-order monotone (donor-cell) fluxes ``fqxl/fqyl/fqzl``;
      2. antidiffusive fluxes ``A = high_order - low_order``;
      3. the low-order updated coupled value ``ph_low`` (PD) / ``ph_upwind``
         (mono) built from ``field_old`` and ``mu_old``;
      4. a per-cell scale (PD: one factor on the OUTGOING antidiffusive flux so
         the cell cannot go negative; mono: separate inflow/outflow factors so
         the cell stays within its neighbour min/max), combined at each FACE as
         the WRF ``min(scale_in, scale_out)`` of the two adjacent cells;
      5. tendency ``-div(A_limited + low_order)``.

    Mass conservation: every flux is a face quantity differenced as
    ``flux(i+1)-flux(i)``; scaling a face value scales the SAME number that enters
    both adjacent cells with opposite sign, so the global sum of the tendency is a
    telescoping (periodic) sum that is zero to round-off -> the limiter only
    REDISTRIBUTES mass, never creates or destroys it.
    """

    nz = int(field.shape[0])
    msftx = _mass_factor_or_one(vel.msftx, field)  # (ny, nx); unity on the unit map
    # Coupled face / mass dry-air masses (c1*mu + c2).  WRF ``mu`` at an x-face is
    # 0.5*(c1*mut(i)+c2 + c1*mut(i-1)+c2); the y-face and mass-level analogues
    # follow.  These match couple_velocities_periodic's mass_u/mass_v exactly.
    mu_h = c1[:, None, None] * mut[None, :, :] + c2[:, None, None]  # (nz, ny, nx)
    muu = _muu_face(mut)  # (ny, nx)
    muv = _muv_face(mut)
    mu_u = c1[:, None, None] * muu[None, :, :] + c2[:, None, None]  # x-face mass
    mu_v = c1[:, None, None] * muv[None, :, :] + c2[:, None, None]  # y-face mass

    dx = 1.0 / rdx
    dy = 1.0 / rdy

    # v0.14 SPECIFIED-boundary degradation (advect_scalar_pd/_mono carry the
    # same degrade blocks as advect_scalar): tiered high-order faces with
    # zero-fill stencils, donor faces zeroed at the never-consumed outermost
    # face, divergences masked to the WRF cell bounds below.
    _specified = bool(getattr(vel, "specified", False))

    # ---- low-order (first-order upwind / donor-cell) fluxes from field_old ----
    fqxl = _flux_upwind_face(field_old, vel.ru, mu_u, dx, dt, axis=2)  # face i
    fqyl = _flux_upwind_face(field_old, vel.rv, mu_v, dy, dt, axis=1)  # face j
    fqzl = _flux_upwind_face_z(field_old, vel.rom, mu_h, rdzw, dt)  # face k (nz+1)
    if _specified:
        fqxl = jnp.where(_axis_index(fqxl, 2) == 0, 0.0, fqxl)
        fqyl = jnp.where(_axis_index(fqyl, 1) == 0, 0.0, fqyl)

    # ---- high-order fluxes (= plain path), then antidiffusive A = hi - lo ----
    if _specified:
        fqx = specified_flux_faces(field, vel.ru, axis=2) - fqxl
        fqy = specified_flux_faces(field, vel.rv, axis=1) - fqyl
    else:
        fqx = _high_order_flux_x(field, vel.ru) - fqxl  # face i
        fqy = _high_order_flux_y(field, vel.rv) - fqyl  # face j
    fqz = _high_order_flux_z(field, vel.rom, fzm, fzp) - fqzl  # face k (nz+1)

    # ---- low-order updated coupled value (WRF ph_low / ph_upwind) ----
    # WRF builds the start-of-step coupled mass from mu_old/mub:
    # ``(c1*mub+c2)+(c1*mu_old)``.  On the project's dry single-domain idealized
    # path mub is folded into mut (no separate base/perturbation split), so the
    # start-of-step coupled mass is ``c1*mu_old+c2``.  Use that for the monotone
    # low-order update exactly as WRF builds it.
    mass_old = c1[:, None, None] * mu_old[None, :, :] + c2[:, None, None]  # (nz, ny, nx)
    _ip1 = (lambda a, ax: _shift0(a, -1, ax)) if _specified else (lambda a, ax: jnp.roll(a, -1, axis=ax))
    fqxl_ip1 = _ip1(fqxl, 2)
    fqyl_jp1 = _ip1(fqyl, 1)
    fqzl_kp1 = fqzl[1:, :, :]
    fqzl_k = fqzl[:nz, :, :]
    div_low = (
        msftx[None, :, :] * msftx[None, :, :] * (rdx * (fqxl_ip1 - fqxl) + rdy * (fqyl_jp1 - fqyl))
        + msftx[None, :, :] * rdzw[:, None, None] * (fqzl_kp1 - fqzl_k)
    )
    ph_low = mass_old * field_old - dt * div_low

    # Antidiffusive face values offset to the i+1 / j+1 / k+1 face for divergence.
    fqx_ip1 = _ip1(fqx, 2)
    fqy_jp1 = _ip1(fqy, 1)
    fqz_kp1 = fqz[1:, :, :]
    fqz_k = fqz[:nz, :, :]

    pos = lambda a: jnp.maximum(0.0, a)
    neg = lambda a: jnp.minimum(0.0, a)

    if int(scalar_adv_opt) == 1:
        # ----- positive-definite (WRF advect_scalar_pd :7744-7779) -----
        # flux_out: mass the antidiffusive fluxes would REMOVE from cell (i,k,j).
        # Note the z-flux opposite sign (mass coordinate decreases with k).
        flux_out = dt * (
            msftx[None, :, :] * msftx[None, :, :]
            * (rdx * (pos(fqx_ip1) - neg(fqx)) + rdy * (pos(fqy_jp1) - neg(fqy)))
            + msftx[None, :, :] * rdzw[:, None, None] * (neg(fqz_kp1) - pos(fqz_k))
        )
        # scale only when the outgoing antidiffusive flux would empty the cell.
        scale = jnp.where(
            flux_out > ph_low,
            jnp.maximum(0.0, ph_low / (flux_out + _PD_EPS)),
            jnp.ones_like(ph_low),
        )
        # WRF scales the OUTGOING component of each antidiffusive face flux by the
        # donor cell's scale (:7765-7773).  At x face i the cell on the -x side is
        # i-1 and on the +x side is i; fqx(i)>0 drains cell i-1 -> scale(i-1);
        # fqx(i)<0 drains cell i -> scale(i).
        scale_im1_x = jnp.roll(scale, 1, axis=2)
        scale_im1_y = jnp.roll(scale, 1, axis=1)
        fqx_lim = jnp.where(fqx > 0.0, scale_im1_x * fqx, jnp.where(fqx < 0.0, scale * fqx, fqx))
        fqy_lim = jnp.where(fqy > 0.0, scale_im1_y * fqy, jnp.where(fqy < 0.0, scale * fqy, fqy))
        # z: at w face m the cell BELOW is m-1, ABOVE is m.  WRF advect_scalar_pd
        # (:7771-7772) scales each face by the DONOR cell's scale -- and in the
        # mass coordinate the flux sign is inverted (the vertical coordinate
        # decreases with increasing k), so a NEGATIVE fqz(m) drains the cell BELOW
        # (m-1) and a POSITIVE fqz(m) drains the cell ABOVE (m):
        #   IF fqz(k+1)<0 -> fqz(k+1)*=scale(k)   [face m=k+1 scaled by cell m-1]
        #   IF fqz(k)  >0 -> fqz(k)  *=scale(k)   [face m=k   scaled by cell m  ]
        scale_pad = jnp.concatenate((scale[:1, :, :], scale, scale[-1:, :, :]), axis=0)  # (nz+2)
        scale_above_face = scale_pad[1 : nz + 2, :, :]  # cell m   at faces 0..nz
        scale_below_face = scale_pad[0 : nz + 1, :, :]  # cell m-1 at faces 0..nz
        fqz_lim = jnp.where(
            fqz < 0.0, scale_below_face * fqz, jnp.where(fqz > 0.0, scale_above_face * fqz, fqz)
        )
    elif int(scalar_adv_opt) == 2:
        # ----- monotonic (WRF advect_scalar_mono :10363-10448) -----
        qmax, qmin = _neighbour_min_max(field_old)
        ph_upwind = ph_low  # same low-order monotone update (no IEVA on this path)
        flux_in = -dt * (
            msftx[None, :, :] * msftx[None, :, :]
            * (rdx * (neg(fqx_ip1) - pos(fqx)) + rdy * (neg(fqy_jp1) - pos(fqy)))
            + msftx[None, :, :] * rdzw[:, None, None] * (pos(fqz_kp1) - neg(fqz_k))
        )
        flux_out = dt * (
            msftx[None, :, :] * msftx[None, :, :]
            * (rdx * (pos(fqx_ip1) - neg(fqx)) + rdy * (pos(fqy_jp1) - neg(fqy)))
            + msftx[None, :, :] * rdzw[:, None, None] * (neg(fqz_kp1) - pos(fqz_k))
        )
        # ieva_corr == mass_old on this (no implicit vertical advection) path.
        ph_hi = qmax * mass_old - ph_upwind
        ph_low_m = ph_upwind - qmin * mass_old
        scale_in = jnp.where(
            flux_in > ph_hi, jnp.maximum(0.0, ph_hi / (flux_in + _PD_EPS)), jnp.ones_like(ph_hi)
        )
        scale_out = jnp.where(
            flux_out > ph_low_m, jnp.maximum(0.0, ph_low_m / (flux_out + _PD_EPS)), jnp.ones_like(ph_low_m)
        )
        # WRF combines at each face (:10363-10444): inflow to a cell uses
        # min(scale_in(target), scale_out(source)); outflow the reverse.
        sin_i = scale_in
        sout_i = scale_out
        sin_im1_x = jnp.roll(scale_in, 1, axis=2)
        sout_im1_x = jnp.roll(scale_out, 1, axis=2)
        fqx_lim = jnp.where(
            fqx > 0.0,
            jnp.minimum(sin_i, sout_im1_x) * fqx,
            jnp.minimum(sout_i, sin_im1_x) * fqx,
        )
        sin_jm1_y = jnp.roll(scale_in, 1, axis=1)
        sout_jm1_y = jnp.roll(scale_out, 1, axis=1)
        fqy_lim = jnp.where(
            fqy > 0.0,
            jnp.minimum(sin_i, sout_jm1_y) * fqy,
            jnp.minimum(sout_i, sin_jm1_y) * fqy,
        )
        # z face k: cell above = k, below = k-1.  WRF (:10437-10444): fqz<0 ->
        # min(scale_in(k), scale_out(k-1)); else min(scale_out(k), scale_in(k-1)).
        sin_pad = jnp.concatenate((scale_in[:1, :, :], scale_in, scale_in[-1:, :, :]), axis=0)
        sout_pad = jnp.concatenate((scale_out[:1, :, :], scale_out, scale_out[-1:, :, :]), axis=0)
        sin_above = sin_pad[1 : nz + 2, :, :]
        sout_above = sout_pad[1 : nz + 2, :, :]
        sin_below = sin_pad[0 : nz + 1, :, :]
        sout_below = sout_pad[0 : nz + 1, :, :]
        fqz_lim = jnp.where(
            fqz < 0.0,
            jnp.minimum(sin_above, sout_below) * fqz,
            jnp.minimum(sout_above, sin_below) * fqz,
        )
    else:
        raise ValueError(
            f"advect_scalar_flux_limited only implements scalar_adv_opt 1 (PD) and 2 "
            f"(monotonic); got {scalar_adv_opt}.  Use advect_scalar_flux for the plain path."
        )

    # ---- tendency = -div(limited antidiffusive + low-order), WRF :7780-7885 ----
    fqx_total = fqx_lim + fqxl
    fqy_total = fqy_lim + fqyl
    fqz_total = fqz_lim + fqzl
    fqz_total_kp1 = fqz_total[1:, :, :]
    fqz_total_k = fqz_total[:nz, :, :]
    if _specified:
        # WRF cell bounds under specified: x [ids+1, ide-2], y [jds+1, jde-2];
        # the vertical divergence covers the full tile.
        tend = -msftx[None, :, :] * rdx * _specified_div(fqx_total, axis=2)
        tend = tend - msftx[None, :, :] * rdy * _specified_div(fqy_total, axis=1)
        tend = tend - rdzw[:, None, None] * (fqz_total_kp1 - fqz_total_k)
        return tend
    fqx_total_ip1 = jnp.roll(fqx_total, -1, axis=2)
    fqy_total_jp1 = jnp.roll(fqy_total, -1, axis=1)
    tend = -msftx[None, :, :] * rdx * (fqx_total_ip1 - fqx_total)
    tend = tend - msftx[None, :, :] * rdy * (fqy_total_jp1 - fqy_total)
    tend = tend - rdzw[:, None, None] * (fqz_total_kp1 - fqz_total_k)
    return tend


# --- moisture-species scalar advection loop (advect_scalar / pd / mono) -------
#
# WRF advances EACH moisture species (qv, qc, qr, qi, qs, qg, ...) with the SAME
# flux-form scalar advection it uses for theta, selected by the namelist option
# ``moist_adv_opt`` (the moisture analogue of ``scalar_adv_opt``):
#
#   solve_em.F:2282-2408 -- ``moist_variable_loop: DO im = PARAM_FIRST_SCALAR,
#   num_3d_m`` calls ``rk_scalar_tend(im, im, ..., config_flags%moist_adv_opt,
#   ...)`` per species, which dispatches to ``advect_scalar`` (opt 0),
#   ``advect_scalar_pd`` (opt 1) or ``advect_scalar_mono`` (opt 2) exactly as the
#   scalar (theta) loop does for ``scalar_adv_opt``.  The limiter is applied on
#   the final RK3 stage only (the start-of-step ``moist_old`` / ``mu_1`` feed the
#   FCT bound), every other stage and ``moist_adv_opt == 0`` use the plain h5/v3
#   path -- identical cadence to the theta scalar.
#
# This helper is field-agnostic: it loops over a tuple of (current, old) moisture
# arrays and returns the matching tuple of COUPLED tendencies ``d(mu*q)/dt``, so a
# caller drops it into the moisture loop exactly as it drops ``advect_scalar_flux``
# / ``advect_scalar_flux_limited`` into the theta loop.  Positivity matters most
# for moisture (negative mixing ratios are unphysical), which is why WRF defaults
# real-case moisture to ``moist_adv_opt = 2`` (monotonic) -- but the DEFAULT here
# stays opt-in: ``moist_adv_opt == 0`` runs the byte-for-byte plain path.


def advect_moisture_scalars(
    fields: tuple[jax.Array, ...],
    fields_old: tuple[jax.Array, ...] | None,
    vel: CoupledVelocities,
    *,
    moist_adv_opt: int,
    is_final_rk_stage: bool,
    mut: jax.Array,
    mu_old: jax.Array,
    c1: jax.Array,
    c2: jax.Array,
    rdx: float,
    rdy: float,
    rdzw: jax.Array,
    fzm: jax.Array,
    fzp: jax.Array,
    dt: float,
) -> tuple[jax.Array, ...]:
    """WRF moisture-species flux-form advection loop (h=5/v=3), per species.

    Source: ``solve_em.F:2282-2408`` ``moist_variable_loop`` -> ``rk_scalar_tend``
    with ``config_flags%moist_adv_opt`` (0 plain / 1 PD / 2 monotonic), the
    moisture analogue of the theta ``scalar_adv_opt`` loop.  Same WRF routines
    (``advect_scalar`` / ``advect_scalar_pd`` / ``advect_scalar_mono``).

    ``fields``     -- tuple of CURRENT-stage moisture mixing ratios (each (nz,ny,nx)).
    ``fields_old`` -- tuple of START-OF-STEP moisture mixing ratios (WRF
                      ``moist_old``); required only when the limiter is active
                      (``moist_adv_opt`` in {1,2} on the final RK3 stage).  Must be
                      the same length / shapes as ``fields``.

    Returns a tuple of COUPLED tendencies ``d(mu*q)/dt`` (same length / order as
    ``fields``), so the caller adds each to its coupled moisture prognostic
    exactly as it does the theta tendency.

    Selection (STATIC, compile-time): the limiter runs ONLY when
    ``moist_adv_opt`` in {1,2} AND ``is_final_rk_stage`` is True AND
    ``fields_old`` is provided -- mirroring WRF's final-stage-only FCT and the
    theta wiring.  For ``moist_adv_opt == 0`` (default) or any non-final stage the
    plain ``advect_scalar_flux`` path runs and ``fields_old`` is ignored, so the
    default moisture transport is BYTE-FOR-BYTE the plain h5/v3 path (the limiter
    function is never even traced).

    Mass conservation: each species is limited independently by the same
    flux-renormalization that conserves theta mass (every flux is a face quantity
    differenced as ``flux(i+1)-flux(i)``; scaling a face value scales the same
    number entering both adjacent cells with opposite sign), so the per-species
    coupled-mass integral telescopes to round-off and the limiter only
    redistributes each species' mass, never creates or destroys it.
    """

    use_limiter = (
        int(moist_adv_opt) in (1, 2)
        and bool(is_final_rk_stage)
        and fields_old is not None
    )

    if use_limiter:
        if len(fields_old) != len(fields):  # type: ignore[arg-type]
            raise ValueError(
                "advect_moisture_scalars: fields_old must match fields length "
                f"({len(fields_old)} vs {len(fields)}) when the limiter is active."  # type: ignore[arg-type]
            )
        return tuple(
            advect_scalar_flux_limited(
                field,
                field_old,
                vel,
                scalar_adv_opt=int(moist_adv_opt),
                mut=mut,
                mu_old=mu_old,
                c1=c1,
                c2=c2,
                rdx=rdx,
                rdy=rdy,
                rdzw=rdzw,
                fzm=fzm,
                fzp=fzp,
                dt=dt,
            )
            for field, field_old in zip(fields, fields_old)  # type: ignore[arg-type]
        )

    # Plain h5/v3 path (moist_adv_opt == 0, or a non-final RK stage): byte-for-byte
    # identical to the unlimited scalar advection used before this sprint.
    return tuple(
        advect_scalar_flux(
            field,
            vel,
            mut=mut,
            c1=c1,
            rdx=rdx,
            rdy=rdy,
            rdzw=rdzw,
            fzm=fzm,
            fzp=fzp,
        )
        for field in fields
    )


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
    """WRF flux-form coupled u advection tendency (h=5, v=3), periodic path.

    ``u`` is on x-faces ``(nz, ny, nx+1)`` (periodic: face nx == face 0).  The
    transporting velocities are ``ru/rv/rom`` averaged onto the u faces.
    Returns the COUPLED tendency ``d(mu*u)/dt`` (so callers add it directly to
    the coupled momentum, exactly like ``advect_scalar_flux``).
    """

    nx = vel.ru.shape[-1]
    if bool(getattr(vel, "specified", False)) and vel.ru_full is not None and vel.rv_full is not None:
        # v0.14 SPECIFIED degraded path (advect_u order-5 degrade blocks).
        # Works on the FULL staggered u (nz, ny, nx+1); the generic tier map in
        # the face-index frame reproduces WRF's staggered bounds exactly
        # (x flux faces F[i] between u(i-1), u(i): i==1 -> 2nd-order with the
        # specified UPSTREAM normal-wind rule, i==2 -> flux3, [3, nx-2] ->
        # flux5, i==nx-1 -> flux3, i==nx -> 2nd+upstream; u faces updated
        # [ids+1, ide-1] = [1, nx-1] in EVERY term -- the ring normal faces are
        # owned by the boundary machinery).
        nxs = int(u.shape[-1])  # nx+1 staggered faces
        if vel.msfux is None:
            msfux_f = jnp.ones((int(u.shape[1]), nxs), dtype=u.dtype)
        else:
            msfux_f = jnp.asarray(vel.msfux, dtype=u.dtype)
            if int(msfux_f.shape[-1]) == nxs - 1:
                # periodic-collapsed factor: faces 0..nx-1 exact; the edge-pad
                # face nx is masked out of the update range anyway.
                msfux_f = jnp.concatenate([msfux_f, msfux_f[:, -1:]], axis=-1)
        velx = 0.5 * (vel.ru_full + _shift0(vel.ru_full, 1, 2))
        fqx = specified_flux_faces(u, velx, axis=2, upstream=True)
        tend = -msfux_f[None, :, :] * rdx * _specified_div(fqx, axis=2)
        # y-flux: vel = 0.5*(rv(i)+rv(i-1)) onto the u stagger (zero-padded
        # columns are masked by the final face mask); flux faces along y follow
        # the scalar tier map, divergence rows [jds+1, jde-2].
        rv_pad = jnp.pad(vel.rv_full, ((0, 0), (0, 0), (1, 1)))
        rv_at_u = 0.5 * (rv_pad[:, :, 1:] + rv_pad[:, :, :-1])  # (nz, ny+1, nx+1)
        vely = rv_at_u[:, : int(u.shape[1]), :]
        fqy = specified_flux_faces(u, vely, axis=1)
        tend = tend - msfux_f[None, :, :] * rdy * _specified_div(fqy, axis=1)
        # z-flux (order 3): rom onto u faces with zero-padded columns.
        rom_pad = jnp.pad(vel.rom, ((0, 0), (0, 0), (1, 1)))
        rom_at_u = 0.5 * (rom_pad[:, :, 1:] + rom_pad[:, :, :-1])
        tend = tend + _vertical_flux_div_3(u, rom_at_u, rdzw, fzm, fzp)
        # WRF advect_u under specified updates ONLY faces [ids+1, ide-1].
        face_idx = jnp.arange(nxs).reshape(1, 1, nxs)
        face_mask = (face_idx >= 1) & (face_idx <= nxs - 2)
        return jnp.where(face_mask, tend, 0.0)

    u_f = _collapse_x_face_periodic(u, nx=nx) if u.shape[-1] == nx + 1 else u  # collapse to nx periodic faces
    msfux = _x_face_factor_or_one(vel.msfux, nx=nx, reference=u_f)
    # x-flux: at u-face i the velocity is 0.5*(ru(i)+ru(i-1)); flux5 stencil on u_f.
    velx = _avg_to_u_face_x(vel.ru)
    fqx = velx * flux5_face_periodic(u_f, velx, axis=2)
    tend = -msfux[None, :, :] * rdx * (jnp.roll(fqx, -1, axis=2) - fqx)
    # y-flux: velocity 0.5*(rv(i)+rv(i-1)) averaged in x onto the u stagger.
    if u_f.shape[1] > 1:
        rv_collapsed = vel.rv[:, : u_f.shape[1], :]
        vely = _avg_to_u_face_x(rv_collapsed)
        fqy = vely * flux5_face_periodic(u_f, vely, axis=1)
        tend = tend - msfux[None, :, :] * rdy * (jnp.roll(fqy, -1, axis=1) - fqy)
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
    if bool(getattr(vel, "specified", False)) and vel.ru_full is not None and vel.rv_full is not None:
        # v0.14 SPECIFIED degraded path (advect_v order-5 degrade blocks; the
        # mirror of advect_u with the upstream rule on the NORMAL y faces).
        nys = int(v.shape[-2])  # ny+1 staggered faces

        def _v_factor(factor):
            if factor is None:
                return jnp.ones((nys, int(v.shape[-1])), dtype=v.dtype)
            arr = jnp.asarray(factor, dtype=v.dtype)
            if int(arr.shape[-2]) == nys - 1:
                # periodic-collapsed factor: rows 0..ny-1 exact; the edge-pad
                # row ny is masked out of the update range anyway.
                arr = jnp.concatenate([arr, arr[-1:, :]], axis=-2)
            return arr

        msfvy_f = _v_factor(vel.msfvy)
        msfvx_f = _v_factor(vel.msfvx)
        # x-flux: vel = 0.5*(ru(j)+ru(j-1)) onto the v stagger; scalar tier map
        # along x, divergence cells [ids+1, ide-2].
        ru_pad = jnp.pad(vel.ru_full, ((0, 0), (1, 1), (0, 0)))
        ru_at_v = 0.5 * (ru_pad[:, 1:, :] + ru_pad[:, :-1, :])  # (nz, ny+1, nx+1)
        velx = ru_at_v[:, :, : int(v.shape[-1])]
        fqx = specified_flux_faces(v, velx, axis=2)
        tend = -msfvy_f[None, :, :] * rdx * _specified_div(fqx, axis=2)
        # y-flux: vel = 0.5*(rv(j)+rv(j-1)); upstream rule on the normal faces.
        vely = 0.5 * (vel.rv_full + _shift0(vel.rv_full, 1, 1))
        fqy = specified_flux_faces(v, vely, axis=1, upstream=True)
        tend = tend - msfvy_f[None, :, :] * rdy * _specified_div(fqy, axis=1)
        # z-flux (order 3): rom onto v faces with zero-padded rows.
        rom_pad = jnp.pad(vel.rom, ((0, 0), (1, 1), (0, 0)))
        rom_at_v = 0.5 * (rom_pad[:, 1:, :] + rom_pad[:, :-1, :])
        tend = tend + (msfvy_f / msfvx_f)[None, :, :] * _vertical_flux_div_3(
            v, rom_at_v, rdzw, fzm, fzp
        )
        # WRF advect_v under specified updates ONLY faces [jds+1, jde-1].
        row_idx = jnp.arange(nys).reshape(1, nys, 1)
        row_mask = (row_idx >= 1) & (row_idx <= nys - 2)
        return jnp.where(row_mask, tend, 0.0)

    v_f = v[:, :ny, :] if v.shape[-2] == ny + 1 else v
    msfvy = _y_face_factor_or_one(vel.msfvy, ny=ny, reference=v_f)
    msfvx = _y_face_factor_or_one(vel.msfvx, ny=ny, reference=v_f)
    # x-flux: velocity 0.5*(ru(j)+ru(j-1)) averaged in y onto the v stagger.
    velx = _avg_to_v_face_y(vel.ru[:, : v_f.shape[1], :])
    fqx = velx * flux5_face_periodic(v_f, velx, axis=2)
    tend = -msfvy[None, :, :] * rdx * (jnp.roll(fqx, -1, axis=2) - fqx)
    # y-flux: velocity 0.5*(rv(j)+rv(j-1)).
    vely = _avg_to_v_face_y(vel.rv)
    fqy = vely * flux5_face_periodic(v_f, vely, axis=1)
    tend = tend - msfvy[None, :, :] * rdy * (jnp.roll(fqy, -1, axis=1) - fqy)
    # z-flux (order 3).
    tend = tend + (msfvy / msfvx)[None, :, :] * _vertical_flux_div_3(
        v_f, _avg_to_v_face_y_w(vel.rom), rdzw, fzm, fzp
    )
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

    msftx = _mass_factor_or_one(vel.msftx, w)
    # ru/rv on mass levels -> interpolate to w (full) levels: rw(k)=fzm(k)*r(k)+fzp(k)*r(k-1).
    ru_w = _mass_to_full_levels(vel.ru, fzm, fzp)  # (nz+1, ny, nx)
    rv_w = _mass_to_full_levels(vel.rv, fzm, fzp)
    if bool(getattr(vel, "specified", False)):
        # v0.14 SPECIFIED degraded path: w is mass-located horizontally, so the
        # scalar tier map + cell bounds apply (advect_w degrade blocks mirror
        # advect_scalar).  The face-0 wrap of the collapsed ru/rv is masked.
        fqx = specified_flux_faces(w, ru_w, axis=2)
        tend = -msftx[None, :, :] * rdx * _specified_div(fqx, axis=2)
        fqy = specified_flux_faces(w, rv_w, axis=1)
        tend = tend - msftx[None, :, :] * rdy * _specified_div(fqy, axis=1)
        tend = tend + _vertical_flux_div_w(w, vel.rom, rdn, top_lid=top_lid)
        return tend
    # x-flux of w by ru_w (w and ru_w both on full levels, mass-located in x).
    fqx = ru_w * flux5_face_periodic(w, ru_w, axis=2)
    tend = -msftx[None, :, :] * rdx * (jnp.roll(fqx, -1, axis=2) - fqx)
    if w.shape[1] > 1:
        fqy = rv_w * flux5_face_periodic(w, rv_w, axis=1)
        tend = tend - msftx[None, :, :] * rdy * (jnp.roll(fqy, -1, axis=1) - fqy)
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
    # Sprint U P0-1: result_type(field, fzm/fzp) so fp64 metric weights do not
    # force a silent fp64->fp32 scatter when ``field_mass`` is fp32.
    out_dtype = jnp.result_type(field_mass.dtype, fzm.dtype, fzp.dtype)
    out = jnp.zeros((nz + 1,) + tuple(field_mass.shape[1:]), dtype=out_dtype)
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
    # Sprint U P0-1: allocate vflux at the promoted dtype of the advected field
    # and the (metric-derived) transporting velocity so an fp64 ``romq`` against
    # an fp32 ``field_mass`` is computed in fp64 instead of triggering a silent
    # fp64->fp32 scatter down-cast.  When both are fp64 (the operational
    # force_fp64 path) this is the identity; it never drops precision.
    out_dtype = jnp.result_type(field_mass.dtype, romq.dtype)
    vflux = jnp.zeros((nz + 1,) + tuple(field_mass.shape[1:]), dtype=out_dtype)
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
    # Sprint U P0-1: carry vflux/tend at result_type(w, rom) so an fp64
    # transporting rom against an fp32 w is computed in fp64 (no silent scatter
    # down-cast); identity when both are fp64 (the operational force_fp64 path).
    out_dtype = jnp.result_type(w.dtype, rom.dtype)
    vflux = jnp.zeros((nzp1,) + tuple(w.shape[1:]), dtype=out_dtype)  # vflux at face k
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
    tend = jnp.zeros((nzp1,) + tuple(w.shape[1:]), dtype=out_dtype)
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
    "couple_uv_specified",
    "flux5_face_periodic",
    "specified_flux_faces",
    "advect_scalar_flux",
    "advect_scalar_flux_limited",
    "advect_moisture_scalars",
    "advect_u_flux",
    "advect_v_flux",
    "advect_w_flux",
]
