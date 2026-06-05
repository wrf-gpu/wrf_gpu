"""Construct a child specified+relaxation boundary package from a parent state.

This is the WRF ``med_nest_force`` / ``bdy_interp1`` boundary-VALUE construction
(``share/interp_fcn.F:2423-2626``, ``share/mediation_force_domain.F:111-206``)
re-expressed for the GPU port:

  * interpolate each parent prognostic field onto the child grid with the
    WRF-faithful cell-centered registration (:mod:`gpuwrf.nesting.interp`);
  * slice the child boundary RING (the outer ``bdy_width`` rows/cols of each of
    the W/E/S/N sides) in the frozen ``State.*_bdy`` layout
    ``(time, side, bdy_width, z, side_len)``;
  * stack a TWO-TIME leaf ``[old_child_ring, new_parent_target]`` so the existing
    :func:`gpuwrf.coupling.boundary_apply.interpolate_boundary_leaf` linear
    time-interpolation across the child subcycle reproduces WRF's
    ``bdy_* + dtbc*bdy_t*`` cadence EXACTLY.

WRF correspondence (``bdy_interp1``, ``interp_fcn.F:2578-2617``):

    bdy_xs(...)  = nfld(ni,k,nj)                  ! the child's CURRENT value
    bdy_txs(...) = rdt*(psca(...) - nfld(ni,k,nj))  ! rdt = 1/cdt = 1/parent_dt

i.e. the boundary START value is the child's current ring (``nfld``), and the
tendency targets the freshly interpolated parent value (``psca``) over one parent
step ``cdt``.  The two-time leaf ``[nfld_ring, psca_ring]`` with
``cadence_s = parent_dt`` realizes ``nfld + (dtbc/cdt)*(psca - nfld)`` exactly,
with ``dtbc`` advancing ``0 -> parent_dt`` over the subcycle.

The forced prognostic set is WRF ``inc/nest_forcedown_interp.inc`` (u_2, v_2, w_2,
ph_2, t_2, mu_2, QVAPOR) plus the base-state leaves the boundary consumer needs
(phb/pb/mub).  The mass coupling ``(c1*mut + c2)`` that WRF applies before
interpolation (``couple_or_uncouple_em.F:270-345``) is an O(mu'-gradient) term that
vanishes for a uniform-mass column and for a constant field; we interpolate the
DECOUPLED prognostics (the same bounded approximation the validated v0.1.0 hourly
replay used) and keep the mass-coupled IN-LOOP ph/w forcing on the child side via
the existing ``boundary_apply.nested_ph_relax_tendency`` path.  The
constant-field conservation gate (:func:`build_child_boundary_package` round-trip
in the proof) asserts this term is bounded.

This module does NOT edit ``boundary_apply.py``, ``lateral_bc.py``, the dycore, or
any physics; it only READS the State interface and writes a new ``*_bdy`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.nesting.interp import (
    InterpWeights,
    build_bilinear_weights,
    build_sint_weights,
    interp_bilinear,
    interp_sint_linear,
)


# WRF ``share/module_bc.F`` side order is W/E/S/N; the State leaf side axis uses
# the same order (``boundary_apply.SIDES``).
_SIDES = ("W", "E", "S", "N")


@dataclass(frozen=True)
class NestForceWeights:
    """Per-staggering parent->child interp weights for one edge (all static).

    The C-grid stagger only changes the parent extent each gather clamps to;
    for the odd (3:1) ratios of our tower the WRF ``sint`` staggered offset
    ``rioff/rjoff`` is ZERO (``sint.F:51-52`` -- staggered offset is set only for
    EVEN ratios), so u/v reuse the mass cell-centered registration and only widen
    the parent extent by one along the staggered axis.
    """

    mass: InterpWeights          # parent (ny, nx)   -> child (ny, nx)
    u: InterpWeights             # parent (ny, nx+1) -> child (ny, nx+1)  C-grid u
    v: InterpWeights             # parent (ny+1, nx) -> child (ny+1, nx)  C-grid v
    registration: str            # "sint" (WRF cell-centered) | "bilinear" (replay)

    def tree_flatten(self):
        return (self.mass, self.u, self.v), self.registration

    @classmethod
    def tree_unflatten(cls, aux, children):
        mass, u, v = children
        return cls(mass, u, v, aux)


jax.tree_util.register_pytree_node_class(NestForceWeights)


def build_nest_force_weights(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_grid: GridSpec,
    child_grid: GridSpec,
    registration: str = "sint",
) -> NestForceWeights:
    """Precompute mass/u/v parent->child gathers for one edge.

    ``registration="sint"`` (default) uses the WRF cell-centered nest
    registration; ``registration="bilinear"`` uses the node-aligned v0.1.0 replay
    convention (kept for the proof cross-check).
    """

    if registration not in ("sint", "bilinear"):
        raise ValueError(f"unknown nest interpolation registration {registration!r}")
    pny, pnx = int(parent_grid.ny), int(parent_grid.nx)
    cny, cnx = int(child_grid.ny), int(child_grid.nx)
    builder = build_sint_weights if registration == "sint" else build_bilinear_weights
    common = dict(
        parent_grid_ratio=parent_grid_ratio,
        i_parent_start=i_parent_start,
        j_parent_start=j_parent_start,
    )
    mass = builder(parent_ny=pny, parent_nx=pnx, child_ny=cny, child_nx=cnx, **common)
    u = builder(parent_ny=pny, parent_nx=pnx + 1, child_ny=cny, child_nx=cnx + 1, **common)
    v = builder(parent_ny=pny + 1, parent_nx=pnx, child_ny=cny + 1, child_nx=cnx, **common)
    return NestForceWeights(mass=mass, u=u, v=v, registration=str(registration))


def _interp(field: jax.Array, weights: InterpWeights, registration: str) -> jax.Array:
    if registration == "sint":
        return interp_sint_linear(field, weights)
    return interp_bilinear(field, weights)


# ---------------------------------------------------------------------------
# Child ring side strips (device).  Layout: (side, bdy_width, z, side_len).
# bdy_width axis runs outer (index 0 = domain edge) -> inner, matching the
# boundary_apply._strip consumer.
# ---------------------------------------------------------------------------


def field_sides_3d(field: jax.Array, width: int, side_len: int) -> jax.Array:
    """WRF-ordered ring strips ``(side, bdy_width, z, side_len)`` for a 3-D field.

    ``field`` is ``(z, ny, nx)``.  W = first ``width`` columns; E = last ``width``
    columns flipped so index 0 is the east edge; S = first ``width`` rows; N = last
    ``width`` rows flipped so index 0 is the north edge.  Each side is padded along
    the tangential axis to ``side_len``.  This matches the State leaf the
    ``boundary_apply`` consumer expects.
    """

    z, ny, nx = field.shape
    xw = min(int(width), nx)
    yw = min(int(width), ny)
    w = jnp.moveaxis(field[:, :, :xw], 2, 0)                       # (xw, z, ny)
    e = jnp.moveaxis(field[:, :, nx - xw:][:, :, ::-1], 2, 0)      # (xw, z, ny)
    s = jnp.moveaxis(field[:, :yw, :], 1, 0)                       # (yw, z, nx)
    n = jnp.moveaxis(field[:, ny - yw:, :][:, ::-1, :], 1, 0)      # (yw, z, nx)

    def _pad(strip: jax.Array) -> jax.Array:  # (bw, z, tan) -> (width, z, side_len)
        bw, zz, tan = strip.shape
        pad_w = int(width) - bw
        pad_t = int(side_len) - tan
        if pad_w > 0 or pad_t > 0:
            strip = jnp.pad(strip, ((0, max(pad_w, 0)), (0, 0), (0, max(pad_t, 0))))
        return strip

    return jnp.stack([_pad(w), _pad(e), _pad(s), _pad(n)], axis=0)


def field_sides_2d(field: jax.Array, width: int, side_len: int) -> jax.Array:
    """WRF-ordered ring strips ``(side, bdy_width, 1, side_len)`` for a 2-D field."""

    ny, nx = field.shape
    xw = min(int(width), nx)
    yw = min(int(width), ny)
    w = jnp.moveaxis(field[:, :xw], 1, 0)                          # (xw, ny)
    e = jnp.moveaxis(field[:, nx - xw:][:, ::-1], 1, 0)            # (xw, ny)
    s = field[:yw, :]                                              # (yw, nx)
    n = field[ny - yw:, :][::-1, :]                                # (yw, nx)

    def _pad(strip: jax.Array) -> jax.Array:  # (bw, tan) -> (width, 1, side_len)
        bw, tan = strip.shape
        pad_w = int(width) - bw
        pad_t = int(side_len) - tan
        if pad_w > 0 or pad_t > 0:
            strip = jnp.pad(strip, ((0, max(pad_w, 0)), (0, max(pad_t, 0))))
        return strip[:, None, :]

    return jnp.stack([_pad(w), _pad(e), _pad(s), _pad(n)], axis=0)


def _child_ring_3d(parent_field, weights, registration, width, side_len) -> jax.Array:
    child = _interp(parent_field, weights, registration)
    return field_sides_3d(child, width, side_len)


def _child_ring_2d(parent_field, weights, registration, width, side_len) -> jax.Array:
    child = _interp(parent_field, weights, registration)
    return field_sides_2d(child, width, side_len)


# ---------------------------------------------------------------------------
# Boundary-VALUE construction: build the full child *_bdy package from a parent.
# ---------------------------------------------------------------------------


def _fit(strips: jax.Array, z_target: int, side_target: int, dtype) -> jax.Array:
    """Trim/pad ``(side, bdy_width, z, side_len)`` strips to the leaf extents."""

    out = strips.astype(dtype)
    if out.shape[-2] != z_target or out.shape[-1] != side_target:
        out = out[..., :z_target, :side_target]
        pad_z = max(z_target - out.shape[-2], 0)
        pad_s = max(side_target - out.shape[-1], 0)
        if pad_z > 0 or pad_s > 0:
            out = jnp.pad(out, ((0, 0), (0, 0), (0, pad_z), (0, pad_s)))
    return out


def build_child_boundary_package(
    child_state: State,
    parent_state: State,
    weights: NestForceWeights,
    *,
    bdy_width: int = 5,
) -> State:
    """Construct the child specified+relaxation ``*_bdy`` package from a parent.

    WRF ``med_nest_force`` -> ``bdy_interp1`` for the forced prognostic set
    (u, v, w, ph, theta, qv, mu) plus the base-state leaves (phb, pb, mub) the
    GPU boundary consumer needs.  Each leaf is the two-time
    ``[old_child_ring, new_parent_target]`` package: the START ring is the child's
    CURRENT field (WRF ``bdy_* = nfld``) and the NEW ring is the parent-interpolated
    target (WRF ``bdy_* + cdt*bdy_t* = psca``).  The caller drives the child for
    ``parent_grid_ratio`` substeps with ``update_cadence_s = parent_dt``.

    The child prognostic INTERIOR is untouched -- only the boundary package is
    filled.  Returns a new ``child_state``.
    """

    if int(bdy_width) <= 0:
        raise ValueError(f"bdy_width must be positive, got {bdy_width}")
    reg = weights.registration
    side_len = int(max(child_state.u_bdy.shape[-1], child_state.v_bdy.shape[-1]))
    w = int(bdy_width)

    def two_time(old_leaf_full, child_field, new_strips, *, is_2d=False):
        ref = old_leaf_full[-1]
        zt, st = int(ref.shape[-2]), int(ref.shape[-1])
        if is_2d:
            old = _fit(field_sides_2d(child_field, w, side_len), zt, st, ref.dtype)
        else:
            old = _fit(field_sides_3d(child_field, w, side_len), zt, st, ref.dtype)
        new = _fit(new_strips, zt, st, ref.dtype)
        return jnp.stack([old, new], axis=0)

    pb_parent = parent_state.p_total - parent_state.p_perturbation
    phb_parent = parent_state.ph_total - parent_state.ph_perturbation
    mub_parent = parent_state.mu_total - parent_state.mu_perturbation

    theta_new = _child_ring_3d(parent_state.theta, weights.mass, reg, w, side_len)
    qv_new = _child_ring_3d(parent_state.qv, weights.mass, reg, w, side_len)
    w_new = _child_ring_3d(parent_state.w, weights.mass, reg, w, side_len)
    p_new = _child_ring_3d(parent_state.p_perturbation, weights.mass, reg, w, side_len)
    pb_new = _child_ring_3d(pb_parent, weights.mass, reg, w, side_len)
    ph_new = _child_ring_3d(parent_state.ph_perturbation, weights.mass, reg, w, side_len)
    phb_new = _child_ring_3d(phb_parent, weights.mass, reg, w, side_len)
    u_new = _child_ring_3d(parent_state.u, weights.u, reg, w, side_len)
    v_new = _child_ring_3d(parent_state.v, weights.v, reg, w, side_len)
    mu_new = _child_ring_2d(parent_state.mu_perturbation, weights.mass, reg, w, side_len)
    mub_new = _child_ring_2d(mub_parent, weights.mass, reg, w, side_len)

    child_phb = child_state.ph_total - child_state.ph_perturbation
    child_pb = child_state.p_total - child_state.p_perturbation
    child_mub = child_state.mu_total - child_state.mu_perturbation

    return child_state.replace(
        u_bdy=two_time(child_state.u_bdy, child_state.u, u_new),
        v_bdy=two_time(child_state.v_bdy, child_state.v, v_new),
        w_bdy=two_time(child_state.w_bdy, child_state.w, w_new),
        theta_bdy=two_time(child_state.theta_bdy, child_state.theta, theta_new),
        qv_bdy=two_time(child_state.qv_bdy, child_state.qv, qv_new),
        ph_bdy=two_time(child_state.ph_bdy, child_state.ph_perturbation, ph_new),
        phb_bdy=two_time(child_state.phb_bdy, child_phb, phb_new),
        p_bdy=two_time(child_state.p_bdy, child_state.p_perturbation, p_new),
        pb_bdy=two_time(child_state.pb_bdy, child_pb, pb_new),
        mu_bdy=two_time(child_state.mu_bdy, child_state.mu_perturbation, mu_new, is_2d=True),
        mub_bdy=two_time(child_state.mub_bdy, child_mub, mub_new, is_2d=True),
    )


def interp_parent_field_to_child(
    field: jax.Array,
    weights: NestForceWeights,
    *,
    staggering: str = "mass",
) -> jax.Array:
    """Interpolate one parent field to the FULL child grid (proof/diagnostic).

    ``staggering`` selects which precomputed gather to use ("mass", "u", "v").
    Returns the full child-grid field (NOT sliced to the ring) so the P0-1a oracle
    can compare interior + boundary against the recorded child wrfout.
    """

    wsel = {"mass": weights.mass, "u": weights.u, "v": weights.v}[staggering]
    return _interp(field, wsel, weights.registration)


__all__ = [
    "NestForceWeights",
    "build_nest_force_weights",
    "build_child_boundary_package",
    "interp_parent_field_to_child",
    "field_sides_3d",
    "field_sides_2d",
]
