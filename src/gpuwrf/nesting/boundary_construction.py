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

import os

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.nesting.interp import (
    InterpWeights,
    build_bilinear_weights,
    build_sint_weights,
    interp_bilinear,
    interp_sint_linear,
    slice_weights_cols,
    slice_weights_rows,
)


def _edge_only_enabled() -> bool:
    """Whether the edge-only (ring-only) boundary gather is active.

    Default ON: the edge-only path is a precision-safe restriction of the
    full-grid gather (proven bit-identical on CPU; see
    ``tests/test_v017_edge_only_boundary.py``).  Set
    ``GPUWRF_EDGE_ONLY_BOUNDARY=0`` to fall back to the full-grid->slice
    reference path (e.g. for an A/B bit-compare).
    """

    return os.environ.get("GPUWRF_EDGE_ONLY_BOUNDARY", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
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
# Edge-only (ring-only) gather.
#
# Instead of interpolating the FULL child grid (``_interp``) and then slicing the
# width-N W/E/S/N strips (``field_sides_3d``/``_2d``), gather ONLY the ring cells:
# subset the precomputed parent->child weights to the ring rows/cols and re-run the
# IDENTICAL per-cell bilinear gather (:func:`interp._gather`).  Because that gather
# is independent per output cell, the strip values are bit-for-bit the same as the
# full-grid->slice path -- only the wasted interior interp work is removed.  The
# strip ASSEMBLY (move-to-strip-axis, edge flips, tangential/width padding) is the
# SAME as ``field_sides_3d``/``_2d`` so the produced
# ``(side, bdy_width, z, side_len)`` leaf is identical.
# ---------------------------------------------------------------------------


def _pad_strip_3d(strip: jax.Array, width: int, side_len: int) -> jax.Array:
    """``(bw, z, tan) -> (width, z, side_len)`` -- the ``field_sides_3d`` pad."""

    bw, _zz, tan = strip.shape
    pad_w = int(width) - bw
    pad_t = int(side_len) - tan
    if pad_w > 0 or pad_t > 0:
        strip = jnp.pad(strip, ((0, max(pad_w, 0)), (0, 0), (0, max(pad_t, 0))))
    return strip


def _pad_strip_2d(strip: jax.Array, width: int, side_len: int) -> jax.Array:
    """``(bw, tan) -> (width, 1, side_len)`` -- the ``field_sides_2d`` pad."""

    bw, tan = strip.shape
    pad_w = int(width) - bw
    pad_t = int(side_len) - tan
    if pad_w > 0 or pad_t > 0:
        strip = jnp.pad(strip, ((0, max(pad_w, 0)), (0, max(pad_t, 0))))
    return strip[:, None, :]


def field_sides_3d_edgeonly(
    parent_field, weights: InterpWeights, registration, width, side_len
) -> jax.Array:
    """Edge-only equivalent of ``field_sides_3d(_interp(parent_field, weights))``.

    Gathers ONLY the four width-``width`` ring strips (W/E column strips, S/N row
    strips) directly from the parent via subset weights, then assembles them into
    the SAME ``(side, bdy_width, z, side_len)`` layout ``field_sides_3d`` produces.
    Bit-identical to the full-grid->slice path; computes ~ring/area as much interp.
    """

    cny = int(weights.child_ny)
    cnx = int(weights.child_nx)
    xw = min(int(width), cnx)
    yw = min(int(width), cny)

    # W: child columns 0..xw-1, all rows -> (z, cny, xw) -> moveaxis -> (xw, z, cny)
    w_full = _interp(parent_field, slice_weights_cols(weights, 0, xw), registration)
    w = jnp.moveaxis(w_full, 2, 0)
    # E: child columns cnx-xw..cnx-1, all rows, flipped so index 0 = east edge.
    e_full = _interp(parent_field, slice_weights_cols(weights, cnx - xw, cnx), registration)
    e = jnp.moveaxis(e_full[:, :, ::-1], 2, 0)
    # S: child rows 0..yw-1, all cols -> (z, yw, cnx) -> moveaxis -> (yw, z, cnx)
    s_full = _interp(parent_field, slice_weights_rows(weights, 0, yw), registration)
    s = jnp.moveaxis(s_full, 1, 0)
    # N: child rows cny-yw..cny-1, all cols, flipped so index 0 = north edge.
    n_full = _interp(parent_field, slice_weights_rows(weights, cny - yw, cny), registration)
    n = jnp.moveaxis(n_full[:, ::-1, :], 1, 0)

    return jnp.stack(
        [
            _pad_strip_3d(w, width, side_len),
            _pad_strip_3d(e, width, side_len),
            _pad_strip_3d(s, width, side_len),
            _pad_strip_3d(n, width, side_len),
        ],
        axis=0,
    )


def field_sides_2d_edgeonly(
    parent_field, weights: InterpWeights, registration, width, side_len
) -> jax.Array:
    """Edge-only equivalent of ``field_sides_2d(_interp(parent_field, weights))``.

    2-D analogue of :func:`field_sides_3d_edgeonly`; produces the SAME
    ``(side, bdy_width, 1, side_len)`` layout as ``field_sides_2d``.
    """

    cny = int(weights.child_ny)
    cnx = int(weights.child_nx)
    xw = min(int(width), cnx)
    yw = min(int(width), cny)

    w_full = _interp(parent_field, slice_weights_cols(weights, 0, xw), registration)
    w = jnp.moveaxis(w_full, 1, 0)                          # (xw, cny)
    e_full = _interp(parent_field, slice_weights_cols(weights, cnx - xw, cnx), registration)
    e = jnp.moveaxis(e_full[:, ::-1], 1, 0)                 # (xw, cny) flipped
    s_full = _interp(parent_field, slice_weights_rows(weights, 0, yw), registration)
    s = s_full                                              # (yw, cnx)
    n_full = _interp(parent_field, slice_weights_rows(weights, cny - yw, cny), registration)
    n = n_full[::-1, :]                                     # (yw, cnx) flipped

    return jnp.stack(
        [
            _pad_strip_2d(w, width, side_len),
            _pad_strip_2d(e, width, side_len),
            _pad_strip_2d(s, width, side_len),
            _pad_strip_2d(n, width, side_len),
        ],
        axis=0,
    )


def _child_ring_3d_edgeonly(parent_field, weights, registration, width, side_len) -> jax.Array:
    return field_sides_3d_edgeonly(parent_field, weights, registration, width, side_len)


def _child_ring_2d_edgeonly(parent_field, weights, registration, width, side_len) -> jax.Array:
    return field_sides_2d_edgeonly(parent_field, weights, registration, width, side_len)


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

    # Edge-only (ring-only) gather is DEFAULT-ON: it is a precision-safe restriction
    # of the full-grid gather (bit-identical, proven on CPU) that interpolates only
    # the width-N ring instead of the whole child grid -- killing the ~area/ring
    # wasted interior interp work.  The full-grid->slice path is kept as the
    # reference/fallback (``GPUWRF_EDGE_ONLY_BOUNDARY=0``) and as the A/B target the
    # parent bit-compares on GPU.
    if _edge_only_enabled():
        ring3d = _child_ring_3d_edgeonly
        ring2d = _child_ring_2d_edgeonly
    else:
        ring3d = _child_ring_3d
        ring2d = _child_ring_2d

    theta_new = ring3d(parent_state.theta, weights.mass, reg, w, side_len)
    qv_new = ring3d(parent_state.qv, weights.mass, reg, w, side_len)
    w_new = ring3d(parent_state.w, weights.mass, reg, w, side_len)
    p_new = ring3d(parent_state.p_perturbation, weights.mass, reg, w, side_len)
    ph_new = ring3d(parent_state.ph_perturbation, weights.mass, reg, w, side_len)
    u_new = ring3d(parent_state.u, weights.u, reg, w, side_len)
    v_new = ring3d(parent_state.v, weights.v, reg, w, side_len)
    mu_new = ring2d(parent_state.mu_perturbation, weights.mass, reg, w, side_len)

    child_phb = child_state.ph_total - child_state.ph_perturbation
    child_pb = child_state.p_total - child_state.p_perturbation
    child_mub = child_state.mu_total - child_state.mu_perturbation

    # WRF NEVER laterally forces the nest BASE state: ``inc/nest_forcedown_interp.inc``
    # forces u_2/v_2/w_2/ph_2/t_2/mu_2/moist only, and the nest keeps the
    # ``start_em`` base (PB/MUB/PHB = base formula of the blended terrain) at every
    # cell including the boundary frame.  Packing SINT-interpolated PARENT base
    # here (as pre-v0.15 did) dragged the child's static MUB/PB ring toward
    # ``interp(parent base)``, which differs from ``formula(blended HGT)`` by the
    # base-formula nonlinearity over steep terrain -- the Canary L2 d02 Atlas
    # ``MUB``/``PB`` ~250 Pa nest-frame seam (CPU-WRF truth has NO such seam).
    # The consumer (`apply_lateral_boundaries`) reconstructs totals as
    # ``base + perturbation``, so the base leaves are packed from the CHILD's OWN
    # static base: the spec/relax application becomes an identity and the child
    # base stays exactly the WRF ``start_domain`` base, matching CPU-WRF.
    pb_new = field_sides_3d(child_pb, w, side_len)
    phb_new = field_sides_3d(child_phb, w, side_len)
    mub_new = field_sides_2d(child_mub, w, side_len)

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
    "field_sides_3d_edgeonly",
    "field_sides_2d_edgeonly",
]
