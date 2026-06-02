"""Parent->child spatial interpolation operators (WRF ``interp_fcn.F`` faithful).

This module implements the WRF down-nest horizontal interpolation that fills a
child grid (or its boundary ring) from a recorded/live parent state.  Two
operators are provided, both as **static-index** device gathers (precompute the
parent corner indices + weights ONCE on the host from the static nest geometry;
the per-call op is a fused ``jnp.take`` with no host transfer):

  * :func:`build_bilinear_weights` / :func:`interp_bilinear` -- the plain
    bilinear interpolation the validated v0.1.0 hourly replay used
    (``integration/d02_replay._interp_parent_horizontal``), kept as the
    cross-check baseline.

  * :func:`build_sint_weights` / :func:`interp_sint_linear` -- the WRF-faithful
    **cell-centered** nest registration.  WRF's default EM-core down-nest
    interpolation is ``SINT`` (``share/interp_fcn.F:2356-2358`` -- the
    ``interp_method_type`` default is ``SINT``), realized by ``bdy_interp1`` ->
    ``sintb`` (``share/sint.F``).  ``sint`` is a monotone 4th-order (TR4)
    residual-advection interpolation with a flux limiter; the limiter and the
    high-order residual are *corrections* that vanish for a locally linear field,
    and the load-bearing structural difference versus the replay bilinear is the
    GRID REGISTRATION, not the limiter:

    WRF places the child cell-centers offset by ``-(nri//2)/nri`` of a parent
    cell relative to the node-aligned ``d02_replay`` convention.  Derivation
    (two independent WRF sources, both give the same constant):
      - ``sint.F:54-59`` ``XIG``/``XJG`` sub-cell offsets:
        ``XIG(J) = (rr-1-rioff)/(2*rr) - (J-1)/rr`` for ``J=1..rr`` -- the child
        sub-cell sits at coarse ``-XIG`` of the parent cell center.
      - ``interp_fcn.F:2527-2529`` ``nj=(j-jpos)*nrj+(nrj/2+1)`` -- the coarse
        cell ``j`` maps to the fine CENTER ``nj``; inverting gives child point
        ``ni1`` at continuous coarse ``(ipos-1)+(ni1-1-nri//2)/nri`` (0-based).
    For ratio 3 this is a constant ``-1/3`` parent-cell shift (verified to
    machine precision: a linear parent field reproduces exactly under both the
    XIG offsets and the center-map).  The replay/bilinear convention used
    ``x=(i_parent_start-1)+i/ratio`` (node-aligned), which is OFF by ``-1/3``
    parent cell from WRF.

    For the (odd) 3:1 ratios used in our 9->3->1 km tower, ``rioff=rjoff=0``
    (``sint.F:51-52`` set the staggered offset only for EVEN ratios), so a
    staggered (u/v) field uses the SAME cell-centered offset as the mass field --
    which is why the replay's "reuse the mass coordinate for u/v" was an
    acceptable approximation for staggering but NOT for registration.

We expose the cell-centered registration as :func:`interp_sint_linear` (the
linear / TR4-low-order limit of ``sint`` -- the part that is unconditionally
WRF-correct).  The full monotone TR4 limiter (:func:`sint_block_reference`,
NumPy host reference) is provided for proof-grade fidelity measurement but is NOT
the default device path: the limiter only engages near sharp gradients and the
P0-1a oracle measures how much it matters.  Bumping the device interp to the full
TR4 limiter is a tracked P0-1b refinement (see proofs/p0_1/FINDINGS.md).

No State, dycore, or boundary code is edited here; this is pure interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Static gather weights (precomputed once on the host; device arrays).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterpWeights:
    """Precomputed static parent->child bilinear gather indices + weights.

    All arrays are device arrays so the per-call interpolation is a single fused
    gather with no host transfer.

      y0,y1: (child_ny,) int32 floor/ceil parent ROW index per child row.
      x0,x1: (child_nx,) int32 floor/ceil parent COL index per child col.
      wy:    (child_ny,) f64 weight toward y1 (1-wy toward y0).
      wx:    (child_nx,) f64 weight toward x1 (1-wx toward x0).
    """

    y0: jax.Array
    y1: jax.Array
    x0: jax.Array
    x1: jax.Array
    wy: jax.Array
    wx: jax.Array
    child_ny: int
    child_nx: int

    def tree_flatten(self):
        return (self.y0, self.y1, self.x0, self.x1, self.wy, self.wx), (
            self.child_ny,
            self.child_nx,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        y0, y1, x0, x1, wy, wx = children
        child_ny, child_nx = aux
        return cls(y0, y1, x0, x1, wy, wx, child_ny, child_nx)


jax.tree_util.register_pytree_node_class(InterpWeights)


def _axis_corner_weights(
    coords: np.ndarray, parent_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Floor/ceil parent indices + the toward-ceil weight for one axis.

    Clamps the continuous parent coordinate to ``[0, parent_size-1]``, floors it,
    and forms the linear weight ``frac - floor`` -- the per-axis half of a
    bilinear gather.  Used by both the node-aligned and the cell-centered builds;
    only the input ``coords`` differ (the registration offset).
    """

    clamped = np.clip(np.asarray(coords, dtype=np.float64), 0.0, float(parent_size - 1))
    lower = np.floor(clamped).astype(np.int64)
    upper = np.clip(lower + 1, 0, parent_size - 1)
    weight = clamped - lower.astype(np.float64)
    return lower.astype(np.int32), upper.astype(np.int32), weight


def _axis_coords(
    *,
    parent_start_1based: int,
    child_len: int,
    ratio: int,
    cell_centered: bool,
) -> np.ndarray:
    """Continuous parent (0-based) coordinate of each child point along one axis.

    ``cell_centered=False``: the node-aligned ``d02_replay`` convention
    ``x = (parent_start-1) + i/ratio``.

    ``cell_centered=True``: the WRF ``sint`` cell-centered registration
    ``x = (parent_start-1) + (i - ratio//2)/ratio`` -- child point 0 sits
    ``-(ratio//2)/ratio`` of a parent cell from the node convention
    (``-1/3`` for ratio 3).  Derived from ``sint.F`` ``XIG`` and ``bdy_interp1``
    ``(nrj/2+1)`` -- see module docstring.
    """

    base = float(int(parent_start_1based) - 1)
    idx = np.arange(int(child_len), dtype=np.float64)
    if cell_centered:
        return base + (idx - float(int(ratio) // 2)) / float(ratio)
    return base + idx / float(ratio)


def _build(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_ny: int,
    parent_nx: int,
    child_ny: int,
    child_nx: int,
    cell_centered: bool,
) -> InterpWeights:
    ratio = int(parent_grid_ratio)
    if ratio <= 1:
        raise ValueError(f"nested child requires parent_grid_ratio > 1, got {ratio}")
    y = _axis_coords(
        parent_start_1based=j_parent_start,
        child_len=child_ny,
        ratio=ratio,
        cell_centered=cell_centered,
    )
    x = _axis_coords(
        parent_start_1based=i_parent_start,
        child_len=child_nx,
        ratio=ratio,
        cell_centered=cell_centered,
    )
    y0, y1, wy = _axis_corner_weights(y, int(parent_ny))
    x0, x1, wx = _axis_corner_weights(x, int(parent_nx))
    return InterpWeights(
        y0=jnp.asarray(y0),
        y1=jnp.asarray(y1),
        x0=jnp.asarray(x0),
        x1=jnp.asarray(x1),
        wy=jnp.asarray(wy, dtype=jnp.float64),
        wx=jnp.asarray(wx, dtype=jnp.float64),
        child_ny=int(child_ny),
        child_nx=int(child_nx),
    )


def build_bilinear_weights(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_ny: int,
    parent_nx: int,
    child_ny: int,
    child_nx: int,
) -> InterpWeights:
    """Node-aligned bilinear gather (the validated v0.1.0 replay convention)."""

    return _build(
        parent_grid_ratio=parent_grid_ratio,
        i_parent_start=i_parent_start,
        j_parent_start=j_parent_start,
        parent_ny=parent_ny,
        parent_nx=parent_nx,
        child_ny=child_ny,
        child_nx=child_nx,
        cell_centered=False,
    )


def build_sint_weights(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_ny: int,
    parent_nx: int,
    child_ny: int,
    child_nx: int,
) -> InterpWeights:
    """WRF cell-centered nest registration (the ``sint``/``bdy_interp1`` target).

    Identical gather machinery to the bilinear build but with the WRF
    cell-centered ``-(ratio//2)/ratio`` registration offset on each axis.  This is
    the linear/low-order limit of ``sint`` -- unconditionally WRF-faithful for the
    grid alignment; the monotone TR4 limiter is the (small) residual measured by
    the P0-1a oracle.
    """

    return _build(
        parent_grid_ratio=parent_grid_ratio,
        i_parent_start=i_parent_start,
        j_parent_start=j_parent_start,
        parent_ny=parent_ny,
        parent_nx=parent_nx,
        child_ny=child_ny,
        child_nx=child_nx,
        cell_centered=True,
    )


def _gather(field: jax.Array, weights: InterpWeights) -> jax.Array:
    """Static-index bilinear gather of a parent field onto the child grid.

    ``field`` is ``(z, parent_ny, parent_nx)`` (3-D) or ``(parent_ny, parent_nx)``
    (2-D).  Returns ``(z, child_ny, child_nx)`` / ``(child_ny, child_nx)``.  Shared
    by both registrations -- only the precomputed indices/weights differ.
    """

    dtype = field.dtype
    wy = weights.wy.astype(dtype)
    wx = weights.wx.astype(dtype)
    if field.ndim == 3:
        f_y0 = jnp.take(field, weights.y0, axis=1)
        f_y1 = jnp.take(field, weights.y1, axis=1)
        wy3 = wy[None, :, None]
        f_y = (1.0 - wy3) * f_y0 + wy3 * f_y1
        f_x0 = jnp.take(f_y, weights.x0, axis=2)
        f_x1 = jnp.take(f_y, weights.x1, axis=2)
        wx3 = wx[None, None, :]
        return (1.0 - wx3) * f_x0 + wx3 * f_x1
    if field.ndim == 2:
        f_y0 = jnp.take(field, weights.y0, axis=0)
        f_y1 = jnp.take(field, weights.y1, axis=0)
        wy2 = wy[:, None]
        f_y = (1.0 - wy2) * f_y0 + wy2 * f_y1
        f_x0 = jnp.take(f_y, weights.x0, axis=1)
        f_x1 = jnp.take(f_y, weights.x1, axis=1)
        wx2 = wx[None, :]
        return (1.0 - wx2) * f_x0 + wx2 * f_x1
    raise ValueError(f"interp expects 2D or 3D field, got {field.shape}")


def interp_bilinear(field: jax.Array, weights: InterpWeights) -> jax.Array:
    """Node-aligned bilinear parent->child interpolation (replay baseline)."""

    return _gather(field, weights)


def interp_sint_linear(field: jax.Array, weights: InterpWeights) -> jax.Array:
    """WRF cell-centered (linear/low-order ``sint``) parent->child interpolation.

    Same gather as :func:`interp_bilinear`; the ``weights`` carry the WRF
    cell-centered registration.  This is the WRF-faithful default device operator
    for P0-1a (the monotone TR4 limiter is a tracked P0-1b refinement).
    """

    return _gather(field, weights)


# ---------------------------------------------------------------------------
# Host reference for the FULL WRF monotone TR4 ``sint`` kernel (fidelity proof
# only -- not on the device hot path).  Faithful transcription of share/sint.F.
# ---------------------------------------------------------------------------


def _sint_sub_offsets(ratio: int, *, xstag: bool, ystag: bool):
    """``XIG``/``XJG`` per-sub-cell offsets (``sint.F:46-59``)."""

    rr = int(round(np.sqrt(float(ratio * ratio))))
    rioff = 1.0 if (xstag and rr % 2 == 0) else 0.0
    rjoff = 1.0 if (ystag and rr % 2 == 0) else 0.0
    nf = rr * rr
    xig = np.zeros(nf)
    xjg = np.zeros(nf)
    for i in range(1, rr + 1):
        for j in range(1, rr + 1):
            xig[j + (i - 1) * rr - 1] = (rr - 1.0 - rioff) / (2.0 * rr) - (j - 1) * 1.0 / rr
            xjg[j + (i - 1) * rr - 1] = (rr - 1.0 - rjoff) / (2.0 * rr) - (i - 1) * 1.0 / rr
    return xig, xjg, rr


def sint_block_reference(
    coarse: np.ndarray,
    ratio: int,
    *,
    xstag: bool = False,
    ystag: bool = False,
) -> np.ndarray:
    """Faithful NumPy transcription of WRF ``sintb`` (``share/sint.F``).

    ``coarse`` is a 2-D parent slice ``(ny, nx)`` with at least a 2-cell halo on
    every side.  Returns ``(ny, nx, ratio*ratio)`` sub-cell values: for parent
    cell ``(j, i)`` the ``ratio*ratio`` monotone-TR4 interpolated children, in the
    WRF ``IIM = jp*ratio + (ip+1)`` sub-cell order.  Cells without a full 2-cell
    halo are returned as NaN (WRF computes them from neighbours; the boundary ring
    always has the halo because the nest is interior to the parent).

    This is the proof-grade fidelity reference -- it includes the DONOR flux and
    the OV/UN monotone limiter exactly as WRF.  It is host-only NumPy; the device
    path uses the linear registration (:func:`interp_sint_linear`).
    """

    one12, one24, ep = 1.0 / 12.0, 1.0 / 24.0, 1.0e-10
    xig, xjg, rr = _sint_sub_offsets(int(ratio), xstag=xstag, ystag=ystag)
    nf = rr * rr
    ny, nx = coarse.shape

    def donor(y1, y2, a):
        return (y1 * max(0.0, np.sign(a)) - y2 * min(0.0, np.sign(a))) * a

    def tr4(ym1, y0, yp1, yp2, a):
        return (
            a * one12 * (7.0 * (yp1 + y0) - (yp2 + ym1))
            - a * a * one24 * (15.0 * (yp1 - y0) - (yp2 - ym1))
            - a * a * a * one12 * ((yp1 + y0) - (yp2 + ym1))
            + a * a * a * a * one24 * (3.0 * (yp1 - y0) - (yp2 - ym1))
        )

    def pp(x):
        return max(0.0, x)

    def pn(x):
        return min(0.0, x)

    def limited(vals, a):
        ym2, ym1, y0, yp1, yp2 = vals
        fl0 = donor(ym1, y0, a)
        fl1 = donor(y0, yp1, a)
        w = y0 - (fl1 - fl0)
        mxm = max(ym1, y0, yp1, w)
        mn = min(ym1, y0, yp1, w)
        f0 = tr4(ym2, ym1, y0, yp1, a) - fl0
        f1 = tr4(ym1, y0, yp1, yp2, a) - fl1
        ov = (mxm - w) / (-pn(f1) + pp(f0) + ep)
        un = (w - mn) / (pp(f1) - pn(f0) + ep)
        f0 = pp(f0) * min(1.0, ov) + pn(f0) * min(1.0, un)
        f1 = pp(f1) * min(1.0, un) + pn(f1) * min(1.0, ov)
        return w - (f1 - f0)

    out = np.full((ny, nx, nf), np.nan, dtype=np.float64)
    cf = np.asarray(coarse, dtype=np.float64)
    for iim in range(nf):
        # x-pass: Z[(jj, ii, J)] for J in -2..2 (jrel + 2)
        z = np.full((ny, nx, 5), np.nan, dtype=np.float64)
        for jj in range(2, ny - 2):
            for jrel in range(-2, 3):
                row = jj + jrel
                for ii in range(2, nx - 2):
                    vals = (
                        cf[row, ii - 2],
                        cf[row, ii - 1],
                        cf[row, ii],
                        cf[row, ii + 1],
                        cf[row, ii + 2],
                    )
                    z[jj, ii, jrel + 2] = limited(vals, xig[iim])
        # y-pass
        for jj in range(2, ny - 2):
            for ii in range(2, nx - 2):
                vals = (
                    z[jj, ii, 0],
                    z[jj, ii, 1],
                    z[jj, ii, 2],
                    z[jj, ii, 3],
                    z[jj, ii, 4],
                )
                out[jj, ii, iim] = limited(vals, xjg[iim])
    return out


def sint_to_child_reference(
    coarse: np.ndarray,
    *,
    ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    child_ny: int,
    child_nx: int,
    xstag: bool = False,
    ystag: bool = False,
) -> np.ndarray:
    """Full-grid child field from the WRF monotone ``sint`` (host reference).

    Maps each child mass point ``(nj, ni)`` (0-based) to its parent cell + sub-cell
    via the WRF ``bdy_interp1`` index math (``ci=ipos+(ni1-1)//nri``,
    ``ip=(ni1-1)%nri``, sub-cell ``ip+1+jp*nri``) and gathers from
    :func:`sint_block_reference`.  ``coarse`` is ``(parent_ny, parent_nx)``.
    Returns ``(child_ny, child_nx)``.  Host-only; used by the P0-1a fidelity proof
    to bound the TR4-limiter residual vs the linear registration.
    """

    rr = int(ratio)
    block = sint_block_reference(coarse, rr, xstag=xstag, ystag=ystag)
    out = np.zeros((int(child_ny), int(child_nx)), dtype=np.float64)
    for nj in range(int(child_ny)):
        nj1 = nj + 1  # 1-based fine index
        cj = int(j_parent_start) + (nj1 - 1) // rr - 1  # 0-based parent row
        jp = (nj1 - 1) % rr
        for ni in range(int(child_nx)):
            ni1 = ni + 1
            ci = int(i_parent_start) + (ni1 - 1) // rr - 1  # 0-based parent col
            ip = (ni1 - 1) % rr
            sub = ip + jp * rr  # 0-based sub-cell (XIG row index ip, XJG block jp)
            out[nj, ni] = block[cj, ci, sub]
    return out


__all__ = [
    "InterpWeights",
    "build_bilinear_weights",
    "build_sint_weights",
    "interp_bilinear",
    "interp_sint_linear",
    "sint_block_reference",
    "sint_to_child_reference",
]
