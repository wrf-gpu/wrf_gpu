"""WPS/metgrid horizontal-interpolation kernels, ported to vectorized JAX.

This is a **faithful** port of WRF/WPS
``WPS/metgrid/src/interp_module.F`` (functions ``interp_sequence``,
``four_pt`` (bilinear), ``sixteen_pt`` (overlapping-parabolic / bicubic-ish via
the 1-D ``oned``), ``four_pt_average``, ``sixteen_pt_average``,
``wt_four_pt_average``, ``wt_sixteen_pt_average``, ``nearest_neighbor``,
``search_extrap``) plus the ``+``-chained dispatcher and the per-field mask
policy from ``METGRID.TBL.ARW`` (parsed by ``interp_option_module.F`` and applied
in ``process_domain_module.F::interp_met_field``).

PORTING MODEL
-------------
The Fortran code is scalar: for each target (mass/U/V) grid point it computes a
fractional **source-grid** index ``(rx, ry)`` (via ``lltoxy`` of the target
lat/lon onto the source projection) and calls ``interp_sequence`` to evaluate the
source slab at ``(rx, ry)``. Here every kernel is vectorized over a flat array of
target points ``(rx[k], ry[k])`` evaluating against the same source slab; the
``+``-chain is realized by computing each method's value + a per-point "valid"
mask and selecting the first valid method in chain order (exactly the Fortran
fall-through, where an inapplicable method returns ``msgval`` and recurses to the
next).

INDEX CONVENTION (the classic interp bug surface -- pinned here)
----------------------------------------------------------------
Fortran ``interp_module.F`` uses **1-based** ``(x, y)`` where ``x`` is the first
(``start_x:end_x``) array dimension and ``y`` the second. The source slab in WPS
is indexed ``slab(i_lon, j_lat)`` -- first dim = longitude (i), second = latitude
(j). We keep that exact convention: kernels take ``slab`` shaped
``(nx_src, ny_src)`` (== ``(i, j)``) and a 1-based fractional ``(rx, ry)`` where
``rx`` indexes the **first** (longitude) axis and ``ry`` the **second**
(latitude) axis. ``rx in [1, nx_src]``. We convert to 0-based only at the gather.
``msgval`` is the source missing value (default ``-1.e30``); a per-point result
equal to ``msgval`` means "this method did not produce a value here".

All math is fp64 (the analytic oracle gate is ``<= 1e-6`` rel).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# x64 is required for the analytic-oracle tolerance; enable at import like the
# rest of the package (mirrors gpuwrf state layout policy).
jax.config.update("jax_enable_x64", True)


# --- METGRID.TBL interp-method enum (mirrors misc_definitions_module) --------
# Values are arbitrary tags; only identity/ordering matters for the dispatcher.
N_NEIGHBOR = 1
FOUR_POINT = 2
SIXTEEN_POINT = 3
AVERAGE4 = 4
AVERAGE16 = 5
W_AVERAGE4 = 6
W_AVERAGE16 = 7
SEARCH = 8

_METHOD_NAMES = {
    "nearest_neighbor": N_NEIGHBOR,
    "four_pt": FOUR_POINT,
    "sixteen_pt": SIXTEEN_POINT,
    "average_4pt": AVERAGE4,
    "average_16pt": AVERAGE16,
    "wt_average_4pt": W_AVERAGE4,
    "wt_average_16pt": W_AVERAGE16,
}

DEFAULT_MSGVAL = -1.0e30
DEFAULT_SEARCH_DEPTH = 1200


def parse_interp_string(interp_string: str) -> list[tuple[int, int]]:
    """Parses a METGRID.TBL ``interp_option`` like
    ``sixteen_pt+four_pt+average_4pt`` into an ordered list of
    ``(method_id, option)`` pairs, faithfully reproducing
    ``interp_module.F::interp_array_from_string`` /
    ``interp_options_from_string`` (the only non-zero option is the ``search``
    max-depth, parsed from ``search(NNNN)``; default 1200).

    Unknown tokens (e.g. ``average_gcell``) are skipped, matching the Fortran
    WARN-and-continue behavior. ``derived``/``static`` strings yield an empty
    list (S3 builds those without horizontal interp).
    """

    out: list[tuple[int, int]] = []
    for raw in interp_string.split("+"):
        tok = raw.strip()
        if tok in _METHOD_NAMES:
            out.append((_METHOD_NAMES[tok], 0))
        elif tok.startswith("search"):
            depth = DEFAULT_SEARCH_DEPTH
            if "(" in tok and ")" in tok:
                inside = tok[tok.index("(") + 1 : tok.index(")")]
                try:
                    depth = int(inside)
                except ValueError:
                    depth = DEFAULT_SEARCH_DEPTH
            out.append((SEARCH, depth))
        # else: unrecognized (average_gcell / derived / static) -> skip
    return out


# =============================================================================
# Mask predicate
# =============================================================================
# In interp_module.F a source cell is "masked off" (excluded) per the
# (mask_relational, maskval) pair:
#   '>'  -> excluded where mask_array >  maskval
#   '<'  -> excluded where mask_array <  maskval
#   ' '  -> excluded where mask_array == maskval   (equality; the AIFS soil case
#                                                    interp_mask=LANDSEA(0))
# We pre-compute a boolean source-cell "usable" array once per call.
def _source_usable(
    slab: jnp.ndarray,
    msgval: float,
    mask_array: jnp.ndarray | None,
    maskval: float | None,
    mask_relational: str | None,
) -> jnp.ndarray:
    """Boolean (nx,ny): True where the source cell is a usable donor."""

    usable = slab != msgval
    if mask_array is not None and maskval is not None:
        if mask_relational == ">":
            usable = usable & ~(mask_array > maskval)
        elif mask_relational == "<":
            usable = usable & ~(mask_array < maskval)
        else:  # ' ' equality (or None relational defaults to equality exclusion)
            usable = usable & (mask_array != maskval)
    return usable


def _gather(slab: jnp.ndarray, ix: jnp.ndarray, iy: jnp.ndarray) -> jnp.ndarray:
    """slab[ix, iy] with 1-based ix/iy clamped into range (caller guarantees
    in-range for the value; clamp only protects the gather)."""

    nx, ny = slab.shape
    ix0 = jnp.clip(ix - 1, 0, nx - 1)
    iy0 = jnp.clip(iy - 1, 0, ny - 1)
    return slab[ix0, iy0]


def _gather_mask(
    usable: jnp.ndarray, ix: jnp.ndarray, iy: jnp.ndarray
) -> jnp.ndarray:
    nx, ny = usable.shape
    ix0 = jnp.clip(ix - 1, 0, nx - 1)
    iy0 = jnp.clip(iy - 1, 0, ny - 1)
    return usable[ix0, iy0]


# =============================================================================
# oned : 1-D overlapping parabolic interpolation (interp_module.F::oned)
# =============================================================================
def oned(x: jnp.ndarray, a, b, c, d) -> jnp.ndarray:
    """Vectorized faithful port of ``oned(x,a,b,c,d)``.

    Reproduces the exact branch structure, including the zero-handling: a, b, c,
    d are treated as "missing" when == 0 (the ``sixteen_pt`` caller substitutes
    1e-20 for true zeros so genuine zeros are not seen as missing here).
    """

    x = jnp.asarray(x, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)

    # Base: 0., overwritten by x==0 -> b, x==1 -> c (the leading block).
    res = jnp.zeros_like(x)
    res = jnp.where(x == 0.0, b, res)
    res = jnp.where(x == 1.0, c, res)

    bc_nonzero = (b * c) != 0.0
    ad_zero = (a * d) == 0.0
    a_zero = a == 0.0
    d_zero = d == 0.0

    # branch: b*c != 0 and a*d == 0
    both_ad_zero = a_zero & d_zero
    val_lin = b * (1.0 - x) + c * x
    val_a = b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b))
    val_d = c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c))
    branch_adzero = jnp.where(
        both_ad_zero,
        val_lin,
        jnp.where(~a_zero, val_a, jnp.where(~d_zero, val_d, res)),
    )

    # branch: b*c != 0 and a*d != 0 (full overlapping parabola)
    val_full = (1.0 - x) * (
        b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b))
    ) + x * (c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c)))

    res = jnp.where(
        bc_nonzero,
        jnp.where(ad_zero, branch_adzero, val_full),
        res,
    )
    return res


# =============================================================================
# nearest_neighbor (interp_module.F::nearest_neighbor)
# =============================================================================
def nearest_neighbor(rx, ry, slab, msgval, usable):
    """Returns (value, produced) for each (rx,ry). ``produced`` is True where this
    method yields a non-msgval value. nint() round, edge-out -> not produced."""

    nx, ny = slab.shape
    ix = jnp.round(rx).astype(jnp.int32)
    iy = jnp.round(ry).astype(jnp.int32)
    in_range = (ix >= 1) & (ix <= nx) & (iy >= 1) & (iy <= ny)
    val = _gather(slab, ix, iy)
    cell_usable = _gather_mask(usable, ix, iy)
    produced = in_range & cell_usable
    val = jnp.where(produced, val, msgval)
    return val, produced


# =============================================================================
# four_pt : bilinear (interp_module.F::four_pt)
# =============================================================================
def four_pt(rx, ry, slab, msgval, usable):
    nx, ny = slab.shape
    min_x = jnp.floor(rx).astype(jnp.int32)
    max_x = jnp.ceil(rx).astype(jnp.int32)
    min_y = jnp.floor(ry).astype(jnp.int32)
    max_y = jnp.ceil(ry).astype(jnp.int32)

    in_range = (min_x >= 1) & (max_x <= nx) & (min_y >= 1) & (max_y <= ny)

    u00 = _gather_mask(usable, min_x, min_y)
    u10 = _gather_mask(usable, max_x, min_y)
    u01 = _gather_mask(usable, min_x, max_y)
    u11 = _gather_mask(usable, max_x, max_y)
    all_usable = u00 & u10 & u01 & u11

    a00 = _gather(slab, min_x, min_y)
    a10 = _gather(slab, max_x, min_y)
    a01 = _gather(slab, min_x, max_y)
    a11 = _gather(slab, max_x, max_y)

    fx = rx
    fy = ry
    mnx = min_x.astype(jnp.float64)
    mxx = max_x.astype(jnp.float64)
    mny = min_y.astype(jnp.float64)
    mxy = max_y.astype(jnp.float64)

    same_x = min_x == max_x
    same_y = min_y == max_y

    # general bilinear (matches the Fortran final else)
    val_general = (fy - mny) * (a01 * (mxx - fx) + a11 * (fx - mnx)) + (
        mxy - fy
    ) * (a00 * (mxx - fx) + a10 * (fx - mnx))
    # min_x==max_x, min_y!=max_y
    val_samex = a00 * (mxy - fy) + a01 * (fy - mny)
    # min_y==max_y, min_x!=max_x
    val_samey = a00 * (mxx - fx) + a10 * (fx - mnx)
    # both same
    val_point = a00

    val = jnp.where(
        same_x & same_y,
        val_point,
        jnp.where(same_x, val_samex, jnp.where(same_y, val_samey, val_general)),
    )

    produced = in_range & all_usable
    val = jnp.where(produced, val, msgval)
    return val, produced


# =============================================================================
# sixteen_pt : overlapping parabolic (interp_module.F::sixteen_pt)
# =============================================================================
def sixteen_pt(rx, ry, slab, msgval, usable):
    """Faithful port. Clamps the 4x4 stencil to the array edges (kk/ll clamp),
    builds stl, replaces genuine zeros by 1e-20 when msgval!=0, applies the
    overlapping-parabolic oned-along-x then oned-along-y. If any of the 16
    stencil cells is unusable (msgval or masked), the point is not produced
    (recurse to next method). The ``n/=16`` averaging branch in Fortran is dead
    (n is always 16), so it is intentionally omitted.

    Also handles the exact-grid-point branch (abs(x)<=1e-4 and abs(y)<=1e-4):
    returns the cell value if usable, else not produced.
    """

    nx, ny = slab.shape

    # int(xx) out of [start,end] -> recurse (not produced). int() truncates
    # toward zero; rx>=1 so int==floor here.
    intx = jnp.floor(rx).astype(jnp.int32)
    inty = jnp.floor(ry).astype(jnp.int32)
    base_in = (intx >= 1) & (intx <= nx) & (inty >= 1) & (inty <= ny)

    i = jnp.floor(rx + 0.00001).astype(jnp.int32)
    j = jnp.floor(ry + 0.00001).astype(jnp.int32)
    x = rx - i.astype(jnp.float64)
    y = ry - j.astype(jnp.float64)

    on_grid = (jnp.abs(x) <= 0.0001) & (jnp.abs(y) <= 0.0001)

    # ---- general (off-grid) 4x4 overlapping-parabola path ----
    # stl[k-1][l-1] for k,l in 1..4 ; kk = i+k-2 clamped, ll = j+l-2 clamped
    stl = [[None] * 4 for _ in range(4)]
    any_unusable = jnp.zeros_like(rx, dtype=bool)
    for k in range(1, 5):
        kk = jnp.clip(i + k - 2, 1, nx)
        for l in range(1, 5):
            ll = jnp.clip(j + l - 2, 1, ny)
            v = _gather(slab, kk, ll)
            cu = _gather_mask(usable, kk, ll)
            any_unusable = any_unusable | (~cu)
            # Fortran: if stl==0 and msgval/=0 -> stl=1e-20 (protect true zeros)
            if msgval != 0.0:
                v = jnp.where(v == 0.0, 1.0e-20, v)
            stl[k - 1][l - 1] = v

    # oned along x: a = oned(x, stl(1,1),stl(2,1),stl(3,1),stl(4,1)) (vary k, l=1)
    a = oned(x, stl[0][0], stl[1][0], stl[2][0], stl[3][0])
    b = oned(x, stl[0][1], stl[1][1], stl[2][1], stl[3][1])
    c = oned(x, stl[0][2], stl[1][2], stl[2][2], stl[3][2])
    d = oned(x, stl[0][3], stl[1][3], stl[2][3], stl[3][3])
    val_general = oned(y, a, b, c, d)
    val_general = jnp.where(val_general == 1.0e-20, 0.0, val_general)

    # ---- on-grid path: return cell (i,j) if usable ----
    on_in = (i >= 1) & (i <= nx) & (j >= 1) & (j <= ny)
    cell_v = _gather(slab, i, j)
    cell_u = _gather_mask(usable, i, j)
    on_grid_produced = on_in & cell_u
    val_on = jnp.where(on_grid_produced, cell_v, msgval)

    # combine
    produced = jnp.where(
        ~base_in,
        False,
        jnp.where(on_grid, on_grid_produced, ~any_unusable),
    )
    val = jnp.where(on_grid, val_on, val_general)
    val = jnp.where(produced, val, msgval)
    return val, produced


# =============================================================================
# four_pt_average (interp_module.F::four_pt_average)
# =============================================================================
def four_pt_average(rx, ry, slab, msgval, usable):
    nx, ny = slab.shape
    ifx = jnp.floor(rx).astype(jnp.int32)
    icx = jnp.ceil(rx).astype(jnp.int32)
    ify = jnp.floor(ry).astype(jnp.int32)
    icy = jnp.ceil(ry).astype(jnp.int32)

    # half-grid-point-out correction
    rxf = rx
    ryf = ry
    out = (ifx < 1) | (icx > nx) | (ify < 1) | (icy > ny)
    fix_lo_x = out & (rxf > 1.0 - 0.5) & (ifx < 1)
    fix_hi_x = out & (rxf < float(nx) + 0.5) & (icx > nx)
    ifx = jnp.where(fix_lo_x, 1, jnp.where(fix_hi_x, nx, ifx))
    icx = jnp.where(fix_lo_x, 1, jnp.where(fix_hi_x, nx, icx))
    fix_lo_y = out & (ryf > 1.0 - 0.5) & (ify < 1)
    fix_hi_y = out & (ryf < float(ny) + 0.5) & (icy > ny)
    ify = jnp.where(fix_lo_y, 1, jnp.where(fix_hi_y, ny, ify))
    icy = jnp.where(fix_lo_y, 1, jnp.where(fix_hi_y, ny, icy))

    still_out = (ifx < 1) | (icx > nx) | (ify < 1) | (icy > ny)

    u00 = _gather_mask(usable, ifx, ify)
    u01 = _gather_mask(usable, ifx, icy)
    u10 = _gather_mask(usable, icx, ify)
    u11 = _gather_mask(usable, icx, icy)
    a00 = _gather(slab, ifx, ify)
    a01 = _gather(slab, ifx, icy)
    a10 = _gather(slab, icx, ify)
    a11 = _gather(slab, icx, icy)

    w00 = jnp.where(u00, 1.0, 0.0)
    w01 = jnp.where(u01, 1.0, 0.0)
    w10 = jnp.where(u10, 1.0, 0.0)
    w11 = jnp.where(u11, 1.0, 0.0)
    wsum = w00 + w01 + w10 + w11
    val = (w00 * a00 + w01 * a01 + w10 * a10 + w11 * a11) / jnp.where(
        wsum > 0, wsum, 1.0
    )
    produced = (~still_out) & (wsum > 0.0)
    val = jnp.where(produced, val, msgval)
    return val, produced


# =============================================================================
# wt_four_pt_average (interp_module.F::wt_four_pt_average)
# =============================================================================
def wt_four_pt_average(rx, ry, slab, msgval, usable):
    nx, ny = slab.shape
    ifx = jnp.floor(rx).astype(jnp.int32)
    icx = jnp.ceil(rx).astype(jnp.int32)
    ify = jnp.floor(ry).astype(jnp.int32)
    icy = jnp.ceil(ry).astype(jnp.int32)

    rxf = rx
    ryf = ry

    # distance weights computed on the ORIGINAL if/ic (Fortran computes them
    # before the boundary correction)
    def dist_w(ax, ay):
        return jnp.maximum(
            0.0,
            1.0
            - jnp.sqrt(
                (rxf - ax.astype(jnp.float64)) ** 2
                + (ryf - ay.astype(jnp.float64)) ** 2
            ),
        )

    w00 = dist_w(ifx, ify)
    w01 = dist_w(ifx, icy)
    w10 = dist_w(icx, ify)
    w11 = dist_w(icx, icy)

    out = (ifx < 1) | (icx > nx) | (ify < 1) | (icy > ny)
    fix_lo_x = out & (rxf > 1.0 - 0.5) & (ifx < 1)
    fix_hi_x = out & (rxf < float(nx) + 0.5) & (icx > nx)
    ifx = jnp.where(fix_lo_x, 1, jnp.where(fix_hi_x, nx, ifx))
    icx = jnp.where(fix_lo_x, 1, jnp.where(fix_hi_x, nx, icx))
    fix_lo_y = out & (ryf > 1.0 - 0.5) & (ify < 1)
    fix_hi_y = out & (ryf < float(ny) + 0.5) & (icy > ny)
    ify = jnp.where(fix_lo_y, 1, jnp.where(fix_hi_y, ny, ify))
    icy = jnp.where(fix_lo_y, 1, jnp.where(fix_hi_y, ny, icy))

    still_out = (ifx < 1) | (icx > nx) | (ify < 1) | (icy > ny)

    u00 = _gather_mask(usable, ifx, ify)
    u01 = _gather_mask(usable, ifx, icy)
    u10 = _gather_mask(usable, icx, ify)
    u11 = _gather_mask(usable, icx, icy)
    a00 = _gather(slab, ifx, ify)
    a01 = _gather(slab, ifx, icy)
    a10 = _gather(slab, icx, ify)
    a11 = _gather(slab, icx, icy)

    w00 = jnp.where(u00, w00, 0.0)
    w01 = jnp.where(u01, w01, 0.0)
    w10 = jnp.where(u10, w10, 0.0)
    w11 = jnp.where(u11, w11, 0.0)
    wsum = w00 + w01 + w10 + w11
    val = (w00 * a00 + w01 * a01 + w10 * a10 + w11 * a11) / jnp.where(
        wsum > 0, wsum, 1.0
    )
    produced = (~still_out) & (wsum > 0.0)
    val = jnp.where(produced, val, msgval)
    return val, produced


# =============================================================================
# sixteen_pt_average / wt_sixteen_pt_average (interp_module.F)
# =============================================================================
def _sixteen_avg(rx, ry, slab, msgval, usable, weighted):
    nx, ny = slab.shape
    ifx = jnp.floor(rx).astype(jnp.int32)
    ify = jnp.floor(ry).astype(jnp.int32)
    # need ifx in [start+1, end-2] etc. (start=1, end=nx)
    far_enough = (ifx >= 2) & (ifx <= nx - 2) & (ify >= 2) & (ify <= ny - 2)

    rxf = rx
    ryf = ry

    wsum = jnp.zeros_like(rxf)
    acc = jnp.zeros_like(rxf)
    for i in range(1, 5):
        for j in range(1, 5):
            xi = ifx + 3 - i
            yj = ify + 3 - j
            cu = _gather_mask(usable, xi, yj)
            v = _gather(slab, xi, yj)
            if weighted:
                w = jnp.maximum(
                    0.0,
                    2.0
                    - jnp.sqrt(
                        (rxf - xi.astype(jnp.float64)) ** 2
                        + (ryf - yj.astype(jnp.float64)) ** 2
                    ),
                )
            else:
                w = jnp.ones_like(rxf)
            w = jnp.where(cu, w, 0.0)
            wsum = wsum + w
            acc = acc + w * v
    val = acc / jnp.where(wsum > 0, wsum, 1.0)
    produced = far_enough & (wsum > 0.0)
    val = jnp.where(produced, val, msgval)
    return val, produced


def sixteen_pt_average(rx, ry, slab, msgval, usable):
    return _sixteen_avg(rx, ry, slab, msgval, usable, weighted=False)


def wt_sixteen_pt_average(rx, ry, slab, msgval, usable):
    return _sixteen_avg(rx, ry, slab, msgval, usable, weighted=True)


# =============================================================================
# search_extrap (interp_module.F::search_extrap)
# =============================================================================
# Fortran does a BFS over the source grid (expanding ring) from nint(rx,ry),
# returning the nearest usable cell within max search depth. The set of cells the
# BFS can reach within ``depth`` is exactly the L1-ball of radius ``depth`` about
# the origin; among those reachable AND usable, the BFS keeps the one minimizing
# squared Euclidean distance to (rx, ry). We compute that directly. For fine
# Canary targets this branch is essentially inactive (only fires for soil over
# wide water gaps); a scan over target points keeps memory bounded.
def search_extrap(rx, ry, slab, msgval, usable, depth):
    nx, ny = slab.shape
    ix = jnp.round(rx).astype(jnp.int32)
    iy = jnp.round(ry).astype(jnp.int32)
    in_range = (ix >= 1) & (ix <= nx) & (iy >= 1) & (iy <= ny)

    ii = jnp.arange(1, nx + 1, dtype=jnp.float64)  # (nx,)
    jj = jnp.arange(1, ny + 1, dtype=jnp.float64)  # (ny,)
    II, JJ = jnp.meshgrid(ii, jj, indexing="ij")  # (nx,ny)
    flat_slab = slab.reshape(-1)

    npts = rx.shape[0]

    def one_point(carry, k):
        rxk = rx[k]
        ryk = ry[k]
        ixk = ix[k].astype(jnp.float64)
        iyk = iy[k].astype(jnp.float64)
        manhattan = jnp.abs(II - ixk) + jnp.abs(JJ - iyk)
        reachable = (manhattan <= float(depth)) & usable
        d2 = (II - rxk) ** 2 + (JJ - ryk) ** 2
        d2m = jnp.where(reachable, d2, jnp.inf)
        flat = jnp.argmin(d2m)
        found = jnp.isfinite(d2m.reshape(-1)[flat]) & in_range[k]
        val = jnp.where(found, flat_slab[flat], msgval)
        return carry, (val, found)

    _, (vals, founds) = jax.lax.scan(one_point, None, jnp.arange(npts))
    return vals, founds


# =============================================================================
# dispatcher: the +-chain (interp_module.F::interp_sequence)
# =============================================================================
_KERNELS = {
    N_NEIGHBOR: lambda rx, ry, slab, msg, usable, opt: nearest_neighbor(
        rx, ry, slab, msg, usable
    ),
    FOUR_POINT: lambda rx, ry, slab, msg, usable, opt: four_pt(rx, ry, slab, msg, usable),
    SIXTEEN_POINT: lambda rx, ry, slab, msg, usable, opt: sixteen_pt(
        rx, ry, slab, msg, usable
    ),
    AVERAGE4: lambda rx, ry, slab, msg, usable, opt: four_pt_average(
        rx, ry, slab, msg, usable
    ),
    AVERAGE16: lambda rx, ry, slab, msg, usable, opt: sixteen_pt_average(
        rx, ry, slab, msg, usable
    ),
    W_AVERAGE4: lambda rx, ry, slab, msg, usable, opt: wt_four_pt_average(
        rx, ry, slab, msg, usable
    ),
    W_AVERAGE16: lambda rx, ry, slab, msg, usable, opt: wt_sixteen_pt_average(
        rx, ry, slab, msg, usable
    ),
    SEARCH: lambda rx, ry, slab, msg, usable, opt: search_extrap(
        rx, ry, slab, msg, usable, opt
    ),
}


def interp_sequence(
    rx: jnp.ndarray,
    ry: jnp.ndarray,
    slab: jnp.ndarray,
    chain: list[tuple[int, int]],
    msgval: float = DEFAULT_MSGVAL,
    mask_array: jnp.ndarray | None = None,
    maskval: float | None = None,
    mask_relational: str | None = None,
) -> jnp.ndarray:
    """Vectorized ``+``-chain dispatcher faithful to ``interp_sequence``.

    For each target point ``(rx[k], ry[k])`` evaluates the methods in ``chain``
    order, taking the **first** method that produces a value (i.e. the Fortran
    recursion's fall-through). Returns ``msgval`` where no method produced a
    value. ``slab`` is the source slab ``(nx_src, ny_src)`` (i=lon, j=lat),
    ``rx`` indexes the first axis (1-based), ``ry`` the second.
    """

    rx = jnp.asarray(rx, dtype=jnp.float64)
    ry = jnp.asarray(ry, dtype=jnp.float64)
    slab = jnp.asarray(slab, dtype=jnp.float64)
    if mask_array is not None:
        mask_array = jnp.asarray(mask_array, dtype=jnp.float64)
    usable = _source_usable(slab, msgval, mask_array, maskval, mask_relational)

    result = jnp.full(rx.shape, msgval, dtype=jnp.float64)
    chosen = jnp.zeros(rx.shape, dtype=bool)
    for method_id, opt in chain:
        kern = _KERNELS[method_id]
        val, produced = kern(rx, ry, slab, msgval, usable, opt)
        take = produced & (~chosen)
        result = jnp.where(take, val, result)
        chosen = chosen | take
    return result


# =============================================================================
# Source-grid lat/lon -> fractional (rx, ry) for a regular_ll source (AIFS)
# =============================================================================
@dataclass(frozen=True)
class LatLonSourceGrid:
    """Descriptor of the AIFS regular_ll source grid (recon §3).

    AIFS GRIB: 0.25 deg global, Ni=1440 Nj=721, lon increasing, lat 90N -> 90S.
    We store the explicit first-point + increments so the lltoxy mapping is
    unambiguous.

    lon0_deg, dlon_deg: longitude of i=1 and the per-i increment (deg, +east).
    lat0_deg, dlat_deg: latitude  of j=1 and the per-j increment (deg). For AIFS
       lat0=90, dlat=-0.25 (lat DECREASES with j).
    nx, ny: source dims (i over lon, j over lat).
    global_wrap: True if the grid spans 360 deg in longitude (enables wrap so a
       target lon just outside [lon0, lon0+360) still maps).
    """

    lon0_deg: float
    dlon_deg: float
    lat0_deg: float
    dlat_deg: float
    nx: int
    ny: int
    global_wrap: bool = True


def latlon_to_source_xy(
    rlat: jnp.ndarray, rlon: jnp.ndarray, grid: LatLonSourceGrid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Maps target (lat, lon) in degrees to 1-based fractional source index
    ``(rx, ry)`` on the regular_ll source grid (the metgrid ``lltoxy`` for
    ``PROJ_LATLON``). ``rx`` indexes lon (first slab axis), ``ry`` lat (second).

    Longitudes are wrapped into ``[lon0, lon0+360)`` for a global source so the
    ``rx`` index lands in ``[1, nx+1)`` (the bounds check is the caller's).
    """

    rlat = jnp.asarray(rlat, dtype=jnp.float64)
    rlon = jnp.asarray(rlon, dtype=jnp.float64)
    lon = rlon
    if grid.global_wrap:
        lon = grid.lon0_deg + jnp.mod(rlon - grid.lon0_deg, 360.0)
    rx = (lon - grid.lon0_deg) / grid.dlon_deg + 1.0
    ry = (rlat - grid.lat0_deg) / grid.dlat_deg + 1.0
    return rx, ry


def interp_field_to_grid(
    src_field: jnp.ndarray,
    target_lat: jnp.ndarray,
    target_lon: jnp.ndarray,
    grid: LatLonSourceGrid,
    chain: list[tuple[int, int]],
    msgval: float = DEFAULT_MSGVAL,
    mask_array: jnp.ndarray | None = None,
    maskval: float | None = None,
    mask_relational: str | None = None,
) -> jnp.ndarray:
    """Horizontally interpolate one 2-D source slab ``(nx_src, ny_src)`` onto a
    target grid given by 2-D ``target_lat/target_lon`` ``(ny_tgt, nx_tgt)`` arrays
    (met_em row-major: south_north then west_east). Returns ``(ny_tgt, nx_tgt)``.

    This is the per-level workhorse; callers (metgrid_assemble) loop levels and
    select the source slab + target lat/lon per output stagger.
    """

    ny_t, nx_t = target_lat.shape
    rlat = jnp.asarray(target_lat, dtype=jnp.float64).reshape(-1)
    rlon = jnp.asarray(target_lon, dtype=jnp.float64).reshape(-1)
    rx, ry = latlon_to_source_xy(rlat, rlon, grid)
    out = interp_sequence(
        rx, ry, src_field, chain, msgval, mask_array, maskval, mask_relational
    )
    return out.reshape(ny_t, nx_t)
