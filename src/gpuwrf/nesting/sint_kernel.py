"""One-kernel pristine-WRF SINT for the live-nest force-down ring.

``build_child_boundary_package`` (coupled force-down) computes
``field_sides_3d(interp_sint_full(parent, ...), width, side_len)``.  Under the
producer JIT that pair runs with XLA fusion disabled (exact eager rounding),
which costs ~366 single-op kernels per field.  This module evaluates the same
cells in one Pallas kernel:

* only the ``(side, width, z, side_len)`` ring cells are computed (each SINT
  output cell depends only on its own child row/column plan, so this is a pure
  restriction of the full-grid result);
* every multiply/add/subtract/divide is an explicitly rounded PTX ``.rn``
  instruction in the expression order of :func:`interp._sint_limited_1d`, so
  neither LLVM nor ptxas can contract it into an FMA.  The result therefore
  reproduces the unfused per-operation rounding instead of approximating it.

Out-of-range parent indices load NaN, matching ``jnp.take``'s fill mode (a
valid WRF nest never reaches them).  GPU only: CPU callers keep the JAX path
except for the explicit ``interpret=True`` logic tests.
"""

from __future__ import annotations

from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.nesting.interp import InterpWeights

_BLOCK = 256
_PTX_TYPE = {np.dtype(np.float32): ("f32", "f"), np.dtype(np.float64): ("f64", "d")}


def _host_subcell_offsets(nchild: int, ratio: int, staggered: bool, dtype) -> np.ndarray:
    """``XIG``/``XJG`` of ``interp._sint_axis_plan`` as a host constant.

    The offsets depend only on the static child index, so evaluating them in
    NumPy (IEEE, true division) leaves no traced multiply-add for a fusing
    compiler to contract and no reciprocal rewrite to vary with the ratio
    (review-fdfuse advisory).  Identical to the traced plan for ratios 2-4.
    """

    rr = int(ratio)
    f = np.dtype(dtype).type
    shift = max((rr - 1) // 2, 1) if bool(staggered) else 0
    subcell = (np.arange(int(nchild)) + shift) % rr
    offset = 1.0 if bool(staggered) and rr % 2 == 0 else 0.0
    return f(float(rr - 1) - offset) / f(float(2 * rr)) - subcell.astype(dtype) / f(float(rr))


def _int_centers(lower: jax.Array, nchild: int, ratio: int, staggered: bool) -> jax.Array:
    """Central parent index of ``interp._sint_axis_plan`` in integer arithmetic.

    SINT weights (``interp._build(cell_centered=True)``) place child 0 at the
    parent coordinate ``(start-1) - (ratio//2)/ratio``, stored as
    ``lower[0] = start-2`` plus a fraction; the plan's
    ``rint(coord[0] + (ratio//2)/ratio)`` is therefore ``lower[0] + 1``.  The
    destination shift and ``fine // ratio`` are static, so no f64 remains.
    """

    rr = int(ratio)
    shift = max((rr - 1) // 2, 1) if bool(staggered) else 0
    steps = (np.arange(int(nchild)) + shift) // rr
    return lower[0].astype(jnp.int32) + jnp.int32(1) + jnp.asarray(steps, dtype=jnp.int32)


def sint_kernel_enabled() -> bool:
    """Default ON for GPU; ``GPUWRF_SINT_KERNEL=0`` restores the unfused path."""

    flag = os.environ.get("GPUWRF_SINT_KERNEL", "1").strip().lower()
    return flag not in ("0", "false", "off", "no") and jax.default_backend() == "gpu"


def _sint_sides_kernel(
    src, xc_ref, xig_ref, yc_ref, xjg_ref, out,
    *, nz, py, px, ny, nx, width, side_len, dtype, interpret, div_mode,
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    np_dtype = np.dtype(dtype)
    ptx, reg = _PTX_TYPE[np_dtype]
    f = np_dtype.type
    total = 4 * width * nz * side_len
    i = pl.program_id(0) * _BLOCK + jnp.arange(_BLOCK, dtype=jnp.int32)
    live = i < total
    t = i % side_len
    z = (i // side_len) % nz
    d = (i // (side_len * nz)) % width
    side = i // (side_len * nz * width)
    # W/E strips run along child rows, S/N strips along child columns; E/N
    # distance 0 is the outer edge (field_sides_3d's flipped strips).
    along_y = side < 2
    y = jnp.where(along_y, t, jnp.where(side == 2, d, ny - 1 - d))
    x = jnp.where(along_y, jnp.where(side == 0, d, nx - 1 - d), t)
    valid = live & jnp.where(along_y, (t < ny) & (d < nx), (t < nx) & (d < ny))
    yy = jnp.where(valid, y, 0)
    xx = jnp.where(valid, x, 0)

    def load_vec(ref, idx):
        return pt.load(ref.at[idx], mask=valid, other=0)

    nan = f(np.nan)

    def rn(op, a, b):
        if interpret:
            return {"mul": a * b, "add": a + b, "sub": a - b, "div": a / b}[op]
        mode = div_mode if op == "div" else "rn"
        return pt.elementwise_inline_asm(
            f"{op}.{mode}.{ptx} $0, $1, $2;", args=[a, b],
            constraints=f"={reg},{reg},{reg}", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, np_dtype)],
        )[0]

    mul = partial(rn, "mul")
    add = partial(rn, "add")
    sub = partial(rn, "sub")
    div = partial(rn, "div")
    zero = jnp.full((_BLOCK,), f(0), np_dtype)
    one = jnp.full((_BLOCK,), f(1), np_dtype)
    one12 = jnp.full((_BLOCK,), f(1) / f(12), np_dtype)
    one24 = jnp.full((_BLOCK,), f(1) / f(24), np_dtype)
    ep = jnp.full((_BLOCK,), f(1.0e-10), np_dtype)
    c3, c7, c15 = (jnp.full((_BLOCK,), f(v), np_dtype) for v in (3.0, 7.0, 15.0))

    def limited(ym2, ym1, y0, yp1, yp2, a):
        # interp._sint_limited_1d, operation for operation.
        sign_a = jnp.where(a >= zero, one, -one)
        sp, sn = jnp.maximum(zero, sign_a), jnp.minimum(zero, sign_a)
        fl0 = mul(sub(mul(ym1, sp), mul(y0, sn)), a)
        fl1 = mul(sub(mul(y0, sp), mul(yp1, sn)), a)
        w = sub(y0, sub(fl1, fl0))
        mxm = jnp.maximum(jnp.maximum(ym1, y0), jnp.maximum(yp1, w))
        mn = jnp.minimum(jnp.minimum(ym1, y0), jnp.minimum(yp1, w))
        aa = mul(a, a)
        aaa = mul(aa, a)
        aaaa = mul(aaa, a)

        def tr4(vm1, v0, vp1, vp2):
            t1 = mul(mul(a, one12), sub(mul(c7, add(vp1, v0)), add(vp2, vm1)))
            t2 = mul(mul(aa, one24), sub(mul(c15, sub(vp1, v0)), sub(vp2, vm1)))
            t3 = mul(mul(aaa, one12), sub(add(vp1, v0), add(vp2, vm1)))
            t4 = mul(mul(aaaa, one24), sub(mul(c3, sub(vp1, v0)), sub(vp2, vm1)))
            return add(sub(sub(t1, t2), t3), t4)

        f0 = sub(tr4(ym2, ym1, y0, yp1), fl0)
        f1 = sub(tr4(ym1, y0, yp1, yp2), fl1)
        ov = div(sub(mxm, w), add(add(-jnp.minimum(zero, f1), jnp.maximum(zero, f0)), ep))
        un = div(sub(w, mn), add(sub(jnp.maximum(zero, f1), jnp.minimum(zero, f0)), ep))
        f0 = add(mul(jnp.maximum(zero, f0), jnp.minimum(one, ov)),
                 mul(jnp.minimum(zero, f0), jnp.minimum(one, un)))
        f1 = add(mul(jnp.maximum(zero, f1), jnp.minimum(one, un)),
                 mul(jnp.minimum(zero, f1), jnp.minimum(one, ov)))
        return sub(w, sub(f1, f0))

    xc = load_vec(xc_ref, (xx,))
    xig = load_vec(xig_ref, (xx,))
    yc = load_vec(yc_ref, (yy,))
    xjg = load_vec(xjg_ref, (yy,))
    rows = []
    for dy in range(-2, 3):
        prow = yc + dy
        row_ok = (prow >= 0) & (prow < py)
        stencil = []
        for dx in range(-2, 3):
            pcol = xc + dx
            ok = valid & row_ok & (pcol >= 0) & (pcol < px)
            stencil.append(pt.load(
                src.at[jnp.where(ok, z, 0), jnp.where(ok, prow, 0), jnp.where(ok, pcol, 0)],
                mask=ok, other=nan,
            ))
        rows.append(limited(*stencil, xig))
    value = limited(*rows, xjg)
    pt.store(out.at[i], jnp.where(valid, value, zero), mask=live)


@partial(
    jax.jit,
    static_argnames=(
        "parent_grid_ratio", "xstag", "ystag", "width", "side_len", "interpret", "_div",
    ),
)
def sint_sides(
    field: jax.Array,
    weights: InterpWeights,
    *,
    parent_grid_ratio: int,
    xstag: bool = False,
    ystag: bool = False,
    width: int,
    side_len: int,
    interpret: bool = False,
    _div: str = "rn",
) -> jax.Array:
    """``field_sides_{3,2}d(interp_sint_full(field, ...))`` in one kernel.

    ``_div`` selects the PTX division ("rn" IEEE; "full" mirrors XLA's f32
    ``-nvptx-prec-divf32=1`` lowering; f64 only supports "rn").
    """

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    was_2d = field.ndim == 2
    source = field[None] if was_2d else field
    nz, py, px = source.shape
    dtype = source.dtype
    ny, nx = int(weights.y0.shape[0]), int(weights.x0.shape[0])
    x_center = _int_centers(weights.x0, nx, parent_grid_ratio, xstag)
    y_center = _int_centers(weights.y0, ny, parent_grid_ratio, ystag)
    # Host offsets in the plan dtype, rounded to the field dtype on the host
    # (same round-to-nearest-even as an XLA convert): no f64 enters the graph.
    plan_dtype = np.dtype(weights.wx.dtype)
    xig = jnp.asarray(_host_subcell_offsets(nx, parent_grid_ratio, xstag, plan_dtype).astype(dtype))
    xjg = jnp.asarray(_host_subcell_offsets(ny, parent_grid_ratio, ystag, plan_dtype).astype(dtype))
    total = 4 * int(width) * nz * int(side_len)
    flat = pl.pallas_call(
        partial(
            _sint_sides_kernel, nz=nz, py=py, px=px, ny=ny, nx=nx,
            width=int(width), side_len=int(side_len), dtype=dtype, interpret=bool(interpret),
            div_mode=str(_div) if np.dtype(dtype) == np.float32 else "rn",
        ),
        out_shape=jax.ShapeDtypeStruct((total,), dtype),
        grid=((total + _BLOCK - 1) // _BLOCK,),
        interpret=bool(interpret),
        compiler_params=pt.CompilerParams(num_warps=4),
        name="nest_sint_sides",
    )(
        source,
        x_center.astype(jnp.int32),
        xig.astype(dtype),
        y_center.astype(jnp.int32),
        xjg.astype(dtype),
    )
    return flat.reshape(4, int(width), nz, int(side_len))
