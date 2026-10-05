"""Noah-MP column routines as ONE Pallas launch (#14, flag ``GPUWRF_NOAHMP_COLUMN_KERNELS=1``, default off).

The VEGE_FLUX / BARE_FLUX Newton loops are pure per-column 2-D arithmetic; under XLA every sweep
is at least one kernel (a sweep's transcendental results feed both the next sweep and the outputs,
so XLA will not duplicate them into one fusion). :func:`column_call` traces the UNCHANGED routine
inside a Triton kernel on ``(block,)`` column vectors: every array leaf that broadcasts to the 2-D
column grid is passed per column, Python scalars stay static, and layered leaves (snow/soil
levels) are replaced by a sentinel that fails loudly if the routine touches them. The
per-sweep ``optimization_barrier`` (a fusion fence for XLA) is skipped while tracing inside the
kernel. Expressions are the routine's own; results match the XLA path up to compiler
contraction/libdevice differences (Noah-gate equivalence, not bitwise).
"""
from __future__ import annotations

import contextvars
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

BLOCK = 128
_IN_KERNEL = contextvars.ContextVar("noahmp_column_kernel", default=False)


def columns_enabled() -> bool:
    from gpuwrf.physics.noahmp.precision import native_real_enabled
    return native_real_enabled() and os.environ.get("GPUWRF_NOAHMP_COLUMN_KERNELS", "0") == "1"


def in_column_kernel() -> bool:
    return _IN_KERNEL.get()


class _Layered:
    """Placeholder for a leaf that is not a per-column 2-D field."""

    def __init__(self, shape):
        self.shape = shape

    def __getattr__(self, name):
        raise TypeError(f"column kernel routine touched a non-column leaf of shape {self.shape}")


def column_call(fn, args, *, grid_shape, name, interpret=False, block=BLOCK, warps=4):
    """``fn(*args)`` over all columns in one Pallas launch; returns ``fn``'s output pytree of 2-D arrays.

    Boolean and int32 column inputs ride in the f32 operand stack (exact: Noah-MP's integer
    column fields are small layer counts/flags, |value| < 2**24) and are restored in-kernel."""
    ny, nx = grid_shape
    ncol = ny * nx
    leaves, treedef = jax.tree.flatten(args)
    dyn, kinds = [], []
    for leaf in leaves:
        if isinstance(leaf, (jax.Array, jnp.ndarray)) or hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
            arr = jnp.asarray(leaf)
            if arr.ndim <= 2 and all(s in (1, t) for s, t in zip(arr.shape[::-1], (nx, ny))):
                col = jnp.broadcast_to(arr, (ny, nx)).reshape(ncol)
                if col.dtype == jnp.bool_:
                    kinds.append(("bool", None))
                    col = col.astype(jnp.float32)
                elif col.dtype == jnp.int32:
                    kinds.append(("int32", None))
                    col = col.astype(jnp.float32)
                else:
                    kinds.append(("col", None))
                dyn.append(col)
            else:
                kinds.append(("layered", arr.shape))
        else:
            kinds.append(("static", leaf))

    def rebuild(columns):
        it = iter(columns)
        out = []
        for kind, extra in kinds:
            if kind == "col":
                out.append(next(it))
            elif kind == "bool":
                out.append(next(it) != 0)
            elif kind == "int32":
                out.append(next(it).astype(jnp.int32))
            elif kind == "layered":
                out.append(_Layered(extra))
            else:
                out.append(extra)
        return jax.tree.unflatten(treedef, out)

    def run(columns):
        token = _IN_KERNEL.set(True)
        try:
            return fn(*rebuild(columns))
        finally:
            _IN_KERNEL.reset(token)

    abstract = [jax.ShapeDtypeStruct((block,), c.dtype) for c in dyn]
    out_struct = jax.eval_shape(run, abstract)
    out_leaves, out_tree = jax.tree.flatten(out_struct)
    # One stacked operand / result per dtype: the producers of all columns (casts, broadcasts)
    # fuse into ONE concatenate instead of one materialised array each; consumers slice rows.
    in_groups = _groups([c.dtype for c in dyn])
    out_is_bool = [o.dtype == jnp.bool_ for o in out_leaves]  # stored as int32, restored below
    out_groups = _groups([jnp.int32 if b else o.dtype for o, b in zip(out_leaves, out_is_bool)])
    stacked = [jnp.stack([dyn[j] for j in members]) for _, members in in_groups]

    def body(*refs):
        i = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
        valid = i < ncol
        cols = [None] * len(dyn)
        for ref, (_, members) in zip(refs[:len(in_groups)], in_groups):
            for row, j in enumerate(members):
                cols[j] = pt.load(ref.at[row, i], mask=valid, other=1)
        outs = jax.tree.leaves(run(cols))
        for ref, (dtype, members) in zip(refs[len(in_groups):], out_groups):
            for row, j in enumerate(members):
                pt.store(ref.at[row, i], jnp.broadcast_to(jnp.asarray(outs[j], dtype), (block,)), mask=valid)

    out_shape = tuple(jax.ShapeDtypeStruct((len(members), ncol), dtype) for dtype, members in out_groups)
    # Pallas interpreter on a CPU backend (release defaults on CPU, like the other release kernels) or on request
    # (CPU validation, trace-time flag); GPU runs always lower to Triton.
    interpret = (interpret or os.environ.get("GPUWRF_NOAHMP_COLUMN_INTERPRET", "0") == "1"
                 or jax.default_backend() == "cpu")
    values = pl.pallas_call(body, grid=((ncol + block - 1) // block,), name=name, interpret=interpret,
                            compiler_params=pt.CompilerParams(num_warps=warps), out_shape=out_shape)(*stacked)
    flat = [None] * len(out_leaves)
    for value, (_, members) in zip(values, out_groups):
        for row, j in enumerate(members):
            flat[j] = value[row].reshape(ny, nx)
            if out_is_bool[j]:
                flat[j] = flat[j] != 0
    return jax.tree.unflatten(out_tree, flat)


def _groups(dtypes):
    """Indices grouped by dtype, first-appearance order: [(dtype, [indices])]."""
    order = {}
    for j, dtype in enumerate(dtypes):
        order.setdefault(jnp.dtype(dtype), []).append(j)
    return list(order.items())


__all__ = ["columns_enabled", "in_column_kernel", "column_call"]
