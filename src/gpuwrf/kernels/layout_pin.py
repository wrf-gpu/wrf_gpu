"""GPUWRF_LAYOUT_PIN (b-core P1, default off): row-major layout constraints at dycore<->physics seams.

XLA layout assignment turns a logical transpose into a bitcast whenever it can -- the physics
column views ``moveaxis(f, 0, -1)``, the z-minor transpose its ReduceWindowRewriter inserts for a
vertical cumsum -- by giving the transpose's operand the permuted layout, and propagates that layout
back into dycore fields and loop carries until it meets a row-major Pallas operand, where it copies
per use (inside the acoustic/RK loops). Which seam wins is a multistable whole-module choice (RC1
d02 248 -> 441 transposes from top_lid + #14, b-core L1). Constraining both sides of a seam to
row-major makes it one isolated physical transpose at the seam. Values are unchanged (layout only).

Parts ("_"-separated): cols (physics column views), cum (vertical cumsums), ac (acoustic scan payload),
carry (the step-loop carry: every >= 2-D leaf row-major at body entry and exit);
scope part ``nested``: pin only while tracing a nested-child step (``ring_select.nested_step``), so the
root keeps its layout (the same pins push the d01 root to a heavier state: GPU HLO d01 1055 -> 1384
transposes, +0.64 ms/root, b-core P1G). "1" = cols_cum_ac_carry_nested.
"""
import os

ALL_PARTS = ("ac", "carry", "cols", "cum", "nested")


def parse_parts(value):
    """Flag value -> enabled parts as a SORTED tuple ("" / "0" none, "1" all, else "_"-separated names).

    A tuple, not a set: PARTS is hashed into the AOT cheap key (IMPORT_TIME_ENV_CONSTANTS) and a set's
    repr order follows the per-process string-hash seed -> a different key in every process (AOT miss)."""
    if value in ("", "0"):
        return ()
    if value == "1":
        return ALL_PARTS
    return tuple(sorted(set(value.split("_"))))


PARTS = parse_parts(os.environ.get("GPUWRF_LAYOUT_PIN", "0"))


def row_major(x):
    """Layout constraint only: the default row-major order for every >= 2-D jax array."""
    import jax
    from jax.experimental.layout import Layout, with_layout_constraint
    if not isinstance(x, jax.Array) or x.ndim < 2:
        return x
    return with_layout_constraint(x, Layout(major_to_minor=tuple(range(x.ndim))))


def pin(part, value):
    """``value`` (any pytree) row-major-constrained when ``part`` is enabled, else unchanged."""
    if part not in PARTS:
        return value
    if "nested" in PARTS:
        from gpuwrf.kernels.ring_select import _NESTED_STEP
        if not _NESTED_STEP.get():
            return value
    import jax
    return jax.tree.map(row_major, value)


# e41-class (Swiss E41): on SMALL grids XLA's GPU fusion cost model (launch cost >> traffic) merges independent work
# of different phases into one multi-output kInput fusion (Swiss 42x42: surface-layer psi lookups + Noah internals +
# dycore curvature = HLO 755; RK stage k small_step_finish + stage k+1 PGF/Coriolis + advection = HLO 590; the
# b6 tree already had five > 500 before any layout pin). An optimization barrier at the phase seams bounds every
# fusion to one phase there. Grids above SMALL_GRID_COLUMNS (every WN/PROD/Monica domain: >= 8400 columns) trace
# NOTHING new: the value is returned as is, so their programs stay byte-identical.
SMALL_GRID_COLUMNS = 4096


def small_grid_firewall(value, horizontal_shape):
    """``value`` (any pytree) with all its jax.Array leaves behind ONE ``lax.optimization_barrier`` when
    ``prod(horizontal_shape)`` (the domain's (ny, nx) or (ncol,)) is <= SMALL_GRID_COLUMNS, else ``value`` itself (same
    object, no op traced). Non-array leaves (Python/NumPy static sentinels) pass through unchanged."""
    columns = 1
    for size in tuple(horizontal_shape):
        columns *= int(size)
    if columns > SMALL_GRID_COLUMNS:
        return value
    import jax
    leaves, treedef = jax.tree_util.tree_flatten(value)
    arrays = [i for i, leaf in enumerate(leaves) if isinstance(leaf, jax.Array)]
    if not arrays:
        return value
    fenced = jax.lax.optimization_barrier(tuple(leaves[i] for i in arrays))
    for i, leaf in zip(arrays, fenced):
        leaves[i] = leaf
    return jax.tree_util.tree_unflatten(treedef, leaves)
