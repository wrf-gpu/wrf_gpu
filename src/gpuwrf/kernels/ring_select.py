"""Exact single-op replacements for per-row boundary-ring write chains (GPUWRF_SPEC_RING_SELECT).

Each helper reproduces a sequence of ``.at[...].set`` writes bit-for-bit: the
values are only selected/copied, never recomputed, and a trace-time numpy map
replays the original write order so overlapping corners keep the LAST writer.
"""
import contextlib
import contextvars
import os

import jax.numpy as jnp
import numpy as np

# True while a NESTED-child step program is being traced (set by operational_mode's step
# builder). The row-major scatter pin repairs the nested step layout (strict guards
# default) but flips the root step's carry layout, so it applies on nested steps only.
_NESTED_STEP = contextvars.ContextVar("gpuwrf_ring_select_nested_step", default=False)


@contextlib.contextmanager
def nested_step(nested):
    token = _NESTED_STEP.set(bool(nested))
    try:
        yield
    finally:
        _NESTED_STEP.reset(token)


def enabled():
    return os.environ.get("GPUWRF_SPEC_RING_SELECT", "0") == "1"


def _row_major(value):
    """Layout constraint only (values unchanged): default row-major minor-to-major order."""
    from jax.experimental.layout import Layout, with_layout_constraint
    return with_layout_constraint(value, Layout(major_to_minor=tuple(range(value.ndim))))


def ring_mask(y_len, x_len, width):
    """Cells with min distance to the lateral edge < width (the union of the ring writes)."""
    yy = np.arange(int(y_len))[:, None]
    xx = np.arange(int(x_len))[None, :]
    w = int(width)
    return (yy < w) | (yy >= int(y_len) - w) | (xx < w) | (xx >= int(x_len) - w)


def scatter_mask(field, target, mask):
    """``field`` with the cells of the static 2-D ``mask`` (last two axes) taken from ``target``.

    One in-place scatter over the masked cells only (static, sorted, unique
    indices; the gather from ``target`` fuses into it). Unlike a full-field
    ``jnp.where`` this keeps the operand buffer -- a select builds a new array
    (G07: +391 transposes on d02 when the loop carried the field column-major).

    On nested-child steps operand and result are pinned to the row-major layout of the REAL carry: the
    z-window scatter otherwise pulls its operand, and through it the RK fields,
    to a z-fastest layout once nothing else anchors them (strict guards default,
    F2C): d02 step 498 -> 233 transposes, legacy guards 230 (b-diff BD69/BD70).
    """
    ys, xs = np.nonzero(np.asarray(mask))
    if ys.size == 0:
        return field
    ys, xs = jnp.asarray(ys, jnp.int32), jnp.asarray(xs, jnp.int32)
    values = jnp.asarray(target)[..., ys, xs].astype(field.dtype)
    if not _NESTED_STEP.get():
        return field.at[..., ys, xs].set(values, indices_are_sorted=True, unique_indices=True)
    out = _row_major(field).at[..., ys, xs].set(values, indices_are_sorted=True, unique_indices=True)
    return _row_major(out)


def select_ring(field, target, width):
    """``field`` with every ring cell replaced by ``target`` (cast like ``.at[].set``)."""
    if int(width) <= 0:
        return field
    return scatter_mask(field, target, ring_mask(field.shape[-2], field.shape[-1], width))


def full_ring_source_map(n_width, side_index, y_len, x_len, pad_len):
    """Flat source index into a leaf viewed as ``(side*n_width*pad_len)`` per level, or -1.

    Replays ``_full_ring_target_from_leaf``'s loop: for b in range(n_width):
    W column b, E column x-1-b, S row b, N row y-1-b (later writes win).
    """
    y_len, x_len = int(y_len), int(x_len)
    src = np.full((y_len, x_len), -1, dtype=np.int64)

    def flat(side, b, pos):
        return (side_index[side] * n_width + b) * pad_len + pos

    ys, xs = np.arange(y_len), np.arange(x_len)
    for b in range(int(n_width)):
        src[:, b] = flat("W", b, ys)
        src[:, x_len - 1 - b] = flat("E", b, ys)
        src[b, :] = flat("S", b, xs)
        src[y_len - 1 - b, :] = flat("N", b, xs)
    return src


def full_ring_target(leaf, z_len, y_len, x_len, dtype, side_index):
    """One gather equivalent of ``_full_ring_target_from_leaf`` for a ``(side, width, z, len)`` leaf."""
    n_side, n_width, _, pad_len = (int(n) for n in leaf.shape)
    src = full_ring_source_map(n_width, side_index, y_len, x_len, pad_len)
    flat = jnp.moveaxis(leaf[:, :, : int(z_len), :], 2, 0).reshape(int(z_len), n_side * n_width * pad_len)
    values = jnp.take(flat.astype(dtype), jnp.asarray(np.maximum(src, 0).reshape(-1)), axis=1)
    values = values.reshape(int(z_len), int(y_len), int(x_len))
    return jnp.where(jnp.asarray(src >= 0)[None], values, jnp.zeros((), dtype=dtype))
