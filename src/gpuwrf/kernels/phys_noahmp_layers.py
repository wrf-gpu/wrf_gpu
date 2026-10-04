"""Noah-MP layer updates without in-place update chains or level scans.

Flag ``GPUWRF_NOAHMP_LAYER_SELECT=1`` (default off), effective only on the
native REAL path (``GPUWRF_NOAHMP_NATIVE_REAL=1``). On the GPU every
``x.at[k].set(v)`` on a layer-leading array is its own in-place
dynamic-update-slice kernel, and the soil ROSR12 ``lax.scan`` launches per row.
Here a static-layer update is an elementwise select over the layer axis (same
values; selects fuse with their producers) and ROSR12 is unrolled over the
static layer count with the scan bodies' operation order. Off the flag the
helpers return exactly the retained ``.at`` / scan operations.
"""
from __future__ import annotations

import os

import jax.numpy as jnp

from gpuwrf.physics.noahmp.precision import native_real_enabled


def layers_enabled() -> bool:
    return native_real_enabled() and os.environ.get("GPUWRF_NOAHMP_LAYER_SELECT", "0") == "1"


def lists_enabled() -> bool:
    """``GPUWRF_NOAHMP_LAYER_LISTS=1`` (on top of the layer-select path, default off): the
    layer-loop routines keep their layer arrays as Python lists of 2-D fields and stack once,
    so XLA sees one shape instead of per-layer select kernels (#14; same expressions)."""
    return layers_enabled() and os.environ.get("GPUWRF_NOAHMP_LAYER_LISTS", "0") == "1"


def to_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x[k] for k in range(x.shape[0])]


def put(layers, k: int, value, dtype):
    """List form of ``set_layer``: the slot takes ``value`` cast to the layer dtype."""
    layers[k] = jnp.asarray(value, dtype)


def _select(x, k, value):
    n = x.shape[0]
    ids = jnp.arange(n, dtype=jnp.int32).reshape((n,) + (1,) * (x.ndim - 1))
    return jnp.where(ids == k % n, jnp.asarray(value, x.dtype)[None], x)


def set_layer(x, k: int, value):
    """``x.at[k].set(value)`` for a static leading index ``k``."""
    if not layers_enabled():
        return x.at[k].set(value)
    return _select(x, k, value)


def add_layer(x, k: int, value):
    """``x.at[k].add(value)`` for a static leading index ``k``.

    Same arithmetic as JAX's scatter-add: the sum is formed in the promoted dtype
    of ``x`` and ``value``, then cast to ``x``'s dtype.
    """
    if not layers_enabled():
        return x.at[k].add(value)
    return _select(x, k, x[k] + value)


def rosr12_forward(carry, rows, eps=1e-30):
    """Unrolled ``soil_thermo._rosr12_fwd_scan`` (same per-row order)."""
    p_prev, delta_prev = carry
    a, b, c, d = rows
    ps, ds = [], []
    for k in range(a.shape[0]):
        denom = b[k] + a[k] * p_prev
        denom = jnp.where(jnp.abs(denom) > eps, denom, eps)
        p_prev = -c[k] / denom
        delta_prev = (d[k] - a[k] * delta_prev) / denom
        ps.append(p_prev)
        ds.append(delta_prev)
    return (p_prev, delta_prev), (jnp.stack(ps), jnp.stack(ds))


def rosr12_backward(x_next, rows):
    """Unrolled ``soil_thermo._rosr12_bwd_scan`` (same per-row order)."""
    p, delta = rows
    xs = []
    for k in range(p.shape[0]):
        x_next = p[k] * x_next + delta[k]
        xs.append(x_next)
    return x_next, jnp.stack(xs)


__all__ = ["layers_enabled", "lists_enabled", "to_list", "put", "set_layer", "add_layer", "rosr12_forward",
           "rosr12_backward"]
