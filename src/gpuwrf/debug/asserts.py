"""JAX debug assertions guarded by Python static booleans."""

from __future__ import annotations

import jax.numpy as jnp


def assert_finite(x, name: str, *, enabled: bool):
    """Returns immediately in production; emits NaNs in debug when finiteness fails."""

    if not enabled:
        return x
    del name
    ok = jnp.all(jnp.isfinite(x))
    return jnp.where(ok, x, x * jnp.nan)


def assert_physical_bounds(x, lo: float, hi: float, name: str, *, enabled: bool):
    """Returns immediately in production; emits NaNs in debug on bound violations."""

    if not enabled:
        return x
    del name
    ok = jnp.all((x >= lo) & (x <= hi) & jnp.isfinite(x))
    return jnp.where(ok, x, x * jnp.nan)
