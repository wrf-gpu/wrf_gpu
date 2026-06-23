"""Positive-definite scalar limiter skeletons for the c2 dycore."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from functools import partial

import jax
from jax import config
import jax.numpy as jnp


configure_jax_x64()


@dataclass(frozen=True)
class LimiterConfig:
    """Config for disabled-by-default scalar limiting."""

    enabled: bool = False
    floor: float = 0.0
    mass_tolerance: float = 1.0e-12


@partial(jax.jit, static_argnames=("config",))
def positive_definite_limiter(scalar: jax.Array, mass: jax.Array, config: LimiterConfig) -> jax.Array:
    """Clips negatives and rescales positives to preserve global scalar mass.

    This follows the architecture pattern of Pace ``FillNegativeTracerValues``:
    limiter repair is explicit, mass-aware, and separately testable. It is not a
    line-by-line WRF port.
    """

    if not config.enabled:
        return scalar
    floor = float(config.floor)
    before = jnp.sum(scalar * mass)
    clipped = jnp.maximum(scalar, floor)
    after = jnp.sum(clipped * mass)
    scale = jnp.where(after > 0.0, before / after, 1.0)
    limited = clipped * scale
    return jnp.maximum(limited, floor)


@partial(jax.jit, static_argnames=("config",))
def monotonic_limiter(scalar: jax.Array, lower: jax.Array, upper: jax.Array, config: LimiterConfig) -> jax.Array:
    """Applies a disabled-by-default monotonic clip to scalar fields."""

    if not config.enabled:
        return scalar
    return jnp.clip(scalar, lower, upper)


@partial(jax.jit, static_argnames=())
def scalar_mass(scalar: jax.Array, mass: jax.Array) -> jax.Array:
    """Returns the mass-weighted scalar integral used in limiter proof objects."""

    return jnp.sum(scalar * mass)


@partial(jax.jit, static_argnames=())
def limiter_diagnostics(before: jax.Array, after: jax.Array, mass: jax.Array) -> jax.Array:
    """Returns min-after and mass residual for proof scripts."""

    before_mass = scalar_mass(before, mass)
    after_mass = scalar_mass(after, mass)
    residual = jnp.abs(after_mass - before_mass)
    relative = residual / jnp.maximum(jnp.abs(before_mass), 1.0e-30)
    return jnp.asarray([jnp.min(after), before_mass, after_mass, relative], dtype=jnp.float64)
