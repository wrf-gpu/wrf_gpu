"""Sixth-order hyperdiffusion skeleton for c2 dycore stabilization."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from functools import partial

import jax
from jax import config
import jax.numpy as jnp


configure_jax_x64()


@dataclass(frozen=True)
class HyperdiffusionConfig:
    """Config for disabled-by-default sixth-order diffusion.

    WRF source anchors: ``module_big_step_utilities_em.F:6506-6511`` passes
    ``diff_6th_opt``, ``diff_6th_factor``, and map factors; line ``6605`` builds
    the coefficient; lines ``6753-6756`` guard up-gradient diffusion.
    """

    enabled: bool = False
    coefficient: float = 0.0
    monotonic_guard: bool = False


@partial(jax.jit, static_argnames=("axis",))
def sixth_derivative_periodic(field: jax.Array, *, axis: int) -> jax.Array:
    """Returns the centered sixth-difference stencil on one axis."""

    return (
        jnp.roll(field, -3, axis=axis)
        - 6.0 * jnp.roll(field, -2, axis=axis)
        + 15.0 * jnp.roll(field, -1, axis=axis)
        - 20.0 * field
        + 15.0 * jnp.roll(field, 1, axis=axis)
        - 6.0 * jnp.roll(field, 2, axis=axis)
        + jnp.roll(field, 3, axis=axis)
    )


@partial(jax.jit, static_argnames=("config", "axis"))
def apply_hyperdiffusion_axis(field: jax.Array, config: HyperdiffusionConfig, *, axis: int) -> jax.Array:
    """Applies a sixth-order damping skeleton along one axis."""

    if not config.enabled or float(config.coefficient) == 0.0:
        return field
    tendency = sixth_derivative_periodic(field, axis=axis)
    if config.monotonic_guard:
        tendency = jnp.where(tendency * field > 0.0, 0.0, tendency)
    return field + float(config.coefficient) * tendency


@partial(jax.jit, static_argnames=("config",))
def apply_horizontal_hyperdiffusion(field: jax.Array, config: HyperdiffusionConfig) -> jax.Array:
    """Applies the c2 horizontal hyperdiffusion skeleton to mass-like fields."""

    field_x = apply_hyperdiffusion_axis(field, config, axis=2)
    return apply_hyperdiffusion_axis(field_x, config, axis=1)
