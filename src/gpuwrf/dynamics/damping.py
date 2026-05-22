"""Divergence and Rayleigh damping skeletons for the c2 WRF dycore."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SmdivConfig:
    """Small-step divergence damping config.

    WRF source anchor: ``module_small_step_em.F:562`` applies
    ``p = p + smdiv * (p - pm1)``.
    """

    enabled: bool = False
    coefficient: float = 0.0


@dataclass(frozen=True)
class RayleighConfig:
    """Upper-level Rayleigh/sponge damping config."""

    enabled: bool = False
    coefficient: float = 0.0
    top_start_fraction: float = 0.75


@partial(jax.jit, static_argnames=("config",))
def apply_smdiv_pressure(pressure: jax.Array, previous_pressure: jax.Array, config: SmdivConfig) -> jax.Array:
    """Applies WRF-shaped smdiv pressure memory when enabled."""

    if not config.enabled or float(config.coefficient) == 0.0:
        return pressure
    return pressure + float(config.coefficient) * (pressure - previous_pressure)


@partial(jax.jit, static_argnames=("config",))
def apply_rayleigh_w(w: jax.Array, config: RayleighConfig) -> jax.Array:
    """Applies a top-ramped Rayleigh damping skeleton to vertical velocity.

    WRF source anchors: ``module_big_step_utilities_em.F:6120`` defines
    ``rk_rayleigh_damp`` and lines ``6256``, ``6311``, ``6333`` apply
    mass-weighted damping tendencies to winds.
    """

    if not config.enabled or float(config.coefficient) == 0.0:
        return w
    z = jnp.linspace(0.0, 1.0, int(w.shape[0]), dtype=w.dtype)[:, None, None]
    denom = max(1.0e-12, 1.0 - float(config.top_start_fraction))
    ramp = jnp.clip((z - float(config.top_start_fraction)) / denom, 0.0, 1.0)
    return w / (1.0 + float(config.coefficient) * ramp)


@partial(jax.jit, static_argnames=("smdiv", "rayleigh"))
def apply_damping_skeleton(
    pressure: jax.Array,
    previous_pressure: jax.Array,
    w: jax.Array,
    smdiv: SmdivConfig,
    rayleigh: RayleighConfig,
) -> tuple[jax.Array, jax.Array]:
    """Composes the disabled-by-default damping skeletons."""

    return apply_smdiv_pressure(pressure, previous_pressure, smdiv), apply_rayleigh_w(w, rayleigh)
