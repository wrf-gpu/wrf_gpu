"""Source-anchored divergence and Rayleigh damping hooks for the c2 dycore.

WRF source anchors: ``module_small_step_em.F:548-563`` for ``smdiv`` pressure
memory and ``module_small_step_em.F:1559-1569`` for the top-layer Rayleigh
vertical-velocity ramp. MPAS source anchor:
``mpas_atm_time_integration.F:2184-2192`` for implicit Rayleigh damping on
``rw_p``.
"""

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
    """Upper-level Rayleigh damping config.

    WRF source anchor: ``module_small_step_em.F:1559-1569`` computes the
    height-dependent ``dampwt`` ramp and applies it to ``w``. MPAS source
    anchor: ``mpas_atm_time_integration.F:2184-2192`` applies the matching
    implicit gravity-wave absorbing layer to ``rw_p``.
    """

    enabled: bool = False
    coefficient: float = 0.0
    top_start_fraction: float = 0.75


@partial(jax.jit, static_argnames=("config",))
def apply_smdiv_pressure(pressure: jax.Array, previous_pressure: jax.Array, config: SmdivConfig) -> jax.Array:
    """Applies WRF-shaped ``smdiv`` pressure memory when enabled.

    WRF source anchor: ``module_small_step_em.F:548-563`` stores ``pm1`` then
    applies ``p = p + smdiv * (p - pm1)`` during small timesteps.
    """

    if not config.enabled or float(config.coefficient) == 0.0:
        return pressure
    return pressure + float(config.coefficient) * (pressure - previous_pressure)


@partial(jax.jit, static_argnames=("config",))
def apply_rayleigh_w(w: jax.Array, config: RayleighConfig) -> jax.Array:
    """Applies top-ramped Rayleigh damping to vertical velocity.

    WRF source anchor: ``module_small_step_em.F:1559-1569`` computes the
    sinusoidal top-layer ``dampwt`` profile and damps ``w`` using saved
    vertical velocity. MPAS source anchor:
    ``mpas_atm_time_integration.F:2184-2192`` applies implicit damping to
    ``rw_p`` with the ``dss`` profile.
    """

    if not config.enabled or float(config.coefficient) == 0.0:
        return w
    z = jnp.linspace(0.0, 1.0, int(w.shape[0]), dtype=w.dtype)[:, None, None]
    denom = max(1.0e-12, 1.0 - float(config.top_start_fraction))
    # WRF source anchor: module_small_step_em.F:1559-1569 bounds the top-ramp
    # to the damping layer before applying it to w.
    ramp = jnp.clip((z - float(config.top_start_fraction)) / denom, 0.0, 1.0)
    return w / (1.0 + float(config.coefficient) * ramp)


# WRF/MPAS source anchors: module_small_step_em.F:548-563,
# module_small_step_em.F:1559-1569, and mpas_atm_time_integration.F:2184-2192.
@partial(jax.jit, static_argnames=("smdiv", "rayleigh"))
def apply_damping_skeleton(
    pressure: jax.Array,
    previous_pressure: jax.Array,
    w: jax.Array,
    smdiv: SmdivConfig,
    rayleigh: RayleighConfig,
) -> tuple[jax.Array, jax.Array]:
    """Composes source-anchored WRF/MPAS damping hooks."""

    # WRF/MPAS source anchors: module_small_step_em.F:548-563,
    # module_small_step_em.F:1559-1569, and mpas_atm_time_integration.F:2184-2192.
    return apply_smdiv_pressure(pressure, previous_pressure, smdiv), apply_rayleigh_w(w, rayleigh)
