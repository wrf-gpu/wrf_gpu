"""JAX implementation of the M2 analytic thermo-column fixture."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp


configure_jax_x64()


@jax.jit
def column_thermo(
    temperature_initial: jax.Array,
    qv_initial: jax.Array,
    pressure_initial: jax.Array,
    saturation_qv: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run one fp64 branch-shaped thermodynamic source update."""

    excess = jnp.maximum(qv_initial - saturation_qv, 0.0)
    deficit = jnp.maximum(0.72 * saturation_qv - qv_initial, 0.0)
    condensation = 0.32 * excess
    evaporation = jnp.minimum(0.04 * deficit, 0.18 * qv_initial)
    qv_next = jnp.maximum(qv_initial - condensation + evaporation, 1.0e-8)

    cp_d = 1004.0
    lv = 2.5e6
    latent_mass = condensation - evaporation
    temperature_next = temperature_initial + (lv / cp_d) * latent_mass
    pressure_next = pressure_initial + jnp.zeros_like(pressure_initial)
    mse_delta = cp_d * (temperature_next - temperature_initial) + lv * (qv_next - qv_initial)
    return temperature_next, qv_next, pressure_next, mse_delta
