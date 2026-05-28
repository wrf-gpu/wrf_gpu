"""WRF hybrid-eta coefficient helpers for pressure and mass weighting."""

from __future__ import annotations

from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics


config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=())
def mass_level_pressure(mu: jax.Array, metrics: DycoreMetrics) -> jax.Array:
    """Reconstructs WRF dry pressure on mass levels from hybrid coefficients.

    WRF source anchor: ``module_big_step_utilities_em.F:1045-1047`` uses
    ``c3h/c4h`` and ``ptop`` for mass-level pressure terms.
    """

    return metrics.c3h[:, None, None] * mu[None, :, :] + metrics.c4h[:, None, None] + metrics.p_top


@partial(jax.jit, static_argnames=())
def face_level_pressure(mu: jax.Array, metrics: DycoreMetrics) -> jax.Array:
    """Reconstructs WRF dry pressure on vertical faces from hybrid coefficients."""

    return metrics.c3f[:, None, None] * mu[None, :, :] + metrics.c4f[:, None, None] + metrics.p_top


@partial(jax.jit, static_argnames=())
def mass_weight(mu: jax.Array, metrics: DycoreMetrics) -> jax.Array:
    """Returns ``c1h * mu + c2h`` on mass levels.

    WRF source anchors: ``module_small_step_em.F:522-542`` and
    ``module_advect_em.F:10374`` use this hybrid mass weight in acoustic and
    flux-form tendencies.
    """

    return metrics.c1h[:, None, None] * mu[None, :, :] + metrics.c2h[:, None, None]


@partial(jax.jit, static_argnames=())
def face_weight(mu: jax.Array, metrics: DycoreMetrics) -> jax.Array:
    """Returns ``c1f * mu + c2f`` on vertical faces."""

    return metrics.c1f[:, None, None] * mu[None, :, :] + metrics.c2f[:, None, None]


@partial(jax.jit, static_argnames=())
def pressure_thickness(mu: jax.Array, metrics: DycoreMetrics) -> jax.Array:
    """Computes positive layer pressure thickness from face pressure differences."""

    face_pressure = face_level_pressure(mu, metrics)
    return jnp.abs(face_pressure[:-1, :, :] - face_pressure[1:, :, :])


@partial(jax.jit, static_argnames=())
def hybrid_summary(mu: jax.Array, metrics: DycoreMetrics) -> jax.Array:
    """JIT-safe pressure summary used by proof scripts."""

    pressure = mass_level_pressure(mu, metrics)
    thickness = pressure_thickness(mu, metrics)
    return jnp.asarray(
        [
            jnp.min(pressure),
            jnp.max(pressure),
            jnp.min(thickness),
            jnp.max(thickness),
        ],
        dtype=jnp.float64,
    )
