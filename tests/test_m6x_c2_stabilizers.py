from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig, apply_rayleigh_w, apply_smdiv_pressure
from gpuwrf.dynamics.hyperdiffusion import HyperdiffusionConfig, apply_horizontal_hyperdiffusion
from gpuwrf.dynamics.limiters import LimiterConfig, limiter_diagnostics, positive_definite_limiter


def test_smdiv_identity_when_disabled_and_effect_when_enabled():
    pressure = jnp.ones((2, 3, 4), dtype=jnp.float64)
    previous = pressure - 0.5

    disabled = apply_smdiv_pressure(pressure, previous, SmdivConfig())
    enabled = apply_smdiv_pressure(pressure, previous, SmdivConfig(enabled=True, coefficient=0.1))

    assert jnp.allclose(disabled, pressure)
    assert jnp.all(jnp.isfinite(enabled))
    assert not jnp.allclose(enabled, pressure)


def test_rayleigh_identity_when_disabled_and_effect_when_enabled():
    w = jnp.ones((6, 3, 4), dtype=jnp.float64)

    disabled = apply_rayleigh_w(w, RayleighConfig())
    enabled = apply_rayleigh_w(w, RayleighConfig(enabled=True, coefficient=0.2, top_start_fraction=0.5))

    assert jnp.allclose(disabled, w)
    assert jnp.all(jnp.isfinite(enabled))
    assert not jnp.allclose(enabled[-1], w[-1])


def test_hyperdiffusion_identity_when_disabled_and_effect_when_enabled():
    field = jnp.zeros((2, 8, 8), dtype=jnp.float64).at[:, 4, 4].set(1.0)

    disabled = apply_horizontal_hyperdiffusion(field, HyperdiffusionConfig())
    enabled = apply_horizontal_hyperdiffusion(field, HyperdiffusionConfig(enabled=True, coefficient=0.01))

    assert jnp.allclose(disabled, field)
    assert jnp.all(jnp.isfinite(enabled))
    assert not jnp.allclose(enabled, field)


def test_positive_definite_limiter_preserves_mass_for_positive_total():
    scalar = jnp.asarray([[[2.0, -0.5, 1.0]]], dtype=jnp.float64)
    mass = jnp.ones_like(scalar)

    disabled = positive_definite_limiter(scalar, mass, LimiterConfig())
    enabled = positive_definite_limiter(scalar, mass, LimiterConfig(enabled=True))
    diagnostics = limiter_diagnostics(scalar, enabled, mass)

    assert jnp.allclose(disabled, scalar)
    assert float(jnp.min(enabled)) >= 0.0
    assert float(diagnostics[3]) < 1.0e-12
