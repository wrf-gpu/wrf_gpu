"""Native WRF-REAL MYNN surface layer (``sf_sfclay_physics=5``).

Pristine ``module_sf_mynn.F`` declares no DOUBLE: SFCLAY_mynn, its psi tables
(psi_init) and the zolrib iteration are WRF REAL. Under
``GPUWRF_SFCLAY_NATIVE_REAL=1`` the unchanged ``physics.surface_layer``
algorithm runs with float32 inputs, tables and arithmetic; outputs stay REAL
(consumers cast at their State/flux seams). Default off.
"""
from __future__ import annotations

import os

import jax.numpy as jnp

_NATIVE_REAL = os.environ.get("GPUWRF_SFCLAY_NATIVE_REAL", "0").strip().lower() in {"1", "true", "yes", "on"}


def native_real_enabled() -> bool:
    return _NATIVE_REAL


def surface_layer_with_diagnostics_real(state, *, first_timestep=False):
    from gpuwrf.physics.surface_layer import _surface_layer_impl

    return _surface_layer_impl(state, first_timestep, jnp.float32)
