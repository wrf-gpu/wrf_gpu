"""WRF-shaped acoustic scan skeleton for the c2 dycore architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.state import State
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig, apply_rayleigh_w, apply_smdiv_pressure


config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class AcousticConfig:
    """Static acoustic-substep config for c2 nested scans."""

    n_substeps: int = 1
    smdiv: SmdivConfig = field(default_factory=SmdivConfig)
    rayleigh: RayleighConfig = field(default_factory=RayleighConfig)


@partial(jax.jit, static_argnames=("config",))
def acoustic_substep(
    state: State,
    previous_pressure: jax.Array,
    metrics: DycoreMetrics,
    config: AcousticConfig,
) -> tuple[State, jax.Array]:
    """Runs one disabled-by-default WRF-shaped acoustic substep.

    This file intentionally does not import the c1 ``acoustic.py`` routines.
    The first c2 version proves scan/carry architecture and stabilizer wiring;
    later c2-A2 implementation sprints fill in WRF small-step numerics from
    ``module_small_step_em.F``.
    """

    metric_zero = jnp.sum(metrics.c1h, dtype=state.p.dtype) * jnp.asarray(0.0, dtype=state.p.dtype)
    pressure_source = state.p + metric_zero
    pressure_next = apply_smdiv_pressure(pressure_source, previous_pressure, config.smdiv)
    w_next = apply_rayleigh_w(state.w, config.rayleigh)
    next_state = state.replace(p=pressure_next, w=w_next)
    return next_state, pressure_source


@partial(jax.jit, static_argnames=("config", "dt"))
def run_acoustic_scan(
    state: State,
    previous_pressure: jax.Array,
    metrics: DycoreMetrics,
    config: AcousticConfig,
    dt: float,
) -> tuple[State, jax.Array]:
    """Runs nested acoustic substeps with previous pressure in the scan carry."""

    del dt

    def body(carry, _):
        sub_state, sub_previous_pressure = carry
        return acoustic_substep(sub_state, sub_previous_pressure, metrics, config), None

    (next_state, next_previous_pressure), _ = jax.lax.scan(
        body,
        (state, previous_pressure),
        xs=None,
        length=int(config.n_substeps),
    )
    return next_state, next_previous_pressure
