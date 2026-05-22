"""c2 RK/acoustic scan orchestrator with explicit diagnostic carry."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.state import State
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig, run_acoustic_scan
from gpuwrf.dynamics.hyperdiffusion import HyperdiffusionConfig, apply_horizontal_hyperdiffusion
from gpuwrf.dynamics.limiters import LimiterConfig, positive_definite_limiter


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class FluxAccumulators:
    """Scan-carried flux/tendency accumulators required by WRF-style steps."""

    __slots__ = ("pressure", "theta")

    def __init__(self, pressure: jax.Array, theta: jax.Array) -> None:
        self.pressure = pressure
        self.theta = theta

    @classmethod
    def zeros_like(cls, state: State) -> "FluxAccumulators":
        """Creates zero accumulators with timestep-resident shapes."""

        return cls(jnp.zeros_like(state.p), jnp.zeros_like(state.theta))

    def tree_flatten(self):
        """Presents accumulators as scan-carry leaves."""

        return (self.pressure, self.theta), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds FluxAccumulators after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class DycoreScanCarry:
    """Outer timestep scan carry: state, previous pressure, and accumulators."""

    __slots__ = ("state", "previous_pressure", "fluxes")

    def __init__(self, state: State, previous_pressure: jax.Array, fluxes: FluxAccumulators) -> None:
        self.state = state
        self.previous_pressure = previous_pressure
        self.fluxes = fluxes

    def tree_flatten(self):
        """Presents scan carry leaves to JAX."""

        return (self.state, self.previous_pressure, self.fluxes), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds DycoreScanCarry after JAX transformations."""

        del aux
        state, previous_pressure, fluxes = children
        return cls(state, previous_pressure, fluxes)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Static knobs for the c2 architecture skeleton."""

    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    hyperdiffusion: HyperdiffusionConfig = field(default_factory=HyperdiffusionConfig)
    limiter: LimiterConfig = field(default_factory=LimiterConfig)


@jax.jit
def initialize_scan_carry(state: State) -> DycoreScanCarry:
    """Creates the explicit c2 scan carry from an already-resident state."""

    return DycoreScanCarry(
        state=state,
        previous_pressure=state.p + jnp.zeros_like(state.p),
        fluxes=FluxAccumulators.zeros_like(state),
    )


@partial(jax.jit, static_argnames=("config", "dt"))
def wrf_em_step(
    carry: DycoreScanCarry,
    metrics: DycoreMetrics,
    config: OrchestratorConfig,
    dt: float,
) -> DycoreScanCarry:
    """Runs one c2 architecture step with nested acoustic scan."""

    acoustic_state, previous_pressure = run_acoustic_scan(
        carry.state,
        carry.previous_pressure,
        metrics,
        config.acoustic,
        dt,
    )
    theta_diffused = apply_horizontal_hyperdiffusion(acoustic_state.theta, config.hyperdiffusion)
    mass = jnp.ones_like(theta_diffused)
    theta_limited = positive_definite_limiter(theta_diffused, mass, config.limiter)
    next_state = acoustic_state.replace(theta=theta_limited)
    next_fluxes = FluxAccumulators(
        pressure=carry.fluxes.pressure + (next_state.p - carry.state.p),
        theta=carry.fluxes.theta + (next_state.theta - carry.state.theta),
    )
    return DycoreScanCarry(next_state, previous_pressure, next_fluxes)


@partial(jax.jit, static_argnames=("config", "dt", "n_steps"))
def run_scan(
    state: State,
    metrics: DycoreMetrics,
    config: OrchestratorConfig,
    dt: float,
    n_steps: int,
) -> DycoreScanCarry:
    """Runs the outer timestep scan while carrying diagnostics explicitly."""

    def body(carry, _):
        return wrf_em_step(carry, metrics, config, dt), None

    final_carry, _ = jax.lax.scan(body, initialize_scan_carry(state), xs=None, length=int(n_steps))
    return final_carry
