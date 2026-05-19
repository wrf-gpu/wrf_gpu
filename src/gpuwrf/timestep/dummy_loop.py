"""M3 dummy timestep loop proving State/Tendencies scan residency."""

from __future__ import annotations

from functools import partial

import jax
from jax import config

from gpuwrf.contracts.state import State, Tendencies


config.update("jax_enable_x64", True)


def dummy_step(state: State, tendencies: Tendencies, dt: float) -> tuple[State, Tendencies]:
    """Exercises one fused hot-path update over preallocated leaves without allocations."""

    theta_work = state.theta + dt * tendencies.theta
    theta_next = theta_work - dt * tendencies.theta
    return state.replace(theta=theta_next), tendencies


@partial(jax.jit, static_argnames=("n_steps",))
def run_dummy_loop(state: State, tendencies: Tendencies, dt: float, n_steps: int) -> tuple[State, Tendencies]:
    """Runs the whole dummy integration as one JITed scan call."""

    def body(carry, _):
        """Keeps the scan body inline-sized while carrying State and Tendencies together."""

        step_state, step_tendencies = carry
        return dummy_step(step_state, step_tendencies, dt), None

    (next_state, next_tendencies), _ = jax.lax.scan(body, (state, tendencies), xs=None, length=n_steps)
    return next_state, next_tendencies
