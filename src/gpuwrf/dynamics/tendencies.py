"""Small state/tendency algebra helpers shared by RK and tests."""

from __future__ import annotations

from gpuwrf.contracts.state import State, Tendencies


def add_scaled_tendencies(state: State, tendencies: Tendencies, dt: float) -> State:
    """Centralizes the eight-field Euler update used at each RK stage."""

    return state.replace(
        u=state.u + dt * tendencies.u,
        v=state.v + dt * tendencies.v,
        w=state.w + dt * tendencies.w,
        theta=state.theta + dt * tendencies.theta,
        qv=state.qv + dt * tendencies.qv,
        p=state.p + dt * tendencies.p,
        ph=state.ph + dt * tendencies.ph,
        mu=state.mu + dt * tendencies.mu,
    )
