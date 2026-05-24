"""Validation-only WRF small-step scratch helpers.

These helpers mirror the scratch carry boundaries used by the M6B3 savepoint
comparator. They are not wired into the operational dycore state API.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ScratchInputs:
    """Arrays needed to reproduce WRF-shaped scratch state boundaries."""

    theta_old: jnp.ndarray
    theta_new: jnp.ndarray
    t_2ave_prev: jnp.ndarray
    ww_old: jnp.ndarray
    ww_new: jnp.ndarray
    mu_old: jnp.ndarray
    mu_new: jnp.ndarray
    mut: jnp.ndarray
    muave_prev: jnp.ndarray
    muts_prev: jnp.ndarray
    ph_tend_old: jnp.ndarray
    ph_tend_increment: jnp.ndarray
    u_current: jnp.ndarray
    v_current: jnp.ndarray
    w_current: jnp.ndarray
    ph_current: jnp.ndarray
    epssm: float


def update_t_2ave(inputs: ScratchInputs) -> jnp.ndarray:
    """Return the validation-mode theta running average."""

    _ = inputs.t_2ave_prev
    return 0.5 * (jnp.asarray(inputs.theta_old) + jnp.asarray(inputs.theta_new))


def update_ww(inputs: ScratchInputs) -> jnp.ndarray:
    """Return the WRF-shaped omega working state for the current substep."""

    _ = inputs.ww_old
    return jnp.asarray(inputs.ww_new)


def update_muave_muts(inputs: ScratchInputs) -> dict[str, jnp.ndarray]:
    """Return WRF ``MUAVE`` and ``MUTS`` scratch fields."""

    _ = (inputs.muave_prev, inputs.muts_prev)
    mu_old = jnp.asarray(inputs.mu_old)
    mu_new = jnp.asarray(inputs.mu_new)
    muave = 0.5 * ((1.0 + float(inputs.epssm)) * mu_new + (1.0 - float(inputs.epssm)) * mu_old)
    muts = jnp.asarray(inputs.mut) + mu_new
    return {"muave": muave, "muts": muts}


def accumulate_ph_tend(inputs: ScratchInputs) -> jnp.ndarray:
    """Return accumulated geopotential tendency scratch."""

    return jnp.asarray(inputs.ph_tend_old) + jnp.asarray(inputs.ph_tend_increment)


def snapshot_save_state(inputs: ScratchInputs) -> dict[str, jnp.ndarray]:
    """Snapshot the WRF ``*_save`` family used across small-step stages."""

    return {
        "u_save": jnp.asarray(inputs.u_current),
        "v_save": jnp.asarray(inputs.v_current),
        "w_save": jnp.asarray(inputs.w_current),
        "t_save": jnp.asarray(inputs.theta_new),
        "ph_save": jnp.asarray(inputs.ph_current),
        "mu_save": jnp.asarray(inputs.mu_new),
        "ww_save": jnp.asarray(inputs.ww_new),
    }


def build_scratch_state(inputs: ScratchInputs) -> dict[str, jnp.ndarray]:
    """Return all M6B3 scratch-family outputs for one acoustic substep."""

    state = {
        "t_2ave": update_t_2ave(inputs),
        "ww": update_ww(inputs),
        "ph_tend": accumulate_ph_tend(inputs),
    }
    state.update(update_muave_muts(inputs))
    state.update(snapshot_save_state(inputs))
    return state
