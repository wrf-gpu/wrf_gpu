"""WRF-shaped small-step finish for the operational RK acoustic path.

The source routine is WRF ``dyn_em/module_small_step_em.F:364-430``.
It reconstructs physical perturbation prognostics from coupled small-step work
arrays and restores the saved mass, geopotential, and omega families.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import State
from gpuwrf.dynamics.core.small_step_prep import SmallStepPrepState


def _safe_denominator(value: jax.Array) -> jax.Array:
    floor = jnp.asarray(1.0e-12, dtype=value.dtype)
    return jnp.where(jnp.abs(value) > floor, value, jnp.where(value >= 0.0, floor, -floor))


def small_step_finish_wrf(prep: SmallStepPrepState, acoustic_out: object) -> State:
    """Return the post-acoustic physical state for one RK stage.

    Source: WRF ``dyn_em/module_small_step_em.F:364-430``.  The current mass
    kernel returns the physical perturbation ``mu`` while carrying the WRF work
    delta through ``muts - mut``; therefore the final dry-mass perturbation is
    read directly from ``acoustic_out.mu`` instead of adding ``mu_save`` again.
    """

    state = prep.entry_state
    u_work = jnp.asarray(getattr(acoustic_out, "u"))
    v_work = jnp.asarray(getattr(acoustic_out, "v"))
    w_work = jnp.asarray(getattr(acoustic_out, "w"))
    theta_work_attr = getattr(acoustic_out, "theta_coupled_work", None)
    theta_work = jnp.asarray(theta_work_attr if theta_work_attr is not None else getattr(acoustic_out, "theta"))
    ph_work = jnp.asarray(getattr(acoustic_out, "ph"))
    mu_perturbation = jnp.asarray(getattr(acoustic_out, "mu"))
    p_perturbation = jnp.asarray(getattr(acoustic_out, "p"))
    muts = jnp.asarray(getattr(acoustic_out, "muts"))

    mass_u_stage = prep.c1h[:, None, None] * prep.muus[None, :, :] + prep.c2h[:, None, None]
    mass_u_current = prep.c1h[:, None, None] * prep.muu[None, :, :] + prep.c2h[:, None, None]
    mass_v_stage = prep.c1h[:, None, None] * prep.muvs[None, :, :] + prep.c2h[:, None, None]
    mass_v_current = prep.c1h[:, None, None] * prep.muv[None, :, :] + prep.c2h[:, None, None]
    mass_w_stage = prep.c1f[:, None, None] * muts[None, :, :] + prep.c2f[:, None, None]
    mass_w_current = prep.c1f[:, None, None] * prep.mut[None, :, :] + prep.c2f[:, None, None]
    mass_theta_stage = prep.c1h[:, None, None] * muts[None, :, :] + prep.c2h[:, None, None]
    mass_theta_current = prep.c1h[:, None, None] * prep.mut[None, :, :] + prep.c2h[:, None, None]

    u = (prep.msfuy[None, :, :] * u_work + prep.u_save * mass_u_current) / _safe_denominator(mass_u_stage)
    v = (prep.msfvx[None, :, :] * v_work + prep.v_save * mass_v_current) / _safe_denominator(mass_v_stage)
    w = (prep.msfty[None, :, :] * w_work + prep.w_save * mass_w_current) / _safe_denominator(mass_w_stage)
    theta_perturbation = (theta_work + prep.t_save * mass_theta_current) / _safe_denominator(mass_theta_stage)
    theta = theta_perturbation + prep.theta_offset
    ph_perturbation = ph_work + prep.ph_save
    ww = jnp.asarray(getattr(acoustic_out, "ww")) + prep.ww_save
    del ww

    # v0.20 fp32 INTEGRATION bit-identity fix: choose the base-field source by
    # storage mode so fp64_default stays BYTE-IDENTICAL to the pre-S4 baseline
    # while the perturbation-authoritative fp32 mode stays cancellation-safe.
    #
    # The S4 merge unconditionally switched p_base/ph_base/mu_base from the
    # historical `state.p_total - state.p_perturbation` reconstruction to the
    # pristine `prep.pb/phb/mub`. Those are mathematically equal but NOT
    # floating-point identical, so the always-on switch broke fp64_default
    # bit-identity (GPU all-7 byte-compare: PB maxΔ~1.6e-2 Pa, P/PH/U/V at fp64
    # round-off). For the mixed_perturb_fp32 mode the pristine fp64 base
    # (`prep.pb`) is REQUIRED -- there `p_perturbation` is stored fp32 and
    # `state.p_total - state.p_perturbation` would be a mixed-dtype subtraction
    # that re-introduces the very fp32 cancellation the perturbation-authoritative
    # design avoids. Gate on the perturbation storage dtype (a compile-time
    # static property -> zero runtime cost; fp64_default re-emits the exact prior
    # HLO -> bit-identical; mixed keeps the pristine base).
    if jnp.dtype(jnp.asarray(state.p_perturbation).dtype) == jnp.dtype(jnp.float32):
        # perturbation-authoritative fp32 storage: use the pristine fp64 base.
        p_base = prep.pb
        ph_base = prep.phb
        mu_base = prep.mub
    else:
        # fp64_default (and every fp64-stored path): historical reconstruction,
        # byte-identical to the pre-S4 baseline.
        p_base = state.p_total - state.p_perturbation
        ph_base = state.ph_total - state.ph_perturbation
        mu_base = state.mu_total - state.mu_perturbation
    return state.replace(
        u=u,
        v=v,
        w=w,
        theta=theta,
        p=p_base + p_perturbation,
        p_total=p_base + p_perturbation,
        p_perturbation=p_perturbation,
        ph=ph_base + ph_perturbation,
        ph_total=ph_base + ph_perturbation,
        ph_perturbation=ph_perturbation,
        mu=mu_base + mu_perturbation,
        mu_total=mu_base + mu_perturbation,
        mu_perturbation=mu_perturbation,
    )


__all__ = ["small_step_finish_wrf"]
