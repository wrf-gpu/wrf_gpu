"""Minimal WRF ``calc_p_rho`` loop-entry pressure preparation.

This sprint implements only ``step=0`` from WRF
``dyn_em/module_small_step_em.F:492-563``.  The per-substep
``calc_p_rho(step=iteration)`` pressure-memory update remains deferred to
F7.B with the full ``advance_w`` rewrite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.dynamics.core.small_step_prep import SmallStepPrepState


@jax.tree_util.register_pytree_node_class
class CalcPRhoStep0:
    """Step-0 ``calc_p_rho`` result and initialized pressure memory."""

    __slots__ = ("p", "al", "pm1")

    def __init__(self, p: jax.Array, al: jax.Array, pm1: jax.Array) -> None:
        self.p = p
        self.al = al
        self.pm1 = pm1

    def tree_flatten(self):
        return (self.p, self.al, self.pm1), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def calc_p_rho_wrf(
    prep: SmallStepPrepState,
    *,
    step: int = 0,
    non_hydrostatic: bool = True,
    t0: float = 300.0,
) -> CalcPRhoStep0:
    """Compute WRF ``calc_p_rho`` for loop entry only.

    Source: WRF ``dyn_em/module_small_step_em.F:522-563``.  This provides the
    pressure and inverse-density perturbation state consumed by
    ``advance_uv_wrf`` before the first acoustic substep.  Divergence-damping
    pressure history is initialized as ``pm1 = p`` for ``step == 0``.
    """

    if int(step) != 0:
        raise NotImplementedError("F7.A implements calc_p_rho_wrf(step=0) only")
    if not bool(non_hydrostatic):
        raise NotImplementedError("F7.A implements the nonhydrostatic calc_p_rho_wrf(step=0) path only")

    mass_h = prep.c1h[:, None, None] * prep.mut[None, :, :] + prep.c2h[:, None, None]
    mu_term = prep.c1h[:, None, None] * prep.mu_work[None, :, :]
    safe_mass = jnp.where(jnp.abs(mass_h) > 1.0e-12, mass_h, jnp.asarray(1.0e-12, dtype=mass_h.dtype))
    theta_total_ref = jnp.maximum(float(t0) + prep.theta_1, jnp.asarray(1.0e-6, dtype=prep.theta_1.dtype))
    al = -(
        prep.alt * mu_term
        + prep.rdnw[:, None, None] * (prep.ph_work[1:, :, :] - prep.ph_work[:-1, :, :])
    ) / safe_mass
    p = prep.c2a * (prep.alt * (prep.theta_work - mu_term * prep.theta_1) / (safe_mass * theta_total_ref) - al)
    return CalcPRhoStep0(p=p, al=al, pm1=p)


__all__ = ["CalcPRhoStep0", "calc_p_rho_wrf"]
