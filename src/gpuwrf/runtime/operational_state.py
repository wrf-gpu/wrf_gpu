"""Operational timestep carry for promoted M6b scratch families.

The M6b real-Gen2 first-step blocker promoted the WRF small-step scratch
families from operational-undecided to operational-required. This module keeps
that production carry separate from validation savepoint modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import State


PROMOTED_CARRY_EVIDENCE = {
    "t_2ave": "M6b proof_bounds.json: all three strict-subset operational runs breached theta bounds after one 10 s step.",
    "ww": "M6b proof_bounds.json: all three runs developed extreme vertical velocity with strict-subset carry.",
    "mudf": "M6b real-IC bisection: advance_mu_t MUDF was computed but not committed with MU/theta.",
    "muave": "M6b failure-critic reviewer-report §1: mass running average absent from operational carry.",
    "muts": "M6b failure-critic reviewer-report §1: substep total mass absent from operational carry.",
    "ph_tend": "M6b failure-critic reviewer-report §1: geopotential tendency scratch absent from operational carry.",
    "_save": "M6b failure-critic reviewer-report §1: RK/acoustic transition save family absent from operational carry.",
}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OperationalCarry:
    """Production scan carry: prognostic ``State`` plus promoted WRF scratch.

    Field rationale:
    - ``t_2ave`` follows WRF ``module_small_step_em.F`` theta half-step
      averaging used by the small-step vertical recurrence.
    - ``ww``, ``muave`` and ``muts`` follow WRF ``advance_mu_t`` scratch
      updates in ``module_small_step_em.F:1066-1175``.
    - ``mudf`` follows WRF ``module_small_step_em.F:1102-1108`` where
      ``advance_mu_t`` updates the divergence-damping mass tendency in place.
    - ``ph_tend`` follows the WRF small-step geopotential tendency accumulator.
    - ``*_save`` fields keep the RK/acoustic transition state consumed across
      WRF small-step stages.
    - ``rthraten`` is the resident WRF radiative potential-temperature tendency
      (K/s, ``module_radiation_driver.F`` ``RTHRATEN``).  WRF refreshes it only
      once per ``radt`` interval and ADDS ``dt*RTHRATEN`` into theta at EVERY
      dynamics step over that interval (``phy_ra_ten`` in
      ``module_physics_addtendc.F``).  Carrying it here (instead of lumping the
      whole interval at one step) makes the radiation cadence WRF-faithful while
      keeping the held rate resident on device (no host transfer in the loop).
      Mass-grid (nz, ny, nx); starts at zero.
    """

    state: State
    t_2ave: jax.Array
    ww: jax.Array
    mudf: jax.Array
    muave: jax.Array
    muts: jax.Array
    ph_tend: jax.Array
    u_save: jax.Array
    v_save: jax.Array
    w_save: jax.Array
    t_save: jax.Array
    ph_save: jax.Array
    mu_save: jax.Array
    ww_save: jax.Array
    rthraten: jax.Array

    def replace(self, **updates) -> "OperationalCarry":
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def _base_mu(state: State) -> jax.Array:
    """Return resident WRF ``MUB`` from explicit total/perturbation fields."""

    return jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)


def initial_operational_carry(state: State) -> OperationalCarry:
    """Build promoted carry from the initialized operational ``State``.

    WRF history output does not expose all small-step scratch directly, so the
    production initial condition mirrors the M6B3 savepoint extractor: ``ww``
    and ``ph_tend`` start at zero, ``muave`` starts from perturbation ``MU``,
    and ``muts`` starts from ``MUB + MU``.
    """

    mu_base = _base_mu(state)
    muts = mu_base + state.mu_perturbation
    ww = jnp.zeros_like(state.w)
    mudf = jnp.zeros_like(state.mu_perturbation)
    ph_tend = jnp.zeros_like(state.ph, dtype=jnp.float64)
    return OperationalCarry(
        state=state,
        # F7G: ``t_2ave`` is the WRF small-step WORK-theta running average
        # (module_small_step_em.F:1341-1344), NOT the full initialized theta.  At a
        # fresh RK stage on a fixed-mass rest thermal the coupled work theta ``t_2``
        # is zero, so ``t_2ave`` must start at ZERO; seeding it with the full
        # initialized theta double-counts the thermal as a spurious advance_w term-B
        # buoyancy source (gpt-council-findings.md §3.5).
        t_2ave=jnp.zeros_like(state.theta, dtype=jnp.float64),
        ww=ww,
        mudf=mudf,
        # F7G: ``muave`` is the small-step mass-WORK running average
        # (module_small_step_em.F:1102-1108).  For a fixed-mass thermal with mu'=0
        # and no mass tendency it is zero; it becomes nonzero only from actual
        # small-step mass evolution, so seed it at zero rather than the full mu'.
        muave=jnp.zeros_like(state.mu_perturbation),
        muts=muts,
        ph_tend=ph_tend,
        u_save=jnp.asarray(state.u),
        v_save=jnp.asarray(state.v),
        w_save=jnp.asarray(state.w),
        t_save=jnp.asarray(state.theta),
        ph_save=jnp.asarray(state.ph),
        mu_save=jnp.asarray(state.mu_perturbation),
        ww_save=ww,
        # Held WRF radiative theta tendency (K/s). Zero until the first radiation
        # call refreshes it; theta += dt*rthraten is applied every dynamics step.
        # Match theta dtype so force_fp64 keeps the held rate fp64.
        rthraten=jnp.zeros_like(state.theta),
    )


__all__ = ["OperationalCarry", "PROMOTED_CARRY_EVIDENCE", "initial_operational_carry"]
