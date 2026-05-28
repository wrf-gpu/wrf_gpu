"""WRF ``calc_p_rho`` perturbation pressure / inverse-density with smdiv memory.

Source: WRF ``dyn_em/module_small_step_em.F:438-568``.

``calc_p_rho`` recomputes perturbation inverse density ``al`` (hydrostatic
relation, all-dry) and the temporally-linearized perturbation pressure ``p``
(linearized equation of state) from the current small-step work state.  ``c2a``
is ``INTENT(IN)`` (computed once in ``small_step_prep``) and is never recomputed
here.  Divergence-damping pressure memory is applied per WRF lines 548-567:

* ``step == 0``: seed ``pm1 = p``.
* ``step > 0``: ``ptmp = p`` ; ``p = p + smdiv*(p - pm1)`` ; ``pm1 = ptmp``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.dynamics.core.small_step_prep import SmallStepPrepState

# WRF EM default external/internal divergence damping coefficient
# (``namelist smdiv``; WRF Registry default ``0.1``).
WRF_SMDIV_DEFAULT = 0.1


@jax.tree_util.register_pytree_node_class
class CalcPRhoStep0:
    """``calc_p_rho`` result and the divergence-damping pressure memory ``pm1``.

    The name is retained for back-compat; it now carries the pressure memory
    for every substep, not just ``step == 0``.
    """

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


def _calc_al_p(
    *,
    mu_work: jax.Array,
    mut: jax.Array,
    ph_work: jax.Array,
    theta_work: jax.Array,
    theta_1: jax.Array,
    c2a: jax.Array,
    alt: jax.Array,
    c1h: jax.Array,
    c2h: jax.Array,
    rdnw: jax.Array,
    t0: float,
) -> tuple[jax.Array, jax.Array]:
    """Return WRF non-hydrostatic ``al`` and ``p`` (``:522-528``).

    ``mu_work`` is the perturbation dry-mass work array (WRF ``mu``),
    ``mut`` the full base dry mass (WRF ``mut``), ``ph_work`` the perturbation
    geopotential work array (WRF ``ph``), ``theta_work`` the coupled theta work
    array (WRF ``t_2``), and ``theta_1`` the perturbation theta (WRF ``t_1``).
    """

    mass_h = c1h[:, None, None] * mut[None, :, :] + c2h[:, None, None]
    safe_mass = jnp.where(jnp.abs(mass_h) > 1.0e-12, mass_h, jnp.asarray(1.0e-12, dtype=mass_h.dtype))
    mu_term = c1h[:, None, None] * mu_work[None, :, :]
    # WRF :522-523 -- al = -1/(c1h*mut+c2h) * ( alt*(c1h*mu) + rdnw*(ph(k+1)-ph(k)) )
    al = -(
        alt * mu_term
        + rdnw[:, None, None] * (ph_work[1:, :, :] - ph_work[:-1, :, :])
    ) / safe_mass
    # WRF :527-528 -- p = c2a*( alt*(t_2 - c1h*mu*t_1)/((c1h*mut+c2h)*(t0+t_1)) - al )
    theta_total_ref = t0 + theta_1
    safe_theta_ref = jnp.where(
        jnp.abs(theta_total_ref) > 1.0e-6, theta_total_ref, jnp.asarray(1.0e-6, dtype=theta_total_ref.dtype)
    )
    p = c2a * (alt * (theta_work - mu_term * theta_1) / (safe_mass * safe_theta_ref) - al)
    return al, p


def calc_p_rho_wrf(
    prep: SmallStepPrepState,
    *,
    step: int = 0,
    non_hydrostatic: bool = True,
    t0: float = 300.0,
) -> CalcPRhoStep0:
    """Compute WRF ``calc_p_rho`` at loop entry (``step == 0``).

    Source: WRF ``dyn_em/module_small_step_em.F:515-567``.  Seeds ``pm1 = p``.
    The per-substep ``step > 0`` divergence-damping refresh lives in
    :func:`calc_p_rho_step`, which operates on the live acoustic work state.
    """

    if int(step) != 0:
        raise NotImplementedError("calc_p_rho_wrf only seeds step=0; use calc_p_rho_step for step>0")
    if not bool(non_hydrostatic):
        raise NotImplementedError("F7.A implements the nonhydrostatic calc_p_rho path only")

    al, p = _calc_al_p(
        mu_work=prep.mu_work,
        mut=prep.mut,
        ph_work=prep.ph_work,
        theta_work=prep.theta_work,
        theta_1=prep.theta_1,
        c2a=prep.c2a,
        alt=prep.alt,
        c1h=prep.c1h,
        c2h=prep.c2h,
        rdnw=prep.rdnw,
        t0=float(t0),
    )
    return CalcPRhoStep0(p=p, al=al, pm1=p)


def calc_p_rho_step(
    *,
    mu_work: jax.Array,
    mut: jax.Array,
    ph_work: jax.Array,
    theta_work: jax.Array,
    theta_1: jax.Array,
    c2a: jax.Array,
    alt: jax.Array,
    c1h: jax.Array,
    c2h: jax.Array,
    rdnw: jax.Array,
    pm1: jax.Array,
    smdiv: float = WRF_SMDIV_DEFAULT,
    t0: float = 300.0,
) -> CalcPRhoStep0:
    """Compute WRF ``calc_p_rho(step=iteration)`` with divergence-damping memory.

    Source: WRF ``dyn_em/module_small_step_em.F:515-567``.  ``c2a`` is
    ``INTENT(IN)`` and never recomputed.  Applies the smdiv pressure memory
    update ``p = p + smdiv*(p - pm1)`` then refreshes ``pm1`` with the
    pre-update pressure (WRF lines 557-567).
    """

    al, p = _calc_al_p(
        mu_work=mu_work,
        mut=mut,
        ph_work=ph_work,
        theta_work=theta_work,
        theta_1=theta_1,
        c2a=c2a,
        alt=alt,
        c1h=c1h,
        c2h=c2h,
        rdnw=rdnw,
        t0=float(t0),
    )
    ptmp = p
    p = p + float(smdiv) * (p - pm1)
    return CalcPRhoStep0(p=p, al=al, pm1=ptmp)


__all__ = ["CalcPRhoStep0", "calc_p_rho_wrf", "calc_p_rho_step", "WRF_SMDIV_DEFAULT"]
