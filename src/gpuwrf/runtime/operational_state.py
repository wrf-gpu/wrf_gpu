"""Operational timestep carry for promoted M6b scratch families.

The M6b real-Gen2 first-step blocker promoted the WRF small-step scratch
families from operational-undecided to operational-required. This module keeps
that production carry separate from validation savepoint modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import BaseState, State


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
    # --- v0.2.0 S6b: prognostic Noah-MP land carry threaded through the scan ---
    # ``noahmp_land`` is the prognostic NoahMPLandState (a registered pytree), or
    # ``None`` when Noah-MP is not activated (the carry pytree then has the same
    # structure as the pre-S6b carry; None is a consistent empty subtree). It
    # EVOLVES each physics step -- the standalone-replacement land fix.
    # ``noahmp_rad`` is the HELD surface radiation forcing for Noah-MP
    # (SOLDN/LWDN/COSZ as a 3-tuple of (ny,nx) device arrays), refreshed at the
    # radiation cadence and held between calls (WRF-faithful), resident on device;
    # ``None`` when Noah-MP is off.
    noahmp_land: Any = field(default=None)
    noahmp_rad: Any = field(default=None)
    # --- v0.6.0 scan-wire: persistent KF (cu=1) cumulus carry ----------------
    # ``cumulus_carry`` is the Kain-Fritsch ``(w0avg, nca)`` persistent state: the
    # running mean of vertical velocity (nz, ny, nx) feeding the KF trigger and the
    # ``nca`` cloud-relaxation countdown (ny, nx). ``None`` when no GPU-scan cumulus
    # scheme is active (the carry pytree is then structurally identical to the
    # pre-v0.6.0 carry). Appended LAST so the prefix of the existing carry leaves
    # keeps its pytree position; ``coupling.scan_adapters.initial_kf_carry`` seeds it.
    cumulus_carry: Any = field(default=None)
    # --- v0.6.0 Noah-classic operational land coupler -----------------------
    # ``noahclassic_land`` is the 4-layer Noah-classic land carry plus last land
    # flux diagnostics. ``noahclassic_rad`` is the held SOLDN/LWDN/COSZ tuple used
    # by the SFLX forcing assembler. Both are ``None`` unless an explicit
    # sf_surface_physics=2 run supplies a WRF-derived NoahClassicStatic/land bundle.
    noahclassic_land: Any = field(default=None)
    noahclassic_rad: Any = field(default=None)
    # --- v0.17 thermal-diffusion slab LSM (sf_surface_physics=1) coupler -------
    # ``slab_land`` is the 5-layer slab land carry (TSLB soil temperatures + last
    # land flux diagnostics, a SlabLandState pytree). ``slab_rad`` is the held
    # GSW/GLW down-radiation forcing the slab energy budget reads. Both are
    # ``None`` unless an explicit sf_surface_physics=1 run supplies a WRF-derived
    # SlabStaticBundle (TMN/THC/EMISS + soil ZS/DZS). Appended LAST so the prefix
    # of the existing carry leaves keeps its pytree position.
    slab_land: Any = field(default=None)
    slab_rad: Any = field(default=None)
    # --- v0.17 Pleim-Xiu (sf_surface_physics=7) 2-layer ISBA LSM coupler --------
    # ``px_land`` is the 2-layer ISBA land carry (TG/T2/WG/W2/WR + last land flux
    # diagnostics, a PleimXiuLandState pytree). ``px_rad`` is the held GSW/GLW
    # down-radiation. Both ``None`` unless an explicit sf_surface_physics=7 run
    # supplies a WRF-derived PleimXiuStaticBundle. Appended LAST.
    px_land: Any = field(default=None)
    px_rad: Any = field(default=None)
    # Explicit fp64 WRF base fields for opt-in perturbation-authoritative mixed
    # precision. Appended last so fp64_default carry prefixes stay unchanged.
    base_state: BaseState | None = field(default=None)
    # WRF radiation driver holds its flux diagnostics between calls. Seeded
    # to concrete arrays before scanning, read only by the output boundary.
    radiation_diagnostics: Any = field(default=None)
    cumulus_tendencies: Any = field(default=None)
    # Optional resident pre-repair/work telemetry; None has no device leaves.
    census: Any = field(default=None)
    # WRF ``h_diabatic`` (K/s): the previous post-RK microphysics theta_m heating
    # (moist_physics_finish_em) that rk_addtend_dry adds at every RK stage and the
    # final small_step_finish removes.  Seeded only under WRF-order microphysics;
    # None keeps the released carry structure.
    h_diabatic: Any = field(default=None)
    # WRF history surface fields from the last step's in-step physics
    # (:class:`HistoryDiagnostics`); None unless GPUWRF_HISTORY_INSTEP_DIAG=1.
    history_diagnostics: Any = field(default=None)
    # Full WRF history only: last Noah-MP outputs and per-DT J/m2 sums.
    # Optional leaves are seeded before scanning, kept on device and restarted.
    land_history: Any = field(default=None)
    energy_accumulators: Any = field(default=None)
    noahmp_precipitation: Any = field(default=None)  # previous-step WRF MP/KF rates, mm/s


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


class HistoryDiagnostics(NamedTuple):
    """WRF history surface fields produced by the last step's in-step physics.

    WRF writes history BEFORE ``solve`` (module_integrate.F:375 vs :393), so the
    surface fields of a frame at time t are the previous step's ``surface_driver``
    values on its step-start state (``PSFC = p8w(kts)``, module_surface_driver.F:1988;
    Noah-MP land T2 overwrite :3469-3473), not a re-solve on the post-step state.
    Leaves are 2-D WRF REAL, seeded once and cast explicitly on every write.
    ``hfx`` is ``None`` when ``State`` itself carries HFX.
    """

    t2: Any
    q2: Any
    u10: Any
    v10: Any
    lh: Any
    psfc: Any
    hfx: Any = None

    def write(self, values: dict[str, Any]) -> "HistoryDiagnostics":
        """Return the container with ``values`` cast to the seeded leaf dtype/shape."""

        def put(old, new):
            if old is None:
                return None
            return jnp.asarray(new, dtype=old.dtype).reshape(old.shape)

        return HistoryDiagnostics(*(put(getattr(self, name), values[name]) for name in self._fields))


def history_instep_diag_enabled() -> bool:
    """``GPUWRF_HISTORY_INSTEP_DIAG=1`` writes history surface fields from in-step physics."""

    return os.environ.get("GPUWRF_HISTORY_INSTEP_DIAG", "0").strip().lower() in {"1", "true", "yes", "on"}


def seed_history_diagnostics(state: State) -> HistoryDiagnostics:
    """Zero WRF REAL container shaped like ``state.t_skin`` (filled by the first step)."""

    zero = jnp.zeros(jnp.shape(state.t_skin), dtype=jnp.float32)
    hfx = None if "hfx" in getattr(type(state), "__slots__", ()) else zero
    return HistoryDiagnostics(t2=zero, q2=zero, u10=zero, v10=zero, lh=zero, psfc=zero, hfx=hfx)


def _base_mu(state: State, base_state: BaseState | None = None) -> jax.Array:
    """Return resident WRF ``MUB`` from explicit total/perturbation fields."""

    if base_state is not None:
        return jnp.asarray(base_state.mub)
    return jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)


def initial_operational_carry(
    state: State,
    *,
    noahmp_land: Any = None,
    noahmp_rad: Any = None,
    cumulus_carry: Any = None,
    noahclassic_land: Any = None,
    noahclassic_rad: Any = None,
    slab_land: Any = None,
    slab_rad: Any = None,
    px_land: Any = None,
    px_rad: Any = None,
    base_state: BaseState | None = None,
) -> OperationalCarry:
    """Build promoted carry from the initialized operational ``State``.

    WRF history output does not expose all small-step scratch directly, so the
    production initial condition mirrors the M6B3 savepoint extractor: ``ww``
    and ``ph_tend`` start at zero, ``muave`` starts from perturbation ``MU``,
    and ``muts`` starts from ``MUB + MU``.

    ``noahmp_land``/``noahmp_rad`` (v0.2.0 S6b) seed the prognostic Noah-MP land
    carry + held surface-radiation forcing when Noah-MP is activated; both default
    to ``None`` (Noah-MP off; the carry is structurally identical to pre-S6b).
    """

    mu_base = _base_mu(state, base_state)
    muts = jnp.asarray(mu_base, dtype=jnp.float64) + jnp.asarray(state.mu_perturbation, dtype=jnp.float64)
    ww = jnp.zeros_like(state.w, dtype=jnp.float64)
    mudf = jnp.zeros_like(state.mu_perturbation, dtype=jnp.float64)
    ph_tend = jnp.zeros_like(state.ph_perturbation, dtype=jnp.float64)
    return OperationalCarry(
        state=state,
        base_state=base_state,
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
        muave=jnp.zeros_like(state.mu_perturbation, dtype=jnp.float64),
        muts=muts,
        ph_tend=ph_tend,
        u_save=jnp.asarray(state.u),
        v_save=jnp.asarray(state.v),
        w_save=jnp.asarray(state.w),
        t_save=jnp.asarray(state.theta),
        ph_save=jnp.asarray(state.ph, dtype=jnp.float64),
        mu_save=jnp.asarray(state.mu_perturbation, dtype=jnp.float64),
        ww_save=ww,
        # Held WRF radiative theta tendency (K/s). Zero until the first radiation
        # call refreshes it; theta += dt*rthraten is applied every dynamics step.
        # Match theta dtype so force_fp64 keeps the held rate fp64.
        rthraten=jnp.zeros_like(state.theta),
        noahmp_land=noahmp_land,
        noahmp_rad=noahmp_rad,
        cumulus_carry=cumulus_carry,
        noahclassic_land=noahclassic_land,
        noahclassic_rad=noahclassic_rad,
        slab_land=slab_land,
        slab_rad=slab_rad,
        px_land=px_land,
        px_rad=px_rad,
    )


__all__ = [
    "HistoryDiagnostics",
    "OperationalCarry",
    "PROMOTED_CARRY_EVIDENCE",
    "history_instep_diag_enabled",
    "initial_operational_carry",
    "seed_history_diagnostics",
]
