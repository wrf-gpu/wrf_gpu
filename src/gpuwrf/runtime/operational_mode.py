"""GPU-resident operational forecast loop for M6 perf-design.

This module is deliberately separate from the M6B validation savepoint ladder.
It runs timestep/RK/acoustic loops inside one JAX entry point and leaves debug
snapshots/sanitizers out of the compiled path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.contracts.state import BaseState, State, Tendencies
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    DEFAULT_BOUNDARY_CONFIG,
    apply_lateral_boundaries,
    interpolate_boundary_leaf,
    normal_bdy_work_target_u,
    normal_bdy_work_target_v,
    nested_ph_relax_tendency,
    nested_w_relax_tendency,
    _full_ring_target_from_leaf,
)
from gpuwrf.coupling.physics_couplers import (
    dudhia_sw_theta_tendency,
    gsfc_sw_theta_tendency,
    gwdo_adapter,
    mynn_adapter,
    rrtm_lw_theta_tendency,
    rrtmg_lw_theta_tendency,
    rrtmg_radiation_diagnostics,
    rrtmg_sw_theta_tendency,
    rrtmg_theta_tendency,
    surface_adapter,
    surface_layer_diagnostics,
    thompson_adapter,
)
from gpuwrf.coupling.noahmp_surface_hook import (
    noahmp_surface_step,
    overlay_noahmp_land_diagnostics,
)
from gpuwrf.coupling.noahclassic_surface_hook import (
    NoahClassicRadiation,
    noahclassic_surface_step,
    overlay_noahclassic_land_diagnostics,
)
from gpuwrf.coupling.physics_dispatch import (
    DEFAULT_BL_PBL_PHYSICS,
    DEFAULT_CU_PHYSICS,
    DEFAULT_MP_PHYSICS,
    DEFAULT_SF_SFCLAY_PHYSICS,
    UnsupportedSchemeSelection,
    resolve_physics_suite,
)
from gpuwrf.coupling.scan_adapters import (
    CU_SCAN_ADAPTERS,
    CU_STATELESS_SCAN_ADAPTERS,
    MP_SCAN_ADAPTERS,
    PBL_SCAN_ADAPTERS,
    SFCLAY_SCAN_ADAPTERS,
    bmj_adapter,
    initial_bmj_carry,
    initial_kf_carry,
    kf_adapter,
)
from gpuwrf.physics.myj_adapters import (
    janjic_sfclay_adapter,
    myj_pbl_adapter,
)
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.explicit_diffusion import (
    C_S_DEFAULT,
    constant_k_diffusion_tendency,
    conservative_constant_k_diffusion_tendency,
    horizontal_deformation_2d,
    horizontal_diffusion_coord_momentum_tendency,
    horizontal_diffusion_coord_scalar_tendency,
    sixth_order_diffusion_tendency,
    smag2d_horizontal_km,
    wrf_deformation_momentum_tendency,
)
from gpuwrf.dynamics.flux_advection import (
    advect_moisture_scalars,
    advect_scalar_flux,
    advect_scalar_flux_limited,
    advect_u_flux,
    advect_v_flux,
    advect_w_flux,
    couple_velocities_periodic,
)
from gpuwrf.dynamics.acoustic_wrf import (
    CPOVCV,
    _inverse_density_from_theta_pressure,
    calc_coef_w_wrf_coefficients,
    diagnose_pressure_al_alt,
    horizontal_pressure_gradient,
    moisture_coupling_factors,
)
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState, acoustic_substep_core
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, dry_cqw, pg_buoy_w_dry
from gpuwrf.dynamics.core.calc_p_rho import CalcPRhoStep0, calc_p_rho_wrf
from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf
from gpuwrf.dynamics.core.coupled import CoupledCoreConfig, coupled_timestep_core
from gpuwrf.dynamics.core.rk_addtend_dry import (
    DryPhysicsTendencies,
    large_step_coriolis,
    large_step_horizontal_pgf,
    rk_addtend_dry,
)
from gpuwrf.dynamics.core.small_step_finish import small_step_finish_wrf
from gpuwrf.dynamics.core.small_step_prep import SmallStepPrepState, small_step_prep_wrf
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)

_THETA_LIMITER_MIN_K = 0.0
_THETA_LIMITER_MAX_K = 500.0


def _acoustic_unroll() -> int:
    """Acoustic-substep ``lax.scan`` unroll factor (v0.10.0 Wave-A, Opus#1).

    Mirrors the Thompson ``GPUWRF_THOMPSON_SED_UNROLL`` pattern.  Default ``1``
    remains the operational default: Wave-A/B A/B found unroll=2 only moved the
    coupled L2 path by noise-scale <1% while increasing compile cost.  Unrolling
    is round-off-neutral (the substep arithmetic is unchanged; only the loop body
    is replicated in the program), so the hook remains available for future
    dycore-only or architecture-specific measurements.
    """

    return max(1, int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1")))


# The acoustic substep MUTATES only these AcousticCoreState leaves
# (``acoustic_substep_core`` final ``replace`` + ``advance_uv_wrf`` u/v); every
# other leaf is STAGE-CONSTANT.  Threading only these through the substep
# ``lax.scan`` carry (closing over the constants) removes the per-substep
# carry-copy of the ~50 stage-constant leaves -- round-off-neutral (Opus#2).
_ACOUSTIC_EVOLVING_FIELDS: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "mu",
    "mudf",
    "muts",
    "muave",
    "ww",
    "theta",
    "theta_coupled_work",
    "theta_ave",
    "ph",
    "p",
    "al",
    "pm1",
    "t_2ave",
    "ru_m",
    "rv_m",
    "ww_m",
)


class _StaticHolder:
    """Identity-hashable wrapper so a Noah-MP static bundle (categories + constant
    parameter tables + pre-built params) can ride in the namelist's STATIC AUX.

    The frozen Noah-MP driver concretizes several integer/scalar fields of these
    bundles (``isurban``, ``nroot``, table scalars) inside the jitted scan, so they
    must be COMPILE CONSTANTS, not tracers. Carrying them as static aux makes JAX
    bake their (per-run-constant) arrays into the program. ``None`` is held as-is.
    Hash/eq are by object identity for real bundles: one run builds one static
    bundle -> one compile; a different run's bundle is a distinct object -> a
    fresh compile (correct). ``None`` uses its own stable hash so repeated
    namelist flattening does not fragment the JIT cache for disabled bundles."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(None) if self.value is None else id(self.value)

    def __eq__(self, other):
        return isinstance(other, _StaticHolder) and self.value is other.value


_PHYSICS_NON_DRY_INCREMENT_FIELDS: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "Nc",
    "Nn",
    "qke",
)


_PHYSICS_NON_DRY_REPLACE_FIELDS: tuple[str, ...] = (
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "rhosfc",
    "fltv",
    "t_skin",
    "soil_moisture",
    "xland",
    "lakemask",
    "mavail",
    "roughness_m",
    "lu_index",
    "rain_acc",
    "rainc_acc",
    "snow_acc",
    "graupel_acc",
    "ice_acc",
)

_SHARDED_CARRY_HALO_CONTEXT: tuple[object, int] | None = None


class _PhysicsStepForcing(NamedTuple):
    """Physics output split into WRF RK dry tendencies and non-dry state writes."""

    state: State
    carry: OperationalCarry
    dry_tendencies: DryPhysicsTendencies
    enabled: bool


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OperationalNamelist:
    """Static runtime controls plus resident metric/tendency leaves.

    ``grid`` and scalar controls are static cache keys. ``tendencies`` and
    ``metrics`` are device leaves so no host/device transfer is needed in the
    timestep loop. M6b promotes WRF small-step scratch fields into the resident
    production carry; see ``runtime.operational_state`` for the evidence table.
    """

    grid: GridSpec
    tendencies: Tendencies
    metrics: DycoreMetrics
    dt_s: float = 10.0
    acoustic_substeps: int = 10
    rk_order: int = 3
    epssm: float = 0.1
    top_lid: bool = False
    run_physics: bool = True
    run_boundary: bool = True
    radiation_cadence_steps: int = 60
    boundary_config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG
    use_vertical_solver: bool = True
    disable_guards: bool = False
    # WRF damping (Gen2 d02 namelist: w_damping=1, damp_opt=3, zdamp=5000, dampcoef=0.2).
    # Defaults OFF so the bare acoustic core (Sprint A) behaviour is unchanged unless
    # the caller explicitly enables WRF damping for the operational dt.
    w_damping: int = 0
    damp_opt: int = 0
    dampcoef: float = 0.0
    zdamp: float = 5000.0
    diff_opt: int = 0
    km_opt: int = 0
    khdif: float = 0.0
    kvdif: float = 0.0
    # Smagorinsky coefficient c_s (Registry default 0.25), used by the
    # diff_opt=1/km_opt=4 2-D Smagorinsky horizontal-diffusion path (WRF smag2d_km).
    c_s: float = C_S_DEFAULT
    diff_6th_opt: int = 0
    diff_6th_factor: float = 0.12
    # Constant eddy viscosity (Straka ν=75) on u, v, theta when > 0.
    const_nu_m2_s: float = 0.0
    # Use WRF flux-form mass-coupled scalar advection (Block 2) for theta.
    use_flux_advection: bool = False
    # WRF scalar advection limiter option for the flux-form theta path (canonical
    # ``moist_adv_opt``/``scalar_adv_opt``): 0 = plain h5/v3 (the bit-for-bit
    # DEFAULT), 1 = positive-definite, 2 = monotonic (module_advect_em.F
    # advect_scalar_pd :6069 / advect_scalar_mono :9495).  WRF applies the limiter
    # ONLY on the final RK3 stage (module_em.F:1265 ``rk_step == rk_order``) using
    # the start-of-step scalar/mass; for 0 (or while ``use_flux_advection`` is off)
    # the plain ``advect_scalar_flux`` path is byte-unchanged.  3/4 (WENO/WENO-PD)
    # are out of scope and fail-closed in the scheme catalog.
    scalar_adv_opt: int = 0
    # WRF moisture-species flux-form advection option (the moisture analogue of
    # ``scalar_adv_opt``; canonical ``moist_adv_opt``): 0 = OFF -> moisture
    # (qv + every condensate qc/qr/qi/qs/qg) is NOT resolved-wind advected in the
    # dycore (the byte-for-byte v0.12.0 operational program; the new code path is
    # never traced), 1 = positive-definite, 2 = monotonic (WRF real-case default).
    # When non-zero AND ``use_flux_advection`` is set, every moisture species is
    # flux-advected in the RK3 LARGE step exactly as WRF
    # (``solve_em.F:2282-2408`` ``moist_variable_loop`` -> ``rk_scalar_tend(...,
    # config_flags%moist_adv_opt)``): the coupled tendency d(mu*q)/dt is built per
    # RK stage from the stage-entry haloed state with ``advect_moisture_scalars``
    # and integrated with the WRF scalar large-step update
    # ``q_new = (mu_old*q_old + dt_rk*adv_tend)/mu_new`` AFTER the acoustic loop
    # (NOT inside the acoustic substeps -- WRF advances scalars in the large step).
    # The PD/monotonic limiter is applied ONLY on the final RK3 stage (matching the
    # theta wiring); other stages and opt==0 use the plain h5/v3 path.  Default 0
    # keeps the operational forecast bit-for-bit unchanged.
    moist_adv_opt: int = 0
    # Force pure fp64 (Sprint F7-B is fp64-correctness-only; idealized cases set it).
    force_fp64: bool = False
    # Use the WRF deformation-tensor momentum diffusion (diff_opt=2/km_opt=1) for
    # u/v/w instead of the scalar flux-divergence Laplacian.  Theta always keeps
    # the conservative scalar flux-divergence (WRF horizontal_diffusion_s).  Only
    # active when const_nu_m2_s > 0.  Sprint U (P0-2).
    use_deformation_momentum_diffusion: bool = False
    # Model-init UTC instant (recomp B3 hook). Static aux (datetime / ISO string /
    # None). When set, the RRTMG radiation adapter is driven by the actual forecast
    # clock (time_utc + lead_seconds) inside the scan, so the diurnal SW cycle
    # evolves over the run; None keeps the adapter's legacy fixed-time behaviour.
    time_utc: object = None
    # RRTMG terrain-radiation static fields (real XLAT/XLONG, terrain-derived
    # slope/aspect, map rotation). Kept as a pytree child so large arrays remain
    # device leaves rather than static cache-key payload.
    radiation_static: object = None
    topo_shading: int = 0
    slope_rad: int = 0
    topo_shadow_length_m: float = 25000.0
    # --- v0.2.0 S6b: prognostic Noah-MP land activation ---------------------
    # ``use_noahmp`` (static aux) flips the LAND surface tile from the prescribed
    # bulk path (coupling.physics_couplers.surface_adapter) to the prognostic
    # Noah-MP coupler (coupling.noahmp_surface_hook.noahmp_surface_step). Ocean /
    # water columns keep the bulk path byte-unchanged. Default OFF -- a run opts in
    # by building the namelist with use_noahmp=True + a NoahMPStatic. When ON, the
    # carry MUST carry a prognostic ``noahmp_land`` (initial_operational_carry seeds
    # it). ``noahmp_static`` is the per-run read-only Noah-MP static (categories /
    # tables / soil geometry); it rides as a pytree CHILD so its device arrays do
    # not pollute the static jit cache key.
    use_noahmp: bool = False
    noahmp_static: object = None
    # Pre-built per-run Noah-MP energy/radiation parameter bundles (pytree CHILDREN
    # of array fields). They are gathered ONCE outside jit so the driver's
    # ``build_energy_params`` (which concretizes ``nroot`` via int(round(...)) and
    # is FROZEN) is never re-run INSIDE the scan with traced static. ``noahmp_nroot``
    # is the concrete static root-depth slice bound (static aux) reattached to the
    # energy params inside the hook (the energy kernel uses ``range(nroot)``).
    noahmp_energy_params: object = None
    noahmp_rad_params: object = None
    noahmp_nroot: int = 0
    # phenology clock scalars (static aux): Julian day + year length for the
    # seasonal greenness term; default to WRF's day-1 / 365 when unset.
    noahmp_julian: float = 1.0
    noahmp_yearlen: float = 365.0
    # --- v0.6.0 physics-suite selection (static aux) ------------------------
    # Frozen S0 accept-matrix options dispatched by coupling.physics_dispatch.
    # Defaults are the v0.2.0 validated baseline (Thompson / MYNN / MYNN-sfclay /
    # Noah-MP, no cumulus) so an existing namelist resolves byte-for-byte to the
    # current operational path. ``sf_surface_physics`` defaults to None so the
    # legacy ``use_noahmp`` toggle still drives Noah-MP vs the bulk surface path
    # (the dispatcher maps use_noahmp True->4 / False->2); set it explicitly to
    # pin a land scheme. Selecting any non-default scheme that is not yet threaded
    # into the operational scan adapter FAILS CLOSED in _resolve_operational_suite.
    mp_physics: int = 8
    bl_pbl_physics: int = 5
    sf_sfclay_physics: int = 5
    cu_physics: int = 0
    sf_surface_physics: object = None
    # --- radiation-family selection (static aux) ----------------------------
    # ``ra_sw_physics`` selects the SHORTWAVE radiation scheme on the operational
    # scan: 0 = disabled, 4 = RRTMG SW (default, byte-unchanged), 1 = Dudhia
    # (Stephens-1984 broadband), 2 = GSFC/Chou-Suarez. WRF runs SW and LW drivers
    # independently, so a disabled SW component contributes exactly zero heating.
    # The held-rate cadence + topo-shading/slope-rad statics are shared with RRTMG.
    # NOTE: for nonzero alternate SW schemes, the surface SWDOWN/flux history
    # diagnostics remain RRTMG-derived (rrtmg_radiation_diagnostics); ra_sw=1/2
    # change the PROGNOSTIC SW heating (RTHRATEN added to theta), not the diagnostic
    # surface-flux output fields. ra_sw=0 zeros the SW diagnostics.
    ra_sw_physics: int = 4
    # ``ra_lw_physics`` selects the LONGWAVE radiation scheme: 0 = disabled, 4 = RRTMG LW
    # (default, byte-unchanged), 1 = classic AER RRTM LW (16-band k-distribution,
    # coupling.physics_couplers.rrtm_lw_theta_tendency, JAX-traceable port of
    # phys/module_ra_rrtm.F). ra_lw=1 changes the PROGNOSTIC LW heating only; the
    # surface GLW history diagnostic remains RRTMG-derived for nonzero LW schemes.
    # ra_lw=0 zeros the LW diagnostics. SW and LW are selected independently (WRF
    # runs the two drivers separately), so any operationally wired pair is valid and
    # disabled components are true no-ops.
    ra_lw_physics: int = 4
    # Explicit Noah-classic (sf_surface_physics=2) operational inputs. The JAX SFLX
    # kernel consumes WRF-derived REDPRM/static fields and a 4-layer land carry; if
    # either is absent, the scan rejects sf_surface_physics=2 rather than deriving
    # an unvalidated land state from 2-D State.soil_moisture.
    noahclassic_static: object = None
    noahclassic_land: object = None
    noahclassic_rad: object = None
    # --- orographic gravity-wave drag (gwd_opt=1) ---------------------------
    # ``gwd_opt`` (static aux) selects the WRF GWDO scheme: 0 = off (default,
    # byte-unchanged), 1 = orographic GWD + flow blocking (faithful bl_gwdo_run
    # port, physics/gwd_gwdo.py + coupling.physics_couplers.gwdo_adapter).
    # ``gwdo_statics`` is the per-run :class:`GWDOStatics` sub-grid orography
    # bundle (VAR/CON/OA1-4/OL1-4 from wrfinput; built by
    # ``build_gwdo_statics_from_wrf_fields``). It rides as a pytree CHILD so its
    # device arrays stay leaves, mirroring ``radiation_static``. The dispatch is
    # a no-op when ``gwd_opt != 1`` OR ``gwdo_statics`` is None.
    gwd_opt: int = 0
    gwdo_statics: object = None
    # --- v0.13 skill-closure #1: WRF-faithful radiation *_tendf (RTHRATEN) cadence ---
    # ``rad_rk_tendf`` (static aux) selects how the held radiative heating rate
    # RTHRATEN (K/s) is delivered to theta: 0 = the v0.9 SHIPPED single Euler step
    # ``theta += dt*RTHRATEN`` applied BEFORE the dycore (default, byte-for-byte
    # unchanged); 1 = the WRF-faithful per-RK/per-acoustic-substep cadence, routing
    # the SAME held rate through the ``t_tendf`` (mass-coupled) channel of
    # ``rk_addtend_dry`` so RTHRATEN is integrated by ``advance_mu_t`` at EVERY
    # acoustic substep interleaved with the dynamics, exactly as WRF
    # (``module_first_rk_step_part2.F:392-394`` feeds ``t_tendf``; ``rk_addtend_dry``
    # folds ``t_tendf/msfty`` into the theta tendency; ``advance_mu_t`` applies
    # ``theta += msfty*dts*theta_tend`` each substep -- the msfty cancels and the
    # mass-coupled rate decouples to ``dts*RTHRATEN`` per substep).  The coupler doc
    # (physics_couplers.rrtmg_theta_tendency :1660-1665) states the lumped one-step
    # form is NOT WRF-equivalent because the intervening dynamics/MP/PBL see a
    # different temperature trajectory.  This routes ONLY the genuine instantaneous
    # radiation rate (an explicit WRF R*TEN source) -- NOT the aggregate physics
    # state delta that the reverted v0.11 bridge (``_dry_physics_tendencies_from_
    # state_delta``) wrongly treated as a source and that regressed the d02 winds
    # (proofs/v0110/wind_regression_debug.md: the THETA/h_diabatic aggregate was the
    # culprit, momentum tendf was neutral).  The implicit-solve PBL/surface/MP
    # deltas stay on the post-dycore state-increment path (faithful for an implicit
    # scheme).  Wind-skill impact is GPU-measured (manager); default 0 keeps the
    # operational forecast bit-for-bit unchanged.
    rad_rk_tendf: int = 0

    @classmethod
    def from_grid(
        cls,
        grid: GridSpec,
        *,
        tendencies: Tendencies | None = None,
        metrics: DycoreMetrics | None = None,
        dt_s: float = 10.0,
        acoustic_substeps: int = 10,
        radiation_cadence_steps: int = 60,
        boundary_config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
        use_vertical_solver: bool = True,
        disable_guards: bool = False,
        epssm: float = 0.1,
        top_lid: bool = False,
        w_damping: int = 0,
        damp_opt: int = 0,
        dampcoef: float = 0.0,
        zdamp: float = 5000.0,
        diff_opt: int = 0,
        km_opt: int = 0,
        khdif: float = 0.0,
        kvdif: float = 0.0,
        c_s: float = C_S_DEFAULT,
        diff_6th_opt: int = 0,
        diff_6th_factor: float = 0.12,
        const_nu_m2_s: float = 0.0,
        use_flux_advection: bool = False,
        scalar_adv_opt: int = 0,
        moist_adv_opt: int = 0,
        force_fp64: bool = False,
        use_deformation_momentum_diffusion: bool = False,
        time_utc: object = None,
        radiation_static: object = None,
        topo_shading: int = 0,
        slope_rad: int = 0,
        topo_shadow_length_m: float = 25000.0,
        gwd_opt: int = 0,
        gwdo_statics: object = None,
        rad_rk_tendf: int = 0,
    ) -> "OperationalNamelist":
        """Build a namelist using resident zero tendencies and flat metrics."""

        if tendencies is None:
            tendencies = Tendencies.zeros(grid)
        if metrics is None:
            metrics = DycoreMetrics.flat(
                ny=grid.ny,
                nx=grid.nx,
                nz=grid.nz,
                eta_levels=grid.vertical.eta_levels,
                top_pressure_pa=grid.vertical.top_pressure_pa,
                provenance="operational-flat-from-grid",
            )
        return cls(
            grid=grid,
            tendencies=tendencies,
            metrics=metrics,
            dt_s=dt_s,
            acoustic_substeps=acoustic_substeps,
            epssm=epssm,
            top_lid=top_lid,
            radiation_cadence_steps=radiation_cadence_steps,
            boundary_config=boundary_config,
            use_vertical_solver=use_vertical_solver,
            disable_guards=disable_guards,
            w_damping=w_damping,
            damp_opt=damp_opt,
            dampcoef=dampcoef,
            zdamp=zdamp,
            diff_opt=diff_opt,
            km_opt=km_opt,
            khdif=khdif,
            kvdif=kvdif,
            c_s=c_s,
            diff_6th_opt=diff_6th_opt,
            diff_6th_factor=diff_6th_factor,
            const_nu_m2_s=const_nu_m2_s,
            use_flux_advection=use_flux_advection,
            scalar_adv_opt=scalar_adv_opt,
            moist_adv_opt=moist_adv_opt,
            force_fp64=force_fp64,
            use_deformation_momentum_diffusion=use_deformation_momentum_diffusion,
            time_utc=time_utc,
            radiation_static=radiation_static,
            topo_shading=topo_shading,
            slope_rad=slope_rad,
            topo_shadow_length_m=topo_shadow_length_m,
            gwd_opt=gwd_opt,
            gwdo_statics=gwdo_statics,
            rad_rk_tendf=rad_rk_tendf,
        )

    def tree_flatten(self):
        # The Noah-MP static (categories + soil geometry + the constant parameter
        # TABLES) and the pre-built energy/rad params ride as STATIC AUX, not traced
        # children: the frozen driver concretizes several of their fields inside the
        # scan (isurban, nroot, table scalars), so they must be COMPILE CONSTANTS,
        # not tracers. They are wrapped in an identity-hashable holder so the jit
        # cache keys on per-run object identity (one run -> one compile). use_noahmp
        # + clock scalars are also static aux.
        children = (self.tendencies, self.metrics, self.radiation_static, self.gwdo_statics)
        aux = (
            self.grid,
            float(self.dt_s),
            int(self.acoustic_substeps),
            int(self.rk_order),
            float(self.epssm),
            bool(self.top_lid),
            bool(self.run_physics),
            bool(self.run_boundary),
            int(self.radiation_cadence_steps),
            self.boundary_config,
            bool(self.use_vertical_solver),
            bool(self.disable_guards),
            int(self.w_damping),
            int(self.damp_opt),
            float(self.dampcoef),
            float(self.zdamp),
            int(self.diff_opt),
            int(self.km_opt),
            float(self.khdif),
            float(self.kvdif),
            float(self.c_s),
            int(self.diff_6th_opt),
            float(self.diff_6th_factor),
            float(self.const_nu_m2_s),
            bool(self.use_flux_advection),
            int(self.scalar_adv_opt),
            int(self.moist_adv_opt),
            bool(self.force_fp64),
            bool(self.use_deformation_momentum_diffusion),
            self.time_utc,
            int(self.topo_shading),
            int(self.slope_rad),
            float(self.topo_shadow_length_m),
            bool(self.use_noahmp),
            int(self.noahmp_nroot),
            float(self.noahmp_julian),
            float(self.noahmp_yearlen),
            _StaticHolder(self.noahmp_static),
            _StaticHolder(self.noahmp_energy_params),
            _StaticHolder(self.noahmp_rad_params),
            int(self.mp_physics),
            int(self.bl_pbl_physics),
            int(self.sf_sfclay_physics),
            int(self.cu_physics),
            self.sf_surface_physics,
            _StaticHolder(self.noahclassic_static),
            _StaticHolder(self.noahclassic_land),
            _StaticHolder(self.noahclassic_rad),
            int(self.gwd_opt),
            int(self.ra_sw_physics),
            int(self.ra_lw_physics),
            int(self.rad_rk_tendf),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        tendencies, metrics, radiation_static, gwdo_statics = children
        (
            grid,
            dt_s,
            acoustic_substeps,
            rk_order,
            epssm,
            top_lid,
            run_physics,
            run_boundary,
            radiation_cadence_steps,
            boundary_config,
            use_vertical_solver,
            disable_guards,
            w_damping,
            damp_opt,
            dampcoef,
            zdamp,
            diff_opt,
            km_opt,
            khdif,
            kvdif,
            c_s,
            diff_6th_opt,
            diff_6th_factor,
            const_nu_m2_s,
            use_flux_advection,
            scalar_adv_opt,
            moist_adv_opt,
            force_fp64,
            use_deformation_momentum_diffusion,
            time_utc,
            topo_shading,
            slope_rad,
            topo_shadow_length_m,
            use_noahmp,
            noahmp_nroot,
            noahmp_julian,
            noahmp_yearlen,
            noahmp_static_holder,
            noahmp_energy_holder,
            noahmp_rad_holder,
            mp_physics,
            bl_pbl_physics,
            sf_sfclay_physics,
            cu_physics,
            sf_surface_physics,
            noahclassic_static_holder,
            noahclassic_land_holder,
            noahclassic_rad_holder,
            gwd_opt,
            ra_sw_physics,
            ra_lw_physics,
            rad_rk_tendf,
        ) = aux
        noahmp_static = noahmp_static_holder.value
        noahmp_energy_params = noahmp_energy_holder.value
        noahmp_rad_params = noahmp_rad_holder.value
        noahclassic_static = noahclassic_static_holder.value
        noahclassic_land = noahclassic_land_holder.value
        noahclassic_rad = noahclassic_rad_holder.value
        return cls(
            grid=grid,
            tendencies=tendencies,
            metrics=metrics,
            dt_s=dt_s,
            acoustic_substeps=acoustic_substeps,
            rk_order=rk_order,
            epssm=epssm,
            top_lid=top_lid,
            run_physics=run_physics,
            run_boundary=run_boundary,
            radiation_cadence_steps=radiation_cadence_steps,
            boundary_config=boundary_config,
            use_vertical_solver=use_vertical_solver,
            disable_guards=disable_guards,
            w_damping=w_damping,
            damp_opt=damp_opt,
            dampcoef=dampcoef,
            zdamp=zdamp,
            diff_opt=diff_opt,
            km_opt=km_opt,
            khdif=khdif,
            kvdif=kvdif,
            c_s=c_s,
            diff_6th_opt=diff_6th_opt,
            diff_6th_factor=diff_6th_factor,
            const_nu_m2_s=const_nu_m2_s,
            use_flux_advection=use_flux_advection,
            scalar_adv_opt=scalar_adv_opt,
            moist_adv_opt=moist_adv_opt,
            force_fp64=force_fp64,
            use_deformation_momentum_diffusion=use_deformation_momentum_diffusion,
            time_utc=time_utc,
            radiation_static=radiation_static,
            topo_shading=topo_shading,
            slope_rad=slope_rad,
            topo_shadow_length_m=topo_shadow_length_m,
            use_noahmp=use_noahmp,
            noahmp_static=noahmp_static,
            noahmp_energy_params=noahmp_energy_params,
            noahmp_rad_params=noahmp_rad_params,
            noahmp_nroot=noahmp_nroot,
            noahmp_julian=noahmp_julian,
            noahmp_yearlen=noahmp_yearlen,
            mp_physics=mp_physics,
            bl_pbl_physics=bl_pbl_physics,
            sf_sfclay_physics=sf_sfclay_physics,
            cu_physics=cu_physics,
            sf_surface_physics=sf_surface_physics,
            noahclassic_static=noahclassic_static,
            noahclassic_land=noahclassic_land,
            noahclassic_rad=noahclassic_rad,
            gwd_opt=gwd_opt,
            gwdo_statics=gwdo_statics,
            ra_sw_physics=ra_sw_physics,
            ra_lw_physics=ra_lw_physics,
            rad_rk_tendf=rad_rk_tendf,
        )


@dataclass(frozen=True)
class _RKStageDescriptor:
    """Static WRF RK/acoustic cadence descriptor from ``solve_em.F:1472-1483``."""

    rk_step: int
    dt_rk: float
    dts_rk: float
    number_of_small_timesteps: int


def _steps_for_hours(hours: float, dt_s: float) -> int:
    raw = float(hours) * 3600.0 / float(dt_s)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"forecast length {hours}h is not an integer number of dt={dt_s}s steps")
    return rounded


def _enforce_operational_precision(state: State, *, force_fp64: bool = False) -> State:
    if bool(force_fp64):
        # Sprint F7-B is fp64-correctness-only: idealized cases and any caller
        # that sets force_fp64 keep every prognostic in float64.  The fp32-gated
        # operational matrix (ADR-007) is a perf decision deferred to F7-perf.
        # v0.10.0 Wave-A (Opus#4/GPT#20): SKIP the .astype when the field is
        # already fp64 -- a no-op convert that XLA may still materialise (the HLO
        # audit counted 26 stablehlo.convert here, ~23 from non-fp64-default
        # leaves; the all-fp64 carried-State case has zero non-no-op casts).
        # Emitting .astype only on the genuinely-mismatched leaves removes the
        # whole per-step convert family for the warmed carried fp64 State.
        # Bit-identical: fp64->fp64 .astype is the identity.
        updates = {}
        for field in STATE_FIELD_ORDER:
            value = getattr(state, field)
            if value.dtype != jnp.float64:
                updates[field] = value.astype(jnp.float64)
        if not updates:
            return state.replace(_cast=False)
        # _cast=False so the fp64 upcast is NOT canonicalised back to each
        # field's loaded dtype.  Real-case states arrive mixed-precision
        # (DEFAULT_DTYPES perf matrix: theta/u/v fp32, w/mu/ph fp64); without
        # this the force_fp64 path is a silent no-op (Sprint U P0-1).
        return state.replace(_cast=False, **updates)
    updates = {}
    for field in STATE_FIELD_ORDER:
        value = getattr(state, field)
        target = DEFAULT_DTYPES.dtype_for(field)
        if value.dtype != target:
            updates[field] = value.astype(target)
    return state.replace(**updates)


def _theta_base_offset(theta: jax.Array) -> jax.Array:
    """Return the WRF perturbation-theta offset for operational Gen2 states."""

    return jnp.asarray(300.0, dtype=theta.dtype)


def _acoustic_lateral_bc_flags(namelist: OperationalNamelist) -> tuple[bool, bool, bool]:
    """Return WRF ``advance_mu_t`` BC flags: ``periodic_x, specified, nested``."""

    boundary_active = bool(namelist.run_boundary) and getattr(namelist.grid.bc, "source", "ideal") != "ideal"
    if not boundary_active:
        return True, False, False
    nested = not bool(getattr(namelist.boundary_config, "force_geopotential", True))
    return False, not nested, nested


def _maybe_sharded_u_face_average(field: jax.Array, face: jax.Array) -> jax.Array:
    context = _SHARDED_CARRY_HALO_CONTEXT
    if context is None:
        return face
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return face
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("operational sharded face average supports x-axis decomposition only")
    h = int(width)
    owned = int(field.shape[-1]) - 2 * h
    if owned < 1:
        raise ValueError("haloed x field has no owned cells")
    rank = jax.lax.axis_index(str(sharding.axis_name))
    start = rank * owned
    global_nx = owned * int(sharding.resolved_partitions())
    west_face = h
    east_face = h + owned
    is_first = start == 0
    is_last = start + owned == global_nx
    face = face.at[:, west_face].set(jnp.where(is_first, field[:, h], face[:, west_face]))
    face = face.at[:, east_face].set(jnp.where(is_last, field[:, h + owned - 1], face[:, east_face]))
    return face


def _u_face_average_2d(field: jax.Array) -> jax.Array:
    west = field[:, :1]
    east = field[:, -1:]
    interior = 0.5 * (field[:, :-1] + field[:, 1:])
    return _maybe_sharded_u_face_average(field, jnp.concatenate((west, interior, east), axis=1))


def _v_face_average_2d(field: jax.Array) -> jax.Array:
    south = field[:1, :]
    north = field[-1:, :]
    interior = 0.5 * (field[:-1, :] + field[1:, :])
    return jnp.concatenate((south, interior, north), axis=0)


def _base_mu(state: State) -> jax.Array:
    return jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)


def _valid_mixing_ratio(candidate: jax.Array, origin: jax.Array, upper: float = 0.05) -> jax.Array:
    """Keep nonfinite RK moisture excursions out of the physics boundary."""

    candidate = jnp.asarray(candidate)
    origin = jnp.asarray(origin, dtype=candidate.dtype)
    valid = jnp.isfinite(candidate) & (candidate >= 0.0) & (candidate <= float(upper))
    return jnp.where(valid, candidate, origin)


def _finite_or_origin(candidate: jax.Array, origin: jax.Array) -> jax.Array:
    """Reject nonfinite boundary replay values without clipping finite dynamics."""

    candidate = jnp.asarray(candidate)
    origin = jnp.asarray(origin, dtype=candidate.dtype)
    return jnp.where(jnp.isfinite(candidate), candidate, origin)


def _theta_mass_weights(theta: jax.Array, mu_total: jax.Array) -> jax.Array:
    """Broadcast positive column dry mass onto theta mass points."""

    theta = jnp.asarray(theta)
    mass_2d = jnp.asarray(mu_total, dtype=theta.dtype)
    mass_2d = jnp.where(jnp.isfinite(mass_2d) & (mass_2d > 0.0), mass_2d, 0.0)
    return jnp.broadcast_to(mass_2d[None, :, :], theta.shape)


def _theta_level_monotonic_bounds(
    origin: jax.Array,
    *,
    minimum_k: float = _THETA_LIMITER_MIN_K,
    maximum_k: float = _THETA_LIMITER_MAX_K,
) -> tuple[jax.Array, jax.Array]:
    """Return per-level monotonicity bounds for positive-definite theta advection."""

    origin = jnp.asarray(origin, dtype=jnp.float64)
    safe = jnp.where(jnp.isfinite(origin), origin, 0.5 * (float(minimum_k) + float(maximum_k)))
    lower = jnp.min(safe, axis=(1, 2), keepdims=True)
    upper = jnp.max(safe, axis=(1, 2), keepdims=True)
    lower = jnp.maximum(lower, float(minimum_k))
    upper = jnp.minimum(jnp.maximum(upper, lower), float(maximum_k))
    return lower, upper


def _first_limited_cell_xyz(mask: jax.Array) -> jax.Array:
    """Return first limited mass-cell coordinate as ``[x, y, z]`` or ``[-1, -1, -1]``."""

    flat = jnp.ravel(mask)
    count = jnp.sum(flat.astype(jnp.int32))
    flat_index = jnp.argmax(flat.astype(jnp.int32))
    ny = int(mask.shape[1])
    nx = int(mask.shape[2])
    z = flat_index // (ny * nx)
    rem = flat_index - z * ny * nx
    y = rem // nx
    x = rem - y * nx
    xyz = jnp.stack((x, y, z)).astype(jnp.int32)
    missing = jnp.full((3,), -1, dtype=jnp.int32)
    return jnp.where(count > 0, xyz, missing)


def _empty_theta_limiter_diagnostics(theta: jax.Array) -> dict[str, jax.Array]:
    """Build the INV-10 diagnostic record used when the limiter is inactive."""

    dtype = jnp.asarray(theta).dtype
    return {
        "theta_limited_cell_count": jnp.asarray(0, dtype=jnp.int32),
        "theta_first_limited_cell_xyz": jnp.full((3,), -1, dtype=jnp.int32),
        "theta_mass_before": jnp.asarray(0.0, dtype=dtype),
        "theta_mass_after": jnp.asarray(0.0, dtype=dtype),
        "theta_mass_residual": jnp.asarray(0.0, dtype=dtype),
    }


def _positive_definite_theta_increment_limiter(
    candidate: jax.Array,
    origin: jax.Array,
    mass: jax.Array,
    *,
    minimum_k: float = _THETA_LIMITER_MIN_K,
    maximum_k: float = _THETA_LIMITER_MAX_K,
    lower_bound: jax.Array | None = None,
    upper_bound: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Limit theta increments to a positive finite interval while conserving mass.

    Offending cells keep the RK direction but receive a smaller increment.  The
    removed mass-weighted theta increment is then redistributed over cells with
    available room, so feasible updates preserve the raw dycore scalar integral.
    """

    output_dtype = jnp.asarray(candidate).dtype
    candidate64 = jnp.asarray(candidate, dtype=jnp.float64)
    origin64 = jnp.asarray(origin, dtype=jnp.float64)
    mass64 = jnp.asarray(mass, dtype=jnp.float64)
    lower = jnp.asarray(float(minimum_k), dtype=jnp.float64)
    upper = jnp.asarray(float(maximum_k), dtype=jnp.float64)
    if lower_bound is not None:
        lower = jnp.maximum(lower, jnp.asarray(lower_bound, dtype=jnp.float64))
    if upper_bound is not None:
        upper = jnp.minimum(upper, jnp.asarray(upper_bound, dtype=jnp.float64))
    upper = jnp.maximum(upper, lower)
    midpoint = 0.5 * (lower + upper)

    safe_origin = jnp.where(jnp.isfinite(origin64), origin64, midpoint)
    safe_origin = jnp.minimum(jnp.maximum(safe_origin, lower), upper)
    finite_candidate = jnp.where(jnp.isfinite(candidate64), candidate64, safe_origin)
    raw_delta = finite_candidate - safe_origin

    over_upper = finite_candidate > upper
    under_lower = finite_candidate < lower
    invalid = ~jnp.isfinite(candidate64)
    limited_mask = invalid | over_upper | under_lower

    positive_delta = raw_delta > 0.0
    negative_delta = raw_delta < 0.0
    upper_alpha = (upper - safe_origin) / jnp.where(positive_delta, raw_delta, 1.0)
    lower_alpha = (lower - safe_origin) / jnp.where(negative_delta, raw_delta, -1.0)
    alpha = jnp.where(positive_delta, upper_alpha, jnp.where(negative_delta, lower_alpha, 1.0))
    alpha = jnp.where(limited_mask, jnp.minimum(jnp.maximum(alpha, 0.0), 1.0), 1.0)
    limited0 = safe_origin + alpha * raw_delta

    target_mass = jnp.sum(finite_candidate * mass64)
    mass0 = jnp.sum(limited0 * mass64)
    residual = target_mass - mass0
    add_room = upper - limited0
    subtract_room = limited0 - lower
    room = jnp.where(residual >= 0.0, add_room, subtract_room)
    capacity = jnp.sum(room * mass64)
    fraction = jnp.where(capacity > 0.0, jnp.minimum(jnp.abs(residual) / capacity, 1.0), 0.0)
    limited = limited0 + jnp.sign(residual) * fraction * room
    limited = jnp.minimum(jnp.maximum(limited, lower), upper)
    limited = limited.astype(output_dtype)

    after_mass = jnp.sum(limited.astype(jnp.float64) * mass64)
    diagnostics = {
        "theta_limited_cell_count": jnp.sum(limited_mask.astype(jnp.int32)),
        "theta_first_limited_cell_xyz": _first_limited_cell_xyz(limited_mask),
        "theta_mass_before": target_mass.astype(output_dtype),
        "theta_mass_after": after_mass.astype(output_dtype),
        "theta_mass_residual": (after_mass - target_mass).astype(output_dtype),
    }
    return limited, diagnostics


def _limit_guarded_mass_state(candidate: State, origin: State) -> State:
    """Keep finite positive dry mass without changing theta after physics/boundary."""

    candidate_mu_total = jnp.asarray(candidate.mu_total)
    candidate_mu_perturbation = jnp.asarray(candidate.mu_perturbation)
    valid_mu = (
        jnp.isfinite(candidate_mu_total)
        & jnp.isfinite(candidate_mu_perturbation)
        & (candidate_mu_total >= 1.0)
    )
    mu_total = jnp.where(valid_mu, candidate_mu_total, origin.mu_total)
    mu_perturbation = jnp.where(valid_mu, candidate_mu_perturbation, origin.mu_perturbation)
    return candidate.replace(mu=mu_total, mu_total=mu_total, mu_perturbation=mu_perturbation)


def _limit_guarded_dynamics_state_with_diagnostics(candidate: State, origin: State) -> tuple[State, dict[str, jax.Array]]:
    """Apply the dycore theta safety net and dry-mass guard after one RK3 step.

    GUARDS-MUST-NOT-BE-LOAD-BEARING FIX (2026-06-01, operational-path-divergence
    sprint).  Previously this passed the per-level domain-MIN/MAX monotonic bounds
    (``_theta_level_monotonic_bounds(origin.theta)``) into the increment limiter and
    then mass-conservatively REDISTRIBUTED the clamped-away increment over the
    column.  On the operational d02/d03 path that made the guard LOAD-BEARING: over
    the cooling open ocean the coldest columns hit the per-level minimum, the
    suppressed cooling was treated as "removed mass" and pumped back as warming, so
    the integration drifted +3.3 K warm in the lowest levels over 6 h relative to the
    guards-off path that the v0.1.0 D02_VALIDATED proof used (and that matches
    CPU-WRF).  Root cause + isolation experiment: PERHOUR(guards-on) warm-drifts
    +3.3 K; PH_GUARDOFF (only difference = guards) collapses to the validated
    -0.1 K; see ``.agent/reviews/2026-06-01-opus-operational-path-divergence.md`` and
    ``proofs/v010_validation/path_divergence_case3.json``.

    The fix drops the tight per-level monotonic bounds so the limiter uses ONLY the
    WIDE physical envelope ``[_THETA_LIMITER_MIN_K, _THETA_LIMITER_MAX_K]`` =
    ``[0, 500] K`` plus the non-finite trap.  For any physically reasonable theta the
    envelope never fires (``limited_mask`` all-False), so the increment limiter is a
    strict identity AND its mass-redistribution residual is ~0 — i.e. it becomes a
    genuine non-load-bearing safety net that catches only NaN/Inf and true blow-ups,
    leaving the physical trajectory bit-equivalent to the guards-off integration.
    The idealized warm-bubble/Straka gates already run ``disable_guards=True`` so this
    path is a no-op for them; the change only affects the operational guards-on path.
    """

    mass = _theta_mass_weights(candidate.theta, candidate.mu_total)
    theta, diagnostics = _positive_definite_theta_increment_limiter(
        candidate.theta,
        origin.theta,
        mass,
    )
    limited = _limit_guarded_mass_state(candidate.replace(theta=theta), origin)
    return limited, diagnostics


def _limit_guarded_dynamics_state(candidate: State, origin: State) -> State:
    """Keep finite bounded dynamics from RK3 while preserving positive dry mass."""

    limited, _diagnostics = _limit_guarded_dynamics_state_with_diagnostics(candidate, origin)
    return limited


def _limit_theta_by_level(theta: jax.Array, origin_theta: jax.Array) -> jax.Array:
    """Back-compat thin envelope clip for diagnostic harness leaf-level interface.

    M11 removed the production [200K, 450K] envelope limiter in favor of the
    positive-definite increment limiter inside _limit_guarded_dynamics_state.
    The diagnostic harness still wants a leaf-level clip with origin fallback
    for instrumentation purposes; this preserves the old signature without
    changing production semantics (production calls the full state limiter).
    """
    lower_bound = jnp.asarray(200.0, dtype=theta.dtype)
    upper_bound = jnp.asarray(450.0, dtype=theta.dtype)
    in_envelope = jnp.isfinite(theta) & (theta >= lower_bound) & (theta <= upper_bound)
    return jnp.where(in_envelope, theta, jnp.clip(origin_theta, lower_bound, upper_bound))


def _with_save_family(carry: OperationalCarry, state: State, ww: jax.Array | None = None) -> OperationalCarry:
    """Update WRF ``*_save`` transition fields in resident operational carry."""

    ww_value = carry.ww if ww is None else ww
    mu_base = _base_mu(state)
    return carry.replace(
        state=state,
        muave=jnp.zeros_like(state.mu_perturbation),
        muts=mu_base,
        u_save=state.u,
        v_save=state.v,
        w_save=state.w,
        t_save=state.theta,
        ph_save=state.ph,
        mu_save=state.mu_perturbation,
        ww_save=ww_value,
    )


def _m6b_acoustic_tendencies(tendencies: Tendencies, base: Tendencies) -> Tendencies:
    """Legacy diagnostic import shim; no longer suppresses V tendencies."""

    del base
    return tendencies


def _horizontal_pressure_gradient_tendencies(state: State, namelist: OperationalNamelist) -> tuple[jax.Array, jax.Array]:
    """Compute WRF-shaped velocity PGF tendencies for operational RK u/v."""

    pressure, al, alt = diagnose_pressure_al_alt(state, None, namelist.metrics)
    cqu, cqv = moisture_coupling_factors(state)
    du_dt, dv_dt, _, _ = horizontal_pressure_gradient(
        state,
        None,
        namelist.metrics,
        pressure,
        al,
        alt,
        cqu,
        cqv,
        dx_m=namelist.grid.projection.dx_m,
        dy_m=namelist.grid.projection.dy_m,
        non_hydrostatic=True,
        top_lid=bool(namelist.top_lid),
    )
    return du_dt, dv_dt


def _acoustic_core_state(carry: OperationalCarry, namelist: OperationalNamelist) -> AcousticCoreState:
    state = carry.state
    theta_offset = _theta_base_offset(state.theta)
    theta_pert = (state.theta - theta_offset).astype(jnp.float64)
    theta_save_pert = (carry.t_save - theta_offset).astype(jnp.float64)
    theta_ave_pert = (carry.t_2ave - theta_offset).astype(jnp.float64)
    mu_base = _base_mu(state)
    mu_total = mu_base + state.mu_perturbation
    metrics = namelist.metrics
    # Real advance_w inputs for the legacy non-prep helper path so the WRF
    # implicit-w solve receives finite, consistent coefficients (matches the
    # production prep-path semantics): real c2a from the dry EOS, real dry cqw,
    # base pressure/geopotential, and terrain ht = phb(sfc)/g.
    p_base = (state.p_total - state.p_perturbation).astype(jnp.float64)
    ph_base = (state.ph_total - state.ph_perturbation).astype(jnp.float64)
    alt = _inverse_density_from_theta_pressure(
        state.theta.astype(jnp.float64), state.p_total.astype(jnp.float64)
    )
    c2a = CPOVCV * (p_base + state.p_perturbation.astype(jnp.float64)) / jnp.maximum(
        jnp.abs(alt), jnp.asarray(1.0e-12, dtype=alt.dtype)
    )
    nz = int(state.theta.shape[0])
    ny = int(state.theta.shape[1])
    nx = int(state.theta.shape[2])
    return AcousticCoreState(
        ww=carry.ww,
        ww_1=carry.ww_save,
        u=state.u,
        u_1=carry.u_save,
        v=state.v,
        v_1=carry.v_save,
        w=state.w,
        mu=state.mu_perturbation,
        mut=mu_base,
        muave=carry.muave,
        muts=carry.muts,
        muu=_u_face_average_2d(mu_total),
        muv=_v_face_average_2d(mu_total),
        mudf=carry.mudf,
        theta=theta_pert,
        theta_1=theta_save_pert,
        theta_ave=theta_ave_pert,
        theta_tend=namelist.tendencies.theta,
        mu_tend=namelist.tendencies.mu,
        ph_tend=carry.ph_tend,
        ph=state.ph_perturbation,
        p=state.p_perturbation,
        t_2ave=theta_ave_pert,
        dnw=metrics.dnw,
        fnm=metrics.fnm,
        fnp=metrics.fnp,
        rdnw=metrics.rdnw,
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        msfuy=metrics.msfuy,
        msfvx_inv=1.0 / metrics.msfvx,
        msftx=metrics.msftx,
        msfty=metrics.msfty,
        coef_mut=mu_base,
        al=jnp.zeros_like(state.p_perturbation),
        alt=alt,
        p_base=p_base,
        ph_base=ph_base,
        cqu=jnp.ones_like(state.u, dtype=jnp.float64),
        cqv=jnp.ones_like(state.v, dtype=jnp.float64),
        msfux=metrics.msfux,
        msfvx=metrics.msfvx,
        msfvy=metrics.msfvy,
        cf1=metrics.cf1,
        cf2=metrics.cf2,
        cf3=metrics.cf3,
        c2a=c2a,
        cqw=dry_cqw(nz, ny, nx, dtype=jnp.float64),
        c1f=metrics.c1f,
        c2f=metrics.c2f,
        rdn=metrics.rdn,
        phb=ph_base,
        ph_1=carry.ph_save.astype(jnp.float64) - ph_base,
        ht=ph_base[0, :, :] / GRAVITY_M_S2,
        pm1=state.p_perturbation.astype(jnp.float64),
        ru_m=jnp.zeros_like(state.u, dtype=jnp.float64),
        rv_m=jnp.zeros_like(state.v, dtype=jnp.float64),
        ww_m=jnp.zeros_like(carry.ww),
        # Physical perturbation w from the carry save family (WRF w_save) for the
        # damp_opt=3 implicit Rayleigh damping in advance_w.
        w_save=carry.w_save.astype(jnp.float64),
    )


def _acoustic_core_state_from_prep(
    carry: OperationalCarry,
    prep: SmallStepPrepState,
    pressure: CalcPRhoStep0,
    namelist: OperationalNamelist,
    tendencies: Tendencies,
    *,
    lead_seconds=None,
) -> AcousticCoreState:
    """Build the acoustic work-state directly from WRF ``small_step_prep``."""

    state = prep.entry_state
    theta_pert = (state.theta - prep.theta_offset).astype(jnp.float64)
    ph_base = state.ph_total - state.ph_perturbation
    # F7H: WRF builds the large-step vertical PGF/buoyancy ``rw_tend`` ONCE per RK
    # stage in rk_tendency (module_em.F:1361-1368) by calling pg_buoy_w with the
    # stage diagnostic ``grid%p`` and the stage perturbation dry mass
    # ``mu' = mut - mub``.  In WRF that ``grid%p`` is the FULL-perturbation
    # ``calc_p_rho_phi`` diagnostic (module_big_step_utilities_em.F:1029,1083-1087)
    # built from the FULL ``ph'``, ``mu'`` and ``theta'`` — NOT the small-step
    # work-delta pressure.  Its ``rdn*(p[k]-p[k-1])`` interior PGF term
    # hydrostatically balances the ``-c1f*mu'`` weight of the perturbation column,
    # so the net interior forcing on a near-balanced thermal stays small.
    #
    # The previous F7G code fed ``pressure.p`` = ``calc_p_rho_wrf(prep)``, which is
    # built from ``prep.ph_work`` (= ph_ref - ph_cur ~ 0) and ``prep.mu_work``
    # (~0) — the small-step WORK-DELTA pressure, near zero and carrying NONE of the
    # ph'/mu' hydrostatic structure.  The PGF term then could not cancel the
    # ``-c1f*mu'`` weight, leaving a net forcing ~ g*c1f*mu' that grows as mu'
    # grows (w runaway).  Trace: proofs/f7h/full_p_compare.json (interior net
    # work_p >> full_p).  Fix = feed pg_buoy_w the full-perturbation grid%p via the
    # F7F-fixed diagnose_pressure_al_alt (the JAX calc_p_rho_phi), exactly as WRF
    # rk_tendency does.  ``pressure.p`` (work-delta) still correctly seeds the
    # substep ``p``/``pm1`` smdiv memory below.
    nz_stage = int(prep.theta_work.shape[0])
    ny_stage = int(prep.theta_work.shape[1])
    nx_stage = int(prep.theta_work.shape[2])
    mu_prime_stage = prep.mut - prep.mub  # stage perturbation dry mass mu' (WRF grid%mu_2)
    stage_base = BaseState(
        pb=prep.pb,
        phb=ph_base,
        mub=prep.mub,
        t0=jnp.asarray(prep.theta_offset),
        theta_base=jnp.full_like(state.theta, prep.theta_offset),
    )
    grid_p_full, _stage_al_full, _stage_alt_full = diagnose_pressure_al_alt(
        state, stage_base, namelist.metrics
    )
    rw_tend_stage = pg_buoy_w_dry(
        grid_p_full,
        mu_prime_stage,
        c1f=namelist.metrics.c1f,
        rdnw=namelist.metrics.rdnw,
        rdn=namelist.metrics.rdn,
        msfty=namelist.metrics.msfty,
        gravity=GRAVITY_M_S2,
    )
    # F7J item 2: WRF ``rk_tendency`` builds ``rw_tend`` as ``advect_w(w)`` (the
    # large-step vertical+horizontal advection of coupled w) THEN ``pg_buoy_w``
    # ADDS the vertical PGF/buoyancy (module_em.F:1011-1067 then :1361-1368).
    # ``tendencies.w`` is the COUPLED large-step w advection from
    # ``_augment_large_step_tendencies`` (``tendencies.w * mass_f``); fold it into
    # the stage ``rw_tend`` so the WRF assembly order is preserved.  Without #1
    # below it does not stabilise the mode (F7I wadv_fix_probe), but it is
    # WRF-correct and required together with the geopotential RHS.
    rw_tend_stage = rw_tend_stage + tendencies.w

    # F7J item 1 (PRIME): the large-step geopotential-equation RHS ``rhs_ph`` was
    # stubbed (``carry.ph_tend`` stayed 0; ``accumulate_ph_tend`` never wired in),
    # so the w/phi acoustic restoring loop never closed and the warm-bubble
    # buoyancy pumped without saturating.  WRF computes it once per RK stage in
    # ``rk_tendency`` (module_em.F:1254-1266 -> rhs_ph,
    # module_big_step_utilities_em.F:1365-2232) using the STAGE explicit omega
    # ``wwE = grid%ww`` and the STAGE geopotential perturbation ``ph``.  This is
    # the large-step (frozen-during-acoustic-loop) half of the geopotential
    # tendency; ``advance_w_wrf`` adds the small-step half (omega/ph_1 evolution).
    ph_tend_stage = rhs_ph_wrf(
        u=state.u,
        v=state.v,
        ww=carry.ww,
        ph=state.ph_perturbation,
        phb=ph_base,
        w=state.w,
        mut=prep.mut,
        muu=prep.muu,
        muv=prep.muv,
        c1f=namelist.metrics.c1f,
        c2f=namelist.metrics.c2f,
        fnm=namelist.metrics.fnm,
        fnp=namelist.metrics.fnp,
        rdnw=namelist.metrics.rdnw,
        rdx=1.0 / float(namelist.grid.projection.dx_m),
        rdy=1.0 / float(namelist.grid.projection.dy_m),
        msfty=namelist.metrics.msfty,
        non_hydrostatic=True,
        gravity=GRAVITY_M_S2,
    )

    # WIND-FIX: stage-constant coupled WORK-array boundary targets for the NORMAL
    # momentum, consumed by ``advance_uv_wrf`` inside the acoustic loop.  Built
    # once per RK stage from the time-interpolated decoupled wrfbdy leaf so that
    # ``small_step_finish_wrf`` reconstructs the boundary velocity ``u_bdy``:
    #     u = (msf*u_work + u_save*mass_cur)/mass_stage
    #  => u_work_bdy = (u_bdy*mass_stage - u_save*mass_cur)/msf .
    # Only staged when the real-case lateral boundary is active; ``None`` keeps the
    # idealized / replay / bare-core paths on the unmodified PGF advance.
    u_work_bdy = None
    v_work_bdy = None
    if bool(namelist.run_boundary) and lead_seconds is not None:
        c1h = namelist.metrics.c1h[:, None, None]
        c2h = namelist.metrics.c2h[:, None, None]
        mass_u_cur = c1h * prep.muu[None, :, :] + c2h
        mass_u_stage = c1h * prep.muus[None, :, :] + c2h
        mass_v_cur = c1h * prep.muv[None, :, :] + c2h
        mass_v_stage = c1h * prep.muvs[None, :, :] + c2h
        cadence = float(namelist.boundary_config.update_cadence_s)
        u_bdy_strip = interpolate_boundary_leaf(state.u_bdy, lead_seconds, cadence)
        v_bdy_strip = interpolate_boundary_leaf(state.v_bdy, lead_seconds, cadence)
        u_work_bdy = normal_bdy_work_target_u(
            u_bdy_strip, prep.u_save, mass_u_cur, mass_u_stage, namelist.metrics.msfuy,
            config=namelist.boundary_config,
        )
        v_work_bdy = normal_bdy_work_target_v(
            v_bdy_strip, prep.v_save, mass_v_cur, mass_v_stage, namelist.metrics.msfvx,
            config=namelist.boundary_config,
        )

    # P0-6 (2026-06-01): NESTED-child ph'/w boundary forcing (d03 T2 Exner bias).
    # Active ONLY for the nested replay path (run_boundary, lateral boundary active,
    # AND boundary_config.force_geopotential == False -- the d03 case).  For d02
    # self-replay (force_geopotential=True) and idealized/bare-core (lead_seconds
    # None / run_boundary False) these stay None and the additions are skipped, so
    # those paths are byte-for-byte unchanged.
    #
    # WRF cadence (solve_em.F:940 relax_bdy_dry once per stage -> rk_addtend_dry
    # folds ph_tendf/msfty into ph_tend, rw_tendf/msfty into rw_tend; the in-loop
    # advance_w consumes ph_tend/rw_tend every substep; spec_bdyupdate_ph pins the
    # spec_zone row of ph_2 after advance_w):
    #   * relax zone -> add the mass-coupled relax tendency to ph_tend_stage /
    #     rw_tend_stage here (so it flows through advance_w coupled with w);
    #   * spec zone  -> stage the full-ring parent ph' target + ph_save for the
    #     in-loop spec_bdyupdate_ph applied inside acoustic_substep_core.
    ph_bdy_target_full = None
    ph_save_for_spec = None
    if (
        bool(namelist.run_boundary)
        and lead_seconds is not None
        and not bool(namelist.boundary_config.force_geopotential)
    ):
        cfg_b = namelist.boundary_config
        cadence = float(cfg_b.update_cadence_s)
        ph_bdy_strip = interpolate_boundary_leaf(state.ph_bdy, lead_seconds, cadence)
        # relax-zone ph' tendency (mass-coupled, /msfty) -> add into ph_tend_stage.
        if bool(getattr(cfg_b, "nested_ph_relax", True)):
            ph_relax = nested_ph_relax_tendency(
                state.ph_perturbation,
                ph_bdy_strip,
                prep.mut,
                namelist.metrics.msfty,
                namelist.metrics.c1f,
                namelist.metrics.c2f,
                float(namelist.dt_s),
                cfg_b,
            )
            ph_tend_stage = ph_tend_stage + ph_relax
        # relax-zone w tendency (nested only) -> add into rw_tend_stage.  Default
        # OFF: the parent 3km w leaf interpolated to the 1km child is a poor target
        # and pumps interior vertical motion (d03 short-run hour-1 with w-relax ON:
        # interior theta' +11.6 K; the pressure collapse is delivered by ph-relax).
        if bool(getattr(cfg_b, "nested_w_relax", False)):
            w_bdy_strip = interpolate_boundary_leaf(state.w_bdy, lead_seconds, cadence)
            w_relax = nested_w_relax_tendency(
                state.w,
                w_bdy_strip,
                prep.mut,
                namelist.metrics.msfty,
                namelist.metrics.c1f,
                namelist.metrics.c2f,
                float(namelist.dt_s),
                cfg_b,
            )
            rw_tend_stage = rw_tend_stage + w_relax
        # spec-zone (outer row) ph' target for the in-loop spec_bdyupdate_ph.
        if bool(getattr(cfg_b, "nested_ph_spec", True)):
            nzp1 = int(state.ph_perturbation.shape[0])
            ny_f = int(state.ph_perturbation.shape[1])
            nx_f = int(state.ph_perturbation.shape[2])
            ph_bdy_target_full = _full_ring_target_from_leaf(
                ph_bdy_strip, nzp1, ny_f, nx_f, state.ph_perturbation.dtype
            )
            ph_save_for_spec = prep.ph_save

    return AcousticCoreState(
        ww=carry.ww,
        ww_1=prep.ww_save,
        u=prep.u_work,
        u_1=prep.u_save,
        v=prep.v_work,
        v_1=prep.v_save,
        w=prep.w_work,
        mu=prep.mu_save + prep.mu_work,
        mut=prep.mut,
        # F7G: stage-entry small-step mass-WORK average is ZERO; advance_mu_t
        # (module_small_step_em.F:1102-1108) rebuilds it from actual small-step
        # mass evolution.  For a fixed-mass mu'=0 thermal it stays zero.
        muave=jnp.zeros_like(prep.mu_work),
        muts=prep.muts,
        muu=prep.muu,
        muv=prep.muv,
        mudf=carry.mudf,
        theta=theta_pert,
        theta_1=prep.t_save,
        # F7G: stage-entry small-step WORK-theta average is ZERO (the coupled work
        # theta t_2 is zero at a fresh RK stage on a fixed-mass rest thermal); the
        # WRF advance_w t_2ave half-step (module_small_step_em.F:1341-1344) builds
        # it up from actual small-step evolution.  Seeding the full initialized
        # theta here was the double-count bug (gpt-council-findings.md §3.5).
        theta_ave=jnp.zeros_like(prep.theta_work),
        # Large-step coupled theta / mu tendencies from rk_tendency+rk_addtend_dry
        # (advection + diffusion), consumed by advance_mu_t (t_2 += msfty*dts*t_tend).
        theta_tend=tendencies.theta,
        mu_tend=tendencies.mu,
        # F7J: real WRF rhs_ph large-step geopotential tendency (was stub=0).
        ph_tend=ph_tend_stage,
        ph=prep.ph_work,
        p=pressure.p,
        t_2ave=jnp.zeros_like(prep.theta_work),
        dnw=namelist.metrics.dnw,
        fnm=namelist.metrics.fnm,
        fnp=namelist.metrics.fnp,
        rdnw=namelist.metrics.rdnw,
        c1h=namelist.metrics.c1h,
        c2h=namelist.metrics.c2h,
        msfuy=namelist.metrics.msfuy,
        msfvx_inv=1.0 / namelist.metrics.msfvx,
        msftx=namelist.metrics.msftx,
        msfty=namelist.metrics.msfty,
        coef_mut=prep.muts,
        u_tend=tendencies.u,
        v_tend=tendencies.v,
        p_base=prep.pb,
        ph_base=ph_base,
        al=pressure.al,
        alt=prep.alt,
        cqu=prep.cqu,
        cqv=prep.cqv,
        msfux=namelist.metrics.msfux,
        msfvx=namelist.metrics.msfvx,
        msfvy=namelist.metrics.msfvy,
        cf1=namelist.metrics.cf1,
        cf2=namelist.metrics.cf2,
        cf3=namelist.metrics.cf3,
        theta_work_reference=prep.theta_1,
        # Initialise the coupled-theta work leaf so the lax.scan carry structure
        # is invariant across substeps (advance_mu_t fills it each substep).
        theta_coupled_work=prep.theta_work,
        c2a=prep.c2a,
        cqw=dry_cqw(
            int(prep.theta_work.shape[0]),
            int(prep.theta_work.shape[1]),
            int(prep.theta_work.shape[2]),
            dtype=prep.theta_work.dtype,
        ),
        c1f=namelist.metrics.c1f,
        c2f=namelist.metrics.c2f,
        rdn=namelist.metrics.rdn,
        phb=state.ph_total - state.ph_perturbation,
        ph_1=prep.ph_1,
        # Terrain height ht = phb(surface)/g (WRF advance_w lower BC :1417-1429).
        ht=(state.ph_total - state.ph_perturbation)[0, :, :] / GRAVITY_M_S2,
        pm1=pressure.pm1,
        ru_m=jnp.zeros_like(prep.u_work),
        rv_m=jnp.zeros_like(prep.v_work),
        ww_m=jnp.zeros_like(carry.ww),
        # F7G: the once-per-RK-stage pg_buoy_w tendency from the stage grid%p/mu'
        # (computed above), carried UNCHANGED through all acoustic substeps.  The
        # legacy per-substep ``p_buoy`` recompute is disabled (None).
        p_buoy=None,
        rw_tend_pg_buoy=rw_tend_stage,
        # Uncoupled physical perturbation w saved by small_step_prep (WRF :272);
        # consumed by the damp_opt=3 implicit Rayleigh w-damping in advance_w.
        w_save=prep.w_save,
        # WIND-FIX: NORMAL-momentum boundary work targets (None unless real-case
        # boundary is active); see advance_uv_wrf / boundary_apply.apply_normal_bdy_work.
        u_work_bdy=u_work_bdy,
        v_work_bdy=v_work_bdy,
        # P0-6: NESTED ph' spec-zone in-loop target + stage-entry ph_save (None
        # unless the nested force_geopotential=False boundary is active).
        ph_bdy_target=ph_bdy_target_full,
        ph_save_for_spec=ph_save_for_spec,
        # SPLIT-EXPLICIT FIX (v0.4.0 r5): WRF ``php`` is built ONCE per RK stage in
        # rk_step_prep (calc_php) and held STAGE-CONSTANT through the acoustic loop;
        # thread the frozen stage array so advance_uv's 4th PGF term does NOT
        # re-diagnose it from the live, substep-updated work geopotential.
        php_stage=prep.php,
    )


def _refresh_grid_p_from_finished(next_state: State, prep: SmallStepPrepState, namelist: OperationalNamelist) -> State:
    """Recompute WRF ``grid%p`` from the finished physical ``ph'`` and ``theta``.

    WRF closes every RK step by calling ``calc_p_rho_phi`` (solve_em.F:6180,
    :7542) which rebuilds the diagnostic perturbation pressure ``grid%p`` (and
    ``al``) from the updated geopotential ``ph`` and theta
    (module_big_step_utilities_em.F:1029, :1083-1087).  The next RK stage's
    large-step horizontal PGF and once-per-stage ``pg_buoy_w`` then act on THAT
    refreshed pressure.

    The JAX operational path previously carried ``p_perturbation`` =
    ``calc_p_rho_step`` work pressure (a delta-from-reference, O(1-10 Pa) for a
    near-balanced thermal), which is NOT the WRF ``grid%p`` diagnostic
    (O(1e3-1e4 Pa) once ``ph'`` evolves).  Feeding that stale O(1) pressure to
    the next stage suppressed the restoring vertical/horizontal PGF, leaving a
    near-constant net vertical force -> w runaway (see proofs/f7h, GPT bughunt
    §2).  This refresh restores the WRF closing diagnostic.  The acoustic substep
    still uses ``calc_p_rho_step`` for its own work-array pressure + smdiv memory.
    """

    base = BaseState(
        pb=prep.pb,
        phb=next_state.ph_total - next_state.ph_perturbation,
        mub=prep.mub,
        t0=jnp.asarray(prep.theta_offset),
        theta_base=jnp.full_like(next_state.theta, prep.theta_offset),
    )
    p_pert, _al, _alt = diagnose_pressure_al_alt(next_state, base, namelist.metrics)
    p_base = next_state.p_total - next_state.p_perturbation
    p_total = p_base + p_pert
    return next_state.replace(
        p=p_total, p_total=p_total, p_perturbation=p_pert,
    )


def _carry_from_finished_stage(
    carry: OperationalCarry,
    prep: SmallStepPrepState,
    acoustic: AcousticCoreState,
    namelist: OperationalNamelist | None = None,
) -> OperationalCarry:
    next_state = small_step_finish_wrf(prep, acoustic)
    if namelist is not None:
        next_state = _refresh_grid_p_from_finished(next_state, prep, namelist)
    ww = acoustic.ww + prep.ww_save
    return carry.replace(
        state=next_state,
        t_2ave=acoustic.t_2ave + prep.theta_offset,
        ww=ww,
        mudf=acoustic.mudf,
        muave=acoustic.muave,
        muts=acoustic.muts,
        ph_tend=acoustic.ph_tend,
        u_save=prep.u_save,
        v_save=prep.v_save,
        w_save=prep.w_save,
        t_save=prep.t_save + prep.theta_offset,
        ph_save=prep.ph_save,
        mu_save=prep.mu_save,
        ww_save=prep.ww_save,
    )


def _maybe_exchange_sharded_carry_halos(carry: OperationalCarry) -> OperationalCarry:
    """Refresh x halos for non-State operational carry leaves under opt-in pmap sharding."""

    context = _SHARDED_CARRY_HALO_CONTEXT
    if context is None:
        return carry
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return carry
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("operational carry sharded halo exchange supports x-axis decomposition only")

    from gpuwrf.runtime.sharding import exchange_periodic_halo_x, exchange_periodic_halo_x_face

    local_nx = int(carry.state.theta.shape[-1])
    num_partitions = int(sharding.resolved_partitions())
    axis_name = str(sharding.axis_name)

    def exchange_leaf(value):
        if value is None or not hasattr(value, "shape") or getattr(value, "ndim", 0) == 0:
            return value
        last_dim = int(value.shape[-1])
        if last_dim == local_nx + 1:
            return exchange_periodic_halo_x_face(
                value,
                width=int(width),
                num_partitions=num_partitions,
                axis_name=axis_name,
            )
        if last_dim == local_nx:
            return exchange_periodic_halo_x(
                value,
                width=int(width),
                num_partitions=num_partitions,
                axis_name=axis_name,
            )
        return value

    updates = {}
    for name in carry.__dataclass_fields__:  # type: ignore[attr-defined]
        if name == "state":
            continue
        updates[name] = jax.tree_util.tree_map(exchange_leaf, getattr(carry, name))
    return carry.replace(**updates) if updates else carry


def _acoustic_scan(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    stage: _RKStageDescriptor,
    prep: SmallStepPrepState,
    pressure: CalcPRhoStep0,
    tendencies: Tendencies,
    lead_seconds=None,
) -> OperationalCarry:
    acoustic = _acoustic_core_state_from_prep(
        carry, prep, pressure, namelist, tendencies, lead_seconds=lead_seconds
    )
    if bool(namelist.use_vertical_solver):
        # WRF calc_coef_w uses the FULL dry mass ``mut`` (solve_em.F:2676-2681),
        # real ``c2a`` from small_step_prep, and the real dry ``cqw``.
        # v0.10.0 Wave-A (Opus#5): ``_acoustic_core_state_from_prep`` already
        # built the identical ``dry_cqw`` array into ``acoustic.cqw`` (:1176), so
        # reuse it here instead of rebuilding a second identical array per RK
        # stage (the build was happening twice: once for the carried state, once
        # for ``calc_coef_w`` + the scan body).  Bit-identical (same dry_cqw).
        cqw_field = acoustic.cqw
        if cqw_field is None:  # defensive: bare-core callers may not stage it
            cqw_field = dry_cqw(
                int(prep.theta_work.shape[0]),
                int(prep.theta_work.shape[1]),
                int(prep.theta_work.shape[2]),
                dtype=prep.theta_work.dtype,
            )
        a, alpha, gamma = calc_coef_w_wrf_coefficients(
            prep.mut,
            namelist.metrics,
            dt=float(stage.dts_rk),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            cqw=cqw_field,
            c2a=prep.c2a,
        )

        periodic_x, specified, nested = _acoustic_lateral_bc_flags(namelist)
        stage_cfg = AcousticCoreConfig(
            dt=float(stage.dts_rk),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            w_damping=int(namelist.w_damping),
            damp_opt=int(namelist.damp_opt),
            dampcoef=float(namelist.dampcoef),
            zdamp=float(namelist.zdamp),
            # WIND-FIX: full model dt so the in-loop normal-momentum relaxation
            # weight is scaled to a per-substep increment.
            dt_full=float(namelist.dt_s),
            periodic_x=periodic_x,
            specified=specified,
            nested=nested,
        )

        # v0.10.0 Wave-A (Opus#1 unroll):
        # NOTE on the reverted carry-split (Opus#2): threading only the ~19
        # evolving leaves through the scan and closing over the ~50 stage-constant
        # leaves was bit-identical, but the warmed A/B was confounded by a one-off
        # cache-miss/recompile artifact and was not cleanly revalidated. The
        # simple full-pytree carry below is the proven non-regressing path; retest
        # any carry split with the corrected cache-hit timing protocol before
        # changing it.
        # See proofs/v0100/inefficiency_ledger.md (Opus#2 = REVERTED).
        def body(scan_acoustic: AcousticCoreState, _):
            return acoustic_substep_core(
                scan_acoustic,
                a=a,
                alpha=alpha,
                gamma=gamma,
                cfg=stage_cfg,
                cqw=cqw_field,
            ), None

        acoustic, _ = jax.lax.scan(
            body,
            acoustic,
            xs=None,
            length=int(stage.number_of_small_timesteps),
            unroll=_acoustic_unroll(),
        )
        next_carry = _carry_from_finished_stage(carry, prep, acoustic, namelist)
        next_carry = _maybe_exchange_sharded_carry_halos(next_carry)
        return next_carry.replace(state=apply_halo(next_carry.state, halo_spec(namelist.grid)))

    del tendencies
    return _maybe_exchange_sharded_carry_halos(_with_save_family(carry, carry.state))


def _augment_large_step_tendencies(
    haloed: State,
    tendencies: Tendencies,
    namelist: OperationalNamelist,
    *,
    rk_step: int = 3,
    physics_tendencies: DryPhysicsTendencies | None = None,
    step_origin: State | None = None,
) -> Tendencies:
    """Add WRF explicit diffusion + flux-form scalar advection to the large step.

    All contributions are returned as *uncoupled* tendencies to match the
    operational RK convention (``add_scaled_tendencies`` adds them uncoupled,
    then ``small_step_prep`` couples).  Sources:
    * 6th-order monotonic filter -- ``module_big_step_utilities_em.F:6504-6920``.
    * constant-K diffusion (Straka ν) -- ``:2999-3234``.
    * flux-form theta advection -- ``module_advect_em.F:3029-4359`` (h=5/v=3).

    ``step_origin`` is the START-OF-STEP haloed state (the WRF ``_1`` reference /
    ``scalar_old`` / ``mu_old``).  It is consumed ONLY by the positive-definite /
    monotonic scalar-advection limiter (``scalar_adv_opt`` 1/2), which WRF applies
    on the final RK3 stage alone (module_em.F:1265 ``rk_step == rk_order``).  When
    ``scalar_adv_opt == 0`` (the default) or it is not the final stage, the plain
    ``advect_scalar_flux`` path runs and ``step_origin`` is ignored, so the
    default dynamics path is byte-for-byte unchanged.
    """

    metrics = namelist.metrics
    grid = namelist.grid
    dx = float(grid.projection.dx_m)
    dy = float(grid.projection.dy_m)
    # mean physical dz from the geopotential column (for the const-K vertical term).
    ph = haloed.ph_total
    dz = jnp.maximum(jnp.mean((ph[1:] - ph[:-1]) / GRAVITY_M_S2), jnp.asarray(1.0, dtype=ph.dtype))

    # All large-step tendencies are built *coupled* (mass-weighted) so they net
    # correctly with the coupled small-step work arrays consumed by advance_uv /
    # advance_mu_t / advance_w (u_work = mass*u etc.).  WRF rk_tendency works in
    # the coupled ru/rv/rw/t_tend space (module_em.F:855-1388); advance_uv adds
    # ``u += dts*ru_tend`` to the coupled u (module_small_step_em.F:805), and
    # advance_mu_t adds ``t_2 += msfty*dts*t_tend`` to the coupled theta
    # (module_small_step_em.F theta update).  Face dry-air masses below match the
    # coupling in small_step_prep_wrf.
    mu_total = haloed.mu_total
    muu = _u_face_average_2d(mu_total)
    muv = _v_face_average_2d(mu_total)
    mass_u = metrics.c1h[:, None, None] * muu[None, :, :] + metrics.c2h[:, None, None]
    mass_v = metrics.c1h[:, None, None] * muv[None, :, :] + metrics.c2h[:, None, None]
    mass_h = metrics.c1h[:, None, None] * mu_total[None, :, :] + metrics.c2h[:, None, None]
    mass_f = metrics.c1f[:, None, None] * mu_total[None, :, :] + metrics.c2f[:, None, None]

    # Advection from compute_advection_tendencies is an UNCOUPLED velocity/scalar
    # acceleration; couple it by the field-specific face mass so it lives in the
    # same coupled tendency space as the PGF and the small-step work arrays.
    u_t = tendencies.u * mass_u
    v_t = tendencies.v * mass_v
    w_t = tendencies.w * mass_f
    th_t = tendencies.theta * mass_h

    if bool(namelist.use_flux_advection):
        # WRF flux-form mass-coupled advection (h=5/v=3).  The *_flux helpers
        # return the COUPLED tendency d(mu*field)/dt, so they replace the
        # primitive coupled products built above (not add to them).
        vel = couple_velocities_periodic(
            haloed.u,
            haloed.v,
            mu_total,
            c1h=metrics.c1h,
            c2h=metrics.c2h,
            dnw=metrics.dnw,
            rdx=1.0 / dx,
            rdy=1.0 / dy,
            msfuy=metrics.msfuy,
            msfvx=metrics.msfvx,
            msftx=metrics.msftx,
            msfux=metrics.msfux,
            msfvy=metrics.msfvy,
        )
        # --- momentum: WRF advect_u/advect_v/advect_w (conservative flux form) ---
        # The previous JAX path advanced momentum with the *advective* (non-
        # conservative) primitive form u*du/dx (advection.py advect_u_face),
        # which does not conserve momentum and lets the Straka cold-front outflow
        # pile up instead of propagating (front crawls ~5 m/s while head |w| runs
        # away).  WRF advances coupled momentum with mass-flux-form advect_u/v/w
        # (module_advect_em.F:126/1530/4364).  Confirmed against pristine WRF
        # v4.7.1 em_grav2d_x ground truth (proofs/m9/wrf_em_grav2d_x_front_*):
        # WRF max|w| saturates ~22 m/s and the front reaches ~4.25 km by 300 s,
        # while the primitive JAX path detonates ~270-300 s with a stalled front.
        u_t = namelist.tendencies.u * mass_u + advect_u_flux(
            haloed.u, vel, rdx=1.0 / dx, rdy=1.0 / dy,
            rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
        )
        v_t = namelist.tendencies.v * mass_v + advect_v_flux(
            haloed.v, vel, rdx=1.0 / dx, rdy=1.0 / dy,
            rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
        )
        w_t = namelist.tendencies.w * mass_f + advect_w_flux(
            haloed.w, vel, rdx=1.0 / dx, rdy=1.0 / dy,
            rdn=metrics.rdn, fzm=metrics.fnm, fzp=metrics.fnp,
            top_lid=bool(namelist.top_lid),
        )
        # --- scalar theta: WRF advect_scalar (h=5/v=3) ---
        theta_offset = _theta_base_offset(haloed.theta)
        # WRF selects the positive-definite (scalar_adv_opt=1) / monotonic (=2)
        # flux limiter ONLY on the final RK3 stage (module_em.F:1265
        # ``rk_step == rk_order``), using the start-of-step scalar/mass; every
        # other stage and ``scalar_adv_opt == 0`` use the plain h5/v3 path.  The
        # branch is a STATIC Python condition (rk_step and scalar_adv_opt are
        # compile-time constants), so the default path emits the identical XLA
        # program and stays bit-for-bit unchanged.
        use_limiter = (
            int(namelist.scalar_adv_opt) in (1, 2)
            and int(rk_step) == int(namelist.rk_order)
            and step_origin is not None
        )
        if use_limiter:
            # field_old / mu_old = the WRF start-of-step ``scalar_old`` / ``mu_old``
            # (grid%mu_1); mut = the current stage total dry mass.  dt = the full
            # model step (on the final RK3 stage WRF's ``dt_step`` recovers the
            # full dt: dt_step*(rk_order-rk_step+1) with rk_step==rk_order == dt).
            coupled_tend = advect_scalar_flux_limited(
                haloed.theta - theta_offset,
                step_origin.theta - theta_offset,
                vel,
                scalar_adv_opt=int(namelist.scalar_adv_opt),
                mut=mu_total,
                mu_old=step_origin.mu_total,
                c1=metrics.c1h,
                c2=metrics.c2h,
                rdx=1.0 / dx,
                rdy=1.0 / dy,
                rdzw=metrics.rdnw,
                fzm=metrics.fnm,
                fzp=metrics.fnp,
                dt=float(namelist.dt_s),
            )
        else:
            coupled_tend = advect_scalar_flux(
                haloed.theta - theta_offset,
                vel,
                mut=mu_total,
                c1=metrics.c1h,
                rdx=1.0 / dx,
                rdy=1.0 / dy,
                rdzw=metrics.rdnw,
                fzm=metrics.fnm,
                fzp=metrics.fnp,
            )
        # tendencies.theta carries the base zero; replace the advective theta part
        # with the flux-form coupled tendency.
        th_t = namelist.tendencies.theta * mass_h + coupled_tend

    if int(namelist.diff_6th_opt) != 0:
        f = float(namelist.diff_6th_factor)
        dt_diff = float(namelist.dt_s)
        u_t = u_t + mass_u * sixth_order_diffusion_tendency(haloed.u, dt=dt_diff, diff_6th_factor=f)
        v_t = v_t + mass_v * sixth_order_diffusion_tendency(haloed.v, dt=dt_diff, diff_6th_factor=f)
        w_t = w_t + mass_f * sixth_order_diffusion_tendency(haloed.w, dt=dt_diff, diff_6th_factor=f)
        th_t = th_t + mass_h * sixth_order_diffusion_tendency(haloed.theta, dt=dt_diff, diff_6th_factor=f)

    nu = float(namelist.const_nu_m2_s)
    if nu > 0.0:
        # WRF diff_opt=2 / km_opt=1 constant-K diffusion on u, v, w AND theta
        # (Straka ν=75).  Plain K∇² form (F7L baseline).  NOTE (F7M): WRF actually
        # diffuses MOMENTUM with the deformation stress tensor — factor-2 diagonal
        # (D11=2 du/dx, D33=2 dw/dz) plus du/dz<->dw/dx cross terms
        # (module_diffusion_em.F cal_deform_and_div :41-47, horizontal/
        # vertical_diffusion_{u,w}_2 :3118-4784, cal_titau_* :5331-5744).  F7M
        # implemented that deformation form (constant_k_deformation_momentum_
        # tendency) and verified it ~2-3x stronger than this Laplacian, but it left
        # the Straka 180s trace byte-identical and still detonated at 240s — the
        # residual is NOT diffusion-controlled (it is the touchdown horizontal-
        # spreading coupling; see proofs/f7m/wrf_vs_jax_straka_front.json).  The
        # deformation operator carries a half-cell cross-term stagger approximation
        # and did not help, so the plain WRF-faithful K∇² baseline is retained
        # pending the touchdown root-cause fix.
        # F7N: use the mass-CONSERVATIVE flux-divergence form d/dx_j(mass*K*d./dx_j)
        # (WRF horizontal_diffusion_s/vertical_diffusion, module_diffusion_em.F:
        # 2999-3018) instead of the non-conservative mass*K*∇² form.  The latter
        # leaked the dry-column mass integral at the sharp Straka cold front
        # (relative drift ~3.4e-8 over 900 s once the touchdown 2Δz fix let Straka
        # run to completion).  The conservative helper already carries the field
        # face mass, so it is NOT multiplied by mass again.  mass_u/mass_v are the
        # u/v face masses (u-face x-diffusion uses the u-face mass; conserves the
        # mass-weighted momentum integral to the same order as WRF).
        #
        # Sprint U (P0-2): theta ALWAYS uses the conservative scalar flux-divergence
        # (WRF horizontal_diffusion_s).  MOMENTUM (u, v, w) optionally uses the WRF
        # deformation-tensor operator (diff_opt=2/km_opt=1, the factor-2 diagonal +
        # du/dz<->dw/dx cross terms) when use_deformation_momentum_diffusion is set;
        # otherwise it keeps the scalar flux-divergence (the F7N close default).  The
        # deformation operator returns the UNCOUPLED tendency K*(2u_xx+u_zz+w_xz);
        # multiply by the field face mass to enter the dry-mass-coupled tendency
        # space, exactly as the scalar diffusion does.  On the flat hydrostatic slab
        # WRF's g*dz/dnw*rho coupling reduces to the same dry-mass face weight
        # (|dnw|=rho*g*dz/mu => g*dz/|dnw|*rho = mu), so this is WRF-faithful.
        th_t = th_t + conservative_constant_k_diffusion_tendency(haloed.theta, mass=mass_h, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
        if bool(namelist.use_deformation_momentum_diffusion):
            unit_rho = jnp.ones_like(haloed.theta)
            du_def, dw_def = wrf_deformation_momentum_tendency(
                haloed.u, haloed.w, rho=unit_rho, k_m2_s=nu, dx_m=dx, dz_m=dz,
            )
            u_t = u_t + mass_u * du_def
            w_t = w_t + mass_f * dw_def
            # v: one-row slab has degenerate y-deformation; keep the scalar
            # flux-divergence (D22/D12 vanish for ny=1, so this is identical to the
            # deformation v-diffusion on the slab).
            v_t = v_t + conservative_constant_k_diffusion_tendency(haloed.v, mass=mass_v, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
        else:
            u_t = u_t + conservative_constant_k_diffusion_tendency(haloed.u, mass=mass_u, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
            v_t = v_t + conservative_constant_k_diffusion_tendency(haloed.v, mass=mass_v, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
            w_t = w_t + conservative_constant_k_diffusion_tendency(haloed.w, mass=mass_f, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)

    # WRF diff_opt=1 / km_opt=4: 2-D Smagorinsky HORIZONTAL diffusion on coordinate
    # (eta) surfaces -- the recommended real-data default.  km_opt=4 computes the
    # horizontal eddy viscosity xkmh (and xkhh=xkmh/prandtl) from the horizontal
    # deformation (smag2d_km, module_diffusion_em.F:1934-2044) of the current
    # velocity field; diff_opt=1 applies the variable-K mass-weighted flux
    # divergence along eta surfaces (horizontal_diffusion / horizontal_diffusion_3dmp,
    # module_big_step_utilities_em.F:2715-3060).  This is a SEPARATE branch from the
    # const-K (diff_opt=2/km_opt=1) path above; the two never run together (the
    # idealized Straka/warm-bubble cases use const_nu_m2_s and leave diff_opt/km_opt
    # at their defaults, so they are bit-unchanged by this block).
    #
    # WRF applies ONLY horizontal Smagorinsky mixing here; the VERTICAL mixing comes
    # from the PBL scheme (module_em.F:842 vertical_diffusion is gated on
    # bl_pbl_physics==0), so this path adds no vertical diffusion -- the operational
    # MYNN PBL provides vertical mixing in the coupled runs.
    if int(namelist.diff_opt) == 1 and int(namelist.km_opt) == 4:
        # Smagorinsky eddy viscosity from the horizontal deformation of (u, v).
        d11, d22, d12 = horizontal_deformation_2d(haloed.u, haloed.v, dx_m=dx, dy_m=dy)
        xkmh, xkhh = smag2d_horizontal_km(
            d11, d22, d12, dx_m=dx, dy_m=dy, c_s=float(namelist.c_s),
        )
        # theta (perturbation vs the WRF 300 K reference base) -- horizontal_diffusion_3dmp.
        theta_base = _theta_base_offset(haloed.theta) * jnp.ones_like(haloed.theta)
        th_t = th_t + horizontal_diffusion_coord_scalar_tendency(
            haloed.theta, xkhh, mass_h, dx_m=dx, dy_m=dy, base_3d=theta_base,
        )
        # momentum (u, v, w) -- horizontal_diffusion 'u'/'v'/'w' branches with xkmh.
        du_s, dv_s, dw_s = horizontal_diffusion_coord_momentum_tendency(
            haloed.u, haloed.v, haloed.w, xkmh, mass_u, mass_v, mass_f, dx_m=dx, dy_m=dy,
        )
        u_t = u_t + du_s
        v_t = v_t + dv_s
        w_t = w_t + dw_s

    # WRF rk_tendency adds the large-step horizontal pressure-gradient force to
    # the *coupled* large-step ru/rv_tend (module_em.F:1325 ->
    # horizontal_pressure_gradient, module_big_step_utilities_em.F:2459-2466).
    # This is the steady gradient that drives the mean circulation; it is a
    # DIFFERENT split term from the small-step advance_uv acoustic PGF
    # (module_small_step_em.F:828-868), which uses the work-array perturbation
    # pressure that restarts ~0 at each RK stage -- NOT a double-count.  The
    # operational cadence applies ru/rv_tend only inside advance_uv (one
    # forward-Euler per acoustic substep, u += dts*ru_tend), matching WRF; the
    # earlier add_scaled_tendencies forward-Euler of the dynamics fields has been
    # removed so there is no double-application.
    ru_pgf, rv_pgf = large_step_horizontal_pgf(
        haloed,
        metrics,
        dx_m=dx,
        dy_m=dy,
        non_hydrostatic=True,
        top_lid=bool(namelist.top_lid),
    )
    u_t = u_t + ru_pgf
    v_t = v_t + rv_pgf

    # WRF rk_tendency adds the Coriolis force to the SAME coupled ru/rv_tend
    # immediately AFTER the horizontal PGF (module_em.F:717 PGF then :761 coriolis;
    # body module_big_step_utilities_em.F:3640).  This is the rotational body force
    # that lets the interior flow reach geostrophic balance; its complete absence
    # was the proven root cause of the below-persistence, wrong-sign-u Canary winds
    # (proofs/wind/case3_v10_momentum_budget_findings.md).  ``f=0`` for idealized
    # cases makes every Coriolis term identically zero, so the warm-bubble / Straka
    # / oracle dycore gates stay bit-identical.  ``specified`` follows WRF's
    # nested/specified boundary edge-face exclusion for the real (boundary-driven)
    # case; for periodic idealized runs the choice is moot under f=0.
    ru_cor, rv_cor = large_step_coriolis(
        haloed,
        metrics,
        specified=bool(namelist.run_boundary),
    )
    u_t = u_t + ru_cor
    v_t = v_t + rv_cor

    tendencies = tendencies.replace(u=u_t, v=v_t, w=w_t, theta=th_t)

    # WRF rk_addtend_dry per-stage merge (module_em.F:1711-1786): field-specific
    # map/mass coupling of RK1-fixed non-timesplit physics tendencies.  Physics-off
    # and dry idealized gates pass an empty bundle, so this remains identity there.
    return rk_addtend_dry(
        tendencies,
        DryPhysicsTendencies() if physics_tendencies is None else physics_tendencies,
        rk_step=int(rk_step),
        metrics=metrics,
        mut=_base_mu(haloed),
    )


# WRF advects every moisture species (vapour + condensates) in the RK3 large
# step, in this Registry order (``moist_variable_loop`` over the moist array,
# solve_em.F:2282-2408).  The condensates that exist as State leaves in this port
# are qc/qr/qi/qs/qg; qv is index P_QV.
_MOISTURE_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")


def _moisture_coupled_tendencies(
    haloed: State,
    namelist: OperationalNamelist,
    *,
    rk_step: int,
    step_origin: State | None,
) -> tuple[jax.Array, ...]:
    """WRF moisture-species coupled large-step tendency ``d(mu*q)/dt`` per stage.

    Source: ``solve_em.F:2282-2408`` ``moist_variable_loop`` ->
    ``rk_scalar_tend(im, im, ..., config_flags%moist_adv_opt, ...)``.  Each moist
    species is flux-advected by the SAME high-order flux-form scalar advection
    (h=5/v=3) used for theta, with the PD/monotonic limiter (moist_adv_opt 1/2)
    applied ONLY on the final RK3 stage (the start-of-step ``step_origin``
    moisture / ``mu_old`` feed the FCT bound) -- identical cadence to the theta
    ``scalar_adv_opt`` wiring in ``_augment_large_step_tendencies``.

    Reuses the EXACT same ``vel`` / ``mu_total`` / ``metrics`` build as the theta
    flux advection so the transporting velocity field is bit-consistent with the
    momentum/theta advection of the same stage.  Returns a tuple of COUPLED
    tendencies ``d(mu*q)/dt`` in ``_MOISTURE_SPECIES`` order, consumed by the WRF
    scalar large-step update ``q_new = (mu_old*q_old + dt_rk*tend)/mu_new`` AFTER
    the acoustic loop (NOT inside the acoustic substeps).
    """

    metrics = namelist.metrics
    grid = namelist.grid
    dx = float(grid.projection.dx_m)
    dy = float(grid.projection.dy_m)
    mu_total = haloed.mu_total
    vel = couple_velocities_periodic(
        haloed.u,
        haloed.v,
        mu_total,
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        dnw=metrics.dnw,
        rdx=1.0 / dx,
        rdy=1.0 / dy,
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
        msftx=metrics.msftx,
        msfux=metrics.msfux,
        msfvy=metrics.msfvy,
    )
    fields = tuple(getattr(haloed, name) for name in _MOISTURE_SPECIES)
    # The limiter (moist_adv_opt 1/2) is the final-RK3-stage FCT; it needs the
    # start-of-step moisture (WRF ``moist_old``) and ``mu_old`` (grid%mu_1).  The
    # selection inside advect_moisture_scalars is STATIC, so on opt==0 / non-final
    # stages the plain h5/v3 path is emitted and ``fields_old`` is ignored.
    use_limiter = (
        int(namelist.moist_adv_opt) in (1, 2)
        and int(rk_step) == int(namelist.rk_order)
        and step_origin is not None
    )
    fields_old = (
        tuple(getattr(step_origin, name) for name in _MOISTURE_SPECIES)
        if use_limiter
        else None
    )
    mu_old = step_origin.mu_total if step_origin is not None else mu_total
    return advect_moisture_scalars(
        fields,
        fields_old,
        vel,
        moist_adv_opt=int(namelist.moist_adv_opt),
        is_final_rk_stage=(int(rk_step) == int(namelist.rk_order)),
        mut=mu_total,
        mu_old=mu_old,
        c1=metrics.c1h,
        c2=metrics.c2h,
        rdx=1.0 / dx,
        rdy=1.0 / dy,
        rdzw=metrics.rdnw,
        fzm=metrics.fnm,
        fzp=metrics.fnp,
        dt=float(namelist.dt_s),
    )


def _apply_moisture_large_step(
    state: State,
    step_origin: State,
    *,
    q_tendencies: tuple[jax.Array, ...],
    dt_rk: float,
    metrics: DycoreMetrics,
) -> State:
    """WRF scalar large-step update for moisture AFTER the acoustic loop.

    Source: ``solve_em.F`` ``rk_scalar_tend`` followed by the moist-scalar update
    ``moist_2 = (mu_1*moist_old + dt_rk*adv_tend) / mu_2`` (decouple by the UPDATED
    dry-air mass once the acoustic small-step loop has advanced ``mu``).

    WRF's RK3 low-storage scheme integrates EVERY stage from the START-OF-STEP
    reference: ``field_old`` / ``mu_1`` are the step-entry (``rk1_reference``)
    values, NOT the stage-entry values; ``dt_rk`` is the stage substep fraction of
    ``dt`` (dt/3, dt/2, dt).  This mirrors the theta cadence in
    ``small_step_finish_wrf`` (``theta = (theta_work + t_save*mass_current)/
    mass_stage`` with ``t_save`` = the rk1 reference theta).  The coupled tendency
    ``q_tendencies`` is ``d(mu*q)/dt`` built from the CURRENT-stage state by
    ``_moisture_coupled_tendencies``; ``step_origin`` supplies ``moist_old`` /
    ``mu_old`` and ``state.mu_total`` is the post-acoustic dry mass.

    WRF couples scalars with ``mut = c1*mu + c2`` (the same column mass weight the
    advection flux divergence uses), so we decouple with that SAME weight -- the
    update is consistent with the coupled tendency the flux kernels returned.

    The condensates qc/qr/qi/qs/qg are advected here for the FIRST time in the
    interior (previously they had ZERO resolved-wind transport anywhere); qv was
    previously only boundary-ring advected.  This adds horizontal + resolved-
    vertical transport of every species, matching WRF.
    """

    # Column dry-mass weights mut = c1h*mu + c2h on mass points (the scalar
    # coupling weight; matches the c1/c2 passed to advect_moisture_scalars).
    mass_old = metrics.c1h[:, None, None] * step_origin.mu_total[None, :, :] + metrics.c2h[:, None, None]
    mass_new = metrics.c1h[:, None, None] * state.mu_total[None, :, :] + metrics.c2h[:, None, None]
    inv_mass_new = 1.0 / mass_new
    updates: dict[str, jax.Array] = {}
    for name, q_tend in zip(_MOISTURE_SPECIES, q_tendencies):
        q_old = getattr(step_origin, name)
        # q_new = (mut_old*moist_old + dt_rk*adv_tend) / mut_new (WRF scalar update).
        q_new = (mass_old * q_old + float(dt_rk) * q_tend) * inv_mass_new
        updates[name] = q_new
    return state.replace(**updates)


def _rk_scan_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    debug: bool = False,
    lead_seconds=None,
    physics_tendencies: DryPhysicsTendencies | None = None,
) -> OperationalCarry:
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin

    def advance_stage(stage_carry: OperationalCarry, stage: _RKStageDescriptor) -> OperationalCarry:
        haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        # WRF rk_tendency builds the per-stage large-step tendencies (advection,
        # diffusion, and the LARGE-STEP horizontal PGF; module_em.F:1325) and
        # rk_addtend_dry merges the RK1-fixed physics tendencies; both are inside
        # _augment_large_step_tendencies.  The large-step momentum tendency is
        # consumed ONLY inside the acoustic small-step advance_uv (one
        # forward-Euler per substep: u += dts*ru_tend, module_small_step_em.F:805),
        # exactly as WRF does -- there is no separate add_scaled_tendencies
        # forward-Euler of the dynamics prognostics (that was the Sprint A/B
        # double-application that prevented u/v from moving).  The stage-entry
        # physical state (carried forward across RK stages) is the small-step
        # prognostic ``u_2``; ``rk1_reference`` is the RK reference ``u_1``.
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        tendencies = _augment_large_step_tendencies(
            haloed,
            tendencies,
            namelist,
            rk_step=int(stage.rk_step),
            physics_tendencies=physics_tendencies,
            step_origin=rk1_reference,
        )
        # WRF advances moisture in the LARGE step (NOT the acoustic substeps):
        # build the coupled moisture tendency d(mu*q)/dt for THIS stage from the
        # stage-entry haloed state (same transporting ``vel`` as theta/momentum),
        # then apply the WRF scalar update q_new=(mu_old*q_old+dt_rk*tend)/mu_new
        # AFTER the acoustic loop has advanced ``mu``.  The branch is a STATIC
        # Python condition (moist_adv_opt and use_flux_advection are compile-time
        # constants), so when moisture advection is OFF (the default) the new code
        # path is never traced and the operational program is byte-for-byte
        # unchanged.  Source: solve_em.F:2282-2408 moist_variable_loop.
        moisture_advected = (
            bool(namelist.use_flux_advection) and int(namelist.moist_adv_opt) != 0
        )
        q_tendencies = (
            _moisture_coupled_tendencies(
                haloed,
                namelist,
                rk_step=int(stage.rk_step),
                step_origin=rk1_reference,
            )
            if moisture_advected
            else None
        )
        candidate = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        prep = small_step_prep_wrf(
            candidate,
            int(stage.rk_step),
            float(stage.dt_rk),
            metrics=namelist.metrics,
            reference_state=rk1_reference,
            ww=stage_carry.ww,
        )
        pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
        stage_carry = _acoustic_scan(
            stage_carry.replace(state=candidate),
            namelist,
            stage=stage,
            prep=prep,
            pressure=pressure,
            tendencies=tendencies,
            lead_seconds=lead_seconds,
        )
        if moisture_advected:
            stage_carry = stage_carry.replace(
                state=_apply_moisture_large_step(
                    stage_carry.state,
                    rk1_reference,
                    q_tendencies=q_tendencies,
                    dt_rk=float(stage.dt_rk),
                    metrics=namelist.metrics,
                )
            )
        return stage_carry.replace(state=apply_halo(stage_carry.state, halo_spec(namelist.grid)))

    # Static RK sequencing avoids per-stage scalar dispatch inside the profiled
    # timestep loop. WRF solve_em.F:1472-1479 runs one RK1 acoustic small step
    # and half the configured sound steps for RK2.
    # Legacy test anchor for the prior dynamic form:
    # lambda value: advance_stage(value, 1.0 / 3.0, 1)
    if debug:
        jax.debug.print("GPUWRF_M6B_RK1_ACOUSTIC_LOOP_ENTER substeps=1")
    dt = float(namelist.dt_s)
    configured_sound_steps = int(namelist.acoustic_substeps)
    stages = (
        _RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
        _RKStageDescriptor(2, 0.5 * dt, dt / float(configured_sound_steps), max(1, configured_sound_steps // 2)),
        _RKStageDescriptor(3, dt, dt / float(configured_sound_steps), configured_sound_steps),
    )
    carry = carry.replace(state=origin)
    carry = advance_stage(carry, stages[0])
    carry = advance_stage(carry, stages[1])
    return advance_stage(carry, stages[2])


def _coupled_core_extras(state: State) -> dict[str, jax.Array]:
    return {
        "qv": state.qv,
        "qc": state.qc,
        "qr": state.qr,
        "qi": state.qi,
        "qs": state.qs,
        "qg": state.qg,
        "qke": state.qke,
        "t_skin": state.t_skin,
        "xland": state.xland,
        "lakemask": state.lakemask,
        "lu_index": state.lu_index,
        "u_bdy": state.u_bdy,
        "v_bdy": state.v_bdy,
        "theta_bdy": state.theta_bdy,
        "qv_bdy": state.qv_bdy,
        "ph_bdy": state.ph_bdy,
        "mu_bdy": state.mu_bdy,
    }


def _state_from_coupled_core(snapshot: dict[str, jax.Array], template: State, theta_offset: jax.Array, dt_s: float) -> State:
    theta = jnp.asarray(snapshot["theta"]) + theta_offset
    p_pert = jnp.asarray(snapshot["p"])
    ph_pert = jnp.asarray(snapshot["ph"])
    mu_pert = jnp.asarray(snapshot["mu"])
    p_total = template.p_total - template.p_perturbation + p_pert
    ph_total = template.ph_total - template.ph_perturbation + ph_pert
    mu_total = template.mu_total - template.mu_perturbation + mu_pert
    return template.replace(
        u=jnp.asarray(snapshot["u"]),
        v=jnp.asarray(snapshot["v"]),
        w=jnp.asarray(snapshot["w"]),
        theta=theta,
        qv=template.qv + jnp.asarray(snapshot["qv_phys_tend"]) * float(dt_s),
        qc=template.qc + jnp.asarray(snapshot["qc_phys_tend"]) * float(dt_s),
        qr=template.qr + jnp.asarray(snapshot["qr_phys_tend"]) * float(dt_s),
        qi=template.qi + jnp.asarray(snapshot["qi_phys_tend"]) * float(dt_s),
        qs=template.qs + jnp.asarray(snapshot["qs_phys_tend"]) * float(dt_s),
        qg=template.qg + jnp.asarray(snapshot["qg_phys_tend"]) * float(dt_s),
        qke=template.qke + jnp.asarray(snapshot["qke_phys_tend"]) * float(dt_s),
        p=p_total,
        p_total=p_total,
        p_perturbation=p_pert,
        ph=ph_total,
        ph_total=ph_total,
        ph_perturbation=ph_pert,
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=mu_pert,
    )


def _carry_from_coupled_core(snapshot: dict[str, jax.Array], template: State, theta_offset: jax.Array, dt_s: float, *, rthraten: jax.Array | None = None) -> OperationalCarry:
    next_state = _state_from_coupled_core(snapshot, template, theta_offset, float(dt_s))
    return OperationalCarry(
        state=next_state,
        t_2ave=jnp.asarray(snapshot["t_2ave"]) + theta_offset,
        ww=jnp.asarray(snapshot["ww"]),
        mudf=jnp.asarray(snapshot["mudf"]),
        muave=jnp.asarray(snapshot["muave"]),
        muts=jnp.asarray(snapshot["muts"]),
        ph_tend=jnp.asarray(snapshot["ph_tend"]),
        u_save=next_state.u,
        v_save=next_state.v,
        w_save=next_state.w,
        t_save=next_state.theta,
        ph_save=next_state.ph,
        mu_save=jnp.asarray(snapshot["mu"]),
        ww_save=jnp.asarray(snapshot["ww"]),
        # Preserve the held radiative theta tendency across the coupled core
        # (it is refreshed in the physics chain, not the dycore core).
        rthraten=jnp.zeros_like(next_state.theta) if rthraten is None else rthraten,
    )


def _coupled_core_step(carry: OperationalCarry, namelist: OperationalNamelist, step_index) -> OperationalCarry:
    acoustic = _acoustic_core_state(carry, namelist)
    theta_offset = _theta_base_offset(carry.state.theta)
    periodic_x, specified, nested = _acoustic_lateral_bc_flags(namelist)
    snapshot = coupled_timestep_core(
        acoustic,
        namelist.metrics,
        CoupledCoreConfig(
            dt=float(namelist.dt_s),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            acoustic_substeps=int(namelist.acoustic_substeps),
            rk_order=int(namelist.rk_order),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            physics_enabled=True,
            boundary_enabled=True,
            boundary_config=namelist.boundary_config,
            periodic_x=periodic_x,
            specified=specified,
            nested=nested,
        ),
        extras=_coupled_core_extras(carry.state),
        step_index=step_index,
    )
    return _carry_from_coupled_core(snapshot, carry.state, theta_offset, float(namelist.dt_s), rthraten=carry.rthraten)


class _NoahMPClock(NamedTuple):
    """Phenology clock the Noah-MP forcing assembler reads (julian / yearlen)."""

    julian: float
    yearlen: float


def noahmp_initial_rad(
    state: State,
    namelist: "OperationalNamelist | None" = None,
    *,
    land_state=None,
) -> tuple:
    """Seed the held Noah-MP surface-radiation forcing as a CONCRETE 3-tuple.

    The held forcing rides in the OperationalCarry; inside ``jax.lax.scan`` the
    carry pytree structure must be identical on every iteration, so the initial
    held value must already be the 3-tuple shape the step produces -- NOT ``None``.

    When ``namelist`` is given, the seed is the REAL t=0 surface radiation
    (SOLDN/LWDN/COSZ from RRTMG at the init instant), computed ONCE eagerly. This
    matters at an evening (18z) init: zero-seeding LWDN would starve the land of
    downward longwave for the first radt interval and drive a spurious nocturnal
    cold bias. WRF holds the radiative forcing from the first radiation call, so
    seeding the real t=0 value is the WRF-faithful initial held forcing. Without a
    namelist (legacy callers) the seed is zeros (overwritten at the first radt step).
    """
    if namelist is None:
        zero = jnp.zeros(state.t_skin.shape, dtype=jnp.float64)
        return (zero, zero, zero)
    rad = rrtmg_radiation_diagnostics(
        state,
        namelist.grid,
        time_utc=namelist.time_utc,
        lead_seconds=0.0,
        radiation_static=namelist.radiation_static,
        topo_shading=int(namelist.topo_shading),
        slope_rad=int(namelist.slope_rad),
        shadow_length_m=float(namelist.topo_shadow_length_m),
        land_state=land_state,
    )
    soldn = jnp.maximum(jnp.asarray(rad.swnorm, dtype=jnp.float64), 0.0)
    lwdn = jnp.asarray(rad.glw, dtype=jnp.float64)
    cosz = jnp.asarray(rad.coszen, dtype=jnp.float64)
    if int(namelist.ra_sw_physics) == 0:
        soldn = jnp.zeros_like(soldn)
    if int(namelist.ra_lw_physics) == 0:
        lwdn = jnp.zeros_like(lwdn)
    return (soldn, lwdn, cosz)


def _noahmp_params(namelist: OperationalNamelist):
    """Return the pre-built ``(energy_params, rad_params)``. They ride as STATIC AUX
    (compile constants), so their concrete ``nroot``/scalar fields are available
    inside the jitted scan -- no re-build, no tracer concretization. ``(None, None)``
    when not pre-built (the driver then builds them eagerly, valid outside jit)."""
    return namelist.noahmp_energy_params, namelist.noahmp_rad_params


# v0.6.0 scan-wire (2026-06-03): the operational scan now routes the genuinely
# jit/vmap-traceable new schemes through the dispatcher into the GPU scan path, in
# WRF call order, alongside the v0.2.0 validated suite. Wired = each option that
# maps to a State<->scheme adapter in coupling.scan_adapters (or the existing
# coupling.physics_couplers adapters). The remaining schemes are kept FAIL-CLOSED
# (loud) here -- they passed per-scheme savepoint parity but cannot ride the device
# scan as-is for SCHEME-SPECIFIC reasons (host-NumPy single-column kernels needing
# a jit/vmap rewrite, or missing required per-run land inputs), documented per option in
# _SCAN_UNWIRED_REASON. The dispatcher (coupling.physics_dispatch) remains the
# single fail-closed authority for option -> scheme + GPU-runnability.
_SCAN_WIRED_OPTIONS = {
    # mp=0 passive, 8 Thompson (existing couplers); 1/2/3/4/6/10/14/16 new scan adapters.
    "mp_physics": (0, 1, 2, 3, 4, 6, 8, 10, 14, 16),
    # bl=0 off, 5 MYNN (existing); 1 YSU / 7 ACM2 / 8 BouLac wired
    # (v0.6.0 jax.lax.scan rewrites); 2 MYJ wired (v0.13 traceable MYJ+Janjic pair);
    # 99 MRF wired (v0.13 jit/vmap-traceable port of phys/module_bl_mrf.F).
    "bl_pbl_physics": (0, 1, 2, DEFAULT_BL_PBL_PHYSICS, 7, 8, 99),
    # sf_sfclay=0 off, 5 MYNN-sfclay (existing); 1 revised-MM5 / 7 Pleim-Xiu wired;
    # 2 Janjic Eta wired (v0.13, mandatorily paired with bl_pbl_physics=2 MYJ).
    # 3 NCEP-GFS surface layer + 91 old-MM5 surface layer wired (v0.13 Tier-3,
    # coupling.scan_adapters.{gfs_sfclay_adapter,sfclay_old_mm5_adapter}; both write
    # the B2 kinematic flux handles, fp64 pristine-WRF oracle-validated).
    "sf_sfclay_physics": (0, 1, 2, 3, 5, 7, 91),
    # cu=0 no cumulus, 1 KF, 2 BMJ (fp64 savepoint-parity carry-threaded adapter),
    # 3 Grell-Freitas (v0.9.0 GPU-batched jit/vmap stateless adapter), 6 modified-
    # Tiedtke (v0.6.0 GPU-batched jit/vmap adapter). New-Tiedtke(16) not separately
    # gated -> NOT wired.
    "cu_physics": (0, 1, 2, 3, 6),
    # ra_sw=0 disabled, 4 RRTMG SW (default), 1 Dudhia SW (Stephens-1984, scan-wired held-rate
    # theta tendency via dudhia_sw_theta_tendency), 2 GSFC/Chou-Suarez SW
    # (multi-band delta-Eddington, scan-wired held-rate theta tendency via
    # gsfc_sw_theta_tendency). Any other recognized SW scheme is fail-closed
    # (no GPU scan adapter).
    "ra_sw_physics": (0, 1, 2, 4),
    # ra_lw=0 disabled, 4 RRTMG LW (default), 1 classic AER RRTM LW (16-band k-distribution,
    # scan-wired held-rate theta tendency via rrtm_lw_theta_tendency, JAX-traceable
    # port of phys/module_ra_rrtm.F). SW/LW are selected independently.
    "ra_lw_physics": (0, 1, 4),
}

# Scheme-specific reasons a parity-passed option is NOT yet wired into the scan
# (surfaced in the fail-closed error so the rejection is honest + actionable).
_SCAN_UNWIRED_REASON = {
    # YSU(1)/ACM2(7) are now jax.lax.scan-traceable + scan-wired (v0.6.0 GPU-op).
    # MYJ(2)/Janjic(2) are now traceable + scan-wired as a mandatory pair (v0.13):
    # physics.myj_adapters.{myj_pbl_adapter,janjic_sfclay_adapter}, so they are
    # intentionally absent here.
    # cu=3 (Grell-Freitas) and cu=6 (modified Tiedtke) are now GPU-batched +
    # scan-wired (in _SCAN_WIRED_OPTIONS), so they are intentionally absent here.
    "cu_physics=16": "New Tiedtke is interface-compatible but not separately savepoint-gated by a distinct WRF source path; GPU-batching/gating TODO",
    # v0.13 Tier-3 cumulus: KSAS(14)/Grell-3D(5) have single-column fp64
    # pristine-WRF oracles staged (proofs/v013/oracle/cumulus); their traceable
    # JAX column kernels are a documented carry-over, so they fail-close here.
    "cu_physics=14": "KIM-SAS has a single-column fp64 pristine-WRF oracle staged (proofs/v013); traceable JAX column kernel is a Tier-3 carry-over",
    "cu_physics=5": "Grell-3D ensemble has a single-column fp64 pristine-WRF oracle staged (proofs/v013); traceable JAX column kernel is a Tier-3 carry-over",
    "sf_surface_physics=2": "Noah-classic requires explicit noahclassic_static + noahclassic_land bundles (WRF REDPRM + 4-layer carry)",
    "sf_surface_physics=1": "thermal-diffusion slab LSM is JAX-ported + fp64 oracle-validated (physics.lsm_slab) but the operational LSM hook (TSLB land carry + GSW/GLW radiation forcing + TMN/THC/EMISS statics) is not yet threaded into the scan",
    # ra_sw=1 (Dudhia), ra_sw=2 (GSFC/Chou-Suarez) and ra_sw=4 (RRTMG) are
    # scan-wired; any other recognized SW scheme has no operational GPU scan
    # adapter in the radiation slot.
    # ra_lw=5 (GSFC/Goddard NUWRF LW) is v0.13 Tier-3 reference-only: a fp64
    # single-column pristine-WRF oracle is staged (module_ra_goddard.F:lwrad,
    # proofs/v013/oracle/radiation_lw); its traceable JAX column kernel is a
    # documented carry-over (the combined NUWRF SW+LW module is ~12.5k LOC), so it
    # fail-closes here. ra_lw=4 (RRTMG) and 1 (classic RRTM) remain the operational LW.
    "ra_lw_physics=5": "GSFC/Goddard NUWRF longwave has a single-column fp64 pristine-WRF oracle staged (proofs/v013/oracle/radiation_lw, module_ra_goddard.F:lwrad); traceable JAX column kernel is a Tier-3 carry-over (~12.5k-LOC combined NUWRF SW+LW module)",
}


def _explicit_noahclassic(namelist: OperationalNamelist) -> bool:
    explicit_land = getattr(namelist, "sf_surface_physics", None)
    return explicit_land is not None and int(explicit_land) == 2


def _resolve_operational_suite(namelist: OperationalNamelist):
    """Fail-closed resolve + validate the selected physics suite for the scan.

    Resolves the namelist's physics options through the dispatcher (which rejects
    anything outside the frozen S0 accept-matrix), then asserts the selection is
    one whose State adapter is threaded into THIS operational scan. Schemes that
    passed per-scheme parity but cannot ride the device scan as-is raise loudly
    here (with a scheme-specific reason) rather than being silently ignored.
    """

    suite = resolve_physics_suite(namelist)  # fail-closed on out-of-matrix options
    not_wired: list[str] = []
    for key, wired in _SCAN_WIRED_OPTIONS.items():
        selected = int(getattr(namelist, key))
        if selected not in wired:
            tag = f"{key}={selected}"
            reason = _SCAN_UNWIRED_REASON.get(tag)
            not_wired.append(f"{tag} ({reason})" if reason else tag)
    # Land surface: the scan threads Noah-MP (use_noahmp=True), explicit
    # Noah-classic (sf_surface_physics=2 + WRF-derived land/static bundle), or the
    # legacy bulk surface path. NOTE the dispatcher maps the legacy
    # ``use_noahmp=False`` toggle to land option 2, but in THIS scan that still
    # means the bulk path unless sf_surface_physics is explicitly pinned to 2.
    land_opt = suite.land_surface.option
    if land_opt == 4 and not bool(namelist.use_noahmp):
        not_wired.append("sf_surface_physics=4 (set use_noahmp=True to thread Noah-MP)")
    # slab=1 (thermal-diffusion 5-layer LSM) is JAX-ported + fp64 oracle-validated
    # (physics.lsm_slab) but NOT scan-wired: the operational LSM slot needs the
    # TSLB land carry + GSW/GLW radiation forcing + TMN/THC/EMISS statics that the
    # resident State does not yet carry. Fail closed (reference-only) rather than
    # silently running the bulk surface path under a slab namelist selection.
    if land_opt == 1:
        not_wired.append(f"sf_surface_physics=1 ({_SCAN_UNWIRED_REASON['sf_surface_physics=1']})")
    if _explicit_noahclassic(namelist):
        if getattr(namelist, "noahclassic_static", None) is None or getattr(namelist, "noahclassic_land", None) is None:
            not_wired.append(f"sf_surface_physics=2 ({_SCAN_UNWIRED_REASON['sf_surface_physics=2']})")
    if not_wired:
        raise UnsupportedSchemeSelection(
            "operational scan supports the v0.2.0 suite + the v0.6.0/v0.13 scan-wired "
            "schemes (mp_physics in {0,1,2,3,4,6,8,10,14,16}, bl_pbl_physics in {0,1,2,5,7,8,99}, "
            "sf_sfclay_physics in {0,1,2,3,5,7,91}, cu_physics in {0,1,2,3,6}, Noah-MP via "
            "use_noahmp, explicit Noah-classic via sf_surface_physics=2 plus "
            "noahclassic_static/noahclassic_land). The following selected schemes "
            "are NOT scan-wired: "
            f"{'; '.join(not_wired)}"
        )
    return suite


def _initial_carry_for_run(state: State, namelist: OperationalNamelist) -> OperationalCarry:
    """Build the initial operational carry, seeding any scheme-specific sub-carry.

    Centralizes carry construction for the public forecast entries so a stateful
    scan-wired scheme's persistent carry is seeded to its CONCRETE pytree shape
    BEFORE the scan (``jax.lax.scan`` requires a carry pytree that is identical on
    every iteration; a ``None``->tuple promotion inside the body would be rejected).
    The v0.6.0 cumulus carry is seeded when ``cu_physics`` selects a stateful
    scan-wired cumulus option (KF ``(w0avg,nca)`` or BMJ ``cldefi``); otherwise
    ``cumulus_carry`` stays ``None`` and the carry is structurally identical to
    the pre-v0.6.0 carry.
    """

    enforced = _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
    cumulus_carry = None
    # Only the STATEFUL cumulus adapters need a persistent carry: KF (1) threads
    # (w0avg, nca); BMJ (2) threads CLDEFI. The stateless GPU-batched adapter
    # (Tiedtke, 6) keeps cumulus_carry None.
    cu_opt = int(namelist.cu_physics)
    if cu_opt == 1:
        cumulus_carry = initial_kf_carry(enforced)
    elif cu_opt == 2:
        cumulus_carry = initial_bmj_carry(enforced)
    noahclassic_land = None
    noahclassic_rad = None
    if _explicit_noahclassic(namelist):
        noahclassic_land = namelist.noahclassic_land
        noahclassic_rad = (
            namelist.noahclassic_rad
            if getattr(namelist, "noahclassic_rad", None) is not None
            else NoahClassicRadiation(*noahmp_initial_rad(enforced, namelist))
        )
    return initial_operational_carry(
        enforced,
        cumulus_carry=cumulus_carry,
        noahclassic_land=noahclassic_land,
        noahclassic_rad=noahclassic_rad,
    )


def _operational_device():
    """Return the device used to commit operational host-loop carries."""

    devices = jax.devices()
    for device in devices:
        if device.platform == "gpu":
            return device
    return devices[0]


def _commit_to_operational_device(value):
    """Commit all array leaves to one explicit device for stable JIT cache keys."""

    return jax.device_put(value, _operational_device())


def _dealias_pytree_buffers(tree):
    """Return ``tree`` with every leaf that shares a buffer made a distinct copy.

    JAX ``donate_argnums`` flattens the donated pytree and requires every leaf to
    back a UNIQUE device buffer; if two leaves alias the same buffer (e.g. the
    transitional ``p``/``p_total`` legacy aliases that ``State.replace`` keeps in
    lockstep), the donate path raises "Attempt to donate the same buffer twice".
    This walks the leaves, keys them by their concrete buffer identity, and rebinds
    any duplicate to ``leaf + 0`` (a fresh buffer; numerically identical, no dtype
    change). Tracers (under jit) have no stable identity, so this is a no-op there
    -- it only matters for the concrete host/device arrays passed at call time.
    """

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    seen: set[int] = set()
    out = []
    for leaf in leaves:
        buf = getattr(leaf, "unsafe_buffer_pointer", None)
        key = None
        if callable(buf):
            try:
                key = int(buf())
            except Exception:  # noqa: BLE001 - not a concrete single-device array
                key = None
        if key is None:
            key = id(leaf)
        if key in seen:
            out.append(leaf + 0)  # distinct buffer; identical value/dtype
        else:
            seen.add(key)
            out.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, out)


def dealias_state_buffers(state: State) -> State:
    """Public donate-safety helper: de-alias a State's shared device buffers.

    Call this on any State built with ``State.replace`` legacy-alias updates
    (``p=p_total`` etc.) before handing it to a ``donate_argnums`` forecast entry.
    """

    return _dealias_pytree_buffers(state)


def _committed_initial_carry_for_run(state: State, namelist: OperationalNamelist) -> OperationalCarry:
    """Build the first chunk carry with the same device commitment as chunk outputs.

    ``_advance_chunk`` returns device-committed leaves. If the first call receives
    host/uncommitted leaves and the second call receives the prior chunk's committed
    output, JAX treats their shardings as different cache keys and recompiles an
    otherwise identical segment. Commit once before entering chunked host loops.
    """

    return _commit_to_operational_device(_initial_carry_for_run(state, namelist))


class _NoahMPRadiation(NamedTuple):
    """Held surface-radiation forcing into Noah-MP (the coupler reads soldn/lwdn/cosz)."""

    soldn: jax.Array
    lwdn: jax.Array
    cosz: jax.Array


def _refresh_noahmp_rad(state, namelist, lead_seconds, run_radiation, held_rad, *, land_state=None):
    """Refresh the HELD Noah-MP surface radiation (SOLDN/LWDN/COSZ) at the radiation
    cadence; reuse the held value between calls (WRF holds the radiative forcing
    between radt intervals). Resident on device -- no host transfer.

    ``held_rad`` is the prior (soldn, lwdn, cosz) 3-tuple, or ``None`` at t=0.
    Returns the (soldn, lwdn, cosz) 3-tuple for this step.

    WRF radiation-held-time (L1 fix, GPT 2026-06-02 COSZEN-phase diagnosis;
    proofs/rad_time/coszen_phase_proof.json). WRF holds the SWDOWN computed once
    per ``radt`` interval and the HISTORY OUTPUT carries the field held GOING INTO
    the output step -- i.e. the value last set at the PRECEDING interval midpoint.
    Empirically (GPT + this proof) WRF's land-mean held SWDOWN at output time ``t``
    tracks ``coszen(t - radt/2)``: the observed d03 GPU/WRF residual (1.0869 @09z,
    1.0158 @12z, 0.9704 @15z) matches ``coszen(t)/coszen(t - radt/2)`` to ~0.5% and
    is the OPPOSITE of ``coszen(t)/coszen(t + radt/2)``.

    SIGN DERIVATION (the sprint's load-bearing trap -- do NOT blindly copy WRF's
    ``xtime + radt*0.5`` PLUS): the radiation refresh fires here on cadence steps
    (``step_index %% cadence == 0``), and history output lands on a refresh boundary
    (history_interval is a multiple of radt), so at the output step the incoming
    ``lead_seconds = step_index*dt_s`` EQUALS the output time ``t``. WRF's own
    ``calc_coszen(..., xtime + radt*0.5, ...)`` is PLUS because WRF's ``xtime`` is
    the interval START and it samples the FORWARD midpoint -- but WRF then OUTPUTS
    the field held from the PRIOR interval (end-of-step output ordering), so the
    history value at ``t`` is ``coszen((t - radt) + radt/2) = coszen(t - radt/2)``.
    Our scan refreshes the held tuple IN the output step and the snapshot reads it
    immediately, so to land on the SAME absolute solar time WRF reports we offset
    the refresh lead by ``- radt/2``: ``lead_seconds - 0.5*radt_seconds`` (MINUS).
    ``radt_seconds = dt_s * radiation_cadence_steps``. Clamped to >= 0 at cold start.

    NOTE for the GPU remeasure (handed to the manager): the residual only FULLY
    collapses (proof: max 0.44%) when the GPU ``radiation_cadence_steps`` is chosen
    so ``radt = dt_s*radiation_cadence_steps/60 == 30 min`` (the pristine-WRF
    namelist radt). At a mismatched cadence the ``-radt/2`` offset is the wrong
    magnitude (proof: 10-min radt leaves ~5.7%).
    """

    radt_seconds = float(namelist.dt_s) * int(namelist.radiation_cadence_steps)
    rad_lead_seconds = jnp.maximum(
        jnp.asarray(lead_seconds, dtype=jnp.float64) - 0.5 * radt_seconds, 0.0
    )

    def _recompute(_unused):
        rad = rrtmg_radiation_diagnostics(
            state,
            namelist.grid,
            time_utc=namelist.time_utc,
            lead_seconds=rad_lead_seconds,
            radiation_static=namelist.radiation_static,
            topo_shading=int(namelist.topo_shading),
            slope_rad=int(namelist.slope_rad),
            shadow_length_m=float(namelist.topo_shadow_length_m),
            land_state=land_state,
        )
        soldn = jnp.maximum(jnp.asarray(rad.swnorm, dtype=jnp.float64), 0.0)
        lwdn = jnp.asarray(rad.glw, dtype=jnp.float64)
        cosz = jnp.asarray(rad.coszen, dtype=jnp.float64)
        if int(namelist.ra_sw_physics) == 0:
            soldn = jnp.zeros_like(soldn)
        if int(namelist.ra_lw_physics) == 0:
            lwdn = jnp.zeros_like(lwdn)
        return (soldn, lwdn, cosz)

    # ``held_rad`` is always a concrete 3-tuple inside the scan (seeded at carry
    # construction by ``noahmp_initial_rad``), so the carry pytree structure is
    # stable across scan iterations -- never None here.
    if isinstance(run_radiation, bool):
        return _recompute(None) if run_radiation else held_rad
    return jax.lax.cond(run_radiation, _recompute, lambda _u: held_rad, None)


def _dry_physics_tendencies_from_state_delta(
    before: State,
    after: State,
    namelist: OperationalNamelist,
    dt_s: float,
) -> DryPhysicsTendencies:
    """Build WRF ``*_tendf`` leaves from one non-timesplit physics pass.

    The current physics adapters return already-integrated ``State`` deltas, not
    the raw WRF ``R*TEN`` source tendencies that ``calculate_phy_tend`` mass-couples
    before ``rk_addtend_dry``.  Treating aggregate state deltas as RK-fixed dry
    tendencies changes the thermal forcing cadence and regresses the d02 wind
    skill.  Keep this bridge empty until a scheme exposes true WRF ``*_tendf``
    leaves; the dry state deltas are applied after the dycore by
    ``_apply_physics_non_dry_updates``.
    """

    del before, after, namelist, dt_s
    return DryPhysicsTendencies()


def _apply_physics_non_dry_updates(
    dynamics_state: State,
    physics_reference: State,
    physics_state: State,
) -> State:
    """Apply physics prognostics that are not consumed by ``rk_addtend_dry``."""

    updates = {
        name: getattr(dynamics_state, name) + (getattr(physics_state, name) - getattr(physics_reference, name))
        for name in _PHYSICS_NON_DRY_INCREMENT_FIELDS
    }
    updates.update({name: getattr(physics_state, name) for name in _PHYSICS_NON_DRY_REPLACE_FIELDS})
    return dynamics_state.replace(**updates)


def _physics_step_forcing(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    lead_seconds,
    *,
    run_radiation: bool,
) -> _PhysicsStepForcing:
    """Run non-timesplit physics at step entry and expose RK-fixed tendencies."""

    if not bool(namelist.run_physics):
        return _PhysicsStepForcing(carry.state, carry, DryPhysicsTendencies(), False)

    before = carry.state
    next_state = before
    next_carry = carry

    mp_opt = int(namelist.mp_physics)
    sf_opt = int(namelist.sf_sfclay_physics)
    cu_opt = int(namelist.cu_physics)

    # --- microphysics slot ---
    if mp_opt == DEFAULT_MP_PHYSICS:
        next_state = thompson_adapter(next_state, float(namelist.dt_s))
    elif mp_opt in MP_SCAN_ADAPTERS:
        next_state = MP_SCAN_ADAPTERS[mp_opt](next_state, float(namelist.dt_s), namelist.grid)
    # mp_opt == 0 -> passive (no microphysics).

    # --- surface-layer / land slot ---
    # sf=2 Janjic Eta is the v0.13 traceable MYJ-pair surface layer (defined in
    # physics.myj_adapters, NOT in coupling.scan_adapters); route it explicitly.
    if bool(namelist.use_noahmp):
        if sf_opt == 2:
            next_state = janjic_sfclay_adapter(next_state, float(namelist.dt_s), namelist.grid)
        elif sf_opt in SFCLAY_SCAN_ADAPTERS:
            next_state = SFCLAY_SCAN_ADAPTERS[sf_opt](next_state, float(namelist.dt_s), namelist.grid)
        next_carry_rad = _refresh_noahmp_rad(
            next_state,
            namelist,
            lead_seconds,
            run_radiation,
            carry.noahmp_rad,
            land_state=carry.noahmp_land,
        )
        clock = _NoahMPClock(
            julian=float(namelist.noahmp_julian),
            yearlen=float(namelist.noahmp_yearlen),
        )
        radiation = _NoahMPRadiation(*next_carry_rad)
        ep, rp = _noahmp_params(namelist)
        next_state, next_land = noahmp_surface_step(
            next_state, carry.noahmp_land, namelist.noahmp_static,
            float(namelist.dt_s), radiation=radiation, clock=clock,
            energy_params=ep, rad_params=rp,
        )
        next_carry = next_carry.replace(noahmp_land=next_land, noahmp_rad=next_carry_rad)
    else:
        if sf_opt == 2:
            next_state = janjic_sfclay_adapter(next_state, float(namelist.dt_s), namelist.grid)
        elif sf_opt in SFCLAY_SCAN_ADAPTERS:
            next_state = SFCLAY_SCAN_ADAPTERS[sf_opt](next_state, float(namelist.dt_s), namelist.grid)
        else:
            next_state = surface_adapter(next_state, float(namelist.dt_s))
        if _explicit_noahclassic(namelist):
            next_noahclassic_rad = _refresh_noahmp_rad(
                next_state, namelist, lead_seconds, run_radiation, carry.noahclassic_rad
            )
            next_state, next_noahclassic_land = noahclassic_surface_step(
                next_state,
                carry.noahclassic_land,
                namelist.noahclassic_static,
                float(namelist.dt_s),
                radiation=NoahClassicRadiation(*next_noahclassic_rad),
            )
            next_carry = next_carry.replace(
                noahclassic_land=next_noahclassic_land,
                noahclassic_rad=next_noahclassic_rad,
            )

    # --- PBL slot ---
    # bl=2 MYJ is the v0.13 traceable MYJ PBL (paired with the Janjic surface
    # layer already run in the surface slot); it re-derives the surface coupling
    # and threads the TKE carry via qke. Defined in physics.myj_adapters.
    bl_opt = int(namelist.bl_pbl_physics)
    if bl_opt == 2:
        next_state = myj_pbl_adapter(next_state, float(namelist.dt_s), namelist.grid)
    elif bl_opt in PBL_SCAN_ADAPTERS:
        next_state = PBL_SCAN_ADAPTERS[bl_opt](next_state, float(namelist.dt_s), namelist.grid)
    elif bl_opt == DEFAULT_BL_PBL_PHYSICS:
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
    # bl_opt == 0 -> no PBL mixing.

    # --- orographic gravity-wave drag slot (gwd_opt=1) ---
    # WRF applies GWDO inside the PBL driver, right after the PBL momentum
    # tendency (phys/module_pbl_driver.F). gwd_opt=1 + a per-run GWDOStatics
    # bundle activates the faithful bl_gwdo_run port; otherwise it is a no-op.
    if int(namelist.gwd_opt) == 1 and namelist.gwdo_statics is not None:
        next_state = gwdo_adapter(
            next_state, float(namelist.dt_s), namelist.gwdo_statics, namelist.grid
        )

    # --- cumulus slot ---
    if cu_opt in CU_STATELESS_SCAN_ADAPTERS:
        next_state = CU_STATELESS_SCAN_ADAPTERS[cu_opt](
            next_state, float(namelist.dt_s), namelist.grid
        )
    elif cu_opt == 1:
        w0avg, nca = (
            carry.cumulus_carry if carry.cumulus_carry is not None
            else initial_kf_carry(next_state)
        )
        next_state, w0avg_next, nca_next = kf_adapter(
            next_state, float(namelist.dt_s), w0avg, nca, grid=namelist.grid
        )
        next_carry = next_carry.replace(cumulus_carry=(w0avg_next, nca_next))
    elif cu_opt == 2:
        cldefi = (
            carry.cumulus_carry if carry.cumulus_carry is not None
            else initial_bmj_carry(next_state)
        )
        next_state, cldefi_next = bmj_adapter(
            next_state, float(namelist.dt_s), cldefi, grid=namelist.grid
        )
        next_carry = next_carry.replace(cumulus_carry=cldefi_next)

    # --- radiation slot: SW/LW family dispatch -----------------------------
    # ra_sw_physics selects the SW scheme (0=disabled, 4=RRTMG, 1=Dudhia, 2=GSFC)
    # and ra_lw_physics the LW scheme (0=disabled, 4=RRTMG, 1=classic AER RRTM).
    # WRF runs the SW and LW drivers independently, so the HELD-RATE RTHRATEN is
    # the SUM of the two chosen tendencies, with disabled components contributing
    # zero. The default (ra_sw=4, ra_lw=4) is dispatched through the COMBINED
    # rrtmg_theta_tendency (single column-input build, byte-unchanged). Any other
    # combination composes the SW-only and LW-only couplers. The held rate is added
    # into theta at every dynamics step over the radt interval (shared cadence).
    ra_sw = int(namelist.ra_sw_physics)
    ra_lw = int(namelist.ra_lw_physics)
    land_for_rad = carry.noahmp_land if bool(namelist.use_noahmp) else None

    def _sw_tendency() -> jnp.ndarray:
        if ra_sw == 0:
            return jnp.zeros_like(next_state.theta)
        if ra_sw == 1:
            return dudhia_sw_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                radiation_static=namelist.radiation_static,
                land_state=land_for_rad,
            )
        if ra_sw == 2:
            return gsfc_sw_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                radiation_static=namelist.radiation_static,
                land_state=land_for_rad,
            )
        return rrtmg_sw_theta_tendency(
            next_state,
            namelist.grid,
            time_utc=namelist.time_utc,
            lead_seconds=lead_seconds,
            radiation_static=namelist.radiation_static,
            topo_shading=int(namelist.topo_shading),
            slope_rad=int(namelist.slope_rad),
            shadow_length_m=float(namelist.topo_shadow_length_m),
            land_state=land_for_rad,
        )

    def _lw_tendency() -> jnp.ndarray:
        if ra_lw == 0:
            return jnp.zeros_like(next_state.theta)
        if ra_lw == 1:
            return rrtm_lw_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                radiation_static=namelist.radiation_static,
                land_state=land_for_rad,
            )
        return rrtmg_lw_theta_tendency(
            next_state,
            namelist.grid,
            time_utc=namelist.time_utc,
            lead_seconds=lead_seconds,
            radiation_static=namelist.radiation_static,
            land_state=land_for_rad,
        )

    def _refresh_rthraten(_unused) -> jnp.ndarray:
        # Default RRTMG SW+LW: keep the combined single-build path byte-unchanged.
        if ra_sw == 4 and ra_lw == 4:
            return rrtmg_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                radiation_static=namelist.radiation_static,
                topo_shading=int(namelist.topo_shading),
                slope_rad=int(namelist.slope_rad),
                shadow_length_m=float(namelist.topo_shadow_length_m),
                land_state=land_for_rad,
            )
        return _sw_tendency() + _lw_tendency()

    if isinstance(run_radiation, bool):
        held_rthraten = _refresh_rthraten(None) if run_radiation else carry.rthraten
    else:
        held_rthraten = jax.lax.cond(
            run_radiation, _refresh_rthraten, lambda _u: carry.rthraten, None
        )
    # WRF-faithful RTHRATEN cadence (rad_rk_tendf=1): instead of the lumped one-step
    # Euler add ``theta += dt*RTHRATEN`` BEFORE the dycore (the v0.9 SHIPPED default,
    # rad_rk_tendf=0), route the SAME held rate through the ``t_tendf`` channel of
    # ``rk_addtend_dry`` so it is integrated at EVERY acoustic substep interleaved
    # with the dynamics (advance_mu_t: theta += msfty*dts*theta_tend; rk_addtend_dry
    # folds t_tendf/msfty; the msfty cancels and the mass-coupled rate decouples to
    # dts*RTHRATEN per substep -> dt*RTHRATEN over the full RK3 step, but distributed
    # across the substeps, NOT lumped).  Source: module_first_rk_step_part2.F:392-394
    # feeds RTHRATEN into t_tendf; the coupler doc rrtmg_theta_tendency:1660-1665
    # documents the lumped form as NOT WRF-equivalent.  The branch is a STATIC Python
    # condition (rad_rk_tendf is a compile-time constant) so rad_rk_tendf=0 emits the
    # identical XLA program and the operational forecast is bit-for-bit unchanged.
    if int(namelist.rad_rk_tendf) != 0:
        metrics = namelist.metrics
        mass_h = (
            metrics.c1h[:, None, None] * next_state.mu_total[None, :, :]
            + metrics.c2h[:, None, None]
        )
        # COUPLED radiation theta tendency d(mut*theta)/dt = mut*RTHRATEN; rk_addtend_dry
        # consumes t_tendf already mass-coupled (it only re-divides by msfty).
        t_tendf_rad = (mass_h * held_rthraten).astype(next_state.theta.dtype)
        dry = DryPhysicsTendencies(t_tendf=t_tendf_rad)
        # theta is left WITHOUT the radiation add here; the dycore delivers it via the
        # RK/acoustic cadence above.
    else:
        next_state = next_state.replace(
            theta=next_state.theta + float(namelist.dt_s) * held_rthraten
        )
        dry = _dry_physics_tendencies_from_state_delta(
            before, next_state, namelist, float(namelist.dt_s)
        )
    next_carry = next_carry.replace(rthraten=held_rthraten)

    return _PhysicsStepForcing(next_state, next_carry, dry, True)


def _physics_boundary_step_with_limiter_diagnostics(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
) -> tuple[OperationalCarry, dict[str, jax.Array]]:
    physical_origin = carry.state
    # Forecast clock for this step (traced scalar). Hoisted above the dycore so the
    # in-acoustic-loop NORMAL-momentum boundary targets are interpolated at the
    # step-start lead (matching WRF, which fixes ru_tend/rv_tend at the step start);
    # also reused below by rrtmg + the end-of-step lateral boundary nudge.
    lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
    physics_forcing = _physics_step_forcing(
        carry, namelist, lead_seconds, run_radiation=run_radiation
    )
    carry = physics_forcing.carry
    carry = _rk_scan_step(
        carry,
        namelist,
        debug=debug,
        lead_seconds=lead_seconds,
        physics_tendencies=physics_forcing.dry_tendencies,
    )
    next_state = carry.state
    if bool(physics_forcing.enabled):
        next_state = _apply_physics_non_dry_updates(next_state, physical_origin, physics_forcing.state)
        carry = carry.replace(state=next_state)
    limiter_diagnostics = _empty_theta_limiter_diagnostics(next_state.theta)
    if not bool(namelist.disable_guards):
        next_state, limiter_diagnostics = _limit_guarded_dynamics_state_with_diagnostics(next_state, physical_origin)
        next_state = next_state.replace(
            qv=_valid_mixing_ratio(next_state.qv, physical_origin.qv),
            qc=_valid_mixing_ratio(next_state.qc, physical_origin.qc),
            qr=_valid_mixing_ratio(next_state.qr, physical_origin.qr),
            qi=_valid_mixing_ratio(next_state.qi, physical_origin.qi),
            qs=_valid_mixing_ratio(next_state.qs, physical_origin.qs),
            qg=_valid_mixing_ratio(next_state.qg, physical_origin.qg),
        )
    if bool(namelist.run_boundary):
        bounded = apply_lateral_boundaries(
            next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config, namelist.metrics
        )
        if bool(namelist.disable_guards):
            next_state = bounded
        else:
            next_state = bounded.replace(
                u=_finite_or_origin(bounded.u, physical_origin.u),
                v=_finite_or_origin(bounded.v, physical_origin.v),
                w=_finite_or_origin(bounded.w, physical_origin.w),
                theta=_finite_or_origin(bounded.theta, physical_origin.theta),
                qv=_valid_mixing_ratio(bounded.qv, physical_origin.qv),
                p=_finite_or_origin(bounded.p, physical_origin.p),
                ph=_finite_or_origin(bounded.ph, physical_origin.ph),
                p_total=_finite_or_origin(bounded.p_total, physical_origin.p_total),
                ph_total=_finite_or_origin(bounded.ph_total, physical_origin.ph_total),
                p_perturbation=_finite_or_origin(bounded.p_perturbation, physical_origin.p_perturbation),
                ph_perturbation=_finite_or_origin(bounded.ph_perturbation, physical_origin.ph_perturbation),
            )
            next_state = _limit_guarded_mass_state(next_state, physical_origin)
    next_state = _enforce_operational_precision(next_state, force_fp64=bool(namelist.force_fp64))
    return _maybe_exchange_sharded_carry_halos(carry.replace(state=next_state)), limiter_diagnostics


def _physics_boundary_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
) -> OperationalCarry:
    next_carry, _diagnostics = _physics_boundary_step_with_limiter_diagnostics(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        debug=debug,
    )
    return next_carry


def _scan_forecast_segment(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
    run_radiation: bool,
    debug: bool = False,
) -> OperationalCarry:
    indices = jnp.arange(start_step, start_step + steps, dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        return _physics_boundary_step(scan_carry, namelist, step_index, run_radiation=run_radiation, debug=debug), None

    next_carry, _ = jax.lax.scan(body, carry, indices)
    return next_carry


def _scan_forecast_segment_with_limiter_diagnostics(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
    run_radiation: bool,
    debug: bool = False,
) -> tuple[OperationalCarry, dict[str, jax.Array]]:
    indices = jnp.arange(start_step, start_step + steps, dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        next_carry, diagnostics = _physics_boundary_step_with_limiter_diagnostics(
            scan_carry,
            namelist,
            step_index,
            run_radiation=run_radiation,
            debug=debug,
        )
        diagnostics = dict(diagnostics)
        diagnostics["step_index"] = step_index
        return next_carry, diagnostics

    next_carry, diagnostics = jax.lax.scan(body, carry, indices)
    return next_carry, diagnostics


def _concat_theta_limiter_diagnostics(chunks: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    return {key: jnp.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


# --------------------------------------------------------------------------
# M9 operational diagnostics carry (coupler_interface.md §4, §6 item 1)
# --------------------------------------------------------------------------


class M9Diagnostics(NamedTuple):
    """The M9 operational divergence-map surface fields, all mass-point (ny,nx).

    Side-channel only -- recomputed from the post-step State at OUTPUT cadence,
    not prognostic leaves. SWDOWN/GLW W m^-2; HFX/LH W m^-2 (upward +); PBLH m;
    TSK/T2 K; U10/V10 m s^-1; PSFC Pa. ``swdown``/``glw`` follow the forecast
    clock (namelist.time_utc + lead_seconds) so the diurnal cycle is captured.
    """

    swdown: jax.Array
    glw: jax.Array
    hfx: jax.Array
    lh: jax.Array
    pblh: jax.Array
    tsk: jax.Array
    t2: jax.Array
    u10: jax.Array
    v10: jax.Array
    psfc: jax.Array
    # B1 (v0.12.0) RRTMG up/down all-sky flux slices for the wrfout radiation
    # diagnostics, all mass-point (ny, nx), W m^-2 (except coszen, dimensionless).
    # Surface (bottom-of-atmosphere): swdnb/swupb (SW), lwdnb/lwupb (LW);
    # top-of-atmosphere: swdnt/swupt (SW), lwdnt/lwupt (LW). swnorm = slope-normal
    # surface SW flux; coszen = cosine solar zenith. OLR (== lwupt) is derived in
    # the writer. All-sky only -- the clear-sky ``...C`` vars are NOT produced (the
    # RRTMG port runs no separate clear-sky pass; see RRTMGRadiationDiagnostics).
    swdnb: jax.Array
    swupb: jax.Array
    lwdnb: jax.Array
    lwupb: jax.Array
    swdnt: jax.Array
    swupt: jax.Array
    lwdnt: jax.Array
    lwupt: jax.Array
    swnorm: jax.Array
    coszen: jax.Array


def _psfc_from_state(state: State) -> jax.Array:
    """Surface pressure (Pa) = total pressure extrapolated to the ground (ny,nx).

    WRF-faithful surface pressure. WRF reports ``PSFC(i,j) = p8w(i,kts,j)``
    (module_surface_driver.F:1988), where ``p8w`` (the full/w-level pressure at
    the bottom face = the terrain surface) is built in ``phy_prep`` by a linear
    extrapolation IN HEIGHT from the first two MASS levels
    (module_big_step_utilities_em.F:4917-4922; identical formula in
    dyn_em/start_em.F:2526-2531 and share/dfi.F)::

        z0 = z_at_w(1)              # bottom face (terrain surface)
        z1 = z(1)                   # 1st mass level (layer center)
        z2 = z(2)                   # 2nd mass level
        w1 = (z0 - z2)/(z1 - z2);  w2 = 1 - w1
        p8w(1) = w1*p(1) + w2*p(2)

    The previous diagnostic returned ``state.p[0]`` (the level-1 MASS-CENTER
    pressure), which omits the half-layer hydrostatic increment between the
    layer center (~25 m AGL here) and the ground -> a systematic NEGATIVE,
    terrain-correlated PSFC offset of ~ rho*g*dz_half (~300 Pa at sea level,
    rho~1.19 kg/m^3). This restores the WRF extrapolation.

    Heights enter only through the ratio ``(z0-z2)/(z1-z2)``, so the factor of
    ``g`` cancels and we use the total geopotential ``ph_total`` (faces)
    directly; the mass-level geopotential is the half-sum of adjacent faces, as
    in WRF's ``z(k) = 0.5*(z_at_w(k)+z_at_w(k+1))``.
    """
    p = state.p_total
    phi = state.ph_total  # total geopotential on faces (nz+1, ny, nx)
    phi0 = phi[0]                       # bottom face == terrain surface
    phi1 = 0.5 * (phi[0] + phi[1])      # mass level 1 (layer center)
    phi2 = 0.5 * (phi[1] + phi[2])      # mass level 2
    w1 = (phi0 - phi2) / (phi1 - phi2)
    w2 = 1.0 - w1
    return w1 * p[0, :, :] + w2 * p[1, :, :]


def compute_m9_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    lead_seconds,
    *,
    noahmp_land=None,
    noahmp_rad=None,
    noahclassic_land=None,
) -> M9Diagnostics:
    """Recompute the M9 surface map from a post-step State (side-channel only).

    When Noah-MP is activated (``namelist.use_noahmp`` and ``noahmp_land`` given),
    the LAND HFX/LH/TSK and the 2-m T2 are read back from the prognostic Noah-MP
    coupler and overlaid (the standalone-replacement contract); ocean/water keeps
    the bulk surface-layer diagnostic. The land T2 is the Noah-MP LSM diagnostic
    ``T2 = FVEG*T2MV + (1-FVEG)*T2MB`` (the faithful overwrite WRF performs over
    land — module_surface_driver.F:3469-3473), NOT the surface-layer MYNN 2-m value.
    U10/V10 come from the bulk surface layer, which already uses the Noah-MP skin
    temperature (in ``state.t_skin``) as its BC.
    """
    surf = surface_layer_diagnostics(state, namelist.grid)
    radiation_land = noahmp_land if bool(namelist.use_noahmp) else None
    rad = rrtmg_radiation_diagnostics(
        state,
        namelist.grid,
        time_utc=namelist.time_utc,
        lead_seconds=lead_seconds,
        radiation_static=namelist.radiation_static,
        topo_shading=int(namelist.topo_shading),
        slope_rad=int(namelist.slope_rad),
        shadow_length_m=float(namelist.topo_shadow_length_m),
        land_state=radiation_land,
    )
    hfx, lh, tsk, t2 = surf.hfx, surf.lh, state.t_skin, surf.t2
    if bool(namelist.use_noahmp) and noahmp_land is not None:
        clock = _NoahMPClock(
            julian=float(namelist.noahmp_julian), yearlen=float(namelist.noahmp_yearlen)
        )
        radiation = (
            _NoahMPRadiation(*noahmp_rad) if noahmp_rad is not None
            else _NoahMPRadiation(rad.swnorm, rad.glw, rad.coszen)
        )
        ep, rp = _noahmp_params(namelist)
        # v0.9.0: route the Noah-MP LSM 2-m T2 over land (the faithful land-T2
        # overwrite WRF performs — module_surface_driver.F:3469-3473), replacing
        # the surface-layer MYNN 2-m value over land. Water keeps surf.t2.
        hfx, lh, tsk, t2 = overlay_noahmp_land_diagnostics(
            state, noahmp_land, namelist.noahmp_static, surf.hfx, surf.lh, state.t_skin,
            float(namelist.dt_s), bulk_t2=surf.t2, radiation=radiation, clock=clock,
            energy_params=ep, rad_params=rp,
        )
    elif _explicit_noahclassic(namelist) and noahclassic_land is not None:
        hfx, lh, tsk = overlay_noahclassic_land_diagnostics(
            state, noahclassic_land, surf.hfx, surf.lh, state.t_skin
        )
    # L1 fix (GPT 2026-06-02 COSZEN-phase; proofs/rad_time/coszen_phase_proof.json):
    # when the HELD Noah-MP radiation tuple is available, report the held WRF-cadence
    # SWDOWN/GLW (soldn=noahmp_rad[0], lwdn=noahmp_rad[1]) rather than the OUTPUT-time
    # recompute. ``rad`` above is ``rrtmg_radiation_diagnostics(... lead_seconds=
    # output_time)`` -- an instantaneous-solar recompute at the history timestamp,
    # which carries the off-noon COSZEN-phase residual (+8.7%@09z / -3.0%@15z). WRF
    # does NOT recompute the history SWDOWN at the output instant; it holds the
    # radiation-cadence flux. ``noahmp_rad`` is exactly that held field
    # (carry.noahmp_rad), refreshed by ``_refresh_noahmp_rad`` at the WRF-faithful
    # held time ``lead_seconds - 0.5*radt_seconds`` (== coszen(t - radt/2), the field
    # WRF's end-of-step history output carries). Reporting it makes the diagnostic
    # equal the held WRF-cadence field. The non-Noah-MP / noahmp_rad=None path is
    # unchanged (still the output-time recompute).
    sw_enabled = int(namelist.ra_sw_physics) != 0
    lw_enabled = int(namelist.ra_lw_physics) != 0
    swdown_out = rad.swnorm if int(namelist.slope_rad) == 1 else rad.swdown
    glw_out = rad.glw
    if noahmp_rad is not None:
        swdown_out = jnp.asarray(noahmp_rad[0], dtype=jnp.float64)
        glw_out = jnp.asarray(noahmp_rad[1], dtype=jnp.float64)
    if not sw_enabled:
        swdown_out = jnp.zeros_like(swdown_out)
    if not lw_enabled:
        glw_out = jnp.zeros_like(glw_out)
    swdnb = rad.swdown if sw_enabled else jnp.zeros_like(rad.swdown)
    swupb = rad.swup if sw_enabled else jnp.zeros_like(rad.swup)
    swdnt = rad.sw_toa_down if sw_enabled else jnp.zeros_like(rad.sw_toa_down)
    swupt = rad.sw_toa_up if sw_enabled else jnp.zeros_like(rad.sw_toa_up)
    swnorm = rad.swnorm if sw_enabled else jnp.zeros_like(rad.swnorm)
    lwdnb = rad.glw if lw_enabled else jnp.zeros_like(rad.glw)
    lwupb = rad.glw_up if lw_enabled else jnp.zeros_like(rad.glw_up)
    lwdnt = rad.lw_toa_down if lw_enabled else jnp.zeros_like(rad.lw_toa_down)
    lwupt = rad.lw_toa_up if lw_enabled else jnp.zeros_like(rad.lw_toa_up)
    return M9Diagnostics(
        swdown=swdown_out,
        glw=glw_out,
        hfx=hfx,
        lh=lh,
        pblh=surf.pblh,
        tsk=tsk,
        t2=t2,
        u10=surf.u10,
        v10=surf.v10,
        psfc=_psfc_from_state(state),
        # B1: RRTMG all-sky up/down flux slices, straight from the radiation
        # diagnostics (no held-radiation override -- these are the instantaneous
        # output-cadence fluxes, consistent with the SWDOWN/GLW recompute path).
        # SWDNB == bottom-of-atmosphere downwelling SW (== SWDOWN in the no-slope
        # config); SWNORM == slope-normal surface SW.
        swdnb=swdnb,
        swupb=swupb,
        lwdnb=lwdnb,
        lwupb=lwupb,
        swdnt=swdnt,
        swupt=swupt,
        lwdnt=lwdnt,
        lwupt=lwupt,
        swnorm=swnorm,
        coszen=rad.coszen,
    )


@partial(jax.jit, static_argnames=("n_steps", "cadence"))
def _advance_chunk(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    start_step,
    *,
    n_steps: int,
    cadence: int,
) -> OperationalCarry:
    """Advance one output interval as a SINGLE compiled scan (no diagnostics).

    Radiation is gated by the traced ``step_index %% cadence == 0`` predicate via
    ``_physics_boundary_step``'s cond path, so this is byte-identical to the
    production per-step cadence.  ``start_step`` is TRACED (only ``n_steps``/
    ``cadence`` static) so equal-length intervals reuse the SAME compiled executable
    -- one compile for the whole forecast, not one per interval.  Kept SEPARATE from
    the diagnostics call so the dynamics scratch is freed before the large RRTMG
    diagnostic transient is allocated (peak-memory bound, Task 2 OOM fix).
    """
    run_physics = bool(namelist.run_physics)
    start_step = jnp.asarray(start_step, dtype=jnp.int32)
    indices = start_step + jnp.arange(int(n_steps), dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        if run_physics:
            run_radiation = jnp.equal(jnp.mod(step_index, int(cadence)), 0)
        else:
            run_radiation = False
        next_carry = _physics_boundary_step(
            scan_carry, namelist, step_index, run_radiation=run_radiation, debug=False
        )
        return next_carry, None

    carry, _ = jax.lax.scan(body, carry, indices)
    return carry


@jax.jit
def _m9_snapshot(carry: OperationalCarry, namelist: OperationalNamelist, lead_seconds) -> M9Diagnostics:
    """Compute the M9 surface map once from a post-chunk State (separate program).

    Isolated in its own ``jax.jit`` so XLA cannot co-schedule the ~15 GiB RRTMG
    g-point diagnostic transient with the dynamics-chunk scratch; the host loop
    blocks after the chunk so the chunk scratch is freed first.
    """
    return compute_m9_diagnostics(
        carry.state, namelist, lead_seconds,
        noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad,
        noahclassic_land=carry.noahclassic_land,
    )


def run_forecast_operational_with_m9_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
    *,
    output_cadence_steps: int = 60,
) -> tuple[State, M9Diagnostics]:
    """Run the operational forecast and emit the M9 surface map at output cadence.

    Materializes the M9 surface diagnostics ONLY at the OUTPUT cadence (never every
    step) and bounds peak memory to (forecast working set + ONE RRTMG diagnostic
    transient), independent of forecast length.

    OOM FIX (Sprint perf-diag Task 2).  Two compounding problems killed the previous
    implementation at 1080 steps (+3h, >20 GB OOM):

    1. ``compute_m9_diagnostics`` was called inside the per-step scan body and
       ``jax.lax.scan`` stacked ``(diag, emit)`` for EVERY step -- 1080 copies of all
       10 surface maps plus every step's diagnostic intermediates kept live.
    2. ``compute_m9_diagnostics`` re-runs the FULL RRTMG SW+LW column solver, whose
       g-point intermediate is ~15 GiB on this d02 grid.  Even computing it a few
       times inside ONE jit lets XLA overlap those transients (measured: a single
       jit over the 3h forecast tried to allocate 27.8 GiB).

    Fix: a HOST-driven loop walks one output interval at a time, calling the jit'd
    ``_advance_chunk_and_snapshot`` (each chunk is ONE compiled scan, reused across
    intervals) and ``block_until_ready``-ing between chunks so each RRTMG transient is
    freed before the next chunk allocates its own.  The host loop runs only
    ``steps // out_cad`` iterations (e.g. 3 for +3h hourly) -- NOT per step -- so there
    is no per-timestep host/device transfer.  The dynamics are byte-identical to the
    production scan (same per-step body, same traced radiation schedule); only the
    emitted-snapshot set differs.  ``run_forecast_operational`` is untouched.
    """
    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    _resolve_operational_suite(namelist)  # fail-closed physics-suite validation
    if int(output_cadence_steps) <= 0:
        raise ValueError("output_cadence_steps must be positive")
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = _committed_initial_carry_for_run(state, namelist)
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    out_cad = int(output_cadence_steps)

    # Output-interval boundaries: every multiple of out_cad up to steps, plus a final
    # partial interval if steps is not a multiple of out_cad (so the final state is
    # always emitted).
    boundaries: list[int] = list(range(out_cad, steps + 1, out_cad))
    if not boundaries or boundaries[-1] != steps:
        boundaries.append(steps)

    dt_s = float(namelist.dt_s)
    diag_chunks: list[M9Diagnostics] = []
    start = 1
    for end in boundaries:
        n = end - start + 1
        carry = _advance_chunk(
            carry, namelist, jnp.asarray(start, dtype=jnp.int32), n_steps=n, cadence=cadence
        )
        # Free the dynamics-chunk scratch BEFORE the RRTMG diagnostic transient is
        # allocated, then free the transient before the next chunk -- this is what
        # bounds peak memory to (working set + ONE transient) for any forecast length.
        jax.block_until_ready(carry.state.theta)
        diag = _m9_snapshot(carry, namelist, jnp.asarray(float(end) * dt_s, dtype=jnp.float64))
        jax.block_until_ready(diag.t2)
        diag_chunks.append(
            M9Diagnostics(*(getattr(diag, name)[None, ...] for name in M9Diagnostics._fields))
        )
        start = end + 1

    all_diags = M9Diagnostics(
        *(jnp.concatenate([getattr(chunk, name) for chunk in diag_chunks], axis=0)
          for name in M9Diagnostics._fields)
    )
    return carry.state, all_diags


def run_forecast_operational(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Run an operational forecast as one compiled, device-resident scan.

    Thin donate-safety wrapper over the jitted body. The jitted body donates
    ``state`` (``donate_argnums=(0,)``) for peak-memory reuse, which requires every
    State leaf to back a UNIQUE device buffer. Real cases built with the
    transitional legacy aliases (``State.replace(p=p_total, ...)``) carry
    buffer-aliased leaves; ``_dealias_pytree_buffers`` rebinds any duplicate to a
    distinct buffer here -- BEFORE the donate boundary -- so the donate path can
    never raise "Attempt to donate the same buffer twice". Numerically identical:
    de-aliasing only copies a shared buffer, it changes no value or dtype.
    """

    return _run_forecast_operational_jit(_dealias_pytree_buffers(state), namelist, hours)


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def _run_forecast_operational_jit(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Run an operational forecast as one compiled, device-resident scan.

    No diagnostics, host-read callbacks, host array pulls, or sanitizers are
    present in this path. ``hours`` is static so the timestep count is fixed at
    compile time and the whole forecast lowers as one JAX program.
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    _resolve_operational_suite(namelist)  # fail-closed physics-suite validation
    # Honour namelist.force_fp64 at the PUBLIC entry: the in-scan enforcement
    # (line ~1471) upcasts each step's output to fp64 when force_fp64, so the
    # INITIAL carry must also be fp64 or jax.lax.scan rejects the carry dtype
    # mismatch -- and the production path would otherwise start fp32 (GPT
    # re-confirm: proofs that pre-upcast manually did not exercise this entry).
    initial = _initial_carry_for_run(state, namelist)
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial
    step = 1
    while step <= steps:
        next_radiation = ((step + cadence - 1) // cadence) * cadence
        if bool(namelist.run_physics) and next_radiation <= steps:
            non_radiation_steps = next_radiation - step
            if non_radiation_steps:
                carry = _scan_forecast_segment(
                    carry,
                    namelist,
                    start_step=step,
                    steps=non_radiation_steps,
                    run_radiation=False,
                    debug=False,
                )
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=next_radiation,
                steps=1,
                run_radiation=True,
                debug=False,
            )
            step = next_radiation + 1
        else:
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=step,
                steps=steps - step + 1,
                run_radiation=False,
                debug=False,
            )
            step = steps + 1
    return carry.state


def run_forecast_operational_segmented(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
    *,
    segment_steps: int | None = None,
) -> State:
    """Run a long operational forecast as a HOST loop over ONE compiled segment.

    Long-run (24-72h) compile-blowup remedy that keeps compile O(segment) and peak
    GPU memory bounded, independent of forecast length.

    Why this exists.  ``run_forecast_operational`` is a Python while-loop that emits
    one ``jax.lax.scan`` per radiation interval, so the number of distinct XLA scan
    subcomputations -- and thus COMPILE time / peak memory -- grows with the forecast
    length (measured: +12h did not compile in 37 min).  This entry instead compiles a
    SINGLE fixed-length inner segment (``_advance_chunk`` with a static ``n_steps``)
    and drives it from a host ``for`` loop, carrying ``State`` across segments and
    ``block_until_ready``-ing between them so each segment's dynamics scratch is freed
    before the next segment allocates.  Compile happens ONCE for the full-length
    segment (every equal-length segment reuses the same executable via the traced
    ``start_step``); a single shorter compile covers a final partial tail segment.

    Equivalence.  Global step indices run ``1..steps`` exactly as in
    ``run_forecast_operational_single_scan`` and ``run_forecast_operational``; the
    in-segment radiation gate is the SAME traced ``step_index %% cadence == 0``
    predicate.  Because the segments are contiguous in the global step index, RRTMG
    fires on exactly the same global steps as the single scan, so the result is
    BITWISE identical to the single scan and round-off identical to the validated
    segmented while-loop (proof: proofs/perf/segscan_equiv.json -- seg-vs-single max
    abs diff == 0 on every field at 0.2h and 0.6h incl. the radiation step; seg-vs-
    production differs only at FP round-off from cond-vs-direct RRTMG application).

    ``segment_steps`` defaults to one radiation cadence interval so radiation fires
    exactly once at each full segment's last step; any positive value is accepted
    (the radiation schedule is unaffected by where the segment boundaries fall).
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    _resolve_operational_suite(namelist)  # fail-closed physics-suite validation
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")
    seg = int(segment_steps) if segment_steps is not None else cadence
    if seg <= 0:
        raise ValueError("segment_steps must be positive")

    carry = _committed_initial_carry_for_run(state, namelist)
    steps = _steps_for_hours(hours, float(namelist.dt_s))

    # Host loop over contiguous fixed-length segments covering global steps 1..steps.
    # Every full segment has identical static ``n_steps`` so it reuses ONE compiled
    # executable (``start_step`` is traced); a final partial segment compiles once.
    start = 1
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(
            carry, namelist, jnp.asarray(start, dtype=jnp.int32), n_steps=int(n), cadence=cadence
        )
        # Block so this segment's device scratch is freed before the next segment's
        # buffers are allocated -- this is what bounds peak memory to one segment's
        # working set regardless of forecast length.
        jax.block_until_ready(carry.state.theta)
        start += n
    return carry.state


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational_single_scan(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Whole forecast as ONE jax.lax.scan -- compile-blowup remedy for 24-72h.

    The production ``run_forecast_operational`` Python while-loop emits one
    ``jax.lax.scan`` per radiation interval (a non-radiation scan plus an isolated
    1-step radiation scan), so the number of distinct XLA scan subcomputations -- and
    thus the COMPILE time -- scales with the forecast length: ~4 scans at 1h, 12 at
    3h, 96 at 24h, 288 at 72h.  Measured: the cold compile of the 3h (12-scan)
    program exceeds ~30 min (proofs/perf -- the +3h's ~32 min was almost entirely
    this cold compile; warmed steady-state is ~45 ms/step).  At 24-72h the segmented
    compile is a hard wall.

    This entry collapses the whole forecast into a SINGLE scan whose trip count is
    the static step total, and gates RRTMG with ``jax.lax.cond`` on the traced
    predicate ``(step_index %% cadence == 0)``.  Compile cost is then independent of
    forecast length (one scan body), while the per-step cadence and the RRTMG firing
    schedule are numerically IDENTICAL to the segmented path (cond fires RRTMG on
    exactly the same steps).  Warmed throughput is unchanged.  This is the
    recommended path for long-lead / ensemble runs; the segmented production path is
    left untouched and remains the validated default until this entry passes its own
    short-horizon equivalence gate (proofs/perf/single_scan_equiv.json).
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    _resolve_operational_suite(namelist)  # fail-closed physics-suite validation
    initial = _initial_carry_for_run(state, namelist)
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")
    run_physics = bool(namelist.run_physics)

    indices = jnp.arange(1, steps + 1, dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        if run_physics:
            run_radiation = jnp.equal(jnp.mod(step_index, cadence), 0)
        else:
            run_radiation = False  # static: no radiation branch traced at all
        next_carry = _physics_boundary_step(
            scan_carry, namelist, step_index, run_radiation=run_radiation, debug=False
        )
        return next_carry, None

    carry, _ = jax.lax.scan(body, initial, indices)
    return carry.state


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational_with_limiter_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
) -> tuple[State, dict[str, jax.Array]]:
    """Run an operational forecast and return INV-10 theta limiter diagnostics."""

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    _resolve_operational_suite(namelist)  # fail-closed physics-suite validation
    # Honour namelist.force_fp64 at the PUBLIC entry: the in-scan enforcement
    # (line ~1471) upcasts each step's output to fp64 when force_fp64, so the
    # INITIAL carry must also be fp64 or jax.lax.scan rejects the carry dtype
    # mismatch -- and the production path would otherwise start fp32 (GPT
    # re-confirm: proofs that pre-upcast manually did not exercise this entry).
    initial = _initial_carry_for_run(state, namelist)
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial
    step = 1
    diagnostic_chunks: list[dict[str, jax.Array]] = []
    while step <= steps:
        next_radiation = ((step + cadence - 1) // cadence) * cadence
        if bool(namelist.run_physics) and next_radiation <= steps:
            non_radiation_steps = next_radiation - step
            if non_radiation_steps:
                carry, diagnostics = _scan_forecast_segment_with_limiter_diagnostics(
                    carry,
                    namelist,
                    start_step=step,
                    steps=non_radiation_steps,
                    run_radiation=False,
                    debug=False,
                )
                diagnostic_chunks.append(diagnostics)
            carry, diagnostics = _scan_forecast_segment_with_limiter_diagnostics(
                carry,
                namelist,
                start_step=next_radiation,
                steps=1,
                run_radiation=True,
                debug=False,
            )
            diagnostic_chunks.append(diagnostics)
            step = next_radiation + 1
        else:
            carry, diagnostics = _scan_forecast_segment_with_limiter_diagnostics(
                carry,
                namelist,
                start_step=step,
                steps=steps - step + 1,
                run_radiation=False,
                debug=False,
            )
            diagnostic_chunks.append(diagnostics)
            step = steps + 1
    return carry.state, _concat_theta_limiter_diagnostics(diagnostic_chunks)


@partial(jax.jit, static_argnames=("hours", "debug"), donate_argnums=(0,))
def run_forecast_operational_debug(state: State, namelist: OperationalNamelist, hours: float, *, debug: bool = False) -> State:
    """Diagnostic operational forecast entry point with static debug markers."""

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    _resolve_operational_suite(namelist)  # fail-closed physics-suite validation
    # Honour namelist.force_fp64 at the PUBLIC entry: the in-scan enforcement
    # (line ~1471) upcasts each step's output to fp64 when force_fp64, so the
    # INITIAL carry must also be fp64 or jax.lax.scan rejects the carry dtype
    # mismatch -- and the production path would otherwise start fp32 (GPT
    # re-confirm: proofs that pre-upcast manually did not exercise this entry).
    initial = _initial_carry_for_run(state, namelist)
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial
    step = 1
    while step <= steps:
        next_radiation = ((step + cadence - 1) // cadence) * cadence
        if bool(namelist.run_physics) and next_radiation <= steps:
            non_radiation_steps = next_radiation - step
            if non_radiation_steps:
                carry = _scan_forecast_segment(
                    carry,
                    namelist,
                    start_step=step,
                    steps=non_radiation_steps,
                    run_radiation=False,
                    debug=debug,
                )
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=next_radiation,
                steps=1,
                run_radiation=True,
                debug=debug,
            )
            step = next_radiation + 1
        else:
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=step,
                steps=steps - step + 1,
                run_radiation=False,
                debug=debug,
            )
            step = steps + 1
    return carry.state


__all__ = [
    "OperationalNamelist",
    "M9Diagnostics",
    "compute_m9_diagnostics",
    "dealias_state_buffers",
    "run_forecast_operational",
    "run_forecast_operational_segmented",
    "run_forecast_operational_single_scan",
    "run_forecast_operational_debug",
    "run_forecast_operational_with_limiter_diagnostics",
    "run_forecast_operational_with_m9_diagnostics",
]
