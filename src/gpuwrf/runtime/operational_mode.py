"""GPU-resident operational forecast loop for M6 perf-design.

This module is deliberately separate from the M6B validation savepoint ladder.
It runs timestep/RK/acoustic loops inside one JAX entry point and leaves debug
snapshots/sanitizers out of the compiled path.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import os
import dataclasses
from dataclasses import dataclass, replace as dataclass_replace
from functools import partial
from typing import Any, NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.kernels.layout_pin import pin as _layout_pin, small_grid_firewall as _small_grid_firewall
from gpuwrf.dynamics.core.w_surface_reset import ENABLED as _W_SURFACE_RESET, reset_surface_w as _reset_surface_w
from gpuwrf.dynamics.core.advance_w import W_DAMP_STAGE as _W_DAMP_STAGE, w_damp_rw_tend_wrf as _w_damp_rw_tend_wrf
from gpuwrf.contracts.state import BaseState, State, Tendencies
from gpuwrf.physics.rrtmg_mp_re import mp_re_active
from gpuwrf.contracts.precision import (
    DEFAULT_ACOUSTIC_PRECISION_MODE,
    DEFAULT_DTYPES,
    STATE_FIELD_ORDER,
    SURFACE_LAYER_CARRY_LEAVES,
    MYNN_DIAGNOSTIC_LEAVES,
    MP_RE_DIAGNOSTIC_LEAVES,
    GWDO_DIAGNOSTIC_LEAVES,
    GWDO_SURFACE_DIAGNOSTIC_LEAVES,
    acoustic_precision_mode_label,
    is_mixed_perturb_fp32_mode,
)
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    DEFAULT_BOUNDARY_CONFIG,
    NESTED_BOUNDARY_SCALAR_SPECIES,
    apply_lateral_boundaries,
    flow_dep_bdy,
    interpolate_boundary_leaf,
    normal_bdy_work_target_u,
    normal_bdy_work_target_v,
    nested_ph_relax_tendency,
    nested_scalar_boundary_tendencies,
    nested_w_relax_tendency,
    root_scalar_bdy_rk1_enabled,
    root_scalar_boundary_tendencies,
    root_scalar_rk1_split,
    specified_boundary_tendency,
    specified_relax_dry_tendencies,
    SpecifiedRelaxTendencies,
    tangential_bdy_work_target_u,
    tangential_bdy_work_target_v,
    _full_ring_target_from_leaf,
)
from gpuwrf.coupling.physics_couplers import (
    dudhia_sw_theta_tendency,
    gsfc_sw_theta_tendency,
    gwdo_adapter,
    gwdo_tendencies,
    held_suarez_theta_tendency,
    mynn_adapter,
    mynn_adapter_with_source_leaves,
    rrtm_lw_theta_tendency,
    rrtmg_lw_theta_tendency,
    rrtmg_radiation_diagnostics,
    rrtmg_sw_theta_tendency,
    rrtmg_ozone_columns,
    rrtmg_theta_tendency,
    surface_adapter,
    surface_layer_diagnostics,
    thompson_adapter,
    thompson_aero_adapter,
    thompson_aero_coldstart_init,
    time_utc_clock_base,
    RadiationClock,
    RRTMGRadiationDiagnostics,
)
from gpuwrf.coupling.noahmp_surface_hook import (
    noahmp_surface_step,
    overlay_noahmp_land_diagnostics,
)
from gpuwrf.coupling.slab_surface_hook import (
    SlabRadiation,
    initial_slab_land,
    slab_surface_step,
)
from gpuwrf.coupling.pleim_xiu_surface_hook import (
    PleimXiuRadiation,
    initial_pleim_xiu_land,
    pleim_xiu_surface_step,
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
    ntiedtke_adapter,
    tiedtke_adapter,
)
from gpuwrf.assimilation.data_assimilation import (
    DigitalFilterConfig,
    add_dry_physics_tendencies,
    apply_nudging_rates,
    data_assimilation_dry_tendencies,
    data_assimilation_rates,
    digital_filter_initialize,
)
from gpuwrf.physics.myj_adapters import (
    janjic_sfclay_adapter,
    myj_pbl_adapter,
)
from gpuwrf.physics.wrf_clwrf_ghg import CLWRFGasClock, clwrf_gas_clock
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.explicit_diffusion import (
    C_S_DEFAULT,
    constant_k_diffusion_tendency,
    conservative_constant_k_diffusion_tendency,
    deformation_components_3d,
    dry_brunt_vaisala_squared,
    horizontal_deformation_2d,
    horizontal_diffusion_coord_momentum_tendency,
    horizontal_diffusion_coord_scalar_tendency,
    sixth_order_diffusion_tendency,
    smag2d_horizontal_km,
    smag3d_km,
    tke3d_km,
    tke_rhs_tendency,
    vertical_diffusion_coord_scalar_tendency,
    wrf_deformation_momentum_tendency,
    wrf_nonperiodic_diffusion_metrics,
    wrf_nested_horizontal_diffusion_momentum_tendency,
    wrf_sixth_order_scalar_tendf,
    wrf_sixth_order_uvw_tendf,
)
from gpuwrf.dynamics.flux_advection import (
    CoupledVelocities,
    advect_moisture_scalars,
    advect_scalar_flux,
    advect_u_flux,
    advect_v_flux,
    advect_w_flux,
    couple_velocities_periodic,
    stage_omega_specified,
    couple_uv_specified,
)
from gpuwrf.dynamics.acoustic_wrf import (
    CPOVCV,
    CVPM,
    P0_PA,
    R_D,
    _inverse_density_from_theta_pressure,
    calc_coef_w_wrf_coefficients,
    diagnose_pressure_al_alt,
    horizontal_pressure_gradient,
    moisture_coupling_factors,
)
from gpuwrf.dynamics.core.acoustic import (
    AcousticCoreConfig,
    AcousticCoreState,
    AcousticPhaseTapSummary,
    MassPrimitiveObservation,
    PHASE_TAP_SCRATCH_FIELDS,
    PHASE_TAP_SUMMARY_METRICS,
    acoustic_substep_core,
)
from gpuwrf.dynamics.core.advance_w import (
    GRAVITY_M_S2,
    dry_cqw,
    moist_cqw_calc_face,
    pg_buoy_w_dry,
    pg_buoy_w_moist,
)
from gpuwrf.dynamics.core.calc_p_rho import CalcPRhoStep0, calc_p_rho_wrf
from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf
from gpuwrf.dynamics.core.coupled import CoupledCoreConfig, coupled_timestep_core
from gpuwrf.dynamics.core.rk_addtend_dry import (
    DryPhysicsTendencies,
    large_step_coriolis,
    large_step_horizontal_curvature,
    large_step_horizontal_pgf,
    large_step_uv_fused,
    large_step_uv_fused_enabled,
    large_step_uv_nested_only,
    rk_addtend_dry,
)
from gpuwrf.dynamics.core.small_step_finish import small_step_finish_wrf
from gpuwrf.dynamics.core.small_step_prep import SmallStepPrepState, small_step_prep_wrf
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


configure_jax_x64()

_M0_EVIDENCE_FLAG = "GPUWRF_M0_EVIDENCE"
_M0_EVIDENCE_PATH = "GPUWRF_M0_EVIDENCE_PATH"
_M0_EVIDENCE_RUN_ID = "GPUWRF_M0_RUN_ID"
_M0_EVIDENCE_SOURCE_SHA256 = "GPUWRF_M0_SOURCE_SHA256"
_M0_EVIDENCE_CONFIG_SHA256 = "GPUWRF_M0_CONFIG_SHA256"
_M0_EVIDENCE_INPUT_SHA256 = "GPUWRF_M0_INPUT_MANIFEST_SHA256"
_M0_EVIDENCE_DEVICE_UUID = "GPUWRF_M0_DEVICE_UUID"
_M0_EVIDENCE_RANGE = "GPUWRF_M0_FORECAST_INTEGRATION"
_M0_EVIDENCE_SCHEMA = "wrf_gpu2.v025.m0.forecast_allocator.v1"
_M0_EVIDENCE_RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$"
_M0_EVIDENCE_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_M0_EVIDENCE_DEVICE_UUID_PATTERN = (
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)

_THETA_LIMITER_MIN_K = 0.0
# v0.14 Switzerland venting ROOT CAUSE (proofs/v014/switzerland_midlevel_momentum_budget,
# switzerland_theta_advection_operator_oracle, the norad/nophys bisect runs):
# the previous 500 K ceiling SAT BELOW the real stratospheric theta of the
# Switzerland d01 top level (k43 theta up to 507.5 K, >500 K over 62-98 % of the
# domain from h33 on; p_top=5000 Pa).  The "non-load-bearing safety net"
# therefore FIRED EVERY STEP: it crushed the interior top-level theta to 500 K
# while apply_lateral_boundaries restored the wrfbdy >500 K band afterwards, so
# each step re-clipped the boundary ring and the limiter's mass-conserving
# redistribution pumped the clipped theta into the whole interior as a steady
# ~+0.5 K/h tropospheric heating (profile proportional to the 500-theta
# headroom, physics-independent: identical with mp=0, radiation off, and ALL
# physics off).  The heated columns expand hydrostatically, diverge at mid/upper
# levels, and export dry mass -- THE -26.5 Pa/cell/h depth-8 venting -- with the
# low-level inflow / mid-level outflow u-dipole as the secondary (heat-low)
# circulation.  WRF has NO theta clamp; the envelope must be unreachable for any
# physical state.  Theta(1 hPa) ~ 870 K, far above any p_top this port runs, so
# 1000 K keeps the guard a genuine NaN/blow-up trap and nothing else.
_THETA_LIMITER_MAX_K = 1000.0
_RVRD = 461.6 / 287.0
_RVRD_REAL = np.float32(np.float32(461.6) / np.float32(287.0))  # WRF REAL r_v/r_d
# WRF reciprocal earth radius (share/module_model_constants.F:43), consumed by
# the rk_tendency curvature term on the vertical momentum.
_W_RERADIUS = 1.0 / 6370.0e3


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


def _moist_cqw_enabled() -> bool:
    """Whether the operational acoustic w-equation uses WRF moist ``cqw``/``pg_buoy_w``.

    Default ON for v0.14.  The once-per-RK-stage large-step vertical
    PGF/buoyancy ``rw_tend`` and the implicit-W ``cqw`` field carry the WRF moist
    water-mass loading (``calc_cq`` + moist ``pg_buoy_w``,
    module_big_step_utilities_em.F:856-870,2474-2497), so the GPU pressure rides
    the MOIST hydrostatic column (CPU/WRF) instead of the dry one.  Set
    ``GPUWRF_MOIST_CQW=0`` only for bisection/back-compat checks.  Bit-identical
    to the dry path wherever total moisture is zero (idealized/dry gates).  See
    proofs/v014/moist_cqw_pressure_dynamics_closure.{py,json,md} and
    proofs/v014/moist_cqw_gpu_h4_validation.{py,json,md}.
    """

    return os.environ.get("GPUWRF_MOIST_CQW", "1").strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
    }


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


# Per-CASE Noah-MP inputs (WRF TMN, VEGFRA): read from each case's wrfinput, not from
# the static geography. Baked as compile constants they made every new forecast case
# recompile the whole step (wn3 W4, 2026-10-03), so the namelist carries them as
# TRACED children; the static aux keeps a case-free NoahMPStatic (these slots None).
_NOAHMP_CASE_FIELDS = ("tbot", "shdfac")
# id(full static) -> (full static, case-free holder); pins the object so ids are not reused.
_NOAHMP_CASE_SPLIT_MEMO: dict[int, tuple[Any, _StaticHolder]] = {}
# content digest -> canonical case-free static: equal content shares one object, so the
# identity-hashed holder (and the jit treedef) is the same for every case of one grid.
_NOAHMP_CASE_FREE_INTERN: dict[str, Any] = {}


def _noahmp_case_free_digest(static) -> str | None:
    """Content digest of the non-case slots; None if any of them is traced."""
    import hashlib

    import numpy as np

    h = hashlib.sha256()
    for name in type(static).__slots__:
        if name in _NOAHMP_CASE_FIELDS:
            continue
        leaves, treedef = jax.tree_util.tree_flatten(getattr(static, name))
        h.update(name.encode())
        h.update(str(treedef).encode())
        for leaf in leaves:
            if isinstance(leaf, jax.core.Tracer):
                return None
            if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                arr = np.ascontiguousarray(np.asarray(leaf))
                h.update(f"{arr.shape}{arr.dtype}".encode())
                h.update(arr.tobytes())
            else:
                h.update(repr(leaf).encode())
    return h.hexdigest()


def _noahmp_case_split(static) -> tuple[_StaticHolder, tuple[Any, Any] | None]:
    """(static-aux holder, traced case tuple) for ``OperationalNamelist.tree_flatten``."""
    if static is None or not all(hasattr(static, f) for f in _NOAHMP_CASE_FIELDS):
        return _StaticHolder(static), None
    hit = _NOAHMP_CASE_SPLIT_MEMO.get(id(static))
    if hit is not None and hit[0] is static:
        return hit[1], tuple(getattr(static, f) for f in _NOAHMP_CASE_FIELDS)
    digest = _noahmp_case_free_digest(static)
    if digest is None:
        # Traced geography (not a product path): keep the old identity holder.
        return _StaticHolder(static), None
    case_free = _NOAHMP_CASE_FREE_INTERN.get(digest)
    if case_free is None:
        case_free = static.replace(**{f: None for f in _NOAHMP_CASE_FIELDS})
        _NOAHMP_CASE_FREE_INTERN[digest] = case_free
    holder = _StaticHolder(case_free)
    _noahmp_case_memo(static, holder)
    return holder, tuple(getattr(static, f) for f in _NOAHMP_CASE_FIELDS)


def _noahmp_case_memo(static, holder: _StaticHolder) -> None:
    if len(_NOAHMP_CASE_SPLIT_MEMO) >= 512:
        _NOAHMP_CASE_SPLIT_MEMO.clear()  # safe: interning keeps holders equal after a miss
    _NOAHMP_CASE_SPLIT_MEMO[id(static)] = (static, holder)


class _DateClockAux:
    """Non-keying carrier for the date-derived clock scalars (#114 fix).

    ``time_utc`` / ``noahmp_julian`` / ``noahmp_yearlen`` are pure functions of the
    forecast init DATE. #91 already threads them into the compiled program via the
    traced ``clock_base`` (see ``build_clock_base`` + ``_resolve_clock_parts``), so on
    the production nest path they are DEAD inside the compiled function -- their values
    are never read. Keeping them as discriminating entries of the namelist STATIC AUX
    therefore only changed the ``@jax.jit`` cache key per forecast date, forcing a full
    re-trace + re-lower of the fused 9-nest megakernel and a fresh persistent-cache key
    every date (the #114 per-date recompile).

    This holder rides in the aux tuple in their place. Its ``__hash__`` is CONSTANT and
    its ``__eq__`` is date-blind (any two ``_DateClockAux`` compare equal), so two
    namelists differing ONLY in date produce an IDENTICAL treedef -> the in-memory jit
    cache HITS and the persistent compile-cache key is date-invariant. The VALUES are
    still carried on the instance and restored in ``tree_unflatten``, so the legacy
    ``clock_base=None`` path (idealized cases / old tests) still reads the exact same
    ``time_utc`` / ``noahmp_julian`` / ``noahmp_yearlen`` -- the change is value-
    preserving and bit-identical on every path.

    Safety of the constant hash: this holder appears EXACTLY ONCE per namelist treedef,
    so there is no within-treedef collision; every OTHER aux entry (grid, options, dt_s,
    table holders, ...) still discriminates compiles correctly. The only merge is the
    intended one (date axis).

    v0.25 S3: the merge is only correct if NO traced code reads these values. The
    RRTMG CLWRF gases used to (host-interpolated from ``time_utc`` at trace time),
    so a cache hit for a new date silently reused the first date's gases. They now
    come from ``_ClockBase.ghg_clock``; a dated RRTMG call with a traced clock but
    no gas clock fails closed.  Clock-less (``clock_base=None``) legacy entries still
    read ``time_utc`` at trace time and are NOT cross-date cache-safe."""

    __slots__ = ("time_utc", "noahmp_julian", "noahmp_yearlen")

    def __init__(self, time_utc, noahmp_julian, noahmp_yearlen):
        self.time_utc = time_utc
        self.noahmp_julian = noahmp_julian
        self.noahmp_yearlen = noahmp_yearlen

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, _DateClockAux)


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
    # v0.17 ADR-032 graupel/hail substrate: the day a hail MP scheme is wired it
    # returns increments for these, and the scan must carry them like every
    # other MP species. With no hail scheme wired the increment is zero, so this
    # is inert (no existing program changes).
    "qh",
    "Nh",
    "qvolg",
    "qvolh",
    # v0.16 aerosol-aware Thompson (mp=28) prognostic aerosol numbers: the
    # thompson_aero_adapter advances nwfa/nifa (activation/scavenging/emission)
    # and the scan must carry those increments like every other mp species.
    "nwfa",
    "nifa",
    "qke",
)


_PHYSICS_NON_DRY_REPLACE_FIELDS: tuple[str, ...] = (
    # MYNN prognosed variance and diagnosed BL clouds must survive RK. WRF
    # radiation_driver consumes the previous PBL call's cloud fields before
    # the next PBL call (icloud_bl=1); dropping them makes radiation clear.
    "qsq",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    *MYNN_DIAGNOSTIC_LEAVES,
    *MP_RE_DIAGNOSTIC_LEAVES,
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
    # B39 MYNN surface-layer carry (WRF grid%MOL/HFX/QFX/QSFC/PBLH).
    "mol",
    "hfx",
    "qfx",
    "qsfc",
    "pblh",
    "dtaux3d",
    "dtauy3d",
    "dusfcg",
    "dvsfcg",
    "sfc_wspd",
)

_SHARDED_CARRY_HALO_CONTEXT: tuple[object, int] | None = None


class _PhysicsStepForcing(NamedTuple):
    """Physics output split into WRF RK dry tendencies and non-dry state writes."""

    state: State
    carry: OperationalCarry
    dry_tendencies: DryPhysicsTendencies
    enabled: bool


class _PreHaloCaptureResult(NamedTuple):
    """Proof-only return carrying the normal post-halo carry plus captured State."""

    carry: OperationalCarry
    pre_halo_state: State


class CorrectedNiPhaseTapResult(NamedTuple):
    """Complete ordinary carry plus the bounded option-2 acoustic summary."""

    carry: OperationalCarry
    summary: AcousticPhaseTapSummary


class FirstIntervalMomentumRecord(NamedTuple):
    """Diagnostic-only first-interval momentum savepoints (v0234).

    Raw A-grid MYNN PBL momentum tendencies (RUBLTEN/RVBLTEN, exactly as the
    adapter returns them before any mass coupling) plus the assembled WRF
    ``update_phy_ten`` momentum tendencies (mass-coupled, face-mapped
    ru_tendf/rv_tendf).  Captured only by the proof-only
    ``advance_one_step_with_first_interval_capture`` entry point; never part
    of the production step program.
    """

    rublten: jax.Array
    rvblten: jax.Array
    ru_tendf: jax.Array
    rv_tendf: jax.Array


class FirstIntervalStepResult(NamedTuple):
    """Complete ordinary carry plus the first-interval momentum record."""

    carry: OperationalCarry
    record: FirstIntervalMomentumRecord


class FirstIntervalLadderRecord(NamedTuple):
    """Diagnostic-only d03 step suboperator-ladder savepoints (v0234).

    Source-bound before/after states spanning the dry-dycore/nest region
    between the assembled physics tendencies (SP3) and the end-of-step exit
    (SP4), aligned 1:1 with the WRF-side ladder dumps:

    * ``rk1_tend_u/v``: RK1 merged large-step coupled momentum tendency
      (WRF ``grid%ru_tend/rv_tend`` after ``rk_addtend_dry``/``spec_bdy_dry``,
      ``solve_em.F`` after ``BENCH_END(relax_bdy_dry_tim)``).
    * ``rk1_relax_u/v``: step-constant boundary-relax bundle
      (WRF ``grid%u_save/v_save`` after rk1-only ``relax_bdy_dry``).
    * ``rk{1,2,3}_fin_u/v``: per-RK-stage finished momentum
      (WRF ``grid%u_2/v_2`` after ``small_step_finish``).
    * ``pre_bdry_u/v``: state just before the end-of-step boundary pass
      (WRF ``grid%u_2/v_2`` immediately before ``spec_bdy_final``).

    Captured only by the proof-only
    ``advance_one_step_with_first_interval_ladder`` entry point; never part
    of the production step program.
    """

    rk1_tend_u: jax.Array
    rk1_tend_v: jax.Array
    rk1_relax_u: jax.Array
    rk1_relax_v: jax.Array
    rk1_fin_u: jax.Array
    rk1_fin_v: jax.Array
    rk2_fin_u: jax.Array
    rk2_fin_v: jax.Array
    rk3_fin_u: jax.Array
    rk3_fin_v: jax.Array
    pre_bdry_u: jax.Array
    pre_bdry_v: jax.Array


class FirstIntervalLadderStepResult(NamedTuple):
    """Complete ordinary carry plus the first-interval and ladder records."""

    carry: OperationalCarry
    record: FirstIntervalMomentumRecord
    ladder: FirstIntervalLadderRecord


class _RkLadderResult(NamedTuple):
    """Internal carrier for the RK-scan leg of the ladder capture."""

    carry: OperationalCarry
    rk1_tend_u: jax.Array
    rk1_tend_v: jax.Array
    rk1_relax_u: jax.Array
    rk1_relax_v: jax.Array
    rk1_fin_u: jax.Array
    rk1_fin_v: jax.Array
    rk2_fin_u: jax.Array
    rk2_fin_v: jax.Array
    rk3_fin_u: jax.Array
    rk3_fin_v: jax.Array


RCA_HEALTH_METRICS = (
    "nonfinite_count",
    "first_nonfinite_flat_index",
    "max_abs_finite",
    "max_abs_flat_index",
    "value_at_max_abs",
    "min_finite",
    "min_finite_flat_index",
    "max_finite",
)
RCA_ACOUSTIC_FIELDS = (
    "pre_uv_u",
    "pre_uv_v",
    "large_u_tend",
    "large_v_tend",
    "pre_p",
    "pre_al",
    "pre_ph",
    "small_dpx",
    "small_dpy",
    "uv_u",
    "uv_v",
    "raw_dvdxi",
    "raw_dmdt",
    "raw_mu_tendency",
    "mu_scale",
    "limited_dvdxi",
    "limited_dmdt",
    "output_mudf",
    "new_muts",
    "post_w",
    "post_ph",
    "post_p",
    "post_al",
    "post_theta",
    "post_u",
    "post_v",
    "post_mu",
    "post_ww",
)
RCA_STATE_FIELDS = (
    "u", "v", "w", "theta", "qv",
    "p_total", "p_perturbation", "ph_total", "ph_perturbation",
    "mu_total", "mu_perturbation",
    "qc", "qr", "qi", "qs", "qg",
    "Ni", "Nr", "Ns", "Ng", "Nc", "Nn",
    "qke", "qsq", "qc_bl", "qi_bl", "cldfra_bl",
    "ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv",
    "t_skin", "soil_moisture", "roughness_m",
    "rain_acc", "rainc_acc", "snow_acc", "graupel_acc", "ice_acc",
)
RCA_BOUNDARY_FIELDS = (
    "u_bdy", "v_bdy", "w_bdy", "theta_bdy", "qv_bdy",
    "p_bdy", "pb_bdy", "ph_bdy", "phb_bdy", "mu_bdy", "mub_bdy",
)
RCA_STATE_PHASES = (
    "step_entry",
    "physics_output",
    "post_rk_pre_non_dry",
    "post_non_dry",
    "post_guard",
    "post_boundary",
    "post_precision",
)
RCA_TARGET_K = 1
RCA_TARGET_Y = 48
RCA_TARGET_X = 78


class _RcaAcousticScanResult(NamedTuple):
    carry: OperationalCarry
    health: jax.Array
    target: jax.Array


class _RcaRkResult(NamedTuple):
    carry: OperationalCarry
    acoustic_health: jax.Array
    acoustic_target: jax.Array


class RcaStepRecord(NamedTuple):
    acoustic_health: jax.Array
    acoustic_target: jax.Array
    state_health: jax.Array
    state_target: jax.Array
    boundary_health: jax.Array


class _RcaStepResult(NamedTuple):
    carry: OperationalCarry
    record: RcaStepRecord


class RcaChunkResult(NamedTuple):
    carry: OperationalCarry
    records: RcaStepRecord


def _rca_array_health(value: jax.Array) -> jax.Array:
    """Eight-scalar finite/range/location summary with no host callback."""

    array = jnp.asarray(value, dtype=jnp.float64)
    flat = jnp.ravel(array)
    finite = jnp.isfinite(flat)
    bad = jnp.logical_not(finite)
    bad_count = jnp.sum(bad, dtype=jnp.int64)
    first_bad = jnp.where(bad_count > 0, jnp.argmax(bad), -1)
    finite_abs = jnp.where(finite, jnp.abs(flat), -jnp.inf)
    max_abs_index = jnp.argmax(finite_abs)
    max_abs = finite_abs[max_abs_index]
    value_at_max = flat[max_abs_index]
    finite_values = jnp.where(finite, flat, jnp.inf)
    min_index = jnp.argmin(finite_values)
    minimum = finite_values[min_index]
    maximum = jnp.max(jnp.where(finite, flat, -jnp.inf))
    return jnp.asarray(
        (
            bad_count, first_bad, max_abs, max_abs_index, value_at_max,
            minimum, min_index, maximum,
        ),
        dtype=jnp.float64,
    )


def _rca_acoustic_health(
    result: AcousticCoreState,
    observation: MassPrimitiveObservation,
) -> jax.Array:
    values = (
        observation.pre_uv_u,
        observation.pre_uv_v,
        observation.large_u_tend,
        observation.large_v_tend,
        observation.pre_p,
        observation.pre_al,
        observation.pre_ph,
        observation.small_dpx,
        observation.small_dpy,
        observation.uv_u,
        observation.uv_v,
        observation.raw_dvdxi,
        observation.raw_dmdt,
        observation.raw_mu_tendency,
        observation.mu_scale,
        observation.limited_dvdxi,
        observation.limited_dmdt,
        observation.output_mudf,
        observation.new_muts,
        result.w,
        result.ph,
        result.p,
        result.al,
        result.theta,
        result.u,
        result.v,
        result.mu,
        result.ww,
    )
    return jnp.stack(tuple(_rca_array_health(value) for value in values), axis=0)


def _rca_target_value(value: jax.Array) -> jax.Array:
    """Return the corrected Ni-cell scalar (shape-clamped for tiny unit grids)."""

    array = jnp.asarray(value, dtype=jnp.float64)
    y = min(RCA_TARGET_Y, int(array.shape[-2]) - 1)
    x = min(RCA_TARGET_X, int(array.shape[-1]) - 1)
    if array.ndim >= 3:
        k = min(RCA_TARGET_K, int(array.shape[-3]) - 1)
        return array[(Ellipsis, k, y, x)]
    return array[(Ellipsis, y, x)]


def _rca_acoustic_target(
    result: AcousticCoreState,
    observation: MassPrimitiveObservation,
) -> jax.Array:
    values = (
        observation.pre_uv_u,
        observation.pre_uv_v,
        observation.large_u_tend,
        observation.large_v_tend,
        observation.pre_p,
        observation.pre_al,
        observation.pre_ph,
        observation.small_dpx,
        observation.small_dpy,
        observation.uv_u,
        observation.uv_v,
        observation.raw_dvdxi,
        observation.raw_dmdt,
        observation.raw_mu_tendency,
        observation.mu_scale,
        observation.limited_dvdxi,
        observation.limited_dmdt,
        observation.output_mudf,
        observation.new_muts,
        result.w,
        result.ph,
        result.p,
        result.al,
        result.theta,
        result.u,
        result.v,
        result.mu,
        result.ww,
    )
    return jnp.stack(tuple(_rca_target_value(value) for value in values), axis=0)


def _rca_state_health(state: State) -> jax.Array:
    return jnp.stack(
        tuple(_rca_array_health(getattr(state, name)) for name in RCA_STATE_FIELDS),
        axis=0,
    )


def _rca_state_target(state: State) -> jax.Array:
    return jnp.stack(
        tuple(_rca_target_value(getattr(state, name)) for name in RCA_STATE_FIELDS),
        axis=0,
    )


def _rca_boundary_health(state: State) -> jax.Array:
    return jnp.stack(
        tuple(_rca_array_health(getattr(state, name)) for name in RCA_BOUNDARY_FIELDS),
        axis=0,
    )


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
    # WRF calc_p_rho_phi specific-volume relation (Registry.EM_COMMON:2285;
    # WRF registry default is 2 = LOG-pressure-thickness).  The dataclass default
    # stays 1 (the legacy linear form) so idealized harnesses -- whose generators
    # carry placeholder c3f/c4f hybrid coefficients (ic_generators/idealized.py)
    # that make the LOG form singular -- remain byte-unchanged; the REAL replay /
    # daily / nested pipelines pass 2 (their metrics carry the true wrfinput
    # C3F/C4F/C3H/C4H/P_TOP).  v0.14 Switzerland h36 root cause: the linear form
    # biases al/alt/p one-signed (~4.2e-4 rel mean) with terrain-modulated
    # structure -> spurious large-step horizontal PGF -> d01 dry mass venting.
    hypsometric_opt: int = 1
    # v0.14 acoustic continuation: WRF &dynamics h_sca_adv_order (Registry
    # default 5).  rhs_ph's horizontal phi advection uses this order; the
    # real-case specified-boundary order<=6 branch is selected when >=4 AND the
    # lateral BCs are specified/nested.  Default 2 keeps every idealized /
    # periodic caller on the legacy byte-identical path.
    h_sca_adv_order: int = 2
    # v0.14 stage3/wrapper-cadence sprint: WRF SPECIFIED-domain lateral-boundary
    # cadence.  ON: per-stage relax_bdy_dry tendencies (u/v/t/ph all stages, mu
    # at rk_step==1 -- the WRF quirk) flow through the acoustic loop, the
    # spec-zone (ring-0) mass/theta/ph/tangential-wind work arrays are pinned to
    # the wrfbdy trajectory every substep (WRF spec_bdyupdate/_ph), w gets the
    # specified zero_grad_bdy ring copy, and the end-of-step lateral pass keeps
    # only the ring-0 spec re-sync + full moisture (no relax-zone value nudge,
    # no p'/pb forcing -- WRF has neither).  OFF (default): byte-identical to
    # the previous once-per-step end-of-step nudge on every path.
    specified_bdy_cadence: bool = False
    # v0.14 stage3/wrapper-cadence sprint, candidate A: WRF SPECIFIED-boundary
    # advection-order degradation (module_advect_em degrade_* blocks) for the
    # flux-form theta/moisture/u/v/w operators.  The periodic implementation
    # wraps every horizontal stencil; on a real specified domain rings 0-2
    # advect with opposite-edge values (ring-1 mu increment err 11.7 Pa/stage
    # vs WRF 0.94, immune to the boundary cadence).  Default OFF.
    specified_adv_degrade: bool = False
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
    # diff_opt=1/km_opt=4 2-D Smagorinsky horizontal-diffusion path (WRF smag2d_km)
    # and diff_opt=2/km_opt=3/5 3-D Smagorinsky/SMS paths.
    c_s: float = C_S_DEFAULT
    # WRF TKE/LES diffusion defaults (Registry.EM_COMMON c_k/mix_isotropic/
    # mix_upper_bound/tke_upper_bound/tke_mix2_off).  These are static compile
    # controls; the timestep loop consumes only device arrays and scalar literals.
    c_k: float = 0.15
    mix_isotropic: int = 0
    mix_upper_bound: float = 0.1
    tke_upper_bound: float = 1000.0
    tke_mix2_off: bool = False
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
    # Optional auxinput4 operands, loaded before stepping; None preserves the
    # legacy namelist pytree/step graph for sst_update=0.
    lower_boundary: object = None
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
    # Optional pre-built prognostic Noah-MP land carry (NoahMPLandState). The
    # production daily/nested pipelines build it from the wrfinput and seed it into
    # the carry via ``carry.replace`` AFTER ``_initial_carry_for_run``. The generic
    # single-domain operational path (``run_forecast_operational`` -> the coverage
    # gate) has no post-replace seam, so when this field IS supplied
    # ``_initial_carry_for_run`` seeds ``noahmp_land`` + the concrete held
    # ``noahmp_rad`` directly. Left None, behaviour is unchanged (the pipelines
    # still post-replace), so this is append-only / non-breaking.
    noahmp_land: object = None
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
    # G3 city/lake switches. Defaults are WRF's disabled path. Active BEP/BEM
    # urban canopy and lake-model selections are recognized but fail-closed in
    # _resolve_operational_suite until their state carry + WRF oracle + JAX
    # kernels are implemented.
    sf_urban_physics: int = 0
    sf_lake_physics: int = 0
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
    # WRF default: Thompson supplies held effective radii to RRTMG.
    use_mp_re: int = 1
    # Explicit Noah-classic (sf_surface_physics=2) operational inputs. The JAX SFLX
    # kernel consumes WRF-derived REDPRM/static fields and a 4-layer land carry; if
    # either is absent, the scan rejects sf_surface_physics=2 rather than deriving
    # an unvalidated land state from 2-D State.soil_moisture.
    noahclassic_static: object = None
    noahclassic_land: object = None
    noahclassic_rad: object = None
    # Explicit thermal-diffusion slab LSM (sf_surface_physics=1) operational
    # inputs (v0.17). The JAX SLAB1D kernel consumes a WRF-derived
    # SlabStaticBundle (soil ZS/DZS + TMN/THC/EMISS/SNOWC) and a 5-layer TSLB
    # land carry. If ``slab_static`` is absent, the scan rejects
    # sf_surface_physics=1 rather than deriving an unvalidated land state.
    slab_static: object = None
    slab_land: object = None
    slab_rad: object = None
    # Explicit Pleim-Xiu LSM (sf_surface_physics=7) operational inputs (v0.17).
    # The JAX SURFPX+QFLUX kernel consumes a WRF-derived PleimXiuStaticBundle (ISBA
    # soil constants + vegetation fields) and a 2-layer ISBA land carry. If
    # ``px_static`` is absent, the scan rejects sf_surface_physics=7.
    px_static: object = None
    px_land: object = None
    px_rad: object = None
    # Explicit RUC LSM (sf_surface_physics=3) operational inputs (v0.3.4, lane
    # o1-ruc): a RucStaticBundle (RucConfig + WRF RUC tables + IVGTYP/ISLTYP/XLAND/
    # XICE/TMN/SHDMIN/SHDMAX/ALBBCK) and the seeded RucLandState (RUC soil levels from
    # wrfinput). If either is absent the scan rejects sf_surface_physics=3.
    ruc_static: object = None
    ruc_land: object = None
    ruc_rad: object = None
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
    # --- v0.22 G1 data assimilation ----------------------------------------
    # Optional resident analysis/obs/spectral nudging bundle.  The arrays ride
    # as pytree CHILDREN, so FDDA targets can live on device and be consumed
    # inside the timestep scan without host callbacks.  ``None`` is the default
    # and leaves all historical programs byte-unchanged.
    data_assimilation: object = None
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
    # v0.14 R0 default-inert acoustic precision contract label (ADR-031 DRAFT).
    # NO timestep code consumes this field yet; it rides in static aux so a
    # future explicit mixed acoustic mode gets a separate JIT/cache variant and
    # report label, and unknown mode strings fail closed at construction.
    acoustic_precision_mode: str = DEFAULT_ACOUSTIC_PRECISION_MODE
    radiation_interval_s: float = 0.0
    cumulus_cadence_steps: int = 1
    cudt_minutes: float = 0.0
    # WRF cam_abs_freq_s (Registry default 21600 s): ra_lw_physics=3 CAM recomputes its LW absorptivities/emissivity
    # only every cam_abs_freq_s (module_radiation_driver.F:1601) and holds them in OperationalCarry.cam_abs.
    cam_abs_freq_s: float = 21600.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acoustic_precision_mode",
            acoustic_precision_mode_label(self.acoustic_precision_mode),
        )
        if (os.environ.get("GPUWRF_CARRY_REAL_ALL", "0") == "1"
                and os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1"
                and os.environ.get("GPUWRF_DYN_REAL_ALL", "0") != "1"):
            # b-carry G10: the REAL carry without the REAL dycore operators splits the
            # Noah-MP fusion (+3 % ordinary root); the release runs the pair only.
            raise ValueError("GPUWRF_CARRY_REAL_ALL requires GPUWRF_DYN_REAL_ALL=1 (the v0.3 "
                             "release pairs them); set both, or GPUWRF_FAST_DEFAULTS=0")
        if os.environ.get("GPUWRF_DYN_REAL_ALL", "0") == "1":
            # S2-DYN: WRF REAL grid metrics as executable parameters (no per-call
            # converts); a no-op when they are already REAL (unflatten in traces).
            from gpuwrf.kernels.dyn_real_fp32 import real, require_native
            require_native()
            if getattr(self.metrics, "precision", "fp32") != "fp32":
                object.__setattr__(self, "metrics", dataclass_replace(self.metrics, precision="fp32"))
            if self.tendencies is not None:
                # Resident base tendencies are WRF REAL too (else they promote augment).
                object.__setattr__(self, "tendencies", jax.tree_util.tree_map(real, self.tendencies))

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
        disable_guards: bool | None = None,
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
        c_k: float = 0.15,
        mix_isotropic: int = 0,
        mix_upper_bound: float = 0.1,
        tke_upper_bound: float = 1000.0,
        tke_mix2_off: bool = False,
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
        data_assimilation: object = None,
        rad_rk_tendf: int = 0,
        acoustic_precision_mode: str = DEFAULT_ACOUSTIC_PRECISION_MODE,
        hypsometric_opt: int = 1,
        h_sca_adv_order: int = 2,
        specified_bdy_cadence: bool = False,
        specified_adv_degrade: bool = False,
        use_mp_re: int = 1,
    ) -> "OperationalNamelist":
        """Build a namelist using resident zero tendencies and flat metrics."""

        if disable_guards is None:
            disable_guards = _strict_guards_enabled()
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
            c_k=c_k,
            mix_isotropic=mix_isotropic,
            mix_upper_bound=mix_upper_bound,
            tke_upper_bound=tke_upper_bound,
            tke_mix2_off=tke_mix2_off,
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
            data_assimilation=data_assimilation,
            rad_rk_tendf=rad_rk_tendf,
            acoustic_precision_mode=acoustic_precision_mode,
            hypsometric_opt=hypsometric_opt,
            h_sca_adv_order=h_sca_adv_order,
            specified_bdy_cadence=specified_bdy_cadence,
            specified_adv_degrade=specified_adv_degrade,
            use_mp_re=use_mp_re,
        )

    def tree_flatten(self):
        # The Noah-MP static (categories + soil geometry + the constant parameter
        # TABLES) and the pre-built energy/rad params ride as STATIC AUX, not traced
        # children: the frozen driver concretizes several of their fields inside the
        # scan (isurban, nroot, table scalars), so they must be COMPILE CONSTANTS,
        # not tracers. They are wrapped in an identity-hashable holder so the jit
        # cache keys on per-run object identity (one run -> one compile). use_noahmp
        # + clock scalars are also static aux. Exception: the per-case tbot/shdfac
        # (TMN/VEGFRA) ride as the traced ``noahmp_case`` child, so one grid -> one
        # compile across forecast cases (see _noahmp_case_split).
        noahmp_static_holder, noahmp_case = _noahmp_case_split(self.noahmp_static)
        children = (
            self.tendencies,
            self.metrics,
            self.radiation_static,
            self.gwdo_statics,
            self.data_assimilation,
            noahmp_case,
        )
        if self.lower_boundary is not None:
            children += (self.lower_boundary,)
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
            float(self.c_k),
            int(self.mix_isotropic),
            float(self.mix_upper_bound),
            float(self.tke_upper_bound),
            bool(self.tke_mix2_off),
            int(self.diff_6th_opt),
            float(self.diff_6th_factor),
            float(self.const_nu_m2_s),
            bool(self.use_flux_advection),
            int(self.scalar_adv_opt),
            int(self.moist_adv_opt),
            bool(self.force_fp64),
            bool(self.use_deformation_momentum_diffusion),
            # #114: the three date-derived scalars (time_utc, noahmp_julian,
            # noahmp_yearlen) ride in a date-BLIND holder instead of as discriminating
            # aux entries. #91 threads them via the traced clock_base, so they are dead
            # inside the compiled fn -> keying on them only forced a per-date recompile
            # of the fused nest. The holder hashes/compares date-blind so the namelist
            # treedef is date-invariant (in-memory jit cache hit + persistent-cache-key
            # stable across dates); tree_unflatten restores the values for the legacy
            # clock_base=None path. Value-preserving and bit-identical on every path.
            _DateClockAux(
                self.time_utc,
                float(self.noahmp_julian),
                float(self.noahmp_yearlen),
            ),
            int(self.topo_shading),
            int(self.slope_rad),
            float(self.topo_shadow_length_m),
            bool(self.use_noahmp),
            int(self.noahmp_nroot),
            noahmp_static_holder,
            _StaticHolder(self.noahmp_energy_params),
            _StaticHolder(self.noahmp_rad_params),
            _StaticHolder(self.noahmp_land),
            int(self.mp_physics),
            int(self.bl_pbl_physics),
            int(self.sf_sfclay_physics),
            int(self.cu_physics),
            self.sf_surface_physics,
            int(self.sf_urban_physics),
            int(self.sf_lake_physics),
            _StaticHolder(self.noahclassic_static),
            _StaticHolder(self.noahclassic_land),
            _StaticHolder(self.noahclassic_rad),
            _StaticHolder(self.slab_static),
            _StaticHolder(self.slab_land),
            _StaticHolder(self.slab_rad),
            _StaticHolder(self.px_static),
            _StaticHolder(self.px_land),
            _StaticHolder(self.px_rad),
            _StaticHolder(self.ruc_static),
            _StaticHolder(self.ruc_land),
            _StaticHolder(self.ruc_rad),
            int(self.gwd_opt),
            int(self.ra_sw_physics),
            int(self.ra_lw_physics),
            int(self.use_mp_re),
            int(self.rad_rk_tendf),
            self.acoustic_precision_mode,
            float(self.radiation_interval_s),
            int(self.cumulus_cadence_steps),
            float(self.cudt_minutes),
            int(self.hypsometric_opt),
            int(self.h_sca_adv_order),
            bool(self.specified_bdy_cadence),
            bool(self.specified_adv_degrade),
            float(self.cam_abs_freq_s),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        lower_boundary = children[6] if len(children) == 7 else None
        tendencies, metrics, radiation_static, gwdo_statics, data_assimilation, noahmp_case = children[:6]
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
            c_k,
            mix_isotropic,
            mix_upper_bound,
            tke_upper_bound,
            tke_mix2_off,
            diff_6th_opt,
            diff_6th_factor,
            const_nu_m2_s,
            use_flux_advection,
            scalar_adv_opt,
            moist_adv_opt,
            force_fp64,
            use_deformation_momentum_diffusion,
            date_clock,
            topo_shading,
            slope_rad,
            topo_shadow_length_m,
            use_noahmp,
            noahmp_nroot,
            noahmp_static_holder,
            noahmp_energy_holder,
            noahmp_rad_holder,
            noahmp_land_holder,
            mp_physics,
            bl_pbl_physics,
            sf_sfclay_physics,
            cu_physics,
            sf_surface_physics,
            sf_urban_physics,
            sf_lake_physics,
            noahclassic_static_holder,
            noahclassic_land_holder,
            noahclassic_rad_holder,
            slab_static_holder,
            slab_land_holder,
            slab_rad_holder,
            px_static_holder,
            px_land_holder,
            px_rad_holder,
            ruc_static_holder,
            ruc_land_holder,
            ruc_rad_holder,
            gwd_opt,
            ra_sw_physics,
            ra_lw_physics,
            use_mp_re,
            rad_rk_tendf,
            acoustic_precision_mode,
            radiation_interval_s,
            cumulus_cadence_steps,
            cudt_minutes,
            hypsometric_opt,
            h_sca_adv_order,
            specified_bdy_cadence,
            specified_adv_degrade,
            cam_abs_freq_s,
        ) = aux
        # #114: recover the date-derived clock scalars from the date-blind holder so the
        # legacy clock_base=None path still reads the exact same values (value-preserving).
        time_utc = date_clock.time_utc
        noahmp_julian = date_clock.noahmp_julian
        noahmp_yearlen = date_clock.noahmp_yearlen
        noahmp_static = noahmp_static_holder.value
        if noahmp_case is not None:
            noahmp_static = noahmp_static.replace(**dict(zip(_NOAHMP_CASE_FIELDS, noahmp_case)))
            _noahmp_case_memo(noahmp_static, noahmp_static_holder)
        noahmp_energy_params = noahmp_energy_holder.value
        noahmp_rad_params = noahmp_rad_holder.value
        noahmp_land = noahmp_land_holder.value
        noahclassic_static = noahclassic_static_holder.value
        noahclassic_land = noahclassic_land_holder.value
        noahclassic_rad = noahclassic_rad_holder.value
        slab_static = slab_static_holder.value
        slab_land = slab_land_holder.value
        slab_rad = slab_rad_holder.value
        px_static = px_static_holder.value
        px_land = px_land_holder.value
        px_rad = px_rad_holder.value
        ruc_static = ruc_static_holder.value
        ruc_land = ruc_land_holder.value
        ruc_rad = ruc_rad_holder.value
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
            c_k=c_k,
            mix_isotropic=mix_isotropic,
            mix_upper_bound=mix_upper_bound,
            tke_upper_bound=tke_upper_bound,
            tke_mix2_off=tke_mix2_off,
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
            lower_boundary=lower_boundary,
            topo_shading=topo_shading,
            slope_rad=slope_rad,
            topo_shadow_length_m=topo_shadow_length_m,
            use_noahmp=use_noahmp,
            noahmp_static=noahmp_static,
            noahmp_energy_params=noahmp_energy_params,
            noahmp_rad_params=noahmp_rad_params,
            noahmp_land=noahmp_land,
            noahmp_nroot=noahmp_nroot,
            noahmp_julian=noahmp_julian,
            noahmp_yearlen=noahmp_yearlen,
            mp_physics=mp_physics,
            bl_pbl_physics=bl_pbl_physics,
            sf_sfclay_physics=sf_sfclay_physics,
            cu_physics=cu_physics,
            sf_surface_physics=sf_surface_physics,
            sf_urban_physics=sf_urban_physics,
            sf_lake_physics=sf_lake_physics,
            noahclassic_static=noahclassic_static,
            noahclassic_land=noahclassic_land,
            noahclassic_rad=noahclassic_rad,
            slab_static=slab_static,
            slab_land=slab_land,
            slab_rad=slab_rad,
            px_static=px_static,
            px_land=px_land,
            px_rad=px_rad,
            ruc_static=ruc_static,
            ruc_land=ruc_land,
            ruc_rad=ruc_rad,
            gwd_opt=gwd_opt,
            gwdo_statics=gwdo_statics,
            data_assimilation=data_assimilation,
            ra_sw_physics=ra_sw_physics,
            ra_lw_physics=ra_lw_physics,
            use_mp_re=use_mp_re,
            rad_rk_tendf=rad_rk_tendf,
            acoustic_precision_mode=acoustic_precision_mode,
            radiation_interval_s=radiation_interval_s,
            cumulus_cadence_steps=cumulus_cadence_steps,
            cudt_minutes=cudt_minutes,
            hypsometric_opt=hypsometric_opt,
            h_sca_adv_order=h_sca_adv_order,
            specified_bdy_cadence=specified_bdy_cadence,
            specified_adv_degrade=specified_adv_degrade,
            cam_abs_freq_s=cam_abs_freq_s,
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


def _base_state_from_totals(state: State) -> BaseState:
    """Build explicit fp64 WRF base leaves from a concrete fp64 total/prime state."""

    theta_base = jnp.full_like(state.theta, jnp.asarray(300.0, dtype=jnp.float64), dtype=jnp.float64)
    return BaseState(
        pb=jnp.asarray(state.p_total, dtype=jnp.float64) - jnp.asarray(state.p_perturbation, dtype=jnp.float64),
        phb=jnp.asarray(state.ph_total, dtype=jnp.float64) - jnp.asarray(state.ph_perturbation, dtype=jnp.float64),
        mub=jnp.asarray(state.mu_total, dtype=jnp.float64) - jnp.asarray(state.mu_perturbation, dtype=jnp.float64),
        t0=jnp.asarray(300.0, dtype=jnp.float64),
        theta_base=theta_base,
    )


# v0.20 fp32 HARDENING (b): moisture / number species stored fp32 under the
# aggressive ADR-007 matrix. The qke promote (precision.py:233-257) handled the
# ONE field that went non-finite in fp32 at 1km by a static FP64 promotion. For
# the bulk moisture/number species (which stay FP32_GATED for the resident-state
# win) we instead apply a per-field RUNTIME finiteness GATE in the
# mixed_perturb_fp32 path: any non-finite fp32 cell is replaced by a finite floor
# (0.0 -- the WRF-faithful clamp for a poisoned mixing ratio / number
# concentration), so a single non-finite excursion guards locally instead of
# poisoning the whole run. Mirrors the existing _valid_mixing_ratio nonfinite
# trap; storage dtype is preserved (no resident-VRAM change). Only reached in the
# opt-in fp32 mode -> fp64_default is byte-unchanged.
_MIXED_FP32_FINITENESS_GATED_SPECIES: tuple[str, ...] = (
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "qh",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "Nc",
    "Nn",
    "Nh",
    "qvolg",
    "qvolh",
    "nwfa",
    "nifa",
)


def _gate_species_finiteness(value: jax.Array) -> jax.Array:
    """Replace non-finite fp32 species cells with a finite zero floor in-place dtype."""

    finite = jnp.isfinite(value)
    floor = jnp.zeros((), dtype=value.dtype)
    return jnp.where(finite, value, floor)


def _apply_mixed_perturb_fp32_storage(state: State, base_state: BaseState) -> State:
    """Store S4 acoustic dynamic carry as fp32 primes with fp64 rebuilt totals."""

    p_prime = jnp.asarray(state.p_perturbation, dtype=jnp.float32)
    ph_prime = jnp.asarray(state.ph_perturbation, dtype=jnp.float32)
    mu_prime = jnp.asarray(state.mu_perturbation, dtype=jnp.float32)
    w = jnp.asarray(state.w, dtype=jnp.float32)
    p_total = jnp.asarray(base_state.pb, dtype=jnp.float64) + p_prime.astype(jnp.float64)
    ph_total = jnp.asarray(base_state.phb, dtype=jnp.float64) + ph_prime.astype(jnp.float64)
    mu_total = jnp.asarray(base_state.mub, dtype=jnp.float64) + mu_prime.astype(jnp.float64)
    # Per-field 1km finiteness gate on the fp32-stored moisture/number species.
    species_updates: dict[str, jax.Array] = {}
    for field in _MIXED_FP32_FINITENESS_GATED_SPECIES:
        value = getattr(state, field, None)
        if value is None:
            continue
        if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
            # Only fp32-stored leaves can carry the fp32 non-finite excursion;
            # leave any (statically promoted) fp64 species untouched.
            continue
        species_updates[field] = _gate_species_finiteness(value)
    return state.replace(
        _cast=False,
        p_total=p_total,
        p_perturbation=p_prime,
        ph_total=ph_total,
        ph_perturbation=ph_prime,
        mu_total=mu_total,
        mu_perturbation=mu_prime,
        w=w,
        **species_updates,
    )


def _enforce_operational_precision(
    state: State,
    *,
    force_fp64: bool = False,
    acoustic_precision_mode: str | None = None,
    base_state: BaseState | None = None,
) -> State:
    if is_mixed_perturb_fp32_mode(acoustic_precision_mode):
        fp64_state = _enforce_operational_precision(state, force_fp64=True)
        base = _base_state_from_totals(fp64_state) if base_state is None else base_state
        return _apply_mixed_perturb_fp32_storage(fp64_state, base)
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
            if value is None or field in (*SURFACE_LAYER_CARRY_LEAVES, *MYNN_DIAGNOSTIC_LEAVES, *GWDO_DIAGNOSTIC_LEAVES, *MP_RE_DIAGNOSTIC_LEAVES):
                # B39: WRF-REAL surface-layer carry is never upcast.
                continue
            if value.dtype != jnp.float64:
                updates[field] = value.astype(jnp.float64)
        # _cast=False so the fp64 upcast is NOT canonicalised back to each
        # field's loaded dtype.  Real-case states arrive mixed-precision
        # (DEFAULT_DTYPES perf matrix: theta/u/v fp32, w/mu/ph fp64); without
        # this the force_fp64 path is a silent no-op (Sprint U P0-1).
        result = state.replace(_cast=False, **updates) if updates else state.replace(_cast=False)
        if base_state is not None and os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1":
            # Default-off WRF REAL carry: the native RK/acoustic step stores the
            # dycore prognostics REAL; the wide copy above only serves BC40's base.
            from gpuwrf.kernels.dyn_carry_fp32 import real_state
            result = real_state(result)
        return result
    updates = {}
    for field in STATE_FIELD_ORDER:
        value = getattr(state, field)
        if value is None:
            continue
        target = DEFAULT_DTYPES.dtype_for(field)
        if value.dtype != target:
            updates[field] = value.astype(target)
    # v0.20 S4-ultracode: _cast=False so the ADR-007 fp32-gated DOWNCAST actually
    # stands.  Without it, State.replace's default _cast=True re-canonicalises each
    # updated leaf back to its CURRENT (fp64) dtype (state.py:861) -- the SAME no-op
    # trap the force_fp64 branch documents at :1034 and fixes with _cast=False.  An
    # fp64 carry would otherwise pass through here UNCHANGED, so the fp32-gated
    # operational matrix was dormant whenever the carry arrived fp64 (i.e. always,
    # since the pipeline force_fp64=True path produces an fp64 carry).  Only reached
    # when force_fp64=False (the GPUWRF_FORCE_FP64=0 measurement hook); the fp64 and
    # mixed branches above are untouched, so the production default is byte-identical.
    return state.replace(_cast=False, **updates)


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


def _kahan_sum_vertical(values: jax.Array) -> jax.Array:
    """Kahan-compensated fp64 sum over the leading vertical axis."""

    arr = jnp.asarray(values, dtype=jnp.float64)
    total = jnp.zeros_like(arr[0], dtype=jnp.float64)
    compensation = jnp.zeros_like(total, dtype=jnp.float64)
    for k in range(int(arr.shape[0])):
        y = arr[k] - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total - compensation


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

    if os.environ.get("GPUWRF_DYN_REAL_ALL", "0") == "1":
        return _positive_definite_theta_increment_limiter_real(
            candidate, origin, mass, minimum_k=minimum_k, maximum_k=maximum_k,
            lower_bound=lower_bound, upper_bound=upper_bound)
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


def _positive_definite_theta_increment_limiter_real(
    candidate: jax.Array,
    origin: jax.Array,
    mass: jax.Array,
    *,
    minimum_k: float,
    maximum_k: float,
    lower_bound: jax.Array | None,
    upper_bound: jax.Array | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """GPUWRF_DYN_REAL_ALL form of the theta limiter in WRF REAL.

    Same algorithm, but the conserved-mass residual is summed from the per-cell
    difference sum((candidate - limited) * mass) -- exactly zero off the limited
    cells -- instead of the difference of two domain-wide sums, which would cancel
    catastrophically in REAL. With no limited cell the output equals the input.
    """

    f32 = jnp.float32
    output_dtype = jnp.asarray(candidate).dtype
    cand = jnp.asarray(candidate, dtype=f32)
    orig = jnp.asarray(origin, dtype=f32)
    weight = jnp.asarray(mass, dtype=f32)
    lower = jnp.asarray(float(minimum_k), dtype=f32)
    upper = jnp.asarray(float(maximum_k), dtype=f32)
    if lower_bound is not None:
        lower = jnp.maximum(lower, jnp.asarray(lower_bound, dtype=f32))
    if upper_bound is not None:
        upper = jnp.minimum(upper, jnp.asarray(upper_bound, dtype=f32))
    upper = jnp.maximum(upper, lower)
    midpoint = f32(0.5) * (lower + upper)

    safe_origin = jnp.where(jnp.isfinite(orig), orig, midpoint)
    safe_origin = jnp.minimum(jnp.maximum(safe_origin, lower), upper)
    finite_candidate = jnp.where(jnp.isfinite(cand), cand, safe_origin)
    raw_delta = finite_candidate - safe_origin

    over_upper = finite_candidate > upper
    under_lower = finite_candidate < lower
    invalid = ~jnp.isfinite(cand)
    limited_mask = invalid | over_upper | under_lower

    positive_delta = raw_delta > 0
    negative_delta = raw_delta < 0
    upper_alpha = (upper - safe_origin) / jnp.where(positive_delta, raw_delta, f32(1))
    lower_alpha = (lower - safe_origin) / jnp.where(negative_delta, raw_delta, f32(-1))
    alpha = jnp.where(positive_delta, upper_alpha, jnp.where(negative_delta, lower_alpha, f32(1)))
    alpha = jnp.where(limited_mask, jnp.minimum(jnp.maximum(alpha, f32(0)), f32(1)), f32(1))
    limited0 = jnp.where(limited_mask, safe_origin + alpha * raw_delta, finite_candidate)

    residual = jnp.sum((finite_candidate - limited0) * weight)
    add_room = upper - limited0
    subtract_room = limited0 - lower
    room = jnp.where(residual >= 0, add_room, subtract_room)
    capacity = jnp.sum(room * weight)
    fraction = jnp.where(capacity > 0, jnp.minimum(jnp.abs(residual) / capacity, f32(1)), f32(0))
    limited = limited0 + jnp.sign(residual) * fraction * room
    limited = jnp.minimum(jnp.maximum(limited, lower), upper)
    limited = limited.astype(output_dtype)

    target_mass = jnp.sum(finite_candidate * weight)
    mass_residual = jnp.sum((limited.astype(f32) - finite_candidate) * weight)
    diagnostics = {
        "theta_limited_cell_count": jnp.sum(limited_mask.astype(jnp.int32)),
        "theta_first_limited_cell_xyz": _first_limited_cell_xyz(limited_mask),
        "theta_mass_before": target_mass.astype(output_dtype),
        "theta_mass_after": (target_mass + mass_residual).astype(output_dtype),
        "theta_mass_residual": mass_residual.astype(output_dtype),
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
    ``[0, 1000] K`` plus the non-finite trap (v0.14: the earlier 500 K ceiling was
    BELOW the real Switzerland-d01 top-level stratospheric theta of up to 507.5 K
    and made the guard load-bearing again -- the steady boundary-ring clip +
    redistribution heat pump behind the -26.5 Pa/cell/h venting; see the
    ``_THETA_LIMITER_MAX_K`` comment).  For any physically reasonable theta the
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
    mu_base = carry.base_state.mub if carry.base_state is not None else _base_mu(state)
    mu_total = mu_base + state.mu_perturbation
    metrics = namelist.metrics
    # Real advance_w inputs for the legacy non-prep helper path so the WRF
    # implicit-w solve receives finite, consistent coefficients (matches the
    # production prep-path semantics): real c2a from the dry EOS, real dry cqw,
    # base pressure/geopotential, and terrain ht = phb(sfc)/g.
    p_base = (
        carry.base_state.pb.astype(jnp.float64)
        if carry.base_state is not None
        else (state.p_total - state.p_perturbation).astype(jnp.float64)
    )
    ph_base = (
        carry.base_state.phb.astype(jnp.float64)
        if carry.base_state is not None
        else (state.ph_total - state.ph_perturbation).astype(jnp.float64)
    )
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
        ph=state.ph_perturbation.astype(jnp.float64),
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



def _w_coriolis_curvature(u, v, muu, muv, mts, dtype, *, wrf_real=False):
    """WRF rk_tendency coriolis + curvature ``rw_tend`` terms on interior w faces 1..nz-1.

    ``ru``/``rv`` follow couple_momentum; ``wrf_real`` (GPUWRF_CARRY_REAL_ALL) uses WRF's
    REAL ``rv = v*(c1h*muv+c2h)*msfv_inv`` (reciprocal multiply) instead of the division.
    """
    c1h_col = mts.c1h[:, None, None]
    c2h_col = mts.c2h[:, None, None]
    u_stage = u.astype(dtype)
    v_stage = v.astype(dtype)
    ru_stage = (c1h_col * muu[None, :, :] + c2h_col) * u_stage / mts.msfuy[None, :, :]
    if wrf_real:
        rv_stage = v_stage * (c1h_col * muv[None, :, :] + c2h_col) * (1.0 / mts.msfvx)[None, :, :]
    else:
        rv_stage = (c1h_col * muv[None, :, :] + c2h_col) * v_stage / mts.msfvx[None, :, :]
    # x/y de-stagger to mass points, then fnm/fnp vertical face average.
    ru_m = 0.5 * (ru_stage[:, :, :-1] + ru_stage[:, :, 1:])
    u_m = 0.5 * (u_stage[:, :, :-1] + u_stage[:, :, 1:])
    rv_m = 0.5 * (rv_stage[:, :-1, :] + rv_stage[:, 1:, :])
    v_m = 0.5 * (v_stage[:, :-1, :] + v_stage[:, 1:, :])
    nzs = int(ru_m.shape[0])
    fnm_f = mts.fnm[1:nzs, None, None]
    fnp_f = mts.fnp[1:nzs, None, None]

    def _face(field):
        return fnm_f * field[1:nzs, :, :] + fnp_f * field[: nzs - 1, :, :]

    ru_f = _face(ru_m)
    u_f = _face(u_m)
    rv_f = _face(rv_m)
    v_f = _face(v_m)
    mxy = (mts.msftx / mts.msfty)[None, :, :]
    cor_f = mts.e[None, :, :] * (
        mts.cosa[None, :, :] * ru_f - mxy * mts.sina[None, :, :] * rv_f
    )
    curv_f = _W_RERADIUS * (ru_f * u_f + mxy * rv_f * v_f)
    return cor_f + curv_f

def _acoustic_core_state_from_prep(
    carry: OperationalCarry,
    prep: SmallStepPrepState,
    pressure: CalcPRhoStep0,
    namelist: OperationalNamelist,
    tendencies: Tendencies,
    *,
    lead_seconds=None,
    bdy_relax: SpecifiedRelaxTendencies | None = None,
) -> AcousticCoreState:
    """Build the acoustic work-state directly from WRF ``small_step_prep``."""

    state = prep.entry_state
    nested_frozen_bundle = _nested_frozen_wrf_boundary_active(namelist)
    # GPUWRF_CARRY_REAL_ALL: the glue below (coupled stage momenta, Coriolis/curvature,
    # boundary work masses, staged work fields) runs in WRF REAL with REAL metrics;
    # the pg_buoy_w/rhs_ph operator inputs keep their owners' precision.
    from gpuwrf.kernels.dyn_carry_fp32 import real_all_enabled
    real_glue = real_all_enabled()
    glue_dtype = jnp.float32 if real_glue else jnp.float64
    if real_glue:
        from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
        glue_metrics = real_metrics(namelist.metrics)
    else:
        glue_metrics = namelist.metrics
    # REAL grid%p / ph' operands for pg_buoy_w and rhs_ph under either REAL flag
    # (b-diff's GPUWRF_DYN_REAL_ALL owns those operators; it composes with this one).
    operator_dtype = (
        jnp.float32 if real_glue or os.environ.get("GPUWRF_DYN_REAL_ALL", "0") == "1" else jnp.float64
    )
    # ``lead_seconds`` is the candidate package endpoint.  The corrected nested
    # path consumes its retained record tendency directly; released specified
    # callers retain their historical stage-end target clock below.
    theta_pert = (state.theta - prep.theta_offset).astype(glue_dtype)
    ph_base = prep.phb
    # F7H: WRF builds the large-step vertical PGF/buoyancy ``rw_tend`` ONCE per RK
    # stage in rk_tendency (module_em.F:1361-1368) by calling pg_buoy_w with the
    # stage diagnostic ``grid%p`` and the stage perturbation dry mass
    # ``mu'`` (grid%mu_2).  In WRF that ``grid%p`` is the FULL-perturbation
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
    #
    # v0.14 WRF-native advance_w oracle (proofs/v014/wrf_native_advance_w_dump):
    # WRF rk_tendency consumes the CARRIED ``grid%p`` — the calc_p_rho_phi
    # diagnostic refreshed at the END of the previous RK stage/step — never a
    # stage-entry recompute.  The JAX carry maintains exactly that leaf
    # (``_refresh_grid_p_from_finished``), so use it directly.  Recomputing here
    # was (a) WRF-unfaithful at RK1, where the recompute sees the re-init/post-
    # physics prognostic fields instead of the carried diagnostic (Switzerland
    # h36 native dump: recompute-vs-carried p differs 0.30 Pa interior / 1.1 Pa
    # at the lowest mass levels, which pg_buoy_w amplifies to the DOMINANT
    # rw_tend error, interior rmse 511 of 1337; with the carried p the JAX
    # pg_buoy_w matches the WRF-native term to 4.6e-4), and (b) a wasted full
    # diagnostics pass per RK stage.
    nz_stage = int(prep.theta_work.shape[0])
    ny_stage = int(prep.theta_work.shape[1])
    nx_stage = int(prep.theta_work.shape[2])
    # WRF rk_tendency passes grid%mu_2 itself (module_em.F:730); small_step_prep saves the same
    # array as mu_save (module_small_step_em.F:1904). mut - mub would difference a rounded total.
    mu_prime_stage = prep.mu_save  # stage perturbation dry mass mu' (WRF grid%mu_2)
    grid_p_full = state.p_perturbation.astype(operator_dtype)
    # MOIST-CQW (default ON, GPUWRF_MOIST_CQW=0 disables for bisection): WRF builds the large-step
    # vertical PGF/buoyancy with the full moist water-mass loading
    # (calc_cq cqw=0.5*qtot then pg_buoy_w cq1/cq2, module_big_step_utilities_em.F:
    # 856-870,2474-2497).  The dry specialization omits -cq2*(c1f*mub+c2f), so the
    # acoustic solver relaxes the column to DRY hydrostatic balance; WRF/CPU rides
    # the MOIST column (proofs/v014/moist_cqw_pressure_dynamics_closure: GPU
    # P+PB(k0) off-dry ~8 Pa / off-moist ~200 Pa; CPU off-moist ~13 Pa).  cqw is
    # computed once per RK stage from the stage-entry total moisture and held
    # constant through the acoustic loop (matches WRF calc_cq cadence); no
    # host/device transfer, no large transients.  Bit-identical to dry when
    # qtot=0.
    moist_cqw_solver_stage = None
    if _moist_cqw_enabled():
        qtot_stage = (
            state.qv + state.qc + state.qr + state.qi + state.qs + state.qg
        ).astype(grid_p_full.dtype)
        cqw_calc_stage = moist_cqw_calc_face(qtot_stage)
        rw_tend_stage, moist_cqw_solver_stage = pg_buoy_w_moist(
            grid_p_full,
            mu_prime_stage,
            prep.mub,
            cqw_calc_stage,
            c1f=namelist.metrics.c1f,
            c2f=namelist.metrics.c2f,
            rdnw=namelist.metrics.rdnw,
            rdn=namelist.metrics.rdn,
            msfty=namelist.metrics.msfty,
            gravity=GRAVITY_M_S2,
        )
    else:
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
    if real_glue:
        rw_tend_stage = rw_tend_stage.astype(glue_dtype)
    rw_tend_stage = rw_tend_stage + tendencies.w

    # v0.14 WRF-native advance_w oracle: WRF rk_tendency ALSO adds the
    # cosine-Coriolis and curvature vertical-momentum terms to ``rw_tend``
    # (coriolis, module_big_step_utilities_em.F:3836-3843; curvature,
    # :4283-4291; both on interior w faces k=2..kde-1 with the coupled stage
    # momenta and the fnm/fnp face weights — module_em.F passes fnm/fnp as the
    # fzm/fzp dummies).  These were missing here: the Switzerland h36 native
    # rk_tendency dump measures them at interior rmse 172 (coriolis) + 7.2
    # (curvature) of the 1318-rms total rw_tend, the second-largest rw_tend gap
    # after the carried-p fix above.  GPUWRF_W_CORIOLIS=0 disables for bisection.
    if os.environ.get("GPUWRF_W_CORIOLIS", "1") != "0":
        nzs = int(state.u.shape[0])
        rw_tend_stage = rw_tend_stage.at[1:nzs, :, :].add(_w_coriolis_curvature(
            state.u, state.v, prep.muu, prep.muv, glue_metrics, glue_dtype, wrf_real=real_glue,
        ))
    if _W_DAMP_STAGE and int(namelist.w_damping) == 1:
        # BD85: WRF rk_tendency adds w_damp ONCE per RK stage (module_em.F:738) with the stage
        # grid%ww (calc_ww_cp = prep.ww_save), grid%w_2, grid%mut and the full grid%dt; the native
        # acoustic never applied it and the legacy in-acoustic form used small-step operands.
        rw_tend_stage = _w_damp_rw_tend_wrf(
            rw_tend_stage, ww=prep.ww_save, w=state.w, mut=prep.mut,
            c1f=glue_metrics.c1f, c2f=glue_metrics.c2f, rdnw=glue_metrics.rdnw, dt=float(namelist.dt_s),
        )

    # F7J item 1 (PRIME): the large-step geopotential-equation RHS ``rhs_ph`` was
    # stubbed (``carry.ph_tend`` stayed 0; ``accumulate_ph_tend`` never wired in),
    # so the w/phi acoustic restoring loop never closed and the warm-bubble
    # buoyancy pumped without saturating.  WRF computes it once per RK stage in
    # ``rk_tendency`` (module_em.F:1254-1266 -> rhs_ph,
    # module_big_step_utilities_em.F:1365-2232) using the STAGE explicit omega
    # ``wwE = grid%ww`` and the STAGE geopotential perturbation ``ph``.  This is
    # the large-step (frozen-during-acoustic-loop) half of the geopotential
    # tendency; ``advance_w_wrf`` adds the small-step half (omega/ph_1 evolution).
    # v0.14 acoustic continuation: WRF top-row extrapolation weights cfn/cfn1
    # (used by the real-case rhs_ph top-face advection row when open-top).
    _dn_top = glue_metrics.dn[-1]
    _dn_safe = jnp.where(jnp.abs(_dn_top) > 1.0e-30, _dn_top, jnp.asarray(1.0, dtype=_dn_top.dtype))
    _cfn = (0.5 * glue_metrics.dnw[-1] + glue_metrics.dn[-1]) / _dn_safe
    _cfn1 = -0.5 * glue_metrics.dnw[-1] / _dn_safe
    _periodic_x, _specified, _nested = _acoustic_lateral_bc_flags(namelist)
    ph_tend_stage = rhs_ph_wrf(
        u=state.u,
        v=state.v,
        # v0.14: WRF rhs_ph consumes the FRESH rk_step_prep calc_ww_cp stage
        # omega (grid%ww), not a carried post-acoustic omega; prep.ww_save now
        # holds exactly that (see advance_stage).
        ww=prep.ww_save,
        ph=state.ph_perturbation.astype(operator_dtype),
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
        # v0.14 real-case horizontal phi advection (the Switzerland p/ph-first
        # stage divergence root): WRF order<=6 with map factors + specified
        # trims when the case's h_sca_adv_order >= 4 and lateral BCs are
        # specified/nested; idealized callers (order 2 default) are unchanged.
        advective_order=int(namelist.h_sca_adv_order),
        specified=bool(_specified or _nested),
        msfux=namelist.metrics.msfux,
        msfvy=namelist.metrics.msfvy,
        cfn=_cfn,
        cfn1=_cfn1,
        top_lid=bool(namelist.top_lid),
    )

    # WIND-FIX: stage coupled boundary operands for momentum, consumed by
    # ``advance_uv_wrf`` inside the acoustic loop.  Released paths retain the
    # historical absolute work targets.  The corrected nested path stages the
    # exact coupled ``spec_bdytend`` arrays for the complete ring.
    # ``small_step_finish_wrf`` reconstructs the boundary velocity ``u_bdy``:
    #     u = (msf*u_work + u_save*mass_cur)/mass_stage
    #  => u_work_bdy = (u_bdy*mass_stage - u_save*mass_cur)/msf .
    # Only staged when the real-case lateral boundary is active; ``None`` keeps the
    # idealized / replay / bare-core paths on the unmodified PGF advance.
    u_work_bdy = None
    v_work_bdy = None
    if bool(namelist.run_boundary) and lead_seconds is not None:
        cadence = float(namelist.boundary_config.update_cadence_s)
        if nested_frozen_bundle:
            u_work_bdy = specified_boundary_tendency(
                state.u_bdy,
                lead_seconds,
                cadence,
                z_len=int(state.u.shape[0]),
                y_len=int(state.u.shape[1]),
                x_len=int(state.u.shape[2]),
                dtype=state.u.dtype,
                config=namelist.boundary_config,
            )
            v_work_bdy = specified_boundary_tendency(
                state.v_bdy,
                lead_seconds,
                cadence,
                z_len=int(state.v.shape[0]),
                y_len=int(state.v.shape[1]),
                x_len=int(state.v.shape[2]),
                dtype=state.v.dtype,
                config=namelist.boundary_config,
            )
        else:
            c1h = glue_metrics.c1h[:, None, None]
            c2h = glue_metrics.c2h[:, None, None]
            mass_u_cur = c1h * prep.muu[None, :, :] + c2h
            # small_step_finish uses the stage-frozen muus/muvs face masses.
            mass_u_stage = c1h * prep.muus[None, :, :] + c2h
            mass_v_cur = c1h * prep.muv[None, :, :] + c2h
            mass_v_stage = c1h * prep.muvs[None, :, :] + c2h
            u_bdy_strip = interpolate_boundary_leaf(state.u_bdy, lead_seconds, cadence)
            v_bdy_strip = interpolate_boundary_leaf(state.v_bdy, lead_seconds, cadence)
            u_work_bdy = normal_bdy_work_target_u(
                u_bdy_strip,
                prep.u_save,
                mass_u_cur,
                mass_u_stage,
                namelist.metrics.msfuy,
                config=namelist.boundary_config,
                coupled_boundary_leaves=False,
            )
            v_work_bdy = normal_bdy_work_target_v(
                v_bdy_strip,
                prep.v_save,
                mass_v_cur,
                mass_v_stage,
                namelist.metrics.msfvx,
                config=namelist.boundary_config,
                coupled_boundary_leaves=False,
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
        and not nested_frozen_bundle
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

    # v0.14 SPECIFIED-domain WRF boundary cadence (stage3/wrapper sprint).
    # Root evidence (proofs/v014/switzerland_stage3_wrapper_cadence.json): with
    # the once-per-step end-of-step nudge, the spec-zone ph drifted 126-276
    # m2/s2 within ONE step (ring-0 p err ~200 Pa; WRF per-stage band increment
    # 0.53) because the JAX small step advances the ring with full dynamics
    # while WRF excludes it and walks it along the wrfbdy trajectory
    # (spec_bdyupdate/_ph, zero_grad_bdy w) every substep, with relax_bdy_dry
    # tendencies owning rings 1..relax_zone-1 per stage.
    mu_spec_target = None
    muts_spec_target = None
    muave_spec_target = None
    theta_spec_target = None
    u_spec_tan_target = None
    v_spec_tan_target = None
    w_spec_target = None
    _spec_cadence = lead_seconds is not None and _specified_bdy_cadence_active(namelist)
    _nested_cadence = lead_seconds is not None and nested_frozen_bundle
    if _spec_cadence or _nested_cadence:
        cfg_b = namelist.boundary_config
        cadence = float(cfg_b.update_cadence_s)
        dtype_s = state.ph_perturbation.dtype
        nz_m = int(state.theta.shape[0])
        ny_m = int(state.theta.shape[1])
        nx_m = int(state.theta.shape[2])
        # (a) relax-zone ph tendency (relax_bdy_dry 'h' from the step-start
        # reference, step-constant) -> flows through advance_w every substep.
        if bdy_relax is not None:
            if real_glue:
                ph_tend_stage = ph_tend_stage.astype(glue_dtype)
            ph_tend_stage = ph_tend_stage + bdy_relax.ph
        if _nested_cadence:
            # Exact pristine-WRF live-nest ring cadence.  spec_bdy_dry writes
            # the coupled boundary-record tendencies once per RK stage;
            # spec_bdyupdate then consumes the same arrays after every acoustic
            # primitive.  No stage-end absolute target or muave operand exists.
            mu_spec_target = specified_boundary_tendency(
                state.mu_bdy,
                lead_seconds,
                cadence,
                z_len=1,
                y_len=ny_m,
                x_len=nx_m,
                dtype=dtype_s,
                config=cfg_b,
            )[0]
            muts_spec_target = mu_spec_target
            theta_spec_target = specified_boundary_tendency(
                state.theta_bdy,
                lead_seconds,
                cadence,
                z_len=nz_m,
                y_len=ny_m,
                x_len=nx_m,
                dtype=dtype_s,
                config=cfg_b,
            )
            ph_bdy_target_full = specified_boundary_tendency(
                state.ph_bdy,
                lead_seconds,
                cadence,
                z_len=int(state.ph_perturbation.shape[0]),
                y_len=ny_m,
                x_len=nx_m,
                dtype=dtype_s,
                config=cfg_b,
            )
            ph_save_for_spec = prep.ph_save
            w_spec_target = specified_boundary_tendency(
                state.w_bdy,
                lead_seconds,
                cadence,
                z_len=int(state.w.shape[0]),
                y_len=ny_m,
                x_len=nx_m,
                dtype=dtype_s,
                config=cfg_b,
            )
        else:
            # Preserve the released SPECIFIED-domain stage-end program exactly.
            lead_stage = lead_seconds + float(prep.dt_rk)
            ph_strip_stage = interpolate_boundary_leaf(
                state.ph_bdy, lead_stage, cadence
            )
            ph_bdy_target_full = _full_ring_target_from_leaf(
                ph_strip_stage,
                int(state.ph_perturbation.shape[0]),
                ny_m,
                nx_m,
                dtype_s,
            )
            ph_save_for_spec = prep.ph_save
            mu_strip = interpolate_boundary_leaf(state.mu_bdy, lead_stage, cadence)
            mu_pin = _full_ring_target_from_leaf(mu_strip, 1, ny_m, nx_m, dtype_s)[0]
            muts_pin = prep.mub + mu_pin
            th_strip = interpolate_boundary_leaf(
                state.theta_bdy, lead_stage, cadence
            )
            th_pin = _full_ring_target_from_leaf(
                th_strip, nz_m, ny_m, nx_m, dtype_s
            )
            _c1h = glue_metrics.c1h[:, None, None]
            _c2h = glue_metrics.c2h[:, None, None]
            mass_pin = _c1h * muts_pin[None, :, :] + _c2h
            mass_cur = _c1h * prep.mut[None, :, :] + _c2h
            mu_spec_target = mu_pin
            muts_spec_target = muts_pin
            muave_spec_target = muts_pin - prep.mut
            theta_spec_target = (
                mass_pin * (th_pin - prep.theta_offset)
                - mass_cur * prep.t_save
            )
            # TANGENTIAL ring-0 wind work pins; normal targets cover W/E u and
            # S/N v only on this released path.
            _c1h3 = glue_metrics.c1h[:, None, None]
            _c2h3 = glue_metrics.c2h[:, None, None]
            mass_u_cur_t = _c1h3 * prep.muu[None, :, :] + _c2h3
            mass_u_stage_t = _c1h3 * prep.muus[None, :, :] + _c2h3
            mass_v_cur_t = _c1h3 * prep.muv[None, :, :] + _c2h3
            mass_v_stage_t = _c1h3 * prep.muvs[None, :, :] + _c2h3
            u_strip_stage = interpolate_boundary_leaf(
                state.u_bdy, lead_stage, cadence
            )
            v_strip_stage = interpolate_boundary_leaf(
                state.v_bdy, lead_stage, cadence
            )
            u_spec_tan_target = tangential_bdy_work_target_u(
                u_strip_stage,
                prep.u_save,
                mass_u_cur_t,
                mass_u_stage_t,
                namelist.metrics.msfuy,
                config=cfg_b,
                coupled_boundary_leaves=False,
            )
            v_spec_tan_target = tangential_bdy_work_target_v(
                v_strip_stage,
                prep.v_save,
                mass_v_cur_t,
                mass_v_stage_t,
                namelist.metrics.msfvx,
                config=cfg_b,
                coupled_boundary_leaves=False,
            )

    return AcousticCoreState(
        # v0.14: the small-step omega work array starts from the FRESH stage
        # diagnostic (WRF grid%ww at loop entry = rk_step_prep calc_ww_cp);
        # advance_mu_t overwrites the interior faces each substep and the
        # bottom/top faces stay zero, exactly as WRF.
        ww=prep.ww_save,
        ww_1=prep.ww_save,
        u=prep.u_work,
        u_1=prep.u_save,
        v=prep.v_work,
        v_1=prep.v_save,
        w=prep.w_work,
        mu=(prep.mu_save + prep.mu_work).astype(glue_dtype),
        mut=prep.mut.astype(glue_dtype),
        # F7G: stage-entry small-step mass-WORK average is ZERO; advance_mu_t
        # (module_small_step_em.F:1102-1108) rebuilds it from actual small-step
        # mass evolution.  For a fixed-mass mu'=0 thermal it stays zero.
        muave=jnp.zeros_like(prep.mu_work, dtype=glue_dtype),
        muts=prep.muts.astype(glue_dtype),
        muu=prep.muu,
        muv=prep.muv,
        # Pristine small_step_prep zeroes mudf at RK1 before the first
        # advance_uv; it is then recurrent only within/across later RK stages.
        # Candidate child only: do not import cross-step divergence memory.
        mudf=_stage_entry_mudf(
            carry.mudf,
            rk_step=int(prep.rk_step),
            nested_frozen_bundle=nested_frozen_bundle,
        ),
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
        ph_tend=ph_tend_stage.astype(glue_dtype),
        ph=prep.ph_work.astype(glue_dtype),
        p=pressure.p,
        t_2ave=jnp.zeros_like(prep.theta_work),
        dnw=namelist.metrics.dnw,
        fnm=namelist.metrics.fnm,
        fnp=namelist.metrics.fnp,
        rdnw=namelist.metrics.rdnw,
        c1h=namelist.metrics.c1h,
        c2h=namelist.metrics.c2h,
        msfuy=namelist.metrics.msfuy,
        msfvx_inv=1.0 / glue_metrics.msfvx,
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
        # MOIST-CQW: the post-pg_buoy_w cqw=cq1 (consumed by calc_coef_w + the
        # advance_w implicit pressure term) when the moist path is enabled; the
        # dry cqw (interior=1) otherwise.  Bit-identical when qtot=0.
        cqw=(
            moist_cqw_solver_stage
            if moist_cqw_solver_stage is not None
            else dry_cqw(
                int(prep.theta_work.shape[0]),
                int(prep.theta_work.shape[1]),
                int(prep.theta_work.shape[2]),
                dtype=prep.theta_work.dtype,
            )
        ),
        c1f=namelist.metrics.c1f,
        c2f=namelist.metrics.c2f,
        rdn=namelist.metrics.rdn,
        phb=prep.phb,
        ph_1=prep.ph_1,
        # Terrain height ht = phb(surface)/g (WRF advance_w lower BC :1417-1429).
        ht=prep.phb[0, :, :] / GRAVITY_M_S2,
        pm1=pressure.pm1,
        ru_m=jnp.zeros_like(prep.u_work),
        rv_m=jnp.zeros_like(prep.v_work),
        ww_m=jnp.zeros_like(carry.ww, dtype=glue_dtype),
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
        # unless the nested force_geopotential=False boundary is active); the
        # v0.14 SPECIFIED cadence reuses the same in-loop spec_bdyupdate_ph slot
        # with the stage-end-interpolated wrfbdy leaf.
        ph_bdy_target=ph_bdy_target_full,
        ph_save_for_spec=ph_save_for_spec,
        # v0.14 SPECIFIED cadence ring-0 work pins (None on every other path).
        mu_spec_target=mu_spec_target,
        muts_spec_target=muts_spec_target,
        muave_spec_target=muave_spec_target,
        theta_spec_target=theta_spec_target,
        u_spec_tan_target=u_spec_tan_target,
        v_spec_tan_target=v_spec_tan_target,
        w_spec_target=w_spec_target,
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

    # v0.20 fp32 INTEGRATION bit-identity fix (the PRIMARY source): choose the
    # base-field source for the closing grid%p diagnostic by storage mode so
    # fp64_default stays BYTE-IDENTICAL to the pre-S4 baseline while the
    # perturbation-authoritative fp32 mode stays cancellation-safe.
    #
    # The S4 merge unconditionally switched phb and the p_total base term from the
    # historical FINISHED-state reconstruction
    # (next_state.{ph_total,p_total} - next_state.{ph_perturbation,p_perturbation})
    # to the pristine ENTRY-state prep.pb/prep.phb. The base is physically
    # constant across the acoustic substeps, but in floating point the FINISHED
    # totals/primes have evolved, so entry-base != finished-base at fp64 round-off.
    # That feeds diagnose_pressure_al_alt and the written-back p_total every RK
    # stage -> cascades into P/PH/U/V/W/QKE and (via total-minus-pert) the output
    # PB/PHB/MU/MUB + downstream radiation/surface, breaking fp64_default
    # bit-identity (GPU all-7 byte-compare: PB maxΔ~1.6e-2 Pa). This refresh runs
    # AFTER small_step_finish and OVERWRITES p/p_total, which is why the earlier
    # small_step_finish gating alone had no effect.
    #
    # For the mixed_perturb_fp32 mode the pristine fp64 base (prep.pb/prep.phb) is
    # REQUIRED -- there next_state.p_perturbation is stored fp32 and the
    # total-minus-pert reconstruction would re-introduce the fp32 cancellation the
    # perturbation-authoritative design avoids. Gate on the static acoustic
    # precision mode (compile-time -> zero runtime cost; the fp64 branch re-emits
    # the exact pre-S4 HLO -> byte-identical).
    _keep_base = (
        is_mixed_perturb_fp32_mode(namelist.acoustic_precision_mode)
        or os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1"
    )
    base = BaseState(
        pb=prep.pb,  # pb was prep.pb pre-merge too (unchanged); only phb regressed.
        phb=prep.phb if _keep_base else (next_state.ph_total - next_state.ph_perturbation),
        mub=prep.mub,
        t0=jnp.asarray(prep.theta_offset),
        theta_base=jnp.full_like(next_state.theta, prep.theta_offset),
    )
    p_pert, _al, _alt = diagnose_pressure_al_alt(
        next_state, base, namelist.metrics, hypsometric_opt=int(namelist.hypsometric_opt)
    )
    if _keep_base:
        p_total = prep.pb + p_pert
    else:
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
    stage: "_RKStageDescriptor | None" = None,
) -> OperationalCarry:
    if (
        carry.h_diabatic is not None
        and stage is not None
        and namelist is not None
        and int(stage.rk_step) == int(namelist.rk_order)
    ):
        # module_small_step_em.F:417-423: final stage removes dts*N*mass*h_diabatic.
        next_state = small_step_finish_wrf(
            prep, acoustic, h_diabatic=carry.h_diabatic,
            h_diabatic_seconds=float(stage.dts_rk) * int(stage.number_of_small_timesteps),
        )
    else:
        next_state = small_step_finish_wrf(prep, acoustic)
    if namelist is not None:
        next_state = _refresh_grid_p_from_finished(next_state, prep, namelist)
    if (
        namelist is not None
        and is_mixed_perturb_fp32_mode(namelist.acoustic_precision_mode)
        and carry.base_state is not None
    ):
        next_state = _apply_mixed_perturb_fp32_storage(next_state, carry.base_state)
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
        ph_save=prep.ph_save.astype(jnp.float64),
        mu_save=prep.mu_save.astype(jnp.float64),
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


class _TimeAveragedScalarMassFluxes(NamedTuple):
    """WRF ``sumflux`` result retained only inside one RK-stage trace."""

    ru_full: jax.Array
    rv_full: jax.Array
    ww: jax.Array


class _AcousticScalarTransportResult(NamedTuple):
    """Finished acoustic result and its scalar-transport mass fluxes."""

    result: object
    mass_fluxes: _TimeAveragedScalarMassFluxes


def _finalize_time_averaged_scalar_mass_fluxes(
    acoustic: AcousticCoreState,
    prep: SmallStepPrepState,
    *,
    number_of_small_timesteps: int,
) -> _TimeAveragedScalarMassFluxes:
    """Finish pristine-WRF ``sumflux`` for moisture/other scalars.

    The acoustic core accumulates the live coupled work arrays after every
    sound step.  WRF divides those accumulators by the number of sound steps
    and adds the stage-entry linear/save flux before ``rk_scalar_tend``
    (``solve_em.F:1566-1581`` and ``module_small_step_em.F:1559-1628``).
    The bundle stays internal to this RK-stage trace; it is not an operational
    carry or executable-result leaf.
    """

    n = int(number_of_small_timesteps)
    if n < 1:
        raise ValueError("sumflux requires at least one acoustic substep")
    if acoustic.ru_m is None or acoustic.rv_m is None or acoustic.ww_m is None:
        raise ValueError("sumflux accumulators are absent from the acoustic state")

    c1h = prep.c1h[:, None, None]
    c2h = prep.c2h[:, None, None]
    ru_full = acoustic.ru_m / n + (
        (c1h * prep.muu[None, :, :] + c2h)
        * prep.u_save
        / prep.msfuy[None, :, :]
    )
    rv_full = acoustic.rv_m / n + (
        (c1h * prep.muv[None, :, :] + c2h)
        * prep.v_save
        / prep.msfvx[None, :, :]
    )
    ww = acoustic.ww_m / n + prep.ww_save
    return _TimeAveragedScalarMassFluxes(ru_full, rv_full, ww)


def _scalar_transport_velocities_from_sumflux(
    stage_velocities: CoupledVelocities,
    mass_fluxes: _TimeAveragedScalarMassFluxes,
) -> CoupledVelocities:
    """Replace only scalar transport fluxes with WRF's acoustic average."""

    nx = int(stage_velocities.ru.shape[-1])
    ny = int(stage_velocities.rv.shape[-2])
    return dataclasses.replace(
        stage_velocities,
        ru=mass_fluxes.ru_full[..., :nx],
        rv=mass_fluxes.rv_full[:, :ny, :],
        rom=mass_fluxes.ww,
        ru_full=(mass_fluxes.ru_full if bool(stage_velocities.specified) else None),
        rv_full=(mass_fluxes.rv_full if bool(stage_velocities.specified) else None),
    )


def _acoustic_scan(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    stage: _RKStageDescriptor,
    prep: SmallStepPrepState,
    pressure: CalcPRhoStep0,
    tendencies: Tendencies,
    lead_seconds=None,
    capture_pre_halo: bool = False,
    capture_rca: bool = False,
    capture_phase_tap: bool = False,
    return_scalar_transport: bool = False,
    bdy_relax: SpecifiedRelaxTendencies | None = None,
) -> (
    OperationalCarry
    | _PreHaloCaptureResult
    | _RcaAcousticScanResult
    | CorrectedNiPhaseTapResult
    | _AcousticScalarTransportResult
):
    from gpuwrf.diagnostics.census import count_work

    if sum(bool(value) for value in (capture_pre_halo, capture_rca, capture_phase_tap)) > 1:
        raise ValueError("pre-halo, RCA, and phase-tap captures are mutually exclusive")
    acoustic = _acoustic_core_state_from_prep(
        carry, prep, pressure, namelist, tendencies, lead_seconds=lead_seconds,
        bdy_relax=bdy_relax,
    )
    native_fp32 = os.environ.get("GPUWRF_DYN_FP32", "0") == "1"
    if native_fp32:
        if capture_rca or capture_phase_tap:
            raise ValueError("GPUWRF_DYN_FP32 does not support legacy RCA/phase-tap capture")
        # Stage exactly once, before rounding. Never recover the small work
        # delta from fp32 total masses inside the acoustic recurrence.
        spec_work = None
        if acoustic.muts_spec_target is not None:
            spec_work = (acoustic.muts_spec_target - prep.mut
                         if not _nested_frozen_wrf_boundary_active(namelist)
                         else acoustic.mu_spec_target)
        acoustic = acoustic.replace(mu_work=prep.mu_work,mu_work_spec_target=spec_work)
        acoustic = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype,jnp.floating) else x,
            acoustic,
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
        # ADR-038/M2 seam (env-gated, default OFF): fused Pallas coefficient kernel,
        # bitwise-exact vs XLA:GPU in the device bake-off (verdict:
        # <DATA_ROOT>/wrf_gpu2/v025/m2/bakeoff_20260918_rerun/
        # m2_bakeoff_verdict_device_final.json, PASS). Metrics namespace flattened
        # to the arrays the wrapper takes; otherwise identical arguments.
        if native_fp32:
            # Built below, once stage_cfg is available.
            a = alpha = gamma = None
        elif os.environ.get("GPUWRF_FUSED_VERTICAL", "0") == "1":
            from gpuwrf.kernels.fused_vertical_implicit import calc_coef_w_pallas
            _m = namelist.metrics
            a, alpha, gamma = calc_coef_w_pallas(
                prep.mut,
                _m.c1h,
                _m.c2h,
                _m.c1f,
                _m.c2f,
                _m.rdn,
                _m.rdnw,
                dt=float(stage.dts_rk),
                epssm=float(namelist.epssm),
                top_lid=bool(namelist.top_lid),
                cqw=cqw_field,
                c2a=prep.c2a,
                interpret=False,
            )
        else:
            a, alpha, gamma = calc_coef_w_wrf_coefficients(
                prep.mut,
                namelist.metrics,
                dt=float(stage.dts_rk),
                epssm=float(namelist.epssm),
                top_lid=bool(namelist.top_lid),
                cqw=cqw_field,
                c2a=prep.c2a,
            )
        # NOTE (v0.15 kernel probe, NEGATIVE result): precomputing the four
        # advance_w stage-constant denominator arrays here (like a/alpha/gamma)
        # and closing them into the substep scan was tried and REVERTED -- it
        # measured SLOWER (inline broadcast-FMA recompute fuses for free;
        # materialized arrays cost DRAM loads) and was NOT bit-identical
        # in-program (XLA FMA contraction is fusion-context-dependent). See
        # proofs/perf/v015/ab_compare_v014_base_vs_streamA.json.

        periodic_x, specified, nested = _acoustic_lateral_bc_flags(namelist)
        stage_cfg = AcousticCoreConfig(
            dt=float(stage.dts_rk),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            # GPUWRF_W_DAMP_STAGE moves w_damp to the stage rw_tend (WRF cadence): no in-acoustic copy.
            w_damping=0 if _W_DAMP_STAGE else int(namelist.w_damping),
            damp_opt=int(namelist.damp_opt),
            dampcoef=float(namelist.dampcoef),
            zdamp=float(namelist.zdamp),
            # WIND-FIX: full model dt so the in-loop normal-momentum relaxation
            # weight is scaled to a per-substep increment.
            dt_full=float(namelist.dt_s),
            normal_bdy_relax_strength=getattr(namelist.boundary_config, "normal_bdy_relax_strength", None),
            nested_frozen_wrf_boundary_bundle=_nested_frozen_wrf_boundary_active(
                namelist
            ),
            periodic_x=periodic_x,
            specified=specified,
            nested=nested,
            # v0.14 SPECIFIED cadence: WRF zero_grad_bdy on the spec-zone w
            # work array after advance_w (specified domains only).
            spec_w_zero_grad=bool(
                lead_seconds is not None and _specified_bdy_cadence_active(namelist)
            ),
            spec_zone=int(namelist.boundary_config.spec_zone),
            # Same predicate as _specified_bdy_relax: WRF's relax tendency is
            # already in ru/rv_tend, so no extra per-substep relax-row blend.
            specified_relax_tendency=bool(
                lead_seconds is not None and _specified_bdy_cadence_active(namelist)
            ),
        )
        if native_fp32:
            from gpuwrf.kernels.dyn_acoustic_fp32 import (
                calc_coef_fp32,acoustic_substep_fp32,evolving_payload,state_from_payload,
            )
            native_coefficients = _layout_pin("ac", calc_coef_fp32(acoustic,stage_cfg,apply_boundary_forcing=True))

        # v0.10.0 Wave-A (Opus#1 unroll):
        # NOTE on the reverted carry-split (Opus#2): threading only the ~19
        # evolving leaves through the scan and closing over the ~50 stage-constant
        # leaves was bit-identical, but the warmed A/B was confounded by a one-off
        # cache-miss/recompile artifact and was not cleanly revalidated. The
        # simple full-pytree carry below is the proven non-regressing path; retest
        # any carry split with the corrected cache-hit timing protocol before
        # changing it.
        # See proofs/v0100/inefficiency_ledger.md (Opus#2 = REVERTED).
        counted = carry.census is not None
        native_template = _layout_pin("ac", acoustic)  # scan constants (loop-invariant while operands)
        def encode_acoustic(value):
            return evolving_payload(value) if native_fp32 else value
        def decode_acoustic(value):
            return state_from_payload(native_template,value) if native_fp32 else value
        seed_value = _layout_pin("ac", encode_acoustic(acoustic))
        scan_seed = (seed_value, carry.census) if counted else seed_value

        def body(scan_value, _):
            encoded, census = scan_value if counted else (scan_value, None)
            scan_acoustic = decode_acoustic(_layout_pin("ac", encoded))
            if native_fp32:
                native_result = acoustic_substep_fp32(
                    scan_acoustic,coefficients=native_coefficients,cfg=stage_cfg,
                    apply_boundary_forcing=True,return_guard_events=counted,
                )
                if counted:
                    from gpuwrf.diagnostics.census import count_native_mass_guard_events
                    next_acoustic, guard_events = native_result
                    census = count_native_mass_guard_events(census,guard_events)
                else:
                    next_acoustic = native_result
            else:
                next_acoustic = acoustic_substep_core(
                    scan_acoustic,a=a,alpha=alpha,gamma=gamma,
                    cfg=stage_cfg,cqw=cqw_field,
                )
            encoded_next = _layout_pin("ac", encode_acoustic(next_acoustic))
            value = (encoded_next,count_work(census,"acoustic_trips")) if counted else encoded_next
            return value, None

        acoustic_health = None
        acoustic_target = None
        phase_tap_summary = None
        if capture_phase_tap:
            acoustic, phase_tap_summary = acoustic_substep_core(
                acoustic,
                a=a,
                alpha=alpha,
                gamma=gamma,
                cfg=stage_cfg,
                cqw=cqw_field,
                capture_phase_tap=True,
            )
            scan_seed = (acoustic, count_work(carry.census, "acoustic_trips")) if counted else acoustic
            remaining_substeps = int(stage.number_of_small_timesteps) - 1
            if remaining_substeps:
                scan_seed, _ = jax.lax.scan(
                    body,
                    scan_seed,
                    xs=None,
                    length=remaining_substeps,
                    unroll=_acoustic_unroll(),
                )
        elif capture_rca:
            def observed_body(scan_value, _):
                scan_acoustic, census = scan_value if counted else (scan_value, None)
                next_acoustic, observation = acoustic_substep_core(
                    scan_acoustic,
                    a=a,
                    alpha=alpha,
                    gamma=gamma,
                    cfg=stage_cfg,
                    cqw=cqw_field,
                    observe_mass_primitive=True,
                )
                value = (next_acoustic, count_work(census, "acoustic_trips")) if counted else next_acoustic
                return value, (
                    _rca_acoustic_health(next_acoustic, observation),
                    _rca_acoustic_target(next_acoustic, observation),
                )

            scan_seed, (acoustic_health, acoustic_target) = jax.lax.scan(
                observed_body,
                scan_seed,
                xs=None,
                length=int(stage.number_of_small_timesteps),
                unroll=_acoustic_unroll(),
            )
        else:
            scan_seed, _ = jax.lax.scan(
                body,
                scan_seed,
                xs=None,
                length=int(stage.number_of_small_timesteps),
                unroll=_acoustic_unroll(),
            )
        if counted:
            encoded, census = scan_seed
            acoustic = decode_acoustic(encoded)
            carry = carry.replace(census=census)
        else:
            acoustic = decode_acoustic(scan_seed)
        scalar_mass_fluxes = (
            _finalize_time_averaged_scalar_mass_fluxes(
                acoustic,
                prep,
                number_of_small_timesteps=int(stage.number_of_small_timesteps),
            )
            if bool(return_scalar_transport)
            else None
        )
        next_carry = _carry_from_finished_stage(carry, prep, acoustic, namelist, stage)
        if native_fp32:
            # The native recurrence is fp32; its retained outer-loop interface
            # must match the declared carry types (including mixed storage).
            next_carry = next_carry.replace(**{
                name: getattr(next_carry,name).astype(getattr(carry,name).dtype)
                for name in ("mudf","muave","muts","ph_tend")
            })
        next_carry = _maybe_exchange_sharded_carry_halos(next_carry)
        post_halo_carry = next_carry.replace(state=apply_halo(next_carry.state, halo_spec(namelist.grid)))
        if capture_pre_halo:
            result = _PreHaloCaptureResult(post_halo_carry, next_carry.state)
        elif capture_rca:
            assert acoustic_health is not None
            assert acoustic_target is not None
            result = _RcaAcousticScanResult(
                post_halo_carry, acoustic_health, acoustic_target,
            )
        elif capture_phase_tap:
            assert phase_tap_summary is not None
            result = CorrectedNiPhaseTapResult(post_halo_carry, phase_tap_summary)
        else:
            result = post_halo_carry
        if bool(return_scalar_transport):
            assert scalar_mass_fluxes is not None
            return _AcousticScalarTransportResult(result, scalar_mass_fluxes)
        return result

    del tendencies
    if bool(return_scalar_transport):
        raise ValueError("WRF sumflux scalar transport requires the vertical acoustic solver")
    next_carry = _maybe_exchange_sharded_carry_halos(_with_save_family(carry, carry.state))
    if capture_pre_halo:
        return _PreHaloCaptureResult(next_carry, next_carry.state)
    if capture_rca:
        raise ValueError("RCA capture requires the vertical acoustic solver")
    if capture_phase_tap:
        raise ValueError("phase-tap capture requires the vertical acoustic solver")
    return next_carry


def _stage_transport_omega_ownership_enabled() -> bool:
    """Bind pristine WRF's single ``calc_ww_cp`` owner for stage transport."""

    return True


def _stage_transport_velocities(
    haloed: State,
    namelist: OperationalNamelist,
) -> CoupledVelocities:
    """Build the stage's WRF mass-coupled transport velocities ``ru/rv/rom``.

    WRF builds ``ru/rv/ww`` ONCE per RK stage (``rk_step_prep``/``calc_ww_cp``,
    solve_em.F) and every flux-form advection of that stage -- momentum, theta,
    and each moist scalar -- transports with the SAME arrays.  This helper is the
    single construction point; ``_augment_large_step_tendencies`` and
    ``_moisture_coupled_tendencies`` consume the shared value instead of each
    rebuilding it from the identical inputs.
    """

    metrics = namelist.metrics
    grid = namelist.grid
    vel = couple_velocities_periodic(
        haloed.u,
        haloed.v,
        haloed.mu_total,
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        dnw=metrics.dnw,
        rdx=1.0 / float(grid.projection.dx_m),
        rdy=1.0 / float(grid.projection.dy_m),
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
        msftx=metrics.msftx,
        msfux=metrics.msfux,
        msfvy=metrics.msfvy,
    )
    # v0.14 SPECIFIED-boundary advection degradation: stage the static switch +
    # the edge-faithful FULL-FACE coupled velocities so every flux-form operator
    # (theta/moisture scalar, u, v, w) runs the WRF degraded-tier boundary
    # branch.  Every other path keeps the periodic vel untouched.
    if _specified_adv_degrade_active(namelist):
        ru_full, rv_full = couple_uv_specified(
            haloed.u,
            haloed.v,
            haloed.mu_total,
            c1h=metrics.c1h,
            c2h=metrics.c2h,
            msfuy=metrics.msfuy,
            msfvx=metrics.msfvx,
        )
        # Pristine rk_step_prep/calc_ww_cp constructs ONE stage ``ww`` from the
        # real, non-periodic staggered faces.  That same array is the ``rom``
        # operand of every stage advection call and the acoustic ``ww_save``.
        # The retained path corrected the latter but accidentally left vertical
        # flux advection on couple_velocities_periodic().rom, so one RK stage had
        # two incompatible omega fields at its physical boundary.  Route the
        # already source/oracle-closed edge-faithful construction into the shared
        # CoupledVelocities object; no new carry leaf or transfer is introduced.
        if _stage_transport_omega_ownership_enabled():
            rom = stage_omega_specified(
                haloed.u,
                haloed.v,
                haloed.mu_total,
                c1h=metrics.c1h,
                c2h=metrics.c2h,
                dnw=metrics.dnw,
                rdx=1.0 / float(grid.projection.dx_m),
                rdy=1.0 / float(grid.projection.dy_m),
                msfuy=metrics.msfuy,
                msfvx=metrics.msfvx,
                msftx=metrics.msftx,
            )
            vel = dataclasses.replace(
                vel,
                specified=True,
                ru_full=ru_full,
                rv_full=rv_full,
                rom=rom,
            )
        else:
            # Exact retained trace surface for the authenticated CPU A arm.
            vel = dataclasses.replace(
                vel, specified=True, ru_full=ru_full, rv_full=rv_full
            )
    return vel


def _specified_bdy_cadence_active(namelist: OperationalNamelist) -> bool:
    """Static gate for the v0.14 SPECIFIED-domain WRF boundary cadence.

    Active only when the namelist flag is on, a real lateral boundary runs, the
    domain is specified (not the nested force_geopotential=False child), so
    idealized / periodic / nested-replay paths stay byte-identical.
    """

    if not bool(getattr(namelist, "specified_bdy_cadence", False)):
        return False
    if not bool(namelist.run_boundary):
        return False
    _per_x, _spec, _nest = _acoustic_lateral_bc_flags(namelist)
    return bool(_spec) and bool(namelist.boundary_config.force_geopotential)


def _nested_frozen_wrf_boundary_active(namelist: OperationalNamelist) -> bool:
    """Static gate for the coherent v0.23.4 live-child boundary bundle."""

    if not bool(
        getattr(
            namelist.boundary_config,
            "nested_frozen_wrf_boundary_bundle",
            False,
        )
    ):
        return False
    if not bool(namelist.run_boundary):
        return False
    _per_x, _spec, nested = _acoustic_lateral_bc_flags(namelist)
    return bool(nested) and not bool(namelist.boundary_config.force_geopotential)


def _root_scalar_bdy_rk1_active(namelist: OperationalNamelist) -> bool:
    """B43 gate: GPUWRF_ROOT_SCALAR_BDY_RK1 on a specified root with explicit have_bcs flags."""

    if not root_scalar_bdy_rk1_enabled():
        return False
    config = namelist.boundary_config
    return (
        _specified_bdy_cadence_active(namelist)
        and not _nested_frozen_wrf_boundary_active(namelist)
        and bool(namelist.use_flux_advection)
        and config.have_bcs_moist is not None
        and config.have_bcs_scalar is not None
    )


def nested_boundary_package_endpoint_seconds(
    completed_child_step,
    *,
    child_dt_s: float,
    parent_cadence_s: float,
):
    """Return WRF ``dtbc`` endpoint within the current parent subcycle.

    Child positions are explicitly the integers ``1..parent_ratio``.  The last
    child step maps to ``parent_cadence_s`` (the exact parent boundary), and the
    following step maps back to ``child_dt_s`` in the newly forced two-record
    package.  This avoids both absolute-lead two-leaf saturation and the fragile
    ``fmod == 0`` special case.
    """

    child_dt = float(child_dt_s)
    parent_dt = float(parent_cadence_s)
    if child_dt <= 0.0 or parent_dt <= 0.0:
        raise ValueError("nested boundary child/parent timesteps must be positive")
    ratio_float = parent_dt / child_dt
    ratio = int(round(ratio_float))
    if ratio < 1 or abs(ratio_float - float(ratio)) > 1.0e-12:
        raise ValueError(
            "nested boundary parent cadence must be an integer child-dt multiple"
        )
    step = jnp.asarray(completed_child_step, dtype=jnp.int32)
    position = jnp.mod(step - jnp.asarray(1, dtype=step.dtype), ratio) + 1
    return position.astype(jnp.float64) * child_dt


def nested_boundary_stage_seconds(
    package_endpoint_seconds,
    *,
    child_dt_s: float,
    rk_dt_s: float,
):
    """Direct spec target time for one RK stage inside a child step."""

    return (
        jnp.asarray(package_endpoint_seconds, dtype=jnp.float64)
        - float(child_dt_s)
        + float(rk_dt_s)
    )


def _stage_entry_mudf(
    carried_mudf,
    *,
    rk_step: int,
    nested_frozen_bundle: bool,
):
    """Mirror pristine RK1 ``mudf=0`` without changing released paths."""

    if bool(nested_frozen_bundle) and int(rk_step) == 1:
        return jnp.zeros_like(carried_mudf, dtype=jnp.float64)
    return carried_mudf.astype(jnp.float64)


def _specified_adv_degrade_active(namelist: OperationalNamelist) -> bool:
    """Static gate for the v0.14 SPECIFIED-boundary advection degradation.

    Pristine ``module_advect_em.F`` selects its degraded horizontal stencils
    for ``config_flags%specified .or. config_flags%nested``.  The coherent
    frozen-WRF live-child bundle therefore selects this already-implemented
    branch without relying on the legacy root-only experiment flag.  Released
    children with that bundle off and the legacy specified-root path retain
    their prior gates exactly.
    """

    if _nested_frozen_wrf_boundary_active(namelist):
        return True
    if not bool(getattr(namelist, "specified_adv_degrade", False)):
        return False
    if not bool(namelist.run_boundary):
        return False
    _per_x, _spec, _nest = _acoustic_lateral_bc_flags(namelist)
    return bool(_spec) and bool(namelist.boundary_config.force_geopotential)


def _specified_bdy_relax(
    reference: State,
    namelist: OperationalNamelist,
    lead_seconds,
) -> SpecifiedRelaxTendencies | None:
    """Step-constant WRF ``relax_bdy_dry`` bundle (None unless the cadence is on).

    Computed from the rk1 reference (= the step-start state, exactly the fields
    WRF's rk_step==1 ``relax_bdy_dry`` reads) at the step-start lead; the values
    are therefore identical at every RK stage, matching the WRF tendf fold.
    """

    if lead_seconds is None or not _specified_bdy_cadence_active(namelist):
        return None
    return specified_relax_dry_tendencies(
        reference,
        lead_seconds,
        namelist.metrics,
        float(namelist.dt_s),
        namelist.boundary_config,
    )


def _nested_frozen_bdy_relax(
    reference: State,
    namelist: OperationalNamelist,
    endpoint_seconds,
) -> SpecifiedRelaxTendencies | None:
    """Construct the live-child RK1 frozen dry relax bundle exactly once."""

    if endpoint_seconds is None or not _nested_frozen_wrf_boundary_active(namelist):
        return None
    return specified_relax_dry_tendencies(
        reference,
        endpoint_seconds,
        namelist.metrics,
        float(namelist.dt_s),
        namelist.boundary_config,
        include_nested_w=True,
        coupled_boundary_leaves=True,
    )


class _DiffOpt2TurbulenceFields(NamedTuple):
    """Shared WRF diff_opt=2 turbulence closure fields for one RK stage."""

    d11: jax.Array
    d22: jax.Array
    d33: jax.Array
    d12: jax.Array
    d13: jax.Array
    d23: jax.Array
    bn2: jax.Array
    xkmh: jax.Array
    xkmv: jax.Array
    xkhh: jax.Array
    xkhv: jax.Array
    rdz: jax.Array
    rdzw: jax.Array


def _diffopt2_turbulence_fields(
    haloed: State,
    namelist: OperationalNamelist,
    *,
    dz: jax.Array,
) -> _DiffOpt2TurbulenceFields:
    """Build WRF ``diff_opt=2`` 3-D Smag/TKE coefficients for km_opt=2/3/5."""

    grid = namelist.grid
    dx = float(grid.projection.dx_m)
    dy = float(grid.projection.dy_m)
    rdzw = jnp.ones_like(haloed.theta) / dz
    rdz = jnp.ones_like(haloed.w) / dz
    d11, d22, d33, d12, d13, d23 = deformation_components_3d(
        haloed.u,
        haloed.v,
        haloed.w,
        dx_m=dx,
        dy_m=dy,
        rdz=rdz,
        rdzw=rdzw,
    )
    bn2 = dry_brunt_vaisala_squared(haloed.theta, dz_m=dz)
    km_opt = int(namelist.km_opt)
    isotropic = int(namelist.mix_isotropic) != 0
    if km_opt == 3:
        xkmh, xkmv, xkhh, xkhv = smag3d_km(
            d11,
            d22,
            d33,
            d12,
            d13,
            d23,
            bn2,
            dx_m=dx,
            dy_m=dy,
            rdzw=rdzw,
            dt_s=float(namelist.dt_s),
            c_s=float(namelist.c_s),
            mix_upper_bound=float(namelist.mix_upper_bound),
            isotropic=isotropic,
        )
    elif km_opt in (2, 5):
        smag_xkmh = None
        smag_xkhh = None
        if km_opt == 5:
            smag_xkmh, smag_xkhh = smag2d_horizontal_km(
                d11,
                d22,
                d12,
                dx_m=dx,
                dy_m=dy,
                c_s=float(namelist.c_s),
                diff_opt_for_slope=2,
            )
        xkmh, xkmv, xkhh, xkhv = tke3d_km(
            haloed.qke,
            haloed.theta,
            bn2,
            dx_m=dx,
            dy_m=dy,
            rdzw=rdzw,
            dt_s=float(namelist.dt_s),
            km_opt=km_opt,
            c_k=float(namelist.c_k),
            mix_upper_bound=float(namelist.mix_upper_bound),
            isotropic=isotropic,
            smag_xkmh=smag_xkmh,
            smag_xkhh=smag_xkhh,
        )
    else:
        raise ValueError(f"diff_opt=2 3-D turbulence does not support km_opt={km_opt}")
    return _DiffOpt2TurbulenceFields(d11, d22, d33, d12, d13, d23, bn2, xkmh, xkmv, xkhh, xkhv, rdz, rdzw)


def _xface_average_3d(field: jax.Array) -> jax.Array:
    """Average mass-cell field to periodic u x-faces."""

    west_faces = 0.5 * (field + jnp.roll(field, 1, axis=2))
    return jnp.concatenate((west_faces, west_faces[:, :, :1]), axis=2)


def _yface_average_3d(field: jax.Array) -> jax.Array:
    """Average mass-cell field to periodic v y-faces."""

    south_faces = 0.5 * (field + jnp.roll(field, 1, axis=1))
    return jnp.concatenate((south_faces, south_faces[:, :1, :]), axis=1)


def _wface_average_3d(field: jax.Array) -> jax.Array:
    """Average mass-cell field to w z-faces, using nearest levels at lids."""

    nz = int(field.shape[0])
    face = jnp.zeros((nz + 1,) + tuple(field.shape[1:]), dtype=field.dtype)
    if nz > 1:
        face = face.at[1:nz, :, :].set(0.5 * (field[1:nz, :, :] + field[0 : nz - 1, :, :]))
    face = face.at[0, :, :].set(field[0, :, :])
    face = face.at[nz, :, :].set(field[nz - 1, :, :])
    return face


def _tke_coupled_tendency(
    haloed: State,
    namelist: OperationalNamelist,
    turbulence: _DiffOpt2TurbulenceFields,
    *,
    mass_h: jax.Array,
    dz: jax.Array,
) -> jax.Array:
    """Build WRF dry TKE coupled tendency for diff_opt=2/km_opt=2/5."""

    grid = namelist.grid
    dx = float(grid.projection.dx_m)
    dy = float(grid.projection.dy_m)
    qke_t = tke_rhs_tendency(
        haloed.qke,
        mass_h,
        turbulence.d11,
        turbulence.d22,
        turbulence.d33,
        turbulence.d12,
        turbulence.d13,
        turbulence.d23,
        turbulence.bn2,
        turbulence.xkmh,
        turbulence.xkmv,
        turbulence.xkhv,
        dx_m=dx,
        dy_m=dy,
        rdzw=turbulence.rdzw,
        dt_s=float(namelist.dt_s),
        c_k=float(namelist.c_k),
        km_opt=int(namelist.km_opt),
    )
    if (int(namelist.km_opt) == 2 and not bool(namelist.tke_mix2_off)) or int(namelist.km_opt) == 5:
        qke_t = qke_t + horizontal_diffusion_coord_scalar_tendency(
            haloed.qke,
            turbulence.xkmh,
            mass_h,
            dx_m=dx,
            dy_m=dy,
        )
    if int(namelist.km_opt) == 2:
        qke_t = qke_t + vertical_diffusion_coord_scalar_tendency(
            haloed.qke,
            turbulence.xkmv,
            mass_h,
            dz_m=dz,
        )
    lower = -mass_h * jnp.maximum(jnp.asarray(0.0, dtype=haloed.qke.dtype), haloed.qke) / float(namelist.dt_s)
    return jnp.maximum(qke_t, lower)


def _apply_tke_large_step(
    state: State,
    step_origin: State,
    *,
    qke_tendency: jax.Array,
    dt_rk: float,
    metrics: DycoreMetrics,
    tke_upper_bound: float,
) -> State:
    """WRF scalar update + ``bound_tke`` for the prognostic TKE carry."""

    mass_old = metrics.c1h[:, None, None] * step_origin.mu_total[None, :, :] + metrics.c2h[:, None, None]
    mass_new = metrics.c1h[:, None, None] * state.mu_total[None, :, :] + metrics.c2h[:, None, None]
    qke_new = (mass_old * step_origin.qke + float(dt_rk) * qke_tendency) / mass_new
    qke_new = jnp.minimum(jnp.maximum(qke_new, 0.0), float(tke_upper_bound))
    return state.replace(qke=qke_new)


def _diffopt1_dry_forward_tendencies(
    haloed: State,
    namelist: OperationalNamelist,
    *,
    base_state: BaseState | None = None,
    return_scalar_xkhh: bool = False,
) -> tuple[jax.Array, ...]:
    """Build WRF's RK1-frozen dry horizontal-diffusion ``*_tendf`` bundle.

    ``return_scalar_xkhh`` appends the scalar diffusivity ``xkhh`` that the
    theta branch used; WRF passes the same ``grid%xkhh`` to rk_scalar_tend's
    moist/other-scalar ``horizontal_diffusion`` (see
    :func:`_diffopt1_scalar_horizontal_diffusion`).

    Pristine ``module_first_rk_step_part2.F`` computes the Smagorinsky
    coefficients from the time-t fields, then ``module_em.F::rk_tendency``
    places all diff_opt=1 dry diffusion inside ``forward_step: IF (rk_step ==
    1)``.  ``rk_addtend_dry`` reuses ``ru/rv/rw/t_tendf`` in RK1, RK2, and
    RK3.  The released momentum operator remains unchanged.  For
    specified/nested real domains the theta branch additionally follows WRF's
    live ``compute_diff_metrics -> cal_deform_and_div -> smag2d_km ->
    horizontal_diffusion_3dmp`` sequence, including physical-edge ownership.
    This deliberately isolates the scalar repair from the already-positive
    momentum discriminator.
    """

    metrics = namelist.metrics
    dx = float(namelist.grid.projection.dx_m)
    dy = float(namelist.grid.projection.dy_m)
    mu_total = haloed.mu_total
    muu = _u_face_average_2d(mu_total)
    muv = _v_face_average_2d(mu_total)
    mass_u = metrics.c1h[:, None, None] * muu[None, :, :] + metrics.c2h[:, None, None]
    mass_v = metrics.c1h[:, None, None] * muv[None, :, :] + metrics.c2h[:, None, None]
    mass_h = (
        metrics.c1h[:, None, None] * mu_total[None, :, :]
        + metrics.c2h[:, None, None]
    )
    mass_f = metrics.c1f[:, None, None] * mu_total[None, :, :] + metrics.c2f[:, None, None]
    # Periodic idealized programs retain the released flat coefficient/operator.
    # Specified/nested domains replace this below with the literal real-map WRF
    # coefficient and U/V/W target-stagger operator.
    d11, d22, d12 = horizontal_deformation_2d(
        haloed.u,
        haloed.v,
        dx_m=dx,
        dy_m=dy,
    )
    xkmh, xkhh = smag2d_horizontal_km(
        d11,
        d22,
        d12,
        dx_m=dx,
        dy_m=dy,
        c_s=float(namelist.c_s),
    )
    _, specified, nested = _acoustic_lateral_bc_flags(namelist)
    source_scalar_path = bool(specified or nested)
    if source_scalar_path:
        zx, zy, rdzw = wrf_nonperiodic_diffusion_metrics(
            haloed.ph_total,
            dx_m=dx,
            dy_m=dy,
        )
        scalar_d11, scalar_d22, scalar_d12 = horizontal_deformation_2d(
            haloed.u,
            haloed.v,
            dx_m=dx,
            dy_m=dy,
            msftx=metrics.msftx,
            msfty=metrics.msfty,
            msfux=metrics.msfux,
            msfuy=metrics.msfuy,
            msfvx=metrics.msfvx,
            msfvy=metrics.msfvy,
            zx=zx,
            zy=zy,
            rdzw=rdzw,
            fnm=metrics.fnm,
            fnp=metrics.fnp,
            cf1=metrics.cf1,
            cf2=metrics.cf2,
            cf3=metrics.cf3,
            dn=metrics.dn,
            dnw=metrics.dnw,
        )
        xkmh, xkhh = smag2d_horizontal_km(
            scalar_d11,
            scalar_d22,
            scalar_d12,
            dx_m=dx,
            dy_m=dy,
            c_s=float(namelist.c_s),
            msftx=metrics.msftx,
            msfty=metrics.msfty,
        )
        # smag2d_km owns ids+1:ide-2 / jds+1:jde-2 for both specified
        # and nested domains.  xkhh is an initialized work array in WRF, so
        # the unowned physical ring remains exactly zero before phy_bc copies
        # values only into the external memory halo.
        for coefficient_name, coefficient in (("xkmh", xkmh), ("xkhh", xkhh)):
            coefficient = coefficient.at[:, 0, :].set(0.0)
            coefficient = coefficient.at[:, -1, :].set(0.0)
            coefficient = coefficient.at[:, :, 0].set(0.0)
            coefficient = coefficient.at[:, :, -1].set(0.0)
            if coefficient_name == "xkmh":
                xkmh = coefficient
            else:
                xkhh = coefficient
    # WRF passes ``t_init`` to horizontal_diffusion_3dmp and diffuses
    # ``t - t_init``.  State.theta is the full (moist when use_theta_m=1)
    # potential temperature while BaseState.theta_base is ``t0+t_init``;
    # subtracting the latter is the identical full-theta representation.  A
    # 106-leaf operational carry has no explicit BaseState, so the alternate
    # branch reconstructs that same profile from its resident base components.
    # Native RK's retained BaseState captures PB/PHB/MUB; its 300 K theta
    # offset is not WRF's terrain-dependent t0+t_init diffusion reference.
    # Preserve the pre-native-RK reconstruction for this consumer.
    _native_rk_base_only = (
        os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1"
        and not is_mixed_perturb_fp32_mode(namelist.acoustic_precision_mode)
    )
    if base_state is not None and not _native_rk_base_only:
        theta_base = jnp.asarray(base_state.theta_base, dtype=haloed.theta.dtype)
    else:
        # The released fp64 carry deliberately has no explicit BaseState leaf
        # (d03 remains the authenticated 106-leaf interface).  Recover WRF's
        # exact t0+t_init profile from the invariant base components already in
        # State.  This is the device transcription of
        # d02_replay._wrf_base_theta_from_loaded_state and inverts WRF's
        # discrete base hydrostatic relation.  WRF uses the linear pressure
        # depth for hypsometric_opt=1 and the LOG-pressure depth for option 2;
        # both close the same EOS
        #   alb = (R_d/p0)*theta_base*(pb/p0)**cvpm.
        pb = haloed.p_total - haloed.p_perturbation
        phb = haloed.ph_total - haloed.ph_perturbation
        mub = haloed.mu_total - haloed.mu_perturbation
        dphb = phb[1:, :, :] - phb[:-1, :, :]
        if int(namelist.hypsometric_opt) == 2:
            p_top = jnp.reshape(metrics.p_top, ()).astype(pb.dtype)
            mub_column = mub[None, :, :]
            pfu = (
                metrics.c3f[1:, None, None] * mub_column
                + metrics.c4f[1:, None, None]
                + p_top
            )
            pfd = (
                metrics.c3f[:-1, None, None] * mub_column
                + metrics.c4f[:-1, None, None]
                + p_top
            )
            phm = (
                metrics.c3h[:, None, None] * mub_column
                + metrics.c4h[:, None, None]
                + p_top
            )
            alb = dphb / (phm * jnp.log(pfd / pfu))
        elif int(namelist.hypsometric_opt) == 1:
            base_mass_h = (
                metrics.c1h[:, None, None] * mub[None, :, :]
                + metrics.c2h[:, None, None]
            )
            denominator = metrics.dnw[:, None, None] * base_mass_h
            alb = -dphb / denominator
        else:
            raise ValueError(
                "diff_opt=1 theta base reconstruction supports "
                f"hypsometric_opt 1 or 2, got {namelist.hypsometric_opt}"
            )
        pressure_ratio = (pb / P0_PA) ** CVPM
        theta_base = alb * (P0_PA / R_D) / pressure_ratio
    theta_diffusion = horizontal_diffusion_coord_scalar_tendency(
        haloed.theta,
        xkhh,
        mass_h,
        dx_m=dx,
        dy_m=dy,
        base_3d=theta_base,
        msftx=metrics.msftx if source_scalar_path else None,
        msfty=metrics.msfty if source_scalar_path else None,
        msfux=metrics.msfux if source_scalar_path else None,
        msfuy=metrics.msfuy if source_scalar_path else None,
        msfvx=metrics.msfvx if source_scalar_path else None,
        msfvy=metrics.msfvy if source_scalar_path else None,
        nonperiodic_owned=source_scalar_path,
    )
    if source_scalar_path:
        du, dv, dw = wrf_nested_horizontal_diffusion_momentum_tendency(
            haloed.u,
            haloed.v,
            haloed.w,
            xkmh,
            mu_total,
            c1h=metrics.c1h,
            c2h=metrics.c2h,
            c1f=metrics.c1f,
            c2f=metrics.c2f,
            msfux=metrics.msfux,
            msfuy=metrics.msfuy,
            msfvx=metrics.msfvx,
            msfvy=metrics.msfvy,
            msftx=metrics.msftx,
            msfty=metrics.msfty,
            dx_m=dx,
            dy_m=dy,
        )
    else:
        du, dv, dw = horizontal_diffusion_coord_momentum_tendency(
            haloed.u,
            haloed.v,
            haloed.w,
            xkmh,
            mass_u,
            mass_v,
            mass_f,
            dx_m=dx,
            dy_m=dy,
        )
    if return_scalar_xkhh:
        return du, dv, dw, theta_diffusion, xkhh
    return du, dv, dw, theta_diffusion


def _diffopt1_scalar_horizontal_diffusion(
    reference: State,
    xkhh: jax.Array,
    muts: jax.Array,
    namelist: OperationalNamelist,
) -> dict[str, jax.Array]:
    """WRF diff_opt=1 2nd-order horizontal mixing of moist and other scalars.

    Pristine ``rk_scalar_tend`` (module_em.F:1380-1390) calls
    ``horizontal_diffusion('m', scalar, scalar_tends, mut, ..., xkmhd)`` at RK1
    for every moist species and every other scalar (QNI/QNR), with
    ``moist_mix2_off``/``scalar_mix2_off`` defaulting to ``.false.``.  The
    caller (solve_em.F:2299-2314, :2869-2884) passes ``grid%muts`` (post-RK1
    acoustic total mass) as ``mut`` and the step's ``grid%xkhh`` as ``xkmhd``.
    The result lands in ``moist_tend``/``scalar_tend`` and is reused by all
    three ``rk_update_scalar`` calls, unscaled by ``msfty``.  The stencil is the
    scalar branch of ``horizontal_diffusion`` (full field, no base profile) with
    the same specified/nested ownership as theta's ``horizontal_diffusion_3dmp``.
    """

    metrics = namelist.metrics
    _, specified, nested = _acoustic_lateral_bc_flags(namelist)
    source_scalar_path = bool(specified or nested)
    mass = metrics.c1h[:, None, None] * muts[None, :, :] + metrics.c2h[:, None, None]
    tendencies: dict[str, jax.Array] = {}
    for name in NESTED_BOUNDARY_SCALAR_SPECIES:
        field = getattr(reference, name, None)
        if field is None:
            continue
        tendencies[name] = horizontal_diffusion_coord_scalar_tendency(
            field,
            xkhh,
            mass,
            dx_m=float(namelist.grid.projection.dx_m),
            dy_m=float(namelist.grid.projection.dy_m),
            msftx=metrics.msftx if source_scalar_path else None,
            msfty=metrics.msfty if source_scalar_path else None,
            msfux=metrics.msfux if source_scalar_path else None,
            msfuy=metrics.msfuy if source_scalar_path else None,
            msfvx=metrics.msfvx if source_scalar_path else None,
            msfvy=metrics.msfvy if source_scalar_path else None,
            nonperiodic_owned=source_scalar_path,
        )
    return tendencies


def _real_theta_sum_fence(theta_tend: jax.Array) -> jax.Array:
    """Output-only fusion fence on a REAL theta-tendency partial sum (E41/BD93).

    Without it XLA packs the whole theta stage sum (coupled advection, sixth-order and horizontal
    diffusion, rk_addtend_dry t_tendf/h_diabatic, root relax_bdy_dry) into ONE Triton transpose
    fusion with 11 operands in three layouts; on the single-domain d01 program that kernel spills
    (STACK 632 B).  The fence sits only where the producer ends in an add and the consumer is an
    add, so no multiply-add pair is split (no FMA change) and the values are unchanged.  The f64
    legacy graph stays byte-identical.
    """
    if theta_tend.dtype != jnp.float32:
        return theta_tend
    return jax.lax.optimization_barrier(theta_tend)


def _augment_large_step_tendencies(
    haloed: State,
    tendencies: Tendencies,
    namelist: OperationalNamelist,
    *,
    rk_step: int = 3,
    physics_tendencies: DryPhysicsTendencies | None = None,
    step_origin: State | None = None,
    transport_velocities: CoupledVelocities | None = None,
    bdy_relax: SpecifiedRelaxTendencies | None = None,
    base_state: BaseState | None = None,
    frozen_diffopt1_tendencies: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    | None = None,
    frozen_diff6_theta_tendency: jax.Array | None = None,
    frozen_diff6_uvw_tendencies: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    frozen_les3d_km3: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None,
) -> Tendencies:
    """Add WRF explicit diffusion + flux-form scalar advection to the large step.

    All contributions are returned as *uncoupled* tendencies to match the
    operational RK convention (``add_scaled_tendencies`` adds them uncoupled,
    then ``small_step_prep`` couples).  Sources:
    * 6th-order monotonic filter -- ``module_big_step_utilities_em.F:6504-6920``.
    * constant-K diffusion (Straka ν) -- ``:2999-3234``.
    * flux-form theta advection -- ``module_advect_em.F:3029-4359`` (h=5/v=3).

    ``step_origin`` remains in this helper's frozen call interface for the other
    RK consumers.  Theta itself does not use it: pristine WRF ``rk_tendency``
    always calls ordinary ``advect_scalar`` for potential temperature when the
    currently wired ``scalar_adv_opt`` is 0, 1, or 2.  The final-RK3
    positive-definite/monotonic selection belongs only to the separate moist and
    other-scalar loops (``rk_scalar_tend``), whose helpers consume
    ``step_origin`` independently below.
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
        # primitive coupled products built above (not add to them).  The
        # transporting ru/rv/rom is the per-stage shared build (WRF builds it
        # once per RK stage); the caller may pass it in so the moisture path
        # reuses the SAME arrays instead of duplicating the construction.
        vel = (
            transport_velocities
            if transport_velocities is not None
            else _stage_transport_velocities(haloed, namelist)
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
            haloed.u, vel, rdx=1.0 / dx, rdy=1.0 / dy, nested=_acoustic_lateral_bc_flags(namelist)[2],
            rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
        ); u_t, haloed, vel, metrics = _small_grid_firewall((u_t, haloed, vel, metrics), haloed.theta.shape[-2:])
        v_t = namelist.tendencies.v * mass_v + advect_v_flux(
            haloed.v, vel, rdx=1.0 / dx, rdy=1.0 / dy, nested=_acoustic_lateral_bc_flags(namelist)[2],
            rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
        ); v_t, haloed, vel, metrics = _small_grid_firewall((v_t, haloed, vel, metrics), haloed.theta.shape[-2:])
        w_t = namelist.tendencies.w * mass_f + advect_w_flux(
            haloed.w, vel, rdx=1.0 / dx, rdy=1.0 / dy,
            rdn=metrics.rdn, fzm=metrics.fnm, fzp=metrics.fnp,
            top_lid=bool(namelist.top_lid),
        )
        # --- scalar theta: WRF advect_scalar (h=5/v=3) ---
        theta_offset = _theta_base_offset(haloed.theta)
        # Pristine module_em.F::rk_tendency does NOT send theta through
        # advect_scalar_pd/mono for scalar_adv_opt 1/2.  Those limiters are for
        # the separate moist/scalar families in rk_scalar_tend.  Applying the PD
        # tracer limiter here was additionally ill-posed because WRF T is a
        # signed perturbation (the operational d03 initial field contains
        # negative values).  For all currently wired options 0/1/2, theta uses
        # the same ordinary h5/v3 advect_scalar call at every RK stage.
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
        if frozen_diff6_uvw_tendencies is not None:
            # Pristine WRF momentum sixth-order diffusion lives in the RK1-only
            # ``ru/rv/rw_tendf`` bundle (module_em.F:882-916) and reaches
            # ``ru_tend`` through rk_addtend_dry's per-field map division at
            # every stage.  The live nested child supplies that already-divided
            # step-persistent bundle here (S1 attribution sprint 2026-07-18:
            # the per-stage periodic form below wrapped opposite-boundary
            # values into rings 0-2 and carried point-mass/no-msf algebra —
            # 97.8% of the frozen S1 nonspec SSE).  The periodic path remains
            # byte-identical for idealized/released programs.
            du6, dv6, dw6 = frozen_diff6_uvw_tendencies
            u_t = u_t + du6
            v_t = v_t + dv6
            w_t = w_t + dw6
        else:
            u_t = u_t + mass_u * sixth_order_diffusion_tendency(haloed.u, dt=dt_diff, diff_6th_factor=f)
            v_t = v_t + mass_v * sixth_order_diffusion_tendency(haloed.v, dt=dt_diff, diff_6th_factor=f)
            w_t = w_t + mass_f * sixth_order_diffusion_tendency(haloed.w, dt=dt_diff, diff_6th_factor=f)
        if frozen_diff6_theta_tendency is None:
            th_t = th_t + mass_h * sixth_order_diffusion_tendency(
                haloed.theta,
                dt=dt_diff,
                diff_6th_factor=f,
            )
        else:
            # Pristine WRF evaluates scalar sixth-order diffusion only inside
            # ``forward_step`` (RK1), stores the mass/map-coupled result in
            # ``t_tendf``, and reuses it in RK2/RK3 through
            # ``rk_addtend_dry(t_tendf/msfty)``.  The live-child candidate
            # supplies that already-divided, immutable effective tendency here.
            # U/V/W intentionally retain their prior bytes so this remains the
            # bounded T/scalar discriminator independent of 2c13b731's wind fix.
            th_t = th_t + frozen_diff6_theta_tendency

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
        # WRF computes this complete dry bundle once at RK1 and retains it in
        # ru/rv/rw/t_tendf.  Direct helper callers without the operational
        # hoist retain the historical current-state construction.
        if frozen_diffopt1_tendencies is None:
            du_s, dv_s, dw_s, theta_diffusion = _diffopt1_dry_forward_tendencies(
                haloed,
                namelist,
                base_state=base_state,
            )
        else:
            du_s, dv_s, dw_s, theta_diffusion = frozen_diffopt1_tendencies
        th_t = th_t + theta_diffusion
        u_t = u_t + du_s
        v_t = v_t + dv_s
        w_t = w_t + dw_s

    # WRF diff_opt=2 / km_opt=2,3,5: 3-D LES/TKE turbulence closure.  The
    # coefficient formulas are WRF's ``smag_km`` / ``tke_km`` periodic-interior
    # reductions.  Momentum diffusion currently uses conservative variable-K
    # scalar flux divergence on each staggered field; the deformation-stress
    # momentum tensor is the remaining parity item called out in the proof report.
    if int(namelist.diff_opt) == 2 and int(namelist.km_opt) == 3:
        # o1-smag3d: literal WRF km_opt=3 (gpuwrf.dynamics.les3d_smagorinsky, pristine-oracle
        # parity): RK1-frozen ru/rv/rw/t_tendf already folded as rk_addtend_dry consumes them.
        if frozen_les3d_km3 is None:
            from gpuwrf.runtime.les3d_km3 import les3d_km3_forward_tendencies

            frozen_les3d_km3 = les3d_km3_forward_tendencies(haloed, namelist, base_state=base_state)[:4]
        du3, dv3, dw3, dth3 = frozen_les3d_km3
        u_t = u_t + du3.astype(u_t.dtype)
        v_t = v_t + dv3.astype(v_t.dtype)
        w_t = w_t + dw3.astype(w_t.dtype)
        th_t = th_t + dth3.astype(th_t.dtype)
    elif int(namelist.diff_opt) == 2 and int(namelist.km_opt) in (2, 3, 5):
        turb = _diffopt2_turbulence_fields(haloed, namelist, dz=dz)
        theta_base = _theta_base_offset(haloed.theta) * jnp.ones_like(haloed.theta)
        th_t = th_t + horizontal_diffusion_coord_scalar_tendency(
            haloed.theta,
            turb.xkhh,
            mass_h,
            dx_m=dx,
            dy_m=dy,
            base_3d=theta_base,
        )
        th_t = th_t + vertical_diffusion_coord_scalar_tendency(
            haloed.theta,
            turb.xkhv,
            mass_h,
            dz_m=dz,
            base_3d=theta_base,
        )
        du_h, dv_h, dw_h = horizontal_diffusion_coord_momentum_tendency(
            haloed.u,
            haloed.v,
            haloed.w,
            turb.xkmh,
            mass_u,
            mass_v,
            mass_f,
            dx_m=dx,
            dy_m=dy,
        )
        u_t = u_t + du_h + vertical_diffusion_coord_scalar_tendency(
            haloed.u,
            _xface_average_3d(turb.xkmv),
            mass_u,
            dz_m=dz,
        )
        v_t = v_t + dv_h + vertical_diffusion_coord_scalar_tendency(
            haloed.v,
            _yface_average_3d(turb.xkmv),
            mass_v,
            dz_m=dz,
        )
        w_t = w_t + dw_h + vertical_diffusion_coord_scalar_tendency(
            haloed.w,
            _wface_average_3d(turb.xkmv),
            mass_f,
            dz_m=dz,
        )

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
    _fused_uv = bool(namelist.run_boundary) and large_step_uv_fused_enabled(u_t, v_t, base_state) and (
        _acoustic_lateral_bc_flags(namelist)[2] or not large_step_uv_nested_only())
    if _fused_uv:
        # GPUWRF_DYN_GLUE_FUSED: PGF + Coriolis + curvature (below) in one REAL
        # stencil per stagger, same operands and add order.
        u_t, v_t = large_step_uv_fused(
            haloed,
            metrics,
            u_t,
            v_t,
            dx_m=dx,
            dy_m=dy,
            top_lid=bool(namelist.top_lid),
            hypsometric_opt=int(namelist.hypsometric_opt),
            base_state=base_state,
        )
    else:
        ru_pgf, rv_pgf = large_step_horizontal_pgf(
            haloed,
            metrics,
            dx_m=dx,
            dy_m=dy,
            non_hydrostatic=True,
            top_lid=bool(namelist.top_lid),
            hypsometric_opt=int(namelist.hypsometric_opt),
            base_state=base_state,
        )
        u_t = u_t + ru_pgf
        v_t = v_t + rv_pgf; u_t, v_t, haloed, metrics = _small_grid_firewall((u_t, v_t, haloed, metrics), haloed.theta.shape[-2:])

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
    if not _fused_uv:
        ru_cor, rv_cor = large_step_coriolis(
            haloed,
            metrics,
            specified=bool(namelist.run_boundary),
        )
        u_t = u_t + ru_cor
        v_t = v_t + rv_cor; u_t, v_t, haloed, metrics = _small_grid_firewall((u_t, v_t, haloed, metrics), haloed.theta.shape[-2:])

    # WRF calls normal-map curvature immediately after Coriolis
    # (module_em.F:773-781; module_big_step_utilities_em.F:4239-4446). The
    # source correction is intentionally bound to the authenticated
    # specified/nested path; periodic idealized programs retain their prior
    # bytes under the contract amendment.
    if bool(namelist.run_boundary) and not _fused_uv:
        ru_curv, rv_curv = large_step_horizontal_curvature(
            haloed,
            metrics,
            dx_m=dx,
            dy_m=dy,
            specified=True,
        )
        u_t = u_t + ru_curv
        v_t = v_t + rv_curv; u_t, v_t, haloed, metrics = _small_grid_firewall((u_t, v_t, haloed, metrics), haloed.theta.shape[-2:])

    tendencies = tendencies.replace(u=u_t, v=v_t, w=w_t, theta=_real_theta_sum_fence(th_t))

    # WRF rk_addtend_dry per-stage merge (module_em.F:1711-1786): field-specific
    # map/mass coupling of RK1-fixed non-timesplit physics tendencies.  Physics-off
    # and dry idealized gates pass an empty bundle, so this remains identity there.
    merged = rk_addtend_dry(
        tendencies,
        DryPhysicsTendencies() if physics_tendencies is None else physics_tendencies,
        rk_step=int(rk_step),
        metrics=metrics,
        # WRF couples h_diabatic with the FULL dry column mass grid%mut
        # (module_em.F:1079 ``(c1(k)*mut(i,j)+c2(k))``), not the base state.
        # ``mut`` is consumed ONLY by the h_diabatic term, which was identically
        # zero before the v0.14 venting-residual fix routed the Thompson heating.
        mut=haloed.mu_total,
    )
    # v0.14 SPECIFIED WRF boundary cadence: relax_bdy_dry tendencies.  WRF folds
    # the coupled u/v/t relax into the step-constant *_tendf lane at rk_step==1
    # (solve_em.F:938-965 -> rk_addtend_dry :1735/:1746/:1770), so the NET
    # per-stage contribution is the coupled relax itself at EVERY stage (the
    # msf couple/uncouple pair cancels); the mu relax went straight into the
    # stage-1 mu_tend and rk_tendency re-zeroes mu_tend at stages 2-3, so it is
    # genuinely stage-1-only (quirk preserved).  The ph half of the bundle is
    # consumed by ph_tend_stage in _acoustic_core_state_from_prep (the
    # operational ph tendency lane is rhs_ph, not tendencies.ph).
    if bdy_relax is not None:
        mu_aug = merged.mu + bdy_relax.mu if int(rk_step) == 1 else merged.mu
        if bdy_relax.w is None:
            # Released specified-root program: preserve the exact replacement
            # surface when the nested-only w lane is absent.
            merged = merged.replace(
                u=merged.u + bdy_relax.ru,
                v=merged.v + bdy_relax.rv,
                theta=_real_theta_sum_fence(merged.theta) + bdy_relax.t,
                mu=mu_aug,
            )
        else:
            merged = merged.replace(
                u=merged.u + bdy_relax.ru,
                v=merged.v + bdy_relax.rv,
                w=merged.w + bdy_relax.w,
                theta=_real_theta_sum_fence(merged.theta) + bdy_relax.t,
                mu=mu_aug,
            )
    if physics_tendencies is not None and physics_tendencies.moist_tendf:
        # E41/E204: fence assembled outputs, preserving WRF arithmetic. Monica's
        # fused cascade otherwise joins U/V PGF and advection into spilling
        # transpose kernels, alongside the BD99 theta fusion.
        u, v, theta = jax.lax.optimization_barrier((merged.u, merged.v, merged.theta))
        merged = merged.replace(u=u, v=v, theta=theta)
    return merged


# WRF advects every moisture species (vapour + condensates) in the RK3 large
# step, in this Registry order (``moist_variable_loop`` over the moist array,
# solve_em.F:2282-2408).  The condensates that exist as State leaves in this port
# are qc/qr/qi/qs/qg; qv is index P_QV.
_MOISTURE_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")

# A mapped species lane is beneficial only in the small-grid regime. Key the
# policy on CELLS, not bytes: release d02 has 797,940 cells and must remain scalar
# in fp64 AND fp32, while d03/d08 have 309,672/378,972 cells. A byte threshold
# would silently move fp32 d02 back onto the failed batched path. Keep the choice
# trace-static and conservative; width selection is finalized by ADR-033.
_NESTED_SCALAR_BATCH_MAX_FIELD_CELLS = 524_288
_NESTED_SCALAR_BATCH_WIDTH = 2


def _nested_scalar_species_batch_width(
    field: jax.Array, *, use_limiter: bool = True
) -> int:
    """Static limited-stage batch width; plain stages always use reference width 1."""

    if not bool(use_limiter):
        return 1

    cells = 1
    for extent in field.shape:
        cells *= int(extent)
    return (
        _NESTED_SCALAR_BATCH_WIDTH
        if cells <= _NESTED_SCALAR_BATCH_MAX_FIELD_CELLS
        else 1
    )


# v0.17 ADR-032 hail microphysics family (mp_physics ids). When one of these is
# selected WRF carries the additional hail/predicted-density scalars
# (qh/qvolg/qvolh) in the moist/scalar arrays and transports them with the SAME
# rk_scalar_tend flux-form machinery. The selection in _advected_scalar_species
# is STATIC (mp_physics is a jit static aux), so NON-hail programs are
# byte-for-byte unchanged. NONE of these ids is in _SCAN_WIRED_OPTIONS yet (the
# scheme kernels are not ported -- ADR-032 is the State substrate only), so the
# hail branch is UNREACHABLE in every currently-runnable program; it is wired
# and unit-tested here so the future hail-scheme worker inherits the transport.
_HAIL_MP_FAMILY: frozenset[int] = frozenset({7, 17, 18, 19, 21, 22, 24, 26, 27, 38})

# Per hail-scheme additional advected species (WRF moist/scalar members beyond
# the six core moist species). Only the leaves a given scheme actually carries
# are transported. Schemes not listed fall back to the core moist set.
_HAIL_MP_ADVECTED_EXTRAS: Mapping[int, tuple[str, ...]] = {
    7: ("qh",),            # nuwrf4icescheme   moist:...,qh
    24: ("qh",),           # wsm7scheme        moist:...,qh
    26: ("qh",),           # wdm7scheme        moist:...,qh
    27: ("qh",),           # udmscheme         moist:...,qh
    38: ("qvolg",),        # thompsongh        scalar:...,qng,qvolg (Ng untransported like mp=8/10)
    # NSSL family (mp 17-22): base nssl_2mom carries qv..qg; the hail/volume
    # leaves are added by the nssl sub-packages. The conservative substrate
    # default transports the hail mass + predicted-density volumes when present.
    17: ("qh", "qvolg", "qvolh"),
    18: ("qh", "qvolg", "qvolh"),
    19: ("qh", "qvolg", "qvolh"),
    21: ("qh", "qvolg", "qvolh"),
    22: ("qh", "qvolg", "qvolh"),
}


def _advected_scalar_species(namelist: "OperationalNamelist") -> tuple[str, ...]:
    """Scalar species transported by the large-step flux advection.

    The six core moist species always advect (WRF ``moist_variable_loop``).
    Two scheme families add transported scalars on top, via the SAME
    ``rk_scalar_tend`` flux-form machinery (``solve_em.F`` scalar loops):

      * Hail microphysics family (ADR-032; WSM7=24, WDM7=26, and the wider
        hail family) carries the scheme's extra hail / predicted-density
        scalars (QHAIL/QVGRAUPEL/QVHAIL) listed in ``_HAIL_MP_ADVECTED_EXTRAS``.
      * Aerosol-aware Thompson (mp=28) carries the two prognostic aerosol
        numbers ``nwfa``/``nifa`` (WRF QNWFA/QNIFA in the ``scalar`` array).

    The selection is STATIC (``mp_physics`` is a jit static aux), so every
    program that selects none of these schemes is byte-for-byte unchanged. The
    Ni/Nr use the separate other-scalar transport loop and ``scalar_adv_opt``.
    The extra leaves also cold-start at zero,
    so even the FIRST enabling of one of these schemes cannot perturb the
    dynamics until that scheme produces the species.
    """

    mp = int(namelist.mp_physics)
    if mp in _HAIL_MP_FAMILY:
        return _MOISTURE_SPECIES + _HAIL_MP_ADVECTED_EXTRAS.get(mp, ())
    if mp == 28:
        return _MOISTURE_SPECIES + ("nwfa", "nifa")
    if mp == 40:
        # v0.3.4 O1: Registry morr_tm_aero scalar qnc,qni,qns,qnr,qng -- Ni/Nr
        # ride the number path below; qns/qng/qnc are transported here with the
        # same flux-form machinery as mp=28's nwfa/nifa (PD family by
        # _SCALAR_FAMILY_SPECIES; stencil = moist_adv_opt, WRF's scalar_adv_opt,
        # identical when both are equal as in PROD/WN3).  Nc is overwritten by
        # the aercu_opt=0 scheme every call, so its transport cannot feed back.
        return _MOISTURE_SPECIES + ("Ns", "Ng", "Nc")
    return _MOISTURE_SPECIES


# NSSL 2-moment (mp=18) number scalars beyond the Ni/Nr pair (WRF ``scalar`` array members
# qndrop/qns/qng/qnh/qnn, Registry nssl2mconc/nssl_hail/nssl_ccn_opt) transported by the SAME root
# rk_scalar_tend flux-form call as Ni/Nr.  Requires moist_adv_opt == scalar_adv_opt (one stacked
# call); qvolg/qvolh ride the hail moist extras above.
_NSSL_NUMBER_SCALARS: tuple[str, ...] = ("Nc", "Ns", "Ng", "Nh", "Nn")


def _nssl_number_scalars(namelist: "OperationalNamelist") -> tuple[str, ...]:
    if int(namelist.moist_adv_opt) != int(namelist.scalar_adv_opt) or int(namelist.moist_adv_opt) == 0:
        raise NotImplementedError(
            "mp_physics=18 (NSSL) transport needs moist_adv_opt == scalar_adv_opt != 0 (one stacked "
            f"scalar transport call); got moist_adv_opt={int(namelist.moist_adv_opt)}, "
            f"scalar_adv_opt={int(namelist.scalar_adv_opt)}")
    return _NSSL_NUMBER_SCALARS


def _nssl_root_flow_only(namelist: "OperationalNamelist") -> dict:
    """mp=18: NSSL extras with no lateral boundary records take WRF's flow_dep_bdy on the root."""
    if getattr(namelist, "mp_physics", None) != 18:
        return {}
    return {"flow_only": ("qh", "qvolg", "qvolh") + _NSSL_NUMBER_SCALARS}


def _scalar_transport_coupled_tendencies(
    haloed: State,
    namelist: OperationalNamelist,
    *,
    rk_step: int,
    step_origin: State | None,
    species: tuple[str, ...],
    advection_opt: int,
    transport_velocities: CoupledVelocities | None = None,
    species_batch_width: int = 1,
    f2_pd_out: dict | None = None,
) -> tuple[jax.Array, ...]:
    """WRF scalar-loop coupled large-step tendency ``d(mu*q)/dt`` per stage.

    Both pristine ``solve_em.F`` loops call ``rk_scalar_tend`` with the same
    transport equation: the moist loop selects ``moist_adv_opt`` and the other
    scalar loop (including Thompson QNI/QNR) selects ``scalar_adv_opt``.  Keeping
    the static species and option explicit also lets equal-option source loops
    share bounded species chunks without changing their WRF equations or
    ordering. ``species_batch_width`` is a static dispatch policy; its default
    1 retains the canonical scalar loop.

    Reuses the EXACT same ``vel`` / ``mu_total`` / ``metrics`` build as the theta
    flux advection so the transporting velocity field is bit-consistent with the
    momentum/theta advection of the same stage.  Returns a tuple of COUPLED
    tendencies ``d(mu*q)/dt`` in ``species`` order, consumed by the WRF
    scalar large-step update ``q_new = (mu_old*q_old + dt_rk*tend)/mu_new`` AFTER
    the acoustic loop (NOT inside the acoustic substeps).
    """

    metrics = namelist.metrics
    grid = namelist.grid
    dx = float(grid.projection.dx_m)
    dy = float(grid.projection.dy_m)
    mu_total = haloed.mu_total
    vel = (
        transport_velocities
        if transport_velocities is not None
        else _stage_transport_velocities(haloed, namelist)
    )
    fields = tuple(getattr(haloed, name) for name in species)
    if f2_pd_out is not None:
        f2_pd_out["species"] = species
    # The limiter (advection option 1/2) is the final-RK3-stage FCT; it needs the
    # start-of-step moisture (WRF ``moist_old``) and ``mu_old`` (grid%mu_1).  The
    # selection inside advect_moisture_scalars is STATIC, so on opt==0 / non-final
    # stages the plain h5/v3 path is emitted and ``fields_old`` is ignored.
    use_limiter = (
        int(advection_opt) in (1, 2)
        and int(rk_step) == int(namelist.rk_order)
        and step_origin is not None
    )
    fields_old = (
        tuple(getattr(step_origin, name) for name in species)
        if use_limiter
        else None
    )
    mu_old = step_origin.mu_total if step_origin is not None else mu_total
    return advect_moisture_scalars(
        fields,
        fields_old,
        vel,
        moist_adv_opt=int(advection_opt),
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
        species_batch_width=int(species_batch_width),
        msfty=metrics.msfty,
        **({"f2_pd_out": f2_pd_out} if f2_pd_out is not None else {}),
    )


def _moisture_coupled_tendencies(
    haloed: State,
    namelist: OperationalNamelist,
    *,
    rk_step: int,
    step_origin: State | None,
    transport_velocities: CoupledVelocities | None = None,
    f2_pd_out: dict | None = None,
) -> tuple[jax.Array, ...]:
    """Released WRF moist-loop coupled tendencies in its existing order."""

    return _scalar_transport_coupled_tendencies(
        haloed,
        namelist,
        rk_step=int(rk_step),
        step_origin=step_origin,
        species=_advected_scalar_species(namelist),
        advection_opt=int(namelist.moist_adv_opt),
        transport_velocities=transport_velocities,
        **({"f2_pd_out": f2_pd_out} if f2_pd_out is not None else {}),
    )


def _nested_number_scalar_coupled_tendencies(
    haloed: State,
    namelist: OperationalNamelist,
    *,
    rk_step: int,
    step_origin: State,
    transport_velocities: CoupledVelocities,
) -> tuple[jax.Array, ...]:
    """Candidate-only pristine ``other_scalar_advance`` for QNI and QNR.

    Thompson's Registry package declares both fields in the ``scalar`` family.
    ``solve_em.F:2774-2869`` therefore transports them with ``scalar_adv_opt``
    before adding the same RK1-frozen boundary tendency and calling
    ``rk_update_scalar``.  The production State already carries both fields;
    this adds no interface leaf or loop transfer.
    """

    return _scalar_transport_coupled_tendencies(
        haloed,
        namelist,
        rk_step=int(rk_step),
        step_origin=step_origin,
        species=("Ni", "Nr"),
        advection_opt=int(namelist.scalar_adv_opt),
        transport_velocities=transport_velocities,
    )


def _nested_scalar_sixth_order_tendencies(
    step_origin: State,
    namelist: OperationalNamelist,
) -> tuple[jax.Array, ...]:
    """Pristine WRF's RK1-frozen moist/other-scalar ``sc_tend`` bundle.

    ``solve_em.F`` calls ``rk_scalar_tend`` for every active moist and other
    scalar.  In ``module_em.F`` the sixth-order contribution is formed only at
    RK1, with that stage's ``dt_step`` (``dt/3`` for the operational RK3), and
    remains in ``moist_tend``/``scalar_tend`` for all three calls to
    ``rk_update_scalar``.  The returned values are already mass/map-coupled
    ``sc_tend`` quantities: unlike advection, ``rk_update_scalar`` does not
    multiply them by ``msfty``.

    The canonical Thompson nest represents the six moist members plus QNI/QNR
    in exactly ``NESTED_BOUNDARY_SCALAR_SPECIES`` order.  Computing the static
    tuple outside ``advance_stage`` adds no carry leaf or loop transfer.
    """

    metrics = namelist.metrics
    rk1_dt = float(namelist.dt_s) / float(namelist.rk_order)
    return tuple(
        wrf_sixth_order_scalar_tendf(
            getattr(step_origin, name),
            step_origin.mu_total,
            c1=metrics.c1h,
            c2=metrics.c2h,
            msftx=metrics.msftx,
            msfty=metrics.msfty,
            dt=rk1_dt,
            diff_6th_factor=float(namelist.diff_6th_factor),
            monotonic=(int(namelist.diff_6th_opt) == 2),
            specified_or_nested=True,
        )
        for name in NESTED_BOUNDARY_SCALAR_SPECIES
    )


def _tiedtke_qvften_from_flux_advection(
    state: State,
    namelist: OperationalNamelist,
) -> jax.Array:
    """Diagnose WRF ``RQVFTEN`` for modified Tiedtke from flux-form qv advection.

    WRF stores the scalar advection tendency in ``RQVFTEN`` via ``set_tend`` and
    mass-normalizes it before ``cumulus_driver`` calls ``CU_TIEDTKE``.  This port's
    operational moisture-advection helper returns the same coupled
    ``d(mut*qv)/dt`` quantity, so divide by the mass-point dry-air weight
    ``c1h*mu + c2h`` to recover the kg kg-1 s-1 forcing the Tiedtke closure uses.
    The helper is called only when the active flux-form moisture-advection path
    is enabled.
    """

    haloed = apply_halo(state, halo_spec(namelist.grid))
    qv_coupled = _moisture_coupled_tendencies(
        haloed,
        namelist,
        rk_step=1,
        step_origin=state,
    )[0]
    metrics = namelist.metrics
    mass = (
        metrics.c1h[:, None, None] * haloed.mu_total[None, :, :]
        + metrics.c2h[:, None, None]
    )
    return qv_coupled / mass


def _apply_moisture_large_step(
    state: State,
    step_origin: State,
    *,
    q_tendencies: tuple[jax.Array, ...],
    dt_rk: float,
    metrics: DycoreMetrics,
    species: tuple[str, ...] = _MOISTURE_SPECIES,
    f2_pd_out: dict | None = None,
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
    for name, q_tend in zip(species, q_tendencies):
        q_old = getattr(step_origin, name)
        # q_new = (mut_old*moist_old + dt_rk*adv_tend) / mut_new (WRF scalar update).
        if f2_pd_out is not None and name == "qc":
            f2_numerator = mass_old * q_old + float(dt_rk) * q_tend
            q_new = f2_numerator * inv_mass_new
            f2_pd_out["update"] = dict(mass_old=mass_old, mass_new=mass_new,
                inverse_mass_new=inv_mass_new, numerator=f2_numerator, result=q_new)
        else:
            q_new = (mass_old * q_old + float(dt_rk) * q_tend) * inv_mass_new
        updates[name] = q_new
    return state.replace(**updates)


def _nested_scalar_stage_tendencies(
    advected_tendencies: tuple[jax.Array, ...] | None,
    advected_species: tuple[str, ...],
    frozen_boundary_tendencies: tuple[jax.Array, ...],
    config: BoundaryConfig,
    msfty: jax.Array,
) -> tuple[tuple[str, ...], tuple[jax.Array, ...]]:
    """Merge evolving scalar advection with RK1-frozen WRF boundary tendency.

    ``rk_update_scalar`` multiplies the raw scalar advection by ``msfty``,
    excludes the outer ``spec_zone`` from that advection, adds the frozen
    ``sc_tend`` unscaled everywhere, and integrates every stage from the
    step-start scalar/mass.  The live nested path is always non-periodic in this
    runtime, matching pristine WRF's four-sided limits.  Optional advected
    species without a represented boundary record retain the same map-scaled
    advection semantics.
    """

    advected = {
        name: tendency
        for name, tendency in zip(
            advected_species,
            advected_tendencies or (),
            strict=True,
        )
    }
    frozen = dict(
        zip(
            NESTED_BOUNDARY_SCALAR_SPECIES,
            frozen_boundary_tendencies,
            strict=True,
        )
    )
    species = tuple(advected_species) + tuple(
        name
        for name in NESTED_BOUNDARY_SCALAR_SPECIES
        if name not in advected
    )
    merged: list[jax.Array] = []
    spec_zone = int(config.spec_zone)
    for name in species:
        adv = advected.get(name)
        bdy = frozen.get(name)
        if adv is not None:
            # module_em.F::rk_update_scalar uses
            #   advect_tend(i,k,j) * msfty(i,j) + sc_tend(i,k,j)
            # inside the advection rectangle.  sc_tend is already the coupled
            # WRF boundary tendency and must not receive the map factor.
            map_factor = jnp.asarray(msfty, dtype=adv.dtype)[None, :, :]
            adv = adv * map_factor
        if adv is not None and bdy is not None and spec_zone > 0:
            # module_em.F rk_update_scalar: advection rectangle excludes the
            # specified rows/columns, while sc_tend covers the full mass grid.
            interior = jnp.zeros_like(adv)
            y_stop = int(adv.shape[-2]) - spec_zone
            x_stop = int(adv.shape[-1]) - spec_zone
            interior = interior.at[
                ...,
                spec_zone:y_stop,
                spec_zone:x_stop,
            ].set(adv[..., spec_zone:y_stop, spec_zone:x_stop])
            adv = interior
        if adv is None:
            assert bdy is not None
            merged.append(bdy)
        elif bdy is None:
            merged.append(adv)
        else:
            merged.append(adv + bdy)
    return species, tuple(merged)


def _root_scalar_stage_tendencies(
    advected_tendencies: tuple[jax.Array, ...],
    species: tuple[str, ...],
    state: State,
    config: BoundaryConfig,
    msfty: jax.Array,
    *,
    bounded: bool,
    sc_tendencies: dict[str, jax.Array] | None = None,
) -> tuple[jax.Array, ...]:
    """Root-domain ``rk_update_scalar`` advection term (module_em.F:1587-1799).

    WRF multiplies the raw ``advect_tend`` by ``msfty`` inside the advection
    rectangle, which excludes the outer ``spec_zone`` only on a specified or
    nested domain (``bounded``; periodic runs advect everywhere).  The root's
    lateral forcing stays the end-of-step ``apply_lateral_boundaries`` pass, so
    the ring is excluded only for species that carry a boundary leaf (the pass
    owns that ring); a species without one keeps its ring advection because the
    root has no ``flow_dep_bdy`` substitute.  ``sc_tendencies`` is WRF's
    RK1-frozen ``sc_tend`` (sixth-order diffusion), added unscaled on the full
    mass grid as rk_update_scalar does.
    """

    spec_zone = int(config.spec_zone) if bool(bounded) else 0
    merged: list[jax.Array] = []
    for name, adv in zip(species, advected_tendencies, strict=True):
        adv = adv * jnp.asarray(msfty, dtype=adv.dtype)[None, :, :]
        if spec_zone > 0 and getattr(state, f"{name}_bdy", None) is not None:
            y_stop = int(adv.shape[-2]) - spec_zone
            x_stop = int(adv.shape[-1]) - spec_zone
            adv = jnp.zeros_like(adv).at[
                ..., spec_zone:y_stop, spec_zone:x_stop
            ].set(adv[..., spec_zone:y_stop, spec_zone:x_stop])
        if sc_tendencies is not None and name in sc_tendencies:
            adv = adv + jnp.asarray(sc_tendencies[name], dtype=adv.dtype)
        merged.append(adv)
    return tuple(merged)


def _root_sc_with_boundary(
    sc_tendencies: dict[str, jax.Array] | None,
    boundary: dict[str, jax.Array],
    spec_zone: int,
) -> dict[str, jax.Array]:
    """B43 root RK1 ``sc_tend``: rk_scalar_tend terms, then relax_bdy_scalar
    (added) and spec_bdy_scalar (overwrites the spec zone), solve_em.F:2299-2380."""

    merged = dict(sc_tendencies or {})
    for name, bdy in boundary.items():
        base = merged.get(name)
        if base is None:
            merged[name] = bdy
            continue
        ny, nx = (int(n) for n in base.shape[-2:])
        y = jnp.arange(ny)[:, None]
        x = jnp.arange(nx)[None, :]
        spec = (jnp.minimum(jnp.minimum(y, ny - 1 - y), jnp.minimum(x, nx - 1 - x)) < int(spec_zone))[None]
        bdy = bdy.astype(base.dtype)
        merged[name] = jnp.where(spec, bdy, base + bdy)
    return merged


def _root_flow_dep(state: State, names: tuple[str, ...], ru_m, rv_m, config: BoundaryConfig) -> State:
    """B43 flow_dep_bdy on the root flow species, fenced from the next stage's species stack.

    Unfenced, XLA merged the seven ring copy/selects into the stacked scalar-update transpose
    fusion that packs the next stage's f32[8,44,70,120] (review-b 03:05Z: input_transpose_fusion_76,
    255 regs, 30 spills, STACK 216 B; the A19/E38 pattern).  One barrier on the OUTPUTS stops that
    consumer-side merge; an extra barrier on the inputs (species, ru_m, rv_m) perturbed layout
    assignment into a new 11-input transposing Triton fusion (STACK 528 B, BD92 fence r1), so it is
    not used.  Values are unchanged (flow_dep_bdy is copy/select only).
    """

    if not names:
        return state
    out = jax.lax.optimization_barrier(
        tuple(flow_dep_bdy(getattr(state, name), ru_m, rv_m, config) for name in names))
    return state.replace(**dict(zip(names, out)))


_SCALAR_FAMILY_SPECIES = frozenset(("Ni", "Nr", "Ns", "Ng", "Nc", "Nn", "Nh", "nwfa", "nifa"))


def _pd_family_species(namelist: OperationalNamelist, species) -> tuple[str, ...]:
    """Species whose WRF loop takes rk_update_scalar_pd at rk_step == rk_order.

    solve_em.F:1932-2000: every moist_adv_opt / scalar_adv_opt except ORIGINAL (0)
    and WENO_SCALAR (3) pre-applies the frozen moist_tend / scalar_tend to the
    step-start field before the limited final-stage advection.
    """

    moist = int(namelist.moist_adv_opt) not in (0, 3)
    scalar = int(namelist.scalar_adv_opt) not in (0, 3)
    return tuple(n for n in species if (scalar if n in _SCALAR_FAMILY_SPECIES else moist))


def _physics_sc_outside_spec(
    physics_sc: dict, owned: tuple[str, ...], spec_zone: int, *, bounded: bool,
) -> dict:
    """Lateral ring ownership of the physics part of sc_tend (WRF caller wiring, E62).

    Moist species reach moist_tend through update_phy_ten -> add_a2a
    (module_physics_addtendc.F:2435-2481), which on a specified or nested domain
    (``bounded``) skips exactly the outermost mass ring (i in [ids+1, ide-2],
    j in [jds+1, jde-2]) whatever spec_zone is.  MYNN number scalars bypass add_a2a:
    pbl_driver writes scalar_tend(P_QNI) = RQNIBLTEN on the full mass tile
    (module_pbl_driver.F:1866-1874) and calculate_phy_tend couples it there
    (module_em.F:2473-2481).  spec_bdy_scalar then OVERWRITES rings < spec_zone of
    every species with a lateral spec (``owned``; solve_em.F:2345-2372, :2892-2920).
    """

    out = {}
    for name, tend in physics_sc.items():
        width = 1 if bool(bounded) and name not in _SCALAR_FAMILY_SPECIES else 0
        if name in owned:
            width = max(width, int(spec_zone))
        if width > 0:
            ny, nx = (int(n) for n in tend.shape[-2:])
            y = jnp.arange(ny)[:, None]
            x = jnp.arange(nx)[None, :]
            ring = jnp.minimum(jnp.minimum(y, ny - 1 - y), jnp.minimum(x, nx - 1 - x))
            tend = jnp.where((ring < width)[None], jnp.zeros_like(tend), tend)
        out[name] = tend
    return out


def _rk_update_scalar_pd(origin: State, species, sources, dt_rk: float, metrics: DycoreMetrics) -> State:
    """module_em.F rk_update_scalar_pd as solve_em.F:1942-1948 calls it.

    ``scalar = ((c1*mu_old+c2)*scalar + dt*sc_tend)/(c1*mu_new+c2)`` with
    mu_old = mu_new = grid%mu_1 (the step-start mass) for every PD-family species;
    ``sc_tend`` is the whole frozen moist_tend (physics + diffusion + boundary), zero
    where none, and is consumed here (the final rk_update_scalar adds advection only).
    """

    m1 = metrics.c1h[:, None, None] * origin.mu_total[None, :, :] + metrics.c2h[:, None, None]
    updates = {}
    for name in species:
        q = getattr(origin, name)
        tend = None
        for src in sources:
            if src is not None and name in src:
                part = jnp.asarray(src[name], q.dtype)
                tend = part if tend is None else tend + part
        tend = jnp.zeros_like(q) if tend is None else tend
        mass = m1.astype(q.dtype)
        updates[name] = (mass * q + jnp.asarray(float(dt_rk), q.dtype) * tend) / mass
    return origin.replace(**updates)


def _with_physics_sc(q_species, q_tendencies, physics_sc: dict | None, skip: tuple[str, ...]) -> tuple:
    """Add the physics part of sc_tend (unscaled, full mass grid) to non-PD stage tendencies."""

    if physics_sc is None:
        return q_tendencies
    missing = sorted(set(physics_sc) - set(q_species))
    if missing:
        raise NotImplementedError(f"GPUWRF_PHYS_TEND_RK_WRF: physics sc_tend for non-advected {missing}")
    return tuple(
        t + jnp.asarray(physics_sc[n], t.dtype) if (n in physics_sc and n not in skip) else t
        for n, t in zip(q_species, q_tendencies, strict=True)
    )


def _rk_scan_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    debug: bool = False,
    lead_seconds=None,
    physics_tendencies: DryPhysicsTendencies | None = None,
    capture_pre_halo: bool = False,
    capture_rca: bool = False,
    capture_phase_tap: bool = False,
    capture_ladder: bool = False,
    root_scalar_flux_out: dict | None = None,
    f2_pd_out: dict | None = None,
) -> OperationalCarry | _PreHaloCaptureResult | _RcaRkResult | CorrectedNiPhaseTapResult | _RkLadderResult:
    if sum(bool(value) for value in (capture_pre_halo, capture_rca, capture_phase_tap, capture_ladder)) > 1:
        raise ValueError("pre-halo, RCA, phase-tap, and ladder captures are mutually exclusive")
    if carry.h_diabatic is not None:
        # WRF h_diabatic pair: the previous microphysics heating enters t_tend at
        # every stage (rk_addtend_dry, module_em.F:1079) and the final-stage
        # finish removes it (_carry_from_finished_stage); one carry leaf feeds both.
        physics_tendencies = dataclass_replace(
            DryPhysicsTendencies() if physics_tendencies is None else physics_tendencies,
            h_diabatic=carry.h_diabatic,
        )
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin
    if f2_pd_out is not None:
        f2_pd_out["rk1_origin"] = rk1_reference
    nested_frozen_bundle = _nested_frozen_wrf_boundary_active(namelist)
    # GPUWRF_PHYS_TEND_RK_WRF: the physics part of moist_tend/scalar_tend (coupled,
    # RK1-frozen); None keeps every released program unchanged.
    phys_moist_sc = (
        physics_tendencies.moist_tendf
        if physics_tendencies is not None and physics_tendencies.moist_tendf
        else None
    )
    # WRF first_rk_step_part2 builds diff_opt=1 coefficients from the time-t
    # fields and module_em.F::rk_tendency adds the dry diffusion only inside
    # ``forward_step`` (rk_step == 1).  rk_addtend_dry then reuses the complete
    # ru/rv/rw/t_tendf bundle in all three RK stages.  Hoist that immutable
    # device bundle beside the other step-origin forcing; no carry leaf or
    # host/device transfer is introduced.
    # The same RK1 Smagorinsky ``xkhh`` drives rk_scalar_tend's moist/other-
    # scalar horizontal_diffusion (module_em.F:1380); keep it beside the bundle.
    rk1_diffopt1_bundle = (
        _diffopt1_dry_forward_tendencies(
            rk1_reference,
            namelist,
            base_state=carry.base_state,
            return_scalar_xkhh=True,
        )
        if int(namelist.diff_opt) == 1 and int(namelist.km_opt) == 4
        else None
    )
    rk1_forward_diffopt1 = None if rk1_diffopt1_bundle is None else rk1_diffopt1_bundle[:4]
    rk1_scalar_xkhh = None if rk1_diffopt1_bundle is None else rk1_diffopt1_bundle[4]
    # o1-smag3d: WRF diff_opt=2/km_opt=3 forms its whole diffusion bundle (dry *_tendf and
    # moist/scalar sc_tend) once in first_rk_step_part2 from the time-t fields.
    rk1_les3d_km3 = None
    if int(namelist.diff_opt) == 2 and int(namelist.km_opt) == 3:
        from gpuwrf.runtime.les3d_km3 import les3d_km3_forward_tendencies

        rk1_les3d_km3 = les3d_km3_forward_tendencies(rk1_reference, namelist, base_state=carry.base_state)
    # Canonical real-data nests select diff_6th_opt=2.  WRF builds theta's
    # sixth-order contribution once from the time-t/RK1 field into ``t_tendf``;
    # the previous operational path rebuilt a periodic, cell-mass approximation
    # in all three stages, including physical rings where WRF owns no stencil.
    # Keep this correction behind the coherent live-child bundle so every
    # released/candidate-off program remains byte-identical.
    # WRF sixth_order_diffusion treats specified exactly like nested
    # (``specified = config_flags%specified .or. config_flags%nested``,
    # module_big_step_utilities_em.F:6334), so the specified root takes the same
    # RK1-frozen ownership-ring form; the released per-stage periodic form wraps
    # opposite-boundary values into rings 0-2 (BC44: 18x theta ring-1 error vs
    # pristine WRF after one d01 step).  Periodic idealized programs keep it.
    sixth_order_frozen = nested_frozen_bundle or _acoustic_lateral_bc_flags(namelist)[1]
    rk1_forward_diff6_theta = None
    if sixth_order_frozen and int(namelist.diff_6th_opt) != 0:
        theta_tendf = wrf_sixth_order_scalar_tendf(
            rk1_reference.theta - _theta_base_offset(rk1_reference.theta),
            rk1_reference.mu_total,
            c1=namelist.metrics.c1h,
            c2=namelist.metrics.c2h,
            msftx=namelist.metrics.msftx,
            msfty=namelist.metrics.msfty,
            dt=float(namelist.dt_s),
            diff_6th_factor=float(namelist.diff_6th_factor),
            monotonic=(int(namelist.diff_6th_opt) == 2),
            specified_or_nested=True,
        )
        rk1_forward_diff6_theta = (
            theta_tendf
            / namelist.metrics.msfty.astype(theta_tendf.dtype)[None, :, :]
        )
    # Momentum shares theta's rk1-only tendf cadence (module_em.F:882-916):
    # WRF forms u/v/w sixth-order diffusion once from the time-t fields with
    # the specified/nested ownership rings, adjacent-face masses, and
    # per-direction map factors, and reuses the folded result at every stage.
    # S1 attribution (sprint 2026-07-18-v0234-s1-dyn-attribution-fable5):
    # the prior per-stage periodic form explained 97.8% of the frozen 8105x
    # rk_tendency nonspec residual (relax band 98.05%, corr 0.992).
    rk1_forward_diff6_uvw = None
    if sixth_order_frozen and int(namelist.diff_6th_opt) != 0:
        rk1_forward_diff6_uvw = wrf_sixth_order_uvw_tendf(
            rk1_reference.u,
            rk1_reference.v,
            rk1_reference.w,
            rk1_reference.mu_total,
            c1h=namelist.metrics.c1h,
            c2h=namelist.metrics.c2h,
            c1f=namelist.metrics.c1f,
            c2f=namelist.metrics.c2f,
            msfux=namelist.metrics.msfux,
            msfuy=namelist.metrics.msfuy,
            msfvx=namelist.metrics.msfvx,
            msfvy=namelist.metrics.msfvy,
            msftx=namelist.metrics.msftx,
            msfty=namelist.metrics.msfty,
            dt=float(namelist.dt_s),
            diff_6th_factor=float(namelist.diff_6th_factor),
            monotonic=(int(namelist.diff_6th_opt) == 2),
        )
    # Root (specified) counterpart of the child's RK1 scalar sc_tend bundle:
    # rk_scalar_tend forms the moist/other-scalar sixth-order diffusion only at
    # RK1 with dt_step (module_em.F:1423) and rk_update_scalar reuses it in all
    # stages.  The root previously had no scalar sixth-order diffusion at all.
    # rk_scalar_tend runs AFTER the RK1 acoustic loop and receives grid%muts
    # (solve_em.F:2303), so the mass weights use the post-acoustic RK1 total
    # mass; the stage function forms it once at RK1 and caches it (stages are
    # Python-unrolled in one trace).
    rk1_scalar_cache: dict[str, object] = {}
    # B43 (GPUWRF_ROOT_SCALAR_BDY_RK1, default off): the specified root's
    # moist/scalar lateral forcing in WRF's RK cadence -- RK1-frozen
    # relax/spec sc_tend for QV (solve_em.F:2345-2380) and flow_dep_bdy after
    # every stage's scalar update for the other species (:2426-2438, :2995).
    root_scalar_rk1 = (not nested_frozen_bundle) and _root_scalar_bdy_rk1_active(namelist)
    if root_scalar_rk1 and lead_seconds is None:
        raise ValueError("GPUWRF_ROOT_SCALAR_BDY_RK1 needs the step boundary lead")
    root_scalar_diff6_active = (
        not nested_frozen_bundle
        and sixth_order_frozen
        and int(namelist.diff_6th_opt) != 0
        and bool(namelist.use_flux_advection)
    )
    # rk_scalar_tend's diff_opt=1 horizontal_diffusion shares that RK1 sc_tend
    # (added before the sixth-order term, module_em.F:1380-1423).
    root_scalar_hdiff_active = (
        not nested_frozen_bundle
        and (rk1_scalar_xkhh is not None or rk1_les3d_km3 is not None)
        and bool(namelist.use_flux_advection)
    )

    def _root_scalar_sc_tend(muts_rk1):
        hdiff = (
            _diffopt1_scalar_horizontal_diffusion(rk1_reference, rk1_scalar_xkhh, muts_rk1, namelist)
            if root_scalar_hdiff_active and rk1_scalar_xkhh is not None
            else (dict(rk1_les3d_km3.scalar_sc) if root_scalar_hdiff_active else {})
        )
        if not root_scalar_diff6_active:
            return hdiff
        diff6 = _root_scalar_diff6(muts_rk1)
        return {
            name: (hdiff[name].astype(d.dtype) + d) if name in hdiff else d
            for name, d in diff6.items()
        }

    def _root_scalar_diff6(muts_rk1):
        rk1_dt = float(namelist.dt_s) / float(namelist.rk_order)
        return {
            name: wrf_sixth_order_scalar_tendf(
                getattr(rk1_reference, name),
                muts_rk1,
                c1=namelist.metrics.c1h,
                c2=namelist.metrics.c2h,
                msftx=namelist.metrics.msftx,
                msfty=namelist.metrics.msfty,
                dt=rk1_dt,
                diff_6th_factor=float(namelist.diff_6th_factor),
                monotonic=(int(namelist.diff_6th_opt) == 2),
                specified_or_nested=True,
            )
            for name in NESTED_BOUNDARY_SCALAR_SPECIES
            if getattr(rk1_reference, name, None) is not None
        }

    # Pristine relax_bdy_dry runs at RK1 and stores *_tendf for reuse.  Hoist the
    # candidate child's construction outside advance_stage so it is traced once
    # from the immutable step-start fields.  The released specified path remains
    # in its historical per-stage Python construction below for inactive-path
    # JAXPR/HLO identity.
    nested_frozen_relax = (
        _nested_frozen_bdy_relax(rk1_reference, namelist, lead_seconds)
        if nested_frozen_bundle
        else None
    )
    # solve_em.F builds scalar relax/spec tendencies only at RK1 and leaves the
    # resulting sc_tend resident for RK2/RK3.  Hoist the candidate bundle beside
    # the dry tendf bundle so every stage consumes identical boundary forcing.
    nested_frozen_scalar = (
        nested_scalar_boundary_tendencies(
            rk1_reference,
            lead_seconds,
            namelist.metrics,
            float(namelist.dt_s),
            namelist.boundary_config,
        )
        if nested_frozen_bundle and lead_seconds is not None
        else None
    )
    # WRF's moist/other-scalar sixth-order lane shares the same step-persistent
    # sc_tend arrays as the RK1 boundary tendency above.  Form it once from the
    # time-t fields with dt/3, then add it without map rescaling before all three
    # scalar updates.  Rings 0--2 remain exact zero by source ownership.
    nested_frozen_scalar_bdy = nested_frozen_scalar

    def _child_frozen_scalar(muts_rk1):
        # Post-acoustic RK1 sc_tend: boundary part (time-t) + diff_opt=1
        # horizontal mixing + sixth-order with grid%muts as rk_scalar_tend
        # receives it (solve_em.F:2303, module_em.F:1380-1423).
        hdiff = (
            _diffopt1_scalar_horizontal_diffusion(rk1_reference, rk1_scalar_xkhh, muts_rk1, namelist)
            if rk1_scalar_xkhh is not None
            else (None if rk1_les3d_km3 is None else dict(rk1_les3d_km3.scalar_sc))
        )
        if int(namelist.diff_6th_opt) == 0:
            if hdiff is None:
                return nested_frozen_scalar_bdy
            return tuple(
                b + hdiff[name].astype(b.dtype)
                for b, name in zip(nested_frozen_scalar_bdy, NESTED_BOUNDARY_SCALAR_SPECIES, strict=True)
            )
        diffusion = _nested_scalar_sixth_order_tendencies(
            rk1_reference.replace(mu_total=muts_rk1), namelist,
        )
        if hdiff is not None:
            diffusion = tuple(
                hdiff[name].astype(d.dtype) + d
                for name, d in zip(NESTED_BOUNDARY_SCALAR_SPECIES, diffusion, strict=True)
            )
        return tuple(b + d for b, d in zip(nested_frozen_scalar_bdy, diffusion, strict=True))

    if nested_frozen_scalar is not None and int(namelist.diff_6th_opt) != 0:
        # C1 compatibility arm (pre-acoustic, both scalar options 0) keeps the
        # released time-t bundle byte-identical.
        rk1_forward_diff6_scalar = _nested_scalar_sixth_order_tendencies(
            rk1_reference,
            namelist,
        )
        nested_frozen_scalar = tuple(
            boundary + diffusion
            for boundary, diffusion in zip(
                nested_frozen_scalar,
                rk1_forward_diff6_scalar,
                strict=True,
            )
        )

    def _rk1_cached(key, stage, muts_now, build):
        if int(stage.rk_step) == 1 or key not in rk1_scalar_cache:
            assert int(stage.rk_step) == 1, "RK1 must form the frozen scalar sc_tend first"
            rk1_scalar_cache[key] = build(muts_now)
        return rk1_scalar_cache[key]

    def advance_stage(
        stage_carry: OperationalCarry,
        stage: _RKStageDescriptor,
        *,
        capture_stage_pre_halo: bool = False,
        capture_stage_rca: bool = False,
        capture_stage_phase_tap: bool = False,
        capture_stage_ladder: bool = False,
    ) -> (
        OperationalCarry
        | _PreHaloCaptureResult
        | _RcaAcousticScanResult
        | CorrectedNiPhaseTapResult
        | tuple[OperationalCarry, tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
    ):
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
        # The stage's ru/rv/rom transport build is shared between the theta/
        # momentum flux advection and the moisture scalar advection (WRF builds
        # them once per RK stage in rk_step_prep/calc_ww_cp).  The branch is a
        # STATIC Python condition, so with flux advection off nothing changes.
        stage_velocities = (
            _stage_transport_velocities(haloed, namelist)
            if bool(namelist.use_flux_advection)
            else None
        )
        # v0.14 SPECIFIED WRF boundary cadence: the step-constant relax_bdy_dry
        # bundle from the rk1 reference (identical values at every stage; WRF
        # computes it once at rk_step==1 and folds it into the *_tendf lane).
        bdy_relax = (
            nested_frozen_relax
            if nested_frozen_bundle
            else _specified_bdy_relax(rk1_reference, namelist, lead_seconds)
        )
        tendencies = _augment_large_step_tendencies(
            haloed,
            tendencies,
            namelist,
            rk_step=int(stage.rk_step),
            physics_tendencies=physics_tendencies,
            step_origin=rk1_reference,
            transport_velocities=stage_velocities,
            bdy_relax=bdy_relax,
            base_state=stage_carry.base_state,
            frozen_diffopt1_tendencies=rk1_forward_diffopt1,
            frozen_diff6_theta_tendency=rk1_forward_diff6_theta,
            frozen_diff6_uvw_tendencies=rk1_forward_diff6_uvw,
            frozen_les3d_km3=None if rk1_les3d_km3 is None else tuple(rk1_les3d_km3[:4]),
        ); tendencies, stage_carry, haloed = _small_grid_firewall((tendencies, stage_carry, haloed), stage_carry.state.theta.shape[-2:])
        # WRF advances moisture/other scalars after acoustic integration and
        # constructs their tendencies with ``sumflux`` -- the time-average of
        # live acoustic ru/rv/ww plus the saved linear stage flux.  The released
        # nested 0/0 program already transported QNI/QNR from the stage-entry
        # operands, however.  Preserve that exact compatibility control when
        # BOTH scalar options are off; activating either source option selects
        # the coherent post-acoustic path for both WRF scalar loops.  Source:
        # solve_em.F:1566-1581, :2282-2317, and :2855-2883.
        moisture_advected = (
            bool(namelist.use_flux_advection) and int(namelist.moist_adv_opt) != 0
        )
        if nested_frozen_bundle and getattr(namelist, "mp_physics", None) == 18:
            raise NotImplementedError(
                "mp_physics=18 (NSSL) is wired for single (root) domains only: nested parent->child forcing of "
                "the NSSL number/volume scalars (qndrop/qns/qng/qnh/qnn, qvolg/qvolh) is not ported")
        if nested_frozen_bundle:
            # WRF's separate ``other_scalar_advance`` always transports the
            # represented Thompson QNI/QNR fields when flux advection is active;
            # option 0 means its ordinary (unlimited) stencil, not no transport.
            number_scalars_advected = bool(namelist.use_flux_advection)
            moist_species = _advected_scalar_species(namelist) if moisture_advected else ()
            post_acoustic_scalar_transport = bool(
                moisture_advected
                or (
                    number_scalars_advected
                    and int(namelist.scalar_adv_opt) != 0
                )
            )
            if post_acoustic_scalar_transport:
                q_species = ()
                q_tendencies = ()
            else:
                # C1 compatibility arm: this is the exact pre-candidate 0/0
                # nested QNI/QNR construction.  It intentionally keeps the
                # stage-entry state/flux operands so the control remains a
                # byte-identical released program rather than a mixed candidate.
                if number_scalars_advected:
                    assert stage_velocities is not None
                    number_tendencies = _nested_number_scalar_coupled_tendencies(
                        haloed,
                        namelist,
                        rk_step=int(stage.rk_step),
                        step_origin=rk1_reference,
                        transport_velocities=stage_velocities,
                    )
                else:
                    number_tendencies = ()
                advected_species = (
                    ("Ni", "Nr") if number_scalars_advected else ()
                )
                assert nested_frozen_scalar is not None
                q_species, q_tendencies = _nested_scalar_stage_tendencies(
                    number_tendencies,
                    advected_species,
                    nested_frozen_scalar,
                    namelist.boundary_config,
                    namelist.metrics.msfty,
                )
        else:
            number_scalars_advected = bool(namelist.use_flux_advection) and (
                haloed.Ni is not None and haloed.Nr is not None
            )
            moist_species = _advected_scalar_species(namelist) if moisture_advected else ()
            q_species = moist_species + (("Ni", "Nr") if number_scalars_advected else ())
            if getattr(namelist, "mp_physics", None) == 18 and number_scalars_advected:
                q_species = q_species + _nssl_number_scalars(namelist)
            q_tendencies = None
            post_acoustic_scalar_transport = bool(moisture_advected or number_scalars_advected)
        tke_advected = int(namelist.diff_opt) == 2 and int(namelist.km_opt) in (2, 5)
        if tke_advected:
            ph = haloed.ph_total
            dz_tke = jnp.maximum(
                jnp.mean((ph[1:] - ph[:-1]) / GRAVITY_M_S2),
                jnp.asarray(1.0, dtype=ph.dtype),
            )
            mass_h_tke = (
                namelist.metrics.c1h[:, None, None] * haloed.mu_total[None, :, :]
                + namelist.metrics.c2h[:, None, None]
            )
            turbulence_tke = _diffopt2_turbulence_fields(haloed, namelist, dz=dz_tke)
            qke_tendency = _tke_coupled_tendency(
                haloed,
                namelist,
                turbulence_tke,
                mass_h=mass_h_tke,
                dz=dz_tke,
            )
        else:
            qke_tendency = None
        candidate = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        # v0.14 ACOUSTIC-SUBSTEP FIX (fresh stage omega): WRF rk_step_prep
        # RE-DIAGNOSES ``grid%ww`` from the STAGE u/v/mu at EVERY RK stage entry
        # (module_em.F:127-133 -> calc_ww_cp) -- it never carries the post-
        # acoustic omega across stages.  That fresh diagnostic is what
        # small_step_prep saves as ``ww_save`` (= advance_mu_t's ``ww_1``
        # subtraction reference and the end-of-stage omega restore) and what
        # rhs_ph consumes.  ``stage_velocities.rom`` IS that calc_ww_cp omega
        # (flux_advection.couple_velocities_periodic builds it from the same
        # stage-entry haloed state).  The previous code threaded the CARRIED
        # ``carry.ww`` (previous stage's acoustic omega; zero on the first step
        # of a re-init) -- a per-stage omega-semantics divergence in the
        # geopotential RHS and the small-step omega reference.  Legacy callers
        # without flux advection keep the carried omega unchanged.
        # v0.14 continuation: ``rom`` from couple_velocities_periodic wraps the
        # x/y edges periodically -- exact in the interior (oracle parity
        # 5.8e-16) but up to ~5x the physical omega on the outermost
        # row/column of a real specified domain (continuation proof D1: band
        # rmse 6.99 / max 116 vs oracle 2.48 / 24).  Threading that wrapped
        # omega into ww_save/rhs_ph poisoned the lateral band and REGRESSED
        # the h36->h37 mass residual (-27.7 -> -35.5 Pa/cell/h).  For
        # specified/nested real domains build the stage omega with the
        # edge-faithful calc_ww_cp port instead.
        _per_x, _spec, _nest = _acoustic_lateral_bc_flags(namelist)
        if stage_velocities is not None and (_spec or _nest):
            ww_stage = stage_omega_specified(
                haloed.u,
                haloed.v,
                haloed.mu_total,
                c1h=namelist.metrics.c1h,
                c2h=namelist.metrics.c2h,
                dnw=namelist.metrics.dnw,
                rdx=1.0 / float(namelist.grid.projection.dx_m),
                rdy=1.0 / float(namelist.grid.projection.dy_m),
                msfuy=namelist.metrics.msfuy,
                msfvx=namelist.metrics.msfvx,
                msftx=namelist.metrics.msftx,
            )
        elif stage_velocities is not None:
            ww_stage = stage_velocities.rom
        else:
            ww_stage = stage_carry.ww
        prep = small_step_prep_wrf(
            candidate,
            int(stage.rk_step),
            float(stage.dt_rk),
            metrics=namelist.metrics,
            reference_state=rk1_reference,
            ww=ww_stage,
            base_state=stage_carry.base_state,
        )
        pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
        acoustic_result = _acoustic_scan(
            stage_carry.replace(state=candidate),
            namelist,
            stage=stage,
            prep=prep,
            pressure=pressure,
            tendencies=tendencies,
            lead_seconds=lead_seconds,
            capture_pre_halo=capture_stage_pre_halo,
            capture_rca=capture_stage_rca,
            capture_phase_tap=capture_stage_phase_tap,
            return_scalar_transport=post_acoustic_scalar_transport,
            bdy_relax=bdy_relax,
        ); acoustic_result, stage_carry = _small_grid_firewall((acoustic_result, stage_carry), stage_carry.state.theta.shape[-2:])
        scalar_transport_velocities = None
        if post_acoustic_scalar_transport:
            assert isinstance(acoustic_result, _AcousticScalarTransportResult)
            assert stage_velocities is not None
            scalar_transport_velocities = _scalar_transport_velocities_from_sumflux(
                stage_velocities,
                acoustic_result.mass_fluxes,
            )
            acoustic_result = acoustic_result.result
        captured_pre_halo_state = None
        captured_rca_health = None
        captured_rca_target = None
        captured_phase_tap_summary = None
        if capture_stage_pre_halo:
            captured_pre_halo_state = acoustic_result.pre_halo_state
            stage_carry = acoustic_result.carry
        elif capture_stage_rca:
            captured_rca_health = acoustic_result.health
            captured_rca_target = acoustic_result.target
            stage_carry = acoustic_result.carry
        elif capture_stage_phase_tap:
            captured_phase_tap_summary = acoustic_result.summary
            stage_carry = acoustic_result.carry
        else:
            stage_carry = acoustic_result
        # The scalar fields themselves do not change in the sound-step loop,
        # while the coupled dry mass and transporting fluxes do.  Match WRF's
        # post-``small_step_finish`` ownership and construct every represented
        # scalar tendency from that state and the finalized acoustic average.
        f2_pd_stage_out = f2_pd_out if int(stage.rk_step) == int(namelist.rk_order) else None
        if f2_pd_stage_out is not None:
            f2_pd_stage_out.update(spec_zone=int(namelist.boundary_config.spec_zone),
                nested=nested_frozen_bundle, root_flow_qc=False, pd_preload=False,
                physics_sc=None, other_sc=None, dt_rk=float(stage.dt_rk))
        scalar_origin = rk1_reference
        pd_sc_species: tuple[str, ...] = ()
        if phys_moist_sc is not None and not post_acoustic_scalar_transport:
            raise NotImplementedError(
                "GPUWRF_PHYS_TEND_RK_WRF needs the post-acoustic scalar transport path")
        pd_stage = phys_moist_sc is not None and int(stage.rk_step) == int(namelist.rk_order)
        if post_acoustic_scalar_transport:
            if nested_frozen_bundle:
                advected_species = moist_species + (
                    ("Ni", "Nr") if number_scalars_advected else ()
                )
                stage_phys_sc = None
                if phys_moist_sc is not None:
                    stage_phys_sc = _physics_sc_outside_spec(
                        phys_moist_sc,
                        tuple(n for n in NESTED_BOUNDARY_SCALAR_SPECIES
                              if getattr(rk1_reference, f"{n}_bdy", None) is not None),
                        int(namelist.boundary_config.spec_zone),
                        bounded=any(_acoustic_lateral_bc_flags(namelist)[1:]),
                    )
                if pd_stage:
                    pd_sc_species = _pd_family_species(namelist, advected_species)
                    child_sc = dict(zip(
                        NESTED_BOUNDARY_SCALAR_SPECIES,
                        _rk1_cached("child", stage, stage_carry.state.mu_total, _child_frozen_scalar),
                        strict=True,
                    ))
                    if f2_pd_stage_out is not None:
                        f2_pd_stage_out.update(pd_preload="qc" in pd_sc_species,
                            physics_sc=None if stage_phys_sc is None else stage_phys_sc.get("qc"),
                            other_sc=child_sc.get("qc"))
                    scalar_origin = _rk_update_scalar_pd(
                        rk1_reference, pd_sc_species, (stage_phys_sc, child_sc),
                        float(stage.dt_rk), namelist.metrics,
                    )
                limiter_stage = (
                    int(namelist.moist_adv_opt) in (1, 2)
                    and int(stage.rk_step) == int(namelist.rk_order)
                    and rk1_reference is not None
                )
                species_batch_width = _nested_scalar_species_batch_width(
                    getattr(stage_carry.state, advected_species[0]),
                    use_limiter=limiter_stage,
                )
                if (
                    moisture_advected
                    and number_scalars_advected
                    and int(namelist.moist_adv_opt)
                    == int(namelist.scalar_adv_opt)
                    and species_batch_width > 1
                ):
                    # Pristine WRF owns these as two source loops, but their
                    # final-stage limiter equation and static option are
                    # identical. Static pair chunks preserve every per-field
                    # operation while reducing graph width on the smallest,
                    # most-subcycled nest. Plain stages stay as the original
                    # two scalar loops: mapped plain transport failed the
                    # enclosing-jit exactness and performance discriminator.
                    scalar_tendencies = _scalar_transport_coupled_tendencies(
                        stage_carry.state,
                        namelist,
                        rk_step=int(stage.rk_step),
                        step_origin=scalar_origin,
                        species=advected_species,
                        advection_opt=int(namelist.moist_adv_opt),
                        transport_velocities=scalar_transport_velocities,
                        species_batch_width=species_batch_width,
                        **({"f2_pd_out": f2_pd_stage_out} if f2_pd_stage_out is not None else {}),
                    )
                else:
                    moist_tendencies = (
                        _moisture_coupled_tendencies(
                            stage_carry.state,
                            namelist,
                            rk_step=int(stage.rk_step),
                            step_origin=scalar_origin,
                            transport_velocities=scalar_transport_velocities,
                            **({"f2_pd_out": f2_pd_stage_out} if f2_pd_stage_out is not None else {}),
                        )
                        if moisture_advected
                        else ()
                    )
                    number_tendencies = (
                        _nested_number_scalar_coupled_tendencies(
                            stage_carry.state,
                            namelist,
                            rk_step=int(stage.rk_step),
                            step_origin=scalar_origin,
                            transport_velocities=scalar_transport_velocities,
                        )
                        if number_scalars_advected
                        else ()
                    )
                    scalar_tendencies = moist_tendencies + number_tendencies
                assert nested_frozen_scalar is not None
                child_frozen = _rk1_cached("child", stage, stage_carry.state.mu_total, _child_frozen_scalar)
                if pd_sc_species:
                    # rk_update_scalar_pd consumed (and zeroed) sc_tend for these.
                    child_frozen = tuple(
                        jnp.zeros_like(f) if n in pd_sc_species else f
                        for n, f in zip(NESTED_BOUNDARY_SCALAR_SPECIES, child_frozen, strict=True)
                    )
                q_species, q_tendencies = _nested_scalar_stage_tendencies(
                    scalar_tendencies,
                    advected_species,
                    child_frozen,
                    namelist.boundary_config,
                    namelist.metrics.msfty,
                )
                q_tendencies = _with_physics_sc(q_species, q_tendencies, stage_phys_sc, pd_sc_species)
            else:
                def _root_stage_sc():
                    root_sc = _rk1_cached(
                        "root", stage, stage_carry.state.mu_total, _root_scalar_sc_tend,
                    ) if (root_scalar_diff6_active or root_scalar_hdiff_active) else None
                    relaxed = flow = ()
                    if root_scalar_rk1:
                        relaxed, flow = root_scalar_rk1_split(namelist.boundary_config, q_species, **_nssl_root_flow_only(namelist))
                        root_sc = _root_sc_with_boundary(
                            root_sc,
                            _rk1_cached("root_bdy", stage, None, lambda _unused: root_scalar_boundary_tendencies(
                                rk1_reference, lead_seconds, namelist.metrics, float(namelist.dt_s),
                                namelist.boundary_config, relaxed,
                            )),
                            int(namelist.boundary_config.spec_zone),
                        )
                    return root_sc, relaxed, flow

                stage_phys_sc = None
                root_sc_early = None
                if phys_moist_sc is not None:
                    owned = (
                        root_scalar_rk1_split(namelist.boundary_config, q_species, **_nssl_root_flow_only(namelist))[0]
                        if root_scalar_rk1 else ()
                    )
                    stage_phys_sc = _physics_sc_outside_spec(
                        phys_moist_sc, tuple(owned), int(namelist.boundary_config.spec_zone),
                        bounded=any(_acoustic_lateral_bc_flags(namelist)[1:]))
                if pd_stage:
                    pd_sc_species = _pd_family_species(namelist, q_species)
                    root_sc_early = _root_stage_sc()
                    if f2_pd_stage_out is not None:
                        f2_pd_stage_out.update(pd_preload="qc" in pd_sc_species,
                            physics_sc=None if stage_phys_sc is None else stage_phys_sc.get("qc"),
                            other_sc=None if root_sc_early[0] is None else root_sc_early[0].get("qc"))
                    scalar_origin = _rk_update_scalar_pd(
                        rk1_reference, pd_sc_species, (stage_phys_sc, root_sc_early[0]),
                        float(stage.dt_rk), namelist.metrics,
                    )
                if (
                    moisture_advected
                    and number_scalars_advected
                    and int(namelist.moist_adv_opt) == int(namelist.scalar_adv_opt)
                ):
                    # Same equation/option for both WRF scalar loops: one call so
                    # the native stencils see all species in one stacked launch.
                    root_advection = _scalar_transport_coupled_tendencies(
                        stage_carry.state, namelist, rk_step=int(stage.rk_step),
                        step_origin=scalar_origin, species=q_species,
                        advection_opt=int(namelist.moist_adv_opt),
                        transport_velocities=scalar_transport_velocities,
                        **({"f2_pd_out": f2_pd_stage_out} if f2_pd_stage_out is not None else {}),
                    )
                else:
                    root_advection = (
                        _moisture_coupled_tendencies(
                            stage_carry.state, namelist, rk_step=int(stage.rk_step),
                            step_origin=scalar_origin,
                            transport_velocities=scalar_transport_velocities,
                            **({"f2_pd_out": f2_pd_stage_out} if f2_pd_stage_out is not None else {}),
                        ) if moisture_advected else ()
                    ) + (
                        _nested_number_scalar_coupled_tendencies(
                            stage_carry.state, namelist, rk_step=int(stage.rk_step),
                            step_origin=scalar_origin,
                            transport_velocities=scalar_transport_velocities,
                        ) if number_scalars_advected else ()
                    )
                root_sc, root_rk1_relaxed, root_rk1_flow = (
                    root_sc_early if root_sc_early is not None else _root_stage_sc()
                )
                if pd_sc_species and root_sc is not None:
                    # rk_update_scalar_pd consumed (and zeroed) sc_tend for these.
                    root_sc = {n: t for n, t in root_sc.items() if n not in pd_sc_species}
                q_tendencies = _root_scalar_stage_tendencies(
                    root_advection,
                    q_species,
                    stage_carry.state,
                    namelist.boundary_config,
                    namelist.metrics.msfty,
                    bounded=any(_acoustic_lateral_bc_flags(namelist)[1:]),
                    sc_tendencies=root_sc,
                )
                q_tendencies = _with_physics_sc(q_species, q_tendencies, stage_phys_sc, pd_sc_species)
        if moisture_advected or number_scalars_advected or nested_frozen_bundle:
            stage_carry = stage_carry.replace(
                state=_apply_moisture_large_step(
                    stage_carry.state,
                    scalar_origin,
                    q_tendencies=q_tendencies,
                    dt_rk=float(stage.dt_rk),
                    metrics=namelist.metrics,
                    # ADR-032: same static species as the coupled-tendency build
                    # above (core six for every wired scheme; + qh for WSM7 mp=24
                    # and the rest of the wired hail family; + nwfa/nifa for mp=28).
                    species=q_species,
                    **({"f2_pd_out": f2_pd_stage_out} if f2_pd_stage_out is not None else {}),
                )
            )
        if f2_pd_stage_out is not None:
            f2_pd_stage_out.update(mu_update=stage_carry.state.mu_total,
                before_flow=stage_carry.state.qc,
                merged_tendency=q_tendencies[q_species.index("qc")])
        if root_scalar_rk1 and (moisture_advected or number_scalars_advected):
            # solve_em.F:2426-2438 / :2995-3015: flow_dep_bdy right after this
            # stage's rk_update_scalar, signed by the acoustic-averaged ru_m/rv_m.
            assert scalar_transport_velocities is not None and scalar_transport_velocities.ru_full is not None
            ru_m, rv_m = scalar_transport_velocities.ru_full, scalar_transport_velocities.rv_full
            stage_carry = stage_carry.replace(state=_root_flow_dep(
                stage_carry.state, root_rk1_flow, ru_m, rv_m, namelist.boundary_config))
            if f2_pd_stage_out is not None:
                f2_pd_stage_out["root_flow_qc"] = "qc" in root_rk1_flow
            if root_scalar_flux_out is not None:
                root_scalar_flux_out.update(ru_m=ru_m, rv_m=rv_m, relaxed=root_rk1_relaxed, flow=root_rk1_flow)
        if tke_advected:
            stage_carry = stage_carry.replace(
                state=_apply_tke_large_step(
                    stage_carry.state,
                    rk1_reference,
                    qke_tendency=qke_tendency,
                    dt_rk=float(stage.dt_rk),
                    metrics=namelist.metrics,
                    tke_upper_bound=float(namelist.tke_upper_bound),
                )
            )
        stage_carry = stage_carry.replace(state=apply_halo(stage_carry.state, halo_spec(namelist.grid)))
        if capture_stage_pre_halo:
            return _PreHaloCaptureResult(stage_carry, captured_pre_halo_state)
        if capture_stage_rca:
            assert captured_rca_health is not None
            assert captured_rca_target is not None
            return _RcaAcousticScanResult(
                stage_carry, captured_rca_health, captured_rca_target,
            )
        if capture_stage_phase_tap:
            assert captured_phase_tap_summary is not None
            return CorrectedNiPhaseTapResult(stage_carry, captured_phase_tap_summary)
        if capture_stage_ladder:
            relax_u = (
                bdy_relax.ru if bdy_relax is not None else jnp.zeros_like(tendencies.u)
            )
            relax_v = (
                bdy_relax.rv if bdy_relax is not None else jnp.zeros_like(tendencies.v)
            )
            return stage_carry, (tendencies.u, tendencies.v), (relax_u, relax_v)
        return stage_carry

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
    if capture_rca:
        stage1 = advance_stage(carry, stages[0], capture_stage_rca=True)
        stage2 = advance_stage(stage1.carry, stages[1], capture_stage_rca=True)
        stage3 = advance_stage(stage2.carry, stages[2], capture_stage_rca=True)
        return _RcaRkResult(
            stage3.carry,
            jnp.concatenate((stage1.health, stage2.health, stage3.health), axis=0),
            jnp.concatenate((stage1.target, stage2.target, stage3.target), axis=0),
        )
    if capture_phase_tap:
        carry = advance_stage(carry, stages[0])
        carry = advance_stage(carry, stages[1])
        return advance_stage(carry, stages[2], capture_stage_phase_tap=True)
    if capture_ladder:
        stage1, rk1_tend, rk1_relax = advance_stage(
            carry, stages[0], capture_stage_ladder=True,
        )
        stage2 = advance_stage(stage1, stages[1])
        stage3 = advance_stage(stage2, stages[2])
        return _RkLadderResult(
            stage3,
            rk1_tend_u=rk1_tend[0],
            rk1_tend_v=rk1_tend[1],
            rk1_relax_u=rk1_relax[0],
            rk1_relax_v=rk1_relax[1],
            rk1_fin_u=stage1.state.u,
            rk1_fin_v=stage1.state.v,
            rk2_fin_u=stage2.state.u,
            rk2_fin_v=stage2.state.v,
            rk3_fin_u=stage3.state.u,
            rk3_fin_v=stage3.state.v,
        )
    carry = advance_stage(carry, stages[0]); fwcar = _small_grid_firewall(carry, carry.state.theta.shape[-2:])
    carry = advance_stage(fwcar, stages[1]); fwcar = _small_grid_firewall(carry, carry.state.theta.shape[-2:])
    return advance_stage(fwcar, stages[2], capture_stage_pre_halo=capture_pre_halo)


def _rk_scan_step_with_pre_halo_capture(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    debug: bool = False,
    lead_seconds=None,
    physics_tendencies: DryPhysicsTendencies | None = None,
) -> _PreHaloCaptureResult:
    """Proof-only helper for final-RK post-refresh state before RK halo exchange."""

    result = _rk_scan_step(
        carry,
        namelist,
        debug=debug,
        lead_seconds=lead_seconds,
        physics_tendencies=physics_tendencies,
        capture_pre_halo=True,
    )
    return result


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
        "qc_bdy": state.qc_bdy,
        "qr_bdy": state.qr_bdy,
        "qi_bdy": state.qi_bdy,
        "qs_bdy": state.qs_bdy,
        "qg_bdy": state.qg_bdy,
        "Ni_bdy": state.Ni_bdy,
        "Nr_bdy": state.Nr_bdy,
    }


def _state_from_coupled_core(
    snapshot: dict[str, jax.Array],
    template: State,
    theta_offset: jax.Array,
    dt_s: float,
    *,
    base_state: BaseState | None = None,
) -> State:
    theta = jnp.asarray(snapshot["theta"]) + theta_offset
    p_pert = jnp.asarray(snapshot["p"])
    ph_pert = jnp.asarray(snapshot["ph"])
    mu_pert = jnp.asarray(snapshot["mu"])
    p_base = base_state.pb if base_state is not None else template.p_total - template.p_perturbation
    ph_base = base_state.phb if base_state is not None else template.ph_total - template.ph_perturbation
    mu_base = base_state.mub if base_state is not None else template.mu_total - template.mu_perturbation
    p_total = p_base + p_pert
    ph_total = ph_base + ph_pert
    mu_total = mu_base + mu_pert
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


def _carry_from_coupled_core(
    snapshot: dict[str, jax.Array],
    template: State,
    theta_offset: jax.Array,
    dt_s: float,
    *,
    rthraten: jax.Array | None = None,
    base_state: BaseState | None = None,
) -> OperationalCarry:
    next_state = _state_from_coupled_core(
        snapshot, template, theta_offset, float(dt_s), base_state=base_state
    )
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
        base_state=base_state,
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
    return _carry_from_coupled_core(
        snapshot,
        carry.state,
        theta_offset,
        float(namelist.dt_s),
        rthraten=carry.rthraten,
        base_state=carry.base_state,
    ).replace(census=carry.census)


class _NoahMPClock(NamedTuple):
    """Phenology clock the Noah-MP forcing assembler reads (julian / yearlen)."""

    julian: float
    yearlen: float


def _noahmp_julian_advance_enabled() -> bool:
    return os.environ.get("GPUWRF_NOAHMP_JULIAN_ADVANCE", "0") == "1"


def _noahmp_clock(namelist, clock_base, lead_seconds, *, output_time=False) -> _NoahMPClock:
    """Noah-MP ``JULIAN``/``YEARLEN`` for an LSM call at ``init + lead_seconds``.

    Default (flag unset): the run-start clock for every step, unchanged. GPUWRF_NOAHMP_JULIAN_ADVANCE=1:
    WRF's CURRENT grid%julian (surface_driver JULIAN_IN=grid%julian, first_rk_step_part1.F:624), i.e. the
    clock at the START of the step, so phenology (module_sf_noahmplsm.F:1301-1316) advances with model time.
    Dated runs read the traced CLWRF anchor (exact seconds of year + year wrap); undated runs advance the
    run-start julian within its year. Output-time recomputes (``output_time``) step back one dt: WRF's frame-t
    land fields come from the LSM call of the step that started at t - dt.
    """

    if clock_base is None:
        julian, yearlen = float(namelist.noahmp_julian), float(namelist.noahmp_yearlen)
    else:
        julian, yearlen = clock_base.noahmp_julian, clock_base.noahmp_yearlen
    if not _noahmp_julian_advance_enabled():
        return _NoahMPClock(julian=julian, yearlen=yearlen)
    from gpuwrf.physics.wrf_clwrf_ghg import wrf_julian_yearlen_at_lead  # noqa: PLC0415

    if output_time:
        lead_seconds = jnp.maximum(jnp.asarray(lead_seconds, jnp.float64) - float(namelist.dt_s), 0.0)
    ghg_clock = getattr(clock_base, "ghg_clock", None)
    if ghg_clock is not None:
        return _NoahMPClock(*wrf_julian_yearlen_at_lead(ghg_clock, lead_seconds))
    seconds = jnp.mod(jnp.asarray(julian, jnp.float64) * 86400.0 + jnp.asarray(lead_seconds, jnp.float64),
                      jnp.asarray(yearlen, jnp.float64) * 86400.0)
    advanced = (1.0 + seconds / 86400.0).astype(jnp.float32) - jnp.float32(1.0)
    return _NoahMPClock(julian=advanced.astype(jnp.float64), yearlen=jnp.asarray(yearlen, jnp.float64))


class _ClockBase(NamedTuple):
    """v0.20.0 #91: the per-run date-derived scalars threaded as TRACED inputs.

    Every value that used to be a Python-float *literal* baked into the radiation /
    phenology HLO (so a new forecast date minted a new XLA cache key and the
    persistent compile cache MISSED across dates) is bundled here and passed into
    the compiled scan as a TRACED argument -- exactly like ``start_step``. Because
    the date enters the program as a runtime input rather than a trace-time
    constant, the compiled HLO is IDENTICAL for every forecast date, so the cache
    HITS across dates with no config. The numeric values are byte-for-byte the same
    as the old host-extracted floats; only their binding time changes, so the fix
    is numerically inert (fp64_default stays bit-identical).

    Fields (all 0-D ``float64`` ``jnp`` arrays):
      ``rad_julian`` / ``rad_minute`` : WRF-style Julian day + UTC minute-of-day for
          the radiation/solar geometry (``physics_couplers._time_utc_parts``).
      ``noahmp_julian`` / ``noahmp_yearlen`` : the Noah-MP phenology greenness clock.

    Tenerife B=4 batch extension (2026-09-18 pilot): the three OPTIONAL trailing
    fields carry the genuinely per-lane Noah-MP static arrays (``tbot`` /
    ``shdfac`` / ``shdmax``) as TRACED per-lane inputs for the batched-ensemble
    vmap. The closed-over batch namelist must keep a lane-INVARIANT static aux
    (the jit treedef keys on it), so these arrays cannot stay on the namelist for
    B>1: they ride the per-lane clock base instead (same traced-argument pattern
    as #91). Defaults ``None``; the B=1 path builds a clock base without them and
    reads the namelist static exactly as before, so single-lane behaviour --
    including the bitwise-sensitive dt54 A/B pattern -- is untouched. Values are
    byte-for-byte what the per-lane static carries; only their binding site
    changes.

    v0.25 S3: ``ghg_clock`` is the CLWRF gas anchor (``None`` for grid-less or
    undated namelists).  RRTMG evaluates the gases at each call's valid time
    from these operands instead of baking init-date constants into the HLO.
    """

    rad_julian: jax.Array
    rad_minute: jax.Array
    noahmp_julian: jax.Array
    noahmp_yearlen: jax.Array
    noahmp_tbot: jax.Array | None = None
    noahmp_shdfac: jax.Array | None = None
    noahmp_shdmax: jax.Array | None = None
    ghg_clock: CLWRFGasClock | None = None
    lower_boundary: object = None


def build_clock_base(
    namelist: "OperationalNamelist",
    noahmp_static_arrays: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> _ClockBase:
    """Build the traced :class:`_ClockBase` for ``namelist`` ONCE on the host.

    Called OUTSIDE ``jax.jit`` (the host chunk loop / M9 snapshot) so the date
    scalars enter the compiled program as runtime inputs, not baked literals.
    ``noahmp_static_arrays`` is the optional ``(tbot, shdfac, shdmax)`` per-lane
    triple for the B>1 batch path (see :class:`_ClockBase`); ``None`` (the B=1
    default) leaves the three fields unset and the step reads the namelist
    static as before.
    """

    rad_julian, rad_minute = time_utc_clock_base(namelist.time_utc)
    # The CLWRF gas anchor exists exactly when RRTMG would read the gas table:
    # a real grid (metrics) and a dated run (``_rrtmg_column_inputs``).
    ghg_clock = None
    if (
        getattr(getattr(namelist, "grid", None), "metrics", None) is not None
        and getattr(namelist, "time_utc", None) is not None
    ):
        ghg_clock = CLWRFGasClock(
            *(jnp.asarray(leaf) for leaf in clwrf_gas_clock(namelist.time_utc))
        )
    return _ClockBase(
        rad_julian=rad_julian,
        rad_minute=rad_minute,
        noahmp_julian=jnp.asarray(float(namelist.noahmp_julian), dtype=jnp.float64),
        noahmp_yearlen=jnp.asarray(float(namelist.noahmp_yearlen), dtype=jnp.float64),
        noahmp_tbot=None if noahmp_static_arrays is None else noahmp_static_arrays[0],
        noahmp_shdfac=None if noahmp_static_arrays is None else noahmp_static_arrays[1],
        noahmp_shdmax=None if noahmp_static_arrays is None else noahmp_static_arrays[2],
        ghg_clock=ghg_clock,
        lower_boundary=getattr(namelist, "lower_boundary", None),
    )


def _lane_noahmp_static(namelist: "OperationalNamelist", clock_base) -> Any:
    """Per-lane Noah-MP static view for the batched-ensemble vmap (Tenerife B=4).

    In the B>1 batch path the closed-over batch namelist must keep a lane-
    INVARIANT static aux (the jit treedef keys on it), so the genuinely per-lane
    ``noahmp_static`` arrays (``tbot`` / ``shdfac`` / ``shdmax``) ride the TRACED
    per-lane clock base instead (see :class:`_ClockBase`). When those arrays are
    present this returns a shallow clone of the namelist static with them swapped
    in; every other static slot is lane-invariant by construction (asserted in
    ``nested_pipeline._load_batched_domains``). With a plain (B=1) clock base the
    namelist static is returned UNCHANGED, so single-lane behaviour -- including
    the bitwise-sensitive dt54 A/B pattern -- is byte-identical.
    """

    static = getattr(namelist, "noahmp_static", None)
    if clock_base is None or static is None:
        return static
    tbot = getattr(clock_base, "noahmp_tbot", None)
    if tbot is None:
        return static
    clone = object.__new__(type(static))
    for slot in type(static).__slots__:
        object.__setattr__(clone, slot, getattr(static, slot))
    object.__setattr__(clone, "tbot", tbot)
    shdfac = getattr(clock_base, "noahmp_shdfac", None)
    if shdfac is not None:
        object.__setattr__(clone, "shdfac", shdfac)
    shdmax = getattr(clock_base, "noahmp_shdmax", None)
    if shdmax is not None:
        object.__setattr__(clone, "shdmax", shdmax)
    return clone


def _rad_clock_base(clock_base):
    """Extract the traced :class:`RadiationClock` (solar ``julian``/``utc_minute``
    plus the CLWRF gas anchor), or ``None`` when no traced clock was threaded
    (legacy host-extraction path)."""

    if clock_base is None:
        return None
    return RadiationClock(
        clock_base.rad_julian,
        clock_base.rad_minute,
        getattr(clock_base, "ghg_clock", None),
    )


def noahmp_initial_rad(
    state: State,
    namelist: "OperationalNamelist | None" = None,
    *,
    land_state=None,
    _with_diagnostics=False,
    _kernel_call=None,
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
        if _with_diagnostics:
            raise ValueError("full radiation initialization requires a namelist")
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
        _kernel_call=_kernel_call,
        with_clear_sky=True,
        use_mp_re=int(mp_re_active(namelist)),
    )
    soldn = jnp.maximum(jnp.asarray(rad.swnorm, dtype=jnp.float64), 0.0)
    lwdn = jnp.asarray(rad.glw, dtype=jnp.float64)
    cosz = jnp.asarray(rad.coszen, dtype=jnp.float64)
    if int(namelist.ra_sw_physics) == 0:
        soldn = jnp.zeros_like(soldn)
    if int(namelist.ra_lw_physics) == 0:
        lwdn = jnp.zeros_like(lwdn)
    forcing = (soldn, lwdn, cosz)
    if _with_diagnostics:
        # WRF publishes t0 before radiation_driver runs: CLDFRA is zero then.
        # The first forecast radiation call publishes its actual input fraction.
        cloud_template = rad.cloud_fraction if rad.cloud_fraction is not None else state.qv
        rad = rad._replace(cloud_fraction=jnp.zeros_like(cloud_template))
    return (forcing, rad) if _with_diagnostics else forcing


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
    # mp=0 passive, 8 Thompson (existing couplers); 1/2/3/4/6/10/13/14/16 new scan
    # adapters; 28 aerosol-aware Thompson (v0.16 thompson_aero_adapter);
    # 97 Goddard GCE single-moment 3-ice (v0.17 goddard_adapter, gsfcgce port).
    # mp=13 SBU-YLin (v0.18-harvested single-moment 5-class, savepoint-parity vs
    # unmodified module_mp_sbu_ylin.F; coupling.scan_adapters.sbu_ylin_adapter).
    # mp=24 WSM7 + 26 WDM7 (v0.17 hail: WSM6/WDM6 + a separate precipitating qh +
    # hail_acc; WDM7 also keeps the WDM6 double-moment Nc/Nn;
    # coupling.scan_adapters.{wsm7_adapter,wdm7_adapter}).
    # mp=40 Morrison-aerosol at WRF's stand-alone point aercu_opt=0 (v0.3.4 O1,
    # coupling.scan_adapters.morrison_aero_adapter; aercu_opt>0 needs cu=11).
    # mp=18 NSSL 2-moment (v0.3.4 o1-nssl: physics.nssl2mom JAX port, CPU-oracle-qualified vs
    # proofs/v034/f2_oracles/nssl_2mom; root-domain transport of Nc/Ns/Ng/Nh/Nn, nested fail-closed).
    "mp_physics": (0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 18, 24, 26, 28, 40, 97),
    # bl=0 off, 5 MYNN (existing); 1 YSU / 7 ACM2 / 8 BouLac wired
    # (v0.6.0 jax.lax.scan rewrites); 2 MYJ wired (v0.13 traceable MYJ+Janjic pair);
    # 3 GFS wired (v0.17 jit/vmap-traceable port of phys/module_bl_gfs.F);
    # 99 MRF wired (v0.13 jit/vmap-traceable port of phys/module_bl_mrf.F).
    # 11 Shin-Hong is the v0.18 scale-aware JAX/vmap port; 12 GBM is the v0.18
    # moist prognostic-TKE JAX/vmap port. 9 CAM-UW is the v0.3.4 faithful r8 port
    # (proofs/v034/camuw_oracle; CPU-oracle-qualified, GPU-unqualified).
    "bl_pbl_physics": (0, 1, 2, 3, DEFAULT_BL_PBL_PHYSICS, 7, 8, 9, 11, 12, 99),
    # sf_sfclay=0 off, 5 MYNN-sfclay (existing); 1 revised-MM5 / 7 Pleim-Xiu wired;
    # 2 Janjic Eta wired (v0.13, mandatorily paired with bl_pbl_physics=2 MYJ).
    # 3 NCEP-GFS surface layer + 91 old-MM5 surface layer wired (v0.13 Tier-3,
    # coupling.scan_adapters.{gfs_sfclay_adapter,sfclay_old_mm5_adapter}; both write
    # the B2 kinematic flux handles, fp64 pristine-WRF oracle-validated).
    "sf_sfclay_physics": (0, 1, 2, 3, 5, 7, 91),
    # cu=0 no cumulus, 1 KF, 2 BMJ (fp64 savepoint-parity carry-threaded adapter),
    # 3 Grell-Freitas (v0.9.0 GPU-batched jit/vmap stateless adapter), 6 modified-
    # Tiedtke (v0.6.0 GPU-batched jit/vmap adapter; requires active flux-form
    # moisture advection so the scan can diagnose WRF RQVFTEN). 16 New-Tiedtke
    # (v0.23 F2: machine-precision fp64 kernel vs the WRF oracle savepoints,
    # scan-wired via coupling.scan_adapters.ntiedtke_adapter; same flux-form
    # moisture-advection requirement as cu=6). SAS 94/95/96 are reference-only /
    # not wired. 5 Grell-3D + 93 Grell-Devenyi: v0.3.4 o1-grell
    # line-faithful JAX ports (coupling.scan_adapters.grell_cumulus_adapter),
    # CPU-oracle-qualified vs pristine WRF (proofs/v034); GPU qualification pending.
    # 4 scale-aware GFS SAS (v0.3.4 o1-sas: bitwise vs the pristine WRF REAL build,
    # held-rate cadence via coupling.scalesas_adapter) -- Python-API only: WRF ARW refuses
    # cu=4 (check_a_mundo.F:671) and so does the CLI (io.namelist_binding, scheme_catalog).
    "cu_physics": (0, 1, 2, 3, 4, 5, 6, 16, 93),
    # ra_sw=0 disabled, 4 RRTMG SW (default), 1 Dudhia SW (Stephens-1984, scan-wired held-rate
    # theta tendency via dudhia_sw_theta_tendency), 2 GSFC/Chou-Suarez SW
    # (multi-band delta-Eddington, scan-wired held-rate theta tendency via
    # gsfc_sw_theta_tendency). Any other recognized SW scheme is fail-closed
    # (no GPU scan adapter). 3 CAM SW (v0.3.4 o1-camrad: physics.ra_cam/ra_cam_sw port of phys/module_ra_cam.F,
    # held-rate theta tendency via coupling.cam_radiation.cam_sw_theta_tendency; CPU-oracle-qualified).
    "ra_sw_physics": (0, 1, 2, 3, 4),
    # ra_lw=0 disabled, 4 RRTMG LW (default), 1 classic AER RRTM LW (16-band k-distribution,
    # scan-wired held-rate theta tendency via rrtm_lw_theta_tendency, JAX-traceable
    # port of phys/module_ra_rrtm.F). 31 Held-Suarez idealized radiation (COMBINED
    # LW+SW Newtonian relaxation, phys/module_ra_hs.F:HSRAD, scan-wired held-rate
    # theta tendency via held_suarez_theta_tendency; requires ra_sw_physics=0 since
    # HSRAD is the only radiative call). SW/LW are otherwise selected independently.
    # 3 CAM LW (v0.3.4 o1-camrad: physics.ra_cam port of phys/module_ra_cam.F, held-rate theta tendency via
    # coupling.cam_radiation.cam_lw_theta_tendency; CPU-oracle-qualified, GPU/coupled qualification pending).
    "ra_lw_physics": (0, 1, 3, 4, 31),
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
    "cu_physics=94": "2015 GFS SAS / HWRF has v0.17 fp64 pristine-WRF savepoints, but the shared JAX endpoint is RED vs oracle (proofs/v017/sas_family_parity.json); operational GPU scan wiring is blocked",
    "cu_physics=95": "Previous GFS SAS / HWRF OSAS has v0.17 fp64 pristine-WRF savepoints, but the shared JAX endpoint is RED vs oracle (proofs/v017/sas_family_parity.json); operational GPU scan wiring is blocked",
    "cu_physics=96": "Previous new GFS SAS / YSU NSAS has v0.17 fp64 pristine-WRF savepoints, but the shared JAX endpoint is RED vs oracle (proofs/v017/sas_family_parity.json); operational GPU scan wiring is blocked",
    # v0.13/v0.18 Tier-3 cumulus: KSAS(14), Grell-3D(5), and
    # Grell-Devenyi(93) have real pristine-WRF oracle artifacts. Their
    # traceable JAX column kernels remain carry-overs, so they fail-close here.
    "cu_physics=14": "KIM-SAS has a single-column fp64 pristine-WRF oracle staged (proofs/v013); traceable JAX column kernel is a Tier-3 carry-over",
    # cu=93 (Grell-Devenyi) / cu=99 (previous Kain-Fritsch) are v0.17 RED/reference-only.
    "cu_physics=99": "previous Kain-Fritsch is accepted for v0.17 oracle work, but the available GPU endpoint is the KF-eta family, not a parity-proven module_cu_kf.F:KFCPS port",
    "sf_surface_physics=2": "Noah-classic requires explicit noahclassic_static + noahclassic_land bundles (WRF REDPRM + 4-layer carry)",
    "sf_surface_physics=1": "thermal-diffusion slab LSM requires an explicit slab_static (SlabStaticBundle: soil ZS/DZS + TMN/THC/EMISS/SNOWC) so the scan can advance the 5-layer TSLB land carry from GSW/GLW radiation forcing",
    "sf_surface_physics=7": "Pleim-Xiu LSM requires an explicit px_static (PleimXiuStaticBundle: ISBA soil constants + vegetation/surface fields) so the scan can advance the 2-layer ISBA land carry from GSW/GLW radiation forcing",
    # v0.17 Tier-3 land surface: RUC(3)/SSiB(8) have single-column fp64 pristine-WRF
    # oracles staged (proofs/v017/oracle/{ruclsm,ssib}, built from the unmodified
    # LSMRUC/SSIB drivers); their faithful traceable JAX column kernels are documented
    # carry-overs (the ~7.5k-LOC RUC soil/snow solver and the ~6.6k-LOC SSiB SiB
    # canopy/soil/snow solver), so both fail-close here.
    "sf_surface_physics=3": "RUC LSM (physics/ruclsm.py, CPU-oracle-qualified vs proofs/v034/oracle/ruclsm) requires an explicit ruc_static (RucStaticBundle: RucConfig + WRF RUC tables + IVGTYP/ISLTYP/XLAND/XICE/TMN/SHDMIN/SHDMAX/ALBBCK) and a seeded ruc_land (RucLandState on the wrfinput RUC soil levels), sf_sfclay_physics=5 (MYNN-SL supplies FLHC/FLQC) and use_noahmp=False; sea-ice/lake points fail closed",
    "sf_surface_physics=8": "SSiB SiB biophysical canopy/soil/snow LSM has a single-column fp64 pristine-WRF oracle staged (proofs/v017/oracle/ssib, the unmodified SSIB driver); traceable JAX column kernel is a Tier-3 carry-over (~6.6k-LOC coupled SiB solver)",
    "sf_urban_physics=1": "single-layer UCM is recognized but no urban canopy state carry, pristine-WRF oracle, or faithful JAX kernel is wired into the operational scan",
    "sf_urban_physics=2": "BEP urban canopy (phys/module_sf_bep.F:BEP) needs the Registry bepscheme carry, urban-map/static tables, pristine-WRF oracle, and faithful JAX kernel before scan wiring",
    "sf_urban_physics=3": "BEP+BEM urban canopy/building-energy model (phys/module_sf_bep.F:BEP + module_sf_bem.F:BEM) needs the Registry bep_bemscheme carry, urban/BEM static tables, pristine-WRF oracle, and faithful JAX kernel before scan wiring",
    "sf_lake_physics=1": "WRF lake model (phys/module_sf_lake.F:Lake/LakeMain/lakeini) needs lake-depth/category initialization, snow/ice/water carry, pristine-WRF oracle, and faithful JAX kernel before scan wiring",
    # ra_sw=1 (Dudhia), ra_sw=2 (GSFC/Chou-Suarez) and ra_sw=4 (RRTMG) are
    # scan-wired; v0.18 recognizes ra_sw=3/5/7/99 for real-WRF oracle/parity work
    # only and fail-closes them here.
    "ra_sw_physics=5": "New Goddard shortwave has a v0.18 exact-driver real-WRF oracle (module_radiation_driver.F -> module_ra_goddard.F:goddardrad, proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json) but no faithful JAX column kernel or operational scan wiring",
    "ra_sw_physics=7": "FLG/UCLA shortwave has a v0.18 exact-driver real-WRF oracle (module_radiation_driver.F -> module_ra_flg.F:RAD_FLG, proofs/v018/savepoints/ra_tail_wrf/ra7_wrf_real.json) but no faithful JAX column kernel or operational scan wiring",
    "ra_sw_physics=99": "GFDL-Eta shortwave has a v0.18 exact-driver real-WRF oracle (module_radiation_driver.F -> module_ra_gfdleta.F:ETARA, proofs/v018/savepoints/ra_tail_wrf/ra99_wrf_real.json) but no faithful JAX column kernel or operational scan wiring",
    # ra_lw=5 (GSFC/Goddard NUWRF LW) is v0.13 Tier-3 reference-only: a fp64
    # single-column pristine-WRF oracle is staged (module_ra_goddard.F:lwrad,
    # proofs/v013/oracle/radiation_lw); its traceable JAX column kernel is a
    # documented carry-over (the combined NUWRF SW+LW module is ~12.5k LOC), so it
    # fail-closes here. ra_lw=4 (RRTMG), 1 (classic RRTM), and 31 (Held-Suarez with
    # ra_sw=0) remain the operational LW.
    "ra_lw_physics=5": "GSFC/Goddard NUWRF longwave has a v0.13 single-column fp64 pristine-WRF oracle and a v0.18 exact-driver paired real-WRF oracle (module_radiation_driver.F -> module_ra_goddard.F:goddardrad, proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json); no faithful JAX column kernel or operational scan wiring",
    "ra_lw_physics=7": "FLG/UCLA longwave has a v0.18 exact-driver real-WRF oracle (module_radiation_driver.F -> module_ra_flg.F:RAD_FLG, proofs/v018/savepoints/ra_tail_wrf/ra7_wrf_real.json) but no faithful JAX column kernel or operational scan wiring",
    "ra_lw_physics=99": "GFDL-Eta longwave has a v0.18 exact-driver real-WRF oracle (module_radiation_driver.F -> module_ra_gfdleta.F:ETARA, proofs/v018/savepoints/ra_tail_wrf/ra99_wrf_real.json) but no faithful JAX column kernel or operational scan wiring",
}


def _explicit_noahclassic(namelist: OperationalNamelist) -> bool:
    explicit_land = getattr(namelist, "sf_surface_physics", None)
    return explicit_land is not None and int(explicit_land) == 2


def _explicit_slab(namelist: OperationalNamelist) -> bool:
    explicit_land = getattr(namelist, "sf_surface_physics", None)
    return explicit_land is not None and int(explicit_land) == 1


def _explicit_pleim_xiu(namelist: OperationalNamelist) -> bool:
    explicit_land = getattr(namelist, "sf_surface_physics", None)
    return explicit_land is not None and int(explicit_land) == 7


def _explicit_ruc(namelist: OperationalNamelist) -> bool:
    explicit_land = getattr(namelist, "sf_surface_physics", None)
    return explicit_land is not None and int(explicit_land) == 3


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
    urban_opt = int(getattr(namelist, "sf_urban_physics", 0))
    if urban_opt != 0:
        tag = f"sf_urban_physics={urban_opt}"
        reason = _SCAN_UNWIRED_REASON.get(tag, "urban canopy physics is not operationally wired")
        not_wired.append(f"{tag} ({reason})")
    lake_opt = int(getattr(namelist, "sf_lake_physics", 0))
    if lake_opt != 0:
        tag = f"sf_lake_physics={lake_opt}"
        reason = _SCAN_UNWIRED_REASON.get(tag, "lake model is not operationally wired")
        not_wired.append(f"{tag} ({reason})")
    tiedtke_lacks_rqvften = (
        int(getattr(namelist, "cu_physics", 0)) in (6, 16)
        and (
            not bool(namelist.use_flux_advection)
            or int(getattr(namelist, "moist_adv_opt", 0)) == 0
        )
    )
    grell_cu = int(getattr(namelist, "cu_physics", 0)) in (5, 93)
    if grell_cu and int(getattr(namelist, "cumulus_cadence_steps", 1)) > 1:
        not_wired.append(
            f"cu_physics={int(getattr(namelist, 'cu_physics', 0))} with cudt>0 "
            "(Grell-3D/Grell-Devenyi held R*CUTEN between cumulus calls is not wired; "
            "set cudt=0 so the scheme runs every step like WRF's stepcu=1)"
        )
    if tiedtke_lacks_rqvften:
        not_wired.append(
            f"cu_physics={int(getattr(namelist, 'cu_physics', 0))} "
            "(the Tiedtke-family schemes require use_flux_advection=True and "
            "moist_adv_opt=1/2 so the operational scan can diagnose WRF RQVFTEN "
            "moisture-convergence forcing)"
        )
    # Land surface: the scan threads Noah-MP (use_noahmp=True), explicit
    # Noah-classic (sf_surface_physics=2 + WRF-derived land/static bundle), or the
    # legacy bulk surface path. NOTE the dispatcher maps the legacy
    # ``use_noahmp=False`` toggle to land option 2, but in THIS scan that still
    # means the bulk path unless sf_surface_physics is explicitly pinned to 2.
    land_opt = suite.land_surface.option
    if land_opt == 4 and not bool(namelist.use_noahmp):
        not_wired.append("sf_surface_physics=4 (set use_noahmp=True to thread Noah-MP)")
    # slab=1 (thermal-diffusion 5-layer LSM) is v0.17 scan-wired
    # (coupling.slab_surface_hook.slab_surface_step): it advances the 5-layer
    # TSLB land carry from GSW/GLW radiation forcing + the TMN/THC/EMISS static
    # bundle. Like Noah-classic it requires an explicit WRF-derived
    # SlabStaticBundle; fail closed if absent rather than deriving an unvalidated
    # land state from the resident State.
    if _explicit_slab(namelist):
        if getattr(namelist, "slab_static", None) is None:
            not_wired.append(f"sf_surface_physics=1 ({_SCAN_UNWIRED_REASON['sf_surface_physics=1']})")
    # px=7 (Pleim-Xiu 2-layer ISBA LSM) is v0.17 scan-wired
    # (coupling.pleim_xiu_surface_hook.pleim_xiu_surface_step); like slab/Noah-classic
    # it requires an explicit WRF-derived PleimXiuStaticBundle, else fail closed.
    if _explicit_pleim_xiu(namelist):
        if getattr(namelist, "px_static", None) is None:
            not_wired.append(f"sf_surface_physics=7 ({_SCAN_UNWIRED_REASON['sf_surface_physics=7']})")
    if _explicit_noahclassic(namelist):
        if getattr(namelist, "noahclassic_static", None) is None or getattr(namelist, "noahclassic_land", None) is None:
            not_wired.append(f"sf_surface_physics=2 ({_SCAN_UNWIRED_REASON['sf_surface_physics=2']})")
    # ruc=3 (RUC) / ssib=8 (SSiB) are v0.17 REFERENCE-ONLY: fp64 pristine-WRF
    # single-column oracle staged, faithful JAX column kernel is a carry-over -- always
    # fail-closed in the operational scan (never silently substituted by another LSM).
    if land_opt == 8:
        not_wired.append(f"sf_surface_physics={land_opt} ({_SCAN_UNWIRED_REASON[f'sf_surface_physics={land_opt}']})")
    # ruc=3 (v0.3.4, lane o1-ruc): opt-in RUC LSM via coupling.ruc_surface_hook; needs the
    # explicit WRF-derived bundles, the MYNN surface layer (exact FLHC/FLQC) and no
    # Noah-MP; columns the port cannot run (sea ice, WRF nroot OOB) fail closed.
    if land_opt == 3:
        ruc_ok = (
            getattr(namelist, "ruc_static", None) is not None
            and getattr(namelist, "ruc_land", None) is not None
            and int(getattr(namelist, "sf_sfclay_physics", 5)) == 5
            and not bool(getattr(namelist, "use_noahmp", False))
        )
        # The previous-step precipitation (RAINBL/RAINNCV/SNOWNCV/GRAUPELNCV/SR) reaches the hook only
        # through the Thompson precipitation carry; any other MP would silently feed zeros (rv-ruc #4).
        # myj=.true. (MYJ PBL) selects LSMRUC's chs-based qkms/tkms path that the hook does not wire (A5).
        if ruc_ok and int(getattr(namelist, "mp_physics", DEFAULT_MP_PHYSICS)) != DEFAULT_MP_PHYSICS:
            not_wired.append(
                f"sf_surface_physics=3 (RUC needs the previous-step precipitation carry, produced only by "
                f"mp_physics={DEFAULT_MP_PHYSICS} Thompson; mp_physics={int(namelist.mp_physics)} would feed zero rain/snow)")
            ruc_ok = None
        if ruc_ok and int(getattr(namelist, "bl_pbl_physics", 5)) == 2:
            not_wired.append("sf_surface_physics=3 (RUC with MYJ PBL needs LSMRUC's myj=.true. chs path; not wired)")
            ruc_ok = None
        if ruc_ok:
            from gpuwrf.coupling.ruc_surface_hook import ruc_unsupported_mask

            bad = ruc_unsupported_mask(namelist.ruc_land, namelist.ruc_static)
            if bool(bad.any()):
                not_wired.append(
                    f"sf_surface_physics=3 ({int(bad.sum())} land columns need RUC sea-ice (SICE/"
                    "SNOWSEAICE) or hit the WRF zshalf(nroot+1) out-of-bounds read; not ported)")
        elif ruc_ok is not None:
            not_wired.append(f"sf_surface_physics=3 ({_SCAN_UNWIRED_REASON['sf_surface_physics=3']})")
    # Held-Suarez (ra_lw=31) is a COMBINED idealized LW+SW Newtonian relaxation:
    # WRF's radiation driver makes the single HSRAD call and NO separate shortwave
    # call. Pairing it with a real SW scheme would double-count radiative heating,
    # so fail closed unless ra_sw_physics=0 (HSRAD is the sole radiative source).
    if int(getattr(namelist, "ra_lw_physics", 0)) == 31 and int(getattr(namelist, "ra_sw_physics", 0)) != 0:
        not_wired.append(
            f"ra_sw_physics={int(namelist.ra_sw_physics)} (Held-Suarez ra_lw_physics=31 is "
            "a COMBINED idealized LW+SW scheme -- WRF makes no separate shortwave call -- "
            "so it must run with ra_sw_physics=0 to avoid double-counting radiative heating)"
        )
    if not_wired:
        raise UnsupportedSchemeSelection(
            "operational scan supports the v0.2.0 suite + the v0.6.0/v0.13/v0.17 scan-wired "
            "schemes (mp_physics in {0,1,2,3,4,6,8,10,13,14,16,18,24,26,28,40,97}, bl_pbl_physics in {0,1,2,3,5,7,8,9,11,12,99}, "
            "sf_sfclay_physics in {0,1,2,3,5,7,91}, cu_physics in {0,1,2,3,4,5,6,16,93}, Noah-MP via "
            "use_noahmp, explicit Noah-classic via sf_surface_physics=2 plus "
            "noahclassic_static/noahclassic_land, ra_sw_physics in {0,1,2,4}, "
            "ra_lw_physics in {0,1,4,31}). The following selected schemes "
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

    from gpuwrf.diagnostics.census import enabled as census_enabled, initial_census, count_work

    state = state.ensure_conditional_leaves(mp_physics=int(namelist.mp_physics), use_mp_re=int(namelist.use_mp_re),
                                             ra_lw_physics=int(namelist.ra_lw_physics),
                                             ra_sw_physics=int(namelist.ra_sw_physics))
    native_real_carry = os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1"
    if native_real_carry:
        if (os.environ.get("GPUWRF_DYN_FP32", "0") != "1"
                or os.environ.get("GPUWRF_DYN_RK_FP32", "0") != "1"):
            raise ValueError("GPUWRF_DYN_CARRY_FP32 requires native acoustic and RK flags")
        if is_mixed_perturb_fp32_mode(namelist.acoustic_precision_mode):
            raise ValueError("native REAL carry requires fp64_default acoustic storage mode")
    mixed_precision = is_mixed_perturb_fp32_mode(namelist.acoustic_precision_mode)
    base_state = None
    if mixed_precision:
        fp64_state = _enforce_operational_precision(state, force_fp64=True)
        base_state = _base_state_from_totals(fp64_state)
        enforced = _apply_mixed_perturb_fp32_storage(fp64_state, base_state)
    else:
        enforced = _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
        if os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1":
            from gpuwrf.kernels.dyn_rk_fp32 import real_base

            # WRF PB/PHB/MUB are immutable REAL fields. Capture the resolved
            # initial base before native RK rounds totals and perturbations
            # separately; recovering their difference each stage causes drift.
            base_state = real_base(_base_state_from_totals(enforced))
    if native_real_carry:
        from gpuwrf.kernels.dyn_carry_fp32 import real_state
        # BC40 captured the resolved base from wide totals above.
        enforced = real_state(enforced)
    if int(namelist.gwd_opt) == 1 and namelist.gwdo_statics is not None:
        # Seed the optional history leaves before the first scan so the GWDO
        # call cannot change the carry pytree. WRF stores these diagnostics REAL.
        enforced = enforced.replace(**{
            name: jnp.zeros_like(enforced.theta[0] if name in GWDO_SURFACE_DIAGNOSTIC_LEAVES
                                 else enforced.theta, dtype=jnp.float32)
            for name in GWDO_DIAGNOSTIC_LEAVES if getattr(enforced, name) is None
        })
    if int(namelist.mp_physics) == 28:
        # v0.16 aerosol-aware Thompson: cold-start nwfa/nifa from the WRF
        # thompson_init climatological profiles when the inputs carry no
        # aerosol state (use_aero_icbc=.false. self-init; init-time only,
        # no-op on restart/forced states).
        enforced = thompson_aero_coldstart_init(enforced, namelist.grid)
    cumulus_carry = None
    # Only the STATEFUL cumulus adapters need a persistent carry: KF (1) threads
    # (w0avg, nca); BMJ (2) threads CLDEFI. The stateless GPU-batched adapter
    # (Tiedtke, 6) keeps cumulus_carry None.
    cu_opt = int(namelist.cu_physics)
    if cu_opt == 1:
        cumulus_carry = initial_kf_carry(enforced)
    elif cu_opt == 2:
        cumulus_carry = initial_bmj_carry(enforced)
    init_radiation_calls = 0
    noahclassic_land = None
    noahclassic_rad = None
    if _explicit_noahclassic(namelist):
        init_radiation_calls += int(getattr(namelist, "noahclassic_rad", None) is None)
        noahclassic_land = namelist.noahclassic_land
        noahclassic_rad = (
            namelist.noahclassic_rad
            if getattr(namelist, "noahclassic_rad", None) is not None
            else NoahClassicRadiation(*noahmp_initial_rad(enforced, namelist))
        )
    slab_land = None
    slab_rad = None
    if _explicit_slab(namelist):
        # Seed the 5-layer TSLB land carry from the supplied static bundle (or a
        # TSK->TMN cold-start) and the held GSW/GLW down-radiation, exactly as the
        # Noah-classic seam seeds its 4-layer carry + held radiation.
        slab_land = (
            namelist.slab_land
            if getattr(namelist, "slab_land", None) is not None
            else initial_slab_land(enforced, namelist.slab_static)
        )
        init_radiation_calls += 1
        soldn, lwdn, _cosz = noahmp_initial_rad(enforced, namelist)
        slab_rad = (
            namelist.slab_rad
            if getattr(namelist, "slab_rad", None) is not None
            else SlabRadiation(soldn, lwdn)
        )
    px_land = None
    px_rad = None
    if _explicit_pleim_xiu(namelist):
        # Seed the 2-layer ISBA land carry (or a TSK/soil-moisture cold-start) and
        # the held GSW/GLW radiation, mirroring the slab/Noah-classic seam.
        px_land = (
            namelist.px_land
            if getattr(namelist, "px_land", None) is not None
            else initial_pleim_xiu_land(enforced, namelist.px_static)
        )
        init_radiation_calls += 1
        px_soldn, px_lwdn, _px_cosz = noahmp_initial_rad(enforced, namelist)
        px_rad = (
            namelist.px_rad
            if getattr(namelist, "px_rad", None) is not None
            else PleimXiuRadiation(px_soldn, px_lwdn)
        )
    ruc_land = None
    ruc_rad = None
    if _explicit_ruc(namelist):
        # RUC (sf_surface_physics=3): the seeded RucLandState (RUCLSMINIT + LSMRUC
        # ktau=1 block, coupling.ruc_surface_hook.initial_ruc_land) rides the namelist;
        # the legacy held radiation is the downward SW/LW like the slab/PX seams.
        from gpuwrf.coupling.ruc_surface_hook import RucRadiation

        ruc_land = namelist.ruc_land
        init_radiation_calls += 1
        ruc_soldn, ruc_lwdn, _ruc_cosz = noahmp_initial_rad(enforced, namelist)
        ruc_rad = (
            namelist.ruc_rad
            if getattr(namelist, "ruc_rad", None) is not None
            else RucRadiation(ruc_soldn, ruc_lwdn)
        )
    # Noah-MP: the production daily/nested pipelines seed noahmp_land + noahmp_rad
    # via carry.replace AFTER this call. The generic single-domain operational path
    # (coverage gate) has no post-replace seam, so when an explicit noahmp_land
    # bundle is supplied on the namelist, seed it + the CONCRETE held radiation
    # 3-tuple HERE -- carry.noahmp_rad must be a 3-tuple, never None
    # (_refresh_noahmp_rad returns it verbatim on a held step and the scan body does
    # _NoahMPRadiation(*carry.noahmp_rad), which crashed on None). Left None, the
    # pipelines' post-replace still owns the seeding (append-only / non-breaking).
    noahmp_land = None
    noahmp_rad = None
    initial_radiation_diagnostics = None
    if bool(namelist.use_noahmp) and getattr(namelist, "noahmp_land", None) is not None:
        noahmp_land = namelist.noahmp_land
        init_radiation_calls += 1
        noahmp_rad, initial_radiation_diagnostics = noahmp_initial_rad(
            enforced, namelist, land_state=noahmp_land, _with_diagnostics=True
        )
    result = initial_operational_carry(
        enforced,
        cumulus_carry=cumulus_carry,
        noahclassic_land=noahclassic_land,
        noahclassic_rad=noahclassic_rad,
        slab_land=slab_land,
        slab_rad=slab_rad,
        px_land=px_land,
        px_rad=px_rad,
        noahmp_land=noahmp_land,
        noahmp_rad=noahmp_rad,
        base_state=base_state,
        ruc_land=ruc_land,
        ruc_rad=ruc_rad,
    )
    if cu_opt == 1:
        from gpuwrf.physics.cumulus_kf import kf_real_enabled

        # Held KF R*CUTEN/PRATEC: WRF REAL on the native KF path (BP55, the rates are
        # produced REAL there); otherwise the historical seeds.
        kf_real = kf_real_enabled()
        result = result.replace(cumulus_tendencies=(
            *(jnp.zeros_like(enforced.theta, dtype=jnp.float32) if kf_real
              else jnp.zeros_like(enforced.theta, dtype=jnp.float64)
              if native_real_carry else jnp.zeros_like(enforced.theta) for _ in range(6)),
            jnp.zeros_like(enforced.t_skin, dtype=jnp.float32) if kf_real else jnp.zeros_like(enforced.t_skin),
        ))
    elif cu_opt == 4:
        # Scale-aware GFS SAS: held WRF REAL R*CUTEN/PRATEC in the KF carry layout.
        from gpuwrf.coupling.scalesas_adapter import initial_scalesas_tendencies
        result = result.replace(cumulus_tendencies=initial_scalesas_tendencies(enforced))
    if int(namelist.ra_sw_physics) == 4 and int(namelist.ra_lw_physics) == 4:
        # Flux diagnostics follow the retained surface/land interface (fp64),
        # while RTHRATEN follows the prognostic theta dtype.
        zero = jnp.zeros(enforced.t_skin.shape, dtype=jnp.float64)
        # The loader replaces this placeholder with its existing init solve.
        # The first 15 fields are all-sky; optional clear-sky fields stay None.
        fields = [zero for _ in range(15)]
        fields[10] = jnp.zeros(enforced.t_skin.shape, dtype=jnp.int32)
        diagnostics = initial_radiation_diagnostics or RRTMGRadiationDiagnostics(
            *fields, *(zero for _ in range(8)))
        from gpuwrf.coupling.physics_couplers import _rrtmg_real_enabled
        from gpuwrf.physics.mynn_sgs_cloud import sgs_cloud_enabled
        cloud_dtype = (enforced.t_skin.dtype if _rrtmg_real_enabled() else
                       jnp.result_type(*(getattr(enforced, name).dtype for name in
                           (("qc", "qi", "qs", "qg", "cldfra_bl") if sgs_cloud_enabled()
                            else ("qc", "qi", "qs", "qg")))))
        result = result.replace(radiation_diagnostics=diagnostics._replace(
            cloud_fraction=jnp.zeros(enforced.qv.shape, dtype=cloud_dtype)))
    if native_real_carry:
        from gpuwrf.kernels.dyn_carry_fp32 import real_carry, real_scratch
        result = real_carry(real_scratch(result))
    if census_enabled():
        result = result.replace(census=count_work(
            initial_census(), "radiation_init_calls", init_radiation_calls
        ))
    if _h_diabatic_pair_enabled(namelist):
        # WRF starts h_diabatic at zero (no microphysics call yet).
        result = result.replace(h_diabatic=jnp.zeros_like(enforced.theta))
    if int(namelist.bl_pbl_physics) == 9:
        # CAM-UW KVM3D/KVH3D/TAURES carry; zeros == WRF itimestep=1 initialisation.
        from gpuwrf.physics.bl_camuw import initial_camuw_carry
        result = result.replace(camuw_pbl=initial_camuw_carry(enforced))
    if int(namelist.ra_lw_physics) == 3:
        from gpuwrf.coupling.cam_radiation import initial_cam_held
        result = result.replace(cam_abs=initial_cam_held(enforced))
    from gpuwrf.runtime.history_accumulators import full_history_enabled, seed_history
    if bool(namelist.use_noahmp) and full_history_enabled():
        land_history, energy_accumulators = seed_history(enforced)
        result = result.replace(land_history=land_history, energy_accumulators=energy_accumulators)
    if (bool(namelist.use_noahmp) or _explicit_ruc(namelist)) and bool(namelist.run_physics) and int(namelist.mp_physics) == DEFAULT_MP_PHYSICS:
        from gpuwrf.runtime.noahmp_precipitation import seed_precipitation
        result = result.replace(noahmp_precipitation=seed_precipitation(enforced))
    return result


def _operational_device():
    """Return the device used to commit operational host-loop carries."""

    devices = jax.devices()
    for device in devices:
        if device.platform == "gpu":
            return device
    return devices[0]


def _assert_nonzero_initial_mu_total(state: State) -> None:
    """Fail loudly if real operational initialization produced zero dry mass.

    The public forecast wrapper is sometimes itself lowered by HLO proof tools.
    In that tracing context the host value is intentionally unavailable; the
    concrete runtime entry still performs the one-time check before forecast
    execution.
    """

    try:
        max_abs = float(jax.device_get(jnp.max(jnp.abs(jnp.asarray(state.mu_total)))))
    except jax.errors.ConcretizationTypeError:
        return
    if not max_abs > 0.0:
        raise ValueError(
            "operational initialization produced zero mu_total; "
            "State.replace must seed authoritative total dry mass before forecast entry"
        )


def _operational_scan_state(state: State, namelist: OperationalNamelist) -> State:
    """Materialize the concrete scan carry tail outside compiled timestep graphs."""

    # The public State contract keeps inactive hail/aerosol leaves absent for
    # storage and restart (#37).  Inside the compiled operational scan, however,
    # a concrete tail preserves the pre-v0.18 carry layout and avoids a slower
    # default mp=8 GPU program shape.  Do it at the concrete forecast entry so
    # XLA does not emit the one-time zeros/materialization work in every lowered
    # forecast executable.
    return state.ensure_conditional_leaves(
        mp_physics=int(namelist.mp_physics),
        include_all_conditional=True,
        use_mp_re=int(namelist.use_mp_re),
        ra_lw_physics=int(namelist.ra_lw_physics),
        ra_sw_physics=int(namelist.ra_sw_physics),
    )


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

    _assert_nonzero_initial_mu_total(state)
    state = _operational_scan_state(state, namelist)
    return _commit_to_operational_device(_initial_carry_for_run(state, namelist))


class _NoahMPRadiation(NamedTuple):
    """Held surface-radiation forcing into Noah-MP (the coupler reads soldn/lwdn/cosz)."""

    soldn: jax.Array
    lwdn: jax.Array
    cosz: jax.Array


def _refresh_rrtmg_driver(carry, namelist, lead_seconds, run_radiation, clock_base):
    """One WRF driver solve, holding heating and all surface/TOA flux slices."""
    from gpuwrf.diagnostics.census import count_work
    interval = float(namelist.radiation_interval_s)
    if interval == 0.0:
        interval = float(namelist.dt_s) * int(namelist.radiation_cadence_steps)

    # nesting/nest_o3.py (GPUWRF_NEST_O3_FROM_PARENT): with a held o3rad leaf a nest (live-child boundary,
    # force_geopotential False) radiates with the parent's force-down field, while the root refreshes its held
    # o3rad from the CAM climatology at its own radiation call (WRF o3input=2 runs on id 1 only).
    held_o3 = getattr(carry, "o3rad", None)
    nest_o3 = held_o3 is not None and not bool(getattr(namelist.boundary_config, "force_geopotential", True))

    def refresh(_unused):
        o3_kwargs, new_o3 = {}, held_o3
        if held_o3 is not None:
            if nest_o3:
                o3_columns = jnp.moveaxis(held_o3, 0, -1)
            else:
                o3_columns = rrtmg_ozone_columns(
                    carry.state, namelist.grid, time_utc=namelist.time_utc, lead_seconds=lead_seconds,
                    clock_base=_rad_clock_base(clock_base), radiation_static=namelist.radiation_static,
                    land_state=carry.noahmp_land,
                )
                if o3_columns is not None:
                    new_o3 = jnp.moveaxis(o3_columns, -1, 0).astype(held_o3.dtype)
            if o3_columns is not None:
                o3_kwargs["ozone_vmr_override"] = o3_columns
        rate, diag = rrtmg_theta_tendency(
            carry.state, namelist.grid, time_utc=namelist.time_utc,
            lead_seconds=lead_seconds,
            # WRF offsets COSZEN only; the GHG/ozone clocks use call time.
            solar_lead_seconds=lead_seconds + 0.5 * interval,
            clock_base=_rad_clock_base(clock_base),
            radiation_static=namelist.radiation_static,
            topo_shading=int(namelist.topo_shading), slope_rad=int(namelist.slope_rad),
            shadow_length_m=float(namelist.topo_shadow_length_m),
            land_state=carry.noahmp_land, _with_diagnostics=True, with_clear_sky=True,
            use_mp_re=int(mp_re_active(namelist)),
            **o3_kwargs,
        )

        from gpuwrf.kernels.dyn_carry_fp32 import like, real_all_enabled
        if real_all_enabled():  # E65: the refreshed slices take the held REAL carry dtypes
            rate, diag = like((rate, diag), (carry.rthraten, carry.radiation_diagnostics))
        out = (rate, diag, count_work(carry.census, "radiation_tendency_calls"))
        return out if held_o3 is None else out + (new_o3,)

    held = (carry.rthraten, carry.radiation_diagnostics, carry.census)
    if held_o3 is not None:
        held = held + (held_o3,)
    if isinstance(run_radiation, bool):
        refreshed = refresh(None) if run_radiation else held
    else:
        refreshed = jax.lax.cond(run_radiation, refresh, lambda _u: held, None)
    if held_o3 is None:
        rate, diag, census = refreshed
        return carry.replace(rthraten=rate, radiation_diagnostics=diag, census=census)
    rate, diag, census, o3rad = refreshed
    return carry.replace(rthraten=rate, radiation_diagnostics=diag, census=census, o3rad=o3rad)


def _refresh_noahmp_rad(state, namelist, lead_seconds, run_radiation, held_rad, *, land_state=None, clock_base=None, census=None, held_diagnostics=None, cam_abs=None):
    """Legacy surface-only refresh; native RRTMG4/4 uses _refresh_rrtmg_driver.

    Refresh the HELD Noah-MP surface radiation (SOLDN/LWDN/COSZ) at the radiation
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
    (``step_index %% cadence == 0`` in that legacy convention), and history output lands on a refresh boundary
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

    from gpuwrf.diagnostics.census import count_work

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
            clock_base=_rad_clock_base(clock_base),
            radiation_static=namelist.radiation_static,
            topo_shading=int(namelist.topo_shading),
            slope_rad=int(namelist.slope_rad),
            shadow_length_m=float(namelist.topo_shadow_length_m),
            land_state=land_state,
            use_mp_re=int(mp_re_active(namelist)),
        )
        soldn = jnp.maximum(jnp.asarray(rad.swnorm, dtype=jnp.float64), 0.0)
        lwdn = jnp.asarray(rad.glw, dtype=jnp.float64)
        cosz = jnp.asarray(rad.coszen, dtype=jnp.float64)
        if int(namelist.ra_sw_physics) == 0:
            soldn = jnp.zeros_like(soldn)
        if int(namelist.ra_lw_physics) == 0:
            lwdn = jnp.zeros_like(lwdn)
        if 3 in (int(namelist.ra_lw_physics), int(namelist.ra_sw_physics)):
            # CAM (o1-camrad): the land surface sees the selected CAM scheme's GLW / SWDOWN, as in WRF.
            from gpuwrf.coupling.cam_radiation import cam_doabsems, cam_surface_forcing

            soldn, lwdn = cam_surface_forcing(
                state, namelist.grid, soldn, lwdn, ra_lw_physics=int(namelist.ra_lw_physics),
                ra_sw_physics=int(namelist.ra_sw_physics), time_utc=namelist.time_utc, lead_seconds=rad_lead_seconds,
                clock_base=_rad_clock_base(clock_base), radiation_static=namelist.radiation_static, land_state=land_state,
                held=cam_abs, doabsems=cam_doabsems(lead_seconds, namelist.dt_s, namelist.radiation_cadence_steps,
                                                    namelist.cam_abs_freq_s, held=cam_abs))
        forcing = (soldn, lwdn, cosz)
        result = forcing if held_diagnostics is None else (forcing, rad)
        from gpuwrf.kernels.dyn_carry_fp32 import like, real_all_enabled
        if real_all_enabled() and held_rad is not None:  # held REAL radiation (E65)
            result = like(result, held_rad if held_diagnostics is None else (held_rad, held_diagnostics))
        return result

    # ``held_rad`` is always a concrete 3-tuple inside the scan (seeded at carry
    # construction by ``noahmp_initial_rad``), so the carry pytree structure is
    # stable across scan iterations -- never None here.
    held_result = held_rad if held_diagnostics is None else (held_rad, held_diagnostics)
    if census is not None:
        def counted_recompute(_unused):
            return _recompute(None), count_work(census, "radiation_surface_calls")

        def held(_unused):
            return held_result, census

        if isinstance(run_radiation, bool):
            return counted_recompute(None) if run_radiation else held(None)
        return jax.lax.cond(run_radiation, counted_recompute, held, None)
    if isinstance(run_radiation, bool):
        return _recompute(None) if run_radiation else held_result
    return jax.lax.cond(run_radiation, _recompute, lambda _u: held_result, None)


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


def _apply_data_assimilation_forcing(
    reference: State,
    physics_state: State,
    dry: DryPhysicsTendencies,
    namelist: OperationalNamelist,
    lead_seconds,
) -> tuple[State, DryPhysicsTendencies, bool]:
    """Add v0.22 G1 FDDA tendencies to the WRF RK tendency lanes.

    WRF grid/spectral FDDA computes target-minus-state rates in ``fddagd`` /
    ``spectral_nudging`` and then feeds u/v/theta/ph/mu through the same
    ``*_tendf`` dry-tendency merge as the other non-timesplit physics sources.
    The current port has no moist ``qv_tendf`` lane, so qv nudging is applied as a
    non-dry state increment and later transferred by
    ``_apply_physics_non_dry_updates``.
    """

    if namelist.data_assimilation is None:
        return physics_state, dry, False
    rates = data_assimilation_rates(reference, namelist.data_assimilation, lead_seconds)
    da_dry = data_assimilation_dry_tendencies(reference, rates, namelist.metrics)
    # qv has no DryPhysicsTendencies field in the current port; keep it resident
    # and additive through the existing physics state-delta lane.
    physics_state = apply_nudging_rates(
        physics_state,
        rates,
        float(namelist.dt_s),
        fields=("qv",),
    )
    return physics_state, add_dry_physics_tendencies(dry, da_dry), True


def _apply_physics_non_dry_updates(
    dynamics_state: State,
    physics_reference: State,
    physics_state: State,
) -> State:
    """Apply physics prognostics that are not consumed by ``rk_addtend_dry``."""

    updates = {}
    for name in _PHYSICS_NON_DRY_INCREMENT_FIELDS:
        dynamics_value = getattr(dynamics_state, name)
        physics_value = getattr(physics_state, name)
        reference_value = getattr(physics_reference, name)
        if dynamics_value is None or physics_value is None or reference_value is None:
            continue
        updates[name] = dynamics_value + (physics_value - reference_value)
    updates.update({
        name: getattr(physics_state, name)
        for name in _PHYSICS_NON_DRY_REPLACE_FIELDS
        if getattr(physics_state, name) is not None
    })
    return dynamics_state.replace(**updates)


def _strict_guards_enabled() -> bool:
    """F2 strict guards, the default (Phase C): no restore-to-origin repairs or positivity floors.

    WRF's default mp_zero_out=0 zeroes nothing; round-off negatives after RK are absorbed by
    Thompson's own entry/final R1 tests (module_mp_thompson.F:1827-1925, 4007-4054), so the
    guards only masked state (ADR-F2C-STRICT-GUARDS-DEFAULT). GPUWRF_STRICT_GUARDS=0 restores
    the legacy guards for A/B runs.
    """
    return os.environ.get("GPUWRF_STRICT_GUARDS", "1") == "1"


def _microphysics_wrf_order_enabled() -> bool:
    """Opt in to solve_em's post-RK time-split microphysics call."""
    return os.environ.get("GPUWRF_MICROPHYSICS_WRF_ORDER", "1") == "1"


def _phys_tend_rk_wrf_enabled() -> bool:
    """D2/D3 physics->RK coupling in WRF form (default off).

    D2: the MYNN moist tendencies enter WRF's moist_tend/scalar_tend coupled with
    the time-n mass (calculate_phy_tend + update_phy_ten) and are integrated by the
    scalar update of every RK stage -- at the final stage of a positive-definite or
    monotonic family through rk_update_scalar_pd (solve_em.F:1932-2000) -- instead
    of a post-RK increment; the MYNN prognostics are restored right after the PBL
    call so later drivers read the step-entry state like WRF's.  D3: the held
    RTHRATEN is coupled with mut at RK1 and decoupled with the end-of-RK muts
    (phy_prep_part2, solve_em.F:3669), so the coupled rate is what stays held.
    """
    return os.environ.get("GPUWRF_PHYS_TEND_RK_WRF", "0") == "1"


def _microphysics_spec_zone(namelist: OperationalNamelist) -> int:
    """solve_em.F:3693-3707 ``sz``: spec_zone on specified/nested domains, else 0."""

    _, specified, nested = _acoustic_lateral_bc_flags(namelist)
    if not (specified or nested):
        return 0
    return int(namelist.boundary_config.spec_zone)


def _microphysics_interior_only(
    before: State, after: State, namelist: OperationalNamelist
) -> State:
    """Keep the outer ``sz`` mass ring untouched by time-split microphysics.

    WRF bounds moist_physics_prep_em, microphysics_driver and
    moist_physics_finish_em to ``ids+sz..ide-1-sz`` / ``jds+sz..jde-1-sz``
    (solve_em.F:3693-3707, module_microphysics_driver.F:809-886): on a
    specified or nested domain the spec zone receives no MP tendency, no
    precipitation and no diabatic heating; its values come from the lateral
    boundary forcing alone.  Column physics is independent per column, so the
    interior is exactly the unrestricted adapter result.
    """

    sz = _microphysics_spec_zone(namelist)
    if sz <= 0 or after is before:
        return after
    ny, nx = (int(n) for n in before.theta.shape[-2:])
    rows = jnp.arange(ny)
    cols = jnp.arange(nx)
    interior = ((rows >= sz) & (rows < ny - sz))[:, None] & (
        (cols >= sz) & (cols < nx - sz)
    )[None, :]
    updates = {}
    for name in State.__slots__:
        new = getattr(after, name)
        old = getattr(before, name)
        if new is old or new is None or old is None:
            continue
        if not hasattr(new, "shape") or tuple(new.shape[-2:]) != (ny, nx):
            raise ValueError(
                f"microphysics changed non-mass-grid State.{name}; spec-zone "
                "restriction needs a mass-point (..., ny, nx) leaf"
            )
        updates[name] = jnp.where(interior, new, jnp.asarray(old, dtype=new.dtype))
    return after.replace(**updates) if updates else after


def _h_diabatic_pair_enabled(namelist: OperationalNamelist) -> bool:
    """WRF h_diabatic pair (B38): only with the post-RK microphysics call.

    ``GPUWRF_MP_H_DIABATIC=0`` is the explicit A/B opt-out.
    """
    return (
        _microphysics_wrf_order_enabled()
        and os.environ.get("GPUWRF_MP_H_DIABATIC", "1") != "0"
        and bool(namelist.run_physics)
        and int(namelist.mp_physics) != 0
    )


def _apply_post_rk_microphysics(state: State, namelist: OperationalNamelist, *, return_precipitation=False,
                                step_index=None) -> State:
    """Update transported prognostics directly, as solve_em.F:3809 does."""
    if not bool(namelist.run_physics):
        return state
    mp_opt = int(namelist.mp_physics)
    if mp_opt == 18:
        # NSSL 2-moment: WRF itimestep == 1 cold start (CN := 0, calcnfromq) on the first own step.
        from gpuwrf.physics.nssl2mom.adapter import nssl2mom_adapter
        first = False if step_index is None else jnp.equal(step_index, 1)
        updated, nssl_precip = nssl2mom_adapter(state, float(namelist.dt_s), namelist.grid, first_step=first,
                                                return_precipitation=True)
        precip = {"rain": nssl_precip["rain_liquid"] + nssl_precip["hailncv"], "snow": nssl_precip["snowncv"],
                  "graupel": nssl_precip["grplncv"], "ice": jnp.zeros_like(nssl_precip["snowncv"])}
    elif mp_opt == DEFAULT_MP_PHYSICS:
        if return_precipitation:
            updated, precip = thompson_adapter(state, float(namelist.dt_s), return_precipitation=True)
        else:
            updated = thompson_adapter(state, float(namelist.dt_s))
    elif mp_opt == 28:
        updated = thompson_aero_adapter(state, float(namelist.dt_s))
    elif mp_opt in MP_SCAN_ADAPTERS:
        updated = MP_SCAN_ADAPTERS[mp_opt](state, float(namelist.dt_s), namelist.grid)
    else:
        return state
    updated = _microphysics_interior_only(state, updated, namelist)
    return (updated, precip) if return_precipitation else updated


def _kf_cadence_step(state, carry, namelist, step_index):
    """WRF KF-eta gate on the one-based production index, with held rates."""
    from gpuwrf.diagnostics.census import count_work
    from gpuwrf.coupling.physics_couplers import _theta_m_tendency_from_dry
    dt = float(namelist.dt_s)
    stepcu = max(1, int(namelist.cumulus_cadence_steps))
    step = jnp.asarray(step_index, dtype=jnp.int32)
    run_cu = (step == 1) | (jnp.mod(step, stepcu) == 0)
    w0avg, nca = carry.cumulus_carry
    held = carry.cumulus_tendencies

    def refresh(_unused):
        values = kf_adapter(
            state, dt, w0avg, nca, grid=namelist.grid,
            stepcu=stepcu, cudt=float(namelist.cudt_minutes),
            held_tendencies=held, return_tendencies=True,
        )
        from gpuwrf.kernels.dyn_carry_fp32 import like, real_all_enabled
        if real_all_enabled():  # E65: KF refreshed rates are WRF REAL like the held carry
            values = like(tuple(values), (held, w0avg, nca))

        return (*values, count_work(carry.census, "kf_calls"))

    rates, w0avg, nca, census = jax.lax.cond(
        run_cu, refresh, lambda _u: (held, w0avg, nca, carry.census), None
    )
    from gpuwrf.coupling.kf_rk import kf_tend_rk_enabled
    if kf_tend_rk_enabled():
        # Driver outputs remain rates. RK consumes a step-local coupled copy;
        # advance_ppt/phy_prep_part2 run only after the last RK consumer.
        return state, carry.replace(
            cumulus_carry=(w0avg, nca), cumulus_tendencies=rates, census=census)
    # KFETASCHEME clears R*CUTEN on the last active cloud step. PRATEC is
    # accumulated each timestep, exactly as module_physics_addtendc.F:2289.
    updates = {
        name: (getattr(state, name) + dt * rate).astype(getattr(state, name).dtype)
        for name, rate in zip(("theta", "qv", "qc", "qr", "qi", "qs"), rates[:6], strict=True)
    }
    # P0/WRF converts held dry-theta/vapour rates at this timestep's OLD state.
    theta_rate = _theta_m_tendency_from_dry(
        rates[0], rates[1], state.theta, state.qv, state.theta.dtype
    )
    updates["theta"] = (state.theta + dt * theta_rate).astype(state.theta.dtype)
    updates["rainc_acc"] = (state.rainc_acc + dt * rates[6]).astype(state.rainc_acc.dtype)
    # solve_em finishes RK consumption before advance_ppt clears the carry.
    clear = (nca > 0) & (jnp.floor(nca / dt + 0.5) <= 1)
    rates = tuple(jnp.where(clear[None], 0.0, rate) for rate in rates[:6]) + (rates[6],)
    nca = jnp.where(nca > 0, nca - dt, nca)
    return state.replace(**updates), carry.replace(
        cumulus_carry=(w0avg, nca), cumulus_tendencies=rates, census=census,
    )


def _source_leaf_dry_tendencies(
    mu_total, held_rthraten, rthblten, rqvblten, rublten, rvblten, qv, theta,
    namelist_metrics, theta_dtype, *, real_glue=False, kf_tendf=None,
) -> DryPhysicsTendencies:
    """WRF calculate_phy_tend + update_phy_ten (add_a2a/add_a2c) + conv_t_tendf_to_moist.

    ``real_glue`` (GPUWRF_CARRY_REAL_ALL): REAL metrics, each source coupled separately and
    summed in update_phy_ten order, WRF's REAL ``R_v/R_d``.
    """
    glue_dtype = jnp.float32 if real_glue else jnp.float64
    metrics = namelist_metrics
    if real_glue:
        from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
        metrics = real_metrics(metrics)
    mass_h = (
        metrics.c1h[:, None, None] * mu_total[None, :, :]
        + metrics.c2h[:, None, None]
    )
    # COUPLED dry theta source d(mut*theta)/dt = mut*(RTHRATEN+RTHBLTEN);
    # rk_addtend_dry consumes t_tendf already mass-coupled (it only re-divides
    # by msfty).  The MYNN theta state delta is removed from the later
    # non-dry update state below so the source is not double-applied.
    if real_glue:
        # WRF calculate_phy_tend couples each source, update_phy_ten sums them.
        t_tendf_source = mass_h * jnp.asarray(held_rthraten, glue_dtype)
        if rthblten is not None:
            t_tendf_source = t_tendf_source + mass_h * jnp.asarray(rthblten, glue_dtype)
    else:
        rth_source = held_rthraten if rthblten is None else held_rthraten + rthblten
        t_tendf_source = mass_h * rth_source
    qv_tendf_source = (
        jnp.zeros_like(t_tendf_source)
        if rqvblten is None
        else mass_h * (jnp.asarray(rqvblten, glue_dtype) if real_glue else rqvblten)
    )
    if kf_tendf is not None:
        # update_phy_ten sums RA, BL, CU before one moist-theta conversion.
        # KF has already undergone calculate_phy_tend and add_a2a ownership.
        t_tendf_source = t_tendf_source + jnp.asarray(kf_tendf[0], glue_dtype)
        qv_tendf_source = qv_tendf_source + jnp.asarray(kf_tendf[1], glue_dtype)
    # WRF use_theta_m=1 converts dry theta forcing to moist theta in
    # conv_t_tendf_to_moist immediately after update_phy_ten.
    rvrd = _RVRD_REAL if real_glue else _RVRD
    theta_m_factor = 1.0 + rvrd * jnp.asarray(qv, glue_dtype)
    t_tendf_source = (
        theta_m_factor * t_tendf_source
        + rvrd
        * jnp.asarray(theta, glue_dtype)
        / theta_m_factor
        * qv_tendf_source
    ).astype(theta_dtype)
    # v0.14 venting-residual fix: WRF PBL momentum fold.  phy_tend couples the
    # A-grid RUBLTEN/RVBLTEN with the dry column mass (module_em.F:2381,
    # ``(c1(k)*mut+c2(k))*R?BLTEN``); update_phy_ten averages mass->face
    # (add_a2c_u/add_a2c_v, phys/module_physics_addtendc.F) with the
    # specified-domain edge exclusions; rk_addtend_dry later divides by
    # msfuy/msfvx.  Without this fold the acoustic loop integrated with ZERO
    # PBL drag (the WRF-native oracle measured the missing term at 57%/72%
    # of ru/rv_tend, the u''/v'' -> ww/mu'' venting creator).
    ru_tendf_source = None
    rv_tendf_source = None
    if rublten is not None and rvblten is not None:
        rub_coupled = mass_h * jnp.asarray(rublten, glue_dtype)
        rvb_coupled = mass_h * jnp.asarray(rvblten, glue_dtype)
        nz_p, ny_p, nx_p = rub_coupled.shape
        ru_tendf_source = jnp.zeros((nz_p, ny_p, nx_p + 1), dtype=rub_coupled.dtype)
        # WRF add_a2c_u (specified): u faces i in [ids+1, ide-1], mass rows
        # j in [jds+1, jde-2] -- 0-based faces 1..nx-1, rows 1..ny-2.
        ru_tendf_source = ru_tendf_source.at[:, 1 : ny_p - 1, 1:nx_p].set(
            0.5
            * (
                rub_coupled[:, 1 : ny_p - 1, : nx_p - 1]
                + rub_coupled[:, 1 : ny_p - 1, 1:nx_p]
            )
        )
        rv_tendf_source = jnp.zeros((nz_p, ny_p + 1, nx_p), dtype=rvb_coupled.dtype)
        # WRF add_a2c_v (specified): v faces j in [jds+1, jde-1], mass cols
        # i in [ids+1, ide-2] -- 0-based faces 1..ny-1, cols 1..nx-2.
        rv_tendf_source = rv_tendf_source.at[:, 1:ny_p, 1 : nx_p - 1].set(
            0.5
            * (
                rvb_coupled[:, : ny_p - 1, 1 : nx_p - 1]
                + rvb_coupled[:, 1:ny_p, 1 : nx_p - 1]
            )
        )
    return DryPhysicsTendencies(
        t_tendf=t_tendf_source,
        ru_tendf=ru_tendf_source,
        rv_tendf=rv_tendf_source,
    )


def _pbl_scalar_rate(post, entry, dt: float, dtype):
    """RQNIBLTEN as the MYNN source-leaf adapter forms RQ?BLTEN: (post - entry) / dt."""

    return (jnp.asarray(post, dtype) - jnp.asarray(entry, dtype)) / float(dt)


def _decouple_held_rthraten(rthraten, mut, muts, namelist_metrics):
    """D3 (GPUWRF_PHYS_TEND_RK_WRF): WRF's held RTHRATEN round trip of one step.

    calculate_phy_tend couples RTHRATEN in place with the time-n mass at RK1
    (module_em.F:2243) and phy_prep_part2 decouples it with the end-of-RK grid%muts
    (solve_em.F:3669, module_big_step_utilities_em.F:5098), so between radiation
    calls the COUPLED rate is what persists.  Same glue precision as t_tendf.
    """
    from gpuwrf.kernels.dyn_carry_fp32 import real_all_enabled
    real = real_all_enabled()
    metrics = namelist_metrics
    if real:
        from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
        metrics = real_metrics(metrics)
    glue = jnp.float32 if real else jnp.float64
    c1, c2 = metrics.c1h[:, None, None], metrics.c2h[:, None, None]
    coupled = (c1 * jnp.asarray(mut, glue)[None] + c2) * jnp.asarray(rthraten, glue)
    return (coupled / (c1 * jnp.asarray(muts, glue)[None] + c2)).astype(rthraten.dtype)


def _coupled_physics_moist_tendf(mu_total, raw, namelist_metrics, *, real_glue=False) -> dict:
    """WRF calculate_phy_tend for the moist/scalar physics tendencies (GPUWRF_PHYS_TEND_RK_WRF).

    RQ?BLTEN (moist_tend, module_em.F calculate_phy_tend) and scalar_tend(P_QNI)
    (the '4d couple scalar tendencies' loop) are multiplied by c1h*mut+c2h with the
    time-n mass, in the same glue precision as the t_tendf source.
    """
    glue_dtype = jnp.float32 if real_glue else jnp.float64
    metrics = namelist_metrics
    if real_glue:
        from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
        metrics = real_metrics(metrics)
    mass_h = metrics.c1h[:, None, None] * mu_total[None, :, :] + metrics.c2h[:, None, None]
    return {name: mass_h * jnp.asarray(rate, glue_dtype) for name, rate in raw.items()}


def _physics_step_forcing(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    lead_seconds,
    *,
    run_radiation: bool,
    first_timestep=False,
    clock_base=None,
    capture_first_interval: bool = False,
    lower_fields=None,
) -> _PhysicsStepForcing | tuple[_PhysicsStepForcing, FirstIntervalMomentumRecord]:
    """Run non-timesplit physics at step entry and expose RK-fixed tendencies."""

    from gpuwrf.diagnostics.census import count_work

    if not bool(namelist.run_physics):
        if capture_first_interval:
            raise ValueError(
                "first-interval momentum capture requires run_physics=True"
            )
        next_state, dry, da_enabled = _apply_data_assimilation_forcing(
            carry.state,
            carry.state,
            DryPhysicsTendencies(),
            namelist,
            lead_seconds,
        )
        return _PhysicsStepForcing(next_state, carry, dry, da_enabled)

    before = carry.state
    next_state = before
    next_carry = carry
    source_leaf_mode = int(namelist.rad_rk_tendf) != 0
    rthblten = None
    rqvblten = None
    rublten = None
    rvblten = None
    pbl_theta_dry_delta = None
    pbl_u_face_delta = None
    pbl_v_face_delta = None
    pbl_after_state = None
    pbl_moist_raw = None
    # GPUWRF_CARRY_REAL_ALL: PBL/GWDO folds and the source-leaf coupling in WRF REAL.
    from gpuwrf.kernels.dyn_carry_fp32 import real_all_enabled
    real_glue = real_all_enabled()
    glue_dtype = jnp.float32 if real_glue else jnp.float64

    mp_opt = int(namelist.mp_physics)
    sf_opt = int(namelist.sf_sfclay_physics)
    cu_opt = int(namelist.cu_physics)
    from gpuwrf.coupling.kf_rk import (
        kf_tend_rk_enabled, couple_kf_rates, kf_add_a2a)
    kf_rk = kf_tend_rk_enabled() and cu_opt == 1
    kf_tendf = None
    if kf_rk and (not source_leaf_mode or carry.cumulus_tendencies is None):
        raise ValueError("KF_TEND_RK_WRF requires source-leaf forcing and held KF rates")

    if carry.radiation_diagnostics is not None:
        next_carry = _refresh_rrtmg_driver(
            next_carry, namelist, lead_seconds, run_radiation, clock_base
        )

    # --- microphysics slot ---
    if not _microphysics_wrf_order_enabled() and mp_opt == DEFAULT_MP_PHYSICS:
        next_state = _microphysics_interior_only(
            next_state, thompson_adapter(next_state, float(namelist.dt_s)), namelist
        )
    elif not _microphysics_wrf_order_enabled() and mp_opt == 28:
        # v0.16 aerosol-aware Thompson: same coupler shape as mp=8 plus the
        # prognostic Nc/nwfa/nifa threading + surface aerosol emission.
        next_state = _microphysics_interior_only(
            next_state, thompson_aero_adapter(next_state, float(namelist.dt_s)), namelist
        )
    elif not _microphysics_wrf_order_enabled() and mp_opt == 18:
        raise NotImplementedError("mp_physics=18 (NSSL) is wired only on the WRF-order post-RK microphysics path")
    elif not _microphysics_wrf_order_enabled() and mp_opt in MP_SCAN_ADAPTERS:
        next_state = _microphysics_interior_only(
            next_state,
            MP_SCAN_ADAPTERS[mp_opt](next_state, float(namelist.dt_s), namelist.grid),
            namelist,
        )
    # mp_opt == 0 -> passive (no microphysics).
    # NOTE (v0.14 venting-residual sprint, NAMED NEXT THETA TERM, NOT ROUTED):
    # WRF folds h_diabatic -- the previous-step microphysics theta_m heating
    # rate captured by moist_physics_finish_em -- into t_tend at EVERY RK stage
    # (rk_addtend_dry, module_em.F:1079).  The WRF-native h36 theta-tendency
    # oracle measures the missing term at rms 20-26 (cloud levels k4-k13) of
    # the 25.6-rms total t_tend.  Capturing it as the step-entry Thompson
    # state delta / dt was tested and REJECTED by the same oracle (10x too
    # large and anti-correlated at the re-init step-1: the adapter's first
    # call re-equilibrates the fp32 wrfout state, which is NOT WRF's settled
    # heating rate; proofs/v014/switzerland_uv_lane_contributors
    # fold_diagnosis).  A faithful route needs the settled per-step heating
    # (e.g. the previous step's mp delta carried across steps), left for the
    # follow-up sprint.

    lower_legacy_rad = None
    if lower_fields is not None:
        # WRF radiation_driver precedes surface_driver. Aux4 fields are already
        # read, but radiation still sees the previous water TSK/top TSLB.
        # The legacy surface-only radiation refresh also stays before SST.
        if bool(namelist.use_noahmp) and next_carry.radiation_diagnostics is None:
            lower_legacy_rad = _refresh_noahmp_rad(
                next_state, namelist, lead_seconds, run_radiation, carry.noahmp_rad,
                land_state=carry.noahmp_land, clock_base=clock_base, census=next_carry.census,
                cam_abs=carry.cam_abs,
            )
            if next_carry.census is not None:
                lower_legacy_rad, census = lower_legacy_rad
                next_carry = next_carry.replace(census=census)
        from gpuwrf.io.lower_boundary import apply_lower_boundary_fields
        next_carry = apply_lower_boundary_fields(next_carry, lower_fields)
        next_state = next_state.replace(t_skin=next_carry.state.t_skin)

    # --- surface-layer / land slot ---
    # sf=2 Janjic Eta is the v0.13 traceable MYJ-pair surface layer (defined in
    # physics.myj_adapters, NOT in coupling.scan_adapters); route it explicitly.
    if bool(namelist.use_noahmp):
        if sf_opt == 2:
            next_state = janjic_sfclay_adapter(next_state, float(namelist.dt_s), namelist.grid)
        elif sf_opt in SFCLAY_SCAN_ADAPTERS:
            next_state = SFCLAY_SCAN_ADAPTERS[sf_opt](next_state, float(namelist.dt_s), namelist.grid)
        if next_carry.radiation_diagnostics is not None:
            rad = next_carry.radiation_diagnostics
            next_carry_rad = (jnp.maximum(rad.swnorm, 0.0), rad.glw, rad.coszen)
        elif lower_legacy_rad is not None:
            next_carry_rad = lower_legacy_rad
        else:
            next_carry_rad = _refresh_noahmp_rad(
                next_state, namelist, lead_seconds, run_radiation, carry.noahmp_rad,
                land_state=carry.noahmp_land, clock_base=clock_base, census=next_carry.census,
                cam_abs=carry.cam_abs,
            )
            if next_carry.census is not None:
                next_carry_rad, census = next_carry_rad
                next_carry = next_carry.replace(census=census)
        clock = _noahmp_clock(namelist, clock_base, lead_seconds)
        radiation = _NoahMPRadiation(*next_carry_rad)
        ep, rp = _noahmp_params(namelist)
        history = getattr(next_carry, "history_diagnostics", None)
        land_history = getattr(next_carry, "land_history", None)
        next_state, next_land, *history_fields = noahmp_surface_step(
            next_state, (carry.noahmp_land if lower_fields is None else next_carry.noahmp_land),
            _lane_noahmp_static(namelist, clock_base),
            float(namelist.dt_s), radiation=radiation, clock=clock,
            energy_params=ep, rad_params=rp, first_timestep=first_timestep,
            grid=namelist.grid, history=history is not None,
            **({"land_history": True} if land_history is not None else {}),
            **({"precipitation": next_carry.noahmp_precipitation}
               if getattr(next_carry, "noahmp_precipitation", None) is not None else {}),
        )
        if history is not None:
            # WRF history fields = this step's surface_driver values (history precedes solve).
            next_carry = next_carry.replace(history_diagnostics=history.write(history_fields[0]))
        if land_history is not None:
            from gpuwrf.runtime.history_accumulators import PackedFields, accumulate_energy
            produced = history_fields[0]
            next_carry = next_carry.replace(
                land_history=PackedFields.pack(
                    {name: jnp.asarray(produced["land_history"][name], land_history.data.dtype)
                     for name in land_history}),
                energy_accumulators=PackedFields.pack(accumulate_energy(
                    next_carry.energy_accumulators, next_carry.radiation_diagnostics,
                    produced, float(namelist.dt_s))),
            )
        next_carry = next_carry.replace(noahmp_land=next_land, noahmp_rad=next_carry_rad)
    else:
        if _explicit_ruc(namelist):
            pass  # RUC: the MYNN surface layer runs inside ruc_surface_step (exact FLHC/FLQC)
        elif sf_opt == 2:
            next_state = janjic_sfclay_adapter(next_state, float(namelist.dt_s), namelist.grid)
        elif sf_opt in SFCLAY_SCAN_ADAPTERS:
            next_state = SFCLAY_SCAN_ADAPTERS[sf_opt](next_state, float(namelist.dt_s), namelist.grid)
        else:
            next_state = surface_adapter(
                next_state,
                float(namelist.dt_s),
                namelist.grid,
                first_timestep=first_timestep,
            )
        if _explicit_noahclassic(namelist):
            next_noahclassic_rad = _refresh_noahmp_rad(
                next_state, namelist, lead_seconds, run_radiation, carry.noahclassic_rad,
                census=next_carry.census,
            )
            if next_carry.census is not None:
                next_noahclassic_rad, census = next_noahclassic_rad
                next_carry = next_carry.replace(census=census)
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
        elif _explicit_slab(namelist):
            # Thermal-diffusion slab LSM (sf_surface_physics=1). The surface-layer
            # adapter above wrote the kinematic flux handles; the slab recovers
            # FLHC/FLQC from them, advances the 5-layer TSLB carry from the held
            # GSW/GLW down-radiation, and overwrites land TSK/HFX/QFX.
            # Hold the GSW/GLW down-radiation across non-radiation-cadence steps
            # via a CONCRETE 3-tuple (gsw, glw, cosz), exactly as the Noah-classic
            # seam passes ``carry.noahclassic_rad``. ``cosz`` is unused by the slab
            # energy solve, so it rides a zeros placeholder. Passing ``None`` here
            # made ``_refresh_noahmp_rad`` return ``None`` on a held step --
            # jax.lax.cond requires both branches return the same 3-tuple, so the
            # unpack crashed (this slab path was never exercised end-to-end until a
            # real slab_static bundle made the operational scan select it).
            held_slab_rad = (
                carry.slab_rad.gsw,
                carry.slab_rad.glw,
                jnp.zeros_like(carry.slab_rad.gsw),
            )
            refreshed_rad = _refresh_noahmp_rad(
                next_state, namelist, lead_seconds, run_radiation, held_slab_rad,
                census=next_carry.census,
            )
            if next_carry.census is not None:
                refreshed_rad, census = refreshed_rad
                next_carry = next_carry.replace(census=census)
            slab_soldn, slab_lwdn, _slab_cosz = refreshed_rad
            next_state, next_slab_land = slab_surface_step(
                next_state,
                carry.slab_land,
                namelist.slab_static,
                float(namelist.dt_s),
                radiation=SlabRadiation(slab_soldn, slab_lwdn),
            )
            next_carry = next_carry.replace(
                slab_land=next_slab_land,
                slab_rad=SlabRadiation(slab_soldn, slab_lwdn),
            )
        elif _explicit_pleim_xiu(namelist):
            # Pleim-Xiu 2-layer ISBA LSM (sf_surface_physics=7). The surface-layer
            # adapter (PX surface layer is the WRF pair) wrote the kinematic flux
            # handles; the PX LSM recovers RMOL from them, advances the 2-layer
            # ISBA carry from the held GSW/GLW down-radiation, and overwrites land
            # TSK/HFX/QFX.
            # Same held-radiation 3-tuple contract as the slab seam above (cosz is
            # unused by the PX ISBA solve; ``None`` would crash on a held step).
            held_px_rad = (
                carry.px_rad.gsw,
                carry.px_rad.glw,
                jnp.zeros_like(carry.px_rad.gsw),
            )
            refreshed_rad = _refresh_noahmp_rad(
                next_state, namelist, lead_seconds, run_radiation, held_px_rad,
                census=next_carry.census,
            )
            if next_carry.census is not None:
                refreshed_rad, census = refreshed_rad
                next_carry = next_carry.replace(census=census)
            px_soldn, px_lwdn, _px_cosz = refreshed_rad
            next_state, next_px_land = pleim_xiu_surface_step(
                next_state,
                carry.px_land,
                namelist.px_static,
                float(namelist.dt_s),
                radiation=PleimXiuRadiation(px_soldn, px_lwdn),
            )
            next_carry = next_carry.replace(
                px_land=next_px_land,
                px_rad=PleimXiuRadiation(px_soldn, px_lwdn),
            )
        elif _explicit_ruc(namelist):
            # RUC LSM (sf_surface_physics=3, lane o1-ruc): MYNN surface layer + LSMRUC
            # (coupling.ruc_surface_hook). Native RRTMG: WRF GSW = SWDOWN-SWUP held in
            # radiation_diagnostics; legacy: held downward SW/LW -> GSW with the RUC albedo.
            from gpuwrf.coupling.ruc_surface_hook import RucRadiation, ruc_net_radiation, ruc_surface_step

            if next_carry.radiation_diagnostics is not None:
                rdiag = next_carry.radiation_diagnostics
                ruc_net = RucRadiation(jnp.asarray(rdiag.swdown) - jnp.asarray(rdiag.swup), rdiag.glw)
                next_ruc_rad = carry.ruc_rad
            else:
                held_ruc_rad = (carry.ruc_rad.gsw, carry.ruc_rad.glw, jnp.zeros_like(carry.ruc_rad.gsw))
                refreshed_rad = _refresh_noahmp_rad(
                    next_state, namelist, lead_seconds, run_radiation, held_ruc_rad,
                    census=next_carry.census,
                )
                if next_carry.census is not None:
                    refreshed_rad, census = refreshed_rad
                    next_carry = next_carry.replace(census=census)
                ruc_soldn, ruc_lwdn, _ruc_cosz = refreshed_rad
                next_ruc_rad = RucRadiation(ruc_soldn, ruc_lwdn)
                ruc_net = ruc_net_radiation(ruc_soldn, ruc_lwdn, carry.ruc_land.alb)
            next_state, next_ruc_land = ruc_surface_step(
                next_state,
                carry.ruc_land,
                namelist.ruc_static,
                float(namelist.dt_s),
                namelist.grid,
                radiation=ruc_net,
                precipitation=next_carry.noahmp_precipitation,
                first_timestep=first_timestep,
            )
            next_carry = next_carry.replace(ruc_land=next_ruc_land, ruc_rad=next_ruc_rad)

    # --- PBL slot ---
    # bl=2 MYJ is the v0.13 traceable MYJ PBL (paired with the Janjic surface
    # layer already run in the surface slot); it re-derives the surface coupling
    # and threads the TKE carry via qke. Defined in physics.myj_adapters.
    bl_opt = int(namelist.bl_pbl_physics)
    next_state, next_carry = _small_grid_firewall((next_state, next_carry), next_state.theta.shape[-2:]); pbl_entry_state = next_state
    if bl_opt == 2:
        next_state = myj_pbl_adapter(next_state, float(namelist.dt_s), namelist.grid)
    elif bl_opt == 9:
        # CAM-UW threads its own carry and reads the held radiation CLDFRA/RTHRATEN.
        from gpuwrf.coupling.scan_adapters import camuw_pbl_adapter
        held_rad = getattr(next_carry, "radiation_diagnostics", None)
        next_state, camuw_carry = camuw_pbl_adapter(
            next_state, float(namelist.dt_s), namelist.grid, camuw=next_carry.camuw_pbl,
            rthraten=next_carry.rthraten,
            cldfra=None if held_rad is None else getattr(held_rad, "cloud_fraction", None))
        next_carry = next_carry.replace(camuw_pbl=camuw_carry)
    elif bl_opt in PBL_SCAN_ADAPTERS:
        next_state = PBL_SCAN_ADAPTERS[bl_opt](next_state, float(namelist.dt_s), namelist.grid)
    elif bl_opt == DEFAULT_BL_PBL_PHYSICS:
        if source_leaf_mode:
            mynn = mynn_adapter_with_source_leaves(
                next_state,
                float(namelist.dt_s),
                namelist.grid,
                first_timestep=first_timestep,
            )
            next_state = mynn.state
            rthblten = mynn.rthblten
            rqvblten = mynn.rqvblten
            # v0.14 venting-residual fix: raw A-grid PBL momentum sources for the
            # WRF ru/rv_tendf fold (the WRF-native advance_uv oracle proved the
            # staged ru/rv_tend were missing the entire PBL drag: 57%/72% rel,
            # proofs/v014/switzerland_uv_lane_decomposition).
            rublten = mynn.rublten
            rvblten = mynn.rvblten
            if _phys_tend_rk_wrf_enabled():
                # WRF's MYNN driver writes tendencies only (module_pbl_driver.F); every
                # later driver (KF :1380/1511) reads the step-entry prognostics.  Restore
                # them now and keep the PBL memory/diagnostic leaves; the dycore gets
                # theta/u/v through t_tendf/ru/rv_tendf and the moist species through
                # DryPhysicsTendencies.moist_tendf at the WRF RK cadence.
                pbl_moist_raw = {"qv": rqvblten}
                restore = {"theta": pbl_entry_state.theta, "u": pbl_entry_state.u,
                           "v": pbl_entry_state.v, "qv": pbl_entry_state.qv}
                if mynn.rqcblten is not None:
                    pbl_moist_raw.update(qc=mynn.rqcblten, qi=mynn.rqiblten)
                    restore.update(qc=pbl_entry_state.qc, qi=pbl_entry_state.qi)
                if next_state.Ni is not pbl_entry_state.Ni:
                    # module_pbl_driver.F:1866-1874 (bl_mynn_mixscalars=1):
                    # scalar_tend(P_QNI) = RQNIBLTEN, coupled later like moist_tend.
                    pbl_moist_raw["Ni"] = _pbl_scalar_rate(
                        next_state.Ni, pbl_entry_state.Ni, float(namelist.dt_s),
                        jnp.float32 if real_glue else jnp.float64)
                    restore["Ni"] = pbl_entry_state.Ni
                next_state = next_state.replace(**restore)
            elif real_glue:
                # Removed below as entry + (post-PBL changes): exact in REAL and equal to
                # the wide delta form (the post-PBL theta change is Sterbenz-exact).
                pbl_after_state = next_state
            else:
                pbl_theta_dry_delta = (
                    jnp.asarray(next_state.theta, jnp.float64)
                    - jnp.asarray(pbl_entry_state.theta, jnp.float64)
                )
                # Face-space MYNN momentum increments actually applied to the state
                # (A2C-coupled inside _state_from_mynn_output); removed again below so
                # the dycore integrates the SAME drag via ru/rv_tendf instead of a
                # step-entry Euler add (WRF cadence, no double-application).
                pbl_u_face_delta = (
                    jnp.asarray(next_state.u, jnp.float64)
                    - jnp.asarray(pbl_entry_state.u, jnp.float64)
                )
                pbl_v_face_delta = (
                    jnp.asarray(next_state.v, jnp.float64)
                    - jnp.asarray(pbl_entry_state.v, jnp.float64)
                )
        else:
            next_state = mynn_adapter(
                next_state,
                float(namelist.dt_s),
                namelist.grid,
                first_timestep=first_timestep,
            )
    # bl_opt == 0 -> no PBL mixing.
    qvpblten = (
        (
            jnp.asarray(next_state.qv, jnp.float64)
            - jnp.asarray(pbl_entry_state.qv, jnp.float64)
        )
        / float(namelist.dt_s)
    ) if pbl_moist_raw is None else jnp.asarray(pbl_moist_raw["qv"], jnp.float64)

    # --- orographic gravity-wave drag slot (gwd_opt=1) ---
    # WRF applies GWDO inside the PBL driver, right after the PBL momentum
    # tendency (phys/module_pbl_driver.F). gwd_opt=1 + a per-run GWDOStatics
    # bundle activates the faithful bl_gwdo_run port; otherwise it is a no-op.
    # B46: WRF gwdo reads the pre-PBL phy_prep state and accumulates into
    # RUBLTEN/RVBLTEN (bl_gwdo.F90:632), integrated via ru/rv_tendf at RK cadence.
    if int(namelist.gwd_opt) == 1 and namelist.gwdo_statics is not None:
        if rublten is not None and rvblten is not None:  # source-leaf PBL fold
            gwd_ru, gwd_rv, dtaux3d, dtauy3d, dusfcg, dvsfcg = gwdo_tendencies(
                pbl_entry_state, float(namelist.dt_s), namelist.gwdo_statics, namelist.grid,
                return_diagnostics=True,
            )
            next_state = next_state.replace(dtaux3d=dtaux3d, dtauy3d=dtauy3d,
                                            dusfcg=dusfcg, dvsfcg=dvsfcg)
            rublten = jnp.asarray(rublten, glue_dtype) + jnp.asarray(gwd_ru, glue_dtype)
            rvblten = jnp.asarray(rvblten, glue_dtype) + jnp.asarray(gwd_rv, glue_dtype)
        else:
            next_state = gwdo_adapter(
                next_state, float(namelist.dt_s), namelist.gwdo_statics, namelist.grid,
                input_state=pbl_entry_state,
            )

    # --- cumulus slot ---
    if cu_opt == 6:
        qvften = _tiedtke_qvften_from_flux_advection(next_state, namelist)
        next_state = tiedtke_adapter(
            next_state,
            float(namelist.dt_s),
            namelist.grid,
            qvften=qvften,
            qvpblten=qvpblten,
        )
    elif cu_opt == 16:
        # New-Tiedtke: WRF RQVFTEN = advective + PBL moisture forcing; WRF
        # RTHFTEN = the accumulated non-convective physics theta forcing of
        # this step (radiation + surface + PBL slots; the advective-theta
        # component is a named coupling caveat -- see ntiedtke_adapter).
        qvften16 = (
            _tiedtke_qvften_from_flux_advection(next_state, namelist) + qvpblten
        )
        thften16 = (
            jnp.asarray(next_state.theta, jnp.float64)
            - jnp.asarray(before.theta, jnp.float64)
        ) / float(namelist.dt_s)
        next_state = ntiedtke_adapter(
            next_state,
            float(namelist.dt_s),
            namelist.grid,
            qvften=qvften16,
            thften=thften16,
        )
    elif cu_opt in (5, 93):
        # Grell-3D / Grell-Devenyi: WRF RTHFTEN+RTHRATEN+RTHBLTEN and RQVFTEN+RQVBLTEN
        # forcing = this step's accumulated non-convective increments since physics
        # entry, plus the MYNN rates only when the PBL state was restored (WRF-RK
        # source leaves); the advective part is a named carry-over.
        periodic_x, _specified, _nested = _acoustic_lateral_bc_flags(namelist)
        rad = next_carry.radiation_diagnostics
        gsw = None if rad is None else (jnp.asarray(rad.swdown) - jnp.asarray(rad.swup))
        pbl_restored = pbl_moist_raw is not None
        next_state = CU_STATELESS_SCAN_ADAPTERS[cu_opt](
            next_state, float(namelist.dt_s), namelist.grid, forcing_entry_state=before,
            rthblten=rthblten if pbl_restored else None,
            rqvblten=rqvblten if pbl_restored else None, gsw=gsw,
            periodic_x=periodic_x, periodic_y=periodic_x,
        )
    elif cu_opt == 4:
        # Scale-aware GFS SAS (v0.3.4 o1-sas): WRF cumulus_driver itimestep/STEPCU gate,
        # inputs from the time-n physics entry, held REAL rates (coupling.scalesas_adapter).
        from gpuwrf.coupling.scalesas_adapter import scalesas_cadence_step
        next_state, sas_rates = scalesas_cadence_step(
            before, next_state, next_carry.cumulus_tendencies, float(namelist.dt_s), namelist.grid,
            stepcu=int(namelist.cumulus_cadence_steps),
            itimestep=1 + jnp.rint(jnp.asarray(lead_seconds) / float(namelist.dt_s)).astype(jnp.int32))
        if sas_rates is not None:
            next_carry = next_carry.replace(cumulus_tendencies=sas_rates)
    elif cu_opt in CU_STATELESS_SCAN_ADAPTERS:
        next_state = CU_STATELESS_SCAN_ADAPTERS[cu_opt](
            next_state, float(namelist.dt_s), namelist.grid
        )
    elif cu_opt == 1:
        if carry.cumulus_tendencies is not None:
            if kf_rk:
                # WRF drivers all read the time-n physics entry. Do not feed
                # PBL's prognostic state replacements into the KF trigger.
                _unused_state, next_carry = _kf_cadence_step(
                    before, next_carry, namelist,
                    1 + jnp.rint(jnp.asarray(lead_seconds) / float(namelist.dt_s)).astype(jnp.int32))
                coupled = couple_kf_rates(next_carry.cumulus_tendencies, before.mu_total, namelist.metrics)
                # Preserve unmasked coupled driver rates for final decoupling;
                # add_a2a alone owns the specified/nested interior restriction.
                next_carry = next_carry.replace(cumulus_tendencies=coupled)
                periodic_x, specified, nested = _acoustic_lateral_bc_flags(namelist)
                kf_tendf = tuple(kf_add_a2a(rate, specified_or_nested=specified or nested,
                                          periodic_x=periodic_x) for rate in coupled[:6])
            else:
                next_state, next_carry = _kf_cadence_step(
                    next_state, next_carry, namelist,
                    1 + jnp.rint(jnp.asarray(lead_seconds) / float(namelist.dt_s)).astype(jnp.int32),
                )
        else:
            w0avg, nca = (
                carry.cumulus_carry if carry.cumulus_carry is not None
                else initial_kf_carry(next_state)
            )
            next_state, w0avg_next, nca_next = kf_adapter(
                next_state, float(namelist.dt_s), w0avg, nca, grid=namelist.grid
            )
            next_carry = next_carry.replace(cumulus_carry=(w0avg_next, nca_next))
            next_carry = next_carry.replace(census=count_work(next_carry.census, "kf_calls"))
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
    ra_sw = int(namelist.ra_sw_physics); next_state, next_carry = _small_grid_firewall((next_state, next_carry), next_state.theta.shape[-2:])
    ra_lw = int(namelist.ra_lw_physics)
    land_for_rad = carry.noahmp_land if bool(namelist.use_noahmp) else None
    # #91/S3: traced (julian, utc_minute) + CLWRF gas anchor for the radiation
    # helpers (None on the legacy host-extraction path). Date-independent HLO ->
    # cross-date cache hit that also uses each date's own valid-time gases.
    rad_clock_base = _rad_clock_base(clock_base)

    def _sw_tendency() -> jnp.ndarray:
        if ra_sw == 0:
            return jnp.zeros_like(next_state.theta)
        if ra_sw == 3:
            from gpuwrf.coupling.cam_radiation import cam_sw_theta_tendency

            return cam_sw_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                clock_base=rad_clock_base,
                radiation_static=namelist.radiation_static,
                land_state=land_for_rad,
            )
        if ra_sw == 1:
            return dudhia_sw_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                clock_base=rad_clock_base,
                radiation_static=namelist.radiation_static,
                land_state=land_for_rad,
            )
        if ra_sw == 2:
            return gsfc_sw_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                clock_base=rad_clock_base,
                radiation_static=namelist.radiation_static,
                land_state=land_for_rad,
            )
        return rrtmg_sw_theta_tendency(
            next_state,
            namelist.grid,
            time_utc=namelist.time_utc,
            lead_seconds=lead_seconds,
            clock_base=rad_clock_base,
            radiation_static=namelist.radiation_static,
            topo_shading=int(namelist.topo_shading),
            slope_rad=int(namelist.slope_rad),
            shadow_length_m=float(namelist.topo_shadow_length_m),
            land_state=land_for_rad,
            use_mp_re=int(mp_re_active(namelist)),
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
        if ra_lw == 31:
            # Held-Suarez is a COMBINED idealized LW+SW Newtonian relaxation
            # (HSRAD is the only radiative call WRF makes); _resolve_operational_suite
            # has already asserted ra_sw_physics=0, so _sw_tendency() is exactly zero
            # and this is the sole radiative source.
            return held_suarez_theta_tendency(
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
            clock_base=rad_clock_base,
            radiation_static=namelist.radiation_static,
            land_state=land_for_rad,
            use_mp_re=int(mp_re_active(namelist)),
        )

    def _refresh_rthraten(_unused) -> jnp.ndarray:
        # Default RRTMG SW+LW: keep the combined single-build path byte-unchanged.
        if ra_sw == 4 and ra_lw == 4:
            return rrtmg_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
                clock_base=rad_clock_base,
                radiation_static=namelist.radiation_static,
                topo_shading=int(namelist.topo_shading),
                slope_rad=int(namelist.slope_rad),
                shadow_length_m=float(namelist.topo_shadow_length_m),
                land_state=land_for_rad,
                use_mp_re=int(mp_re_active(namelist)),
            )
        if ra_lw == 3:
            # CAM LW (o1-camrad): the held REAL absorptivities ride with the held rate (WRF cam_abs_freq_s).
            from gpuwrf.coupling.cam_radiation import cam_doabsems, cam_lw_theta_tendency, initial_cam_held

            cam_prev = carry.cam_abs if carry.cam_abs is not None else initial_cam_held(next_state)
            rth_lw, cam_next = cam_lw_theta_tendency(
                next_state, namelist.grid, time_utc=namelist.time_utc, lead_seconds=lead_seconds,
                clock_base=rad_clock_base, radiation_static=namelist.radiation_static, land_state=land_for_rad,
                held=cam_prev, doabsems=cam_doabsems(lead_seconds, namelist.dt_s, namelist.radiation_cadence_steps,
                                                     namelist.cam_abs_freq_s, held=cam_prev))
            return _sw_tendency() + rth_lw, cam_next
        return _sw_tendency() + _lw_tendency()

    held_value = carry.rthraten
    if ra_lw == 3:
        from gpuwrf.coupling.cam_radiation import initial_cam_held

        held_value = (carry.rthraten, carry.cam_abs if carry.cam_abs is not None else initial_cam_held(next_state))
    if next_carry.radiation_diagnostics is None and getattr(next_carry, "o3rad", None) is not None:
        raise NotImplementedError(
            "GPUWRF_NEST_O3_FROM_PARENT needs the RRTMG driver refresh (radiation_diagnostics seeded); "
            "the legacy RTHRATEN refresh would ignore the held o3rad"
        )
    if next_carry.radiation_diagnostics is not None:
        held_rthraten = next_carry.rthraten
    elif next_carry.census is not None:
        def counted_refresh(_unused):
            return _refresh_rthraten(None), count_work(next_carry.census, "radiation_tendency_calls")

        def held_refresh(_unused):
            return held_value, next_carry.census

        if isinstance(run_radiation, bool):
            held_rthraten, census = counted_refresh(None) if run_radiation else held_refresh(None)
        else:
            held_rthraten, census = jax.lax.cond(run_radiation, counted_refresh, held_refresh, None)
        next_carry = next_carry.replace(census=census)
    elif isinstance(run_radiation, bool):
        held_rthraten = _refresh_rthraten(None) if run_radiation else held_value
    else:
        held_rthraten = jax.lax.cond(
            run_radiation, _refresh_rthraten, lambda _u: held_value, None
        )
    if ra_lw == 3 and next_carry.radiation_diagnostics is None:
        held_rthraten, cam_abs_next = held_rthraten
        next_carry = next_carry.replace(cam_abs=cam_abs_next)
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
    if source_leaf_mode:
        dry = _source_leaf_dry_tendencies(
            next_state.mu_total, held_rthraten, rthblten, rqvblten, rublten, rvblten,
            before.qv, before.theta, namelist.metrics, next_state.theta.dtype,
            real_glue=real_glue, kf_tendf=kf_tendf,
        )
        if pbl_moist_raw is not None:
            dry = dataclass_replace(dry, moist_tendf=_coupled_physics_moist_tendf(
                next_state.mu_total, pbl_moist_raw, namelist.metrics, real_glue=real_glue))
        if kf_tendf is not None:
            # The shared RK scalar consumer is supplied by BD99. These sources
            # are already coupled and must not receive an advection map factor.
            scalars = dict(dry.moist_tendf or {})
            for name, rate in zip(("qv", "qc", "qr", "qi", "qs"), kf_tendf[1:6], strict=True):
                scalars[name] = scalars[name] + rate if name in scalars else rate
            dry = dataclass_replace(dry, moist_tendf=scalars)
        if pbl_after_state is not None:
            # REAL: entry + (changes after the PBL call, e.g. KF theta); u/v return to entry.
            next_state = next_state.replace(
                theta=pbl_entry_state.theta + (next_state.theta - pbl_after_state.theta),
                u=pbl_entry_state.u + (next_state.u - pbl_after_state.u),
                v=pbl_entry_state.v + (next_state.v - pbl_after_state.v),
            )
        if pbl_theta_dry_delta is not None:
            next_state = next_state.replace(
                theta=(
                    jnp.asarray(next_state.theta, jnp.float64) - pbl_theta_dry_delta
                ).astype(next_state.theta.dtype)
            )
        if pbl_u_face_delta is not None and pbl_v_face_delta is not None:
            # Remove the step-entry Euler momentum add; the dycore now delivers
            # the identical drag through ru/rv_tendf at the WRF RK/acoustic
            # cadence.  Subtracting the captured delta (rather than restoring the
            # PBL-entry winds) keeps any later u/v modifiers (e.g. GWDO) intact.
            next_state = next_state.replace(
                u=(
                    jnp.asarray(next_state.u, jnp.float64) - pbl_u_face_delta
                ).astype(next_state.u.dtype),
                v=(
                    jnp.asarray(next_state.v, jnp.float64) - pbl_v_face_delta
                ).astype(next_state.v.dtype),
            )
        # theta is left WITHOUT the radiation add here; the dycore delivers it via
        # the RK/acoustic cadence above.
    else:
        next_state = next_state.replace(
            theta=next_state.theta + float(namelist.dt_s) * held_rthraten
        )
        dry = _dry_physics_tendencies_from_state_delta(
            before, next_state, namelist, float(namelist.dt_s)
        )
    next_state, dry, _da_enabled = _apply_data_assimilation_forcing(
        before,
        next_state,
        dry,
        namelist,
        lead_seconds,
    )
    next_carry = next_carry.replace(rthraten=held_rthraten)

    forcing = _PhysicsStepForcing(next_state, next_carry, dry, True)
    if capture_first_interval:
        if (
            rublten is None
            or rvblten is None
            or dry.ru_tendf is None
            or dry.rv_tendf is None
        ):
            raise ValueError(
                "first-interval momentum capture requires the MYNN source-leaf "
                "PBL path (rad_rk_tendf=1 with raw RUBLTEN/RVBLTEN)"
            )
        return forcing, FirstIntervalMomentumRecord(
            rublten=jnp.asarray(rublten, jnp.float64),
            rvblten=jnp.asarray(rvblten, jnp.float64),
            ru_tendf=jnp.asarray(dry.ru_tendf, jnp.float64),
            rv_tendf=jnp.asarray(dry.rv_tendf, jnp.float64),
        )
    return forcing


def _physics_boundary_step_with_limiter_diagnostics(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
    clock_base=None,
    capture_rca: bool = False,
    capture_phase_tap: bool = False,
    capture_first_interval: bool = False,
    capture_ladder: bool = False,
) -> (
    tuple[OperationalCarry, dict[str, jax.Array]]
    | _RcaStepResult
    | CorrectedNiPhaseTapResult
    | FirstIntervalStepResult
    | FirstIntervalLadderStepResult
):
    from gpuwrf.diagnostics.census import (
        count_boundary_guards, count_dynamics_guards, count_work,
        observe_f2_state, count_water_repair, guard_dry_mass_kg,
    )

    lower_fields = None
    if namelist.lower_boundary is not None:
        from gpuwrf.io.lower_boundary import lower_boundary_fields
        lower_boundary = getattr(clock_base, "lower_boundary", None)
        if lower_boundary is None:
            lower_boundary = namelist.lower_boundary
        lower_fields = lower_boundary_fields(lower_boundary, step_index - 1)
        if getattr(clock_base, "noahmp_tbot", None) is not None:
            # The batched lane's dynamic static-view takes precedence over the
            # namelist holder; refresh that view's VEGFRA operand too.
            clock_base = clock_base._replace(
                noahmp_shdfac=lower_fields["VEGFRA"].astype(clock_base.noahmp_shdfac.dtype) / 100.0
            )
        if namelist.noahmp_static is not None:
            # A local dynamic view, before physics: VEGFRA is a prescribed
            # auxinput4 field (percent); Noah-MP receives SHDFAC in [0,1].
            namelist = dataclass_replace(
                namelist,
                noahmp_static=namelist.noahmp_static.replace(
                    shdfac=lower_fields["VEGFRA"].astype(namelist.noahmp_static.shdfac.dtype) / 100.0
                ),
            )

    if bool(capture_rca) and bool(capture_phase_tap):
        raise ValueError("RCA and phase-tap captures are mutually exclusive")
    if bool(capture_first_interval) and (bool(capture_rca) or bool(capture_phase_tap)):
        raise ValueError("first-interval capture is mutually exclusive with RCA/phase-tap")
    if bool(capture_ladder) and (bool(capture_rca) or bool(capture_phase_tap)):
        raise ValueError("ladder capture is mutually exclusive with RCA/phase-tap")
    physical_origin = carry.state
    carry = carry.replace(census=observe_f2_state(
        carry.census, "step_entry", physical_origin, step_index))
    state_health = [_rca_state_health(physical_origin)] if capture_rca else None
    state_target = [_rca_state_target(physical_origin)] if capture_rca else None
    boundary_health = _rca_boundary_health(physical_origin) if capture_rca else None
    # Preserve the existing endpoint clock for boundary/dycore interpolation.
    # Physics has its own step-entry clock below; frozen nests separately select
    # the boundary-package endpoint.
    lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
    boundary_lead_seconds = lead_seconds
    # domain_tree dispatches steps1..N; physics sees the start of this step.
    physics_lead_seconds = (step_index.astype(jnp.float64) - 1.0) * float(namelist.dt_s)
    if _nested_frozen_wrf_boundary_active(namelist):
        boundary_lead_seconds = nested_boundary_package_endpoint_seconds(
            step_index,
            child_dt_s=float(namelist.dt_s),
            parent_cadence_s=float(namelist.boundary_config.update_cadence_s),
        )
    if capture_first_interval:
        physics_forcing, first_interval_record = _physics_step_forcing(
            carry,
            namelist,
            physics_lead_seconds,
            run_radiation=run_radiation,
            first_timestep=jnp.equal(step_index, 1),
            clock_base=clock_base,
            capture_first_interval=True,
            lower_fields=lower_fields,
        )
    else:
        physics_forcing = _physics_step_forcing(
            carry,
            namelist,
            physics_lead_seconds,
            run_radiation=run_radiation,
            first_timestep=jnp.equal(step_index, 1),
            clock_base=clock_base,
            lower_fields=lower_fields,
        )
        first_interval_record = None
    if capture_rca:
        state_health.append(_rca_state_health(physics_forcing.state))
        state_target.append(_rca_state_target(physics_forcing.state))
    carry, physics_dry_tendencies, namelist_fw = _small_grid_firewall((physics_forcing.carry, physics_forcing.dry_tendencies, namelist), physics_forcing.carry.state.theta.shape[-2:])
    # B43: the RK3 stage's acoustic-averaged ru_m/rv_m and the root species
    # split, filled inside the same trace when GPUWRF_ROOT_SCALAR_BDY_RK1 is active.
    root_scalar_flux: dict = {}
    f2_pd_context = None
    if carry.census is not None and getattr(carry.census, "f2_qc_packet", None) is not None:
        f2_pd_context = dict(physical_origin=physical_origin,
            boundary_lead_seconds=boundary_lead_seconds, physics_lead_seconds=physics_lead_seconds)
    rk_result = _rk_scan_step(
        carry,
        namelist_fw,
        debug=debug,
        lead_seconds=boundary_lead_seconds,
        physics_tendencies=physics_dry_tendencies,
        capture_rca=capture_rca,
        capture_phase_tap=capture_phase_tap,
        capture_ladder=capture_ladder,
        root_scalar_flux_out=root_scalar_flux,
        **({"f2_pd_out": f2_pd_context} if f2_pd_context is not None else {}),
    )
    acoustic_health = None
    acoustic_target = None
    phase_tap_summary = None
    ladder_partial = None
    if capture_rca:
        acoustic_health = rk_result.acoustic_health
        acoustic_target = rk_result.acoustic_target
        carry = rk_result.carry
    elif capture_phase_tap:
        phase_tap_summary = rk_result.summary
        carry = rk_result.carry
    elif capture_ladder:
        ladder_partial = rk_result
        carry = rk_result.carry
    else:
        carry = rk_result
    next_state = carry.state
    carry = carry.replace(census=observe_f2_state(
        carry.census, "post_rk", next_state, step_index,
        **({"pd_context": f2_pd_context} if f2_pd_context is not None else {})))
    if (
        _phys_tend_rk_wrf_enabled()
        and bool(physics_forcing.enabled)
        and int(namelist.rad_rk_tendf) != 0
        and carry.rthraten is not None
    ):
        carry = carry.replace(rthraten=_decouple_held_rthraten(
            carry.rthraten, physical_origin.mu_total, next_state.mu_total, namelist.metrics))
    if capture_rca:
        state_health.append(_rca_state_health(next_state))
        state_target.append(_rca_state_target(next_state))
    if bool(physics_forcing.enabled):
        next_state = _apply_physics_non_dry_updates(next_state, physical_origin, physics_forcing.state)
        carry = carry.replace(state=next_state)
    f2_post_nondry_qc = next_state.qc if f2_pd_context is not None else None
    from gpuwrf.coupling.kf_rk import kf_tend_rk_enabled, finish_kf_rates
    if kf_tend_rk_enabled() and int(namelist.cu_physics) == 1 and bool(physics_forcing.enabled):
        rates, nca, rain_increment = finish_kf_rates(
            carry.cumulus_tendencies, carry.cumulus_carry[1], next_state.mu_total,
            namelist.metrics, float(namelist.dt_s))
        next_state = next_state.replace(
            rainc_acc=(next_state.rainc_acc + rain_increment).astype(next_state.rainc_acc.dtype))
        carry = carry.replace(state=next_state,
            cumulus_carry=(carry.cumulus_carry[0], nca), cumulus_tendencies=rates)
    if root_scalar_flux.get("flow"):
        # B43: WRF's RK3 flow_dep_bdy copies an interior that already holds the
        # physics tendencies (moist_tend), so repeat it on the post-physics
        # state with the RK3 ru_m/rv_m, before microphysics (which skips the
        # spec zone, solve_em.F:3693-3707).
        next_state = _root_flow_dep(next_state, root_scalar_flux["flow"], root_scalar_flux["ru_m"],
                                    root_scalar_flux["rv_m"], namelist.boundary_config)
        carry = carry.replace(state=next_state)
    if f2_pd_context is not None:
        from gpuwrf.diagnostics.f2_scalar_pd_packet import finish_seam
        carry = carry.replace(census=carry.census._replace(f2_qc_packet=finish_seam(
            carry.census.f2_qc_packet, step_index, f2_post_nondry_qc, next_state.qc)))
    if _microphysics_wrf_order_enabled():
        pre_microphysics_theta = next_state.theta
        if carry.noahmp_precipitation is not None:
            next_state, precip = _apply_post_rk_microphysics(
                next_state, namelist, return_precipitation=True,
                **({"step_index": step_index} if getattr(namelist, "mp_physics", None) == 18 else {}))
            from gpuwrf.runtime.noahmp_precipitation import precipitation_from_step
            rates = carry.cumulus_tendencies
            convective_rate = jnp.zeros_like(next_state.t_skin) if rates is None else rates[6]
            carry = carry.replace(noahmp_precipitation=precipitation_from_step(
                precip, convective_rate, float(namelist.dt_s),
                spec_zone=_microphysics_spec_zone(namelist)))
        else:
            next_state = _apply_post_rk_microphysics(
                next_state, namelist, **({"step_index": step_index} if getattr(namelist, "mp_physics", None) == 18 else {}))
        carry = carry.replace(state=next_state)
        if carry.h_diabatic is not None:
            # moist_physics_finish_em (use_theta_m=1): h_diabatic*dt is exactly the
            # theta_m increment of the microphysics call (:5735-5741).
            carry = carry.replace(h_diabatic=(
                (next_state.theta - pre_microphysics_theta) / float(namelist.dt_s)
            ).astype(carry.h_diabatic.dtype))
    if capture_rca:
        state_health.append(_rca_state_health(next_state))
        state_target.append(_rca_state_target(next_state))
    limiter_diagnostics = _empty_theta_limiter_diagnostics(next_state.theta)
    # Census observes the guard candidates whether or not the guards repair them.
    carry = carry.replace(census=observe_f2_state(
        carry.census, "pre_dynamics_guard", next_state, step_index))
    carry = carry.replace(census=count_dynamics_guards(
        carry.census, next_state, theta_min=_THETA_LIMITER_MIN_K,
        theta_max=_THETA_LIMITER_MAX_K,
    ))
    if not bool(namelist.disable_guards):
        guard_before = next_state
        next_state, limiter_diagnostics = _limit_guarded_dynamics_state_with_diagnostics(next_state, physical_origin)
        next_state = next_state.replace(
            qv=_valid_mixing_ratio(next_state.qv, physical_origin.qv),
            qc=_valid_mixing_ratio(next_state.qc, physical_origin.qc),
            qr=_valid_mixing_ratio(next_state.qr, physical_origin.qr),
            qi=_valid_mixing_ratio(next_state.qi, physical_origin.qi),
            qs=_valid_mixing_ratio(next_state.qs, physical_origin.qs),
            qg=_valid_mixing_ratio(next_state.qg, physical_origin.qg),
        )
        if carry.census is not None and carry.census.water_changes is not None:
            carry = carry.replace(census=count_water_repair(
                carry.census, "dynamics", guard_before, next_state,
                guard_dry_mass_kg(next_state, namelist)))
    if capture_rca:
        state_health.append(_rca_state_health(next_state))
        state_target.append(_rca_state_target(next_state))
    if capture_ladder:
        # WRF L5 rung: state immediately before the end-of-step boundary pass
        # (solve_em.F immediately before the spec_bdy_final IF block).
        pre_bdry_u = next_state.u
        pre_bdry_v = next_state.v
    if bool(namelist.run_boundary):
        # WRF cadence: in-loop pins + frozen RK1 relax own the dry boundary band.
        # Released specified/replay paths retain their ring-0 end sync.  The
        # exact live-nest candidate performs no dry end-step overwrite and only
        # consumes the coupled moist/scalar records (WRF never forces p'/pb).
        bounded = apply_lateral_boundaries(
            next_state, boundary_lead_seconds, float(namelist.dt_s), namelist.boundary_config, namelist.metrics,
            dry_spec_only=(
                _specified_bdy_cadence_active(namelist)
                or _nested_frozen_wrf_boundary_active(namelist)
            ),
            # Guards-disabled (strict) runs keep WRF's unclamped moist/scalar
            # boundary result; census then observes the true values.
            positivity_floor=not bool(namelist.disable_guards),
            root_scalar_rk1=(
                (root_scalar_flux["relaxed"], root_scalar_flux["flow"]) if root_scalar_flux else None
            ),
        )
        carry = carry.replace(census=observe_f2_state(
            carry.census, "pre_boundary_guard", bounded, step_index))
        carry = carry.replace(census=count_boundary_guards(carry.census, bounded))
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
            if carry.census is not None and carry.census.water_changes is not None:
                carry = carry.replace(census=count_water_repair(
                    carry.census, "boundary", bounded, next_state,
                    guard_dry_mass_kg(next_state, namelist)))
    if _W_SURFACE_RESET:
        # WRF solve_em.F:4818-4834: after all RK stages, the microphysics and the boundary
        # updates, every step ends with set_w_surface(fill_w_flag=.false.) from the final u/v.
        if _SHARDED_CARRY_HALO_CONTEXT is not None:
            raise NotImplementedError("GPUWRF_W_SURFACE_RESET does not support the sharded carry")
        next_state = next_state.replace(w=_reset_surface_w(
            next_state, namelist.grid, namelist.metrics,
            periodic_x=_acoustic_lateral_bc_flags(namelist)[0]))
    if capture_rca:
        state_health.append(_rca_state_health(next_state))
        state_target.append(_rca_state_target(next_state))
    next_state = _enforce_operational_precision(
        next_state,
        force_fp64=bool(namelist.force_fp64),
        acoustic_precision_mode=namelist.acoustic_precision_mode,
        base_state=carry.base_state,
    )
    final_carry = _maybe_exchange_sharded_carry_halos(carry.replace(state=next_state))
    final_carry = final_carry.replace(census=count_work(final_carry.census, "steps"))
    if capture_rca:
        state_health.append(_rca_state_health(next_state))
        state_target.append(_rca_state_target(next_state))
        assert acoustic_health is not None
        assert acoustic_target is not None
        assert boundary_health is not None
        return _RcaStepResult(
            final_carry,
            RcaStepRecord(
                acoustic_health,
                acoustic_target,
                jnp.stack(tuple(state_health), axis=0),
                jnp.stack(tuple(state_target), axis=0),
                boundary_health,
            ),
        )
    if capture_phase_tap:
        assert phase_tap_summary is not None
        return CorrectedNiPhaseTapResult(final_carry, phase_tap_summary)
    if capture_first_interval and capture_ladder:
        assert first_interval_record is not None
        assert ladder_partial is not None
        return FirstIntervalLadderStepResult(
            final_carry,
            first_interval_record,
            FirstIntervalLadderRecord(
                rk1_tend_u=ladder_partial.rk1_tend_u,
                rk1_tend_v=ladder_partial.rk1_tend_v,
                rk1_relax_u=ladder_partial.rk1_relax_u,
                rk1_relax_v=ladder_partial.rk1_relax_v,
                rk1_fin_u=ladder_partial.rk1_fin_u,
                rk1_fin_v=ladder_partial.rk1_fin_v,
                rk2_fin_u=ladder_partial.rk2_fin_u,
                rk2_fin_v=ladder_partial.rk2_fin_v,
                rk3_fin_u=ladder_partial.rk3_fin_u,
                rk3_fin_v=ladder_partial.rk3_fin_v,
                pre_bdry_u=pre_bdry_u,
                pre_bdry_v=pre_bdry_v,
            ),
        )
    if capture_first_interval:
        assert first_interval_record is not None
        return FirstIntervalStepResult(final_carry, first_interval_record)
    return final_carry, limiter_diagnostics


def _physics_boundary_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
    clock_base=None,
) -> OperationalCarry:
    next_carry, _diagnostics = _physics_boundary_step_with_limiter_diagnostics(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        debug=debug,
        clock_base=clock_base,
    )
    if os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1":
        # Native RK scratch is REAL32; the retained scan carries its input ABI.
        # Widen after all native arithmetic so each computed value stays exact.
        scratch = ("t_2ave", "ww", "u_save", "v_save", "w_save", "t_save", "ww_save")
        next_carry = next_carry.replace(**{
            name: getattr(next_carry, name).astype(getattr(carry, name).dtype)
            for name in scratch
        })
    if os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1":
        from gpuwrf.kernels.dyn_carry_fp32 import (
            REAL_ALL_CARRY_FIELDS, SCRATCH_REAL_FIELDS, like, real_all_enabled,
        )
        next_carry = next_carry.replace(**{
            name: jnp.asarray(getattr(next_carry, name), dtype=getattr(carry, name).dtype)
            for name in SCRATCH_REAL_FIELDS
        })
        if real_all_enabled():
            # Held radiation / KF leaves keep their seeded REAL dtype across the step.
            next_carry = next_carry.replace(**{
                name: like(getattr(next_carry, name), getattr(carry, name))
                for name in REAL_ALL_CARRY_FIELDS
            })
    return next_carry


def _physics_boundary_step_with_rca(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    clock_base=None,
) -> _RcaStepResult:
    """Run one normal step while returning bounded device-only RCA summaries."""

    result = _physics_boundary_step_with_limiter_diagnostics(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        debug=False,
        clock_base=clock_base,
        capture_rca=True,
    )
    return result


def _physics_boundary_step_with_phase_tap(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    clock_base=None,
) -> CorrectedNiPhaseTapResult:
    """Run one normal step with only the authorized RK3/substep-1 phase tap."""

    result = _physics_boundary_step_with_limiter_diagnostics(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        debug=False,
        clock_base=clock_base,
        capture_phase_tap=True,
    )
    return result


def _physics_boundary_step_with_first_interval(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    clock_base=None,
) -> FirstIntervalStepResult:
    """Run one normal step with the first-interval momentum savepoint capture."""

    result = _physics_boundary_step_with_limiter_diagnostics(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        debug=False,
        clock_base=clock_base,
        capture_first_interval=True,
    )
    return result


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


_M9_RRTMG_COLUMN_TILE_COLS = 512


def _history_pblh(state: State, grid) -> jax.Array:
    """PBLH for in-step history output: the in-step MYNN value once State carries it.

    Until then this is the existing post-step MYNN diagnosis (same calls as
    ``surface_layer_diagnostics``) without the surface-layer re-solve.
    """
    if "pblh" in State.__slots__:
        return state.pblh
    from gpuwrf.coupling.physics_couplers import (  # noqa: PLC0415
        _mynn_column_from_state,
        _mynn_pblh_for_output,
        _surface_fluxes_from_state,
    )

    return _mynn_pblh_for_output(_mynn_column_from_state(state, grid), _surface_fluxes_from_state(state))


def _psfc_from_state(state: State, metrics: DycoreMetrics) -> jax.Array:
    """Surface pressure (Pa) = WRF runtime PSFC = moist hydrostatic p_hyd_w(kts).

    WRF's history PSFC is NOT a height extrapolation of the nonhydrostatic
    total pressure. The surface driver sets ``PSFC(i,j) = p8w(i,kts,j)``
    (module_surface_driver.F:1988) and is called with ``P8W = grid%p_hyd_w``
    (module_first_rk_step_part1.F:1400); ``p_hyd_w`` is built in ``phy_prep``
    (module_big_step_utilities_em.F:4946-4958) by integrating the MOIST
    hydrostatic column in the hybrid dry-mass coordinate from the model top::

        p_hyd_w(kte) = p_top
        qtot(k)      = sum over ALL moist species of q(k)
        p_hyd_w(k)   = p_hyd_w(k+1) - (1+qtot(k))*(c1h(k)*MUT + c2h(k))*dnw(k)

    so ``PSFC = p_top + sum_k (1+qtot_k)*(c1h_k*MUT + c2h_k)*(-dnw_k)`` with
    ``MUT = mu + mub`` the FULL dry column mass (``State.mu_total``).

    The previous height extrapolation of ``p_total`` (phy_prep's OTHER ``p8w``
    field, module_big_step_utilities_em.F:4917-4922 — never the one bound to
    PSFC at runtime) matches WRF only to ~14 Pa on a moist-consistent pressure
    state, and under-reports PSFC by the full vapor column load (~200-230 Pa,
    the v0.14 fixed-Canary PSFC floor) on the current dry-balanced GPU pressure
    state. The dry-mass + moisture integral is the exact WRF diagnostic path:
    CPU-truth residual RMSE <= 0.18 Pa across h1/h4/h10/h24
    (proofs/v014/psfc_moist_pressure_state_closure.*).

    Constant memory per column; runs inside the jitted M9 snapshot with
    device-resident ``metrics`` leaves (no timestep-loop host transfer).
    """
    qtot = state.qv + state.qc + state.qr + state.qi + state.qs + state.qg
    dp_dry = (
        metrics.c1h[:, None, None] * state.mu_total[None, :, :]
        + metrics.c2h[:, None, None]
    ) * (-metrics.dnw[:, None, None])
    p_top = jnp.reshape(jnp.asarray(metrics.p_top), ())
    column = (1.0 + qtot) * dp_dry
    # v0.20 fp32 INTEGRATION bit-identity fix (secondary source): the S4 merge
    # introduced Kahan compensated summation here, which rounds differently from
    # the historical plain .sum(axis=0) and broke fp64_default PSFC bit-identity.
    # Kahan is only needed for the perturbation-authoritative fp32 mode; gate on
    # the perturbation storage dtype so fp64_default re-emits the exact pre-S4 sum
    # (compile-time static -> zero runtime cost, byte-identical HLO).
    if jnp.dtype(jnp.asarray(state.mu_perturbation).dtype) == jnp.dtype(jnp.float32):
        return p_top + _kahan_sum_vertical(column)
    return p_top + column.sum(axis=0)


_HISTORY_SWDOWN_HORIZONTAL = os.environ.get("GPUWRF_HISTORY_SWDOWN_HORIZONTAL", "0") == "1"


def compute_m9_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    lead_seconds,
    *,
    noahmp_land=None,
    noahmp_rad=None,
    noahclassic_land=None,
    clock_base=None,
    radiation_diagnostics=None,
    surface_diagnostics=None,
    history_diagnostics=None,
    _with_radiation_count: bool = False,
) -> M9Diagnostics:
    """Recompute the M9 surface map from a post-step State (side-channel only).

    ``history_diagnostics`` (carry.history_diagnostics, GPUWRF_HISTORY_INSTEP_DIAG)
    replaces the post-step surface re-solve: HFX/LH/T2/U10/V10/PSFC are the last
    step's in-step values and TSK is ``state.t_skin``, as WRF writes them.

    When Noah-MP is activated (``namelist.use_noahmp`` and ``noahmp_land`` given),
    the LAND HFX/LH/TSK and the 2-m T2 are read back from the prognostic Noah-MP
    coupler and overlaid (the standalone-replacement contract); ocean/water keeps
    the bulk surface-layer diagnostic. The land T2 is the Noah-MP LSM diagnostic
    ``T2 = FVEG*T2MV + (1-FVEG)*T2MB`` (the faithful overwrite WRF performs over
    land — module_surface_driver.F:3469-3473), NOT the surface-layer MYNN 2-m value.
    U10/V10 come from the bulk surface layer, which already uses the Noah-MP skin
    temperature (in ``state.t_skin``) as its BC.
    """
    # Full-output callers also need the Q2 lookup on this same surface result.
    # Reuse it without changing the eager arithmetic or its fusion boundaries.
    history = history_diagnostics
    surf = (
        None
        if history is not None
        else surface_diagnostics
        if surface_diagnostics is not None
        else surface_layer_diagnostics(state, namelist.grid)
    )
    radiation_land = noahmp_land if bool(namelist.use_noahmp) else None
    rad = radiation_diagnostics
    radiation_calls = 0
    if rad is None:
        rad = rrtmg_radiation_diagnostics(
            state,
            namelist.grid,
            time_utc=namelist.time_utc,
            lead_seconds=lead_seconds,
            clock_base=_rad_clock_base(clock_base),
            radiation_static=namelist.radiation_static,
            topo_shading=int(namelist.topo_shading),
            slope_rad=int(namelist.slope_rad),
            shadow_length_m=float(namelist.topo_shadow_length_m),
            land_state=radiation_land,
            column_tile_cols=_M9_RRTMG_COLUMN_TILE_COLS,
            _m9_flux_slices_only=True,
            use_mp_re=int(mp_re_active(namelist)),
        )
        # This output solve executed; return its count with the live diagnostics.
        radiation_calls += 1
    if history is not None:
        hfx = state.hfx if history.hfx is None else history.hfx
        lh, tsk, t2 = history.lh, state.t_skin, history.t2
    else:
        hfx, lh, tsk, t2 = surf.hfx, surf.lh, state.t_skin, surf.t2
    if history is None and bool(namelist.use_noahmp) and noahmp_land is not None:
        clock = _noahmp_clock(namelist, clock_base, lead_seconds, output_time=True)
        radiation = (
            _NoahMPRadiation(*noahmp_rad) if noahmp_rad is not None
            else _NoahMPRadiation(rad.swnorm, rad.glw, rad.coszen)
        )
        ep, rp = _noahmp_params(namelist)
        # v0.9.0: route the Noah-MP LSM 2-m T2 over land (the faithful land-T2
        # overwrite WRF performs — module_surface_driver.F:3469-3473), replacing
        # the surface-layer MYNN 2-m value over land. Water keeps surf.t2.
        hfx, lh, tsk, t2 = overlay_noahmp_land_diagnostics(
            state, noahmp_land, _lane_noahmp_static(namelist, clock_base), surf.hfx, surf.lh, state.t_skin,
            float(namelist.dt_s), bulk_t2=surf.t2, radiation=radiation, clock=clock,
            energy_params=ep, rad_params=rp,
        )
    elif history is None and _explicit_noahclassic(namelist) and noahclassic_land is not None:
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
    if _HISTORY_SWDOWN_HORIZONTAL:
        # WRF surface_driver restores horizontal SWDOWN after the LSM; the
        # slope-adjusted forcing remains SWNORM. Only history changes here:
        # Noah still receives the unchanged held noahmp_rad (rad.swnorm).
        swdown_out = rad.swdown
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
    diag = M9Diagnostics(
        swdown=swdown_out,
        glw=glw_out,
        hfx=hfx,
        lh=lh,
        pblh=surf.pblh if history is None else _history_pblh(state, namelist.grid),
        tsk=tsk,
        t2=t2,
        u10=surf.u10 if history is None else history.u10,
        v10=surf.v10 if history is None else history.v10,
        psfc=_psfc_from_state(state, namelist.metrics) if history is None else history.psfc,
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

    return (diag, radiation_calls) if _with_radiation_count else diag


@partial(jax.jit, static_argnames=("attrs", "_with_radiation_count"))
def compute_m9_selected_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    lead_seconds,
    clock_base,
    attrs: tuple[str, ...],
    *,
    noahmp_land=None,
    noahmp_rad=None,
    noahclassic_land=None,
    radiation_diagnostics=None,
    _with_radiation_count: bool = False,
) -> tuple[jax.Array, ...]:
    """Compute only selected M9 leaves so output-subset HLO can DCE the rest."""

    diag = compute_m9_diagnostics(
        state,
        namelist,
        lead_seconds,
        noahmp_land=noahmp_land,
        noahmp_rad=noahmp_rad,
        noahclassic_land=noahclassic_land,
        clock_base=clock_base,
        radiation_diagnostics=radiation_diagnostics,
        _with_radiation_count=_with_radiation_count,
    )
    if _with_radiation_count:
        diag, radiation_calls = diag
        return tuple(getattr(diag, attr) for attr in attrs), radiation_calls
    return tuple(getattr(diag, attr) for attr in attrs)


def _advance_chunk_loop_mode() -> str:
    """Select the compiled loop lowering for `_advance_chunk`.

    Default is the traced-count fori_loop path.  The static scan path is kept as
    an opt-in diagnostic because it was proven to make tiny all-7 leaf domains
    pathologically slow.
    """

    return os.environ.get("GPUWRF_ADVANCE_CHUNK_LOOP", "fori").strip().lower()


@jax.jit
def _advance_chunk_fori(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    start_step,
    clock_base=None,
    *,
    n_steps: int,
    cadence: int,
) -> OperationalCarry:
    """Advance one output interval as a SINGLE compiled loop (no diagnostics).

    Radiation is gated by the traced ``(step_index - 1) %% cadence == 0`` predicate via
    ``_physics_boundary_step``'s cond path, so this is byte-identical to the
    production per-step cadence.  ``start_step``, ``n_steps``, and the explicit
    ``cadence`` argument are TRACED scalars, so interval-length/cadence variation at
    the caller does not mint a new XLA cache key.  Kept SEPARATE from the diagnostics
    call so the dynamics scratch is freed before the large RRTMG diagnostic transient
    is allocated (peak-memory bound, Task 2 OOM fix).
    """
    run_physics = bool(namelist.run_physics)
    start_step = jnp.asarray(start_step, dtype=jnp.int32)
    n_steps = jnp.asarray(n_steps, dtype=jnp.int32)
    cadence = jnp.asarray(cadence, dtype=jnp.int32)
    if (os.environ.get("GPUWRF_DYN_METRICS_FP32", "0") == "1"
            and os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1"):
        # WRF stores grid metrics as REAL; cast once per call (outside the loop)
        # so mixed REAL-carry glue no longer promotes to fp64.
        from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
        namelist = dataclass_replace(namelist, metrics=real_metrics(namelist.metrics))

    def body(offset, scan_carry: OperationalCarry):
        step_index = start_step + offset
        if run_physics:
            run_radiation = jnp.equal(jnp.mod(step_index - 1, cadence), 0)
        else:
            run_radiation = False
        return _layout_pin("carry", _physics_boundary_step(
            _layout_pin("carry", scan_carry), namelist, step_index, run_radiation=run_radiation,
            debug=False, clock_base=clock_base,
        ))

    from gpuwrf.kernels.ring_select import nested_step

    with nested_step(_acoustic_lateral_bc_flags(namelist)[2]):  # ring-scatter layout pin (nested only)
        return jax.lax.fori_loop(
            jnp.asarray(0, dtype=jnp.int32),
            n_steps,
            body,
            carry,
        )


# Resolved at import (registered in aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS).
_CARRY_DONATE = os.environ.get("GPUWRF_CARRY_DONATE", "0") == "1"
if _CARRY_DONATE:
    # Default-off: donate the carry so the step loop reuses the input buffers
    # instead of copying every non-donated parameter per call. Callers must not
    # read a pre-advance carry afterwards (misuse raises "Array has been deleted").
    _advance_chunk_fori = jax.jit(_advance_chunk_fori.__wrapped__, donate_argnums=(0,))


def _donation_buffer_key(leaf: Any) -> Any:
    try:
        return leaf.unsafe_buffer_pointer()
    except Exception:  # noqa: BLE001 - deleted or multi-device: fall back to identity
        return ("id", id(leaf))


def unalias_donated_carry(carry: Any, *, seen: set | None = None, by_buffer: bool = False) -> Any:
    """Give every carry buffer one owner before a donating advance.

    The initial carry holds some State arrays at two positions, and PjRt refuses
    "Attempt to donate the same buffer twice" (dispatch D13). Repeats are copied
    once. ``by_buffer`` also catches distinct arrays that share one buffer (a
    device_put of an uncommitted array is a committed alias); it costs a pointer
    query per leaf, so callers use it only where carries enter a run (``seen``
    shared across domains). Executable outputs are distinct buffers.
    """
    if not _CARRY_DONATE:
        return carry
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return carry  # inside a trace (vmapped lanes): donation applies only at top-level dispatch
    seen = set() if seen is None else seen
    repeated = False
    for index, leaf in enumerate(leaves):
        if isinstance(leaf, jax.Array):
            key = _donation_buffer_key(leaf) if by_buffer else id(leaf)
            if key in seen:
                leaves[index] = jnp.copy(leaf)
                repeated = True
            else:
                seen.add(key)
    return jax.tree_util.tree_unflatten(treedef, leaves) if repeated else carry


@partial(jax.jit, static_argnames=("n_steps", "cadence"))
def _advance_chunk_static_scan(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    start_step,
    clock_base=None,
    *,
    n_steps: int,
    cadence: int,
) -> OperationalCarry:
    """Opt-in static-scan lowering retained for diagnostics/repro only."""

    run_physics = bool(namelist.run_physics)
    start_step = jnp.asarray(start_step, dtype=jnp.int32)
    indices = start_step + jnp.arange(int(n_steps), dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        if run_physics:
            run_radiation = jnp.equal(jnp.mod(step_index - 1, int(cadence)), 0)
        else:
            run_radiation = False
        next_carry = _physics_boundary_step(
            scan_carry, namelist, step_index, run_radiation=run_radiation,
            debug=False, clock_base=clock_base,
        )
        return next_carry, None

    carry, _ = jax.lax.scan(body, carry, indices)
    return carry


def _advance_chunk(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    start_step,
    clock_base=None,
    *,
    n_steps: int,
    cadence: int,
) -> OperationalCarry:
    """Advance one output interval; default to the fast traced-count loop.

    ``clock_base`` (a TRACED :class:`_ClockBase` built once on the host via
    :func:`build_clock_base`) carries the per-run date scalars as runtime inputs
    so the compiled HLO is date-independent (#91 cross-date cache hit). ``None``
    keeps the legacy host-extraction-from-``namelist.time_utc`` behaviour.
    """

    # Every runner reaches the donating jit through here (single-domain loops, tree fallbacks).
    carry = unalias_donated_carry(carry, by_buffer=True)
    mode = _advance_chunk_loop_mode()
    if mode in {"scan", "static_scan", "static-scan"}:
        return _advance_chunk_static_scan(
            carry, namelist, start_step, clock_base, n_steps=int(n_steps), cadence=int(cadence)
        )
    if mode not in {"", "fori", "fori_loop", "fori-loop"}:
        raise ValueError(
            "GPUWRF_ADVANCE_CHUNK_LOOP must be 'fori' (default) or 'static_scan'; "
            f"got {mode!r}"
        )
    return _advance_chunk_fori(
        carry, namelist, start_step, clock_base, n_steps=n_steps, cadence=cadence
    )


@jax.jit
def advance_one_step_with_corrected_ni_phase_tap(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    clock_base=None,
    *,
    cadence: int,
) -> CorrectedNiPhaseTapResult:
    """Proof-only one-step option-2 tap with the ordinary radiation predicate."""

    step_index = jnp.asarray(step_index, dtype=jnp.int32)
    cadence = jnp.asarray(cadence, dtype=jnp.int32)
    if bool(namelist.run_physics):
        run_radiation = jnp.equal(jnp.mod(step_index - 1, cadence), 0)
    else:
        run_radiation = False
    return _physics_boundary_step_with_phase_tap(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        clock_base=clock_base,
    )


@jax.jit
def advance_one_step_with_first_interval_capture(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    clock_base=None,
    *,
    cadence: int,
) -> FirstIntervalStepResult:
    """Proof-only one-step first-interval momentum savepoint capture.

    Diagnostic-only entry point for the v0234 first-interval momentum
    isolation sprint: executes the ordinary step and additionally returns the
    raw MYNN RUBLTEN/RVBLTEN and the assembled ru_tendf/rv_tendf physics
    momentum tendencies.  It is a separate jit program; the production
    one-step program is untouched and byte-identical.
    """

    step_index = jnp.asarray(step_index, dtype=jnp.int32)
    cadence = jnp.asarray(cadence, dtype=jnp.int32)
    if bool(namelist.run_physics):
        run_radiation = jnp.equal(jnp.mod(step_index - 1, cadence), 0)
    else:
        run_radiation = False
    return _physics_boundary_step_with_first_interval(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        clock_base=clock_base,
    )


def _physics_boundary_step_with_first_interval_ladder(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    clock_base=None,
) -> FirstIntervalLadderStepResult:
    """Run one normal step with the first-interval + suboperator-ladder capture."""

    return _physics_boundary_step_with_limiter_diagnostics(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        debug=False,
        clock_base=clock_base,
        capture_first_interval=True,
        capture_ladder=True,
    )


@jax.jit
def advance_one_step_with_first_interval_ladder(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    clock_base=None,
    *,
    cadence: int,
) -> FirstIntervalLadderStepResult:
    """Proof-only one-step d03 dry-dycore/nest suboperator-ladder capture.

    Diagnostic-only entry point for the v0234 dycore-suboperator sprint:
    executes the ordinary step and additionally returns the first-interval
    momentum record plus the source-bound ladder record (RK1 merged tendency,
    RK1 boundary-relax bundle, per-RK-stage finished momentum, and the state
    just before the end-of-step boundary pass).  It is a separate jit
    program; the production one-step program is untouched and byte-identical.
    """

    step_index = jnp.asarray(step_index, dtype=jnp.int32)
    cadence = jnp.asarray(cadence, dtype=jnp.int32)
    if bool(namelist.run_physics):
        run_radiation = jnp.equal(jnp.mod(step_index - 1, cadence), 0)
    else:
        run_radiation = False
    return _physics_boundary_step_with_first_interval_ladder(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        clock_base=clock_base,
    )


@partial(jax.jit, static_argnames=("n_steps", "cadence"))
def advance_chunk_with_corrected_ni_rca(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    start_step,
    clock_base=None,
    *,
    n_steps: int,
    cadence: int,
) -> RcaChunkResult:
    """Proof-only chunk: canonical step outputs plus bounded on-device records."""

    run_physics = bool(namelist.run_physics)
    start_step = jnp.asarray(start_step, dtype=jnp.int32)
    indices = start_step + jnp.arange(int(n_steps), dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        if run_physics:
            run_radiation = jnp.equal(jnp.mod(step_index - 1, int(cadence)), 0)
        else:
            run_radiation = False
        result = _physics_boundary_step_with_rca(
            scan_carry,
            namelist,
            step_index,
            run_radiation=run_radiation,
            clock_base=clock_base,
        )
        return result.carry, result.record

    next_carry, records = jax.lax.scan(body, carry, indices)
    return RcaChunkResult(next_carry, records)


@jax.jit
def _m9_snapshot(
    carry: OperationalCarry, namelist: OperationalNamelist, lead_seconds, clock_base=None
) -> M9Diagnostics:
    """Compute the M9 surface map once from a post-chunk State (separate program).

    Isolated in its own ``jax.jit`` so XLA cannot co-schedule the ~15 GiB RRTMG
    g-point diagnostic transient with the dynamics-chunk scratch; the host loop
    blocks after the chunk so the chunk scratch is freed first.

    ``clock_base`` (traced :class:`_ClockBase`) makes the diagnostic HLO
    date-independent too (#91); ``None`` keeps the legacy host-extraction path.
    """
    return compute_m9_diagnostics(
        carry.state, namelist, lead_seconds,
        noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad,
        noahclassic_land=carry.noahclassic_land,
        clock_base=clock_base,
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
    # #91: build the per-run date scalars ONCE on the host and thread them into the
    # jitted chunk + snapshot as TRACED inputs so the compiled HLO is date-independent
    # (cross-date compile-cache hit; numerically identical to the baked-literal path).
    clock_base = build_clock_base(namelist)
    diag_chunks: list[M9Diagnostics] = []
    start = 1
    for end in boundaries:
        n = end - start + 1
        carry = _advance_chunk(
            carry, namelist, jnp.asarray(start, dtype=jnp.int32), clock_base,
            n_steps=n, cadence=cadence,
        )
        # Free the dynamics-chunk scratch BEFORE the RRTMG diagnostic transient is
        # allocated, then free the transient before the next chunk -- this is what
        # bounds peak memory to (working set + ONE transient) for any forecast length.
        jax.block_until_ready(carry.state.theta)
        diag = _m9_snapshot(
            carry, namelist, jnp.asarray(float(end) * dt_s, dtype=jnp.float64), clock_base
        )
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


def _m0_evidence_config_from_env() -> dict[str, object]:
    """Validate the private M0 evidence envelope before integration starts."""

    import re
    from pathlib import Path

    required = {
        "sidecar_path": _M0_EVIDENCE_PATH,
        "run_id": _M0_EVIDENCE_RUN_ID,
        "source_sha256": _M0_EVIDENCE_SOURCE_SHA256,
        "config_sha256": _M0_EVIDENCE_CONFIG_SHA256,
        "input_manifest_sha256": _M0_EVIDENCE_INPUT_SHA256,
        "device_uuid": _M0_EVIDENCE_DEVICE_UUID,
    }
    values: dict[str, str] = {}
    for field, variable in required.items():
        value = os.environ.get(variable)
        if value is None or not value.strip():
            raise RuntimeError(
                f"{_M0_EVIDENCE_FLAG}=1 requires non-empty {variable}"
            )
        values[field] = value.strip()

    if not re.fullmatch(_M0_EVIDENCE_RUN_ID_PATTERN, values["run_id"]):
        raise RuntimeError(
            f"{_M0_EVIDENCE_RUN_ID} must be 8-128 filename-safe characters"
        )
    for field in ("source_sha256", "config_sha256", "input_manifest_sha256"):
        if not re.fullmatch(_M0_EVIDENCE_SHA256_PATTERN, values[field]):
            raise RuntimeError(
                f"{required[field]} must be a lowercase full SHA-256 digest"
            )
    if not re.fullmatch(_M0_EVIDENCE_DEVICE_UUID_PATTERN, values["device_uuid"]):
        raise RuntimeError(
            f"{_M0_EVIDENCE_DEVICE_UUID} must be a full GPU- UUID"
        )

    sidecar_path = Path(values.pop("sidecar_path"))
    if not sidecar_path.is_absolute():
        raise RuntimeError(f"{_M0_EVIDENCE_PATH} must be an absolute path")
    if not sidecar_path.parent.is_dir():
        raise RuntimeError(
            f"{_M0_EVIDENCE_PATH} parent must already exist: {sidecar_path.parent}"
        )
    if os.path.lexists(sidecar_path):
        raise RuntimeError(
            f"refusing to overwrite existing M0 evidence sidecar: {sidecar_path}"
        )
    return {"sidecar_path": sidecar_path, **values}


def _m0_evidence_allocator_stats(result: State) -> dict[str, object]:
    """Read allocator peaks through a result leaf's already-active device."""

    device = None
    for leaf in jax.tree_util.tree_leaves(result):
        candidate = getattr(leaf, "device", None)
        if candidate is None:
            continue
        device = candidate() if callable(candidate) else candidate
        break
    if device is None:
        raise RuntimeError(
            "M0 evidence could not identify the forecast result's existing device"
        )

    memory_stats = getattr(device, "memory_stats", None)
    if not callable(memory_stats):
        raise RuntimeError(
            "the forecast result's existing device does not expose memory_stats()"
        )
    stats = memory_stats()
    if not isinstance(stats, dict):
        raise RuntimeError("forecast allocator memory_stats() returned no mapping")

    peaks: dict[str, int] = {}
    for key in ("peak_bytes_in_use", "peak_bytes_reserved"):
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"forecast allocator statistic {key} is missing")
        integer = int(value)
        if integer < 0:
            raise RuntimeError(f"forecast allocator statistic {key} is invalid")
        peaks[key] = integer
    if peaks["peak_bytes_reserved"] < peaks["peak_bytes_in_use"]:
        raise RuntimeError(
            "forecast allocator peak_bytes_reserved is below peak_bytes_in_use"
        )

    local_ordinal = getattr(device, "id", None)
    if callable(local_ordinal):
        local_ordinal = local_ordinal()
    platform = getattr(device, "platform", None)
    if not isinstance(platform, str) or not platform:
        raise RuntimeError("forecast result device platform is missing")
    if (
        isinstance(local_ordinal, bool)
        or not isinstance(local_ordinal, (int, str, type(None)))
    ):
        raise RuntimeError("forecast result device local ordinal is invalid")
    return {
        **peaks,
        "device_platform": platform,
        "device_local_ordinal": local_ordinal,
    }


def _m0_evidence_atomic_json(path, payload: dict[str, object]) -> None:
    """Publish a complete sidecar atomically and without replacing prior evidence."""

    import json
    import tempfile
    from pathlib import Path

    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _m0_evidence_run(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
) -> State:
    """Run one whole integration range and emit same-process allocator evidence."""

    import time
    from datetime import datetime, timezone

    evidence = _m0_evidence_config_from_env()
    prepared_state = _dealias_pytree_buffers(state)
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(timezone.utc).isoformat()
    # ADR-038: the evidence range must wrap the entry the product actually runs
    # -- run_forecast_operational_segmented with the SAME segment-length
    # resolution and positivity guard as the default-off dispatch in
    # run_forecast_operational -- not the retired monolithic escape hatch
    # (GPUWRF_FORECAST_ENTRY=monolithic), which misses the frozen cold-compile
    # gate. Enabled-vs-default numerical identity therefore holds by
    # construction: both paths run the identical default entry.
    seg = int(os.environ.get("GPUWRF_FORECAST_SEGMENT_STEPS", "34"))
    if seg <= 0:
        raise RuntimeError("GPUWRF_FORECAST_SEGMENT_STEPS must be positive")
    with jax.profiler.TraceAnnotation(_M0_EVIDENCE_RANGE):
        result = run_forecast_operational_segmented(
            prepared_state, namelist, hours, segment_steps=seg
        )
        jax.block_until_ready(result)
    finished_ns = time.monotonic_ns()
    finished_utc = datetime.now(timezone.utc).isoformat()

    allocator = _m0_evidence_allocator_stats(result)
    payload = {
        "schema": _M0_EVIDENCE_SCHEMA,
        "emitter_process_role": "forecast_process",
        "instrumentation": {
            "enabled": True,
            "opt_in_environment": f"{_M0_EVIDENCE_FLAG}=1",
            "default_when_unset": "original-direct-call-no-range-no-sidecar",
            "range_name": _M0_EVIDENCE_RANGE,
            "range_scope": "full-forecast-integration",
            "final_synchronization": "jax.block_until_ready(result)",
            "allocator_read": "after-range-on-result-device",
        },
        "run_id": evidence["run_id"],
        "forecast_pid": os.getpid(),
        "source_sha256": evidence["source_sha256"],
        "config_sha256": evidence["config_sha256"],
        "input_manifest_sha256": evidence["input_manifest_sha256"],
        "device_uuid": evidence["device_uuid"],
        "measurement_start_ns": started_ns,
        "measurement_end_ns": finished_ns,
        "measurement_start_utc": started_utc,
        "measurement_end_utc": finished_utc,
        **allocator,
    }
    _m0_evidence_atomic_json(evidence["sidecar_path"], payload)
    return result


def run_forecast_operational(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Run an operational forecast on the compile-bounded default entry (ADR-038).

    Default entry is ``run_forecast_operational_segmented`` with ``segment_steps=34``:
    compile O(one segment body) instead of O(whole forecast), bitwise-identical
    seg-vs-seg, round-off-class vs the monolithic entry (cond-vs-direct RRTMG fusion,
    ADR-038/R6), warm cost ~2%, peak RSS 20.6 -> 8.3 GiB on the FAST arm. The
    monolithic whole-forecast jit remains available as an explicit escape hatch
    (``GPUWRF_FORECAST_ENTRY=monolithic``); it violates the frozen FAST cold `<=600 s`
    (872.8 s uncapped) and cached `<=60 s` (101.7 s) product gates.

    Donate-safety is preserved on both entries: the jitted monolithic body donates
    ``state`` (``donate_argnums=(0,)``), which requires every State leaf to back a
    UNIQUE device buffer; ``_dealias_pytree_buffers`` rebinds duplicate buffers BEFORE
    the boundary (numerically identical: de-aliasing only copies a shared buffer).

    ``GPUWRF_FORECAST_SEGMENT_STEPS`` overrides the default segment length (positive
    integer; every equal-length segment reuses ONE compiled executable).
    """

    _assert_nonzero_initial_mu_total(state)
    state = _operational_scan_state(state, namelist)
    entry = os.environ.get("GPUWRF_FORECAST_ENTRY", "segmented").strip().lower()
    if entry not in ("segmented", "monolithic"):
        raise RuntimeError(
            "GPUWRF_FORECAST_ENTRY must be 'segmented' (default, ADR-038) or "
            f"'monolithic', got {entry!r}"
        )
    evidence_flag = os.environ.get(_M0_EVIDENCE_FLAG)
    if evidence_flag is None:
        if entry == "segmented":
            seg = int(os.environ.get("GPUWRF_FORECAST_SEGMENT_STEPS", "34"))
            if seg <= 0:
                raise RuntimeError("GPUWRF_FORECAST_SEGMENT_STEPS must be positive")
            return run_forecast_operational_segmented(
                _dealias_pytree_buffers(state), namelist, hours, segment_steps=seg
            )
        return _run_forecast_operational_jit(_dealias_pytree_buffers(state), namelist, hours)
    if evidence_flag != "1":
        raise RuntimeError(
            f"{_M0_EVIDENCE_FLAG} must be unset (default-off) or exactly '1'"
        )
    return _m0_evidence_run(state, namelist, hours)


def dfi_initialize_operational_state(
    state: State,
    namelist: OperationalNamelist,
    *,
    half_window_steps: int = 3,
    cutoff_s: float = 3600.0,
    filter_id: int = 1,
) -> State:
    """Apply a forward digital-filter initialization launch.

    This is the v0.22 G1 resident DFI path.  It uses the WRF ``dfcoef`` filter
    family and accumulates filtered prognostic state samples on device.  The
    current port exposes the practical forward-DFI half of WRF's
    ``dfi_fwd_init``; the full backward+forward ``dfi_bck_init`` choreography
    still needs reverse-time boundary/timekeeping support.
    """

    _resolve_operational_suite(namelist)
    state = _operational_scan_state(state, namelist)
    cfg = DigitalFilterConfig(
        half_window_steps=int(half_window_steps),
        dt_s=float(namelist.dt_s),
        cutoff_s=float(cutoff_s),
        filter_id=int(filter_id),
    )
    one_step_hours = float(namelist.dt_s) / 3600.0

    def advance_one(sample: State) -> State:
        return run_forecast_operational_segmented(
            sample,
            namelist,
            one_step_hours,
            segment_steps=1,
        )

    return digital_filter_initialize(state, advance_one, cfg)


def run_forecast_operational_dfi(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
    *,
    half_window_steps: int = 3,
    cutoff_s: float = 3600.0,
    filter_id: int = 1,
) -> State:
    """DFI-initialize and then launch the normal operational forecast."""

    initialized = dfi_initialize_operational_state(
        state,
        namelist,
        half_window_steps=half_window_steps,
        cutoff_s=cutoff_s,
        filter_id=filter_id,
    )
    return run_forecast_operational(initialized, namelist, hours)


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
        next_radiation = 1 + ((step + cadence - 2) // cadence) * cadence
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
    in-segment radiation gate is the SAME traced ``(step_index - 1) %% cadence == 0``
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

    # Sync cadence: ``GPUWRF_SEGMENT_SYNC_EVERY`` (default 1 = sync after every
    # segment, the historical behaviour) blocks only every K-th segment instead.
    # Pure scheduling: device work and op ordering are unchanged, so outputs are
    # bitwise identical for any K; a larger K lets the host enqueue segment N+1
    # while segment N still executes, hiding launch gaps, at the cost of more
    # in-flight segment working sets (K in flight instead of one).  The final
    # segment always syncs so the returned state is materialized.
    sync_every = int(os.environ.get("GPUWRF_SEGMENT_SYNC_EVERY", "1") or "1")
    if sync_every <= 0:
        raise ValueError("GPUWRF_SEGMENT_SYNC_EVERY must be a positive integer")

    # Host loop over contiguous fixed-length segments covering global steps 1..steps.
    # Every full segment has identical static ``n_steps`` so it reuses ONE compiled
    # executable (``start_step`` is traced); a final partial segment compiles once.
    # #91: traced per-run date scalars so each segment's HLO is date-independent.
    clock_base = build_clock_base(namelist)
    start = 1
    segment_index = 0
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(
            carry, namelist, jnp.asarray(start, dtype=jnp.int32), clock_base,
            n_steps=int(n), cadence=cadence,
        )
        segment_index += 1
        # Block so this segment's device scratch is freed before the next segment's
        # buffers are allocated -- this is what bounds peak memory to one segment's
        # working set regardless of forecast length.  With ``sync_every > 1`` this
        # happens only every K-th segment (and on the final segment), trading
        # bounded memory for host run-ahead overlap.
        if segment_index % sync_every == 0 or start + n > steps:
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
    predicate ``((step_index - 1) %% cadence == 0)``.  Compile cost is then independent of
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
            run_radiation = jnp.equal(jnp.mod(step_index - 1, cadence), 0)
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
        next_radiation = 1 + ((step + cadence - 2) // cadence) * cadence
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
        next_radiation = 1 + ((step + cadence - 2) // cadence) * cadence
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
    "PHASE_TAP_SCRATCH_FIELDS",
    "PHASE_TAP_SUMMARY_METRICS",
    "CorrectedNiPhaseTapResult",
    "RCA_HEALTH_METRICS",
    "RCA_ACOUSTIC_FIELDS",
    "RCA_STATE_FIELDS",
    "RCA_BOUNDARY_FIELDS",
    "RCA_STATE_PHASES",
    "RCA_TARGET_K",
    "RCA_TARGET_Y",
    "RCA_TARGET_X",
    "RcaChunkResult",
    "RcaStepRecord",
    "advance_one_step_with_corrected_ni_phase_tap",
    "advance_chunk_with_corrected_ni_rca",
    "compute_m9_diagnostics",
    "compute_m9_selected_diagnostics",
    "dealias_state_buffers",
    "run_forecast_operational",
    "dfi_initialize_operational_state",
    "run_forecast_operational_dfi",
    "run_forecast_operational_segmented",
    "run_forecast_operational_single_scan",
    "run_forecast_operational_debug",
    "run_forecast_operational_with_limiter_diagnostics",
    "run_forecast_operational_with_m9_diagnostics",
]
