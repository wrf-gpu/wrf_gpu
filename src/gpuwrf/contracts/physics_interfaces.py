"""Frozen v0.6.0 physics adapter interfaces.

This module defines the typed, allocation-free contract that per-scheme lanes
must implement before they are wired into ``runtime.operational_mode``. It does
not run a scheme and does not import JAX. The actual arrays are intentionally
typed as ``Any`` because JAX, NumPy, and savepoint replay payloads all need to
flow through the same interface during oracle work.

The contract separates:

* ``PhysicsTendency``: rates, direct replacements, and accumulator increments
  returned by one scheme call.
* ``PhysicsCarry``: persistent WRF ``state:`` members that are not dycore
  ``State`` leaves, following the existing Noah-MP sibling-carry pattern.
* ``PhysicsStepSpec``: per-option ownership, WRF call slot, State reads/writes,
  carry reads/writes, diagnostics, and required oracle gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_RA_LW_PHYSICS,
    ACCEPTED_RA_SW_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
    ACCEPTED_SF_SURFACE_PHYSICS,
    ACCUMULATORS,
    CUMULUS_CARRY_MEMBERS,
    CUMULUS_TENDENCY_MEMBERS,
    LAND_CARRY_MEMBERS,
    MOIST_SPECIES_ALL,
    NOAH_CLASSIC_NUM_SOIL_LAYERS,
    NUMBER_SPECIES,
    VOLUME_SPECIES,
    PBL_CARRY_MEMBERS,
    PBL_DIAGNOSTIC_MEMBERS,
    PHYSICS_REGISTRY_VERSION,
    assert_registry_consistent,
    state_leaves_for_mp,
)


PHYSICS_INTERFACE_VERSION = "v0.6.0-S0-frozen-2026-06-04-consolidation3-bmj2-extension"

ArrayLike = Any
_EMPTY: Mapping[str, Any] = MappingProxyType({})


def _empty_mapping() -> Mapping[str, Any]:
    return _EMPTY


# WRF ARW call slots for the physics drivers. The current port's operational
# loop bridges these slots in its own order until a scheme lane proves parity;
# savepoint oracle gates must use the WRF slot named here.
WRF_CALL_ORDER_SLOTS: tuple[str, ...] = (
    "first_rk_pre_radiation_driver",
    "first_rk_radiation_driver",
    "first_rk_surface_driver",
    "first_rk_pbl_driver",
    "first_rk_cumulus_driver",
    "solve_em_microphysics_driver",
)

STATE_TENDENCY_KEYS: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "theta",
    # MOIST_SPECIES_ALL = the v0.2.0..v0.16 six-class core PLUS the v0.17 hail
    # class qh; VOLUME_SPECIES (qvolg/qvolh) are the predicted-density volumes.
    # A hail microphysics scheme (WSM7/WDM7) replaces qh (and a future
    # predicted-density scheme the volumes) in its PhysicsTendency.
    *MOIST_SPECIES_ALL,
    *VOLUME_SPECIES,
    *NUMBER_SPECIES,
    "qke",
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "rhosfc",
    "fltv",
    "t_skin",
    "soil_moisture",
    "mavail",
)
ACCUMULATOR_UPDATE_KEYS: tuple[str, ...] = ACCUMULATORS


@dataclass(frozen=True)
class PhysicsTendency:
    """One scheme-call update payload.

    ``state_tendencies`` contains per-second tendencies keyed by State leaf.
    ``state_replacements`` contains post-scheme replacement arrays for schemes
    ported in WRF's in-place style. An adapter may use one style per leaf, never
    both for the same leaf in the same call. ``accumulator_increments`` contains
    per-call millimeter increments for precipitation accumulators.
    """

    state_tendencies: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    state_replacements: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    accumulator_increments: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    diagnostics: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)

    def validate_keys(self) -> None:
        """Fail if an adapter returns an unknown State/accumulator key."""

        allowed_state = set(STATE_TENDENCY_KEYS)
        overlap = set(self.state_tendencies).intersection(self.state_replacements)
        if overlap:
            raise ValueError(f"PhysicsTendency leaf returned as both tendency and replacement: {sorted(overlap)}")
        for key in self.state_tendencies:
            if key not in allowed_state:
                raise ValueError(f"unknown state_tendency key {key!r}")
        for key in self.state_replacements:
            if key not in allowed_state:
                raise ValueError(f"unknown state_replacement key {key!r}")
        for key in self.accumulator_increments:
            if key not in ACCUMULATOR_UPDATE_KEYS:
                raise ValueError(f"unknown accumulator increment key {key!r}")


@dataclass(frozen=True)
class PhysicsDiagnostics:
    """Optional diagnostic side channels grouped by physics family."""

    microphysics: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    surface_layer: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    pbl: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    cumulus: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    land_surface: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    radiation: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)


@dataclass(frozen=True)
class PhysicsCarry:
    """Persistent WRF ``state:`` members outside the dycore State contract."""

    microphysics: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    surface_layer: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    pbl: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    cumulus: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    land_surface: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    radiation: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)


@dataclass(frozen=True)
class PhysicsStepResult:
    """Adapter return object for one physics scheme call."""

    tendency: PhysicsTendency
    carry: PhysicsCarry = field(default_factory=PhysicsCarry)
    diagnostics: PhysicsDiagnostics = field(default_factory=PhysicsDiagnostics)


@dataclass(frozen=True)
class PhysicsStepSpec:
    """Frozen per-scheme adapter contract and file-ownership metadata."""

    family: str
    option: int
    name: str
    wrf_slot: str
    owner_module: str
    oracle: str
    reads_state: tuple[str, ...]
    writes_state: tuple[str, ...]
    reads_carry: tuple[str, ...] = ()
    writes_carry: tuple[str, ...] = ()
    returns_accumulators: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    notes: str = ""
    # ``variant`` disambiguates families where one namelist option maps to more
    # than one independent adapter sharing a kernel. Radiation is the only such
    # case in the v0.6.0 menu: ra_lw_physics=4 and ra_sw_physics=4 are distinct
    # namelist switches that both select the RRTMG column code, so the freeze
    # carries two specs under option 4 keyed by variant ``"lw"`` / ``"sw"``.
    variant: str = ""


def _mp_spec(
    option: int,
    name: str,
    owner: str,
    oracle: str,
    *,
    diagnostics: tuple[str, ...] = (),
    accumulators: tuple[str, ...] = ("rain_acc", "snow_acc", "graupel_acc", "ice_acc"),
) -> PhysicsStepSpec:
    leaves = state_leaves_for_mp(option)
    return PhysicsStepSpec(
        family="microphysics",
        option=option,
        name=name,
        wrf_slot="solve_em_microphysics_driver",
        owner_module=owner,
        oracle=oracle,
        reads_state=("theta", "p", "pb", "ph", "mu", *leaves),
        writes_state=("theta", *leaves),
        returns_accumulators=accumulators,
        diagnostics=diagnostics,
    )


SCHEME_STEP_SPECS: tuple[PhysicsStepSpec, ...] = (
    _mp_spec(
        1,
        "Kessler warm rain",
        "src/gpuwrf/physics/microphysics_kessler.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:kessler",
    ),
    _mp_spec(
        2,
        "Purdue-Lin",
        "src/gpuwrf/physics/microphysics_lin.py",
        "v0.6.0 Purdue-Lin pristine-WRF savepoint parity gate at module_microphysics_driver.F:lin",
    ),
    _mp_spec(
        3,
        "WSM3 simple ice",
        "src/gpuwrf/physics/microphysics_wsm3.py",
        "v0.6.0 WSM3 pristine-WRF savepoint parity gate at module_microphysics_driver.F:wsm3",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        4,
        "WSM5",
        "src/gpuwrf/physics/microphysics_wsm5.py",
        "v0.6.0 WSM5 pristine-WRF savepoint parity gate at module_microphysics_driver.F:wsm5",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        6,
        "WSM6",
        "src/gpuwrf/physics/microphysics_wsm6.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:wsm6",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        8,
        "Thompson",
        "src/gpuwrf/physics/thompson_column.py",
        "operational / Tier-4 RMSE validated vs CPU-WRF corpus, NOT isolated-unmodified-WRF-savepoint-proven",
        diagnostics=("re_cloud", "re_ice", "re_snow", "ThompsonTendencySideChannel"),
    ),
    _mp_spec(
        10,
        "Morrison two-moment",
        "src/gpuwrf/physics/microphysics_morrison.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:morrison two moment",
    ),
    _mp_spec(
        13,
        "SBU-YLin",
        "src/gpuwrf/physics/microphysics_sbu_ylin.py",
        "v0.17 SBU-YLin pristine-WRF single-column savepoint parity gate at "
        "module_mp_sbu_ylin.F (proofs/v017/run_sbu_ylin_parity.py / "
        "savepoints_sbu_ylin)",
        diagnostics=("ri3d",),
    ),
    _mp_spec(
        14,
        "WDM5",
        "src/gpuwrf/physics/microphysics_wdm5.py",
        "v0.13 WDM5 pristine-WRF single-column savepoint parity gate at "
        "module_mp_wdm5.F (proofs/v013/t3_wdm5_oracle.py/.json)",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        16,
        "WDM6",
        "src/gpuwrf/physics/microphysics_wdm6.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:wdm6",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        28,
        "Thompson aerosol-aware",
        "src/gpuwrf/physics/thompson_aero_column.py",
        "v0.16 aerosol-aware Thompson pristine-WRF grid savepoint parity gate at "
        "module_mp_thompson.F:mp_gt_driver (mp_physics=28; "
        "proofs/v016/thompson_aero_savepoint_parity.py/.json)",
    ),
    _mp_spec(
        97,
        "Goddard GCE",
        "src/gpuwrf/physics/microphysics_goddard.py",
        "v0.17 Goddard GCE (gsfcgcescheme) single-column savepoint parity gate vs "
        "unmodified phys/module_mp_gsfcgce.F (proofs/v090/run_goddard_parity.py, "
        "proofs/v090/goddard_mp_r2_savepoint_parity.json; 5/5 cases, ~machine "
        "precision vs the fp64 transparency oracle)",
    ),
    # v0.17 WSM7 = WSM6 + a separate precipitating hail class (qh leaf + hail_acc
    # surface accumulator). Savepoint-parity-proven against the unmodified
    # phys/module_mp_wsm7.F (proofs/v013/run_wsm7_parity.py, 6/6 PASS).
    _mp_spec(
        24,
        "WSM7",
        "src/gpuwrf/physics/microphysics_wsm7.py",
        "v0.17 WSM7 pristine-WRF single-column savepoint parity gate at "
        "module_mp_wsm7.F (proofs/v013/run_wsm7_parity.py/.json)",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc"),
    ),
    # v0.17 WDM7 = WDM6 double-moment warm rain (Nc/Nr/Nn) + a separate
    # single-moment precipitating hail class (qh + hail_acc; no Nh).
    # Savepoint-parity-proven against the unmodified phys/module_mp_wdm7.F
    # (proofs/v013_wdm7/run_wdm7_parity.py, 6/6 PASS).
    _mp_spec(
        26,
        "WDM7",
        "src/gpuwrf/physics/microphysics_wdm7.py",
        "v0.17 WDM7 pristine-WRF single-column savepoint parity gate at "
        "module_mp_wdm7.F (proofs/v013_wdm7/run_wdm7_parity.py/.json)",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc"),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=1,
        name="YSU",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_ysu.py",
        oracle="M20 physics-oracle factory savepoint at module_pbl_driver.F:YSU",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        diagnostics=("pblh",),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=2,
        name="MYJ",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_myj.py",
        oracle="v0.13 operational savepoint parity vs unmodified WRF module_bl_myjpbl.F "
        "(proofs/v013/myj_janjic_oracle.py/.json; reuses v0.6.0 fp64 savepoints)",
        reads_state=("u", "v", "theta", "qv", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[2],
        writes_carry=PBL_CARRY_MEMBERS[2],
        diagnostics=("pblh", "kpbl", "mixht", "tke_pbl", "exch_h", "exch_m", "el_pbl"),
        notes="v0.13 OPERATIONAL (jit/vmap-traceable bl_myj.myj_columns, scan-wired via "
        "physics.myj_adapters.myj_pbl_adapter; host-NumPy reference in pbl_myj.py). "
        "Mandatory pair with sf_sfclay_physics=2; the TKE carry rides State.qke (q^2 "
        "convention). Pairing enforced by namelist_check and dispatcher resolution.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=3,
        name="GFS",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_gfs.py",
        oracle="v0.17 operational savepoint parity vs unmodified WRF module_bl_gfs.F "
        "(BL_GFS -> MONINP -> TRIDI2/TRIDIN/TRIDIT; proofs/v017/gfs_oracle.py/.json, "
        "fp64 kind_phys-native ~1e-13, 6 regimes, NOT a self-compare)",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        diagnostics=("pblh", "kpbl"),
        notes="v0.17 OPERATIONAL (jit/vmap-traceable bl_gfs.gfs_columns, scan-wired via "
        "coupling.scan_adapters.gfs_pbl_adapter -> PBL_SCAN_ADAPTERS[3]). NCEP-GFS "
        "Hybrid-EDMF-ancestor nonlocal-K PBL (Hong-Pan lineage); re-derives the "
        "revised-MM5 surface forcing (sf_sfclay=1), nonlocal-K so no prognostic carry.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=4,
        name="QNSE-EDMF",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_reference_only.py",
        oracle="v0.18 fp64 pristine-WRF savepoint oracle vs unmodified "
        "phys/module_bl_qnsepbl.F (proofs/v018/qnse_pbl4_reference_oracle.json; "
        "proofs/v018/savepoints_fp64/qnse)",
        reads_state=("u", "v", "theta", "qv", "qc", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qc"),
        reads_carry=PBL_CARRY_MEMBERS[4],
        writes_carry=PBL_CARRY_MEMBERS[4],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[4],
        notes="v0.18 REFERENCE-ONLY: real fp64 WRF oracle/savepoints staged, but "
        "no traceable JAX column kernel is scan-wired. Operational scan fail-closes.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=11,
        name="Shin-Hong",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_shinhong.py",
        oracle="v0.18 savepoint-derived JAX/vmap kernel checked against the v090 faithful host-NumPy "
        "reference port of unmodified phys/module_bl_shinhong.F; forecast-driving "
        "tendencies/EXCH_H/PBLH/KPBL/WSTAR/DELTA are parity-gated. TKE/EL are "
        "diagnostic-only here and retain explicit residuals vs the v090 PARTIAL "
        "reference (TKE rel ~=0.285, EL rel ~=0.013; fp32-sensitive upstream oracle)",
        reads_state=("u", "v", "theta", "qv", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[11],
        writes_carry=PBL_CARRY_MEMBERS[11],
        diagnostics=("pblh", "kpbl", "tke_pbl", "el_pbl", "exch_h"),
        notes="v0.18 OPERATIONAL dynamics-green: scale-aware (grid-dependent) YSU-family PBL "
        "(Shin-Hong 2015), scan-wired via coupling.scan_adapters.shinhong_pbl_adapter "
        "-> PBL_SCAN_ADAPTERS[11]. Consumes revised-MM5 surface forcing (sf_sfclay=1) "
        "and grid dx/dy for pu/pq/pthnl/pthl/ptke partition functions. Follow up by "
        "refining TKE/EL if a faithful pristine-WRF Shin-Hong TKE oracle is built.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=12,
        name="GBM TKE",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_gbm.py",
        oracle="v0.18 fp64 parity-green vs single-column pristine-WRF savepoints "
        "from unmodified phys/module_bl_gbmpbl.F (proofs/v018/gbm_pbl12_jax_parity.json)",
        reads_state=("u", "v", "theta", "qv", "qc", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qc", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[12],
        writes_carry=PBL_CARRY_MEMBERS[12],
        diagnostics=("pblh", "kpbl", "tke_pbl", "el_pbl", "exch_tke"),
        notes="v0.18 OPERATIONAL: Grenier-Bretherton-McCaa moist prognostic-TKE PBL, "
        "scan-wired via coupling.scan_adapters.gbm_pbl_adapter -> PBL_SCAN_ADAPTERS[12]. "
        "The JAX/vmap kernel preserves GBMPBL's two-pass wrapper, advances qc/qke, and "
        "matches the fp64 pristine-WRF savepoint oracle at roundoff.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=5,
        name="MYNN",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/mynn_pbl.py",
        oracle="operational / Tier-4 RMSE validated vs CPU-WRF corpus, NOT isolated-unmodified-WRF-savepoint-proven",
        reads_state=("u", "v", "theta", "qv", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[5],
        writes_carry=PBL_CARRY_MEMBERS[5],
        diagnostics=("pblh", "tke_pbl", "sh3d", "sm3d", "tsq", "qsq", "cov", "el_pbl"),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=7,
        name="ACM2",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_acm2.py",
        oracle="M20 physics-oracle factory savepoint at module_pbl_driver.F:ACM2",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        diagnostics=("pblh",),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=8,
        name="BouLac",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_boulac.py",
        oracle="v0.6.0 BouLac WRF savepoint parity gate at module_bl_boulac.F",
        reads_state=("u", "v", "theta", "qv", "qc", "qke", "p", "pb", "ph", "mu", "ustar"),
        writes_state=("u", "v", "theta", "qv", "qc", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[8],
        writes_carry=PBL_CARRY_MEMBERS[8],
        diagnostics=("pblh", "tke_pbl", "dlk", "exch_h", "exch_m"),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=10,
        name="TEMF",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_reference_only.py",
        oracle="v0.18 fp64 pristine-WRF savepoint oracle vs unmodified "
        "phys/module_bl_temf.F (proofs/v018/temf_pbl10_reference_oracle.json; "
        "proofs/v018/savepoints_fp64/temf)",
        reads_state=("u", "v", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux", "t_skin"),
        writes_state=("u", "v", "theta", "qv", "qc"),
        reads_carry=PBL_CARRY_MEMBERS[10],
        writes_carry=PBL_CARRY_MEMBERS[10],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[10],
        notes="v0.18 REFERENCE-ONLY: real fp64 WRF oracle/savepoints staged, but "
        "no traceable JAX column kernel is scan-wired. Operational scan fail-closes.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=16,
        name="EEPS epsilon",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_reference_only.py",
        oracle="v0.18 fp64 pristine-WRF savepoint oracle vs unmodified "
        "phys/module_bl_eepsilon.F (proofs/v018/eeps_pbl16_reference_oracle.json; "
        "proofs/v018/savepoints_fp64/eeps)",
        reads_state=("u", "v", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi"),
        reads_carry=PBL_CARRY_MEMBERS[16],
        writes_carry=PBL_CARRY_MEMBERS[16],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[16],
        notes="v0.18 REFERENCE-ONLY: real fp64 WRF oracle/savepoints staged, but "
        "no traceable JAX column kernel is scan-wired. Operational scan fail-closes.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=17,
        name="KEPS k-epsilon",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_reference_only.py",
        oracle="v0.18 fp64 pristine-WRF savepoint oracle vs unmodified "
        "phys/module_bl_keps.F (proofs/v018/keps_pbl17_reference_oracle.json; "
        "proofs/v018/savepoints_fp64/keps)",
        reads_state=("u", "v", "theta", "qv", "qc", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qc"),
        reads_carry=PBL_CARRY_MEMBERS[17],
        writes_carry=PBL_CARRY_MEMBERS[17],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[17],
        notes="v0.18 REFERENCE-ONLY: real fp64 WRF oracle/savepoints staged, but "
        "no traceable JAX column kernel is scan-wired. Operational scan fail-closes.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=99,
        name="MRF",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_mrf.py",
        oracle="v0.13 MRF WRF savepoint parity gate at phys/module_bl_mrf.F (proofs/v013/mrf_oracle.py)",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "t_skin", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        reads_carry=PBL_CARRY_MEMBERS[99],
        writes_carry=PBL_CARRY_MEMBERS[99],
        diagnostics=("pblh", "kpbl"),
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=1,
        name="sfclayrev",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/surface_layer_sfclayrev.py",
        oracle="M20 physics-oracle factory savepoint at module_surface_driver.F:sfclayrev",
        reads_state=("u", "v", "theta", "qv", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "LH", "UST"),
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=2,
        name="Janjic Eta surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/sfclay_janjic.py",
        oracle="v0.13 operational savepoint parity vs unmodified WRF module_sf_myjsfc.F "
        "(proofs/v013/myj_janjic_oracle.py/.json; reuses v0.6.0 fp64 savepoints)",
        reads_state=("u", "v", "theta", "qv", "qke", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "QFX", "LH", "UST", "AKHS", "AKMS"),
        notes="v0.13 OPERATIONAL (jit/vmap-batched via sf_myj.myjsfc_columns over the "
        "traceable sfclay_janjic kernel, scan-wired via physics.myj_adapters."
        "janjic_sfclay_adapter). Mandatory pair with bl_pbl_physics=2; runs FIRST in "
        "the WRF call chain and supplies the MYJ PBL coupling. Pairing enforced by "
        "namelist_check and dispatcher resolution.",
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=5,
        name="MYNN surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/surface_layer.py",
        oracle="operational / Tier-4 RMSE validated vs CPU-WRF corpus, NOT isolated-unmodified-WRF-savepoint-proven",
        reads_state=("u", "v", "theta", "qv", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "LH", "UST"),
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=7,
        name="Pleim-Xiu surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/surface_layer_px.py",
        oracle="M20 physics-oracle factory savepoint at module_surface_driver.F:Pleim-Xiu sfclay",
        reads_state=("u", "v", "theta", "qv", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "LH", "UST"),
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=3,
        name="NCEP GFS surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/sfclay_gfs.py",
        oracle="v0.13 Tier-3 fp64 savepoint parity vs unmodified WRF module_sf_gfs.F "
        "(proofs/v013/t3_surface_lsm_oracle.py/.json; faithful fpvs table)",
        reads_state=("u", "v", "theta", "qv", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("U10", "V10", "PSFC", "HFX", "QFX", "LH", "UST", "CHS", "CHS2", "CQS2"),
        notes="v0.13 OPERATIONAL (jit/vmap-batched sf_gfs_columns, scan-wired via "
        "coupling.scan_adapters.gfs_sfclay_adapter -> B2 kinematic flux handles). "
        "NCEP Monin-Obukhov bulk-Richardson exchange-coefficient solve; the GFS "
        "land/soil/canopy blocks are bypassed in standalone surface-layer mode.",
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=91,
        name="old MM5 surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/sfclay_old_mm5.py",
        oracle="v0.13 Tier-3 fp64 savepoint parity vs unmodified WRF module_sf_sfclay.F "
        "(proofs/v013/t3_surface_lsm_oracle.py/.json)",
        reads_state=("u", "v", "theta", "qv", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("T2", "Q2", "TH2", "U10", "V10", "HFX", "QFX", "LH", "UST", "REGIME"),
        notes="v0.13 OPERATIONAL (jit/vmap-batched sfclay_old_mm5_columns, scan-wired via "
        "coupling.scan_adapters.sfclay_old_mm5_adapter -> B2 kinematic flux handles). "
        "Classic MM5 4-regime Monin-Obukhov surface layer (predecessor of sfclayrev=1).",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=1,
        name="Kain-Fritsch",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_kf.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:KF",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qr", "qi", "qs"),
        reads_carry=CUMULUS_CARRY_MEMBERS[1],
        writes_carry=CUMULUS_CARRY_MEMBERS[1],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[1]),
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=2,
        name="Betts-Miller-Janjic",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_bmj.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:BMJ",
        reads_state=("theta", "qv", "p", "pb", "ph", "mu", "xland"),
        writes_state=("theta", "qv"),
        reads_carry=CUMULUS_CARRY_MEMBERS[2],
        writes_carry=CUMULUS_CARRY_MEMBERS[2],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "cutop", "cubot", *CUMULUS_TENDENCY_MEMBERS[2]),
        notes="Frozen-contract extension for WRF cu_physics=2. BMJ is an adjustment scheme and returns only RTHCUTEN/RQVCUTEN plus cumulus precipitation.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=3,
        name="Grell-Freitas",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_grell_freitas.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:Grell-Freitas",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        reads_carry=CUMULUS_CARRY_MEMBERS[3],
        writes_carry=CUMULUS_CARRY_MEMBERS[3],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[3]),
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=6,
        name="Tiedtke",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_tiedtke.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:Tiedtke",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[6]),
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=4,
        name="Scale-aware GFS SAS",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_sas.py",
        oracle="v0.17 single-column fp64 pristine-WRF savepoint "
        "(proofs/v017/oracle/cumulus_sas) vs unmodified "
        "phys/module_cu_scalesas.F:CU_SCALESAS; current shared JAX endpoint is "
        "RED in proofs/v017/sas_family_parity.json",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[4]),
        notes="v0.17 reference-only/fail-closed. Do not wire into CU_SCAN_ADAPTERS "
        "until the shared SAS JAX kernel is GREEN vs pristine WRF.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=16,
        name="New Tiedtke",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_tiedtke.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:New Tiedtke",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[16]),
        notes="v0.13 Tier-3 reference-only: single-column fp64 pristine-WRF oracle "
        "staged (proofs/v013/oracle/cumulus/ntiedtke_*); traceable JAX kernel is a "
        "carry-over (fail-closed in the operational scan).",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=94,
        name="2015 GFS SAS / HWRF",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_sas.py",
        oracle="v0.17 single-column fp64 pristine-WRF savepoint "
        "(proofs/v017/oracle/cumulus_sas) vs unmodified phys/module_cu_sas.F:CU_SAS; "
        "current shared JAX endpoint is RED in proofs/v017/sas_family_parity.json",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[94]),
        notes="v0.17 reference-only/fail-closed.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=95,
        name="Previous GFS SAS / HWRF OSAS",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_sas.py",
        oracle="v0.17 single-column fp64 pristine-WRF savepoint "
        "(proofs/v017/oracle/cumulus_sas) vs unmodified phys/module_cu_osas.F:CU_OSAS; "
        "current shared JAX endpoint is RED in proofs/v017/sas_family_parity.json",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[95]),
        notes="v0.17 reference-only/fail-closed.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=96,
        name="Previous new GFS SAS / YSU NSAS",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_sas.py",
        oracle="v0.17 single-column fp64 pristine-WRF savepoint "
        "(proofs/v017/oracle/cumulus_sas) vs unmodified phys/module_cu_nsas.F:CU_NSAS; "
        "current shared JAX endpoint is RED in proofs/v017/sas_family_parity.json",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[96]),
        notes="v0.17 reference-only/fail-closed.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=5,
        name="Grell-3D ensemble",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_g3.py",
        oracle="v0.13 single-column fp64 pristine-WRF savepoint harness "
        "(proofs/v013/oracle/cumulus) vs unmodified phys/module_cu_g3.F:G3DRV "
        "(module verified to compile standalone; G3DRV driver + savepoints are a "
        "Tier-3 carry-over)",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[5]),
        notes="v0.13 Tier-3 reference-only: oracle harness scaffolded (G3DRV driver "
        "+ savepoints + traceable JAX kernel are a carry-over); fail-closed in the "
        "operational scan.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=14,
        name="KIM Simplified Arakawa-Schubert",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_ksas.py",
        oracle="v0.13 single-column fp64 pristine-WRF savepoint "
        "(proofs/v013/oracle/cumulus/ksas_*) vs unmodified phys/module_cu_ksas.F:cu_ksas",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[14]),
        notes="v0.13 Tier-3 reference-only: oracle staged; traceable JAX column "
        "kernel is a carry-over (fail-closed in the operational scan).",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=93,
        name="Grell-Devenyi ensemble",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_grell_devenyi.py",
        oracle="v0.17 RED: real-WRF source target is unmodified "
        "phys/module_cu_gd.F:GRELLDRV; no committed single-column savepoint "
        "exists yet (proofs/v017/run_cu_kfgrell_parity.py records the gap).",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", *CUMULUS_TENDENCY_MEMBERS[93]),
        notes="v0.17 reference-only / RED. The Grell-Devenyi ensemble has a "
        "distinct WRF source path from Grell-Freitas; it is accepted for oracle "
        "work but fail-closed in the operational scan until a source-specific "
        "traceable JAX column endpoint passes pristine-WRF parity.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=99,
        name="previous Kain-Fritsch",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_kf_previous.py",
        oracle="v0.17 real-WRF single-column savepoint gate target is "
        "unmodified phys/module_cu_kf.F:KFCPS; candidate wrapper reuses the "
        "cu=1 KF-eta family endpoint and is intentionally RED until the "
        "source-specific parity gate passes "
        "(proofs/v017/run_cu_kfgrell_parity.py).",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        reads_carry=CUMULUS_CARRY_MEMBERS[99],
        writes_carry=CUMULUS_CARRY_MEMBERS[99],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", *CUMULUS_TENDENCY_MEMBERS[99]),
        notes="v0.17 reference-only / RED. This is WRF's previous "
        "Kain-Fritsch path (module_cu_kf.F), not the already-green KF-eta "
        "path (module_cu_kfeta.F). Reusing the KF-eta family code is only a "
        "candidate, not operational scan wiring.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=1,
        name="thermal-diffusion slab LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/lsm_slab.py",
        oracle="v0.13 Tier-3 fp64 savepoint parity vs unmodified WRF module_sf_slab.F "
        "(proofs/v013/t3_surface_lsm_oracle.py/.json)",
        reads_state=("t_skin", "mavail", "xland"),
        writes_state=("t_skin",),
        reads_carry=LAND_CARRY_MEMBERS[1],
        writes_carry=LAND_CARRY_MEMBERS[1],
        diagnostics=("TSK", "HFX", "QFX", "LH", "QSFC", "CAPG"),
        notes="5-layer Blackadar thermal-diffusion slab (SLAB1D). JAX-ported + fp64 "
        "oracle-validated (physics.lsm_slab.slab_columns) and v0.17 operationally "
        "scan-wired via coupling.slab_surface_hook.slab_surface_step: advances the "
        "5-layer TSLB soil-temperature land carry from GSW/GLW radiation forcing + an "
        "explicit TMN/THC/EMISS SlabStaticBundle (FLHC/FLQC recovered from the resident "
        "surface-layer kinematic flux handles).",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=2,
        name="Noah classic",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/noah_classic.py",
        oracle="M20 physics-oracle factory savepoint at module_surface_driver.F:Noah LSM",
        reads_state=("t_skin", "soil_moisture", "xland", "mavail", "roughness_m", "lu_index"),
        writes_state=("t_skin", "soil_moisture", "mavail"),
        reads_carry=LAND_CARRY_MEMBERS[2],
        writes_carry=LAND_CARRY_MEMBERS[2],
        diagnostics=("TSK", "GRDFLX", "QFX", "ALBEDO", "EMISS"),
        notes=f"Noah classic initializes a {NOAH_CLASSIC_NUM_SOIL_LAYERS}-layer land carry; "
        "do not reinterpret State.soil_moisture as a 4-layer field.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=3,
        name="RUC LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/lsm_ruc.py",
        oracle="v0.17 fp64 single-column savepoint vs unmodified WRF module_sf_ruclsm.F "
        "(LSMRUC->SOILVEGIN->SFCTMP; proofs/v017/oracle/ruclsm + "
        "savepoints/ruclsm/fp64/ruclsm_fp64.json, 5 regimes, REFERENCE-ONLY)",
        reads_state=("t_skin", "soil_moisture", "xland", "mavail", "roughness_m", "lu_index"),
        writes_state=("t_skin", "soil_moisture", "mavail"),
        reads_carry=LAND_CARRY_MEMBERS[3],
        writes_carry=LAND_CARRY_MEMBERS[3],
        diagnostics=("TSK", "HFX", "QFX", "LH", "GRDFLX", "SFCRUNOFF", "UDRUNOFF", "SNOW", "SNOWH"),
        notes="RUC multi-layer soil/snow LSM (sf_surface_physics=3). STATUS: REFERENCE-ONLY -- "
        "a fp64 pristine-WRF single-column oracle is staged (proofs/v017/oracle/ruclsm, "
        "LSMRUC driver with SOILVEGIN reading the unmodified VEGPARM/SOILPARM/GENPARM "
        "tables; NOT a self-compare), but a faithful traceable JAX column port of the "
        "~7.5k-LOC multi-layer soil/snow solver (SFCTMP + SOIL/SNOWSOIL + SOILTEMP/"
        "SNOWTEMP/SOILMOIST/SOILPROP/TRANSF/VILKA) is a documented carry-over, so NO "
        "operational kernel is shipped (avoiding a silently-wrong port) and RUC fail-closes "
        "in the operational scan (not in _SCAN_WIRED_OPTIONS). owner_module is the JAX "
        "column endpoint stub that exposes the carry shapes for the future port.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=4,
        name="Noah-MP",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/noah_mp.py",
        oracle="existing Noah-MP WRF savepoint parity gate; rerun before mixed-suite integration",
        reads_state=("t_skin", "soil_moisture", "xland", "mavail", "roughness_m", "lu_index"),
        writes_state=("t_skin", "soil_moisture", "mavail"),
        reads_carry=LAND_CARRY_MEMBERS[4],
        writes_carry=LAND_CARRY_MEMBERS[4],
        diagnostics=("TSK", "GRDFLX", "QFX", "ALBEDO", "EMISS"),
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=7,
        name="Pleim-Xiu LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/lsm_pleim_xiu.py",
        oracle="v0.17 fp64 savepoint parity vs unmodified WRF module_sf_pxlsm.F "
        "(SURFPX+QFLUX; proofs/v017/oracle/pxlsm + pxlsm_savepoint_parity_report.json, "
        "worst |jax-oracle| 4.4e-11 over 5 regimes)",
        reads_state=("t_skin", "soil_moisture", "xland", "ustar", "roughness_m", "theta_flux", "qv_flux"),
        writes_state=("t_skin",),
        reads_carry=LAND_CARRY_MEMBERS[7],
        writes_carry=LAND_CARRY_MEMBERS[7],
        diagnostics=("TSK", "HFX", "QFX", "LH", "GRDFLX", "CAPG", "T2", "Q2"),
        notes="Pleim-Xiu 2-layer ISBA LSM (SURFPX+QFLUX, NUDGEX=0). fp64-oracle-validated "
        "(physics.lsm_pleim_xiu.pxlsm_columns) and v0.17 operationally scan-wired via "
        "coupling.pleim_xiu_surface_hook.pleim_xiu_surface_step: advances the 2-layer ISBA "
        "carry (TG/T2/WG/W2/WR) from GSW/GLW radiation + an explicit PleimXiuStaticBundle "
        "(ISBA soil constants + vegetation fields); pairs with the PX surface layer "
        "(sf_sfclay_physics=7). RMOL recovered from the resident surface-layer flux handles.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=8,
        name="SSiB LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/lsm_ssib.py",
        oracle="v0.17 fp64 single-column savepoint vs unmodified WRF module_sf_ssib.F "
        "(SSIB driver + ~30 internal subroutines; proofs/v017/oracle/ssib + "
        "savepoints/ssib/fp64/ssib_case_{1..5}.json, 5 regimes, REFERENCE-ONLY)",
        reads_state=("t_skin", "soil_moisture", "xland", "mavail", "roughness_m", "lu_index"),
        writes_state=("t_skin", "soil_moisture", "mavail"),
        reads_carry=LAND_CARRY_MEMBERS[8],
        writes_carry=LAND_CARRY_MEMBERS[8],
        diagnostics=("TSK", "HFX", "QFX", "LH", "GRDFLX", "ALBEDO", "T2", "Q2"),
        notes="SSiB SiB biophysical canopy/soil/snow LSM (sf_surface_physics=8). STATUS: "
        "REFERENCE-ONLY -- a fp64 pristine-WRF single-column oracle is staged "
        "(proofs/v017/oracle/ssib, the unmodified SSIB driver; vegetation parameters from "
        "module-level DATA arrays, no external table; NOT a self-compare), but a faithful "
        "traceable JAX column port of the ~6.6k-LOC coupled SiB canopy/soil/4-level-snow "
        "solver (TEMRS1/TEMRS2 + UPDAT1 + RADAB + STOMA1 + INTERC + STRES1 + NEWTON) is a "
        "documented carry-over, so NO operational kernel is shipped and SSiB fail-closes in "
        "the operational scan (not in _SCAN_WIRED_OPTIONS). owner_module is the JAX column "
        "endpoint stub exposing the carry shapes for the future port.",
    ),
    # Radiation (RRTMG, ra_lw/ra_sw option 4). Unlike the other families, the
    # radiation drivers write a HELD-RATE potential-temperature tendency
    # (WRF RTHRATEN) applied at every dynamics step over the radt interval; the
    # column adapter (coupling/physics_couplers.rrtmg_held_rate) emits a `theta`
    # state_tendency rather than an in-place replacement. SW and LW are distinct
    # namelist options but share one RRTMG column adapter; LW reads cloud
    # hydrometeors + qv + the LSM surface emissivity/skin temperature, SW adds
    # the solar-geometry/coszen inputs and surface albedo. Surface fluxes
    # (GLW/SWDOWN/GSW) are returned as radiation diagnostics consumed by the LSM.
    PhysicsStepSpec(
        family="radiation",
        option=4,
        name="RRTMG longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/rrtmg_lw.py",
        oracle="M20 physics-oracle factory savepoint at module_radiation_driver.F:RRTMG LW",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "LWUPB", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=4. Held-rate RTHRATEN theta tendency; SW pairs with LW under "
        "RRTMG. cu_rad_feedback couples active cumulus cloud fraction into the column when on.",
    ),
    PhysicsStepSpec(
        family="radiation",
        option=4,
        name="RRTMG shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/rrtmg_sw.py",
        oracle="M20 physics-oracle factory savepoint at module_radiation_driver.F:RRTMG SW",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "SWUPB", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=4. Shares the RRTMG column adapter with LW; needs solar geometry "
        "(coszen/declination/equation-of-time) and surface albedo. Held-rate RTHRATEN.",
    ),
    # Classic Dudhia shortwave (ra_sw_physics=1) -- Stephens-1984 broadband
    # SW. Self-contained column kernel (no external table file); reads cloud
    # hydrometeors + qv + solar geometry (coszen) + surface albedo, emits a
    # held-rate RTHRATEN theta tendency and the surface net SW flux GSW.
    PhysicsStepSpec(
        family="radiation",
        option=1,
        name="Dudhia shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_sw_dudhia.py",
        oracle="v0.6.0 physics-oracle factory savepoint at module_ra_sw.F:SWRAD (Dudhia SW)",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=1. Stephens-1984 broadband Dudhia SW; held-rate RTHRATEN theta "
        "tendency. No external lookup-table asset. Solar geometry via coszen + surface albedo. "
        "STATUS: isolated-WRF-savepoint parity-proven AND operational-scan-wired: "
        "OperationalNamelist.ra_sw_physics=1 routes the radiation slot in runtime.operational_mode "
        "to coupling.physics_couplers.dudhia_sw_theta_tendency (Dudhia SW) + rrtmg_lw_theta_tendency "
        "(RRTMG LW), composed as the held-rate RTHRATEN (WRF runs SW/LW drivers independently). "
        "Wired-coupler oracle proof: proofs/radiation/cdudhia_sw_oracle.py. The surface SWDOWN/flux "
        "history diagnostics remain RRTMG-derived (rrtmg_radiation_diagnostics).",
    ),
    # GSFC/Chou-Suarez shortwave (ra_sw_physics=2) -- v0.13 Tier-3 faithful port of
    # phys/module_ra_gsfcsw.F (Chou-Suarez 8-band SW with k-distribution + the 5x75
    # ozone climatology). Self-contained column kernel; held-rate RTHRATEN theta
    # tendency + surface net SW flux GSW.
    PhysicsStepSpec(
        family="radiation",
        option=2,
        name="GSFC/Chou-Suarez shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_sw_gsfc.py",
        oracle="pristine-WRF fp64 savepoint at module_ra_gsfcsw.F (proofs/v013/t3_radiation_oracle.py, PASS ~1e-12, 7 regimes, NOT self-compare)",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=2. GSFC/Chou-Suarez 8-band SW; held-rate RTHRATEN theta tendency. "
        "STATUS: pristine-WRF-savepoint parity-proven (fp64 ~1e-12) AND operational-scan-wired: "
        "OperationalNamelist.ra_sw_physics=2 routes the radiation slot in runtime.operational_mode "
        "to coupling.physics_couplers.gsfc_sw_theta_tendency (GSFC SW) + rrtmg_lw_theta_tendency "
        "(RRTMG LW), composed as the held-rate RTHRATEN. Surface SWDOWN/flux history diagnostics "
        "remain RRTMG-derived. v0.13 Tier-3 (kernel/coupler/oracle validated; full multi-step "
        "forecast gate is a follow-on).",
    ),
    # New Goddard shortwave (ra_sw_physics=5) -- v0.18 RA-tail
    # REFERENCE-ONLY. This is backed by the paired real-WRF exact-driver oracle
    # for ra_lw/sw=5, generated from the upstream-identical module through the
    # physics-pristine WRFGPU2_ORACLE-instrumented executable.
    # This is the SW half of phys/module_ra_goddard.F:goddardrad('sw'), not the
    # older standalone module_ra_gsfcsw.F path above. It shares the Goddard
    # cloud optical LUT / ozone / aerosol conventions with the LW half. No JAX
    # kernel is shipped in v0.18 because the faithful full-module port is still
    # outstanding; selecting ra_sw=5 is accepted for oracle work but fail-closes
    # operationally.
    PhysicsStepSpec(
        family="radiation",
        option=5,
        name="Goddard shortwave (new)",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_goddard.py",
        oracle="v0.18 exact-driver real-WRF savepoint at module_radiation_driver.F -> "
        "module_ra_goddard.F:goddardrad (proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_sw_physics=5 paired with ra_lw_physics=5; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=5. New Goddard SW shares module_ra_goddard.F tables with "
        "Goddard LW. STATUS: REFERENCE-ONLY. Namelist-accepted for a reference "
        "comparison, fail-closed in the operational scan; no faithful JAX kernel is shipped.",
    ),
    # Classic RRTM longwave (ra_lw_physics=1) -- 16-band k-distribution from
    # AER, loaded from the RRTM_DATA asset. Reads cloud hydrometeors + qv +
    # interface T/p (t8w/p8w) + LSM surface emissivity/skin temperature; emits
    # a held-rate RTHRATEN theta tendency, surface downwelling GLW, and TOA OLR.
    PhysicsStepSpec(
        family="radiation",
        option=1,
        name="RRTM longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_lw_rrtm.py",
        oracle="v0.6.0 physics-oracle factory savepoint at module_ra_rrtm.F:RRTMLWRAD (classic RRTM LW)",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=1. AER classic RRTM 16-band LW; held-rate RTHRATEN. Loads the "
        "RRTM_DATA k-distribution asset and passes the proofs/v060/run_rrtm_lw_parity.py "
        "fp64 savepoint gate. STATUS: isolated-WRF-savepoint parity-proven AND "
        "operational-scan-wired: the host-NumPy reference kernel (physics.ra_lw_rrtm) was "
        "re-expressed as a JIT/vmap-traceable JAX column kernel (physics.ra_lw_rrtm_jax, "
        "reproducing the reference to fp64 round-off), and OperationalNamelist.ra_lw_physics=1 "
        "routes the radiation slot in runtime.operational_mode to "
        "coupling.physics_couplers.rrtm_lw_theta_tendency (classic RRTM LW) summed with the "
        "selected SW tendency (WRF runs SW/LW drivers independently). Wired-coupler oracle proof: "
        "proofs/radiation/rrtm_lw_oracle.py (pristine-WRF, NOT a self-compare). The surface GLW "
        "history diagnostic remains RRTMG-derived.",
    ),
    # GSFC/Goddard NUWRF longwave (ra_lw_physics=5) -- v0.18 RA-tail
    # REFERENCE-ONLY.
    # The Goddard LW core (phys/module_ra_goddard.F:lwrad, the Chou-Suarez 1994
    # 10-band correlated-k IR transfer) is the LW half of the ~12.5k-LOC combined
    # NUWRF SW+LW module (~11.8k hardcoded LW coefficients). A faithful traceable
    # JAX column port of that volume is a documented carry-over; NO kernel is wired.
    # A paired exact-driver real-WRF oracle IS staged (NOT a self-compare) so a
    # future port has a ready reference. This spec exists so the interface freeze
    # stays consistent with the reference-only accept-matrix; the operational scan
    # fail-closes ra_lw=5 (it is NOT in _SCAN_WIRED_OPTIONS["ra_lw_physics"]).
    PhysicsStepSpec(
        family="radiation",
        option=5,
        name="GSFC/Goddard NUWRF longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/rrtmg_lw.py",
        oracle="v0.18 exact-driver real-WRF savepoint at module_radiation_driver.F -> "
        "module_ra_goddard.F:goddardrad (proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_lw_physics=5 paired with ra_sw_physics=5; NOT a self-compare). Prior v0.13 "
        "single-column lwrad oracle remains useful background evidence.",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=5. GSFC/Goddard Chou-Suarez 10-band correlated-k IR (held-rate "
        "RTHRATEN theta tendency + surface downwelling GLW). STATUS: REFERENCE-ONLY -- the "
        "Goddard LW core lwrad is the LW half of the ~12.5k-LOC combined NUWRF SW+LW module "
        "(~11.8k hardcoded LW coefficients), so a faithful traceable JAX column kernel is a "
        "documented v0.13 carry-over and NO kernel is shipped (avoiding a silently-wrong port). "
        "A paired v0.18 exact-driver real-WRF oracle IS staged "
        "(proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json), with the older v0.13 "
        "single-column lwrad oracle retained as background evidence, so a future faithful port "
        "has a ready non-self-compare reference. ra_lw=5 is "
        "namelist-accepted (selectable for a single-column reference comparison) but FAIL-CLOSES "
        "in the operational scan (OperationalNamelist.ra_lw_physics=5 is not in "
        "_SCAN_WIRED_OPTIONS); the operational default remains ra_lw=4 (RRTMG), byte-unchanged. "
        "owner_module points at the RRTMG LW slot used as the operational fallback path.",
    ),
    # GFDL-Eta radiation pair (ra_lw_physics=99 / ra_sw_physics=99) -- v0.18
    # RA-tail REFERENCE-ONLY. WRF's ETARA computes both SW and LW together and
    # then exposes the selected components through THRATENLW/THRATENSW. A paired
    # real-WRF exact-driver oracle is staged; no faithful JAX kernel is shipped.
    PhysicsStepSpec(
        family="radiation",
        option=99,
        name="GFDL (Eta) longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_gfdleta.py",
        oracle="v0.18 exact-driver real-WRF savepoint at module_radiation_driver.F -> "
        "module_ra_gfdleta.F:ETARA (proofs/v018/savepoints/ra_tail_wrf/ra99_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_lw_physics=99 paired with ra_sw_physics=99; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qi", "qs", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=99. GFDL-Eta LW half of ETARA; STATUS: REFERENCE-ONLY. "
        "Paired with ra_sw=99 because WRF ETARA shares tables, cloud diagnostics, "
        "and call state between SW/LW; fail-closed in the operational scan.",
    ),
    PhysicsStepSpec(
        family="radiation",
        option=99,
        name="GFDL (Eta) shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_gfdleta.py",
        oracle="v0.18 exact-driver real-WRF savepoint at module_radiation_driver.F -> "
        "module_ra_gfdleta.F:ETARA (proofs/v018/savepoints/ra_tail_wrf/ra99_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_sw_physics=99 paired with ra_lw_physics=99; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qi", "qs", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=99. GFDL-Eta SW half of ETARA; STATUS: REFERENCE-ONLY. "
        "Paired with ra_lw=99 because WRF ETARA computes and stores both radiative "
        "components in one shared driver; fail-closed in the operational scan.",
    ),
    # CAM radiation pair (ra_lw_physics=3 / ra_sw_physics=3) -- v0.18 RA-tail
    # REFERENCE-ONLY. WRF's CAMRAD (phys/module_ra_cam.F, the NCAR CAM 3.0 radiation,
    # ~8.1k LOC) computes the LW and SW components selected through the camlwscheme /
    # camswscheme slots. No faithful traceable JAX kernel is shipped (the volume +
    # monthly ozone/aerosol climatology coupling makes an in-scope faithful port a
    # self-compare risk), so both specs are fail-closed in the operational scan and
    # lean on the v0.18 exact-driver real-WRF savepoint oracle.
    PhysicsStepSpec(
        family="radiation",
        option=3,
        name="CAM longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_cam.py",
        oracle="v0.18 exact-driver real-WRF column savepoint at module_radiation_driver.F -> "
        "module_ra_cam.F:CAMRAD (proofs/v018/savepoints/ra_tail_wrf/ra3_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_lw_physics=3; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=3. CAM 3.0 LW half of CAMRAD; STATUS: REFERENCE-ONLY. Paired with "
        "ra_sw=3 (CAMRAD computes SW+LW in one driver). Namelist-accepted for a reference "
        "comparison, fail-closed in the operational scan; operational default stays ra_lw=4 (RRTMG).",
    ),
    PhysicsStepSpec(
        family="radiation",
        option=3,
        name="CAM shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_cam.py",
        oracle="v0.18 exact-driver real-WRF column savepoint at module_radiation_driver.F -> "
        "module_ra_cam.F:CAMRAD (proofs/v018/savepoints/ra_tail_wrf/ra3_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_sw_physics=3; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=3. CAM 3.0 SW half of CAMRAD; STATUS: REFERENCE-ONLY. Shares the "
        "CAMRAD driver + ozone/aerosol climatology with ra_lw=3, fail-closed in the operational scan.",
    ),
    # FLG/UCLA radiation pair (ra_lw_physics=7 / ra_sw_physics=7) -- v0.18 RA-tail
    # REFERENCE-ONLY. WRF's RAD_FLG (phys/module_ra_flg.F, the UCLA/Fu-Liou-Gu
    # radiation, ~15.3k LOC) is the largest of the tail modules. No faithful JAX
    # kernel is shipped; both specs fail-close operationally and lean on the v0.18
    # exact-driver real-WRF savepoint oracle.
    PhysicsStepSpec(
        family="radiation",
        option=7,
        name="FLG (UCLA) longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_flg.py",
        oracle="v0.18 exact-driver real-WRF column savepoint at module_radiation_driver.F -> "
        "module_ra_flg.F:RAD_FLG (proofs/v018/savepoints/ra_tail_wrf/ra7_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_lw_physics=7; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=7. FLG/UCLA (Fu-Liou-Gu) LW half of RAD_FLG; STATUS: REFERENCE-ONLY. "
        "Paired with ra_sw=7, fail-closed in the operational scan; operational default stays ra_lw=4.",
    ),
    PhysicsStepSpec(
        family="radiation",
        option=7,
        name="FLG (UCLA) shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_flg.py",
        oracle="v0.18 exact-driver real-WRF column savepoint at module_radiation_driver.F -> "
        "module_ra_flg.F:RAD_FLG (proofs/v018/savepoints/ra_tail_wrf/ra7_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_sw_physics=7; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=7. FLG/UCLA (Fu-Liou-Gu) SW half of RAD_FLG; STATUS: REFERENCE-ONLY. "
        "Shares the RAD_FLG driver with ra_lw=7, fail-closed in the operational scan.",
    ),
    # Held-Suarez idealized radiation (ra_lw_physics=31) -- v0.17 GPU-op endpoint.
    # A COMBINED LW+SW Newtonian relaxation toward an analytic radiative-equilibrium
    # temperature (phys/module_ra_hs.F:HSRAD; Held & Suarez 1994). Selected through
    # the LW slot; WRF makes NO separate shortwave call, so the operational dispatch
    # requires ra_sw_physics=0. Carries NO prognostic state and reads only T, p,
    # surface interface pressure, and latitude -> a true no-kernel-change port.
    PhysicsStepSpec(
        family="radiation",
        option=31,
        name="Held-Suarez idealized radiation",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_lw_hs.py",
        oracle="pristine-WRF fp64 single-column savepoint at module_ra_hs.F:HSRAD "
        "(proofs/v017/oracle/hs_build_and_run.sh + proofs/v017/held_suarez_lw_savepoint_parity.json; "
        "NOT a self-compare; fp64 worst rel 1.1e-14 across 7 latitude regimes).",
        reads_state=("theta", "p", "pb"),
        writes_state=("theta",),
        diagnostics=("RTHRATEN",),
        variant="lw",
        notes="ra_lw_physics=31. Held-Suarez idealized COMBINED LW+SW Newtonian relaxation "
        "(held-rate RTHRATEN theta tendency). STATUS: IMPLEMENTED + scan-wired "
        "(coupling.physics_couplers.held_suarez_theta_tendency, stateless State->RTHRATEN). "
        "HSRAD is the SOLE radiative call WRF makes for this scheme, so the operational scan "
        "REQUIRES ra_sw_physics=0 (a real SW selection fail-closes to avoid double-counting). "
        "No-kernel-change endpoint: no prognostic state, no solar geometry/moisture/cloud/ozone; "
        "savepoint-parity-proven against the unmodified WRF source at fp64.",
    ),
)

SCHEME_STEP_SPECS_BY_KEY: Mapping[tuple[str, int, str], PhysicsStepSpec] = {
    (spec.family, spec.option, spec.variant): spec for spec in SCHEME_STEP_SPECS
}


def scheme_step_spec(family: str, option: int, variant: str = "") -> PhysicsStepSpec:
    """Return a frozen step spec for a nonzero physics option.

    ``variant`` is only needed for radiation, where one option (4) selects both
    the RRTMG LW (``variant="lw"``) and SW (``variant="sw"``) adapters.
    """

    return SCHEME_STEP_SPECS_BY_KEY[(family, int(option), variant)]


def assert_interfaces_consistent() -> None:
    """Validate the interface freeze against the Registry freeze."""

    assert_registry_consistent()

    expected = {
        ("microphysics", opt, "") for opt in ACCEPTED_MP_PHYSICS if opt != 0
    } | {
        ("pbl", opt, "") for opt in ACCEPTED_BL_PBL_PHYSICS if opt != 0
    } | {
        ("surface_layer", opt, "") for opt in ACCEPTED_SF_SFCLAY_PHYSICS if opt != 0
    } | {
        ("cumulus", opt, "") for opt in ACCEPTED_CU_PHYSICS if opt != 0
    } | {
        ("land_surface", opt, "") for opt in ACCEPTED_SF_SURFACE_PHYSICS if opt != 0
    } | {
        ("radiation", opt, "lw") for opt in ACCEPTED_RA_LW_PHYSICS if opt != 0
    } | {
        ("radiation", opt, "sw") for opt in ACCEPTED_RA_SW_PHYSICS if opt != 0
    }
    actual = set(SCHEME_STEP_SPECS_BY_KEY)
    missing = expected - actual
    if missing:
        raise AssertionError(f"missing PhysicsStepSpec entries: {sorted(missing)}")

    slots = set(WRF_CALL_ORDER_SLOTS)
    allowed_writes = set(STATE_TENDENCY_KEYS)
    for spec in SCHEME_STEP_SPECS:
        if spec.wrf_slot not in slots:
            raise AssertionError(f"{spec.family}:{spec.option} has unknown WRF slot {spec.wrf_slot!r}")
        if not spec.owner_module.startswith("src/gpuwrf/"):
            raise AssertionError(f"{spec.family}:{spec.option} owner is not a source module")
        if "savepoint" not in spec.oracle.lower():
            raise AssertionError(f"{spec.family}:{spec.option} oracle must be a WRF savepoint gate")
        unknown_writes = set(spec.writes_state) - allowed_writes
        if unknown_writes:
            raise AssertionError(
                f"{spec.family}:{spec.option} writes unknown PhysicsTendency State keys: {sorted(unknown_writes)}"
            )

    sample = PhysicsTendency(
        state_tendencies={"theta": object()},
        accumulator_increments={"rainc_acc": object()},
    )
    sample.validate_keys()


__all__ = [
    "PHYSICS_INTERFACE_VERSION",
    "PHYSICS_REGISTRY_VERSION",
    "ArrayLike",
    "WRF_CALL_ORDER_SLOTS",
    "STATE_TENDENCY_KEYS",
    "ACCUMULATOR_UPDATE_KEYS",
    "PhysicsTendency",
    "PhysicsDiagnostics",
    "PhysicsCarry",
    "PhysicsStepResult",
    "PhysicsStepSpec",
    "SCHEME_STEP_SPECS",
    "SCHEME_STEP_SPECS_BY_KEY",
    "scheme_step_spec",
    "assert_interfaces_consistent",
]


if __name__ == "__main__":
    assert_interfaces_consistent()
    families = sorted({spec.family for spec in SCHEME_STEP_SPECS})
    print(
        "ok physics_interfaces consistent:",
        f"version={PHYSICS_INTERFACE_VERSION}",
        f"scheme_specs={len(SCHEME_STEP_SPECS)}",
        f"families={len(families)}({','.join(families)})",
        f"slots={len(WRF_CALL_ORDER_SLOTS)}",
    )
