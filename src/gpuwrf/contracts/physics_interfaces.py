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
    MOIST_SPECIES,
    NOAH_CLASSIC_NUM_SOIL_LAYERS,
    NUMBER_SPECIES,
    PBL_CARRY_MEMBERS,
    PHYSICS_REGISTRY_VERSION,
    assert_registry_consistent,
    state_leaves_for_mp,
)


PHYSICS_INTERFACE_VERSION = "v0.6.0-S0-frozen-2026-06-03-rad"

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
    *MOIST_SPECIES,
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


def _mp_spec(option: int, name: str, owner: str, oracle: str, *, diagnostics: tuple[str, ...] = ()) -> PhysicsStepSpec:
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
        returns_accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc"),
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
        "existing Thompson WRF savepoint parity gate; rerun before mixed-suite integration",
        diagnostics=("re_cloud", "re_ice", "re_snow", "ThompsonTendencySideChannel"),
    ),
    _mp_spec(
        10,
        "Morrison two-moment",
        "src/gpuwrf/physics/microphysics_morrison.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:morrison two moment",
    ),
    _mp_spec(
        16,
        "WDM6",
        "src/gpuwrf/physics/microphysics_wdm6.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:wdm6",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
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
        option=5,
        name="MYNN",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/mynn_pbl.py",
        oracle="existing MYNN WRF savepoint parity gate; rerun before mixed-suite integration",
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
        option=5,
        name="MYNN surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/surface_layer.py",
        oracle="existing surface-layer WRF savepoint parity gate; rerun before mixed-suite integration",
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
        option=16,
        name="New Tiedtke",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_tiedtke.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:New Tiedtke",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[16]),
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
