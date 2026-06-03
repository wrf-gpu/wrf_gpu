"""v0.6.0 Registry-driven physics field list: the single source of truth.

This module is an interface-freeze artifact only. It allocates no arrays and
implements no physics kernels. It mirrors the relevant WRF
``Registry/Registry.EM_COMMON`` package memberships for the v0.6.0 accepted
physics menu so that scheme lanes, wrfout, restart, and nesting consume one
field list instead of re-creating Thompson-era assumptions locally.

WRF Registry lines verified against
``/home/enric/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON`` on
2026-06-02:

* Kessler(1): ``moist:qv,qc,qr``.
* Purdue-Lin(2): ``moist:qv,qc,qr,qi,qs,qg`` (single-moment, no scalar number
  species; graupel via F_QG=.true.).
* WSM3(3): ``moist:qv,qc,qr;state:re_cloud,re_ice,re_snow``.
* WSM5(4): ``moist:qv,qc,qr,qi,qs;state:re_cloud,re_ice,re_snow``.
* WSM6(6): ``moist:qv,qc,qr,qi,qs,qg;state:re_cloud,re_ice,re_snow``.
* Thompson(8): ``moist:qv,qc,qr,qi,qs,qg;scalar:qni,qnr;state:re_*``.
* Morrison(10): ``moist:qv..qg;scalar:qni,qns,qnr,qng`` plus cuten state.
* WDM6(16): ``moist:qv..qg;scalar:qnn,qnc,qnr;state:re_*``.
* MYJ(2): ``state:tke_pbl,el_pbl``; requires Janjic Eta sfclay(2).
* MYNN(5): ``scalar:qke_adv;state:qke,tke_pbl,sh3d,sm3d,tsq,qsq,cov,el_pbl``.
* BouLac(8): ``state:qke`` reused as the prognostic TKE storage plus PBLH/K
  diagnostics (frozen-contract extension, 2026-06-04).
* Noah classic(2): ``state:flx4,fvb,fbur,fgsn,smcrel,xlaidyn``.
* Cumulus options KF(1), Grell-Freitas(3), Tiedtke(6/16) use the common
  ``R*CUTEN`` tendency family and scheme-specific carry listed below.

Append-only State rule:
    Existing ``State.__slots__`` order is preserved. v0.6.0 additive dycore
    leaves are frozen as ``Nc``, ``Nn``, and ``rainc_acc``; lanes that need
    them must append them to ``State.__slots__`` / ``STATE_FIELD_ORDER`` /
    ``PRECISION_MATRIX`` and bump restart schema deliberately. Land/PBL/cumulus
    save-state fields stay in ``PhysicsCarry`` sibling trees, following the
    existing Noah-MP carry pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


PHYSICS_REGISTRY_VERSION = "v0.6.0-S0-frozen-2026-06-03-wsm-sm-extension"


@dataclass(frozen=True)
class SchemeOption:
    """One accepted namelist option with its WRF Registry package source."""

    key: str
    option: int
    name: str
    wrf_package: str
    status: str
    owner_family: str


ACCEPTED_MP_PHYSICS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 10, 16)
ACCEPTED_BL_PBL_PHYSICS: tuple[int, ...] = (0, 1, 2, 5, 7, 8)
ACCEPTED_SF_SFCLAY_PHYSICS: tuple[int, ...] = (0, 1, 2, 5, 7)
ACCEPTED_CU_PHYSICS: tuple[int, ...] = (0, 1, 3, 6, 16)
ACCEPTED_SF_SURFACE_PHYSICS: tuple[int, ...] = (0, 2, 4)
ACCEPTED_RA_SW_PHYSICS: tuple[int, ...] = (0, 4)
ACCEPTED_RA_LW_PHYSICS: tuple[int, ...] = (0, 4)

ACCEPTED_NAMELIST_OPTIONS: Mapping[str, tuple[int, ...]] = {
    "mp_physics": ACCEPTED_MP_PHYSICS,
    "bl_pbl_physics": ACCEPTED_BL_PBL_PHYSICS,
    "sf_sfclay_physics": ACCEPTED_SF_SFCLAY_PHYSICS,
    "cu_physics": ACCEPTED_CU_PHYSICS,
    "sf_surface_physics": ACCEPTED_SF_SURFACE_PHYSICS,
    "ra_sw_physics": ACCEPTED_RA_SW_PHYSICS,
    "ra_lw_physics": ACCEPTED_RA_LW_PHYSICS,
}

MP_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("mp_physics", 0, "disabled/passive qv", "passiveqv", "accepted", "microphysics"),
    1: SchemeOption("mp_physics", 1, "Kessler warm rain", "kesslerscheme", "accepted", "microphysics"),
    2: SchemeOption("mp_physics", 2, "Purdue-Lin", "linscheme", "accepted", "microphysics"),
    3: SchemeOption("mp_physics", 3, "WSM3 simple ice", "wsm3scheme", "accepted", "microphysics"),
    4: SchemeOption("mp_physics", 4, "WSM5", "wsm5scheme", "accepted", "microphysics"),
    6: SchemeOption("mp_physics", 6, "WSM6", "wsm6scheme", "accepted", "microphysics"),
    8: SchemeOption("mp_physics", 8, "Thompson", "thompson", "implemented", "microphysics"),
    10: SchemeOption("mp_physics", 10, "Morrison two-moment", "morr_two_moment", "accepted", "microphysics"),
    16: SchemeOption("mp_physics", 16, "WDM6", "wdm6scheme", "accepted", "microphysics"),
}

PBL_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("bl_pbl_physics", 0, "disabled", "none", "accepted", "pbl"),
    # YSU(1)/ACM2(7) status bumped to "implemented" (v0.6.0 GPU-op): the host-NumPy
    # single-column kernels were rewritten as jax.lax.scan-traceable / vmap-batched
    # JAX and scan-wired into the operational PBL slot, with no savepoint-parity
    # regression (proofs/v060/{ysu,acm2}_gpuop_savepoint_parity.json).
    1: SchemeOption("bl_pbl_physics", 1, "YSU", "ysuscheme", "implemented", "pbl"),
    # MYJ(2): savepoint-parity-proven CPU reference, NOT yet GPU-scan-wired
    # (fail-closed in the operational scan); requires Janjic Eta sfclay(2).
    2: SchemeOption("bl_pbl_physics", 2, "MYJ", "myjpblscheme", "accepted", "pbl"),
    5: SchemeOption("bl_pbl_physics", 5, "MYNN", "mynnpblscheme", "implemented", "pbl"),
    7: SchemeOption("bl_pbl_physics", 7, "ACM2", "acmpblscheme", "implemented", "pbl"),
    8: SchemeOption("bl_pbl_physics", 8, "BouLac", "boulacscheme", "accepted", "pbl"),
}

SFCLAY_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("sf_sfclay_physics", 0, "disabled", "none", "accepted", "surface_layer"),
    1: SchemeOption("sf_sfclay_physics", 1, "sfclayrev", "sfclayscheme", "accepted", "surface_layer"),
    2: SchemeOption("sf_sfclay_physics", 2, "Janjic Eta surface layer", "myjsfcscheme", "accepted", "surface_layer"),
    5: SchemeOption("sf_sfclay_physics", 5, "MYNN surface layer", "mynnsfclayscheme", "implemented", "surface_layer"),
    7: SchemeOption("sf_sfclay_physics", 7, "Pleim-Xiu surface layer", "pxsfclayscheme", "accepted", "surface_layer"),
}

CU_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("cu_physics", 0, "disabled", "no_cumulus", "accepted", "cumulus"),
    1: SchemeOption("cu_physics", 1, "Kain-Fritsch", "kfetascheme", "accepted", "cumulus"),
    3: SchemeOption("cu_physics", 3, "Grell-Freitas", "gfscheme", "accepted", "cumulus"),
    6: SchemeOption("cu_physics", 6, "Tiedtke", "tiedtkescheme", "accepted", "cumulus"),
    16: SchemeOption("cu_physics", 16, "New Tiedtke", "ntiedtkescheme", "accepted", "cumulus"),
}

SURFACE_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("sf_surface_physics", 0, "disabled", "none", "accepted", "land_surface"),
    2: SchemeOption("sf_surface_physics", 2, "Noah classic", "lsmscheme", "accepted", "land_surface"),
    4: SchemeOption("sf_surface_physics", 4, "Noah-MP", "noahmpscheme", "implemented", "land_surface"),
}


# Moisture species: WRF 4-D ``moist`` array members. State leaf name equals the
# lowercase WRF moist-array member name used throughout this port.
MOIST_SPECIES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")

MOIST_WRFOUT_NAME: Mapping[str, str] = {
    "qv": "QVAPOR",
    "qc": "QCLOUD",
    "qr": "QRAIN",
    "qi": "QICE",
    "qs": "QSNOW",
    "qg": "QGRAUP",
}

MP_MOIST_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: ("qv",),
    1: ("qv", "qc", "qr"),
    2: ("qv", "qc", "qr", "qi", "qs", "qg"),
    3: ("qv", "qc", "qr"),
    4: ("qv", "qc", "qr", "qi", "qs"),
    6: ("qv", "qc", "qr", "qi", "qs", "qg"),
    8: ("qv", "qc", "qr", "qi", "qs", "qg"),
    10: ("qv", "qc", "qr", "qi", "qs", "qg"),
    16: ("qv", "qc", "qr", "qi", "qs", "qg"),
}


# Number concentrations / WRF ``scalar`` array members. Existing Thompson-era
# State leaves are preserved; WDM6 adds ``Nc`` and ``Nn`` append-only.
NUMBER_SPECIES_EXISTING: tuple[str, ...] = ("Ni", "Nr", "Ns", "Ng")
NUMBER_SPECIES_ADDITIVE: tuple[str, ...] = ("Nc", "Nn")
NUMBER_SPECIES: tuple[str, ...] = NUMBER_SPECIES_EXISTING + NUMBER_SPECIES_ADDITIVE

NUMBER_REGISTRY_MEMBER: Mapping[str, str] = {
    "Ni": "qni",
    "Nr": "qnr",
    "Ns": "qns",
    "Ng": "qng",
    "Nc": "qnc",
    "Nn": "qnn",
}

NUMBER_WRFOUT_NAME: Mapping[str, str] = {
    "Ni": "QNICE",
    "Nr": "QNRAIN",
    "Ns": "QNSNOW",
    "Ng": "QNGRAUPEL",
    "Nc": "QNCLOUD",
    "Nn": "QNCCN",
}

MP_NUMBER_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: (),
    2: (),
    3: (),
    4: (),
    6: (),
    8: ("Ni", "Nr"),
    10: ("Ni", "Ns", "Nr", "Ng"),
    16: ("Nn", "Nc", "Nr"),
}


# Precipitation accumulators. ``rainc_acc`` is additive State because cumulus
# precipitation is a prognostic history/restart quantity in WRF (RAINC).
ACCUMULATORS_EXISTING: tuple[str, ...] = ("rain_acc", "snow_acc", "graupel_acc", "ice_acc")
ACCUMULATORS_ADDITIVE: tuple[str, ...] = ("rainc_acc",)
ACCUMULATORS: tuple[str, ...] = ACCUMULATORS_EXISTING + ACCUMULATORS_ADDITIVE

ACCUMULATOR_WRFOUT_NAME: Mapping[str, str] = {
    "rain_acc": "RAINNC",
    "snow_acc": "SNOWNC",
    "graupel_acc": "GRAUPELNC",
    "ice_acc": "SNOWNC",
    "rainc_acc": "RAINC",
}

V060_EXISTING_STATE_PHYSICS_LEAVES: tuple[str, ...] = (
    *MOIST_SPECIES,
    *NUMBER_SPECIES_EXISTING,
    "qke",
    *ACCUMULATORS_EXISTING,
)
V060_ADDITIVE_STATE_LEAVES: tuple[str, ...] = (*NUMBER_SPECIES_ADDITIVE, *ACCUMULATORS_ADDITIVE)


@dataclass(frozen=True)
class RegistryFieldSpec:
    """Frozen metadata for a WRF Registry-backed physics field."""

    leaf: str
    wrf_name: str
    registry_member: str
    kind: str
    shape: str
    storage: str
    existing_state: bool
    additive_state: bool
    restart_required: bool
    wrfout_required: bool
    nest_forcedown: bool
    nest_feedback: bool
    lateral_bc: bool
    schemes: tuple[str, ...]
    notes: str = ""


def _field(
    leaf: str,
    wrf_name: str,
    registry_member: str,
    kind: str,
    shape: str,
    storage: str,
    schemes: tuple[str, ...],
    *,
    existing_state: bool = False,
    additive_state: bool = False,
    restart_required: bool = False,
    wrfout_required: bool = False,
    nest_forcedown: bool = False,
    nest_feedback: bool = False,
    lateral_bc: bool = False,
    notes: str = "",
) -> RegistryFieldSpec:
    return RegistryFieldSpec(
        leaf=leaf,
        wrf_name=wrf_name,
        registry_member=registry_member,
        kind=kind,
        shape=shape,
        storage=storage,
        existing_state=existing_state,
        additive_state=additive_state,
        restart_required=restart_required,
        wrfout_required=wrfout_required,
        nest_forcedown=nest_forcedown,
        nest_feedback=nest_feedback,
        lateral_bc=lateral_bc,
        schemes=schemes,
        notes=notes,
    )


FIELD_SPECS: tuple[RegistryFieldSpec, ...] = (
    *(
        _field(
            leaf,
            MOIST_WRFOUT_NAME[leaf],
            leaf,
            "moist",
            "mass_3d",
            "State",
            tuple(f"mp{opt}" for opt, members in MP_MOIST_MEMBERS.items() if leaf in members),
            existing_state=True,
            restart_required=True,
            wrfout_required=leaf in {"qv", "qc", "qr", "qi", "qs", "qg"},
            nest_forcedown=True,
            nest_feedback=True,
            lateral_bc=True,
        )
        for leaf in MOIST_SPECIES
    ),
    *(
        _field(
            leaf,
            NUMBER_WRFOUT_NAME[leaf],
            NUMBER_REGISTRY_MEMBER[leaf],
            "scalar_number",
            "mass_3d",
            "State",
            tuple(f"mp{opt}" for opt, members in MP_NUMBER_MEMBERS.items() if leaf in members),
            existing_state=leaf in NUMBER_SPECIES_EXISTING,
            additive_state=leaf in NUMBER_SPECIES_ADDITIVE,
            restart_required=True,
            wrfout_required=True,
            nest_forcedown=True,
            nest_feedback=True,
            lateral_bc=False,
            notes="WRF have_bcs_scalar controls lateral scalar bdy; nests force/feedback active scalars.",
        )
        for leaf in NUMBER_SPECIES
    ),
    *(
        _field(
            leaf,
            ACCUMULATOR_WRFOUT_NAME[leaf],
            ACCUMULATOR_WRFOUT_NAME[leaf],
            "accumulator",
            "surface_2d",
            "State",
            ("microphysics", "cumulus") if leaf == "rainc_acc" else ("microphysics",),
            existing_state=leaf in ACCUMULATORS_EXISTING,
            additive_state=leaf in ACCUMULATORS_ADDITIVE,
            restart_required=True,
            wrfout_required=True,
            notes="History/restart accumulator; updated as increments by scheme adapters.",
        )
        for leaf in ACCUMULATORS
    ),
    _field(
        "qke",
        "QKE",
        "qke",
        "pbl_scalar",
        "mass_3d",
        "State",
        ("pbl5", "pbl8"),
        existing_state=True,
        restart_required=True,
        wrfout_required=True,
        nest_forcedown=True,
        nest_feedback=True,
        lateral_bc=False,
        notes="Persistent TKE leaf for TKE-based PBL schemes; WRF MYNN advected scalar member is qke_adv.",
    ),
    *(
        _field(
            leaf,
            leaf.upper(),
            leaf,
            "microphysics_diagnostic",
            "mass_3d",
            "PhysicsDiagnostics",
            ("mp3", "mp4", "mp6", "mp8", "mp16"),
            notes="Effective radius diagnostic/carry; not a dycore State leaf.",
        )
        for leaf in ("re_cloud", "re_ice", "re_snow")
    ),
    *(
        _field(
            leaf.lower(),
            leaf,
            leaf.lower(),
            "cumulus_tendency",
            "mass_3d",
            "PhysicsCarry",
            ("cu1", "cu3", "cu6", "cu16", "mp10"),
            notes="WRF R*CUTEN state/tendency family carried between physics driver calls.",
        )
        for leaf in (
            "RUCUTEN",
            "RVCUTEN",
            "RTHCUTEN",
            "RQVCUTEN",
            "RQRCUTEN",
            "RQCCUTEN",
            "RQSCUTEN",
            "RQICUTEN",
            "RQCNCUTEN",
            "RQINCUTEN",
        )
    ),
    _field("raincv", "RAINCV", "RAINCV", "cumulus_diagnostic", "surface_2d", "PhysicsDiagnostics", ("cu1", "cu3", "cu6", "cu16")),
    _field("rainshv", "RAINSHV", "RAINSHV", "cumulus_diagnostic", "surface_2d", "PhysicsDiagnostics", ("cu1", "cu3", "cu6", "cu16")),
    _field("nca", "NCA", "NCA", "cumulus_carry", "surface_2d", "PhysicsCarry", ("cu1",), notes="KF relaxation counter."),
    _field("w0avg", "W0AVG", "w0avg", "cumulus_carry", "mass_3d", "PhysicsCarry", ("cu1",), notes="KF average vertical velocity."),
    *(
        _field(leaf, leaf.upper(), leaf, "cumulus_carry", "mass_3d", "PhysicsCarry", ("cu3",))
        for leaf in ("cugd_qvten", "cugd_tten", "cugd_qvtens", "cugd_ttens", "cugd_qcten")
    ),
    *(
        _field(leaf, leaf.upper(), leaf, "cumulus_carry", "surface_2d", "PhysicsCarry", ("cu3",))
        for leaf in ("xmb_shallow", "k22_shallow", "kbcon_shallow", "ktop_shallow")
    ),
    _field("tke_pbl", "TKE_PBL", "tke_pbl", "pbl_diagnostic", "mass_3d", "PhysicsDiagnostics", ("pbl2", "pbl5")),
    _field("el_pbl", "EL_PBL", "el_pbl", "pbl_diagnostic", "mass_3d", "PhysicsDiagnostics", ("pbl2", "pbl5")),
    *(
        _field(leaf, leaf.upper(), leaf, "pbl_diagnostic", "mass_3d", "PhysicsDiagnostics", ("pbl5",))
        for leaf in ("sh3d", "sm3d", "tsq", "qsq", "cov")
    ),
    *(
        _field(leaf, leaf.upper(), leaf, "land_carry", "surface_2d", "PhysicsCarry", ("surface2",))
        for leaf in ("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn")
    ),
)

FIELD_SPECS_BY_LEAF: Mapping[str, RegistryFieldSpec] = {spec.leaf: spec for spec in FIELD_SPECS}


CUMULUS_TENDENCY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: ("rucuten", "rvcuten", "rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
    3: ("rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
    6: ("rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
    16: ("rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
}

CUMULUS_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: ("w0avg", "nca"),
    3: (
        "cugd_qvten",
        "cugd_tten",
        "cugd_qvtens",
        "cugd_ttens",
        "cugd_qcten",
        "xmb_shallow",
        "k22_shallow",
        "kbcon_shallow",
        "ktop_shallow",
    ),
    6: (),
    16: (),
}

PBL_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: (),
    2: ("tke_pbl", "el_pbl"),
    5: ("qke",),
    7: (),
    8: ("qke",),
}

PBL_DIAGNOSTIC_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: ("pblh",),
    2: ("pblh", "kpbl", "mixht", "tke_pbl", "exch_h", "exch_m", "el_pbl"),
    5: ("pblh", "tke_pbl", "sh3d", "sm3d", "tsq", "qsq", "cov", "el_pbl"),
    7: ("pblh",),
    8: ("pblh", "tke_pbl", "dlk", "exch_h", "exch_m"),
}

LAND_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    2: ("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn"),
    4: ("NoahMPLandState",),
}

NOAH_CLASSIC_NUM_SOIL_LAYERS = 4


@dataclass(frozen=True)
class NestFieldEntry:
    """One field in the Registry-driven nest forcedown/feedback/boundary list."""

    leaf: str
    wrf_name: str
    kind: str
    stagger: str
    forcedown: bool
    feedback: bool
    lateral_bc: bool


# Core prognostic/base list. ``qv`` is deliberately core because WRF nests force
# water vapor even when no microphysics package is active.
_NEST_CORE: tuple[NestFieldEntry, ...] = (
    NestFieldEntry("u", "U", "prognostic", "X", True, True, True),
    NestFieldEntry("v", "V", "prognostic", "Y", True, True, True),
    NestFieldEntry("w", "W", "prognostic", "Z", True, True, True),
    NestFieldEntry("ph_perturbation", "PH", "prognostic", "Z", True, True, True),
    NestFieldEntry("theta", "T", "prognostic", "", True, True, True),
    NestFieldEntry("mu_perturbation", "MU", "prognostic", "", True, True, True),
    NestFieldEntry("qv", "QVAPOR", "moist", "", True, True, True),
    NestFieldEntry("phb_bdy", "PHB", "base", "Z", True, False, True),
    NestFieldEntry("pb_bdy", "PB", "base", "", True, False, True),
    NestFieldEntry("mub_bdy", "MUB", "base", "", True, False, True),
)


def moist_species_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return active WRF moist-array members for an accepted microphysics option."""

    return MP_MOIST_MEMBERS[int(mp_physics)]


def number_species_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return active WRF scalar number leaves for an accepted microphysics option."""

    return MP_NUMBER_MEMBERS[int(mp_physics)]


def state_leaves_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return State leaves an MP option may read/write."""

    return (*moist_species_for_mp(mp_physics), *number_species_for_mp(mp_physics))


def wrfout_names_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return wrfout names introduced by the active MP option."""

    leaves = state_leaves_for_mp(mp_physics)
    names: list[str] = []
    for leaf in leaves:
        if leaf in MOIST_WRFOUT_NAME:
            names.append(MOIST_WRFOUT_NAME[leaf])
        elif leaf in NUMBER_WRFOUT_NAME:
            names.append(NUMBER_WRFOUT_NAME[leaf])
    return tuple(names)


def physics_state_append_order() -> tuple[str, ...]:
    """Return the append-only State leaves frozen by v0.6.0 S0."""

    return V060_ADDITIVE_STATE_LEAVES


def nest_field_list(
    mp_physics: int = 8,
    *,
    bl_pbl_physics: int = 5,
    cu_physics: int = 0,
    include_numbers: bool = True,
) -> tuple[NestFieldEntry, ...]:
    """Return the frozen, scheme-aware nest forcedown/feedback/boundary list.

    v0.5.0 nesting and v0.6.0 physics-expansion must call this function instead
    of hardcoding the active moisture/scalar set. Moist members get forcedown,
    feedback, and lateral boundary carry. Scalar number fields and MYNN ``qke``
    get forcedown/feedback but default to no lateral scalar bdy unless a later
    lane explicitly enables WRF ``have_bcs_scalar`` semantics.
    """

    int(mp_physics)
    int(bl_pbl_physics)
    int(cu_physics)
    entries: list[NestFieldEntry] = list(_NEST_CORE)
    seen = {entry.leaf for entry in entries}

    for leaf in moist_species_for_mp(mp_physics):
        if leaf in seen:
            continue
        entries.append(NestFieldEntry(leaf, MOIST_WRFOUT_NAME[leaf], "moist", "", True, True, True))
        seen.add(leaf)

    if include_numbers:
        for leaf in number_species_for_mp(mp_physics):
            if leaf in seen:
                continue
            entries.append(NestFieldEntry(leaf, NUMBER_WRFOUT_NAME[leaf], "scalar", "", True, True, False))
            seen.add(leaf)
        if int(bl_pbl_physics) in (5, 8) and "qke" not in seen:
            entries.append(NestFieldEntry("qke", "QKE", "scalar", "", True, True, False))

    return tuple(entries)


def assert_registry_consistent() -> None:
    """Fail-closed internal consistency checks for the frozen registry."""

    for key, accepted in ACCEPTED_NAMELIST_OPTIONS.items():
        assert len(set(accepted)) == len(accepted), f"{key} has duplicate accepted values"

    for opt in ACCEPTED_MP_PHYSICS:
        assert opt in MP_SCHEMES, f"missing MP scheme metadata for {opt}"
        assert opt in MP_MOIST_MEMBERS, f"missing moist members for mp={opt}"
        assert opt in MP_NUMBER_MEMBERS, f"missing number members for mp={opt}"
    for opt in ACCEPTED_BL_PBL_PHYSICS:
        assert opt in PBL_SCHEMES, f"missing PBL metadata for {opt}"
        assert opt in PBL_CARRY_MEMBERS, f"missing PBL carry metadata for {opt}"
    for opt in ACCEPTED_SF_SFCLAY_PHYSICS:
        assert opt in SFCLAY_SCHEMES, f"missing surface-layer metadata for {opt}"
    for opt in ACCEPTED_CU_PHYSICS:
        assert opt in CU_SCHEMES, f"missing cumulus metadata for {opt}"
        assert opt in CUMULUS_CARRY_MEMBERS, f"missing cumulus carry metadata for {opt}"
        assert opt in CUMULUS_TENDENCY_MEMBERS, f"missing cumulus tendency metadata for {opt}"
    for opt in ACCEPTED_SF_SURFACE_PHYSICS:
        assert opt in SURFACE_SCHEMES, f"missing surface metadata for {opt}"
        assert opt in LAND_CARRY_MEMBERS, f"missing land carry metadata for {opt}"

    for opt, members in MP_MOIST_MEMBERS.items():
        for leaf in members:
            assert leaf in MOIST_SPECIES, f"mp={opt} moist member {leaf!r} not in MOIST_SPECIES"
            assert leaf in MOIST_WRFOUT_NAME, f"moist {leaf!r} missing wrfout name"
    for opt, members in MP_NUMBER_MEMBERS.items():
        for leaf in members:
            assert leaf in NUMBER_SPECIES, f"mp={opt} number member {leaf!r} not in NUMBER_SPECIES"
            assert leaf in NUMBER_REGISTRY_MEMBER, f"number {leaf!r} missing Registry scalar member"
            assert leaf in NUMBER_WRFOUT_NAME, f"number {leaf!r} missing wrfout name"
    for leaf in ACCUMULATORS:
        assert leaf in ACCUMULATOR_WRFOUT_NAME, f"accumulator {leaf!r} missing wrfout name"

    field_names = [spec.leaf for spec in FIELD_SPECS]
    assert len(field_names) == len(set(field_names)), "FIELD_SPECS has duplicate leaves"
    for leaf in V060_EXISTING_STATE_PHYSICS_LEAVES + V060_ADDITIVE_STATE_LEAVES:
        assert leaf in FIELD_SPECS_BY_LEAF, f"{leaf!r} missing from FIELD_SPECS"

    nested = nest_field_list(mp_physics=10, bl_pbl_physics=5, cu_physics=1)
    assert "qv" in {entry.leaf for entry in nested}, "qv must remain in nest core"
    assert len(nested) == len({entry.leaf for entry in nested}), "duplicate nest field entry"
    for entry in nested:
        assert entry.kind in {"prognostic", "moist", "scalar", "base"}


__all__ = [
    "PHYSICS_REGISTRY_VERSION",
    "SchemeOption",
    "ACCEPTED_MP_PHYSICS",
    "ACCEPTED_BL_PBL_PHYSICS",
    "ACCEPTED_SF_SFCLAY_PHYSICS",
    "ACCEPTED_CU_PHYSICS",
    "ACCEPTED_SF_SURFACE_PHYSICS",
    "ACCEPTED_RA_SW_PHYSICS",
    "ACCEPTED_RA_LW_PHYSICS",
    "ACCEPTED_NAMELIST_OPTIONS",
    "MP_SCHEMES",
    "PBL_SCHEMES",
    "SFCLAY_SCHEMES",
    "CU_SCHEMES",
    "SURFACE_SCHEMES",
    "MOIST_SPECIES",
    "MOIST_WRFOUT_NAME",
    "MP_MOIST_MEMBERS",
    "NUMBER_SPECIES",
    "NUMBER_SPECIES_EXISTING",
    "NUMBER_SPECIES_ADDITIVE",
    "NUMBER_REGISTRY_MEMBER",
    "NUMBER_WRFOUT_NAME",
    "MP_NUMBER_MEMBERS",
    "ACCUMULATORS",
    "ACCUMULATORS_EXISTING",
    "ACCUMULATORS_ADDITIVE",
    "ACCUMULATOR_WRFOUT_NAME",
    "V060_EXISTING_STATE_PHYSICS_LEAVES",
    "V060_ADDITIVE_STATE_LEAVES",
    "RegistryFieldSpec",
    "FIELD_SPECS",
    "FIELD_SPECS_BY_LEAF",
    "CUMULUS_TENDENCY_MEMBERS",
    "CUMULUS_CARRY_MEMBERS",
    "PBL_CARRY_MEMBERS",
    "PBL_DIAGNOSTIC_MEMBERS",
    "LAND_CARRY_MEMBERS",
    "NOAH_CLASSIC_NUM_SOIL_LAYERS",
    "NestFieldEntry",
    "moist_species_for_mp",
    "number_species_for_mp",
    "state_leaves_for_mp",
    "wrfout_names_for_mp",
    "physics_state_append_order",
    "nest_field_list",
    "assert_registry_consistent",
]


if __name__ == "__main__":
    assert_registry_consistent()
    print(
        "ok physics_registry consistent:",
        f"version={PHYSICS_REGISTRY_VERSION}",
        f"moist={len(MOIST_SPECIES)}",
        f"numbers={len(NUMBER_SPECIES)}",
        f"accumulators={len(ACCUMULATORS)}",
        f"append={','.join(physics_state_append_order())}",
        f"nest_fields(mp=10,cu=1)={len(nest_field_list(mp_physics=10, cu_physics=1))}",
    )
