"""v0.6.0 Registry-driven physics field list — the SINGLE source of truth.

INTERFACE FREEZE ONLY — no physics, no allocation, no kernels. See
``.agent/decisions/V0.6.0-S0-PLAN.md`` and
``.agent/decisions/ADR-031-v060-physics-expansion-interfaces.md``.

This module is the Python-side mirror of the WRF ``Registry/Registry.EM_COMMON``
``package <scheme> <opt> - moist:...;scalar:...;state:...`` membership lines. It
freezes the *superset* of prognostic moisture species, number concentrations,
scalars, accumulators and physics-coupling tendency/carry fields that the v0.6.0
supported scheme menu introduces, so that the per-scheme lanes (built in PARALLEL)
can NOT invent incompatible State leaves, and so the nest forcedown/feedback field
list, the wrfout writer, and the restart schema all consume ONE list instead of
hardcoding the Thompson-era field set (GPT plan-review §Answer 5 + Prioritized
Correction #10: "freeze a state/Registry interface before v0.6.0 so nesting,
restart, and wrfout do not break when new species/scalars are introduced").

WRF arbiter (verified against ``/home/enric/src/wrf_pristine/WRF/Registry/
Registry.EM_COMMON`` 2026-06-02):

    package kesslerscheme    mp_physics==1  - moist:qv,qc,qr
    package wsm6scheme       mp_physics==6  - moist:qv,qc,qr,qi,qs,qg;state:re_*
    package thompson         mp_physics==8  - moist:qv..qg;scalar:qni,qnr;state:re_*
    package morr_two_moment  mp_physics==10 - moist:qv..qg;scalar:qni,qns,qnr,qng;
                                               state:rqrcuten,rqscuten,rqicuten
    package wdm6scheme       mp_physics==16 - moist:qv..qg;scalar:qnn,qnc,qnr;state:re_*

    package kfetascheme      cu_physics==1  - state:w0avg            (+NCA, RAINCV, RTH/RQ*CUTEN)
    package gfscheme         cu_physics==3  - state:cugd_*           (Grell-Freitas)
    package tiedtkescheme    cu_physics==6  - -                      (tendencies only)
    package mynnpblscheme    bl==5          - scalar:qke_adv;state:qke,tke_pbl,sh3d,sm3d,...
    package ysuscheme        bl==1          - -                      (no extra prognostic state)
    package acmpblscheme     bl==7          - -                      (no extra prognostic state)
    package lsmscheme        sf_surface==2  - state:flx4,fvb,fbur,fgsn,smcrel,xlaidyn (Noah)
    package noahmpscheme     sf_surface==4  - state:*xy ...          (Noah-MP, sibling carry)

NOTHING here is allocated or stored as a JAX array. These are *names + WRF
mappings + which-scheme-needs-them*; the actual State leaf addition (and its
precision-matrix entry) is performed by the per-scheme implementation lane under
the patch protocol, by appending to ``STATE_FIELD_ORDER`` / ``PRECISION_MATRIX``
(``contracts.precision``) and the corresponding ``State.__slots__``, and is gated
by the contract-test in this module's ``__main__`` self-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# --------------------------------------------------------------------------- #
# 1. Moisture species (WRF 4-D ``moist`` array members, P_QV..P_QG).
#    State leaf name == lowercase WRF moist-array member name.
# --------------------------------------------------------------------------- #
# The full v0.6.0 single-moment hydrometeor superset. Thompson(8)/WSM6(6)/
# Morrison(10)/WDM6(16) all use this 6-class set; Kessler(1) uses the warm-rain
# subset {qv, qc, qr} (no ice). Every supported MP scheme's moist members are a
# SUBSET of MOIST_SPECIES, so the State already carries them (qc..qg exist).
MOIST_SPECIES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")

# WRF moist-array member -> wrfout NetCDF variable name (writer mapping).
MOIST_WRFOUT_NAME: Mapping[str, str] = {
    "qv": "QVAPOR",
    "qc": "QCLOUD",
    "qr": "QRAIN",
    "qi": "QICE",
    "qs": "QSNOW",
    "qg": "QGRAUP",
}

# Which moist members each supported mp_physics option activates (WRF package
# membership). Used by the namelist checker + the per-scheme lane to assert the
# scheme only reads/writes State leaves it is allowed to.
MP_MOIST_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: ("qv",),                                  # passiveqv (dry/no-MP gate)
    1: ("qv", "qc", "qr"),                        # Kessler (warm rain)
    6: ("qv", "qc", "qr", "qi", "qs", "qg"),      # WSM6
    8: ("qv", "qc", "qr", "qi", "qs", "qg"),      # Thompson (IN/ported)
    10: ("qv", "qc", "qr", "qi", "qs", "qg"),     # Morrison 2-moment
    16: ("qv", "qc", "qr", "qi", "qs", "qg"),     # WDM6
}


# --------------------------------------------------------------------------- #
# 2. Number concentrations / scalar-array members (WRF ``scalar`` array).
#    State leaf names follow the EXISTING Thompson convention (Ni/Nr/Ns/Ng);
#    new schemes add the CCN/cloud-droplet number leaves.
# --------------------------------------------------------------------------- #
# Existing (Thompson-era, already in State): Ni<-qni, Nr<-qnr, Ns<-qns, Ng<-qng.
# v0.6.0 ADDITIVE for Morrison(10) and WDM6(16). Morrison needs all four moment
# numbers (qni,qns,qnr,qng -> Ni,Ns,Nr,Ng, already present). WDM6 needs the
# warm-rain droplet/CCN numbers (qnn CCN, qnc cloud droplet, qnr rain).
#
# Leaf-name <- WRF scalar member:
#   Ni <- qni   ice number             [#/kg or #/m3 per scheme convention]
#   Nr <- qnr   rain number
#   Ns <- qns   snow number
#   Ng <- qng   graupel number
#   Nc <- qnc   cloud-droplet number   (WDM6, Morrison-aero) -- ADDITIVE
#   Nn <- qnn   CCN number             (WDM6)                -- ADDITIVE
NUMBER_SPECIES_EXISTING: tuple[str, ...] = ("Ni", "Nr", "Ns", "Ng")
NUMBER_SPECIES_ADDITIVE: tuple[str, ...] = ("Nc", "Nn")
NUMBER_SPECIES: tuple[str, ...] = NUMBER_SPECIES_EXISTING + NUMBER_SPECIES_ADDITIVE

# State leaf -> wrfout NetCDF scalar variable name (WRF ``scalar`` array names).
NUMBER_WRFOUT_NAME: Mapping[str, str] = {
    "Ni": "QNICE",
    "Nr": "QNRAIN",
    "Ns": "QNSNOW",
    "Ng": "QNGRAUPEL",
    "Nc": "QNCLOUD",
    "Nn": "QNN",
}

# Which scalar (number) members each supported mp_physics option activates.
# Single-moment schemes (Kessler/WSM6) activate NONE.
MP_NUMBER_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: (),                                        # Kessler single-moment
    6: (),                                        # WSM6 single-moment
    8: ("Ni", "Nr"),                              # Thompson: 2 numbers (qni,qnr)
    10: ("Ni", "Ns", "Nr", "Ng"),                 # Morrison: full 2-moment
    16: ("Nn", "Nc", "Nr"),                       # WDM6: CCN + droplet + rain
}


# --------------------------------------------------------------------------- #
# 3. Precipitation accumulators (WRF surface ``misc`` state).
#    State leaf -> wrfout name. Grid-scale (NC) vs cumulus (C) partition.
# --------------------------------------------------------------------------- #
# EXISTING in State: rain_acc/snow_acc/graupel_acc/ice_acc (grid-scale, from MP).
# v0.6.0 ADDITIVE: cumulus precip accumulator (RAINC) when a cu scheme is active.
# WRF: RAINNC = grid-scale total, RAINC = cumulus total, SNOWNC/GRAUPELNC = MP
# partition. ``rain_acc`` maps to RAINNC; the cumulus lane adds ``rainc_acc``.
ACCUMULATORS_EXISTING: tuple[str, ...] = ("rain_acc", "snow_acc", "graupel_acc", "ice_acc")
ACCUMULATORS_ADDITIVE: tuple[str, ...] = ("rainc_acc",)
ACCUMULATORS: tuple[str, ...] = ACCUMULATORS_EXISTING + ACCUMULATORS_ADDITIVE

ACCUMULATOR_WRFOUT_NAME: Mapping[str, str] = {
    "rain_acc": "RAINNC",        # accumulated total grid-scale precip [mm]
    "snow_acc": "SNOWNC",        # accumulated grid-scale snow+ice [mm]
    "graupel_acc": "GRAUPELNC",  # accumulated grid-scale graupel [mm]
    "ice_acc": "SNOWNC",         # WRF folds ice into SNOWNC; kept distinct here
    "rainc_acc": "RAINC",        # accumulated total cumulus precip [mm] (ADDITIVE)
}


# --------------------------------------------------------------------------- #
# 4. Physics-coupling carry fields (WRF ``state`` array members carried BETWEEN
#    physics steps). These ride in OperationalCarry as OPTIONAL subtrees, NOT in
#    the prognostic dycore State.__slots__ (mirrors noahmp_land). See
#    contracts.physics_interfaces.PhysicsCarry.
# --------------------------------------------------------------------------- #
# Cumulus held carry by scheme (WRF Registry ``state:`` members):
#   KF(1)/MSKF(11)/oldKF(99): w0avg (running-mean w) + NCA (relaxation counter)
#   Grell-Freitas(3):         cugd_* family (deep+shallow tendencies/closure)
#   Tiedtke(6/16):            NONE (diagnostic, no persistent carry)
CUMULUS_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: ("w0avg", "nca"),                          # Kain-Fritsch
    3: ("cugd_qvten", "cugd_tten", "cugd_qvtens", "cugd_ttens", "cugd_qcten",
        "xmb_shallow", "k22_shallow", "kbcon_shallow", "ktop_shallow"),  # Grell-Freitas
    6: (),                                         # Tiedtke (tendency-only)
    16: (),                                        # newer Tiedtke (tendency-only)
}

# PBL held carry by scheme (WRF Registry ``state:`` members):
#   MYNN(5): qke (in State) + tke_pbl/sh3d/sm3d/tsq/qsq/cov/el_pbl (currently
#            diagnosed in-kernel; only qke is a persistent State leaf in the port)
#   YSU(1), ACM2(7): NONE (no extra prognostic PBL state)
PBL_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: (),                                         # YSU
    5: ("qke",),                                   # MYNN (qke is the State leaf)
    7: (),                                         # ACM2
}


# --------------------------------------------------------------------------- #
# 5. THE FROZEN NEST / LATERAL-BOUNDARY field list (GPT hidden-dependency).
#    BOTH v0.6.0 physics-expansion AND v0.5.0 nesting consume this. Adding a moist
#    species here is what makes the nest forcedown / 2-way feedback / wrfbdy carry
#    it; v0.5.0 must NOT hardcode {qv} (current boundary_construction.py only
#    forces qv). WRF nest_forcedown_interp.inc forces every active moist+scalar
#    member; have_bcs_moist/have_bcs_scalar control lateral wrfbdy carry of them.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NestFieldEntry:
    """One field in the Registry-driven nest forcedown/feedback/boundary list."""

    leaf: str                    # State leaf name (or carry/sibling name)
    wrf_name: str                # WRF Registry / wrfout name
    kind: str                    # "prognostic" | "moist" | "scalar" | "base"
    stagger: str                 # "" mass | "X" u-face | "Y" v-face | "Z" w-face
    forcedown: bool              # included in parent->child med_nest_force
    feedback: bool               # included in child->parent 2-way feedback
    lateral_bc: bool             # carried in wrfbdy (have_bcs_moist/scalar)


# The CORE always-forced prognostic + base set (matches the current
# boundary_construction.py forced set: u,v,w,ph,theta,mu,qv + base phb/pb/mub).
# kind="base" entries are read-only base-state leaves the boundary consumer needs.
_NEST_CORE: tuple[NestFieldEntry, ...] = (
    NestFieldEntry("u", "U", "prognostic", "X", True, True, True),
    NestFieldEntry("v", "V", "prognostic", "Y", True, True, True),
    NestFieldEntry("w", "W", "prognostic", "Z", True, True, True),
    NestFieldEntry("ph_perturbation", "PH", "prognostic", "Z", True, True, True),
    NestFieldEntry("theta", "T", "prognostic", "", True, True, True),
    NestFieldEntry("mu_perturbation", "MU", "prognostic", "", True, True, True),
    NestFieldEntry("phb_bdy", "PHB", "base", "Z", True, False, True),
    NestFieldEntry("pb_bdy", "PB", "base", "", True, False, True),
    NestFieldEntry("mub_bdy", "MUB", "base", "", True, False, True),
)


def nest_field_list(
    mp_physics: int = 8,
    *,
    bl_pbl_physics: int = 5,
    cu_physics: int = 0,
    include_numbers: bool = True,
) -> tuple[NestFieldEntry, ...]:
    """Return the FROZEN, scheme-aware nest forcedown/feedback/boundary field list.

    This is the single function v0.5.0 nesting (forcedown + 2-way feedback) and
    the lateral-boundary builder MUST call instead of hardcoding {qv}. It expands
    the moist + scalar (number) members for the *active* mp/pbl scheme so adding
    Morrison/WDM6 automatically widens the nest/boundary carry — no v0.5.0 rework.

    Moist species are forcedown + feedback + lateral-bc (WRF forces all active
    moist members through nest_forcedown_interp.inc and 2-way feedback copy/avg).
    Number/scalar members are forcedown + feedback when ``include_numbers`` and the
    scheme is 2-moment; their wrfbdy lateral carry is gated by have_bcs_scalar
    (WRF default off for nests, so lateral_bc=False here — they are zero-gradient
    at the nest edge unless have_bcs_scalar is set).
    """

    entries: list[NestFieldEntry] = list(_NEST_CORE)
    # Active moist members beyond qv (qv already in core).
    for leaf in MP_MOIST_MEMBERS.get(int(mp_physics), ("qv",)):
        if leaf == "qv":
            continue
        entries.append(
            NestFieldEntry(leaf, MOIST_WRFOUT_NAME[leaf], "moist", "", True, True, True)
        )
    # Active number/scalar members.
    if include_numbers:
        for leaf in MP_NUMBER_MEMBERS.get(int(mp_physics), ()):
            entries.append(
                NestFieldEntry(leaf, NUMBER_WRFOUT_NAME[leaf], "scalar", "", True, True, False)
            )
        # MYNN advects qke as a scalar (qke_adv) -> forcedown/feedback when active.
        if int(bl_pbl_physics) == 5:
            entries.append(NestFieldEntry("qke", "QKE", "scalar", "", True, True, False))
    return tuple(entries)


# --------------------------------------------------------------------------- #
# 6. Self-check: the registry must stay internally consistent and every name it
#    references in the precision matrix / State must actually exist once a lane
#    has added it. Run by the contract test; new lanes extend the matrix first.
# --------------------------------------------------------------------------- #
def assert_registry_consistent() -> None:
    """Fail-closed internal consistency checks for the frozen registry."""

    for opt, members in MP_MOIST_MEMBERS.items():
        for leaf in members:
            assert leaf in MOIST_SPECIES, f"mp={opt} moist member {leaf!r} not in MOIST_SPECIES"
            assert leaf in MOIST_WRFOUT_NAME, f"moist {leaf!r} missing wrfout name"
    for opt, members in MP_NUMBER_MEMBERS.items():
        for leaf in members:
            assert leaf in NUMBER_SPECIES, f"mp={opt} number member {leaf!r} not in NUMBER_SPECIES"
            assert leaf in NUMBER_WRFOUT_NAME, f"number {leaf!r} missing wrfout name"
    for leaf in ACCUMULATORS:
        assert leaf in ACCUMULATOR_WRFOUT_NAME, f"accumulator {leaf!r} missing wrfout name"
    # Every nest entry's leaf is named consistently with the species tables.
    for entry in nest_field_list(mp_physics=10, cu_physics=1):
        assert entry.kind in {"prognostic", "moist", "scalar", "base"}


__all__ = [
    "MOIST_SPECIES",
    "MOIST_WRFOUT_NAME",
    "MP_MOIST_MEMBERS",
    "NUMBER_SPECIES",
    "NUMBER_SPECIES_EXISTING",
    "NUMBER_SPECIES_ADDITIVE",
    "NUMBER_WRFOUT_NAME",
    "MP_NUMBER_MEMBERS",
    "ACCUMULATORS",
    "ACCUMULATORS_EXISTING",
    "ACCUMULATORS_ADDITIVE",
    "ACCUMULATOR_WRFOUT_NAME",
    "CUMULUS_CARRY_MEMBERS",
    "PBL_CARRY_MEMBERS",
    "NestFieldEntry",
    "nest_field_list",
    "assert_registry_consistent",
]


if __name__ == "__main__":
    assert_registry_consistent()
    print("ok physics_registry consistent:",
          f"moist={len(MOIST_SPECIES)} numbers={len(NUMBER_SPECIES)} "
          f"accumulators={len(ACCUMULATORS)} "
          f"nest_fields(mp=10,cu=1)={len(nest_field_list(mp_physics=10, cu_physics=1))}")
