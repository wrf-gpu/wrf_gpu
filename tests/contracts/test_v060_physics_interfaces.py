"""CPU-only contract tests for the v0.6.0 physics interface freeze."""

from __future__ import annotations

import pytest

from gpuwrf.contracts.physics_interfaces import (
    ACCUMULATOR_UPDATE_KEYS,
    SCHEME_STEP_SPECS,
    SCHEME_STEP_SPECS_BY_KEY,
    PhysicsTendency,
    assert_interfaces_consistent,
    scheme_step_spec,
)
from gpuwrf.contracts.physics_registry import (
    NUMBER_WRFOUT_NAME,
    V060_ADDITIVE_STATE_LEAVES,
    assert_registry_consistent,
    nest_field_list,
    physics_state_append_order,
    state_leaves_for_mp,
    wrfout_names_for_mp,
)
from gpuwrf.io.namelist_check import validate_supported_namelist
from gpuwrf.io.wrfout_writer import MICROPHYSICS_EXTRA_VARIABLES, WRFOUT_VARIABLE_SPECS

# v0.18 trunk: the v0.17 trunk shipped 52 SCHEME_STEP_SPECS (six merged lanes).
# The v0.18 harvest adds six Phase-1 specs plus four PBL-family reference specs:
#   Goddard-GCE mp=97, aerosol-aware Thompson mp=28, Held-Suarez ra_lw=31,
#   SBU-YLin mp=13, WSM7 mp=24, WDM7 mp=26;
#   reference-only PBL4/PBL10/PBL16/PBL17. PBL11/PBL12 specs already existed
#   as reference specs and this sprint promotes their status/evidence.
# The v0.18 RADIATION (RA) tail adds four reference-only radiation specs:
#   CAM ra_lw=3 + ra_sw=3, FLG/UCLA ra_lw=7 + ra_sw=7 (each backed by a real-WRF
#   exact-driver savepoint oracle; ra_sw=5 + ra_lw/sw=99 already had specs).
_V018_TRUNK_BASE_SPEC_COUNT = 52
_V018_HARVESTED_SPECS = 6
_V018_PBL_FAMILY_SPECS = 4
_V018_RA_TAIL_SPECS = 4
_V022_F3_CAMUW_PBL_SPECS = 1
_V018_EXPECTED_SPEC_COUNT = (
    _V018_TRUNK_BASE_SPEC_COUNT
    + _V018_HARVESTED_SPECS
    + _V018_PBL_FAMILY_SPECS
    + _V018_RA_TAIL_SPECS
    + _V022_F3_CAMUW_PBL_SPECS
)


def test_registry_self_check_and_state_append_order() -> None:
    assert_registry_consistent()
    # v0.18 trunk additive set = the registry-computed UNION of the number,
    # moist, volume and accumulator additive families:
    #   NUMBER_SPECIES_ADDITIVE (Nc, Nn, Nh [v0.17 ADR-032], nwfa, nifa [v0.16])
    #   + MOIST_SPECIES_ADDITIVE (qh [v0.17 ADR-032])
    #   + VOLUME_SPECIES (qvolg, qvolh [v0.17 ADR-032])
    #   + ACCUMULATORS_ADDITIVE (rainc_acc, hail_acc [v0.17 WSM7/WDM7 hail])
    # The State pytree appends the hail substrate, then the aerosol leaves, then
    # the hail_acc accumulator before optional runtime-only boundary leaves.
    _expected_additive = (
        "Nc", "Nn", "Nh", "nwfa", "nifa", "qh", "qvolg", "qvolh", "rainc_acc", "hail_acc",
    )
    assert physics_state_append_order() == _expected_additive
    assert V060_ADDITIVE_STATE_LEAVES == _expected_additive


def test_nest_field_list_is_registry_driven_for_two_moment_schemes() -> None:
    morrison = {entry.leaf for entry in nest_field_list(mp_physics=10, bl_pbl_physics=5)}
    assert {"qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Ns", "Nr", "Ng", "qke"} <= morrison

    wdm6 = {entry.leaf for entry in nest_field_list(mp_physics=16, bl_pbl_physics=0)}
    assert {"qv", "qc", "qr", "qi", "qs", "qg", "Nn", "Nc", "Nr"} <= wdm6
    assert "Ni" not in wdm6


def test_mp_registry_names_match_expected_wrfout_variables() -> None:
    assert state_leaves_for_mp(1) == ("qv", "qc", "qr")
    assert state_leaves_for_mp(3) == ("qv", "qc", "qr")
    assert state_leaves_for_mp(4) == ("qv", "qc", "qr", "qi", "qs")
    assert state_leaves_for_mp(10) == ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Ns", "Nr", "Ng")
    # WDM5 (mp=14): 5-class moist (no graupel) + the WDM6 Nn/Nc/Nr number leaves.
    assert state_leaves_for_mp(14) == ("qv", "qc", "qr", "qi", "qs", "Nn", "Nc", "Nr")
    assert state_leaves_for_mp(16) == ("qv", "qc", "qr", "qi", "qs", "qg", "Nn", "Nc", "Nr")
    # v0.16 aerosol-aware Thompson (mp=28): Registry thompsonaero scalars.
    assert state_leaves_for_mp(28) == ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Nc", "nwfa", "nifa")
    assert NUMBER_WRFOUT_NAME["Nn"] == "QNCCN"
    assert NUMBER_WRFOUT_NAME["nwfa"] == "QNWFA"
    assert NUMBER_WRFOUT_NAME["nifa"] == "QNIFA"
    assert "QNCLOUD" in wrfout_names_for_mp(16)
    assert "QNCCN" in wrfout_names_for_mp(16)
    assert "QNWFA" in wrfout_names_for_mp(28)
    assert "QNIFA" in wrfout_names_for_mp(28)


def test_interfaces_self_check_and_scheme_specs_cover_v060_options() -> None:
    assert_interfaces_consistent()
    # v0.18 trunk integrated count = the UNION of the v0.17 trunk (52 specs from
    #   the six merged lanes: qh + pbl + lsm-adv + cu-sas + cu-kfgrell + rad) PLUS
    #   the v0.18 harvested schemes added here:
    #   + Goddard-GCE microphysics (mp=97)
    #   + aerosol-aware Thompson microphysics (mp=28)
    #   + Held-Suarez idealized radiation (ra_lw=31)
    #   + SBU-YLin microphysics (mp=13)
    #   + WSM7 hail microphysics (mp=24)
    #   + WDM7 double-moment hail microphysics (mp=26)
    # The exact integer is asserted via the canonical-count helper above so it
    # stays in lockstep with SCHEME_STEP_SPECS as schemes are harvested.
    assert len(SCHEME_STEP_SPECS) == len(set(SCHEME_STEP_SPECS_BY_KEY))
    assert len(SCHEME_STEP_SPECS) == _V018_EXPECTED_SPEC_COUNT
    assert scheme_step_spec("microphysics", 16).writes_state[-3:] == ("Nn", "Nc", "Nr")
    assert scheme_step_spec("pbl", 2).writes_carry == ("tke_pbl", "el_pbl")
    # v0.17 GFS PBL + v0.18 Shin-Hong/GBM PBL operational specs present.
    assert scheme_step_spec("pbl", 3).owner_module.endswith("bl_gfs.py")
    assert "savepoint" in scheme_step_spec("pbl", 3).oracle.lower()
    assert scheme_step_spec("pbl", 11).owner_module.endswith("bl_shinhong.py")
    assert scheme_step_spec("pbl", 12).owner_module.endswith("bl_gbm.py")
    assert scheme_step_spec("pbl", 12).writes_state == ("u", "v", "theta", "qv", "qc", "qke")
    # v0.18 PBL reference-only specs present, each tied to a real WRF savepoint
    # oracle but not scan-wired into operational mode.
    for opt in (4, 10, 16, 17):
        spec = scheme_step_spec("pbl", opt)
        assert spec.owner_module.endswith("pbl_reference_only.py")
        assert "REFERENCE-ONLY" in spec.notes
        assert "savepoint" in spec.oracle.lower()
    assert scheme_step_spec("surface_layer", 2).owner_module.endswith("sfclay_janjic.py")
    assert scheme_step_spec("cumulus", 1).returns_accumulators == ("rainc_acc",)
    assert scheme_step_spec("cumulus", 2).writes_carry == ("cldefi",)
    assert scheme_step_spec("cumulus", 93).owner_module.endswith("cumulus_grell_devenyi.py")
    assert scheme_step_spec("cumulus", 99).writes_carry == ("w0avg", "nca")
    assert scheme_step_spec("land_surface", 2).writes_carry == ("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn")


def test_radiation_specs_are_held_rate_theta_tendencies() -> None:
    lw = scheme_step_spec("radiation", 4, "lw")
    sw = scheme_step_spec("radiation", 4, "sw")
    # Radiation is a column endpoint: it only writes a held-rate theta tendency
    # (WRF RTHRATEN), never an in-place State replacement or a new species leaf.
    assert lw.writes_state == ("theta",)
    assert sw.writes_state == ("theta",)
    assert lw.wrf_slot == "first_rk_radiation_driver"
    assert sw.wrf_slot == "first_rk_radiation_driver"
    assert "SWDOWN" in sw.diagnostics and "GLW" in lw.diagnostics

    # Classic Dudhia SW (ra_sw=1) and classic RRTM LW (ra_lw=1) follow the same
    # held-rate theta-endpoint contract.
    dudhia = scheme_step_spec("radiation", 1, "sw")
    rrtm = scheme_step_spec("radiation", 1, "lw")
    assert dudhia.writes_state == ("theta",)
    assert rrtm.writes_state == ("theta",)
    assert dudhia.wrf_slot == "first_rk_radiation_driver"
    assert rrtm.wrf_slot == "first_rk_radiation_driver"
    assert "GSW" in dudhia.diagnostics and "GLW" in rrtm.diagnostics
    assert dudhia.owner_module == "src/gpuwrf/physics/ra_sw_dudhia.py"

    for code in (3, 5, 7, 99):
        lw_tail = scheme_step_spec("radiation", code, "lw")
        sw_tail = scheme_step_spec("radiation", code, "sw")
        for spec in (lw_tail, sw_tail):
            assert spec.writes_state == ("theta",)
            assert spec.wrf_slot == "first_rk_radiation_driver"
            assert "REFERENCE-ONLY" in spec.notes
            assert f"proofs/v018/savepoints/ra_tail_wrf/ra{code}_wrf_real.json" in spec.oracle
            assert "v0.18 exact-driver real-WRF" in spec.oracle
            assert "NOT a self-compare" in spec.oracle
            assert "REFERENCE-ONLY / RED" not in spec.notes
            assert "STATUS: REFERENCE-ONLY / RED" not in spec.notes

    for spec in (
        scheme_step_spec("radiation", 5, "sw"),
        scheme_step_spec("radiation", 5, "lw"),
        scheme_step_spec("radiation", 99, "sw"),
        scheme_step_spec("radiation", 99, "lw"),
    ):
        assert spec.writes_state == ("theta",)
        assert spec.wrf_slot == "first_rk_radiation_driver"
        assert "REFERENCE-ONLY" in spec.notes


def test_physics_tendency_validates_unknown_keys() -> None:
    assert "rainc_acc" in ACCUMULATOR_UPDATE_KEYS
    PhysicsTendency(state_tendencies={"theta": object()}, accumulator_increments={"rainc_acc": object()}).validate_keys()
    with pytest.raises(ValueError, match="unknown state_tendency"):
        PhysicsTendency(state_tendencies={"bad_leaf": object()}).validate_keys()


def test_v060_namelist_accept_matrix_and_wrfout_forward_names() -> None:
    validate_supported_namelist(
        {
            "physics": {
                "mp_physics": [1, 2, 3, 4, 6, 8, 10, 16],
                "cu_physics": [0, 1, 2, 3, 4, 5, 6, 14, 16, 93, 94, 95, 96, 99],
                "bl_pbl_physics": [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 16, 17],
                "sf_sfclay_physics": [0, 1, 2, 5, 7],
                "sf_surface_physics": [0, 2, 3, 4, 7, 8],
                "ra_sw_physics": [0, 1, 2, 4, 5, 99],
                "ra_lw_physics": [0, 1, 4, 5, 99],
            }
        }
    )
    for name in ("QNSNOW", "QNGRAUPEL", "QNCLOUD", "QNCCN"):
        assert name in MICROPHYSICS_EXTRA_VARIABLES
        assert name in WRFOUT_VARIABLE_SPECS
