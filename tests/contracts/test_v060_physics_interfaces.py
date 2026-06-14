"""CPU-only contract tests for the v0.6.0 physics interface freeze."""

from __future__ import annotations

import pytest

from gpuwrf.contracts.physics_interfaces import (
    ACCUMULATOR_UPDATE_KEYS,
    SCHEME_STEP_SPECS,
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


def test_registry_self_check_and_state_append_order() -> None:
    assert_registry_consistent()
    # v0.16 appends the aerosol-aware Thompson (mp=28) nwfa/nifa leaves to the
    # additive set (append-only AFTER Nc/Nn; the State pytree appends them at
    # the very END of __slots__, after rainc_acc -- see contracts/state.py).
    assert physics_state_append_order() == ("Nc", "Nn", "nwfa", "nifa", "rainc_acc")
    assert V060_ADDITIVE_STATE_LEAVES == ("Nc", "Nn", "nwfa", "nifa", "rainc_acc")


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
    # 31 single-option specs (9 microphysics incl. Purdue-Lin + WSM3/WSM5 + v0.13
    # Tier-3 WDM5(14) + 6 PBL incl. BouLac + MRF(99) + 6 surface-layer incl. v0.13
    # Tier-3 GFS(3) + old-MM5(91) + 7 cumulus incl. BMJ cu=2 + v0.13 Tier-3
    # reference-only Grell-3D cu=5 + KSAS cu=14 + 3 land-surface incl. v0.13 Tier-3
    # slab(1)) + 6 radiation variants (RRTMG LW/SW under option 4, classic RRTM LW +
    # Dudhia SW under option 1, v0.13 Tier-3 GSFC/Chou-Suarez SW under option 2, v0.13
    # Tier-3 reference-only GSFC/Goddard NUWRF LW under option 5).
    # +1 in v0.16: aerosol-aware Thompson microphysics (mp=28).
    assert len(SCHEME_STEP_SPECS) == 38
    assert scheme_step_spec("microphysics", 16).writes_state[-3:] == ("Nn", "Nc", "Nr")
    assert scheme_step_spec("pbl", 2).writes_carry == ("tke_pbl", "el_pbl")
    assert scheme_step_spec("surface_layer", 2).owner_module.endswith("sfclay_janjic.py")
    assert scheme_step_spec("cumulus", 1).returns_accumulators == ("rainc_acc",)
    assert scheme_step_spec("cumulus", 2).writes_carry == ("cldefi",)
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
                "cu_physics": [0, 1, 2, 3, 5, 6, 14, 16],
                "bl_pbl_physics": [0, 1, 2, 5, 7],
                "sf_sfclay_physics": [0, 1, 2, 5, 7],
                "sf_surface_physics": [0, 2, 4],
                "ra_sw_physics": [0, 1, 4],
                "ra_lw_physics": [0, 1, 4],
            }
        }
    )
    for name in ("QNSNOW", "QNGRAUPEL", "QNCLOUD", "QNCCN"):
        assert name in MICROPHYSICS_EXTRA_VARIABLES
        assert name in WRFOUT_VARIABLE_SPECS
