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
    assert physics_state_append_order() == ("Nc", "Nn", "rainc_acc")
    assert V060_ADDITIVE_STATE_LEAVES == ("Nc", "Nn", "rainc_acc")


def test_nest_field_list_is_registry_driven_for_two_moment_schemes() -> None:
    morrison = {entry.leaf for entry in nest_field_list(mp_physics=10, bl_pbl_physics=5)}
    assert {"qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Ns", "Nr", "Ng", "qke"} <= morrison

    wdm6 = {entry.leaf for entry in nest_field_list(mp_physics=16, bl_pbl_physics=0)}
    assert {"qv", "qc", "qr", "qi", "qs", "qg", "Nn", "Nc", "Nr"} <= wdm6
    assert "Ni" not in wdm6


def test_mp_registry_names_match_expected_wrfout_variables() -> None:
    assert state_leaves_for_mp(1) == ("qv", "qc", "qr")
    assert state_leaves_for_mp(10) == ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Ns", "Nr", "Ng")
    assert state_leaves_for_mp(16) == ("qv", "qc", "qr", "qi", "qs", "qg", "Nn", "Nc", "Nr")
    assert NUMBER_WRFOUT_NAME["Nn"] == "QNCCN"
    assert "QNCLOUD" in wrfout_names_for_mp(16)
    assert "QNCCN" in wrfout_names_for_mp(16)


def test_interfaces_self_check_and_scheme_specs_cover_v060_options() -> None:
    assert_interfaces_consistent()
    assert len(SCHEME_STEP_SPECS) == 17
    assert scheme_step_spec("microphysics", 16).writes_state[-3:] == ("Nn", "Nc", "Nr")
    assert scheme_step_spec("cumulus", 1).returns_accumulators == ("rainc_acc",)
    assert scheme_step_spec("land_surface", 2).writes_carry == ("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn")


def test_physics_tendency_validates_unknown_keys() -> None:
    assert "rainc_acc" in ACCUMULATOR_UPDATE_KEYS
    PhysicsTendency(state_tendencies={"theta": object()}, accumulator_increments={"rainc_acc": object()}).validate_keys()
    with pytest.raises(ValueError, match="unknown state_tendency"):
        PhysicsTendency(state_tendencies={"bad_leaf": object()}).validate_keys()


def test_v060_namelist_accept_matrix_and_wrfout_forward_names() -> None:
    validate_supported_namelist(
        {
            "physics": {
                "mp_physics": [1, 6, 8, 10, 16],
                "cu_physics": [0, 1, 3, 6, 16],
                "bl_pbl_physics": [0, 1, 5, 7],
                "sf_sfclay_physics": [0, 1, 5, 7],
                "sf_surface_physics": [0, 2, 4],
                "ra_sw_physics": [0, 4],
                "ra_lw_physics": [0, 4],
            }
        }
    )
    for name in ("QNSNOW", "QNGRAUPEL", "QNCLOUD", "QNCCN"):
        assert name in MICROPHYSICS_EXTRA_VARIABLES
        assert name in WRFOUT_VARIABLE_SPECS
