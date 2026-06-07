"""Fail-closed contract tests for the three-way WRF scheme support catalog.

These assert the v0.12.0 "no silent gaps" honesty contract:

* an implemented WRF suite validates silently;
* the Smagorinsky-vs-constant-K dynamics path is classified honestly;
* every recognized-but-unimplemented WRF scheme fails closed by name;
* every out-of-scope feature (WRF-Chem, WRF-Fire, FDDA, stochastic physics,
  moving nests, urban BEP/BEM) fails closed as a named scope decision;
* the catalog's ``implemented`` set never over-claims a reference-only scheme
  and never drifts from the frozen accept-matrix.
"""

from __future__ import annotations

import pytest

from gpuwrf.contracts.physics_registry import ACCEPTED_NAMELIST_OPTIONS
from gpuwrf.io.namelist_check import (
    UnsupportedNamelistOption,
    UnsupportedSchemeError,
    validate_namelist,
)
from gpuwrf.io.scheme_catalog import (
    OUT_OF_SCOPE_FEATURES,
    SupportStatus,
    assert_catalog_consistent,
    classify_feature_switch,
    classify_scheme,
    iter_full_catalog,
    status_counts,
)


# --------------------------------------------------------------------------- #
# Catalog internal consistency (the honesty invariants).                      #
# --------------------------------------------------------------------------- #
def test_catalog_is_internally_consistent() -> None:
    assert_catalog_consistent()


def test_implemented_set_matches_frozen_accept_matrix() -> None:
    """``IMPLEMENTED ∪ REFERENCE_ONLY`` must equal the frozen accept-matrix.

    This is the anti-over-claim guard: the public catalog cannot mark something
    implemented that the port has not accepted, nor silently drop an accepted
    scheme.
    """

    for key, accepted in ACCEPTED_NAMELIST_OPTIONS.items():
        passing = {
            s.code
            for s in iter_full_catalog()
            if s.key == key and s.status.passes_namelist_check
        }
        assert passing == set(accepted), (
            f"{key}: catalog-passing {sorted(passing)} != accepted {sorted(accepted)}"
        )


def test_every_enumerated_code_classified_into_one_status() -> None:
    counts = status_counts()
    assert counts[SupportStatus.IMPLEMENTED] > 0
    assert counts[SupportStatus.RECOGNIZED_FAIL_CLOSED] > 0
    # Total must equal the sum of the full WRF v4 enumeration we classify.
    assert sum(counts.values()) == sum(1 for _ in iter_full_catalog())


# --------------------------------------------------------------------------- #
# Contract requirement (1): an implemented combo validates OK.                #
# --------------------------------------------------------------------------- #
def test_implemented_suite_validates_ok() -> None:
    validate_namelist(
        {
            "physics": {
                "mp_physics": [8],
                "cu_physics": [1],
                "bl_pbl_physics": [5],
                "sf_sfclay_physics": [5],
                "sf_surface_physics": [4],
                "ra_lw_physics": [4],
                "ra_sw_physics": [4],
                "chem_opt": [0],  # explicitly-off out-of-scope switch is fine
            },
            "dynamics": {"rk_order": 3, "diff_opt": 2, "km_opt": 1},
        }
    )


# --------------------------------------------------------------------------- #
# Contract requirement (2): Smagorinsky / constant-K dynamics honesty.        #
# --------------------------------------------------------------------------- #
def test_smagorinsky_and_constant_k_classification_is_honest() -> None:
    """diff_opt=1/km_opt=4 (2-D Smag) is an IMPLEMENTED operational path; the
    3-D LES closures (km_opt=2/3/5) fail closed and the constant-K transition
    recipe is surfaced in the message."""

    # 2-D Smagorinsky horizontal diffusion is the wired real-data default.
    assert classify_scheme("diff_opt", 1).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("km_opt", 4).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("km_opt", 1).status is SupportStatus.IMPLEMENTED
    validate_namelist({"dynamics": {"diff_opt": 1, "km_opt": 4}})
    validate_namelist({"dynamics": {"diff_opt": 2, "km_opt": 1}})

    # The 3-D TKE / 3-D Smagorinsky / SMS-3DTKE closures are NOT implemented and
    # fail closed, naming the constant-K / 2-D-Smagorinsky alternative.
    for unimplemented in (2, 3, 5):
        support = classify_scheme("km_opt", unimplemented)
        assert support.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
        assert "constant-K" in support.alternative or "Smagorinsky" in support.alternative

    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"dynamics": {"diff_opt": 1, "km_opt": 2}})
    message = str(excinfo.value)
    assert "km_opt" in message
    # The transition recipe to a supported eddy-coefficient is named.
    assert "diff_opt=1" in message or "constant-K" in message


# --------------------------------------------------------------------------- #
# Contract requirement (3 & 4): out-of-scope fails closed.                    #
# --------------------------------------------------------------------------- #
def test_wrf_chem_option_fails_closed_out_of_scope() -> None:
    support = classify_feature_switch("chem_opt", 401)
    assert support is not None
    assert support.status is SupportStatus.OUT_OF_SCOPE

    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"chem": {"chem_opt": 401}})
    message = str(excinfo.value)
    assert "WRF-Chem" in message
    assert "out-of-scope" in message
    assert "chem_opt=0" in message  # the named alternative


@pytest.mark.parametrize(
    "key, value, feature_substring",
    [
        ("ifire", 2, "WRF-Fire"),
        ("wrf_hydro", 1, "WRF-Hydro"),
        ("grid_fdda", 1, "FDDA"),
        ("sppt", 1, "SPPT"),
        ("skebs", 1, "SKEBS"),
        ("spp", 1, "SPP"),
        ("num_moves", 3, "Moving"),
        ("sf_ocean_physics", 1, "ocean"),
    ],
)
def test_out_of_scope_feature_switches_fail_closed(
    key: str, value: int, feature_substring: str
) -> None:
    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"physics": {key: value}})
    sel = [s for s in excinfo.value.selections if s.key == key]
    assert sel, f"expected {key} to be rejected"
    assert sel[0].outcome == "out_of_scope"
    assert feature_substring in str(excinfo.value)


def test_out_of_scope_switch_off_passes() -> None:
    """A meteorology-only namelist that leaves chem/fire/fdda off passes."""

    validate_namelist(
        {
            "physics": {"mp_physics": [8], "bl_pbl_physics": [5]},
            "chem": {"chem_opt": 0},
            "fdda": {"grid_fdda": 0, "obs_nudge_opt": 0},
            "stoch": {"sppt": 0, "skebs": 0, "spp": 0},
        }
    )


def test_urban_bep_bem_is_out_of_scope() -> None:
    """Multi-layer urban canopy (BEP=2 / BEM=3) is a documented scope decision."""

    assert classify_scheme("sf_urban_physics", 2).status is SupportStatus.OUT_OF_SCOPE
    assert classify_scheme("sf_urban_physics", 3).status is SupportStatus.OUT_OF_SCOPE
    assert classify_scheme("sf_urban_physics", 0).status is SupportStatus.IMPLEMENTED

    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"physics": {"sf_urban_physics": 2}})
    assert "BEP" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Recognized-but-unimplemented + reference-only behavior under validate_namelist.
# --------------------------------------------------------------------------- #
def test_recognized_unimplemented_scheme_fails_closed_by_name() -> None:
    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"physics": {"mp_physics": [10, 28]}})  # domain 2 = aero-Thompson
    message = str(excinfo.value)
    assert "Thompson" in message
    assert "NOT YET IMPLEMENTED" in message


def test_reference_only_scheme_passes_namelist_layer() -> None:
    """Reference-only schemes (MYJ/Janjic, classic RRTM, Dudhia, New-Tiedtke)
    are accepted at the namelist layer (the operational scan fail-closes them
    downstream with a named reason)."""

    assert classify_scheme("bl_pbl_physics", 2).status is SupportStatus.REFERENCE_ONLY
    assert classify_scheme("cu_physics", 16).status is SupportStatus.REFERENCE_ONLY
    assert classify_scheme("ra_lw_physics", 1).status is SupportStatus.REFERENCE_ONLY
    validate_namelist({"physics": {"cu_physics": [16]}})
    validate_namelist({"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [2]}})


def test_unsupported_namelist_option_is_an_unsupported_scheme_error() -> None:
    """The CLI catches UnsupportedNamelistOption; it must subclass the umbrella."""

    assert issubclass(UnsupportedNamelistOption, UnsupportedSchemeError)


def test_combined_scheme_and_out_of_scope_failures_are_reported_together() -> None:
    """A namelist with both an unimplemented scheme and an out-of-scope feature
    reports BOTH in one message (no fail-on-first-category)."""

    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist(
            {"physics": {"mp_physics": [9]}, "chem": {"chem_opt": 401}}
        )
    keys = {s.key for s in excinfo.value.selections}
    assert "mp_physics" in keys
    assert "chem_opt" in keys


def test_out_of_scope_features_have_alternatives() -> None:
    """Every out-of-scope feature must name a concrete disable/alternative."""

    for feature in OUT_OF_SCOPE_FEATURES:
        assert feature.alternative.strip(), f"{feature.key} missing alternative"
        assert feature.reason.strip(), f"{feature.key} missing reason"
