"""H2/H3 must answer the question they were pre-registered to answer.

Manager review on main `6a497c0e`: XLA phase-family attribution (codegen,
autotune, hlo_pass) does not answer radiation-vs-dycore. These tests hold the two
apart, pin the pre-registered bars, and -- most importantly -- keep the H3
structural proxy from ever being presented as measured per-family seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import application_attribution as aa  # noqa: E402

_A6 = REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json"


@pytest.fixture
def family_map():
    if not _A6.exists():
        pytest.skip("A6 census not on this machine")
    return aa.load_family_map()


# --------------------------------------------------------------------------- #
# the pre-registered bars                                                      #
# --------------------------------------------------------------------------- #
def test_the_bars_are_the_pre_registered_numbers():
    assert aa.H2_MAX_PASS_SHARE == 0.15
    assert aa.H3_MIN_RADIATION_SHARE == 0.25


def test_the_family_map_covers_the_frozen_inventory(family_map):
    assert len(family_map) == 49
    assert family_map["rrtmg_lw"] == "physics.radiation"
    assert family_map["rrtmg_sw"] == "physics.radiation"
    assert family_map["advect_u_flux"] == "dycore.advection"


# --------------------------------------------------------------------------- #
# operator -> family matching                                                  #
# --------------------------------------------------------------------------- #
def test_an_hlo_op_name_maps_to_its_application_family(family_map):
    name = "jit(_run_forecast)/jit(main)/rrtmg_lw/broadcast_in_dim"
    assert aa.family_of(name, family_map) == "physics.radiation"


def test_longest_match_wins_so_prefixes_do_not_swallow_families(family_map):
    """`calc_p_rho` is a prefix of `calc_p_rho_step`; first-match would merge them."""
    assert "calc_p_rho" in family_map and "calc_p_rho_step" in family_map
    name = "jit(main)/calc_p_rho_step/add"
    matched = max((op for op in family_map if op in name), key=len)
    assert matched == "calc_p_rho_step"
    assert aa.family_of(name, family_map) == family_map["calc_p_rho_step"]


def test_an_unrecognised_op_name_is_not_forced_into_a_family(family_map):
    assert aa.family_of("jit(main)/some_helper/add", family_map) is None


# --------------------------------------------------------------------------- #
# HLO attribution                                                              #
# --------------------------------------------------------------------------- #
HLO = """
ENTRY %main {
  %a = f32[8]{0} add(), metadata={op_name="jit(main)/rrtmg_lw/add" source_line=1}
  %b = f32[8]{0} mul(), metadata={op_name="jit(main)/rrtmg_lw/mul" source_line=2}
  %c = f32[8]{0} sub(), metadata={op_name="jit(main)/rrtmg_sw/sub" source_line=3}
  %d = f32[8]{0} add(), metadata={op_name="jit(main)/advect_u_flux/add" source_line=4}
}
"""


def test_instructions_are_attributed_to_families(family_map):
    result = aa.attribute_hlo(HLO, family_map)
    assert result["instructions_with_metadata"] == 4
    assert result["unattributed"] == 0
    assert result["per_family_instructions"]["physics.radiation"] == 3
    assert result["per_family_share"]["physics.radiation"] == pytest.approx(0.75)


def test_unattributed_instructions_are_counted_not_dropped(family_map):
    hlo = HLO + '  %e = f32[8]{0} add(), metadata={op_name="jit(main)/mystery/add"}\n'
    result = aa.attribute_hlo(hlo, family_map)
    assert result["unattributed"] == 1
    assert result["unattributed_share"] == pytest.approx(0.2)


def test_the_attribution_declares_itself_a_proxy(family_map):
    result = aa.attribute_hlo(HLO, family_map)
    assert "IS_A_PROXY" in result
    assert "not a measurement of" in result["IS_A_PROXY"]
    assert result["weight"] == "instruction count"


# --------------------------------------------------------------------------- #
# H2                                                                           #
# --------------------------------------------------------------------------- #
def test_h2_is_supported_when_a_pass_exceeds_the_bar():
    result = aa.h2_verdict({"fusion": 30.0, "layout": 5.0}, 100.0)
    assert result["verdict"] == "SUPPORTED"
    assert result["largest_pass"]["share"] == pytest.approx(0.30)


def test_h2_is_falsified_when_cost_is_diffuse():
    passes = {f"pass{i}": 10.0 for i in range(10)}
    result = aa.h2_verdict(passes, 100.0)
    assert result["verdict"] == "FALSIFIED"
    assert "diffuse" in result["reason"]


def test_h2_sits_exactly_on_its_bar():
    assert aa.h2_verdict({"p": 15.0}, 100.0)["verdict"] == "FALSIFIED"   # not >15%
    assert aa.h2_verdict({"p": 15.1}, 100.0)["verdict"] == "SUPPORTED"


def test_h2_blocks_without_timings():
    assert aa.h2_verdict({}, 100.0)["verdict"] == "BLOCKED"
    assert aa.h2_verdict({"p": 1.0}, 0.0)["verdict"] == "BLOCKED"


def test_h2_rests_on_measured_seconds_not_a_proxy():
    assert "measured" in aa.h2_verdict({"p": 30.0}, 100.0)["basis"]


# --------------------------------------------------------------------------- #
# H3                                                                           #
# --------------------------------------------------------------------------- #
def test_h3_is_supported_when_radiation_clears_the_bar():
    result = aa.h3_verdict({"physics.radiation": 0.40, "dycore.advection": 0.60},
                           unattributed_share=0.0)
    assert result["verdict"] == "SUPPORTED"
    assert result["radiation_share"] == pytest.approx(0.40)


def test_h3_is_falsified_below_the_bar():
    result = aa.h3_verdict({"physics.radiation": 0.10, "dycore.advection": 0.90},
                           unattributed_share=0.0)
    assert result["verdict"] == "FALSIFIED"


def test_h3_sits_exactly_on_its_bar():
    assert aa.h3_verdict({"physics.radiation": 0.25}, unattributed_share=0.0)["verdict"] \
        == "SUPPORTED"
    assert aa.h3_verdict({"physics.radiation": 0.249}, unattributed_share=0.0)["verdict"] \
        == "FALSIFIED"


def test_h3_blocks_when_too_much_hlo_is_unattributed():
    """A verdict over a poorly attributed module is not a verdict."""
    result = aa.h3_verdict({"physics.radiation": 0.9}, unattributed_share=0.30)
    assert result["verdict"] == "BLOCKED"
    assert "could not be attributed" in result["reason"]


def test_h3_never_claims_to_be_measured_seconds():
    result = aa.h3_verdict({"physics.radiation": 0.40}, unattributed_share=0.0)
    assert "PROXY" in result["basis"]
    assert "not measured per-family seconds" in result["basis"]


# --------------------------------------------------------------------------- #
# the CPU-side prior                                                           #
# --------------------------------------------------------------------------- #
def test_the_cpu_prior_is_computed_from_real_a6_timings(family_map):
    prior = aa.cpu_side_family_compile_seconds()
    assert prior["total_seconds"] > 0
    assert len(prior["per_family_seconds"]) == 18
    assert prior["radiation_share"] > 0


def test_the_cpu_prior_refuses_to_stand_in_for_the_fused_module(family_map):
    """It is a prior. Saying so is the difference between evidence and a claim."""
    scope = aa.cpu_side_family_compile_seconds()["SCOPE"]
    assert "NOT the fused GPU module" in scope
    assert "A prior, not an answer" in scope


def test_the_two_attributions_are_not_interchangeable():
    """The category error the manager corrected, pinned as a test.

    `nvtx_exclusive` families are compiler phases; `application_attribution`
    families are WRF components. No name may appear in both vocabularies.
    """
    import nvtx_exclusive as nx
    phase_families = {name for name, _ in nx.FAMILY_PATTERNS} | {"unknown"}
    application_families = set(aa.load_family_map().values()) if _A6.exists() else set()
    assert phase_families.isdisjoint(application_families)


def test_the_prior_names_which_timing_field_it_summed(family_map):
    """`seconds` is lower+compile; calling it 'compile' would overstate it."""
    prior = aa.cpu_side_family_compile_seconds()
    assert prior["timing_field_used"]
    if "seconds" in prior["timing_field_used"]:
        assert "lower+compile combined" in prior["timing_field_note"]
        assert "not" in prior["timing_field_note"].lower()
