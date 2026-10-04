"""Integrity tests for the GPT M1 Round-3 terminal critic evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PROOF = REPO / "proofs/v025/m1/m1_round3_gpt_final_critic_cpu_evidence.json"
SIDECAR = PROOF.with_suffix(".sha256")
SCRIPT = REPO / "scripts/v025/m1_round3_gpt_final_critic.py"
VERDICT = "REJECT_ROUND3_BLOCKED_PROOF_UNSOUND"


def _proof():
    return json.loads(PROOF.read_text(encoding="utf-8"))


def test_sidecar_binds_exact_critic_proof_bytes():
    digest = hashlib.sha256(PROOF.read_bytes()).hexdigest()
    assert SIDECAR.read_text(encoding="utf-8").split()[0] == digest


def test_terminal_verdict_separates_all_four_decision_axes():
    proof = _proof()
    assert proof["terminal_verdict"] == VERDICT
    assert set(proof["axes"]) == {
        "mechanism_reality",
        "candidate_acceptance",
        "proof_integrity",
        "architecture_testability",
    }
    assert proof["axes"]["mechanism_reality"].startswith("SUPPORTED_CPU_STATIC")
    assert proof["axes"]["candidate_acceptance"].startswith("REJECTED")
    assert proof["axes"]["proof_integrity"].startswith("REJECTED")
    assert proof["axes"]["architecture_testability"].startswith(
        "PLAUSIBLE_NOT_PROVEN"
    )


def test_all_recorded_proof_binding_mutations_false_pass_author_gate():
    attack = _proof()["proof_binding_mutations"]
    assert attack["gate_is_sound"] is False
    assert attack["attack_count"] >= 8
    assert attack["false_pass_count"] == attack["attack_count"]
    assert all(attack["mutations_that_false_pass"].values())


def test_author_binding_is_stale_even_though_custom_source_digest_corresponds():
    binding = _proof()["binding_audit"]
    assert binding["author_proof"]["sidecar_matches"] is True
    assert binding["all_raw_hashes_match"] is True
    assert binding["recorded_head_is_exact_candidate"] is False
    assert binding["recorded_git_tree_is_exact_candidate"] is False
    assert binding["recorded_git_tree_resolves_to"]["matches_f95daee0"] is True
    assert (
        binding["custom_python_tree_digest"]["current_matches_recorded"] is True
    )
    assert binding["production_diff_after_exact_candidate"] == []


def test_real_wrfbdy_input_is_used_but_not_hash_bound():
    inputs = _proof()["input_binding"]
    assert inputs["wrfbdy_is_used"] is True
    assert inputs["wrfbdy_is_hash_bound_in_author_raw"] is False
    assert inputs["complete_input_binding"] is False
    assert inputs["assets"]["wrfbdy"]["actual_sha256"] == (
        "296cef912e17dc0dfff4b78103dedfb1329cd184de5bc2cd0d44cc6edf823599"
    )


def test_independent_complete_step_reproduction_fails_both_arena_gates():
    step = _proof()["independent_reproduction"]["three_arm_complete_step"]
    assert step["pre_candidate"]["temporary_bytes"] == 1_318_841_144
    assert step["candidate"]["temporary_bytes"] == 1_256_262_880
    assert step["fp64_control"]["temporary_bytes"] == 1_379_251_432
    assert step["incremental_10_percent_gate_passes"] is False
    assert step["fp64_control_20_percent_gate_passes"] is False
    assert step["launch_delta"] == 11
    assert step["launch_resolution_band"] == 18
    assert step["launch_status"] == (
        "STATIC_PROXY_REGRESSION_REQUIRES_DEVICE_FALSIFICATION"
    )


def test_conversion_count_reproduces_but_source_frame_is_not_dataflow_proof():
    reproduced = _proof()["independent_reproduction"]
    loop = reproduced["optimized_loop_conversion"]
    assert loop["in_loop"] == 65
    assert loop["in_loop_converted_elements_per_substep"] == 494_216
    attribution = reproduced["source_jaxpr_conversion_attribution"]
    assert attribution["elements_from_lead_interpolated_boundary_data"] == 0
    assert "does not trace dataflow" in reproduced["attribution_limit"]


def test_fresh_fp64_identity_matches_immutable_pre_for_core13_and_full_step():
    identity = _proof()["independent_reproduction"]["default_identity"]
    assert identity["core13_count"] == 13
    assert identity["core13_all_stablehlo_equal"] is True
    assert identity["core13_all_outputs_equal"] is True
    assert all(identity["full_step_comparison_to_immutable_pre"].values())


def test_reported_arena_ratio_is_arithmetic_not_a_valid_scope_ceiling():
    ceiling = _proof()["ceiling_attack"]
    isolated = ceiling["independent_isolated_reproduction"]
    assert isolated["same_instrument_reproduces_author_values"] is True
    assert isolated["pre_candidate"]["temporary_bytes"] == 181_395_480
    assert isolated["candidate"]["temporary_bytes"] == 90_744_064
    assert ceiling["arithmetic_recomputed"]["matches_reported_share"] is True
    assert (
        ceiling["arithmetic_recomputed"]["matches_reported_measured_ceiling"]
        is True
    )
    assert ceiling["denominator_compatibility"]["compatible"] is False
    assert ceiling["bounds_adr037_scope"] is False
    assert ceiling["proofs_fp32_scope_insufficient"] is False


def test_author_floor_differs_from_round_after_real_preparation_for_every_family():
    attack = _proof()["typed_work_floor_attack"]
    assert attack["source"]["placement_matches_contract"] is False
    assert attack["source"]["raw_gate_false_pass_count"] == 5
    assert all(
        attack["source"]["raw_gate_mutations_that_false_pass"].values()
    )
    numeric = attack["numeric_boundary_probe"]
    assert numeric["contracted_floor_reproduced_by_author"] is False
    assert numeric["environment"]["jax_platforms"] == ["cpu"]
    assert set(numeric["fields"]) == {
        "u_work",
        "v_work",
        "theta_work",
        "w_work",
        "ph_work",
        "mu_work",
        "pressure_work",
    }
    assert all(
        entry["author_equals_round_after_prep"] is False
        and entry["unequal_elements"] > 0
        and entry["rms_difference"] > 0.0
        for entry in numeric["fields"].values()
    )


def test_all_ten_mandatory_attacks_have_terminal_outcomes():
    attacks = _proof()["mandatory_attack_outcomes"]
    assert len(attacks) == 10
    assert {int(name.split("_", 1)[0]) for name in attacks} == set(range(1, 11))
    assert all(entry["outcome"] for entry in attacks.values())


def test_critic_entry_point_pins_cpu_before_any_local_jax_import():
    source = SCRIPT.read_text(encoding="utf-8")
    real_state = source.index(
        "from scripts.v025.real_state import assert_cpu_only"
    )
    jax = source.index("    import jax")
    assert real_state < jax
    assert "nvidia-smi" not in source
    assert 'os.environ["JAX_PLATFORMS"]' not in source
