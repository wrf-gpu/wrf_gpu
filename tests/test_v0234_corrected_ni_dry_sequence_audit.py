from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from scripts import v0234_corrected_ni_dry_sequence_audit as audit


PROOF_PATH = Path(
    ".agent/sprints/2026-07-13-v0234-corrected-ni-dry-sequence-audit/"
    "offline-dry-sequence-proof.json"
)


def test_audit_imports_no_jax_or_gpuwrf() -> None:
    module = ast.parse(Path(audit.__file__).read_text())
    imported = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "jax" not in imported
    assert "gpuwrf" not in imported


def test_global_clock_saturates_while_package_clock_cycles() -> None:
    expected = (1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    for step, phase in zip(range(9313, 9319), expected):
        assert audit.live_boundary_alpha(step, 6.0, 18.0) == 1.0
        assert audit.wrf_child_phase(step) == phase
        assert audit.local_boundary_lead_seconds(step) / 18.0 == phase
    assert np.isclose(
        audit.live_boundary_alpha(9314, 6.0, 18.0) - audit.wrf_child_phase(9314),
        1.0 / 3.0,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert audit.wrf_rk_spec_endpoint_phases(9314) == (4.0 / 9.0, 0.5, 2.0 / 3.0)
    assert audit.naive_localized_live_stage_phases(9314) == (7.0 / 9.0, 5.0 / 6.0, 1.0)


def test_wrf_corner_ownership_is_single_and_live_normal_path_has_apex_holes() -> None:
    wrf_u = audit.wrf_relax_coverage(93, 112)
    wrf_v = audit.wrf_relax_coverage(94, 111)
    live_u = audit.live_normal_relax_coverage(93, 112, "u")
    live_v = audit.live_normal_relax_coverage(94, 111, "v")
    assert np.max(wrf_u) == np.max(wrf_v) == 1
    assert np.count_nonzero(wrf_u) == np.count_nonzero(wrf_v) == 1170
    assert np.count_nonzero((wrf_u > 0) & (live_u == 0)) == 648
    assert np.count_nonzero((wrf_v > 0) & (live_v == 0)) == 528

    for y, x in audit.FINAL_APICES:
        faces = (
            (wrf_u[y, x], live_u[y, x]),
            (wrf_u[y, x + 1], live_u[y, x + 1]),
            (wrf_v[y, x], live_v[y, x]),
            (wrf_v[y + 1, x], live_v[y + 1, x]),
        )
        assert sum(bool(wrf) and not bool(live) for wrf, live in faces) == 3


def test_numpy_relax_null_space_response_and_continuity_conservation() -> None:
    proof = audit.algebraic_oracles()
    assert proof["zero_residual_null_space_exact"] is True
    assert proof["corner_ownership"] == {
        "max_owner_count": 1,
        "min_active_owner_count": 1,
        "duplicate_cells": 0,
    }
    response = proof["uniform_residual_integrated_response_by_b_dist"]
    assert response["1"]["actual"] == 0.1
    assert np.isclose(response["2"]["actual"], 0.1 * 2.0 / 3.0, rtol=0.0, atol=1.0e-15)
    assert np.isclose(response["3"]["actual"], 0.1 / 3.0, rtol=0.0, atol=1.0e-15)
    assert proof["parent_interval_endpoint_conservation"]["final_exact"] is True
    assert proof["continuity_flux_telescope"]["absolute_closure"] <= 2.0e-14


def test_offline_proof_binds_retained_package_and_gain_falsifier() -> None:
    payload = json.loads(PROOF_PATH.read_text())
    assert audit.verify_embedded_digest(payload)
    assert payload["status"] == "OFFLINE_COMPLETE_GPU_HELD"
    assert payload["verdict"] == "COMPLETE_NESTED_FROZEN_BUNDLE_CANDIDATE_PENDING_FABLE"
    step_9314 = payload["clock_discrepancy"]["step_9314"]
    assert step_9314["production_alpha"] == 1.0
    assert step_9314["wrf_alpha"] == 2.0 / 3.0
    assert np.isclose(
        step_9314["production_minus_wrf_alpha"], 1.0 / 3.0, rtol=0.0, atol=1.0e-15
    )
    assert step_9314["wrf_rk_spec_endpoint_phases"] == [4.0 / 9.0, 0.5, 2.0 / 3.0]
    assert step_9314["naive_modulo_only_rk_spec_endpoint_phases"] == [
        7.0 / 9.0,
        5.0 / 6.0,
        1.0,
    ]
    expected_errors = {
        "u": 17.21556824077861,
        "v": 11.06330527647538,
        "w": 14.476338343609381,
        "theta": 29.98006820766513,
        "ph_perturbation": 1344.650983117705,
        "mu_perturbation": 4456.075791772779,
    }
    for name, expected in expected_errors.items():
        row = payload["retained_step9313_package"][name]
        assert row["all_completed_step_9313_spec_sides_equal_record_1_exact"] is True
        assert row["max_abs_live_minus_wrf_target_step_9314"] == expected

    coupled = payload["independent_numpy_oracles"]["retained_mass_coupled_apex_response"]
    missing_responses = []
    for apex in coupled["apices"].values():
        for face in apex["faces"].values():
            if face["wrf_relaxed"] and not face["released_live_normal_relaxed"]:
                missing_responses.append(face["max_abs_six_second_velocity_equivalent_m_s"])
    assert len(missing_responses) == 6
    assert min(missing_responses) > 1.0
    assert max(missing_responses) > 7.0

    gain = payload["retained_dynamic_compatibility"]["gain_discriminator"]
    assert gain["gain20_A"]["projected_cells"] == 110
    assert gain["gain1_B"]["projected_cells"] == 29
    assert gain["gain20_A"]["contains_both_final_apices"] is True
    assert gain["gain1_B"]["contains_both_final_apices"] is True


def test_candidate_is_fail_closed_pending_fable_and_keeps_v10_separate() -> None:
    payload = json.loads(PROOF_PATH.read_text())
    assert payload["gpu_gate"]["gpu_eligible"] is False
    assert payload["gpu_gate"]["hard_hold"] is True
    assert payload["gpu_gate"]["gpu_commands_run"] == 0
    assert payload["scope"]["model_or_numerical_edits"] == 0
    assert payload["scope"]["agents_or_models_launched"] == 0
    assert payload["mechanism_separation"]["verdict"] == (
        "V10_REMAINS_SEPARATE; NO_TEMPORAL_CAUSAL_LINK_ADDED"
    )
    ranking = payload["candidate_ranking"]
    assert ranking[0]["candidate"] == (
        "complete_nested_frozen_RK1_boundary_bundle_with_package_local_phase"
    )
    assert ranking[1]["candidate"] == "package_local_phase_only"
    assert "INSUFFICIENT_STANDALONE" in ranking[1]["classification"]
