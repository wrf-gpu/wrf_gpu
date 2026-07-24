from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from scripts import v0234_corrected_ni_gain1_ab as gain1


SPRINT_DIR = Path(".agent/sprints/2026-07-13-v0234-corrected-ni-gain1-ab")


def test_preimport_environment_has_exactly_one_numerical_difference() -> None:
    environment = dict(gain1.REQUIRED_ORDINARY_ENV)
    environment.update({
        gain1.GAIN_ENV_NAME: gain1.GAIN_ENV_VALUE,
        "GPUWRF_JAX_CACHE_DIR": "/cache",
        "JAX_COMPILATION_CACHE_DIR": "/cache",
        "GPUWRF_WRF_ROOT": "/authority",
    })
    row = gain1.validate_preimport_environment(environment)
    assert row["validated_before_jax_or_gpuwrf_import"] is True
    assert row["ordinary_environment"] == gain1.REQUIRED_ORDINARY_ENV
    assert row["only_numerical_environment_difference"] == {
        "GPUWRF_NORMAL_BDY_RELAX_STRENGTH": "1.0"
    }

    environment["GPUWRF_NORMAL_BDY_RELAX_STRENGTH"] = "1"
    try:
        gain1.validate_preimport_environment(environment)
    except RuntimeError as error:
        assert "exact string" in str(error)
    else:
        raise AssertionError("non-exact gain must fail closed")

    environment["GPUWRF_NORMAL_BDY_RELAX_STRENGTH"] = "1.0"
    environment["GPUWRF_UNAUTHORIZED_NUMERICAL_ARM"] = "1"
    try:
        gain1.validate_preimport_environment(environment)
    except RuntimeError as error:
        assert "unapproved GPUWRF" in str(error)
    else:
        raise AssertionError("an extra numerical arm must fail closed")


def test_runner_imports_only_ordinary_production_carry_callable() -> None:
    module = ast.parse(Path("scripts/v0234_corrected_ni_gain1_ab.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        and node.module == "gpuwrf.runtime.operational_mode"
        for alias in node.names
    }
    assert imported == {"_advance_chunk_fori", "build_clock_base"}
    source = inspect.getsource(gain1.main)
    dispatch9314 = source.index("b9314 = executable(")
    health9314 = source.index("materialize_completed_health(b9314, baseline)")
    conditional = source.index('if health9314["passed"]:')
    dispatch9315 = source.index("b9315 = executable(")
    health9315 = source.index("materialize_completed_health(b9315, baseline)")
    assert dispatch9314 < health9314 < conditional < dispatch9315 < health9315
    assert '"ordinary_A_model_calls": 0' in source
    assert "phase_tap" not in imported
    assert "advance_chunk_with_corrected_ni_rca" not in imported


def _healthy_arrays() -> dict[str, np.ndarray]:
    arrays = {
        name: np.asarray([10.0], dtype=np.float64)
        for name in gain1.ALL_SCALE_FIELDS
    }
    arrays["state.p_total"] = np.asarray([90_000.0], dtype=np.float64)
    arrays["state.mu_total"] = np.asarray([80_000.0], dtype=np.float64)
    return arrays


def test_physical_health_rejects_finite_guard_masked_scale() -> None:
    baseline_arrays = _healthy_arrays()
    baseline = gain1.baseline_health(baseline_arrays)
    passed = gain1.evaluate_physical_health(
        baseline_arrays, baseline, all_carry_nonfinite_count=0
    )
    assert passed["passed"] is True
    assert passed["violations"] == []

    unphysical = {name: value.copy() for name, value in baseline_arrays.items()}
    unphysical["ww"] = np.asarray([1.0e8], dtype=np.float64)
    failed = gain1.evaluate_physical_health(
        unphysical, baseline, all_carry_nonfinite_count=0
    )
    assert failed["passed"] is False
    assert {row["gate"] for row in failed["violations"]} == {
        "absolute_scale_le_1e6",
        "amplification_vs_step9313_le_1000",
    }

    nonfinite = gain1.evaluate_physical_health(
        baseline_arrays, baseline, all_carry_nonfinite_count=1
    )
    assert nonfinite["passed"] is False
    assert nonfinite["violations"][0]["field"] == "complete_106_leaf_carry"


def test_boundary_discriminator_oracle_is_independent_and_exact() -> None:
    oracle = gain1.boundary_response_oracle()
    assert oracle["pristine_wrf_frozen_tendency_response"] == 0.1
    assert oracle["released_moving_residual_strength20_response"] == 0.8926258175999999
    assert oracle["candidate_moving_residual_strength1_response"] == 0.09561792499119559
    assert oracle["candidate_to_wrf_ratio"] == 0.9561792499119559


def test_contract_freezes_a_and_b_scope_and_falsification_endpoint() -> None:
    contract = (SPRINT_DIR / "sprint-contract.md").read_text()
    assert gain1.REPORT_COMMIT in contract
    assert gain1.INPUT_FILE_SHA256 in contract
    assert gain1.REFERENCE_MANIFEST_SHA256 in contract
    assert "never rerun A" in contract
    assert "GPUWRF_NORMAL_BDY_RELAX_STRENGTH=1.0" in contract
    assert "GAIN1_CANDIDATE_FALSIFIED" in contract
    assert "broad V10 drift remain separate" in contract


def test_gpu_proof_falsifies_gain1_and_stops_before_ni_step() -> None:
    proof_path = SPRINT_DIR / "gain1-ab-proof.json"
    raw = proof_path.read_bytes()
    proof = json.loads(raw)
    embedded = proof.pop("proof_sha256")
    canonical = json.dumps(
        proof, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()

    assert hashlib.sha256(raw).hexdigest() == (
        "cb8643d7e24b139331a3fad780ccb35564809cbfc1ffb44c37fd981372486ebb"
    )
    assert embedded == hashlib.sha256(canonical).hexdigest() == (
        "4f21e31400688592e379c2fda8d31302afdd18f4c3585bf99f3d5263761f326a"
    )
    assert proof["verdict"] == "GAIN1_CANDIDATE_FALSIFIED"
    execution = proof["execution"]
    assert execution["ordinary_A_model_calls"] == 0
    assert execution["gain1_B_model_calls"] == 1
    assert execution["lower_calls"] == execution["compile_calls"] == 1
    assert [row["native_step"] for row in execution["steps"]] == [9314]
    step = execution["steps"][0]
    assert step["manifest"]["manifest_sha256"] == (
        "b960ca07c36652ebff4934610252f4672102c7233ed3b0148627498604398b81"
    )
    assert step["manifest"]["floating_nonfinite_count"] == 2610
    assert step["physical_health"]["passed"] is False
    named = step["physical_health"]["named_fields"]
    assert {name: named[name]["nonfinite_count"] for name in gain1.SCRATCH_HEALTH_FIELDS} == {
        "t_2ave": 1276,
        "ww": 1247,
        "mudf": 29,
        "muave": 29,
        "muts": 29,
    }
    assert len(step["physical_health"]["violations"]) == 49
    assert execution["comparisons"][
        "retained_strength20_A_vs_gain1_B_step9314"
    ]["different_leaf_count"] == 22
    assert "step_9315" not in execution["retained_outputs"]
    assert proof["causal_decision"] == {
        "both_steps_healthy": False,
        "full_pristine_wrf_frozen_rk1_bundle_next": False,
        "gain1_candidate_falsified": True,
        "gain20_incident_trigger_localized": False,
        "strength_only_fix_authorized": False,
    }
    program = proof["program_audit"]
    assert program["input_leaf_count"] == program["output_leaf_count"] == 106
    assert program["input_output_structure_and_avals_identical"] is True
    assert program["forbidden_tokens_found"] == []
    assert proof["mechanism_separation"]["common_root_proved"] is False
