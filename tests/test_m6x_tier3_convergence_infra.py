from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

from gpuwrf.validation import tier3_envelope


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "m6_tier3_convergence_runner.py"
CASE_PATH = ROOT / "data" / "fixtures" / "tier3_idealized" / "case_definition.json"
ALLOWED_VERDICTS = {"PASS_TIER3", "FAIL_DRIFT", "FAIL_NONFINITE", "FAIL_INSUFFICIENT_DT_PAIRS"}
REQUIRED_TOP_LEVEL = {
    "artifact_type",
    "case",
    "config",
    "dt_pairs",
    "checkpoints_s",
    "per_dt_run_metadata",
    "norms",
    "convergence_verdict",
    "rationale",
}


def _runner_module():
    spec = importlib.util.spec_from_file_location("m6_tier3_convergence_runner", RUNNER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _numeric_leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numeric_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numeric_leaves(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def test_case_definition_documents_idealized_tier3_case() -> None:
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))

    assert case["case_name"] == "flat_warm_bubble_tier3"
    assert case["boundary_conditions"]["mode"] in {"periodic", "open", "damped"}
    assert case["physics_toggles"]["scope"] in {"dycore_only", "dycore_plus_micro"}
    assert case["dt_refinement"]["refinement_factors"] == [1.0, 0.5, 0.25]
    assert case["variables_to_track"] == ["U", "V", "W", "theta", "p_perturbation", "mu_perturbation"]
    assert "d02" in case["case_choice_rationale"]["why_not_d02"]


def test_tier3_helper_classifies_missing_second_pair() -> None:
    verdict, rationale = tier3_envelope.classify_convergence(
        {
            "theta": {
                "pair_0": {
                    "1s": {"l2": 0.0, "linf": 0.0, "rmse": 0.0, "dt_coarse": 1.0, "dt_fine": 0.5}
                }
            }
        },
        dt_pairs=[{"dt_coarse": 1.0, "dt_fine": 0.5, "pair_index": 0}],
        per_dt_run_metadata=[{"dt_s": 1.0, "first_nonfinite_step": None}],
    )

    assert verdict == "FAIL_INSUFFICIENT_DT_PAIRS"
    assert "dt, dt/2, and dt/4" in rationale


def test_runner_smoke_produces_contract_schema(tmp_path: Path) -> None:
    runner = _runner_module()
    output = tmp_path / "tsc_envelope.json"
    start = time.perf_counter()

    assert runner.main(["--case", "flat_warm_bubble_tier3", "--dt", "1.0", "--output", str(output)]) == 0

    elapsed_s = time.perf_counter() - start
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert elapsed_s < 60.0
    assert REQUIRED_TOP_LEVEL <= set(payload)
    assert payload["artifact_type"] == "m6_tier3_tsc_envelope"
    assert payload["case"] == "flat_warm_bubble_tier3"
    assert payload["config"]["boundary_mode"] == "open"
    assert payload["config"]["physics"] == "dycore_only"
    assert payload["config"]["variables"] == ["U", "V", "W", "theta", "p_perturbation", "mu_perturbation"]
    assert len(payload["dt_pairs"]) == 2
    assert payload["dt_pairs"][0] == {"dt_coarse": 1.0, "dt_fine": 0.5, "pair_index": 0}
    assert payload["dt_pairs"][1] == {"dt_coarse": 0.5, "dt_fine": 0.25, "pair_index": 1}
    assert payload["convergence_verdict"] in ALLOWED_VERDICTS
    assert len(payload["per_dt_run_metadata"]) == 3

    norm_numbers = list(_numeric_leaves(payload["norms"]))
    assert norm_numbers
    assert all(math.isfinite(value) for value in norm_numbers)
    assert set(payload["norms"]) == set(payload["config"]["variables"])
    tier3_envelope.validate_tsc_payload(payload)
