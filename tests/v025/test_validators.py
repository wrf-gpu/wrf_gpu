"""The validators must fail closed, not degrade to warnings.

Contract §12: "Missing evidence is never PASS." These tests pin that a missing
object, an absent required field, and a hash that no longer matches the file all
produce a non-PASS verdict and a non-zero exit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

V025 = Path(__file__).resolve().parents[2] / "scripts" / "v025"

VALIDATORS = [
    ("validate_m0_manifest.py", []),
    ("validate_kernel_census.py", ["--min-attribution", "0.95"]),
    ("validate_cancellation_map.py", ["--require-complete"]),
    ("validate_pallas_viability.py", []),
    ("recompute_efficiency.py", ["--check"]),
    ("validate_fast_pair.py", ["--require-fresh-cpu"]),
]


def run(script: str, path: Path, extra: list[str]) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(V025 / script), str(path), *extra],
        capture_output=True,
        text=True,
    )
    try:
        return proc.returncode, json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.returncode, {"stdout": proc.stdout, "stderr": proc.stderr}


@pytest.mark.parametrize("script,extra", VALIDATORS)
def test_missing_object_is_missing_not_pass(script, extra, tmp_path):
    code, report = run(script, tmp_path / "absent.json", extra)
    assert code != 0
    assert report["status"] == "MISSING"


@pytest.mark.parametrize("script,extra", VALIDATORS)
def test_empty_object_never_passes(script, extra, tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    code, report = run(script, path, extra)
    assert code != 0
    assert report["status"] in {"MISSING", "FAIL"}


def test_efficiency_recompute_catches_wrong_arithmetic(tmp_path):
    """The critic must not have to trust the stored T_ceiling / E_baseline."""
    path = tmp_path / "efficiency_model.json"
    # F/R = 2e10/1e13 s = 2 ms; stored value is deliberately wrong.
    path.write_text(
        json.dumps(
            {
                "T_control_ms_per_step": 100.0,
                "F_measured_flop_per_step": 2e10,
                "R_achievable_flop_per_second": 1e13,
                "T_ceiling_ms_per_step": 5.0,
                "E_baseline": 0.05,
                "R_achievable_source": "measured_calibration_kernel",
                "calibration_repeats": 5,
                "step_classes": {"ordinary": {}, "event": {}},
                "uncertainty": {"bootstrap": True},
            }
        )
    )
    code, report = run("recompute_efficiency.py", path, ["--check"])
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["recompute:T_ceiling"] == "FAIL"
    assert code != 0


def test_efficiency_recompute_accepts_consistent_arithmetic(tmp_path):
    path = tmp_path / "efficiency_model.json"
    path.write_text(
        json.dumps(
            {
                "T_control_ms_per_step": 100.0,
                "F_measured_flop_per_step": 2e10,
                "R_achievable_flop_per_second": 1e13,
                "T_ceiling_ms_per_step": 2.0,
                "E_baseline": 0.02,
                "R_achievable_source": "measured_calibration_kernel",
                "calibration_repeats": 5,
                "step_classes": {"ordinary": {}, "event": {}},
                "uncertainty": {"bootstrap": True},
            }
        )
    )
    code, report = run("recompute_efficiency.py", path, ["--check"])
    assert report["status"] == "PASS", report
    assert code == 0


def test_efficiency_rejects_vendor_peak_fraction(tmp_path):
    """§8: R_achievable is measured on this card, not a % of vendor peak."""
    path = tmp_path / "efficiency_model.json"
    path.write_text(
        json.dumps(
            {
                "T_control_ms_per_step": 100.0,
                "F_measured_flop_per_step": 2e10,
                "R_achievable_flop_per_second": 1e13,
                "T_ceiling_ms_per_step": 2.0,
                "E_baseline": 0.02,
                "R_achievable_source": "vendor_peak_fraction",
                "calibration_repeats": 5,
                "step_classes": {"ordinary": {}, "event": {}},
                "uncertainty": {"bootstrap": True},
            }
        )
    )
    code, report = run("recompute_efficiency.py", path, ["--check"])
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["R_achievable_is_measured"] == "FAIL"
    assert code != 0


def test_pallas_interpreter_result_is_not_viability_evidence(tmp_path):
    path = tmp_path / "pallas.json"
    path.write_text(
        json.dumps(
            {
                "verdict": "PALLAS_GREEN",
                "backend": "interpreter",
                "operator": "advance_w_thomas",
                "state_source_is_real": True,
                "correctness": {
                    "cpu_reference_pass": True,
                    "native_output_finite": True,
                    "within_wrf_fp32_vs_fp64_envelope": True,
                    "no_hidden_transfer_in_timed_loop": True,
                },
                "performance": {
                    "paired_warm_median_speedup": 2.0,
                    "bootstrap_lower_95": 1.5,
                    "warm_iterations_per_arm": 100,
                    "alternating_pairs": 5,
                    "raw_artifacts": ["x.nsys-rep"],
                },
            }
        )
    )
    code, report = run("validate_pallas_viability.py", path, [])
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["backend:native_mosaic_not_interpreter"] == "FAIL"
    assert code != 0


def test_cancellation_map_rejects_synthetic_state(tmp_path):
    path = tmp_path / "cancellation_map.json"
    path.write_text(
        json.dumps(
            {
                "state_source": "synthetic_random_arrays",
                "coverage_fraction": 1.0,
                "operators": [{"operator": "x", "active": True, "classification": "SAFE_FP32"}],
                "proposed_fp64_islands": [],
            }
        )
    )
    code, report = run("validate_cancellation_map.py", path, ["--require-complete"])
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["state_source_is_real"] == "FAIL"
    assert code != 0


def test_cancellation_map_rejects_keep_all_islands_rationale(tmp_path):
    path = tmp_path / "cancellation_map.json"
    path.write_text(
        json.dumps(
            {
                "state_source": "20260725_18z_production_snapshot",
                "coverage_fraction": 1.0,
                "operators": [
                    {"operator": "x", "active": True, "classification": "SAFE_FP32"}
                ],
                "proposed_fp64_islands": [{"site": "a", "evidence": "e"}],
                "rationale": "keep all current islands for safety",
            }
        )
    )
    code, report = run("validate_cancellation_map.py", path, ["--require-complete"])
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["islands:not_keep_all_current"] == "FAIL"


def test_manifest_detects_a_stale_hash(tmp_path):
    """A proof object that drifted from its recorded hash must not validate."""
    artifact = tmp_path / "thing.json"
    artifact.write_text('{"v": 1}')
    manifest = tmp_path / "PROOF_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": str(artifact),
                        "sha256": "0" * 64,
                        "producing_command": "echo",
                        "returncode": 0,
                    }
                ],
                "gates": {"g": {"status": "PASS", "evidence": "thing.json"}},
            }
        )
    )
    code, report = run("validate_m0_manifest.py", manifest, [])
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks[f"artifact:{artifact}:hash_matches"] == "FAIL"
    assert code != 0


def test_manifest_rejects_pass_without_evidence(tmp_path):
    manifest = tmp_path / "PROOF_MANIFEST.json"
    manifest.write_text(
        json.dumps({"artifacts": [], "gates": {"g": {"status": "PASS"}}})
    )
    code, report = run("validate_m0_manifest.py", manifest, [])
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["gate:g:pass_has_evidence"] == "FAIL"
