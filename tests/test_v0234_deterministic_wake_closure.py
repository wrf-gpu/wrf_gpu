from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset

from scripts import v0234_deterministic_wake_admission as admission
from scripts import v0234_deterministic_wake_rca as rca
from scripts import v0234_deterministic_wake_terminal_proof as terminal


def _frame_pair(run: str) -> dict:
    path = admission.LINEAGE_ROOT / run / "frame-pairs/d03-step-09000.json"
    return json.loads(path.read_text())


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(np=np, Dataset=Dataset)


def test_kimi_authority_and_external_frames_authenticate() -> None:
    authority, row = admission.authenticate_authority()
    assert authority["model_tree"] == "835dcc29bf316c0715b41a72e064985e9cf099df"
    assert authority["release_gate_green"] is False
    assert authority["isolation_only"] is True
    assert row["kimi_proof_file_sha256"] == admission.KIMI_PROOF_FILE_SHA256
    assert row["kimi_proof_canonical_sha256"] == admission.KIMI_PROOF_CANONICAL_SHA256
    assert row["authenticated_frame_sha256"] == {
        "retained_A": admission.RETAINED_A_SHA256,
        "retained_B": admission.RETAINED_B_SHA256,
        "cpu_wrf": admission.CPU_WRF_SHA256,
    }


def test_both_authenticated_realizations_match_metric_and_wake_signature() -> None:
    authority, _ = admission.authenticate_authority()
    cases = (
        (
            "nested_stage_omega_transport_470e6111_full18h_toolingrepair2",
            admission.RETAINED_A,
        ),
        (
            "nested_stage_omega_transport_470e6111_post_fable_corner_window_"
            "gpt56_resource_retry1",
            admission.RETAINED_B,
        ),
    )
    for run, candidate in cases:
        pair = _frame_pair(run)
        result = admission.classify_known_wake_observation(
            pair["decisive_1500"],
            candidate_sha256=pair["candidate"]["sha256"],
            authority=authority,
            candidate_path=candidate,
            cpu_path=admission.CPU_WRF,
            runtime=_runtime(),
        )
        assert result["passed"] is True
        assert result["classification"] == (
            "KNOWN_WAKE_RELEASE_BLOCKER_METRIC_AND_FINGERPRINT_MATCH"
        )
        assert result["metric_signature"]["passed"] is True
        assert result["spatial_fingerprint"]["policy"]["passed"] is True
        assert result["release_gate_green"] is False
        assert result["release_blocker_remains"] is True
        assert result["waiver_or_reclassification"] is False
        assert result["tolerance_changed"] is False


def test_metric_classifier_rejects_other_field_and_envelope_drift() -> None:
    pair = _frame_pair(
        "nested_stage_omega_transport_470e6111_post_fable_corner_window_"
        "gpt56_resource_retry1"
    )
    other_field = copy.deepcopy(pair["decisive_1500"])
    other_field["strict_fields"]["U"]["no_worse"] = False
    other_field["violations"].append({"field": "U", "gate": "synthetic-test"})
    result = admission.metric_signature(other_field)
    assert result["passed"] is False
    assert result["checks"]["only_v_v10_may_be_red"] is False

    v_drift = copy.deepcopy(pair["decisive_1500"])
    v_drift["strict_fields"]["V"]["candidate_rmse"] = admission.V_ENVELOPE[1] + 1e-12
    assert admission.metric_signature(v_drift)["passed"] is False

    v10_drift = copy.deepcopy(pair["decisive_1500"])
    v10_drift["strict_fields"]["V10"]["candidate_rmse"] = admission.V10_ENVELOPE[1] + 1e-12
    assert admission.metric_signature(v10_drift)["passed"] is False


def test_spatial_classifier_rejects_changed_mechanism() -> None:
    authority, _ = admission.authenticate_authority()
    pair = _frame_pair(
        "nested_stage_omega_transport_470e6111_full18h_toolingrepair2"
    )
    fingerprint = admission.spatial_fingerprint(
        _runtime(),
        candidate_path=admission.RETAINED_A,
        cpu_path=admission.CPU_WRF,
        candidate_sha256=pair["candidate"]["sha256"],
        authority=authority,
    )
    assert admission.evaluate_spatial_policy(fingerprint)["passed"] is True
    changed = copy.deepcopy(fingerprint)
    changed["v10_max_error_location"] = [0, 0]
    verdict = admission.evaluate_spatial_policy(changed)
    assert verdict["passed"] is False
    assert verdict["checks"]["v10_max_in_southwest_wake"] is False


def test_reference_profile_binds_dump_mode_and_fail_closed_launcher() -> None:
    code = """
import json
from scripts import v0234_nested_frozen_wrf_boundary_window as r
print(json.dumps({
  'namespace': r.FULL_REPLAY_NAMESPACE,
  'label': r.LOCK_LABEL,
  'known': r.REQUIRE_KNOWN_1500_V10_RECORD,
  'terminal': r.ALLOW_DIAGNOSTIC_TERMINAL_WAKE_RED,
  'allowed': sorted(r.DIAGNOSTIC_TERMINAL_ALLOWED_FIELDS),
  'audit': r.audit_exact_launch_command(r.LAUNCH_COMMAND),
}))
"""
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(admission.REPO_ROOT),
        "GPUWRF_DETERMINISTIC_WAKE_CLOSURE": "1",
        "GPUWRF_DETERMINISTIC_AUTOTUNE_MODE": "reference",
        "GPUWRF_NESTED_BUNDLE_APPROVED_SHA": (
            "470e6111d516479bed4bc0c3b2be1007bb082afd"
        ),
    })
    completed = subprocess.run(
        ["<USER_HOME>/miniconda3/bin/python", "-c", code],
        cwd=admission.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["namespace"].endswith("deterministic_wake_reference1")
    assert payload["label"] == "v0234-deterministic-fulltree-reference"
    assert payload["known"] is True
    assert payload["terminal"] is True
    assert payload["allowed"] == ["V", "V10"]
    assert payload["audit"]["passed"] is True
    assert payload["audit"]["deterministic_dump_then_pin_pair"] is True
    assert payload["audit"]["tooling_critic_accept_required"] is False


def test_launcher_has_one_canonical_lock_and_no_cache_or_gpu_query() -> None:
    launcher = admission.SPRINT / "deterministic-fulltree-exact-launch-command.sh"
    text = launcher.read_text()
    assert text.count("/scripts/with_gpu_lock.sh") == 1
    assert "--intent production-preemptible" in text
    assert "--xla_gpu_dump_autotune_results_to=$DUMP_PIN" in text
    assert "--xla_gpu_load_autotune_results_from=$REFERENCE_PIN" in text
    assert text.count('mv -- "$DUMP_PIN" "$PIN"') == 1
    assert "JAX_ENABLE_COMPILATION_CACHE=false" in text
    assert "nvidia-smi" not in text
    assert "rocm-smi" not in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def _wind_fields(*, low_u: float, low_v: float, ratio: float) -> dict:
    return {
        "U": np.full((1, 2, 3), low_u, dtype=np.float64),
        "V": np.full((1, 3, 2), low_v, dtype=np.float64),
        "U10": np.full((2, 2), low_u * ratio, dtype=np.float64),
        "V10": np.full((2, 2), low_v * ratio, dtype=np.float64),
    }


def test_rca_ratio_decomposition_separates_diagnostic_from_wind() -> None:
    cpu = _wind_fields(low_u=2.0, low_v=4.0, ratio=0.5)
    ratio_only = _wind_fields(low_u=2.0, low_v=4.0, ratio=0.75)
    result = rca.ten_m_ratio_decomposition(ratio_only, cpu)
    assert result["ratio_only_v10_rmse"] == result["actual_v10_rmse"] == 1.0
    assert result["wind_only_v10_rmse"] == 0.0

    wind_only = _wind_fields(low_u=3.0, low_v=6.0, ratio=0.5)
    result = rca.ten_m_ratio_decomposition(wind_only, cpu)
    assert result["wind_only_v10_rmse"] == result["actual_v10_rmse"] == 1.0
    assert result["ratio_only_v10_rmse"] == 0.0


def test_rca_binds_pristine_wrf_and_unchanged_model_tree() -> None:
    authority = rca.source_authority()
    assert authority["model_tree"] == terminal.MODEL_TREE
    assert authority["wrf_mynn_surface_source"]["file_sha256"] == rca.WRF_MYNN_SHA256
    assert authority["gpu_surface_source"]["file_sha256"] == rca.GPU_SURFACE_SHA256
    assert terminal.FIXED_FILES[terminal.STAGED_PIN].startswith("edd3b127")
