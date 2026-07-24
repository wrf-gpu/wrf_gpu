"""CPU-only adversarial tests for the held v0234 S1 diff6 GPU arm."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import v0234_s1_diff6_gpu_arm_profile as arm

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "scripts/v0234_dycore_suboperator_kimi_gpu_ladder.py"
ENTRY = REPO / "scripts/v0234_nested_frozen_wrf_boundary_window.py"


def _authorization(path: Path, **overrides):
    payload = {
        "schema": arm.AUTHORIZATION_SCHEMA,
        "decision": "RELEASE_GPU",
        "issuer": "chief-0:3",
        "intent": "production-preemptible",
        "one_arm": True,
        "sprint": arm.SPRINT_NAME,
        "namespace": arm.NAMESPACE,
        "lock_label": arm.LOCK_LABEL,
        "candidate_head": arm.CANDIDATE_HEAD,
        "expires_utc": "2099-01-01T00:00:00+00:00",
        "nonce": "unit-test-one-run-nonce",
    }
    payload.update(overrides)
    payload["proof_sha256"] = arm._canonical_digest(payload)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def test_profile_top_level_is_runtime_import_free():
    tree = ast.parse(Path(arm.__file__).read_text())
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not {
        name.split(".")[0] for name in imports
    } & {"jax", "gpuwrf", "numpy", "netCDF4"}
    assert "jax" not in sys.modules
    assert not any(name.startswith("gpuwrf") for name in sys.modules)


def test_authorization_absent_blocks_before_any_runtime_import(tmp_path, monkeypatch):
    path = tmp_path / "authorization.json"
    monkeypatch.setattr(arm, "AUTHORIZATION_PATH", path)
    with pytest.raises(ValueError, match="absent"):
        arm.validate_authorization(path, expected_head=arm.CANDIDATE_HEAD)
    assert "jax" not in sys.modules
    assert not any(name.startswith("gpuwrf") for name in sys.modules)


def test_authorization_exact_valid_and_self_hashed(tmp_path, monkeypatch):
    path = tmp_path / "authorization.json"
    monkeypatch.setattr(arm, "AUTHORIZATION_PATH", path)
    payload = _authorization(path)
    row = arm.validate_authorization(
        path,
        expected_head=arm.CANDIDATE_HEAD,
        now=datetime(2026, 7, 18, tzinfo=timezone.utc),
    )
    assert row["proof_sha256"] == payload["proof_sha256"]
    assert row["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert row["validated_before_runtime_import"] is True


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"candidate_head": "0" * 40}, "candidate_head"),
        ({"issuer": "not-chief"}, "issuer"),
        ({"one_arm": False}, "one_arm"),
        ({"intent": "diagnostic"}, "intent"),
        ({"expires_utc": "2020-01-01T00:00:00+00:00"}, "expired"),
        ({"nonce": ""}, "nonce"),
    ],
)
def test_authorization_semantic_mutations_fail(tmp_path, monkeypatch, overrides, match):
    path = tmp_path / "authorization.json"
    monkeypatch.setattr(arm, "AUTHORIZATION_PATH", path)
    _authorization(path, **overrides)
    with pytest.raises(ValueError, match=match):
        arm.validate_authorization(
            path,
            expected_head=arm.CANDIDATE_HEAD,
            now=datetime(2026, 7, 18, tzinfo=timezone.utc),
        )


def test_authorization_extra_field_and_hash_tamper_fail(tmp_path, monkeypatch):
    path = tmp_path / "authorization.json"
    monkeypatch.setattr(arm, "AUTHORIZATION_PATH", path)
    payload = _authorization(path)
    payload["retry"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fields"):
        arm.validate_authorization(path, expected_head=arm.CANDIDATE_HEAD)
    payload.pop("retry")
    payload["proof_sha256"] = "f" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self-hash"):
        arm.validate_authorization(path, expected_head=arm.CANDIDATE_HEAD)


def test_threshold_boundaries_are_inclusive_and_step2_is_withheld_on_first_red():
    green = arm.classify_step1_metrics({
        "nonspec": 4.0,
        "relax_rows_1_4": 9.5,
        "interior_ge_5": 1.4,
    })
    assert green["passed"] is True
    assert green["step2_dispatch_admitted"] is True

    red = arm.classify_step1_metrics({
        "nonspec": 4.0 + 1e-12,
        "relax_rows_1_4": 0.0,
        "interior_ge_5": 0.0,
    })
    assert red["passed"] is False
    assert red["first_red"] == "nonspec"
    assert red["step2_dispatch_admitted"] is False


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_threshold_input_is_red(bad):
    result = arm.classify_step1_metrics({
        "nonspec": bad,
        "relax_rows_1_4": 0.0,
        "interior_ge_5": 0.0,
    })
    assert result["passed"] is False
    assert result["first_red"] == "nonspec"
    assert result["step2_dispatch_admitted"] is False


def test_fresh_hlo_is_retained_before_policy_and_ordinary_program_never_compiles():
    source = LADDER.read_text()
    runtime = source[source.index("def _ladder_runtime_main("):]
    assert runtime.index("fresh_hlo_retainer(") < runtime.index(
        "evaluate_stablehlo_policy(production_stablehlo)"
    )
    assert "production_lowered.compile(" not in runtime
    assert "production_lowered(" not in runtime
    assert "ordinary-one-step-fixed-reference-lowered-hlo.json" in runtime


def test_step1_savepoint_gate_precedes_any_next_loop_dispatch():
    source = LADDER.read_text()
    window = source[
        source.index("def _run_ladder_window("):
        source.index("def _ladder_runtime_main(")
    ]
    assert window.index("_write_step_savepoints(") < window.index("after_step_gate(")
    assert "S1_DIFF6_THRESHOLD_RED" in window
    assert '"next_dispatch_withheld": True' in window
    assert window.index("after_step_gate(") < window.index("return value")


def test_profile_selection_precedes_legacy_ladder_selection():
    source = ENTRY.read_text()
    assert source.index("GPUWRF_S1_DIFF6_VALIDATION") < source.index(
        "GPUWRF_DYCORE_SUBOPERATOR_LADDER"
    )


def test_exact_two_file_production_pins_and_no_reviewer_numerics_edit():
    actual = subprocess.run(
        ("git", "-C", str(REPO), "diff", "--name-only", arm.FABLE_PARENT,
         arm.CANDIDATE_HEAD, "--", "src/gpuwrf"),
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    assert actual == sorted(arm.MODEL_HASHES)
    for relative, expected in arm.MODEL_HASHES.items():
        frozen = subprocess.run(
            ("git", "-C", str(REPO), "show", f"{arm.CANDIDATE_HEAD}:{relative}"),
            check=True, stdout=subprocess.PIPE,
        ).stdout
        assert hashlib.sha256(frozen).hexdigest() == expected


def test_launcher_is_one_lock_one_model_and_shell_valid():
    audit = arm.audit_exact_launcher()
    assert audit["passed"] is True
    assert audit["canonical_lock_invocations"] == 1
    assert audit["model_processes"] == 1
    assert audit["authorization_before_freshness_before_lock"] is True
    assert audit["release_receipt_before_green_finalization"] is True
    assert subprocess.run(
        ("bash", "-n", str(arm.LAUNCHER)), check=False,
    ).returncode == 0


def test_release_receipt_is_self_hashed_and_requires_exact_namespace(tmp_path):
    run_dir = tmp_path / arm.NAMESPACE
    run_dir.mkdir()
    assert arm._main([
        "--seal-release-receipt", str(run_dir),
        "--returncode", "3",
        "--authorization-sha", "a" * 64,
    ]) == 0
    receipt = json.loads((run_dir / "gpu-lock-release-receipt.json").read_text())
    assert receipt["wrapper_returned"] is True
    assert receipt["file_descriptor_lease_released_before_receipt"] is True
    assert receipt["model_returncode"] == 3
    assert receipt["gpu_queries_run_by_receipt"] == 0
    assert receipt["proof_sha256"] == arm._canonical_digest(
        receipt, omit="proof_sha256"
    )
    with pytest.raises(ValueError, match="already exists"):
        arm._main([
            "--seal-release-receipt", str(run_dir),
            "--returncode", "3",
            "--authorization-sha", "a" * 64,
        ])


def test_arm_is_exactly_one_root_step_and_excludes_broader_scope():
    contract = arm.CONTRACT.read_text()
    launcher = arm.LAUNCHER.read_text()
    assert arm.NAMESPACE in contract and arm.NAMESPACE in launcher
    assert arm.THRESHOLDS == {
        "nonspec": 4.0,
        "relax_rows_1_4": 9.5,
        "interior_ge_5": 1.4,
    }
    assert "steps `1..9`" in contract
    for forbidden in (
        "--direct-terminal", "--parent-join-resume",
        "--record-known-1500-v10-red", "nvidia-smi", "rocm-smi",
    ):
        assert forbidden not in launcher
