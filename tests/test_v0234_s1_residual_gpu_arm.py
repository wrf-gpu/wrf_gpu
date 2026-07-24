"""CPU-only adversarial tests for the held S1 residual-closure GPU arm."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts import v0234_s1_residual_gpu_arm_closeout as closeout
from scripts import v0234_s1_residual_gpu_arm_profile as arm

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "scripts/v0234_nested_frozen_wrf_boundary_window.py"
LADDER_PROFILE = REPO / "scripts/v0234_dycore_suboperator_kimi_gpu_ladder.py"


def authorization(path: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": arm.AUTHORIZATION_SCHEMA,
        "decision": "RELEASE_GPU",
        "issuer": "manager-0:1",
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
    payload["proof_sha256"] = arm._canonical(payload)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def test_profile_top_level_has_no_runtime_import() -> None:
    tree = ast.parse(Path(arm.__file__).read_text())
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not {name.split(".")[0] for name in imports} & {
        "jax", "gpuwrf", "numpy", "netCDF4",
    }
    completed = subprocess.run(
        (sys.executable, "-c", "import sys; import scripts.v0234_s1_residual_gpu_arm_profile; assert 'jax' not in sys.modules; assert not any(n.startswith('gpuwrf') for n in sys.modules)"),
        cwd=REPO, check=False,
    )
    assert completed.returncode == 0


def test_exact_source_candidate_and_hashes() -> None:
    assert subprocess.run(
        ("git", "-C", str(REPO), "merge-base", "--is-ancestor", arm.CANDIDATE_HEAD, "HEAD"),
        check=False,
    ).returncode == 0
    actual = subprocess.run(
        ("git", "-C", str(REPO), "diff", "--name-only", arm.MODEL_PARENT,
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


def test_contract_and_launcher_are_exact_and_held() -> None:
    assert hashlib.sha256(arm.CONTRACT.read_bytes()).hexdigest() == arm.CONTRACT_SHA256
    audit = arm.audit_exact_launcher()
    assert audit["passed"] is True
    assert audit["authorization_before_freshness_before_lock"] is True
    assert audit["closeout_after_lock_before_receipt"] is True
    assert subprocess.run(("bash", "-n", str(arm.LAUNCHER)), check=False).returncode == 0
    source = arm.LAUNCHER.read_text()
    assert source.count("/scripts/with_gpu_lock.sh") == 1
    assert source.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1
    assert "HOLD" in source


def test_residual_profile_precedes_diff6_and_ladder_profiles() -> None:
    source = ENTRY.read_text()
    residual = source.index("GPUWRF_S1_RESIDUAL_VALIDATION")
    diff6 = source.index("GPUWRF_S1_DIFF6_VALIDATION")
    ladder = source.index("GPUWRF_DYCORE_SUBOPERATOR_LADDER")
    assert residual < diff6 < ladder
    ladder_profile = LADDER_PROFILE.read_text()
    assert 'os.environ.get("GPUWRF_S1_RESIDUAL_VALIDATION") != "1"' in ladder_profile
    assert "S1_RESIDUAL_MODEL_HASHES" in ladder_profile


def test_authorization_absence_and_semantic_mutation_are_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "authorization.json"
    monkeypatch.setattr(arm, "AUTHORIZATION_PATH", path)
    runtime_before = {
        name for name in sys.modules if name == "jax" or name.startswith("gpuwrf")
    }
    with pytest.raises(ValueError, match="absent"):
        arm.validate_authorization(path, expected_head=arm.CANDIDATE_HEAD)
    authorization(path, issuer="not-manager")
    with pytest.raises(ValueError, match="issuer"):
        arm.validate_authorization(
            path, expected_head=arm.CANDIDATE_HEAD,
            now=datetime(2026, 7, 18, tzinfo=timezone.utc),
        )
    runtime_after = {
        name for name in sys.modules if name == "jax" or name.startswith("gpuwrf")
    }
    assert runtime_after == runtime_before


def test_authorization_exact_valid_and_self_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "authorization.json"
    monkeypatch.setattr(arm, "AUTHORIZATION_PATH", path)
    payload = authorization(path)
    row = arm.validate_authorization(
        path, expected_head=arm.CANDIDATE_HEAD,
        now=datetime(2026, 7, 18, tzinfo=timezone.utc),
    )
    assert row["proof_sha256"] == payload["proof_sha256"]
    assert row["validated_before_runtime_import"] is True


def test_raw_step1_classifier_is_inclusive_and_nonfinite_red() -> None:
    green = arm.classify_step1_metrics(dict(arm.PREDICTED))
    assert green["passed"] is True and green["step2_dispatch_admitted"] is True
    red_values = dict(arm.PREDICTED)
    red_values["nonspec"] += arm.ABSOLUTE_BAND * 1.01
    red = arm.classify_step1_metrics(red_values)
    assert red["passed"] is False and red["first_red"] == "nonspec"
    red_values["nonspec"] = math.nan
    assert arm.classify_step1_metrics(red_values)["passed"] is False


def test_closeout_four_zone_raw_and_intrinsic_green() -> None:
    shape = (1, 20, 20)
    raw: dict[str, np.ndarray] = {}
    core = {name: np.zeros(shape, dtype=np.float64) for name in ("u", "v")}
    for name in ("u", "v"):
        distance = closeout.ledger.cpu.distance_to_edge(shape)
        value = np.full(shape, closeout.RAW_PREDICTIONS["interior_ge_5"])
        value[(distance >= 1) & (distance <= 4)] = closeout.RAW_PREDICTIONS["relax_rows_1_4"]
        value[distance == 1] = closeout.RAW_PREDICTIONS["ring_1"]
        raw[name] = value
    gate = closeout.classify_fields(raw, core)
    assert gate["passed"] is True
    assert set(gate["rows"]) == set(closeout.RAW_PREDICTIONS)


def test_closeout_rejects_intrinsic_and_raw_deviation() -> None:
    shape = (1, 20, 20)
    raw = {name: np.full(shape, 2.0) for name in ("u", "v")}
    core = {name: np.full(shape, 0.02) for name in ("u", "v")}
    gate = closeout.classify_fields(raw, core)
    assert gate["passed"] is False
    assert gate["first_red"] == "nonspec"


def test_closeout_binds_archive_inventory_and_cpu_only_scope() -> None:
    assert closeout.sha256_file(closeout.PREDICTIONS) == closeout.PREDICTIONS_SHA256
    source = Path(closeout.__file__).read_text()
    assert "== 180" in source
    assert "gpu_queries\": 0" in source
    assert "gpu_compiles\": 0" in source
    assert "gpu_dispatches\": 0" in source
    assert "import jax" not in source
    assert "import gpuwrf" not in source
    for forbidden in ("nvidia-smi", "rocm-smi", "with_gpu_lock.sh"):
        assert forbidden not in source
