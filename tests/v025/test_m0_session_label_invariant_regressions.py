"""Active invariant regressions migrated from the original GPT critic suite.

The original mixed critic file is immutable historical rejection evidence.  Its
test helpers remain the authority for these byte-faithful invariant bodies, but
only the twelve durable invariant cases below are collected here.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
HISTORICAL_CRITIC = (
    REPO
    / ".agent/sprints/2026-07-30-v0250-m0-session-label-gpt-critic"
    / "historical_evidence/test_m0_session_label_gpt_critic.py"
)
_historical = types.ModuleType("_m0_retired_gpt_critic_evidence")
_historical.__file__ = str(
    REPO / "tests/v025/test_m0_session_label_gpt_critic.py"
)
exec(
    compile(
        HISTORICAL_CRITIC.read_bytes(),
        _historical.__file__,
        "exec",
    ),
    _historical.__dict__,
)

LIVE_DEFECT_COMMIT = _historical.LIVE_DEFECT_COMMIT
LIVE_MESSAGE = _historical.LIVE_MESSAGE
SESSION_LABEL = _historical.SESSION_LABEL
SCRIPTS = _historical.SCRIPTS
STAGE_LABELS = _historical.STAGE_LABELS
_require_synthetic = _historical._require_synthetic
_run_direct_child = _historical._run_direct_child
_run_legacy_authorise = _historical._run_legacy_authorise
_run_real_parent = _historical._run_real_parent


def test_archived_bb1_parent_reproduces_the_live_failure(tmp_path):
    """Gate 1: archived parent producer -> archived child, not a lower helper."""

    archive_root = _require_synthetic(tmp_path / "bb1-archive", tmp_path)
    archive_root.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", LIVE_DEFECT_COMMIT, "scripts/v025"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(archive_root)],
        input=archive.stdout,
        check=True,
    )
    outcome = _run_real_parent(
        _require_synthetic(tmp_path / "bb1-run", tmp_path),
        script_root=archive_root / "scripts/v025",
        label=SESSION_LABEL,
        stage_identity="W1",
        stage="gpt_critic_archived_parent",
    )
    assert LIVE_MESSAGE in str(outcome["log"])
    assert "expected_label=expected_window" in str(outcome["log"])
    assert "child returned 1" in str(outcome["caught"])
    assert outcome["handoff_remaining"] is True
    assert outcome["handoff_spent"] is False
    assert outcome["result_written"] is False


@pytest.mark.parametrize("stage_identity", ["W1", "W2", "W3"])
def test_current_real_parent_reaches_the_real_child_for_every_stage(
    tmp_path, stage_identity
):
    """Gate 2: execute the actual parent command builder and actual child CLI."""

    root = _require_synthetic(tmp_path / f"current-{stage_identity}", tmp_path)
    outcome = _run_real_parent(
        root,
        script_root=SCRIPTS,
        label=SESSION_LABEL,
        stage_identity=stage_identity,
        stage=f"gpt_critic_real_parent_{stage_identity}",
    )
    payload = json.loads(str(outcome["log"]))
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["reason"].startswith("WRF source authority:")
    assert payload["session_label"] == SESSION_LABEL
    assert payload["window"] == stage_identity
    assert payload["handoff_consumed"] is True
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert payload["device_touched"] is False
    assert outcome["handoff_remaining"] is False
    assert outcome["handoff_spent"] is True
    assert outcome["result_written"] is False


@pytest.mark.parametrize("stage_identity", ["W1", "W2", "W3"])
def test_legacy_outer_spender_binds_the_exact_stage_coordination_label(
    tmp_path, stage_identity
):
    """The existing legacy outer path itself remains exact and synthetic."""

    label = STAGE_LABELS[stage_identity]
    outcome = _run_legacy_authorise(
        tmp_path / stage_identity,
        stage_identity=stage_identity,
        receipt_label=label,
        holder_label=label,
    )
    authorization = outcome["authorization"]
    assert outcome["error"] is None
    assert authorization["window"] == stage_identity
    assert authorization["label"] == label
    assert authorization["canonical_lock"]["exported_label"] == label
    assert outcome["receipt"]["spent"]["window"] == label
    assert len(outcome["ledger"]["spent"]) == 1


def test_legacy_outer_spender_refuses_a_cross_stage_receipt_before_spend(tmp_path):
    wrong_label = STAGE_LABELS["W2"]
    outcome = _run_legacy_authorise(
        tmp_path / "legacy-cross-stage",
        stage_identity="W1",
        receipt_label=wrong_label,
        holder_label=wrong_label,
    )
    assert "WindowNotAuthorised" in outcome["error"]
    assert "not 'm0-autotune0-qualification'" in outcome["error"]
    assert "spent" not in outcome["receipt"]
    assert outcome["ledger"] is None


def test_old_v1_payload_fails_closed_and_is_not_consumed(tmp_path):
    root = _require_synthetic(tmp_path / "old-v1", tmp_path)
    root.mkdir()

    def old_v1(payload):
        payload["schema"] = "wrf_gpu2.v025.m0.authorized_child_handoff.v1"
        payload.pop("session_label")
        return payload

    outcome = _run_direct_child(root, payload_edit=old_v1)
    assert outcome["returncode"] == 2
    assert "fields are incomplete" in outcome["payload"]["reason"]
    assert outcome["payload"]["handoff_consumed"] is False
    assert outcome["handoff_remaining"] is True
    assert outcome["handoff_spent"] is False


def test_lock_refusal_is_structured_and_preserves_the_fresh_handoff(tmp_path):
    root = _require_synthetic(tmp_path / "lock-refusal", tmp_path)
    root.mkdir()

    def wrong_export(environment):
        environment["GPUWRF_GPU_LOCK_LABEL"] = STAGE_LABELS["W2"]

    outcome = _run_direct_child(root, environment_edit=wrong_export)
    payload = outcome["payload"]
    assert outcome["returncode"] == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["reason"].startswith("canonical lock refused the session authority")
    assert payload["handoff_consumed"] is False
    assert outcome["handoff_remaining"] is True
    assert outcome["handoff_spent"] is False
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert payload["device_touched"] is False


def test_malformed_holder_is_structured_and_preserves_the_handoff(tmp_path):
    root = _require_synthetic(tmp_path / "malformed-holder", tmp_path)
    root.mkdir()
    outcome = _run_direct_child(
        root,
        holder_text=(
            f"holder={SESSION_LABEL} token=gpt-critic-synthetic-token "
            "pid=1 cmd=wrong-order\n"
        ),
    )
    assert outcome["returncode"] == 2
    assert "not one exact canonical-wrapper record" in outcome["payload"]["reason"]
    assert outcome["payload"]["handoff_consumed"] is False
    assert outcome["handoff_remaining"] is True
    assert outcome["handoff_spent"] is False


def test_critic_probes_name_no_real_receipt_lock_or_mnt_write():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    external_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name in {"run", "Popen"}:
            external_calls.append(ast.unparse(node))
    assert not any("with_gpu_lock" in call for call in external_calls)
    assert not any("nvidia-smi" in call for call in external_calls)
    assert not any("GPU_COORDINATION_RECEIPT" in call for call in external_calls)
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
