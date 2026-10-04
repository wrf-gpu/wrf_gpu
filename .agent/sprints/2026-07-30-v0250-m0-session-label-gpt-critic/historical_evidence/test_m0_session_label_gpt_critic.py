"""Independent GPT critic attacks for the M0 session-label repair.

These tests are deliberately separate from the candidate's mutation vocabulary.
They execute the archived pre-repair parent, the current real parent-to-child
transition, the legacy synthetic receipt path, and the Pallas native handoff.
Every write stays below a pytest-owned temporary root.  The test process is
required to run with ``JAX_PLATFORMS=cpu``, an empty ``CUDA_VISIBLE_DEVICES``,
and OS ``DevicePolicy=closed``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpu_window_registry as registry  # noqa: E402
import m0_core_session_protocol as core_session  # noqa: E402
import m0_exact_boundary_child as child  # noqa: E402
import m0_window_parent as parent  # noqa: E402


SESSION_LABEL = core_session.SESSION_LABEL
LIVE_DEFECT_COMMIT = "bb1b1cd5"
LIVE_MESSAGE = (
    "the exported canonical-lock label differs from the exact authorized "
    f"label: '{SESSION_LABEL}' != 'W1'"
)
STAGE_LABELS = {
    stage: str(configuration["window_label"])
    for stage, configuration in registry.SESSION_STAGE_IDENTITIES.items()
}
STAGE_RUN_IDS = {
    stage: str(configuration["run_id"])
    for stage, configuration in registry.SESSION_STAGE_IDENTITIES.items()
}
FORBIDDEN_WRITE_TARGETS = (
    Path("/tmp/wrf_gpu2_gpu.lock"),
    Path("/tmp/wrf_gpu2_gpu.lock.holder"),
    Path("<DATA_ROOT>"),
    REPO,
    Path("<USER_HOME>/src/wrf_gpu2"),
    Path("<USER_HOME>/src/wrf_gpu2_wt"),
)


class CriticProbeRefusal(RuntimeError):
    """The critic probe refused a real coordination or repository write path."""


def _require_synthetic(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    synthetic_root = Path(root).resolve()
    if synthetic_root == Path("/tmp"):
        raise CriticProbeRefusal("a dedicated temporary root is required")
    if resolved != synthetic_root and synthetic_root not in resolved.parents:
        raise CriticProbeRefusal(f"{resolved} is outside {synthetic_root}")
    for forbidden in FORBIDDEN_WRITE_TARGETS:
        candidate = forbidden.resolve()
        if resolved == candidate or candidate in resolved.parents:
            raise CriticProbeRefusal(f"refusing real path {resolved}")
    return resolved


def _clean_cpu_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GPUWRF_", "XLA_", "JAX_", "CUDA_"))
    }
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
        }
    )
    return environment


def _holder_environment(
    root: Path,
    *,
    label: str,
    token: str = "gpt-critic-synthetic-token",
    holder_text: str | None = None,
) -> dict[str, str]:
    holder = _require_synthetic(root / "synthetic.holder", root)
    holder.write_text(
        holder_text
        or (
            f"holder={label} pid={os.getpid()} "
            f"since=2026-07-30T00:00:00Z token={token} cmd=gpt-critic\n"
        ),
        encoding="utf-8",
    )
    environment = _clean_cpu_environment()
    environment.update(
        {
            "GPUWRF_GPU_LOCK_HELD": "1",
            "GPUWRF_GPU_LOCK_TOKEN": token,
            "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
            "GPUWRF_GPU_LOCK_LABEL": label,
            "GPUWRF_M0_PARENT_LAUNCH_NS": "1",
        }
    )
    return environment


def _write_parent_handoff(
    root: Path,
    *,
    label: str = SESSION_LABEL,
    stage_identity: str = "W1",
    stage: str = "gpt_critic_direct_child",
    token: str = "gpt-critic-synthetic-token",
    edit=None,
    symlink: bool = False,
) -> Path:
    previous = os.environ.get("GPUWRF_GPU_LOCK_TOKEN")
    os.environ["GPUWRF_GPU_LOCK_TOKEN"] = token
    try:
        payload = parent._handoff_payload(
            session_label=label,
            window_id=stage_identity,
            stage=stage,
            run_id=STAGE_RUN_IDS[stage_identity],
            authorization={
                "receipt_fingerprint": "c" * 64,
                "receipt_spent_at_utc": "2026-07-30T00:00:00+00:00",
            },
        )
    finally:
        if previous is None:
            os.environ.pop("GPUWRF_GPU_LOCK_TOKEN", None)
        else:
            os.environ["GPUWRF_GPU_LOCK_TOKEN"] = previous
    if edit is not None:
        payload = edit(dict(payload))
    target = _require_synthetic(root / "authorization.target.json", root)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not symlink:
        handoff = _require_synthetic(root / "authorization.json", root)
        target.replace(handoff)
        return handoff
    handoff = _require_synthetic(root / "authorization.json", root)
    handoff.symlink_to(target)
    return handoff


def _run_direct_child(
    root: Path,
    *,
    label: str = SESSION_LABEL,
    stage_identity: str = "W1",
    stage: str = "gpt_critic_direct_child",
    payload_edit=None,
    environment_edit=None,
    holder_text: str | None = None,
    symlink_handoff: bool = False,
) -> dict[str, object]:
    root = _require_synthetic(root, root)
    token = "gpt-critic-synthetic-token"
    environment = _holder_environment(
        root, label=label, token=token, holder_text=holder_text
    )
    if environment_edit is not None:
        environment_edit(environment)
    handoff = _write_parent_handoff(
        root,
        label=label,
        stage_identity=stage_identity,
        stage=stage,
        token=token,
        edit=payload_edit,
        symlink=symlink_handoff,
    )
    result = _require_synthetic(root / "result.json", root)
    command = [
        sys.executable,
        str(SCRIPTS / "m0_exact_boundary_child.py"),
        "--session-label",
        label,
        "--window",
        stage_identity,
        "--stage",
        stage,
        "--run-id",
        STAGE_RUN_IDS[stage_identity],
        "--mode",
        "compile-only",
        "--handoff",
        str(handoff),
        "--result",
        str(result),
    ]
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout) if completed.stdout.strip() else None
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
        "handoff_remaining": handoff.exists() or handoff.is_symlink(),
        "handoff_spent": Path(f"{handoff}.spent").exists(),
        "result_written": result.exists(),
    }


_REAL_PARENT_PROBE = textwrap.dedent(
    r"""
    import json
    import os
    import pathlib
    import sys
    from datetime import datetime, timezone

    script_root = pathlib.Path(os.environ["GPT_CRITIC_SCRIPT_ROOT"])
    root = pathlib.Path(os.environ["GPT_CRITIC_ROOT"])
    label = os.environ["GPT_CRITIC_LABEL"]
    stage_identity = os.environ["GPT_CRITIC_STAGE_IDENTITY"]
    stage = os.environ["GPT_CRITIC_STAGE"]
    sys.path.insert(0, str(script_root))
    import m0_window_parent as parent

    token = "gpt-critic-real-parent-token"
    holder = root / "synthetic.holder"
    holder.write_text(
        f"holder={label} pid={os.getpid()} since=2026-07-30T00:00:00Z "
        f"token={token} cmd=gpt-critic-real-parent\n",
        encoding="utf-8",
    )
    child_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GPUWRF_", "XLA_", "JAX_", "CUDA_"))
    }
    child_environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "GPUWRF_GPU_LOCK_HELD": "1",
            "GPUWRF_GPU_LOCK_TOKEN": token,
            "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
            "GPUWRF_GPU_LOCK_LABEL": label,
        }
    )
    parent._base_child_environment = (
        lambda cache_path, extra_xla_flags=(): dict(child_environment)
    )
    previous = os.environ.get("GPUWRF_GPU_LOCK_TOKEN")
    os.environ["GPUWRF_GPU_LOCK_TOKEN"] = token
    caught = None
    try:
        parent._launch_authorized_child(
            window_id=stage_identity,
            stage=stage,
            run_id=parent.WINDOWS[stage_identity]["run_id"],
            mode="compile-only",
            cache_path=root / "cache",
            run_root=root / "run",
            authorization={
                "window": os.environ.get(
                    "GPT_CRITIC_AUTHORIZATION_SCOPE", "M0_CORE_SESSION"
                ),
                "label": label,
                "receipt_fingerprint": "d" * 64,
                "receipt_spent_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            timeout_seconds=10.0,
        )
    except Exception as exc:
        caught = f"{type(exc).__name__}: {exc}"
    finally:
        if previous is None:
            os.environ.pop("GPUWRF_GPU_LOCK_TOKEN", None)
        else:
            os.environ["GPUWRF_GPU_LOCK_TOKEN"] = previous
    stage_root = root / "run" / stage
    log_path = stage_root / "child.log"
    handoff = stage_root / "authorization.json"
    print(
        json.dumps(
            {
                "caught": caught,
                "log": log_path.read_text(encoding="utf-8")
                if log_path.exists()
                else None,
                "handoff_remaining": handoff.exists(),
                "handoff_spent": pathlib.Path(f"{handoff}.spent").exists(),
                "result_written": (stage_root / "exact_boundary.json").exists(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    """
)


def _run_real_parent(
    root: Path,
    *,
    script_root: Path,
    label: str,
    stage_identity: str,
    stage: str,
    authorization_scope: str = "M0_CORE_SESSION",
) -> dict[str, object]:
    root = _require_synthetic(root, root)
    root.mkdir(parents=True, exist_ok=False)
    environment = _clean_cpu_environment()
    environment.update(
        {
            "GPT_CRITIC_SCRIPT_ROOT": str(script_root),
            "GPT_CRITIC_ROOT": str(root),
            "GPT_CRITIC_LABEL": label,
            "GPT_CRITIC_STAGE_IDENTITY": stage_identity,
            "GPT_CRITIC_STAGE": stage,
            "GPT_CRITIC_AUTHORIZATION_SCOPE": authorization_scope,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _REAL_PARENT_PROBE],
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _fresh_receipt(path: Path, *, label: str) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "window": label,
        "requested_at_utc": now,
        "request_text": f"synthetic critic request for {label}",
        "replies": {
            "0:2": {
                "affirmative": True,
                "verbatim": "synthetic yes from 0:2",
                "received_at_utc": now,
            },
            "0:3": {
                "affirmative": True,
                "verbatim": "synthetic yes from 0:3",
                "received_at_utc": now,
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


_LEGACY_AUTHORISE_PROBE = textwrap.dedent(
    r"""
    import json
    import os
    import pathlib
    import sys
    from datetime import datetime, timezone

    root = pathlib.Path(os.environ["GPT_CRITIC_ROOT"])
    stage_identity = os.environ["GPT_CRITIC_STAGE_IDENTITY"]
    receipt_label = os.environ["GPT_CRITIC_RECEIPT_LABEL"]
    holder_label = os.environ["GPT_CRITIC_HOLDER_LABEL"]
    sys.path.insert(0, os.environ["GPT_CRITIC_SCRIPT_ROOT"])
    import m0_window_parent as parent

    now = datetime.now(timezone.utc).isoformat()
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "window": receipt_label,
                "requested_at_utc": now,
                "request_text": "synthetic critic legacy request",
                "replies": {
                    "0:2": {
                        "affirmative": True,
                        "verbatim": "synthetic yes from 0:2",
                        "received_at_utc": now,
                    },
                    "0:3": {
                        "affirmative": True,
                        "verbatim": "synthetic yes from 0:3",
                        "received_at_utc": now,
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    token = "gpt-critic-legacy-token"
    holder = root / "synthetic.holder"
    holder.write_text(
        f"holder={holder_label} pid={os.getpid()} "
        f"since=2026-07-30T00:00:00Z token={token} cmd=gpt-critic\n",
        encoding="utf-8",
    )
    lock_environment = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_TOKEN": token,
        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
        "GPUWRF_GPU_LOCK_LABEL": holder_label,
    }
    ledger = root / "spent.json"
    authorization = None
    error = None
    try:
        authorization = parent.authorise_window(
            stage_identity,
            receipt_path=receipt,
            ledger_path=ledger,
            env=lock_environment,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    print(
        json.dumps(
            {
                "authorization": authorization,
                "error": error,
                "receipt": json.loads(receipt.read_text(encoding="utf-8")),
                "ledger": json.loads(ledger.read_text(encoding="utf-8"))
                if ledger.exists()
                else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    """
)


def _run_legacy_authorise(
    root: Path,
    *,
    stage_identity: str,
    receipt_label: str,
    holder_label: str,
) -> dict[str, object]:
    root = _require_synthetic(root, root)
    root.mkdir(parents=True, exist_ok=False)
    environment = _clean_cpu_environment()
    environment.update(
        {
            "GPT_CRITIC_ROOT": str(root),
            "GPT_CRITIC_SCRIPT_ROOT": str(SCRIPTS),
            "GPT_CRITIC_STAGE_IDENTITY": stage_identity,
            "GPT_CRITIC_RECEIPT_LABEL": receipt_label,
            "GPT_CRITIC_HOLDER_LABEL": holder_label,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _LEGACY_AUTHORISE_PROBE],
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


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


@pytest.mark.parametrize(
    ("stage_identity", "wrong_registered_label"),
    [
        ("W1", STAGE_LABELS["W2"]),
        ("W2", STAGE_LABELS["W3"]),
        ("W3", STAGE_LABELS["W1"]),
        ("W1", "baseline-census"),
    ],
)
def test_real_parent_accepts_a_coherent_wrong_registered_label(
    tmp_path, stage_identity, wrong_registered_label
):
    """Critic-owned attack: a wrong registered label crosses and is consumed.

    The expected safe behavior is an authority refusal.  The observed candidate
    behavior reaches the unrelated WRF gate and consumes the child handoff,
    proving that the child enforces only membership in the coordination-label
    namespace, not the stage/context-specific label relation.
    """

    root = _require_synthetic(
        tmp_path / f"wrong-{stage_identity}-{wrong_registered_label}", tmp_path
    )
    outcome = _run_real_parent(
        root,
        script_root=SCRIPTS,
        label=wrong_registered_label,
        stage_identity=stage_identity,
        stage=f"gpt_critic_wrong_label_{stage_identity}",
    )
    payload = json.loads(str(outcome["log"]))
    assert payload["reason"].startswith("WRF source authority:")
    assert payload["session_label"] == wrong_registered_label
    assert payload["window"] == stage_identity
    assert payload["handoff_consumed"] is True
    assert outcome["handoff_spent"] is True


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


@pytest.mark.parametrize(
    ("attack", "payload_edit", "environment_edit", "symlink", "reason"),
    [
        (
            "handoff_stage_mismatch",
            lambda payload: {**payload, "stage": "a_different_stage"},
            None,
            False,
            "authorization handoff stage=",
        ),
        (
            "missing_parent_launch_endpoint",
            None,
            lambda environment: environment.pop("GPUWRF_M0_PARENT_LAUNCH_NS"),
            False,
            "parent launch monotonic endpoint is missing/invalid",
        ),
        (
            "empty_receipt_fingerprint",
            lambda payload: {**payload, "receipt_fingerprint": "   "},
            None,
            False,
            "authorization receipt fingerprint is empty",
        ),
        (
            "symlink_handoff",
            None,
            None,
            True,
            "authorization handoff must not be a symlink",
        ),
    ],
)
def test_candidate_gate_inventory_omits_enforced_guards(
    tmp_path, attack, payload_edit, environment_edit, symlink, reason
):
    """Gate 6: dynamically prove four enforced checks absent from the list."""

    root = _require_synthetic(tmp_path / attack, tmp_path)
    root.mkdir()
    outcome = _run_direct_child(
        root,
        payload_edit=payload_edit,
        environment_edit=environment_edit,
        symlink_handoff=symlink,
    )
    payload = outcome["payload"]
    assert outcome["returncode"] == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert reason in payload["reason"]
    assert payload["handoff_consumed"] is False
    assert outcome["handoff_spent"] is False
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert payload["device_touched"] is False
    declared = set(child.SESSION_AUTHORITY_HARD_GATES)
    assert {
        "handoff_stage_matches_cli",
        "parent_launch_endpoint_valid",
        "receipt_fingerprint_nonempty",
        "handoff_regular_nonsymlink",
    }.isdisjoint(declared)


def test_candidate_16_gate_25_mutation_equality_is_only_self_declared():
    """The submitted set equality is true but does not equal implementation."""

    sys.path.insert(0, str(REPO / "tests/v025"))
    import test_m0_session_label_authority as candidate_suite

    declared = set(child.SESSION_AUTHORITY_HARD_GATES)
    covered = {gate for _name, _kwargs, gate, _expected in candidate_suite.MUTATIONS}
    assert len(declared) == 16
    assert len(candidate_suite.MUTATIONS) == 25
    assert covered == declared
    source = Path(child.__file__).read_text(encoding="utf-8")
    for independently_enforced_source in (
        '"stage": expected_stage',
        'os.environ.get("GPUWRF_M0_PARENT_LAUNCH_NS", "")',
        'payload["receipt_fingerprint"]',
        "authorization handoff must not be a symlink",
    ):
        assert independently_enforced_source in source


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


def _run_pallas_probe(
    root: Path, *, label: str, stage_identity: str = "W3"
) -> dict[str, object]:
    root = _require_synthetic(root, root)
    root.mkdir(parents=True, exist_ok=False)
    environment = _holder_environment(root, label=label)
    handoff = _write_parent_handoff(
        root,
        label=label,
        stage_identity=stage_identity,
        stage="gpt_critic_pallas_native",
    )
    result = _require_synthetic(root / "pallas-result.json", root)
    wrapper = textwrap.dedent(
        """
        import json
        import sys
        import pallas_sm120_spike as pallas

        def stop_after_real_authorization(args, authorization):
            raise RuntimeError(
                "GPT_CRITIC_POST_AUTH_SENTINEL:"
                + json.dumps(
                    {
                        "session_label": authorization["session_label"],
                        "window": authorization["window"],
                        "stage_identity": authorization["stage_identity"],
                    },
                    sort_keys=True,
                )
            )

        pallas.run_native = stop_after_real_authorization
        raise SystemExit(pallas.main())
        """
    )
    command = [
        sys.executable,
        "-c",
        wrapper,
        "--native",
        "--session-label",
        label,
        "--window",
        stage_identity,
        "--stage",
        "gpt_critic_pallas_native",
        "--run-id",
        STAGE_RUN_IDS[stage_identity],
        "--handoff",
        str(handoff),
        "--out",
        str(result),
    ]
    completed = subprocess.run(
        command,
        cwd=str(root),
        env={**environment, "PYTHONPATH": str(SCRIPTS)},
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
        "handoff_remaining": handoff.exists(),
        "handoff_spent": Path(f"{handoff}.spent").exists(),
    }


def test_pallas_native_handoff_accepts_a_coherent_cross_stage_label(tmp_path):
    """Gate 4: the real Pallas parser/handoff reaches a post-auth sentinel."""

    wrong_label = STAGE_LABELS["W1"]
    outcome = _run_pallas_probe(
        tmp_path / "pallas-cross-stage",
        label=wrong_label,
        stage_identity="W3",
    )
    payload = outcome["payload"]
    assert outcome["returncode"] == 1
    assert payload["status"] == "FAILED"
    assert "GPT_CRITIC_POST_AUTH_SENTINEL" in payload["toolchain_boundary"]
    assert payload["authorization"]["session_label"] == wrong_label
    assert payload["authorization"]["window"] == "W3"
    assert payload["authorization"]["stage_identity"] == "W3"
    assert outcome["handoff_spent"] is True
    assert outcome["handoff_remaining"] is False
    assert '"jax_imported"' not in outcome["stdout"]


def test_complete_reachable_authority_call_graph_has_one_unbound_join():
    """Independent call-graph check, without the candidate's AST vocabulary."""

    executor_source = (
        SCRIPTS / "m0_three_window_executor.py"
    ).read_text(encoding="utf-8")
    executor_tree = ast.parse(executor_source)
    protected_calls = []
    for node in ast.walk(executor_tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name not in {"check", "authorise"}:
            continue
        rendered = [ast.unparse(argument) for argument in node.args]
        rendered += [
            f"{keyword.arg}={ast.unparse(keyword.value)}"
            for keyword in node.keywords
        ]
        if any("SESSION_LABEL" in item for item in rendered):
            protected_calls.append({"name": name, "arguments": rendered})
    assert len(protected_calls) == 3

    launch_source = ast.get_source_segment(
        (SCRIPTS / "m0_window_parent.py").read_text(encoding="utf-8"),
        next(
            node
            for node in ast.walk(
                ast.parse(
                    (SCRIPTS / "m0_window_parent.py").read_text(encoding="utf-8")
                )
            )
            if isinstance(node, ast.FunctionDef)
            and node.name == "_launch_authorized_child"
        ),
    )
    assert launch_source is not None
    assert "session_label = session_label_of(authorization)" in launch_source
    assert "assert_stage_identity(" in launch_source
    assert "stage_configuration[\"window_label\"]" not in launch_source
    assert "authorization.get(\"window\")" not in launch_source
    assert "--session-label" in launch_source
    assert "--window" in launch_source


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
