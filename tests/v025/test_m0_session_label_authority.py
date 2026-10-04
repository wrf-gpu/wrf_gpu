"""CPU-only proofs that the session lock authority is never a stage identity.

This suite exists because of one committed live failure, not as a precaution.
The held M0-CORE device session recorded by ``bb1b1cd5`` acquired the canonical
zero-wait lock for label ``m0-core-w1-w2-w3-session``, spent its one receipt,
and then lost the entire window because the first reachable W1 child compared
that exported session label with the *stage identity* ``W1``:

    run_gpu_arm.WindowNotAuthorised: the exported canonical-lock label differs
    from the exact authorized label: 'm0-core-w1-w2-w3-session' != 'W1'

Every earlier CPU proof and the independent critic passed because the only
end-to-end session test replaced W1/C1/W2/W3 with stubs, so nothing ever drove
the real parent handoff into the real child CLI.  This suite therefore drives
the **real** transition -- the real ``_handoff_payload``, the real child
process, the real ``check_canonical_lock`` -- and mutates it.

Nothing here touches the GPU, the live holder file, the live launcher, any real
receipt, spend ledger, or canonical lock.  Every probe writes into a fresh
temporary root and mechanically refuses live paths before mutating anything.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpu_window_registry as registry  # noqa: E402
import m0_core_session_protocol as core_session  # noqa: E402
import m0_authority_guard_inventory as guard_inventory  # noqa: E402
import m0_exact_boundary_child as child  # noqa: E402
import m0_window_parent as window_parent  # noqa: E402
import run_gpu_arm  # noqa: E402


CHILD = SCRIPTS / "m0_exact_boundary_child.py"
SESSION_LABEL = core_session.SESSION_LABEL
HELD_MODE = registry.HELD_LAUNCH_MODE
LEGACY_MODE = registry.LEGACY_LAUNCH_MODE

#: The commit that recorded the live refusal under repair.
LIVE_DEFECT_COMMIT = "bb1b1cd5"
LIVE_SESSION_PROOF = (
    REPO / "proofs/v025/m0/autotune0_three_window/held_session/m0_core_session.json"
)
#: The verbatim message the live window died on.
LIVE_REFUSAL_MESSAGE = (
    "run_gpu_arm.WindowNotAuthorised: the exported canonical-lock label "
    "differs from the exact authorized label: "
    f"'{SESSION_LABEL}' != 'W1'"
)

#: Live paths no probe in this file may ever write to.
FORBIDDEN_ROOTS = (
    Path("/tmp/wrf_gpu2_gpu.lock"),
    Path("/tmp/wrf_gpu2_gpu.lock.holder"),
    Path("<DATA_ROOT>/wrf_gpu2"),
    Path("<USER_HOME>/src/wrf_gpu2"),
    Path("<USER_HOME>/src/wrf_gpu2_wt"),
    REPO,
)

DEVICE_FREE_ENV = {
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
}


class ProbeRefusal(RuntimeError):
    """The probe refused to touch a real holder/launcher/worktree path."""


def _require_synthetic(path: Path, root: Path) -> Path:
    """Mechanically refuse the real holder/launcher/worktree paths."""

    resolved = Path(path).resolve()
    synthetic_root = Path(root).resolve()
    if synthetic_root == Path(tempfile.gettempdir()).resolve():
        raise ProbeRefusal("the synthetic root must not be the shared temp root")
    for forbidden in FORBIDDEN_ROOTS:
        candidate = Path(forbidden)
        if resolved == candidate or candidate in resolved.parents:
            raise ProbeRefusal(f"refusing to touch live path {resolved}")
    if resolved != synthetic_root and synthetic_root not in resolved.parents:
        raise ProbeRefusal(f"{resolved} is outside the fresh root {synthetic_root}")
    return resolved


def _exited_pid() -> int:
    """A PID that has certainly never been an ancestor of a later child."""

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return int(process.pid)


def _stage_run_id(stage_identity: str) -> str:
    return registry.SESSION_STAGE_IDENTITIES[stage_identity]["run_id"]


def _drive_child(
    root: Path,
    *,
    launch_mode: str = HELD_MODE,
    cli_launch_mode: str | None = None,
    stage_identity: str = "W1",
    stage: str = "cold_empty_cache_readiness_1",
    session_label: str = SESSION_LABEL,
    cli_session_label: str | None = None,
    cli_stage_identity: str | None = None,
    run_id: str | None = None,
    cli_run_id: str | None = None,
    holder_label: str | None = None,
    holder_text: str | None = None,
    holder_token: str | None = None,
    exported_label: str | None = None,
    handoff_edit=None,
    environment_edit=None,
    handoff_shape: str = "json",
    root_read_only: bool = False,
    pre_consume: bool = False,
    child_path: Path | None = None,
) -> dict[str, object]:
    """Run the real child CLI over a real parent handoff in a fresh root.

    The handoff is produced by the production ``_handoff_payload`` so the test
    cannot drift from what the parent actually writes.  Mutations are applied
    to that real object rather than to a hand-rolled imitation.
    """

    root = Path(root)
    _require_synthetic(root, root)
    token = "gpuwrf-lock-synthetic-session-probe"
    holder = _require_synthetic(root / "synthetic.lock.holder", root)
    if holder_text is None:
        holder_text = (
            f"holder={holder_label or session_label} pid={os.getpid()} "
            f"since=2026-07-30T06:25:58Z token={holder_token or token} "
            "cmd=cpu-only-probe\n"
        )
    holder.write_text(holder_text, encoding="utf-8")

    run_id = run_id if run_id is not None else _stage_run_id(stage_identity)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GPUWRF_", "XLA_", "JAX_", "CUDA_"))
    }
    environment.update(DEVICE_FREE_ENV)
    environment.update(
        {
            "GPUWRF_GPU_LOCK_HELD": "1",
            "GPUWRF_GPU_LOCK_TOKEN": token,
            "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
            "GPUWRF_GPU_LOCK_LABEL": (
                session_label if exported_label is None else exported_label
            ),
            "GPUWRF_M0_PARENT_LAUNCH_NS": "1",
        }
    )
    if environment_edit is not None:
        environment_edit(environment)

    # The real control-plane producer, reading the same lock token the child
    # will re-check.  ``_handoff_payload`` consults os.environ directly.
    previous_token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN")
    os.environ["GPUWRF_GPU_LOCK_TOKEN"] = token
    try:
        handoff = window_parent._handoff_payload(
            session_label=window_parent.session_label_of({"label": session_label}),
            launch_mode=launch_mode,
            window_id=stage_identity,
            stage=stage,
            run_id=run_id,
            authorization={
                "receipt_fingerprint": "0" * 64,
                "receipt_spent_at_utc": "2026-07-30T06:25:58.736734+00:00",
            },
        )
    finally:
        if previous_token is None:
            os.environ.pop("GPUWRF_GPU_LOCK_TOKEN", None)
        else:
            os.environ["GPUWRF_GPU_LOCK_TOKEN"] = previous_token

    if handoff_edit is not None:
        handoff = handoff_edit(dict(handoff))
    handoff_path = _require_synthetic(root / "authorization.json", root)
    if handoff_shape == "json":
        handoff_path.write_text(
            json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif handoff_shape == "missing":
        pass
    elif handoff_shape == "directory":
        handoff_path.mkdir()
    elif handoff_shape == "malformed":
        handoff_path.write_text("{not-json", encoding="utf-8")
    elif handoff_shape == "nonobject":
        handoff_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    elif handoff_shape == "symlink":
        target = _require_synthetic(root / "authorization.target.json", root)
        target.write_text(
            json.dumps(handoff, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        handoff_path.symlink_to(target)
    else:
        raise ValueError(f"unknown handoff shape {handoff_shape!r}")
    if pre_consume:
        _require_synthetic(root / "authorization.json.spent", root).write_text(
            "already consumed", encoding="utf-8"
        )

    result_path = _require_synthetic(root / "exact_boundary.json", root)
    command = [
        sys.executable,
        str(child_path or CHILD),
        "--launch-mode",
        cli_launch_mode if cli_launch_mode is not None else launch_mode,
        "--session-label",
        cli_session_label if cli_session_label is not None else session_label,
        "--window",
        cli_stage_identity if cli_stage_identity is not None else stage_identity,
        "--stage",
        stage,
        "--run-id",
        cli_run_id if cli_run_id is not None else run_id,
        "--mode",
        "compile-only",
        "--handoff",
        str(handoff_path),
        "--result",
        str(result_path),
    ]
    if root_read_only:
        root.chmod(0o555)
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if root_read_only:
            root.chmod(0o755)
    payload = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
        "result_written": result_path.exists(),
        "handoff_remaining": handoff_path.exists(),
        "handoff_spent": (root / "authorization.json.spent").exists(),
    }


# --------------------------------------------------------------------------- #
# 1. the committed live failure                                                #
# --------------------------------------------------------------------------- #
def test_the_live_failure_is_imported_from_the_committed_session_proof():
    """The regression fixture is the real artifact, not a retyped string."""

    proof = json.loads(LIVE_SESSION_PROOF.read_text(encoding="utf-8"))
    failure = proof["first_failure"]["error"]
    assert proof["status"] == "BLOCKED"
    assert failure.count("m0_exact_boundary_child.py") >= 1
    assert "consume_authorization_handoff(" in failure
    assert "run_gpu_arm.check_canonical_lock(" in failure
    assert "expected_label=expected_window" in failure
    assert LIVE_REFUSAL_MESSAGE in failure
    assert proof["graph"]["suppressed"] == [
        core_session.C1_STAGE,
        "W2",
        "W3",
    ]


def test_the_committed_pre_repair_child_still_reproduces_the_live_refusal(tmp_path):
    """Extract the defect commit's own sources and re-run the exact transition.

    This does not depend on the working tree being pre-repair, so the
    reproduction stays re-runnable after the fix lands.
    """

    old_tree = tmp_path / "pre-repair"
    old_tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", LIVE_DEFECT_COMMIT, "scripts/v025"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(old_tree)], input=archive.stdout, check=True
    )
    old_child = old_tree / "scripts/v025/m0_exact_boundary_child.py"
    assert old_child.is_file()
    assert "expected_label=expected_window" in old_child.read_text(encoding="utf-8")

    root = tmp_path / "prerepair-root"
    root.mkdir()
    token = "gpuwrf-lock-synthetic-prerepair"
    holder = root / "synthetic.lock.holder"
    holder.write_text(
        f"holder={SESSION_LABEL} pid={os.getpid()} token={token} cmd=cpu-only\n",
        encoding="utf-8",
    )
    handoff = {
        "schema": "wrf_gpu2.v025.m0.authorized_child_handoff.v1",
        "window": "W1",
        "stage": "cold_empty_cache_readiness_1",
        "run_id": _stage_run_id("W1"),
        "parent_pid": os.getpid(),
        "receipt_fingerprint": "0" * 64,
        "receipt_spent_at_utc": "2026-07-30T06:25:58.736734+00:00",
        "lock_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "child_nonce": "prerepair-probe",
    }
    handoff_path = root / "authorization.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GPUWRF_", "XLA_", "JAX_", "CUDA_"))
    }
    environment.update(DEVICE_FREE_ENV)
    environment.update(
        {
            "GPUWRF_GPU_LOCK_HELD": "1",
            "GPUWRF_GPU_LOCK_TOKEN": token,
            "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
            "GPUWRF_GPU_LOCK_LABEL": SESSION_LABEL,
            "GPUWRF_M0_PARENT_LAUNCH_NS": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(old_child),
            "--window",
            "W1",
            "--stage",
            "cold_empty_cache_readiness_1",
            "--run-id",
            _stage_run_id("W1"),
            "--mode",
            "compile-only",
            "--handoff",
            str(handoff_path),
            "--result",
            str(root / "exact_boundary.json"),
        ],
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert LIVE_REFUSAL_MESSAGE in completed.stderr
    assert "expected_label=expected_window" in completed.stderr
    assert not (root / "exact_boundary.json").exists()


def test_the_repaired_child_cannot_be_invoked_the_pre_repair_way(tmp_path):
    """The defective command shape no longer parses at all."""

    completed = subprocess.run(
        [
            sys.executable,
            str(CHILD),
            "--window",
            "W1",
            "--stage",
            "cold_empty_cache_readiness_1",
            "--run-id",
            _stage_run_id("W1"),
            "--mode",
            "compile-only",
            "--handoff",
            str(tmp_path / "absent.json"),
            "--result",
            str(tmp_path / "must-not-exist.json"),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "--session-label" in completed.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


# --------------------------------------------------------------------------- #
# 2. the two namespaces                                                        #
# --------------------------------------------------------------------------- #
def test_coordination_labels_and_stage_identities_are_disjoint():
    assert set(registry.REGISTERED_WINDOWS) & set(
        registry.SESSION_STAGE_IDENTITIES
    ) == set()
    assert registry.STAGE_IDENTITIES == ("W1", "W2", "W3")
    assert SESSION_LABEL in registry.REGISTERED_WINDOWS


@pytest.mark.parametrize("stage_identity", ["W1", "W2", "W3"])
def test_a_stage_identity_can_never_authorize_a_lock(stage_identity):
    with pytest.raises(registry.WindowLabelNotRegistered, match="stage identity"):
        registry.assert_session_label(stage_identity, context="probe")


def test_the_session_label_can_never_be_used_as_a_stage_identity():
    with pytest.raises(
        registry.StageIdentityNotRegistered, match="coordination label"
    ):
        registry.assert_stage_identity(
            SESSION_LABEL, expected_run_id=_stage_run_id("W1"), context="probe"
        )


@pytest.mark.parametrize("stage_identity", ["W1", "W2", "W3"])
def test_each_stage_identity_is_bound_to_its_own_run_id(stage_identity):
    configuration = registry.assert_stage_identity(
        stage_identity,
        expected_run_id=_stage_run_id(stage_identity),
        context="probe",
    )
    assert configuration["run_id"] == _stage_run_id(stage_identity)
    other = next(s for s in registry.STAGE_IDENTITIES if s != stage_identity)
    with pytest.raises(registry.StageIdentityNotRegistered, match="bound to run ID"):
        registry.assert_stage_identity(
            stage_identity, expected_run_id=_stage_run_id(other), context="probe"
        )


def test_the_parent_window_table_agrees_with_the_frozen_stage_registry():
    consistency = window_parent.assert_stage_registry_consistency()
    assert consistency["stage_identities"] == ["W1", "W2", "W3"]
    for stage_identity, configuration in registry.SESSION_STAGE_IDENTITIES.items():
        assert window_parent.WINDOWS[stage_identity]["run_id"] == configuration[
            "run_id"
        ]
        assert window_parent.WINDOWS[stage_identity]["label"] == configuration[
            "window_label"
        ]


@pytest.mark.parametrize("stage_identity", ["W1", "W2", "W3"])
def test_the_parent_refuses_to_build_a_handoff_authorized_by_a_stage_name(
    stage_identity,
):
    """Defence in depth: the defective handoff is unbuildable, not just unused."""

    with pytest.raises(window_parent.WindowRefusal, match="stage identity"):
        window_parent.session_label_of({"label": stage_identity})


def test_parent_and_child_agree_on_the_handoff_schema():
    assert window_parent.HANDOFF_SCHEMA == child.HANDOFF_SCHEMA
    assert child.HANDOFF_SCHEMA.endswith(".v3")


# --------------------------------------------------------------------------- #
# 3. the session authority at every point of the reachable chain               #
# --------------------------------------------------------------------------- #
def test_the_outer_spender_and_held_executor_spend_only_the_session_label():
    """No spend/receipt/lock call in the control plane may name a stage."""

    executor_source = (SCRIPTS / "m0_three_window_executor.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(executor_source)
    spending_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = getattr(function, "attr", getattr(function, "id", ""))
        if name not in {"authorise", "check", "validate_spending_identity"}:
            continue
        arguments = list(node.args) + [kw.value for kw in node.keywords]
        for argument in arguments:
            text = ast.unparse(argument)
            if text.endswith("SESSION_LABEL"):
                spending_calls.append((name, text))
    assert spending_calls, "no receipt/authorise call was found to check"
    assert all(text.endswith("SESSION_LABEL") for _, text in spending_calls)
    for stage_identity in registry.STAGE_IDENTITIES:
        assert f'authorise("{stage_identity}"' not in executor_source
        assert f'check("{stage_identity}"' not in executor_source


def test_no_control_plane_call_passes_a_stage_identity_as_a_lock_label():
    """Repo-wide structural gate against the exact live confusion returning."""

    offenders: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name not in {
                "check_canonical_lock",
                "assert_lock_label",
                "assert_session_label",
                "validate_spending_identity",
                "authorise",
                "spend_receipt",
            }:
                continue
            for keyword in node.keywords:
                if keyword.arg not in {"expected_label", "expected", "window"}:
                    continue
                text = ast.unparse(keyword.value)
                if text.strip("\"'") in registry.STAGE_IDENTITIES:
                    offenders.append(f"{path.name}:{node.lineno} {name}({text})")
            for argument in node.args:
                text = ast.unparse(argument)
                if text.strip("\"'") in registry.STAGE_IDENTITIES:
                    offenders.append(f"{path.name}:{node.lineno} {name}({text})")
    assert offenders == []


def test_the_parent_command_carries_both_values_separately():
    """The launched command must name the session label and the stage."""

    source = (SCRIPTS / "m0_window_parent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_launch_authorized_child"
    )
    text = ast.unparse(launch)
    assert '"--session-label", session_label' in text.replace("\n", " ").replace(
        "'", '"'
    )
    assert '"--launch-mode", launch_mode' in text.replace("\n", " ").replace(
        "'", '"'
    )
    assert '"--window", window_id' in text.replace("\n", " ").replace("'", '"')
    assert "authorized_launch_context(" in text


@pytest.mark.parametrize("stage_identity", ["W1", "W2", "W3"])
def test_the_exact_session_label_passes_the_real_child_lock_check(
    tmp_path, stage_identity
):
    """Gate 4/5: drive the real W1/W2/W3 handoff path to the real child.

    A pass here means the child consumed the parent authorization and crossed
    ``check_canonical_lock`` with the session label.  It then stops for an
    unrelated, non-device reason (no WRF source authority is bound in a test),
    which keeps the probe accelerator-free while still exercising the exact
    transition that failed live.
    """

    root = tmp_path / f"root-{stage_identity}"
    root.mkdir()
    outcome = _drive_child(root, stage_identity=stage_identity)
    payload = outcome["payload"]
    assert payload is not None, outcome["stderr"]
    assert outcome["returncode"] == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["session_label"] == SESSION_LABEL
    assert payload["window"] == stage_identity
    assert payload["run_id"] == _stage_run_id(stage_identity)
    # The session-label gate was crossed: the single-use handoff was consumed.
    assert payload["handoff_consumed"] is True
    assert payload["reason"].startswith("WRF source authority:")
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert payload["device_touched"] is False
    assert outcome["handoff_spent"] is True
    assert outcome["handoff_remaining"] is False
    assert outcome["result_written"] is False
    # Every path the probe wrote stayed inside its fresh synthetic root.
    assert Path(outcome["command"][-1]).parent == root
    assert str(root) in outcome["command"][-3]


def test_the_probe_refuses_the_real_holder_and_launcher_paths(tmp_path):
    for forbidden in (
        Path("/tmp/wrf_gpu2_gpu.lock.holder"),
        REPO / "scripts/with_gpu_lock.sh",
        REPO / "proofs/v025/m0/autotune0_three_window/held_session/held_wrapper.log",
        Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw"),
    ):
        with pytest.raises(ProbeRefusal):
            _require_synthetic(forbidden, tmp_path)
    with pytest.raises(ProbeRefusal):
        _require_synthetic(tmp_path / "ok", Path(tempfile.gettempdir()))


# --------------------------------------------------------------------------- #
# 4. the complete held/legacy context matrix and Pallas consumer              #
# --------------------------------------------------------------------------- #
MATRIX_CASES = tuple(
    (launch_mode, stage_identity, label)
    for launch_mode in registry.LAUNCH_MODES
    for stage_identity in registry.STAGE_IDENTITIES
    for label in registry.WINDOW_LABELS
)


def _allowable_label(launch_mode: str, stage_identity: str) -> str:
    if launch_mode == HELD_MODE:
        return SESSION_LABEL
    return str(
        registry.SESSION_STAGE_IDENTITIES[stage_identity]["window_label"]
    )


@pytest.mark.parametrize(
    ("launch_mode", "stage_identity", "label"),
    MATRIX_CASES,
    ids=[
        f"{launch_mode}-{stage_identity}-{label}"
        for launch_mode, stage_identity, label in MATRIX_CASES
    ],
)
def test_real_parent_payload_into_real_child_has_exact_6_by_7_matrix(
    tmp_path, launch_mode, stage_identity, label
):
    """G2: six accepts and 36 relation refusals, through the real child CLI."""

    root = tmp_path / f"{launch_mode}-{stage_identity}-{label}"
    root.mkdir()
    outcome = _drive_child(
        root,
        launch_mode=launch_mode,
        stage_identity=stage_identity,
        session_label=label,
    )
    payload = outcome["payload"]
    assert outcome["returncode"] == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert payload["device_touched"] is False
    if label == _allowable_label(launch_mode, stage_identity):
        assert payload["reason"].startswith("WRF source authority:")
        assert payload["handoff_consumed"] is True
        assert outcome["handoff_spent"] is True
        assert outcome["handoff_remaining"] is False
    else:
        assert "allow coordination label" in payload["reason"]
        assert payload["handoff_consumed"] is False
        assert outcome["handoff_spent"] is False
        assert outcome["handoff_remaining"] is True


@pytest.mark.parametrize(
    ("launch_mode", "stage_identity", "label"),
    MATRIX_CASES,
    ids=[
        f"parent-{launch_mode}-{stage_identity}-{label}"
        for launch_mode, stage_identity, label in MATRIX_CASES
    ],
)
def test_parent_enforces_the_same_complete_launch_context_relation(
    launch_mode, stage_identity, label
):
    """G1 parent half: the launcher resolves one relation before payload write."""

    authorization = {
        "window": (
            "M0_CORE_SESSION" if launch_mode == HELD_MODE else stage_identity
        ),
        "label": label,
    }
    if label == _allowable_label(launch_mode, stage_identity):
        context = window_parent.authorized_launch_context(
            authorization,
            stage_identity=stage_identity,
            run_id=_stage_run_id(stage_identity),
        )
        assert context == {
            "launch_mode": launch_mode,
            "stage_identity": stage_identity,
            "run_id": _stage_run_id(stage_identity),
            "coordination_label": label,
            "allowable_coordination_label": label,
            "window_label": registry.SESSION_STAGE_IDENTITIES[stage_identity][
                "window_label"
            ],
        }
    else:
        with pytest.raises(window_parent.WindowRefusal, match="allow coordination"):
            window_parent.authorized_launch_context(
                authorization,
                stage_identity=stage_identity,
                run_id=_stage_run_id(stage_identity),
            )


def _drive_pallas_native_consumer(
    root: Path,
    *,
    label: str,
    stage_identity: str = "W3",
) -> dict[str, object]:
    """Drive the real Pallas native authorization consumer, stopping after it."""

    _require_synthetic(root, root)
    token = "gpuwrf-lock-synthetic-pallas-closure"
    holder = _require_synthetic(root / "synthetic.lock.holder", root)
    holder.write_text(
        f"holder={label} pid={os.getpid()} since=2026-07-30T06:25:58Z "
        f"token={token} cmd=cpu-only-pallas-closure\n",
        encoding="utf-8",
    )
    previous_token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN")
    os.environ["GPUWRF_GPU_LOCK_TOKEN"] = token
    try:
        handoff = window_parent._handoff_payload(
            session_label=label,
            launch_mode=HELD_MODE,
            window_id=stage_identity,
            stage="closure_pallas_native",
            run_id=_stage_run_id(stage_identity),
            authorization={
                "receipt_fingerprint": "9" * 64,
                "receipt_spent_at_utc": "2026-07-30T06:25:58.736734+00:00",
            },
        )
    finally:
        if previous_token is None:
            os.environ.pop("GPUWRF_GPU_LOCK_TOKEN", None)
        else:
            os.environ["GPUWRF_GPU_LOCK_TOKEN"] = previous_token
    handoff_path = _require_synthetic(root / "authorization.json", root)
    handoff_path.write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = _require_synthetic(root / "pallas-result.json", root)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GPUWRF_", "XLA_", "JAX_", "CUDA_"))
    }
    environment.update(DEVICE_FREE_ENV)
    environment.update(
        {
            "PYTHONPATH": str(SCRIPTS),
            "GPUWRF_GPU_LOCK_HELD": "1",
            "GPUWRF_GPU_LOCK_TOKEN": token,
            "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
            "GPUWRF_GPU_LOCK_LABEL": label,
            "GPUWRF_M0_PARENT_LAUNCH_NS": "1",
        }
    )
    wrapper = (
        "import json,sys; import pallas_sm120_spike as p; "
        "p.run_native=lambda args,authorization: "
        "(_ for _ in ()).throw(RuntimeError('CLOSURE_POST_AUTH_SENTINEL')); "
        "raise SystemExit(p.main())"
    )
    command = [
        sys.executable,
        "-c",
        wrapper,
        "--native",
        "--launch-mode",
        HELD_MODE,
        "--session-label",
        label,
        "--window",
        stage_identity,
        "--stage",
        "closure_pallas_native",
        "--run-id",
        _stage_run_id(stage_identity),
        "--handoff",
        str(handoff_path),
        "--out",
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
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": (
            json.loads(result.read_text(encoding="utf-8"))
            if result.exists()
            else json.loads(completed.stdout)
        ),
        "handoff_remaining": handoff_path.exists(),
        "handoff_spent": Path(f"{handoff_path}.spent").exists(),
    }


@pytest.mark.parametrize("label", registry.WINDOW_LABELS)
def test_pallas_native_consumer_has_one_correct_and_six_wrong_labels(
    tmp_path, label
):
    """G3: real Pallas native consumer applies the same held-context relation."""

    root = tmp_path / f"pallas-{label}"
    root.mkdir()
    outcome = _drive_pallas_native_consumer(root, label=label)
    payload = outcome["payload"]
    if label == SESSION_LABEL:
        assert outcome["returncode"] == 1
        assert payload["status"] == "FAILED"
        assert "CLOSURE_POST_AUTH_SENTINEL" in payload["toolchain_boundary"]
        assert outcome["handoff_spent"] is True
        assert outcome["handoff_remaining"] is False
    else:
        assert outcome["returncode"] == 2
        assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
        assert "allow coordination label" in payload["reason"]
        assert payload["handoff_consumed"] is False
        assert payload["jax_imported"] is False
        assert payload["gpuwrf_imported"] is False
        assert payload["device_touched"] is False
        assert outcome["handoff_spent"] is False
        assert outcome["handoff_remaining"] is True


# --------------------------------------------------------------------------- #
# 5. the mechanically derived hard gates and their mutations                   #
# --------------------------------------------------------------------------- #
def _extra_field(handoff):
    handoff["extra"] = "unrecognized"
    return handoff


def _drop_session_label(handoff):
    handoff.pop("session_label")
    return handoff


def _old_schema(handoff):
    handoff["schema"] = "wrf_gpu2.v025.m0.authorized_child_handoff.v2"
    return handoff


def _handoff_value(key, value):
    def edit(handoff):
        handoff[key] = value
        return handoff

    return edit


def _environment_drop(key):
    def edit(environment):
        environment.pop(key, None)

    return edit


def _environment_value(key, value):
    def edit(environment):
        environment[key] = value

    return edit


def _missing_holder(environment):
    environment["GPUWRF_GPU_LOCK_HOLDER_FILE"] += ".missing"


def _foreign_token_digest(handoff):
    handoff["lock_token_sha256"] = hashlib.sha256(b"someone-elses-token").hexdigest()
    return handoff


def _foreign_parent(handoff):
    handoff["parent_pid"] = _exited_pid()
    return handoff


#: (mutation name, kwargs for :func:`_drive_child`, the hard gate it must trip,
#: the exact refusal text that proves *which* gate fired).
MUTATIONS: tuple[tuple[str, dict, str, str], ...] = (
    (
        "stage_name_used_as_the_lock_label",
        {"cli_session_label": "W1"},
        "session_label_is_not_a_stage_identity",
        "is a held-session stage identity, not a coordination label",
    ),
    (
        "unknown_session_label",
        {"cli_session_label": "m0-core-w1-w2-w3-sessio"},
        "session_label_is_registered_coordination_label",
        "is not registered",
    ),
    (
        "coordination_label_used_as_the_stage_identity",
        {"cli_stage_identity": "m0-autotune0-qualification"},
        "stage_identity_is_not_a_coordination_label",
        "is a coordination label, not a stage identity",
    ),
    (
        "unknown_stage_identity",
        {"cli_stage_identity": "W9", "stage_identity": "W1"},
        "stage_identity_is_registered",
        "unknown stage identity",
    ),
    (
        "stage_identity_carries_another_stages_run_id",
        {"cli_run_id": _stage_run_id("W2")},
        "stage_identity_bound_to_run_id",
        "is bound to run ID",
    ),
    (
        "unknown_launch_mode",
        {"cli_launch_mode": "unknown-mode"},
        "launch_mode_is_registered",
        "unknown launch mode",
    ),
    (
        "held_mode_carries_a_wrong_registered_label",
        {
            "session_label": registry.SESSION_STAGE_IDENTITIES["W2"][
                "window_label"
            ]
        },
        "launch_context_allows_exact_coordination_label",
        "allow coordination label",
    ),
    (
        "handoff_is_a_symlink",
        {"handoff_shape": "symlink"},
        "handoff_is_not_a_symlink",
        "authorization handoff must not be a symlink",
    ),
    (
        "handoff_is_missing",
        {"handoff_shape": "missing"},
        "handoff_is_openable",
        "authorization handoff could not be opened/read",
    ),
    (
        "handoff_is_a_directory",
        {"handoff_shape": "directory"},
        "handoff_is_a_regular_file",
        "authorization handoff is not a regular file",
    ),
    (
        "handoff_is_malformed_json",
        {"handoff_shape": "malformed"},
        "handoff_is_valid_json",
        "authorization handoff is malformed",
    ),
    (
        "handoff_is_a_json_array",
        {"handoff_shape": "nonobject"},
        "handoff_is_a_json_object",
        "authorization handoff is not a JSON object",
    ),
    (
        "handoff_carries_an_extra_field",
        {"handoff_edit": _extra_field},
        "handoff_field_set_exact",
        "authorization handoff fields are incomplete or unrecognized",
    ),
    (
        "handoff_declares_the_v2_schema",
        {"handoff_edit": _old_schema},
        "handoff_schema_exact",
        "authorization handoff schema mismatch",
    ),
    (
        "handoff_names_a_different_launch_mode",
        {"handoff_edit": _handoff_value("launch_mode", LEGACY_MODE)},
        "launch_mode_matches_handoff",
        "authorization handoff launch_mode=",
    ),
    (
        "handoff_names_a_different_session_label",
        {"handoff_edit": _handoff_value("session_label", "baseline-census")},
        "session_label_matches_handoff",
        "authorization handoff session_label=",
    ),
    (
        "handoff_names_a_different_stage_identity",
        {"handoff_edit": _handoff_value("window", "W2")},
        "stage_identity_matches_handoff",
        "authorization handoff window=",
    ),
    (
        "handoff_names_a_different_stage_name",
        {"handoff_edit": _handoff_value("stage", "a-different-stage")},
        "stage_name_matches_handoff",
        "authorization handoff stage=",
    ),
    (
        "handoff_names_a_different_run_id",
        {"handoff_edit": _handoff_value("run_id", _stage_run_id("W2"))},
        "run_id_matches_handoff",
        "authorization handoff run_id=",
    ),
    (
        "handoff_parent_pid_is_not_an_integer",
        {"handoff_edit": _handoff_value("parent_pid", "not-an-integer")},
        "parent_pid_is_integer",
        "authorization parent_pid is invalid",
    ),
    (
        "handoff_creator_is_not_an_ancestor",
        {"handoff_edit": _foreign_parent},
        "parent_is_ancestor",
        "authorization creator is not an ancestor of this device child",
    ),
    (
        "parent_launch_endpoint_is_missing",
        {
            "environment_edit": _environment_drop(
                "GPUWRF_M0_PARENT_LAUNCH_NS"
            )
        },
        "parent_launch_endpoint_is_integer",
        "parent launch monotonic endpoint is missing/invalid",
    ),
    (
        "parent_launch_endpoint_is_zero",
        {
            "environment_edit": _environment_value(
                "GPUWRF_M0_PARENT_LAUNCH_NS", "0"
            )
        },
        "parent_launch_endpoint_is_positive",
        "authorization launch endpoint is invalid",
    ),
    (
        "receipt_fingerprint_is_empty",
        {"handoff_edit": _handoff_value("receipt_fingerprint", "   ")},
        "receipt_fingerprint_nonempty",
        "authorization receipt fingerprint is empty",
    ),
    (
        "receipt_spent_timestamp_is_not_iso8601",
        {"handoff_edit": _handoff_value("receipt_spent_at_utc", "not-a-time")},
        "receipt_spent_timestamp_is_iso8601",
        "authorization receipt spent timestamp is not ISO-8601",
    ),
    (
        "canonical_lock_wrapper_flag_is_missing",
        {
            "environment_edit": _environment_drop(
                "GPUWRF_GPU_LOCK_HELD"
            )
        },
        "canonical_lock_wrapper_is_held",
        "GPUWRF_GPU_LOCK_HELD is not set",
    ),
    (
        "canonical_lock_environment_has_no_token",
        {
            "environment_edit": _environment_drop(
                "GPUWRF_GPU_LOCK_TOKEN"
            )
        },
        "canonical_lock_environment_is_complete",
        "the lock environment is incomplete",
    ),
    (
        "canonical_lock_holder_file_is_missing",
        {"environment_edit": _missing_holder},
        "canonical_lock_holder_file_exists",
        "lock holder file",
    ),
    (
        "holder_record_fields_are_reordered",
        {
            "holder_text": (
                f"pid=1 holder={SESSION_LABEL} "
                "token=gpuwrf-lock-synthetic-session-probe cmd=cpu-only-probe\n"
            )
        },
        "holder_record_is_one_canonical_record",
        "the holder file is not one exact canonical-wrapper record",
    ),
    (
        "holder_record_carries_another_token",
        {"holder_token": "someone-elses-token"},
        "holder_record_token_equals_exported_token",
        "the holder file does not carry this process's lock token",
    ),
    (
        "exported_lock_label_is_a_different_window",
        {"exported_label": "baseline-census"},
        "exported_lock_label_equals_session_label",
        "the exported canonical-lock label differs from the exact authorized",
    ),
    (
        "holder_record_carries_another_label",
        {"holder_label": "baseline-census"},
        "holder_record_label_equals_session_label",
        "the holder record does not carry the exact authorized label",
    ),
    (
        "handoff_pins_another_lock_token",
        {"handoff_edit": _foreign_token_digest},
        "lock_token_sha256_matches_handoff",
        "authorization lock token does not match held lock",
    ),
    (
        "handoff_was_already_consumed",
        {"pre_consume": True},
        "handoff_consumed_exactly_once",
        "authorization handoff was already consumed",
    ),
    (
        "handoff_directory_becomes_read_only_before_consumption",
        {"root_read_only": True},
        "handoff_consumption_is_atomic",
        "could not atomically consume authorization handoff",
    ),
)


def test_derived_declared_and_mutated_guard_sets_have_empty_differences():
    """G4/G5: actual refusal sites, declaration, and real mutations coincide."""

    declared = set(child.SESSION_AUTHORITY_HARD_GATES)
    derived = set(guard_inventory.derive_pre_device_guard_inventory())
    covered = {gate for _, _, gate, _ in MUTATIONS}
    assert len(child.SESSION_AUTHORITY_HARD_GATES) == len(declared)
    assert derived - declared == set()
    assert declared - derived == set()
    assert covered - declared == set()
    assert declared - covered == set()
    assert len({name for name, _, _, _ in MUTATIONS}) == len(MUTATIONS)


@pytest.mark.parametrize(
    ("name", "kwargs", "gate", "expected"),
    MUTATIONS,
    ids=[mutation[0] for mutation in MUTATIONS],
)
def test_every_mutation_is_refused_before_any_device_import(
    tmp_path, name, kwargs, gate, expected
):
    root = tmp_path / name
    root.mkdir()
    outcome = _drive_child(root, **kwargs)
    payload = outcome["payload"]
    assert payload is not None, outcome["stderr"]
    assert outcome["returncode"] == 2, outcome["stdout"]
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert expected in payload["reason"], payload["reason"]
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert payload["device_touched"] is False
    assert outcome["result_written"] is False
    if gate != "handoff_consumed_exactly_once":
        # A refused authorization is never consumed, so a corrected retry is
        # still possible without new coordination.
        assert payload["handoff_consumed"] is False


def test_the_unmutated_probe_is_the_only_passing_case(tmp_path):
    """A mutation matrix is worthless if the baseline also refuses."""

    root = tmp_path / "baseline"
    root.mkdir()
    outcome = _drive_child(root)
    assert outcome["payload"]["handoff_consumed"] is True
    assert "canonical lock refused" not in outcome["payload"]["reason"]


# --------------------------------------------------------------------------- #
# 6. no live state was touched                                                 #
# --------------------------------------------------------------------------- #
def test_this_suite_holds_no_lock_and_names_no_live_receipt():
    # Checked over call sites, not raw text, so naming a live path in a
    # read-only freeze check does not read as using it.
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    invoked = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    for forbidden in (
        "with_gpu_lock",
        "spend_receipt",
        "nvidia-smi",
        "GPU_COORDINATION_RECEIPT",
        "authorise(",
    ):
        assert not any(forbidden in text for text in invoked), forbidden
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") is None
    assert run_gpu_arm.registry.SESSION_LABEL == SESSION_LABEL


def test_no_live_source_or_proof_artifact_was_modified():
    """Gate 10 plus the live-artifact freeze, checked mechanically."""

    frozen = (
        "src/gpuwrf",
        "proofs/v025/m0/autotune0_three_window",
        ".agent/decisions",
        ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_spent.json",
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", *frozen],
        capture_output=True,
        text=True,
        check=True,
    )
    assert dirty.stdout.strip() == "", dirty.stdout
