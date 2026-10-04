"""Receipt creation stays downstream of fresh exact kernel-vacancy evidence."""

from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m0_kernel_lock_authority as lock_authority  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402


BUILDER_PATH = REPO / (
    ".agent/sprints/2026-08-09-v0250-m0-r5-kernel-lock-authority/"
    "build_fresh_r5_coordination_receipt.py"
)
SPEC = importlib.util.spec_from_file_location("r5_receipt_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


HEAD = "a" * 40
PACKET_SHA = "b" * 64


def _write_manager_proof(tmp_path: Path, kernel: dict) -> tuple[dict, dict]:
    proof = {
        "schema": "wrf_gpu2.v025.m0.r5_manager_kernel_lock_preflight.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {"head": HEAD},
        "packet": {"actual_sha256": PACKET_SHA},
        "kernel_snapshot": kernel,
        "verdict": kernel["verdict"],
        "permission_granted": False,
        "receipt_created": False,
    }
    proof["content_address"] = builder.canonical_sha256(proof)
    path = tmp_path / "lock-proof.json"
    path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.write_text(f"{builder.sha256_file(path)}  {path.name}\n", encoding="utf-8")
    loaded, meta = builder.load_sidecar_json(
        path, schema="wrf_gpu2.v025.m0.r5_manager_kernel_lock_preflight.v1"
    )
    return loaded, meta


def _vacant_fixture(tmp_path: Path) -> tuple[Path, dict, dict, datetime]:
    lock = tmp_path / "private-vacant.lock"
    lock.touch()
    kernel = lock_authority.capture_kernel_lock_status(lock)
    assert kernel["verdict"] == "VACANT_UNAUTHORIZED"
    proof, meta = _write_manager_proof(tmp_path, kernel)
    captured = datetime.fromisoformat(kernel["captured_at_utc"]).astimezone(timezone.utc)
    return lock, proof, meta, captured


def _texts(meta: dict, lock_key: str) -> tuple[str, dict]:
    tokens = builder.binding_tokens(
        candidate_head=HEAD,
        packet_sha256=PACKET_SHA,
        proof_meta=meta,
        lock_key=lock_key,
    )
    text = "R5 " + " ".join(tokens)
    return text, {
        "0:2": {"verbatim": f"AFFIRMATIVE {text}"},
        "0:3": {"verbatim": f"AFFIRMATIVE {text}"},
    }


def _validate(
    lock: Path,
    proof: dict,
    meta: dict,
    captured: datetime,
    *,
    now_offset: float = 2.0,
    request_override: str | None = None,
    candidate_head: str = HEAD,
    packet_sha: str = PACKET_SHA,
) -> dict:
    key = proof["kernel_snapshot"]["lock_identity_before"]["proc_locks_key"]
    text, replies = _texts(meta, key)
    return builder.validate_lock_proof_for_receipt(
        proof,
        meta,
        expected_candidate_head=candidate_head,
        expected_packet_sha256=packet_sha,
        expected_lock_path=lock,
        release_received=captured - timedelta(seconds=1),
        requested=captured + timedelta(seconds=1),
        request_text=request_override if request_override is not None else text,
        replies=replies,
        now=captured + timedelta(seconds=now_offset),
    )


def _executor_binding(lock: Path) -> dict:
    kernel = lock_authority.capture_kernel_lock_status(lock)
    identity = kernel["lock_identity_before"]
    return {
        "kernel_lock_binding": {
            "preflight": {
                "status": "PASS",
                "kernel_identity": identity,
                "vacancy_grants_permission": False,
            },
            "immediate_live_recheck": {
                "status": "PASS",
                "identity": copy.deepcopy(identity),
                "vacancy_grants_permission": False,
            },
            "vacancy_grants_permission": False,
        }
    }


def test_fresh_content_bound_vacancy_is_accepted_but_grants_no_permission(tmp_path):
    lock, proof, meta, captured = _vacant_fixture(tmp_path)
    accepted = _validate(lock, proof, meta, captured)
    assert accepted["status"] == "PASS"
    assert accepted["vacancy_grants_permission"] is False
    assert accepted["kernel_identity"]["path"] == str(lock)
    assert accepted["age_seconds"] == pytest.approx(2.0)


def test_occupied_proof_refuses_before_any_receipt_path_is_created(tmp_path):
    control = lock_authority.private_flock_control(
        tmp_path / "control", wrapper_path=REPO / "scripts/with_gpu_lock.sh"
    )
    occupied = control["occupied_snapshot"]
    proof, meta = _write_manager_proof(tmp_path, occupied)
    captured = datetime.fromisoformat(occupied["captured_at_utc"]).astimezone(timezone.utc)
    output = tmp_path / "receipt.json"
    with pytest.raises(builder.ReceiptRefusal, match="not vacant"):
        _validate(Path(occupied["lock_identity_before"]["path"]), proof, meta, captured)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["stale", "candidate", "packet", "missing_token"])
def test_stale_or_misbound_proof_refuses_before_output(tmp_path, mutation):
    lock, proof, meta, captured = _vacant_fixture(tmp_path)
    output = tmp_path / "receipt.json"
    kwargs = {}
    if mutation == "stale":
        kwargs["now_offset"] = 121.0
    elif mutation == "candidate":
        kwargs["candidate_head"] = "c" * 40
    elif mutation == "packet":
        kwargs["packet_sha"] = "d" * 64
    else:
        kwargs["request_override"] = "request-without-bindings"
    with pytest.raises(builder.ReceiptRefusal):
        _validate(lock, proof, meta, captured, **kwargs)
    assert not output.exists()


def test_tampered_nested_content_address_refuses_before_output(tmp_path):
    lock, proof, meta, captured = _vacant_fixture(tmp_path)
    proof = copy.deepcopy(proof)
    proof["kernel_snapshot"]["content_address"] = "0" * 64
    output = tmp_path / "receipt.json"
    with pytest.raises(builder.ReceiptRefusal, match="nested"):
        _validate(lock, proof, meta, captured)
    assert not output.exists()


def test_immediate_occupied_or_wrong_inode_recheck_never_opens_output(tmp_path):
    lock, proof, meta, captured = _vacant_fixture(tmp_path)
    accepted = _validate(lock, proof, meta, captured)
    payload = {"kernel_lock_binding": {"immediate_live_recheck": None}}

    control = lock_authority.private_flock_control(
        tmp_path / "occupied", wrapper_path=REPO / "scripts/with_gpu_lock.sh"
    )
    output = tmp_path / "occupied-receipt.json"
    with pytest.raises(builder.ReceiptRefusal, match="live recheck"):
        builder.write_receipt_exclusive(
            output,
            copy.deepcopy(payload),
            proof_validation=accepted,
            live_snapshot=control["occupied_snapshot"],
        )
    assert not output.exists()

    other = tmp_path / "other.lock"
    other.touch()
    wrong_identity = lock_authority.capture_kernel_lock_status(other)
    output = tmp_path / "wrong-inode-receipt.json"
    with pytest.raises(builder.ReceiptRefusal, match="another lock identity"):
        builder.write_receipt_exclusive(
            output,
            copy.deepcopy(payload),
            proof_validation=accepted,
            live_snapshot=wrong_identity,
        )
    assert not output.exists()


def test_only_two_green_kernel_gates_allow_exclusive_write(tmp_path):
    lock, proof, meta, captured = _vacant_fixture(tmp_path)
    accepted = _validate(lock, proof, meta, captured)
    live = lock_authority.capture_kernel_lock_status(lock)
    payload = {
        "schema": "test-receipt",
        "kernel_lock_binding": {"immediate_live_recheck": None},
    }
    output = tmp_path / "receipt.json"
    digest = builder.write_receipt_exclusive(
        output,
        payload,
        proof_validation=accepted,
        live_snapshot=live,
    )
    assert output.is_file()
    assert digest == builder.sha256_file(output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["kernel_lock_binding"]["immediate_live_recheck"]["status"] == "PASS"
    assert written["kernel_lock_binding"]["immediate_live_recheck"]["vacancy_grants_permission"] is False


def test_packet_may_use_sidecar_without_a_packet_content_address(tmp_path):
    packet = {"schema": "wrf_gpu2.v025.m0.r5_launch_packet.v1", "status": "test"}
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet) + "\n", encoding="utf-8")
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.write_text(f"{builder.sha256_file(path)}  {path.name}\n", encoding="utf-8")
    loaded, meta = builder.load_sidecar_json(
        path,
        schema="wrf_gpu2.v025.m0.r5_launch_packet.v1",
        require_content_address=False,
    )
    assert loaded == packet
    assert meta["content_address"] is None


def test_held_executor_binds_receipt_to_actual_locked_fd(tmp_path, monkeypatch):
    lock = tmp_path / "held.lock"
    lock.touch()
    receipt = _executor_binding(lock)
    fd = os.open(lock, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", str(fd))
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FILE", str(lock.absolute()))
        result = executor._validate_r5_kernel_lock_receipt_binding(
            receipt,
            require_held_fd=True,
            expected_lock_path=lock,
        )
        assert result["held_fd_checked"] is True
        assert result["held_kernel_record_count"] >= 1
        assert result["inode"] == lock.stat().st_ino
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_held_executor_refuses_wrong_or_unlocked_fd_before_spend(
    tmp_path, monkeypatch
):
    expected = tmp_path / "expected.lock"
    wrong = tmp_path / "wrong.lock"
    expected.touch()
    wrong.touch()
    receipt = _executor_binding(expected)
    wrong_fd = os.open(wrong, os.O_RDWR | os.O_APPEND)
    try:
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", str(wrong_fd))
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FILE", str(expected.absolute()))
        with pytest.raises(
            executor.core_session.SessionRefusal, match="fd/path"
        ):
            executor._validate_r5_kernel_lock_receipt_binding(
                receipt,
                require_held_fd=True,
                expected_lock_path=expected,
            )
    finally:
        os.close(wrong_fd)

    expected_fd = os.open(expected, os.O_RDWR | os.O_APPEND)
    try:
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", str(expected_fd))
        with pytest.raises(
            executor.core_session.SessionRefusal, match="no exact kernel"
        ):
            executor._validate_r5_kernel_lock_receipt_binding(
                receipt,
                require_held_fd=True,
                expected_lock_path=expected,
            )
    finally:
        os.close(expected_fd)


def test_executor_refuses_missing_or_changed_receipt_identity(tmp_path):
    lock = tmp_path / "identity.lock"
    lock.touch()
    with pytest.raises(
        executor.core_session.SessionRefusal, match="lacks.*binding"
    ):
        executor._validate_r5_kernel_lock_receipt_binding(
            {}, require_held_fd=False, expected_lock_path=lock
        )

    receipt = _executor_binding(lock)
    receipt["kernel_lock_binding"]["immediate_live_recheck"]["identity"][
        "inode"
    ] += 1
    with pytest.raises(
        executor.core_session.SessionRefusal, match="identities differ"
    ):
        executor._validate_r5_kernel_lock_receipt_binding(
            receipt, require_held_fd=False, expected_lock_path=lock
        )
