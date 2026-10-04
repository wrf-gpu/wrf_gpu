"""Device-denied falsification tests for cross-user kernel-lock evidence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m0_kernel_lock_authority as authority  # noqa: E402


def _line(identity: dict, *, pid: int = 4242, waiter: bool = False) -> str:
    arrow = "-> " if waiter else ""
    return (
        f"12: {arrow}FLOCK  ADVISORY  WRITE {pid} "
        f"{identity['device_major']:02x}:{identity['device_minor']:02x}:"
        f"{identity['inode']} 0 EOF"
    )


def _legacy_false_free() -> dict:
    return {
        "authoritative": False,
        "cannot_promote_vacancy": True,
        "holder_sidecar": {"text": ""},
        "fuser": {"stdout": "", "stderr": "", "returncode": 1},
        "lslocks": {"matching_lines": []},
    }


def _without_address(proof: dict) -> dict:
    return {key: value for key, value in proof.items() if key != "content_address"}


def test_exact_device_inode_record_blocks_even_when_every_legacy_view_is_empty(tmp_path):
    lock = tmp_path / "lock"
    lock.write_bytes(b"")
    identity = authority._stat_lock(lock)
    proof = authority.capture_kernel_lock_status(
        lock,
        proc_locks_text=_line(identity),
        legacy_observations=_legacy_false_free(),
    )
    assert proof["verdict"] == "BLOCKED_OCCUPIED", json.dumps(proof, indent=2)
    assert proof["matching_record_count"] == 1
    assert proof["matching_records"][0]["pid"] == 4242
    assert proof["legacy_cannot_change_verdict"] is True
    assert proof["permission_granted"] is False
    assert proof["content_address"] == authority.canonical_sha256(
        _without_address(proof)
    )


def test_waiter_is_also_fail_closed_occupied(tmp_path):
    lock = tmp_path / "lock"
    lock.touch()
    identity = authority._stat_lock(lock)
    proof = authority.capture_kernel_lock_status(
        lock, proc_locks_text=_line(identity, waiter=True)
    )
    assert proof["verdict"] == "BLOCKED_OCCUPIED"
    assert proof["matching_records"][0]["waiter"] is True


def test_zero_matching_records_is_vacant_but_never_permission(tmp_path):
    lock = tmp_path / "lock"
    lock.touch()
    proof = authority.capture_kernel_lock_status(
        lock, proc_locks_text="", legacy_observations=_legacy_false_free()
    )
    assert proof["verdict"] == "VACANT_UNAUTHORIZED"
    assert proof["matching_records"] == []
    assert proof["permission_granted"] is False


@pytest.mark.parametrize("field", ["device_major", "device_minor", "inode"])
def test_major_minor_and_inode_each_participate_in_identity(tmp_path, field):
    lock = tmp_path / "lock"
    lock.touch()
    identity = authority._stat_lock(lock)
    mutated = dict(identity)
    mutated[field] += 1
    proof = authority.capture_kernel_lock_status(
        lock, proc_locks_text=_line(mutated)
    )
    assert proof["verdict"] == "VACANT_UNAUTHORIZED"
    assert proof["matching_records"] == []


def test_malformed_line_naming_the_exact_key_is_unverifiable(tmp_path):
    lock = tmp_path / "lock"
    lock.touch()
    key = authority._stat_lock(lock)["proc_locks_key"]
    proof = authority.capture_kernel_lock_status(
        lock, proc_locks_text=f"malformed lock row {key} still names identity\n"
    )
    assert proof["verdict"] == "BLOCKED_UNVERIFIABLE"
    assert proof["malformed_matching_lines"]


def test_missing_nonregular_and_symlink_paths_fail_closed(tmp_path):
    missing = authority.capture_kernel_lock_status(
        tmp_path / "missing", proc_locks_text=""
    )
    assert missing["verdict"] == "BLOCKED_UNVERIFIABLE"

    directory = tmp_path / "directory"
    directory.mkdir()
    nonregular = authority.capture_kernel_lock_status(
        directory, proc_locks_text=""
    )
    assert nonregular["verdict"] == "BLOCKED_UNVERIFIABLE"

    real = tmp_path / "real"
    real.touch()
    link = tmp_path / "link"
    link.symlink_to(real)
    symlink = authority.capture_kernel_lock_status(link, proc_locks_text="")
    assert symlink["verdict"] == "BLOCKED_UNVERIFIABLE"


def test_identity_change_during_snapshot_is_unverifiable(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.touch()
    before = authority._stat_lock(lock)
    after = dict(before)
    after["inode"] += 1
    after["proc_locks_key"] = (
        f"{after['device_major']:02x}:{after['device_minor']:02x}:{after['inode']}"
    )
    observations = iter((before, after))
    monkeypatch.setattr(authority, "_stat_lock", lambda _path: next(observations))
    proof = authority.capture_kernel_lock_status(lock, proc_locks_text="")
    assert proof["verdict"] == "BLOCKED_UNVERIFIABLE"
    assert "changed" in " ".join(proof["reasons"])


def test_injected_snapshot_can_never_adjudicate_canonical_lock():
    proof = authority.capture_kernel_lock_status(
        authority.CANONICAL_LOCK, proc_locks_text=""
    )
    assert proof["verdict"] == "BLOCKED_UNVERIFIABLE"
    assert "injected" in " ".join(proof["reasons"])


def test_holder_provenance_binds_comm_uid_and_systemd_unit(tmp_path):
    lock = tmp_path / "lock"
    lock.touch()
    identity = authority._stat_lock(lock)
    proc_root = tmp_path / "proc"
    process = proc_root / "4242"
    process.mkdir(parents=True)
    (process / "comm").write_text("listener\n", encoding="utf-8")
    (process / "status").write_text(
        "Name:\tlistener\nUid:\t1001\t1001\t1001\t1001\n", encoding="utf-8"
    )
    (process / "cgroup").write_text(
        "0::/system.slice/alisios-nightly-compute-listener.service\n",
        encoding="utf-8",
    )
    proof = authority.capture_kernel_lock_status(
        lock,
        proc_locks_text=_line(identity),
        proc_root=proc_root,
    )
    provenance = proof["holder_provenance"][0]
    assert provenance["status"] == "BOUND"
    assert provenance["comm"] == "listener"
    assert provenance["uids"] == [1001, 1001, 1001, 1001]
    assert provenance["systemd_units"] == [
        "alisios-nightly-compute-listener.service",
        "system.slice",
    ]
    assert provenance["resource_job_receipt"] == "MISSING_NOT_SUPPLIED"
    assert proof["verdict"] == "BLOCKED_OCCUPIED"


def test_real_private_flock_is_detected_without_touching_canonical_lock(tmp_path):
    wrapper = REPO / "scripts/with_gpu_lock.sh"
    canonical_before = os.lstat(authority.CANONICAL_LOCK)
    wrapper_before = authority.sha256_file(wrapper)
    control = authority.private_flock_control(
        tmp_path / "control", wrapper_path=wrapper
    )
    canonical_after = os.lstat(authority.CANONICAL_LOCK)
    assert control["status"] == "PASS", json.dumps(control, indent=2)[:8000]
    assert control["detected_exact_child_pid"] is True
    assert control["occupied_snapshot"]["verdict"] == "BLOCKED_OCCUPIED"
    assert control["post_release_snapshot"]["verdict"] == "VACANT_UNAUTHORIZED"
    assert control["canonical_wrapper_unchanged"] is True
    assert authority.sha256_file(wrapper) == wrapper_before
    assert (
        canonical_before.st_dev,
        canonical_before.st_ino,
        canonical_before.st_size,
        canonical_before.st_mtime_ns,
    ) == (
        canonical_after.st_dev,
        canonical_after.st_ino,
        canonical_after.st_size,
        canonical_after.st_mtime_ns,
    )


def test_parser_rejects_partial_or_extra_fields():
    assert authority.parse_proc_lock_line("not a lock") is None
    assert authority.parse_proc_lock_line(
        "12: FLOCK ADVISORY WRITE 1 00:27:294 0 EOF extra"
    ) is None
