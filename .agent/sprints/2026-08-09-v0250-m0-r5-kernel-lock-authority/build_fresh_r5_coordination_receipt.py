#!/usr/bin/env python3
"""Build R5's one receipt only after a fresh, bound kernel-vacancy proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SPRINT = Path(__file__).resolve().parent
PACKET = SPRINT / "M0_R5_LAUNCH_PACKET.json"
OUTPUT = SPRINT / "M0_CORE_W1_W2_W3_SESSION_RECEIPT_R5.json"
LEDGER = REPO / ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_spent.json"
WINDOW = "m0-core-w1-w2-w3-session"
EXPECTED_BRANCH = "manager/v0250-m0-r5-lock-authority"
LOCK_PROOF_MAX_AGE = timedelta(seconds=120)
BASE_REPLY_TOKENS = (
    "AFFIRMATIVE",
    WINDOW,
    "W1->C1->W2->W3",
    "4220",
    "4500",
    "exactly-one-receipt",
    "exactly-one-zero-wait-lock",
    "no-retry",
    "no-queue",
    "no-refund",
    "no-extra-arm",
    "preemption",
)


class ReceiptRefusal(RuntimeError):
    """A pre-receipt gate failed; the output path must stay absent."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str, role: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ReceiptRefusal(f"{role} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ReceiptRefusal(f"{role} lacks timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def read_exact(path: Path, role: str) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ReceiptRefusal(f"{role} is missing/not regular: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ReceiptRefusal(f"{role} is empty: {path}")
    return text


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout.strip()


def load_sidecar_json(
    path: Path, *, schema: str, require_content_address: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    if path.is_symlink() or not path.is_file() or sidecar.is_symlink() or not sidecar.is_file():
        raise ReceiptRefusal(f"proof/sidecar missing or not regular: {path}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    actual = sha256_file(path)
    if len(fields) != 2 or fields[0] != actual or fields[1] != path.name:
        raise ReceiptRefusal(f"sidecar does not bind exact file bytes/name: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReceiptRefusal(f"JSON proof is malformed: {path}") from exc
    if payload.get("schema") != schema:
        raise ReceiptRefusal(f"unexpected proof schema at {path}")
    supplied = payload.get("content_address")
    if require_content_address:
        expected = canonical_sha256(
            {key: value for key, value in payload.items() if key != "content_address"}
        )
        if supplied != expected:
            raise ReceiptRefusal(f"content address mismatch at {path}")
    return payload, {
        "path": str(path),
        "file_sha256": actual,
        "content_address": supplied,
    }


def binding_tokens(
    *, candidate_head: str, packet_sha256: str, proof_meta: dict[str, str], lock_key: str
) -> tuple[str, ...]:
    return (
        f"candidate={candidate_head}",
        f"packet-sha256={packet_sha256}",
        f"kernel-lock-content={proof_meta['content_address']}",
        f"kernel-lock-sha256={proof_meta['file_sha256']}",
        f"kernel-lock-key={lock_key}",
    )


def validate_lock_proof_for_receipt(
    proof: dict[str, Any],
    proof_meta: dict[str, str],
    *,
    expected_candidate_head: str,
    expected_packet_sha256: str,
    expected_lock_path: Path,
    release_received: datetime,
    requested: datetime,
    request_text: str,
    replies: dict[str, dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Validate every content/timeline binding before any receipt open."""

    kernel = proof.get("kernel_snapshot")
    if not isinstance(kernel, dict):
        raise ReceiptRefusal("manager lock proof lacks nested kernel snapshot")
    nested_address = kernel.get("content_address")
    if nested_address != canonical_sha256(
        {key: value for key, value in kernel.items() if key != "content_address"}
    ):
        raise ReceiptRefusal("nested kernel snapshot content address changed")
    if kernel.get("schema") != "wrf_gpu2.v025.m0.kernel_lock_snapshot.v1":
        raise ReceiptRefusal("nested kernel snapshot schema changed")
    if proof.get("verdict") != "VACANT_UNAUTHORIZED" or kernel.get("verdict") != "VACANT_UNAUTHORIZED":
        raise ReceiptRefusal("kernel lock proof is not vacant")
    if proof.get("permission_granted") is not False or kernel.get("permission_granted") is not False:
        raise ReceiptRefusal("vacancy proof incorrectly grants permission")
    if kernel.get("snapshot_injected_for_test") is not False:
        raise ReceiptRefusal("an injected kernel snapshot cannot authorize a receipt")
    if kernel.get("matching_record_count") != 0 or kernel.get("matching_records") != []:
        raise ReceiptRefusal("vacancy proof carries matching kernel records")
    if kernel.get("malformed_matching_lines") != []:
        raise ReceiptRefusal("vacancy proof carries malformed matching records")
    if kernel.get("canonical_lock_reacquire_or_probe") is not False:
        raise ReceiptRefusal("preflight claims a lock acquisition/probe")

    before = kernel.get("lock_identity_before") or {}
    after = kernel.get("lock_identity_after") or {}
    expected_path = str(Path(expected_lock_path))
    identity_fields = (
        "path",
        "mode",
        "device_decimal",
        "device_major",
        "device_minor",
        "inode",
        "proc_locks_key",
    )
    if before.get("path") != expected_path or any(
        before.get(field) != after.get(field) for field in identity_fields
    ):
        raise ReceiptRefusal("kernel proof path/identity is wrong or changed")
    if not before.get("is_regular") or before.get("is_symlink"):
        raise ReceiptRefusal("kernel proof lock object is not one regular file")
    if proof.get("candidate", {}).get("head") != expected_candidate_head:
        raise ReceiptRefusal("kernel proof binds another candidate HEAD")
    if proof.get("packet", {}).get("actual_sha256") != expected_packet_sha256:
        raise ReceiptRefusal("kernel proof binds another packet")

    captured = parse_time(kernel.get("captured_at_utc"), "kernel capture time")
    if captured < release_received:
        raise ReceiptRefusal("kernel snapshot predates exact release")
    if requested < captured:
        raise ReceiptRefusal("request predates the kernel proof it claims to bind")
    age = now - captured
    if not (timedelta(0) <= age <= LOCK_PROOF_MAX_AGE):
        raise ReceiptRefusal(f"kernel proof age {age} exceeds {LOCK_PROOF_MAX_AGE}")

    tokens = binding_tokens(
        candidate_head=expected_candidate_head,
        packet_sha256=expected_packet_sha256,
        proof_meta=proof_meta,
        lock_key=before["proc_locks_key"],
    )
    for role, text in {
        "request": request_text,
        "0:2 reply": replies["0:2"]["verbatim"],
        "0:3 reply": replies["0:3"]["verbatim"],
    }.items():
        missing = [token for token in tokens if token not in text]
        if missing:
            raise ReceiptRefusal(f"{role} does not bind lock proof: missing {missing}")
    return {
        "status": "PASS",
        "proof": proof_meta,
        "kernel_content_address": nested_address,
        "kernel_identity": before,
        "captured_at_utc": captured.isoformat(),
        "age_seconds": age.total_seconds(),
        "binding_tokens": list(tokens),
        "vacancy_grants_permission": False,
    }


def validate_live_recheck(
    live: dict[str, Any], *, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    if live.get("verdict") != "VACANT_UNAUTHORIZED":
        raise ReceiptRefusal(f"immediate live recheck is {live.get('verdict')}")
    if live.get("permission_granted") is not False or live.get("matching_record_count") != 0:
        raise ReceiptRefusal("immediate live recheck is not zero-record/unprivileged")
    before = live.get("lock_identity_before") or {}
    after = live.get("lock_identity_after") or {}
    fields = ("path", "device_decimal", "device_major", "device_minor", "inode", "proc_locks_key")
    if any(before.get(field) != expected_identity.get(field) for field in fields):
        raise ReceiptRefusal("immediate live recheck names another lock identity")
    if any(before.get(field) != after.get(field) for field in fields):
        raise ReceiptRefusal("lock identity changed during immediate live recheck")
    return {
        "status": "PASS",
        "content_address": live.get("content_address"),
        "captured_at_utc": live.get("captured_at_utc"),
        "identity": before,
        "vacancy_grants_permission": False,
    }


def write_receipt_exclusive(
    output: Path,
    payload: dict[str, Any],
    *,
    proof_validation: dict[str, Any],
    live_snapshot: dict[str, Any],
) -> str:
    """Perform the only output open, strictly after both kernel gates."""

    if proof_validation.get("status") != "PASS":
        raise ReceiptRefusal("content-addressed preflight was not accepted")
    live_validation = validate_live_recheck(
        live_snapshot, expected_identity=proof_validation["kernel_identity"]
    )
    output = Path(output)
    if os.path.lexists(output):
        raise ReceiptRefusal(f"receipt path is already occupied: {output}")
    payload["kernel_lock_binding"]["immediate_live_recheck"] = live_validation
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return sha256_file(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--principal-authority-file", type=Path, required=True)
    parser.add_argument("--principal-authority-at-utc", required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--release-received-at-utc", required=True)
    parser.add_argument("--lock-proof", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--requested-at-utc", required=True)
    parser.add_argument("--reply-0-2-file", type=Path, required=True)
    parser.add_argument("--reply-0-2-at-utc", required=True)
    parser.add_argument("--reply-0-3-file", type=Path, required=True)
    parser.add_argument("--reply-0-3-at-utc", required=True)
    args = parser.parse_args()

    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("exact device denial is required")
    if any(name.startswith("GPUWRF_GPU_LOCK_") for name in os.environ):
        raise SystemExit("receipt builder inherited canonical lock authority")
    if os.path.lexists(OUTPUT):
        raise SystemExit(f"fixed R5 receipt path is not vacant: {OUTPUT}")

    sys.path.insert(0, str(REPO / "scripts/v025"))
    import m0_core_session_protocol as core_session  # noqa: PLC0415
    import m0_kernel_lock_authority as lock_authority  # noqa: PLC0415
    import run_gpu_arm as gpu_auth  # noqa: PLC0415

    core_session.assert_accelerator_free()
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise SystemExit("R5 receipt builder is on the wrong branch")
    if git("diff", "--name-only", "--", "src/gpuwrf", "scripts/v025", "tests/v025"):
        raise SystemExit("candidate executable worktree is dirty")

    packet, packet_meta = load_sidecar_json(
        PACKET,
        schema="wrf_gpu2.v025.m0.r5_launch_packet.v1",
        require_content_address=False,
    )
    implementation = packet.get("checkout", {}).get("implementation_commit")
    if not implementation or git(
        "diff", "--name-only", f"{implementation}..HEAD", "--", "src/gpuwrf", "scripts/v025", "tests/v025"
    ):
        raise SystemExit("candidate executable tree differs from packet implementation")

    principal_text = read_exact(args.principal_authority_file, "fresh principal authority")
    release_text = read_exact(args.release_file, "terminal ALISIOS release")
    request_text = read_exact(args.request_file, "fresh R5 request")
    replies = {
        "0:2": {
            "affirmative": True,
            "verbatim": read_exact(args.reply_0_2_file, "0:2 reply"),
            "received_at_utc": args.reply_0_2_at_utc,
        },
        "0:3": {
            "affirmative": True,
            "verbatim": read_exact(args.reply_0_3_file, "0:3 reply"),
            "received_at_utc": args.reply_0_3_at_utc,
        },
    }
    if "AUTHORIZE_M0_R5_DEVICE" not in principal_text:
        raise SystemExit("fresh principal authority lacks AUTHORIZE_M0_R5_DEVICE")
    if "RELEASE_GPU" not in release_text:
        raise SystemExit("ALISIOS evidence lacks terminal RELEASE_GPU")
    for manager, reply in replies.items():
        missing = [token for token in (*BASE_REPLY_TOKENS, head) if token not in reply["verbatim"]]
        if missing:
            raise SystemExit(f"{manager} reply is incomplete: missing {missing}")

    principal_at = parse_time(args.principal_authority_at_utc, "principal authority time")
    release_at = parse_time(args.release_received_at_utc, "release receipt time")
    requested = parse_time(args.requested_at_utc, "request time")
    reply_times = {
        manager: parse_time(reply["received_at_utc"], f"{manager} reply time")
        for manager, reply in replies.items()
    }
    now = datetime.now(timezone.utc)
    if requested < max(principal_at, release_at):
        raise SystemExit("request predates fresh principal authority or terminal release")
    if any(stamp < requested for stamp in reply_times.values()):
        raise SystemExit("an affirmative predates the fresh request")
    if not (timedelta(0) <= now - requested <= gpu_auth.RECEIPT_MAX_AGE):
        raise SystemExit("request is future-dated or stale")

    lock_proof, lock_meta = load_sidecar_json(
        args.lock_proof,
        schema="wrf_gpu2.v025.m0.r5_manager_kernel_lock_preflight.v1",
    )
    lock_validation = validate_lock_proof_for_receipt(
        lock_proof,
        lock_meta,
        expected_candidate_head=head,
        expected_packet_sha256=packet_meta["file_sha256"],
        expected_lock_path=lock_authority.CANONICAL_LOCK,
        release_received=release_at,
        requested=requested,
        request_text=request_text,
        replies=replies,
        now=now,
    )

    payload = {
        "schema": "wrf_gpu2.v025.m0.gpu_coordination_receipt.v1",
        "window": WINDOW,
        "requested_at_utc": args.requested_at_utc,
        "request_text": request_text,
        "principal_authority": {
            "message": principal_text,
            "received_at_utc": args.principal_authority_at_utc,
            "file_sha256": sha256_file(args.principal_authority_file),
        },
        "release_message": release_text,
        "release_received_at_utc": args.release_received_at_utc,
        "replies": replies,
        "kernel_lock_binding": {
            "preflight": lock_validation,
            "immediate_live_recheck": None,
            "vacancy_grants_permission": False,
        },
        "candidate_head": head,
        "packet_sha256": packet_meta["file_sha256"],
        "spent": None,
    }
    shape = core_session.validate_receipt_shape(payload)
    payload["fingerprint"] = shape["fingerprint"]
    coordination = gpu_auth.CoordinationReceipt(
        window=payload["window"],
        requested_at_utc=payload["requested_at_utc"],
        replies=payload["replies"],
        request_text=payload["request_text"],
        spent=None,
        source_path=None,
    )
    coordination.check(WINDOW, now=now, ledger_path=LEDGER)
    if coordination.fingerprint() != payload["fingerprint"]:
        raise SystemExit("receipt validators disagree on fingerprint")

    live = lock_authority.capture_kernel_lock_status(lock_authority.CANONICAL_LOCK)
    receipt_sha = write_receipt_exclusive(
        OUTPUT,
        payload,
        proof_validation=lock_validation,
        live_snapshot=live,
    )
    loaded = gpu_auth.CoordinationReceipt.load(OUTPUT)
    loaded.check(WINDOW, now=now, ledger_path=LEDGER)
    if loaded.fingerprint() != payload["fingerprint"]:
        raise SystemExit("written receipt fingerprint changed")

    proof = {
        "schema": "wrf_gpu2.v025.m0.r5_manager_receipt_creation.v1",
        "created_at_utc": now.isoformat(),
        "candidate_head": head,
        "packet": packet_meta,
        "receipt_path": str(OUTPUT),
        "receipt_pre_spend_sha256": receipt_sha,
        "receipt_fingerprint": payload["fingerprint"],
        "kernel_lock_preflight": lock_validation,
        "kernel_lock_immediate_live_recheck": payload["kernel_lock_binding"]["immediate_live_recheck"],
        "ledger_path": str(LEDGER),
        "ledger_pre_spend_sha256": sha256_file(LEDGER),
        "status": "FRESH_R5_RECEIPT_VALID_UNSPENT",
        "device_action": False,
    }
    proof["content_address"] = canonical_sha256(proof)
    proof_path = SPRINT / f"MANAGER_R5_RECEIPT_{proof['content_address']}.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar = proof_path.with_name(f"{proof_path.name}.sha256")
    sidecar.write_text(f"{sha256_file(proof_path)}  {proof_path.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": proof["status"],
                "receipt": str(OUTPUT),
                "receipt_sha256": receipt_sha,
                "fingerprint": payload["fingerprint"],
                "proof": str(proof_path),
                "device_action": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReceiptRefusal as exc:
        raise SystemExit(f"R5 receipt refused before creation: {exc}") from exc
