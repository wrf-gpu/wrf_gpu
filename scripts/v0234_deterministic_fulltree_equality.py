#!/usr/bin/env python3
"""Prove bit identity between the v0234 reference and pinned full-tree replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COUNTS = {"d01": 19, "d02": 19, "d03": 55}
REFERENCE_NAMESPACE = (
    "nested_stage_omega_transport_470e6111_deterministic_wake_reference1"
)
PINNED_NAMESPACE = (
    "nested_stage_omega_transport_470e6111_deterministic_wake_pinned1"
)
MODEL_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_self_hashed(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/symlinked proof: {path}")
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    if embedded != observed:
        raise RuntimeError(f"canonical proof mismatch: {path}")
    return payload, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "canonical_sha256": observed,
    }


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def compare_file(left: Path, right: Path, *, role: str) -> dict[str, Any]:
    for path in (left, right):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/symlinked equality artifact: {path}")
    left_hash = sha256_file(left)
    right_hash = sha256_file(right)
    return {
        "role": role,
        "reference": {"path": str(left.resolve()), "bytes": left.stat().st_size, "sha256": left_hash},
        "pinned": {"path": str(right.resolve()), "bytes": right.stat().st_size, "sha256": right_hash},
        "bit_identical": left_hash == right_hash and left.stat().st_size == right.stat().st_size,
    }


def output_inventory(run_dir: Path) -> dict[str, dict[str, Path]]:
    output = run_dir / "gpu-output"
    rows: dict[str, dict[str, Path]] = {}
    for domain in ("d01", "d02", "d03"):
        rows[domain] = {
            path.name: path for path in sorted(output.glob(f"wrfout_{domain}_*"))
        }
    counts = {domain: len(paths) for domain, paths in rows.items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"terminal frame inventory changed for {run_dir}: {counts}")
    return rows


def first_mismatch(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("bit_identical") is not True), None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--pinned-run-dir", type=Path, required=True)
    parser.add_argument("--pin-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_run_dir.resolve()
    pinned = args.pinned_run_dir.resolve()
    if reference.name != REFERENCE_NAMESPACE or pinned.name != PINNED_NAMESPACE:
        raise RuntimeError("unexpected deterministic replay namespaces")

    ref_terminal, ref_terminal_row = read_self_hashed(reference / "full-terminal-proof.json")
    pin_terminal, pin_terminal_row = read_self_hashed(pinned / "full-terminal-proof.json")
    ref_window, ref_window_row = read_self_hashed(
        Path(ref_terminal["window_proof"]["path"])
    )
    pin_window, pin_window_row = read_self_hashed(
        Path(pin_terminal["window_proof"]["path"])
    )
    pin_manifest, pin_manifest_row = read_self_hashed(args.pin_manifest.resolve())
    for terminal, mode in ((ref_terminal, "reference"), (pin_terminal, "pinned")):
        deterministic = terminal["authority"]["preimport"]["deterministic_autotune"]
        if (
            terminal.get("verdict")
            != "FULL_18H_LATE_NI_COMPLETE_KNOWN_V10_RED_REMAINS"
            or terminal.get("terminal_output_counts") != EXPECTED_COUNTS
            or terminal.get("all_finite_identity_pairs_passed") is not True
            or terminal.get("known_1500_v10_red_remains_release_blocker") is not True
            or deterministic.get("mode") != mode
        ):
            raise RuntimeError(f"{mode} terminal proof is not admissible")
    if (
        pin_manifest.get("verdict") != "REFERENCE_AUTOTUNE_PIN_AUTHENTICATED"
        or pin_manifest.get("model_tree") != MODEL_TREE
    ):
        raise RuntimeError("reference pin manifest authority changed")

    comparisons: list[dict[str, Any]] = []
    comparisons.append(compare_file(
        reference / "autotune-results.pb",
        pinned / "autotune-results.pb",
        role="complete_autotune_pin_regeneration",
    ))

    ref_outputs = output_inventory(reference)
    pin_outputs = output_inventory(pinned)
    for domain in ("d01", "d02", "d03"):
        if set(ref_outputs[domain]) != set(pin_outputs[domain]):
            raise RuntimeError(f"output filename inventory mismatch: {domain}")
        for name in sorted(ref_outputs[domain]):
            comparisons.append(compare_file(
                ref_outputs[domain][name], pin_outputs[domain][name],
                role=f"terminal_output/{domain}/{name}",
            ))

    for step in (8800, 9000):
        comparisons.append(compare_file(
            reference / "checkpoints" / f"authenticated-d03-step-{step}.pkl",
            pinned / "checkpoints" / f"authenticated-d03-step-{step}.pkl",
            role=f"checkpoint_carry/d03/{step}",
        ))

    ref_late = ref_window["window"]["retained_late_window"]
    pin_late = pin_window["window"]["retained_late_window"]
    if set(ref_late) != {"9313", "9314", "9405"} or set(pin_late) != set(ref_late):
        raise RuntimeError("late-window retained inventory changed")
    for step in ("9313", "9314", "9405"):
        comparisons.append(compare_file(
            Path(ref_late[step]["carry"]["path"]),
            Path(pin_late[step]["carry"]["path"]),
            role=f"late_window_carry/d03/{step}",
        ))
        comparisons.append(compare_file(
            Path(ref_late[step]["frame"]["path"]),
            Path(pin_late[step]["frame"]["path"]),
            role=f"late_window_frame/d03/{step}",
        ))

    for domain in ("d01", "d02", "d03"):
        comparisons.append(compare_file(
            Path(ref_terminal["terminal_carries"][domain]["path"]),
            Path(pin_terminal["terminal_carries"][domain]["path"]),
            role=f"terminal_carry/{domain}",
        ))

    mismatch = first_mismatch(comparisons)
    output_rows = [row for row in comparisons if row["role"].startswith("terminal_output/")]
    checkpoint = {
        step: all(
            row["bit_identical"]
            for row in comparisons
            if row["role"] == f"checkpoint_carry/d03/{step}"
        )
        for step in (8800, 9000)
    }
    late = {
        step: all(
            row["bit_identical"]
            for row in comparisons
            if row["role"].endswith(f"/d03/{step}")
        )
        for step in (9313, 9314, 9405)
    }
    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.deterministic-fulltree-equality.v1",
        "verdict": (
            "DETERMINISTIC_FULLTREE_BIT_IDENTITY_PROVED"
            if mismatch is None
            else "DETERMINISTIC_FULLTREE_IDENTITY_RED"
        ),
        "model_tree": MODEL_TREE,
        "reference_terminal_proof": ref_terminal_row,
        "pinned_terminal_proof": pin_terminal_row,
        "reference_window_proof": ref_window_row,
        "pinned_window_proof": pin_window_row,
        "reference_pin_manifest": pin_manifest_row,
        "autotune_pin_byte_identical": comparisons[0]["bit_identical"],
        "checkpoint_identity": checkpoint,
        "late_window_identity": late,
        "terminal_output_identity": {
            "counts": EXPECTED_COUNTS,
            "compared": len(output_rows),
            "all_bit_identical": all(row["bit_identical"] for row in output_rows),
        },
        "terminal_carry_identity": {
            domain: next(
                row["bit_identical"] for row in comparisons
                if row["role"] == f"terminal_carry/{domain}"
            )
            for domain in ("d01", "d02", "d03")
        },
        "comparisons": comparisons,
        "first_mismatch": mismatch,
        "deterministic_equality_passed": mismatch is None,
        "release_gate_green": False,
        "wake_release_blocker_remains": True,
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(args.output.resolve(), proof)
    print(json.dumps({
        "verdict": proof["verdict"],
        "output": str(args.output.resolve()),
        "file_sha256": sha256_file(args.output.resolve()),
        "proof_sha256": proof["proof_sha256"],
        "first_mismatch": mismatch,
    }, sort_keys=True))
    return 0 if mismatch is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
