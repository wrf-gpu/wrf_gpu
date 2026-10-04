#!/usr/bin/env python3
"""CPU-only proof builders for the v0234 V/V10 deterministic replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


SCHEMA_ARM = "gpuwrf.v0234.v10-pinned-replay-arm.v1"
EXPECTED_COUNTS = {"d01": 16, "d02": 16, "d03": 46}
MODEL_TREE = "5a6298fba90e76c3cafe674e1413d38efb694532"
OLD_PROOF = Path(
    ".agent/sprints/2026-07-18-v0234-deterministic-wake-closure-gpt/proof.json"
)


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


def load_self_hashed(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or symlinked proof: {path}")
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    if embedded != observed:
        raise RuntimeError(
            f"self hash mismatch for {path}: embedded={embedded} observed={observed}"
        )
    return payload, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "canonical_sha256": observed,
    }


def write_self_hashed(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    payload = dict(payload)
    payload["proof_sha256"] = canonical_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_arm(payload: Mapping[str, Any], *, mode: str) -> None:
    if (
        payload.get("schema") != SCHEMA_ARM
        or payload.get("mode") != mode
        or payload.get("src_gpuwrf_tree") != MODEL_TREE
        or payload.get("reference_or_pinned_scoped_horizon_complete") is not True
        or (payload.get("incremental_frame_pairs") or {}).get("counts") != EXPECTED_COUNTS
        or (payload.get("scope") or {}).get("d03_terminal_step") != 9000
        or (payload.get("scope") or {}).get("late_ni_claimed") is not False
        or (payload.get("scope") or {}).get("tolerance_changed") is not False
    ):
        raise RuntimeError(f"invalid {mode} arm semantics")
    strict = payload.get("strict_rmse") or {}
    if set(strict) != {"T", "U", "V", "W", "T2", "U10", "V10", "PSFC"}:
        raise RuntimeError(f"invalid strict field inventory for {mode}")
    if not all(math.isfinite(float(value)) for value in strict.values()):
        raise RuntimeError(f"nonfinite strict metric in {mode}")


def build_pin_manifest(
    *, reference_proof: Path, pin: Path, output: Path,
) -> dict[str, Any]:
    reference, reference_row = load_self_hashed(reference_proof)
    validate_arm(reference, mode="reference")
    if not pin.is_file() or pin.is_symlink() or pin.stat().st_size <= 0:
        raise RuntimeError(f"invalid reference pin: {pin}")
    payload = {
        "schema": "gpuwrf.v0234.v10-reference-pin-manifest.v1",
        "verdict": "V10_REFERENCE_PIN_AUTHENTICATED",
        "nonce": reference["nonce"],
        "model_tree": MODEL_TREE,
        "reference_run": {
            "namespace": reference["namespace"],
            "proof": reference_row,
            "verdict": reference["verdict"],
            "release_gate_green": reference["release_gate_green"],
            "strict_rmse": reference["strict_rmse"],
            "red_fields": reference["red_fields"],
            "output_counts": reference["incremental_frame_pairs"]["counts"],
            "d03_terminal_step": 9000,
        },
        "autotune_pin": {
            "path": str(pin.resolve()),
            "bytes": pin.stat().st_size,
            "file_sha256": sha256_file(pin),
        },
        "scoped_horizon_complete": True,
        "scientific_gate_reclassified": False,
        "late_ni_claimed": False,
        "full_18h_claimed": False,
    }
    write_self_hashed(output, payload)
    return payload


def _frame_rows(payload: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = (payload.get("incremental_frame_pairs") or {}).get("rows") or []
    mapped = {(str(row["domain"]), int(row["own_step"])): row for row in rows}
    if len(mapped) != sum(EXPECTED_COUNTS.values()):
        raise RuntimeError("duplicate or incomplete frame rows")
    return mapped


def _checkpoint_manifest(payload: Mapping[str, Any], step: str) -> Mapping[str, Any]:
    checkpoint = (payload.get("checkpoint_carries") or {}).get(step) or {}
    carry = checkpoint.get("carry") or {}
    manifest = carry.get("manifest") or {}
    if not manifest:
        raise RuntimeError(f"missing checkpoint manifest {step}")
    return manifest


def build_equality(
    *,
    reference_proof: Path,
    pinned_proof: Path,
    reference_pin: Path,
    pinned_pin: Path,
    output: Path,
) -> dict[str, Any]:
    reference, reference_row = load_self_hashed(reference_proof)
    pinned, pinned_row = load_self_hashed(pinned_proof)
    validate_arm(reference, mode="reference")
    validate_arm(pinned, mode="pinned")
    if reference["nonce"] != pinned["nonce"]:
        raise RuntimeError("reference/pinned nonce mismatch")
    ref_rows = _frame_rows(reference)
    pin_rows = _frame_rows(pinned)
    if set(ref_rows) != set(pin_rows):
        raise RuntimeError("reference/pinned frame key mismatch")
    frame_comparisons = []
    for key in sorted(ref_rows):
        left = ref_rows[key]
        right = pin_rows[key]
        equal = left.get("candidate_sha256") == right.get("candidate_sha256")
        frame_comparisons.append({
            "domain": key[0],
            "own_step": key[1],
            "reference_sha256": left.get("candidate_sha256"),
            "pinned_sha256": right.get("candidate_sha256"),
            "bit_equal": equal,
        })
    checkpoint_comparisons = {}
    for step in ("8800", "9000"):
        left = _checkpoint_manifest(reference, step)
        right = _checkpoint_manifest(pinned, step)
        left_hash = canonical_digest(left)
        right_hash = canonical_digest(right)
        checkpoint_comparisons[step] = {
            "reference_manifest_sha256": left_hash,
            "pinned_manifest_sha256": right_hash,
            "all_leaf_bytes_equal": left_hash == right_hash,
        }
    pins = []
    for role, path in (("reference", reference_pin), ("pinned", pinned_pin)):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise RuntimeError(f"invalid {role} pin: {path}")
        pins.append({
            "role": role,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
        })
    strict_exact = reference["strict_rmse"] == pinned["strict_rmse"]
    red_exact = reference["red_fields"] == pinned["red_fields"]
    frame_exact = all(row["bit_equal"] for row in frame_comparisons)
    carry_exact = all(
        row["all_leaf_bytes_equal"] for row in checkpoint_comparisons.values()
    )
    pin_exact = pins[0]["file_sha256"] == pins[1]["file_sha256"]
    passed = bool(frame_exact and carry_exact and pin_exact and strict_exact and red_exact)
    payload = {
        "schema": "gpuwrf.v0234.v10-pinned-equality.v1",
        "verdict": "V10_PINNED_BIT_EQUAL" if passed else "V10_PINNED_EQUALITY_FAILED",
        "passed": passed,
        "nonce": reference["nonce"],
        "reference": reference_row,
        "pinned": pinned_row,
        "frame_comparisons": frame_comparisons,
        "all_frames_bit_equal": frame_exact,
        "checkpoint_comparisons": checkpoint_comparisons,
        "all_checkpoint_leaves_bit_equal": carry_exact,
        "autotune_pins": pins,
        "autotune_dump_byte_equal": pin_exact,
        "strict_rmse_exact": strict_exact,
        "red_fields_exact": red_exact,
        "reference_release_gate_green": reference["release_gate_green"],
        "pinned_release_gate_green": pinned["release_gate_green"],
        "late_ni_claimed": False,
    }
    write_self_hashed(output, payload)
    return payload


def build_wake_analysis(*, reference_proof: Path, output: Path) -> dict[str, Any]:
    reference, reference_row = load_self_hashed(reference_proof)
    validate_arm(reference, mode="reference")
    decisive = reference["decisive_1500"]
    artifact_path = Path(decisive["artifact"])
    artifact, artifact_row = load_self_hashed(artifact_path)
    candidate_path = Path(artifact["candidate"]["path"])
    cpu_path = Path(artifact["cpu"]["path"])
    candidate_sha = artifact["candidate"]["sha256"]
    if sha256_file(candidate_path) != candidate_sha:
        raise RuntimeError("decisive candidate frame changed")

    import numpy as np
    from netCDF4 import Dataset

    from scripts import v0234_deterministic_wake_admission as admission
    from scripts import v0234_deterministic_wake_rca as rca

    authority, authority_row = admission.authenticate_authority()
    runtime = SimpleNamespace(np=np, Dataset=Dataset)
    fingerprint = admission.spatial_fingerprint(
        runtime,
        candidate_path=candidate_path,
        cpu_path=cpu_path,
        candidate_sha256=candidate_sha,
        authority=authority,
    )
    candidate, candidate_fields = rca.read_fields(candidate_path)
    cpu, cpu_fields = rca.read_fields(cpu_path)
    ratio = rca.ten_m_ratio_decomposition(candidate, cpu)
    momentum = rca.momentum_fingerprint(candidate, cpu)
    old, old_row = load_self_hashed(OLD_PROOF)
    old_metrics = (old.get("reference_replay") or {}).get(
        "strict_rmse_all_required_fields", {}
    )
    current_metrics = reference["strict_rmse"]
    metric_delta = {
        field: float(current_metrics[field]) - float(old_metrics[field])
        for field in current_metrics
    }
    trajectory = [
        {
            "step": int(row["own_step"]),
            "V": float(row["strict_rmse"]["V"]),
            "V10": float(row["strict_rmse"]["V10"]),
        }
        for row in reference["incremental_frame_pairs"]["rows"]
        if row["domain"] == "d03" and row.get("strict_rmse")
    ]
    payload = {
        "schema": "gpuwrf.v0234.v10-current-tree-wake-analysis.v1",
        "verdict": (
            "V10_CURRENT_TREE_GREEN_WAKE_MEASURED"
            if reference["release_gate_green"]
            else "V10_CURRENT_TREE_RED_WAKE_LOCALIZED"
        ),
        "reference": reference_row,
        "decisive_pair": artifact_row,
        "candidate_frame": candidate_fields,
        "cpu_frame": cpu_fields,
        "prior_authority": authority_row,
        "spatial_fingerprint": fingerprint,
        "ten_m_ratio_decomposition": ratio,
        "momentum_fingerprint": momentum,
        "old_reference_proof": old_row,
        "old_reference_strict_rmse": old_metrics,
        "current_reference_strict_rmse": current_metrics,
        "current_minus_old_strict_rmse": metric_delta,
        "trajectory_v_v10": trajectory,
        "release_gate_green": reference["release_gate_green"],
        "red_fields": reference["red_fields"],
        "late_ni_claimed": False,
    }
    write_self_hashed(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    pin = sub.add_parser("pin-manifest")
    pin.add_argument("--reference-proof", type=Path, required=True)
    pin.add_argument("--pin", type=Path, required=True)
    pin.add_argument("--output", type=Path, required=True)

    equality = sub.add_parser("equality")
    equality.add_argument("--reference-proof", type=Path, required=True)
    equality.add_argument("--pinned-proof", type=Path, required=True)
    equality.add_argument("--reference-pin", type=Path, required=True)
    equality.add_argument("--pinned-pin", type=Path, required=True)
    equality.add_argument("--output", type=Path, required=True)

    wake = sub.add_parser("wake")
    wake.add_argument("--reference-proof", type=Path, required=True)
    wake.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "pin-manifest":
        result = build_pin_manifest(
            reference_proof=args.reference_proof,
            pin=args.pin,
            output=args.output,
        )
    elif args.command == "equality":
        result = build_equality(
            reference_proof=args.reference_proof,
            pinned_proof=args.pinned_proof,
            reference_pin=args.reference_pin,
            pinned_pin=args.pinned_pin,
            output=args.output,
        )
    else:
        result = build_wake_analysis(
            reference_proof=args.reference_proof,
            output=args.output,
        )
    # Builders intentionally preserve their unsigned in-memory payload for
    # direct callers, while write_self_hashed signs the serialized copy.  Read
    # the proof back through the fail-closed verifier before reporting the
    # embedded digest so the CLI cannot fail after a successful write.
    result, _ = load_self_hashed(args.output)
    print(json.dumps({
        "verdict": result["verdict"],
        "proof": str(args.output.resolve()),
        "proof_sha256": result["proof_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
