#!/usr/bin/env python3
"""Seal the fail-closed disposition of the first AOT-seeded nine-nest arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "canonical_payload_sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    authorization = json.loads(args.authorization.read_text())
    preflight = json.loads(args.preflight.read_text())
    seed = json.loads(args.seed_manifest.read_text())
    arm_wall_path = args.namespace / "monitor/arm_wall.json"
    log_path = args.namespace / "monitor/run.stderr.log"
    receipt_path = args.namespace / "aot_seed_receipt.json"
    arm_wall = json.loads(arm_wall_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    log = log_path.read_text(errors="replace")
    match = re.search(r"domain=d01 loaded=false source=fallback:missing .*?/k_([0-9a-f]{64})\.xlaexec", log)
    requested_key = match.group(1) if match else None
    sealed_key = next(
        (
            row["cheap_key"]
            for row in seed["files"]
            if row["kind"] == "aot_meta" and "/d01/" in row["relative_path"]
        ),
        None,
    )
    destination_rows = []
    for row in seed["files"]:
        path = args.namespace / "jax_cache" / row["relative_path"]
        destination_rows.append(
            {
                "relative_path": row["relative_path"],
                "expected_sha256": row["sha256"],
                "observed_sha256": sha256(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
                "pass": path.is_file() and sha256(path) == row["sha256"],
            }
        )

    wrfouts = sorted(str(path) for path in (args.namespace / "gpu_output").glob("wrfout_d??_*") if path.is_file())
    proof_files = sorted(str(path) for path in (args.namespace / "pipeline_proofs").rglob("*") if path.is_file())
    aot_files = sorted(path for path in (args.namespace / "jax_cache/aot").rglob("*") if path.is_file())
    checks = {
        "authorization_matches_arm": authorization["nonce"] == arm_wall["nonce"]
        and authorization["namespace"] == arm_wall["namespace"],
        "preflight_passed_for_arm": preflight["verdict"] == "STATIC_PREFLIGHT_PASS"
        and preflight["nonce"] == arm_wall["nonce"],
        "seed_receipt_has_six_verified_files": len(receipt["files"]) == 6
        and receipt["all_source_hashes_verified"] is True
        and receipt["all_destination_hashes_verified"] is True,
        "all_six_destination_hashes_still_match": len(destination_rows) == 6
        and all(row["pass"] for row in destination_rows),
        "no_extra_aot_files_were_captured": len(aot_files) == 6,
        "nested_gpu_preflight_passed": "nested GPU preflight PASS" in log,
        "d01_missing_key_was_logged": requested_key is not None,
        "requested_key_differs_from_sealed_key": bool(requested_key and sealed_key and requested_key != sealed_key),
        "no_aot_capture_completed": "aot-captured" not in log,
        "no_model_output_landed": not wrfouts,
        "no_pipeline_proof_was_emitted": not proof_files,
        "stopped_before_timeout": arm_wall["returncode"] == 137
        and 0 < float(arm_wall["command_wall_s"]) < float(arm_wall["model_timeout_s"]),
    }
    verdict = "WARM_AOT_KEY_MISS_STOP_PASS" if all(checks.values()) else "WARM_AOT_KEY_MISS_STOP_FAIL"
    artifacts = (args.authorization, args.preflight, args.seed_manifest, arm_wall_path, log_path, receipt_path)
    payload = {
        "schema": "wrfgpu2.v0234.ninenest-warm-key-miss-audit.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": checks,
        "namespace": str(args.namespace),
        "nonce": arm_wall["nonce"],
        "returncode": arm_wall["returncode"],
        "command_wall_s": arm_wall["command_wall_s"],
        "sealed_d01_cheap_key": sealed_key,
        "requested_d01_cheap_key": requested_key,
        "wrfout_count": len(wrfouts),
        "pipeline_proof_file_count": len(proof_files),
        "aot_file_count": len(aot_files),
        "destination_seed_files": destination_rows,
        "artifacts": {str(path): sha256(path) for path in artifacts},
        "interpretation": (
            "The arm authenticated all six seed files and passed live GPU preflight, then was stopped after "
            "the first d01 lookup requested a different key. It produced no model output, pipeline proof, or "
            "replacement AOT artifact; it is a fail-closed harness disposition, not a science falsification."
        ),
    }
    payload["canonical_payload_sha256"] = canonical(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "canonical_payload_sha256": payload["canonical_payload_sha256"]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
