#!/usr/bin/env python3
"""Seal the exact cold-arm AOT artifacts authorized for a warm nine-nest replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpt-ninenest-replay")


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


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-namespace", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase-audit", type=Path, required=True)
    parser.add_argument("--cold-authorization", type=Path, required=True)
    parser.add_argument("--cold-preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = args.source_namespace / "jax_cache"
    arm_wall_path = args.source_namespace / "monitor/arm_wall.json"
    phase_audit = json.loads(args.phase_audit.read_text())
    arm_wall = json.loads(arm_wall_path.read_text())
    cold_authorization = json.loads(args.cold_authorization.read_text())
    cold_preflight = json.loads(args.cold_preflight.read_text())

    tag_root = cache / "aot/0.23.3-jax0.10.0-jaxlib0.10.0-cuda_sm120"
    meta_paths = sorted((tag_root / "d01").glob("k_*.meta")) + sorted(
        (tag_root / "fused_d02").glob("k_*.meta")
    )
    rows = []
    errors: list[str] = []
    for meta_path in meta_paths:
        blob_path = meta_path.with_suffix(".xlaexec")
        if not blob_path.is_file():
            errors.append(f"missing blob for {meta_path}")
            continue
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)  # noqa: S301 - trusted, locally produced AOT metadata
        blob_digest = sha256(blob_path)
        try:
            meta_rel = meta_path.relative_to(cache).as_posix()
            blob_rel = blob_path.relative_to(cache).as_posix()
        except ValueError as exc:
            errors.append(str(exc))
            continue
        rows.extend(
            (
                {
                    "relative_path": meta_rel,
                    "kind": "aot_meta",
                    "bytes": meta_path.stat().st_size,
                    "sha256": sha256(meta_path),
                    "cheap_key": meta.cheap_key,
                    "hlo_sha256": meta.hlo_sha256,
                    "key_schema": meta.key_schema,
                },
                {
                    "relative_path": blob_rel,
                    "kind": "aot_executable",
                    "bytes": blob_path.stat().st_size,
                    "sha256": blob_digest,
                    "sha256_from_meta": meta.blob_sha256,
                    "blob_hash_matches_meta": blob_digest == meta.blob_sha256,
                    "cheap_key": meta.cheap_key,
                    "hlo_sha256": meta.hlo_sha256,
                    "key_schema": meta.key_schema,
                },
            )
        )

    source_tree = git("rev-parse", f"{args.source_head}:src/gpuwrf")
    current_tree = git("rev-parse", "HEAD:src/gpuwrf")
    current_dirty = git("status", "--porcelain", "--", "src/gpuwrf")
    checks = {
        "source_namespace_matches_cold_arm": arm_wall.get("namespace") == str(args.source_namespace),
        "cold_arm_nonce_matches_authorization": arm_wall.get("nonce") == cold_authorization.get("nonce"),
        "cold_arm_timed_out_rc137": arm_wall.get("returncode") == 137,
        "cold_arm_wall_is_exact_bound": 7260.0 <= float(arm_wall.get("command_wall_s", 0)) <= 7261.0,
        "cold_authorization_head_matches_source": cold_authorization.get("authorized_head") == args.source_head,
        "cold_preflight_head_matches_source": cold_preflight.get("authorized_head") == args.source_head,
        "phase_audit_pass": phase_audit.get("verdict") == "AOT_PHASE_AUDIT_PASS",
        "phase_audit_same_hlo": phase_audit.get("checks", {}).get("same_stablehlo_sha256") is True,
        "phase_audit_zero_changed_kept": phase_audit.get("checks", {}).get("zero_differing_avals_consumed") is True,
        "exactly_three_aot_metas": len(meta_paths) == 3,
        "exactly_six_seed_files": len(rows) == 6,
        "all_blob_hashes_match_authenticated_meta": all(
            row.get("blob_hash_matches_meta") is True
            for row in rows
            if row.get("kind") == "aot_executable"
        ),
        "source_gpuwrf_tree_matches_current": source_tree == current_tree,
        "current_gpuwrf_tree_clean": current_dirty == "",
    }
    verdict = "AOT_SEED_MANIFEST_PASS" if not errors and all(checks.values()) else "AOT_SEED_MANIFEST_FAIL"
    payload = {
        "schema": "wrfgpu2.v0234.ninenest-aot-seed-manifest.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "source_namespace": str(args.source_namespace),
        "source_cache": str(cache),
        "source_head": args.source_head,
        "source_gpuwrf_tree": source_tree,
        "current_head_at_seal": git("rev-parse", "HEAD"),
        "current_gpuwrf_tree": current_tree,
        "checks": checks,
        "errors": errors,
        "files": rows,
        "authorities": {
            "arm_wall": str(arm_wall_path),
            "arm_wall_sha256": sha256(arm_wall_path),
            "phase_audit": str(args.phase_audit),
            "phase_audit_sha256": sha256(args.phase_audit),
            "phase_audit_canonical_payload_sha256": phase_audit.get("canonical_payload_sha256"),
            "cold_authorization": str(args.cold_authorization),
            "cold_authorization_sha256": sha256(args.cold_authorization),
            "cold_preflight": str(args.cold_preflight),
            "cold_preflight_sha256": sha256(args.cold_preflight),
        },
        "copy_policy": "fresh writable destination cloned from six hash-sealed AOT files only",
        "cpu_only": True,
        "gpu_queries": 0,
    }
    payload["canonical_payload_sha256"] = canonical(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "out": str(args.out),
                "file_count": len(rows),
                "bytes": sum(int(row["bytes"]) for row in rows),
                "canonical_payload_sha256": payload["canonical_payload_sha256"],
            },
            indent=2,
        )
    )
    return 0 if verdict == "AOT_SEED_MANIFEST_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
