#!/usr/bin/env python3
"""Prepare, but never execute, the separate M0 profiled/clean matched pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/v025"))

import m0_vram_sampler as mvs  # noqa: E402
import m0_stale_pair as stale_pair  # noqa: E402
import m0_window_parent as exact_parent  # noqa: E402
import step1_driver as drv  # noqa: E402

CANONICAL_OUTPUT_DIR = Path(
    exact_parent.WINDOWS["W2"]["identity_path"]
).parent
CANONICAL_CACHE_SEED = Path(exact_parent.WINDOWS["W1"]["cache_path"])
CANONICAL_CACHE_SNAPSHOT_ROOT = exact_parent.PAIR_CACHE_ROOT
CANONICAL_QUALIFICATION = exact_parent.QUALIFICATION
CANONICAL_SOURCE_ROOT = REPO / "src/gpuwrf"
CANONICAL_RUN_DIR = exact_parent.FAST_RUN_DIR
CANONICAL_PROFILED_RUN_ID = str(exact_parent.WINDOWS["W2"]["run_id"])
CANONICAL_CLEAN_RUN_ID = str(exact_parent.WINDOWS["W3"]["run_id"])


def _held_session_callback(window_id: str, receipt: Path) -> list[str]:
    """Describe the already-held callback; never construct a second wrapper."""

    return [
        "HELD_SESSION_CALLBACK",
        "m0-core-w1-w2-w3-session",
        window_id,
        str(receipt),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to replace prepared pair artifact: {path}")
    mvs._atomic_json_no_replace(path, payload)


def _clone_cache(
    seed: Path,
    target: Path,
    *,
    seed_identity: dict[str, Any],
) -> dict[str, Any]:
    """Create an O(metadata) private directory over immutable seed files.

    The M0 child environment disables LRU-atime writes, the XLA autotune cache,
    and cache eviction.  A hit therefore reads an existing inode without
    modifying it; a miss creates a new file in that arm's private directory.
    Hard links avoid two 220 MiB payload copies on a filesystem that does not
    support reflinks, while the terminal re-hash still rejects any mutation.
    """

    if os.path.lexists(target):
        raise RuntimeError(f"prepared cache target already exists: {target}")
    completed = subprocess.run(
        ["cp", "--archive", "--link", str(seed), str(target)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "prepared cache requires same-filesystem hard-link snapshots so "
            "C1 cannot perform two full cache copies inside the held-lock budget: "
            f"{(completed.stdout + completed.stderr)[-1000:]}"
        )
    required_identity_fields = {
        "schema",
        "files",
        "directories",
        "bytes",
        "sha256",
    }
    if (
        set(seed_identity) != required_identity_fields
        or seed_identity.get("schema")
        != "wrf_gpu2.v025.m0.directory_tree_identity.v1"
    ):
        raise RuntimeError("prepared cache seed identity is incomplete")
    seed_files = {
        path.relative_to(seed): path
        for path in seed.rglob("*")
        if path.is_file()
    }
    target_files = {
        path.relative_to(target): path
        for path in target.rglob("*")
        if path.is_file()
    }
    seed_directories = {
        path.relative_to(seed)
        for path in seed.rglob("*")
        if path.is_dir()
    }
    target_directories = {
        path.relative_to(target)
        for path in target.rglob("*")
        if path.is_dir()
    }
    if (
        set(seed_files) != set(target_files)
        or seed_directories != target_directories
        or len(seed_files) != seed_identity["files"]
        or len(seed_directories) != seed_identity["directories"]
        or any(
            (
                seed_files[name].stat().st_dev,
                seed_files[name].stat().st_ino,
            )
            != (
                target_files[name].stat().st_dev,
                target_files[name].stat().st_ino,
            )
            for name in seed_files
        )
    ):
        raise RuntimeError(
            "prepared cache clone is not an exact hard-link snapshot"
        )
    # Every target payload is the already-hashed seed inode. Re-hashing the
    # profiled and clean names would read the same ~220 MiB bytes twice inside
    # C1 while adding no independent evidence.
    return dict(seed_identity)


def _require_frozen_manager_binding(
    *,
    output_dir: Path,
    profiled_run_id: str,
    clean_run_id: str,
    cache_seed: Path,
    cache_snapshot_root: Path,
    qualification_manifest: Path,
    source_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Reject a prepared pair that the fixed W2/W3 manager would not consume."""

    checks = {
        "output_dir": (
            Path(output_dir).resolve(strict=False)
            == Path(CANONICAL_OUTPUT_DIR).resolve(strict=False)
        ),
        "profiled_run_id": profiled_run_id == CANONICAL_PROFILED_RUN_ID,
        "clean_run_id": clean_run_id == CANONICAL_CLEAN_RUN_ID,
        "cache_seed": (
            Path(cache_seed).resolve(strict=False)
            == Path(CANONICAL_CACHE_SEED).resolve(strict=False)
        ),
        "cache_snapshot_root": (
            Path(cache_snapshot_root).resolve(strict=False)
            == Path(CANONICAL_CACHE_SNAPSHOT_ROOT).resolve(strict=False)
        ),
        "qualification_manifest": (
            Path(qualification_manifest).resolve(strict=False)
            == Path(CANONICAL_QUALIFICATION).resolve(strict=False)
        ),
        "source_root": (
            Path(source_root).resolve(strict=False)
            == Path(CANONICAL_SOURCE_ROOT).resolve(strict=False)
        ),
        "run_dir": (
            Path(run_dir).resolve(strict=False)
            == Path(CANONICAL_RUN_DIR).resolve(strict=False)
        ),
    }
    mismatches = sorted(name for name, matched in checks.items() if not matched)
    if mismatches:
        raise RuntimeError(
            "prepared pair does not bind the frozen W2/W3 manager graph: "
            + ", ".join(mismatches)
        )
    return {
        "status": "FROZEN_MANAGER_BINDING_CONFIRMED",
        "checks": checks,
        "profiled_window": "W2",
        "clean_window": "W3",
    }


def _qualified_cache_seed(
    path: Path,
    *,
    cache_seed: Path,
    cache_seed_identity: dict[str, Any],
) -> dict[str, Any]:
    """Require W1 qualification before creating either matched-arm snapshot."""

    if not path.is_file():
        raise RuntimeError(
            "autotune0 qualification manifest is missing; cache snapshots may not "
            "exist before successful W1 qualification"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid autotune0 qualification manifest: {exc}") from exc
    required = set(exact_parent.QUALIFICATION_FIELDS)
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError(
            "autotune0 qualification manifest fields are incomplete or unrecognised"
        )
    if (
        payload["schema"] != "wrf_gpu2.v025.m0.autotune0_qualification.v1"
        or payload["status"] != "AUTOTUNE0_QUALIFIED"
        or payload["run_id"] != exact_parent.WINDOWS["W1"]["run_id"]
        or payload["cache_seed_path"] != str(cache_seed)
        or payload["cache_seed_sha256"] != cache_seed_identity["sha256"]
        or payload["integration_clock"]
        != "synchronized-integration-only-excluding-compile-cache-load-and-io"
        or payload["selected_cold_stage"]
        != (
            "cold_empty_cache_readiness_"
            f"{payload['selected_cold_attempt_index']}"
        )
    ):
        raise RuntimeError(
            "autotune0 qualification does not bind this exact cache and timing scope"
        )
    fast_pair_path = Path(str(payload["fast_pair_path"]))
    if (
        not fast_pair_path.is_file()
        or not mvs.SHA256_PATTERN.fullmatch(str(payload["fast_pair_sha256"]))
        or _sha256(fast_pair_path) != payload["fast_pair_sha256"]
    ):
        raise RuntimeError(
            "autotune0 qualification FAST pair is missing or hash-mismatched"
        )
    exact_binding = payload["w1_exact_boundary"]
    if (
        not isinstance(exact_binding, dict)
        or exact_binding.get("run_id") != exact_parent.WINDOWS["W1"]["run_id"]
        or any(
            not mvs.SHA256_PATTERN.fullmatch(
                str(exact_binding.get(field, ""))
            )
            for field in (
                "cold_result_sha256",
                "cached_result_sha256",
                "gpu_result_sha256",
            )
        )
    ):
        raise RuntimeError(
            "autotune0 qualification exact W1 result binding is invalid"
        )
    completeness = payload["pair_completeness"]
    required_flags = {
        "fresh_cpu_arm_this_invocation",
        "fresh_gpu_arm_this_invocation",
        "comparator_result_present",
        "provenance_present",
        "arms_non_overlapping",
    }
    if (
        not isinstance(completeness, dict)
        or set(completeness) != required_flags | {"completeness_percent"}
        or completeness.get("completeness_percent") != 100
        or any(completeness.get(name) is not True for name in required_flags)
    ):
        raise RuntimeError("autotune0 qualification FAST pair is incomplete")
    limits = (
        ("cold_readiness_seconds", 600.0),
        ("cached_readiness_seconds", 60.0),
        ("warm_integration_seconds", 300.0),
    )
    for field, limit in limits:
        value = payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= limit
        ):
            raise RuntimeError(
                f"autotune0 qualification {field} exceeds its frozen {limit:g}s gate"
            )
    return payload


def prepare(
    *,
    output_dir: Path,
    device_uuid: str,
    profiled_run_id: str,
    clean_run_id: str,
    profiled_receipt: Path,
    clean_receipt: Path,
    cache_seed: Path,
    cache_snapshot_root: Path,
    qualification_manifest: Path,
    source_root: Path = REPO / "src/gpuwrf",
    run_dir: Path = Path(drv.FAST_CASE["run_dir"]),
    raw_root: Path = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/profiler-pair"),
) -> dict[str, Any]:
    """Write immutable identities and exact future commands without launching."""

    manager_binding = _require_frozen_manager_binding(
        output_dir=output_dir,
        profiled_run_id=profiled_run_id,
        clean_run_id=clean_run_id,
        cache_seed=cache_seed,
        cache_snapshot_root=cache_snapshot_root,
        qualification_manifest=qualification_manifest,
        source_root=source_root,
        run_dir=run_dir,
    )
    try:
        stale_pair.reject_stale_prepared_pair(run_id=profiled_run_id)
        stale_pair.reject_stale_prepared_pair(run_id=clean_run_id)
    except stale_pair.StalePairError as exc:
        raise RuntimeError(str(exc)) from exc
    if profiled_run_id == clean_run_id:
        raise RuntimeError("matched pair run IDs must be distinct")
    if output_dir.exists():
        raise RuntimeError(f"prepared pair directory already exists: {output_dir}")
    if cache_snapshot_root.exists():
        raise RuntimeError(
            f"prepared cache snapshot root already exists: {cache_snapshot_root}"
        )
    if not cache_seed.is_absolute() or not cache_snapshot_root.is_absolute():
        raise RuntimeError("cache seed and snapshot root must be absolute paths")
    cache_seed = cache_seed.resolve(strict=True)
    cache_snapshot_root = cache_snapshot_root.resolve(strict=False)
    if (
        cache_seed == cache_snapshot_root
        or cache_seed in cache_snapshot_root.parents
        or cache_snapshot_root in cache_seed.parents
    ):
        raise RuntimeError("cache seed and snapshot root must not overlap")

    profiled_identity = mvs.derive_identity(
        run_id=profiled_run_id,
        device_uuid=device_uuid,
        source_root=source_root,
        run_dir=run_dir,
    )
    # All expensive source/input/config hashes are independent of run_id.
    # Derive them once, then change only the clean arm's identity label.
    clean_identity = replace(profiled_identity, run_id=clean_run_id)
    comparable_fields = (
        "source_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "device_uuid",
        "workload_identity_sha256",
        "integration_scope_sha256",
        "event_mix_sha256",
    )
    mismatches = [
        field
        for field in comparable_fields
        if getattr(profiled_identity, field) != getattr(clean_identity, field)
    ]
    if mismatches:
        raise RuntimeError(f"prepared pair identity mismatch: {mismatches}")

    cache_seed_identity = mvs.directory_tree_identity(cache_seed)
    try:
        stale_pair.reject_stale_prepared_pair(
            prepared_cache_sha256=cache_seed_identity["sha256"]
        )
    except stale_pair.StalePairError as exc:
        raise RuntimeError(str(exc)) from exc
    qualification = _qualified_cache_seed(
        qualification_manifest,
        cache_seed=cache_seed,
        cache_seed_identity=cache_seed_identity,
    )
    cache_snapshot_root.mkdir(parents=True)
    profiled_cache = cache_snapshot_root / profiled_run_id
    clean_cache = cache_snapshot_root / clean_run_id
    profiled_cache_identity = _clone_cache(
        cache_seed,
        profiled_cache,
        seed_identity=cache_seed_identity,
    )
    clean_cache_identity = _clone_cache(
        cache_seed,
        clean_cache,
        seed_identity=cache_seed_identity,
    )
    if not (
        cache_seed_identity
        == profiled_cache_identity
        == clean_cache_identity
    ):
        raise RuntimeError("prepared cache snapshots are not byte-identical")

    output_dir.mkdir(parents=True)
    profiled_identity_path = output_dir / "profiled_identity.json"
    clean_identity_path = output_dir / "clean_identity.json"
    _atomic_json(
        profiled_identity_path,
        {"schema": mvs.IDENTITY_SCHEMA, **profiled_identity.binding()},
    )
    _atomic_json(
        clean_identity_path,
        {"schema": mvs.IDENTITY_SCHEMA, **clean_identity.binding()},
    )

    if Path(profiled_receipt) != Path(clean_receipt):
        raise RuntimeError(
            "Amendment-6 matched arms require the same single-use session receipt"
        )
    profiled_command = _held_session_callback("W2", profiled_receipt)
    clean_command = _held_session_callback("W3", clean_receipt)
    plan = {
        "schema": "wrf_gpu2.v025.m0.prepared_profiler_pair.v1",
        "status": "PREPARED_NOT_AUTHORISED_NOT_RUN",
        "profiler_gate_evidence_status": "MISSING",
        "device_touched": False,
        "device_uuid_source": (
            "caller-supplied expected physical UUID; external sampler must match it "
            "inside the separately coordinated window"
        ),
        "manager_binding": manager_binding,
        "held_session_execution": {
            "label": "m0-core-w1-w2-w3-session",
            "receipt_path": str(profiled_receipt),
            "receipt_spends": 1,
            "canonical_lock_acquisitions": 1,
            "callbacks": ["W2_PROFILED_CAPTURE", "W3_CLEAN_MATCHED_ARM"],
            "second_wrapper_reachable": False,
        },
        "frozen_arm_order": [profiled_run_id, clean_run_id],
        "identical_fields": {
            field: getattr(profiled_identity, field) for field in comparable_fields
        },
        "distinct_run_ids": True,
        "prepared_cache": {
            "status": "BYTE_IDENTICAL_SNAPSHOTS_FROM_QUALIFIED_AUTOTUNE0_CACHE",
            "seed": str(cache_seed),
            "seed_identity": cache_seed_identity,
            "qualification_manifest": str(qualification_manifest),
            "qualification_manifest_sha256": _sha256(qualification_manifest),
            "qualification": qualification,
            "profiled_snapshot": str(profiled_cache),
            "clean_snapshot": str(clean_cache),
            "profiled_identity": profiled_cache_identity,
            "clean_identity": clean_cache_identity,
            "private_directory_per_arm": True,
            "shared_seed_files": (
                "hard links; child cache policy makes existing entries "
                "read-only-on-hit and terminal hashing rejects mutation"
            ),
            "new_miss_files_private_by_directory": True,
            "ram_scaling": "O(1) streaming metadata walk; no cache payload retained",
            "disk_scaling": (
                "O(seed bytes + per-arm misses), not three full cache payloads"
            ),
        },
        "profiled": {
            "run_id": profiled_run_id,
            "instrumentation": "nsys",
            "evidence_hook": "sprint-local exact compiled boundary",
            "identity_path": str(profiled_identity_path),
            "identity_sha256": _sha256(profiled_identity_path),
            "raw_output": str(raw_root / profiled_run_id),
            "receipt_path": str(profiled_receipt),
            "prepared_cache_path": str(profiled_cache),
            "prepared_cache_sha256": cache_seed_identity["sha256"],
            "command": profiled_command,
            "command_shell": " ".join(shlex.quote(part) for part in profiled_command),
        },
        "clean": {
            "run_id": clean_run_id,
            "instrumentation": "none",
            "evidence_hook": (
                "no production hook; sprint-local exact compiled boundary with "
                "every GPUWRF_M0_* variable unset"
            ),
            "identity_path": str(clean_identity_path),
            "identity_sha256": _sha256(clean_identity_path),
            "raw_output": str(raw_root / clean_run_id),
            "receipt_path": str(clean_receipt),
            "prepared_cache_path": str(clean_cache),
            "prepared_cache_sha256": cache_seed_identity["sha256"],
            "command": clean_command,
            "command_shell": " ".join(shlex.quote(part) for part in clean_command),
        },
        "budget": drv.profiler_matched_pair_plan(),
        "execution_preconditions": [
            "independent GPT implementation review accepted",
            "one fresh affirmative session receipt",
            "outer owner acquired one exact zero-wait canonical session lock",
            "W2 and W3 are callbacks inside that already-held session",
            "identity derived before lock and revalidated before each callback",
            "each private cache snapshot re-hashed before its callback",
            "LRU atime writes/eviction and XLA autotune-cache writes disabled",
            "cached executable readiness/hit must be proved before integration timing",
            "profiled arm first; clean arm starts only after W2 succeeds",
        ],
        "timing_boundary": {
            "profiled_observable": (
                "parent launch -> lower().compile() readiness, then exact compiled "
                "invocation + block_until_ready under the forecast NVTX range"
            ),
            "clean_observable": (
                "parent launch -> lower().compile() readiness, then the same exact "
                "compiled invocation + block_until_ready with no range/allocator"
            ),
            "known_gap": None,
            "implementation":
                "scripts/v025/m0_exact_boundary_child.py",
            "phase_subtraction": False,
        },
        "prohibitions": [
            "do not execute from this CPU-only sprint",
            "do not acquire a second lock or spend a second receipt",
            "do not reuse either run directory",
            "do not label first-call wall time integration-only",
            "do not use profiled timing for clean runtime gates",
        ],
    }
    _atomic_json(output_dir / "plan.json", plan)
    return plan


def inspection_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Describe preparation without reading, hashing, creating, or consuming."""

    return {
        "schema": "wrf_gpu2.v025.m0.prepared_profiler_pair_inspection.v1",
        "status": "INSPECTION_ONLY_NO_FILES_READ_OR_WRITTEN",
        "device_touched": False,
        "receipts_consumed": [],
        "inputs": {
            "output_dir": str(args.output_dir) if args.output_dir else None,
            "profiled_run_id": args.profiled_run_id,
            "clean_run_id": args.clean_run_id,
            "cache_seed": str(args.cache_seed) if args.cache_seed else None,
            "cache_snapshot_root": (
                str(args.cache_snapshot_root) if args.cache_snapshot_root else None
            ),
            "qualification_manifest": (
                str(args.qualification_manifest)
                if args.qualification_manifest
                else None
            ),
        },
        "would_require_before_any_snapshot": {
            "qualification_status": "AUTOTUNE0_QUALIFIED",
            "cold_readiness_max_seconds": 600.0,
            "cached_readiness_max_seconds": 60.0,
            "warm_integration_max_seconds": 300.0,
            "cache_seed_content_hash_match": True,
        },
        "future_order": ["profiled", "clean"],
        "note": (
            "This mode deliberately performs no stat/read/hash/mkdir/copy and does "
            "not inspect or consume coordination receipts."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device-uuid")
    parser.add_argument("--profiled-run-id")
    parser.add_argument("--clean-run-id")
    parser.add_argument("--profiled-receipt", type=Path)
    parser.add_argument("--clean-receipt", type=Path)
    parser.add_argument("--cache-seed", type=Path)
    parser.add_argument("--cache-snapshot-root", type=Path)
    parser.add_argument("--qualification-manifest", type=Path)
    parser.add_argument("--source-root", type=Path, default=REPO / "src/gpuwrf")
    parser.add_argument("--run-dir", type=Path, default=Path(drv.FAST_CASE["run_dir"]))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/profiler-pair"),
    )
    args = parser.parse_args()
    if args.print_plan:
        print(json.dumps(inspection_plan(args), indent=2, sort_keys=True))
        return 0
    required = (
        "output_dir",
        "device_uuid",
        "profiled_run_id",
        "clean_run_id",
        "profiled_receipt",
        "clean_receipt",
        "cache_seed",
        "cache_snapshot_root",
        "qualification_manifest",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("preparation requires: " + ", ".join(f"--{x.replace('_', '-')}" for x in missing))
    plan = prepare(
        output_dir=args.output_dir,
        device_uuid=args.device_uuid,
        profiled_run_id=args.profiled_run_id,
        clean_run_id=args.clean_run_id,
        profiled_receipt=args.profiled_receipt,
        clean_receipt=args.clean_receipt,
        cache_seed=args.cache_seed,
        cache_snapshot_root=args.cache_snapshot_root,
        qualification_manifest=args.qualification_manifest,
        source_root=args.source_root,
        run_dir=args.run_dir,
        raw_root=args.raw_root,
    )
    print(json.dumps({
        "status": plan["status"],
        "profiler_gate_evidence_status": plan["profiler_gate_evidence_status"],
        "plan": str(args.output_dir / "plan.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
