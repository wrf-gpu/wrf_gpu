#!/usr/bin/env python3
"""Authenticate the complete v0234 reference replay and its XLA autotune pin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
EXPECTED_COUNTS = {"d01": 19, "d02": 19, "d03": 55}
REFERENCE_NAMESPACE = (
    "nested_stage_omega_transport_470e6111_deterministic_wake_reference1"
)
AUTOTUNE_STAGE_NAME = ".v0234-deterministic-wake-autotune-v1"


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


def file_row(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/symlinked artifact: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.reference_run_dir.resolve()
    output = args.output.resolve()
    if run_dir.name != REFERENCE_NAMESPACE or not run_dir.is_dir() or run_dir.is_symlink():
        raise RuntimeError(f"invalid reference namespace: {run_dir}")

    terminal, terminal_row = read_self_hashed(run_dir / "full-terminal-proof.json")
    window_path = Path(str((terminal.get("window_proof") or {}).get("path", "")))
    window, window_row = read_self_hashed(window_path)
    if (
        terminal.get("verdict")
        != "FULL_18H_LATE_NI_COMPLETE_KNOWN_V10_RED_REMAINS"
        or terminal.get("terminal_output_counts") != EXPECTED_COUNTS
        or terminal.get("terminal_own_steps")
        != {"d01": 1200, "d02": 3600, "d03": 10800}
        or terminal.get("all_finite_identity_pairs_passed") is not True
        or terminal.get("known_1500_v10_red_remains_release_blocker") is not True
        or terminal.get("model_or_numerical_edit") is not False
        or window.get("status") != "WINDOW_COMPLETE"
        or window.get("known_1500_v10_red_remains") is not True
    ):
        raise RuntimeError("reference terminal/window semantics are not admissible")
    deterministic = (
        terminal.get("authority", {}).get("preimport", {}).get("deterministic_autotune", {})
    )
    pin = run_dir / "autotune-results.pb"
    staged_pin = run_dir.parent / AUTOTUNE_STAGE_NAME / "reference-autotune-results.pb"
    expected_flags = f"--xla_gpu_dump_autotune_results_to={staged_pin.resolve()}"
    if (
        deterministic.get("mode") != "reference"
        or deterministic.get("xla_flags") != expected_flags
        or deterministic.get("output_pin") != str(staged_pin.resolve())
        or deterministic.get("persistent_compilation_cache_used") is not False
        or staged_pin.exists()
        or staged_pin.is_symlink()
    ):
        raise RuntimeError("reference XLA dump authority changed")

    decisive = window.get("decisive_1500") or {}
    observation = decisive.get("known_1500_v10_record") or {}
    if (
        observation.get("passed") is not True
        or observation.get("classification")
        != "KNOWN_WAKE_RELEASE_BLOCKER_METRIC_AND_FINGERPRINT_MATCH"
        or observation.get("release_gate_green") is not False
        or observation.get("waiver_or_reclassification") is not False
        or observation.get("isolation_decision_only") is not True
    ):
        raise RuntimeError("reference 15:00 classifier did not match the frozen signature")
    retained_late = (window.get("window") or {}).get("retained_late_window") or {}
    if set(retained_late) != {"9313", "9314", "9405"} or not all(
        row.get("passed") is True for row in retained_late.values()
    ):
        raise RuntimeError("reference late-Ni retained window is incomplete/red")

    output_counts = {
        domain: len(list((run_dir / "gpu-output").glob(f"wrfout_{domain}_*")))
        for domain in ("d01", "d02", "d03")
    }
    if output_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"reference output inventory changed: {output_counts}")
    terminal_carries = terminal.get("terminal_carries") or {}
    if set(terminal_carries) != {"d01", "d02", "d03"}:
        raise RuntimeError("reference terminal carry inventory is incomplete")

    lowered = sorted(
        path for path in run_dir.rglob("*lowered-hlo*.json")
        if path.is_file() and not path.is_symlink()
    )
    compiled_programs = {
        "lowered_hlo_artifacts": [file_row(path) for path in lowered],
        "ordinary_one_step_program": (window.get("ordinary_one_step_program") or {}),
        "health_program": (window.get("health_program") or {}),
    }
    if (
        not compiled_programs["lowered_hlo_artifacts"]
        or compiled_programs["ordinary_one_step_program"].get("compile_calls") != 1
        or compiled_programs["health_program"].get("compile_calls") != 1
    ):
        raise RuntimeError("reference compiled-program inventory is incomplete")

    manifest: dict[str, Any] = {
        "schema": "gpuwrf.v0234.deterministic-reference-pin-manifest.v1",
        "verdict": "REFERENCE_AUTOTUNE_PIN_AUTHENTICATED",
        "model_tree": MODEL_TREE,
        "reference_run": {
            "run_dir": str(run_dir),
            "terminal_proof": terminal_row,
            "window_proof": window_row,
            "terminal_verdict": terminal["verdict"],
            "terminal_output_counts": terminal["terminal_output_counts"],
            "terminal_own_steps": terminal["terminal_own_steps"],
            "late_ni_steps": {
                step: {
                    "carry_file_sha256": row["carry"]["file_sha256"],
                    "carry_manifest_sha256": row["carry"]["manifest"]["manifest_sha256"],
                    "frame_file_sha256": row["frame"]["sha256"],
                    "health_passed": row["health"]["passed"],
                }
                for step, row in retained_late.items()
            },
            "checkpoint_carries": {
                step: {
                    "file_sha256": row["carry"]["file_sha256"],
                    "manifest_sha256": row["carry"]["manifest"]["manifest_sha256"],
                }
                for step, row in (window.get("checkpoint_carries") or {}).items()
            },
            "terminal_carries": {
                domain: {
                    "file_sha256": row["file_sha256"],
                    "manifest_sha256": row["manifest"]["manifest_sha256"],
                }
                for domain, row in terminal_carries.items()
            },
        },
        "autotune_pin": file_row(pin),
        "autotune_dump_promotion": {
            "staging_path": str(staged_pin.resolve()),
            "staging_path_absent_after_success": True,
            "canonical_destination": str(pin.resolve()),
            "promoted_only_after_complete_reference_process": True,
        },
        "compiled_programs": compiled_programs,
        "xla_flags": expected_flags,
        "output_inventory": output_counts,
        "complete_reference_process": True,
        "persistent_compilation_cache_used": False,
        "diagnostic_admission_only": True,
        "release_gate_green": False,
    }
    manifest["proof_sha256"] = canonical_digest(manifest)
    atomic_write_json(output, manifest)
    print(json.dumps({
        "verdict": manifest["verdict"],
        "output": str(output),
        "file_sha256": sha256_file(output),
        "proof_sha256": manifest["proof_sha256"],
        "pin_sha256": manifest["autotune_pin"]["file_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
