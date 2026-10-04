#!/usr/bin/env python3
"""Fail-closed policy interpretation of the retained bounded GPU arm.

The inherited raw probe incorrectly treated the released pooled-RMSE limits as
independent per-frame limits.  Its 00:00 file is byte-identical to the already
authenticated terminal1 cold-start frame, whose canonical proof records T2
RMSE 2.143740471999841 K and nevertheless passes the finite/static frame gate.
The manager-frozen policy explicitly applies the unchanged limits to pooled
samples across exact same-valid frames; per-frame metrics localize only.

This CPU-only postprocessor never runs, queries, or modifies the model/GPU.  It
preserves and authenticates the raw proof and every output byte, requires the
00:00 frame to be the exact known baseline, requires BOTH post-dispatch frames
00:20/00:40 to satisfy every strict limit independently (a stronger bounded
gate), then applies the same released limits to the correctly pooled 3-frame
samples.  No post-step red can be reclassified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
GPU_ROOT = (
    CASE_ROOT
    / "corrected_ni_rca_max_22c2bd7a/v0234_final_ni_fable5_terminal3"
)
RAW_PROOF = GPU_ROOT / "bounded-gpu-proof.json"
TERMINAL1 = (
    CASE_ROOT
    / "corrected_ni_rca_max_22c2bd7a/v0234_1500_science_60659a2e_terminal1"
)
INITIAL_FRAME_PROOF = TERMINAL1 / "science-runtime/frame-pairs/d03-step-00000.json"
INITIAL_FRAME_PROOF_SHA = "07e590b8502e042a90bdbcf7ffa620dcad9458a4f7fde5fccc0c1b732fa58107"
INITIAL_GPU_FRAME_SHA = "95fa068b4712090254366b16ca01367c9b4cbab61b09ec621a5d1c431757484d"
FROZEN_PAIR_STATE = TERMINAL1 / "incremental-pairs.json"
FROZEN_PAIR_STATE_SHA = "57308a1a53bc6bd1442cb6c69c392fc1393b1b6556092fd02dd6137e6dafefb5"
MANAGER_POLICY = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v024-real1km-perf/"
    ".agent/decisions/V0234-CORRECTED-IDENTITY-POLICY.md"
)
MANAGER_POLICY_SHA = "c991f2c83d814eb0efb1d99071b39a0b2e636e92b70bd90268e066af60612df2"
GPU_COMMIT = "66a5fbde1a1591cfadbcce74e481bbbc28f5057b"
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
STATIC_FIELDS = ("XLAT", "XLONG", "HGT", "LANDMASK")
FRAME_LABELS = ("00:00", "00:20", "00:40")
SCHEMA = "gpuwrf.v0234.final-ni-fable5-bounded-gpu-policy.v1"


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled_rmse(sum_squares: Sequence[float], counts: Sequence[int]) -> float:
    if len(sum_squares) != len(counts) or not sum_squares:
        raise ValueError("pooled RMSE needs equal non-empty sequences")
    if any(count <= 0 for count in counts):
        raise ValueError("pooled RMSE counts must be positive")
    total = math.fsum(float(value) for value in sum_squares)
    count = sum(int(value) for value in counts)
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("pooled RMSE sum of squares must be finite/non-negative")
    return math.sqrt(total / float(count))


def raw_red_coordinates(red_gates: Sequence[str]) -> tuple[tuple[str, str], ...]:
    coordinates: list[tuple[str, str]] = []
    for gate in red_gates:
        parts = gate.split(":")
        if len(parts) < 4:
            raise ValueError(f"malformed raw red gate: {gate!r}")
        coordinates.append((f"{parts[0]}:{parts[1]}", parts[2]))
    return tuple(coordinates)


def _load_canonical(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("proof_sha256") != canonical_hash(payload):
        raise RuntimeError(f"canonical proof mismatch: {path}")
    return payload


def build_proof() -> dict[str, Any]:
    if sha256_file(MANAGER_POLICY) != MANAGER_POLICY_SHA:
        raise RuntimeError("manager-frozen identity policy changed")
    if sha256_file(FROZEN_PAIR_STATE) != FROZEN_PAIR_STATE_SHA:
        raise RuntimeError("frozen terminal1 pairing state changed")
    frozen_state = json.loads(FROZEN_PAIR_STATE.read_text())
    policy = frozen_state["identity_policy"]
    if policy["manager_sha256"] != MANAGER_POLICY_SHA:
        raise RuntimeError("terminal1 policy authority differs from manager file")
    if not str(policy["aggregation"]).startswith("pooled_rmse_across_all_"):
        raise RuntimeError(f"frozen policy is not pooled: {policy['aggregation']!r}")
    if tuple(policy["strict_fields"]) != STRICT_FIELDS:
        raise RuntimeError("strict field set changed")
    limits = {name: float(policy["strict_rmse_limits"][name]) for name in STRICT_FIELDS}

    initial_authority = _load_canonical(INITIAL_FRAME_PROOF)
    if initial_authority["proof_sha256"] != INITIAL_FRAME_PROOF_SHA:
        raise RuntimeError("initial canonical frame authority changed")
    if not initial_authority["passed"]:
        raise RuntimeError("authenticated terminal1 initial frame was not green")
    if initial_authority["candidate"]["sha256"] != INITIAL_GPU_FRAME_SHA:
        raise RuntimeError("terminal1 initial GPU frame hash changed")
    initial_t2 = float(initial_authority["d03_full_pair"]["strict_rmse"]["T2"])
    if not initial_t2 > limits["T2"]:
        raise RuntimeError("expected known per-frame T2 baseline is no longer red")

    raw = _load_canonical(RAW_PROOF)
    if raw["model"]["commit"] != GPU_COMMIT:
        raise RuntimeError("raw GPU proof model commit changed")
    if raw["bounds"] != {
        "root_steps": 45,
        "d03_own_steps": 405,
        "d03_output_alarms": [200, 400],
        "fresh_full_history": False,
    }:
        raise RuntimeError(f"raw GPU bounds changed: {raw['bounds']!r}")
    if raw["retained_state_anchor"]["mismatched_state_arrays"]:
        raise RuntimeError("raw GPU cold-start carry anchor differs")
    if raw["lock_v2"]["intent"] != "production-preemptible" or not raw["lock_v2"]["live_verified"]:
        raise RuntimeError("raw GPU lock-v2 authority is not green")
    if raw["cpu_affinity"] != [12, 13, 14, 15]:
        raise RuntimeError("raw GPU CPU affinity changed")
    if raw["gpu_accounting"]["gpu_locks_acquired_or_verified"] != 1 or raw["gpu_accounting"]["model_processes"] != 1:
        raise RuntimeError("raw GPU accounting changed")
    if raw["verdict"] != "GPU_DISCRIMINATOR_RED":
        raise RuntimeError(f"expected preserved raw false-red proof, got {raw['verdict']!r}")
    raw_reds = raw_red_coordinates(raw["red_gates"])
    if raw_reds != (("00:00", "T2"),):
        raise RuntimeError(f"raw proof has a scientific/post-step red: {raw['red_gates']!r}")

    from netCDF4 import Dataset

    frame_rows: dict[str, Any] = {}
    pooled_parts: dict[str, dict[str, list[Any]]] = {
        field: {"sum_squares": [], "counts": []} for field in STRICT_FIELDS
    }
    post_dispatch_green = True
    static_green = True
    for label in FRAME_LABELS:
        raw_frame = raw["frames"][label]
        model_path = Path(raw_frame["model_frame"])
        cpu_path = Path(raw_frame["cpu_frame"])
        if sha256_file(model_path) != raw_frame["model_frame_sha256"]:
            raise RuntimeError(f"{label}: model frame bytes changed")
        if sha256_file(cpu_path) != raw_frame["cpu_frame_sha256"]:
            raise RuntimeError(f"{label}: CPU frame bytes changed")
        strict: dict[str, Any] = {}
        static: dict[str, bool] = {}
        with Dataset(str(model_path), "r") as model, Dataset(str(cpu_path), "r") as cpu:
            for field in STATIC_FIELDS:
                exact = np.array_equal(
                    np.asarray(model.variables[field][:]),
                    np.asarray(cpu.variables[field][:]),
                )
                static[field] = bool(exact)
                static_green = static_green and bool(exact)
            for field in STRICT_FIELDS:
                model_value = np.asarray(model.variables[field][:], dtype=np.float64)
                cpu_value = np.asarray(cpu.variables[field][:], dtype=np.float64)
                if model_value.shape != cpu_value.shape:
                    raise RuntimeError(f"{label}:{field}: incompatible shapes")
                if not np.isfinite(model_value).all() or not np.isfinite(cpu_value).all():
                    raise RuntimeError(f"{label}:{field}: non-finite value")
                delta = model_value - cpu_value
                sum_squares = float(np.sum(delta * delta, dtype=np.float64))
                count = int(delta.size)
                rmse = math.sqrt(sum_squares / float(count))
                raw_rmse = float(raw_frame["strict_rmse"][field]["rmse"])
                # The raw probe uses ``sqrt(mean(delta*delta))`` while this
                # policy proof retains an explicit float64 sum-of-squares for
                # pooling.  Exact frame hashes bind the values; allow only the
                # harmless final-reduction rounding difference here.
                if not math.isclose(rmse, raw_rmse, rel_tol=2.0e-13, abs_tol=2.0e-13):
                    raise RuntimeError(
                        f"{label}:{field}: recomputed RMSE {rmse} != raw {raw_rmse}"
                    )
                passed = rmse <= limits[field]
                if label != "00:00":
                    post_dispatch_green = post_dispatch_green and passed
                strict[field] = {
                    "n": count,
                    "sum_squares": sum_squares,
                    "rmse": rmse,
                    "max_abs": float(np.max(np.abs(delta))),
                    "limit": limits[field],
                    "per_frame_pass": passed,
                    "gate_status": (
                        "POST_DISPATCH_REQUIRED" if label != "00:00" else "KNOWN_EXACT_INITIAL_BASELINE"
                    ),
                }
                pooled_parts[field]["sum_squares"].append(sum_squares)
                pooled_parts[field]["counts"].append(count)
        frame_rows[label] = {
            "own_step": raw_frame["own_step"],
            "model_frame": str(model_path),
            "model_frame_sha256": raw_frame["model_frame_sha256"],
            "cpu_frame": str(cpu_path),
            "cpu_frame_sha256": raw_frame["cpu_frame_sha256"],
            "static_exact": static,
            "strict": strict,
        }

    if frame_rows["00:00"]["model_frame_sha256"] != INITIAL_GPU_FRAME_SHA:
        raise RuntimeError("new 00:00 frame is not byte-identical to terminal1 authority")
    if not static_green:
        raise RuntimeError("a bounded frame has a static geometry mismatch")
    if not post_dispatch_green:
        raise RuntimeError("00:20/00:40 has a strict scientific red")

    pooled: dict[str, Any] = {}
    pooled_green = True
    for field in STRICT_FIELDS:
        parts = pooled_parts[field]
        rmse = pooled_rmse(parts["sum_squares"], parts["counts"])
        passed = rmse <= limits[field]
        pooled_green = pooled_green and passed
        pooled[field] = {
            "n": sum(parts["counts"]),
            "sum_squares": math.fsum(parts["sum_squares"]),
            "rmse": rmse,
            "limit": limits[field],
            "pass": passed,
        }
    if not pooled_green:
        raise RuntimeError(f"bounded pooled identity is red: {pooled!r}")

    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "verdict": "GPU_DISCRIMINATOR_POLICY_GREEN_RAW_GATE_MISAPPLIED",
        "raw_proof": {
            "path": str(RAW_PROOF),
            "proof_sha256": raw["proof_sha256"],
            "verdict": raw["verdict"],
            "red_gates": raw["red_gates"],
            "preserved_unchanged": True,
        },
        "policy_authority": {
            "manager_path": str(MANAGER_POLICY),
            "manager_commit": policy["manager_commit"],
            "manager_sha256": MANAGER_POLICY_SHA,
            "frozen_pair_state_path": str(FROZEN_PAIR_STATE),
            "frozen_pair_state_sha256": FROZEN_PAIR_STATE_SHA,
            "aggregation": policy["aggregation"],
            "strict_fields": list(STRICT_FIELDS),
            "strict_rmse_limits": limits,
            "threshold_changes": 0,
        },
        "initial_frame_authority": {
            "canonical_proof": str(INITIAL_FRAME_PROOF),
            "proof_sha256": INITIAL_FRAME_PROOF_SHA,
            "terminal1_candidate_sha256": INITIAL_GPU_FRAME_SHA,
            "new_candidate_sha256": frame_rows["00:00"]["model_frame_sha256"],
            "byte_identical": True,
            "known_t2_rmse": initial_t2,
            "known_t2_per_frame_limit": limits["T2"],
        },
        "frames": frame_rows,
        "pooled_strict_identity": pooled,
        "gates": {
            "raw_red_is_exact_known_0000_t2_only": True,
            "initial_frame_byte_identical_to_authenticated_terminal1": True,
            "static_geometry_exact_all_frames": static_green,
            "post_dispatch_0020_0040_each_strict_field_green": post_dispatch_green,
            "three_frame_pooled_strict_identity_green": pooled_green,
            "raw_lock_environment_bounds_and_carry_anchor_green": True,
            "model_and_output_preserved_unchanged": True,
            "no_new_gpu_queries_compiles_dispatches_or_arms": True,
        },
        "postprocessor_gpu_accounting": {
            "queries": 0,
            "compiles": 0,
            "dispatches": 0,
            "arms": 0,
        },
        "scientific_scope": {
            "proves": "finite and strict-quality bounded d03 window through own step 400 (00:40), including the previously fatal step 200",
            "does_not_prove": "late step-9400 Ni, 15:00 no-worse, full 19/19/55 identity, or 18-hour completion",
            "v10_causal_status": "bounded identity metric only; no causal link to Ni inferred",
        },
        "commands": [
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_gpu_policy_proof.py"
        ],
    }
    proof["proof_sha256"] = canonical_hash(proof)
    return proof


def main() -> int:
    proof = build_proof()
    out_path = SPRINT_DIR / "bounded-gpu-policy-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"verdict={proof['verdict']}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
