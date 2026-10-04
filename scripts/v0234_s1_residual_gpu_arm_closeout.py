#!/usr/bin/env python3
"""CPU-only scientific closeout for the frozen S1 residual GPU arm.

This runs only after the canonical lock wrapper has returned.  It re-hashes
the complete retained run, reconstructs authentic WRF-vs-GPU S1 from the
step-1 savepoints, and applies the preregistered raw and intrinsic gates.  It
does not import JAX/gpuwrf, inspect a device, acquire a lock, compile, or run
the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_dycore_internal_split_kimi as split  # noqa: E402
from scripts import v0234_first_interval_momentum_wrf_reassemble as wrf  # noqa: E402
from scripts import v0234_s1_dyn_attribution_fable5 as s1  # noqa: E402
from scripts import v0234_s1_operator_ledger as ledger  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt"
CANDIDATE_PROOF = SPRINT / "candidate-cpu-proof.json"
CANDIDATE_SELF = "500db72270e5bd0d5eceb5c37a807cc530d41d38bc716f1c71c754906ea6e8b9"
PREDICTIONS = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_s1_residual_closure_gpt/operator_ledger1/"
    "current-source-predictions-step1.npz"
)
PREDICTIONS_SHA256 = "73087c007e556be03a5a1cb40ab975d31d52a6bdb8e08d789ea78fa9714dca3a"
FROZEN_RUN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
)
NAMESPACE = "nested_stage_omega_transport_470e6111_s1_residual_closure_gpt1"
RAW_PREDICTIONS = {
    "nonspec": 0.7955252696331928,
    "relax_rows_1_4": 0.7826322158654425,
    "interior_ge_5": 0.7978517274341745,
    "ring_1": 0.7806456132691051,
}
FROZEN_RAW = {
    "nonspec": 3.22385592467145,
    "relax_rows_1_4": 7.765706866136884,
    "interior_ge_5": 1.1401580540399108,
    "ring_1": 15.237017605044933,
}
RAW_ABSOLUTE_BAND = 0.01
INTRINSIC_LIMIT = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def authenticate_json(path: Path, expected: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    observed = canonical({key: value for key, value in payload.items() if key != "proof_sha256"})
    require(payload.get("proof_sha256") == observed, f"canonical self-hash mismatch: {path}")
    if expected is not None:
        require(observed == expected, f"unexpected canonical identity: {path}")
    return payload


def zone_masks(fields: dict[str, np.ndarray], lo: int, hi: int) -> dict[str, np.ndarray]:
    return {
        name: (ledger.cpu.distance_to_edge(value.shape) >= lo)
        & (ledger.cpu.distance_to_edge(value.shape) <= hi)
        for name, value in fields.items()
    }


def combined_metric(
    fields: dict[str, np.ndarray], selected: dict[str, np.ndarray],
) -> dict[str, Any]:
    values = [np.asarray(fields[name][selected[name]], dtype=np.float64) for name in ("u", "v")]
    count = sum(value.size for value in values)
    finite = all(bool(np.isfinite(value).all()) for value in values)
    sse = sum(float(np.sum(value * value, dtype=np.float64)) for value in values)
    return {
        "cells": count,
        "finite": finite,
        "sse": sse,
        "rmse": math.sqrt(sse / count),
    }


def classify_fields(
    residual: dict[str, np.ndarray], intrinsic: dict[str, np.ndarray],
) -> dict[str, Any]:
    zone_ranges = {
        "nonspec": (1, 999),
        "relax_rows_1_4": (1, 4),
        "interior_ge_5": (5, 999),
        "ring_1": (1, 1),
    }
    rows: dict[str, Any] = {}
    first_red: str | None = None
    for name, (lo, hi) in zone_ranges.items():
        selected = zone_masks(residual, lo, hi)
        raw = combined_metric(residual, selected)
        core = combined_metric(intrinsic, selected)
        raw_delta = abs(raw["rmse"] - RAW_PREDICTIONS[name])
        passed = bool(
            raw["finite"] and core["finite"]
            and raw_delta <= RAW_ABSOLUTE_BAND
            and core["rmse"] <= INTRINSIC_LIMIT
            and raw["rmse"] <= FROZEN_RAW[name]
        )
        rows[name] = {
            "raw": raw,
            "raw_prediction": RAW_PREDICTIONS[name],
            "raw_absolute_delta": raw_delta,
            "raw_absolute_delta_le": RAW_ABSOLUTE_BAND,
            "intrinsic": core,
            "intrinsic_rmse_le": INTRINSIC_LIMIT,
            "frozen_raw_rmse": FROZEN_RAW[name],
            "no_worse": raw["rmse"] <= FROZEN_RAW[name],
            "passed": passed,
        }
        if first_red is None and not passed:
            first_red = name
    return {"passed": first_red is None, "first_red": first_red, "rows": rows}


def load_step1(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    indexed = {
        (row["tag"], row["field"]): row for row in rows if row["step"] == 1
    }
    require(len(indexed) == 20, "step-1 savepoint inventory is not exactly twenty")

    def load(tag: str, field: str) -> np.ndarray:
        row = indexed[(tag, field)]
        path = Path(row["path"])
        require(path.parent == (run_dir / "savepoints").resolve(), "savepoint escaped namespace")
        require(path.is_file() and not path.is_symlink(), f"savepoint missing/symlink: {path}")
        require(sha256_file(path) == row["file_sha256"], f"savepoint hash drift: {path}")
        value = np.load(path, allow_pickle=False)
        require(list(value.shape) == row["shape"], f"savepoint shape drift: {path}")
        require(bool(np.isfinite(value).all()), f"savepoint nonfinite: {path}")
        return value

    return {
        "ru_tend": load("l1_rk1_tend", "ru_tend"),
        "rv_tend": load("l1_rk1_tend", "rv_tend"),
        "ru_tendf": load("sp3_tendf", "ru_tendf"),
        "rv_tendf": load("sp3_tendf", "rv_tendf"),
        "u_save": load("l1_rk1_relax", "u_save"),
        "v_save": load("l1_rk1_relax", "v_save"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    require(run_dir.name == NAMESPACE and run_dir.is_dir() and not run_dir.is_symlink(), "wrong run namespace")
    require("jax" not in sys.modules and not any(name.startswith("gpuwrf") for name in sys.modules), "forbidden runtime import")
    require(sha256_file(PREDICTIONS) == PREDICTIONS_SHA256, "sealed source prediction archive drifted")
    candidate = authenticate_json(CANDIDATE_PROOF, CANDIDATE_SELF)

    terminal_paths = sorted(run_dir.glob("*terminal-proof.json"))
    require(len(terminal_paths) == 1, "expected exactly one terminal proof")
    terminal = authenticate_json(terminal_paths[0])
    require(terminal["namespace"] == NAMESPACE, "terminal namespace mismatch")
    require(terminal["window"]["model_dispatches"] == 9, "dispatch count is not nine")
    require(terminal["window"]["own_steps"] == {"d01": 1, "d02": 3, "d03": 9}, "schedule drifted")
    require(terminal["gpu_released_at_exit"] is True, "GPU not released at model exit")
    require(terminal["release_gate_unchanged"] is True and terminal["tolerance_changed"] is False, "release/tolerance drift")
    require(all(row["passed"] and not row["violations"] for row in terminal["health_rows"]), "health regression")

    manifest = terminal["savepoint_manifest"]
    rows = manifest["rows"]
    require(len(rows) == manifest["count"] == manifest["expected_count"] == 180, "savepoint count is not 180")
    require(canonical(rows) == manifest["rows_sha256"], "savepoint manifest hash mismatch")
    expected_names: set[str] = set()
    for row in rows:
        path = Path(row["path"])
        require(path.parent == run_dir / "savepoints", "savepoint escaped run directory")
        require(path.is_file() and not path.is_symlink(), f"missing/symlink savepoint: {path}")
        require(sha256_file(path) == row["file_sha256"], f"savepoint bytes drifted: {path}")
        expected_names.add(path.name)
    actual_names = {path.name for path in (run_dir / "savepoints").glob("*.npy")}
    require(actual_names == expected_names, "savepoint filesystem inventory differs from manifest")

    gpu = load_step1(run_dir, rows)
    ranks = wrf.load_ranks(split.RUNS / "control/momsp_dumps")
    control = {
        "ru_tend": wrf.reassemble3d("l1_rk1_tend__ru_tend", 1, ranks),
        "rv_tend": wrf.reassemble3d("l1_rk1_tend__rv_tend", 1, ranks),
        "ru_tendf": wrf.reassemble3d("sp3_tendf__ru_tendf", 1, ranks),
        "rv_tendf": wrf.reassemble3d("sp3_tendf__rv_tendf", 1, ranks),
        "u_save": wrf.reassemble3d("l1_rk1_tend__u_save", 1, ranks),
        "v_save": wrf.reassemble3d("l1_rk1_tend__v_save", 1, ranks),
    }
    split.GPU_INIT_FRAME = FROZEN_RUN / "gpu-output/wrfout_d03_2025-03-01_00:00:00"
    residual = s1.s1_residual(gpu, control, split.load_constants())
    with np.load(PREDICTIONS, allow_pickle=False) as archive:
        intrinsic = {
            name: residual[name]
            - np.asarray(archive[f"pgf_{name}"])
            - np.asarray(archive[f"adv_{name}"])
            - np.asarray(archive[f"cor_{name}"])
            for name in ("u", "v")
        }
    gate = classify_fields(residual, intrinsic)

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.s1-residual-scientific-closeout.v1",
        "verdict": "S1_RESIDUAL_GPU_ARM_GREEN" if gate["passed"] else "S1_RESIDUAL_GPU_ARM_FALSIFIED",
        "namespace": NAMESPACE,
        "source_candidate_commit": "d4b03077f2e8858163d1fbdf99c090fab902e0ac",
        "candidate_cpu_proof_sha256": candidate["proof_sha256"],
        "prediction_archive": {"path": str(PREDICTIONS), "file_sha256": PREDICTIONS_SHA256},
        "terminal_proof": {"path": str(terminal_paths[0]), "proof_sha256": terminal["proof_sha256"], "file_sha256": sha256_file(terminal_paths[0])},
        "savepoints": {"count": len(rows), "rows_sha256": manifest["rows_sha256"], "all_rehashed": True},
        "scientific_gate": gate,
        "non_target_and_resource_gate": {
            "all_health_rows_passed": True,
            "release_gate_unchanged": True,
            "tolerance_changed": False,
            "gpu_lock_acquired": False,
            "gpu_queries": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "jax_imported": False,
            "gpuwrf_imported": False,
        },
    }
    payload["proof_sha256"] = canonical(payload)
    output = run_dir / "s1-residual-scientific-closeout.json"
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"verdict": payload["verdict"], "proof_sha256": payload["proof_sha256"], "path": str(output)}, sort_keys=True))
    return 0 if gate["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
