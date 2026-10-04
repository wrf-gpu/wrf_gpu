"""Compare GPU first-interval momentum savepoints against pristine-WRF truth.

Pairs, per d03 step in 1..200, the GPU capture arm's per-step savepoints
(SP1 incoming U/V, SP2 raw MYNN RUBLTEN/RVBLTEN, SP3 assembled
ru_tendf/rv_tendf, SP4 end-of-step U/V) against the reassembled
instrumented-WRF dumps of the same operators, and isolates the first
divergent operator.

Truth chain (authenticated):
  WRF instrumented run (isolated tree, output-neutrality proven by
  byte-identical wrfout vs the retained CPU authority) -> per-rank dumps
  -> reassembled global arrays (validated bitwise vs wrfout 00:00 U/V).

Discriminator (pre-declared, no after-the-fact tolerance):
  floor(step) = max over the four SP-RMSEs at that step.  The step-1 floor
  is the fp32-truth/fp64-GPU representation floor.  The first step at
  which an operator's RMSE exceeds DIVERGENCE_FACTOR * floor(1)
  (and exceeds ABS_FLOOR) marks the first divergent update; the operator
  with the earliest breach is the first divergent operator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import v0234_first_interval_momentum_wrf_reassemble as wrf

GPU_RUN_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_first_interval_momentum2"
)
GPU_SP_DIR = GPU_RUN_DIR / "savepoints"
OUT_DIR = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/compare"
)

DIVERGENCE_FACTOR = 100.0
ABS_FLOOR = {
    "sp1_entry": 1e-5,   # m s-1 state fields
    "sp4_exit": 1e-5,
    "sp2_pbl": 1e-7,     # raw tendencies m s-2 (per-step values ~1e-3..1e-1)
    "sp3_tendf": 1e-4,   # mass-coupled tendencies (scaled by mu ~ 1e4)
}

FIELD_PAIRS = (
    ("sp1_entry", "u", "u"),
    ("sp1_entry", "v", "v"),
    ("sp2_pbl", "rublten", "rublten"),
    ("sp2_pbl", "rvblten", "rvblten"),
    ("sp3_tendf", "ru_tendf", "ru_tendf"),
    ("sp3_tendf", "rv_tendf", "rv_tendf"),
    ("sp4_exit", "u", "u"),
    ("sp4_exit", "v", "v"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_savepoint(tag: str, field: str, step: int) -> np.ndarray:
    path = GPU_SP_DIR / f"step{step:06d}_{tag}__{field}.npy"
    return np.load(path)


def wrf_savepoint(tag: str, field: str, step: int, ranks) -> np.ndarray:
    cache = OUT_DIR / "wrf_global_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"step{step:06d}_{tag}__{field}.npy"
    if cached.is_file():
        return np.load(cached)
    arr = wrf.reassemble3d(f"{tag}__{field}", step, ranks)
    np.save(cached, arr)
    return arr


def align(gpu: np.ndarray, truth: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if gpu.shape == truth.shape:
        return gpu, truth
    if gpu.ndim == 3 and truth.ndim == 3 and gpu.shape[1:] == truth.shape[1:]:
        if gpu.shape[0] == truth.shape[0] + 1:
            return gpu[: truth.shape[0]], truth
    raise ValueError(f"shape mismatch {name}: gpu={gpu.shape} wrf={truth.shape}")


def rmse(diff: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranks = wrf.load_ranks()
    gpu_hashes = {}
    per_step = []
    for step in range(1, 201):
        row = {"step": step, "fields": {}}
        for tag, gpu_field, wrf_field in FIELD_PAIRS:
            g = gpu_savepoint(tag, gpu_field, step)
            w = wrf_savepoint(tag, wrf_field, step, ranks)
            g, w = align(g, w, f"{tag}__{gpu_field}@{step}")
            diff = g - w
            interior = diff[:, 5:-5, 5:-5] if diff.ndim == 3 else diff[5:-5, 5:-5]
            full = row["fields"][f"{tag}__{gpu_field}"] = {
                "rmse": rmse(diff),
                "maxabs": float(np.max(np.abs(diff))),
                "interior_rmse": rmse(interior),
                "gpu_absmax": float(np.max(np.abs(g))),
                "wrf_absmax": float(np.max(np.abs(w))),
            }
            if diff.ndim == 3:
                full["lowest_level_rmse"] = rmse(diff[0])
        per_step.append(row)
        if step % 25 == 0:
            print(f"compared step {step}", flush=True)

    # --- first-divergence isolation --------------------------------------
    operator_series: dict[str, list[float]] = {}
    for tag, gpu_field, _ in FIELD_PAIRS:
        key = f"{tag}__{gpu_field}"
        operator_series[key] = [
            row["fields"][key]["rmse"] for row in per_step
        ]
    floor1 = {
        key: series[0] for key, series in operator_series.items()
    }
    first_divergent = None
    for index, row in enumerate(per_step):
        step = row["step"]
        for tag, gpu_field, _ in FIELD_PAIRS:
            key = f"{tag}__{gpu_field}"
            metric = row["fields"][key]["rmse"]
            threshold = max(
                DIVERGENCE_FACTOR * floor1[key], ABS_FLOOR[tag]
            )
            if metric > threshold:
                first_divergent = {
                    "step": step,
                    "operator": key,
                    "rmse": metric,
                    "threshold": threshold,
                    "floor_step1": floor1[key],
                    "row": row["fields"][key],
                }
                break
        if first_divergent is not None:
            break

    # --- trajectory summary -------------------------------------------------
    summary = {
        "floor_step1": floor1,
        "first_divergent": first_divergent,
        "series_last": {k: v[-1] for k, v in operator_series.items()},
        "series_max": {k: max(v) for k, v in operator_series.items()},
    }
    payload = {
        "schema": "gpuwrf.v0234.first-interval-momentum-compare.v1",
        "gpu_run_dir": str(GPU_RUN_DIR),
        "divergence_factor": DIVERGENCE_FACTOR,
        "abs_floor": ABS_FLOOR,
        "per_step": per_step,
        "summary": summary,
    }
    out = OUT_DIR / "first-interval-momentum-compare.json"
    payload["proof_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in payload.items() if k != "proof_sha256"},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
