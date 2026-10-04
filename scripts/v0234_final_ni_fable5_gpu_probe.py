"""Shortest decisive GPU discriminator for the corrected v0.23.4 candidate.

Bounded BY CONSTRUCTION: exactly 45 root steps (d03 own step 405), one model
process, one ordinary compile per domain (fori advance), the production
``_PerDomainWrfoutWriter`` with its unmodified finite guard, and d03 output
alarms only at own steps 200 (00:20 — the exact frame that scientifically
falsified `60659a2e`) and 400 (00:40 — the successor frame terminal1 never
reached).  No fresh 18-hour history.

Gates (stop first red — any failure raises, the process dies, and the
lock-v2 wrapper releases fail-closed):

1. preemption sentinels absent + nightly inactive, re-checked after the run;
2. live lock-v2 lease verified (production-preemptible, exact accepted
   wrapper/verifier hashes) via the reviewed ordinary-lane verifier;
3. cold-start d03 prognostic state identical to the retained terminal2
   step-0 carry (the retained-state anchor);
4. production finite guard green at d03 steps 200 and 400 (the guard that
   raised NonFiniteStateError Ni (0,1,1) in terminal2);
5. strict-field RMSE of the written d03 00:00/00:20/00:40 wrfout frames
   against the frozen CPU-WRF oracle frames within the frozen identity-policy
   limits (T/T2/U10/V10 1.5, U/V 1.8, W 0.3, PSFC 120; full-field float64
   RMSE exactly as the reviewed gate's ``_metric``).

Must be launched through the accepted lock-v2 wrapper with
``--intent production-preemptible`` on CPUs 12-15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REQUIRED_ENV = {
    "JAX_PLATFORMS": "cuda",
    "JAX_ENABLE_X64": "true",
    "CUDA_VISIBLE_DEVICES": "0",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "GPUWRF_ALLOCATOR": "cuda_async",
    "GPUWRF_FINITE_CHECK": "1",
    "GPUWRF_NESTED_FUSE": "0",
    "GPUWRF_NESTED_DEFUSE_COMPILE": "0",
    "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
    "GPUWRF_NESTED_AOT": "0",
    "GPUWRF_AOT_VERIFY": "0",
    "GPUWRF_NESTED_ASYNC_OUTPUT": "0",
    "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
    "GPUWRF_FULL_WRFOUT": "1",
    "GPUWRF_TRAINING_OUTPUT_SUBSET": "0",
    "GPUWRF_BATCH_ENSEMBLE": "1",
    "GPUWRF_NESTED_SYNC_MODE": "root",
    "GPUWRF_BITWISE": "1",
    "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}

SCHEMA = "gpuwrf.v0234.final-ni-fable5-gpu-discriminator.v1"
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
INPUT_DIR = CASE_ROOT / "run/wrf"
RETAINED_STEP0 = (
    CASE_ROOT
    / "corrected_ni_rca_max_22c2bd7a/v0234_1500_science_60659a2e_terminal2"
    / "science-runtime/failure/last-healthy-d03-step-0.pkl"
)
RETAINED_STEP0_SHA = (
    "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
)
ROOT_STEPS = 45
D03_ALARMS = (200, 400)
D03_DT_S = 6.0
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
STRICT_RMSE_LIMITS = {
    "T": 1.5,
    "U": 1.8,
    "V": 1.8,
    "W": 0.3,
    "T2": 1.5,
    "U10": 1.5,
    "V10": 1.5,
    "PSFC": 120.0,
}
CPU_FRAMES = {
    "00:00": INPUT_DIR / "wrfout_d03_2025-03-01_00:00:00",
    "00:20": INPUT_DIR / "wrfout_d03_2025-03-01_00:20:00",
    "00:40": INPUT_DIR / "wrfout_d03_2025-03-01_00:40:00",
}


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


def _validate_env() -> dict[str, str]:
    actual = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual != REQUIRED_ENV:
        raise RuntimeError(f"pre-import env mismatch: {actual!r}")
    if "jax" in sys.modules or "gpuwrf" in sys.modules:
        raise RuntimeError("jax/gpuwrf imported before environment validation")
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [12, 13, 14, 15]:
        raise RuntimeError(f"CPU affinity {affinity} != [12,13,14,15]")
    return dict(REQUIRED_ENV)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    env_authority = _validate_env()
    started = time.time()

    model_root = args.model_root.resolve()
    head = _git(model_root, "rev-parse", "HEAD")
    tree_sha = _git(model_root, "rev-parse", "HEAD^{tree}")
    if head != args.expected_commit:
        raise RuntimeError(f"model HEAD {head} != expected {args.expected_commit}")
    if _git(model_root, "status", "--porcelain"):
        raise RuntimeError("model worktree must be clean")

    # Reviewed ordinary-lane lock/preemption gates (constants + live lease
    # verification through the accepted lock-v2 verifier module).
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    preempt_pre = ordinary.assert_preemption_clear("gpu-discriminator-pre")
    lock_authority = ordinary.assert_lock_authority()
    cuda_authority = ordinary.assert_cuda_runtime()

    import numpy as np

    import gpuwrf

    gpuwrf_file = Path(gpuwrf.__file__).resolve()
    if model_root not in gpuwrf_file.parents:
        raise RuntimeError(f"gpuwrf resolved outside model root: {gpuwrf_file}")

    import jax

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"backend {jax.default_backend()} != gpu")

    from gpuwrf.integration.nested_pipeline import (
        NestedPipelineConfig,
        _PerDomainWrfoutWriter,
        _load_domains,
        _nested_sync_mode_from_env,
        domain_names_for,
    )
    from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dir = out_dir / "gpu-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = out_dir / "runtime"

    names = domain_names_for(3)
    config = NestedPipelineConfig(
        input_dir=INPUT_DIR,
        output_dir=runtime_root / "unused-output",
        proof_dir=runtime_root / "unused-pipeline-proof",
        hours=0,
        max_dom=3,
        feedback=False,
    )
    t_load0 = time.time()
    hierarchy, bundles, _meta, run_start, dt_by_domain, carries = _load_domains(
        config, names
    )
    load_seconds = time.time() - t_load0
    if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
    if run_start.isoformat() != "2025-03-01T00:00:00+00:00":
        raise RuntimeError(f"run start changed: {run_start.isoformat()}")
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)

    # Retained-state anchor: the cold-start d03 prognostic state must be the
    # retained terminal2 step-0 carry, bit for bit on every common State array.
    retained_sha = sha256_file(RETAINED_STEP0)
    if retained_sha != RETAINED_STEP0_SHA:
        raise RuntimeError("retained step-0 carry hash changed")
    import pickle

    with RETAINED_STEP0.open("rb") as handle:
        retained0 = pickle.load(handle)

    def _state_arrays(state: Any) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for attr in sorted(dir(state)):
            if attr.startswith("_"):
                continue
            value = getattr(state, attr, None)
            if value is None or callable(value):
                continue
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                arr = np.asarray(value)
                if np.issubdtype(arr.dtype, np.floating):
                    out[attr] = arr
        return out

    cold = _state_arrays(jax.device_get(carries["d03"]).state)
    kept = _state_arrays(retained0.state)
    common = sorted(set(cold) & set(kept))
    mismatched = [k for k in common if not np.array_equal(cold[k], kept[k])]
    retained_anchor = {
        "retained_sha256": retained_sha,
        "compared_state_arrays": len(common),
        "mismatched_state_arrays": mismatched,
    }
    if mismatched:
        raise RuntimeError(
            f"cold start does not reproduce the retained step-0 carry: {mismatched}"
        )

    writer = _PerDomainWrfoutWriter(
        output_dir=output_dir,
        input_dir=INPUT_DIR,
        run_start=run_start,
        bundles=bundles,
        output_cadence_steps={"d01": 0, "d02": 0, "d03": 0},
        dt_by_domain=dt_by_domain,
    )
    # Initial 00:00 d03 frame (the terminal1 identity-green frame).
    writer("d03", 0, carries["d03"])

    block_between, root_sync_cadence = _nested_sync_mode_from_env()
    t_run0 = time.time()
    result = run_operational_domain_tree(
        tree,
        root_steps=ROOT_STEPS,
        feedback_enabled=False,
        output=writer,
        output_alarm_steps={"d03": D03_ALARMS},
        block_between=block_between,
        root_sync_cadence=root_sync_cadence,
        carries=carries,
    )
    jax.block_until_ready(result.carries["d03"].state.theta)
    run_seconds = time.time() - t_run0

    preempt_post = ordinary.assert_preemption_clear("gpu-discriminator-post")

    # Strict-field RMSE versus the frozen CPU-WRF oracle frames (full-field
    # float64 RMSE, exactly the reviewed gate's _metric).
    from netCDF4 import Dataset

    def frame_metrics(model_path: Path, cpu_path: Path) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        with Dataset(str(model_path), "r") as model, Dataset(str(cpu_path), "r") as cpu:
            for field in STRICT_FIELDS:
                gpu_arr = np.asarray(model.variables[field][:], dtype=np.float64)
                cpu_arr = np.asarray(cpu.variables[field][:], dtype=np.float64)
                if gpu_arr.shape != cpu_arr.shape:
                    raise RuntimeError(
                        f"{field}: shape {gpu_arr.shape} != {cpu_arr.shape}"
                    )
                if not np.isfinite(gpu_arr).all():
                    raise RuntimeError(f"{field}: model frame non-finite")
                delta = gpu_arr - cpu_arr
                rmse = float(np.sqrt(np.mean(delta * delta)))
                rows[field] = {
                    "rmse": rmse,
                    "limit": STRICT_RMSE_LIMITS[field],
                    "pass": bool(rmse <= STRICT_RMSE_LIMITS[field]),
                }
        return rows

    frames: dict[str, Any] = {}
    reds: list[str] = []
    for label, own_step in (("00:00", 0), ("00:20", 200), ("00:40", 400)):
        model_path = output_dir / f"wrfout_d03_2025-03-01_{label}:00"
        if not model_path.exists():
            raise RuntimeError(f"model frame missing: {model_path}")
        metrics = frame_metrics(model_path, CPU_FRAMES[label])
        frames[label] = {
            "own_step": own_step,
            "model_frame": str(model_path),
            "model_frame_sha256": sha256_file(model_path),
            "cpu_frame": str(CPU_FRAMES[label]),
            "cpu_frame_sha256": sha256_file(CPU_FRAMES[label]),
            "strict_rmse": metrics,
        }
        for field, row in metrics.items():
            if not row["pass"]:
                reds.append(f"{label}:{field}:rmse={row['rmse']}")

    verdict = "GPU_DISCRIMINATOR_GREEN" if not reds else "GPU_DISCRIMINATOR_RED"
    wall_seconds = time.time() - started
    proof = {
        "schema": SCHEMA,
        "verdict": verdict,
        "red_gates": reds,
        "model": {"root": str(model_root), "commit": head, "tree": tree_sha},
        "environment": env_authority,
        "cpu_affinity": [12, 13, 14, 15],
        "lock_v2": lock_authority,
        "cuda_runtime": cuda_authority,
        "preemption": {"pre": preempt_pre, "post": preempt_post},
        "retained_state_anchor": retained_anchor,
        "bounds": {
            "root_steps": ROOT_STEPS,
            "d03_own_steps": ROOT_STEPS * 9,
            "d03_output_alarms": list(D03_ALARMS),
            "fresh_full_history": False,
        },
        "frames": frames,
        "previous_falsification": {
            "namespace": str(RETAINED_STEP0.parent.parent.parent),
            "first_red": "d03 step 200 Ni level 0 first_index (0,1,1)",
            "failure_proof_sha256": "f1abd1c6e2c99af4bc2e74003c09d30898cf84c7efa73834d8ef9705f194fadf",
        },
        "gpu_accounting": {
            "gpu_locks_acquired_or_verified": 1,
            "model_processes": 1,
            "domain_load_seconds": load_seconds,
            "tree_run_seconds": run_seconds,
            "wall_seconds": wall_seconds,
            "compile_policy": "one ordinary fori advance compile per domain plus writer diagnostics; caches disabled",
        },
        "written_outputs": {k: v for k, v in writer.written.items()},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    proof["proof_sha256"] = canonical_hash(proof)
    out_path = out_dir / "bounded-gpu-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"verdict={verdict}")
    print(f"red_gates={reds}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0 if verdict == "GPU_DISCRIMINATOR_GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
