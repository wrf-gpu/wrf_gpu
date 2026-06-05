#!/usr/bin/env python
"""Short operational chunk JIT-cache diagnostic for v0.11.0.

Runs consecutive d02 ``_advance_chunk`` calls in one Python process, records
per-chunk wall time, and hashes the final state/diagnostic outputs so a cache-key
fix can be proven bit-identical.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"
L3_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"


def _sha256(array: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _array_record(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    numeric = arr.astype(np.float64) if np.issubdtype(arr.dtype, np.number) else None
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _sha256(arr),
        "finite": bool(np.isfinite(numeric).all()) if numeric is not None else None,
        "min": float(np.nanmin(numeric)) if numeric is not None and numeric.size else None,
        "max": float(np.nanmax(numeric)) if numeric is not None and numeric.size else None,
    }


def _state_hashes(state: Any, jax_mod: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in getattr(state, "__slots__", ()):
        value = getattr(state, name, None)
        if value is None:
            continue
        out[name] = _array_record(jax_mod.device_get(value))
    return out


def _namedtuple_hashes(value: Any, jax_mod: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in getattr(value, "_fields", ()):
        out[name] = _array_record(jax_mod.device_get(getattr(value, name)))
    return out


def _static_holder_fingerprint(namelist: Any) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for repeat in range(2):
        _children, aux = namelist.tree_flatten()
        holders = []
        for idx, item in enumerate(aux):
            if type(item).__name__ != "_StaticHolder":
                continue
            value = getattr(item, "value", None)
            holders.append(
                {
                    "aux_index": int(idx),
                    "value_is_none": value is None,
                    "value_type": type(value).__name__ if value is not None else "NoneType",
                    "value_id": None if value is None else int(id(value)),
                    "holder_hash": int(hash(item)),
                }
            )
        fingerprints.append({"repeat": repeat, "holders": holders})
    return fingerprints


def _compare_hashes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    for section in ("state_hashes", "diagnostic_hashes"):
        left_section = left.get(section, {})
        right_section = right.get(section, {})
        fields = sorted(set(left_section) | set(right_section))
        section_mismatches = {}
        for name in fields:
            lrec = left_section.get(name)
            rrec = right_section.get(name)
            ok = (
                lrec is not None
                and rrec is not None
                and lrec.get("shape") == rrec.get("shape")
                and lrec.get("dtype") == rrec.get("dtype")
                and lrec.get("sha256") == rrec.get("sha256")
            )
            if not ok:
                section_mismatches[name] = {
                    "left": None if lrec is None else {
                        "shape": lrec.get("shape"),
                        "dtype": lrec.get("dtype"),
                        "sha256": lrec.get("sha256"),
                    },
                    "right": None if rrec is None else {
                        "shape": rrec.get("shape"),
                        "dtype": rrec.get("dtype"),
                        "sha256": rrec.get("sha256"),
                    },
                }
        mismatches[section] = section_mismatches
    return {
        "bit_identical": not any(mismatches[section] for section in mismatches),
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=L2_RUN_ID)
    parser.add_argument("--run-root-kind", choices=("l2", "l3"), default="l2")
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--cadence", type=int, default=180)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.chunks) < 1:
        raise ValueError("--chunks must be positive")
    if int(args.steps) < 1:
        raise ValueError("--steps must be positive")

    import jax
    import jax.numpy as jnp

    from gpuwrf.config import paths
    from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk,
        _committed_initial_carry_for_run,
        _m9_snapshot,
    )

    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpus:
        raise RuntimeError("No JAX GPU backend visible; refusing to produce timing proof")

    run_root = paths.wrf_l2_root() if args.run_root_kind == "l2" else paths.wrf_l3_root()
    cfg = DailyPipelineConfig(
        run_id=args.run_id,
        hours=1,
        run_root=run_root,
        domain=args.domain,
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=int(args.cadence),
    )
    case, run_dir = _build_real_case(cfg)
    namelist = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=False,
        radiation_cadence_steps=int(args.cadence),
        time_utc=case.run_start,
    )
    carry = _committed_initial_carry_for_run(case.state, namelist)

    print(
        "recompile_diag "
        f"device={gpus[0]} run_id={args.run_id} domain={args.domain} "
        f"chunks={args.chunks} steps={args.steps} cadence={args.cadence}",
        flush=True,
    )

    chunk_records: list[dict[str, Any]] = []
    for idx in range(int(args.chunks)):
        start_step = 1 + idx * int(args.steps)
        print(f"BEGIN_CHUNK {idx + 1} start_step={start_step}", flush=True)
        t0 = time.perf_counter()
        carry = _advance_chunk(
            carry,
            namelist,
            jnp.asarray(start_step, dtype=jnp.int32),
            n_steps=int(args.steps),
            cadence=int(args.cadence),
        )
        jax.block_until_ready(carry.state.theta)
        wall_s = time.perf_counter() - t0
        rec = {
            "chunk_index": int(idx + 1),
            "start_step": int(start_step),
            "steps": int(args.steps),
            "wall_s": float(wall_s),
            "per_step_ms": float(wall_s * 1000.0 / float(args.steps)),
        }
        chunk_records.append(rec)
        print(
            f"END_CHUNK {idx + 1} wall_s={wall_s:.6f} "
            f"per_step_ms={rec['per_step_ms']:.3f}",
            flush=True,
        )

    lead_seconds = float(int(args.chunks) * int(args.steps)) * float(namelist.dt_s)
    diag_t0 = time.perf_counter()
    diag = _m9_snapshot(carry, namelist, jnp.asarray(lead_seconds, dtype=jnp.float64))
    jax.block_until_ready(diag.t2)
    diag_wall_s = time.perf_counter() - diag_t0

    payload: dict[str, Any] = {
        "schema": "V0110RecompileDiagnostic",
        "schema_version": 1,
        "status": "PASS",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "domain": args.domain,
        "device": str(gpus[0]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "env": {
            "JAX_LOG_COMPILES": os.environ.get("JAX_LOG_COMPILES"),
            "JAX_EXPLAIN_CACHE_MISSES": os.environ.get("JAX_EXPLAIN_CACHE_MISSES"),
            "GPUWRF_JAX_CACHE": os.environ.get("GPUWRF_JAX_CACHE"),
            "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "GPUWRF_THOMPSON_NSED": os.environ.get("GPUWRF_THOMPSON_NSED"),
        },
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "chunks": int(args.chunks),
        "steps_per_chunk": int(args.steps),
        "radiation_cadence_steps": int(args.cadence),
        "chunk_records": chunk_records,
        "diagnostic_wall_s": float(diag_wall_s),
        "static_holder_fingerprint": _static_holder_fingerprint(namelist),
        "state_hashes": _state_hashes(carry.state, jax),
        "diagnostic_hashes": _namedtuple_hashes(diag, jax),
    }
    if args.compare_json is not None:
        before = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
        payload["comparison_to"] = str(args.compare_json)
        payload["hash_comparison"] = _compare_hashes(before, payload)
        payload["status"] = "PASS" if payload["hash_comparison"]["bit_identical"] else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "chunk_wall_s": [round(r["wall_s"], 6) for r in chunk_records],
                "diagnostic_wall_s": round(diag_wall_s, 6),
                "bit_identical": payload.get("hash_comparison", {}).get("bit_identical"),
                "out": str(args.out),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
