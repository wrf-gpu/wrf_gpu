"""Single-process unroll A/B: compile + warm-time one segment per unroll value.

Runs unroll in {1,2,4} (set via GPUWRF_ACOUSTIC_UNROLL re-import is not possible
in-process, so this script takes ONE unroll via env and reports compile + warmed
per-step + peak mem + finiteness). Driver loops call it once per unroll so each
gets a fresh process (avoids cross-unroll cache/state bleed) but each process does
exactly ONE compile + a few warm timings -> minimal allocation churn.

Times warm calls of ONE compiled non-radiation segment (block between calls).

Run (per unroll):
  ... GPUWRF_ACOUSTIC_UNROLL=2 python proofs/v0100/wave_a_unroll_ab.py --steps 60 --out ab_u2.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/v0100")
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"


def _peak_mb() -> float:
    try:
        return float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cadence", type=int, default=180)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", type=str, default="wave_a_ab.json")
    args = ap.parse_args()
    steps = int(args.steps)
    cadence = int(args.cadence)
    unroll = int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10,
                              run_id=L2_RUN_ID, run_root=paths.wrf_l2_root(),
                              domain="d02", radiation_cadence_steps=cadence)
    case, _ = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=False, radiation_cadence_steps=cadence,
                             time_utc=case.run_start)
    carry0 = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64)))

    print(f"unroll={unroll} steps={steps} compiling", flush=True)
    t0 = time.perf_counter()
    c = _advance_chunk(carry0, nl, jnp.asarray(1, dtype=jnp.int32),
                       n_steps=steps, cadence=cadence)
    jax.block_until_ready(c.state.theta)
    compile_s = time.perf_counter() - t0
    peak = _peak_mb()
    print(f"compiled {compile_s:.1f}s; timing {args.repeats} warm passes", flush=True)

    ms = []
    for r in range(int(args.repeats)):
        start = jnp.asarray(1 + (r + 1) * steps, dtype=jnp.int32)
        t0 = time.perf_counter()
        c2 = _advance_chunk(c, nl, start, n_steps=steps, cadence=cadence)
        jax.block_until_ready(c2.state.theta)
        ms.append((time.perf_counter() - t0) / steps * 1000.0)
        print(f"  pass {r}: {ms[-1]:.2f} ms/step", flush=True)
        c = c2
    per_step = float(min(ms))
    th = np.asarray(jax.device_get(c.state.theta))
    out = {"unroll": unroll, "steps": steps, "compile_s": compile_s,
           "warmed_per_step_ms": per_step, "samples_ms": ms, "peak_mb": peak,
           "theta_finite": bool(np.isfinite(th).all())}
    PROOF.mkdir(parents=True, exist_ok=True)
    (PROOF / args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
