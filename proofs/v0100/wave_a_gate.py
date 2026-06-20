"""Wave-A canonical gate: warmed per-step + coupled stability on the GREEN d02 L2 init.

The shipped ``proofs/perf/segscan_24h.py`` defaults to the L3 run_id
(``wrf_l3/20260521_18z_l3_24h_...``), whose d02 extraction is the known-unstable
finer-nest init (v0.9.0 "d03-1km steep-terrain dynamics instability", carried-over
OPEN).  The v0.9.0 GREEN d02 stability + skill path uses the **L2** run
``wrf_l2/20260521_18z_l2_72h_20260522T133443Z`` (d02_replay_2to3h_reverify.json:
FINITE_THROUGH_3H_PLUS).  This harness pins that L2 init so the BEFORE/AFTER
baseline is a stable, reproducible, validated workload.

Drives the TRUSTED ``run_forecast_operational_segmented`` (180-step segments,
same as the daily pipeline / segscan) and reports:
  * warmed per-step ms (full_wall / total_steps, steady-state),
  * cold compile-s of the inner segment,
  * peak GPU memory,
  * final-state finiteness + physical ranges (the stability gate).

Run:
  PYTHONPATH=src GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo OMP_NUM_THREADS=4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async GPUWRF_ACOUSTIC_UNROLL=2 taskset -c 0-3 \
    python proofs/v0100/wave_a_gate.py --hours 1 --out wave_a_after_u2_1h.json
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
from gpuwrf.runtime.operational_mode import (
    _advance_chunk,
    _enforce_operational_precision,
    run_forecast_operational_segmented,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/v0100")
# The v0.9.0 GREEN d02 stability/skill init (L2 72h run).
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"


def _peak_mb() -> float:
    try:
        return float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--segment-steps", type=int, default=180)
    ap.add_argument("--cadence", type=int, default=180)
    ap.add_argument("--out", type=str, default="wave_a_gate.json")
    ap.add_argument("--disable-guards", action="store_true",
                    help="bare-dycore stress test (production keeps guards ON)")
    args = ap.parse_args()
    hours = float(args.hours)
    seg_steps = int(args.segment_steps)
    cadence = int(args.cadence)
    disable_guards = bool(args.disable_guards)
    acoustic_unroll = int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))

    cfg = DailyPipelineConfig(
        hours=int(max(1, round(hours))), dt_s=10.0, acoustic_substeps=10,
        run_id=L2_RUN_ID, run_root=paths.wrf_l2_root(), domain="d02",
        radiation_cadence_steps=cadence,
    )
    case, run_dir = _build_real_case(cfg)
    # PRODUCTION config keeps guards ON (the operational finite-or-origin safety
    # net; _build_real_case comment).  The segscan/dycore-stress harness sets
    # disable_guards=True, which exposes a localized steep-terrain NaN that
    # PREDATES Wave-A (reproduced on pristine v0.9.0).  Default here = guards ON.
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True,
        disable_guards=disable_guards,
        radiation_cadence_steps=cadence, time_utc=case.run_start,
    )
    dt_s = float(nl.dt_s)
    steps = int(round(hours * 3600.0 / dt_s))

    print(f"=== Wave-A gate: L2 d02 {case.grid.ny}x{case.grid.nx}x{case.grid.nz} "
          f"+{hours}h={steps} steps unroll={acoustic_unroll} device={jax.devices()[0]} ===",
          flush=True)

    # Cold compile of one NON-radiation segment (the warmed-timing target; a
    # segment with no cadence hit isolates the dynamics+physics step launch cost).
    # indices start_step..start_step+seg-1 with no multiple of cadence.
    nonrad_seg = min(seg_steps, cadence - 1)
    carry0 = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )
    t0 = time.perf_counter()
    c1 = _advance_chunk(carry0, nl, jnp.asarray(1, dtype=jnp.int32),
                        n_steps=nonrad_seg, cadence=cadence)
    jax.block_until_ready(c1.state.theta)
    cold_seg_wall = time.perf_counter() - t0
    peak_compile = _peak_mb()

    # Warmed per-step: time warm calls of the SAME compiled non-radiation segment
    # (cache hits via traced start_step), min of repeats -> excludes compile.
    warm_ms = []
    c = c1
    for r in range(3):
        start = jnp.asarray(1 + (r + 1) * nonrad_seg, dtype=jnp.int32)
        t0 = time.perf_counter()
        c2 = _advance_chunk(c, nl, start, n_steps=nonrad_seg, cadence=cadence)
        jax.block_until_ready(c2.state.theta)
        warm_ms.append((time.perf_counter() - t0) / nonrad_seg * 1000.0)
        c = c2
    per_step_ms = float(min(warm_ms))

    # Full stability run via the trusted segmented entry (the real path).
    state0 = _build_real_case(cfg)[0].state
    t0 = time.perf_counter()
    final = run_forecast_operational_segmented(state0, nl, hours, segment_steps=seg_steps)
    jax.block_until_ready(final.theta)
    full_wall = time.perf_counter() - t0
    peak_full = _peak_mb()
    per_step_ms_amortized = (full_wall / float(steps)) * 1000.0

    def stat(name):
        a = np.asarray(jax.device_get(getattr(final, name)), dtype=np.float64)
        return {"finite": bool(np.isfinite(a).all()),
                "min": float(np.nanmin(a)), "max": float(np.nanmax(a))}

    ranges = {f: stat(f) for f in ("u", "v", "w", "theta", "qv", "p_total", "ph_total", "mu_total")}
    all_finite = all(r["finite"] for r in ranges.values())
    physical = (
        all_finite
        and abs(ranges["u"]["min"]) < 150 and abs(ranges["u"]["max"]) < 150
        and abs(ranges["v"]["min"]) < 150 and abs(ranges["v"]["max"]) < 150
        and abs(ranges["w"]["min"]) < 60 and abs(ranges["w"]["max"]) < 60
        and ranges["theta"]["min"] > 150 and ranges["theta"]["max"] < 600
        and ranges["mu_total"]["min"] > 0 and ranges["p_total"]["min"] > 0
        and ranges["qv"]["min"] >= -1e-6 and ranges["qv"]["max"] < 0.1
    )

    out = {
        "scope": "Wave-A gate: warmed per-step + coupled stability, GREEN d02 L2 init",
        "run_dir": str(run_dir),
        "run_id": L2_RUN_ID,
        "device": str(jax.devices()[0]),
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "config": {
            "force_fp64": bool(nl.force_fp64), "use_flux_advection": bool(nl.use_flux_advection),
            "epssm": float(nl.epssm), "top_lid": bool(nl.top_lid),
            "radiation_cadence_steps": cadence, "acoustic_unroll": acoustic_unroll,
        },
        "disable_guards": disable_guards,
        "hours": hours, "steps": steps, "segment_steps": seg_steps,
        "nonrad_segment_len": nonrad_seg,
        "inner_segment_cold_compile_s": cold_seg_wall,
        "warmed_per_step_ms": per_step_ms,
        "warmed_per_step_ms_samples": warm_ms,
        "warmed_per_step_ms_amortized_incl_radiation": per_step_ms_amortized,
        "full_wall_s_measured": full_wall,
        "peak_gpu_mem_mb_after_compile": peak_compile,
        "peak_gpu_mem_mb_after_full_run": peak_full,
        "final_state_ranges": ranges,
        "all_finite": all_finite,
        "physically_plausible": physical,
        "status": "PASS" if (all_finite and physical) else "FAIL",
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / args.out
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "warmed_per_step_ms", "inner_segment_cold_compile_s",
        "peak_gpu_mem_mb_after_full_run", "all_finite", "physically_plausible",
        "status")}, indent=2), flush=True)
    print(f"wrote {fn}", flush=True)
    return 0 if out["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
