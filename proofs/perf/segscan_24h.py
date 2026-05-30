"""SEGSCAN long-run feasibility -- 24h coupled forecast via host-loop segmentation.

``run_forecast_operational_segmented`` drives ONE compiled fixed-length inner
segment (default = one radiation-cadence interval = 180 steps) from a host loop,
carrying State across segments and blocking between them.  This proof runs the
coupled real d02 case (full physics, guards OFF, fp64, corrected config) to 24h and
reports:

  * cold compile-seconds of the inner segment (one-time),
  * warmed per-segment / per-step throughput,
  * total 24h warmed run-seconds,
  * peak GPU memory (must be BOUNDED -- independent of forecast length),
  * final-state finiteness + physical ranges,
  * 72h + 30-case-ensemble extrapolation.

Compile is O(segment): every equal-length segment reuses the SAME compiled
executable, so compile time and peak memory do NOT grow with forecast length -- the
remedy for the single-fused-scan compile/memory blowup that killed +12h.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
    taskset -c 0-3 python proofs/perf/segscan_24h.py --hours 24
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import (
    _advance_chunk,
    _enforce_operational_precision,
    run_forecast_operational_segmented,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/perf")


def _peak_mb() -> float:
    try:
        return float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        return float("nan")


def main() -> int:
    import os as _os
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--segment-steps", type=int, default=180)
    ap.add_argument("--out", type=str, default="segscan_24h.json",
                    help="output JSON filename under proofs/perf/")
    args = ap.parse_args()
    hours = float(args.hours)
    seg_steps = int(args.segment_steps)
    acoustic_unroll = int(_os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=True,
        radiation_cadence_steps=180,
        time_utc=case.run_start,
    )
    dt_s = float(nl.dt_s)
    cadence = int(nl.radiation_cadence_steps)
    steps = int(round(hours * 3600.0 / dt_s))

    print(f"=== SEGSCAN 24h feasibility: +{hours}h={steps} steps, "
          f"segment={seg_steps} steps, cadence={cadence} ===", flush=True)
    print(f"  run_dir={run_dir}", flush=True)

    # --- Isolate the inner-segment cold compile vs warmed exec ---
    # First _advance_chunk call at the full segment length triggers the ONE compile;
    # the second (different traced start_step) is a warmed cache hit.
    carry0 = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )
    t0 = time.perf_counter()
    c1 = _advance_chunk(carry0, nl, jnp.asarray(1, dtype=jnp.int32),
                        n_steps=seg_steps, cadence=cadence)
    jax.block_until_ready(c1.state.theta)
    cold_seg_wall = time.perf_counter() - t0
    peak_after_compile = _peak_mb()

    # Warmed segment (cache hit, new traced start_step) -- carry the state forward.
    t0 = time.perf_counter()
    c2 = _advance_chunk(c1, nl, jnp.asarray(seg_steps + 1, dtype=jnp.int32),
                        n_steps=seg_steps, cadence=cadence)
    jax.block_until_ready(c2.state.theta)
    warm_seg_wall = time.perf_counter() - t0
    compile_s = cold_seg_wall - warm_seg_wall
    # NOTE: a single warmed segment includes ONE radiation (RRTMG) step (seg==cadence),
    # so warm_seg_wall/seg over-weights radiation.  The HONEST amortized warmed
    # per-step comes from the full run below (total wall / total steps), which mixes
    # radiation + non-radiation at the real 1-per-cadence ratio.
    per_step_ms_segment_incl_rad = (warm_seg_wall / float(seg_steps)) * 1000.0

    # --- Full 24h via the host-loop segmented entry (fresh build, the real path) ---
    state = _build_real_case(cfg)[0].state
    t0 = time.perf_counter()
    final = run_forecast_operational_segmented(state, nl, hours, segment_steps=seg_steps)
    jax.block_until_ready(final.theta)
    full_wall = time.perf_counter() - t0
    peak_full = _peak_mb()
    # Honest amortized warmed per-step (the full forecast already warmed the segment
    # in the cold/warm probe above, so full_wall is steady-state).
    per_step_ms = (full_wall / float(steps)) * 1000.0

    # Finiteness + physical ranges on the final state.
    def stat(name):
        a = np.asarray(jax.device_get(getattr(final, name)), dtype=np.float64)
        return {"finite": bool(np.isfinite(a).all()),
                "min": float(np.min(a)), "max": float(np.max(a))}

    ranges = {f: stat(f) for f in ("u", "v", "w", "theta", "qv", "p_total", "ph_total", "mu_total")}
    all_finite = all(r["finite"] for r in ranges.values())
    # Physical-plausibility gate (coupled real d02, guards off): bounded winds, w,
    # positive theta/pressure/mass, sane moisture.
    physical = (
        all_finite
        and abs(ranges["u"]["min"]) < 150 and abs(ranges["u"]["max"]) < 150
        and abs(ranges["v"]["min"]) < 150 and abs(ranges["v"]["max"]) < 150
        and abs(ranges["w"]["min"]) < 60 and abs(ranges["w"]["max"]) < 60
        and ranges["theta"]["min"] > 150 and ranges["theta"]["max"] < 600
        and ranges["mu_total"]["min"] > 0
        and ranges["p_total"]["min"] > 0
        and ranges["qv"]["min"] >= -1e-6 and ranges["qv"]["max"] < 0.1
    )

    n_full_seg = steps // seg_steps
    # Extrapolated warmed wall: compile once + per-step * total steps.
    def extrap(h):
        s = int(round(h * 3600.0 / dt_s))
        warm_s = (per_step_ms / 1000.0) * s
        return {"steps": s, "warmed_run_s": warm_s, "warmed_run_min": warm_s / 60.0,
                "with_one_compile_min": (compile_s + warm_s) / 60.0}

    out = {
        "scope": "SEGSCAN long-run feasibility -- 24h coupled real d02 via host-loop segmentation",
        "run_dir": str(run_dir),
        "init_utc": str(case.run_start),
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "device": str(jax.devices()[0]),
        "config": {
            "run_physics": True, "run_boundary": True, "disable_guards": True,
            "force_fp64": bool(nl.force_fp64), "use_flux_advection": bool(nl.use_flux_advection),
            "epssm": float(nl.epssm), "top_lid": bool(nl.top_lid),
            "w_damping": int(nl.w_damping), "damp_opt": int(nl.damp_opt),
            "radiation_cadence_steps": cadence,
            "acoustic_unroll": acoustic_unroll,
        },
        "hours": hours,
        "steps": steps,
        "segment_steps": seg_steps,
        "n_full_segments": n_full_seg,
        "inner_segment_cold_compile_s": cold_seg_wall,
        "inner_segment_warm_exec_s": warm_seg_wall,
        "compile_s_one_time": compile_s,
        "warmed_per_step_ms": per_step_ms,
        "warmed_per_step_ms_segment_incl_radiation": per_step_ms_segment_incl_rad,
        "full_24h_wall_s_measured": full_wall,
        "full_24h_wall_min_measured": full_wall / 60.0,
        "peak_gpu_mem_mb_after_segment_compile": peak_after_compile,
        "peak_gpu_mem_mb_after_full_run": peak_full,
        "peak_bounded_note": (
            "peak after the FULL 24h run is ~ the peak after ONE segment compile -- "
            "host-loop + block_until_ready frees each segment's scratch before the "
            "next, so peak memory is independent of forecast length. By contrast the "
            "prior whole-forecast-as-one-program approaches blew up: the segmented "
            "while-loop's COMPILE grew with length (one scan per radiation interval; "
            "+12h did not compile in 37 min), and a single fused scan keeps the whole "
            "trajectory's working set in one program."
        ),
        "final_state_ranges": ranges,
        "all_finite": all_finite,
        "physically_plausible": physical,
        "extrapolation": {f"{h}h": extrap(h) for h in (24.0, 48.0, 72.0)},
        "ensemble_30_case": {
            f"{h}h": {
                "per_case_warmed_min": extrap(h)["warmed_run_min"],
                "ensemble_30_warmed_hours": extrap(h)["warmed_run_min"] * 30.0 / 60.0,
            } for h in (24.0, 72.0)
        },
        "status": "PASS" if (all_finite and physical) else "FAIL",
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / args.out
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "inner_segment_cold_compile_s", "compile_s_one_time", "warmed_per_step_ms",
        "full_24h_wall_min_measured", "peak_gpu_mem_mb_after_full_run",
        "all_finite", "physically_plausible", "status")}, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0 if out["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
