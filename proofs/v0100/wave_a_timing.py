"""Wave-A warmed-timing + kernel-count proxy + daily-wrapper host breakdown.

Captures a clean BEFORE/AFTER for the v0.10.0 Wave-A kernel-optimization sprint.

Workload: coupled real d02 (159x66x44), fp64 (force_fp64=True), dt=10s,
10 acoustic substeps, RRTMG cadence 180 (the operational config from
``_build_real_case`` -> the SAME path daily_pipeline drives).

Measures:
  * warmed per-step ms via a long single compiled segment (>=200 steps), amortized.
    The segment spans an integer number of radiation cadences so the per-step
    number mixes radiation + non-radiation at the true 1/cadence ratio.
  * a NON-radiation warmed per-step (segment with no cadence hit) to isolate the
    dynamics+physics step from the RRTMG outlier (the launch-count target).
  * kernel-count PROXY: count XLA HLO fusion/op instances in the compiled
    ``_advance_chunk`` (one-step) lowering -- a deterministic, nsys-free launch
    proxy that drops when fusion lands.
  * daily-wrapper host breakdown: forecast / finite-summary-D2H / M9-diag /
    output-pack-D2H / netcdf / land-refresh for one production-style forecast hour.

Run:
  PYTHONPATH=src GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo OMP_NUM_THREADS=4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async taskset -c 0-3 \
    python proofs/v0100/wave_a_timing.py --out wave_a_before.json
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import (
    _build_real_case,
    DailyPipelineConfig,
    finite_summary,
)
from gpuwrf.runtime.operational_mode import (
    _advance_chunk,
    _enforce_operational_precision,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/v0100")


def _peak_mb() -> float:
    try:
        return float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        return float("nan")


# Coarse XLA HLO op-instance taxonomy for a launch-count proxy.  We count
# fusion/loop/elementwise/copy/transpose ops in the optimized HLO text.  This is
# a deterministic, nsys-free proxy: the absolute number is NOT a kernel count,
# but its DELTA across a fusion change tracks the launch-count drop.
_OP_PATTERNS = {
    "fusion": re.compile(r"\bfusion\b|fusion\("),
    "loop_fusion": re.compile(r"kLoop|loop_.*fusion"),
    "copy": re.compile(r"\bcopy\b|copy-start|copy-done|\bcopy\("),
    "transpose": re.compile(r"\btranspose\b"),
    "convert": re.compile(r"\bconvert\b"),
    "dynamic_update_slice": re.compile(r"dynamic-update-slice"),
    "pad": re.compile(r"\bpad\b"),
    "concatenate": re.compile(r"\bconcatenate\b"),
}


def _hlo_op_counts(advance_step_fn, carry, nl) -> dict:
    """Count HLO op instances in the optimized one-step lowering (launch proxy)."""
    try:
        lowered = advance_step_fn.lower(
            carry, nl, jnp.asarray(1, dtype=jnp.int32)
        )
        compiled = lowered.compile()
        # Optimized HLO (post-fusion) -- the closest static proxy to launches.
        try:
            hlo = compiled.as_text()
        except Exception:
            hlo = lowered.as_text()
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}
    counts = {}
    total_lines = 0
    for line in hlo.splitlines():
        total_lines += 1
    for name, pat in _OP_PATTERNS.items():
        counts[name] = len(pat.findall(hlo))
    # Fusion-instruction count: lines that are a root fusion computation call.
    fusion_calls = len(re.findall(r"=\s*\S+\s+fusion\(", hlo))
    counts["fusion_call_sites"] = fusion_calls
    counts["hlo_text_lines"] = total_lines
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="wave_a_before.json")
    ap.add_argument("--warm-steps", type=int, default=360,
                    help="long-segment length for amortized per-step (>=200; "
                         "default 360 = 2 cadences)")
    ap.add_argument("--cadence", type=int, default=180)
    ap.add_argument("--repeats", type=int, default=3,
                    help="warmed timing repeats (min reported)")
    args = ap.parse_args()

    cadence = int(args.cadence)
    warm_steps = int(args.warm_steps)
    acoustic_unroll = int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=cadence, time_utc=case.run_start,
    )
    dt_s = float(nl.dt_s)

    print(f"=== Wave-A timing: grid {case.grid.ny}x{case.grid.nx}x{case.grid.nz} "
          f"device={jax.devices()[0]} unroll={acoustic_unroll} ===", flush=True)

    carry0 = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )

    # --- HLO op-count proxy on a ONE-step compiled advance (launch proxy) ---
    one_step = lambda c, n, s: _advance_chunk(c, n, s, n_steps=1, cadence=cadence)
    one_step_j = jax.jit(one_step, static_argnames=())
    hlo_counts = _hlo_op_counts(one_step_j, carry0, nl)
    print("HLO one-step op counts:", json.dumps(hlo_counts), flush=True)

    # --- warmed per-step: long NON-radiation segment (start AFTER a cadence) ---
    # start_step=1: indices 1..warm_steps, cadence hit only at multiples of 180.
    # To get a pure non-radiation segment, start at 1 and length 179 (no mult of 180).
    nonrad_len = cadence - 1  # 179 steps, indices 1..179, no multiple of 180
    seg_nonrad = jax.jit(
        lambda c, n, s: _advance_chunk(c, n, s, n_steps=nonrad_len, cadence=cadence)
    )
    # compile + 1 warm pass
    c = seg_nonrad(carry0, nl, jnp.asarray(1, dtype=jnp.int32))
    jax.block_until_ready(c.state.theta)
    peak_compile = _peak_mb()
    nonrad_ms = []
    for r in range(int(args.repeats)):
        t0 = time.perf_counter()
        c2 = seg_nonrad(c, nl, jnp.asarray(1 + (r + 1) * nonrad_len, dtype=jnp.int32))
        jax.block_until_ready(c2.state.theta)
        nonrad_ms.append((time.perf_counter() - t0) / nonrad_len * 1000.0)
        c = c2
    per_step_ms_nonrad = float(min(nonrad_ms))

    # --- warmed per-step: amortized over a cadence-spanning segment (incl rad) ---
    seg_full = jax.jit(
        lambda c, n, s: _advance_chunk(c, n, s, n_steps=warm_steps, cadence=cadence)
    )
    # use a fresh carry so start_step lands cadence hits at the right phase
    carry_fresh = initial_operational_carry(
        _enforce_operational_precision(_build_real_case(cfg)[0].state,
                                       force_fp64=bool(nl.force_fp64))
    )
    cf = seg_full(carry_fresh, nl, jnp.asarray(cadence, dtype=jnp.int32))  # compile+warm
    jax.block_until_ready(cf.state.theta)
    peak_full = _peak_mb()
    full_ms = []
    for r in range(int(args.repeats)):
        t0 = time.perf_counter()
        cf2 = seg_full(cf, nl, jnp.asarray(cadence + (r + 1) * warm_steps, dtype=jnp.int32))
        jax.block_until_ready(cf2.state.theta)
        full_ms.append((time.perf_counter() - t0) / warm_steps * 1000.0)
        cf = cf2
    per_step_ms_amortized = float(min(full_ms))

    # --- daily-wrapper host breakdown for one production-style forecast hour ---
    # Re-uses the live operational outputs path components. We time:
    #   forecast (GPU, 360 steps=1h via 2 segments) / finite-summary (full-State
    #   D2H) / output-pack D2H (prepare payload device_get) / netcdf write.
    host_break = {}
    try:
        from gpuwrf.integration.daily_pipeline import (
            _surface_diagnostics_for_output,
        )
        from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
        from datetime import timedelta
        # advance 1 forecast hour (360 steps) through the same segment exec
        steps_1h = int(round(3600.0 / dt_s))
        state_h = case.state
        state_h = _enforce_operational_precision(state_h, force_fp64=bool(nl.force_fp64))
        carry_h = initial_operational_carry(state_h)
        t0 = time.perf_counter()
        ch = seg_full(carry_h, nl, jnp.asarray(1, dtype=jnp.int32))
        jax.block_until_ready(ch.state.theta)
        # one more partial segment to reach 360 if warm_steps<360
        n_done = warm_steps
        while n_done < steps_1h:
            take = min(warm_steps, steps_1h - n_done)
            if take == warm_steps:
                ch = seg_full(ch, nl, jnp.asarray(1 + n_done, dtype=jnp.int32))
            else:
                ch = _advance_chunk(ch, nl, jnp.asarray(1 + n_done, dtype=jnp.int32),
                                    n_steps=take, cadence=cadence)
            jax.block_until_ready(ch.state.theta)
            n_done += take
        host_break["forecast_1h_s"] = time.perf_counter() - t0
        state_out = ch.state
        # finite summary (full-State D2H)
        t0 = time.perf_counter()
        fs = finite_summary(state_out)
        host_break["finite_summary_d2h_s"] = time.perf_counter() - t0
        host_break["finite_summary_all_finite"] = bool(fs["all_finite"])
        # M9 surface diagnostics
        t0 = time.perf_counter()
        diags = _surface_diagnostics_for_output(
            state_out, case.namelist, case.run_start, lead_seconds=3600.0)
        jax.block_until_ready(getattr(diags, "t2", getattr(diags, "T2", jnp.asarray(0.0)))
                              if diags is not None else jnp.asarray(0.0))
        host_break["m9_diagnostics_s"] = time.perf_counter() - t0
        # output pack D2H + netcdf write
        valid_time = case.run_start + timedelta(hours=1)
        tmp_out = PROOF / "_tmp_wrfout_timing"
        t0 = time.perf_counter()
        write_wrfout_netcdf(
            state_out, case.grid, case.namelist, tmp_out,
            valid_time=valid_time, lead_hours=1.0, run_start=case.run_start,
            diagnostics=diags,
        )
        host_break["output_pack_and_netcdf_s"] = time.perf_counter() - t0
        try:
            tmp_out.unlink()
        except Exception:
            pass
    except Exception as exc:
        host_break["error"] = repr(exc)

    out = {
        "scope": "Wave-A warmed-timing + HLO launch proxy + daily-wrapper host breakdown",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "config": {
            "force_fp64": bool(nl.force_fp64),
            "acoustic_substeps": int(nl.acoustic_substeps),
            "radiation_cadence_steps": cadence,
            "acoustic_unroll": acoustic_unroll,
            "dt_s": dt_s,
        },
        "warm_steps_segment": warm_steps,
        "nonrad_segment_len": nonrad_len,
        "repeats": int(args.repeats),
        "warmed_per_step_ms_nonradiation": per_step_ms_nonrad,
        "warmed_per_step_ms_amortized_incl_radiation": per_step_ms_amortized,
        "nonrad_ms_samples": nonrad_ms,
        "amortized_ms_samples": full_ms,
        "peak_gpu_mem_mb_after_compile": peak_compile,
        "peak_gpu_mem_mb_after_full": peak_full,
        "hlo_one_step_op_counts": hlo_counts,
        "daily_wrapper_host_breakdown_1h": host_break,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / args.out
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "warmed_per_step_ms_nonradiation",
        "warmed_per_step_ms_amortized_incl_radiation",
        "peak_gpu_mem_mb_after_full", "hlo_one_step_op_counts",
        "daily_wrapper_host_breakdown_1h")}, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
