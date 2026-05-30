"""XLA scheduling-flag A/B probe for the launch-bound coupled dycore step.

The compute-cycle analysis (proofs/perf/compute_cycle_analysis.md) found the
per-step forecast is launch-bound: ~6900 tiny (~1us) elementwise stencil kernels
+ ~3900 memory ops/step, GPU idle 43-68% waiting between dependent launches. The
ONLY safe >=10x lever is cutting that kernel/launch COUNT -- which is precision-
invariant.  This harness measures the effect of XLA GPU SCHEDULING flags (CUDA
command buffers / graphs + the latency-hiding scheduler) that batch many tiny
dependent launches into one graph submission, removing per-launch host overhead
and the idle gaps -- WITHOUT changing any arithmetic.

It is a pure-scheduling A/B: each invocation runs under whatever ``XLA_FLAGS`` the
process was launched with, measures warmed per-step (marginal of two segmented
chunk lengths so the fixed dispatch/compile is differenced out), and saves the
final-state field arrays.  Two runs (different XLA_FLAGS, same GPUWRF_ACOUSTIC_
UNROLL) are then compared: a scheduling-only flag MUST be bitwise-identical.

Memory-robust: uses the SEGMENTED entry (``_advance_chunk``, the bounded-memory
path), and the measured window uses a LARGE radiation cadence so the ~15 GiB
RRTMG g-point transient never co-allocates with the launch-bound dynamics+couplers
we are probing (command buffers help the dynamics, not the isolated RRTMG jit).
The full dynamics + Thompson + surface + MYNN couplers (the launch-bound bulk)
still run every step.

Run (one config per invocation):
  XLA_FLAGS="" PYTHONPATH=src OMP_NUM_THREADS=2 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async taskset -c 0-3 \
    python proofs/perf/fusion_flag_probe.py --tag base
  XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL --xla_gpu_graph_min_graph_size=1 --xla_gpu_enable_latency_hiding_scheduler=true" \
    ... python proofs/perf/fusion_flag_probe.py --tag cb
  python proofs/perf/fusion_flag_probe.py --compare base cb
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

PROOF = Path("proofs/perf")
FIELDS = ("u", "v", "w", "theta", "qv", "p_total", "ph_total", "mu_total")


def _run_one(tag: str, n_small: int, n_big: int, reps: int, physics: bool) -> int:
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk,
        _enforce_operational_precision,
    )
    from gpuwrf.runtime.operational_state import initial_operational_carry

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    # ``physics=True``  -> full coupled launch profile (dynamics + Thompson + surface
    #                      + MYNN; radiation cadence raised so the ~15 GiB RRTMG
    #                      transient never co-allocates), needs an ~8 GiB physics
    #                      transient -> OOMs when the GPU is shared with a 15 GiB agent.
    # ``physics=False`` -> the launch-bound DYCORE ALONE (the ~6900 micro-kernels that
    #                      dominate the launch tax and are the PRIMARY command-buffer
    #                      target), ~3-5 GiB -> fits under contention.  This is the
    #                      cleaner isolation of the scheduling-flag effect.
    nl = dataclasses.replace(
        case.namelist, run_physics=bool(physics), run_boundary=True, disable_guards=True,
        radiation_cadence_steps=10_000_000, time_utc=case.run_start,
    )
    cadence = int(nl.radiation_cadence_steps)

    # _advance_chunk does NOT donate its carry (only the public run_forecast_*
    # entries donate), so ONE built carry can be reused across all timed runs --
    # avoids the slow repeated _build_real_case (netCDF read + interpolation).
    base_carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )

    def chunk(carry, start, n):
        return _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                              n_steps=int(n), cadence=cadence)

    # Warm both compiled lengths (start_step is TRACED -> one compile per length).
    c = chunk(base_carry, 1, n_small); jax.block_until_ready(c.state.theta)
    c = chunk(base_carry, 1 + n_small, n_big); jax.block_until_ready(c.state.theta)

    def best_wall(n, start, k):
        best = 1e9
        for _ in range(k):
            jax.block_until_ready(base_carry.state.theta)
            t0 = time.perf_counter()
            c1 = chunk(base_carry, start, n)
            jax.block_until_ready(c1.state.theta)
            best = min(best, time.perf_counter() - t0)
        return best

    ws = best_wall(n_small, 1, reps)
    wb = best_wall(n_big, 1 + n_small, reps)
    per_step_ms = (wb - ws) / float(n_big - n_small) * 1000.0

    # Final state after n_small steps (the SAME compiled chunk) for cross-flag diff.
    cf = chunk(base_carry, 1, n_small)
    jax.block_until_ready(cf.state.theta)
    arrs = {f: np.asarray(jax.device_get(getattr(cf.state, f)), dtype=np.float64) for f in FIELDS}

    peak_mb = float("nan")
    try:
        peak_mb = float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        pass

    PROOF.mkdir(parents=True, exist_ok=True)
    np.savez(PROOF / f"fusion_flag_probe_{tag}.npz", **arrs)
    meta = {
        "tag": tag,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "acoustic_unroll": os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"),
        "run_physics": bool(physics),
        "device": str(jax.devices()[0]),
        "n_small": n_small, "n_big": n_big, "reps": reps,
        "warm_small_s": ws, "warm_big_s": wb,
        "warmed_per_step_ms": per_step_ms,
        "peak_gpu_mem_mb": peak_mb,
        "run_dir": str(run_dir),
        "radiation_in_window": False,
    }
    (PROOF / f"fusion_flag_probe_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


def _compare(tag_a: str, tag_b: str) -> int:
    a = np.load(PROOF / f"fusion_flag_probe_{tag_a}.npz")
    b = np.load(PROOF / f"fusion_flag_probe_{tag_b}.npz")
    ma = json.loads((PROOF / f"fusion_flag_probe_{tag_a}.json").read_text())
    mb = json.loads((PROOF / f"fusion_flag_probe_{tag_b}.json").read_text())
    max_abs = {f: float(np.max(np.abs(a[f] - b[f]))) for f in FIELDS}
    max_rel = {}
    for f in FIELDS:
        denom = float(np.max(np.abs(a[f]))) or 1.0
        max_rel[f] = max_abs[f] / denom
    bitwise = max(max_abs.values()) == 0.0
    speedup = ma["warmed_per_step_ms"] / mb["warmed_per_step_ms"] if mb["warmed_per_step_ms"] else float("nan")
    out = {
        "scope": f"XLA scheduling-flag A/B: {tag_a} vs {tag_b} (coupled real d02, segmented, no-radiation window)",
        "flags_a": ma["xla_flags"], "flags_b": mb["xla_flags"],
        "unroll_a": ma["acoustic_unroll"], "unroll_b": mb["acoustic_unroll"],
        "baseline_per_step_ms": ma["warmed_per_step_ms"],
        "candidate_per_step_ms": mb["warmed_per_step_ms"],
        "speedup": speedup,
        "max_abs_diff_per_field": max_abs,
        "max_rel_diff_per_field": max_rel,
        "bitwise_identical": bitwise,
        "verdict": ("SAFE+FASTER" if (bitwise and speedup > 1.02)
                    else "SAFE-NO-GAIN" if (bitwise and speedup <= 1.02)
                    else "ROUNDOFF+FASTER" if (max(max_rel.values()) < 1e-12 and speedup > 1.02)
                    else "ROUNDOFF-NO-GAIN" if max(max_rel.values()) < 1e-12
                    else "RESULT-CHANGED"),
    }
    (PROOF / f"fusion_flag_probe_verdict_{tag_a}_{tag_b}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--n-small", type=int, default=20)
    ap.add_argument("--n-big", type=int, default=80)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--no-physics", action="store_true",
                    help="isolate the launch-bound dycore (fits under GPU contention)")
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        return _compare(args.compare[0], args.compare[1])
    tag = args.tag or "probe"
    return _run_one(tag, args.n_small, args.n_big, args.reps, physics=not args.no_physics)


if __name__ == "__main__":
    raise SystemExit(main())
