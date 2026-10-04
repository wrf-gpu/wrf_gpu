"""Compile-pathology kill-gate (a): BouLac O(nz) as DEFAULT in the full operational jit.

Reproduces / falsifies the v0.15 XLA "Very slow compile" pathology documented in
proofs/perf/v015/boulac_nz_optimization.json. Forces the O(nz) free-atmosphere
length-scale search (`_boulac_length_onz`) as the BouLac algorithm for the WHOLE
operational forecast jit and measures, against the frozen v0.14 manifest:

  * full-pipeline COLD compile wall  (s)        -- PASS gate: < ~120 s
  * peak process RSS during compile  (GB)
  * steady runtime per step          (s/step)   -- vs the dense baseline (no regression)
  * tiered field identity            (RMSE)     -- O(nz) vs dense (== frozen v0.14 default)

The harness runs in TWO process-isolated phases (driven by run_killgate.sh):
  --algo dense : the production default (== v0.14-frozen behavior), reference dump + baseline compile/runtime
  --algo onz   : GPUWRF_MYNN_BOULAC_ONZ engaged, the pathology candidate

A hard --compile-timeout (default 1800 s) caps the cold-compile probe so the
known-pathological case fails fast instead of hanging the GPU lock. If the cold
compile exceeds the timeout the run is recorded as STILL-PATHOLOGICAL.

GPU lock required (run only via scripts/with_gpu_lock.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
HERE = Path(__file__).resolve().parent
MANIFEST = Path("proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json").resolve()


def _peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


class _RssSampler(threading.Thread):
    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _peak_rss_gb())
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.peak = max(self.peak, _peak_rss_gb())


class _CompileTimeout(Exception):
    pass


def _force_algo(algo: str) -> str:
    """Force the BouLac length-scale algorithm for the WHOLE operational jit.

    Patches the module-level gate BEFORE daily_pipeline/mynn_pbl is used to build
    or trace the forecast, so the chosen branch is the one XLA lowers.
    """
    # Env var first (in case anything snapshots it at import); module global is
    # the authoritative call-time gate the dispatcher reads.
    os.environ["GPUWRF_MYNN_BOULAC_ONZ"] = "1" if algo == "onz" else "0"
    import gpuwrf.physics.mynn_pbl as m
    if algo == "onz":
        m._MYNN_BOULAC_ONZ = True
        return "onz_default_in_full_operational_jit"
    m._MYNN_BOULAC_ONZ = False
    return "dense_nznz_frozen_v014_default"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dense", "onz"], required=True)
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--compile-timeout", type=int, default=1800)
    args = ap.parse_args()

    os.environ.setdefault("GPUWRF_MYNN_BOULAC_FP32", "0")  # production default

    sampler = _RssSampler()
    sampler.start()

    algo_desc = _force_algo(args.algo)
    from gpuwrf.integration import daily_pipeline as dp

    # The real case lives in PROBE/run_h36 (the v0.15 tiered-gate Switzerland d01
    # reinit-h36 case); run_id selects that subdir, run_root=PROBE is its parent.
    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=args.hours,
        output_dir=Path(f"/tmp/v022_boulac_killgate/{args.algo}"),
        proof_dir=Path(f"/tmp/v022_boulac_killgate/{args.algo}/proofs"),
        run_root=PROBE, domain="d01",
    )
    def _fresh_segment_input(hour: int):
        # The operational forecast donates state buffers to JAX. Build a fresh
        # input tree per call so warm timing does not reuse deleted arrays.
        case, _run_dir = dp._build_real_case(config)
        state = case.state
        boundary_leaves = dp._capture_boundary_leaves(state, case.namelist)
        window_s = dp._boundary_window_cadence_s(case.namelist)
        record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)
        if boundary_leaves:
            state = dp._rewindow_boundary_leaves(
                state, boundary_leaves, segment_start_s=(hour - 1) * 3600.0,
                record_cadence_s=record_s, window_s=window_s,
            )
        return case, state

    # --- COLD compile + first run (with a hard timeout to fail fast on pathology) ---
    slow_alarm = {"fired": False}
    compile_timed_out = False
    cold_wall = None

    def _on_timeout(signum, frame):
        raise _CompileTimeout()

    prev = signal.signal(signal.SIGALRM, _on_timeout)
    try:
        case1, st1 = _fresh_segment_input(1)
        signal.alarm(args.compile_timeout)
        t0 = time.perf_counter()
        out1 = dp._default_forecast_fn(st1, case1.namelist, 1.0)
        cold_wall = round(time.perf_counter() - t0, 3)
    except _CompileTimeout:
        compile_timed_out = True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)

    summary = {
        "algo": args.algo,
        "algo_desc": algo_desc,
        "hours": args.hours,
        "compile_timeout_s": args.compile_timeout,
    }

    if compile_timed_out:
        sampler.stop()
        summary.update({
            "verdict": "STILL-PATHOLOGICAL",
            "cold_compile_plus_run_s": None,
            "cold_compile_timed_out": True,
            "peak_rss_gb": round(sampler.peak, 3),
            "note": (
                f"cold compile+run did not complete within {args.compile_timeout}s "
                "-> XLA slow-compile pathology reproduced (or absolute compile wall "
                "exceeds the probe cap)"
            ),
        })
        (HERE / f"killgate_{args.algo}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    # --- Steady runtime (warm): second hour reuses the compiled program ---
    warm_walls = []
    n_warm = 2
    last_case = case1
    for hour in range(1, n_warm + 1):
        case, st = _fresh_segment_input(hour)
        t0 = time.perf_counter()
        out = dp._default_forecast_fn(st, case.namelist, 1.0)
        warm_walls.append(round(time.perf_counter() - t0, 3))
        last_case = case
    sampler.stop()

    # 1 forecast-hour at dt=18 s == 200 steps.
    steps_per_hour = 200
    dt_s = float(getattr(last_case.namelist, "dt_s", 18.0))
    if dt_s > 0:
        steps_per_hour = int(round(3600.0 / dt_s))
    warm_min = min(warm_walls)
    steady_s_per_step = round(warm_min / steps_per_hour, 5)
    # Cold compile wall = cold(compile+run) - warm(run-only); the warm hour is the
    # same program with no recompile, so the delta is the lowering+codegen cost.
    compile_wall = round(cold_wall - warm_min, 3)

    # --- Dump final state for tiered identity ---
    leaves = {}
    for name, value in dp._field_items(out):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            leaves[name] = arr
    out_npz = HERE / f"killgate_{args.algo}_state.npz"
    np.savez_compressed(out_npz, **leaves)

    summary.update({
        "verdict": "COMPLETED",
        "cold_compile_plus_run_s": cold_wall,
        "cold_compile_timed_out": False,
        "compile_wall_s": compile_wall,
        "warm_run_walls_s": warm_walls,
        "warm_run_min_s": warm_min,
        "steps_per_hour": steps_per_hour,
        "dt_s": dt_s,
        "steady_s_per_step": steady_s_per_step,
        "peak_rss_gb": round(sampler.peak, 3),
        "state_npz": str(out_npz),
        "n_leaves": len(leaves),
        "warm_input_mode": "fresh_real_case_per_call_to_avoid_reusing_jax_donated_arrays",
    })
    (HERE / f"killgate_{args.algo}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
