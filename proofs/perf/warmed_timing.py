"""Task 1 -- WARMED timing breakdown of the coupled real-case operational scan.

Separates (a) one-time COMPILE cost from (b) warmed steady-state throughput so we
can answer: is the +3h's ~32 min cost compile-bound or throughput-bound?

Method (all GPU/JAX, taskset -c 0-3, OMP_NUM_THREADS=4):
  * Build the SAME real d02 case the +1h/+3h skill proof used
    (gpuwrf.integration.daily_pipeline._build_real_case): physics ON, guards ON,
    fp64, flux advection, damp_opt=3, top_lid, epssm=0.5, radiation cadence 180.
  * ``run_forecast_operational(state, nl, hours)`` is ONE compiled scan whose step
    count is a STATIC arg, so each distinct ``hours`` is a distinct compile. To
    isolate per-step throughput from the fixed compile we:
      - measure COLD wall (compile + execute) at a small hours value h1,
      - measure WARM wall (cache hit, re-execute) at the SAME h1 (=> compile_s =
        cold - warm at h1, and warm_h1 = fixed_exec_overhead + n1*per_step),
      - measure COLD+WARM at a second hours value h2 (different static => its own
        compile); the WARM marginal (warm_h2 - warm_h1)/(n2 - n1) = per-step
        warmed wallclock, free of the fixed per-call overhead.
  * Peak device memory via device.memory_stats()['peak_bytes_in_use'].

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 \
    python proofs/perf/warmed_timing.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational

PROOF = Path("proofs/perf")
CPU_WRF_DENOM_NOTE = (
    "Speedup denominator = 28-rank CPU WRF on this workstation (memory baseline: "
    "previous Gen2 attempt was ~4.8x SLOWER than 28-rank CPU WRF on an incomplete "
    "dycore). No CPU-WRF wallclock is re-measured here; speedup framing is deferred "
    "to the dedicated perf sprint. This artifact reports ABSOLUTE warmed GPU wall."
)


def _block(state) -> None:
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state
    )


def _peak_mb() -> float:
    dev = jax.devices()[0]
    try:
        return float(dev.memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        return float("nan")


def _time_run(state, nl, hours: float) -> float:
    t0 = time.perf_counter()
    out = run_forecast_operational(state, nl, float(hours))
    _block(out)
    return time.perf_counter() - t0


def main() -> int:
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    nl = case.namelist
    dt_s = float(nl.dt_s)
    state = case.state

    # Two static-hours points. h1 small enough to compile fast; (h2-h1) large
    # enough that the warmed marginal per-step is the dominant term. Both are
    # exact integer-step multiples of dt=10s. donate_argnums(0) means each call
    # consumes its input buffer; rebuild the case state per timed call so we
    # always feed a live buffer.
    h1 = 0.1   # 36 steps
    h2 = 0.5   # 180 steps
    n1 = int(round(h1 * 3600.0 / dt_s))
    n2 = int(round(h2 * 3600.0 / dt_s))

    results: dict[str, object] = {
        "scope": "Task 1 warmed timing breakdown (compile vs throughput) -- coupled real d02",
        "run_dir": str(run_dir),
        "init_utc": str(case.run_start),
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "device": str(jax.devices()[0]),
        "config": {
            "dt_s": dt_s,
            "acoustic_substeps": int(nl.acoustic_substeps),
            "radiation_cadence_steps": int(nl.radiation_cadence_steps),
            "run_physics": bool(nl.run_physics),
            "run_boundary": bool(nl.run_boundary),
            "disable_guards": bool(nl.disable_guards),
            "force_fp64": bool(nl.force_fp64),
            "use_flux_advection": bool(nl.use_flux_advection),
            "epssm": float(nl.epssm),
            "top_lid": bool(nl.top_lid),
            "w_damping": int(nl.w_damping),
            "damp_opt": int(nl.damp_opt),
        },
        "cpu_wrf_denominator": CPU_WRF_DENOM_NOTE,
    }

    # --- h1: cold then warm ---
    cold_h1 = _time_run(state, nl, h1)
    peak_after_h1_compile = _peak_mb()
    # rebuild state (donated buffer consumed)
    state = _build_real_case(cfg)[0].state
    warm_h1 = _time_run(state, nl, h1)

    # --- h2: cold then warm ---
    state = _build_real_case(cfg)[0].state
    cold_h2 = _time_run(state, nl, h2)
    peak_after_h2_compile = _peak_mb()
    state = _build_real_case(cfg)[0].state
    warm_h2 = _time_run(state, nl, h2)
    # second warm sample at h2 for variance
    state = _build_real_case(cfg)[0].state
    warm_h2_b = _time_run(state, nl, h2)

    compile_h1 = cold_h1 - warm_h1
    compile_h2 = cold_h2 - warm_h2
    warm_h2_mean = 0.5 * (warm_h2 + warm_h2_b)

    # Marginal warmed per-step: removes fixed per-call exec overhead (dispatch,
    # initial_operational_carry, halo, output materialization).
    per_step_warm_s = (warm_h2_mean - warm_h1) / float(n2 - n1)
    per_step_warm_ms = per_step_warm_s * 1000.0
    fixed_call_overhead_s = warm_h1 - n1 * per_step_warm_s
    ms_per_forecast_hour = per_step_warm_ms * (3600.0 / dt_s)

    def extrap_hours(fc_hours: float) -> dict[str, float]:
        steps = fc_hours * 3600.0 / dt_s
        warm_s = fixed_call_overhead_s + steps * per_step_warm_s
        return {
            "steps": steps,
            "warmed_wall_s": warm_s,
            "warmed_wall_min": warm_s / 60.0,
        }

    extrap = {f"{h}h": extrap_hours(h) for h in (3.0, 6.0, 12.0, 24.0, 48.0, 72.0)}
    # NOTE: a single compiled scan for 24-72h is one static-hours compile. Compile
    # scales weakly with step count (the body is identical; only the trip count
    # differs), so we report the measured compile at h1/h2 and treat 24-72h
    # compile as ~ the same one-time cost (re-measured in the dedicated run below
    # would dominate only once). Per-case warmed wall is what multiplies by 30.
    ensemble_30 = {
        f"{h}h": {
            "per_case_warmed_min": extrap[f"{h}h"]["warmed_wall_min"],
            "ensemble_30_warmed_min": extrap[f"{h}h"]["warmed_wall_min"] * 30.0,
            "ensemble_30_warmed_hours": extrap[f"{h}h"]["warmed_wall_min"] * 30.0 / 60.0,
        }
        for h in (24.0, 72.0)
    }

    results["timing"] = {
        "h1_hours": h1, "n1_steps": n1,
        "h2_hours": h2, "n2_steps": n2,
        "cold_h1_wall_s": cold_h1, "warm_h1_wall_s": warm_h1,
        "cold_h2_wall_s": cold_h2, "warm_h2_wall_s": warm_h2, "warm_h2_b_wall_s": warm_h2_b,
        "compile_s_at_h1": compile_h1,
        "compile_s_at_h2": compile_h2,
        "fixed_per_call_exec_overhead_s": fixed_call_overhead_s,
        "warmed_per_step_ms": per_step_warm_ms,
        "warmed_ms_per_forecast_hour": ms_per_forecast_hour,
    }
    results["peak_gpu_mem_mb"] = {
        "after_h1_compile": peak_after_h1_compile,
        "after_h2_compile": peak_after_h2_compile,
    }
    results["extrapolation_warmed"] = extrap
    results["ensemble_30_case"] = ensemble_30
    results["verdict_inputs"] = {
        "compile_dominates_at_short_runs": bool(compile_h1 > warm_h1),
        "warmed_per_step_ms": per_step_warm_ms,
    }

    PROOF.mkdir(parents=True, exist_ok=True)
    out_path = PROOF / "warmed_timing.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["timing"], indent=2))
    print(json.dumps(results["peak_gpu_mem_mb"], indent=2))
    print("24h:", json.dumps(extrap["24.0h"]))
    print("72h:", json.dumps(extrap["72.0h"]))
    print("ensemble:", json.dumps(ensemble_30, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
