"""v0.15 VIABILITY: efficient fp32-vs-fp64 grid ladder (ONE compile per point).

Decisive question: does fp32 give a real speedup on the REAL coupled operational
step, and does the win GROW with grid size?  And does fp32 fit a MUCH larger grid
than fp64?

This is the compile-budget-efficient companion to fp32_fp64_ab_bench.py.  The
warmed two-hours-point method there needs TWO cold compiles per (grid,precision)
(~400 s each) -> impractical across a long grid ladder.  Here we compile ONCE per
(grid,precision) at a fixed step count N, then time warmed steps.  The fixed
per-call exec overhead is identical across precisions, so it cancels in the
fp32/fp64 RATIO (the number that matters).

Method (mirrors km_bench tiling + run_forecast_operational, the REAL merged step):
  * anchor = 128x128x44 dt=18s Switzerland d01 reinit-h36 (the kernel-char headline
    case: fp64 core ~70 ms/step here, full production 119.8 ms/step).
  * tile the State + namelist to larger (ny,nx) COST proxies; boundary/GWD/NoahMP
    OFF uniformly (apples-to-apples, identical to grid_scaling.json).
  * fp64 = force_fp64=True all-fp64 state (the SHIPPED default step).
  * fp32 = force_fp64=False, DEFAULT_DTYPES fp32-gated storage matrix (theta/u/v/qx
    fp32; acoustic perturbation solve stays fp64-PINNED by construction).  This is
    the RUNNABLE fp32 the architecture has today, NOT all-fp32 acoustic.
  * timing: compile at N steps (1 cold call), then 2 warm calls, ms/step =
    mean(warm)/N.  Push each precision up the ladder until OOM.

Run (wrapped by scripts/with_gpu_lock.sh):
  PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/perf/v015/viability/fp32_fp64_ladder.py
"""
from __future__ import annotations

import dataclasses
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER

import proofs.perf.v015.viability.fp32_fp64_ab_bench as AB  # reuse tilers

OUT_DIR = Path("proofs/perf/v015/viability")
OUT_JSON = OUT_DIR / "fp32_fp64_ladder.json"

ANCHOR_RUN_ROOT = "<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable"
ANCHOR_RUN_ID = "run_h36"
ANCHOR_DOMAIN = "d01"
ANCHOR_DT_S = 18.0

# ncol = fy*fx*16384.  Push past the fp64 ceiling into fp32-only territory.
TILE_FACTORS = [
    (1, 1),   # 16,384
    (2, 2),   # 65,536
    (2, 3),   # 98,304
    (3, 3),   # 147,456
    (3, 4),   # 196,608  (fp64 likely OOM near here)
    (4, 4),   # 262,144
    (4, 5),   # 327,680
    (5, 5),   # 409,600
    (5, 6),   # 491,520
    (6, 6),   # 589,824
    (6, 7),   # 688,128
]

N_STEPS = 36  # h = N*dt/3600 = 36*18/3600 = 0.18 h


def _block(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree
    )


def _peak_gib():
    dev = jax.devices()[0]
    try:
        return float(dev.memory_stats()["peak_bytes_in_use"]) / (1024.0**3)
    except Exception:
        return float("nan")


def _reset_peak():
    dev = jax.devices()[0]
    if hasattr(dev, "reset_memory_stats"):
        try:
            dev.reset_memory_stats(); return True
        except Exception:
            return False
    return False


def measure(base_state, base_nl, ny0, nx0, fy, fx, precision):
    ny, nx = fy * ny0, fx * nx0
    ncol = ny * nx
    hours = N_STEPS * ANCHOR_DT_S / 3600.0

    force_fp64 = (precision == "fp64")
    nl = AB._tile_namelist(base_nl, ny0, nx0, fy, fx)
    nl = dataclasses.replace(nl, force_fp64=force_fp64)

    def fresh_state():
        st = AB._tile_state(base_state, ny0, nx0, fy, fx)
        st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
        st = AB._cast_state_all_fp64(st) if force_fp64 else AB._cast_state_fp32_matrix(st)
        _block(st)
        return st

    rec = {"precision": precision, "ny": ny, "nx": nx, "nz": int(base_nl.grid.nz),
           "ncol": ncol, "tile_factor": [fy, fx], "n_steps": N_STEPS}
    try:
        _reset_peak()
        st = fresh_state()
        t0 = time.perf_counter(); out = run_forecast_operational(st, nl, hours); _block(out)
        cold_s = time.perf_counter() - t0
        warms = []
        for _ in range(2):
            st = fresh_state()
            t0 = time.perf_counter(); out = run_forecast_operational(st, nl, hours); _block(out)
            warms.append(time.perf_counter() - t0)
        warm = min(warms)  # best-of-2 removes a stray scheduler hiccup
        ms_per_step = warm / N_STEPS * 1000.0
        # finiteness sentinel on a couple of leaves
        finite = bool(jnp.all(jnp.isfinite(out.theta)) and jnp.all(jnp.isfinite(out.u)))
        rec.update({
            "ran_ok": True, "oom": False, "cold_s": cold_s, "warm_s": warms,
            "warm_min_s": warm, "ms_per_step": ms_per_step,
            "ms_per_forecast_hour": ms_per_step * (3600.0 / ANCHOR_DT_S),
            "peak_vram_gib": _peak_gib(), "out_finite": finite,
            "out_theta_dtype": str(out.theta.dtype), "out_u_dtype": str(out.u.dtype),
        })
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        is_oom = ("RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower()
                  or "OOM" in str(exc))
        rec.update({"ran_ok": False, "oom": bool(is_oom), "error": msg[:500],
                    "peak_vram_gib": _peak_gib()})
        rec["_tb"] = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
    return rec


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = jax.devices()[0]
    cfg = DailyPipelineConfig(run_id=ANCHOR_RUN_ID, run_root=ANCHOR_RUN_ROOT,
                              domain=ANCHOR_DOMAIN, hours=1, dt_s=ANCHOR_DT_S, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    base_nl = case.namelist
    base_state = case.state
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    print(f"[base] grid {ny0}x{nx0}x{nz} dt={ANCHOR_DT_S}s device={dev} N={N_STEPS}", flush=True)

    records = []
    fp64_dead = fp32_dead = False
    for (fy, fx) in TILE_FACTORS:
        ny, nx = fy * ny0, fx * nx0
        for precision in ("fp64", "fp32"):
            if (precision == "fp64" and fp64_dead) or (precision == "fp32" and fp32_dead):
                continue
            print(f"[measure] {precision} ({fy},{fx}) {ny}x{nx} ncol={ny*nx} ...", flush=True)
            rec = measure(base_state, base_nl, ny0, nx0, fy, fx, precision)
            records.append(rec)
            # flush to disk after each point so partial progress is never lost
            OUT_JSON.write_text(json.dumps({"device": str(dev), "records": records}, indent=2) + "\n")
            if rec["ran_ok"]:
                print(f"  OK {precision} ms/step={rec['ms_per_step']:.2f} VRAM={rec['peak_vram_gib']:.2f}G "
                      f"finite={rec['out_finite']}", flush=True)
            else:
                print(f"  FAIL {precision} oom={rec['oom']} :: {rec.get('error','')[:140]}", flush=True)
                if rec["oom"]:
                    if precision == "fp64": fp64_dead = True
                    else: fp32_dead = True
        if fp64_dead and fp32_dead:
            break

    # pair ratios
    by = {}
    for r in records:
        if r.get("ran_ok"):
            by.setdefault(r["ncol"], {})[r["precision"]] = r
    ratios = []
    for ncol in sorted(by):
        p = by[ncol]
        if "fp32" in p and "fp64" in p:
            ratios.append({
                "ncol": ncol,
                "fp64_ms": p["fp64"]["ms_per_step"], "fp32_ms": p["fp32"]["ms_per_step"],
                "fp32_speedup_over_fp64": p["fp64"]["ms_per_step"] / p["fp32"]["ms_per_step"],
                "fp64_gib": p["fp64"]["peak_vram_gib"], "fp32_gib": p["fp32"]["peak_vram_gib"],
                "fp64_over_fp32_vram": (p["fp64"]["peak_vram_gib"] / p["fp32"]["peak_vram_gib"]
                                        if p["fp32"]["peak_vram_gib"] else None),
            })
    fp64_ok = [r for r in records if r.get("ran_ok") and r["precision"] == "fp64"]
    fp32_ok = [r for r in records if r.get("ran_ok") and r["precision"] == "fp32"]

    payload = {
        "scope": "v0.15 VIABILITY: fp32 vs fp64 grid ladder on the REAL merged operational step (1 compile/point)",
        "device": str(dev),
        "anchor": {"run_root": ANCHOR_RUN_ROOT, "run_id": ANCHOR_RUN_ID, "domain": ANCHOR_DOMAIN,
                   "base_grid": f"{ny0}x{nx0}x{nz}", "base_ncol": ny0 * nx0, "dt_s": ANCHOR_DT_S},
        "precision_semantics": {
            "fp64": "force_fp64=True, all-fp64 carried state = the SHIPPED v0.15 default step",
            "fp32": "force_fp64=False, DEFAULT_DTYPES fp32-gated storage matrix; acoustic perturbation solve fp64-PINNED by construction. Runnable mixed-precision, NOT all-fp32 acoustic.",
        },
        "timing_method": f"compile at N={N_STEPS} steps; ms/step = best-of-2 warm wall / N. Fixed per-call overhead cancels in the fp32/fp64 ratio.",
        "config_notes": {"run_boundary": False, "gwd_opt": 0, "use_noahmp": False,
                         "acoustic_substeps": int(base_nl.acoustic_substeps)},
        "records": records,
        "fp32_vs_fp64_ratios": ratios,
        "largest_fp64_grid": ({"ncol": fp64_ok[-1]["ncol"], "ny": fp64_ok[-1]["ny"], "nx": fp64_ok[-1]["nx"],
                               "peak_vram_gib": fp64_ok[-1]["peak_vram_gib"], "ms_per_step": fp64_ok[-1]["ms_per_step"]}
                              if fp64_ok else None),
        "largest_fp32_grid": ({"ncol": fp32_ok[-1]["ncol"], "ny": fp32_ok[-1]["ny"], "nx": fp32_ok[-1]["nx"],
                               "peak_vram_gib": fp32_ok[-1]["peak_vram_gib"], "ms_per_step": fp32_ok[-1]["ms_per_step"]}
                              if fp32_ok else None),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n============ fp32 vs fp64 LADDER ============", flush=True)
    print(f"{'ncol':>9} {'fp64 ms':>9} {'fp32 ms':>9} {'fp32 x':>8} {'fp64 GiB':>9} {'fp32 GiB':>9} {'VRAM x':>7}")
    for r in ratios:
        vr = r["fp64_over_fp32_vram"]
        print(f"{r['ncol']:>9d} {r['fp64_ms']:>9.2f} {r['fp32_ms']:>9.2f} {r['fp32_speedup_over_fp64']:>8.3f} "
              f"{r['fp64_gib']:>9.2f} {r['fp32_gib']:>9.2f} {('%.3f'%vr) if vr else '-':>7}")
    if payload["largest_fp64_grid"]:
        g = payload["largest_fp64_grid"]; print(f"\nlargest fp64: {g['ncol']} cols @ {g['peak_vram_gib']:.2f} GiB")
    if payload["largest_fp32_grid"]:
        g = payload["largest_fp32_grid"]; print(f"largest fp32: {g['ncol']} cols @ {g['peak_vram_gib']:.2f} GiB")
    print(f"\nwrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
