"""v0.15 VIABILITY: TRUE all-fp32 step COST + VRAM proxy (x64 globally OFF).

The storage-matrix fp32 path that v0.15 actually wires (fp32_fp64_ladder.py)
leaves the acoustic perturbation solve fp64-PINNED (operational_mode.py hardcodes
~58 .astype(jnp.float64) calls), so it gives ~no speedup and ~no VRAM saving at
standard grid -- because the binding ~394-temporary arena stays fp64.

THIS probe answers the *other* half of the principal's question: if the acoustic
were un-pinned (a true all-fp32 operational step -- the unbuilt ADR-031 milestone),
what would the STEP COST and PEAK VRAM be?  We get a faithful COST/VRAM proxy by
disabling jax_enable_x64 globally: with x64 OFF, every .astype(jnp.float64) in the
step silently truncates to float32 (verified), so the WHOLE step -- acoustic arena
included -- runs in fp32.  The NUMERICS are wrong (the acoustic detonates, which is
exactly why the project pins it), but kernel COST and the VRAM working-set are a
faithful proxy for a true fp32 step.  We measure ms/step + peak VRAM and the
fp32/fp64 ratio at the 128x128 anchor and at large grids, and push to the OOM
ceiling -- the true-fp32 ceiling vs the fp64 ceiling is the scalability headline.

We must stop gpuwrf's ~20 import-time `config.update("jax_enable_x64", True)` calls
from re-enabling x64.  We do that by (a) NOT setting JAX_ENABLE_X64 and (b)
monkey-patching jax.config.update to ignore jax_enable_x64=True BEFORE importing
gpuwrf.

Run (wrapped by scripts/with_gpu_lock.sh) -- NOTE: JAX_ENABLE_X64 must be UNSET:
  PYTHONPATH=src:. XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/perf/v015/viability/true_fp32_cost_proxy.py
"""
from __future__ import annotations

import os
# Build the case with x64 ON (so the fp64-enforcing GridSpec/DycoreMetrics guards
# pass and the metrics/grid are valid).  We then TOGGLE x64 OFF for the step
# execution: with x64 off, every in-step .astype(jnp.float64) silently truncates
# to float32 (verified), so the WHOLE step -- the fp64-pinned acoustic arena
# included -- runs in fp32.  The numerics are GARBAGE by design (the acoustic
# detonates, which is exactly why the project pins it); only kernel COST and the
# VRAM working-set are measured, as a faithful proxy for a TRUE all-fp32 step.
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)   # ON for construction

import dataclasses
import json
import time
import traceback
from pathlib import Path

# Neutralize the fp64-ENFORCING contract guards so that after we toggle x64 OFF
# and cast to fp32, re-validation inside the step does not reject fp32 metrics.
# These guards (GridSpec/DycoreMetrics require fp64) are themselves part of WHY
# the fp64 architecture is embedded -- documenting that is part of the finding.
import gpuwrf.contracts.grid as _grid
def _dm_post_nofp64(self):
    try:
        if tuple(self.p_top.shape) not in ((), (1,)):
            raise ValueError("DycoreMetrics.p_top must be scalar or shape (1,)")
    except Exception:
        pass
_grid.DycoreMetrics.__post_init__ = _dm_post_nofp64
# GridSpec.__post_init__ also enforces fp64; wrap it to swallow that TypeError.
_orig_gs_post = _grid.GridSpec.__post_init__
def _gs_post_nofp64(self):
    try:
        _orig_gs_post(self)
    except TypeError as e:
        if "fp64" not in str(e):
            raise
_grid.GridSpec.__post_init__ = _gs_post_nofp64

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational

import proofs.perf.v015.viability.fp32_fp64_ab_bench as AB

OUT_DIR = Path("proofs/perf/v015/viability")
OUT_JSON = OUT_DIR / "true_fp32_cost_proxy.json"

ANCHOR_RUN_ROOT = "/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable"
ANCHOR_RUN_ID = "run_h36"
ANCHOR_DOMAIN = "d01"
ANCHOR_DT_S = 18.0

TILE_FACTORS = [
    (1, 1),   # 16,384
    (3, 3),   # 147,456
    (4, 4),   # 262,144  (past the fp64 ~168k OOM ceiling)
    (5, 5),   # 409,600
    (6, 6),   # 589,824
    (7, 7),   # 802,816
    (8, 8),   # 1,048,576
]
N_STEPS = 36


def _block(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


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


def _cast_all_fp32(state):
    from gpuwrf.contracts.state import State
    upd = {}
    for name in State.__slots__:
        v = getattr(state, name)
        if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating) and v.dtype != jnp.float32:
            upd[name] = v.astype(jnp.float32)
    return state.replace(**upd) if upd else state


def _deep_fp32(obj):
    """Cast every floating leaf of an arbitrary pytree to fp32 (x64 must be OFF)."""
    flat, treedef = jax.tree_util.tree_flatten(obj)
    out = []
    for x in flat:
        if hasattr(x, "dtype") and jnp.issubdtype(getattr(x, "dtype", jnp.int32), jnp.floating) and x.dtype != jnp.float32:
            out.append(x.astype(jnp.float32))
        else:
            out.append(x)
    return jax.tree_util.tree_unflatten(treedef, out)


def measure(base_state, base_nl, ny0, nx0, fy, fx):
    ny, nx = fy * ny0, fx * nx0
    ncol = ny * nx
    hours = N_STEPS * ANCHOR_DT_S / 3600.0
    # Tile while x64 is ON (preserves fp64 metrics), then toggle OFF and cast.
    nl = AB._tile_namelist(base_nl, ny0, nx0, fy, fx)
    nl = dataclasses.replace(nl, force_fp64=False)

    # >>> TOGGLE x64 OFF for the fp32 step execution <<<
    jax.config.update("jax_enable_x64", False)
    # Cast the namelist's grid/metrics/tendencies/radiation pytrees to fp32.
    nl = _deep_fp32(nl)

    def fresh_state():
        st = AB._tile_state(base_state, ny0, nx0, fy, fx)
        st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
        st = _deep_fp32(_cast_all_fp32(st))
        _block(st)
        return st

    rec = {"ny": ny, "nx": nx, "nz": int(base_nl.grid.nz), "ncol": ncol,
           "tile_factor": [fy, fx], "n_steps": N_STEPS,
           "x64_during_step": bool(jax.config.read("jax_enable_x64"))}
    try:
        _reset_peak()
        st = fresh_state()
        rec["state_theta_dtype_in"] = str(st.theta.dtype)
        rec["metrics_msftx_dtype_in"] = str(getattr(nl.grid.metrics, "msftx").dtype)
        t0 = time.perf_counter(); out = run_forecast_operational(st, nl, hours); _block(out)
        cold_s = time.perf_counter() - t0
        warms = []
        for _ in range(2):
            st = fresh_state()
            t0 = time.perf_counter(); out = run_forecast_operational(st, nl, hours); _block(out)
            warms.append(time.perf_counter() - t0)
        warm = min(warms)
        ms = warm / N_STEPS * 1000.0
        rec.update({
            "ran_ok": True, "oom": False, "cold_s": cold_s, "warm_s": warms,
            "ms_per_step": ms, "peak_vram_gib": _peak_gib(),
            "out_theta_dtype": str(out.theta.dtype),
            "out_theta_finite": bool(jnp.all(jnp.isfinite(out.theta))),
        })
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        is_oom = ("RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower()
                  or "OOM" in str(exc))
        rec.update({"ran_ok": False, "oom": bool(is_oom), "error": msg[:500],
                    "peak_vram_gib": _peak_gib()})
        rec["_tb"] = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
    finally:
        # toggle x64 back ON so the NEXT grid's tiling preserves fp64 metrics
        jax.config.update("jax_enable_x64", True)
    return rec


def measure_fp64_ref(base_state, base_nl, ny0, nx0, fy, fx):
    """All-fp64 reference in IDENTICAL harness/conditions (x64 stays ON)."""
    ny, nx = fy * ny0, fx * nx0
    ncol = ny * nx
    hours = N_STEPS * ANCHOR_DT_S / 3600.0
    nl = AB._tile_namelist(base_nl, ny0, nx0, fy, fx)
    nl = dataclasses.replace(nl, force_fp64=True)

    def fresh_state():
        st = AB._tile_state(base_state, ny0, nx0, fy, fx)
        st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
        st = AB._cast_state_all_fp64(st)
        _block(st)
        return st

    rec = {"ny": ny, "nx": nx, "ncol": ncol, "tile_factor": [fy, fx],
           "precision": "fp64_ref", "x64_during_step": bool(jax.config.read("jax_enable_x64"))}
    try:
        _reset_peak()
        st = fresh_state()
        t0 = time.perf_counter(); out = run_forecast_operational(st, nl, hours); _block(out)
        warms = []
        for _ in range(2):
            st = fresh_state()
            t0 = time.perf_counter(); out = run_forecast_operational(st, nl, hours); _block(out)
            warms.append(time.perf_counter() - t0)
        warm = min(warms)
        rec.update({"ran_ok": True, "oom": False, "ms_per_step": warm / N_STEPS * 1000.0,
                    "peak_vram_gib": _peak_gib(), "out_theta_dtype": str(out.theta.dtype),
                    "out_theta_finite": bool(jnp.all(jnp.isfinite(out.theta)))})
    except Exception as exc:  # noqa: BLE001
        is_oom = ("RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower())
        rec.update({"ran_ok": False, "oom": bool(is_oom), "error": f"{type(exc).__name__}: {exc}"[:400],
                    "peak_vram_gib": _peak_gib()})
    return rec


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = jax.devices()[0]
    # Build with x64 ON (guards pass); measure() toggles it OFF per-point for the
    # fp32 step and back ON afterwards.
    x64_build = bool(jax.config.read("jax_enable_x64"))
    print(f"[x64] build-time jax_enable_x64={x64_build} (measure() flips OFF for each fp32 step)", flush=True)
    cfg = DailyPipelineConfig(run_id=ANCHOR_RUN_ID, run_root=ANCHOR_RUN_ROOT,
                              domain=ANCHOR_DOMAIN, hours=1, dt_s=ANCHOR_DT_S, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    base_nl = case.namelist
    base_state = case.state
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    print(f"[base] grid {ny0}x{nx0}x{nz} dt={ANCHOR_DT_S}s device={dev} N={N_STEPS}", flush=True)

    records = []
    fp64_refs = []
    # In-harness fp64 references at the first 2 grids for an apples-to-apples ratio.
    for (fy, fx) in TILE_FACTORS[:2]:
        ny, nx = fy * ny0, fx * nx0
        print(f"[fp64-ref] ({fy},{fx}) {ny}x{nx} ncol={ny*nx} ...", flush=True)
        r64 = measure_fp64_ref(base_state, base_nl, ny0, nx0, fy, fx)
        fp64_refs.append(r64)
        if r64.get("ran_ok"):
            print(f"  fp64-ref ms/step={r64['ms_per_step']:.2f} VRAM={r64['peak_vram_gib']:.2f}G "
                  f"finite={r64['out_theta_finite']}", flush=True)
        else:
            print(f"  fp64-ref FAIL oom={r64.get('oom')} :: {r64.get('error','')[:120]}", flush=True)

    for (fy, fx) in TILE_FACTORS:
        ny, nx = fy * ny0, fx * nx0
        print(f"[measure] true-fp32 ({fy},{fx}) {ny}x{nx} ncol={ny*nx} ...", flush=True)
        rec = measure(base_state, base_nl, ny0, nx0, fy, fx)
        records.append(rec)
        OUT_JSON.write_text(json.dumps({"device": str(dev), "x64_build": x64_build,
                                        "fp64_refs": fp64_refs, "records": records}, indent=2) + "\n")
        if rec["ran_ok"]:
            print(f"  OK ms/step={rec['ms_per_step']:.2f} VRAM={rec['peak_vram_gib']:.2f}G "
                  f"theta={rec['out_theta_dtype']} finite={rec['out_theta_finite']}", flush=True)
        else:
            print(f"  FAIL oom={rec['oom']} :: {rec.get('error','')[:140]}", flush=True)
            if rec["oom"]:
                break

    # in-harness ratios at the matched grids
    inharness_ratios = []
    fp32_by_ncol = {r["ncol"]: r for r in records if r.get("ran_ok")}
    for r64 in fp64_refs:
        if r64.get("ran_ok") and r64["ncol"] in fp32_by_ncol:
            r32 = fp32_by_ncol[r64["ncol"]]
            inharness_ratios.append({
                "ncol": r64["ncol"],
                "fp64_ms": r64["ms_per_step"], "true_fp32_ms": r32["ms_per_step"],
                "true_fp32_speedup": r64["ms_per_step"] / r32["ms_per_step"],
                "fp64_gib": r64["peak_vram_gib"], "true_fp32_gib": r32["peak_vram_gib"],
                "fp64_over_fp32_vram": (r64["peak_vram_gib"] / r32["peak_vram_gib"]
                                        if r32["peak_vram_gib"] else None),
            })

    ok = [r for r in records if r.get("ran_ok")]
    payload = {
        "scope": "v0.15 VIABILITY: TRUE all-fp32 step COST+VRAM proxy (x64 globally OFF; numerics intentionally invalid, cost/VRAM faithful)",
        "device": str(dev), "x64_build_time": x64_build, "x64_during_step": False,
        "anchor": {"base_grid": f"{ny0}x{nx0}x{nz}", "base_ncol": ny0 * nx0, "dt_s": ANCHOR_DT_S},
        "caveat": "Numerics are GARBAGE by design (acoustic detonates in fp32); this measures ONLY the kernel COST and VRAM working-set a true all-fp32 step would have. Cross-reference fp64 numbers from fp32_fp64_ladder.json / grid_scaling.json.",
        "timing_method": f"compile at N={N_STEPS} steps; ms/step = best-of-2 warm wall / N.",
        "fp64_refs": fp64_refs,
        "inharness_true_fp32_vs_fp64_ratios": inharness_ratios,
        "records": records,
        "largest_true_fp32_grid": ({"ncol": ok[-1]["ncol"], "ny": ok[-1]["ny"], "nx": ok[-1]["nx"],
                                    "peak_vram_gib": ok[-1]["peak_vram_gib"], "ms_per_step": ok[-1]["ms_per_step"]}
                                   if ok else None),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    if inharness_ratios:
        print("\n--- in-harness fp64-ref vs true-fp32 (apples-to-apples) ---", flush=True)
        for r in inharness_ratios:
            print(f"  {r['ncol']:>7} cols: fp64 {r['fp64_ms']:.1f}ms / fp32 {r['true_fp32_ms']:.1f}ms "
                  f"= {r['true_fp32_speedup']:.2f}x  VRAM {r['fp64_gib']:.2f}->{r['true_fp32_gib']:.2f} GiB "
                  f"({r['fp64_over_fp32_vram']:.2f}x)" if r['fp64_over_fp32_vram'] else "", flush=True)

    print("\n============ TRUE fp32 cost/VRAM proxy ============", flush=True)
    print(f"{'ncol':>9} {'ms/step':>9} {'VRAM GiB':>9} {'finite':>7}")
    for r in records:
        if r.get("ran_ok"):
            print(f"{r['ncol']:>9d} {r['ms_per_step']:>9.2f} {r['peak_vram_gib']:>9.2f} {str(r['out_theta_finite']):>7}")
        else:
            print(f"{r['ncol']:>9d} {'OOM' if r['oom'] else 'ERR':>9} {r['peak_vram_gib']:>9.2f}")
    if payload["largest_true_fp32_grid"]:
        g = payload["largest_true_fp32_grid"]
        print(f"\nlargest true-fp32 grid: {g['ncol']} cols @ {g['peak_vram_gib']:.2f} GiB")
    print(f"\nwrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
