"""v0.15 VIABILITY: the decisive fp32-vs-fp64 A/B on the REAL merged operational step.

Question the principal needs answered with MEASURED evidence:
  Does fp32 give a real speedup on the *real* coupled workload (dycore + physics),
  not just micro-kernels, and does the win grow with grid size?  And does fp32 fit
  a MUCH larger grid than fp64's ~126-168k-col ceiling?

METHOD (mirrors the trusted km_bench/grid_scaling_bench.py exactly):
  * Build the real Switzerland/d02 case ONCE via daily_pipeline._build_real_case.
  * Spatially TILE the State + namelist to larger (ny,nx); physics/dycore kernel
    COST depends only on array SHAPES (ncol,nz), not values -> a faithful cost proxy.
  * boundary/GWD/NoahMP OFF uniformly so the law is apples-to-apples on the
    dycore+radiation+PBL core (identical to the committed grid_scaling.json).
  * Warmed two-hours-point marginal timing: per_step=(warm_h2-warm_h1)/(n2-n1).

PRECISION AXIS (the new variable):
  * "fp64"  = namelist.force_fp64=True, state carried all-fp64.  This is EXACTLY
              the shipped v0.15 default operational step.
  * "fp32"  = namelist.force_fp64=False, state leaves cast to fp32 where the
              operational DEFAULT_DTYPES matrix authorises fp32 storage.  This is
              the *runnable* mixed-precision path actually wired in v0.15
              (theta/u/v/qx fp32 storage; the acoustic perturbation solve stays
              fp64-pinned by construction -- operational_mode.py lines ~1253-1455
              hardcode .astype(float64) because "fp32 detonates the acoustic").
  So the "fp32" column here is the HONEST best-case the current architecture can
  actually RUN today, NOT a hypothetical all-fp32 acoustic (which the project has
  documented detonates).  The gap between this and a true all-fp32 step is the
  acoustic pin, quantified separately by the dycore micro-roofline.

At each grid we record BOTH precisions' warmed ms/step + peak VRAM, the fp32/fp64
ratio, and we push each precision up the grid ladder until OOM to expose the
ceiling difference.

Run (wrapped by scripts/with_gpu_lock.sh):
  PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/perf/v015/viability/fp32_fp64_ab_bench.py
"""
from __future__ import annotations

import dataclasses
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER

OUT_DIR = Path("proofs/perf/v015/viability")
OUT_JSON = OUT_DIR / "fp32_fp64_ab.json"

# Anchor case = the SAME 128x128x44 dt=18s Switzerland d01 reinit-h36 case behind
# the kernel-characterization headline (119.8 ms/step fp64; CPU 24-rank = 200.5
# ms/step -> 1.67x).  base ncol = 16384.
ANCHOR_RUN_ROOT = "/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable"
ANCHOR_RUN_ID = "run_h36"
ANCHOR_DOMAIN = "d01"
ANCHOR_DT_S = 18.0

# Tile ladder from the 128x128=16384 base.  Push past the fp64 ceiling so fp32
# can show a bigger ceiling.  ncol = fy*fx*16384.
TILE_FACTORS = [
    (1, 1),   # 16,384   (the 128x128 anchor)
    (1, 2),   # 32,768
    (2, 2),   # 65,536
    (2, 3),   # 98,304
    (3, 3),   # 147,456  (near the fp64 ~126-168k ceiling)
    (3, 4),   # 196,608  (fp64 likely OOMs here)
    (4, 4),   # 262,144  (fp32-only territory)
    (4, 5),   # 327,680
    (5, 5),   # 409,600
    (5, 6),   # 491,520
    (6, 6),   # 589,824
]


# --------------------------------------------------------------------------- #
# Tiling helpers (copied verbatim from km_bench/grid_scaling_bench.py)
# --------------------------------------------------------------------------- #
def _tile_axis(arr, axis, factor, staggered):
    if factor == 1:
        return arr
    if not staggered:
        reps = [1] * arr.ndim
        reps[axis] = factor
        return jnp.tile(arr, reps)
    n = arr.shape[axis] - 1
    interior = jax.lax.slice_in_dim(arr, 0, n, axis=axis)
    edge = jax.lax.slice_in_dim(arr, n, n + 1, axis=axis)
    reps = [1] * arr.ndim
    reps[axis] = factor
    tiled_interior = jnp.tile(interior, reps)
    return jnp.concatenate([tiled_interior, edge], axis=axis)


def _tile_horizontal(arr, ny, nx, fy, fx):
    if arr.ndim < 2:
        return arr
    yax, xax = arr.ndim - 2, arr.ndim - 1
    ysz, xsz = arr.shape[yax], arr.shape[xax]
    y_ok = ysz in (ny, ny + 1)
    x_ok = xsz in (nx, nx + 1)
    if not (y_ok and x_ok):
        return arr
    out = _tile_axis(arr, yax, fy, staggered=(ysz == ny + 1))
    out = _tile_axis(out, xax, fx, staggered=(xsz == nx + 1))
    return out


def _tile_state(state, ny, nx, fy, fx):
    from gpuwrf.contracts.state import State
    old_side = max(nx + 1, ny + 1)
    new_side = max(fx * nx + 1, fy * ny + 1)
    leaves = {}
    for name in State.__slots__:
        v = getattr(state, name)
        if not hasattr(v, "shape"):
            leaves[name] = v
            continue
        shp = tuple(v.shape)
        if name.endswith("_bdy"):
            if shp[-1] == old_side and new_side != old_side:
                lastax = v.ndim - 1
                interior = jax.lax.slice_in_dim(v, 0, old_side - 1, axis=lastax)
                edge = jax.lax.slice_in_dim(v, old_side - 1, old_side, axis=lastax)
                target_interior = new_side - 1
                base_interior = old_side - 1
                reps = [1] * v.ndim
                reps[lastax] = (target_interior + base_interior - 1) // base_interior
                tiled = jnp.tile(interior, reps)
                tiled = jax.lax.slice_in_dim(tiled, 0, target_interior, axis=lastax)
                leaves[name] = jnp.concatenate([tiled, edge], axis=lastax)
            else:
                leaves[name] = v
            continue
        leaves[name] = _tile_horizontal(v, ny, nx, fy, fx)
    return State(**leaves)


def _tile_child_pytree(child, ny, nx, fy, fx):
    if child is None:
        return None
    flat, treedef = jax.tree_util.tree_flatten(child)
    new = [
        _tile_horizontal(x, ny, nx, fy, fx) if hasattr(x, "shape") else x
        for x in flat
    ]
    return jax.tree_util.tree_unflatten(treedef, new)


def _scaled_grid(grid, ny, nx, fy, fx):
    proj = dataclasses.replace(grid.projection, nx=fx * nx, ny=fy * ny)
    terr = dataclasses.replace(grid.terrain, shape=(fy * ny, fx * nx))
    terrain_height = _tile_horizontal(grid.terrain_height, ny, nx, fy, fx)
    metrics = _tile_child_pytree(grid.metrics, ny, nx, fy, fx)
    return dataclasses.replace(
        grid, projection=proj, terrain=terr,
        terrain_height=terrain_height, metrics=metrics,
    )


def _tile_namelist(nl, ny, nx, fy, fx):
    new_grid = _scaled_grid(nl.grid, ny, nx, fy, fx)
    new_metrics = _tile_child_pytree(nl.metrics, ny, nx, fy, fx)
    new_tend = _tile_child_pytree(nl.tendencies, ny, nx, fy, fx)
    new_rad = _tile_child_pytree(nl.radiation_static, ny, nx, fy, fx)
    return dataclasses.replace(
        nl, grid=new_grid, metrics=new_metrics, tendencies=new_tend,
        radiation_static=new_rad, gwdo_statics=None, gwd_opt=0,
        use_noahmp=False, noahmp_static=None, run_boundary=False,
    )


# --------------------------------------------------------------------------- #
# Precision: cast the tiled state to the operational fp32 matrix
# --------------------------------------------------------------------------- #
def _cast_state_fp32_matrix(state):
    """Cast state leaves to the operational DEFAULT_DTYPES fp32-gated matrix.

    This is the SAME precision matrix _enforce_operational_precision applies when
    force_fp64=False: fp32-authorised fields (theta/u/v/qx...) -> fp32, locked
    fields (mass/pressure/geopotential/accumulation) stay fp64.  Running this on
    a tiled state + force_fp64=False is the runnable mixed-precision operational
    step.
    """
    updates = {}
    for field in STATE_FIELD_ORDER:
        if not hasattr(state, field):
            continue
        value = getattr(state, field)
        if not hasattr(value, "dtype"):
            continue
        target = DEFAULT_DTYPES.dtype_for(field)
        if value.dtype != target:
            updates[field] = value.astype(target)
    return state.replace(**updates) if updates else state


def _cast_state_all_fp64(state):
    from gpuwrf.contracts.state import State
    updates = {}
    for name in State.__slots__:
        v = getattr(state, name)
        if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating) and v.dtype != jnp.float64:
            updates[name] = v.astype(jnp.float64)
    return state.replace(_cast=False, **updates) if updates else state.replace(_cast=False)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
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
            dev.reset_memory_stats()
            return True
        except Exception:
            return False
    return False


def _time_run(state, nl, hours):
    t0 = time.perf_counter()
    out = run_forecast_operational(state, nl, float(hours))
    _block(out)
    return time.perf_counter() - t0


def measure(base_state, base_nl, ny0, nx0, fy, fx, dt_s, precision):
    ny, nx = fy * ny0, fx * nx0
    ncol = ny * nx
    h1, h2 = 0.05, 0.2
    n1 = int(round(h1 * 3600.0 / dt_s))
    n2 = int(round(h2 * 3600.0 / dt_s))

    force_fp64 = (precision == "fp64")
    nl = _tile_namelist(base_nl, ny0, nx0, fy, fx)
    nl = dataclasses.replace(nl, force_fp64=force_fp64)

    def fresh_state():
        st = _tile_state(base_state, ny0, nx0, fy, fx)
        st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
        st = _cast_state_all_fp64(st) if force_fp64 else _cast_state_fp32_matrix(st)
        _block(st)
        return st

    rec = {"precision": precision, "ny": ny, "nx": nx, "nz": int(base_nl.grid.nz),
           "ncol": ncol, "tile_factor": [fy, fx]}
    try:
        _reset_peak()
        st = fresh_state(); _time_run(st, nl, h1)          # cold compile h1
        st = fresh_state(); warm_h1 = _time_run(st, nl, h1)
        st = fresh_state(); _time_run(st, nl, h2)          # cold compile h2
        st = fresh_state(); wa = _time_run(st, nl, h2)
        st = fresh_state(); wb = _time_run(st, nl, h2)
        warm_h2 = 0.5 * (wa + wb)
        per_step_ms = (warm_h2 - warm_h1) / float(n2 - n1) * 1000.0
        rec.update({
            "ran_ok": True, "oom": False,
            "warm_h1_s": warm_h1, "warm_h2a_s": wa, "warm_h2b_s": wb,
            "warmed_ms_per_step": per_step_ms,
            "ms_per_forecast_hour": per_step_ms * (3600.0 / dt_s),
            "peak_vram_gib": _peak_gib(),
        })
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        is_oom = ("RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower()
                  or "OOM" in str(exc))
        rec.update({"ran_ok": False, "oom": bool(is_oom), "error": msg[:600],
                    "peak_vram_gib": _peak_gib()})
        rec["_tb"] = "".join(traceback.format_exc().splitlines(keepends=True)[-8:])
    return rec


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = jax.devices()[0]
    cfg = DailyPipelineConfig(
        run_id=ANCHOR_RUN_ID, run_root=ANCHOR_RUN_ROOT, domain=ANCHOR_DOMAIN,
        hours=1, dt_s=ANCHOR_DT_S, acoustic_substeps=10,
    )
    case, _ = _build_real_case(cfg)
    base_nl = case.namelist
    base_state = case.state
    dt_s = float(base_nl.dt_s)
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    print(f"[base] grid {ny0}x{nx0}x{nz} dt={dt_s}s device={dev}", flush=True)

    records = []
    fp64_dead = False
    fp32_dead = False
    for (fy, fx) in TILE_FACTORS:
        ny, nx = fy * ny0, fx * nx0
        for precision in ("fp64", "fp32"):
            if precision == "fp64" and fp64_dead:
                continue
            if precision == "fp32" and fp32_dead:
                continue
            print(f"[measure] {precision} tile ({fy},{fx}) -> {ny}x{nx} ncol={ny*nx} ...", flush=True)
            rec = measure(base_state, base_nl, ny0, nx0, fy, fx, dt_s, precision)
            records.append(rec)
            if rec["ran_ok"]:
                print(f"  OK  {precision} ms/step={rec['warmed_ms_per_step']:.2f} "
                      f"peakVRAM={rec['peak_vram_gib']:.2f}G", flush=True)
            else:
                print(f"  FAIL {precision} oom={rec['oom']} :: {rec.get('error','')[:160]}", flush=True)
                if rec["oom"]:
                    if precision == "fp64":
                        fp64_dead = True
                    else:
                        fp32_dead = True
        if fp64_dead and fp32_dead:
            break

    by_ncol = {}
    for r in records:
        if r.get("ran_ok"):
            by_ncol.setdefault(r["ncol"], {})[r["precision"]] = r
    ratios = []
    for ncol in sorted(by_ncol):
        pair = by_ncol[ncol]
        if "fp32" in pair and "fp64" in pair:
            ratio = pair["fp64"]["warmed_ms_per_step"] / pair["fp32"]["warmed_ms_per_step"]
            vram_ratio = pair["fp64"]["peak_vram_gib"] / pair["fp32"]["peak_vram_gib"]
            ratios.append({
                "ncol": ncol,
                "fp64_ms_per_step": pair["fp64"]["warmed_ms_per_step"],
                "fp32_ms_per_step": pair["fp32"]["warmed_ms_per_step"],
                "fp32_speedup_over_fp64": ratio,
                "fp64_peak_gib": pair["fp64"]["peak_vram_gib"],
                "fp32_peak_gib": pair["fp32"]["peak_vram_gib"],
                "fp64_over_fp32_vram": vram_ratio,
            })

    fp64_ok = [r for r in records if r.get("ran_ok") and r["precision"] == "fp64"]
    fp32_ok = [r for r in records if r.get("ran_ok") and r["precision"] == "fp32"]
    payload = {
        "scope": "v0.15 VIABILITY: fp32 vs fp64 A/B on the REAL merged operational step",
        "device": str(dev),
        "base_grid": {"ny": ny0, "nx": nx0, "nz": nz, "ncol": ny0 * nx0},
        "dt_s": dt_s,
        "precision_semantics": {
            "fp64": "force_fp64=True, all-fp64 carried state = the SHIPPED v0.15 default step",
            "fp32": "force_fp64=False, DEFAULT_DTYPES fp32-gated matrix (theta/u/v/qx fp32 storage); acoustic perturbation solve stays fp64-PINNED by construction (operational_mode.py ~1253-1455). This is the runnable mixed-precision step, NOT all-fp32 acoustic.",
        },
        "config_notes": {
            "run_boundary": False, "gwd_opt": 0, "use_noahmp": False,
            "acoustic_substeps": int(base_nl.acoustic_substeps),
            "note": "Same COST-proxy tiling + warmed two-hours-point method as committed km_bench/grid_scaling.json; only the precision axis is added.",
        },
        "records": records,
        "fp32_vs_fp64_ratios": ratios,
        "largest_fp64_grid": (
            {"ncol": fp64_ok[-1]["ncol"], "ny": fp64_ok[-1]["ny"], "nx": fp64_ok[-1]["nx"],
             "peak_vram_gib": fp64_ok[-1]["peak_vram_gib"], "ms_per_step": fp64_ok[-1]["warmed_ms_per_step"]}
            if fp64_ok else None),
        "largest_fp32_grid": (
            {"ncol": fp32_ok[-1]["ncol"], "ny": fp32_ok[-1]["ny"], "nx": fp32_ok[-1]["nx"],
             "peak_vram_gib": fp32_ok[-1]["peak_vram_gib"], "ms_per_step": fp32_ok[-1]["warmed_ms_per_step"]}
            if fp32_ok else None),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n==================== fp32 vs fp64 A/B ====================", flush=True)
    print(f"{'ncol':>9} {'fp64 ms':>9} {'fp32 ms':>9} {'fp32 x':>8} {'fp64 GiB':>9} {'fp32 GiB':>9}")
    for r in ratios:
        print(f"{r['ncol']:>9d} {r['fp64_ms_per_step']:>9.2f} {r['fp32_ms_per_step']:>9.2f} "
              f"{r['fp32_speedup_over_fp64']:>8.3f} {r['fp64_peak_gib']:>9.2f} {r['fp32_peak_gib']:>9.2f}")
    if payload["largest_fp64_grid"]:
        g = payload["largest_fp64_grid"]
        print(f"\nlargest fp64 grid: {g['ncol']} cols @ {g['peak_vram_gib']:.2f} GiB")
    if payload["largest_fp32_grid"]:
        g = payload["largest_fp32_grid"]
        print(f"largest fp32 grid: {g['ncol']} cols @ {g['peak_vram_gib']:.2f} GiB")
    print(f"\nwrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
