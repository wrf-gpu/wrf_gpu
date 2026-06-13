"""v0.15 km-bench: GPU step-throughput + peak-VRAM scaling vs column count.

Question: does a LARGER horizontal grid saturate the GPU better than the small
128x128 test grid, or is the warmed production step launch-bound (sublinear in
ncol)?  We MEASURE warmed ms/step and peak VRAM at several grid sizes and fit a
per-column scaling law ms/step ~= a + b*ncol.

METHOD (all GPU/JAX, taskset -c 0-3, OMP_NUM_THREADS=4, MEM_FRACTION 0.92, no
prealloc so peak_bytes_in_use is the true peak):
  * Build the real d02 128x128 case ONCE via daily_pipeline._build_real_case
    (cfg hours=1, dt_s=10, acoustic_substeps=10).  This gives a physically
    non-degenerate State + namelist (radiation cadence 180, fp64, flux
    advection, damp_opt=3, top_lid, epssm=0.5).
  * To get LARGER grids we spatially TILE/replicate the 128x128 State and the
    namelist's grid-shaped leaves (metrics, tendencies, radiation_static, the
    GridSpec metrics/terrain/projection) to bigger (ny,nx).  Physics kernel COST
    depends only on array SHAPES (ncol=ny*nx, nz), not on values, so a tiled
    state is a faithful COST proxy.  Staggered +1 faces are tiled on the
    interior then given one extra edge row/col so shapes are exactly
    (nz, fy*ny, fx*nx+1) etc.
  * run_boundary is set OFF for the cost measurement at EVERY size (including the
    128x128 re-measure) so the scaling law is apples-to-apples; the lateral
    boundary update is a tiny fraction of the step and its forcing arrays are
    awkward to tile faithfully.  GWD and NoahMP are likewise disabled uniformly
    (gwd_opt=0, use_noahmp=False) so all sizes run the identical dycore+radiation
    +PBL cost core.  This means the 128x128 number here is the DYCORE+RAD+PBL
    core cost, slightly below the full production ~178 ms/step that also pays
    boundary+GWD+NoahMP; we report both framings.
  * Warmed marginal timing = the two-hours-point method from
    proofs/perf/warmed_timing.py: time warmed runs at h1 and h2 (distinct static
    compiles), per_step = (warm_h2 - warm_h1)/(n2 - n1), which removes the fixed
    per-call exec overhead.  donate_argnums(0) consumes the input buffer, so the
    state is re-tiled before each timed call.
  * Peak VRAM via jax.devices()[0].memory_stats()['peak_bytes_in_use'] read
    after the largest (h2) warmed run of each size.  peak_bytes_in_use is
    MONOTONIC across the process, so to attribute a per-size peak we reset it
    between sizes when the runtime supports reset_memory_stats; otherwise we
    record the running monotonic peak and flag it.

Run (wrapped by scripts/with_gpu_lock.sh):
  PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/perf/v015/km_bench/grid_scaling_bench.py
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

OUT_DIR = Path("proofs/perf/v015/km_bench")
OUT_JSON = OUT_DIR / "grid_scaling.json"

# Base grid is 128x128; tile factors (fy, fx) -> target (ny, nx).
# 128x128(1,1), 256x256(2,2), 320x320(2.5 -> not integer) -> use 384x384(3,3),
# 448x448(3.5 -> not integer).  To hit the requested column counts with integer
# tiling of a 128 base we use (1,1),(2,2),(3,3),(4,4) = 128,256,384,512 and also
# (3,4)=384x512 to bracket the 200k-column target.  512x512=262144 cols is the
# top end.  We stop early on OOM.
# Finer ladder so we get >2 successful points and bracket the OOM cleanly.
# base 66x159 -> ncol grows as fy*fx*10494.
TILE_FACTORS = [
    (1, 1),   # 66x159   = 10,494
    (1, 2),   # 66x318   = 20,988
    (2, 2),   # 132x318  = 41,976
    (2, 3),   # 132x477  = 62,964
    (3, 3),   # 198x477  = 94,446
    (3, 4),   # 198x636  = 125,928
    (4, 4),   # 264x636  = 167,904
]


# --------------------------------------------------------------------------- #
# Tiling helpers
# --------------------------------------------------------------------------- #
def _tile_axis(arr: jax.Array, axis: int, factor: int, staggered: bool) -> jax.Array:
    """Tile `arr` along `axis` by integer `factor`.

    For an UN-staggered axis of size N -> factor*N (plain jnp.tile of that axis).
    For a STAGGERED axis of size N+1 (one extra face), tile the interior N rows
    `factor` times then append the final edge row -> factor*N + 1.
    """
    if factor == 1:
        return arr
    if not staggered:
        reps = [1] * arr.ndim
        reps[axis] = factor
        return jnp.tile(arr, reps)
    # staggered: size is N+1.  interior = first N along axis, edge = last 1.
    n = arr.shape[axis] - 1
    interior = jax.lax.slice_in_dim(arr, 0, n, axis=axis)
    edge = jax.lax.slice_in_dim(arr, n, n + 1, axis=axis)
    reps = [1] * arr.ndim
    reps[axis] = factor
    tiled_interior = jnp.tile(interior, reps)
    return jnp.concatenate([tiled_interior, edge], axis=axis)


def _tile_horizontal(
    arr: jax.Array, ny: int, nx: int, fy: int, fx: int
) -> jax.Array:
    """Tile any leaf whose trailing-2 dims are a horizontal (y,x) pattern.

    Recognises trailing pairs (ny,nx),(ny,nx+1),(ny+1,nx),(ny+1,nx+1).  The y
    axis is arr.ndim-2, the x axis is arr.ndim-1.  Returns arr unchanged if the
    trailing-2 dims are not a horizontal pattern.
    """
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


def _tile_state(state, ny: int, nx: int, fy: int, fx: int):
    """Tile the State pytree using the explicit per-field shape contract.

    Mass/face 3-D and surface 2-D fields are tiled on (y,x).  Boundary `*_bdy`
    leaves have shape (1,4,bw,nz(+1),boundary_side) where boundary_side =
    max(nx+1,ny+1); only the LAST axis is horizontal, tiled to
    max(fx*nx+1, fy*ny+1) by replicating the interior then keeping the final
    edge.  run_boundary is OFF so these are never consumed; we only need a
    self-consistent shape so the carry pytree is valid.
    """
    from gpuwrf.contracts.state import State  # local import: needs GPU backend

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
            # tile only the last (boundary_side) axis from old_side -> new_side.
            if shp[-1] == old_side and new_side != old_side:
                lastax = v.ndim - 1
                interior = jax.lax.slice_in_dim(v, 0, old_side - 1, axis=lastax)
                edge = jax.lax.slice_in_dim(v, old_side - 1, old_side, axis=lastax)
                # how many interior copies to reach new_side-1 (= new interior)?
                target_interior = new_side - 1
                base_interior = old_side - 1
                # tile enough then slice exactly to target_interior.
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


def _tile_child_pytree(child, ny: int, nx: int, fy: int, fx: int):
    """Tile every horizontal leaf of an arbitrary child pytree (metrics,
    tendencies, RRTMGRadiationStatic NamedTuple, ...).  Vertical-only and
    scalar leaves pass through unchanged."""
    if child is None:
        return None
    flat, treedef = jax.tree_util.tree_flatten(child)
    new = [
        _tile_horizontal(x, ny, nx, fy, fx) if hasattr(x, "shape") else x
        for x in flat
    ]
    return jax.tree_util.tree_unflatten(treedef, new)


def _scaled_grid(grid: GridSpec, ny: int, nx: int, fy: int, fx: int) -> GridSpec:
    """Rebuild GridSpec for the tiled grid: scale projection nx/ny, tile the
    terrain/metric arrays, keep eta unchanged.  GridSpec.__post_init__ validates
    shapes against the new projection dims."""
    proj = dataclasses.replace(grid.projection, nx=fx * nx, ny=fy * ny)
    terr = dataclasses.replace(grid.terrain, shape=(fy * ny, fx * nx))
    terrain_height = _tile_horizontal(grid.terrain_height, ny, nx, fy, fx)
    metrics = _tile_child_pytree(grid.metrics, ny, nx, fy, fx)
    return dataclasses.replace(
        grid,
        projection=proj,
        terrain=terr,
        terrain_height=terrain_height,
        metrics=metrics,
    )


def _tile_namelist(nl, ny: int, nx: int, fy: int, fx: int):
    """Rebuild the namelist for the tiled grid.

    Tiles the pytree children that carry grid-shaped arrays (tendencies,
    metrics, radiation_static).  Disables GWD/NoahMP/boundary uniformly so every
    size runs the identical dycore+radiation+PBL cost core.
    """
    new_grid = _scaled_grid(nl.grid, ny, nx, fy, fx)
    new_metrics = _tile_child_pytree(nl.metrics, ny, nx, fy, fx)
    new_tend = _tile_child_pytree(nl.tendencies, ny, nx, fy, fx)
    new_rad = _tile_child_pytree(nl.radiation_static, ny, nx, fy, fx)
    return dataclasses.replace(
        nl,
        grid=new_grid,
        metrics=new_metrics,
        tendencies=new_tend,
        radiation_static=new_rad,
        gwdo_statics=None,
        gwd_opt=0,
        use_noahmp=False,
        noahmp_static=None,
        run_boundary=False,
    )


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _block(tree) -> None:
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree
    )


def _peak_gib() -> float:
    dev = jax.devices()[0]
    try:
        return float(dev.memory_stats()["peak_bytes_in_use"]) / (1024.0**3)
    except Exception:
        return float("nan")


def _reset_peak() -> bool:
    dev = jax.devices()[0]
    for fn in ("reset_memory_stats",):
        if hasattr(dev, fn):
            try:
                getattr(dev, fn)()
                return True
            except Exception:
                return False
    return False


def _time_run(state, nl, hours: float) -> float:
    t0 = time.perf_counter()
    out = run_forecast_operational(state, nl, float(hours))
    _block(out)
    return time.perf_counter() - t0


def measure_size(base_state, base_nl, ny0, nx0, fy, fx, dt_s):
    """Tile to (fy*ny0, fx*nx0) and return a measured record (or an OOM record)."""
    ny, nx = fy * ny0, fx * nx0
    ncol = ny * nx
    h1, h2 = 0.05, 0.2  # 18 and 72 steps at dt=10s
    n1 = int(round(h1 * 3600.0 / dt_s))
    n2 = int(round(h2 * 3600.0 / dt_s))

    nl = _tile_namelist(base_nl, ny0, nx0, fy, fx)

    def fresh_state():
        # CRITICAL: run_forecast_operational donates (deletes) the state buffers.
        # The tiler must NEVER alias base_state's live buffers, or the 2nd timed
        # call references a deleted buffer.  jnp.tile/concatenate already produce
        # fresh arrays for fy/fx>1, but the (1,1) pass-through path returns the
        # base leaf as-is -- so we force an independent copy of EVERY leaf here.
        st = _tile_state(base_state, ny0, nx0, fy, fx)
        st = jax.tree_util.tree_map(
            lambda x: (x + 0) if hasattr(x, "shape") else x, st
        )
        _block(st)
        return st

    rec = {
        "ny": ny, "nx": nx, "nz": int(base_nl.grid.nz), "ncol": ncol,
        "tile_factor": [fy, fx], "h1": h1, "h2": h2, "n1": n1, "n2": n2,
    }
    try:
        _reset_peak()
        # h1: cold (compile) then warm
        st = fresh_state()
        _time_run(st, nl, h1)               # cold compile h1
        st = fresh_state()
        warm_h1 = _time_run(st, nl, h1)     # warm h1
        # h2: cold then warm (x2 for variance)
        st = fresh_state()
        _time_run(st, nl, h2)               # cold compile h2
        st = fresh_state()
        warm_h2a = _time_run(st, nl, h2)
        st = fresh_state()
        warm_h2b = _time_run(st, nl, h2)
        warm_h2 = 0.5 * (warm_h2a + warm_h2b)

        per_step_s = (warm_h2 - warm_h1) / float(n2 - n1)
        per_step_ms = per_step_s * 1000.0
        ms_per_fc_hour = per_step_ms * (3600.0 / dt_s)
        peak_gib = _peak_gib()

        rec.update({
            "ran_ok": True, "oom": False,
            "warm_h1_s": warm_h1, "warm_h2a_s": warm_h2a, "warm_h2b_s": warm_h2b,
            "warmed_ms_per_step": per_step_ms,
            "ms_per_forecast_hour": ms_per_fc_hour,
            "peak_vram_gib": peak_gib,
        })
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        is_oom = "RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower() \
            or "Out of memory" in str(exc) or "OOM" in str(exc)
        rec.update({
            "ran_ok": False, "oom": bool(is_oom),
            "error": msg[:600], "peak_vram_gib": _peak_gib(),
        })
        rec["_traceback_tail"] = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
    return rec


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = jax.devices()[0]
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    base_nl = case.namelist
    base_state = case.state
    dt_s = float(base_nl.dt_s)
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    print(f"[base] grid {ny0}x{nx0}x{nz} dt={dt_s}s device={dev}", flush=True)

    records = []
    for (fy, fx) in TILE_FACTORS:
        ny, nx = fy * ny0, fx * nx0
        print(f"[measure] tile ({fy},{fx}) -> {ny}x{nx} ncol={ny*nx} ...", flush=True)
        rec = measure_size(base_state, base_nl, ny0, nx0, fy, fx, dt_s)
        records.append(rec)
        if rec["ran_ok"]:
            print(f"  OK  ms/step={rec['warmed_ms_per_step']:.2f}  "
                  f"ms/fc-hr={rec['ms_per_forecast_hour']:.0f}  "
                  f"peakVRAM={rec['peak_vram_gib']:.2f} GiB", flush=True)
        else:
            print(f"  FAIL oom={rec['oom']} :: {rec.get('error','')[:160]}", flush=True)
            if rec["oom"]:
                print("  -> OOM: stop scaling up.", flush=True)
                break

    # Fit ms/step ~= a + b*ncol over the successful records.
    ok = [r for r in records if r.get("ran_ok")]
    fit = {}
    if len(ok) >= 2:
        xs = np.array([r["ncol"] for r in ok], dtype=float)
        ys = np.array([r["warmed_ms_per_step"] for r in ok], dtype=float)
        b, a = np.polyfit(xs, ys, 1)  # ys = b*xs + a
        # ratio test for sublinearity: compare ms/step ratio to ncol ratio
        r_small, r_big = ok[0], ok[-1]
        ncol_ratio = r_big["ncol"] / r_small["ncol"]
        ms_ratio = r_big["warmed_ms_per_step"] / r_small["warmed_ms_per_step"]
        fit = {
            "model": "ms_per_step = a + b*ncol",
            "a_intercept_ms": float(a),
            "b_slope_ms_per_col": float(b),
            "b_slope_ms_per_1000col": float(b * 1000.0),
            "ncol_range": [int(xs.min()), int(xs.max())],
            "smallest": {"ncol": r_small["ncol"], "ms": r_small["warmed_ms_per_step"]},
            "largest_ok": {"ncol": r_big["ncol"], "ms": r_big["warmed_ms_per_step"]},
            "ncol_ratio_small_to_big": float(ncol_ratio),
            "ms_ratio_small_to_big": float(ms_ratio),
            "sublinear_saturation_confirmed": bool(ms_ratio < ncol_ratio),
        }

    largest_ok = ok[-1] if ok else None
    payload = {
        "scope": "v0.15 km-bench: GPU step-throughput + peak-VRAM vs column count",
        "device": str(dev),
        "base_grid": {"ny": ny0, "nx": nx0, "nz": nz, "ncol": ny0 * nx0},
        "dt_s": dt_s,
        "steps_per_forecast_hour": 3600.0 / dt_s,
        "config_notes": {
            "run_boundary": False,
            "gwd_opt": 0,
            "use_noahmp": False,
            "radiation_cadence_steps": int(base_nl.radiation_cadence_steps),
            "force_fp64": bool(base_nl.force_fp64),
            "acoustic_substeps": int(base_nl.acoustic_substeps),
            "rrtmg_lw_col_tile": int(__import__("os").environ.get("GPUWRF_RRTMG_LW_COLUMN_TILE_COLS", 16384)),
            "rrtmg_sw_col_tile": int(__import__("os").environ.get("GPUWRF_RRTMG_SW_COLUMN_TILE_COLS", 16384)),
            "mynn_col_tile": int(__import__("os").environ.get("GPUWRF_MYNN_COLUMN_TILE_COLS", 16384)),
            "note": (
                "Larger grids are COST proxies built by spatially tiling the real "
                "128x128 state/grid; physics kernel cost depends on array shapes "
                "(ncol,nz) not values. boundary/GWD/NoahMP disabled uniformly at "
                "ALL sizes so the scaling law is apples-to-apples on the "
                "dycore+radiation+PBL core. This core 128x128 ms/step is BELOW the "
                "full production ~178 ms/step that also pays boundary+GWD+NoahMP."
            ),
        },
        "warmed_method": (
            "two-hours-point marginal: per_step=(warm_h2-warm_h1)/(n2-n1), removes "
            "fixed per-call exec overhead; warm_h2 = mean of two samples; "
            "donate-safe re-tiled state per timed call."
        ),
        "records": records,
        "fit": fit,
        "largest_grid_that_fit": (
            {"ny": largest_ok["ny"], "nx": largest_ok["nx"], "ncol": largest_ok["ncol"],
             "peak_vram_gib": largest_ok["peak_vram_gib"],
             "warmed_ms_per_step": largest_ok["warmed_ms_per_step"]}
            if largest_ok else None
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # Summary table
    print("\n==================== SUMMARY ====================", flush=True)
    print(f"{'grid':>11} {'ncol':>9} {'ms/step':>9} {'ms/fc-hr':>10} {'peakVRAM':>10} {'status':>7}")
    for r in records:
        grid = f"{r['ny']}x{r['nx']}"
        if r.get("ran_ok"):
            print(f"{grid:>11} {r['ncol']:>9d} {r['warmed_ms_per_step']:>9.2f} "
                  f"{r['ms_per_forecast_hour']:>10.0f} {r['peak_vram_gib']:>9.2f}G {'OK':>7}")
        else:
            st = "OOM" if r.get("oom") else "ERR"
            print(f"{grid:>11} {r['ncol']:>9d} {'-':>9} {'-':>10} "
                  f"{r.get('peak_vram_gib', float('nan')):>9.2f}G {st:>7}")
    if fit:
        print(f"\nfit: ms/step = {fit['a_intercept_ms']:.2f} + "
              f"{fit['b_slope_ms_per_1000col']:.4f} * (ncol/1000)")
        print(f"ncol x{fit['ncol_ratio_small_to_big']:.1f} -> "
              f"ms/step x{fit['ms_ratio_small_to_big']:.2f}  "
              f"=> sublinear saturation: {fit['sublinear_saturation_confirmed']}")
    print(f"\nwrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
