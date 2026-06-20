"""Wave-B2 STEP 1: decompose the surface-layer + MYNN/PBL block on the GREEN d02 L2 init.

Times, in isolation on the operational State, the components that make up the
~33.8 ms / 45.5% MYNN-PBL share of the coupled step:

  * surface_adapter (revised surface layer, writes 7 flux handles)
  * mynn_adapter (column build + EDMF + closure + reassemble)
  * the column-view BUILD only (_mynn_column_from_state) -- the _to_columns
    transposes + rho/dz diagnostics
  * the column-view BUILD for surface (_surface_column_view) -- the redundant
    _u_mass/_v_mass/_to_columns(theta/qv/p) that mynn ALSO builds
  * the MYNN kernel ONLY (step_mynn_pbl_column on prebuilt columns) -- the
    irreducible closure compute
  * the State reassemble only (_state_from_mynn_output) -- the _from_columns
    transposes + A2C increment

Each is jitted and timed warmed (min of repeats, first sample discarded).
Counts the real transposes (moveaxis) on the surface+MYNN path.

Run:
  PYTHONPATH=src GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo OMP_NUM_THREADS=4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async taskset -c 0-3 \
    python proofs/v0100/wave_b2_mynn_profile.py
"""
from __future__ import annotations

import dataclasses
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision
from gpuwrf.coupling import physics_couplers as pc

PROOF = Path("proofs/v0100")
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"
DT = 10.0


def _bench(fn, *args, n_warm=2, n_rep=8, label=""):
    """Warm then time min over repeats; discard the first (compile) sample."""
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples_sorted = sorted(samples)
    return {
        "label": label,
        "min_ms": float(samples_sorted[0]),
        "median_ms": float(np.median(samples)),
        "samples_ms": [round(s, 4) for s in samples],
    }


def main() -> int:
    cfg = DailyPipelineConfig(
        hours=1, dt_s=DT, acoustic_substeps=10,
        run_id=L2_RUN_ID, run_root=paths.wrf_l2_root(), domain="d02",
        radiation_cadence_steps=180,
    )
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=False, radiation_cadence_steps=180,
                             time_utc=case.run_start)
    grid = nl.grid
    state = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    # Put state on device.
    state = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if hasattr(a, "shape") else a, state)
    jax.block_until_ready(state.theta)

    print(f"=== Wave-B2 MYNN profile: L2 d02 {grid.ny}x{grid.nx}x{grid.nz} "
          f"device={jax.devices()[0]} ===", flush=True)

    results = {}

    # 1) surface_adapter (full)
    surf_jit = jax.jit(partial(pc.surface_adapter, dt=DT))
    results["surface_adapter"] = _bench(surf_jit, state, label="surface_adapter")

    # State with surface fluxes written (the input MYNN sees).
    state_with_flux = surf_jit(state)
    jax.block_until_ready(state_with_flux.theta)

    # 2) mynn_adapter (full)
    mynn_jit = jax.jit(partial(pc.mynn_adapter, dt=DT, grid=grid))
    results["mynn_adapter"] = _bench(mynn_jit, state_with_flux, label="mynn_adapter")

    # 3) mynn_adapter_with_diagnostics (full, incl pblh)
    mynnd_jit = jax.jit(partial(pc.mynn_adapter_with_diagnostics, dt=DT, grid=grid))
    results["mynn_adapter_with_diagnostics"] = _bench(
        mynnd_jit, state_with_flux, label="mynn_adapter_with_diagnostics")

    # 4) the MYNN column BUILD only (_mynn_column_from_state) -- transposes + rho/dz
    build_jit = jax.jit(partial(pc._mynn_column_from_state, grid=grid))
    results["mynn_column_build"] = _bench(build_jit, state_with_flux, label="mynn_column_build")

    # 5) the surface column-view BUILD only (the redundant duplicate of the same view)
    surfview_jit = jax.jit(pc._surface_column_view)
    results["surface_column_view_build"] = _bench(surfview_jit, state, label="surface_column_view_build")

    # 6) the surface flux read-back from state (_surface_fluxes_from_state)
    fluxread_jit = jax.jit(pc._surface_fluxes_from_state)
    results["surface_fluxes_from_state"] = _bench(fluxread_jit, state_with_flux, label="surface_fluxes_from_state")

    # 7) the MYNN closure kernel ONLY (on prebuilt + flattened columns) -- irreducible
    column = pc._mynn_column_from_state(state_with_flux, grid)
    surface = pc._surface_fluxes_from_state(state_with_flux)
    ny, nx = column.theta.shape[0], column.theta.shape[1]
    column_b = pc._flatten_columns_to_batch(column, ny, nx)
    surface_b = pc._flatten_columns_to_batch(surface, ny, nx)
    column_b = jax.tree_util.tree_map(lambda a: jnp.asarray(a), column_b)
    surface_b = jax.tree_util.tree_map(lambda a: jnp.asarray(a), surface_b)
    from gpuwrf.physics.mynn_pbl import step_mynn_pbl_column

    def _kernel(col_b, surf_b):
        return step_mynn_pbl_column(col_b, DT, debug=False, surface=surf_b,
                                    edmf=pc._MYNN_EDMF, dx=pc._mynn_dx(grid))
    kernel_jit = jax.jit(_kernel)
    results["mynn_closure_kernel_only"] = _bench(kernel_jit, column_b, surface_b,
                                                 label="mynn_closure_kernel_only")

    # 8) the State reassemble ONLY (_state_from_mynn_output) -- _from_columns + A2C
    out_b = kernel_jit(column_b, surface_b)
    out = pc._unflatten_batch_to_columns(out_b, ny, nx)
    out = jax.tree_util.tree_map(lambda a: jnp.asarray(a), out)
    reassemble_jit = jax.jit(pc._state_from_mynn_output)
    results["state_from_mynn_output"] = _bench(reassemble_jit, state_with_flux, out,
                                               label="state_from_mynn_output")

    # 9) surface+MYNN COMBINED (full chain, as operational_mode runs it)
    def _combined(s):
        s2 = pc.surface_adapter(s, DT)
        return pc.mynn_adapter(s2, DT, grid)
    combined_jit = jax.jit(_combined)
    results["surface_plus_mynn_combined"] = _bench(combined_jit, state, label="surface_plus_mynn_combined")

    # Count transposes on the surface+MYNN path via compiled HLO text.
    def _count_ops(jitted, *args):
        try:
            txt = jitted.lower(*args).compile().as_text()
        except Exception:
            try:
                txt = jitted.lower(*args).as_text()
            except Exception as e:  # pragma: no cover
                return {"error": str(e)}
        return {
            "transpose": txt.count("transpose("),
            "reshape": txt.count("reshape("),
            "convert": txt.count("convert("),
            "fusion": txt.count("fusion("),
        }

    hlo_counts = {
        "surface_column_view": _count_ops(surfview_jit, state),
        "mynn_column_from_state": _count_ops(build_jit, state_with_flux),
        "state_from_mynn_output": _count_ops(reassemble_jit, state_with_flux, out),
        "surface_plus_mynn_combined": _count_ops(combined_jit, state),
        "mynn_closure_kernel_only": _count_ops(kernel_jit, column_b, surface_b),
    }

    summary = {
        "scope": "Wave-B2 STEP 1: surface+MYNN block decomposition (GREEN d02 L2)",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "grid": {"ny": int(grid.ny), "nx": int(grid.nx), "nz": int(grid.nz)},
        "dt_s": DT,
        "edmf": bool(pc._MYNN_EDMF),
        "components_ms": results,
        "hlo_op_counts": hlo_counts,
        "notes": {
            "surface_adapter": "revised surface layer; writes 7 flux handles",
            "mynn_column_build_redundant_with_surface": (
                "_mynn_column_from_state and _surface_column_view both compute "
                "_u_mass/_v_mass/_to_columns(theta,qv,p): redundant duplicate transposes"
            ),
            "mynn_closure_kernel_only": "irreducible MYNN2.5 + EDMF closure compute",
        },
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "wave_b2_mynn_profile.json"
    fn.write_text(json.dumps(summary, indent=2) + "\n")
    # Print a compact table.
    print("\n--- component min_ms ---", flush=True)
    for k, v in results.items():
        print(f"  {k:38s} {v['min_ms']:8.4f} ms", flush=True)
    print("\n--- hlo op counts ---", flush=True)
    for k, v in hlo_counts.items():
        print(f"  {k:38s} {v}", flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
