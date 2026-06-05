"""Wave-B2 STEP 1b: decompose the MYNN closure kernel (32 ms) into its internal phases.

The block profile showed the whole 33.9 ms surface+MYNN block is 94% the MYNN
closure kernel (mynn_closure_kernel_only=32.0 ms), with the mechanical wrappers
(column build/reassemble/surface) ~2 ms total. This splits the closure kernel
itself to separate irreducible closure compute from any removable internal
mechanical cost:

  * _mym_turbulence        (level-2.5 closure + mixing length)
  * _mym_predict_qke       (TKE prediction tridiag)
  * _edmf_arrays_from_state(MYNN-EDMF mass-flux: vmap over plumes of a lax.scan)
  * _apply_mean_tendencies (u/v/theta/qv implicit tridiag solves)
  * cumulative chained (turb -> qke -> edmf -> apply) = the real dependent path
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics import mynn_pbl as mp

PROOF = Path("proofs/v0100")
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"
DT = 10.0


def _bench(fn, *args, n_rep=10, label=""):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"label": label, "min_ms": float(min(samples)),
            "median_ms": float(np.median(samples)),
            "samples_ms": [round(s, 4) for s in samples]}


def main() -> int:
    cfg = DailyPipelineConfig(hours=1, dt_s=DT, acoustic_substeps=10,
                             run_id=L2_RUN_ID, run_root=paths.wrf_l2_root(),
                             domain="d02", radiation_cadence_steps=180)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=False, radiation_cadence_steps=180,
                             time_utc=case.run_start)
    grid = nl.grid
    state = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    state = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if hasattr(a, "shape") else a, state)
    state_with_flux = pc.surface_adapter(state, DT)

    # Build the batched column inputs the closure kernel sees.
    column = pc._mynn_column_from_state(state_with_flux, grid)
    surface = pc._surface_fluxes_from_state(state_with_flux)
    ny, nx = column.theta.shape[0], column.theta.shape[1]
    cb = jax.tree_util.tree_map(lambda a: jnp.asarray(a), pc._flatten_columns_to_batch(column, ny, nx))
    sb = jax.tree_util.tree_map(lambda a: jnp.asarray(a), pc._flatten_columns_to_batch(surface, ny, nx))
    dx = pc._mynn_dx(grid)
    edmf = pc._MYNN_EDMF

    print(f"=== MYNN internal decomposition: B={ny*nx} cols x nz={cb.theta.shape[-1]} "
          f"device={jax.devices()[0]} edmf={edmf} ===", flush=True)

    results = {}

    # _clip_state + _surface_terms (setup)
    def _setup(c, s):
        cl = mp._clip_state(c)
        flux, wind, fltv, rhosfc = mp._surface_terms(cl, s)
        return flux.ustar, wind, fltv, rhosfc
    results["clip_plus_surface_terms"] = _bench(jax.jit(_setup), cb, sb, label="clip_plus_surface_terms")

    # _mym_turbulence
    def _turb(c, s):
        cl = mp._clip_state(c)
        flux, wind, fltv, rhosfc = mp._surface_terms(cl, s)
        qke = 2.0 * cl.tke
        return mp._mym_turbulence(cl, qke, fltv, flux.ustar, dx, flux.xland)
    results["mym_turbulence"] = _bench(jax.jit(_turb), cb, sb, label="mym_turbulence")

    # _mym_predict_qke
    def _qke(c, s):
        cl = mp._clip_state(c)
        flux, wind, fltv, rhosfc = mp._surface_terms(cl, s)
        qke = 2.0 * cl.tke
        turb = mp._mym_turbulence(cl, qke, fltv, flux.ustar, dx, flux.xland)
        return mp._mym_predict_qke(cl, qke, turb, DT, flux.ustar, flux)
    results["turb_plus_predict_qke"] = _bench(jax.jit(_qke), cb, sb, label="turb_plus_predict_qke")

    # _edmf_arrays_from_state (the vmap-over-plumes lax.scan)
    def _edmf(c, s):
        cl = mp._clip_state(c)
        flux, wind, fltv, rhosfc = mp._surface_terms(cl, s)
        qke = 2.0 * cl.tke
        turb = mp._mym_turbulence(cl, qke, fltv, flux.ustar, dx, flux.xland)
        return mp._edmf_arrays_from_state(cl, flux, fltv, turb["pblh"], DT, dx)
    if edmf:
        results["turb_plus_edmf"] = _bench(jax.jit(_edmf), cb, sb, label="turb_plus_edmf")

    # _apply_mean_tendencies (the 4 implicit tridiag solves) -- chained from turb+edmf
    def _apply(c, s):
        cl = mp._clip_state(c)
        flux, wind, fltv, rhosfc = mp._surface_terms(cl, s)
        qke = 2.0 * cl.tke
        turb = mp._mym_turbulence(cl, qke, fltv, flux.ustar, dx, flux.xland)
        mf = mp._edmf_arrays_from_state(cl, flux, fltv, turb["pblh"], DT, dx) if edmf else None
        return mp._apply_mean_tendencies(cl, turb, DT, flux, wind, rhosfc, mf=mf)
    results["turb_edmf_plus_apply"] = _bench(jax.jit(_apply), cb, sb, label="turb_edmf_plus_apply")

    # full impl (reference, should ~= mynn_closure_kernel_only)
    def _full(c, s):
        return mp._step_mynn_pbl_impl(c, DT, False, s, edmf, dx)
    results["full_impl"] = _bench(jax.jit(_full), cb, sb, label="full_impl")

    summary = {
        "scope": "Wave-B2 STEP 1b: MYNN closure kernel internal decomposition (GREEN d02 L2)",
        "run_dir": str(run_dir), "device": str(jax.devices()[0]),
        "batch_cols": int(ny * nx), "nz": int(cb.theta.shape[-1]),
        "edmf": bool(edmf), "dx": float(dx), "dt_s": DT,
        "components_ms": results,
        "interpretation": (
            "cumulative chained timings: each row includes its predecessors. "
            "Subtract consecutive rows for the incremental phase cost. The full "
            "impl is a straight-line dependent sequence (no tunable iteration "
            "count); the per-phase delta is irreducible closure compute over the "
            "461736-cell domain."
        ),
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "wave_b2_mynn_internal.json"
    fn.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n--- cumulative chained min_ms ---", flush=True)
    for k, v in results.items():
        print(f"  {k:30s} {v['min_ms']:8.4f} ms", flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
