"""case3 V10 momentum-budget ATTRIBUTION harness (worker/opus/v10-momentum).

Pinpoints which component weakens the lowest-level prognostic wind over water on
case3 (init 2026-05-21 18z L3), distinguishing the DYCORE interior flow from the
MYNN-PBL vertical momentum mixing (and confirming the residual boundary plume is
NOT the cause, per the boundary-frame skill decomposition done separately).

THREE evidence products, from at most two 24 h GPU forecasts:

  A. VERTICAL PROFILE (1 standard run). The full water-mean u/v profile k0..kN vs
     CPU-WRF destaggered U/V at the lead valid time. If only k0 is weak relative to
     aloft -> PBL over-mixing (a SAFE mynn_pbl.py target). If the whole column is
     uniformly weak / wrong-direction -> dycore interior flow (sacrosanct -> defer).

  B. SINGLE-STEP LOWEST-LEVEL MYNN MOMENTUM INCREMENT, evaluated eagerly on the
     post-dycore final-state column (mynn_adapter applied once, outside the scan).
     The mean over water of the MYNN du/dv at k0 (the friction sink + the
     vertical-mixing redistribution). Quantifies how hard MYNN pulls the k0 wind
     per step relative to the dynamics.

  C. MYNN-MOMENTUM-OFF COUNTERFACTUAL (1 extra run). Zero ONLY the PBL u/v increment
     (keep theta/qv/qke + surface fluxes + dycore + boundary). Re-score V10/U10 water
     skill + dump the k0 wind. If V10 improves toward WRF -> MYNN momentum is (part of)
     the cause. If unchanged/worse -> the deficit is the dycore interior flow (defer).

CPU-WRF truth = corpus wrfout_d02 ONLY (no new WRF runs).

USAGE
  PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/wind/v10_momentum_budget.py --lead 24
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# case3 L3
RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DOMAIN = "d02"
INIT = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)


def masks_from_xland(xland):
    xland = np.asarray(xland, dtype=np.float64)
    land = xland < 1.5
    water = ~land
    return land, water


def rmse(a, b, m=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    if m is not None:
        d = d[m]
    return float(np.sqrt(np.mean(d**2)))


def wmean(arr, m):
    a = np.asarray(arr, dtype=np.float64)[m]
    a = a[np.isfinite(a)]
    return None if a.size == 0 else float(np.mean(a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--segment-steps", type=int, default=180)
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--nz-profile", type=int, default=12)
    ap.add_argument("--mode", choices=("standard", "mynn_off"), default="standard",
                    help="standard = full physics; mynn_off = zero the PBL u/v "
                         "increment only (run as a SEPARATE process so JAX never "
                         "reuses the standard-mode compiled scan).")
    ap.add_argument("--out", type=Path, default=Path("proofs/wind/v10_momentum_budget.json"))
    args = ap.parse_args()

    import jax
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    from gpuwrf.runtime.operational_mode import (
        compute_m9_diagnostics,
        run_forecast_operational_segmented,
    )
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.coupling.physics_couplers import _u_mass, _v_mass
    from gpuwrf.physics.mynn_pbl import step_mynn_pbl_column

    _state_from_mynn = pc._state_from_mynn_output  # the unpatched original

    cfg = DailyPipelineConfig(
        run_id=RUN_ID, run_root=RUN_ROOT, domain=DOMAIN,
        dt_s=args.dt_s, acoustic_substeps=args.acoustic_substeps,
        radiation_cadence_steps=args.radiation_cadence_steps,
    )
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=args.radiation_cadence_steps, time_utc=case.run_start,
    )

    init_static = read_wrfout_file(
        run_dir / f"wrfout_{DOMAIN}_{INIT:%Y-%m-%d_%H:%M:%S}", fields=("XLAND",))["fields"]
    land, water = masks_from_xland(init_static["XLAND"])
    ny, nx = land.shape

    valid = INIT + timedelta(hours=args.lead)
    wrfout = run_dir / f"wrfout_{DOMAIN}_{valid:%Y-%m-%d_%H:%M:%S}"
    wrf2d = read_wrfout_file(wrfout, fields=("U10", "V10", "T2"))["fields"]
    wrf3d = read_wrfout_file(wrfout, fields=("U", "V"))["fields"]
    # destagger WRF U (nz, ny, nx+1) -> mass (nz, ny, nx); V (nz, ny+1, nx) -> mass
    wrf_u_mass = 0.5 * (np.asarray(wrf3d["U"])[:, :, :-1] + np.asarray(wrf3d["U"])[:, :, 1:])
    wrf_v_mass = 0.5 * (np.asarray(wrf3d["V"])[:, :-1, :] + np.asarray(wrf3d["V"])[:, 1:, :])

    init2d = read_wrfout_file(
        run_dir / f"wrfout_{DOMAIN}_{INIT:%Y-%m-%d_%H:%M:%S}", fields=("U10", "V10"))["fields"]

    get = lambda x: np.asarray(jax.device_get(x))
    nzp = int(args.nz_profile)
    nz = min(nzp, wrf_u_mass.shape[0])

    def skill(field_gpu, field_truth, field_init, m):
        p = rmse(field_init, field_truth, m)
        g = rmse(field_gpu, field_truth, m)
        return (1 - g / p) if p > 0 else None, p, g

    payload: dict[str, Any] = {
        "_doc": "case3 V10 momentum-budget attribution. mode=standard dumps the "
                "water-mean vertical u/v profile vs CPU-WRF + the single-step MYNN "
                "k0 momentum increment; mode=mynn_off zeroes ONLY the PBL u/v "
                "increment (separate process, no stale jit cache) to isolate "
                "dycore vs MYNN momentum. Truth = corpus CPU-WRF wrfout_d02.",
        "case": "case3_0521_18z_L3",
        "init_utc": INIT.isoformat(),
        "lead_h": args.lead,
        "mode": args.mode,
        "grid": {"all": ny * nx, "land": int(land.sum()), "water": int(water.sum())},
        "config": {"dt_s": args.dt_s, "acoustic_substeps": args.acoustic_substeps,
                   "segment_steps": args.segment_steps},
    }

    if args.mode == "mynn_off":
        # Suppress ONLY the PBL u/v increment; keep theta/qv/qke writes + surface
        # fluxes + dycore + boundary. Patched BEFORE the forecast traces the scan,
        # in a fresh process, so the compiled scan body bakes in this variant.
        def _state_from_mynn_no_momentum(state, out):
            return state.replace(
                theta=pc._from_columns(out.theta).astype(pc._output_dtype(state, "theta")),
                qv=pc._from_columns(out.qv).astype(pc._output_dtype(state, "qv")),
                qke=(2.0 * pc._from_columns(out.tke)).astype(pc._output_dtype(state, "qke")),
            )
        pc._state_from_mynn_output = _state_from_mynn_no_momentum

    t0 = time.time()
    final = run_forecast_operational_segmented(
        case.state, nl, float(args.lead), segment_steps=args.segment_steps)
    jax.block_until_ready(final.theta)
    diags = compute_m9_diagnostics(final, nl, float(args.lead) * 3600.0)
    wall = round(time.time() - t0, 1)

    gpu_u10 = get(diags.u10)
    gpu_v10 = get(diags.v10)
    gpu_u_col = get(_u_mass(final))  # (nz, ny, nx) mass
    gpu_v_col = get(_v_mass(final))
    finite = bool(np.isfinite(gpu_v10).all() and np.isfinite(gpu_u10).all())

    sk = {}
    for fld, g_diag, truth, ini in [("U10", gpu_u10, wrf2d["U10"], init2d["U10"]),
                                     ("V10", gpu_v10, wrf2d["V10"], init2d["V10"])]:
        sk[fld] = {}
        for rname, m in [("all", np.ones_like(water)), ("water", water)]:
            s, p, g = skill(g_diag, truth, ini, m)
            sk[fld][rname] = {"skill": s, "pers_rmse": p, "gpu_rmse": g}

    prof = []
    for k in range(min(nz, gpu_u_col.shape[0])):
        prof.append({
            "k": k,
            "gpu_u": wmean(gpu_u_col[k], water), "wrf_u": wmean(wrf_u_mass[k], water),
            "gpu_v": wmean(gpu_v_col[k], water), "wrf_v": wmean(wrf_v_mass[k], water),
            "gpu_wspd": wmean(np.hypot(gpu_u_col[k], gpu_v_col[k]), water),
            "wrf_wspd": wmean(np.hypot(wrf_u_mass[k], wrf_v_mass[k]), water),
        })

    payload["result"] = {
        "wall_s": wall, "finite": finite, "k0_diag_skill": sk, "water_profile": prof,
    }

    # single-step MYNN k0 momentum increment, evaluated eagerly on the post-dycore
    # final state (surface_adapter first for the Gate-1 flux hand-off). Reports how
    # hard MYNN pulls the k0 wind per step (dt*RUBLTEN). Computed in BOTH modes; in
    # mynn_off the patch is restored first so the true increment is measured.
    pc._state_from_mynn_output = _state_from_mynn  # ensure original is in place
    sfc_state = pc.surface_adapter(final, float(args.dt_s))
    col = pc._mynn_column_from_state(sfc_state, nl.grid)
    surface = pc._surface_fluxes_from_state(sfc_state)
    out_pbl = step_mynn_pbl_column(col, float(args.dt_s), debug=False, surface=surface)
    u_in0 = _u_mass(sfc_state)[0]
    v_in0 = _v_mass(sfc_state)[0]
    du0 = get(pc._from_columns(out_pbl.u)[0] - u_in0)
    dv0 = get(pc._from_columns(out_pbl.v)[0] - v_in0)

    # Falsifier #4 (sidecar): split the FULL MYNN momentum increment into
    # (a) bottom-drag-only (vertical diffusivity dfm=0, keep surface stress) and
    # (b) vertical-diffusion-only (dfm active, bottom_drag=0). Reuses the MYNN
    # internals read-only; identifies WHICH term (if any) is the candidate lever.
    import jax.numpy as _jnp
    from gpuwrf.physics import mynn_pbl as mp
    clip = mp._clip_state(col)
    flux, wind, fltv, rhosfc = mp._surface_terms(clip, surface)
    qke = 2.0 * clip.tke
    turb = mp._mym_turbulence(clip, qke, fltv)
    drag = rhosfc * flux.ustar * flux.ustar / wind
    zeros = _jnp.zeros_like(wind)
    zerodfm = _jnp.zeros_like(turb["dfm"])
    # drag-only: dfm=0 so no vertical redistribution, only the lower-BC stress sink
    u_drag = mp._diffusion_solve_with_surface(clip.u, zerodfm, clip, float(args.dt_s), zeros, drag)
    v_drag = mp._diffusion_solve_with_surface(clip.v, zerodfm, clip, float(args.dt_s), zeros, drag)
    # diffusion-only: full dfm, no bottom drag
    u_diff = mp._diffusion_solve_with_surface(clip.u, turb["dfm"], clip, float(args.dt_s), zeros, 0.0)
    v_diff = mp._diffusion_solve_with_surface(clip.v, turb["dfm"], clip, float(args.dt_s), zeros, 0.0)
    du0_drag = get(pc._from_columns(u_drag)[0] - u_in0)
    dv0_drag = get(pc._from_columns(v_drag)[0] - v_in0)
    du0_diff = get(pc._from_columns(u_diff)[0] - u_in0)
    dv0_diff = get(pc._from_columns(v_diff)[0] - v_in0)

    payload["result"]["mynn_k0_increment_single_step_water"] = {
        "full": {"mean_du0": wmean(du0, water), "mean_dv0": wmean(dv0, water),
                 "mean_abs_du0": wmean(np.abs(du0), water), "mean_abs_dv0": wmean(np.abs(dv0), water)},
        "drag_only": {"mean_du0": wmean(du0_drag, water), "mean_dv0": wmean(dv0_drag, water)},
        "diffusion_only": {"mean_du0": wmean(du0_diff, water), "mean_dv0": wmean(dv0_diff, water)},
        "_doc": "one mynn step on the post-dycore state, mean over water; dt*RUBLTEN "
                "at k0. full=drag+vertical diffusion; drag_only=dfm=0; "
                "diffusion_only=bottom_drag=0. negative=decelerating that component. "
                "ERROR VECTOR to beat: GPU-WRF k0 ~ (+u too positive, +v too weak); a "
                "harmful PBL term has increment aligned with (+du0,+dv0).",
    }

    print(f"[{args.mode} {wall}s finite={finite}] "
          f"V10 water skill={sk['V10']['water']['skill']:+.4f} "
          f"U10 water skill={sk['U10']['water']['skill']:+.4f}", flush=True)
    print("  water profile (k: gpu_v / wrf_v | gpu_u / wrf_u | gpu_wspd / wrf_wspd):")
    for r in prof:
        print(f"    k={r['k']:2d}  v {r['gpu_v']:+6.2f}/{r['wrf_v']:+6.2f}   "
              f"u {r['gpu_u']:+6.2f}/{r['wrf_u']:+6.2f}   "
              f"wspd {r['gpu_wspd']:5.2f}/{r['wrf_wspd']:5.2f}", flush=True)
    mi = payload["result"]["mynn_k0_increment_single_step_water"]
    print(f"  MYNN k0 incr/step water FULL:  du0={mi['full']['mean_du0']:+.3e} "
          f"dv0={mi['full']['mean_dv0']:+.3e} (|du0|={mi['full']['mean_abs_du0']:.3e} "
          f"|dv0|={mi['full']['mean_abs_dv0']:.3e})", flush=True)
    print(f"  MYNN k0 incr/step water DRAG:  du0={mi['drag_only']['mean_du0']:+.3e} "
          f"dv0={mi['drag_only']['mean_dv0']:+.3e}", flush=True)
    print(f"  MYNN k0 incr/step water DIFF:  du0={mi['diffusion_only']['mean_du0']:+.3e} "
          f"dv0={mi['diffusion_only']['mean_dv0']:+.3e}", flush=True)

    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
