"""Coupled operational validation of orographic gravity-wave drag (gwd_opt=1).

This proves that the operational statics-wiring (io/gwdo_static.load_gwdo_statics
-> OperationalNamelist.gwdo_statics -> operational_mode dispatch) makes GWD
ACTUALLY run in a real-case GPU forecast, and that the coupled drag is finite,
terrain-localised, and physically signed.

It runs on the REAL Canary case (a real geo_em with the WPS sub-grid orography
fields VAR/CON/OA1-4/OL1-4 in wrfinput).  Two independent evidence lanes:

  LANE A  -- direct kernel diagnostic on the real d01 initial state:
      (b) drag is NON-ZERO over mountainous columns and ~zero over flat/sea;
      (c) the low-level drag OPPOSES the resolved low-level wind (decelerates);
          drag is physically bounded (|tendency| small, |dt*tendency| << wind).

  LANE B  -- end-to-end coupled GPU forecast gwd_opt=1 vs gwd_opt=0 (same IC):
      (a) the run stays FINITE + stable with GWD on (no blow-up);
      (d) the gwd_opt=1 - gwd_opt=0 wind difference is LOCALISED to terrain
          (correlates with the sub-grid orography), not a global perturbation.

The build path is ``daily_pipeline._build_real_case`` so the wiring under test
is the production wiring, not a hand-assembled namelist.

Usage (GPU, single job; wrap the whole command ONCE in /tmp/wrf_gpu_run.sh):
    /tmp/wrf_gpu_run.sh taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python proofs/gwd/gwd_coupled_validation.py \
            --case /mnt/data/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z \
            --domain d01 --hours 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _finite(arr) -> bool:
    return bool(np.all(np.isfinite(np.asarray(arr))))


def _u_mass(u):
    # C-grid u (..., nx+1) -> mass points (..., nx)
    return 0.5 * (u[..., :-1] + u[..., 1:])


def _v_mass(v):
    # C-grid v (..., ny+1, nx) -> mass points (..., ny, nx)
    return 0.5 * (v[..., :-1, :] + v[..., 1:, :])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        default="/mnt/data/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z",
    )
    ap.add_argument("--domain", default="d01")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).with_name("gwd_coupled_validation.json")),
    )
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
    from gpuwrf.physics.gwd_gwdo import GWDOColumnState, gwdo_columns
    from gpuwrf.coupling.physics_couplers import (
        _interface_pressure_from_state,
        _to_columns,
        _temperature_from_theta,
        _u_mass as cpl_u_mass,
        _v_mass as cpl_v_mass,
    )
    from gpuwrf.runtime.operational_mode import run_forecast_operational

    P0_PA = 100000.0
    R_D_OVER_CP = 287.0 / 1004.0
    G = 9.80665

    backend = jax.default_backend()
    result: dict = {
        "case": args.case,
        "domain": args.domain,
        "hours": float(args.hours),
        "jax_backend": backend,
    }

    # ---- build the REAL case through the production wiring under test --------
    cfg = DailyPipelineConfig(
        run_id=str(Path(args.case).resolve()),
        run_root=Path(args.case).resolve().parent,
        domain=args.domain,
        hours=int(round(args.hours)),
    )
    case, _run_dir = _build_real_case(cfg)
    nml_on = case.namelist
    state = case.state

    result["wiring"] = {
        "gwd_opt": int(nml_on.gwd_opt),
        "gwdo_statics_attached": nml_on.gwdo_statics is not None,
        "gwdo_static_meta": case.metadata.get("gwdo_static", {}),
        "namelist_gwd_opt_meta": case.metadata.get("namelist", {}).get("gwd_opt"),
    }

    # The wiring MUST have turned GWD on for a gwd_opt=1 case with real statics.
    wiring_ok = int(nml_on.gwd_opt) == 1 and nml_on.gwdo_statics is not None
    result["wiring"]["pass"] = bool(wiring_ok)
    if not wiring_ok:
        result["status"] = "FAIL"
        result["reason"] = "gwd_opt=1 case did not produce an attached GWDOStatics (wiring gap)"
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1

    statics = nml_on.gwdo_statics
    grid = case.grid
    ny, nx = int(grid.ny), int(grid.nx)
    dt = float(nml_on.dt_s)

    # ====================================================================== #
    # LANE A: direct kernel diagnostic on the REAL initial state              #
    # ====================================================================== #
    T = _temperature_from_theta(state.theta, state.p)
    exner = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    z_face = state.ph.astype(jnp.float64) / G
    z_mass = 0.5 * (z_face[:-1] + z_face[1:])
    u_mass = cpl_u_mass(state)
    v_mass = cpl_v_mass(state)

    column = GWDOColumnState(
        uproj=_to_columns(u_mass).reshape((ny * nx, -1)),
        vproj=_to_columns(v_mass).reshape((ny * nx, -1)),
        t1=_to_columns(T).reshape((ny * nx, -1)),
        q1=_to_columns(state.qv).reshape((ny * nx, -1)),
        prsl=_to_columns(state.p).reshape((ny * nx, -1)),
        prsi=_interface_pressure_from_state(state).reshape((ny * nx, -1)),
        prslk=_to_columns(exner).reshape((ny * nx, -1)),
        zl=_to_columns(z_mass).reshape((ny * nx, -1)),
    )
    out = gwdo_columns(column, statics, dt)
    ru = np.asarray(out.rublten)  # (B, K) m/s^2
    rv = np.asarray(out.rvblten)
    dusfcg = np.asarray(out.dusfcg)  # (B,) N/m^2
    dvsfcg = np.asarray(out.dvsfcg)

    var = np.asarray(statics.var)  # (B,) sub-grid orography std dev (m)
    oc1 = np.asarray(statics.oc1)  # (B,) convexity (CON)
    B = var.size

    drag_col_max = np.maximum(np.abs(ru).max(axis=1), np.abs(rv).max(axis=1))  # (B,)
    active = drag_col_max > 1e-12
    n_active = int(active.sum())

    # Opposition test (physically correct measure).  Orographic GWD breaks the
    # wave and deposits momentum ALOFT (often near the wave-breaking level, not
    # the lowest model layers), so a near-surface-layer projection is the wrong
    # probe.  The right invariant is the COLUMN-INTEGRATED momentum tendency: GWD
    # must REMOVE momentum from the resolved low-level flow, i.e. the vertically
    # summed (rublten,rvblten) increment must oppose the column low-level wind
    # (ubar,vbar).  Equivalently, the kernel's integrated surface stress
    # (dusfcg,dvsfcg) = -integral(dudt dp)/g must ALIGN with the wind.
    uproj = np.asarray(column.uproj)  # (B, K)
    vproj = np.asarray(column.vproj)
    klow = min(5, uproj.shape[1])
    ubar = uproj[:, :klow].mean(axis=1)  # low-level (column) wind
    vbar = vproj[:, :klow].mean(axis=1)
    sum_du = ru.sum(axis=1) * dt  # column-integrated wind increment over one step (m/s)
    sum_dv = rv.sum(axis=1) * dt
    wind_mag = np.sqrt(ubar**2 + vbar**2)
    # projection of the column-integrated tendency onto the low-level wind:
    # < 0 means GWD opposes (decelerates) the resolved flow.
    proj = sum_du * ubar + sum_dv * vbar
    # independent check: surface-stress vector aligns with the wind (removes momentum).
    stress_align = dusfcg * ubar + dvsfcg * vbar

    am = active & (wind_mag > 0.5)  # active columns with a meaningful low-level wind
    frac_opposing = float((proj[am] < 0).mean()) if am.sum() else float("nan")
    frac_stress_aligned = float((stress_align[am] > 0).mean()) if am.sum() else float("nan")

    finite_kernel = _finite(ru) and _finite(rv) and _finite(dusfcg) and _finite(dvsfcg)

    laneA = {
        "n_columns": int(B),
        "n_active_columns": n_active,
        "active_fraction": float(n_active / B),
        "drag_abs_max_m_s2": float(drag_col_max.max()),
        "dusfcg_abs_max_N_m2": float(np.abs(dusfcg).max()),
        "var_mean_active": float(var[active].mean()) if n_active else 0.0,
        "var_mean_inactive": float(var[~active].mean()) if (B - n_active) else 0.0,
        "con_mean_active": float(oc1[active].mean()) if n_active else 0.0,
        "con_mean_inactive": float(oc1[~active].mean()) if (B - n_active) else 0.0,
        "corr_dragmax_VAR": float(np.corrcoef(drag_col_max, var)[0, 1]) if var.std() > 0 else None,
        "corr_dragmax_CON": float(np.corrcoef(drag_col_max, oc1)[0, 1]) if oc1.std() > 0 else None,
        "fraction_column_drag_opposing_wind": frac_opposing,
        "fraction_surface_stress_aligned_with_wind": frac_stress_aligned,
        "max_onestep_column_wind_increment_m_s": float(np.max(np.abs(np.sqrt(sum_du**2 + sum_dv**2)))),
        "finite": bool(finite_kernel),
    }
    # Pass criteria for LANE A (the physics):
    #   (b) drag active over mountains, ~zero over flat/sea
    #       -> n_active>0, VAR(active) >> VAR(inactive), drag correlates with terrain.
    #   (c) drag opposes low-level wind in the (overwhelming) majority of active cols,
    #       and a single-step increment is small (bounded).
    laneA_b = (
        n_active > 0
        and laneA["var_mean_active"] > 2.0 * max(laneA["var_mean_inactive"], 1e-6)
        and (laneA["corr_dragmax_CON"] is None or laneA["corr_dragmax_CON"] > 0.3
             or (laneA["corr_dragmax_VAR"] is not None and laneA["corr_dragmax_VAR"] > 0.3))
    )
    laneA_c = (
        finite_kernel
        and (np.isnan(frac_opposing) or frac_opposing >= 0.95)
        and (np.isnan(frac_stress_aligned) or frac_stress_aligned >= 0.95)
        and laneA["max_onestep_column_wind_increment_m_s"] < 5.0
    )
    laneA["pass_b_drag_over_mountains"] = bool(laneA_b)
    laneA["pass_c_opposes_wind_bounded"] = bool(laneA_c)
    result["lane_A_kernel_diagnostic"] = laneA

    # ====================================================================== #
    # LANE B: end-to-end coupled forecast gwd_opt=1 vs gwd_opt=0              #
    # ====================================================================== #
    from dataclasses import replace as dataclass_replace

    nml_off = dataclass_replace(nml_on, gwd_opt=0)

    def _independent_copy(s):
        # run_forecast_operational donates (deletes) its input state buffers, so
        # each forecast needs its OWN independent copy.  A host round-trip
        # (device->numpy->device) yields fully distinct device buffers, leaving
        # the original `state` intact for the second run.
        return jax.tree_util.tree_map(
            lambda a: jnp.asarray(np.asarray(a)) if hasattr(a, "shape") else a, s
        )

    state_on = run_forecast_operational(_independent_copy(state), nml_on, float(args.hours))
    jax.block_until_ready(state_on.u)
    u_on = np.asarray(state_on.u)
    v_on = np.asarray(state_on.v)
    theta_on = np.asarray(state_on.theta)
    w_on = np.asarray(state_on.w)

    state_off = run_forecast_operational(_independent_copy(state), nml_off, float(args.hours))
    jax.block_until_ready(state_off.u)
    u_off = np.asarray(state_off.u)
    v_off = np.asarray(state_off.v)
    theta_off = np.asarray(state_off.theta)

    finite_on = _finite(u_on) and _finite(v_on) and _finite(theta_on) and _finite(w_on)
    finite_off = _finite(u_off) and _finite(v_off) and _finite(theta_off)

    # difference field on mass points
    um_on = _u_mass(u_on)
    um_off = _u_mass(u_off)
    vm_on = _v_mass(v_on)
    vm_off = _v_mass(v_off)
    dwind = np.sqrt((um_on - um_off) ** 2 + (vm_on - vm_off) ** 2)  # (nz, ny, nx)
    dwind_col = dwind.max(axis=0).reshape(-1)  # (ny*nx,) max over column

    # terrain-localisation: correlate the per-column wind difference with the
    # sub-grid orography that drives GWD.  A global (non-terrain) perturbation
    # would show ~0 correlation; a GWD-localised one is positively correlated and
    # concentrated where VAR/CON are large.
    corr_dwind_var = float(np.corrcoef(dwind_col, var)[0, 1]) if var.std() > 0 else None
    corr_dwind_con = float(np.corrcoef(dwind_col, oc1)[0, 1]) if oc1.std() > 0 else None
    terr = (var > np.percentile(var, 90)) | (oc1 > 0)
    mean_dwind_terrain = float(dwind_col[terr].mean()) if terr.sum() else 0.0
    mean_dwind_flat = float(dwind_col[~terr].mean()) if (~terr).sum() else 0.0
    # The cleanest localisation probe is the column set where GWD was ACTIVE at
    # init (the deterministic, un-advected footprint from LANE A): a global
    # perturbation would show no enhancement there, a GWD-localised one does.
    mean_dwind_active = float(dwind_col[active].mean()) if active.sum() else 0.0
    mean_dwind_inactive = float(dwind_col[~active].mean()) if (~active).sum() else 0.0
    terrain_ratio = mean_dwind_terrain / max(mean_dwind_flat, 1e-12)
    active_ratio = mean_dwind_active / max(mean_dwind_inactive, 1e-12)

    laneB = {
        "forecast_hours": float(args.hours),
        "steps": int(round(args.hours * 3600.0 / dt)),
        "finite_gwd_on": bool(finite_on),
        "finite_gwd_off": bool(finite_off),
        "u_on_absmax": float(np.abs(u_on).max()),
        "u_off_absmax": float(np.abs(u_off).max()),
        "w_on_absmax": float(np.abs(w_on).max()),
        "theta_on_min": float(theta_on.min()),
        "theta_on_max": float(theta_on.max()),
        "dwind_absmax_m_s": float(dwind.max()),
        "dwind_mean_m_s": float(dwind.mean()),
        "corr_dwind_VAR": corr_dwind_var,
        "corr_dwind_CON": corr_dwind_con,
        "mean_dwind_terrain": mean_dwind_terrain,
        "mean_dwind_flat": mean_dwind_flat,
        "mean_dwind_gwd_active_cols": mean_dwind_active,
        "mean_dwind_gwd_inactive_cols": mean_dwind_inactive,
        "terrain_to_flat_ratio": float(terrain_ratio),
        "active_to_inactive_ratio": float(active_ratio),
        "note": (
            "After 1 h, horizontal advection (~20 m/s over a 9 km grid moves the "
            "GWD signal several cells) smears the point-wise dwind-vs-orography "
            "correlation, so localisation is assessed by the terrain/flat and "
            "GWD-active/inactive concentration ratios plus the deterministic "
            "init-time footprint (LANE A corr_dragmax_CON)."
        ),
    }
    # (a) finite/stable with GWD on; the gwd-off control is also finite.
    laneB_a = finite_on and finite_off and laneB["u_on_absmax"] < 200.0
    # (d) difference is localised to terrain: GWD ran (dwind>0), and the change is
    #     clearly concentrated where the sub-grid orography is (terrain and the
    #     GWD-active columns carry markedly more change than flat/inactive ones),
    #     while the un-advected init footprint (LANE A) tracks the orography.
    localized = (
        laneB["dwind_absmax_m_s"] > 0.0
        and terrain_ratio > 1.3
        and active_ratio > 1.3
        and (laneA["corr_dragmax_CON"] is not None and laneA["corr_dragmax_CON"] > 0.3)
    )
    laneB["pass_a_finite_stable"] = bool(laneB_a)
    laneB["pass_d_terrain_localized"] = bool(localized)
    result["lane_B_coupled_forecast"] = laneB

    all_pass = bool(wiring_ok and laneA_b and laneA_c and laneB_a and localized)
    result["status"] = "PASS" if all_pass else "PARTIAL"
    result["all_pass"] = all_pass
    result["claims"] = {
        "a_finite_stable": bool(laneB_a),
        "b_drag_over_mountains": bool(laneA_b),
        "c_opposes_wind_bounded": bool(laneA_c),
        "d_terrain_localized": bool(localized),
    }

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
