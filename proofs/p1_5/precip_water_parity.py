"""P1-5 Thompson PRECIP/WATER parity validation against the WRF precipitating oracle.

Sprint P1-5 closes the remaining Thompson surface-precip / hydrometeor gap to WRF
via three WRF-faithful items ported from ``module_mp_thompson.F``:

  (1) Adaptive micro sub-stepping ``nstep`` (per-species, per-column CFL):
      ``nstep = MAX_k INT(DT/(dzq/vt)+1)`` -- module_mp_thompson.F:3634-3641 etc.
      (already in the integration base, commit 822e8db; re-verified here.)
  (2) The ``rr(kts) > R1*1000 = 1e-9 kg/m3`` surface-accumulation threshold WRF
      applies before tallying the bottom-face flux as precip -- F:3818,3868,3895,3936.
      (already in the integration base; re-verified here.)
  (3) Cloud-water sedimentation (the cloud-w fall term) -- module_mp_thompson.F:
      3646-3667 (vtc), 3824-3837 (single full-DT explicit-upwind pass below
      500 m AGL, NOT counted as surface precip).  ADDED in this sprint.

This harness validates the SHIPPED faithful-explicit kernel (cloud-w sed = default
ON) against the WRF ``mp_gt_driver`` single-column precipitating oracle, and ALSO
runs the kernel with cloud-w sedimentation disabled (monkeypatched no-op) to
report the BEFORE->AFTER effect of fix (3) as an honest isolation.

Run (GPU-FREE, pinned):
  JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 PYTHONPATH=src:proofs/thompson_perf \
    taskset -c 0-3 python3 proofs/p1_5/precip_water_parity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))

import gpuwrf.physics.thompson_column as tc  # noqa: E402
from precip_oracle_validate import (  # noqa: E402
    build_state,
    field_errors,
    wrf_reference_precip,
    _meta,
    _rd,
    ORACLE_TO_KERNEL,
)

# --- PREDECLARED tolerances (frozen BEFORE scoring) --------------------------
# Surface precip = WRF RAINNCV (rain+snow+graupel+ice only; cloud-w sed is NOT
# surface precip in WRF).  Water closure is the FULL budget incl. the cloud-w
# bottom-face sink.  Hydrometeor field bands match the shipped oracle test plus a
# tightened qc band that fix (3) is expected to satisfy.
TOL = {
    "surface_precip_ratio": (0.97, 1.03),     # +-3% of WRF RAINNCV
    "per_column_max_rel": 0.03,               # every column within 3%
    "qr_mean_rel": 0.01,
    "qr_max_rel": 0.02,
    "qv_mean_rel": 0.01,
    "qc_mean_rel": 0.001,                     # fix (3) target: qc essentially exact
    "qc_max_rel": 0.01,
    "qs_mean_rel": 0.15,                      # WRF single-mode snow closure band
    "qi_mean_rel": 0.05,
    "water_closure_rel": 1e-5,                # full budget incl cloud-w sink
}

DT = 18.0
MASS_FIELDS = ("qc", "qr", "qi", "qs", "qg")


def _full_water_budget(state_in, out_state, precip_dict):
    """Column water budget: dQ_total + (all surface sinks incl cloud-w) == 0.

    Returns per-col precip (rain+snow+graupel+ice, the WRF surface precip), the
    cloud-w bottom-face sink (a water sink WRF does not count as precip), and the
    closure residual using the FULL sink set so the budget must close exactly.
    """
    rho = np.asarray(state_in.rho, dtype=np.float64)
    dz = np.asarray(state_in.dz, dtype=np.float64)

    def colmass(s):
        cond = sum(np.asarray(getattr(s, ORACLE_TO_KERNEL[f]), dtype=np.float64) for f in MASS_FIELDS)
        return np.sum((cond + np.asarray(s.qv, dtype=np.float64)) * rho * dz, axis=-1)

    total_in = colmass(state_in)
    total_out = colmass(out_state)

    surf_precip = sum(
        np.asarray(precip_dict[k], dtype=np.float64) for k in ("rain", "snow", "graupel", "ice")
    )
    cloudw = np.asarray(precip_dict.get("cloudw", 0.0), dtype=np.float64)
    all_sinks = surf_precip + cloudw  # every water-leaving channel
    closure = (total_out - total_in) + all_sinks
    return {
        "surface_precip_mm_per_col": surf_precip.tolist(),
        "total_surface_precip_mm": float(surf_precip.sum()),
        "cloudw_surface_sink_mm_per_col": cloudw.reshape(-1).tolist() if cloudw.ndim else [float(cloudw)],
        "total_cloudw_surface_sink_mm": float(np.sum(cloudw)),
        "water_closure_max_abs_residual_kg_m2": float(np.max(np.abs(closure))),
        "water_closure_max_rel_residual": float(np.max(np.abs(closure) / np.maximum(total_in, 1e-30))),
    }


def run(cloudw_on: bool):
    """Run the FULL faithful Thompson column path; cloud-w sed ON or OFF.

    When ``cloudw_on`` is False, ``_sed_cloud_water`` is monkeypatched to a no-op
    (qc untouched, zero cloud-w sink) so we can isolate fix (3)'s effect.  The
    rest of the kernel (adaptive nstep, rr>1e-9 threshold) is unchanged.
    """
    ni, nk, nj = _meta()
    state_in = build_state("in", ni, nk, nj)
    ref_post = {n: _rd("out", n, ni, nk, nj) for n in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")}

    orig = tc._sed_cloud_water
    try:
        if not cloudw_on:
            def _noop(state, dt):
                return state.qc, jnp.zeros(state.qc.shape[:-1], dtype=jnp.float64)
            tc._sed_cloud_water = _noop
        out_state, precip = tc._step_thompson_column_full_impl(state_in, DT, False)
    finally:
        tc._sed_cloud_water = orig

    fe = field_errors(out_state, ref_post, ni, nk, nj)
    budget = _full_water_budget(state_in, out_state, precip)
    species = {k: float(np.asarray(v).sum()) for k, v in precip.items()}
    return {"per_field": fe, "budget": budget, "precip_by_species_mm": species,
            "n_columns": int(state_in.qv.shape[0]), "n_levels": nk}


def _nstep_audit():
    """Per-species adaptive nstep range + NSED_MAX clip count (no silent masking)."""
    ni, nk, nj = _meta()
    state_in = build_state("in", ni, nk, nj)
    vt_r_m, vt_r_n, vt_i_m, vt_i_n, vt_s_m, vt_g_m, vt_g_n = tc._fall_speeds(state_in)
    dz = jnp.maximum(state_in.dz, 1.0)
    out = {}
    for name, a, b in (("rain", vt_r_m, vt_r_n), ("ice", vt_i_m, vt_i_m),
                       ("snow", vt_s_m, vt_s_m), ("graupel", vt_g_m, vt_g_m)):
        ns = np.asarray(tc._nstep_per_column(a, b, dz, DT))
        out[name] = {"nstep_min": int(ns.min()), "nstep_max": int(ns.max()),
                     "clipped_at_NSED_MAX": int((ns >= tc.NSED_MAX).sum()),
                     "n_columns": int(ns.size)}
    return out


def main():
    wrf = wrf_reference_precip()
    after = run(cloudw_on=True)    # shipped default
    before = run(cloudw_on=False)  # fix (3) disabled (isolation)

    sp = after["budget"]["total_surface_precip_mm"]
    ratio = sp / wrf["wrf_total_rainncv_mm"]
    pf = after["per_field"]

    jper = np.asarray(after["budget"]["surface_precip_mm_per_col"])
    wper = np.asarray(wrf["wrf_rainncv_mm_per_col"])
    per_col_max_rel = float(np.max(np.abs(jper - wper) / np.maximum(wper, 1e-12)))

    gates = {
        "G1_surface_precip_ratio": {"value": ratio, "tol": list(TOL["surface_precip_ratio"]),
                                    "pass": TOL["surface_precip_ratio"][0] <= ratio <= TOL["surface_precip_ratio"][1]},
        "G2_per_column_max_rel": {"value": per_col_max_rel, "tol": TOL["per_column_max_rel"],
                                  "pass": per_col_max_rel <= TOL["per_column_max_rel"]},
        "G3_qr_parity": {"mean_rel": pf["qr"]["mean_rel"], "max_rel": pf["qr"]["max_rel"],
                         "tol_mean": TOL["qr_mean_rel"], "tol_max": TOL["qr_max_rel"],
                         "pass": pf["qr"]["mean_rel"] <= TOL["qr_mean_rel"] and pf["qr"]["max_rel"] <= TOL["qr_max_rel"]},
        "G4_qv_parity": {"mean_rel": pf["qv"]["mean_rel"], "tol_mean": TOL["qv_mean_rel"],
                         "pass": pf["qv"]["mean_rel"] <= TOL["qv_mean_rel"]},
        "G5_qc_parity": {"mean_rel": pf["qc"]["mean_rel"], "max_rel": pf["qc"]["max_rel"],
                         "tol_mean": TOL["qc_mean_rel"], "tol_max": TOL["qc_max_rel"],
                         "pass": pf["qc"]["mean_rel"] <= TOL["qc_mean_rel"] and pf["qc"]["max_rel"] <= TOL["qc_max_rel"]},
        "G6_qs_parity": {"mean_rel": pf["qs"]["mean_rel"], "tol_mean": TOL["qs_mean_rel"],
                         "pass": pf["qs"]["mean_rel"] <= TOL["qs_mean_rel"]},
        "G7_qi_parity": {"mean_rel": pf["qi"]["mean_rel"], "tol_mean": TOL["qi_mean_rel"],
                         "pass": pf["qi"]["mean_rel"] <= TOL["qi_mean_rel"]},
        "G8_water_closure": {"value": after["budget"]["water_closure_max_rel_residual"],
                             "tol": TOL["water_closure_rel"],
                             "pass": after["budget"]["water_closure_max_rel_residual"] <= TOL["water_closure_rel"]},
    }
    all_pass = all(g["pass"] for g in gates.values())

    record = {
        "proof": "P1-5 Thompson PRECIP/WATER parity vs WRF precipitating oracle",
        "oracle": "WRF mp_gt_driver single-column PRECIPITATING (8 cols x 44 lev, DT=18 s)",
        "fixes": {
            "1_adaptive_nstep": "module_mp_thompson.F:3634-3641,3693-3696,3732-3735,3773-3776 "
                                "(per-species nstep = MAX_k INT(DT/(dzq/vt)+1)); base 822e8db, re-verified",
            "2_rr_surf_threshold": "module_mp_thompson.F:3818,3868,3895,3936 (accumulate surface flux only "
                                   "when updated rr(kts) > R1*1000 = 1e-9 kg/m3); base 822e8db, re-verified",
            "3_cloud_water_sed": "module_mp_thompson.F:3646-3667 (vtc=rhof*av_c*ccg(5,nu_c)*ocg2(nu_c)*ilamc^bv_c, "
                                 "nu_c=12, active only where rc>R1 & w<0.1) + 3824-3837 (single full-DT explicit-"
                                 "upwind pass below 500 m AGL; NOT counted as surface precip). ADDED this sprint",
        },
        "predeclared_tolerances": TOL,
        "wrf_reference": wrf,
        "after_all_three_fixes": after,
        "before_fix3_cloudw_off": before,
        "fix3_isolation": {
            "qc_mean_rel_before": before["per_field"]["qc"]["mean_rel"],
            "qc_mean_rel_after": after["per_field"]["qc"]["mean_rel"],
            "qc_max_rel_before": before["per_field"]["qc"]["max_rel"],
            "qc_max_rel_after": after["per_field"]["qc"]["max_rel"],
            "surface_precip_ratio_before": before["budget"]["total_surface_precip_mm"] / wrf["wrf_total_rainncv_mm"],
            "surface_precip_ratio_after": ratio,
            "cloudw_surface_sink_mm_after": after["budget"]["total_cloudw_surface_sink_mm"],
        },
        "nstep_audit_no_silent_masking": _nstep_audit(),
        "gates": gates,
        "all_pass": all_pass,
    }
    out = ROOT / "proofs" / "p1_5" / "precip_water_parity.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(gates, indent=2))
    print("ALL_PASS:", all_pass)
    print("wrote", out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
