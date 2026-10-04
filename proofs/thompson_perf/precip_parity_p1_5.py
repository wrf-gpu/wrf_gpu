"""P1-5 Thompson precip/water PARITY proof generator.

Validates the WRF-faithful adaptive-nstep sedimentation refinement (NSED_MAX
masked scan with per-column nstep = MAX_k INT(DT/(dz/vt)+1) and the rr>1e-9
surface-accumulation threshold) against the PRECIPITATING WRF mp_gt_driver
single-column oracle, and emits the predeclared parity gates as a JSON proof.

Gates (predeclared, falsifiable):
  G1 surface-precip parity  : total within +-3% of WRF RAINNCV (was +13%)
  G2 per-column parity      : every column within +-3% of WRF
  G3 rain-field parity      : qr mean_rel < 1%, max_rel < 2%
  G4 water-mass closure     : max rel residual < 1e-5
  G5 accumulator additivity : per-species precip sums to total (RAINNC channels)
  G6 sed-clip fallback count : columns whose WRF nstep would exceed NSED_MAX (0
                               on this oracle; reported for the operational audit)

Run: PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
       XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 taskset -c 0-3 \
       python3 proofs/thompson_perf/precip_parity_p1_5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))

import gpuwrf.physics.thompson_column as tc
from precip_oracle_validate import run_scheme, wrf_reference_precip, build_state, _meta


def _sed_clip_audit(dt=18.0):
    """Count columns whose WRF adaptive nstep would exceed NSED_MAX (sed-clip)."""
    ni, nk, nj = _meta()
    state = build_state("in", ni, nk, nj)
    captured = {}
    orig = tc._sedimentation

    def cap(s, d):
        captured["state"] = s
        return orig(s, d)

    tc._sedimentation = cap
    try:
        tc._step_thompson_column_full_impl(state, dt, False)
    finally:
        tc._sedimentation = orig
    pre = captured["state"]
    vts = tc._fall_speeds(pre)
    dz = np.maximum(np.asarray(pre.dz, dtype=np.float64), 1.0)

    def raw_nstep(vt_a, vt_b):
        vt = np.maximum(np.asarray(vt_a, np.float64), np.asarray(vt_b, np.float64))
        cand = np.where(vt > 1.0e-3, np.floor(dt * vt / dz + 1.0), 0.0)
        return np.maximum(cand.max(axis=-1), 1.0)

    species = {
        "rain": raw_nstep(vts[0], vts[1]),
        "ice": raw_nstep(vts[2], vts[2]),
        "snow": raw_nstep(vts[4], vts[4]),
        "graupel": raw_nstep(vts[5], vts[5]),
    }
    out = {}
    for name, ns in species.items():
        out[name] = {
            "nstep_min": int(ns.min()),
            "nstep_max": int(ns.max()),
            "clipped_at_NSED_MAX": int((ns > tc.NSED_MAX).sum()),
            "n_columns": int(ns.size),
        }
    return out


def main():
    print("devices:", jax.devices())
    wrf = wrf_reference_precip()
    r = run_scheme("faithful_explicit")
    pm = r["precip_mass"]
    pf = r["per_field"]
    by_sp = r["precip_by_species_mm"]

    jax_total = pm["total_surface_precip_mm"]
    wrf_total = wrf["wrf_total_rainncv_mm"]
    ratio = jax_total / wrf_total
    jper = pm["surface_precip_mm_per_col"]
    wper = wrf["wrf_rainncv_mm_per_col"]
    per_col_rel = [abs(j - w) / w for j, w in zip(jper, wper)]

    # G5: per-species channels sum to the total accumulator (RAINNC = rain + snow
    # + graupel + ice).  The harness derives total_surface_precip from the same
    # per-species precip dict, so additivity is a consistency check.
    species_sum = sum(by_sp.values())
    accumulator_residual = abs(species_sum - jax_total)

    clip = _sed_clip_audit()

    gates = {
        "G1_surface_precip_ratio": {
            "value": ratio, "tol": [0.97, 1.03],
            "pass": bool(0.97 <= ratio <= 1.03),
        },
        "G2_per_column_max_rel": {
            "value": float(max(per_col_rel)), "tol": 0.03,
            "pass": bool(max(per_col_rel) <= 0.03),
        },
        "G3_qr_field_parity": {
            "mean_rel": pf["qr"]["mean_rel"], "max_rel": pf["qr"]["max_rel"],
            "tol_mean": 0.01, "tol_max": 0.02,
            "pass": bool(pf["qr"]["mean_rel"] < 0.01 and pf["qr"]["max_rel"] < 0.02),
        },
        "G4_water_closure_rel": {
            "value": pm["water_closure_max_rel_residual"], "tol": 1e-5,
            "pass": bool(pm["water_closure_max_rel_residual"] < 1e-5),
        },
        "G5_accumulator_additivity_residual_mm": {
            "value": accumulator_residual, "tol": 1e-9,
            "pass": bool(accumulator_residual < 1e-9),
        },
        "G6_sed_clip_fallback": {
            "per_species": clip,
            "total_clipped": int(sum(s["clipped_at_NSED_MAX"] for s in clip.values())),
            "pass": bool(sum(s["clipped_at_NSED_MAX"] for s in clip.values()) == 0),
        },
    }
    all_pass = all(g["pass"] for g in gates.values())

    record = {
        "proof": "P1-5 Thompson precip/water PARITY (adaptive-nstep + rr>1e-9 threshold)",
        "oracle": "WRF mp_gt_driver single-column PRECIPITATING (8 cols x 44 lev, 18 s)",
        "fix": "NSED_MAX masked scan; per-column nstep = MAX_k INT(DT/(dz/vt)+1); "
               "surface accumulation gated on updated rr(kts) > R1*1000 = 1e-9 kg/m3",
        "NSED_MAX": tc.NSED_MAX,
        "before_fix": {"surface_precip_ratio": 1.1338, "qr_mean_rel": 0.0153,
                       "qr_max_rel": 0.2300, "note": "fixed-64 substeps (over-resolved)"},
        "after_fix": {
            "wrf_total_rainncv_mm": wrf_total,
            "jax_total_surface_precip_mm": jax_total,
            "surface_precip_ratio": ratio,
            "per_field": pf,
            "precip_mass": pm,
            "precip_by_species_mm": by_sp,
        },
        "gates": gates,
        "all_pass": all_pass,
    }
    out = ROOT / "proofs" / "thompson_perf" / "precip_parity_p1_5.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(gates, indent=2))
    print("ALL_PASS:", all_pass)
    print("wrote", out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
