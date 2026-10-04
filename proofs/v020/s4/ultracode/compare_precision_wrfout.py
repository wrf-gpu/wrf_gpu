#!/usr/bin/env python3
"""Compare an fp32-arm wrfout against the fp64 baseline wrfout and check the
v0.20 fp32 acceptance bands (proofs/v020/fp32_proto/acceptance_bands.py).

fp64 is treated as truth; the fp32 arm's field diffs are the "error increase"
attributable to fp32. At a short lead (1-2 h) these should sit well inside the
24h band. Also runs the universal hard gates (finiteness + physical bounds).

Usage:
  python compare_precision_wrfout.py --fp64 <wrfout> --fp32 <wrfout> --lead-h 1 \
      --label mixed --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, "proofs/v020/fp32_proto")
from acceptance_bands import (  # noqa: E402
    check_metric_band,
    check_universal_hard_gates,
    field_group,
    FIELD_GROUPS,
)

# wrfout variable -> acceptance field name (acceptance_bands FIELD_GROUPS key)
VARS = {
    "U": "U", "V": "V", "W": "W", "U10": "U10", "V10": "V10",
    "T": "THETA", "T2": "T2",
    "P": "P", "PB": "PB", "PH": "PH", "PHB": "PHB", "MU": "MU", "MUB": "MUB", "PSFC": "PSFC",
    "QVAPOR": "QVAPOR", "QCLOUD": "QCLOUD", "QICE": "QICE", "QSNOW": "QSNOW", "QGRAUP": "QGRAUP",
}


def _read(ds, name):
    if name not in ds.variables:
        return None
    a = np.asarray(ds.variables[name][:], dtype=np.float64)
    return a[0] if a.ndim and a.shape[0] == 1 else a


def _destagger(a, axis):
    sl1 = [slice(None)] * a.ndim; sl2 = [slice(None)] * a.ndim
    sl1[axis] = slice(0, -1); sl2[axis] = slice(1, None)
    return 0.5 * (a[tuple(sl1)] + a[tuple(sl2)])


def diff_metrics(fp64, fp32, group, var=None):
    d = fp32 - fp64
    out = {
        "rmse_increase": float(np.sqrt(np.mean(d ** 2))),
        "bias_abs": float(abs(np.mean(d))),
        "p95_abs_diff": float(np.percentile(np.abs(d), 95)),
        "max_abs_diff": float(np.max(np.abs(d))),
    }
    # group-specific drift metrics keyed to the acceptance bands
    if group == "temperature":
        out["domain_mean_theta_drift"] = float(abs(np.mean(fp32) - np.mean(fp64)))
    if group == "pressure_mass_geopotential":
        out["psfc_rmse_increase"] = out["rmse_increase"]
        out["psfc_bias_abs"] = out["bias_abs"]
    if group == "qvapor":
        m64 = float(np.mean(np.abs(fp64))) or 1.0
        out["domain_mean_water_vapor_rel_diff"] = float(abs(np.mean(fp32) - np.mean(fp64)) / m64)
        out["qvapor_p999"] = float(np.percentile(np.abs(d), 99.9))
    if group == "cloud_radiation":
        # column_condensate_nrmse: NRMSE of the field diff vs the fp64 field scale (genuine).
        # cloud_fraction_abs_diff / swdown_lwdown_bias_abs need fields not in this condensate
        # field-diff comparison and are left unchecked (noted in the report), NOT faked.
        denom = float(np.sqrt(np.mean(fp64 ** 2))) or 1.0
        out["column_condensate_nrmse"] = out["rmse_increase"] / denom
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp64", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--lead-h", type=float, default=1.0)
    ap.add_argument("--label", default="fp32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = Dataset(args.fp64, "r")
    b = Dataset(args.fp32, "r")

    results = {}
    samples64, samples32 = {}, {}
    overall_pass = True

    # wind-speed domain drift (destagger U,V to mass)
    U64, V64 = _read(a, "U"), _read(a, "V")
    U32, V32 = _read(b, "U"), _read(b, "V")
    if U64 is not None and V64 is not None:
        sp64 = np.sqrt(_destagger(U64, -1) ** 2 + _destagger(V64, -2) ** 2)
        sp32 = np.sqrt(_destagger(U32, -1) ** 2 + _destagger(V32, -2) ** 2)
        wind_speed_drift = float(abs(np.mean(sp32) - np.mean(sp64)))
    else:
        wind_speed_drift = None

    for var, fld in VARS.items():
        f64 = _read(a, var); f32 = _read(b, var)
        if f64 is None or f32 is None:
            continue
        if f64.shape != f32.shape:
            results[var] = {"status": "SHAPE_MISMATCH", "fp64": f64.shape, "fp32": f32.shape}
            continue
        grp = field_group(fld)
        m = diff_metrics(f64, f32, grp)
        if grp == "wind" and wind_speed_drift is not None:
            m["domain_mean_speed_drift"] = wind_speed_drift
        chk = check_metric_band(fld, args.lead_h, m)
        results[var] = {
            "group": grp, "tier": chk.tier, "passes": chk.passes,
            "checked": chk.checked, "failures": list(chk.failures),
            "rmse_increase": round(m["rmse_increase"], 6),
            "bias_abs": round(m["bias_abs"], 6),
            "p95_abs_diff": round(m["p95_abs_diff"], 6),
            "max_abs_diff": round(m["max_abs_diff"], 6),
        }
        if not chk.passes:
            overall_pass = False
        # collect for universal hard gates (subsample to keep it light)
        flat = f32.ravel(); flat64 = f64.ravel()
        samples32[fld] = flat[:: max(1, flat.size // 200000)]
        samples64[fld] = flat64[:: max(1, flat64.size // 200000)]

    hard_ok, hard_fail = check_universal_hard_gates(samples32, baseline=samples64)
    if not hard_ok:
        overall_pass = False

    verdict = {
        "label": args.label,
        "fp64_file": args.fp64,
        "fp32_file": args.fp32,
        "lead_h": args.lead_h,
        "overall_tolerance_pass": overall_pass,
        "universal_hard_gates_pass": hard_ok,
        "universal_hard_failures": list(hard_fail),
        "wind_speed_domain_drift_m_s": wind_speed_drift,
        "per_field": results,
    }
    json.dump(verdict, open(args.out, "w"), indent=2)

    print(f"=== precision compare [{args.label}] lead={args.lead_h}h -> {args.out} ===")
    print(f"OVERALL tolerance pass = {overall_pass}   hard gates = {hard_ok}")
    if hard_fail:
        print("  hard failures:", hard_fail)
    for var, r in results.items():
        if "passes" in r:
            flag = "OK " if r["passes"] else "XX "
            print(f"  {flag}{var:7s} ({r['group']:>24s}) rmse={r['rmse_increase']:.4g} "
                  f"bias={r['bias_abs']:.4g} p95={r['p95_abs_diff']:.4g} max={r['max_abs_diff']:.4g}"
                  + ("" if r["passes"] else f"  FAIL {r['failures']}"))


if __name__ == "__main__":
    main()
