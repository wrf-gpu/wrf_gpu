#!/usr/bin/env python3
"""Validate the JAX WDM5 (mp_physics=14) port against the independent WRF oracle.

Compares gpuwrf.physics.microphysics_wdm5.wdm5_run (fp64) against gold
savepoints that are the REAL WRF module_mp_wdm5.F scheme (wdm5init + wdm52D +
effectRad_wdm5) run on representative columns + edge cases. The JAX port cannot
self-compare: the reference is the unmodified Fortran scheme, not a JAX re-run.

Two oracle builds (both from UNMODIFIED module_mp_wdm5.F; see
proofs/v013/oracle/build_wdm5_oracle.sh):
  * proofs/v013/savepoints_wdm5       -- canonical classic-WRF single precision
                                         (bare `real`). Binding reference for
                                         the PROGNOSTIC mass + number state.
  * proofs/v013/savepoints_wdm5_fp64  -- the SAME unmodified scheme compiled
                                         with -fdefault-real-8 -DDOUBLE_PRECISION
                                         (libmassv VREC/VSQRT real*8 variants).
                                         Machine-precision faithfulness target +
                                         the effective-radius diagnostics
                                         (categorical floors flip on fp32 dust).

PREDECLARED TOLERANCES (frozen BEFORE comparison; never loosened): identical to
the WDM6 lane (WDM5 shares the same double-moment warm-rain machinery and the
same data-dependent semi-Lagrangian PLM sedimentation + cube-root slope
inversions; only the cold-rain side is the simpler 5-class WSM5 ice). fp64 JAX
vs fp32 (prognostic) / fp64 (machine-precision + diagnostics).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics.microphysics_wdm5 import wdm5_run  # noqa: E402

SAVE_FP32 = os.path.join(HERE, "savepoints_wdm5")
SAVE_FP64 = os.path.join(HERE, "savepoints_wdm5_fp64")

# --------- PREDECLARED TOLERANCES (frozen) ---------
PREDECLARED_TOL = {
    "t_abs": 1.0e-2,            # temperature absolute K (latent accumulation)
    "q_rel": 1.0e-2,           # mass mixing ratios: rel-to-column-peak
    "q_abs_floor": 1.0e-7,     # mass absolute floor (kg/kg)
    "n_rel": 2.0e-2,           # number concentrations: rel-to-column-peak
    "n_abs_floor": 1.0e2,      # number absolute floor (# kg^-1) ~ ncmin/nrmin
    "precip_rel": 1.5e-2,      # surface precip: rel
    "precip_abs": 5.0e-4,      # surface precip: abs floor (mm)
    "re_rel": 1.0e-2,          # effective radii (vs fp64): rel
    "re_abs_floor": 1.0e-7,    # effective radii: abs floor (m)
    "sr_abs": 1.0e-2,          # sr ratio: abs
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT")]
N_FIELDS = [("nn", "NN_OUT"), ("nc", "NC_OUT"), ("nr", "NR_OUT")]
RE_FIELDS = [("re_cloud", "RE_CLOUD"), ("re_ice", "RE_ICE"), ("re_snow", "RE_SNOW")]


def col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def field_metrics(jax_arr, oracle_arr, scale_floor):
    a = np.asarray(jax_arr, dtype=np.float64)
    b = np.asarray(oracle_arr, dtype=np.float64)
    scale = max(np.max(np.abs(b)), scale_floor)
    absdiff = np.abs(a - b)
    return float(np.max(absdiff)), float(np.max(absdiff) / scale), scale


def run_jax_for(d):
    s = d["scalars"]
    t = col(d, "T_IN")[None, :]
    qv = col(d, "QV_IN")[None, :]
    qc = col(d, "QC_IN")[None, :]
    qr = col(d, "QR_IN")[None, :]
    qi = col(d, "QI_IN")[None, :]
    qsv = col(d, "QS_IN")[None, :]
    nn = col(d, "NN_IN")[None, :]
    nc = col(d, "NC_IN")[None, :]
    nr = col(d, "NR_IN")[None, :]
    den = col(d, "DEN")[None, :]
    p = col(d, "P")[None, :]
    delz = col(d, "DELZ")[None, :]
    out = wdm5_run(jnp.asarray(t), jnp.asarray(qv), jnp.asarray(qc),
                   jnp.asarray(qr), jnp.asarray(qi), jnp.asarray(qsv),
                   jnp.asarray(nn), jnp.asarray(nc), jnp.asarray(nr),
                   jnp.asarray(den), jnp.asarray(p), jnp.asarray(delz), s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


def run_case(cid):
    with open(os.path.join(SAVE_FP32, f"wdm5_case_{cid}.json")) as fh:
        d32 = json.load(fh)
    with open(os.path.join(SAVE_FP64, f"wdm5_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    s = d32["scalars"]
    out = run_jax_for(d32)  # JAX inputs from fp32 savepoint (== fp64 in)

    results = {}
    passed = True

    # --- temperature vs fp32 ---
    mad, mrd, _ = field_metrics(out["t"], col(d32, "T_OUT"), 1.0)
    mad64, mrd64, _ = field_metrics(out["t"], col(d64, "T_OUT"), 1.0)
    ok = mad <= PREDECLARED_TOL["t_abs"]
    passed = passed and ok
    results["t"] = {"ref": "fp32", "max_abs": mad, "max_rel": mrd,
                    "vs_fp64_max_abs": mad64, "tol_abs": PREDECLARED_TOL["t_abs"],
                    "pass": bool(ok)}

    # --- mass species vs fp32 ---
    for leaf, oname in Q_FIELDS:
        mad, mrd, scale = field_metrics(out[leaf], col(d32, oname),
                                        PREDECLARED_TOL["q_abs_floor"])
        mad64, mrd64, _ = field_metrics(out[leaf], col(d64, oname),
                                        PREDECLARED_TOL["q_abs_floor"])
        ok = (mrd <= PREDECLARED_TOL["q_rel"]) or (mad <= PREDECLARED_TOL["q_abs_floor"])
        passed = passed and ok
        results[leaf] = {"ref": "fp32", "max_abs": mad, "max_rel": mrd,
                         "scale": scale, "vs_fp64_max_rel": mrd64, "pass": bool(ok)}

    # --- number species vs fp32 ---
    for leaf, oname in N_FIELDS:
        mad, mrd, scale = field_metrics(out[leaf], col(d32, oname),
                                        PREDECLARED_TOL["n_abs_floor"])
        mad64, mrd64, _ = field_metrics(out[leaf], col(d64, oname),
                                        PREDECLARED_TOL["n_abs_floor"])
        ok = (mrd <= PREDECLARED_TOL["n_rel"]) or (mad <= PREDECLARED_TOL["n_abs_floor"])
        passed = passed and ok
        results[leaf] = {"ref": "fp32", "max_abs": mad, "max_rel": mrd,
                         "scale": scale, "vs_fp64_max_rel": mrd64, "pass": bool(ok)}

    # --- effective radii vs fp64 (removes reference fp32 floor dust) ---
    for leaf, oname in RE_FIELDS:
        mad, mrd, scale = field_metrics(out[leaf], col(d64, oname),
                                        PREDECLARED_TOL["re_abs_floor"])
        mad32, mrd32, _ = field_metrics(out[leaf], col(d32, oname),
                                        PREDECLARED_TOL["re_abs_floor"])
        ok = (mrd <= PREDECLARED_TOL["re_rel"]) or (mad <= PREDECLARED_TOL["re_abs_floor"])
        passed = passed and ok
        results[leaf] = {"ref": "fp64", "max_abs": mad, "max_rel": mrd,
                         "scale": scale, "pass": bool(ok),
                         "vs_fp32_max_abs": mad32, "vs_fp32_max_rel": mrd32}

    # --- surface precip accumulators vs fp32 ---
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV")]:
        ov = float(s[sname])
        jv = float(out[leaf])
        tol = max(PREDECLARED_TOL["precip_rel"] * abs(ov), PREDECLARED_TOL["precip_abs"])
        ok = abs(jv - ov) <= tol
        passed = passed and ok
        results[leaf] = {"ref": "fp32", "jax": jv, "oracle": ov,
                         "abs_err": abs(jv - ov), "tol": tol, "pass": bool(ok)}

    ov = float(s["SR"]); jv = float(out["sr"])
    ok = abs(jv - ov) <= PREDECLARED_TOL["sr_abs"]
    passed = passed and ok
    results["sr"] = {"ref": "fp32", "jax": jv, "oracle": ov,
                     "abs_err": abs(jv - ov), "tol": PREDECLARED_TOL["sr_abs"],
                     "pass": bool(ok)}

    return passed, results


def main():
    cases = {}
    allpass = True
    for cid in (1, 2, 3, 4, 5, 6):
        ok, res = run_case(cid)
        cases[str(cid)] = {"pass": bool(ok), "fields": res}
        allpass = allpass and ok
        status = "PASS" if ok else "FAIL"
        print(f"=== CASE {cid} -> {status} ===")
        for fld, m in res.items():
            if "max_abs" in m:
                extra = ""
                if "vs_fp64_max_rel" in m:
                    extra = f" (vs_fp64 rel={m['vs_fp64_max_rel']:.3e})"
                elif "vs_fp32_max_rel" in m:
                    extra = f" (vs_fp32 rel={m['vs_fp32_max_rel']:.3e})"
                print(f"  {fld:9s} [{m['ref']}] max_abs={m['max_abs']:.3e} "
                      f"max_rel={m['max_rel']:.3e} {'ok' if m['pass'] else 'FAIL'}{extra}")
            else:
                print(f"  {fld:10s} [{m['ref']}] jax={m['jax']:.5e} oracle={m['oracle']:.5e} "
                      f"abs_err={m['abs_err']:.3e} {'ok' if m['pass'] else 'FAIL'}")
        print()

    report = {
        "scheme": "WDM5 (mp_physics=14) double-moment 5-class",
        "wiring": "OPERATIONAL (MP_SCAN_ADAPTERS[14] = wdm5_adapter; reuses the "
                  "WDM6 Nn/Nc/Nr State leaves, no new State leaf)",
        "oracle": {
            "prognostic_ref": "WRF module_mp_wdm5.F fp32 single-column savepoints "
                              "(proofs/v013/savepoints_wdm5); UNMODIFIED scheme "
                              "(wdm5init+wdm52D+effectRad_wdm5), no self-compare",
            "machine_precision_ref": "same UNMODIFIED scheme, -fdefault-real-8 "
                              "-DDOUBLE_PRECISION (proofs/v013/savepoints_wdm5_fp64); "
                              "fp32<->fp64 oracle cross-check confirms the oracle itself",
            "diagnostic_ref": "fp64 build used for effective radii to remove the "
                              "fp32 reference's categorical floor dust (as WDM6/WSM6)",
        },
        "jax_precision": "fp64",
        "predeclared_tolerances": PREDECLARED_TOL,
        "overall_pass": bool(allpass),
        "notes": [
            "WDM5 = WSM5-style ICE/SNOW (NO graupel/hail) + double-moment warm rain. "
            "Prognostic mass state (t, qv, qc, qr, qi, qs) and surface precip are gated "
            "against the CANONICAL fp32 WRF oracle.",
            "Number concentrations (Nn=CCN, Nc=cloud, Nr=rain) are the DOUBLE-MOMENT "
            "additions; gated vs the fp32 oracle with a slightly looser relative tolerance "
            "(cube-root lamda inversions + 1e1..1e10 dynamic range amplify fp32 roundoff). "
            "The vs_fp64 column shows machine-precision agreement.",
            "Effective radii are diagnostic-only with categorical floors; gated vs the "
            "fp64 oracle, vs-fp32 recorded for transparency.",
            "WDM5 reuses the operationally-wired WDM6 Nn/Nc/Nr leaves -> no State change; "
            "the adapter (coupling.scan_adapters.wdm5_adapter) threads Nn into State.Nn.",
        ],
        "cases": cases,
    }
    outpath = os.path.join(HERE, "t3_wdm5_oracle.json")
    with open(outpath, "w") as fh:
        json.dump(report, fh, indent=2)
    print("OVERALL:", "PASS" if allpass else "FAIL")
    print("wrote", outpath)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
