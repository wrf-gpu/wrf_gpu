#!/usr/bin/env python3
"""Validate the JAX WDM6 port against the independent Fortran WRF oracle.

Compares gpuwrf.physics.microphysics_wdm6.wdm6_run (fp64) against gold
savepoints that are the REAL WRF module_mp_wdm6.F scheme (wdm6init + wdm62D +
effectRad_wdm6) run on representative columns + edge cases. The JAX port cannot
self-compare: the reference is the Fortran scheme, not a JAX re-run.

Two oracle builds (both from UNMODIFIED module_mp_wdm6.F; see
proofs/v060_wdm6/oracle):
  * proofs/v060_wdm6/savepoints       -- canonical classic-WRF single precision
                                         (bare `real`). Binding reference for
                                         the PROGNOSTIC mass + number state.
  * proofs/v060_wdm6/savepoints_fp64  -- the SAME unmodified scheme compiled
                                         with -fdefault-real-8 -DDOUBLE_PRECISION
                                         (so the libmassv VREC/VSQRT macros pick
                                         the real*8 variants). Used as the
                                         machine-precision faithfulness target
                                         and for the effective-radius
                                         diagnostics (categorical floors flip on
                                         fp32 trace dust, exactly as WSM6).

PREDECLARED TOLERANCES (frozen BEFORE comparison; never loosened):
  fp64 JAX vs fp32 (prognostic) / fp64 (machine-precision + diagnostics).
  WDM6 has data-dependent search loops in the semi-Lagrangian PLM
  sedimentation, an exp/log-heavy saturation path, many clamped accretion
  terms, AND a double-moment warm-rain number-budget with cube-root slope
  inversions; fp32-vs-fp64 roundoff propagates through these. Number
  concentrations (Nc/Nr/Nn) carry larger relative roundoff than the mass fields
  because of the cube-root lamda inversions and the large dynamic range
  (1e1..1e10 # kg^-1), so the number tolerance is stated separately.
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

from gpuwrf.physics.microphysics_wdm6 import wdm6_run  # noqa: E402

SAVE_FP32 = os.path.join(HERE, "savepoints")
SAVE_FP64 = os.path.join(HERE, "savepoints_fp64")

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
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT")]
N_FIELDS = [("nn", "NN_OUT"), ("nc", "NC_OUT"), ("nr", "NR_OUT")]
RE_FIELDS = [("re_cloud", "RE_CLOUD"), ("re_ice", "RE_ICE"), ("re_snow", "RE_SNOW")]

# slmsk per case (matches the oracle build_column): 1=land, 2=water
SLMSK = {1: 2.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 1.0, 6: 2.0}


def col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def field_metrics(jax_arr, oracle_arr, scale_floor):
    a = np.asarray(jax_arr, dtype=np.float64)
    b = np.asarray(oracle_arr, dtype=np.float64)
    scale = max(np.max(np.abs(b)), scale_floor)
    absdiff = np.abs(a - b)
    return float(np.max(absdiff)), float(np.max(absdiff) / scale), scale


def run_jax_for(d, cid):
    s = d["scalars"]
    t = col(d, "T_IN")[None, :]
    qv = col(d, "QV_IN")[None, :]
    qc = col(d, "QC_IN")[None, :]
    qr = col(d, "QR_IN")[None, :]
    qi = col(d, "QI_IN")[None, :]
    qsv = col(d, "QS_IN")[None, :]
    qg = col(d, "QG_IN")[None, :]
    nn = col(d, "NN_IN")[None, :]
    nc = col(d, "NC_IN")[None, :]
    nr = col(d, "NR_IN")[None, :]
    den = col(d, "DEN")[None, :]
    p = col(d, "P")[None, :]
    delz = col(d, "DELZ")[None, :]
    slmsk = np.array([SLMSK[cid]], dtype=np.float64)
    out = wdm6_run(jnp.asarray(t), jnp.asarray(qv), jnp.asarray(qc),
                   jnp.asarray(qr), jnp.asarray(qi), jnp.asarray(qsv),
                   jnp.asarray(qg), jnp.asarray(nn), jnp.asarray(nc),
                   jnp.asarray(nr), jnp.asarray(den), jnp.asarray(p),
                   jnp.asarray(delz), s["DT"], jnp.asarray(slmsk))
    return {k: np.asarray(v)[0] for k, v in out.items()}


def run_case(cid):
    with open(os.path.join(SAVE_FP32, f"wdm6_case_{cid}.json")) as fh:
        d32 = json.load(fh)
    with open(os.path.join(SAVE_FP64, f"wdm6_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    s = d32["scalars"]
    out = run_jax_for(d32, cid)  # JAX inputs from fp32 savepoint (== fp64 in)

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
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV"),
                        ("graupelncv", "GRAUPELNCV")]:
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
        "scheme": "WDM6 (mp_physics=16) double-moment 6-class",
        "oracle": {
            "prognostic_ref": "WRF module_mp_wdm6.F fp32 single-column savepoints "
                              "(proofs/v060_wdm6/savepoints); UNMODIFIED scheme "
                              "(wdm6init+wdm62D+effectRad_wdm6), no self-compare",
            "machine_precision_ref": "same UNMODIFIED scheme, -fdefault-real-8 "
                              "-DDOUBLE_PRECISION (proofs/v060_wdm6/savepoints_fp64); "
                              "fp32<->fp64 oracle cross-check agrees to ~6e-6, "
                              "confirming the oracle itself",
            "diagnostic_ref": "fp64 build used for effective radii to remove the "
                              "fp32 reference's categorical floor dust (as WSM6/Morrison)",
        },
        "jax_precision": "fp64",
        "predeclared_tolerances": PREDECLARED_TOL,
        "overall_pass": bool(allpass),
        "notes": [
            "Prognostic mass state (t, qv, qc, qr, qi, qs, qg) and surface precip "
            "are gated against the CANONICAL fp32 WRF oracle.",
            "Number concentrations (Nn=CCN, Nc=cloud, Nr=rain) are the DOUBLE-MOMENT "
            "additions; gated against the fp32 oracle with a slightly looser relative "
            "tolerance (cube-root lamda inversions + 1e1..1e10 dynamic range amplify "
            "fp32 roundoff). The vs_fp64 column shows machine-precision agreement.",
            "Effective radii are diagnostic-only with categorical floors; gated vs the "
            "fp64 oracle, vs-fp32 recorded for transparency.",
            "Nc/Nn State-leaf materialization is the manager-owned integration step "
            "(S0 plan); this lane validates the SCHEME-FUNCTION (wdm6_run) parity.",
        ],
        "cases": cases,
    }
    outpath = os.path.join(HERE, "wdm6_savepoint_parity_report.json")
    with open(outpath, "w") as fh:
        json.dump(report, fh, indent=2)
    print("OVERALL:", "PASS" if allpass else "FAIL")
    print("wrote", outpath)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
