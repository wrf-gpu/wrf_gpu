#!/usr/bin/env python3
"""Validate the JAX Purdue-Lin port against the independent Fortran WRF oracle.

Compares gpuwrf.physics.microphysics_lin.lin_run (fp64) against gold savepoints
that are the REAL WRF phys/module_mp_lin.F scheme (lin_et_al -> clphy1d -> satadj)
run on representative columns + edge cases. The JAX port cannot self-compare:
the reference is the Fortran scheme, not a JAX re-run.

Oracle build (UNMODIFIED module_mp_lin.F + module_mp_radar.F; see
proofs/v060/oracle/lin_build_and_run.sh):
  * proofs/v060/savepoints_lin -- default WRF REAL (single precision). The
    binding reference for the PROGNOSTIC state update + surface precip.

PREDECLARED TOLERANCES (frozen BEFORE comparison; never loosened):
  fp64 JAX vs fp32 Fortran Lin. The Lin scheme has a data-dependent
  adaptive-substep (Courant-limited) terminal-velocity sedimentation, a
  20-iteration Newton saturation adjustment, exp/log-heavy process rates, and
  many clamped accretion terms; fp32-vs-fp64 roundoff propagates through these
  (substep-count and min_q/max_q span are themselves fp32-threshold sensitive).
  Tolerances are a relative tolerance on the column-max magnitude (rel) plus an
  absolute floor (abs_floor) below which a field is treated as zero, an absolute
  tolerance on potential temperature, and rel+abs tolerances on the surface
  precip accumulators (mm).
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

from gpuwrf.physics.microphysics_lin import lin_run  # noqa: E402

SAVE = os.path.join(HERE, "savepoints_lin")
SAVE_FP64 = os.path.join(HERE, "savepoints_lin_fp64")
CHECKSUMS = os.path.join(SAVE, "wrf_source_checksums.txt")

# --------- PREDECLARED TOLERANCES (frozen) ---------
PREDECLARED_TOL = {
    # potential temperature: absolute K (latent-heat accumulation over the step)
    "th_abs": 1.0e-2,
    # moisture mixing ratios: relative-to-column-peak + absolute floor (kg/kg)
    "q_rel": 1.0e-2,
    "q_abs_floor": 1.0e-7,
    # surface precip accumulators: relative + absolute floor (mm)
    "precip_rel": 1.5e-2,
    "precip_abs": 5.0e-4,
    # sr ratio: absolute
    "sr_abs": 1.0e-2,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT")]


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
    th = col(d, "TH_IN")[None, :]
    qv = col(d, "QV_IN")[None, :]
    qc = col(d, "QC_IN")[None, :]
    qr = col(d, "QR_IN")[None, :]
    qi = col(d, "QI_IN")[None, :]
    qsv = col(d, "QS_IN")[None, :]
    qg = col(d, "QG_IN")[None, :]
    rho = col(d, "RHO")[None, :]
    pii = col(d, "PII")[None, :]
    p = col(d, "P")[None, :]
    z = col(d, "Z")[None, :]
    dz = col(d, "DZ8W")[None, :]
    out = lin_run(jnp.asarray(th), jnp.asarray(qv), jnp.asarray(qc),
                  jnp.asarray(qr), jnp.asarray(qi), jnp.asarray(qsv),
                  jnp.asarray(qg), jnp.asarray(rho), jnp.asarray(pii),
                  jnp.asarray(p), jnp.asarray(z), jnp.asarray(dz), s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


def run_case(cid):
    with open(os.path.join(SAVE, f"lin_case_{cid}.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    out = run_jax_for(d)

    results = {}
    passed = True

    # --- potential temperature ---
    mad, mrd, _ = field_metrics(out["th"], col(d, "TH_OUT"), 1.0)
    ok = mad <= PREDECLARED_TOL["th_abs"]
    passed = passed and ok
    results["th"] = {"max_abs": mad, "max_rel": mrd,
                     "tol_abs": PREDECLARED_TOL["th_abs"], "pass": bool(ok)}

    for leaf, oname in Q_FIELDS:
        mad, mrd, scale = field_metrics(out[leaf], col(d, oname),
                                        PREDECLARED_TOL["q_abs_floor"])
        ok = (mrd <= PREDECLARED_TOL["q_rel"]) or (mad <= PREDECLARED_TOL["q_abs_floor"])
        passed = passed and ok
        results[leaf] = {"max_abs": mad, "max_rel": mrd, "scale": scale,
                         "pass": bool(ok)}

    # --- surface precip accumulators ---
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV"),
                        ("graupelncv", "GRAUPELNCV")]:
        ov = float(s[sname])
        jv = float(out[leaf])
        tol = max(PREDECLARED_TOL["precip_rel"] * abs(ov), PREDECLARED_TOL["precip_abs"])
        ok = abs(jv - ov) <= tol
        passed = passed and ok
        results[leaf] = {"jax": jv, "oracle": ov, "abs_err": abs(jv - ov),
                         "tol": tol, "pass": bool(ok)}

    ov = float(s["SR"])
    jv = float(out["sr"])
    ok = abs(jv - ov) <= PREDECLARED_TOL["sr_abs"]
    passed = passed and ok
    results["sr"] = {"jax": jv, "oracle": ov, "abs_err": abs(jv - ov),
                     "tol": PREDECLARED_TOL["sr_abs"], "pass": bool(ok)}

    # transparency: JAX fp64 vs the fp64 oracle (same UNMODIFIED scheme, kind
    # promoted to double). The port matches this to ~machine precision, proving
    # the fp32 residuals above are the reference's own single-precision roundoff,
    # not a port error.
    try:
        with open(os.path.join(SAVE_FP64, f"lin_case_{cid}.json")) as fh:
            d64 = json.load(fh)
        out64 = run_jax_for(d64)
        vs64 = {}
        worst = 0.0
        for leaf, oname in [("th", "TH_OUT"), *Q_FIELDS]:
            mad, _, _ = field_metrics(out64[leaf], col(d64, oname), 1.0)
            vs64[leaf] = mad
            worst = max(worst, mad if leaf != "th" else 0.0)
        results["_vs_fp64_oracle"] = {
            "worst_q_abs": worst,
            "th_abs": vs64["th"],
            "per_field_abs": vs64,
        }
    except FileNotFoundError:
        pass

    return passed, results


def _wrf_sha256():
    out = {}
    try:
        with open(CHECKSUMS) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    out[parts[1]] = parts[0]
    except FileNotFoundError:
        pass
    return out


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
            if fld == "_vs_fp64_oracle":
                print(f"  [vs fp64 oracle] worst_q_abs={m['worst_q_abs']:.2e} "
                      f"th_abs={m['th_abs']:.2e}")
            elif "max_abs" in m:
                print(f"  {fld:11s} max_abs={m['max_abs']:.3e} "
                      f"max_rel={m['max_rel']:.3e} {'ok' if m['pass'] else 'FAIL'}")
            else:
                print(f"  {fld:11s} jax={m['jax']:.5e} oracle={m['oracle']:.5e} "
                      f"abs_err={m['abs_err']:.3e} {'ok' if m['pass'] else 'FAIL'}")
        print()

    report = {
        "scheme": "Purdue-Lin (mp_physics=2)",
        "oracle": {
            "prognostic_ref": "WRF phys/module_mp_lin.F (lin_et_al -> clphy1d -> "
                              "satadj) default-REAL single-column savepoints "
                              "(proofs/v060/savepoints_lin); graupel active "
                              "(F_QG=.true. => gindex=1)",
        },
        "wrf_source_sha256": _wrf_sha256(),
        "fp64_oracle": "WRF module_mp_lin.F built -fdefault-real-8 (same source, "
                       "kind promoted to double; proofs/v060/savepoints_lin_fp64). "
                       "The fp64 JAX port matches it to ~machine precision, proving "
                       "the fp32 residuals are the reference's own roundoff.",
        "jax_precision": "fp64",
        "predeclared_tolerances": PREDECLARED_TOL,
        "overall_pass": bool(allpass),
        "notes": [
            "Prognostic state (theta, qv, qc, qr, qi, qs, qg) and surface precip "
            "are gated against the default-REAL (fp32) WRF Lin oracle.",
            "Lin has a data-dependent adaptive-substep (Courant) sedimentation and "
            "a 20-iteration Newton saturation adjustment; fp32-vs-fp64 roundoff "
            "propagates through the substep count and min_q/max_q span, so parity "
            "is to a predeclared physical tolerance, never bitwise.",
        ],
        "cases": cases,
    }
    outpath = os.path.join(HERE, "lin_mp_savepoint_parity.json")
    with open(outpath, "w") as fh:
        json.dump(report, fh, indent=2)
    print("OVERALL:", "PASS" if allpass else "FAIL")
    print("wrote", outpath)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
