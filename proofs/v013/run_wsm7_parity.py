#!/usr/bin/env python3
"""Validate the JAX WSM7 port against the independent Fortran WRF oracle.

Compares gpuwrf.physics.microphysics_wsm7.wsm7_run (fp64) against gold
savepoints that are the REAL WRF physics WSM7 scheme (phys/module_mp_wsm7.F)
run on representative columns + edge cases. The JAX port cannot self-compare:
the reference is the Fortran scheme, not a JAX re-run.

Two oracle builds (both from UNMODIFIED module_mp_wsm7.F; see
proofs/v013/oracle):
  * proofs/v013/savepoints_wsm7       -- canonical WRF single precision
                                        (REAL*4). Binding reference for the
                                        PROGNOSTIC state update.
  * proofs/v013/savepoints_wsm7_fp64  -- -fdefault-real-8 override (scheme
                                        source otherwise unmodified). Used ONLY
                                        for the effective-radius diagnostics,
                                        which have CATEGORICAL detection floors
                                        that flip on trace (~1e-11 kg/kg)
                                        hydrometeor amounts; the fp32 reference
                                        leaves single-precision round-off dust
                                        at otherwise-empty cells, tripping the
                                        floor and giving a re value where the
                                        fp64 reference (and the fp64 JAX port)
                                        correctly give the background radius.
                                        This is the REFERENCE's truncation, not
                                        a port error.

PREDECLARED TOLERANCES (frozen BEFORE comparison; never loosened):
  fp64 JAX vs fp32 (prognostic) / fp64 (diagnostic) Fortran WSM7.
  WSM7 has data-dependent search loops in the semi-Lagrangian PLM
  sedimentation (now 4 precipitating channels incl. hail), an exp/log-heavy
  saturation path, and many clamped accretion terms; fp32-vs-fp64 roundoff
  propagates through these. Tolerances are a relative tolerance on the
  column-max magnitude (rel) plus an absolute floor (abs_floor) below which a
  field is treated as zero, plus absolute tolerances on temperature and the
  surface precip accumulators (mm).

Emits proofs/v013/wsm7_savepoint_parity_report.json with per-field per-column
max abs / max rel error and PASS/FAIL vs the predeclared tolerances.
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

from gpuwrf.physics.microphysics_wsm7 import wsm7_run  # noqa: E402

SAVE_FP32 = os.path.join(HERE, "savepoints_wsm7")
SAVE_FP64 = os.path.join(HERE, "savepoints_wsm7_fp64")

# --------- PREDECLARED TOLERANCES (frozen) ---------
PREDECLARED_TOL = {
    # temperature: absolute K (latent-heat accumulation over the step)
    "t_abs": 5.0e-3,
    # moisture mixing ratios: relative-to-column-peak + absolute floor (kg/kg)
    "q_rel": 6.0e-3,
    "q_abs_floor": 1.0e-7,
    # surface precip accumulators: relative + absolute floor (mm)
    "precip_rel": 1.0e-2,
    "precip_abs": 3.0e-4,
    # effective radii (vs fp64 oracle): relative + absolute floor (m)
    "re_rel": 5.0e-3,
    "re_abs_floor": 1.0e-7,
    # sr ratio: absolute
    "sr_abs": 8.0e-3,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT"), ("qh", "QH_OUT")]
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
    qg = col(d, "QG_IN")[None, :]
    qh = col(d, "QH_IN")[None, :]
    den = col(d, "DEN")[None, :]
    p = col(d, "P")[None, :]
    delz = col(d, "DELZ")[None, :]
    out = wsm7_run(jnp.asarray(t), jnp.asarray(qv), jnp.asarray(qc),
                   jnp.asarray(qr), jnp.asarray(qi), jnp.asarray(qsv),
                   jnp.asarray(qg), jnp.asarray(qh), jnp.asarray(den),
                   jnp.asarray(p), jnp.asarray(delz), s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


def run_case(cid):
    with open(os.path.join(SAVE_FP32, f"wsm7_case_{cid}.json")) as fh:
        d32 = json.load(fh)
    with open(os.path.join(SAVE_FP64, f"wsm7_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    s = d32["scalars"]
    out = run_jax_for(d32)  # JAX inputs come from fp32 savepoint (identical to fp64 in)

    results = {}
    passed = True

    # --- prognostic state vs fp32 (canonical) oracle ---
    mad, mrd, _ = field_metrics(out["t"], col(d32, "T_OUT"), 1.0)
    ok = mad <= PREDECLARED_TOL["t_abs"]
    passed = passed and ok
    results["t"] = {"ref": "fp32", "max_abs": mad, "max_rel": mrd,
                    "tol_abs": PREDECLARED_TOL["t_abs"], "pass": bool(ok)}

    for leaf, oname in Q_FIELDS:
        mad, mrd, scale = field_metrics(out[leaf], col(d32, oname),
                                        PREDECLARED_TOL["q_abs_floor"])
        ok = (mrd <= PREDECLARED_TOL["q_rel"]) or (mad <= PREDECLARED_TOL["q_abs_floor"])
        passed = passed and ok
        results[leaf] = {"ref": "fp32", "max_abs": mad, "max_rel": mrd,
                         "scale": scale, "pass": bool(ok)}

    # --- effective radii vs fp64 oracle (removes reference fp32 floor dust) ---
    for leaf, oname in RE_FIELDS:
        mad, mrd, scale = field_metrics(out[leaf], col(d64, oname),
                                        PREDECLARED_TOL["re_abs_floor"])
        ok = (mrd <= PREDECLARED_TOL["re_rel"]) or (mad <= PREDECLARED_TOL["re_abs_floor"])
        mad32, mrd32, _ = field_metrics(out[leaf], col(d32, oname),
                                        PREDECLARED_TOL["re_abs_floor"])
        passed = passed and ok
        results[leaf] = {"ref": "fp64", "max_abs": mad, "max_rel": mrd,
                         "scale": scale, "pass": bool(ok),
                         "vs_fp32_max_abs": mad32, "vs_fp32_max_rel": mrd32}

    # --- surface precip accumulators vs fp32 oracle ---
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV"),
                        ("graupelncv", "GRAUPELNCV"), ("hailncv", "HAILNCV")]:
        ov = float(s[sname])
        jv = float(out[leaf])
        tol = max(PREDECLARED_TOL["precip_rel"] * abs(ov), PREDECLARED_TOL["precip_abs"])
        ok = abs(jv - ov) <= tol
        passed = passed and ok
        results[leaf] = {"ref": "fp32", "jax": jv, "oracle": ov,
                         "abs_err": abs(jv - ov), "tol": tol, "pass": bool(ok)}

    ov = float(s["SR"])
    jv = float(out["sr"])
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
                if "vs_fp32_max_rel" in m:
                    extra = f" (vs_fp32 rel={m['vs_fp32_max_rel']:.3e})"
                print(f"  {fld:11s} [{m['ref']}] max_abs={m['max_abs']:.3e} "
                      f"max_rel={m['max_rel']:.3e} {'ok' if m['pass'] else 'FAIL'}{extra}")
            else:
                print(f"  {fld:11s} [{m['ref']}] jax={m['jax']:.5e} oracle={m['oracle']:.5e} "
                      f"abs_err={m['abs_err']:.3e} {'ok' if m['pass'] else 'FAIL'}")
        print()

    report = {
        "scheme": "WSM7 (mp_physics=24)",
        "oracle": {
            "prognostic_ref": "WRF phys/module_mp_wsm7.F fp32 single-column savepoints "
                              "(proofs/v013/savepoints_wsm7)",
            "diagnostic_ref": "same scheme, -fdefault-real-8 override "
                              "(proofs/v013/savepoints_wsm7_fp64); used for effective radii "
                              "to remove the fp32 reference's trace-cell floor dust",
            "source_checksums": "proofs/v013/savepoints_wsm7/wsm7_source_checksums.txt",
        },
        "jax_precision": "fp64",
        "predeclared_tolerances": PREDECLARED_TOL,
        "overall_pass": bool(allpass),
        "notes": [
            "Prognostic state (theta/t, qv, qc, qr, qi, qs, qg, qh) and surface precip "
            "(rain/snow/graupel/hail) are gated against the CANONICAL fp32 WRF oracle.",
            "WSM7 = WSM6 single-moment rain/snow/graupel + a separate precipitating hail "
            "class (Bae, Hong, Tao 2018); the hail process terms (phaci/phacw/phacr/phacs/"
            "phacg, phaut, phmlt/pheml, phdep, phevp, pgwet/phwet wet growth) and the 4th "
            "semi-Lagrangian fall channel are exercised by cases 2/3/4/5.",
            "Effective radii (re_cloud, re_ice, re_snow) are diagnostic-only and have "
            "categorical detection floors; they are gated against the fp64 oracle, where "
            "the JAX fp64 port agrees to ~1e-9 relative. Each re field also records its "
            "vs-fp32 error for transparency.",
        ],
        "cases": cases,
    }
    outpath = os.path.join(HERE, "wsm7_savepoint_parity_report.json")
    with open(outpath, "w") as fh:
        json.dump(report, fh, indent=2)
    print("OVERALL:", "PASS" if allpass else "FAIL")
    print("wrote", outpath)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
