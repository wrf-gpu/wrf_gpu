#!/usr/bin/env python3
"""Validate the NumPy KF reference (cumulus_kf_reference.kf_eta_para_np) against
the independent Fortran oracle savepoints (proofs/p0_4/savepoints/*.json).

The reference is fp64; the Fortran oracle is REAL*4. Parity is therefore to a
PREDECLARED physical tolerance (see PREDECLARED_TOL below), not bitwise.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from gpuwrf.physics import cumulus_kf_reference as ref  # noqa: E402

SAVE = os.path.join(os.path.dirname(__file__), "savepoints")

# --------- PREDECLARED TOLERANCES (frozen BEFORE comparison) ---------
# fp64 reference vs fp32 Fortran oracle. KF has an iterative CAPE-removal
# closure (AINC secant) + lookup-table interpolation; fp32 vs fp64 roundoff in
# the closure can shift AINC by ~1e-3 relative, which scales all tendencies.
# We therefore declare a RELATIVE tolerance on the column-peak tendency plus a
# small absolute floor, and an absolute tolerance on integral scalars.
PREDECLARED_TOL = {
    "tend_rel": 2.0e-3,    # relative, on |field| normalized by column max
    "tend_abs_floor": 1.0e-9,   # absolute floor (K/s or kg/kg/s) below which we ignore
    "raincv_rel": 3.0e-3,  # convective precip (mm) relative
    "raincv_abs": 1.0e-4,  # mm absolute floor
    "int_scalar_abs": 0.5,  # CUTOP/CUBOT/ISHALL exact-ish (must match exactly)
}

TEND_FIELDS = ["RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN"]


def field_metrics(jax_or_ref, oracle, tol):
    a = np.asarray(jax_or_ref, dtype=np.float64)
    b = np.asarray(oracle, dtype=np.float64)
    scale = max(np.max(np.abs(b)), tol["tend_abs_floor"])
    absdiff = np.abs(a - b)
    reldiff = absdiff / scale
    return float(np.max(absdiff)), float(np.max(reldiff)), scale


def run_case(cid, verbose=True):
    with open(os.path.join(SAVE, f"kf_case_{cid}.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    c = d["columns"]
    kx = s["KX"]
    dt = s["DT"]
    dx = s["DX"]
    t0 = np.array(c["T"]); qv0 = np.array(c["QV"]); p0 = np.array(c["P"])
    dz = np.array(c["DZ"]); rho = np.array(c["RHO"]); w0 = np.array(c["W0AVG"])
    u0 = np.array(c["U"]); v0 = np.array(c["V"])
    pi_ex = (p0 / 1.0e5) ** (ref.R_D / ref.CP)

    out = ref.kf_eta_para_np(t0, qv0, p0, dz, rho, w0, u0, v0, dt, dx,
                             pi_exner=pi_ex, cudt=0.0,
                             warm_rain=False, f_qi=True, f_qs=True, trigger=1)

    passed = True
    msgs = []
    # integer-ish scalars must match exactly (cloud-top/base/regime are categorical)
    for name in ("CUTOP", "CUBOT"):
        ov = s[name]
        rv = out[name]
        if abs(ov - rv) > PREDECLARED_TOL["int_scalar_abs"]:
            passed = False
            msgs.append(f"  {name}: oracle={ov} ref={rv}  MISMATCH")
        else:
            msgs.append(f"  {name}: oracle={ov} ref={rv}  ok")
    ish_oracle = int(round(s["SHALL"]))
    if ish_oracle != out["ISHALL"]:
        passed = False
        msgs.append(f"  ISHALL: oracle={ish_oracle} ref={out['ISHALL']}  MISMATCH")
    else:
        msgs.append(f"  ISHALL: oracle={ish_oracle} ref={out['ISHALL']}  ok")

    # tendency fields
    for f in TEND_FIELDS:
        mad, mrd, scale = field_metrics(out[f], c[f], PREDECLARED_TOL)
        ok = (mrd <= PREDECLARED_TOL["tend_rel"]) or (mad <= PREDECLARED_TOL["tend_abs_floor"])
        passed = passed and ok
        msgs.append(f"  {f}: max_abs={mad:.3e} max_rel={mrd:.3e} scale={scale:.3e} {'ok' if ok else 'FAIL'}")

    # precip
    ov = s["RAINCV"]; rv = out["RAINCV"]
    rok = abs(rv - ov) <= max(PREDECLARED_TOL["raincv_rel"] * abs(ov), PREDECLARED_TOL["raincv_abs"])
    passed = passed and rok
    msgs.append(f"  RAINCV(mm): oracle={ov:.6e} ref={rv:.6e} {'ok' if rok else 'FAIL'}")
    msgs.append(f"  PRATEC: oracle={s['PRATEC']:.6e} ref={out['PRATEC']:.6e}")
    msgs.append(f"  NCA: oracle={s['NCA']:.3f} ref={out['NCA']:.3f}")
    msgs.append(f"  TIMEC: oracle={s['TIMEC']:.1f} ref={out['TIMEC']:.1f}")
    msgs.append(f"  moisture-budget ERR2(%)={out.get('ERR2', float('nan')):.4f}")

    if verbose:
        print(f"=== CASE {cid}  ({'DEEP' if ish_oracle==0 else 'SHALLOW' if ish_oracle==1 else 'NONE'}) "
              f"-> {'PASS' if passed else 'FAIL'} ===")
        print("\n".join(msgs))
    return passed, out, s


def main():
    results = {}
    allpass = True
    for cid in (1, 2, 3, 4):
        ok, out, s = run_case(cid)
        results[cid] = ok
        allpass = allpass and ok
        print()
    print("PREDECLARED_TOL =", json.dumps(PREDECLARED_TOL))
    print("OVERALL:", "PASS" if allpass else "FAIL", results)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
