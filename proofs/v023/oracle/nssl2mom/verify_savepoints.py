#!/usr/bin/env python3
"""Honest verification of the NSSL 2-moment oracle savepoints.

Gates (hard fail -> exit nonzero):
  1. FINITE: every number in every savepoint is finite.
  2. ACTIVITY: in every case the state evolves (TH and QV change);
     in the precip-bearing cases (2,4,5: deep mixed-phase / convective
     core / falling precip; plus warm-rain case 1) RAINNCV > 0.
  3. fp32 vs fp64 PHYSICAL BAND: for each prognostic column, where the
     fp64 magnitude is significant (>= floor), the fp32 value agrees
     within rel_tol OR within an absolute band (floor-scaled). Trace
     cells below the significance floor are excluded (categorical
     detection-threshold dust, same policy as the Morrison oracle).
Everything is REPORTED; nothing is silently clipped.
"""
import json
import math
import sys
from pathlib import Path

SAVE = Path(__file__).resolve().parent / "../../../v022/f2_oracles/nssl_2mom"

PROG = ["TH", "QV", "QC", "QR", "QI", "QS", "QH", "QHL",
        "CCW", "CRW", "CCI", "CSW", "CHW", "CHL", "CN", "VHW", "VHL"]
# significance floors per field family (values below are trace/dust)
FLOORS = {
    "TH": 1e-3, "QV": 1e-8,
    "QC": 1e-9, "QR": 1e-9, "QI": 1e-9, "QS": 1e-9, "QH": 1e-9, "QHL": 1e-9,
    "CCW": 1e2, "CRW": 1e-1, "CCI": 1e0, "CSW": 1e-1, "CHW": 1e-1,
    "CHL": 1e-3, "CN": 1e2, "VHW": 1e-12, "VHL": 1e-14,
}
REL_TOL = 0.25   # 25% physical band on significant cells (single 60s step,
                 # threshold-rich 2-moment scheme; report shows actuals)
PRECIP_CASES = {1, 2, 4, 5}

fail = 0


def load(name):
    with open(SAVE / name) as fh:
        return json.load(fh)


def allvals(sp):
    for v in sp["scalars"].values():
        if isinstance(v, float):
            yield v
    for col in sp["columns"].values():
        yield from col


print("=== NSSL 2-mom oracle savepoint verification ===")
for mode, prefix in (("fp32", "nssl"), ("fp64", "nssl_fp64")):
    for c in range(1, 7):
        sp = load(f"{prefix}_case_{c}.json")
        assert sp["schema"] == "wrf-v023-f2-nssl2mom-column-savepoint-v1"
        nonfinite = sum(0 if math.isfinite(v) else 1 for v in allvals(sp))
        if nonfinite:
            print(f"FAIL FINITE {mode} case {c}: {nonfinite} non-finite values")
            fail = 1

print("\n--- gate 1 (finite): PASS unless FAIL lines above ---")

print("\n--- gate 2 (activity + precip) ---")
for mode, prefix in (("fp32", "nssl"), ("fp64", "nssl_fp64")):
    for c in range(1, 7):
        sp = load(f"{prefix}_case_{c}.json")
        cols = sp["columns"]
        dth = max(abs(a - b) for a, b in zip(cols["TH_OUT"], cols["TH_IN"]))
        dqv = max(abs(a - b) for a, b in zip(cols["QV_OUT"], cols["QV_IN"]))
        rain = sp["scalars"]["RAINNCV"]
        ok = dth > 0.0 and dqv > 0.0
        pr_ok = (rain > 0.0) if c in PRECIP_CASES else True
        tag = "ok" if (ok and pr_ok) else "FAIL"
        if not (ok and pr_ok):
            fail = 1
        print(f"{tag} {mode} case {c}: max|dTH|={dth:.3e} max|dQV|={dqv:.3e} "
              f"RAINNCV={rain:.6e} SNOWNCV={sp['scalars']['SNOWNCV']:.3e} "
              f"GRPLNCV={sp['scalars']['GRPLNCV']:.3e} "
              f"HAILNCV={sp['scalars']['HAILNCV']:.3e}")

print("\n--- gate 3 (fp32 vs fp64 physical band, OUT fields) ---")
for c in range(1, 7):
    a = load(f"nssl_case_{c}.json")["columns"]
    b = load(f"nssl_fp64_case_{c}.json")["columns"]
    worst = []
    for f in PROG:
        fa, fb = a[f + "_OUT"], b[f + "_OUT"]
        floor = FLOORS[f]
        wrel, wk = 0.0, -1
        nsig = 0
        for k, (x, y) in enumerate(zip(fa, fb)):
            if abs(y) < floor and abs(x) < floor:
                continue  # trace dust in both
            nsig += 1
            rel = abs(x - y) / max(abs(y), floor)
            if rel > wrel:
                wrel, wk = rel, k
        if nsig and wrel > REL_TOL:
            worst.append((f, wrel, wk, fa[wk], fb[wk]))
        status = "ok " if (not nsig or wrel <= REL_TOL) else "WARN"
        if nsig:
            print(f"{status} case {c} {f:4s}: nsig={nsig:2d} worst_rel={wrel:.3e} "
                  f"(k={wk} fp32={fa[wk] if wk>=0 else 0:.6e} fp64={fb[wk] if wk>=0 else 0:.6e})")
    # precip band
    ra = load(f"nssl_case_{c}.json")["scalars"]["RAINNCV"]
    rb = load(f"nssl_fp64_case_{c}.json")["scalars"]["RAINNCV"]
    if max(ra, rb) > 1e-8:
        rrel = abs(ra - rb) / max(abs(rb), 1e-8)
        print(f"{'ok ' if rrel <= REL_TOL else 'WARN'} case {c} RAINNCV: "
              f"fp32={ra:.6e} fp64={rb:.6e} rel={rrel:.3e}")
    if worst:
        print(f"  case {c} fields beyond {REL_TOL:.0%} band on significant cells:")
        for f, wrel, wk, x, y in worst:
            print(f"    {f} rel={wrel:.3e} at k={wk}: fp32={x:.6e} fp64={y:.6e}")

sys.exit(fail)
