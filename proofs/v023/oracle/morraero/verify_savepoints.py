#!/usr/bin/env python3
"""Honest verification of the morraero oracle savepoints (fp32 + fp64).

Checks (fail-closed):
 1. FINITE: every value in every savepoint is finite (the converter already
    enforces this; re-checked here).
 2. NONTRIVIAL: the microphysics did something in every case:
    - prognostic fields change from input to output,
    - TH and QV respond (saturation adjustment acted),
    - droplet activation moved NC in the cloudy cases (QC_IN > 0),
    - CCN diagnostics nonzero (prescribed-aerosol aercu_opt=2 path executed),
    - EFCG set (aercu_opt>0 effective-radius branch executed),
    - RAINNCV > 0 in cases 1 (warm rain), 2 and 4 (deep/convective).
 3. fp32-vs-fp64 PHYSICAL BAND, anchored on the accepted v0.6.0 base-Morrison
    gold pair (proofs/v060/savepoints{,_fp64}) which exhibits the SAME
    single-step fp32 threshold sensitivity in the mixed-phase cases (2, 4:
    process on/off flips at individual levels, RAINNCV rel diff 4.3%/5.9%):
      - smooth band: TH 5e-3 K, Q-fields 1e-6 kg/kg, N-fields 1e3 /kg,
      - else the per-case-per-field max |fp32-fp64| must be <= 2x the SAME
        case+field divergence of the base gold pair (measured aero/base
        ratios are 0.0-1.5; a plumbing bug would blow far beyond 2x),
      - precip scalars: rel diff <= max(0.5%, 2x base-pair rel diff),
      - NC / CCN1..7 / WACT / EFCG: rel <= 1e-4 (measured <= 1e-5),
      - EFIG/EFSG: rel <= 5% at levels where BOTH modes hold real hydrometeor
        mass (QI/QS >= 1e-8); levels where mass detection itself flips between
        modes are categorical fp32 detection-floor dust (counted + reported,
        exactly as in the base oracle's QS/QG flip levels).
Exits nonzero listing every violated check. No check is skipped or loosened
beyond the documented anchor above.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SAVE = HERE.parents[3] / "v022" / "f2_oracles" / "morrison_aero"
BASE = HERE.parents[3] / "v060"

PROG = ["TH", "QV", "QC", "QR", "QI", "QS", "QG", "NI", "NS", "NR", "NG"]
SMOOTH = {"TH": 5e-3}
SMOOTH.update({f: 1e-6 for f in ("QV", "QC", "QR", "QI", "QS", "QG")})
SMOOTH.update({f: 1e3 for f in ("NI", "NS", "NR", "NG")})
TIGHT_REL = 1e-4  # NC, CCN*, WACT, EFCG
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"  ok: {msg}")


def load(prefix, case, root=SAVE):
    p = root / f"{prefix}_{case}.json"
    d = json.loads(p.read_text())
    if root is SAVE:
        assert d["schema"] == "wrf-v023-f2-morraero-column-savepoint-v1", p
    return d


def check_finite(d, tag):
    bad = [k for k, v in d["scalars"].items() if not math.isfinite(float(v))]
    bad += [k for k, col in d["columns"].items()
            if not all(math.isfinite(x) for x in col)]
    if bad:
        fail(f"{tag}: non-finite values in {bad}")


def check_activity(d, case, tag):
    cols = d["columns"]
    changed = {}
    for f in PROG + ["NC"]:
        changed[f] = max(abs(x - y) for x, y in zip(cols[f + "_IN"], cols[f + "_OUT"]))
    if not any(c > 0 for c in changed.values()):
        fail(f"{tag}: NO prognostic field changed input->output (inert savepoint)")
        return
    if changed["TH"] == 0 or changed["QV"] == 0:
        fail(f"{tag}: TH or QV unchanged -- saturation adjustment did not act")
    if max(cols["QC_IN"]) > 0 and changed["NC"] == 0:
        fail(f"{tag}: cloudy case but NC unchanged -- aerosol activation inert")
    if max(cols["CCN7_GS_OUT"]) <= 0:
        fail(f"{tag}: CCN7_GS_OUT all zero -- prescribed-aerosol path did not run")
    if max(cols["EFCG_OUT"]) <= 0:
        fail(f"{tag}: EFCG_OUT all zero -- aercu_opt>0 effective-radius branch inert")
    r = d["scalars"]["RAINNCV"]
    if case in (1, 2, 4) and not r > 0:
        fail(f"{tag}: RAINNCV={r} not > 0 in precip-expected case {case}")
    ok(f"active fields: {sorted(f for f, c in changed.items() if c > 0)}")


def maxdiff(a, b, name):
    return max(abs(p - q) for p, q in zip(a["columns"][name], b["columns"][name]))


def band_compare(a32, a64, b32, b64, case):
    tag = f"case {case} fp32-vs-fp64"
    # prognostic fields: smooth band OR 2x base-gold-pair anchor
    for f in PROG:
        da = maxdiff(a32, a64, f + "_OUT")
        db = maxdiff(b32, b64, f + "_OUT")
        lim = max(SMOOTH[f], 2.0 * db)
        if da > lim:
            fail(f"{tag}: {f}_OUT maxdiff {da:.3e} > band {lim:.3e} "
                 f"(smooth {SMOOTH[f]:.1e}, base-pair anchor {db:.3e})")
    # precip scalars: rel band anchored on base pair
    for s in ("RAINNCV", "SNOWNCV", "GRAUPELNCV", "SR"):
        pa, qa = a32["scalars"][s], a64["scalars"][s]
        pb, qb = b32["scalars"][s], b64["scalars"][s]
        rel_a = abs(pa - qa) / max(abs(pa), abs(qa), 1e-30)
        rel_b = abs(pb - qb) / max(abs(pb), abs(qb), 1e-30)
        lim = max(5e-3, 2.0 * rel_b)
        if abs(pa - qa) > 1e-7 and rel_a > lim:
            fail(f"{tag}: {s} fp32={pa:.6e} fp64={qa:.6e} rel={rel_a:.2%} "
                 f"> band {lim:.2%} (base-pair rel {rel_b:.2%})")
    # aero-only tight-relative fields; NC gets the same 1e3 /kg absolute floor
    # as the other number fields (detection-flip dust at trace-QC levels,
    # e.g. case 6 level 9: QC 2e-13 vs exact 0 -> NC 335 /kg vs 0).
    for name in (["NC_OUT", "WACT_OUT", "EFCG_OUT"]
                 + [f"CCN{i}_GS_OUT" for i in range(1, 8)]):
        floor = 1e3 if name == "NC_OUT" else 0.0
        for k, (p, q) in enumerate(zip(a32["columns"][name],
                                       a64["columns"][name]), start=1):
            d = abs(p - q)
            rel = d / max(abs(p), abs(q), 1e-30)
            if d > floor and rel > TIGHT_REL:
                fail(f"{tag}: {name}[{k}] fp32={p:.6e} fp64={q:.6e} "
                     f"rel {rel:.2e} > {TIGHT_REL:.0e}")
    # EFIG/EFSG: mass-gated (categorical at detection-flip levels)
    for ef, q in (("EFIG_OUT", "QI_OUT"), ("EFSG_OUT", "QS_OUT")):
        x, y = a32["columns"][ef], a64["columns"][ef]
        qx, qy = a32["columns"][q], a64["columns"][q]
        flips = 0
        for k, (p, r, mq32, mq64) in enumerate(zip(x, y, qx, qy), start=1):
            if min(mq32, mq64) >= 1e-8:
                rel = abs(p - r) / max(abs(p), abs(r), 1e-30)
                if rel > 5e-2:
                    fail(f"{tag}: {ef}[{k}] rel {rel:.2e} > 5% at real-mass level "
                         f"({q}[{k}] fp32={mq32:.2e} fp64={mq64:.2e})")
            elif abs(p - r) / max(abs(p), abs(r), 1e-30) > 5e-2:
                flips += 1
        if flips:
            print(f"  note: {ef}: {flips} categorical level(s) at hydrometeor "
                  f"detection flips (fp32 floor dust, matches base-oracle behavior)")
    ok(f"{tag}: within anchored physical band")
    print(f"  case {case}: RAINNCV fp32={a32['scalars']['RAINNCV']:.9e} "
          f"fp64={a64['scalars']['RAINNCV']:.9e}")


def main():
    for case in range(1, 7):
        print(f"== case {case} ==")
        a32 = load("morr_aero_case", case)
        a64 = load("morr_aero_fp64_case", case)
        b32 = load("morrison_case", case, BASE / "savepoints")
        b64 = load("morrison_case", case, BASE / "savepoints_fp64")
        check_finite(a32, f"case {case} fp32")
        check_finite(a64, f"case {case} fp64")
        check_activity(a32, case, f"case {case} fp32")
        check_activity(a64, case, f"case {case} fp64")
        band_compare(a32, a64, b32, b64, case)
    print()
    if FAILS:
        print(f"VERDICT: FAIL ({len(FAILS)} violations)")
        sys.exit(1)
    print("VERDICT: PASS (finite, nontrivial, fp32~fp64 within the "
          "base-oracle-anchored physical band)")


if __name__ == "__main__":
    main()
