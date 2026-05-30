#!/usr/bin/env python3
"""Render the M19 verdict_result.json into a human-readable per-(case,level,lead)
table. READ-ONLY reporting -- recomputes the ceiling locally only to label
pass/fail per field (same formula as the harness; does NOT alter any scoring or
the verdict already baked into the JSON)."""
import json
import sys
from pathlib import Path

RMSE_CEILING = {"T2": (4.0, 1.0), "U10": (5.0, 1.0), "V10": (6.0, 1.0)}
FIELDS = ("T2", "U10", "V10")


def ceil(field, lead_h):
    b, g = RMSE_CEILING[field]
    return b + g * (lead_h / 24.0)


def main(path):
    d = json.loads(Path(path).read_text())
    print(f"MODE={d['mode']}  VERDICT={d['verdict']}  stopped_at={d.get('stopped_at')}")
    print(f"leads_h={d['leads_h']}  wall_s(total)={d.get('wall_s')}")
    print("ceiling formula:", d["thresholds"]["rmse_ceiling_formula"])
    print()
    hdr = (f"{'case':6} {'lvl':4} {'lead':5} | "
           f"{'fld':4} {'rmse':8} {'bias':8} {'ceil':6} {'finite':6} {'PF':4}")
    print(hdr)
    print("-" * len(hdr))
    for cid, cres in d["results"].items():
        for lvl, lres in cres["levels"].items():
            scores = lres["scores"]
            walls = lres.get("wall_s_by_lead", {})
            nt = lres.get("no_truth_leads", [])
            for lead_s in sorted(scores, key=lambda x: int(x)):
                lead = int(lead_s)
                s = scores[lead_s]
                w = walls.get(lead_s, walls.get(str(lead), "-"))
                for fi, f in enumerate(FIELDS):
                    fs = s[f]
                    c = ceil(f, lead)
                    ok = fs["gpu_finite"] and fs["rmse"] <= c
                    prefix = (f"{cid:6} {lvl:4} {lead:>4}h" if fi == 0
                              else f"{'':6} {'':4} {'':5}")
                    print(f"{prefix} | {f:4} {fs['rmse']:8.3f} {fs['bias']:+8.3f} "
                          f"{c:6.2f} {str(fs['gpu_finite']):6} "
                          f"{'PASS' if ok else 'FAIL':4}"
                          + (f"  wall={w}s" if fi == 0 else ""))
            if nt:
                print(f"{cid:6} {lvl:4}  no-truth leads (skipped): {nt}")
        print(f"  -> {cid} verdict: {'PASS' if cres['passed'] else 'FAIL'}")
    if d.get("probe_pointer_file"):
        print("\nPROBE POINTER:", d["probe_pointer_file"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "proofs/m19/verdict_result.json")
