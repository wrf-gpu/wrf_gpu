"""Render the v0.1.0 d02 validation result JSON as compact human tables.

USAGE
-----
  python proofs/v010_validation/render_table.py \
      --result proofs/v010_validation/v010_d02_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("T2", "U10", "V10", "PRECIP")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path,
                    default=Path("proofs/v010_validation/v010_d02_result.json"))
    args = ap.parse_args()
    d = json.loads(args.result.read_text())

    print(f"verdict={d.get('verdict')}  all_pass={d.get('all_pass')}  "
          f"no_blowup={d.get('no_blowup')}  wall_s={d.get('wall_s')}")
    print(f"base_model: {d.get('base_model')}")
    print(f"tenerife_box: {d.get('tenerife_box')}")
    print()

    # ---- RMSE tables (full + tenerife) ----
    for region in ("full", "tenerife"):
        print(f"===== {region.upper()} DOMAIN — GPU-vs-nightly-WRF RMSE (bias) =====")
        hdr = (f"{'case':6} {'lvl':3} {'lead':>4} "
               f"{'T2 rmse(bias)':>16} {'U10 rmse(bias)':>16} "
               f"{'V10 rmse(bias)':>16} {'PRECIP rmse':>12} {'fin':>4}")
        print(hdr)
        print("-" * len(hdr))
        for cid, c in d["results"].items():
            for lvl, lr in c["levels"].items():
                for lead in sorted(lr["scores"], key=lambda x: int(x)):
                    s = lr["scores"][lead][region]
                    fin = all(s[f]["gpu_finite"] for f in FIELDS)
                    def cell(f):
                        return f"{s[f]['rmse']:6.2f}({s[f]['bias']:+5.2f})"
                    print(f"{cid:6} {lvl:3} {lead:>4} "
                          f"{cell('T2'):>16} {cell('U10'):>16} {cell('V10'):>16} "
                          f"{s['PRECIP']['rmse']:>12.3f} {('Y' if fin else 'N'):>4}")
        print()

    # ---- persistence skill ----
    print("===== SKILL vs PERSISTENCE (1 - GPU_RMSE/pers_RMSE; >0 GPU wins) =====")
    for cid, c in d["results"].items():
        for lvl, lr in c["levels"].items():
            sk = lr.get("skill_vs_persistence")
            if not sk:
                continue
            for region in ("full", "tenerife"):
                bf = sk[region]["by_field"]
                parts = []
                for f in ("T2", "U10", "V10"):
                    b = bf[f]
                    ms = b["mean_skill"]
                    parts.append(f"{f} skill={ms:+.2f}(W{b['wins']}/T{b['ties']}/L{b['losses']})"
                                 if ms is not None else f"{f} n/a")
                print(f"{cid}/{lvl} {region:8}: " + "  ".join(parts))
    print()

    # ---- physical-plausibility / no-blowup summary ----
    print("===== NO-BLOW-UP (gpu_mean / gpu_std over FULL run) =====")
    for cid, c in d["results"].items():
        for lvl, lr in c["levels"].items():
            leads = sorted(lr["scores"], key=lambda x: int(x))
            last = lr["scores"][leads[-1]]["full"]
            print(f"{cid}/{lvl} @+{leads[-1]}h: "
                  f"T2 mean={last['T2']['gpu_mean']:.1f} std={last['T2']['gpu_std']:.1f} | "
                  f"U10 mean={last['U10']['gpu_mean']:+.1f} std={last['U10']['gpu_std']:.1f} | "
                  f"V10 mean={last['V10']['gpu_mean']:+.1f} std={last['V10']['gpu_std']:.1f} | "
                  f"finite={all(last[f]['gpu_finite'] for f in FIELDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
