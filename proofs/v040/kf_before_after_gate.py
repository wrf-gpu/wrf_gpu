#!/usr/bin/env python3
"""KF before/after 2-date forecast gate driver.

Runs the v0.4.0 24h native-init forecast gate on the two target l2 d01 cases
(20260429 + 20260521) twice on the SAME commit/numerics, differing ONLY in
cu_physics: 0 (pre-KF baseline) vs 1 (Kain-Fritsch-eta wired). Emits a U10/PSFC
mean+worst before/after comparison vs the cu_physics=1 CPU-WRF reference, and
writes both full reports. WRF-faithful: oracle = unmodified CPU-WRF wrfout.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "proofs/v040/run_forecast_gate_24h.py"

CASES = [
    "20260429_18z_l2_72h_20260524T204451Z",
    "20260521_18z_l2_72h_20260522T133443Z",
]


def run_gate(cu: int, hours: int, dt_s: float, out: Path, output_root: Path) -> Path:
    cmd = [
        sys.executable, str(GATE),
        "--hours", str(hours),
        "--dt-s", str(dt_s),
        "--cu-physics", str(cu),
        "--cudt", "5.0",
        "--out", str(out),
        "--output-root", str(output_root),
    ]
    for c in CASES:
        cmd += ["--case-id", c]
    print(f"\n=== RUN cu_physics={cu} hours={hours} dt={dt_s} -> {out.name} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT / "proofs/v040"))
    return out


def field_table(report: Path, fields=("U10", "V10", "T2", "PSFC")):
    d = json.loads(report.read_text())
    recs = d.get("cases") or d.get("records") or []
    out = {}
    for rec in recs:
        cid = rec.get("case_id", "?")
        pf = rec.get("per_field_summary", {})
        out[cid] = {}
        for f in fields:
            s = pf.get(f, {})
            if isinstance(s, dict) and s.get("status") == "scored":
                out[cid][f] = {
                    "mean_bias": s.get("mean_bias"),
                    "worst_abs_bias": s.get("worst_abs_bias"),
                    "margin": s.get("regression_margin"),
                    "within": s.get("within_margin"),
                }
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--dt-s", type=float, default=60.0)
    ap.add_argument("--tag", default="kf2date")
    args = ap.parse_args()

    base = ROOT / "proofs/v040"
    before = run_gate(0, args.hours, args.dt_s,
                      base / f"forecast_gate_{args.tag}_cu0_BEFORE.json",
                      Path(f"/tmp/v040_kf_gate_cu0"))
    after = run_gate(1, args.hours, args.dt_s,
                     base / f"forecast_gate_{args.tag}_cu1_AFTER.json",
                     Path(f"/tmp/v040_kf_gate_cu1"))

    tb = field_table(before)
    ta = field_table(after)
    print("\n================ KF BEFORE/AFTER (vs cu_physics=1 CPU-WRF) ================")
    print(f"{'case':<20} {'field':<5} {'mean_before':>12} {'mean_after':>12} "
          f"{'worst_before':>13} {'worst_after':>12} {'margin':>9} {'within_after':>12}")
    for cid in CASES:
        for f in ("U10", "V10", "T2", "PSFC"):
            b = tb.get(cid, {}).get(f, {})
            a = ta.get(cid, {}).get(f, {})
            def fmt(x):
                return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
            print(f"{cid[:20]:<20} {f:<5} {fmt(b.get('mean_bias')):>12} {fmt(a.get('mean_bias')):>12} "
                  f"{fmt(b.get('worst_abs_bias')):>13} {fmt(a.get('worst_abs_bias')):>12} "
                  f"{fmt(a.get('margin')):>9} {str(a.get('within')):>12}")
    summary = {"before": str(before), "after": str(after),
               "before_table": tb, "after_table": ta}
    (base / f"forecast_gate_{args.tag}_COMPARE.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote compare:", base / f"forecast_gate_{args.tag}_COMPARE.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
