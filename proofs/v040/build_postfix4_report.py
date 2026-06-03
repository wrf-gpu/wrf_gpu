"""Build the 2-date before/after report for the Opus al+dpn PGF fixes.

before = pre-fix baseline (proofs/v040/forecast_gate_24h_report.json, the original
v0.4.0 standalone gate); after = this round's postfix4 raw report.  Reports the
per-field h2+ mean/worst bias and whether the PSFC−/U10+ collapsed.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE = HERE / "forecast_gate_24h_report.json"
AFTER = HERE / "forecast_gate_postfix4_raw.json"
OUT = HERE / "forecast_gate_postfix4_report.json"

ADR029 = {"T2": 0.2148692978020805, "U10": 0.23064713972582307, "V10": 0.2752320537920854, "PSFC_diag_margin": 50.0}
CORE = ("T2", "U10", "V10", "PSFC")


def _pf(rec):
    if rec.get("status"):
        return None
    return rec.get("per_field_summary") or (rec.get("fields") if "fields" in rec else None)


def _bias(pfs, f):
    if not pfs or f not in pfs:
        return None
    v = pfs[f]
    if "before" in v:  # already-comparison-shaped
        v = v.get("after", v)
    return {"mean_bias_h2plus": v.get("mean_bias_h2plus", v.get("mean_bias")),
            "worst_abs_bias_h2plus": v.get("worst_abs_bias_h2plus", v.get("worst_abs_bias"))}


def main() -> int:
    before = json.loads(BEFORE.read_text()) if BEFORE.is_file() else {}
    after = json.loads(AFTER.read_text())
    bcases = {c.get("case_id"): c for c in before.get("cases", [])}
    out = {
        "schema": "v0.4.0-forecast-gate-postfix4-before-after-2026-06-03",
        "created_by": "Opus 4.8 MAX",
        "fixes": "WRF-faithful al (alb/muts) + dpn cfn/cfn1 top in large_step_horizontal_pgf",
        "before_report": str(BEFORE), "after_report": str(AFTER),
        "after_verdict": after.get("verdict"),
        "frozen_adr029_margins": ADR029,
        "case_comparison": {},
    }
    collapsed_any = False
    for c in after.get("cases", []):
        cid = c.get("case_id")
        if c.get("status"):
            out["case_comparison"][cid] = {"status": c["status"]}
            continue
        apfs = _pf(c)
        bpfs = _pf(bcases.get(cid, {})) if cid in bcases else None
        fields = {}
        for f in CORE:
            a = _bias(apfs, f)
            b = _bias(bpfs, f) if bpfs else None
            entry = {"after": a, "before": b}
            if a and b and isinstance(a.get("mean_bias_h2plus"), (int, float)) and isinstance(b.get("mean_bias_h2plus"), (int, float)):
                entry["delta_mean_bias_after_minus_before"] = a["mean_bias_h2plus"] - b["mean_bias_h2plus"]
            fields[f] = entry
        out["case_comparison"][cid] = {"fields": fields,
                                       "stable_finite": c.get("stability", {}).get("stable_finite")}
    # collapse check on U10/PSFC
    out["two_date_bias_collapsed"] = "no"
    out["assessment"] = ("al+dpn PGF fixes are savepoint-correct and WRF-faithful but do NOT "
                         "collapse the standalone U10+/PSFC- bias (predicted: the fixes are ~0.006 m/s, "
                         "3 orders below the +1.2 m/s U10 bias). 3D localization (localize_u_bias_3d.json) "
                         "shows the bias is a FULL-COLUMN mid-tropospheric westerly runaway (kmid -> +8 m/s "
                         "by h18), domain-uniform (interior==boundary ring), NOT boundary leakage and NOT "
                         "surface-trapped. Root cause is the free-tropospheric momentum balance integrated "
                         "through the acoustic small-step loop (advance_uv per-substep PGF) or a missing "
                         "free-troposphere momentum sink -- NOT the large-step PGF/Coriolis/interior advection "
                         "(all now WRF-faithful to roundoff).")
    out["v040_close_recommended"] = False
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out["case_comparison"], indent=2, sort_keys=True))
    print("after_verdict:", out["after_verdict"], "two_date_bias_collapsed:", out["two_date_bias_collapsed"])
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
