#!/usr/bin/env python
"""Build the Round-4 s_aw floor before/after forecast-gate report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BEFORE = HERE / "forecast_gate_24h_report.json"
AFTER = HERE / "forecast_gate_postfix6_raw.json"
SAVEPOINT = HERE / "r4_saw_floor_savepoint_parity.json"
OUT = HERE / "forecast_gate_postfix6_report.json"

ADR029 = {
    "T2": 0.2148692978020805,
    "U10": 0.23064713972582307,
    "V10": 0.2752320537920854,
    "PSFC_diag_margin": 50.0,
}
FIELDS = ("T2", "U10", "V10", "PSFC")


def _cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c.get("case_id")): c for c in report.get("cases", [])}


def _per_fields(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not rec or rec.get("status"):
        return None
    return rec.get("per_field_summary") or rec.get("fields")


def _bias(pfs: dict[str, Any] | None, field: str) -> dict[str, Any] | None:
    if not pfs or field not in pfs:
        return None
    row = pfs[field]
    if "after" in row:
        row = row["after"]
    mean_bias = row.get("mean_bias_h2plus", row.get("mean_bias"))
    worst_abs = row.get("worst_abs_bias_h2plus", row.get("worst_abs_bias"))
    if mean_bias is None or worst_abs is None:
        return None
    return {
        "mean_bias_h2plus": float(mean_bias),
        "worst_abs_bias_h2plus": float(worst_abs),
        "within_margin": row.get("within_margin"),
        "regression_margin": row.get("regression_margin"),
    }


def _delta(after: dict[str, Any] | None, before: dict[str, Any] | None) -> dict[str, Any]:
    out = {"after": after, "before": before}
    if after and before:
        out["delta_mean_bias_after_minus_before"] = (
            after["mean_bias_h2plus"] - before["mean_bias_h2plus"]
        )
        out["delta_worst_abs_after_minus_before"] = (
            after["worst_abs_bias_h2plus"] - before["worst_abs_bias_h2plus"]
        )
    return out


def _field_within(field: str, row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    margin = ADR029["PSFC_diag_margin"] if field == "PSFC" else ADR029[field]
    return abs(row["mean_bias_h2plus"]) <= margin and row["worst_abs_bias_h2plus"] <= margin


def _collapse(case_comparison: dict[str, Any]) -> tuple[str, str]:
    scored_cases = [case for case in case_comparison.values() if "fields" in case]
    if not scored_cases:
        return "no", "No scored after rows were available."

    all_fields_within = True
    any_clear_reduction = False
    any_clear_worse = False
    for case in case_comparison.values():
        fields = case.get("fields") or {}
        for field in FIELDS:
            row = fields.get(field) or {}
            after = row.get("after")
            before = row.get("before")
            all_fields_within = all_fields_within and _field_within(field, after)
            if after and before:
                before_abs = abs(before["mean_bias_h2plus"])
                after_abs = abs(after["mean_bias_h2plus"])
                if before_abs > 0:
                    reduction = (before_abs - after_abs) / before_abs
                    any_clear_reduction = any_clear_reduction or reduction >= 0.25
                    any_clear_worse = any_clear_worse or reduction <= -0.25
    if all_fields_within:
        return "yes", "All two-date h2+ mean and worst biases are within ADR-029/PSFC diagnostic margins."
    if any_clear_reduction and not any_clear_worse:
        return "partial", "At least one h2+ mean bias dropped by >=25%, but the two-date gate did not collapse to margins."
    return "no", "The two-date h2+ wind/pressure biases did not collapse toward ADR-029 margins."


def main() -> int:
    before = json.loads(BEFORE.read_text(encoding="utf-8")) if BEFORE.is_file() else {}
    after = json.loads(AFTER.read_text(encoding="utf-8"))
    savepoint = json.loads(SAVEPOINT.read_text(encoding="utf-8")) if SAVEPOINT.is_file() else {}

    before_cases = _cases(before)
    case_comparison: dict[str, Any] = {}
    for case in after.get("cases", []):
        cid = str(case.get("case_id"))
        if case.get("status"):
            case_comparison[cid] = {"status": case.get("status"), "error": case.get("error")}
            continue
        apfs = _per_fields(case)
        bpfs = _per_fields(before_cases.get(cid))
        fields = {
            field: _delta(_bias(apfs, field), _bias(bpfs, field))
            for field in FIELDS
        }
        case_comparison[cid] = {
            "stable_finite": (case.get("stability") or {}).get("stable_finite"),
            "physical_range_ok": (case.get("stability") or {}).get("physical_range_ok"),
            "core_within_margin": case.get("core_within_margin"),
            "fields": fields,
        }

    collapsed, collapse_basis = _collapse(case_comparison)
    report = {
        "schema": "v0.4.0-round4-saw-floor-before-after-2026-06-03",
        "created_by": "GPT-5.5 xhigh",
        "question": (
            "Does adding WRF's kmdz=max(kmdz,0.5*s_aw) momentum eddy-diffusivity "
            "floor collapse the two-date standalone forecast wind/pressure bias?"
        ),
        "before_report": str(BEFORE),
        "after_report": str(AFTER),
        "savepoint_parity_report": str(SAVEPOINT),
        "after_verdict": after.get("verdict"),
        "s_aw_floor_omission": "real",
        "fix_applied": True,
        "wrf_faithful_scope": (
            "Momentum mass-flux source terms s_awu/s_awv remain disabled for "
            "bl_mynn_edmf_mom=0; WRF still applies the s_aw stability floor to "
            "kmdz before the U/V diffusion solve."
        ),
        "savepoint_verdict": savepoint.get("verdict"),
        "frozen_adr029_margins": ADR029,
        "case_comparison": case_comparison,
        "two_date_bias_collapsed": collapsed,
        "collapse_basis": collapse_basis,
        "v040_close_recommended": collapsed in {"yes", "partial"},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "after_verdict": report["after_verdict"],
        "two_date_bias_collapsed": collapsed,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
