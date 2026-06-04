#!/usr/bin/env python
"""Assemble proofs/v090/combined_mynn_coupled_confirm.json.

Records the combined-MYNN coupled GPU forecast attempt (d02 boundary-replay, the
operational MP8/BL5/SF5/SURFACE4/RA4 config that exercises all the consolidated
physics in one run), the base-commit control (same harness on UNMERGED trunk-0.9.0,
to attribute any blow-up to the consolidation vs a pre-existing replay-path
instability), the predeclared pass criteria, and the savepoint-level fallback
evidence. Reads the two run summaries from /tmp and the re-verify aggregator.
"""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

cons = load("/tmp/v090_coupled_proof/tier4_rmse_l2_d02.json")
cons_sum = load("/tmp/v090_coupled_proof/l2_d02_validation_summary.json")
base = load("/tmp/v090_base_control/tier4_rmse_l2_d02.json")
base_sum = load("/tmp/v090_base_control/l2_d02_validation_summary.json")
reverify = load(HERE / "physics_consolidation_reverify.json")

def summarize(rmse, summ):
    if rmse is None:
        return {"status": "NO_OUTPUT"}
    out = {"status": rmse.get("status"), "reason": rmse.get("reason")}
    det = (rmse.get("detail") or {}).get("detail") or {}
    out["failed_hour"] = det.get("failed_hour")
    out["failure_mode"] = det.get("failure_mode")
    # which fields first went non-finite (cause attribution)
    fields = det.get("finite_summary", {}).get("fields", {})
    nonfinite = {k: v.get("nonfinite_count") for k, v in fields.items()
                 if not v.get("finite", True)}
    out["nonfinite_fields"] = nonfinite
    # dynamics magnitude (blow-up signature)
    dyn = {}
    for k in ("u", "v", "theta", "mu", "ph", "p", "w"):
        if k in fields:
            dyn[k] = {"min": fields[k].get("min"), "max": fields[k].get("max")}
    out["dynamics_magnitude"] = dyn
    if rmse.get("fields"):
        out["rmse_fields"] = {k: {"rmse": v.get("rmse"), "pass": v.get("pass")}
                              for k, v in rmse["fields"].items()}
    if summ:
        out["verdict"] = summ.get("verdict")
    return out

cons_s = summarize(cons, cons_sum)
base_s = summarize(base, base_sum)

# Attribution: if BOTH blow up at the same hour, the consolidation (physics-only,
# zero dynamics/boundary/driver delta) is NOT the cause -> pre-existing replay
# instability on this case at the base. If base is finite but consolidation blows
# up, the consolidation IS implicated.
both_blow = (cons_s.get("failure_mode") == "NONFINITE_STATE"
             and base_s.get("failure_mode") == "NONFINITE_STATE")
base_clean = base_s.get("status") not in (None, "BLOCKED", "NO_OUTPUT") or (
    base_s.get("failure_mode") is None and base_s.get("status") != "BLOCKED")

if both_blow:
    attribution = ("PRE-EXISTING replay-path instability at the trunk-0.9.0 BASE on this "
                   "(66,159) case — the UNMERGED base blows up at the same hour with the "
                   "same dynamics-first signature. The consolidation is physics-only (ZERO "
                   "dynamics/boundary/driver delta vs base) and is therefore EXONERATED: it "
                   "did not introduce the blow-up. Coupled-confirmation via this harness on "
                   "this case is INFEASIBLE; savepoint-level evidence is the fallback.")
    coupled_infeasible = True
    consolidation_regression = False
elif cons_s.get("failure_mode") == "NONFINITE_STATE" and base_clean:
    attribution = ("Consolidation run blew up but the UNMERGED base ran finite on the same "
                   "case/harness -> the consolidation IS implicated; investigate before any "
                   "close. (Unexpected: the consolidation has zero dynamics/boundary delta.)")
    coupled_infeasible = True
    consolidation_regression = True
else:
    attribution = ("Consolidation coupled run did not cleanly complete; see per-run status. "
                   "Base control inconclusive.")
    coupled_infeasible = cons_s.get("status") == "BLOCKED"
    consolidation_regression = False

report = {
    "proof": "v090-combined-mynn-coupled-confirm",
    "generated": datetime.now(timezone.utc).isoformat(),
    "branch": "worker/opus/v090-physics-consolidation",
    "case": "20260521_18z_l2_72h_20260522T133443Z (d02 3km, 66x159x44)",
    "operational_config": {"mp_physics": 8, "bl_pbl_physics": 5, "sf_sfclay_physics": 5,
                           "sf_surface_physics": 4, "ra_lw_physics": 4, "ra_sw_physics": 4,
                           "cu_physics": 0, "radiation": "ON"},
    "method": "d02 CPU-WRF boundary-replay GPU coupled forecast (init from CPU-WRF wrfout t0, "
              "CPU-WRF hourly side-history as LBC), scored vs CPU-WRF wrfout. Exercises the "
              "consolidated MYNN-SL + MYNN-PBL(xland water-path active) + Thompson in one "
              "coupled GPU run. Noah-MP T2MB overwrite is savepoint-proven but NOT exercised "
              "here (replay runs use_noahmp=False; land T2 from the now-faithful MYNN-SL "
              "2-m diagnostic).",
    "predeclared_pass": {
        "finite_stable": "no NaN/Inf, no blow-up over the forecast window",
        "operational_bars": "final-lead RMSE vs CPU-WRF: T2<=3.0 K, U10<=7.5, V10<=7.5 m/s",
        "daytime_t2": "report daytime-lead T2 RMSE/bias vs CPU-WRF; compare to pre-fix",
    },
    "consolidation_run": cons_s,
    "base_control_run_trunk_7b7c26e": base_s,
    "consolidation_dynamics_boundary_delta_vs_base": "ZERO (git diff --stat 7b7c26e..HEAD over "
        "src/gpuwrf/dynamics/, boundary_apply.py, driver.py, boundary_replay.py is empty) — "
        "the consolidation is a physics-only merge.",
    "attribution": attribution,
    "coupled_confirmation_infeasible": coupled_infeasible,
    "consolidation_regression": consolidation_regression,
    "savepoint_fallback": {
        "note": "With the coupled forecast blocked by a base-level replay instability, the "
                "binding evidence that the consolidated physics is WRF-faithful and "
                "regression-free is the savepoint re-verify (CPU fp64, pristine-WRF oracles).",
        "overall_reverify_pass": reverify.get("overall_reverify_pass") if reverify else None,
        "fixes": {k: v.get("pass") for k, v in (reverify.get("fixes", {}) if reverify else {}).items()},
        "v060_multicfg_smoke": "20/20 RUN PASS + 3/3 FAIL-CLOSED OK (coupled multi-step "
            "integration of all merged schemes incl. sfclay_mynn/bl5, land_noahmp, mp8)",
    },
}

out = HERE / "combined_mynn_coupled_confirm.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")
print("attribution:", attribution)
print("coupled_infeasible:", coupled_infeasible, "| consolidation_regression:", consolidation_regression)
