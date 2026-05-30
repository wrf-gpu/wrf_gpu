#!/usr/bin/env python3
"""Mini-ensemble (5-case) selection for the anti-overfit gate before M19 is durable.

Reads case_manifest.json + station_join_manifest.json (built by build_corpus.py) and
records a regime-diverse 5-case selection with explicit rationale. Honest about the fact
that the currently-scoreable set is small (output-purged corpus) and entirely MAM season.

Run after build_corpus.py:
  taskset -c 0-3 python3 proofs/m20/build_mini_ensemble.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path("/mnt/data/wrf_gpu2/corpus")
PROOF = Path(__file__).resolve().parent

# Domain-mean diagnostics measured from the surviving full-output wrfout (d02 3km),
# computed by inspecting U10/V10/T2/RAINNC over the forecast (see report).
DIAG = {
    "20260509_18z_l2_72h_20260511T190519Z": {"mean_ws_ms": 3.36, "t2_mean_c": 19.1, "acc_precip_mm": 0.31},
    "20260528_18z_l2_72h_20260529T002423Z": {"mean_ws_ms": 8.26, "t2_mean_c": 19.8, "acc_precip_mm": 0.09},
    "20260521_18z_l3_24h_20260522T133443Z": {"mean_ws_ms": 6.48, "t2_mean_c": 20.0, "acc_precip_mm": 0.01},
    "20260528_18z_l3_24h_20260529T002423Z": {"mean_ws_ms": 8.00, "t2_mean_c": 19.9, "acc_precip_mm": 0.01},
    "20260509_18z_l3_24h_20260511T190519Z": {"mean_ws_ms": 3.22, "t2_mean_c": 19.2, "acc_precip_mm": 0.19},
}


def main() -> None:
    cases = json.loads((CORPUS / "case_manifest.json").read_text())["cases"]
    cov = {c["case_id"]: c for c in json.loads((CORPUS / "station_join_manifest.json").read_text())["case_coverage"]}

    scoreable_now = [
        c for c in cases
        if c["has_full_output"] and cov.get(c["case_id"], {}).get("obs_hours_in_window", 0) > 0
    ]

    # The mini-ensemble must be 5 regime-diverse cases. Today only 4 case-dates are
    # scoreable-now (20260509-L2, 20260521-L3, 20260528-L2, 20260528-L3); the 5th slot is
    # a target backfill date chosen for regime contrast (mid-window moderate regime).
    selection = [
        {
            "case_id": "20260528_18z_l2_72h_20260529T002423Z",
            "level": "L2", "regime": "strong trade-wind, dry",
            "mean_ws_ms": 8.26, "status": "scoreable_now",
            "rationale": "Strongest sustained 10m wind in the surviving set; 72h lead; "
                         "tests momentum/PBL skill at the windy end.",
        },
        {
            "case_id": "20260521_18z_l3_24h_20260522T133443Z",
            "level": "L3", "regime": "moderate trade-wind, 1km nests",
            "mean_ws_ms": 6.48, "status": "scoreable_now",
            "rationale": "Moderate regime AND the only fully-obs-covered L3 (1km) case; this is the "
                         "exact CPU run used to set the ADR-029 TOST benchmark, so it anchors the "
                         "margin derivation. 25/25 obs hours.",
        },
        {
            "case_id": "20260509_18z_l2_72h_20260511T190519Z",
            "level": "L2", "regime": "weak-wind, light precip",
            "mean_ws_ms": 3.36, "status": "scoreable_now",
            "rationale": "Weakest-wind, wettest case in the surviving set (0.31mm acc); tests the "
                         "calm/convective end. Obs only cover its 48-72h block (hourly obs begin "
                         "2026-05-11 09z), so score the 48-72h lead window only.",
        },
        {
            "case_id": "20260528_18z_l3_24h_20260529T002423Z",
            "level": "L3", "regime": "strong trade-wind, 1km nests",
            "mean_ws_ms": 8.00, "status": "scoreable_now",
            "rationale": "Same synoptic day as the strong-wind L2 case but at 1km; cross-checks "
                         "nest-resolution sensitivity of the equivalence claim at high wind.",
        },
        {
            "case_id": "TARGET_BACKFILL:20260516_18z_72h",
            "level": "L2/L3", "regime": "mid-window transitional (to be measured post-backfill)",
            "mean_ws_ms": None, "status": "needs_backfill",
            "rationale": "5th slot reserved for a mid-window date (2026-05-16) whose AIFS forcing is "
                         "preserved in forcing_cases/ but whose wrfout was purged. Re-run via "
                         "WPS->real->wrf to fill a regime/temporal gap between the 05-09 calm and "
                         "05-21/05-28 windy cases. Selected for temporal spacing + independence from "
                         "the 4 retained dates.",
        },
    ]

    payload = {
        "schema": "M20MiniEnsembleSelection",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Anti-overfit gate: before M19 single-case skill recovery is declared durable, "
                   "the GPU must recover comparable skill on 5 regime-diverse cases (not just the "
                   "one tuned case). Guards against overfitting the dycore/physics to a single day.",
        "selection_criteria": [
            "maximize spread in domain-mean 10m wind speed (calm 3.4 -> strong 8.3 m/s)",
            "include both L2 (3km final) and L3 (1km final) domain families",
            "include both 24h and 72h forecast lengths",
            "include the ADR-029 benchmark-anchor case (20260521 L3)",
            "temporal spacing across the obs window to reduce synoptic correlation",
            "every selected case must have hourly-obs overlap for at least one lead block",
        ],
        "scoreable_now_count": len(scoreable_now),
        "scoreable_now_case_ids": [c["case_id"] for c in scoreable_now],
        "selection": selection,
        "honesty_notes": [
            "ALL candidate cases are MAM (spring) 2026 - the mini-ensemble proves regime diversity "
            "in WIND/PRECIP but NOT seasonal diversity.",
            "4 of 5 slots are scoreable today; the 5th requires a ~1.8GB WRF re-run from preserved "
            "AIFS forcing (no external data fetch needed).",
            "The 20260509 case can only be scored on its 48-72h block because hourly obs start "
            "2026-05-11 09z. Its 0-48h leads have no hourly obs.",
            "This mini-ensemble is the M19-durability gate, NOT the M20/M21 TOST corpus (which needs "
            ">=15, target >=30 cases).",
        ],
    }
    for base in (CORPUS, PROOF):
        (base / "mini_ensemble_selection.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("wrote mini_ensemble_selection.json")
    print("scoreable now:", len(scoreable_now), [c["case_id"] for c in scoreable_now])


if __name__ == "__main__":
    main()
