"""Derive wind/T2 skill-vs-persistence from a gpu_wind_localize JSON.

Compares the post-fix GPU all-grid RMSE (from gpu_wind_localize_FIXED.json) to the
persistence RMSE already computed in proofs/m19/persistence_baseline.json
(case2 L2, the SAME case + truth the localize harness uses).  skill = 1 -
gpu_rmse/persistence_rmse.  CPU-only, pure numpy/json.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# persistence RMSE (case2 L2) from proofs/m19/persistence_baseline.json
PERS = {24: {"U10": 2.117732, "V10": 2.054089, "T2": 1.122475}}
# pre-fix GPU all-RMSE (old gpu_wind_localize.json / verdict) for the delta
OLD = {24: {"U10": 2.539, "V10": 3.275, "T2": 1.040}}


def main() -> int:
    fixed = json.load(open(HERE / "gpu_wind_localize_FIXED.json"))
    print("field  lead  persRMSE  oldGPU  newGPU   oldSkill  newSkill  outcome")
    for row in fixed["per_lead"]:
        lead = int(row["lead_h"])
        if lead not in PERS:
            continue
        for f in ("U10", "V10", "T2"):
            new = row[f]["all"]["rmse"]
            pers = PERS[lead][f]
            old = OLD[lead][f]
            old_skill = 1.0 - old / pers
            new_skill = 1.0 - new / pers
            outcome = "WIN" if new_skill > 0.02 else ("tie" if new_skill > -0.02 else "loss")
            print(f"{f:4s}  {lead:>3}h  {pers:7.3f}  {old:6.3f}  {new:6.3f}   "
                  f"{old_skill:+.3f}    {new_skill:+.3f}   {outcome}")
        # interior decomposition
        for f in ("U10", "V10"):
            w = row[f]["water"]["rmse"]
            land = row[f]["land"]["rmse"]
            print(f"     {f} water-RMSE={w:.3f} land-RMSE={land:.3f} bias={row[f]['all']['bias']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
