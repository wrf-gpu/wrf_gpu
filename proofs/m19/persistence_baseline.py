"""M19 PERSISTENCE BASELINE — does the GPU forecast have real skill, or does it
merely score as well as a trivial "hold t=0" persistence forecast?

WHY (closes .agent/reviews/2026-05-30-agy-verdict-crosscheck.md §A.2)
--------------------------------------------------------------------
The 3-case verdict uses 18z inits scored at 24/48/72h leads — all valid at 18z,
the SAME diurnal phase. In the steady Canary trade-wind regime a "persistence"
forecast (hold the t=0 init state constant) might ALSO score ~1 K T2 / ~2-3 m/s
wind. If so, the verdict PASS would be partly hollow. This script PROVES whether
the GPU forecast BEATS persistence.

METHOD (identical metric to verdict_3case.score_fields)
-------------------------------------------------------
For each verdict case / level / scored lead:
  * PERSISTENCE forecast = the CPU-WRF t=0 (init) wrfout_d02 T2/U10/V10 field,
    held constant (no time integration).
  * Truth = the CPU-WRF wrfout_d02 at the lead valid-time — the SAME truth the
    GPU was scored against in verdict_result.json.
  * persistence RMSE = sqrt(mean((init - truth)**2)) over the full d02 grid
    (pure numpy, float64, identical to verdict_3case.score_fields).
  * GPU RMSE is read from proofs/m19/verdict_result.json.
  * skill score = 1 - (GPU_RMSE / persistence_RMSE):
      > 0  GPU BEATS persistence (real dynamical skill)
      ~ 0  GPU ties persistence (skill is persistence-level — honest caveat)
      < 0  GPU is WORSE than persistence

CPU-ONLY. numpy + netCDF reads only. NO GPU, NO forecast runs, NO JAX.

USAGE
-----
  JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/m19/persistence_baseline.py \
      --out proofs/m19/persistence_baseline.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "cases_manifest.json"
VERDICT = HERE / "verdict_result.json"
FIELDS = ("T2", "U10", "V10")


def rmse(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Gridded whole-domain RMSE, float64 — identical to verdict_3case.score_fields."""
    f = np.asarray(forecast, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((f - t) ** 2)))


def lead_truth_path(level_spec: dict, lead_h: int) -> Path | None:
    """Resolve the truth wrfout path for a lead, preferring the manifest mapping
    (which is what the verdict actually scored against) and falling back to the
    deterministic init+lead filename."""
    truth = level_spec.get("wrfout_d02_truth", {})
    key = str(lead_h)
    if key in truth:
        return Path(truth[key])
    # Fall back to the deterministic filename (init + lead).
    init = Path(level_spec["gpu_init_source"])
    stamp = init.name.split(f"wrfout_{level_spec['domain']}_", 1)[-1]
    t0 = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    valid = t0 + timedelta(hours=lead_h)
    return init.parent / f"wrfout_{level_spec['domain']}_{valid:%Y-%m-%d_%H:%M:%S}"


def gpu_rmse_for(verdict: dict, case_id: str, level: str, lead_h: int,
                 field: str) -> float | None:
    try:
        return float(
            verdict["results"][case_id]["levels"][level]["scores"][str(lead_h)][field]["rmse"]
        )
    except (KeyError, TypeError):
        return None


def scored_leads_for(verdict: dict, case_id: str, level: str) -> list[int]:
    try:
        return [int(x) for x in verdict["results"][case_id]["levels"][level]["scored_leads_h"]]
    except (KeyError, TypeError):
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="M19 persistence baseline vs GPU verdict")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--verdict", type=Path, default=VERDICT)
    ap.add_argument("--out", type=Path, default=HERE / "persistence_baseline.json")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    verdict = json.loads(Path(args.verdict).read_text())
    cases = {c["case_id"]: c for c in manifest["cases"]}

    per_entry: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    agg: dict[str, dict[str, Any]] = {
        f: {"skill_scores": [], "gpu_rmse": [], "pers_rmse": [],
            "gpu_wins": 0, "gpu_ties": 0, "gpu_losses": 0}
        for f in FIELDS
    }

    for case_id, case in cases.items():
        for level, level_spec in case["levels"].items():
            # Score ONLY the leads the GPU was actually scored at (the verdict's
            # real gate), so the comparison is apples-to-apples.
            leads = scored_leads_for(verdict, case_id, level)
            if not leads:
                continue
            init_path = Path(level_spec["gpu_init_source"])
            if not init_path.is_file():
                missing.append({"case": case_id, "level": level, "lead_h": "ALL",
                                "reason": f"init (t=0) wrfout absent: {init_path}"})
                continue
            init = read_wrfout_file(init_path, fields=FIELDS)["fields"]
            for lead_h in leads:
                truth_path = lead_truth_path(level_spec, lead_h)
                if truth_path is None or not truth_path.is_file():
                    missing.append({"case": case_id, "level": level, "lead_h": lead_h,
                                    "reason": f"truth wrfout absent: {truth_path}"})
                    continue
                truth = read_wrfout_file(truth_path, fields=FIELDS)["fields"]
                for field in FIELDS:
                    pers = rmse(init[field], truth[field])
                    gpu = gpu_rmse_for(verdict, case_id, level, lead_h, field)
                    skill = None
                    outcome = "no_gpu_rmse"
                    if gpu is not None and pers > 0.0:
                        skill = 1.0 - (gpu / pers)
                        # win if GPU is meaningfully better (>2% lower RMSE), tie
                        # within +/-2%, loss if meaningfully worse.
                        if skill > 0.02:
                            outcome = "gpu_wins"
                            agg[field]["gpu_wins"] += 1
                        elif skill < -0.02:
                            outcome = "gpu_loss"
                            agg[field]["gpu_losses"] += 1
                        else:
                            outcome = "tie"
                            agg[field]["gpu_ties"] += 1
                        agg[field]["skill_scores"].append(skill)
                        agg[field]["gpu_rmse"].append(gpu)
                        agg[field]["pers_rmse"].append(pers)
                    per_entry.append({
                        "case": case_id,
                        "init_utc": case["init_utc"],
                        "level": level,
                        "lead_h": lead_h,
                        "field": field,
                        "persistence_rmse": pers,
                        "gpu_rmse": gpu,
                        "skill_score": skill,
                        "outcome": outcome,
                        "init_file": str(init_path),
                        "truth_file": str(truth_path),
                    })

    summary: dict[str, Any] = {}
    for field in FIELDS:
        a = agg[field]
        n = len(a["skill_scores"])
        summary[field] = {
            "n_comparisons": n,
            "mean_skill_score": (float(np.mean(a["skill_scores"])) if n else None),
            "median_skill_score": (float(np.median(a["skill_scores"])) if n else None),
            "min_skill_score": (float(np.min(a["skill_scores"])) if n else None),
            "max_skill_score": (float(np.max(a["skill_scores"])) if n else None),
            "mean_gpu_rmse": (float(np.mean(a["gpu_rmse"])) if n else None),
            "mean_persistence_rmse": (float(np.mean(a["pers_rmse"])) if n else None),
            "gpu_wins": a["gpu_wins"],
            "gpu_ties": a["gpu_ties"],
            "gpu_losses": a["gpu_losses"],
        }

    def verdict_word(field: str) -> str:
        s = summary[field]
        if s["n_comparisons"] == 0:
            return "NO_DATA"
        if s["gpu_losses"] == 0 and s["gpu_wins"] > 0 and s["gpu_ties"] == 0:
            return "GPU_BEATS_PERSISTENCE_AT_ALL_LEADS"
        if s["gpu_losses"] == 0 and s["gpu_wins"] >= s["gpu_ties"]:
            return "GPU_BEATS_OR_TIES_PERSISTENCE"
        if s["gpu_losses"] > 0:
            return "GPU_LOSES_TO_PERSISTENCE_ON_SOME_LEADS"
        return "GPU_TIES_PERSISTENCE"

    field_verdicts = {f: verdict_word(f) for f in FIELDS}
    any_loss = any(summary[f]["gpu_losses"] > 0 for f in FIELDS)
    all_win = all(
        summary[f]["n_comparisons"] > 0
        and summary[f]["gpu_losses"] == 0
        and summary[f]["gpu_ties"] == 0
        for f in FIELDS
    )
    if all_win:
        overall = "GPU_BEATS_PERSISTENCE_ON_ALL_FIELDS_ALL_LEADS"
    elif not any_loss:
        overall = "GPU_BEATS_OR_TIES_PERSISTENCE_NO_LOSSES"
    else:
        overall = "GPU_LOSES_TO_PERSISTENCE_ON_SOME_FIELD_LEAD"

    payload = {
        "_doc": "M19 persistence baseline. Persistence = hold t=0 CPU-WRF "
                "wrfout_d02 field constant; scored vs the CPU-WRF wrfout_d02 at "
                "the lead valid time with the SAME gridded whole-domain RMSE as "
                "verdict_3case.score_fields. skill_score = 1 - GPU_RMSE/persistence_RMSE "
                "(>0 = GPU beats persistence). Closes "
                ".agent/reviews/2026-05-30-agy-verdict-crosscheck.md §A.2.",
        "metric": "rmse = sqrt(mean((forecast - truth)**2)) over full d02 grid (66x159), float64",
        "skill_score_def": "1 - (GPU_RMSE / persistence_RMSE); >0.02 GPU win, [-0.02,0.02] tie, <-0.02 GPU loss",
        "gpu_rmse_source": str(args.verdict),
        "scored_leads_note": "only leads the GPU was actually scored at in "
                             "verdict_result.json are compared (apples-to-apples).",
        "data_availability_note":
            "case1 (20260528 init) wrfout_d02 history was purged from the corpus "
            "after the 2026-05-30T13:42 verdict run (only static met_em/table "
            "symlinks remain on disk), so case1 persistence cannot be recomputed. "
            "case2 (L2 72h + L3 24h) and case3 (L3 24h) retain full d02 history "
            "and cover the 24/48/72h leads.",
        "per_entry": per_entry,
        "missing": missing,
        "summary_by_field": summary,
        "field_verdicts": field_verdicts,
        "overall_verdict": overall,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote persistence baseline -> {args.out}")
    print(f"overall: {overall}")
    print()
    hdr = f"{'case':6} {'lvl':3} {'lead':>4} {'fld':4} {'pers_RMSE':>10} {'gpu_RMSE':>10} {'skill':>7} {'outcome':>9}"
    print(hdr)
    print("-" * len(hdr))
    for e in per_entry:
        sk = "n/a" if e["skill_score"] is None else f"{e['skill_score']:+.3f}"
        g = "n/a" if e["gpu_rmse"] is None else f"{e['gpu_rmse']:.3f}"
        print(f"{e['case']:6} {e['level']:3} {e['lead_h']:>4} {e['field']:4} "
              f"{e['persistence_rmse']:>10.3f} {g:>10} {sk:>7} {e['outcome']:>9}")
    print()
    for f in FIELDS:
        s = summary[f]
        if s["n_comparisons"]:
            print(f"{f}: n={s['n_comparisons']} mean_skill={s['mean_skill_score']:+.3f} "
                  f"mean_gpu_RMSE={s['mean_gpu_rmse']:.3f} mean_pers_RMSE={s['mean_persistence_rmse']:.3f} "
                  f"(wins {s['gpu_wins']} / ties {s['gpu_ties']} / losses {s['gpu_losses']}) "
                  f"-> {field_verdicts[f]}")
        else:
            print(f"{f}: NO DATA")
    if missing:
        print()
        print("MISSING (could not compute persistence):")
        for m in missing:
            print(f"  {m['case']}/{m['level']} lead={m['lead_h']}: {m['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
