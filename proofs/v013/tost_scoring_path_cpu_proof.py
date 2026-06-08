#!/usr/bin/env python3
"""
v0.13 — CPU proof that the powered-TOST SCORING path runs rc=0 end-to-end.

This exercises the EXACT scoring functions used by the powered-n=15 campaign
(``run_powered_tost_n15_v0120.score_one_case`` -> ``paired_score`` +
``score_cell_level`` + ``aggregate_tost``) on RETAINED REAL GPU wrfout data,
on CPU only (``JAX_PLATFORMS=cpu``; no GPU context — the scoring path is pure
numpy / netCDF4 / pandas, it never touches JAX or a GPU).

It is NOT an equivalence claim: the retained GPU set and the CPU-WRF truth here
come from DIFFERENT initialisations (GPU init 2026-05-29 18z, CPU init
2026-05-30 18z), so the deltas are genuine and non-zero. The sole purpose is to
prove the scoring CODE PATH executes rc=0 on real GPU data — closing the v0.12.0
rc=2 ``L2_D02_BLOCKED`` defect for the scoring lane.

Usage:
    env JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 \\
        python proofs/v013/tost_scoring_path_cpu_proof.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the campaign's OWN scoring functions so the proof binds the real code.
sys.path.insert(0, str(ROOT / "proofs/v0120/powered_tost_n15"))
import run_powered_tost_n15_v0120 as camp  # noqa: E402

GPU_DIR = Path("/tmp/wrf_gpu2_v090_for_v0100_validate/proofs/m20/tost_run/gpu_wrfout/case1_L2")
CPU_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260530_18z_l2_72h_20260531T161057Z")
OUT_JSON = ROOT / "proofs/v013/tost_scoring_path_cpu_proof.json"

# init_time anchored to the CPU-truth init so lead-block bucketing is meaningful.
INIT_TIME = datetime(2026, 5, 30, 18, 0, 0, tzinfo=timezone.utc)


def _json_default(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return str(v)


def main() -> int:
    if not GPU_DIR.is_dir() or not CPU_DIR.is_dir():
        print(f"[ABORT] missing inputs: gpu={GPU_DIR.is_dir()} cpu={CPU_DIR.is_dir()}")
        return 2

    run_id = "v013_scoring_path_proof_realgpu_vs_cpu"

    # validate_gpu_output exercises finite checks + file count over the real GPU set.
    validation = camp.validate_gpu_output(GPU_DIR, run_id)
    print(f"[validate] n_files={validation['n_files']} "
          f"nonfinite={len(validation['nonfinite_hour_files'])}")

    # The exact per-case scorer the campaign calls (TOST pairs + cell-level grid).
    case_result = camp.score_one_case(run_id, GPU_DIR, CPU_DIR, INIT_TIME, validation)

    cell = case_result["cell_level"]
    print(f"[cell] common lead hours={cell['n_common_lead_hours']}")
    deltas_nonzero = False
    for f in camp.FIELDS:
        fs = cell["field_stats"][f]
        n = fs.get("n_cells", 0)
        rmse = fs.get("rmse")
        bias = fs.get("bias")
        print(f"  {f}: n_cells={n} rmse={rmse} bias={bias}")
        if n and rmse is not None and rmse > 0.0:
            deltas_nonzero = True

    # aggregate_tost over a (synthetic) 2-element list of the same real pairs:
    # this proves the cross-case aggregator runs rc=0 (the n>=2 lane). Labelled
    # clearly as a code-path exercise, NOT an equivalence result.
    tost_pairs = case_result["tost_pairs"]
    try:
        agg = camp.aggregate_tost([tost_pairs, tost_pairs])
        agg_ok = "tost" in agg
    except Exception as exc:  # pragma: no cover
        agg = {"error": f"{type(exc).__name__}: {exc}"}
        agg_ok = False
    print(f"[aggregate] aggregate_tost ran rc=0: {agg_ok}")

    n_common = cell["n_common_lead_hours"]
    total_cells = sum(cell["field_stats"][f].get("n_cells", 0) for f in camp.FIELDS)
    finite_ok = all(
        math.isfinite(cell["field_stats"][f]["rmse"])
        for f in camp.FIELDS
        if cell["field_stats"][f].get("n_cells", 0)
    )

    rc0 = bool(n_common > 0 and total_cells > 0 and finite_ok and agg_ok)
    verdict = "SCORING_PATH_RC0_PROVEN" if rc0 else "SCORING_PATH_FAILED"

    payload = {
        "schema": "V013TOSTScoringPathCPUProof",
        "schema_version": 1,
        "verdict": verdict,
        "purpose": (
            "Prove the powered-TOST scoring code path (score_one_case -> "
            "paired_score + score_cell_level + aggregate_tost) runs rc=0 on real "
            "retained GPU wrfout data, on CPU only. NOT an equivalence claim: GPU "
            "init 2026-05-29 18z vs CPU-WRF init 2026-05-30 18z (different IC) -> "
            "deltas are genuine/non-zero by construction."
        ),
        "platform": os.environ.get("JAX_PLATFORMS", "<unset>"),
        "gpu_dir": str(GPU_DIR),
        "cpu_dir": str(CPU_DIR),
        "init_time_utc": INIT_TIME.isoformat(),
        "n_common_lead_hours": n_common,
        "total_cells_scored": total_cells,
        "deltas_nonzero": deltas_nonzero,
        "all_rmse_finite": finite_ok,
        "aggregate_tost_ran": agg_ok,
        "gpu_validation": validation,
        "cell_level_field_stats": {
            f: {
                "n_cells": cell["field_stats"][f].get("n_cells"),
                "rmse": cell["field_stats"][f].get("rmse"),
                "bias": cell["field_stats"][f].get("bias"),
                "mae": cell["field_stats"][f].get("mae"),
                "pearson_r": cell["field_stats"][f].get("pearson_r"),
            }
            for f in camp.FIELDS
        },
        "tost_pairs_total_complete_pairs": tost_pairs.get("total_complete_pairs"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"\n[{verdict}] wrote {OUT_JSON}")
    return 0 if rc0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
