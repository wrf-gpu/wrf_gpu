#!/usr/bin/env python3
"""
Run ONE v0.12.0 d02 24h GPU forecast for the powered-TOST campaign.

Uses run_forecast_operational_segmented to avoid the donate-aliasing crash.
Adapted from proofs/v0110/run_one_case_dealiased.py — root path fixed to
resolve from __file__ so it always uses THIS worktree's src/gpuwrf.

Usage (called by run_powered_tost_n15_v0120.py via GPU lock wrapper):
    /tmp/wrf_gpu_run_lowprio.sh taskset -c 0-3 \\
        env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \\
        --run-root /tmp/v0120_merged_run_root \\
        --run-id <RUN_ID> \\
        --hours 24 \\
        --output-root /tmp/v0120_powered_tost_runs \\
        --proof-dir /path/to/proof/subdir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Root relative to __file__ → .../worktrees/v0120-tostprep
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for _p in [str(SRC), str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented  # noqa: E402
from gpuwrf.profiling.transfer_audit import block_until_ready                   # noqa: E402
from gpuwrf.integration.daily_pipeline import (                                  # noqa: E402
    DailyPipelineConfig,
    execute_daily_pipeline,
    resolve_run_dir,
    write_json,
)
from scripts.m7_l2_d02_replay import (                                           # noqa: E402
    build_l2_daily_case,
    _pin_orchestration_cpus,
    _write_blocked_proofs,
    _write_json,
    write_tier4_rmse,
    write_bounds_check,
    write_wall_clock,
    L2_RUN_ROOT,
    OUTPUT_ROOT,
    RMSE_THRESHOLDS,
)


def _segmented_forecast_fn(state, namelist, hours: float):
    """Segmented forecast — no outer donate_argnums, avoids aliasing crash."""
    result = run_forecast_operational_segmented(state, namelist, float(hours))
    block_until_ready(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",       type=Path, default=L2_RUN_ROOT)
    parser.add_argument("--cpu-truth-root", type=Path,
                        default=Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output"),
                        help="Root for CPU-WRF reference wrfouts")
    parser.add_argument("--run-id",    required=True)
    parser.add_argument("--hours",     type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--proof-dir", type=Path)
    args = parser.parse_args(argv)

    affinity  = _pin_orchestration_cpus()
    proof_dir = args.proof_dir or (ROOT / ".agent/sprints/powered_tost_n15_v0120" / args.run_id)
    proof_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_root / f"l2_d02_{args.run_id}"

    config = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=proof_dir,
        run_root=Path(args.run_root),
        score=False,
        domain="d02",
    )

    run_dir = resolve_run_dir(args.run_id, args.run_root)
    cpu_truth_dir = args.cpu_truth_root / args.run_id
    if not cpu_truth_dir.is_dir():
        cpu_truth_dir = run_dir  # fallback

    pipeline_payload = execute_daily_pipeline(
        config,
        case_builder=build_l2_daily_case,
        forecast_fn=_segmented_forecast_fn,
    )
    if affinity is not None:
        pipeline_payload["orchestration_cpu_affinity"] = affinity
        write_json(proof_dir / "pipeline_run_l2_d02.json", pipeline_payload)

    blocked_reason: str | None = None
    try:
        wrfout_files = [Path(p) for p in pipeline_payload.get("wrfout_files", [])]
        if pipeline_payload.get("verdict") == "PIPELINE_BLOCKED" or not wrfout_files:
            reason = pipeline_payload.get("reason", "pipeline did not produce wrfouts")
            blocked_reason = str(reason)
            _write_blocked_proofs(proof_dir, reason=str(reason), detail=pipeline_payload)
            verdict = "L2_D02_BLOCKED"
            rmse = bounds = wall = {"status": "BLOCKED"}
        else:
            final_wrfout = wrfout_files[-1]
            rmse   = write_tier4_rmse(
                final_wrfout=final_wrfout,
                reference_run_dir=cpu_truth_dir,
                proof_path=proof_dir / "tier4_rmse_l2_d02.json",
            )
            bounds = write_bounds_check(
                wrfout_files=wrfout_files,
                proof_path=proof_dir / "bounds_check_l2_d02.json",
            )
            wall   = write_wall_clock(
                pipeline_payload=pipeline_payload,
                proof_path=proof_dir / "wall_clock_l2_d02.json",
                run_dir=run_dir,
                affinity=affinity,
            )
            verdict = (
                "L2_D02_GREEN"
                if rmse["status"] == "PASS" and bounds["status"] == "PASS"
                else "L2_D02_BOUNDED_FAIL"
            )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        blocked_reason = reason
        detail = {"pipeline_payload": dict(pipeline_payload), "run_dir": str(run_dir)}
        _write_blocked_proofs(proof_dir, reason=reason, detail=detail)
        verdict = "L2_D02_BLOCKED"
        rmse = bounds = wall = {"status": "BLOCKED", "reason": reason}

    summary = {
        "schema": "M7L2D02ReplayValidationSummary",
        "schema_version": 1,
        "verdict": verdict,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        # Surface the upstream blocked reason so the orchestrator / manager can
        # diagnose a GPU-forecast failure WITHOUT having to open the nested
        # pipeline_run JSON. None on a successful (GREEN / BOUNDED_FAIL) case.
        "blocked_reason": blocked_reason,
        "proofs": {
            "tier4_rmse":  str(proof_dir / "tier4_rmse_l2_d02.json"),
            "bounds":      str(proof_dir / "bounds_check_l2_d02.json"),
            "wall_clock":  str(proof_dir / "wall_clock_l2_d02.json"),
        },
        "statuses": {
            "pipeline":  pipeline_payload.get("verdict"),
            "rmse":      rmse.get("status"),
            "bounds":    bounds.get("status"),
            "wall_clock": wall.get("status"),
        },
    }
    _write_json(proof_dir / "l2_d02_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verdict in ("L2_D02_GREEN", "L2_D02_BOUNDED_FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
