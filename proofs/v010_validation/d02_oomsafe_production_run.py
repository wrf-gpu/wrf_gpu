#!/usr/bin/env python
"""OOM-safe d02 production-wrfout run for the post-fix consolidation checkpoint.

The production ``run_forecast_operational`` single-scan compiles ONE jax program
per forecast hour; on the 66x159 (Grid B) d02 mass grid that program needs a
single 16 GiB XLA intermediate, which OOMs the 32 GiB RTX 5090 even at
MEM_FRACTION 0.80 (documented in 2026-06-01-opus-pressure-drift-rootcause.md
Gate c).  This is an INFRASTRUCTURE limit of the single-scan compile, NOT a model
defect.

This driver runs the SAME production wrfout path -- ``execute_daily_pipeline`` +
``_build_real_case`` (domain=d02, ``force_geopotential=True``, all Sprint-U
numerics, M9 surface diagnostics into the writer, hourly land-state refresh,
guards-on which are now non-load-bearing after 13bdef4/512a40e) -- but swaps the
per-hour ``forecast_fn`` from ``run_forecast_operational`` (single scan) to
``run_forecast_operational_segmented`` with a small ``segment_steps``.  The
segmented path is BITWISE identical to the single scan (proofs/perf/
segscan_equiv.json: max abs diff == 0 on every field incl. the radiation step)
while bounding peak GPU memory to one segment's working set.

So this exercises EVERY operational operator the product runs -- including the
``_refresh_grid_p_from_finished`` -> ``diagnose_pressure_al_alt`` alb-from-phb
pressure fix and the de-load-beared theta limiter -- on the wrfout-writing
product path, just without the single-scan compile OOM.  It writes real
wrfout_d02 files for the cheap, re-runnable Exner/T2 scoring phase of
``scripts/diag/d02_t2bias_diagnosis.py score``.

USAGE
  PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
    taskset -c 0-3 python proofs/v010_validation/d02_oomsafe_production_run.py \
      --run-id <id> --run-root <root> --hours 24 --segment-steps 60 --tag <tag>
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

if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUT_ROOT = Path("/tmp/v010_d02_oomsafe_runs")
PROOF_DIR = ROOT / "proofs" / "v010_validation"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--segment-steps", type=int, default=60)
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    from gpuwrf.integration.daily_pipeline import (
        DailyPipelineConfig,
        _build_real_case,
        execute_daily_pipeline,
        resolve_run_dir,
        write_json,
    )
    from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented
    from jax import block_until_ready

    seg = int(args.segment_steps)

    def segmented_forecast_fn(state, namelist, hours):
        result = run_forecast_operational_segmented(
            state, namelist, float(hours), segment_steps=seg
        )
        block_until_ready(result)
        return result

    run_root = Path(args.run_root)
    run_dir = resolve_run_dir(args.run_id, run_root)
    tag = f"_{args.tag}" if args.tag else ""
    output_dir = Path(args.output_root) / f"d02_{args.run_id}{tag}"
    config = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=PROOF_DIR,
        run_root=run_root,
        score=False,
        domain="d02",
        dt_s=float(args.dt_s),
        acoustic_substeps=int(args.acoustic_substeps),
        radiation_cadence_steps=int(args.radiation_cadence_steps),
    )
    print(
        f"=== d02 OOM-safe production pipeline (segment_steps={seg}): "
        f"run_id={args.run_id} hours={args.hours} -> {output_dir} ===",
        flush=True,
    )
    payload = execute_daily_pipeline(
        config, case_builder=_build_real_case, forecast_fn=segmented_forecast_fn
    )
    payload["forecast_fn"] = f"run_forecast_operational_segmented(segment_steps={seg})"
    payload["orchestration_cpu_affinity"] = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    write_json(PROOF_DIR / f"pipeline_run_d02_oomsafe{tag}.json", payload)
    print(
        json.dumps(
            {
                "verdict": payload.get("verdict"),
                "all_finite": (payload.get("all_finite_check") or {}).get("all_finite"),
                "n_wrfout": len(payload.get("wrfout_files", [])),
                "output_dir": str(output_dir),
                "wall_clock_total_s": payload.get("wall_clock_total_s"),
            },
            indent=2,
        )
    )
    return 0 if payload.get("verdict") not in ("PIPELINE_BLOCKED",) else 2


if __name__ == "__main__":
    raise SystemExit(main())
