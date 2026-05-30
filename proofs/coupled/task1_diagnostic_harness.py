"""Task 1 coupled diagnostic harness on the STABLE real-case config.

Uses `_build_real_case` (top_lid=True, epssm=0.5, force_fp64, flux advection,
damping -- the F7-closed dycore config) with all physics ON and guards OFF, runs
the instrumented forecast, and emits per-operator verdicts. The acceptance bar is
no UNEXPLAINED MISSING/NOISY_ZERO: a zero microphysics delta on this dusk-init d02
case is INACTIVE_PHYSICAL (no supersaturation reached), NOT a coupling bug.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/coupled/task1_diagnostic_harness.py --hours 0.5
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

from gpuwrf.diagnostics.comprehensive_harness import (
    DIAGNOSTIC_SCHEMA_VERSION,
    build_diagnostic_report,
    initial_diagnostic_accumulator,
    run_diagnostic_forecast,
)
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _steps_for_hours

PROOF = Path("proofs/coupled")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=0.5)
    ap.add_argument("--rad-cadence", type=int, default=90)
    args = ap.parse_args()

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=True,
        radiation_cadence_steps=int(args.rad_cadence),
        time_utc=case.run_start,
    )

    steps_total = _steps_for_hours(float(args.hours), float(nl.dt_s))
    print(f"=== Task 1 DIAGNOSTIC HARNESS (+{args.hours}h={steps_total} steps, "
          f"guards OFF, all physics ON) ===", flush=True)

    acc = initial_diagnostic_accumulator(steps_total)
    t0 = time.time()
    # Single instrumented pass. run_diagnostic_forecast DONATES argnums (0,2) =
    # (state, accumulator), so it must be called exactly once on the live buffers.
    final_state, acc = run_diagnostic_forecast(
        case.state, nl, acc, float(args.hours), diagnostic_on=True
    )
    jax.block_until_ready(final_state.theta)
    t1 = time.time()
    t_base = t0

    report = build_diagnostic_report(
        accumulator=acc,
        namelist=nl,
        steps_total=steps_total,
        run_config={
            "run_dir": str(run_dir),
            "hours": float(args.hours),
            "dt_s": float(nl.dt_s),
            "acoustic_substeps": int(nl.acoustic_substeps),
            "run_physics": bool(nl.run_physics),
            "run_boundary": bool(nl.run_boundary),
            "disable_guards": bool(nl.disable_guards),
            "top_lid": bool(nl.top_lid),
            "epssm": float(nl.epssm),
            "force_fp64": bool(nl.force_fp64),
            "radiation_cadence_steps": int(nl.radiation_cadence_steps),
        },
        wrf_anchor_payload=None,
        commit="worker/opus/coupled",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds_total=t1 - t0,
        wall_seconds_diagnostic_overhead=(t1 - t_base) - (t_base - t0),
    )

    payload = report.to_dict() if hasattr(report, "to_dict") else dataclasses.asdict(report)
    fn = PROOF / "task1_diagnostic_harness.json"
    fn.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {fn}", flush=True)

    # Print per-operator verdicts for the report.
    ops = payload.get("operator_attribution_24h") or payload.get("operators") or {}
    if isinstance(ops, dict):
        for name, info in ops.items():
            if isinstance(info, dict) and "verdict" in info:
                print(f"  {name:24s}: {info['verdict']}", flush=True)
    print(f"\nschema={DIAGNOSTIC_SCHEMA_VERSION}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
