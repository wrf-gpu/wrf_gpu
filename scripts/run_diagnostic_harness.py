#!/usr/bin/env python
"""Driver for the comprehensive diagnostic harness.

Runs the operational forecast loop with per-operator + per-step instrumentation
and emits the single ``diagnostic_report.json`` artifact that the manager reads
to decide next-sprint priorities.

Usage
-----

CPU smoke (default, 1h, the pinned 20260521 case)::

    taskset -c 0-3 python scripts/run_diagnostic_harness.py --hours 0.5

GPU production (24h Canary)::

    taskset -c 0-3 python scripts/run_diagnostic_harness.py \\
        --hours 24 --jax-platform gpu \\
        --output proofs/diagnostic_harness/diagnostic_report_24h.json

The artifact is always written to ``--output``; stdout gets a single
``HARNESS REPORT`` JSON line for tmux log scraping.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")


DEFAULT_RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
DEFAULT_OUTPUT = ROOT / "proofs/diagnostic_harness/diagnostic_report.json"
DEFAULT_WRF_ANCHOR = ROOT / "proofs/m9/operational_trace_hourly.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _load_wrf_anchor(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload["_source_path"] = str(path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Canary run directory")
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--dt-s", type=float, default=10.0)
    parser.add_argument("--acoustic-substeps", type=int, default=10)
    parser.add_argument(
        "--radiation-cadence-steps",
        type=int,
        default=60,
        help="set to >= steps_total to disable radiation; 60 = 10min cadence at dt=10s",
    )
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--disable-guards", action="store_true")
    parser.add_argument(
        "--measure-overhead",
        action="store_true",
        help="run twice: once with diagnostic_on=False, once with True, and report the ratio",
    )
    parser.add_argument("--jax-platform", default=None, help="cpu or gpu (sets JAX_PLATFORMS)")
    parser.add_argument("--wrf-anchor", type=Path, default=DEFAULT_WRF_ANCHOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.jax_platform is not None:
        os.environ["JAX_PLATFORMS"] = args.jax_platform
        os.environ["JAX_PLATFORM_NAME"] = args.jax_platform

    # Import only after env vars are set so JAX picks up the platform.
    import jax
    from jax import config as jax_config
    import numpy as np

    jax_config.update("jax_enable_x64", True)

    from gpuwrf.diagnostics.comprehensive_harness import (
        DIAGNOSTIC_SCHEMA_VERSION,
        build_diagnostic_report,
        initial_diagnostic_accumulator,
        run_diagnostic_forecast,
    )
    from gpuwrf.integration.d02_replay import build_replay_case
    from gpuwrf.runtime.operational_mode import (
        OperationalNamelist,
        _steps_for_hours,
        run_forecast_operational,
    )

    if not args.run_dir.is_dir():
        sys.stderr.write(f"ERROR: run-dir does not exist: {args.run_dir}\n")
        sys.exit(2)

    if not args.quiet:
        print(f"[harness] loading replay case from {args.run_dir} ...", flush=True)
    case = build_replay_case(args.run_dir, domain=args.domain)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=float(args.dt_s),
        acoustic_substeps=int(args.acoustic_substeps),
        radiation_cadence_steps=int(args.radiation_cadence_steps),
        use_vertical_solver=True,
        disable_guards=bool(args.disable_guards),
    )
    namelist = namelist.__class__(
        grid=namelist.grid,
        tendencies=namelist.tendencies,
        metrics=namelist.metrics,
        dt_s=namelist.dt_s,
        acoustic_substeps=namelist.acoustic_substeps,
        rk_order=namelist.rk_order,
        epssm=namelist.epssm,
        top_lid=namelist.top_lid,
        run_physics=not bool(args.no_physics),
        run_boundary=not bool(args.no_boundary),
        radiation_cadence_steps=namelist.radiation_cadence_steps,
        boundary_config=namelist.boundary_config,
        use_vertical_solver=namelist.use_vertical_solver,
        disable_guards=namelist.disable_guards,
    )

    steps_total = _steps_for_hours(float(args.hours), float(namelist.dt_s))
    accumulator = initial_diagnostic_accumulator(steps_total)

    if not args.quiet:
        print(
            f"[harness] running diagnostic forecast: hours={args.hours} steps={steps_total} "
            f"physics={namelist.run_physics} boundary={namelist.run_boundary} guards={not namelist.disable_guards}",
            flush=True,
        )

    diag_overhead_s: float | None = None
    if args.measure_overhead:
        # Cold-compile a no-diagnostic run for the overhead baseline.
        baseline_acc = initial_diagnostic_accumulator(steps_total)
        # Warm cache
        _ = run_diagnostic_forecast(state, namelist, baseline_acc, float(args.hours), diagnostic_on=False)
        jax.block_until_ready(_)
        t0 = time.perf_counter()
        baseline_state, _ = run_diagnostic_forecast(state, namelist, baseline_acc, float(args.hours), diagnostic_on=False)
        jax.block_until_ready(baseline_state)
        baseline_wall = time.perf_counter() - t0
        # Warm cache for diagnostic-on
        _ = run_diagnostic_forecast(state, namelist, accumulator, float(args.hours), diagnostic_on=True)
        jax.block_until_ready(_)
        t1 = time.perf_counter()
        final_state, final_acc = run_diagnostic_forecast(
            state, namelist, accumulator, float(args.hours), diagnostic_on=True
        )
        jax.block_until_ready(final_state)
        diag_wall = time.perf_counter() - t1
        diag_overhead_s = diag_wall - baseline_wall
        wall_seconds_total = diag_wall
        if not args.quiet:
            ratio = diag_wall / max(baseline_wall, 1.0e-9)
            print(
                f"[harness] overhead: baseline={baseline_wall:.2f}s diagnostic={diag_wall:.2f}s ratio={ratio:.3f}",
                flush=True,
            )
    else:
        t0 = time.perf_counter()
        final_state, final_acc = run_diagnostic_forecast(
            state, namelist, accumulator, float(args.hours), diagnostic_on=True
        )
        jax.block_until_ready(final_state)
        wall_seconds_total = time.perf_counter() - t0

    if not args.quiet:
        print(f"[harness] forecast complete in {wall_seconds_total:.2f}s; building report ...", flush=True)

    wrf_anchor = _load_wrf_anchor(args.wrf_anchor) if args.wrf_anchor else None

    run_config: dict[str, Any] = {
        "case_run_dir": str(args.run_dir),
        "domain": args.domain,
        "hours": float(args.hours),
        "dt_s": float(args.dt_s),
        "steps_total": int(steps_total),
        "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
        "rk_order": int(namelist.rk_order),
        "acoustic_substeps": int(namelist.acoustic_substeps),
        "diagnostic_on": True,
        "disable_guards": bool(namelist.disable_guards),
        "run_physics": bool(namelist.run_physics),
        "run_boundary": bool(namelist.run_boundary),
        "platform": os.environ.get("JAX_PLATFORMS", "default"),
        "jax_devices": [str(d) for d in jax.devices()],
    }

    report = build_diagnostic_report(
        accumulator=final_acc,
        namelist=namelist,
        steps_total=steps_total,
        run_config=run_config,
        wrf_anchor_payload=wrf_anchor,
        commit=_git_commit(),
        generated_utc=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        wall_seconds_total=wall_seconds_total,
        wall_seconds_diagnostic_overhead=diag_overhead_s,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"[harness] report written to {args.output}", flush=True)
        print(f"[harness] headline: {report.to_dict()['headline_diagnosis']}", flush=True)
    print(
        f"HARNESS REPORT verdict={report.to_dict()['verdict']} "
        f"schema={DIAGNOSTIC_SCHEMA_VERSION} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
