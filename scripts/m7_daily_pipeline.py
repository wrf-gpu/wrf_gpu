#!/usr/bin/env python
"""M7 daily pipeline CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.integration.daily_pipeline import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_ID,
    RUN_ROOT,
    SPRINT_DIR,
    DailyPipelineConfig,
    execute_daily_pipeline,
)
from gpuwrf.validation.forecast_vs_obs import DEFAULT_AEMET_ROOT  # noqa: E402


def _pin_orchestration_cpus() -> list[int] | None:
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpus = {0, 1, 2, 3}
    try:
        os.sched_setaffinity(0, cpus)
    except OSError:
        pass
    return sorted(os.sched_getaffinity(0))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--proof-dir", type=Path, default=SPRINT_DIR)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--aemet-root", type=Path, default=DEFAULT_AEMET_ROOT)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--restart-at-hour", type=int, default=None)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--domain", default="d02")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    affinity = _pin_orchestration_cpus()
    config = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(args.hours),
        output_dir=args.output_dir,
        proof_dir=args.proof_dir,
        run_root=args.run_root,
        aemet_root=args.aemet_root,
        score=bool(args.score),
        restart_at_hour=args.restart_at_hour,
        repeat=bool(args.repeat),
        domain=args.domain,
    )
    payload = execute_daily_pipeline(config)
    if affinity is not None:
        payload["orchestration_cpu_affinity"] = affinity
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("verdict") == "PIPELINE_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
