#!/usr/bin/env python
"""Run M6-S6 Tier-3 TSC1.0 drift-envelope proof generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.gen2_accessor import DEFAULT_M6_BOUNDARY_REPLAY, DEFAULT_M6_GEN2_RUN_DIR
from gpuwrf.io.proof_schemas import Tier3DriftEnvelope
from gpuwrf.validation.tier3_coupled import DEFAULT_ARTIFACT, build_tier3_artifact


def _lead_hours(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in text.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_M6_GEN2_RUN_DIR))
    parser.add_argument("--boundary", default=str(DEFAULT_M6_BOUNDARY_REPLAY))
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--lead-hours", type=_lead_hours, default=(6.0, 12.0, 24.0))
    parser.add_argument("--d02-dt-s", type=float, default=18.0)
    parser.add_argument("--radiation-cadence-s", type=float, default=540.0)
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--skip-d02", action="store_true", help="Only build reduced TSC + oracle proof; mark d02 drift BLOCKED.")
    parser.add_argument("--blocked-reason", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_tier3_artifact(
        output_path=args.output,
        run_dir=args.run_dir,
        boundary_path=args.boundary,
        domain=args.domain,
        leads_h=args.lead_hours,
        run_d02=not args.skip_d02,
        d02_blocked_reason=args.blocked_reason,
        d02_dt_s=args.d02_dt_s,
        radiation_cadence_s=args.radiation_cadence_s,
    )
    Tier3DriftEnvelope.validate_dict(payload)
    print(json.dumps({"status": payload["status"], "output": str(Path(args.output))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
