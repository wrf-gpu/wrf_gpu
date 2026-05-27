#!/usr/bin/env python
"""Run the M7 Tier-4 RMSE corpus bridge harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.validation.tier4_probtest import DEFAULT_ENDING_CYCLE, DEFAULT_HELDOUT_CYCLE
from gpuwrf.validation.tier4_rmse_harness import (
    DEFAULT_DOMAIN,
    DEFAULT_GEN2_ROOTS,
    DEFAULT_LEADS_H,
    DEFAULT_PINNED_GRID_YX,
    DEFAULT_VARIABLES,
    run_tier4_rmse_harness,
)


DEFAULT_OUTPUT = ROOT / ".agent/sprints/2026-05-27-m7-corpus-bridge/probationary_smoke.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-operational", action="store_true", help="enable the bounded N=5 probationary bridge")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gen2-root", action="append", default=None)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--ending-cycle", default=DEFAULT_ENDING_CYCLE)
    parser.add_argument("--heldout-cycle", default=DEFAULT_HELDOUT_CYCLE)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--leads-h", nargs="+", type=int, default=list(DEFAULT_LEADS_H))
    parser.add_argument("--pinned-grid-yx", nargs=2, type=int, default=list(DEFAULT_PINNED_GRID_YX))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(item) for item in args.gen2_root] if args.gen2_root else list(DEFAULT_GEN2_ROOTS)
    payload = run_tier4_rmse_harness(
        roots=roots,
        output_path=args.output,
        non_operational=args.non_operational,
        domain=args.domain,
        ending_cycle=args.ending_cycle,
        heldout_cycle=args.heldout_cycle,
        variables=args.variables,
        leads_h=args.leads_h,
        pinned_grid_yx=(int(args.pinned_grid_yx[0]), int(args.pinned_grid_yx[1])),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
