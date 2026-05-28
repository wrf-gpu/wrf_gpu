#!/usr/bin/env python
"""Prepare analytic idealized-case IC summaries for WRF/GPU publication tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR, write_case_summary, write_json, wrf_provenance
from gpuwrf.fixtures.idealized_cases import build_schaer_mountain_wave, build_warmbubble


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("warmbubble", "em_hill2d_x", "schaer"), default="warmbubble")
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "inputs" / "warmbubble_ic_summary.json")
    args = parser.parse_args(argv)
    if args.case in {"schaer", "em_hill2d_x"}:
        summary = write_case_summary(args.output, build_schaer_mountain_wave())
    else:
        summary = write_case_summary(args.output, build_warmbubble())
    write_json(args.output.with_suffix(".provenance.json"), {"case": args.case, "wrf_provenance": wrf_provenance()})
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
