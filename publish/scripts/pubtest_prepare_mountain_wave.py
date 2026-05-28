#!/usr/bin/env python
"""Prepare Schaer/em_hill2d_x mountain-wave IC summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR, write_case_summary
from gpuwrf.fixtures.idealized_cases import build_schaer_mountain_wave


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("schaer", "em_hill2d_x"), default="schaer")
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "inputs" / "schaer_mountain_wave_ic_summary.json")
    args = parser.parse_args(argv)
    write_case_summary(args.output, build_schaer_mountain_wave())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
