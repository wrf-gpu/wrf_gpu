#!/usr/bin/env python
"""Prepare the Straka density-current analytic IC summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR, write_case_summary
from gpuwrf.fixtures.idealized_cases import build_density_current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "inputs" / "density_current_ic_summary.json")
    args = parser.parse_args(argv)
    write_case_summary(args.output, build_density_current())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
