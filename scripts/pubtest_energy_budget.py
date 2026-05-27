#!/usr/bin/env python
"""Run the energy-budget publication wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR
from pubtest_execute_high_priority import _write_conservation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="warmbubble")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--closed-domain", action="store_true")
    parser.add_argument("--flux-corrected", action="store_true")
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "conservation_energy_24h.json")
    args = parser.parse_args(argv)
    _write_conservation(args.output.parent)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
