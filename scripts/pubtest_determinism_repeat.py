#!/usr/bin/env python
"""Run the determinism-repeat publication wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR
from pubtest_execute_high_priority import _write_determinism


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "determinism_repeat.json")
    args = parser.parse_args(argv)
    _write_determinism(args.output.parent)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
