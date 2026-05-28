#!/usr/bin/env python
"""Run the savepoint-parity-deep publication wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR
from pubtest_execute_high_priority import _write_savepoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "savepoint_parity_deep.json")
    args = parser.parse_args(argv)
    _write_savepoint(args.output.parent)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
