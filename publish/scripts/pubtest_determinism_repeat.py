#!/usr/bin/env python
"""Run the determinism-repeat publication wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR
from pubtest_execute_high_priority import DEFAULT_EXECUTION_ROOT, _write_determinism


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "determinism_repeat.json")
    parser.add_argument("--run-heavy", action="store_true")
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    args = parser.parse_args(argv)
    _write_determinism(args.output.parent, run_heavy=bool(args.run_heavy), execution_root=args.execution_root)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
