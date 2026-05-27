#!/usr/bin/env python
"""Select a continuous Canary CPU-run window for multi-day proof execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import CANARY_RUN_ROOT, SPRINT_DIR, discover_canary_cases, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--run-root", type=Path, default=CANARY_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "canary_case_manifest.json")
    args = parser.parse_args(argv)
    write_json(args.output, discover_canary_cases(args.run_root, window_days=int(args.window_days)))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
