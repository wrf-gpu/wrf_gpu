#!/usr/bin/env python3
"""Generate the M1 Canary WRF-derived fixture."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.fixtures.wrf_slice import extraction_summary, extract_fixture  # noqa: E402


def main() -> int:
    paths = extract_fixture()
    print(extraction_summary(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
