#!/usr/bin/env python
"""Compatibility entry point for the M6B0-R HDF5 savepoint extractor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m6b0r_wrf_savepoint_extract import emit_tier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = emit_tier(args.tier, args.steps, args.output)
    print(f"tier={manifest['tier']}")
    print(f"savepoint_count={len(manifest['files'])}")
    print(f"total_bytes={manifest['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
