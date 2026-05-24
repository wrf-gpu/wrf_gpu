#!/usr/bin/env python
"""Emit the mandatory M6B0-R golden small-domain savepoint slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m6b0r_wrf_savepoint_extract import SPRINT, emit_tier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=SPRINT / "savepoints" / "golden")
    args = parser.parse_args()
    manifest = emit_tier("golden", args.steps, args.output)
    runid_path = SPRINT / "proof_golden_slice_runid.txt"
    runid_path.write_text(str(manifest["run_id"]) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"golden_run_id={manifest['run_id']}")
    print(f"proof_golden_slice_runid={runid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
