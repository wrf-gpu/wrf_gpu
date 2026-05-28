#!/usr/bin/env python
"""Compare idealized proof objects already emitted by the high-priority runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR, read_json, write_json
from pubtest_execute_high_priority import execute


CASE_TO_FILE = {
    "warmbubble": "idealized_warmbubble.json",
    "density_current": "idealized_density_current.json",
    "schaer": "idealized_mountain_wave.json",
    "em_hill2d_x": "idealized_mountain_wave.json",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_TO_FILE), default="warmbubble")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--proof-dir", type=Path, default=SPRINT_DIR)
    parser.add_argument("--skip-gpu-probe", action="store_true")
    args = parser.parse_args(argv)
    source = args.proof_dir / CASE_TO_FILE[args.case]
    if not source.exists():
        execute(args.proof_dir, skip_gpu_probe=bool(args.skip_gpu_probe))
    payload = read_json(source) or {"status": "MISSING", "verdict": "MISSING"}
    output = args.output or source
    write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
