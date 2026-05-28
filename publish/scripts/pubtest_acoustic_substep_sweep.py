#!/usr/bin/env python
"""Run the acoustic-substep-sweep publication wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR, gpu_probe
from pubtest_execute_high_priority import _write_stability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "stability_acoustic_substep.json")
    parser.add_argument("--skip-gpu-probe", action="store_true")
    args = parser.parse_args(argv)
    _write_stability(args.output.parent, gpu=gpu_probe(skip=bool(args.skip_gpu_probe)))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
