#!/usr/bin/env python
"""Record GPU idealized-run preflight for publication tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR, gpu_probe, proof_header, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "gpu_ideal_preflight.json")
    parser.add_argument("--skip-gpu-probe", action="store_true")
    args = parser.parse_args(argv)
    probe = gpu_probe(skip=bool(args.skip_gpu_probe))
    payload = proof_header(f"GPU-IDEAL-{args.case.upper()}", "BLOCKED", "GPU_IDEAL_RUN_NOT_EXECUTED")
    payload.update(
        {
            "case": args.case,
            "minutes": int(args.minutes),
            "gpu_preflight": probe,
            "reason": "No idealized GPU integration runner was executed by this preflight wrapper.",
        }
    )
    write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
