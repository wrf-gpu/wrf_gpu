#!/usr/bin/env python3
"""Write the v0.22 G1 data-assimilation small-grid proof object."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

from gpuwrf.validation.data_assimilation_gate import run_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("proofs/v022/feature_push/G1_DATA_ASSIMILATION_GATE.json"),
    )
    args = parser.parse_args(argv)
    payload = run_gate(output=args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
