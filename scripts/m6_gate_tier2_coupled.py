#!/usr/bin/env python
"""Gate the M6-S4 Tier-2 coupled invariant proof object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.proof_schemas import Tier2CoupledInvariants


DEFAULT_ARTIFACT = ROOT / "artifacts" / "m6" / "tier2_coupled_invariants.json"


def run(path: Path) -> dict:
    data = Tier2CoupledInvariants.validate_file(path)
    threshold_failures = [name for name, record in data["thresholds"].items() if not record.get("pass", False)]
    if data["status"] != "PASS" or threshold_failures:
        raise RuntimeError(
            f"Tier-2 coupled invariant gate failed: status={data['status']} threshold_failures={threshold_failures}"
        )
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.artifact)
    data = run(path)
    print(json.dumps({"status": data["status"], "artifact": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
