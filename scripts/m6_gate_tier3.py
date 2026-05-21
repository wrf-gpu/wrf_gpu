#!/usr/bin/env python
"""Gate the M6-S6 Tier-3 TSC1.0 proof object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.proof_schemas import Tier3DriftEnvelope
from gpuwrf.validation.tier3_coupled import DEFAULT_ARTIFACT


def run(path: Path, *, allow_partial: bool = False) -> dict:
    data = Tier3DriftEnvelope.validate_file(path)
    allowed = {"GREEN"} | ({"PARTIAL"} if allow_partial else set())
    bad_leads = []
    for var, record in data["per_variable_status"].items():
        for lead, lead_record in record.get("leads", {}).items():
            if lead_record.get("status") != "GREEN":
                bad_leads.append(f"{var}{lead}:{lead_record.get('status')}")
    if data["status"] not in allowed or (bad_leads and not allow_partial):
        raise RuntimeError(f"Tier-3 gate failed: status={data['status']} bad_leads={bad_leads}")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.artifact)
    data = run(path, allow_partial=args.allow_partial)
    print(json.dumps({"status": data["status"], "artifact": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
