#!/usr/bin/env python
"""Record WRF idealized-reference provenance without mutating the WRF source tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubtest_common import SPRINT_DIR, proof_header, write_json, wrf_provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--output", type=Path, default=SPRINT_DIR / "wrf_reference_preflight.json")
    args = parser.parse_args(argv)
    payload = proof_header(f"WRF-REFERENCE-{args.case.upper()}", "BLOCKED", "REFERENCE_NOT_RUN")
    payload.update(
        {
            "case": args.case,
            "minutes": int(args.minutes),
            "wrf_provenance": wrf_provenance(),
            "reason": "This sprint did not compile or run stock WRF idealized targets; proof objects stamp blocked/failed gates honestly.",
        }
    )
    write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
