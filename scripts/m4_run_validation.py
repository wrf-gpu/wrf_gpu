#!/usr/bin/env python3
"""Run M4 tier-1, tier-2, and tier-3 validators and emit proof JSON files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.validation.tier1 import run_tier1  # noqa: E402
from gpuwrf.validation.tier2 import run_tier2  # noqa: E402
from gpuwrf.validation.tier3 import run_tier3  # noqa: E402


def main() -> int:
    """Executes every M4 validation tier as an idempotent CLI."""

    records = {"tier1": run_tier1(), "tier2": run_tier2(), "tier3": run_tier3()}
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0 if all(record.get("pass") for record in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
