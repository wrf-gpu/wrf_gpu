#!/usr/bin/env python3
"""Parse a simple JSON profiler export or record an unsupported artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--benchmark", default="unknown")
    parser.add_argument("--backend", default="unknown")
    args = parser.parse_args()
    result = {
        "ok": args.report.exists(),
        "benchmark": args.benchmark,
        "backend": args.backend,
        "artifact_paths": [str(args.report)],
    }
    if args.report.exists() and args.report.suffix == ".json":
        result["raw"] = json.loads(args.report.read_text(encoding="utf-8"))
    elif args.report.exists():
        result["note"] = "Binary or unsupported report; keep raw artifact and extract metrics manually."
    else:
        result["error"] = "report not found"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
