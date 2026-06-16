#!/usr/bin/env python3
"""Lightweight fixture manifest validator without requiring PyYAML."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REQUIRED = ["fixture_id:", "source:", "source_commit:", "scenario:", "variables:", "files:", "license_notes:"]


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "usage: validate_fixture_manifest.py <manifest.yaml>"}))
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps({"ok": False, "error": f"missing {path}"}))
        return 1
    text = path.read_text(encoding="utf-8")
    missing = [field for field in REQUIRED if field not in text]
    result = {"ok": not missing, "missing": missing}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
