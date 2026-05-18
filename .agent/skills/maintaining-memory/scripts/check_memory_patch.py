#!/usr/bin/env python3
"""Validate a memory patch from inside the skill folder."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "usage: check_memory_patch.py <patch>"}))
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps({"ok": False, "error": f"missing {path}"}))
        return 1
    text = path.read_text(encoding="utf-8").lower()
    required = ["scope", "evidence", "proposed destination", "reviewer status"]
    missing = [item for item in required if item not in text]
    print(json.dumps({"ok": not missing, "missing": missing}, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
