#!/usr/bin/env python3
"""Create a sprint folder from the AgentOS template."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".agent" / "sprints" / "SPRINT-template"


def slug_ok(slug: str) -> bool:
    return slug and all(ch.isalnum() or ch in "-_" for ch in slug)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sprint_id", help="short slug, for example m2-backend-bakeoff")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    if not slug_ok(args.sprint_id):
        print(json.dumps({"ok": False, "error": "sprint_id must be alnum/dash/underscore"}))
        return 1
    if not TEMPLATE.exists():
        print(json.dumps({"ok": False, "error": f"missing template: {TEMPLATE}"}))
        return 1

    target = ROOT / ".agent" / "sprints" / f"{args.date}-{args.sprint_id}"
    if target.exists():
        print(json.dumps({"ok": False, "error": f"sprint exists: {target}"}))
        return 1
    shutil.copytree(TEMPLATE, target)
    print(json.dumps({"ok": True, "sprint": str(target.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
