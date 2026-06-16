#!/usr/bin/env python3
"""Small Nsight Compute wrapper that fails gracefully when ncu is absent."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="ncu-report")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ncu = shutil.which("ncu")
    if not ncu:
        print(json.dumps({"ok": False, "error": "ncu not found", "command": args.command}, indent=2))
        return 0
    planned = [ncu, "--set", "full", "--export", args.output, *args.command]
    if args.dry_run or not args.command:
        print(json.dumps({"ok": True, "dry_run": True, "planned": planned}, indent=2))
        return 0
    proc = subprocess.run(planned)
    print(json.dumps({"ok": proc.returncode == 0, "returncode": proc.returncode, "output": args.output}, indent=2))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
