#!/usr/bin/env python3
"""Safe wrapper skeleton for local agent CLIs.

This script detects installed tools and prints planned calls. It does not assume
exact CLI syntax and does not execute model agents by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys


TOOLS = ["claude", "codex", "gemini", "opencode"]


def detect() -> dict[str, str | None]:
    return {tool: shutil.which(tool) for tool in TOOLS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list detected tools")
    parser.add_argument("--tool", choices=TOOLS)
    parser.add_argument("--prompt-file")
    parser.add_argument("--resume-id", help="reserved for configured CLIs; not interpreted here")
    args = parser.parse_args()

    tools = detect()
    result = {"ok": True, "detected_tools": tools}
    if args.tool:
        result["requested_tool"] = args.tool
        result["available"] = tools[args.tool] is not None
        result["note"] = "Configure exact CLI syntax before execution. This wrapper is non-destructive."
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
