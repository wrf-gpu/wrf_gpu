#!/usr/bin/env python3
"""Static scan for common GPU-kernel anti-patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PATTERNS = [
    ".cpu()",
    ".numpy()",
    ".item()",
    "device_get(",
    "copy_to_host",
    "cudaMemcpy",
    "asnumpy(",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    findings = []
    for path in args.paths:
        if path.is_dir():
            files = list(path.rglob("*.py"))
        else:
            files = [path]
        for file_path in files:
            if not file_path.exists():
                findings.append({"file": str(file_path), "pattern": "missing"})
                continue
            for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                for pattern in PATTERNS:
                    if pattern in line:
                        findings.append({"file": str(file_path), "line": lineno, "pattern": pattern})
    print(json.dumps({"ok": not findings, "findings": findings}, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
