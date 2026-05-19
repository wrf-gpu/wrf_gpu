#!/usr/bin/env python3
"""Validate a committed fixture manifest against the pinned M1 schema."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.validation.compare_fixture import ManifestError, load_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: validate_fixture_manifest.py <manifest.yaml|manifest.json>", file=sys.stderr)
        return 2
    path = Path(args[0])
    try:
        load_manifest(path)
    except (ManifestError, OSError, ValueError) as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    print(f"{path}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
