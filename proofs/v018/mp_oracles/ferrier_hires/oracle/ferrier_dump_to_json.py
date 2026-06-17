#!/usr/bin/env python3
"""Convert Ferrier Fortran oracle text dumps into JSON savepoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def parse_dump(path: Path) -> dict:
    metadata: dict[str, str | int] = {}
    scalars: dict[str, float] = {}
    columns: dict[str, list[float]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "META":
            key, value = parts[1].split("=", 1)
            metadata[key] = int(value) if value.isdigit() else value
        elif parts[0] == "SCALAR":
            scalars[parts[1]] = float(parts[2])
        elif parts[0] == "FIELD":
            columns[parts[1]] = [float(x) for x in parts[2:]]
    if "scheme" not in metadata or "case" not in metadata:
        raise ValueError(f"{path} is missing scheme/case metadata")
    return {"metadata": metadata, "scalars": scalars, "columns": columns}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ferrier_dump_to_json.py input.dump output.json", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    data = parse_dump(src)
    dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
