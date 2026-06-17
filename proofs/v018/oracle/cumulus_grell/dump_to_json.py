#!/usr/bin/env python3
"""Convert the v0.18 Grell standalone oracle text dump to JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def parse(path: Path) -> dict:
    scalars: dict[str, float] = {}
    columns: dict[str, list[float]] = {}
    case_id: int | None = None
    scheme: str | None = None
    regime: str | None = None
    current: str | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("CASE "):
            case_id = int(line.split()[1])
            current = None
            continue
        if line.startswith("SCHEME "):
            scheme = line.split()[1]
            current = None
            continue
        if line.startswith("REGIME "):
            regime = line.split(maxsplit=1)[1]
            current = None
            continue
        if line.startswith("SCALAR "):
            _, name, value = line.split()
            scalars[name] = float(value)
            current = None
            continue
        if line.startswith("COLUMN "):
            current = line.split()[1]
            columns[current] = []
            continue
        if current is not None:
            _k, value = line.split()
            columns[current].append(float(value))
    return {
        "schema": "wrf-v018-cumulus-grell-column-savepoint-v1",
        "case": case_id,
        "regime": regime,
        "scheme": scheme,
        "scalars": scalars,
        "columns": columns,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: dump_to_json.py INPUT.txt OUTPUT.json", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = parse(src)
    dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {dst}: scalars={len(data['scalars'])} cols={list(data['columns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
