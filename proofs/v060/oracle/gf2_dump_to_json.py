#!/usr/bin/env python3
"""Parse the flat key=value dump from gf_oracle into structured JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ARRAY_RE = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]=(.*)$")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: dump_to_json.py case.txt out.json", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    scalars: dict[str, float | int | str] = {}
    columns: dict[str, dict[int, float]] = {}

    for raw in src.read_text().splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        m = ARRAY_RE.match(line)
        if m:
            name, idx, value = m.groups()
            columns.setdefault(name, {})[int(idx)] = float(value)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            if any(ch in value.lower() for ch in (".", "e")):
                scalars[key] = float(value)
            else:
                scalars[key] = int(value)
        except ValueError:
            scalars[key] = value

    packed_columns = {
        name: [vals[k] for k in sorted(vals)]
        for name, vals in sorted(columns.items())
    }
    out = {
        "schema": "gpuwrf.v060.grellfreitas_oracle_savepoint.v1",
        "scalars": scalars,
        "columns": packed_columns,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
