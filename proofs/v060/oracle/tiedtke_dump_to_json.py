#!/usr/bin/env python3
"""Parse the flat key=value dump from tiedtke_oracle into JSON savepoints."""

from __future__ import annotations

import json
import re
import sys


SCALARS = {
    "CASE",
    "KX",
    "DT",
    "DX",
    "STEPCU",
    "ITIMESTEP",
    "RAINCV",
    "PRATEC",
    "RAINCV_DIRECT",
    "KTYPE",
    "CU_ACT_FLAG",
    "QFX",
    "XLAND",
}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")
INT_SCALARS = {"CASE", "KX", "STEPCU", "ITIMESTEP", "KTYPE", "CU_ACT_FLAG"}


def _scalar(name: str, value: str) -> int | float:
    return int(value) if name in INT_SCALARS else float(value)


def main(infile: str, outfile: str) -> None:
    scalars: dict[str, int | float] = {}
    cols: dict[str, dict[int, float]] = {}
    with open(infile, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = COL_RE.match(line)
            if m:
                name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
                cols.setdefault(name, {})[idx] = val
                continue
            m = SCAL_RE.match(line)
            if m and m.group(1) in SCALARS:
                scalars[m.group(1)] = _scalar(m.group(1), m.group(2))

    kx = int(scalars["KX"])
    out = {
        "schema": "wrf-v060-tiedtke-column-savepoint-v1",
        "scalars": scalars,
        "columns": {},
    }
    for name, values in cols.items():
        n = kx + 1 if name in {"P8W", "W"} else kx
        out["columns"][name] = [values[k] for k in range(1, n + 1)]

    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
