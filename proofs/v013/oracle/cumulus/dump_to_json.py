#!/usr/bin/env python3
"""Parse the flat key=value dump from a cumulus oracle into a JSON savepoint.

Shared by the v0.13 New-Tiedtke / KSAS / Grell-3D single-column oracle drivers
(all use the same flat ``NAME=val`` scalar + ``NAME[k]=val`` column dump format).
Interface arrays (length KX+1) are detected by name (``P8W`` / ``W``).
"""

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
    "CU_ACT_FLAG",
    "QFX",
    "HFX",
    "XLAND",
}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")
INT_SCALARS = {"CASE", "KX", "STEPCU", "ITIMESTEP", "CU_ACT_FLAG"}
IFACE_COLS = {"P8W", "W"}


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
        "schema": "wrf-v013-cumulus-column-savepoint-v1",
        "scalars": scalars,
        "columns": {},
    }
    for name, values in cols.items():
        n = kx + 1 if name in IFACE_COLS else kx
        out["columns"][name] = [values[k] for k in range(1, n + 1)]

    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
