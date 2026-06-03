#!/usr/bin/env python3
"""Parse the flat classic-RRTM LW oracle dump into a structured JSON savepoint."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


STRING_SCALARS = {"REGIME"}
INT_SCALARS = {"CASE", "KX", "ICLOUD", "YR", "FULL_WRF_EXE"}
FLOAT_SCALARS = {"PTOP", "JULIAN", "EMISS", "TSK", "GLW", "OLR"}
SCALARS = STRING_SCALARS | INT_SCALARS | FLOAT_SCALARS
INTERFACE_COLS = {"T8W", "P8W"}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str) -> None:
    scalars: dict[str, int | float | str] = {}
    cols: dict[str, dict[int, float]] = {}
    with Path(infile).open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            match = COL_RE.match(line)
            if match:
                name, idx, value = match.group(1), int(match.group(2)), float(match.group(3))
                cols.setdefault(name, {})[idx] = value
                continue
            match = SCAL_RE.match(line)
            if match and match.group(1) in SCALARS:
                name, value = match.group(1), match.group(2)
                if name in STRING_SCALARS:
                    scalars[name] = value
                elif name in INT_SCALARS:
                    scalars[name] = int(value)
                else:
                    scalars[name] = float(value)

    kx = int(scalars["KX"])
    out = {"scalars": scalars, "columns": {}}
    for name, values in cols.items():
        upper = kx + 1 if name in INTERFACE_COLS else kx
        out["columns"][name] = [values[k] for k in range(1, upper + 1)]

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
