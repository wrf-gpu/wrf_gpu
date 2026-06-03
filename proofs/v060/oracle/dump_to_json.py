#!/usr/bin/env python3
"""Parse the flat revised-MM5 surface-layer oracle dump into JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


INT_SCALARS = {"CASE", "N", "FULL_WRF_EXE", "ISFFLX", "SHALWATER_Z0", "ISFTCFLX", "IZ0TLND"}
STRING_SCALARS = {"REGIME_NAME"}
FLOAT_SCALARS = {"DT"}
SCALARS = INT_SCALARS | STRING_SCALARS | FLOAT_SCALARS
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
                if name in INT_SCALARS:
                    scalars[name] = int(value)
                elif name in FLOAT_SCALARS:
                    scalars[name] = float(value)
                else:
                    scalars[name] = value

    ncol = int(scalars["N"])
    out = {"scalars": scalars, "columns": {}}
    for name, values in cols.items():
        out["columns"][name] = [values[i] for i in range(1, ncol + 1)]

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path}: scalars={len(scalars)} cols={len(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
