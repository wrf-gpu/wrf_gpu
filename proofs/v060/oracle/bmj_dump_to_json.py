#!/usr/bin/env python3
"""Parse the flat key=value dump from bmj_oracle into JSON savepoints."""

from __future__ import annotations

import json
import re
import sys

SCALARS = {
    "CASE",
    "KX",
    "REGIME",
    "DT",
    "STEPCU",
    "XLAND",
    "KPBL",
    "LOWLYR",
    "CLDEFI_OUT",
    "RAINCV",
    "PRATEC",
    "CUTOP",
    "CUBOT",
}

INTEGER_SCALARS = {"CASE", "KX", "STEPCU", "KPBL", "LOWLYR"}
STRING_SCALARS = {"REGIME"}

COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str) -> None:
    scalars: dict = {}
    cols: dict = {}
    with open(infile) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(("INIT_FATAL", "RUN_FATAL", "FATAL")):
                raise SystemExit(f"oracle reported fatal: {line}")
            m = COL_RE.match(line)
            if m:
                name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
                cols.setdefault(name, {})[idx] = val
                continue
            m = SCAL_RE.match(line)
            if m and m.group(1) in SCALARS:
                name, val = m.group(1), m.group(2).strip()
                if name in STRING_SCALARS:
                    scalars[name] = val
                elif name in INTEGER_SCALARS:
                    scalars[name] = int(val)
                else:
                    scalars[name] = float(val)

    kx = scalars["KX"]
    out = {"scalars": scalars, "columns": {}}
    for name, d in cols.items():
        n = kx + 1 if name == "PINT" else kx
        out["columns"][name] = [d[k] for k in range(1, n + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
