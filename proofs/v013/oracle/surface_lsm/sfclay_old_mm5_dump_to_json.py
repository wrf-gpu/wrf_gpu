#!/usr/bin/env python3
"""Parse the flat key=value dump from sfclay_old_mm5_oracle into JSON savepoints.

All per-column fields (length N) are stored under columns[NAME] (0-based lists).
Top scalars: CASE, REGIME (case name), N, ISFFLX, ITIMESTEP.
"""
import json
import re
import sys

INT_SCALARS = {"CASE", "N", "ISFFLX", "ITIMESTEP", "FULL_WRF_EXE"}
STR_SCALARS = {"REGIME", "PRECISION_MODE"}

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
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            if name in INT_SCALARS:
                scalars[name] = int(val)
            elif name in STR_SCALARS:
                scalars[name] = val.strip()

    n = scalars["N"]
    out = {"scalars": scalars, "columns": {}}
    for name, d in cols.items():
        out["columns"][name] = [d[k] for k in range(1, n + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={len(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
