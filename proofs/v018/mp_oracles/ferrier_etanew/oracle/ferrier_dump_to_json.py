#!/usr/bin/env python3
"""Parse Ferrier (etampnew) flat oracle dumps into structured JSON savepoints."""

from __future__ import annotations

import json
import re
import sys

SCALARS = {"CASE", "KX", "DT", "RAINNC", "RAINNCV", "SR"}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str, scheme: str) -> None:
    scalars: dict[str, int | float | str] = {"SCHEME": scheme}
    cols: dict[str, dict[int, float]] = {}
    with open(infile) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(("INIT_FATAL", "RUN_FATAL", "FATAL", "WRF_ERROR_FATAL")):
                raise SystemExit(f"oracle reported fatal: {line}")
            m = COL_RE.match(line)
            if m:
                name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
                cols.setdefault(name, {})[idx] = val
                continue
            m = SCAL_RE.match(line)
            if m and m.group(1) in SCALARS:
                name, val = m.group(1), m.group(2)
                scalars[name] = int(val) if name in {"CASE", "KX"} else float(val)

    kx = int(scalars["KX"])
    out = {"scalars": scalars, "columns": {}}
    for name, values in cols.items():
        out["columns"][name] = [values[k] for k in range(1, kx + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: ferrier_dump_to_json.py INFILE OUTFILE SCHEME")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
