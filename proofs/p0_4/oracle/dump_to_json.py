#!/usr/bin/env python3
"""Parse the flat key=value dump from kf_oracle into a structured JSON savepoint.

Scalars: CASE, KX, DT, DX, STEPCU, TRIGGER, RAINCV, PRATEC, NCA, CUTOP, CUBOT,
         SHALL, TIMEC.
Columns (length KX, index 1..KX -> list index 0..KX-1):
         T, QV, P, DZ, RHO, U, V, W0AVG,
         RTHCUTEN, RQVCUTEN, RQCCUTEN, RQRCUTEN, RQICUTEN, RQSCUTEN.
"""
import json
import re
import sys

SCALARS = {"CASE", "KX", "DT", "DX", "STEPCU", "TRIGGER", "RAINCV", "PRATEC",
           "NCA", "CUTOP", "CUBOT", "SHALL", "TIMEC"}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str) -> None:
    scalars: dict = {}
    cols: dict = {}
    with open(infile) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = COL_RE.match(line)
            if m:
                name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
                cols.setdefault(name, {})[idx] = val
                continue
            m = SCAL_RE.match(line)
            if m and m.group(1) in SCALARS:
                name, val = m.group(1), m.group(2)
                scalars[name] = int(val) if name in {"CASE", "KX", "STEPCU", "TRIGGER"} else float(val)

    kx = scalars["KX"]
    out = {"scalars": scalars, "columns": {}}
    for name, d in cols.items():
        out["columns"][name] = [d[k] for k in range(1, kx + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
