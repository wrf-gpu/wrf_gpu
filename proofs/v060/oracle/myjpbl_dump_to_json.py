#!/usr/bin/env python3
"""Parse the flat key=value dump from myjpbl_oracle into a structured JSON savepoint.

Scalars: CASE, KX, ITIMESTEP, STEPBL, KPBL (int); DT and all 2-D surface fields (float).
Columns (length KX): U,V,T,TH,QV,QC,PMID,DZ,EXNER,RHO,TKE_MYJ,EXCH_H,EL_MYJ,
                     RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN.
Interface column (length KX+1): PINT.
"""
import json
import re
import sys

INT_SCALARS = {"CASE", "KX", "ITIMESTEP", "STEPBL", "KPBL"}
STR_SCALARS = {"REGIME"}
FLOAT_SCALARS = {
    "DT", "HT", "TSK", "XLAND",
    "USTAR", "ZNT", "AKHS", "AKMS", "CHKLOWQ", "ELFLX",
    "THZ0", "QZ0", "UZ0", "VZ0", "QSFC", "CT",
    "PBLH", "MIXHT",
}
INTERFACE_COLS = {"PINT"}

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
            elif name in FLOAT_SCALARS:
                scalars[name] = float(val)

    kx = scalars["KX"]
    out = {"scalars": scalars, "columns": {}}
    for name, d in cols.items():
        n = kx + 1 if name in INTERFACE_COLS else kx
        out["columns"][name] = [d[k] for k in range(1, n + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
