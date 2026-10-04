#!/usr/bin/env python3
"""Parse the flat key=value dump from wsm6_oracle into a structured JSON savepoint.

Scalars: CASE, KX, DT, RAIN, RAINNCV, SNOW, SNOWNCV, GRAUPEL, GRAUPELNCV, SR.
Columns (length KX, Fortran index 1..KX -> JSON list index 0..KX-1):
  inputs : T_IN, QV_IN, QC_IN, QR_IN, QI_IN, QS_IN, QG_IN, PII, DEN, P, DELZ
  outputs: T_OUT, QV_OUT, QC_OUT, QR_OUT, QI_OUT, QS_OUT, QG_OUT,
           RE_CLOUD, RE_ICE, RE_SNOW
"""
import json
import re
import sys

SCALARS = {"CASE", "KX", "DT", "RAIN", "RAINNCV", "SNOW", "SNOWNCV",
           "GRAUPEL", "GRAUPELNCV", "SR"}
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
                name, val = m.group(1), m.group(2)
                scalars[name] = int(val) if name in {"CASE", "KX"} else float(val)

    kx = scalars["KX"]
    out = {"scalars": scalars, "columns": {}}
    for name, d in cols.items():
        out["columns"][name] = [d[k] for k in range(1, kx + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
