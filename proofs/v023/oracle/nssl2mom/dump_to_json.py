#!/usr/bin/env python3
"""Parse the flat key=value dump from nssl2mom_oracle into a JSON savepoint.

Savepoint layout (schema wrf-v023-f2-nssl2mom-column-savepoint-v1):
  {"schema": "...", "scalars": {...}, "columns": {...}}

Scalars: CASE, KX, DT, CCN_IS_CCNA, QCCN, RAINNCV, SNOWNCV, GRPLNCV,
         HAILNCV, SR.
Columns (length KX, Fortran k=1..KX -> JSON list index 0..KX-1):
  inputs : TH_IN QV_IN QC_IN QR_IN QI_IN QS_IN QH_IN QHL_IN CCW_IN CRW_IN
           CCI_IN CSW_IN CHW_IN CHL_IN CN_IN VHW_IN VHL_IN PII P DZ W DN
  outputs: TH_OUT ... VHL_OUT DBZ_OUT RE_CLOUD_OUT RE_ICE_OUT RE_SNOW_OUT
NSSL naming: QH = graupel (WRF qg), QHL = hail (WRF qh).
"""
import json
import re
import sys

SCHEMA = "wrf-v023-f2-nssl2mom-column-savepoint-v1"
SCALARS = {"CASE", "KX", "DT", "CCN_IS_CCNA", "QCCN",
           "RAINNCV", "SNOWNCV", "GRPLNCV", "HAILNCV", "SR"}
INT_SCALARS = {"CASE", "KX", "CCN_IS_CCNA"}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str) -> None:
    scalars: dict = {}
    cols: dict = {}
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
                scalars[name] = int(val) if name in INT_SCALARS else float(val)

    kx = scalars["KX"]
    out = {"schema": SCHEMA, "scalars": scalars, "columns": {}}
    for name, d in cols.items():
        if len(d) != kx:
            raise SystemExit(f"column {name}: {len(d)} values, expected {kx}")
        out["columns"][name] = [d[k] for k in range(1, kx + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={len(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
