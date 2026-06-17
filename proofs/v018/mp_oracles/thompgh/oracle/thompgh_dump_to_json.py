#!/usr/bin/env python3
"""Parse the flat key=value dump from thompgh_oracle into a JSON savepoint.

Scalars: CASE, KX, DT, RAINNC, RAINNCV, SNOWNC, SNOWNCV, GRAUPELNC,
GRAUPELNCV, SR.
Columns (length KX, Fortran index 1..KX -> JSON list index 0..KX-1):
  inputs : T_IN, QV_IN, QC_IN, QR_IN, QI_IN, QS_IN, QG_IN, NI_IN, NR_IN,
           NC_IN, NG_IN, QB_IN, NWFA_IN, NIFA_IN, NBCA_IN, PII, P, DELZ, DEN
  outputs: T_OUT, QV_OUT, QC_OUT, QR_OUT, QI_OUT, QS_OUT, QG_OUT, NI_OUT,
           NR_OUT, NC_OUT, NG_OUT, QB_OUT, RE_CLOUD, RE_ICE, RE_SNOW

QB = qvolg (graupel volume mixing ratio); the variable-density graupel-hail
substrate leaf. NG = qng (graupel number). There is NO separate hail (qh/Nh)
category in mp=38 -- hail is variable-density graupel.

Third positional arg (label) is accepted and ignored (matches the WSM7/WDM7
dumper invocation in the build scripts).
"""
import json
import re
import sys

SCALARS = {"CASE", "KX", "DT", "RAINNC", "RAINNCV", "SNOWNC", "SNOWNCV",
           "GRAUPELNC", "GRAUPELNCV", "SR"}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str, label: str = "") -> None:
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
                scalars[name] = int(val) if name in {"CASE", "KX"} else float(val)

    if "KX" not in scalars:
        raise SystemExit(f"oracle dump missing KX (truncated/failed run): {infile}")
    kx = scalars["KX"]
    out = {"label": label, "scalars": scalars, "columns": {}}
    for name, d in cols.items():
        out["columns"][name] = [d[k] for k in range(1, kx + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    main(sys.argv[1], sys.argv[2], label)
