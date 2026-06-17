#!/usr/bin/env python3
"""Parse the flat key=value dump from goddard4ice_oracle into a JSON savepoint.

Goddard GCE 4-ice (WRF mp_physics=7, GSFCGCE_4ICE_NUWRF). Hail is its own
prognostic category (qh), separate from graupel (qg) -- the 4-ice distinction.

Scalars: CASE, KX, DT, RAINNC, RAINNCV, SNOWNC, SNOWNCV, GRAUPELNC,
GRAUPELNCV, HAILNC, HAILNCV, SR.
Columns (length KX, Fortran index 1..KX -> JSON list index 0..KX-1):
  inputs : T_IN, QV_IN, QC_IN, QR_IN, QI_IN, QS_IN, QG_IN, QH_IN,
           PII, DEN, P, DELZ
  outputs: T_OUT, QV_OUT, QC_OUT, QR_OUT, QI_OUT, QS_OUT, QG_OUT, QH_OUT,
           RE_CLOUD, RE_RAIN, RE_ICE, RE_SNOW, RE_GRAUPEL, RE_HAIL

Usage:
  goddard4ice_dump_to_json.py <infile.txt> <outfile.json> [scheme_label]
"""
import json
import re
import sys

SCALARS = {"CASE", "KX", "DT", "RAINNC", "RAINNCV", "SNOWNC", "SNOWNCV",
           "GRAUPELNC", "GRAUPELNCV", "HAILNC", "HAILNCV", "SR"}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str, label: str = "") -> None:
    scalars: dict = {}
    cols: dict = {}
    with open(infile) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(("INIT_FATAL", "RUN_FATAL", "FATAL",
                                "WRF_ERROR_FATAL")):
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
        raise SystemExit(f"no KX scalar parsed from {infile} (empty/garbled dump)")
    kx = scalars["KX"]
    out = {"scheme": label, "scalars": scalars, "columns": {}}
    for name, d in cols.items():
        out["columns"][name] = [d[k] for k in range(1, kx + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    main(sys.argv[1], sys.argv[2], label)
