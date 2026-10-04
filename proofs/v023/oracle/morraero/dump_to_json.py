#!/usr/bin/env python3
"""Parse the flat key=value dump from morraero_oracle into a JSON savepoint.

Adapted from proofs/v060/oracle/dump_to_json.py (base Morrison oracle) with:
  * top-level "schema" string ("wrf-v023-f2-morraero-column-savepoint-v1"),
  * per-column length inference (KZH_IN is KX+1 long; everything else KX),
  * fail-closed finiteness check on every parsed value,
  * fail-closed required-column/required-scalar completeness check (a
    truncated dump -- e.g. a Fortran STOP inside the scheme -- must never
    silently produce a savepoint).
"""
import json
import math
import re
import sys

SCHEMA = "wrf-v023-f2-morraero-column-savepoint-v1"

INT_SCALARS = {"CASE", "KX", "KXP1", "AERCU_OPT", "NO_SRC_TYPES_CU", "PBL"}
REQ_SCALARS = INT_SCALARS | {
    "DT", "AERCU_FCT", "HT", "CU_UAF",
    "RAINNC", "RAINNCV", "SNOWNC", "SNOWNCV",
    "GRAUPELNC", "GRAUPELNCV", "SR",
}
REQ_COLS_KX = [
    # inputs
    "TH_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "QS_IN", "QG_IN",
    "NI_IN", "NS_IN", "NR_IN", "NG_IN", "NC_IN",
    "PII", "P", "DZ", "W",
    "QRCUTEN_IN", "QSCUTEN_IN", "QICUTEN_IN",
    "NR_CU_IN", "QR_CU_IN", "NS_CU_IN", "QS_CU_IN",
    "AEROCU_DUST1", "AEROCU_DUST2", "AEROCU_DUST3", "AEROCU_DUST4",
    "AEROCU_SEASALT", "AEROCU_SULFATE", "AEROCU_BCPHOB", "AEROCU_BCPHIL",
    "AEROCU_OCPHOB", "AEROCU_OCPHIL",
    # outputs
    "TH_OUT", "QV_OUT", "QC_OUT", "QR_OUT", "QI_OUT", "QS_OUT", "QG_OUT",
    "NI_OUT", "NS_OUT", "NR_OUT", "NG_OUT", "NC_OUT",
    "EFCG_OUT", "EFIG_OUT", "EFSG_OUT", "WACT_OUT",
    "CCN1_GS_OUT", "CCN2_GS_OUT", "CCN3_GS_OUT", "CCN4_GS_OUT",
    "CCN5_GS_OUT", "CCN6_GS_OUT", "CCN7_GS_OUT",
]
REQ_COLS_KXP1 = ["KZH_IN"]

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
                if not math.isfinite(val):
                    raise SystemExit(f"non-finite {name}[{idx}]={m.group(3)!r} in {infile}")
                cols.setdefault(name, {})[idx] = val
                continue
            m = SCAL_RE.match(line)
            if m and m.group(1) in REQ_SCALARS:
                name, raw = m.group(1), m.group(2)
                val = int(raw) if name in INT_SCALARS else float(raw)
                if not math.isfinite(float(val)):
                    raise SystemExit(f"non-finite scalar {name}={raw!r} in {infile}")
                scalars[name] = val

    missing_s = REQ_SCALARS - set(scalars)
    if missing_s:
        raise SystemExit(f"missing scalars {sorted(missing_s)} in {infile} (truncated dump?)")
    kx = scalars["KX"]
    expected = {name: kx for name in REQ_COLS_KX}
    expected.update({name: kx + 1 for name in REQ_COLS_KXP1})
    missing_c = set(expected) - set(cols)
    if missing_c:
        raise SystemExit(f"missing columns {sorted(missing_c)} in {infile} (truncated dump?)")

    out = {"schema": SCHEMA, "scalars": scalars, "columns": {}}
    for name, d in cols.items():
        n = expected.get(name, max(d))
        if sorted(d) != list(range(1, n + 1)):
            raise SystemExit(f"column {name} incomplete: has {len(d)} entries, want 1..{n}")
        out["columns"][name] = [d[k] for k in range(1, n + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={len(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
