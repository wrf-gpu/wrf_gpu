#!/usr/bin/env python3
"""Parse the flat key=value dump from the Goddard LW oracle into a JSON savepoint.

The ``goddard_lw_oracle`` driver emits, one per line, either a scalar
``name=value`` or a comma-separated column ``name=v1,v2,...,vn`` (top-down,
k=1 TOA .. k=NP/NP+1 BOA). Scalars: case/np/nband_lw/ict/icb/tb/ts/glw/olr.
Columns: emiss(nband_lw), pl(np+1), ta/wa/oa/fcld/taucl_b1/tten(np),
flx/acflxd/acflxu(np+1).
"""

from __future__ import annotations

import json
import sys


INT_SCALARS = {"case", "np", "nband_lw", "ict", "icb"}
FLOAT_SCALARS = {"tb", "ts", "glw", "olr"}


def main(infile: str, outfile: str) -> None:
    scalars: dict[str, int | float] = {}
    columns: dict[str, list[float]] = {}
    with open(infile, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            name, _, rhs = line.partition("=")
            name = name.strip()
            rhs = rhs.strip()
            if "," in rhs:
                columns[name] = [float(tok) for tok in rhs.split(",") if tok.strip()]
            elif name in INT_SCALARS:
                scalars[name] = int(rhs)
            elif name in FLOAT_SCALARS:
                scalars[name] = float(rhs)
            else:
                # unknown single value -> store as float best-effort
                try:
                    scalars[name] = float(rhs)
                except ValueError:
                    scalars[name] = rhs  # type: ignore[assignment]

    out = {
        "schema": "wrf-v013-goddard-lw-column-savepoint-v1",
        "scheme": "ra_lw_physics=5 GSFC/Goddard NUWRF longwave (module_ra_goddard.F:lwrad)",
        "orientation": "top-down: k=1 TOA .. k=NP/NP+1 BOA (lwrad internal order)",
        "scalars": scalars,
        "columns": columns,
    }
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {outfile}: scalars={list(scalars)} cols={list(columns)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
