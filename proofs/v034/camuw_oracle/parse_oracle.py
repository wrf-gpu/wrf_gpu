#!/usr/bin/env python3
"""Parse camuw_oracle.exe text output into JSON: {const, estbl, records:[{col,name,step,in:{},out:{}}]}."""
import json
import re
import sys

LINE = re.compile(r"^(IN|OUT) ([A-Z0-9_]+)(?:\[(\d+)\])?=\s*(\S+)$")


def main(src, names_path, dst):
    names = [n.strip() for n in open(names_path) if n.strip()]
    const, estbl, recs, cur = {}, {}, [], None
    for raw in open(src):
        line = raw.strip()
        if line.startswith("CONST "):
            k, v = line[6:].split("=")
            const[k.strip()] = float(v)
        elif line.startswith("ESTBL["):
            m = re.match(r"ESTBL\[(\d+)\]=\s*(\S+)", line)
            estbl[int(m.group(1))] = float(m.group(2))
        elif line.startswith("BEGIN"):
            m = re.match(r"BEGIN col=(\d+) step=(\d+)", line)
            c = int(m.group(1))
            cur = {"col": c, "name": names[c - 1], "step": int(m.group(2)), "in": {}, "out": {}}
        elif line == "END":
            recs.append(cur)
            cur = None
        elif cur is not None:
            m = LINE.match(line)
            if not m:
                continue
            sect, key, idx, val = m.groups()
            d = cur["in" if sect == "IN" else "out"]
            v = float(val)
            if idx is None:
                d[key] = v
            else:
                d.setdefault(key, []).append(v)
    out = {"schema": "gpuwrf.v034.camuw_oracle.v1", "const": const,
           "estbl": [estbl[i] for i in sorted(estbl)], "records": recs}
    json.dump(out, open(dst, "w"))
    print(f"wrote {dst}: {len(recs)} records")


if __name__ == "__main__":
    main(*sys.argv[1:4])
