"""Summarize an onset_localize JSONL trace into a compact growth table."""
import json
import sys

path = sys.argv[1]
fields = sys.argv[2].split(",") if len(sys.argv) > 2 else ["theta", "w", "u", "v"]
rows = []
for line in open(path):
    r = json.loads(line)
    if r.get("kind") != "hour":
        continue
    cell = {"h": r["lead_hours"], "finite": r["all_finite"]}
    for f in fields:
        fr = r["fields"].get(f, {})
        cell[f] = (round(fr.get("absmax", float("nan")), 3), fr.get("worst_level"))
    rows.append(cell)

hdr = "  h   fin  " + "  ".join(f"{f}(absmax@k)" for f in fields)
print(hdr)
for c in rows:
    line = f"{c['h']:5.1f} {str(c['finite'])[0]}   "
    for f in fields:
        v, k = c[f]
        line += f"  {v}@k{k}"
    print(line)
