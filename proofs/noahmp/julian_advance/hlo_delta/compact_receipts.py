"""Compact the six hlo_delta.py receipts (arm x case) into receipts.json: revision, flag, per-program raw lowered HLO
sha256/bytes, lower seconds and AOT cheap-key components (raw .hlo.gz stay in the lane dir)."""
import json, sys
from pathlib import Path
J = Path(sys.argv[1]); out = {}
for arm in ("main_off", "br_off", "br_on"):
    for case in ("prod", "wn3"):
        r = json.load(open(J / f"{arm}_{case}/receipt.json"))
        out[f"{arm}_{case}"] = {"revision": r["revision"], "julian_flag": r["julian_flag"], "case": r["case"],
                                "scope": r["scope"], "programs": {k: {f: v[f] for f in ("raw_lower_hlo_sha256",
                                "raw_lower_hlo_bytes", "lower_s", "components")} for k, v in r["domains"].items()}}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
