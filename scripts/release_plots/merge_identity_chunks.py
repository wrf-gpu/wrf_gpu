#!/usr/bin/env python3
"""Extract ten plotted fields from consecutive original CPU-WRF scorer chunks.

This is a plot input, not a replacement D6 gate report. Read one chunk at a time
to bound memory; retain original per-lead numbers and source report hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIELDS = ("T2", "U10", "V10", "PSFC", "RAINNC", "T", "U", "V", "W", "QVAPOR")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", action="append", required=True, type=Path)
    parser.add_argument("--min-lead", type=int, required=True)
    parser.add_argument("--max-lead", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merged = {"schema": "release-docs.identity-chunks.v1",
              "scope": "ten plotted fields only; original reports govern the D6 gate",
              "source_reports": [], "field_summaries": {}}
    inputs = None
    for path in args.chunk:
        raw = path.read_bytes()
        report = json.loads(raw)
        source_inputs = report["inputs"]
        identity = {k: source_inputs[k] for k in ("cpu_dir", "gpu_dir", "domain")}
        if inputs is not None and identity != inputs:
            raise ValueError("chunk comparison inputs differ")
        inputs = identity
        merged["inputs"] = source_inputs
        merged["source_reports"].append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for field in FIELDS:
            data = report["field_summaries"][field]
            if data.get("missing_leads") or data.get("incompatible_leads"):
                raise ValueError(f"{path}: incomplete {field}")
            dest = merged["field_summaries"].setdefault(field, {"by_lead": []})
            dest["by_lead"].extend(data["by_lead"])
    expected = list(range(args.min_lead, args.max_lead + 1))
    for field, data in merged["field_summaries"].items():
        data["by_lead"].sort(key=lambda row: row["lead_h"])
        if [r["lead_h"] for r in data["by_lead"]] != expected:
            raise ValueError(f"{field}: missing, repeated or unexpected leads")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n")
    print(args.out, len(FIELDS), "fields", len(expected), "hourly leads")


if __name__ == "__main__":
    main()
