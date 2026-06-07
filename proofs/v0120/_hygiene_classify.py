#!/usr/bin/env python3
"""Classify per-file test results into env-failure categories for review."""
import json
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "proofs/v0120/_hygiene_baseline.jsonl"
recs = [json.loads(l) for l in open(path) if l.strip()]

PATTERNS = [
    ("GPU_REQUIRED", re.compile(r"requires a GPU device|no JAX GPU backend|requires a GPU|GPU device", re.I)),
    ("MISSING_MODULE", re.compile(r"ModuleNotFoundError|No module named|ImportError", re.I)),
    ("MISSING_DATA", re.compile(r"FileNotFoundError|No such file|does not exist|unavailable|not found|cannot access", re.I)),
    ("CALLED_PROCESS", re.compile(r"CalledProcessError|returned non-zero exit status|subprocess", re.I)),
    ("SEGV", None),
    ("TIMEOUT", None),
]


def classify(rec):
    if rec["rc"] == 139 or rec["rc"] == -11:
        return "SEGV"
    if rec["rc"] == -99:
        return "TIMEOUT"
    blob = "\n".join(rec.get("fail_lines", [])) + "\n" + rec.get("tail", "")
    for name, pat in PATTERNS:
        if pat is not None and pat.search(blob):
            return name
    return "UNKNOWN_REVIEW"


cats = {}
for r in recs:
    if r["rc"] == 0:
        c = r.get("counts", {})
        # rc==0 could still be all-passed or skipped; record
        key = "PASS_OR_SKIP_rc0"
    else:
        key = classify(r)
    cats.setdefault(key, []).append(r)

print(f"total files: {len(recs)}")
for k in sorted(cats):
    print(f"\n##### {k}: {len(cats[k])} files")
    for r in sorted(cats[k], key=lambda x: x["file"]):
        if k == "PASS_OR_SKIP_rc0":
            print(f"  rc0  {r['file']}  {r.get('counts',{})}")
        else:
            fl = (r.get("fail_lines") or ["<no fail_lines>"])[0][:140]
            print(f"  rc={r['rc']:>4} {r['file']}")
            print(f"        | {fl}")
