#!/usr/bin/env python3
"""Compare BEFORE vs AFTER per-test outcomes on the touched files.

Proves:
  (1) no nodeid that PASSED before is now skipped/failed/errored (no over-masking,
      no newly-broken passers);
  (2) every nodeid that FAILED/ERRORED before is now passed or SKIPPED (env masked)
      or still failed (intentionally-left real failure).

Prints any violations and a transition matrix.
"""
import json
import sys
from collections import defaultdict

before = {}
for l in open("proofs/v0120/_hygiene_before_pertest.jsonl"):
    r = json.loads(l)
    before[r["nodeid"]] = r["outcome"]
after = {}
for l in open("proofs/v0120/_hygiene_after_pertest.jsonl"):
    r = json.loads(l)
    after[r["nodeid"]] = r["outcome"]

# transition matrix
trans = defaultdict(int)
violations = []  # passed -> not-passed
new_only_after = []
for nid, bo in before.items():
    ao = after.get(nid, "<absent-after>")
    trans[(bo, ao)] += 1
    if bo == "passed" and ao != "passed":
        violations.append((nid, bo, ao))
# nodeids present after but not before (e.g. previously crashed-before-report in SEGV file)
for nid, ao in after.items():
    if nid not in before:
        new_only_after.append((nid, ao))

print("=== BEFORE->AFTER transition matrix (touched files) ===")
for (bo, ao), n in sorted(trans.items(), key=lambda x: (-x[1])):
    print(f"  {n:>4}  {bo:>8} -> {ao}")

print(f"\nbefore nodeids: {len(before)} | after nodeids: {len(after)}")

print(f"\n=== VIOLATIONS (passed-before now NOT passed): {len(violations)} ===")
for nid, bo, ao in violations:
    print(f"  !! {bo} -> {ao}  {nid}")

print(f"\n=== nodeids only-in-after (e.g. SEGV file now reports): {len(new_only_after)} ===")
for nid, ao in sorted(new_only_after):
    print(f"  +  {ao:>8}  {nid}")
