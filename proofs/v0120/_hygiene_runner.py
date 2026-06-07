#!/usr/bin/env python3
"""Per-file isolated CPU-only test runner for test-hygiene baseline/after.

Runs every tests/**/test_*.py file in its OWN subprocess with JAX_PLATFORMS=cpu,
captures rc + pass/fail/skip/xfail/error counts + a short failure summary.
Writes a JSON line per file to the output path.
NEVER initializes a GPU JAX context in THIS process (no import jax here).
"""
import json
import os
import re
import subprocess
import sys
import glob

OUT = sys.argv[1] if len(sys.argv) > 1 else "proofs/v0120/_hygiene_results.jsonl"

files = sorted(
    glob.glob("tests/**/test_*.py", recursive=True)
    + glob.glob("evals/agentos/**/test_*.py", recursive=True)
)

env = dict(os.environ)
env["JAX_PLATFORMS"] = "cpu"
env["CUDA_VISIBLE_DEVICES"] = ""  # belt-and-suspenders: no GPU visible
env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
env.setdefault("OMP_NUM_THREADS", "2")

SUMMARY_RE = re.compile(
    r"(?P<n>\d+)\s+(?P<kind>passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?)"
)


def parse_counts(text):
    counts = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("=") and (
            "passed" in s or "failed" in s or "error" in s
            or "skipped" in s or "no tests ran" in s
        ):
            block = {}
            for m in SUMMARY_RE.finditer(s):
                k = m.group("kind")
                k = "error" if k.startswith("error") else ("warning" if k.startswith("warning") else k)
                block[k] = int(m.group("n"))
            if block:
                counts = block
    return counts


results = []
with open(OUT, "w") as fh:
    for f in files:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", f, "-q", "--no-header",
                 "-p", "no:cacheprovider", "--tb=line", "-rfEsxX"],
                env=env, capture_output=True, text=True, timeout=1800,
            )
            rc = proc.returncode
            out = proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired as e:
            rc = -99
            out = "TIMEOUT\n" + (e.stdout or "") + "\n" + (e.stderr or "")
        counts = parse_counts(out)
        fail_lines = [l for l in out.splitlines()
                      if re.search(r"(FAILED|ERROR) ", l) or l.strip().startswith("E   ")][:25]
        skip_lines = [l for l in out.splitlines()
                      if re.search(r"(SKIPPED|XFAIL|XPASS) ", l)][:25]
        rec = {
            "file": f,
            "rc": rc,
            "counts": counts,
            "fail_lines": fail_lines,
            "skip_lines": skip_lines,
            "tail": "\n".join(out.splitlines()[-25:]),
        }
        results.append(rec)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"{rc:>4}  {f}  {counts}", flush=True)

agg = {"total_files": len(results)}
by_rc = {}
for r in results:
    by_rc[str(r["rc"])] = by_rc.get(str(r["rc"]), 0) + 1
agg["by_rc"] = by_rc
print("\n=== AGGREGATE ===")
print(json.dumps(agg, indent=2))
