#!/usr/bin/env python3
"""Per-TEST outcome capture for the failing files, with the exception type/message.

Runs the failing files with a small plugin that records, for every failed test,
the repr of the exception (type + first line), so we can classify each
individual test as GPU-required / missing-data / missing-backend / real-assert.
JAX_PLATFORMS=cpu enforced by caller. No GPU context in THIS process.
"""
import json
import os
import subprocess
import sys

FAIL_FILES = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else []
OUT = sys.argv[2] if len(sys.argv) > 2 else "proofs/v0120/_hygiene_pertest.jsonl"

PLUGIN = r'''
import json, os
_OUT = os.environ["PERTEST_OUT"]
def pytest_runtest_logreport(report):
    if report.when in ("call", "setup") and report.outcome in ("failed", "error"):
        exc = ""
        if report.longrepr is not None:
            try:
                cr = report.longrepr.reprcrash
                exc = f"{cr.message}"
            except Exception:
                exc = str(report.longrepr)[:300]
        rec = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
               "exc": exc.splitlines()[0][:300] if exc else ""}
        with open(_OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
'''
plugin_path = "/tmp/_hygiene_pertest_plugin.py"
open(plugin_path, "w").write(PLUGIN)

env = dict(os.environ)
env["JAX_PLATFORMS"] = "cpu"
env["CUDA_VISIBLE_DEVICES"] = ""
env["PYTHONPATH"] = "src" + (os.pathsep + env.get("PYTHONPATH", ""))
env["PERTEST_OUT"] = OUT
env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + "/tmp"

open(OUT, "w").close()
for f in FAIL_FILES:
    subprocess.run(
        [sys.executable, "-m", "pytest", f, "-q", "--no-header",
         "-p", "no:cacheprovider", "-p", "_hygiene_pertest_plugin", "--tb=no"],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    print("done", f, flush=True)
