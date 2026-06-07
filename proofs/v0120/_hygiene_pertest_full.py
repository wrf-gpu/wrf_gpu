#!/usr/bin/env python3
"""Capture FULL per-test outcomes (passed/skipped/failed/error) for a file list.

Used to prove no previously-passing test became skipped after marking.
JAX_PLATFORMS=cpu enforced by caller; no GPU context in THIS process.
"""
import json
import os
import subprocess
import sys

FILES = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

PLUGIN = r'''
import json, os
_OUT = os.environ["PERTEST_OUT"]
def pytest_runtest_logreport(report):
    # record the decisive phase: call for passed/failed, setup for skip/error-at-setup
    if report.when == "call" or (report.when == "setup" and report.outcome in ("skipped","failed","error")):
        rec = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
        with open(_OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
'''
open("/tmp/_hygiene_full_plugin.py", "w").write(PLUGIN)

env = dict(os.environ)
env["JAX_PLATFORMS"] = "cpu"
env["CUDA_VISIBLE_DEVICES"] = ""
env["PERTEST_OUT"] = OUT
env["PYTHONPATH"] = "src" + os.pathsep + "/tmp" + os.pathsep + env.get("PYTHONPATH", "")

open(OUT, "w").close()
for f in FILES:
    subprocess.run(
        [sys.executable, "-m", "pytest", f, "-q", "--no-header",
         "-p", "no:cacheprovider", "-p", "_hygiene_full_plugin", "--tb=no"],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    print("done", f, flush=True)
