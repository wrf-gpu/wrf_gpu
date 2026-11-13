"""Run one or more `gpuwrf` CLI invocations in ONE process with JAX compile accounting (listener only).

usage: jax_acct_run.py --receipt OUT.json -- <gpuwrf args>            (one run)
       jax_acct_run.py --receipt OUT.json --argv-json RUNS.json        (list of argv lists, in-process series)

Counts per run: persistent-cache requests/hits/misses, cache-disabled compiles, jaxpr traces, MLIR lowerings and
backend compile events. ``backend_compile_duration`` fires for every compile-or-load, hit or not, so
``fresh_compiles = backend_compile_events - cache_hits`` also catches compiles that never issued a cache request
(cache disabled for that program), which ``cache_misses = requests - hits`` cannot see.
The platform is read AFTER the runs: touching ``jax.devices()`` first would initialize the CUDA backend before the
CLI's nested-allocator re-exec (it re-execs ``sys.orig_argv`` once on CUDA) and its VRAM preflight.
"""
import argparse
from collections import Counter
import json
import os
import resource
import sys
import time

T0 = time.perf_counter()
p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument("--receipt", required=True)
p.add_argument("--argv-json")
a, rest = p.parse_known_args()
if rest[:1] == ["--"]:
    rest = rest[1:]
runs = json.load(open(a.argv_json)) if a.argv_json else [rest]

import jax  # noqa: E402  (gpuwrf import also imports jax; no backend init here)
from jax import monitoring  # noqa: E402

current = Counter()
EVENTS = {"/jax/compilation_cache/compile_requests_use_cache": "cache_requests",
          "/jax/compilation_cache/cache_hits": "cache_hits",
          "/jax/compilation_cache/task_disabled_cache": "cache_disabled"}
DURATIONS = {"/jax/core/compile/jaxpr_trace_duration": "traces",
             "/jax/core/compile/jaxpr_to_mlir_module_duration": "lowerings",
             "/jax/core/compile/backend_compile_duration": "backend_compile_events"}
secs = Counter()


def on_event(name, **_):
    if name in EVENTS:
        current[EVENTS[name]] += 1


def on_duration(name, value, **_):
    if name in DURATIONS:
        current[DURATIONS[name]] += 1
        secs[DURATIONS[name] + "_s"] += float(value)


monitoring.register_event_listener(on_event)
monitoring.register_event_duration_secs_listener(on_duration)
import gpuwrf.cli as cli  # noqa: E402

receipt = {"schema": "gpuwrf.jax_acct_run.v1", "platform": None, "import_s": time.perf_counter() - T0,
           "pid": os.getpid(), "reexec": os.environ.get("_GPUWRF_NESTED_ALLOC_REEXEC") == "1", "runs": []}
for argv in runs:
    current.clear()
    secs.clear()
    t = time.perf_counter()
    rc = cli.main(argv)
    row = {"argv": argv, "rc": rc, "wall_s": time.perf_counter() - t, **{k: 0 for k in (*EVENTS.values(), *DURATIONS.values())},
           **dict(current), **{k: round(v, 3) for k, v in secs.items()}}
    row["cache_misses"] = row["cache_requests"] - row["cache_hits"]
    row["fresh_compiles"] = row["backend_compile_events"] - row["cache_hits"]
    receipt["runs"].append(row)
    print("[acct] " + json.dumps({k: v for k, v in row.items() if k != "argv"}), file=sys.stderr, flush=True)
    if rc != 0:
        break
receipt["platform"] = jax.devices()[0].platform
receipt["devices"] = [str(d) for d in jax.devices()]
receipt["maxrss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
receipt["total_s"] = time.perf_counter() - T0
json.dump(receipt, open(a.receipt, "w"), indent=1)
os._exit(int(max((r["rc"] for r in receipt["runs"]), default=1)))
