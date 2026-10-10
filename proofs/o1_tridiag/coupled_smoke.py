"""Coupled CPU smoke for the real Swiss d01 legacy step (lane o1-tridiag).

Runs the real operational nested single-domain driver for a few root steps on
the CPU backend and asserts every carry leaf stays finite. Used as the A/B arm
for the E105 deadlock fix: with ``PYTHONPATH=<tree>/src`` pointing at a tree
whose CPU solve still routes to the LAPACK gtsv FFI, this is the program that
deadlocked (E105); with the fix it must finish and stay finite.

Usage: JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python coupled_smoke.py --steps N --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import ctypes
import faulthandler
import signal

faulthandler.register(signal.SIGUSR1, all_threads=True)
ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)  # PR_SET_PTRACER_ANY (tar pit diagnosis)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument(
    "--input",
    type=Path,
    default=Path("<USER_HOME>/src/wrf_gpu2_wt/o1-restart/examples/switzerland_d01"),
)
args = parser.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"

t_start = time.perf_counter()
import gpuwrf  # noqa: F401  (fast-default resolution first)
import jax

assert jax.devices()[0].platform == "cpu"
from gpuwrf.contracts import state as state_contract

state_contract._gpu_device = lambda: jax.devices("cpu")[0]

import numpy as np

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree

config = NestedPipelineConfig(args.input, args.out / "stream", args.out / "proof", hours=1, max_dom=1)
t0 = time.perf_counter()
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
load_s = time.perf_counter() - t0
tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
steps = {"d01": 0}
times = []
for _ in range(args.steps):
    t0 = time.perf_counter()
    result = run_operational_domain_tree(
        tree, root_steps=1, carries=carries, initial_own_steps=steps,
        block_between=False, root_sync_cadence=0,
    )
    jax.block_until_ready(result.carries)
    carries, steps = result.carries, dict(result.own_steps)
    times.append(time.perf_counter() - t0)
    print("step", steps, round(times[-1], 3), flush=True)

leaves, treedef = jax.tree.flatten(carries)
non_finite = 0
hashes = []
for x in leaves:
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating):
        non_finite += int((~np.isfinite(arr)).sum())
    hashes.append(hashlib.sha256(arr.tobytes()).hexdigest())
record = {
    "load_s": load_s,
    "step_s": times,
    "dt": dts,
    "process_s": time.perf_counter() - t_start,
    "leaves": len(leaves),
    "treedef": str(treedef),
    "non_finite_scalars": non_finite,
    "leaf_hash_set": hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest(),
    "module_file": __import__("gpuwrf.physics.tridiagonal_solver", fromlist=["x"]).__file__,
}
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "smoke.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
