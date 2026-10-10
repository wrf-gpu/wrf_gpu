"""Real-case CPU restart proof: Swiss 42x42 single domain (no GPU, never via pytest).

Actions (each a fresh process):
  control      run N root steps in ONE segment, record every carry-leaf hash
  interrupted  run K steps, publish a verified RestartStore generation, SIGKILL itself
  resume       fresh process: read latest verified generation, run N-K steps, compare to control

Usage: JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python cpu_swiss_restart.py --action A --root DIR [--steps N --cut K]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

t_start = time.perf_counter()
import ctypes, faulthandler
faulthandler.register(signal.SIGUSR1, all_threads=True)
ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)  # PR_SET_PTRACER_ANY: allow py-spy dump (ptrace_scope=1)
parser = argparse.ArgumentParser()
parser.add_argument("--action", choices=("control", "interrupted", "resume"), required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--input", type=Path, default=Path("<USER_HOME>/src/wrf_gpu2_wt/o1-restart/examples/switzerland_d01"))
parser.add_argument("--steps", type=int, default=36)
parser.add_argument("--cut", type=int, default=20)
parser.add_argument("--interpret", action="store_true", help="Pallas interpret mode (fast defaults on CPU)")
args = parser.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert 0 < args.cut < args.steps

import gpuwrf  # noqa: F401  (fast-default resolution first)
import jax

jax.config.update("jax_cpu_enable_async_dispatch", False)
assert jax.devices()[0].platform == "cpu"
from gpuwrf.contracts import state as state_contract

state_contract._gpu_device = lambda: jax.devices("cpu")[0]
if args.interpret:
    from jax.experimental import pallas as pl

    _orig = pl.pallas_call
    pl.pallas_call = lambda *a, **k: _orig(*a, **{**k, "interpret": True})

import numpy as np
import jax.numpy as jnp
from jax import lax


def _thomas_tridiagonal_solve(dl, d, du, b):
    """Pure-JAX Thomas solve, same contract as jax.lax.linalg.tridiagonal_solve.

    CPU HARNESS ONLY: jaxlib's LAPACK dgtsv FFI deadlocks the affinity-sized XLA:CPU
    pool (ParallelBatchMap waits on its own pool, E105). Identical in every arm.
    """
    dl_t, d_t, du_t = (jnp.moveaxis(x, -1, 0)[..., None] for x in (dl, d, du))
    b_t = jnp.moveaxis(b, -2, 0)

    def forward(carry, row):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, r_i = row
        denom = b_i - a_i * cp_prev
        cp = c_i / denom
        dp = (r_i - a_i * dp_prev) / denom
        return (cp, dp), (cp, dp)

    init = (jnp.zeros_like(d_t[0]), jnp.zeros_like(b_t[0]))
    _, (cp, dp) = lax.scan(forward, init, (dl_t, d_t, du_t, b_t))

    def backward(x_next, row):
        cp_i, dp_i = row
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, xs = lax.scan(backward, jnp.zeros_like(b_t[0]), (cp, dp), reverse=True)
    return jnp.moveaxis(xs, 0, -2)


jax.lax.linalg.tridiagonal_solve = _thomas_tridiagonal_solve

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree
from gpuwrf.runtime.operational_mode import _commit_to_operational_device
from gpuwrf.runtime.restart_store import RestartStore, nested_identity

out = args.root / args.action
out.mkdir(parents=True, exist_ok=True)
config = NestedPipelineConfig(args.input, args.root / "stream", args.root / "proof", hours=1, max_dom=1,
                              emit_initial_history=True)
t0 = time.perf_counter()
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
load_s = time.perf_counter() - t0
identity = nested_identity(config, bundles, run_start)
identity["short_horizon_root_steps"] = args.steps
store = RestartStore(args.root / "generations", 8 * 1024**3, 2, 10 * 1024**3)
steps = {"d01": 0}
resume = {}
if args.action == "resume":
    t0 = time.perf_counter()
    generation = store.latest(expected_identity=identity)
    snapshot, receipt = store.read(generation, expected_identity=identity)
    carries = {name: _commit_to_operational_device(value) for name, value in snapshot["carries"].items()}
    steps = dict(snapshot["own_steps"])
    resume = {"generation": str(generation), "read_s": time.perf_counter() - t0, "own_steps": steps}
tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)


def leaf_hashes(tree_):
    leaves, treedef = jax.tree.flatten(tree_)
    return {"treedef": str(treedef), "leaves": [
        [str(np.asarray(x).dtype), list(np.asarray(x).shape), hashlib.sha256(np.asarray(x).tobytes()).hexdigest()]
        for x in leaves]}


def advance(n):
    global carries, steps
    t0 = time.perf_counter()
    result = run_operational_domain_tree(tree, root_steps=n, carries=carries, initial_own_steps=steps,
                                         block_between=False, root_sync_cadence=0)
    jax.block_until_ready(result.carries)
    carries, steps = result.carries, dict(result.own_steps)
    return time.perf_counter() - t0


segments = []
if args.action == "control":
    segments.append(advance(args.steps))
elif args.action == "interrupted":
    segments.append(advance(args.cut))
    t0 = time.perf_counter()
    generation, storage = store.save(carries, steps, dts, identity, driver_state={"forecast_hours": 1})
    record = {"action": "interrupted", "pid": os.getpid(), "own_steps": steps, "generation": str(generation),
              "save_s": time.perf_counter() - t0, "storage": storage, "load_s": load_s, "segment_s": segments,
              "process_s": time.perf_counter() - t_start, "hashes_at_cut": leaf_hashes(carries)}
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print("SIGKILL self after verified generation", generation, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
else:
    segments.append(advance(args.steps - args.cut))

hashes = leaf_hashes(carries)
finite = all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves(carries)
             if np.asarray(x).dtype.kind in "fc")
record = {"action": args.action, "pid": os.getpid(), "own_steps": steps, "load_s": load_s, "segment_s": segments,
          "process_s": time.perf_counter() - t_start, "resume": resume, "finite": finite,
          "census_guards": np.asarray(jax.device_get(carries["d01"].census.guards)).tolist()
          if getattr(carries["d01"], "census", None) is not None else None,
          "backend": jax.devices()[0].platform, "interpret": args.interpret,
          "xla_flags": os.environ.get("XLA_FLAGS"), "hashes": hashes}
if args.action == "resume":
    control = json.loads((args.root / "control" / "result.json").read_text())["hashes"]
    a, b = control["leaves"], hashes["leaves"]
    record["compare"] = {"treedef_equal": control["treedef"] == hashes["treedef"], "leaves": len(a),
                         "changed": [i for i, (x, y) in enumerate(zip(a, b)) if x != y],
                         "byte_exact": control == hashes}
    print("COMPARE", json.dumps(record["compare"]), flush=True)
(out / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "hashes"}, default=str)[:2000], flush=True)
