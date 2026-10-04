"""Direct fresh-process GPU P1 probe (never run through pytest/conftest).

Uses real PROD input loading, normal nested scheduler and runtime executables.
The controller externally SIGKILLs the interrupted action after READY_TO_KILL.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

started = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument("--action", choices=("setup", "control", "interrupted", "resume"), required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--cut", type=int, default=1)
args = parser.parse_args()

import gpuwrf
import jax
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime.domain_tree import DomainTree, _prepare_operational_domain_tree_runtime, run_operational_domain_tree, nested_aot_report
from gpuwrf.runtime.operational_mode import _commit_to_operational_device
from gpuwrf.runtime.restart_store import RestartStore, nested_identity

assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
assert jax.devices()[0].platform == "gpu"
assert 0 < args.cut < args.steps
args.root.mkdir(parents=True, exist_ok=True)
out = args.root / args.action
out.mkdir(parents=True, exist_ok=True)
config = NestedPipelineConfig(
    Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725"), args.root / "same-stream", args.root / "proof",
    hours=3, max_dom=2,
)
hierarchy, bundles, metadata, run_start, dts, carries = _load_domains(config, ("d01", "d02"))
identity = nested_identity(config, bundles, run_start)
identity["short_horizon_root_steps"] = args.steps
store = RestartStore(args.root / "generations", 8 * 1024**3, 2, 10 * 1024**3)
steps = {name: 0 for name in carries}
resume_read_s = None
if args.action == "resume":
    t0 = time.perf_counter()
    generation = store.latest(expected_identity=identity)
    snapshot, receipt = store.read(generation, expected_identity=identity)
    carries = {name: _commit_to_operational_device(value) for name, value in snapshot["carries"].items()}
    steps = snapshot["own_steps"]
    resume_read_s = time.perf_counter() - t0
tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
runtime = replace(_prepare_operational_domain_tree_runtime(tree, feedback_enabled=False), fused_cascade=None)
initialization_s = time.perf_counter() - started
segment_times = []
checkpoint = None
while steps["d01"] < args.steps:
    t0 = time.perf_counter()
    result = run_operational_domain_tree(
        tree, root_steps=1, carries=carries, initial_own_steps=steps,
        prepared_runtime=runtime, block_between=False, root_sync_cadence=0,
    )
    jax.block_until_ready(result.carries)
    carries, steps = result.carries, result.own_steps
    segment_times.append(time.perf_counter() - t0)
    if args.action == "interrupted" and steps["d01"] == args.cut:
        generation, storage = store.save(carries, steps, dts, identity)
        checkpoint = {"generation": str(generation), **storage}
        record = dict(action=args.action, pid=os.getpid(), backend=jax.devices()[0].platform,
                      initialization_s=initialization_s, segment_wall_s=segment_times,
                      own_steps=steps, checkpoint=checkpoint, aot=nested_aot_report())
        (out / "result.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        (out / "READY_TO_KILL").write_text(str(os.getpid()) + "\n")
        print("READY_TO_KILL", os.getpid(), generation, flush=True)
        # The external controller must kill this process, not a simulated return.
        while True:
            time.sleep(1)
leaves = {}
for name, carry in carries.items():
    for path, value in jax.tree_util.tree_flatten_with_path(carry)[0]:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.inexact):
            assert np.isfinite(array).all(), name
        key = name + "/" + jax.tree_util.keystr(path)
        leaves[key] = dict(dtype=str(array.dtype), shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())
record = dict(action=args.action, pid=os.getpid(), backend=jax.devices()[0].platform,
              device=jax.devices()[0].device_kind, initialization_s=initialization_s,
              resume_read_s=resume_read_s, resume_to_first_segment_ready_s=(initialization_s + segment_times[0]) if args.action == "resume" else None,
              segment_wall_s=segment_times, total_s=time.perf_counter()-started,
              own_steps=steps, leaves=leaves, aot=nested_aot_report(), identity=identity,
              scope="real PROD d01+d02 dt54/18,3 root steps,normal live scheduler,all carry/accumulator bytes; no wrfout writer in short probe")
record["census"] = {name: None if carry.census is None else {"guards": np.asarray(carry.census.guards).tolist(), "work": np.asarray(carry.census.work).tolist()} for name, carry in carries.items()}
(out / "result.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
print(json.dumps({key: record[key] for key in ("action", "backend", "own_steps", "total_s", "initialization_s", "resume_read_s")}), flush=True)
