"""Force-down anatomy + fused-variant A/B on DEVICE-resident real operands.

Cases PROD d01->d02 and WN3 d02->d03 (B36 records seeded).  Variants:
``unfused`` = single fusion-disabled jit (main default), ``fused`` =
GPUWRF_FORCEDOWN_FUSED=1 (fusion ON, c1*mu products behind barriers).  Every
output leaf of ``fused`` is byte-compared with ``unfused``; per variant: median
wall with block (n=20, interleaved) and host time to return; one
cudaProfilerApi window of 5 force-downs per (case, variant) for nsys.
"""
import argparse
import ctypes
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
sys.path[:0] = [str(args.source / "src")]

import jax

assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
assert jax.devices()[0].platform == "gpu", jax.devices()

from gpuwrf.nesting.boundary_construction import (
    build_child_boundary_package, build_nest_force_weights, initialize_child_scalar_boundaries,
)
from gpuwrf.runtime.domain_tree import DomainTree
from gpuwrf.runtime.operational_mode import _initial_carry_for_run

dev = jax.devices()[0]
put = lambda tree: jax.block_until_ready(jax.device_put(tree, dev))
cases = []
with Path("<USER_HOME>/wrf_gpu2_lanes/integrate/diagnose/common_inputs.pkl").open("rb") as fh:
    h, b, _, _, _, _ = pickle.load(fh)
edge = DomainTree.from_domains(h, b).edges["d01"][0]
cases.append(dict(name="PROD-d01-d02",
                  parent=put(_initial_carry_for_run(b["d01"].state, b["d01"].namelist).state),
                  child=put(initialize_child_scalar_boundaries(_initial_carry_for_run(b["d02"].state, b["d02"].namelist).state)),
                  pm=b["d01"].namelist.metrics, cm=b["d02"].namelist.metrics, weights=edge.weights))
rain = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/W3/ni_probe/rain01")
pl = [pickle.load((rain / f"d0{n}_first_exceed.pkl").open("rb")) for n in (2, 3)]
cases.append(dict(name="WN3-d02-d03",
                  parent=put(_initial_carry_for_run(pl[0]["before"].state, pl[0]["namelist"]).state),
                  child=put(initialize_child_scalar_boundaries(_initial_carry_for_run(pl[1]["before"].state, pl[1]["namelist"]).state)),
                  pm=put(pl[0]["namelist"].metrics), cm=put(pl[1]["namelist"].metrics),
                  weights=put(build_nest_force_weights(parent_grid_ratio=3, i_parent_start=92, j_parent_start=36,
                                                       parent_grid=pl[0]["grid"], child_grid=pl[1]["grid"]))))

cudart = ctypes.CDLL("libcudart.so")
report = dict(device=str(dev), affinity=sorted(os.sched_getaffinity(0)), cases=[])
for case in cases:
    w = int(case["child"].u_bdy.shape[2])
    def call(variant):
        os.environ["GPUWRF_FORCEDOWN_FUSED"] = "1" if variant == "fused" else "0"
        return build_child_boundary_package(
            case["child"], case["parent"], case["weights"], bdy_width=w, parent_metrics=case["pm"],
            child_metrics=case["cm"], coupled_forcedown=True, parent_grid_ratio=3, _compiled_producers=True)
    outs = {v: jax.block_until_ready(call(v)) for v in ("unfused", "fused")}
    for _ in range(2):
        for v in ("unfused", "fused"):
            jax.block_until_ready(call(v))
    ref = jax.tree_util.tree_flatten_with_path(outs["unfused"])[0]
    new = jax.tree_util.tree_flatten_with_path(outs["fused"])[0]
    changed = []
    for (path, a), (_, b2) in zip(ref, new):
        a, b2 = np.asarray(a), np.asarray(b2)
        if a.dtype != b2.dtype or a.tobytes() != b2.tobytes():
            changed.append(dict(leaf=jax.tree_util.keystr(path), n_diff=int(np.count_nonzero(a != b2)),
                                max_abs=float(np.max(np.abs(a.astype(np.float64) - b2.astype(np.float64))))))
    timing = {v: dict(total=[], host=[]) for v in ("unfused", "fused")}
    for i in range(20):
        for v in (("unfused", "fused") if i % 2 == 0 else ("fused", "unfused")):
            t0 = time.perf_counter()
            out = call(v)
            t1 = time.perf_counter()
            jax.block_until_ready(out)
            t2 = time.perf_counter()
            timing[v]["host"].append(1e3 * (t1 - t0))
            timing[v]["total"].append(1e3 * (t2 - t0))
    for v in ("unfused", "fused"):
        cudart.cudaProfilerStart()
        for _ in range(5):
            jax.block_until_ready(call(v))
        cudart.cudaProfilerStop()
    row = dict(name=case["name"], fused_bytes_equal=not changed, changed=changed,
               **{f"{v}_{k}_median_ms": statistics.median(timing[v][k]) for v in timing for k in ("total", "host")},
               timing=timing)
    report["cases"].append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "timing"}), flush=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
