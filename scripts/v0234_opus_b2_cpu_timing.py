#!/usr/bin/env python3
"""CPU-only directional timing for B2 (mirrors the probe methodology).

Warm both arms once, then `repeats` paired (K1, K3) measurements per arm; report
median K3 wall gain and paired-incremental-per-root-step gain. CPU timing on a
tiny minigrid is dispatch-overhead dominated (NOT the GPU kernel-launch/compute
win), so this is a DIRECTIONAL check only; admission-grade speed needs the
manager-gated GPU arm. Uses the corrected finite fixture (control=K0, candidate=K1),
alarm@2 (the interior-event case), matching the sealed probe.
"""
from __future__ import annotations
import json, os, statistics, sys, time
from pathlib import Path

REPO = Path(os.environ.get("B2_REPO", Path(__file__).resolve().parent.parent)).resolve()
sys.path.insert(0, str(REPO / "scripts"))
import jax
from v0234_event_aware_fusion_probe import _build_all7_minigrid, SYNTHETIC_LEAF_ALARM_STEP
from gpuwrf.runtime.domain_tree import (
    _prepare_operational_domain_tree_runtime, run_operational_domain_tree,
)
from gpuwrf.runtime.operational_mode import _initial_carry_for_run

REPEATS = int(os.environ.get("B2_TIMING_REPEATS", "7"))


def main():
    tree = _build_all7_minigrid()
    carries = {n: _initial_carry_for_run(b.state, b.namelist) for n, b in tree.domains.items()}
    jax.block_until_ready(tuple(c.state.theta for c in carries.values()))
    runtime = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    names = tuple(tree.hierarchy.order); leaf_names = names[2:]
    initial_steps = {n: 0 for n in names}
    alarms = {n: (SYNTHETIC_LEAF_ALARM_STEP,) for n in leaf_names}

    def out_cb(name, step, _s): return (name, int(step))

    def run(k, root_steps):
        t0 = time.perf_counter()
        r = run_operational_domain_tree(
            tree, root_steps=root_steps, feedback_enabled=False, output=out_cb,
            output_alarm_steps=alarms, block_between=False, root_sync_cadence=1,
            carries=carries, initial_own_steps=initial_steps, max_event_tail=None,
            prepared_runtime=runtime, event_aware_fusion_k=k)
        jax.block_until_ready(tuple(s.theta for s in r.states.values()))
        return r, time.perf_counter() - t0

    # warm both graphs (untimed)
    wc, _ = run(0, 1); run(0, 3); run(1, 1)
    cand_warm, _ = run(1, 3)
    assert cand_warm.cascade_counts == {"fused:d02": 8, "event_fallback:d02": 1}, cand_warm.cascade_counts

    rec = {"control": [], "candidate": []}
    for rep in range(REPEATS):
        order = [0, 1] if rep % 2 == 0 else [1, 0]
        for k in order:
            _, k1 = run(k, 1)
            _, k3 = run(k, 3)
            paired = (k3 - k1) / 2.0
            rec["control" if k == 0 else "candidate"].append({"k1": k1, "k3": k3, "paired": paired})

    def med(arm, key): return statistics.median(r[key] for r in rec[arm])
    k3_gain = 1.0 - med("candidate", "k3") / med("control", "k3")
    paired_gain = 1.0 - med("candidate", "paired") / med("control", "paired")
    print(json.dumps({
        "backend": jax.default_backend(),
        "repeats": REPEATS,
        "median_control_k3_s": med("control", "k3"),
        "median_candidate_k3_s": med("candidate", "k3"),
        "median_control_paired_s": med("control", "paired"),
        "median_candidate_paired_s": med("candidate", "paired"),
        "k3_gain_fraction": k3_gain,
        "paired_incremental_gain_fraction": paired_gain,
        "note": "CPU dispatch-overhead-dominated directional check; not the GPU perf proof",
    }, indent=2))


if __name__ == "__main__":
    main()
