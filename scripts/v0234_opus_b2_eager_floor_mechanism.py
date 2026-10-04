#!/usr/bin/env python3
"""Isolate the model's own eager-vs-eager reassociation floor on the B2 fixture.

Tests three axes that could produce eager reassociation, to find which is the
reliable nonzero self-calibrating floor for the bounded terminal criterion:
  - FUSED-PATHWAY TOGGLE : eager(fused_cascade=None) vs eager(fused_cascade=present)
  - RE-COMPILE           : fused program compiled in-process vs not (both eager)
  - RE-CHUNK             : leaf advance chunked 2+1 vs 1+2 (both eager)

Finding on the corrected finite all-7 fixture (fp64): re-compile and re-chunk are
BIT-IDENTICAL (0.0); the fused-pathway toggle is the reliable ~1e-10 floor. CPU-only.
"""
from __future__ import annotations
import json, os, sys
from dataclasses import replace as dataclass_replace
from pathlib import Path

REPO = Path(os.environ.get("B2_REPO", Path(__file__).resolve().parent.parent)).resolve()
sys.path.insert(0, str(REPO / "scripts"))
import jax
import numpy as np
from v0234_event_aware_fusion_probe import _build_all7_minigrid, _tree_maxabs
from gpuwrf.runtime.domain_tree import _prepare_operational_domain_tree_runtime, run_operational_domain_tree
from gpuwrf.runtime.operational_mode import _initial_carry_for_run


def main():
    tree = _build_all7_minigrid()
    carries = {n: _initial_carry_for_run(b.state, b.namelist) for n, b in tree.domains.items()}
    names = tuple(tree.hierarchy.order)
    init = {n: 0 for n in names}
    al2 = {n: (2,) for n in names[2:]}
    al1 = {n: (1,) for n in names[2:]}

    def oc(n, s, _):
        return (n, int(s))

    def run(rt, k, al):
        r = run_operational_domain_tree(
            tree, root_steps=3, feedback_enabled=False, output=oc, output_alarm_steps=al,
            block_between=False, root_sync_cadence=1, carries=carries, initial_own_steps=init,
            max_event_tail=None, prepared_runtime=rt, event_aware_fusion_k=k)
        jax.block_until_ready(tuple(s.theta for s in r.states.values()))
        return r

    rt_none = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    a = run(dataclass_replace(rt_none, fused_cascade=None), 0, al2)      # fused pathway OFF
    rt_b = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    b = run(rt_b, 0, al2)                                                # present, uncompiled
    rt_c = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    rt_c.fused_cascade("d02")
    c = run(rt_c, 0, al2)                                                # present, compiled
    rechunk = run(rt_c, 0, al1)                                          # eager, 1+2 chunking

    print(json.dumps({
        "toggle__none_vs_present_state": _tree_maxabs(jax, np, a.states, b.states),
        "toggle__none_vs_present_carry": _tree_maxabs(jax, np, a.carries, b.carries),
        "recompile__uncompiled_vs_compiled_state": _tree_maxabs(jax, np, b.states, c.states),
        "rechunk__alarm2_vs_alarm1_state": _tree_maxabs(jax, np, c.states, rechunk.states),
    }, indent=2))


if __name__ == "__main__":
    main()
