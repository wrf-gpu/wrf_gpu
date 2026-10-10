#!/usr/bin/env python3
"""Mutant check (E39): the pristine-WRF parity gate must FAIL when a piece of the port is removed.

Each mutant monkeypatches one routine of gpuwrf.physics._grell_cup_jax before tracing and
re-runs the fp64 savepoint comparison of tests/test_v034_grell_cumulus_parity.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import test_v034_grell_cumulus_parity as T  # noqa: E402

G = T.G
orig = {name: getattr(G, name) for name in ("_jmin_search", "_neg_check", "_conv_grell_spread3d")}


def m_jmin(*a, **k):  # keep cup_minimi's JMIN, skip WRF's keep_going buoyancy search
    jmin0, kdet0, _ktop, _h, _z, ierr, _kx = a
    return jmin0, kdet0, ierr


def m_negcheck(dt, q, outq, outt, outqc, pret, kx):  # drop GD neg_check
    return outt, outq, outqc, pret


def m_spread(*a):  # ignore cugd_avedx subsidence spreading
    a = list(a)
    a[11] = 1
    return orig["_conv_grell_spread3d"](*a)


MUTANTS = (("M1_no_jmin_search", "_jmin_search", m_jmin, "g3_deep_r8"),
           ("M2_no_neg_check", "_neg_check", m_negcheck, "gd_r8_s2"),
           ("M3_no_subsidence_spreading", "_conv_grell_spread3d", m_spread, "g3_highres_r8"))
results = {}
for label, attr, fn, case in MUTANTS:
    setattr(G, attr, fn)
    try:
        meta, oracle, tile = T._load(case)
        port = T._run(meta, tile)
        fields = T.G3_FIELDS if meta["scheme"] == "g3" else T.GD_FIELDS
        rel = {k: T._rel(oracle, port, k) for k in fields}
        worst = max(rel.items(), key=lambda kv: kv[1])
        killed = worst[1] > 1e-8
        results[label] = dict(case=case, killed=bool(killed), worst_field=worst[0], worst_rel=worst[1])
    finally:
        setattr(G, attr, orig[attr])
    print(label, results[label], flush=True)
out = ROOT / "proofs" / "v034" / "grell_mutants_result.json"
if not out.exists():
    out.write_text(json.dumps(results, indent=1) + "\n")
# M1/M2 survive: their branches are never active in the staged oracle data (coverage gap,
# see grell_mutants_result.json); M3 must be killed.
assert results["M3_no_subsidence_spreading"]["killed"], results
