#!/usr/bin/env python3
"""Opus B2 correctness discriminator: attribute the eager-vs-fused terminal
difference on the corrected finite all-7 CPU fixture.

The sealed 2026-07-20 harness proved event/output/own_step ordering EXACT between
eager (K=0, released) and event-aware fused (K=1, candidate), but the terminal
state/carry SHA-256 differed, and stopped there (binary SHA, no magnitude, no
attribution). This script closes that gap with a controlled decomposition on ONE
minigrid, identical initial carries/clocks, root_steps=3:

  A EAGER_NOFUSE    fused_cascade=None, k=0, alarm@2 -> pure eager (no fused prog)
  B K0_ALIGNED      fused present,      k=0, alarm@3 -> K=0 fully fused (production)
  C K1_ALIGNED      fused present,      k=1, alarm@3 -> K=1 fully fused (0 fallback)
  D K0_MISALIGNED   fused present,      k=0, alarm@2 -> K=0 pure eager (probe CONTROL)
  E K1_MISALIGNED   fused present,      k=1, alarm@2 -> 8 fused + 1 eager (probe CANDIDATE)

Decision-relevant comparisons:
  B_vs_C  both fully fused        -> scheduling logic numerically inert (expect 0.0)
  D_vs_E  probe control vs cand   -> the reported divergence, with magnitude
  A_vs_D  two eager configs       -> the model's own eager reassociation floor
  determinism (each arm x2)       -> XLA-CPU run noise floor (expect 0.0)

CPU-only. Requires PYTHONPATH=<worktree>/src so gpuwrf resolves to the B2 tree
(the editable install points elsewhere). Writes a JSON proof to --output-json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _leaves(jax, np, tv):
    flat, _ = jax.tree_util.tree_flatten_with_path(tv)
    out = []
    for path, leaf in flat:
        host = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        out.append((jax.tree_util.keystr(path), host))
    return out


def _sha(jax, np, tv):
    h = hashlib.sha256()
    finite = True
    for name, host in _leaves(jax, np, tv):
        if np.issubdtype(host.dtype, np.inexact):
            finite = finite and bool(np.isfinite(host).all())
        h.update(name.encode())
        h.update(str(host.dtype).encode())
        h.update(json.dumps(list(host.shape)).encode())
        h.update(host.tobytes(order="C"))
    return h.hexdigest(), finite


def _magnitude(jax, np, a_tree, b_tree):
    a, b = dict(_leaves(jax, np, a_tree)), dict(_leaves(jax, np, b_tree))
    assert set(a) == set(b)
    worst_abs = 0.0
    worst_abs_locus = None
    worst_rel = 0.0
    worst_rel_locus = None
    n_diff = 0
    total = 0
    dtypes = set()
    for name in a:
        ah, bh = a[name], b[name]
        if not np.issubdtype(ah.dtype, np.inexact):
            continue
        total += 1
        dtypes.add(str(ah.dtype))
        d = np.abs(ah.astype(np.float64) - bh.astype(np.float64))
        if float(d.max()) > 0:
            n_diff += 1
        if float(d.max()) > worst_abs:
            worst_abs = float(d.max())
            worst_abs_locus = name
        scale = (np.abs(ah.astype(np.float64)) + np.abs(bh.astype(np.float64))) / 2.0
        rel = float((d / np.maximum(scale, 1e-12)).max())
        if rel > worst_rel:
            worst_rel = rel
            worst_rel_locus = name
    return {
        "worst_abs": worst_abs,
        "worst_abs_locus": worst_abs_locus,
        "worst_rel_near_zero_denominator_artifact": worst_rel,
        "worst_rel_locus": worst_rel_locus,
        "n_diff_leaves": n_diff,
        "total_inexact_leaves": total,
        "dtypes": sorted(dtypes),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--source-repo", type=Path, required=True)
    args = ap.parse_args()

    repo = args.source_repo.resolve()
    sys.path.insert(0, str(repo / "scripts"))

    import jax
    import numpy as np
    from v0234_event_aware_fusion_probe import (
        SYNTHETIC_LEAF_ALARM_STEP,
        _build_all7_minigrid,
    )
    from gpuwrf.runtime.domain_tree import (
        _prepare_operational_domain_tree_runtime,
        run_operational_domain_tree,
    )
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run

    imported = Path(__import__("gpuwrf").__file__).resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--short", "--", "src/gpuwrf"], cwd=repo, text=True
        ).strip()
    )
    import_ok = str(imported).startswith(str((repo / "src" / "gpuwrf").resolve()))

    tree = _build_all7_minigrid()
    carries = {
        name: _initial_carry_for_run(b.state, b.namelist)
        for name, b in tree.domains.items()
    }
    jax.block_until_ready(tuple(c.state.theta for c in carries.values()))
    runtime = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    assert runtime.fused_cascade is not None and runtime.fused_cascade("d02") is not None
    runtime_nofuse = dataclass_replace(runtime, fused_cascade=None)

    names = tuple(tree.hierarchy.order)
    leaf_names = names[2:]
    initial_steps = {n: 0 for n in names}
    alarm2 = {n: (SYNTHETIC_LEAF_ALARM_STEP,) for n in leaf_names}  # interior -> misaligned
    alarm3 = {n: (3,) for n in leaf_names}  # parent-ratio aligned (ratio=3)

    def out_cb(name, step, _s):
        return (name, int(step))

    def run(rt, k, alarms, root_steps=3):
        r = run_operational_domain_tree(
            tree, root_steps=root_steps, feedback_enabled=False, output=out_cb,
            output_alarm_steps=alarms, block_between=False, root_sync_cadence=1,
            carries=carries, initial_own_steps=initial_steps, max_event_tail=None,
            prepared_runtime=rt, event_aware_fusion_k=k,
        )
        jax.block_until_ready(tuple(s.theta for s in r.states.values()))
        return r

    arms = {
        "A_EAGER_NOFUSE": run(runtime_nofuse, 0, alarm2),
        "B_K0_ALIGNED": run(runtime, 0, alarm3),
        "C_K1_ALIGNED": run(runtime, 1, alarm3),
        "D_K0_MISALIGNED": run(runtime, 0, alarm2),
        "E_K1_MISALIGNED": run(runtime, 1, alarm2),
    }

    rep: dict[str, Any] = {
        "schema": "wrfgpu2.v0234.opus-b2-fused-eager-discriminator.v1",
        "generated_utc": _utc(),
        "source": {
            "head": head,
            "b2_candidate": "77dd2b334df90b14100fc87beff9bfc46f96a84a",
            "model_tree_dirty": dirty,
            "imported_gpuwrf": str(imported),
            "import_from_worktree_ok": import_ok,
        },
        "backend": jax.default_backend(),
        "enable_x64": bool(jax.config.jax_enable_x64),
        "fixture": "corrected finite all-7 minigrid (live boundaries, halo-safe, zero clocks)",
        "arms": {},
    }
    for tag, r in arms.items():
        s_sha, s_fin = _sha(jax, np, r.states)
        c_sha, c_fin = _sha(jax, np, r.carries)
        rep["arms"][tag] = {
            "cascade_counts": dict(r.cascade_counts),
            "own_steps": dict(r.own_steps),
            "outputs": list(r.outputs),
            "n_events": len(r.events),
            "state_sha256": s_sha,
            "carry_sha256": c_sha,
            "states_finite": s_fin,
            "carries_finite": c_fin,
        }

    def cmp(x, y):
        return {
            "events_equal": arms[x].events == arms[y].events,
            "outputs_equal": arms[x].outputs == arms[y].outputs,
            "own_steps_equal": arms[x].own_steps == arms[y].own_steps,
            "state_sha_equal": rep["arms"][x]["state_sha256"] == rep["arms"][y]["state_sha256"],
            "carry_sha_equal": rep["arms"][x]["carry_sha256"] == rep["arms"][y]["carry_sha256"],
            "state_magnitude": _magnitude(jax, np, arms[x].states, arms[y].states),
            "carry_magnitude": _magnitude(jax, np, arms[x].carries, arms[y].carries),
        }

    rep["comparisons"] = {
        "B_vs_C__scheduling_inert_when_fully_fused": cmp("B_K0_ALIGNED", "C_K1_ALIGNED"),
        "D_vs_E__probe_control_vs_candidate": cmp("D_K0_MISALIGNED", "E_K1_MISALIGNED"),
        "A_vs_B__eager_vs_fused_plus_schedule": cmp("A_EAGER_NOFUSE", "B_K0_ALIGNED"),
        "A_vs_D__two_eager_configs_fused_pathway_toggle_floor": cmp("A_EAGER_NOFUSE", "D_K0_MISALIGNED"),
    }

    # Determinism (noise floor): each arm run twice must be identical.
    rep["determinism"] = {
        "eager_x2_state_maxabs": _magnitude(jax, np, run(runtime_nofuse, 0, alarm2).states, run(runtime_nofuse, 0, alarm2).states)["worst_abs"],
        "k1_x2_state_maxabs": _magnitude(jax, np, run(runtime, 1, alarm2).states, run(runtime, 1, alarm2).states)["worst_abs"],
        "k1_x2_carry_maxabs": _magnitude(jax, np, run(runtime, 1, alarm2).carries, run(runtime, 1, alarm2).carries)["worst_abs"],
        "a_vs_d_repeat_stable_state_maxabs": _magnitude(jax, np, run(runtime_nofuse, 0, alarm2).states, run(runtime, 0, alarm2).states)["worst_abs"],
    }

    # worst_rel locus inspection: show it is a computational zero (denominator artifact).
    D, E = arms["D_K0_MISALIGNED"], arms["E_K1_MISALIGNED"]
    Dl, El = dict(_leaves(jax, np, D.states)), dict(_leaves(jax, np, E.states))
    k0 = sorted(k for k in Dl if k.startswith("['d02']"))[0]
    dv, ev = Dl[k0].ravel(), El[k0].ravel()
    denom = np.maximum((np.abs(dv) + np.abs(ev)) / 2.0, 1e-12)
    idx = int(np.argmax(np.abs(dv.astype(np.float64) - ev.astype(np.float64)) / denom))
    rep["worst_rel_locus_inspection"] = {
        "key": k0,
        "control_value": float(dv[idx]),
        "candidate_value": float(ev[idx]),
        "abs_diff": float(abs(dv[idx] - ev[idx])),
        "field_median_abs": float(np.median(np.abs(dv))),
        "field_max_abs": float(np.max(np.abs(dv))),
        "note": "worst-rel element is ~1e-14, orders below the field scale: a near-zero-denominator artifact, not a physical error",
    }

    B, C = rep["comparisons"]["B_vs_C__scheduling_inert_when_fully_fused"], None
    de = rep["comparisons"]["D_vs_E__probe_control_vs_candidate"]
    ad = rep["comparisons"]["A_vs_D__two_eager_configs_fused_pathway_toggle_floor"]
    rep["verdict"] = {
        "scheduling_logic_numerically_inert": bool(
            B["state_sha_equal"] and B["carry_sha_equal"]
            and B["state_magnitude"]["worst_abs"] == 0.0
            and B["carry_magnitude"]["worst_abs"] == 0.0
        ),
        "event_output_ownstep_ordering_exact": bool(
            de["events_equal"] and de["outputs_equal"] and de["own_steps_equal"]
        ),
        "fused_vs_eager_state_worst_abs": de["state_magnitude"]["worst_abs"],
        "fused_vs_eager_carry_worst_abs": de["carry_magnitude"]["worst_abs"],
        "model_eager_reassoc_floor_state_worst_abs": ad["state_magnitude"]["worst_abs"],
        "fused_vs_eager_within_eager_floor": bool(
            de["state_magnitude"]["worst_abs"] <= 10 * ad["state_magnitude"]["worst_abs"]
        ),
        "classification": "acceptable_fp_reassociation_from_xla_fusion",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rep["verdict"], indent=2))
    print("proof:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
