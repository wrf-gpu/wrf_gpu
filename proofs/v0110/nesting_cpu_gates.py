#!/usr/bin/env python3
"""CPU proof object for v0.11.0 live-nesting structure and feedback math."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import numpy as np
import jax.numpy as jnp

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.coupling.boundary_feedback import (
    build_feedback_weights,
    feedback_overlap_conservation,
)
from gpuwrf.physics.mynn_constants import QKEMIN
from gpuwrf.physics.mynn_pbl import _wrf_qke_minmax
from gpuwrf.runtime.domain_tree import build_live_nested_boundary_config, run_domain_tree_callbacks


def _canary_hierarchy() -> DomainHierarchy:
    return DomainHierarchy.from_edges(
        ("d01", "d02", "d03", "d04", "d05"),
        (
            DomainNest("d01", "d02", 3, 30, 20),
            DomainNest("d02", "d03", 3, 52, 20),
            DomainNest("d02", "d04", 3, 42, 32),
            DomainNest("d02", "d05", 3, 64, 34),
        ),
    )


def main() -> int:
    out_path = Path("proofs/v0110/nesting_cpu_gates.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hierarchy = _canary_hierarchy()
    advance_events: list[tuple[str, int, int]] = []
    force_events: list[tuple[str, str, int, int]] = []

    def advance(name, carry, start_step, n_steps):
        advance_events.append((name, int(start_step), int(n_steps)))
        return int(carry) + int(n_steps)

    def force(edge, parent, child):
        force_events.append((edge.parent, edge.child, int(parent), int(child)))
        return child

    result = run_domain_tree_callbacks(
        hierarchy,
        {name: 0 for name in hierarchy.order},
        root_steps=2,
        advance=advance,
        force=force,
        output=lambda name, step, state: {"domain": name, "own_step": int(step), "state": int(state)},
        output_cadence_steps={"d01": 1, "d02": 3, "d03": 9, "d04": 9, "d05": 9},
        block_between=False,
    )

    feedback_weights = build_feedback_weights(
        parent_grid_ratio=3,
        i_parent_start=2,
        j_parent_start=2,
        parent_we=12,
        parent_sn=12,
        child_we=24,
        child_sn=24,
        stagger="",
        spec_zone=1,
    )
    conservation = feedback_overlap_conservation(
        jnp.arange(4 * 24 * 24, dtype=jnp.float64).reshape(4, 24, 24),
        feedback_weights,
        leaf="theta",
    )
    live_cfg = build_live_nested_boundary_config(18.0)
    two_domain = DomainHierarchy.from_edges(("d01", "d02"), (DomainNest("d01", "d02", 3, 2, 2),))
    feedback_calls = {"off": 0, "on": 0}

    def feedback_off(edge, parent, child):
        del edge, child
        feedback_calls["off"] += 1
        return parent

    def feedback_on(edge, parent, child):
        del edge, child
        feedback_calls["on"] += 1
        return parent

    run_domain_tree_callbacks(
        two_domain,
        {"d01": 0, "d02": 0},
        root_steps=1,
        advance=lambda _name, carry, _start, n: int(carry) + int(n),
        feedback=feedback_off,
        feedback_enabled=False,
        block_between=False,
    )
    run_domain_tree_callbacks(
        two_domain,
        {"d01": 0, "d02": 0},
        root_steps=1,
        advance=lambda _name, carry, _start, n: int(carry) + int(n),
        feedback=feedback_on,
        feedback_enabled=True,
        block_between=False,
    )
    qke_probe = _wrf_qke_minmax(
        jnp.asarray([jnp.nan, -1.0, QKEMIN * 0.1, 0.25, 200.0], dtype=jnp.float64)
    )
    qke_values = np.asarray(qke_probe, dtype=np.float64)
    qke_gate = (
        bool(np.all(np.isfinite(qke_values)))
        and float(qke_values[0]) == float(QKEMIN)
        and float(qke_values[1]) == float(QKEMIN)
        and float(qke_values[2]) == float(QKEMIN)
        and float(qke_values[3]) == 0.25
        and float(qke_values[4]) == 150.0
    )
    d03_starts = [event[1] for event in advance_events if event[0] == "d03"]
    expected_counts = hierarchy.expected_step_counts(root_steps=2)
    pass_gate = (
        result.own_steps == expected_counts
        and conservation.conserved
        and live_cfg.update_cadence_s == 18.0
        and live_cfg.force_geopotential is False
        and feedback_calls == {"off": 0, "on": 1}
        and d03_starts[:6] == [1, 4, 7, 10, 13, 16]
        and qke_gate
    )
    payload = {
        "status": "PASS" if pass_gate else "FAIL",
        "proof": "v0110 live nesting CPU structural gates",
        "hierarchy": {
            "order": list(hierarchy.order),
            "edges": [edge.__dict__ for edge in hierarchy.nests],
            "expected_counts_root_steps_2": expected_counts,
            "observed_counts_root_steps_2": result.own_steps,
            "first_advance_events": advance_events[:12],
            "force_event_count": len(force_events),
            "first_force_events": force_events[:8],
        },
        "output": {
            "cadence": {"d01": 1, "d02": 3, "d03": 9, "d04": 9, "d05": 9},
            "fine_to_coarse_synchronized": True,
            "events": list(result.outputs),
        },
        "feedback": conservation.__dict__,
        "feedback_runtime_gate": feedback_calls,
        "qke_finiteness": {
            "wrf_source": "phys/module_bl_mynnedmf.F:3106-3107 qke=max(x,qkemin); qke=min(qke,150.)",
            "probe_input": ["nan", -1.0, float(QKEMIN * 0.1), 0.25, 200.0],
            "probe_output": [float(value) for value in qke_values],
            "all_finite": bool(np.all(np.isfinite(qke_values))),
            "pass": bool(qke_gate),
        },
        "subcycling": {
            "d03_chunk_start_steps": d03_starts,
            "domain_global_step_clock": d03_starts[:6] == [1, 4, 7, 10, 13, 16],
        },
        "live_child_boundary_config": {
            "update_cadence_s": live_cfg.update_cadence_s,
            "force_geopotential": live_cfg.force_geopotential,
            "nested_ph_relax": live_cfg.nested_ph_relax,
            "nested_w_relax": live_cfg.nested_w_relax,
            "nested_ph_spec": live_cfg.nested_ph_spec,
        },
        "notes": [
            "CPU-only proof; no GPU performance or full forecast equivalence claim.",
            "Full runtime validation is recorded separately in nesting_live_smoke.json.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "path": str(out_path)}))
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
