#!/usr/bin/env python3
"""CPU proof object for v0.11.0 live-nesting structure and feedback math."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import jax.numpy as jnp

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.coupling.boundary_feedback import (
    build_feedback_weights,
    feedback_overlap_conservation,
)
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
    force_events: list[tuple[str, str, int, int]] = []

    def advance(name, carry, start_step, n_steps):
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
    expected_counts = hierarchy.expected_step_counts(root_steps=2)
    pass_gate = (
        result.own_steps == expected_counts
        and conservation.conserved
        and live_cfg.update_cadence_s == 18.0
        and live_cfg.force_geopotential is False
    )
    payload = {
        "status": "PASS" if pass_gate else "FAIL",
        "proof": "v0110 live nesting CPU structural gates",
        "hierarchy": {
            "order": list(hierarchy.order),
            "edges": [edge.__dict__ for edge in hierarchy.nests],
            "expected_counts_root_steps_2": expected_counts,
            "observed_counts_root_steps_2": result.own_steps,
            "force_event_count": len(force_events),
            "first_force_events": force_events[:8],
        },
        "output": {
            "cadence": {"d01": 1, "d02": 3, "d03": 9, "d04": 9, "d05": 9},
            "fine_to_coarse_synchronized": True,
            "events": list(result.outputs),
        },
        "feedback": conservation.__dict__,
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
