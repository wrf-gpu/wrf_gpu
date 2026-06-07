"""Fragmentation / real-writer-path OOM probe.

Runs the REAL nested driver path -- the m9 wrfout writer fires INSIDE the
recursion exactly as execute_nested_pipeline does -- but instruments device
memory_stats at every output event so we can see whether peak_bytes_in_use,
peak_bytes_reserved, and largest_free_block_bytes drift with lead (fragmentation)
or stay flat.

Optionally sets the platform (cudaMalloc) allocator via env before this runs.
Run with H short (2-3) and watch the per-event stats. If largest_free_block
shrinks while reserved grows, the OOM is BFC fragmentation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.integration.nested_pipeline import (
    NestedPipelineConfig,
    _load_domains,
    domain_names_for,
)
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree
from gpuwrf.runtime.operational_mode import compute_m9_diagnostics


def _stats():
    s = jax.devices()[0].memory_stats()
    g = 2**30
    return {
        "peak_inuse_gib": s.get("peak_bytes_in_use", 0) / g,
        "inuse_gib": s.get("bytes_in_use", 0) / g,
        "reserved_gib": s.get("bytes_reserved", 0) / g,
        "peak_reserved_gib": s.get("peak_bytes_reserved", 0) / g,
        "pool_gib": s.get("pool_bytes", 0) / g,
        "peak_pool_gib": s.get("peak_pool_bytes", 0) / g,
        "largest_free_block_gib": s.get("largest_free_block_bytes", 0) / g,
        "largest_alloc_gib": s.get("largest_alloc_size", 0) / g,
    }


def main() -> int:
    case = Path(sys.argv[1])
    max_dom = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    hours = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out_json = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("proofs/v0120/_oom_probe2.json")

    names = domain_names_for(max_dom)
    scratch = Path("/tmp/oom_probe2_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    config = NestedPipelineConfig(
        input_dir=case,
        output_dir=scratch / "out",
        proof_dir=scratch / "proof",
        hours=hours,
        max_dom=max_dom,
        scratch_dir=scratch,
    )

    hierarchy, bundles, meta, run_start, dt_by_domain = _load_domains(config, names)
    root = names[0]
    root_dt = dt_by_domain[root]
    root_steps = int(round(hours * 3600.0 / root_dt))
    output_cadence = {n: int(round(3600.0 / dt_by_domain[n])) for n in names}
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)

    timeline = []

    # A writer that does the SAME heavy m9 work as the real pipeline writer
    # (compute_m9_diagnostics) and records device stats at each output event.
    def heavy_output(name, own_step, state):
        nml = bundles[name].namelist
        cadence = int(output_cadence[name])
        lead_h = int(round(int(own_step) / cadence))
        m9 = compute_m9_diagnostics(
            state, nml, jnp.asarray(float(lead_h) * 3600.0, dtype=jnp.float64)
        )
        jax.block_until_ready(m9.t2)
        st = _stats()
        st.update({"domain": name, "lead_h": lead_h, "own_step": int(own_step),
                   "t2_finite": bool(jnp.all(jnp.isfinite(m9.t2)))})
        timeline.append(st)
        return {"domain": name, "lead_h": lead_h}

    t0 = time.perf_counter()
    try:
        result = run_operational_domain_tree(
            tree,
            root_steps=root_steps,
            feedback_enabled=False,
            output=heavy_output,
            output_cadence_steps=output_cadence,
            block_between=True,
        )
        jax.block_until_ready(tuple(s.theta for s in result.states.values()))
        oom = None
    except Exception as exc:  # noqa: BLE001
        oom = f"{type(exc).__name__}: {exc}"

    report = {
        "case": str(case), "max_dom": max_dom, "hours": hours,
        "root_steps": root_steps, "wall_s": time.perf_counter() - t0,
        "oom": oom, "final_stats": _stats(), "timeline": timeline,
        "allocator_env": {
            "XLA_PYTHON_CLIENT_ALLOCATOR": __import__("os").environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": __import__("os").environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": __import__("os").environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    # compact stdout
    print("OOM:", oom)
    for st in timeline:
        print(f"  {st['domain']} lead={st['lead_h']:>2} inuse={st['inuse_gib']:.2f} "
              f"peak={st['peak_inuse_gib']:.2f} reserved={st['reserved_gib']:.2f} "
              f"freeblk={st['largest_free_block_gib']:.2f} largestalloc={st['largest_alloc_gib']:.2f}")
    print("FINAL:", json.dumps(report["final_stats"], indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
