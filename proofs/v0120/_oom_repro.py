"""Empirical OOM root-cause repro for the standalone live-nested 24 h forecast.

Goal: prove WHERE the 9.24 GiB allocation comes from and whether peak VRAM grows
with lead. Loads the exact failing case, advances a short horizon with the real
nested driver, and measures device peak_bytes_in_use / largest_alloc_size at three
probe points:

  A. after loading all domains (persistent state working set);
  B. after a short nested advance with a LIGHTWEIGHT output callback
     (surface_layer_diagnostics only -- the v0.11.0 path);
  C. after computing the FULL compute_m9_diagnostics (RRTMG g-point transient,
     the v0.12.0 nested_pipeline writer path) on each domain.

If C >> B, the OOM is the per-output RRTMG diagnostic transient, not state growth.
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
from gpuwrf.runtime.operational_mode import (
    compute_m9_diagnostics,
    surface_layer_diagnostics,
)


def _dev():
    return jax.devices()[0]


def _peak_gib() -> float:
    s = _dev().memory_stats()
    return float(s.get("peak_bytes_in_use", 0)) / 2**30


def _inuse_gib() -> float:
    s = _dev().memory_stats()
    return float(s.get("bytes_in_use", 0)) / 2**30


def _largest_gib() -> float:
    s = _dev().memory_stats()
    return float(s.get("largest_alloc_size", 0)) / 2**30


def _reset_peak() -> None:
    # JAX has no direct peak reset; we record deltas vs. a baseline snapshot instead.
    pass


def main() -> int:
    case = Path(sys.argv[1])
    max_dom = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    hours = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    out_json = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("proofs/v0120/_oom_repro.json")

    names = domain_names_for(max_dom)
    scratch = Path("/tmp/oom_repro_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    config = NestedPipelineConfig(
        input_dir=case,
        output_dir=scratch / "out",
        proof_dir=scratch / "proof",
        hours=int(hours) if float(hours).is_integer() else 1,
        max_dom=max_dom,
        scratch_dir=scratch,
    )

    report: dict = {"case": str(case), "max_dom": max_dom, "hours": hours, "names": list(names)}

    t0 = time.perf_counter()
    hierarchy, bundles, meta, run_start, dt_by_domain = _load_domains(config, names)
    # force the states onto device
    for b in bundles.values():
        jax.block_until_ready(b.state.theta)
    report["load_wall_s"] = time.perf_counter() - t0
    report["A_after_load"] = {
        "peak_gib": _peak_gib(),
        "inuse_gib": _inuse_gib(),
        "state_bytes_gib": {k: int(v.state.bytes()) / 2**30 for k, v in bundles.items()},
    }

    root = names[0]
    root_dt = dt_by_domain[root]
    root_steps = int(round(hours * 3600.0 / root_dt))
    output_cadence = {n: int(round(3600.0 / dt_by_domain[n])) for n in names}
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)

    # --- B: nested advance with a LIGHTWEIGHT output callback (v0.11.0 path) ---
    light_outputs = []

    def light_output(name, own_step, state):
        surf = surface_layer_diagnostics(state, bundles[name].namelist.grid)
        jax.block_until_ready(surf.t2)
        light_outputs.append((name, int(own_step)))
        return {"domain": name, "own_step": int(own_step)}

    tB = time.perf_counter()
    resultB = run_operational_domain_tree(
        tree,
        root_steps=root_steps,
        feedback_enabled=False,
        output=light_output,
        output_cadence_steps=output_cadence,
        block_between=True,
    )
    jax.block_until_ready(tuple(s.theta for s in resultB.states.values()))
    report["B_advance_lightoutput"] = {
        "wall_s": time.perf_counter() - tB,
        "peak_gib": _peak_gib(),
        "inuse_gib": _inuse_gib(),
        "largest_alloc_gib": _largest_gib(),
        "n_outputs": len(light_outputs),
        "all_finite": {
            n: bool(jnp.all(jnp.isfinite(s.theta))) for n, s in resultB.states.items()
        },
    }

    # --- C: full compute_m9_diagnostics (RRTMG transient) per domain ---------
    peak_before_C = _peak_gib()
    c_largest = {}
    for name, state in resultB.states.items():
        nml = bundles[name].namelist
        m9 = compute_m9_diagnostics(state, nml, jnp.asarray(float(hours) * 3600.0, dtype=jnp.float64))
        jax.block_until_ready(m9.t2)
        c_largest[name] = {
            "peak_gib_after": _peak_gib(),
            "largest_alloc_gib": _largest_gib(),
            "t2_finite": bool(jnp.all(jnp.isfinite(m9.t2))),
        }
    report["C_full_m9_diagnostics"] = {
        "peak_before_gib": peak_before_C,
        "per_domain": c_largest,
        "peak_after_gib": _peak_gib(),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
