#!/usr/bin/env python3
"""CPU coupled smoke: real Swiss 42x42 d01 with cu_physics=5 (Grell-3D) or 93 (Grell-Devenyi).

Release suite (Noah-MP, MYNN-SL, MYNN, Thompson, RRTMG) with the cumulus option swapped to the
Grell scheme (cudt=0 -> every step), a few root steps through the operational domain tree on
CPU; checks every carry leaf finite, (with GPUWRF_CENSUS=1) zero guard events, and records the
convective activity (raining columns, RAINC, theta tendency) honestly.

CPU harness: ``jax.lax.linalg.tridiagonal_solve`` is replaced by a pure-JAX Thomas scan
(jaxlib LAPACK dgtsv deadlocks the affinity-sized XLA:CPU pool, E105; NOT the product solver).

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 GPUWRF_CENSUS=1 \\
      PYTHONPATH=src taskset -c 13 python proofs/v034/grell_swiss_smoke.py --cu 5 --steps 4 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

t_start = time.perf_counter()
ap = argparse.ArgumentParser()
ap.add_argument("--cu", type=int, choices=(5, 93), required=True)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[2] / "examples" / "switzerland_d01")
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"

import gpuwrf  # noqa: F401,E402
import jax  # noqa: E402

jax.config.update("jax_cpu_enable_async_dispatch", False)
assert jax.devices()[0].platform == "cpu"
from gpuwrf.contracts import state as state_contract  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]
from jax.experimental import pallas as pl  # noqa: E402

_orig_pallas_call = pl.pallas_call
pl.pallas_call = lambda *a, **k: _orig_pallas_call(*a, **{**k, "interpret": True})

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402


def _thomas_tridiagonal_solve(dl, d, du, b):
    """Pure-JAX Thomas solve (CPU harness only; see module doc)."""

    dl_t, d_t, du_t = (jnp.moveaxis(x, -1, 0)[..., None] for x in (dl, d, du))
    b_t = jnp.moveaxis(b, -2, 0)

    def forward(carry, row):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, r_i = row
        denom = b_i - a_i * cp_prev
        cp = c_i / denom
        dp = (r_i - a_i * dp_prev) / denom
        return (cp, dp), (cp, dp)

    init = (jnp.zeros_like(d_t[0]), jnp.zeros_like(b_t[0]))
    _, (cp, dp) = lax.scan(forward, init, (dl_t, d_t, du_t, b_t))

    def backward(x_next, row):
        cp_i, dp_i = row
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, xs = lax.scan(backward, jnp.zeros_like(b_t[0]), (cp, dp), reverse=True)
    return jnp.moveaxis(xs, 0, -2)


jax.lax.linalg.tridiagonal_solve = _thomas_tridiagonal_solve

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    _commit_to_operational_device,
    _initial_carry_for_run,
    _resolve_operational_suite,
)

args.out.mkdir(parents=True, exist_ok=True)
config = NestedPipelineConfig(args.input, args.out / "stream", args.out / "proof", hours=1, max_dom=1)
t0 = time.perf_counter()
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
load_s = time.perf_counter() - t0
bundle = bundles["d01"]
nml = replace(bundle.namelist, cu_physics=args.cu, cumulus_cadence_steps=1, cudt_minutes=0.0)
_resolve_operational_suite(nml)
carry0 = carries["d01"]
carry = _commit_to_operational_device(_initial_carry_for_run(bundle.state, nml).replace(
    noahmp_land=carry0.noahmp_land, noahmp_rad=carry0.noahmp_rad,
    radiation_diagnostics=carry0.radiation_diagnostics))
bundles["d01"] = replace(bundle, namelist=nml)
tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
carries = {"d01": carry}
own = {"d01": 0}
rainc0 = np.asarray(carry.state.rainc_acc, np.float64)
theta0 = np.asarray(carry.state.theta, np.float64)
times = []
for _ in range(args.steps):
    t0 = time.perf_counter()
    result = run_operational_domain_tree(tree, root_steps=1, carries=carries, initial_own_steps=own,
                                         block_between=False, root_sync_cadence=0)
    jax.block_until_ready(result.carries)
    carries, own = result.carries, dict(result.own_steps)
    times.append(time.perf_counter() - t0)
    print("step", own, round(times[-1], 2), flush=True)

c = carries["d01"]
nonfinite = []
for path, leaf in jax.tree_util.tree_leaves_with_path(c):
    a = np.asarray(leaf)
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        nonfinite.append(jax.tree_util.keystr(path))
census = None
if c.census is not None:
    census = {"guards": np.asarray(c.census.guards).tolist(),
              "guards_total": int(np.sum(np.asarray(c.census.guards)))}
drain = np.asarray(c.state.rainc_acc, np.float64) - rainc0
record = {
    "case": str(args.input), "cu_physics": args.cu, "steps": args.steps, "dt_s": float(nml.dt_s),
    "dx_m": float(nml.grid.projection.dx_m),
    "physics": {k: int(getattr(nml, k)) for k in ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics",
                                                     "ra_lw_physics", "ra_sw_physics", "cu_physics")},
    "use_noahmp": bool(nml.use_noahmp), "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"),
    "load_s": load_s, "step_s": times, "process_s": time.perf_counter() - t_start,
    "nonfinite_leaves": nonfinite, "census": census,
    "rainc_increment_mm": {"max": float(drain.max()), "min": float(drain.min()),
                           "raining_columns": int((drain > 0).sum())},
    "theta_change_max_K": float(np.max(np.abs(np.asarray(c.state.theta, np.float64) - theta0))),
    "tridiagonal_solve": "pure-JAX Thomas scan (CPU harness, E105)",
}
(args.out / "smoke.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({k: record[k] for k in ("cu_physics", "steps", "step_s", "nonfinite_leaves", "census",
                                         "rainc_increment_mm")}))
