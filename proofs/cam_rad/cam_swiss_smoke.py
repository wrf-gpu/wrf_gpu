#!/usr/bin/env python3
"""CPU coupled smoke: real Swiss 42x42 d01 with CAM radiation (ra_lw_physics / ra_sw_physics = 3) on the operational scan.

Noah-MP + MYNN + Thompson + CAM radiation (selected through the operational namelist API: the CLI does not bind
ra_* yet, lane o1-nlbind), a few root steps on CPU; checks every carry leaf finite and (GPUWRF_CENSUS=1) the guard
census.  Also reports the held RTHRATEN and the land-surface forcing (SWDOWN/GLW) next to an RRTMG 4/4 reference
run of the same steps (not a fidelity gate: different schemes).

CPU harness only: ``jax.lax.linalg.tridiagonal_solve`` is replaced by a pure-JAX Thomas scan in BOTH arms (jaxlib
LAPACK dgtsv deadlocks the affinity-sized XLA:CPU pool, E105; not the product solver).

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_CENSUS=1 PYTHONPATH=src taskset -c 28 \\
        python proofs/cam_rad/cam_swiss_smoke.py --steps 3 --ra-lw 3 --ra-sw 3 --out DIR
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
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--ra-lw", type=int, default=3)
ap.add_argument("--ra-sw", type=int, default=3)
ap.add_argument("--reference", action="store_true", help="also run the RRTMG 4/4 arm for comparison")
ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[2] / "examples" / "switzerland_d01")
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"

import gpuwrf  # noqa: F401,E402
import jax  # noqa: E402

jax.config.update("jax_cpu_enable_async_dispatch", False)
assert jax.devices()[0].platform == "cpu"
from gpuwrf.contracts import state as state_contract  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]

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
hierarchy, bundles, meta, run_start, dts, carries0 = _load_domains(config, ("d01",))
load_s = time.perf_counter() - t0
bundle = bundles["d01"]
nml = bundle.namelist


def run_arm(ra_lw, ra_sw):
    nml_arm = replace(nml, ra_lw_physics=ra_lw, ra_sw_physics=ra_sw)
    _resolve_operational_suite(nml_arm)
    carry = _initial_carry_for_run(bundle.state, nml_arm)
    if (ra_lw, ra_sw) != (4, 4):
        carry = carry.replace(radiation_diagnostics=None)
    carry = carry.replace(noahmp_land=carries0["d01"].noahmp_land, noahmp_rad=carries0["d01"].noahmp_rad)
    carry = _commit_to_operational_device(carry)
    tree = DomainTree.from_domains(hierarchy, {"d01": replace(bundle, namelist=nml_arm)}, feedback_enabled=False)
    carries, own, times = {"d01": carry}, {"d01": 0}, []
    for _ in range(args.steps):
        t1 = time.perf_counter()
        result = run_operational_domain_tree(tree, root_steps=1, carries=carries, initial_own_steps=own,
                                             block_between=False, root_sync_cadence=0)
        jax.block_until_ready(result.carries)
        carries, own = result.carries, dict(result.own_steps)
        times.append(time.perf_counter() - t1)
        print("arm", (ra_lw, ra_sw), "step", own, round(times[-1], 1), flush=True)
    c = carries["d01"]
    nonfinite = [jax.tree_util.keystr(p) for p, leaf in jax.tree_util.tree_leaves_with_path(c)
                 if np.asarray(leaf).dtype.kind == "f" and not np.all(np.isfinite(np.asarray(leaf)))]
    census = None
    if c.census is not None:
        guards = np.asarray(c.census.guards)
        census = {"guards_total": int(guards.sum()), "guards": guards.tolist()}
    rng = lambda a: [float(np.min(np.asarray(a))), float(np.max(np.asarray(a)))]
    rad = c.noahmp_rad
    return {"ra": [ra_lw, ra_sw], "step_s": times, "nonfinite_leaves": nonfinite, "census": census,
            "rthraten_K_per_day": [x * 86400.0 for x in rng(c.rthraten)],
            "surface_soldn": rng(rad[0]) if rad is not None else None,
            "surface_lwdn": rng(rad[1]) if rad is not None else None,
            "theta_range": rng(c.state.theta), "t_skin_range": rng(c.state.t_skin)}, c


record = {"case": str(args.input), "steps": args.steps, "dt_s": float(nml.dt_s),
          "radiation_cadence_steps": int(nml.radiation_cadence_steps), "load_s": load_s,
          "physics": {k: int(getattr(nml, k)) for k in ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics",
                                                        "cu_physics")},
          "use_noahmp": bool(nml.use_noahmp), "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"),
          "tridiagonal_solve": "pure-JAX Thomas scan in every arm (CPU harness, E105)",
          "ra_binding": "operational namelist API (CLI does not bind ra_* yet: lane o1-nlbind)", "arms": {}}
cam, c_cam = run_arm(args.ra_lw, args.ra_sw)
record["arms"]["cam"] = cam
if args.reference:
    ref, c_ref = run_arm(4, 4)
    record["arms"]["rrtmg44"] = ref
    d = np.asarray(c_cam.rthraten) - np.asarray(c_ref.rthraten)
    record["rthraten_cam_minus_rrtmg_K_per_day"] = [float(d.min() * 86400.0), float(d.max() * 86400.0),
                                                    float(np.sqrt(np.mean(d * d)) * 86400.0)]
record["process_s"] = time.perf_counter() - t_start
(args.out / "smoke.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({k: v for k, v in record.items() if k in ("arms", "rthraten_cam_minus_rrtmg_K_per_day")}, indent=1))
