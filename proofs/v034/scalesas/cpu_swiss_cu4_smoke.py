"""Real-case CPU coupled smoke for cu_physics=4: Swiss 42x42 d01, N root steps (legacy CPU path).

Adapted from o1-restart/tools/cpu_swiss_restart.py. CPU HARNESS ONLY: jax.lax.linalg.tridiagonal_solve is
monkeypatched with a pure-JAX Thomas solve (jaxlib LAPACK dgtsv FFI deadlocks the affinity-sized XLA:CPU
pool, E105) -- not the product solver. Usage:
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 python cpu_swiss_cu4_smoke.py --input DIR --out F --steps N
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
t_start = time.perf_counter()
ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--steps", type=int, default=6)
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"
import gpuwrf  # noqa: F401
import jax
jax.config.update("jax_cpu_enable_async_dispatch", False)
assert jax.devices()[0].platform == "cpu"
from gpuwrf.contracts import state as state_contract
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
import numpy as np
import jax.numpy as jnp
from jax import lax

def _thomas_tridiagonal_solve(dl, d, du, b):
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

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree
root = args.out.parent
config = NestedPipelineConfig(args.input, root / "stream", root / "proof", hours=1, max_dom=1,
                              emit_initial_history=False)
t0 = time.perf_counter()
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
load_s = time.perf_counter() - t0
nml = bundles["d01"].namelist if hasattr(bundles["d01"], "namelist") else None
tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
c0 = carries["d01"]
rec = {"load_s": load_s, "source": os.environ.get("PYTHONPATH"), "cu_physics": int(getattr(nml, "cu_physics", -1)) if nml is not None else None,
       "stepcu": int(getattr(nml, "cumulus_cadence_steps", -1)) if nml is not None else None,
       "held_seed": None if c0.cumulus_tendencies is None else [str(np.asarray(h).dtype) for h in c0.cumulus_tendencies],
       "steps": []}
t0 = time.perf_counter()
res = run_operational_domain_tree(tree, root_steps=args.steps, carries=carries, initial_own_steps={"d01": 0},
                                  block_between=False, root_sync_cadence=0)
jax.block_until_ready(res.carries)
d = res.carries["d01"]
import pickle
(args.out.parent / "held.pkl").write_bytes(pickle.dumps({"held": [np.asarray(h) for h in d.cumulus_tendencies] if d.cumulus_tendencies is not None else None, "rainc": np.asarray(d.state.rainc_acc), "steps": dict(res.own_steps)}))
held = d.cumulus_tendencies
finite = all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves(d) if np.asarray(x).dtype.kind in "fc")
entry = {"steps": dict(res.own_steps), "wall_s": time.perf_counter() - t0, "finite": finite,
         "max_abs_RTHCUTEN": float(np.max(np.abs(np.asarray(held[0])))) if held is not None else None,
         "max_abs_RQVCUTEN": float(np.max(np.abs(np.asarray(held[1])))) if held is not None else None,
         "cols_PRATEC_pos": int(np.sum(np.asarray(held[6]) > 0)) if held is not None else None,
         "max_rainc_mm": float(np.max(np.asarray(d.state.rainc_acc))),
         "max_abs_theta_step_change_vs_init": float(np.max(np.abs(np.asarray(d.state.theta, np.float64) - np.asarray(carries["d01"].state.theta, np.float64)))),
         "census_guards": np.asarray(jax.device_get(d.census.guards)).tolist() if getattr(d, "census", None) is not None else None}
rec["result"] = entry
print(json.dumps(entry, default=str), flush=True)
rec["process_s"] = time.perf_counter() - t_start
args.out.write_text(json.dumps(rec, indent=1, default=str))
print("DONE", json.dumps({k: v for k, v in rec.items() if k != "steps"}, default=str), flush=True)
