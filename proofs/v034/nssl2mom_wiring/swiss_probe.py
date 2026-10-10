"""Real Swiss 42x42x44 single-domain CPU probe for the mp=18 wiring (no GPU).

Modes
  jaxpr <input_dir> <out>   anonymized jaxpr (E151) of the product per-step function ``_physics_boundary_step`` (the
                            body of ``_advance_chunk_fori``; run_radiation=False) built from the real case (OFF-path
                            identity: run once per source tree via PYTHONPATH and compare sha)
  smoke <input_dir> <out> N run N real root steps through ``run_operational_domain_tree`` (product path) and report
                            finiteness of every carry leaf + hydrometeor/number/precip statistics

CPU harness patches (disclosed, from lane o1-restart tools/cpu_swiss_probe.py + cli_cpu.py): State GPU guard ->
CPU device; jax.lax.linalg.tridiagonal_solve -> pure-JAX Thomas scan (jaxlib dgtsv FFI deadlock on pinned cores,
E105); optional Pallas interpret mode (GPUWRF_PROBE_INTERPRET=1) for fast-default Pallas kernels.
Usage: JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<tree>/src python swiss_probe.py MODE IN OUT [N]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
mode, input_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
out.mkdir(parents=True, exist_ok=True)

import gpuwrf  # noqa: F401,E402  (fast-default resolution first)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

assert jax.devices()[0].platform == "cpu"
from gpuwrf.contracts import state as state_contract  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]


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
    _, (cps, dps) = lax.scan(forward, init, (dl_t, d_t, du_t, b_t))

    def backward(x_next, row):
        cp, dp = row
        x = dp - cp * x_next
        return x, x

    _, xs = lax.scan(backward, jnp.zeros_like(b_t[0]), (cps, dps), reverse=True)
    return jnp.moveaxis(xs, 0, -2)


lax.linalg.tridiagonal_solve = _thomas_tridiagonal_solve
jax.lax.linalg.tridiagonal_solve = _thomas_tridiagonal_solve
if os.environ.get("GPUWRF_PROBE_INTERPRET") == "1":
    from jax.experimental import pallas as pl

    _orig = pl.pallas_call
    pl.pallas_call = lambda *a, **k: _orig(*a, **{**k, "interpret": True})

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree  # noqa: E402

t_start = time.perf_counter()
config = NestedPipelineConfig(input_dir, out / "out", out / "proof", hours=1, max_dom=1)
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
if os.environ.get("GPUWRF_PROBE_MP"):
    # Disclosed harness override: on this base the native pipeline does not bind mp_physics from
    # namelist.input (separate namelist-binding work), so select the scheme on the loaded bundle and seed
    # its conditional State leaves BEFORE the first scan (frozen carry pytree structure).
    import dataclasses
    mp = int(os.environ["GPUWRF_PROBE_MP"])
    bundles["d01"] = dataclasses.replace(bundles["d01"], namelist=dataclasses.replace(bundles["d01"].namelist, mp_physics=mp))
    st = carries["d01"].state.ensure_conditional_leaves(mp_physics=mp)
    # match the live (precision-enforced) carry dtypes, as _initial_carry_for_run does for a native mp=18 run
    st = st.replace(_cast=False, qh=st.qh.astype(st.qc.dtype), Nh=st.Nh.astype(st.Nr.dtype),
                    qvolg=st.qvolg.astype(st.qg.dtype), qvolh=st.qvolh.astype(st.qg.dtype),
                    hail_acc=st.hail_acc.astype(st.rain_acc.dtype))
    carries["d01"] = carries["d01"].replace(state=st)
nml = bundles["d01"].namelist
record = {"mode": mode, "gpuwrf": gpuwrf.__file__, "mp_physics": int(nml.mp_physics), "dt": dts,
          "load_s": time.perf_counter() - t_start,
          "fast_env": sorted(k for k, v in os.environ.items() if k.startswith("GPUWRF_") and v not in ("", "0"))}


def anon(txt):
    txt = re.sub(r"/[^\s\"':]+\.py:\d+(:\d+)?", "SRC", txt)
    return re.sub(r"name_and_src_info=[^,\]\)]*", "NSI", txt)


if mode == "jaxpr":
    from gpuwrf.runtime.operational_mode import _physics_boundary_step
    t0 = time.perf_counter()
    jx = jax.make_jaxpr(lambda c, s: _physics_boundary_step(c, nml, s, run_radiation=False, debug=False,
                                                            clock_base=None))(carries["d01"], jnp.int32(1))
    a = anon(str(jx))
    (out / "step.jaxpr.txt").write_text(a)
    record.update(jaxpr_sha=hashlib.sha256(a.encode()).hexdigest(), eqns=len(jx.jaxpr.eqns), chars=len(a),
                  trace_s=time.perf_counter() - t0)
elif mode == "smoke":
    import numpy as np
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    steps = {"d01": 0}
    rows = []
    for _ in range(nsteps):
        t0 = time.perf_counter()
        result = run_operational_domain_tree(tree, root_steps=1, carries=carries, initial_own_steps=steps,
                                             block_between=False, root_sync_cadence=0)
        jax.block_until_ready(result.carries)
        carries, steps = result.carries, dict(result.own_steps)
        st = carries["d01"].state
        leaves = jax.tree_util.tree_leaves(carries["d01"])
        nonfinite = int(sum(int(np.sum(~np.isfinite(np.asarray(x)))) for x in leaves
                            if hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.floating)))
        row = {"own_step": steps["d01"], "wall_s": round(time.perf_counter() - t0, 2), "nonfinite": nonfinite}
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "Nc", "Nr", "Ni", "Ns", "Ng", "Nh", "Nn",
                     "qvolg", "qvolh", "rain_acc", "snow_acc", "graupel_acc", "hail_acc"):
            v = getattr(st, name, None)
            if v is not None:
                a = np.asarray(v, np.float64)
                row[name] = [float(a.min()), float(a.max())]
        rows.append(row)
        print(json.dumps(row), flush=True)
    record["steps"] = rows
record["process_s"] = time.perf_counter() - t_start
(out / f"{mode}.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "steps"}))
