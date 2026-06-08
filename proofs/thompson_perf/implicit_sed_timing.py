"""Time the full Thompson kernel: faithful explicit (default) vs implicit
backward-Euler at nsub=1/2/4, on the REAL d02 oracle operational columns.

Answers: does any acceptable implicit nsub keep the kernel speedup? (The
microphysics validation showed nsub=1 is too diffusive; nsub>=4 is needed to
approach faithful accuracy, but more sweeps cost more.)

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 taskset -c 0-3 \
    python3 proofs/thompson_perf/implicit_sed_timing.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gpuwrf.physics.thompson_column as tc
from thompson_timing import _build_operational_columns  # noqa: E402  (same dir on sys.path)

PROOF = Path("proofs/thompson_perf")
DT = 18.0
REPS = 100


def _tile(state, factor):
    # Tile the columns to the full d02 microphysics workload (~20748 cols).
    def rep(x):
        return jnp.concatenate([x] * factor, axis=0)
    import dataclasses
    fields = {n: rep(getattr(state, n)) for n in tc.ThompsonColumnState.__slots__}
    return tc.ThompsonColumnState(**fields)


def time_kernel(state, label):
    fn = jax.jit(lambda s: tc._step_thompson_column_full_impl(s, DT, False))
    out = fn(state)
    jax.block_until_ready(out[0].qr)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn(state)
        jax.block_until_ready(out[0].qr)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return {"label": label, "median_ms": float(np.median(ts)), "min_ms": float(ts.min())}


def main():
    print("devices:", jax.devices())
    base = _build_operational_columns()
    ncol0 = int(base.qv.shape[0])
    factor = max(1, 20748 // ncol0)
    state = _tile(base, factor)
    print(f"workload: {int(state.qv.shape[0])} cols x {state.qv.shape[1]} lev "
          f"(base {ncol0} x{factor})")

    results = {}
    # faithful (default, implicit OFF)
    os.environ.pop("GPUWRF_THOMPSON_IMPLICIT_SED", None)
    results["faithful_explicit"] = time_kernel(state, "faithful_explicit")
    # implicit nsub 1/2/4
    for nsub in (1, 2, 4):
        os.environ["GPUWRF_THOMPSON_IMPLICIT_SED"] = str(nsub)
        results[f"implicit_nsub{nsub}"] = time_kernel(state, f"implicit_nsub{nsub}")
    os.environ.pop("GPUWRF_THOMPSON_IMPLICIT_SED", None)

    base_ms = results["faithful_explicit"]["median_ms"]
    for k, v in results.items():
        v["speedup_vs_faithful"] = round(base_ms / v["median_ms"], 3)

    rec = {"workload_cols": int(state.qv.shape[0]), "levels": int(state.qv.shape[1]),
           "reps": REPS, "dt_s": DT, "results": results}
    out = PROOF / "implicit_sed_timing.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
