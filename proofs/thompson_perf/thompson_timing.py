"""Warmed timing of the Thompson column kernel on the REAL WRF-oracle pre-state.

Times ``step_thompson_column_with_precip`` on the actual operational workload
(5187 columns x 44 levels = the d02 microphysics call) under the OPERATIONAL
input dtypes (hydrometeors fp32-storage, p fp64) so the measured ms is what the
coupled step actually pays. Reports median/min ms, and the XLA cost-analysis
FLOPs/bytes so we can read the arithmetic intensity.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 \
    python proofs/thompson_perf/thompson_timing.py [tag]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    step_thompson_column_with_precip,
    density_from_pressure_temperature,
)
from gpuwrf.validation.tier1_thompson import _load_f64_oracle_arrays, _columns_from_oracle, ORACLE_DIR

PROOF = Path("proofs/thompson_perf")
DT = 18.0


def _block(x):
    jax.block_until_ready(x)


def _build_operational_columns():
    """Real oracle pre-state, cast to the OPERATIONAL storage dtypes.

    Operational matrix: hydrometeors/qv/dz/w are fp32 storage; p stays fp64
    (acoustic-locked). T/rho are derived (promote to fp64). This is exactly what
    ``_thompson_column_from_state`` feeds the kernel in the coupled fp32 run.
    """
    pre, _post = _load_f64_oracle_arrays(ORACLE_DIR)
    col64 = _columns_from_oracle(pre)  # everything fp64

    def f32(x):
        return jnp.asarray(x, jnp.float32)

    p = jnp.asarray(col64.p, jnp.float64)
    qv = f32(col64.qv)
    T = jnp.asarray(col64.T, jnp.float64)  # T = theta*exner promotes to fp64 operationally
    rho = density_from_pressure_temperature(p, T, qv)
    return ThompsonColumnState(
        qv=qv,
        qc=f32(col64.qc),
        qr=f32(col64.qr),
        qi=f32(col64.qi),
        qs=f32(col64.qs),
        qg=f32(col64.qg),
        Ni=f32(col64.Ni),
        Nr=f32(col64.Nr),
        T=T,
        p=p,
        rho=rho,
        Ns=f32(col64.Ns),
        Ng=f32(col64.Ng),
        dz=f32(col64.dz),
        w=f32(col64.w),
    )


def _time(fn, state, reps=50, warm=5):
    f = jax.jit(fn)
    for _ in range(warm):
        _block(f(state))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _block(f(state))
        samples.append(time.perf_counter() - t0)
    samples.sort()
    try:
        ca = f.lower(state).compile().cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        flops = float(ca.get("flops", float("nan")))
        byts = float(ca.get("bytes accessed", ca.get("bytes_accessed", float("nan"))))
    except Exception:
        flops = byts = float("nan")
    return {
        "median_ms": samples[len(samples) // 2] * 1000.0,
        "min_ms": samples[0] * 1000.0,
        "p10_ms": samples[max(0, len(samples) // 10)] * 1000.0,
        "flops": flops,
        "gbytes": byts / 1e9,
        "reps": reps,
    }


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    state = _build_operational_columns()
    n_cols, n_lev = int(state.qv.shape[0]), int(state.qv.shape[1])

    def call(s):
        out, precip = step_thompson_column_with_precip(s, DT, debug=False)
        return out, precip

    rec = _time(call, state)
    o, _ = call(state)
    rec.update(
        {
            "tag": tag,
            "scope": "Thompson kernel warmed timing on REAL WRF-oracle pre-state (operational dtypes)",
            "n_columns": n_cols,
            "n_levels": n_lev,
            "dt_s": DT,
            "device": str(jax.devices()[0]),
            "input_dtypes": {
                "qv": str(state.qv.dtype),
                "p": str(state.p.dtype),
                "T": str(state.T.dtype),
                "rho": str(state.rho.dtype),
            },
            "output_qr_dtype": str(o.qr.dtype),
        }
    )

    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / f"thompson_timing_{tag}.json"
    fn.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=2, sort_keys=True))
    print(f"\nwrote {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
