"""BP56 asserted-GPU check: Pallas WRF ``tridiag2`` sweep vs the literal REAL4 recurrence.

PTX RN32 operations (no contraction, IEEE division) should reproduce NumPy float32
(the WRF REAL build, no FMA) bitwise. Also times one PROD-d02-sized solve batch
(31,239 x 44) against XLA's tridiagonal_solve (cuSPARSE pcrGtsv). Writes --output.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

assert jax.devices()[0].platform == "gpu" and os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
from gpuwrf.kernels import phys_mynn_tridiag as T  # noqa: E402
from test_mynn_tridiag_real import mynn_like, wrf_tridiag2  # noqa: E402

receipt = dict(device=str(jax.devices()[0]), cases=[])
for seed, shape in ((0, (3, 70, 44)), (1, (31239, 44)), (2, (8400, 44)), (3, (131, 44))):
    a, b, c, d = mynn_like(seed, shape)
    got = np.asarray(jax.jit(T.solve_tridiagonal_real)(a, b, c, d))
    ref = wrf_tridiag2(a, b, c, d)
    ulp = np.abs(got.view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64))
    receipt["cases"].append(dict(seed=seed, shape=list(shape), bitwise_fraction=float(np.mean(got == ref)),
                                 max_ulp=int(ulp.max()), finite=bool(np.isfinite(got).all())))
a, b, c, d = (jnp.asarray(v) for v in mynn_like(5, (31239, 44)))
fns = dict(pallas=jax.jit(T.solve_tridiagonal_real),
           xla=jax.jit(lambda a, b, c, d: jax.lax.linalg.tridiagonal_solve(a, b, c, d[..., None])[..., 0]))
for name, fn in fns.items():
    jax.block_until_ready(fn(a, b, c, d))
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(a, b, c, d))
        samples.append(time.perf_counter() - t0)
    receipt[f"{name}_median_ms"] = float(np.median(samples) * 1e3)
receipt["bitwise_all"] = all(c["bitwise_fraction"] == 1.0 for c in receipt["cases"])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps(receipt))
