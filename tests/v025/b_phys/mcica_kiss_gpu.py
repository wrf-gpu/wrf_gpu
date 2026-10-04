"""BP61 asserted-GPU check: lane-wise KISS kernel vs the jump-ahead stream (bitwise) + timing."""
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

assert jax.devices()[0].platform == "gpu" and os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
from gpuwrf.kernels import rad_mcica as M  # noqa: E402
from test_mcica_kiss_lanes import _pressures  # noqa: E402

receipt = dict(device=str(jax.devices()[0]), cases=[])
for first, ng, shape in ((151, 140, (4096, 57)), (2, 224, (4096, 47)), (151, 140, (3, 37, 57))):
    p = _pressures(first + ng, shape)
    out = {}
    for flag in ("0", "1"):
        os.environ["GPUWRF_MCICA_KISS_KERNEL"] = flag
        fn = jax.jit(lambda q: M._random_values(q, first=first, ng=ng, legacy_fp64=False))  # fresh lambda: env read at trace
        out[flag] = np.asarray(jax.block_until_ready(fn(p)))
        samples = []
        for _ in range(20):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(p))
            samples.append(time.perf_counter() - t0)
        out[flag + "_ms"] = float(np.median(samples) * 1e3)
    receipt["cases"].append(dict(first=first, ng=ng, shape=list(shape), bitwise=bool(np.array_equal(out["0"], out["1"])),
                                 jump_ms=out["0_ms"], lanes_ms=out["1_ms"]))
receipt["bitwise_all"] = all(c["bitwise"] for c in receipt["cases"])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps(receipt))
