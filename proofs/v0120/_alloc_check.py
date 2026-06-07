"""Definitive check: does setting XLA_PYTHON_CLIENT_ALLOCATOR at RUNTIME (after
`import jax`, before first device op) actually switch the GPU allocator?

Strategy: allocate a 2 GiB array then free it, repeatedly with different sizes,
and report this process's nvidia-smi footprint. Under BFC (PREALLOCATE=false) the
pool grows and is retained (footprint stays at the max). Under platform/cudaMalloc
the footprint tracks the live set and DROPS after frees. We set the allocator at
runtime per argv[1] in {'runtime','none'}.
"""

from __future__ import annotations

import os
import subprocess
import sys


def smi_self() -> float:
    pid = os.getpid()
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], text=True)
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2 and p[0].isdigit() and int(p[0]) == pid:
            return float(p[1])
    return 0.0


mode = sys.argv[1] if len(sys.argv) > 1 else "runtime"

import jax  # module import (no backend init)

if mode == "runtime":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # set AFTER import jax

import jax.numpy as jnp

# First device op -> backend init.
x = jnp.ones((1, 1)); x.block_until_ready()
base = smi_self()

# Allocate ~6 GiB, free it, then allocate ~1 GiB. Report footprints.
big = jnp.ones((6 * 1024**3 // 8,), dtype=jnp.float64)  # 6 GiB
big.block_until_ready()
after_big = smi_self()
del big
import gc; gc.collect()
small = jnp.ones((1024**3 // 8,), dtype=jnp.float64)  # 1 GiB
small.block_until_ready()
after_free = smi_self()

print(f"mode={mode} allocator_env={os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')} "
      f"base={base:.0f} after_big={after_big:.0f} after_free_then_1g={after_free:.0f}")
print("VERDICT:",
      "PLATFORM-LIKE (footprint dropped after free)" if after_free < after_big - 2000
      else "BFC-LIKE (footprint retained after free)")
