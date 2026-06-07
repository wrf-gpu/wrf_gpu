#!/usr/bin/env python3
"""v0.13 compile-speed proof: cold-vs-warm compile + identical-results (CPU).

WHAT THIS PROVES
----------------
1. **Warm compile << cold compile.** A representative fp64 jitted graph (a
   ``lax.scan`` over substeps mixing stencil + GEMM ops -- the same XLA compile
   cost drivers as the WRF acoustic-substep / RK3 timestep: fusion analysis,
   layout assignment, codegen, LLVM) is compiled in a COLD process (empty cache)
   and then in a WARM process (cache populated by the cold run). The warm
   ``.lower().compile()`` is a disk-read of the identical executable, so it is
   far cheaper than the cold compile.

2. **Identical results.** The output array from the cold-compiled executable is
   bit-for-bit equal to the warm-compiled one. The persistent cache returns the
   *same* XLA program, so this MUST hold; the assert guards against any future
   regression that silently changed the cached object.

3. **AOT entrypoint warms the cache.** ``runtime.aot_precompile.precompile``
   (``jit(fn).lower(args).compile()``) is the mechanism used to warm the cache,
   and ``runtime.compile_cache.warm_hit_for`` detects a warm hit robustly (no new
   cache entry written => served from disk).

SCOPE / HONESTY
---------------
- **CPU backend** (the GPU is reserved for the validation gate). The cold/warm
  *ratio* and the identical-results property are backend-independent; the
  absolute seconds are CPU numbers. **GPU compile numbers (where the multi-
  minute fp64 cold compile + autotuning lives) are a v0.13-GPU-followup** and
  are NOT claimed here.
- This is a *representative* graph, not the full forecast graph (which would be
  an enormous CPU compile). It exercises the same compile machinery; it is not a
  forecast-accuracy artifact.
- Numerically inert: this proof is about *when* compilation happens, never
  *what* is computed.

Run (cold process spawns a fresh warm subprocess so the cache truly starts
empty and the warm run is a genuine separate process):

    PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 \
        python proofs/v0130/compile_speed.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# This script is run twice: as the orchestrator (no _COMPILE_PHASE) which spawns
# a cold child and a warm child; and as a child (_COMPILE_PHASE in {cold,warm})
# which does one compile and prints a JSON line. Splitting into subprocesses is
# the only way to get a TRUE cold compile (XLA caches in-process, and our import
# hook configures the persistent cache once at first import).

REPR_SUBSTEPS = 40  # acoustic-substep-like scan length (heavier => clearer cold compile)
REPR_N = 96  # field edge size (fp64; bounded for CPU)


def _build_and_compile(cache_dir: str, phase: str) -> dict:
    """Child phase: configure the cache at the given dir, build the
    representative graph, lower+compile it, and report timing + a result hash."""
    # Point the persistent compile cache at the shared dir BEFORE importing
    # anything that compiles. gpuwrf's import hook reads these env vars.
    os.environ["GPUWRF_JAX_CACHE_DIR"] = cache_dir
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # Disable the GPU autotune flags on this CPU proof (they would fatally abort
    # a CPU jaxlib build); the guard already does this, belt-and-braces here.
    os.environ["GPUWRF_XLA_AUTOTUNE_CACHE"] = "0"

    import gpuwrf  # noqa: F401  (enables x64 + configures the persistent cache)
    import jax
    import jax.numpy as jnp
    from gpuwrf.runtime.aot_precompile import precompile
    from gpuwrf.runtime.compile_cache import (
        CACHE_STATUS,
        cache_entry_count,
        warm_hit_for,
    )

    def representative_step(state, _x):
        """One substep: a 5-point stencil + a GEMM mix in fp64 -- the same op
        classes (elementwise stencil fusion + matmul autotune target) as a WRF
        timestep, so XLA pays comparable fusion/codegen/layout cost. ``lax.scan``
        passes (carry, x) even when xs=None (x is None), so the signature has the
        unused ``_x`` slot."""
        u, k = state
        # 5-point Laplacian-style stencil (advection/diffusion analogue)
        lap = (
            jnp.roll(u, 1, 0)
            + jnp.roll(u, -1, 0)
            + jnp.roll(u, 1, 1)
            + jnp.roll(u, -1, 1)
            - 4.0 * u
        )
        # GEMM (PGF / vertical-solve analogue; autotune target on GPU)
        mixed = jnp.tanh(u @ u.T) * 1e-3
        u = u + 0.01 * lap + mixed + jnp.sin(u + k)
        return (u, k + 1.0), None

    def graph(u0):
        (u, _), _ = jax.lax.scan(representative_step, (u0, 0.0), xs=None, length=REPR_SUBSTEPS)
        return u

    u0 = jnp.ones((REPR_N, REPR_N), dtype=jnp.float64)

    n_before = cache_entry_count()

    # AOT precompile via the new entrypoint (jit + lower + compile).
    compiled, result = precompile(graph, u0, key=f"repr-{phase}")

    n_after = cache_entry_count()

    out = compiled(u0)
    out.block_until_ready()
    # Stable bitwise fingerprint of the result for the identical-results check.
    # Use sha256 (NOT builtin hash(), which is per-process salted) so the cold
    # and warm child processes produce comparable digests for identical bytes.
    import hashlib

    import numpy as np

    arr = np.asarray(out)
    result_hash = hashlib.sha256(arr.tobytes()).hexdigest()

    return {
        "phase": phase,
        "compile_seconds": result.compile_seconds,
        "cache_enabled": bool(CACHE_STATUS.get("enabled")),
        "cache_dir": CACHE_STATUS.get("dir"),
        "entries_before": n_before,
        "entries_after": n_after,
        "new_entries": n_after - n_before,
        "warm_hit": (n_after == n_before and n_before > 0),
        "result_hash": result_hash,
        "result_sum": float(arr.sum()),
        "result_shape": list(arr.shape),
        "backend": jax.default_backend(),
        "precompile_error": result.error,
    }


def _run_child(cache_dir: str, phase: str) -> dict:
    """Spawn a fresh python process for one compile phase (true cold/warm)."""
    env = dict(os.environ)
    env["_COMPILE_PHASE"] = phase
    env["_COMPILE_CACHE_DIR"] = cache_dir
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
    )
    # The child prints exactly one JSON line prefixed with RESULT_JSON=.
    line = None
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT_JSON="):
            line = ln[len("RESULT_JSON=") :]
    if line is None:
        raise RuntimeError(
            f"child phase={phase} produced no result.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-2000:]}"
        )
    return json.loads(line)


def main() -> int:
    # Child entry?
    phase = os.environ.get("_COMPILE_PHASE")
    if phase in ("cold", "warm"):
        cache_dir = os.environ["_COMPILE_CACHE_DIR"]
        res = _build_and_compile(cache_dir, phase)
        print("RESULT_JSON=" + json.dumps(res))
        return 0

    # Orchestrator entry.
    out_path = Path(__file__).with_name("compile_speed.json")
    cache_dir = tempfile.mkdtemp(prefix="gpuwrf_v0130_compile_")
    try:
        # Start with an empty cache for a genuine cold compile.
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)

        t0 = time.perf_counter()
        cold = _run_child(cache_dir, "cold")
        warm = _run_child(cache_dir, "warm")
        wall = time.perf_counter() - t0

        cold_s = cold["compile_seconds"]
        warm_s = warm["compile_seconds"]
        speedup = (cold_s / warm_s) if warm_s > 0 else float("inf")

        identical = (
            cold["result_hash"] == warm["result_hash"]
            and cold["result_shape"] == warm["result_shape"]
        )

        report = {
            "title": "v0.13 compile-speed proof (cold vs warm, CPU)",
            "backend": cold.get("backend"),
            "representative_graph": {
                "kind": "fp64 lax.scan(stencil + GEMM)",
                "substeps": REPR_SUBSTEPS,
                "field_edge": REPR_N,
                "note": "Representative of WRF timestep XLA compile cost drivers; "
                "NOT a forecast-accuracy artifact.",
            },
            "cold": cold,
            "warm": warm,
            "cold_compile_seconds": cold_s,
            "warm_compile_seconds": warm_s,
            "warm_speedup_x": speedup,
            "warm_is_cache_hit": bool(warm["warm_hit"]),
            "cold_wrote_entries": cold["new_entries"],
            "identical_results": bool(identical),
            "orchestrator_wall_seconds": wall,
            "gpu_followup": (
                "GPU cold-compile (multi-minute fp64 + autotuning) and the "
                "persistent autotune-cache numbers are a v0.13-GPU-followup; not "
                "claimed here. CPU backend used because the GPU is reserved for "
                "the validation gate."
            ),
            "passed": bool(
                identical
                and warm["warm_hit"]
                and warm_s < cold_s
                and cold["precompile_error"] is None
                and warm["precompile_error"] is None
            ),
        }

        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"\nWROTE {out_path}")
        print(
            f"COLD={cold_s:.4f}s  WARM={warm_s:.4f}s  speedup={speedup:.2f}x  "
            f"warm_hit={warm['warm_hit']}  identical={identical}  "
            f"PASSED={report['passed']}"
        )
        return 0 if report["passed"] else 1
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
