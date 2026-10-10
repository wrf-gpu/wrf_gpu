"""Run the PRODUCT CLI (gpuwrf.cli.main) on CPU for the real Swiss case (no GPU, never via pytest).

CPU harness patches (identical in every arm, disclosed): State constructors' GPU guard -> CPU device;
jax.lax.linalg.tridiagonal_solve -> pure-JAX Thomas scan (jaxlib dgtsv FFI deadlock, E105).
Opt-in for external kill: prctl(PR_SET_PTRACER_ANY) for stack dumps.
Usage: JAX_PLATFORMS=cpu python cli_cpu.py run --input-dir ... (any gpuwrf CLI args)
"""
from __future__ import annotations

import ctypes
import faulthandler
import os
import signal
import sys

faulthandler.register(signal.SIGUSR1, all_threads=True)
ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)
assert os.environ.get("JAX_PLATFORMS") == "cpu"

import gpuwrf  # noqa: F401
import jax
import jax.numpy as jnp
from jax import lax

assert jax.devices()[0].platform == "cpu"
jax.config.update("jax_cpu_enable_async_dispatch", False)
from gpuwrf.contracts import state as state_contract

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
    _, (cp, dp) = lax.scan(forward, init, (dl_t, d_t, du_t, b_t))

    def backward(x_next, row):
        cp_i, dp_i = row
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, xs = lax.scan(backward, jnp.zeros_like(b_t[0]), (cp, dp), reverse=True)
    return jnp.moveaxis(xs, 0, -2)


jax.lax.linalg.tridiagonal_solve = _thomas_tridiagonal_solve
print(f"CLI_CPU_PID {os.getpid()}", file=sys.stderr, flush=True)

from gpuwrf import cli

sys.exit(cli.main(sys.argv[1:]))
