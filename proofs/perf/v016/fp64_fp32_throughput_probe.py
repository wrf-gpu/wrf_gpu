"""Measure the TRUE compute-bound fp64:fp32 throughput ratio on this GPU,
and the double-single cost, controlling for the latency-bound trap.

The first probe (eft_double_single_gpu_probe) used a serial dependency chain and
measured fp64 only 3.44x slower than fp32 -- a latency ratio, not throughput.
This decides whether the project's "fp64 is 1/64 -> catastrophic" premise is the
real regime, by:
  - using a high-ILP kernel (independent lanes) so the GPU is throughput-bound,
  - varying the per-iteration coefficient with the loop index so XLA cannot
    close-form the linear recurrence,
  - confirming wall scales ~linearly with K (proof real iterations execute).
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

F32 = jnp.float32
F64 = jnp.float64
_b = lax.optimization_barrier


def _bench(fn, *args, warmup=3, reps=20):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


# Each element is an independent chain; coefficient varies with i (anti-fold).
@partial(jax.jit, static_argnums=(1,))
def loop32(x, K):
    def body(i, acc):
        c = F32(0.999) + F32(1e-4) * (i % 4).astype(F32)
        return _b(acc * c) + F32(1e-3)
    return jnp.sum(lax.fori_loop(0, K, body, x))


@partial(jax.jit, static_argnums=(1,))
def loop64(x, K):
    def body(i, acc):
        c = F64(0.999) + F64(1e-4) * (i % 4).astype(F64)
        return _b(acc * c) + F64(1e-3)
    return jnp.sum(lax.fori_loop(0, K, body, x))


def two_sum(a, b):
    s = _b(a + b); bb = _b(s - a)
    return s, _b((a - _b(s - bb)) + (b - bb))


_SPLIT = F32(4097.0)


def _split(a):
    c = _b(_SPLIT * a); big = _b(c - a); hi = _b(c - big)
    return hi, _b(a - hi)


def two_prod(a, b):
    p = _b(a * b); ah, al = _split(a); bh, bl = _split(b)
    return p, _b((((ah * bh - p) + ah * bl) + al * bh) + al * bl)


@partial(jax.jit, static_argnums=(1,))
def loopdd(x, K):
    def body(i, acc):
        xh, xl = acc
        c = F32(0.999) + F32(1e-4) * (i % 4).astype(F32)
        p, e = two_prod(xh, c)
        e = _b(e + xl * c)
        xh, xl = two_sum(p, e)              # *c
        s, e2 = two_sum(xh, F32(1e-3))
        e2 = _b(e2 + xl)
        xh, xl = two_sum(s, e2)             # + 1e-3
        return (xh, xl)
    xh, xl = lax.fori_loop(0, K, body, (x, jnp.zeros_like(x)))
    return jnp.sum(xh) + jnp.sum(xl)


def main():
    dev = jax.devices()[0]
    out = {"device": str(dev), "platform": dev.platform}
    N = 1 << 23  # 8.4M independent chains -> ample ILP for throughput
    x32 = jax.random.uniform(jax.random.PRNGKey(3), (N,), dtype=F32, minval=0.5, maxval=1.5)
    x64 = x32.astype(F64)

    rows = {}
    for K in (128, 512, 2048):
        t32 = _bench(loop32, x32, K)
        t64 = _bench(loop64, x64, K)
        tdd = _bench(loopdd, x32, K)
        rows[str(K)] = {
            "fp32_s": t32, "fp64_s": t64, "dd_s": tdd,
            "fp64_over_fp32": t64 / t32,
            "dd_over_fp32": tdd / t32,
            "fp64_over_dd": t64 / tdd,
        }
    out["throughput_by_K"] = rows

    # K-scaling sanity: fp32 wall(512)/wall(128) should be ~4 if real iterations run
    out["k_scaling_fp32_512_over_128"] = rows["512"]["fp32_s"] / rows["128"]["fp32_s"]
    out["k_scaling_fp64_512_over_128"] = rows["512"]["fp64_s"] / rows["128"]["fp64_s"]

    big = rows["2048"]
    out["verdict"] = {
        "true_fp64_penalty_x": big["fp64_over_fp32"],
        "dd_cost_x_fp32": big["dd_over_fp32"],
        "dd_beats_fp64": big["fp64_over_dd"] > 1.0,
        "fp64_over_dd": big["fp64_over_dd"],
        "reading": (
            "If true_fp64_penalty_x >> dd_cost_x_fp32, DD-fp32 wins on throughput-bound "
            "compute. If true_fp64_penalty_x is small (~3-4x), fp64 is cheap on this GPU "
            "and the fp32 win is VRAM/storage, not arithmetic speed."
        ),
    }
    print(json.dumps(out, indent=2))
    Path("proofs/perf/v016/fp64_fp32_throughput_probe.json").write_text(
        json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
