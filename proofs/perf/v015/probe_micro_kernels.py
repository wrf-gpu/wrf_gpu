#!/usr/bin/env python
"""v0.15 kernel probe — microbenchmarks for roofline + scaling + scan variants.

All synthetic (no case build), GPU-only. Measures, per (grid, dtype):
  1. advance_w_wrf (production import, exact operational flags) wall per call
  2. Thomas fwd+back standalone: lax.scan unroll=1/4/8/full + associative_scan
     + bytes-touch bandwidth floor; bitwise check unroll-vs-baseline
  3. bandwidth probes: 1R1W elementwise, 6R1W vertical-stencil chain
  4. ALU probes at 512^2x45: div / exp / pow / sqrt / 16-FMA chain, fp64 vs fp32

Artifact: proofs/perf/v015/micro_kernels.json
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.core.advance_w import advance_w_wrf

HERE = Path(__file__).resolve().parent
PROOF = HERE / "micro_kernels.json"

NZ = 44  # mass levels; faces NZ+1 — matches the Switzerland d01 case


def _timed(fn, *args, reps=20, warm=3):
    for _ in range(warm):
        out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)  # ms


def _rng_fields(n, dtype, seed=0):
    rng = np.random.default_rng(seed)

    def f3(nlev):
        return jnp.asarray(1.0 + 0.01 * rng.standard_normal((nlev, n, n)), dtype=dtype)

    def f2():
        return jnp.asarray(1.0 + 0.01 * rng.standard_normal((n, n)), dtype=dtype)

    return f3, f2, rng


def bench_advance_w(n, dtype):
    f3, f2, rng = _rng_fields(n, dtype)
    nzf = NZ + 1
    kw = dict(
        w=f3(nzf), rw_tend=0.001 * f3(nzf), ww=0.01 * f3(nzf),
        u=jnp.asarray(1.0 + 0.01 * rng.standard_normal((NZ, n, n + 1)), dtype=dtype),
        v=jnp.asarray(1.0 + 0.01 * rng.standard_normal((NZ, n + 1, n)), dtype=dtype),
        mu_work=0.01 * f2(), mut=1e4 * f2(), muave=0.01 * f2(), muts=1e4 * f2(),
        t_2ave=f3(NZ), t_2=f3(NZ), t_1=f3(NZ),
        ph=f3(nzf), ph_1=f3(nzf), phb=1e3 * f3(nzf), ph_tend=0.001 * f3(nzf),
        ht=100.0 * f2(), c2a=f3(NZ), cqw=f3(NZ), alt=f3(NZ),
        a=0.1 * f3(nzf), alpha=0.9 * f3(nzf), gamma=0.1 * f3(nzf),
        c1h=jnp.asarray(np.linspace(1.0, 0.1, NZ), dtype=dtype),
        c2h=jnp.asarray(np.linspace(0.0, 100.0, NZ), dtype=dtype),
        c1f=jnp.asarray(np.linspace(1.0, 0.1, nzf), dtype=dtype),
        c2f=jnp.asarray(np.linspace(0.0, 100.0, nzf), dtype=dtype),
        rdnw=jnp.asarray(np.full(NZ, 44.0), dtype=dtype),
        rdn=jnp.asarray(np.full(NZ, 44.0), dtype=dtype),
        fnm=jnp.asarray(np.full(NZ, 0.5), dtype=dtype),
        fnp=jnp.asarray(np.full(NZ, 0.5), dtype=dtype),
        cf1=jnp.asarray(1.5, dtype=dtype), cf2=jnp.asarray(-0.5, dtype=dtype),
        cf3=jnp.asarray(0.0, dtype=dtype),
        msftx=f2(), msfty=f2(),
        w_save=0.01 * f3(nzf),
    )
    fixed = dict(
        rdx=1.0 / 7700.0, rdy=1.0 / 7700.0, dts=1.8, epssm=0.5, top_lid=True,
        damp_opt=3, dampcoef=0.2, zdamp=5000.0, w_damping=1,
    )
    fn = jax.jit(lambda kw_: advance_w_wrf(**kw_, **fixed))
    ms = _timed(fn, kw)
    # ideal-bandwidth floor: 17 x 3D-field reads + 3 writes
    bytes_touched = (11 * (NZ + 1) + 6 * NZ + 3 * (NZ + 1)) * n * n * jnp.dtype(dtype).itemsize
    return ms, int(bytes_touched)


# ---- Thomas standalone variants (exact advance_w formulation) ----
def _thomas(a, alpha, gamma, w_next, unroll):
    nz = w_next.shape[0] - 1

    def _fwd(prev, e):
        a_k, al_k, w_k = e
        out = (w_k - a_k * prev) * al_k
        return out, out

    _, tail = jax.lax.scan(_fwd, w_next[0], (a[1:], alpha[1:], w_next[1:]), unroll=unroll)
    w_fwd = jnp.concatenate((w_next[0][None], tail), axis=0)

    def _back(nxt, e):
        g_k, w_k = e
        out = w_k - g_k * nxt
        return out, out

    gamma_rev = gamma[1:nz][::-1]
    w_rev = w_fwd[1:nz][::-1]
    _, rev = jax.lax.scan(_back, w_fwd[nz], (gamma_rev, w_rev), unroll=unroll)
    interior = rev[::-1]
    return jnp.concatenate((w_fwd[0][None], interior, w_fwd[nz][None]), axis=0)


def _thomas_assoc(a, alpha, gamma, w_next):
    """Associative-scan formulation of the same two recurrences (Stream-B only:
    reassociates fp arithmetic). x_k = A_k*x_{k-1} + B_k with A=-a*alpha, B=w*alpha."""
    nz = w_next.shape[0] - 1
    A = -a[1:] * alpha[1:]
    B = w_next[1:] * alpha[1:]

    def comp(c1, c2):
        a1, b1 = c1
        a2, b2 = c2
        return a2 * a1, a2 * b1 + b2

    Acum, Bcum = jax.lax.associative_scan(comp, (A, B), axis=0)
    tail = Acum * w_next[0][None] + Bcum
    w_fwd = jnp.concatenate((w_next[0][None], tail), axis=0)
    A2 = -gamma[1:nz][::-1]
    B2 = w_fwd[1:nz][::-1]
    A2c, B2c = jax.lax.associative_scan(comp, (A2, B2), axis=0)
    rev = A2c * w_fwd[nz][None] + B2c
    interior = rev[::-1]
    return jnp.concatenate((w_fwd[0][None], interior, w_fwd[nz][None]), axis=0)


def bench_thomas(n, dtype):
    f3, _, _ = _rng_fields(n, dtype, seed=1)
    nzf = NZ + 1
    a = 0.1 * f3(nzf)
    alpha = 0.9 * f3(nzf)
    gamma = 0.1 * f3(nzf)
    w = f3(nzf)
    out = {}
    ref = None
    for unroll in (1, 4, 8, NZ + 1):
        fn = jax.jit(partial(_thomas, unroll=unroll))
        out[f"scan_unroll{unroll}_ms"] = _timed(fn, a, alpha, gamma, w)
        val = np.asarray(fn(a, alpha, gamma, w))
        if ref is None:
            ref = val
            out[f"bitwise_vs_unroll1_u{unroll}"] = True
        else:
            out[f"bitwise_vs_unroll1_u{unroll}"] = bool((val == ref).all())
    fn = jax.jit(_thomas_assoc)
    out["assoc_scan_ms"] = _timed(fn, a, alpha, gamma, w)
    val = np.asarray(fn(a, alpha, gamma, w))
    err = np.abs(val - ref) / np.maximum(np.abs(ref), 1e-30)
    out["assoc_rel_err_max"] = float(err.max())
    # bandwidth floor: touch the same 4 arrays once
    fn_bw = jax.jit(lambda a_, al_, g_, w_: a_ + al_ + g_ + w_)
    out["bytes_touch_floor_ms"] = _timed(fn_bw, a, alpha, gamma, w)
    out["bytes"] = int(4 * nzf * n * n * jnp.dtype(dtype).itemsize)
    return out


def bench_bandwidth(n, dtype):
    f3, _, _ = _rng_fields(n, dtype, seed=2)
    x = f3(NZ + 1)
    o = {}
    fn1 = jax.jit(lambda x_: x_ * 1.0001 + 0.5)
    o["copy_1r1w_ms"] = _timed(fn1, x)
    o["copy_bytes"] = int(2 * x.size * x.dtype.itemsize)

    def stencil(x_):
        return (
            x_
            + 0.5 * jnp.roll(x_, 1, 0)
            + 0.25 * jnp.roll(x_, -1, 0)
            + 0.125 * jnp.roll(x_, 1, 1)
            + 0.0625 * jnp.roll(x_, -1, 1)
            + 0.03125 * jnp.roll(x_, 1, 2)
        )

    fn2 = jax.jit(stencil)
    o["stencil6_ms"] = _timed(fn2, x)
    o["stencil_bytes"] = int(2 * x.size * x.dtype.itemsize)  # 1R (cached neighbors) + 1W
    return o


def bench_alu(n, dtype):
    f3, _, _ = _rng_fields(n, dtype, seed=3)
    a = f3(NZ + 1)
    b = f3(NZ + 1) + 0.5
    o = {"elements": int(a.size)}
    o["div_ms"] = _timed(jax.jit(lambda a_, b_: a_ / b_), a, b)
    o["exp_ms"] = _timed(jax.jit(lambda a_: jnp.exp(a_)), a)
    o["pow1p4_ms"] = _timed(jax.jit(lambda a_: a_ ** 1.4), a)
    o["sqrt_ms"] = _timed(jax.jit(lambda a_: jnp.sqrt(a_)), a)

    def fma16(a_, b_):
        x = a_
        for _ in range(16):
            x = x * b_ + a_
        return x

    o["fma16_ms"] = _timed(jax.jit(fma16), a, b)
    return o


def main():
    results = {"schema": "V015KernelProbeMicro", "gpu": jax.devices()[0].device_kind, "nz": NZ}
    for n in (128, 256, 512):
        for dt_name, dt in (("fp64", jnp.float64), ("fp32", jnp.float32)):
            key = f"{n}x{n}_{dt_name}"
            ms, bts = bench_advance_w(n, dt)
            results[key] = {
                "advance_w_ms": ms,
                "advance_w_min_bytes": bts,
                "advance_w_achieved_GBps_lower_bound": round(bts / ms / 1e6, 1),
                "thomas": bench_thomas(n, dt),
                "bandwidth": bench_bandwidth(n, dt),
            }
            print(key, json.dumps(results[key]["thomas"], indent=None)[:240], flush=True)
    results["alu_512_fp64"] = bench_alu(512, jnp.float64)
    results["alu_512_fp32"] = bench_alu(512, jnp.float32)
    PROOF.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
