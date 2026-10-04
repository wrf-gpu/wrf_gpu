"""Corrected GPU probe for error-free-transformation (double-single) arithmetic
as a numerically-valid fp32 substitute for the fp64 islands.

This is the GPU+dycore-pattern counterpart to `gpt_double_single_probe.py`,
which was (a) CPU-only, where fp32==fp64 wall so its "16x" cost is measured
against a baseline on which fp64 is free; (b) limited to one-shot spatial
cancellation, never the time-accumulation pattern; (c) never run on the GPU.

We test three claims on the real device:

  E0  Raw fp64:fp32 throughput ratio on THIS GPU (compute-bound).
      If ~1/30..1/64, native fp64 is the bottleneck and an fp32-based
      emulation that costs <ratio flops is a net win.

  E1  ACCURACY. The two binding dynamical-core precision patterns:
        (a) pressure-gradient: (P0+aR) - (P0+aL) with P0~1e5, a~O(1).
        (b) acoustic accumulation: X += delta over many substeps, X~1e5.
      Compare fp64 (truth), naive-fp32, perturbation-fp32 (project's current
      "valid" frame), and double-single/compensated-fp32 (the proposed identity).

  E2  SPEED on the GPU. A compute-bound elementwise recurrence in fp32, fp64,
      and double-single. Claim: DD-fp32 wall < native-fp64 wall on the GeForce.

  E3  LONG-HORIZON drift. 200k-step accumulation: does compensated-fp32 stay
      pinned to fp64 while naive-fp32 drifts (the conservation/drift failure)?

Every error-free transformation is wrapped in lax.optimization_barrier and then
EMPIRICALLY VERIFIED exact (a+b == s+e in fp64). If XLA breaks the EFT, the
self-check fails and that is itself the reported blocker.

CPU fallback runs if no GPU is visible (accuracy still valid; speed is a GPU claim).
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # need fp64 for the truth reference
import jax.numpy as jnp
from jax import lax

F32 = jnp.float32
F64 = jnp.float64
_b = lax.optimization_barrier  # opaque barrier: stops XLA reassociating EFTs


# --------------------------------------------------------------------------- #
# Error-free transformations (Knuth TwoSum, Dekker TwoProduct), XLA-safe.
# --------------------------------------------------------------------------- #
def two_sum(a, b):
    """Exact: a + b = s + e, all fp32. 6 flops, branch-free (Knuth/Moller)."""
    s = _b(a + b)
    bb = _b(s - a)
    e = _b((a - _b(s - bb)) + (b - bb))
    return s, e


_SPLIT = F32(4097.0)  # 2^12 + 1 ; fp32 has 24 mantissa bits -> split at 12


def _veltkamp_split(a):
    c = _b(_SPLIT * a)
    big = _b(c - a)
    hi = _b(c - big)
    lo = _b(a - hi)
    return hi, lo


def two_prod(a, b):
    """Exact: a * b = p + e, all fp32, via Dekker split (no FMA dependency)."""
    p = _b(a * b)
    ahi, alo = _veltkamp_split(a)
    bhi, blo = _veltkamp_split(b)
    e = _b((((ahi * bhi - p) + ahi * blo) + alo * bhi) + alo * blo)
    return p, e


# double-single (hi, lo) value = hi + lo, |lo| <= 0.5 ulp(hi)
def dd_add_f32(xh, xl, y):
    """(xh,xl) + fp32 y  ->  (zh, zl)."""
    s, e = two_sum(xh, y)
    e = _b(e + xl)
    zh, zl = two_sum(s, e)  # renormalize
    return zh, zl


def dd_mul_f32(xh, xl, y):
    """(xh,xl) * fp32 y  ->  (zh, zl)."""
    p, e = two_prod(xh, y)
    e = _b(e + xl * y)
    zh, zl = two_sum(p, e)
    return zh, zl


def dd_sub_dd(ah, al, bh, bl):
    """(ah,al) - (bh,bl) -> (zh,zl)  (used for the exact gradient)."""
    s, e = two_sum(ah, -bh)
    e = _b((e + al) - bl)
    zh, zl = two_sum(s, e)
    return zh, zl


# --------------------------------------------------------------------------- #
# E0 + E2  GPU timing helpers
# --------------------------------------------------------------------------- #
def _bench(fn, *args, warmup=3, reps=30):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


@partial(jax.jit, static_argnums=(2,))
def _loop_scalar(x, d, K):
    def body(_, acc):
        return _b(acc * F32(0.9999)) + d
    out = lax.fori_loop(0, K, body, x)
    return jnp.sum(out)


@partial(jax.jit, static_argnums=(3,))
def _loop_scalar64(x, d, c, K):
    def body(_, acc):
        return _b(acc * c) + d
    out = lax.fori_loop(0, K, body, x)
    return jnp.sum(out)


@partial(jax.jit, static_argnums=(2,))
def _loop_dd(x, d, K):
    def body(_, acc):
        xh, xl = acc
        xh, xl = dd_mul_f32(xh, xl, F32(0.9999))
        xh, xl = dd_add_f32(xh, xl, d)
        return (xh, xl)
    xh, xl = lax.fori_loop(0, K, body, (x, jnp.zeros_like(x)))
    return jnp.sum(xh) + jnp.sum(xl)


def main():
    out = {"device": str(jax.devices()[0]), "platform": jax.devices()[0].platform}

    # ---- self-check: did XLA preserve the EFTs? -------------------------- #
    key = jax.random.PRNGKey(20260614)
    ka, kb = jax.random.split(key)
    a = (jax.random.normal(ka, (1_000_000,), dtype=F32) * F32(1e5))
    b = (jax.random.normal(kb, (1_000_000,), dtype=F32) * F32(1e2))
    s, e = jax.jit(two_sum)(a, b)
    exact = a.astype(F64) + b.astype(F64)
    recon = s.astype(F64) + e.astype(F64)
    two_sum_err = float(jnp.max(jnp.abs(exact - recon)))
    p, pe = jax.jit(two_prod)(a, b)
    exactp = a.astype(F64) * b.astype(F64)
    reconp = p.astype(F64) + pe.astype(F64)
    two_prod_err = float(jnp.max(jnp.abs(exactp - reconp) / jnp.abs(exactp)))
    out["eft_selfcheck"] = {
        "two_sum_max_abs_residual": two_sum_err,
        "two_prod_max_rel_residual": two_prod_err,
        "two_sum_exact": two_sum_err == 0.0,
        "two_prod_exact": two_prod_err == 0.0,
        "note": "0.0 == XLA preserved the error-free transformation under optimization_barrier",
    }

    # ---- E0: raw fp64:fp32 compute-bound throughput on THIS GPU ---------- #
    N = 1 << 24
    x32 = jax.random.normal(jax.random.PRNGKey(1), (N,), dtype=F32)
    x64 = x32.astype(F64)
    d32 = F32(1e-3)
    K = 256
    t32 = _bench(_loop_scalar, x32, d32, K)
    t64 = _bench(_loop_scalar64, x64, F64(1e-3), F64(0.9999), K)
    out["E0_raw_throughput"] = {
        "N": N, "K": K,
        "fp32_wall_s": t32, "fp64_wall_s": t64,
        "fp64_over_fp32_slowdown": t64 / t32,
        "interpretation": "how many x slower native fp64 is than fp32 on this device (compute-bound)",
    }

    # ---- E1: accuracy on the two dynamical patterns --------------------- #
    # (a) pressure-gradient cancellation: p = P0 + anomaly, P0 ~ 1e5
    P0 = 1.0e5
    kL, kR = jax.random.split(jax.random.PRNGKey(7))
    aL = jax.random.normal(kL, (2_000_000,), dtype=F64) * 3.0   # O(1) anomaly
    aR = jax.random.normal(kR, (2_000_000,), dtype=F64) * 3.0
    grad_truth = aR - aL                                        # fp64 truth

    pL64, pR64 = P0 + aL, P0 + aR
    # naive fp32: store the TOTAL in fp32, then difference
    grad_naive = (pR64.astype(F32) - pL64.astype(F32)).astype(F64)
    # perturbation fp32: store (total - base) in fp32, base fp64 scalar
    grad_pert = ((pR64 - P0).astype(F32) - (pL64 - P0).astype(F32)).astype(F64)
    # double-single: store total as DD pair, exact DD difference
    def dd_from64(v):
        hi = v.astype(F32)
        lo = (v - hi.astype(F64)).astype(F32)
        return hi, lo
    Lh, Ll = dd_from64(pL64)
    Rh, Rl = dd_from64(pR64)
    gh, gl = jax.jit(dd_sub_dd)(Rh, Rl, Lh, Ll)
    grad_dd = (gh.astype(F64) + gl.astype(F64))

    def rmse(x):
        return float(jnp.sqrt(jnp.mean((x - grad_truth) ** 2)))
    out["E1a_pressure_gradient_rmse_vs_fp64"] = {
        "magnitude_P0": P0, "anomaly_sigma": 3.0,
        "naive_fp32": rmse(grad_naive),
        "perturbation_fp32": rmse(grad_pert),
        "double_single_fp32": rmse(grad_dd),
    }

    # (b) acoustic accumulation: X = X0 (~1e5) += delta over many substeps
    Nsub = 20_000
    kd = jax.random.PRNGKey(11)
    deltas = jax.random.normal(kd, (Nsub,), dtype=F64) * 0.05      # acoustic ±0.05
    drift = 2.0e-4                                                  # slow signal/step
    deltas = deltas + drift
    X0 = 1.0e5

    # fp64 truth
    xt = X0
    for chunk in jnp.array_split(deltas, 1):
        xt = xt + jnp.sum(chunk)
    truth_final = float(X0 + jnp.sum(deltas))

    # naive fp32 sequential
    def acc_fp32(carry, d):
        return (carry + F32(d)), None
    xf32, _ = lax.scan(acc_fp32, F32(X0), deltas.astype(F32))
    # compensated (Kahan) fp32 sequential
    def acc_kahan(carry, d):
        s, c = carry
        y = _b(F32(d) - c)
        t = _b(s + y)
        c = _b(_b(t - s) - y)
        return (t, c), None
    (xk, ck), _ = lax.scan(acc_kahan, (F32(X0), F32(0.0)), deltas.astype(F32))
    out["E1b_acoustic_accumulation"] = {
        "X0": X0, "Nsub": Nsub, "true_total_drift_signal": Nsub * drift,
        "fp64_final": truth_final,
        "naive_fp32_final": float(xf32),
        "naive_fp32_abs_err": abs(float(xf32) - truth_final),
        "compensated_fp32_final": float(xk) + float(ck),
        "compensated_fp32_abs_err": abs((float(xk) + float(ck)) - truth_final),
    }

    # ---- E2: GPU speed, DD-fp32 vs native fp64 -------------------------- #
    tdd = _bench(_loop_dd, x32, d32, K)
    out["E2_gpu_speed"] = {
        "N": N, "K": K,
        "fp32_wall_s": t32, "fp64_wall_s": t64, "double_single_wall_s": tdd,
        "dd_over_fp32_cost": tdd / t32,
        "fp64_over_dd_speedup": t64 / tdd,
        "claim": "fp64_over_dd_speedup > 1 means DD-fp32 BEATS native fp64 on this GPU",
    }

    # ---- E3: long-horizon drift ----------------------------------------- #
    Nlong = 200_000
    dl = jax.random.normal(jax.random.PRNGKey(99), (Nlong,), dtype=F64) * 0.05 + 1.0e-4
    truth_long = float(1.0e5 + jnp.sum(dl))
    xf, _ = lax.scan(acc_fp32, F32(1.0e5), dl.astype(F32))
    (xkl, ckl), _ = lax.scan(acc_kahan, (F32(1.0e5), F32(0.0)), dl.astype(F32))
    out["E3_long_horizon_200k"] = {
        "Nsteps": Nlong,
        "fp64_final": truth_long,
        "naive_fp32_abs_err": abs(float(xf) - truth_long),
        "compensated_fp32_abs_err": abs((float(xkl) + float(ckl)) - truth_long),
        "true_signal_magnitude": Nlong * 1.0e-4,
    }

    print(json.dumps(out, indent=2))
    Path("proofs/perf/v016/eft_double_single_gpu_probe.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    return out


if __name__ == "__main__":
    main()
