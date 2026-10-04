"""Independent audit for FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md.

This script treats the challenge document and its JSONs as untrusted.  It runs
small independent GPU probes, then recomputes the performance implications from
raw v015/v016 artifacts.
"""
from __future__ import annotations

import csv
import json
import math
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "proofs/perf/v016/fp32_challenge_independent_audit.json"

F32 = jnp.float32
F64 = jnp.float64
barrier = lax.optimization_barrier


def bench(fn, *args, warmup: int = 2, reps: int = 7) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


@partial(jax.jit, static_argnums=(1,))
def fma_chain32(x, k: int):
    def body(i, acc):
        c = F32(0.9975) + F32(3.0e-5) * (i % 7).astype(F32)
        return barrier(acc * c + F32(1.0e-3))

    return jnp.sum(lax.fori_loop(0, k, body, x))


@partial(jax.jit, static_argnums=(1,))
def fma_chain64(x, k: int):
    def body(i, acc):
        c = F64(0.9975) + F64(3.0e-5) * (i % 7).astype(F64)
        return barrier(acc * c + F64(1.0e-3))

    return jnp.sum(lax.fori_loop(0, k, body, x))


def two_sum(a, b):
    s = barrier(a + b)
    bb = barrier(s - a)
    e = barrier((a - barrier(s - bb)) + (b - bb))
    return s, e


SPLIT = F32(4097.0)


def split_f32(a):
    c = barrier(SPLIT * a)
    big = barrier(c - a)
    hi = barrier(c - big)
    return hi, barrier(a - hi)


def two_prod(a, b):
    p = barrier(a * b)
    ah, al = split_f32(a)
    bh, bl = split_f32(b)
    e = barrier((((ah * bh - p) + ah * bl) + al * bh) + al * bl)
    return p, e


@partial(jax.jit, static_argnums=(1,))
def dd_chain(x, k: int):
    def body(i, acc):
        hi, lo = acc
        c = F32(0.9975) + F32(3.0e-5) * (i % 7).astype(F32)
        p, pe = two_prod(hi, c)
        pe = barrier(pe + lo * c)
        hi, lo = two_sum(p, pe)
        s, se = two_sum(hi, F32(1.0e-3))
        se = barrier(se + lo)
        hi, lo = two_sum(s, se)
        return hi, lo

    hi, lo = lax.fori_loop(0, k, body, (x, jnp.zeros_like(x)))
    return jnp.sum(hi) + jnp.sum(lo)


@jax.jit
def memory_add32(a, b):
    return jnp.sum(barrier(a + b))


@jax.jit
def memory_add64(a, b):
    return jnp.sum(barrier(a + b))


@jax.jit
def dd_sub(ah, al, bh, bl):
    s, e = two_sum(ah, -bh)
    e = barrier((e + al) - bl)
    return two_sum(s, e)


def rmse(x, y) -> float:
    return float(jnp.sqrt(jnp.mean((x - y) ** 2)))


def accuracy_probe() -> dict:
    n = 1_000_000
    p0 = 100_000.0
    k1, k2 = jax.random.split(jax.random.PRNGKey(901))
    a_l = jax.random.normal(k1, (n,), dtype=F64) * 3.0
    a_r = jax.random.normal(k2, (n,), dtype=F64) * 3.0
    truth = a_r - a_l

    total_l = p0 + a_l
    total_r = p0 + a_r
    naive = (total_r.astype(F32) - total_l.astype(F32)).astype(F64)
    perturb = (a_r.astype(F32) - a_l.astype(F32)).astype(F64)

    l_hi = total_l.astype(F32)
    l_lo = (total_l - l_hi.astype(F64)).astype(F32)
    r_hi = total_r.astype(F32)
    r_lo = (total_r - r_hi.astype(F64)).astype(F32)
    d_hi, d_lo = dd_sub(r_hi, r_lo, l_hi, l_lo)
    dd = d_hi.astype(F64) + d_lo.astype(F64)

    # Accumulation probe: large state plus many small increments.
    nsteps = 100_000
    deltas64 = jax.random.normal(jax.random.PRNGKey(902), (nsteps,), dtype=F64) * 0.05 + 1.0e-4
    truth_final = float(F64(p0) + jnp.sum(deltas64))

    def acc_naive(carry, d):
        return barrier(carry + d), None

    def acc_kahan(carry, d):
        s, c = carry
        y = barrier(d - c)
        t = barrier(s + y)
        c = barrier((t - s) - y)
        return (t, c), None

    naive_final, _ = lax.scan(acc_naive, F32(p0), deltas64.astype(F32))
    (k_sum, k_comp), _ = lax.scan(acc_kahan, (F32(p0), F32(0.0)), deltas64.astype(F32))
    kahan_final = float(k_sum) + float(k_comp)

    return {
        "pressure_gradient_rmse_vs_fp64": {
            "naive_total_fp32": rmse(naive, truth),
            "perturbation_fp32": rmse(perturb, truth),
            "double_single_total_fp32": rmse(dd, truth),
            "naive_over_perturbation": rmse(naive, truth) / rmse(perturb, truth),
            "perturbation_over_dd": rmse(perturb, truth) / rmse(dd, truth),
        },
        "accumulation": {
            "nsteps": nsteps,
            "signal_magnitude": nsteps * 1.0e-4,
            "truth_final": truth_final,
            "naive_fp32_abs_err": abs(float(naive_final) - truth_final),
            "kahan_fp32_abs_err": abs(kahan_final - truth_final),
            "naive_over_kahan_err": abs(float(naive_final) - truth_final)
            / abs(kahan_final - truth_final),
        },
    }


def microbench_probe() -> dict:
    n = 1 << 22
    key = jax.random.PRNGKey(903)
    x32 = jax.random.uniform(key, (n,), dtype=F32, minval=0.5, maxval=1.5)
    x64 = x32.astype(F64)
    y32 = jax.random.uniform(jax.random.PRNGKey(904), (n,), dtype=F32)
    y64 = y32.astype(F64)

    rows = {}
    for k in (256, 1024):
        t32 = bench(fma_chain32, x32, k)
        t64 = bench(fma_chain64, x64, k)
        tdd = bench(dd_chain, x32, k, reps=5)
        rows[str(k)] = {
            "fp32_s": t32,
            "fp64_s": t64,
            "dd_s": tdd,
            "fp64_over_fp32": t64 / t32,
            "dd_over_fp32": tdd / t32,
            "dd_over_fp64": tdd / t64,
        }

    m32 = bench(memory_add32, x32, y32)
    m64 = bench(memory_add64, x64, y64)

    # Same recurrence as the challenge throughput probe, swept over sizes.  This
    # is intentionally fp32/fp64 only: it tests whether the "4.3x" ratio is a
    # stable hardware constant or a shape/codegen-dependent kernel property.
    ratio_sweep = {}
    for n_sweep in (1 << 22, 1 << 23, 1 << 24):
        sx32 = jax.random.uniform(
            jax.random.PRNGKey(n_sweep), (n_sweep,), dtype=F32, minval=0.5, maxval=1.5
        )
        sx64 = sx32.astype(F64)
        for k_sweep in (512, 2048):
            t32 = bench(fma_chain32, sx32, k_sweep, reps=5)
            t64 = bench(fma_chain64, sx64, k_sweep, reps=5)
            ratio_sweep[f"N{n_sweep}_K{k_sweep}"] = {
                "fp32_s": t32,
                "fp64_s": t64,
                "fp64_over_fp32": t64 / t32,
            }

    return {
        "device": str(jax.devices()[0]),
        "platform": jax.devices()[0].platform,
        "n_elements": n,
        "fma_chain": rows,
        "memory_add": {
            "fp32_s": m32,
            "fp64_s": m64,
            "fp64_over_fp32": m64 / m32,
        },
        "fp64_fp32_ratio_sweep": ratio_sweep,
    }


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def csv_total_ns(path: str) -> tuple[float, int]:
    total = 0.0
    count = 0
    with (ROOT / path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += float(row["Total Time (ns)"])
            count += int(row.get("Instances") or row.get("Count") or row.get("Num Calls") or 0)
    return total, count


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    return ((sy * sxx - sx * sxy) / denom, (n * sxy - sx * sy) / denom)


def artifact_analysis() -> dict:
    true_fp32 = load_json("proofs/perf/v015/viability/true_fp32_cost_proxy.json")
    s2 = load_json("proofs/perf/v016/fp32_s2_mixed_ladder_final_2x2.json")
    hlo = load_json("proofs/perf/v016/s2_hlo_stats_2x2_s1_final.json")

    true_records = [(r["ncol"], r["ms_per_step"]) for r in true_fp32["records"] if r.get("ran_ok")]
    true_intercept, true_slope = linear_fit([(float(x), float(y)) for x, y in true_records])

    fp64_ref = true_fp32["fp64_refs"][0]["ms_per_step"]
    fp32_core = true_fp32["records"][0]["ms_per_step"]
    full_step = 119.8
    remainder = full_step - fp64_ref

    am = {}
    for rem_speed in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        new_ms = fp32_core + remainder / rem_speed
        am[str(rem_speed)] = {"new_ms": new_ms, "speedup": full_step / new_ms}

    fp64_record, mixed_record = s2["records"]
    fp64_hlo, mixed_hlo = hlo["records"]
    fp64_kernel_ns, fp64_kernel_instances = csv_total_ns(
        "proofs/perf/v016/nsys_fp64_2x2_s12_stats_cuda_gpu_kern_sum.csv"
    )
    mixed_kernel_ns, mixed_kernel_instances = csv_total_ns(
        "proofs/perf/v016/nsys_mixed_s2_2x2_s12_final_stats_cuda_gpu_kern_sum.csv"
    )

    return {
        "true_fp32_cost_proxy_fit": {
            "points": true_records,
            "intercept_ms": true_intercept,
            "slope_ms_per_1000_columns": true_slope * 1000.0,
            "fp64_core_ms_16k": fp64_ref,
            "true_fp32_core_ms_16k": fp32_core,
            "core_speedup_16k": fp64_ref / fp32_core,
        },
        "amdahl_from_full_step_119p8_ms": {
            "current_full_ms": full_step,
            "fp64_proxy_core_ms": fp64_ref,
            "true_fp32_proxy_core_ms": fp32_core,
            "unchanged_remainder_ms": remainder,
            "remainder_speed_scenarios": am,
        },
        "s2_measured": {
            "fp64_ms": fp64_record["ms_per_step"],
            "mixed_ms": mixed_record["ms_per_step"],
            "speedup": fp64_record["ms_per_step"] / mixed_record["ms_per_step"],
            "vram_ratio": fp64_record["peak_vram_gib"] / mixed_record["peak_vram_gib"],
            "fp64_kernel_total_s_12step": fp64_kernel_ns / 1e9,
            "mixed_kernel_total_s_12step": mixed_kernel_ns / 1e9,
            "kernel_speedup": fp64_kernel_ns / mixed_kernel_ns,
            "fp64_kernel_instances": fp64_kernel_instances,
            "mixed_kernel_instances": mixed_kernel_instances,
        },
        "s2_hlo_delta": {
            "temp_gib_fp64": fp64_hlo["memory_analysis"]["temp_size_in_bytes"] / 1024**3,
            "temp_gib_mixed": mixed_hlo["memory_analysis"]["temp_size_in_bytes"] / 1024**3,
            "temp_ratio": fp64_hlo["memory_analysis"]["temp_size_in_bytes"]
            / mixed_hlo["memory_analysis"]["temp_size_in_bytes"],
            "bytes_accessed_ratio": fp64_hlo["cost_analysis"]["bytes accessed"]
            / mixed_hlo["cost_analysis"]["bytes accessed"],
            "flops_ratio": fp64_hlo["cost_analysis"]["flops"] / mixed_hlo["cost_analysis"]["flops"],
            "f64_token_ratio": fp64_hlo["hlo_counts"]["dtype_tokens"]["f64"]
            / mixed_hlo["hlo_counts"]["dtype_tokens"]["f64"],
            "convert_ratio_mixed_over_fp64": mixed_hlo["hlo_counts"]["top_ops"]["convert"]
            / fp64_hlo["hlo_counts"]["top_ops"]["convert"],
        },
    }


def main() -> dict:
    result = {
        "schema": "FP32ChallengeIndependentAudit",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accuracy_probe": accuracy_probe(),
        "microbench_probe": microbench_probe(),
        "artifact_analysis": artifact_analysis(),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
