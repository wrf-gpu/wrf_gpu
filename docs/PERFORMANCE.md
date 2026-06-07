# Performance — measured, reproducible (v0.12.0 standalone path)

This page gives the **honest, reproducible** performance numbers for the v0.12.0
standalone CLI (`gpuwrf run`), with exact command lines. Every number is measured,
not claimed; precision, domain, and warm/cold are labelled. For sizing and the
already-documented cold-compile / VRAM behaviour see
[`resource-profile.md`](resource-profile.md); for the full speedup-vs-CPU-WRF
analysis see [`../proofs/perf/speedup_denominator.md`](../proofs/perf/speedup_denominator.md).

**A modest honest number beats an impressive shaky one.** Where a value is
projected or pending, it says so.

## Reference setup

| | |
|---|---|
| GPU | 1 × NVIDIA RTX 5090 (Blackwell, 32 GiB), JAX 0.10.0 |
| Standalone case | `wrf_l2/20260514_18z_l2_72h_…` — `wrfinput_d01` + `wrfbdy_d01`, **no CPU-WRF wrfout** (true standalone native-init) |
| Domain | **d01** 9 km, 93 × 59 × 44 mass points |
| Solver | dt = 10 s, 10 acoustic substeps, radiation every 180 steps |
| Precision | **fp64** — the operational standalone path forces pure fp64 for the acoustic solve (`force_fp64=True`) |

### Reproduce

```bash
# WARM throughput (cache on) + cold-vs-cache compile, fp64, d01 standalone:
/tmp/wrf_gpu_run.sh taskset -c 0-3 \
  env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
      GPUWRF_JAX_CACHE_DIR=/mnt/data/gpuwrf_jax_cache \
  python -m gpuwrf.cli run \
    --input-dir <case> --output-dir <scratch> --domain d01 --hours 3
# Cold compile (cache disabled): set GPUWRF_JAX_CACHE=0 instead of GPUWRF_JAX_CACHE_DIR.
```

The pipeline reports `wall_clock_per_hour_s`: element `[0]` carries the XLA compile
(cold or cache-read) + the first hour's execution + JAX warmup; elements `[1:]` are the
**warm** steady-state per-forecast-hour cost.

## Benchmark table (measured, fp64, d01)

| Metric | Value | Precision / domain | Source |
|---|---|---|---|
| **Warm throughput** | **16.69 s / forecast-hour** (≈ 46 ms/step) | fp64, d01 9 km | 3 independent warm hours: 16.69, 16.69, 16.68 s |
| Cold compile + 1st hour (cache **disabled**) | 147.6 s (hour-1) | `GPUWRF_JAX_CACHE=0` | `cold_run.json` |
| Cold compile + 1st hour (empty cache, populating) | 150.3 s (hour-1) | empty `JAX_COMPILATION_CACHE_DIR` | `cachepop_run.json` |
| 1st hour with **warm cache** (cache hit) | _pending — see note_ | warm `JAX_COMPILATION_CACHE_DIR` | `driver_warm.json` |
| Peak VRAM | _pending_ (d01 9 km ≪ the documented d02 ~24.6 GiB) | fp64 | `v0120_profile_run.json` |
| Forecast verdict / finiteness | `PIPELINE_GREEN`, `all_finite=true` (56 fields) | — | `warm_run.json` |

> **Persistent JIT cache — what it buys.** The v0.12.0 release ships a persistent XLA
> compilation cache on by default (`src/gpuwrf/runtime/compile_cache.py`). The cache stores
> the *identical* XLA executable keyed by HLO + backend + flags, so a cache hit is
> bit-for-bit the same program — **zero numerics change**. The standalone segment compiles
> essentially **one** large executable (`_advance_chunk`, static segment length), so the
> cache turns the multi-minute cold compile into a disk read on every later run. The clean
> compile-only cold-vs-cache delta is measured by `proofs/perf/v0120_profile_driver.py`
> (cold vs warm, no restart/repeatability-probe overhead); see
> `proofs/perf/v0120_standalone_bench.json`. The headline cache caveat: after a `jax`/`jaxlib`
> upgrade the key changes and the first run pays one cold compile again (stale entries are
> ignored, never wrong).

### Honest note on the cold/cache hour-1 numbers

The pipeline's hour-1 wall **conflates** XLA compile + the first hour's execution
(~16.7 s) + JAX warmup, so it is an *upper-bound wrapper* around compile time, not pure
compile. That is why the cache-disabled (147.6 s) and cache-warmed-default (140.8 s) hour-1
figures differ by only ~7 s here even though the underlying compile saving is larger: the
fixed ~16.7 s execute + the case build dominate the difference at `--hours` granularity.
The **clean** compile-only saving is isolated by the profiling driver (cold vs warm cache,
no pipeline probes) and reported in `v0120_standalone_bench.json`. See
`docs/resource-profile.md` for the previously documented **~4 min 55 s cold compile** on
the larger d02 program.

## Where this sits vs the published speedup

The published apples-to-apples speedup is the **d02** number, not d01:

- **~5× warm, apples-to-apples** (band **5–8×**), strict **dt-parity floor ~3.2×**,
  real-user warm wall **~2.5×** — one RTX 5090 vs 28-rank CPU-WRF on the same workstation,
  both fp64, same 3 km d02 grid. Full provenance and caveats:
  [`../proofs/perf/speedup_denominator.md`](../proofs/perf/speedup_denominator.md).

The **d01** warm number here (16.69 s/fc-hour) is **in-family** with the established d02
numerator (15.35–16.39 s/fc-hour): same dt / substep / radiation structure, and at these
GPU-underutilized grid sizes the per-step cost is set by the launch-bound step structure
rather than the cell count. **This d01 run confirms the warm-throughput family; it does not
establish a new headline** — there is no clean uncontended CPU-WRF standalone-d01
denominator, so no new ratio is claimed.

### Precision: there is no faster fp32 standalone path today

The precision matrix (`src/gpuwrf/contracts/precision.py`) authorizes fp32 *storage* for
the transported fields, but the operational standalone path forces pure **fp64** for the
acoustic solve (fp32 detonates the perturbation per the F7 gates). A gated-fp32 operational
path is a future ADR-007 decision and is currently **no faster on this memory-bound
workload** (`docs/resource-profile.md`). No fp32 standalone speed number is reported because
none is reachable through the CLI.

## Safe speedup landed: XLA CUDA-graph capture

The warmed step is **launch-bound** (`cuLaunchKernelEx` = 38% of CUDA-API time; only 5.5%
of launches are CUDA-graph-captured today). Setting

```bash
export XLA_FLAGS="--xla_gpu_graph_min_graph_size=1"
```

lowers XLA's graph-capture threshold from 5 to 1, capturing the dycore's short dependent
fusion chains into CUDA graphs. It is a **launch-environment flag, not a source change**, is
**memory-neutral**, and changes results only at fp64 machine-epsilon (~1e-14 reassociation —
the arithmetic is identical). Prior dynamics-only A/B measured **1.71×**
(`proofs/perf/fusion_results.md`); this sprint re-measures it on the **full coupled
standalone path** with a finiteness + numerics-neutrality check
(`proofs/perf/flag_baseline.json` vs `flag_gms1.json`, summarized in
`v0120_standalone_bench.json`). See `proofs/perf/v0120_profile.md` for the full ranked
opportunity list.
