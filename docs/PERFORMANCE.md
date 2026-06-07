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
| 1st hour with **warm cache** (clean cache hit) | **29.3 s** (≈ compile-via-cache ~10 s + first-execute ~16.7 s) | warm `JAX_COMPILATION_CACHE_DIR`, no pipeline probes | `driver_warm.json` |
| **Peak VRAM** | **4.7 GiB** (4727 MB) | fp64, d01 9 km (≪ documented d02 ~24.6 GiB) | `driver_warm.json` |
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

> ¹ **Now measured (was deferred under GPU contention).** The clean cache-hit hour-1 (29.3 s),
> peak VRAM (4.7 GiB), and the coupled graph-capture A/B were all captured once the concurrent
> agents' 24 h jobs cleared the GPU lock. The clean cache-hit hour-1 of **29.3 s** (vs a >100 s
> cold compile — the full-pipeline cold hour-1 was 147.6 s, ~130 s of it compile) is the
> load-bearing cache win: the persistent cache turns a multi-minute cold compile into a
> **~10 s disk read**, bit-identical executable. Earlier deferral text below is superseded.

> ~~Deferred~~ The clean compile-only cache delta, peak VRAM, and the
> coupled graph-capture A/B all needed a free GPU; they were queued
> (`/mnt/data/wrf_perf_scratch/run_suite.sh`) but blocked on the one-GPU-at-a-time lock
> behind a concurrent agent's long-running 24 h job. The discipline was respected (no
> contended measurement). The directional cache evidence (147.6 s cache-off vs 140.8 s
> cache-on hour-1, plus the documented d02 ~4 min 55 s cold compile) and the bit-identical
> cache mechanism stand; the clean isolated numbers fill in when the GPU frees.

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

### Speedup reconciliation — why there are three different numbers (and none is dishonest)

Readers see three speedup numbers in this project. They differ because they measure
**different things**; this is the one place that reconciles them so a skeptic cannot claim
one contradicts another. All are one RTX 5090 vs 28-rank CPU-WRF on the same workstation,
both fp64, same d02 3 km grid.

| Number | Value | What it measures | Source |
|---|---|---|---|
| **Warm kernel (apples-to-apples)** | **~5×** (band **5–8×**, strict **dt-parity floor ~3.2×**) | **Compute-only** per-forecast-hour: warmed GPU step (compile excluded, IO excluded), GPU dt=10 s vs CPU dt=6 s, normalized to the same model time. The defensible engineering speedup. | [`../proofs/perf/speedup_denominator.md`](../proofs/perf/speedup_denominator.md) |
| **Warm real-user wall** | **~2.5×** | Full command-to-finish wall after the JIT cache is warm: includes IO + case build + JAX dispatch overhead, not just the kernel. | warm real-user d02, inherited |
| **Equivalence-demo real-user** | **~4.26× warm-cached**, **~1.70× cold** | The `equivalence_demo.py` 24 h d02 run: end-to-end GPU wall vs the retained CPU-WRF d02 solver wall (CPU dt=6 s × 14 400 steps from the RSL logs). | `proofs/v0120/equivalence_demo_20260509_d02_FINAL.json` (4.26×) and `…_d02.json` (1.70×) |

**How they relate, precisely:**

- **Cold vs warm in the equivalence demo (1.70× → 4.26×).** The *same* demo run took **1408.6 s**
  the first time (cold ~5-minute XLA compile included → **1.70×** vs CPU 2393.2 s) and
  **561.3 s** with a warm persistent JIT cache (cold compile replaced by a ~10 s cache read →
  **4.26×**). The difference is **entirely the persistent JIT cache plus IO/case-build**, not a
  numerics change — the cached executable is bit-identical to the cold one. So the cold number
  is the honest *first-run* experience; the warm number is the honest *every-subsequent-run*
  experience.
- **Equivalence-demo warm-cached (4.26×) vs warm real-user (~2.5×).** The equivalence demo's GPU
  wall is the **forecast-only** integration (561.3 s) divided by the CPU **d02-solver-only** wall
  (the RSL per-step timing for d02, excluding the CPU's own IO and the d01 parent). The ~2.5×
  warm real-user headline is a stricter, fuller-overhead command-to-finish ratio. They bound the
  real-user experience from both sides; the demo's solver-vs-solver framing is more favourable
  because it strips matched overhead from both numerator and denominator.
- **Warm real-user (~2.5×) vs warm kernel (~5×).** The kernel number is compute-only per model
  time; the real-user number adds IO, case build, and dispatch tax that the kernel number
  excludes. The gap between them is overhead, **not** a precision or correctness difference.
- **The dt trap.** CPU-WRF runs d02 at **dt=6 s** (parent_time_step_ratio=3); the warm-kernel
  numerator runs the GPU at **dt=10 s**. All speedup numbers are computed **per forecast-hour
  (same model time)**, never per-step — a per-step comparison would be meaningless and is
  reported only to expose the trap. The **dt-parity floor ~3.2×** is what remains if the GPU is
  forced to the CPU's 6 s step.

**Bottom line:** the honest, defensible engineering number is **~5× warm kernel (5–8× band, ~3.2×
dt-parity floor)**; the honest real-user experience is **~2.5× warm wall** (first run slower by the
cold compile, which the persistent JIT cache removes thereafter). The 4.26× / 1.70× equivalence-demo
numbers are the same physics measured solver-to-solver, warm and cold respectively. None contradicts
another.

### Precision: there is no faster fp32 standalone path today

The precision matrix (`src/gpuwrf/contracts/precision.py`) authorizes fp32 *storage* for
the transported fields, but the operational standalone path forces pure **fp64** for the
acoustic solve (fp32 detonates the perturbation per the F7 gates). A gated-fp32 operational
path is a future ADR-007 decision and is currently **no faster on this memory-bound
workload** (`docs/resource-profile.md`). No fp32 standalone speed number is reported because
none is reachable through the CLI.

## Safe-speedup candidate MEASURED — and rejected on the coupled path

The warmed step is **launch-bound** (`cuLaunchKernelEx` = 38% of CUDA-API time; only 5.5%
of launches are CUDA-graph-captured today), so `XLA_FLAGS=--xla_gpu_graph_min_graph_size=1`
(lower the CUDA-graph capture threshold 5→1) was the leading candidate — it gave **1.71×**
on the **dynamics-only** dycore in a prior sprint (`proofs/perf/fusion_results.md`).

**This sprint re-measured it on the full coupled standalone path and it is a regression, so
it was NOT landed:**

| Config | warm s/forecast-hour | finite | final state |
|---|---|---|---|
| Baseline (XLA defaults) | **16.71** | yes | reference |
| `--xla_gpu_graph_min_graph_size=1` | **19.81** (≈ **0.84×, ~19 % slower**) | yes | **bit-identical** (Δmax = Δmean = 0 on u/v/θ/w/φ/μ/qv) |

The dynamics-only 1.71× **does not carry to the coupled step**: with physics on, lowering
the capture threshold does not help the physics couplers and the extra graph-capture
overhead makes the step slower. It also lengthened cold compile (29 s → 137 s, the flag
changes the cache key). The flag is numerically safe (bit-identical) but simply not a
speedup here. **Recommendation: keep XLA defaults; do not set this flag.** Provenance:
`proofs/perf/flag_baseline.json` vs `flag_gms1.json`. The launch-tax reduction must instead
target the *coupled* graph (future work). See `proofs/perf/v0120_profile.md` for the full
ranked opportunity list.
