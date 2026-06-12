# Performance — measured, reproducible (v0.14)

**v0.14 headline: the GPU port runs the same 72 h forecast at parity with
28-rank CPU-WRF (~1.05×–1.06×).** v0.14 is a **memory + WRF-identity release, not
a performance release.** Completing the fully WRF-faithful dycore + physics
(v0.13/v0.14) raised per-step compute to parity — so the earlier multi-×
speedup numbers, which were measured on an **incomplete/faster dycore**, **no
longer reflect the shipped code and are not v0.14 claims.** Performance recovery
is the dedicated focus of **v0.15**.

This page gives the **honest, reproducible** performance numbers, with exact
command lines. Every number is measured, not claimed; precision, domain, and
warm/cold are labelled. For sizing and the cold-compile / VRAM behaviour see
[`resource-profile.md`](resource-profile.md).

**A modest honest number beats an impressive shaky one.** Where a value is
projected, pending, or superseded, it says so.

## v0.14 measured — at parity with CPU-WRF (the load-bearing number)

The two final 72 h GPU-vs-CPU-WRF field-parity gates were timed end-to-end on the
reference RTX 5090 workstation, both fp64, against retained 28-rank CPU-WRF
(V4.7.1) truth:

| Region | GPU wall | CPU wall (28-rank) | Ratio | Peak VRAM |
|---|---|---|---|---|
| **Switzerland d01** 72 h | **~2762 s** | **2906 s** | **~1.05×** | ~19.8 GiB |
| **Canary L2 d02** 72 h | **~8200 s** | **8713 s** | **~1.06×** | ~20.3 GiB |

This is **at parity** with CPU-WRF — not a multi-× speedup. The v0.14 perf-triage
(`proofs/perf/v014_perf_regression_triage.json`, Switzerland d01 128×128×44,
dt=18 s, 10 acoustic substeps) attributes the wall-clock cleanly and proves the
parity is real, not an instrumentation artifact:

| Component | Share of wall | What it is |
|---|---|---|
| **Deep-kernel steady state** | **~90%** | Genuine per-step compute, **~173 ms/step** (200 steps/forecast-hour) — the dominant cost on the finished dycore. |
| **Per-hour host overhead** | **~7%** | `finite_summary` full-state pulls (×2/hr) + `wrfout` write + land refresh + boundary rewindow — per **hour**, not per step. |
| **Compile (cold start)** | **~2.3%** | One-time XLA compile (~63 s of a 72 h run), removed on later runs by the persistent JIT cache. |

Because ~90% of the wall is genuine deep-kernel steady state, a multi-× warm-kernel
ratio is **mathematically incompatible** with the measured ~1.05× end-to-end: the
old "5× warm kernel" figure cannot coexist with a 1.05× wall on the same code, and
does **not** reflect v0.14.

**Why the regression vs earlier versions.** Earlier releases were timed on an
**incomplete/faster dycore** (fewer operators in the acoustic/`advance_w` solve,
a lighter physics fold). v0.13/v0.14 completed the WRF-faithful dynamics +
physics (per-substep `advance_w` with `w_damp`, the physics-`tendf` fold + 2-D
Smagorinsky on the default path, full guard passes), which raised per-step compute
to parity. The triage also found an **intrinsic double-compile** when a
mixed-precision (fp32/fp64) raw state is fed into the `force_fp64` forecast; the
identity-safe collapse of it shifts hour-1 by fp32-ε, so it was **not** landed in
v0.14. Making the **operational state fp64 end-to-end** removes both the
double-compile and the per-step precision converts — that is the flagged
**highest-leverage v0.15 lever** (a precision-matrix ADR), deferred precisely
because it changes the current baseline at fp32-ε.

**v0.15 performance plan (where the recovery comes from).** The triage's deferred
deep causes (`deep_causes_deferred_v015`) are the v0.15 worklist: the fp64
operational-state ADR (removes the double-compile + per-step converts), hoisting
the per-substep mass-denominator rebuild and `safe_*` floors out of the acoustic
inner loop where identity-safe, A/B-profiling the two Thomas `lax.scan` unrolls,
folding the per-hour double full-state host pull into the writer payload, and the
optional hand-fused-kernel branch. No v0.15 number is claimed here — it will be
re-measured honestly.

## Reference setup

| | |
|---|---|
| GPU | 1 × NVIDIA RTX 5090 (Blackwell, 32 GiB), JAX 0.10.0 |
| Standalone case | `wrf_l2/20260514_18z_l2_72h_…` — `wrfinput_d01` + `wrfbdy_d01`, **no CPU-WRF wrfout** (true standalone native-init) |
| Domain | **d01** 9 km, 93 × 59 × 44 mass points |
| Solver | dt = 10 s, 10 acoustic substeps, radiation every 180 steps |
| Precision | **fp64** — the operational standalone path forces pure fp64 for the acoustic solve (`force_fp64=True`) |

### Reproduce

No internal scheduler or GPU mutex is needed — on a single-GPU machine just run
`python -m gpuwrf.cli run` directly:

```bash
# WARM throughput (cache on) + cold-vs-cache compile, fp64, d01 standalone:
env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    GPUWRF_JAX_CACHE_DIR=/path/to/local/nvme/gpuwrf_jax_cache \
  python -m gpuwrf.cli run \
    --input-dir <case> --output-dir <scratch> --domain d01 --hours 3
# Cold compile (cache disabled): set GPUWRF_JAX_CACHE=0 instead of GPUWRF_JAX_CACHE_DIR.
```

> On the project's *shared* workstation the maintainers prefix this with their
> internal multi-job GPU serializer (`/tmp/wrf_gpu_run.sh taskset -c 0-3 ...`).
> That wrapper is **not** required to reproduce these numbers; prefix it only if
> it already exists on your machine.

The pipeline reports `wall_clock_per_hour_s`: element `[0]` carries the XLA compile
(cold or cache-read) + the first hour's execution + JAX warmup; elements `[1:]` are the
**warm** steady-state per-forecast-hour cost.

## Standalone-path benchmark table (d01) — JIT-cache + VRAM mechanism

> **Note on the throughput row.** The **16.69 s/forecast-hour** figure below was
> measured on the **earlier (v0.12.0) standalone d01 path**, before the
> v0.13/v0.14 dycore completion raised per-step compute. It is **superseded by the
> v0.14 parity measurement above** for the current shipped code and is retained
> here only as standalone-path/JIT-cache provenance. The **persistent JIT cache,
> cold/warm compile, and VRAM** behaviour in this table still holds in v0.14.

| Metric | Value | Precision / domain | Source |
|---|---|---|---|
| Warm throughput (**earlier v0.12.0 dycore — superseded**) | 16.69 s / forecast-hour (≈ 46 ms/step) | fp64, d01 9 km | 3 independent warm hours: 16.69, 16.69, 16.68 s |
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

## Historical speedup numbers — superseded by the v0.14 measurement

> **These numbers do NOT describe v0.14.** Earlier releases reported a warm-kernel
> "~5×" (band 5–8×, dt-parity floor ~3.2×), a warm real-user "~2.5×", and
> equivalence-demo "~4.26× warm-cached / ~1.70× cold" — all measured on an
> **incomplete/faster dycore**. Completing the WRF-faithful dynamics + physics in
> v0.13/v0.14 raised per-step compute to **parity** (the ~1.05×–1.06× measured
> above). A multi-× warm-kernel ratio is **mathematically incompatible** with the
> measured ~1.05× end-to-end on the shipped code, so **these figures are
> superseded and are not v0.14 claims.** They are kept below only as historical
> context for the project's trajectory; the binding v0.14 number is the parity
> measurement and triage at the top of this page.

For the record, the earlier-version figures (one RTX 5090 vs 28-rank CPU-WRF,
same workstation, both fp64, d02 3 km grid; **incomplete/faster dycore,
superseded**):

| Number (earlier version, **superseded**) | Value (historical) | What it measured |
|---|---|---|
| Warm kernel (apples-to-apples) | ~5× (band 5–8×, dt-parity floor ~3.2×) | Compute-only per-forecast-hour on the earlier dycore (`proofs/perf/speedup_denominator.md`, dated 2026-05-30). |
| Warm real-user wall | ~2.5× | Full command-to-finish wall after the JIT cache is warm, earlier dycore. |
| Equivalence-demo real-user | ~4.26× warm-cached / ~1.70× cold | `equivalence_demo.py` 24 h d02, earlier dycore (`proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`). |

`proofs/perf/speedup_denominator.md` remains in the repo as a dated (2026-05-30)
record of the **earlier-version** measurement; it is **not** the current
authority. The current authority is the v0.14 parity table + the perf-triage
breakdown at the top of this page. The **cold→warm JIT-cache** behaviour the demo
exposed (cold compile vs ~10 s bit-identical cache read) still holds in v0.14 — it
is a compile-time effect, not a forecast-speed claim.

### Precision: there is no faster fp32 standalone path today

The precision matrix (`src/gpuwrf/contracts/precision.py`) authorizes fp32 *storage* for
the transported fields, but the operational standalone path forces pure **fp64** for the
acoustic solve (fp32 detonates the perturbation per the F7 gates). A gated-fp32 operational
path is a future ADR-007 decision and is currently **no faster on this memory-bound
workload** (`docs/resource-profile.md`). No fp32 standalone speed number is reported because
none is reachable through the CLI.

## Safe-speedup candidate MEASURED — and rejected on the coupled path

> Historical record of one rejected flag (earlier-version baseline; the 16.71
> s/forecast-hour baseline below is the pre-v0.14 dycore, **superseded** by the
> v0.14 parity measurement at the top). Kept because the *conclusion* — keep XLA
> defaults — still holds, and the launch-tax target it identifies feeds the v0.15
> performance work.

The warmed step is **launch-bound** (`cuLaunchKernelEx` = 38% of CUDA-API time; only 5.5%
of launches are CUDA-graph-captured today), so `XLA_FLAGS=--xla_gpu_graph_min_graph_size=1`
(lower the CUDA-graph capture threshold 5→1) was the leading candidate — it gave **1.71×**
on the **dynamics-only** dycore in a prior sprint (`proofs/perf/fusion_results.md`), but
**did not carry to the coupled step** (below) and was not landed.

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
