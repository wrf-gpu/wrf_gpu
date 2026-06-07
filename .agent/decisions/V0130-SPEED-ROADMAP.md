# v0.13 SPEED ROADMAP — compile + dispatch acceleration (principal-directed 2026-06-07)

**Motto:** speed is a core goal of this port; **compile time counts as part of the speed story**, not just kernel compute. Observed: for SHORT runs (1h validation) XLA compile ≈ 5× the compute (fixed cost dominates); for the full 24h run compute ≈ 5× compile (amortizes). The dev/validation regime (many short, new-config runs) is where compile dominates — and where these levers pay off.

## Why compile is expensive (root cause)
XLA AOT-compiles the whole jitted timestep graph (acoustic substeps + RK3 + microphysics + radiation + PBL + surface + GWD + nesting) before running: fusion analysis, layout assignment, **autotuning (runs candidate GEMM/conv kernels at compile time)**, codegen + LLVM/PTX. Mostly single-threaded, scales with graph size; **fp64 makes the graph bigger + autotuning heavier** (and fp64 is forced operationally — fp32 detonates the acoustic solver, so fp64-inherent compile cost stays).

## v0.13 work items (ranked by value × feasibility)
1. **AOT precompile for the fixed production grids** (HIGH) — `jax.jit(f).lower(args).compile()` + serialize the executable; ship/load a precompiled artifact for the standard Canary 9/3/1km + Switzerland configs → **near-zero compile at runtime** for known grids. The right answer for steady-state production. (Doesn't help arbitrary new grids.)
2. **Persistent XLA autotuning cache** (HIGH) — persist autotune results to disk (`XLA_FLAGS=--xla_gpu_per_fusion_autotune_cache_dir` / `--xla_gpu_dump_autotune_results_to` + load) so even new-but-similar graphs reuse tuned kernels. Often-overlooked, big chunk of fp64 compile.
3. **Strengthen the persistent compile cache** (MED-HIGH) — we have `runtime/compile_cache.py` (cold ~147s → warm ~29s); make it robust + config-keyed so the standard grids/dom-counts/gwd-on-off variants all hit warm. Pre-warm on install.
4. **Parallel compilation** (MED) — `--xla_gpu_force_compilation_parallelism=N` (+ enough host cores) parallelizes kernel compiles + autotuning. ~1.5-2×, not transformative.
5. **Dev/validation autotune-level knob** (MED, dev-only) — `--xla_gpu_autotune_level=0/1` for short validation runs (slightly slower kernels OK when compute is short) → much faster compile. Gate behind a `--fast-compile` dev flag; NEVER for production (production wants the fastest kernels).
6. **Graph reformulation / sub-jit splitting** (MED, careful) — the whole timestep is one mega-jit. Splitting into a few jitted blocks (dycore-step / physics-step) compiles faster + caches better + recompiles less when one part changes — BUT loses cross-block fusion + adds dispatch overhead. `lax.scan` over timesteps already bounds the graph (one step compiled, not 4800) — keep that. Audit for accidental recompiles (unstable shapes / non-static args).
7. **Recompile hygiene** (MED) — audit `static_argnums` / shape stability so config tweaks don't silently retrigger full compiles; `donate_argnums` to cut allocation. Make compile-triggering explicit + logged.
8. **Measurement** (prereq) — a cold-vs-warm compile benchmark + a compile-time line in the perf report, so every lever is quantified (ties into the existing perf-instrumentation backlog #62/#33).

## Framing
- Production (long runs, fixed grids): compile amortizes → AOT + warm autotune-cache make it ~zero. Mostly "solved" by #1+#2.
- Dev/validation (short, many-config — tonight's pattern): compile dominates → #2+#3+#5 + parallel-compile slash iteration time.
- A focused v0.13 "compile-speed" sprint should land #1 (AOT) + #2 (autotune-cache) + #3 (cache hardening) first; #4-#7 as follow-ups; #8 throughout.

## Addendum (principal Q 2026-06-07, "use all cores?"): core-count levers
- **Reality:** XLA compile is single-threaded-dominated; fp64 autotuning runs candidate kernels on the GPU (GPU-serial), so host core count gives only a modest host-codegen win. Core count is NOT a big lever; cache+AOT (#1-3) dwarf it (10-50× on repeat vs ~1.5-2× from parallelism).
- **#4 sharpened — host compile parallelism:** `--xla_gpu_force_compilation_parallelism=N` + enough free host cores → faster host-side LLVM/PTX codegen. Add WITH GPU validation (ad-hoc XLA flag injection broke the GPU path once — the compile-speed revert; never inject XLA flags untested on GPU).
- **#9 NEW — "CPU-flock" for idle nightly cores (structural, careful):** cores 4-31 are reserved for the nightly 28-rank CPU-WRF (speedup baseline + validation corpus) but are IDLE most of the time (scheduler polls, fires periodically). Build a CPU-flock analogous to the GPU flock (`/tmp/wrf_gpu_run.sh`): opportunistically borrow idle nightly cores for our parallel CPU work, yield INSTANTLY when the nightly fires. Real throughput win for CPU testing/compile. RISK: contention would corrupt the nightly's speedup-baseline + corpus → needs the same hiprio/lowprio token discipline as the GPU flock. v0.13+ infra, not a quick win.
