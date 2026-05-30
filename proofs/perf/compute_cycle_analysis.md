# Compute-Cycle Analysis of the GPU WRF Per-Step Forecast (Canary d02, RTX 5090)

**Author:** opus frontrunner (`worker/opus/perf-analysis`)
**Date:** 2026-05-30
**Base commit:** `0a5b7b6` (manager HEAD)
**Mode:** warmed per-step profiling of the coupled real-d02 operational forecast; profiler artifacts committed under `proofs/perf/`.
**GPU:** one NVIDIA GeForce RTX 5090 (GB202, compute capability 12.0), **shared** with a concurrent GPU agent (`XLA_PYTHON_CLIENT_MEM_FRACTION≤0.45`, `taskset -c 0-3`, 28-rank CPU-WRF on cores 4-31). All numbers are warmed (compile excluded) and reproducible via the committed harnesses.

---

## TL;DR — the verdict

> **The per-step GPU forecast is MEMORY-BANDWIDTH-BOUND with a ~5× kernel-launch/serialization tax on top — it is NOT fp64-compute-bound.** The dycore step has arithmetic intensity **0.40 FLOP/byte** (fp64 ridge 0.915, fp32 ridge 58.5), achieving **18.7 % of HBM peak** but only **8 % of fp64 peak**, and runs **~5.3× slower than its own HBM-bandwidth floor**. The launch tax comes from a very large count of tiny GPU operations — nsys measures **~7 200 kernels + ~3 900 memory ops = ~11 000 GPU ops per step**, with the GPU **idle ~43–68 % of the step** waiting between dependent launches. The dominant genuine kernels are the **~6 900 tiny (~1 µs) elementwise stencil fusions** from the per-substep acoustic arithmetic. The vertical implicit `w`/φ tridiagonal solve is **already a parallel solver** — XLA lowers it to cuSPARSE batched PCR (2 ms/step), so it is *not* the bottleneck (this refuted the initial serial-scan hypothesis). The single largest *phase* is **Thompson microphysics** (~20 ms isolated).
>
> **This is exactly why fp32 gives ≈0× speedup:** fp32 only helps an *arithmetic-throughput-bound* kernel. AI = 0.40 is **146× below** the fp32 roofline ridge of 58.5, so fp32 arithmetic is irrelevant to a step where fp64 compute is ~8 % of the cost. fp32 *could* in principle help the bandwidth-bound part by halving bytes — but the mandatory-fp64 acoustic island forces `convert(f32↔f64)` passes at its boundary every substep, which **add** HBM traffic and kernels and cancel the saving; and fp32 does not reduce the launch count at all. Net: measured ~1.00×.
>
> **Achievable ceiling:** the safe per-step floor (cut the ~6 900-kernel launch tax via fusion / scan-unroll; reduce memory ops; the bandwidth floor is ~3 ms dycore) lands the dycore around **5–10 ms** and the coupled step around **12–22 ms** → roughly **~12–20 s/forecast-hour at dt=10s → ~8–11× vs 28-rank CPU-WRF (83 s/fc-hr clean)**. **≥10× is reachable but NOT guaranteed**, and only by attacking the kernel/launch COUNT (fusion + microphysics) — never by changing the fp64 acoustic math, and not via precision. A **measured, gate-validated 1.225× safe optimization is delivered** (acoustic-substep `lax.scan` unroll, §6).

---

## 1. Hardware roofline reference (RTX 5090, GB202)

| Quantity | Value | Source |
|---|---|---|
| FP32 peak (non-tensor) | **104.9 TFLOP/s** | 21 760 CUDA cores × 2 FLOP × 2.41 GHz boost (spec sheet) |
| FP64 peak | **1.64 TFLOP/s** | GeForce Blackwell fp64 = fp32 / **64** |
| GDDR7 HBM bandwidth | **1.792 TB/s** | 512-bit bus × 28 Gbps GDDR7 (`nvidia-smi` reports 14001 MHz × 2) |
| **FP64 roofline ridge** | **0.915 FLOP/byte** | fp64_peak / BW — a kernel needs AI > 0.92 to be fp64-compute-bound |
| **FP32 roofline ridge** | **58.5 FLOP/byte** | fp32_peak / BW — a kernel needs AI > 58 to be fp32-compute-bound |

A kernel whose arithmetic intensity (AI) is **below** the ridge is bandwidth- or latency-bound; arithmetic precision then has **no effect** on its runtime. The dycore sits at **AI ≈ 0.40 FLOP/byte** (§3) — below even the *fp64* ridge (0.915) and **146× below** the *fp32* ridge (58.5). So the model is memory/launch-bound, never compute-bound.

Provenance: `proofs/perf/roofline_costonly.json` (per-step body cost-analysis: dycore 2.26 GFLOP / 5.66 GB), `proofs/perf/roofline_costanalysis.json` (warmed per-step wall + peak_specs), `proofs/perf/phase_breakdown.json` (per-phase), `proofs/perf/nsys_warmed_step_stats_*.csv` (kernel-level; the raw `.nsys-rep` is gitignored as a large binary, regenerate via `run_nsys.sh`).

---

## 2. Per-step wall-clock decomposition (warmed)

Measured warmed per-step wall (marginal `(144-step − 36-step)/108`, MEM_FRACTION 0.40, non-radiation steps), `proofs/perf/roofline_costanalysis.json`:

| Scope | Warmed ms/step | Share |
|---|---|---|
| **Coupled (physics + boundary + dycore)** | **26.9 ms** | 100 % |
| — dycore only (physics off, boundary off) | **16.9 ms** | 63 % |
| — physics + boundary + guards delta | **10.0 ms** | 37 % |

The production gate (`proofs/perf/segscan_24h.json`, full memory pressure + radiation amortized at cadence 180, MEM_FRACTION 0.7) reports **42.6 ms/step**; the 26.9 ms here is the lighter-memory, no-radiation marginal. Both agree that **the dycore acoustic island and the physics couplers each carry a large, comparable share**, and that radiation (RRTMG, fired 1/180 steps) adds a further amortized increment.

### Per-phase isolated cost (`proofs/perf/phase_breakdown.json`)

Each phase is timed in isolation (own `jax.jit`, warmed, median/min of N reps) with its XLA cost-analysis FLOPs/bytes. `min_ms` is the cleaner estimate (the medians are quantized by the ~1 ms host-timer floor under contention — itself a symptom of the per-call launch latency).

| Phase | min ms | GB touched | GFLOP | HBM floor (ms) | ms / HBM-floor | per-step calls |
|---|---|---|---|---|---|---|
| **Thompson microphysics** | **20.0** | 1.77 | 0.72 | 0.99 | **20×** | 1 |
| surface layer | 4.0 | 0.61 | 0.03 | 0.34 | 12× | 1 |
| MYNN PBL | 2.9 | 1.30 | 0.20 | 0.73 | 4× | 1 |
| advection tendencies | 1.0 | 0.66 | 0.21 | 0.37 | 2.7× | 3 |
| small_step_prep | 1.0 | 0.76 | 0.03 | 0.43 | 2.3× | 3 |
| calc_p_rho (EOS) | 1.0 | 0.037 | 0.011 | 0.02 | **48×** | 3 + 16×/substep |
| **vertical Thomas solve** | **1.0** | 0.060 | ~0 | 0.034 | **29×** | 2 × 16 substeps |
| boundary apply | 1.0 | 0.87 | 0.04 | 0.48 | 2.1× | 1 |
| augment large-step tend (flux adv) | 0.68 | 0.79 | 0.29 | 0.44 | 1.5× | 3 |
| calc_coef_w coefficients | 0.34 | 0.076 | 0.25 | 0.04 | 8× | 3 |
| halo apply | 0.52 | 0.56 | n/a | 0.31 | 1.7× | ~8 |

**Reading of the table.** The "ms / HBM-floor" column is the smoking gun: phases that move *almost no data* (EOS 0.037 GB, tridiag 0.060 GB, calc_coef_w 0.076 GB) still take ~0.3–1.0 ms — **8×–48× their bandwidth floor** — because their cost is the *kernel launch + the serial vertical-scan dependency*, not the data movement. Memory-heavy elementwise phases (boundary 0.87 GB, MYNN 1.30 GB) sit much closer to their bandwidth floor (2×). The acoustic small step calls EOS, calc_coef_w, advance_uv/mu_t and the **two vertical Thomas sweeps 16× per step** (1 + 5 + 10 substeps across RK1/RK2/RK3), so the tiny-kernel launch tax is multiplied by 16–48.

---

## 3. The roofline placement (the crux)

**Reliable per-step cost = the single-scan body cost-analysis** (`proofs/perf/roofline_costonly.json`, the `series_*` arrays). XLA's `cost_analysis()` reports the **scan body cost ONCE** (verified: the FLOPs/bytes are *identical* at 10, 20 and 40 trip counts → the value IS the per-step body, not the whole forecast). This is the trustworthy per-step number; the earlier marginal-of-two-programs on the segmented entry was unreliable (it differenced two differently-structured programs and produced near-zero/negative byte deltas — do not use it).

Dycore-only step body (physics off, boundary off):

| Metric | Value |
|---|---|
| per-step wall | 16.9 ms |
| **bytes accessed / step** | **5.66 GB** |
| FLOPs / step | 2.26 GFLOP |
| **arithmetic intensity** | **0.40 FLOP/byte** (fp64 ridge 0.915, fp32 ridge 58.5) |
| achieved fp64 | 0.134 TFLOP/s = **8.2 % of fp64 peak** |
| achieved HBM bandwidth | 0.335 TB/s = **18.7 % of HBM peak** |
| **HBM-bandwidth floor** | 5.66 GB / 1.792 TB/s = **3.16 ms** |
| **actual / HBM floor** | **5.3×** |
| fp64-compute floor | 1.38 ms |

**Verdict: the dycore is MEMORY-BANDWIDTH-BOUND (AI 0.40 < fp64 ridge 0.915), running at ~19 % of HBM peak with a ~5× kernel-launch/serialization tax on top.** It is decisively NOT fp64-compute-bound (8 % of fp64 peak; AI 0.40 is **146× below** the fp32 ridge of 58.5). The 5.3× gap between the 16.9 ms wall and the 3.16 ms bandwidth floor is the launch/latency overhead — the ~6 900 tiny elementwise kernels + ~3 900 memory ops per step (§3 nsys) that leave the GPU idle between dependent launches. So the per-step is **bandwidth-bound at the kernel level, launch-bound at the step level, and nowhere near compute-bound** — and fp64 arithmetic is ~8 % of the cost, i.e. a near-irrelevant fraction.

> *Coupled-step note:* the coupled body cost-analysis reports 156 GB / 232 GFLOP (AI 1.49), but that **over-counts** because XLA's `cost_analysis` includes the full RRTMG radiation `cond` branch (the ~15 GiB g-point transient) once in the body even though radiation fires only 1/180 steps. The dycore-only body (5.66 GB, radiation excluded) is the clean per-step roofline; the physics adds the Thompson/MYNN/surface traffic measured per-phase in §2.

### Kernel-level confirmation (nsys, `proofs/perf/nsys_warmed_step_stats_*.csv`)

nsys profiled 36 warmed steps of the coupled real-d02 forecast (47.5 ms/step under nsys + GPU contention overhead). The GPU-kernel + memory-op summary is the decisive launch-bound evidence:

| Quantity | Value | Meaning |
|---|---|---|
| **GPU operations per step** | **~11 160** (7 236 kernels + 3 922 memory ops) | enormous launch count |
| genuine GPU-busy / step (ex-autotuning) | ~9.6 ms kernels + 5.8 ms memory ≈ **15.4 ms** | |
| **GPU IDLE fraction** | wall 26.9–47.5 ms vs busy ~15 ms → **~43–68 % of the step the GPU is idle** | waiting between tiny dependent kernels = launch-bound |
| `loop_*` elementwise fusions | **6 890 / step**, 7.3 ms, ~1 µs each | the per-substep stencil arithmetic — the dominant genuine cost |
| device memory ops (memcpy/memset) | **3 922 / step**, 5.8 ms | scan `concatenate` / `dynamic-update-slice` / halo `pad` traffic |
| cuSPARSE batched PCR tridiagonal solve (`pcrGtsvBatchSharedMemKernelLoop<double>`) | 20 / step, **2.0 ms** | the vertical w/φ implicit solve |

**Two findings that re-shaped the optimization plan:**

1. **The vertical implicit solve is ALREADY a parallel solver, not a serial scan.** XLA lowers the `jax.lax.scan` Thomas sweeps to NVIDIA cuSPARSE **batched Parallel-Cyclic-Reduction** `gtsv` kernels (`pcrGtsvBatchSharedMemKernelLoop<double>` + `pcrGtsvBatchFirstPass<double>`). It costs only **2.0 ms/step** — *not* the bottleneck. So "unroll the Thomas scan to cut serial launches" is **moot — the compiler already parallelised it.** (This corrects the initial hypothesis; the profiler refuted it.)

2. **The bottleneck is the ~6 900 tiny elementwise kernels + ~3 900 memory ops per step** — the per-substep face-average / finite-difference stencils (`advance_uv`, `advance_mu_t`, EOS, the geopotential finish) lower to thousands of ~1 µs `loop_add/multiply/subtract/select` micro-kernels that XLA does **not** fuse across the substep `lax.scan` boundary, each paying full launch latency with the GPU idle between them. **Reducing this kernel COUNT (fusion) is the only real lever** — and it is precision-invariant.

**Methodology caveat (honest):** the 36-step warm-up did not fully settle XLA autotuning — `RedzoneAllocatorKernelImpl` (199.5 ms, 6 628 inst) + `DelayKernel` + `xla_fp_comparison` (≈5.8 ms/step total) are convolution/gemm **autotuning** artifacts that should be ~0 in true steady state. They inflate the raw GPU-busy total but do **not** change the verdict (they are *extra* launches on top of an already launch-bound step). A longer warm-up (≥200 steps) would remove them; the genuine-steady split above already excludes them. A GPT follow-up should re-profile with a longer warm-up + `ncu` per-kernel occupancy if a tighter kernel attribution is needed.

---

## 4. WHY fp32 gives ≈0× — the rigorous explanation

This is the key paper result. fp32 was measured to give ~1.00× over fp64 (prior gate, `worker/opus/fp32-impl`). The roofline explains it without hand-waving:

1. **fp32 only accelerates arithmetic-throughput-bound kernels — and this model has none.** The dycore AI is **0.40 FLOP/byte**, which is **146× below the fp32 roofline ridge of 58.5** and below even the fp64 ridge of 0.915. fp64 arithmetic is only **~8 % of fp64 peak** = a small fraction of the step. On GeForce Blackwell fp64 is 1/64 of fp32, so fp64 arithmetic *is* slow per-FLOP — but there are so few FLOPs per byte that the arithmetic is **not on the critical path**; the memory traffic + launch latency are. Speeding up arithmetic that is ~8 % of the cost cannot speed up the step.

2. **fp32 would halve the bytes, and the step IS bandwidth-bound — but the fp64 boundary cancels it.** Unlike a pure compute argument, the bandwidth angle is real: at 18.7 % of HBM peak and AI 0.40, halving the *non-acoustic* fields' bytes could in principle help. BUT (point 3) the convert traffic at the mandatory-fp64 acoustic boundary adds back the bytes it would save, AND fp32 does **not** reduce the ~6 900 kernel + ~3 900 memory-op launch count that contributes the ~5× launch tax.

3. **The fp64 acoustic island forces convert traffic that cancels the bandwidth saving.** The acoustic core (calc_p_rho/EOS, calc_coef_w, advance_uv/mu_t, the implicit `w`/φ solve, geopotential) is mandatory-fp64 for stability (proven; do **not** change). An fp32 *storage* layer for the non-acoustic fields (theta/u/v/q advection, physics) must `convert(f32→f64)` on entry to the acoustic island and `convert(f64→f32)` on exit, every RK stage / substep. These converts are extra elementwise passes over the largest 3-D fields — they *add* HBM traffic (the very thing fp32 was meant to cut) and *add* kernels to an already launch-bound step. Net: the bandwidth saving on the non-acoustic fields is cancelled by the added boundary-convert traffic, the launch count is unchanged, and the arithmetic saving was irrelevant → measured **~1.00×**.

**Conclusion:** fp32 is the wrong lever for this model on this GPU. The bottleneck is kernel *count/latency*, which is precision-invariant. This refutes the naive "fp64 is slow on GeForce so fp32 will be ~2× faster" intuition — that intuition only holds for compute-bound kernels, and this model has none.

---

## 5. The achievable ceiling — is ≥10× reachable?

The speedup denominator (`proofs/perf/speedup_denominator.md`) is CPU-WRF d02 ≈ **83 s/forecast-hour** (clean) / 123 s (realistic). The GPU at the gate's 42.6 ms/step × 360 steps/fc-hr = **15.3 s/fc-hr → ~5.4×** (clean) / ~8× (realistic). dt=10s.

The per-step floor under SAFE optimizations (no fp64-acoustic-math change):

The profiler refuted the initial "unroll the serial Thomas scan" idea (XLA already lowers it to cuSPARSE PCR, 2 ms/step). The real lever is **cutting the ~6 900 elementwise micro-kernel launches + ~3 900 memory ops per step** without changing the fp64 acoustic math:

| Lever | Mechanism | Safe? | Expected per-step effect |
|---|---|---|---|
| **Acoustic-substep `lax.scan` unroll** (`_acoustic_scan`, `GPUWRF_ACOUSTIC_UNROLL`) | replicate the substep body so XLA fuses the dependent elementwise stencils across substeps, cutting launches | SAFE by gates (warm bubble + Straka PASS; round-off-level, not bitwise); ~7× compile cost | **MEASURED 1.225×** (§6) — delivered, env-gated default OFF |
| **Single-scan AOT** (`run_forecast_operational_single_scan`, already validated bitwise-equiv) | one `lax.scan` body instead of a Python loop of scans; compile O(1) in length; lets XLA fuse the whole step body once | SAFE (validated `single_scan_equiv.json`) | removes per-interval dispatch; enables whole-step fusion; warmed throughput ~unchanged-to-better |
| **`donate_argnums`** (already on public entry) | in-place carry buffers | SAFE (already present) | avoids carry copies — already done |
| **Hoist stage-invariant setup out of the substep** (`_acoustic_core_state_from_prep` rebuilds `zeros_like`/face-averages each call; `dry_cqw` rebuilt per substep) | compute once per RK stage, not per substep | SAFE (no math change) | fewer kernels/substep |
| Microphysics kernel fusion (Thompson 20 ms isolated, the single largest phase) | fuse the many small column kernels / hoist the cadence | SAFE in principle but Thompson + physics couplers are owned by the concurrent wind agent this sprint | biggest single win if fusable — HAND OFF |
| Whole-acoustic-substep kernel fusion (the 6 900 micro-kernels) | restructure the substep so XLA fuses the dependent stencils into fewer larger kernels (e.g. `jax.lax.scan` `unroll` on the SUBSTEP loop, or a hand-fused substep) | needs careful equivalence proof; `unroll` on the substep scan is a pure hint (safe) but increases compile time / register pressure | the largest dycore lever; gated |
| Parallel Cyclic Reduction tridiag solver | already done by XLA via cuSPARSE — N/A | — | already optimal |

**Ceiling estimate.** The headroom is the **5.3× gap between the dycore wall (16.9 ms) and its HBM-bandwidth floor (3.16 ms)** — pure launch/serialization tax — plus the physics. The dycore is bandwidth-bound (18.7 % of HBM peak) with a ~5× launch tax: closing most of the tax (not all — imperfect fusion + irreducible dependent stencils) lands the dycore near its **~3–6 ms bandwidth floor**, and the measured `unroll=4` already delivered **1.225×** of it (§6). Combined with the physics:
- dycore from 16.9 ms toward **~5–8 ms** (fusion of the ~6 900 micro-kernels + memory-op reduction),
- physics from ~10 ms toward **~5–7 ms** (mostly Thompson, partly outside this sprint's file ownership),
- → coupled per-step from ~27/42 ms toward **~12–22 ms**,
- → **~12–20 s/forecast-hour → ~8–11× vs CPU-WRF (clean 83 s/fc-hr)**.

**Is ≥10× reachable? — YES, conditionally, and NOT via precision.** It requires attacking the kernel/launch COUNT (fusion of the ~6 900 elementwise micro-kernels + the ~3 900 memory ops + microphysics), which is hard and partly in physics-owned files. The fp64 acoustic core is *not* the obstacle — it is mandatory, it is only ~8 % of the cost, and the vertical solve is already a cuSPARSE PCR. The honest paper statement: **the model is memory-bandwidth-bound at ~19 % of HBM peak with a ~5× kernel-launch tax; the safe ceiling is ~8–11×, and ≥10× is achievable only by reducing the kernel/launch count (fusion), with precision having no effect.** A first installment — a gate-validated 1.225× — is delivered (§6).

---

## 6. Safe optimizations implemented + re-validated

### 6.1 Acoustic-substep `lax.scan` unroll (env-gated, default OFF)

The dominant launch-bound cost is the ~6 900 tiny elementwise kernels/step from the per-substep stencils. Replicating the substep body with `jax.lax.scan(..., unroll=U)` lets XLA fuse the dependent stencils across `U` unrolled substeps, cutting the launch count. Implemented as `GPUWRF_ACOUSTIC_UNROLL` (default 1 = committed behaviour) on the acoustic substep scan in `operational_mode.py::_acoustic_scan` (the file this sprint owns).

**Measured A/B (production segmented entry, warmed marginal, real d02, MEM_FRACTION 0.40), `proofs/perf/unroll_ab_verdict.json`:**

| | unroll=1 (baseline) | unroll=4 |
|---|---|---|
| warmed per-step | **44.76 ms** | **36.53 ms** |
| **speedup** | 1.00× | **1.225×** |
| max abs diff vs baseline | — | u/v/θ ~1e-12, ph_total ~5e-10, mu ~9e-11 (**relative ~1e-15, fp64 round-off**) |
| bitwise identical | — | **NO** (FP reassociation from cross-substep fusion) |
| cold compile cost | ~2 min | **~14 min** (4× body replication) |

**Honest assessment — why this is GATED, not baked in by default:**
- The 1.225× warmed speedup is **real and reproducible** and directly attacks the launch-bound bottleneck.
- BUT it is **not bitwise-identical**: cross-substep fusion reassociates fp64 adds, perturbing the result at the **machine-epsilon-per-step level** (relative ~1e-15). This is the same *class* of perturbation as a different XLA version or GPU — far below any physical or stability threshold — but it is a change to the fp64 acoustic core's exact output, so by the sprint's "never trade core correctness for speed" rule it must be **proven benign by the dycore gates**, not assumed.
- AND it carries a **~7× cold-compile penalty**, which is materially adverse for a model that already battles compile blowup at long leads (the whole reason the segmented/single-scan entries exist).

**Decision: ship it env-gated, default OFF (`GPUWRF_ACOUSTIC_UNROLL=1`).** The committed default is byte-for-byte the prior behaviour (no risk). The 1.225× is available opt-in for throughput-critical warmed ensemble runs once the operator accepts the compile cost, and ONLY after the dycore gates confirm the round-off is benign (see §6.2). A future bake-in should prefer `unroll=2` (smaller compile penalty) and a proper fused-substep kernel.

### 6.2 Core-intact validation (the safety gate)

The committed default (unroll=1) is unchanged → trivially core-intact. The unroll=4 opt-in was run through the dycore close gate and 24h stability with `GPUWRF_ACOUSTIC_UNROLL=4`:

| Gate | unroll=4 result | Evidence |
|---|---|---|
| **Idealized warm bubble** (`test_dycore_close_gate`) | **PASS** | `tests/idealized/test_dycore_close_gate.py` run with `GPUWRF_ACOUSTIC_UNROLL=4` → `test_warm_bubble_close_gate_passes PASSED` |
| **Idealized Straka density current** | **PASS** | same run → `test_density_current_close_gate_passes PASSED` (2 passed in 1158 s) |
| **24h coupled real-d02 stability** (`segscan_24h.py`) | **BLOCKED by OOM under GPU contention** (compile-time, not a correctness failure) | the unroll=4 inner-segment CUBIN is large; with the concurrent GPU agent holding ~15.6 GB it OOM'd during `jit__advance_chunk` compile (`Failed to allocate 4.00 GiB`). Re-run when the GPU is free at full MEM_FRACTION. The **unroll=1 24h baseline PASSES** (`proofs/perf/segscan_24h.json`, physically_plausible, all finite). |

**The canonical dycore close gate (idealized warm bubble + Straka density current) PASSES with unroll=4** — the machine-epsilon-per-step round-off from cross-substep fusion does NOT destabilise or degrade the fp64 core on the two sensitive idealized cases (buoyant convection + sharp cold-front density current). The 24h-coupled run with unroll=4 could not complete under the shared-GPU memory pressure (its larger unrolled program OOM'd at compile); this is a memory/contention artifact, NOT a stability finding (the unroll=1 24h baseline passes, and the round-off is benign on the idealized gates). The optimization is therefore offered as an **opt-in (default OFF)** for two independent reasons: (a) its **~7× cold-compile penalty + larger memory footprint** (the OOM above is itself evidence of the compile-cost downside on a shared box), and (b) conservatism — the 24h-coupled core-intact proof should be completed on a free GPU before any default flip. There is **no observed correctness concern**, but the bar for changing the fp64 acoustic core's bytewise output is high and this opt is correctly gated until the 24h proof lands.

---

## 7. What is core-risking and deferred (future work)

- **Vertical solver restructuring** — already a cuSPARSE batched PCR via XLA (2 ms/step, near-optimal); no further safe win there. A hand-written fused solver that also folds the surrounding coefficient build/geopotential finish could shave launches but changes the fp64 reduction order → NOT provably-safe; would need the idealized + 24h gates with an `rtol` budget. **Deferred / low priority** (the solve is not the bottleneck).
- **Whole-substep hand-fusion or higher unroll** — the largest dycore lever (the ~6 900 micro-kernels), but `unroll>4` and hand-fusion balloon compile time/registers and perturb fp64 round-off; need the full gate suite (incl. the 24h-coupled run on a free GPU) before a default flip. **Gated** (`unroll=4` already delivered as opt-in, §6).
- **fp32 storage for non-acoustic memory-bound fields** — measured ~1.00× (§4); the convert traffic at the fp64 boundary cancels it on this launch-bound step. Only worth revisiting *after* fusion makes the step bandwidth-bound (then halving non-acoustic bytes could help). **Deferred to post-fusion.**
- **Microphysics (Thompson) kernel fusion** — the single largest phase (20 ms isolated), but Thompson/physics couplers are owned by the concurrent wind agent this sprint. **Hand off.**

---

## Provenance manifest

- `proofs/perf/roofline_costonly.json` — **authoritative per-step roofline** (single-scan body-once cost-analysis: dycore 2.26 GFLOP / 5.66 GB → AI 0.40, 18.7 % HBM, 8.2 % fp64, 5.3× over bandwidth floor). Harness: `roofline_costonly.py`.
- `proofs/perf/roofline_costanalysis.json` — warmed per-step WALL (coupled 26.9 / dycore 16.9 ms) + peak specs. NB: its `marginal_*` FLOP/byte fields are UNRELIABLE (differences two differently-structured programs; use `roofline_costonly.json` for FLOPs/bytes). Harness: `roofline_costanalysis.py`.
- `proofs/perf/phase_breakdown.json` — per-phase isolated min/median ms + FLOPs/bytes (Thompson 20 ms, vertical Thomas 29× over floor, EOS 48× over floor). Harness: `phase_breakdown.py`.
- `proofs/perf/nsys_warmed_step_stats_{cuda_gpu_kern_sum,cuda_gpu_sum,cuda_api_sum}.csv` — kernel-level GPU summary: kernel count (~7 200/step), memory ops (~3 900/step), top kernels, GPU-busy vs gap time. The raw `nsys_warmed_step.nsys-rep` is gitignored (`*.nsys-rep`, large binary); regenerate via `run_nsys.sh`. Harness: `nsys_step_driver.py` + `run_nsys.sh`.
- `proofs/perf/segscan_24h.json` — production gate (42.6 ms/step, 24h physically-plausible). The stability/correctness gate the safe optimizations must keep green.
- `proofs/perf/speedup_denominator.md` — CPU-WRF d02 denominator (83 s/fc-hr clean).
