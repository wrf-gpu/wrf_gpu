# Runtime & Compute-Cycle Analysis of the GPU WRF Forecast

**A roofline-grounded account of where the per-step time goes, why the core is near-optimal, and why four candidate optimizations cannot faithfully improve it further.**

Companion document to the wrf_gpu2 v0.1.0 paper. Target case: the operational **Canary d02 (3 km, 159 × 66 × 44)** coupled real-case forecast on a single **NVIDIA RTX 5090** (GB202, compute capability 12.0), **fp64**, dt = 10 s, 10 acoustic substeps, RRTMG radiation at cadence 180.

Every number below is sourced to a committed proof artifact under `proofs/perf/` or `proofs/thompson_perf/` with a `file:line` citation. No new runs were performed for this document; it synthesizes the committed profiling, roofline, and A/B evidence. All GPU timings are **warmed** (compile excluded) and reproducible via the committed harnesses listed in the provenance manifest (§7).

---

## 1. Headline result

> **One RTX 5090 (fp64, single domain) runs the 3 km Canary d02 forecast ~5.3× faster (clean) / ~7.8× faster (realistic) than 28-rank CPU-WRF v4.7.1 on the same workstation and the same domain.**

| Framing | CPU-WRF d02 (s/forecast-hour) | GPU d02 (s/forecast-hour) | Speedup |
|---|---|---|---|
| **Conservative — CPU clean compute** | 83 | 15.68 | **5.29×** |
| **Realistic — CPU incl. radiation + IO** | 123 | 15.68 | **7.84×** |
| dt-matched floor (GPU forced to CPU's dt = 6 s) | 83 | ~26.1 | ~3.2× |

- **GPU numerator** = 43.556 ms/step → **15.68 s/forecast-hour** at dt = 10 s, with the shipped bit-identical Thompson sedimentation scan-unroll (§4.4). Source: `proofs/thompson_perf/coupled_timing_base_vs_opt.json:8-9,23` (corroborated by `proofs/perf/warmed_timing.json:39-40`, 43.556 ms/step / 15680 ms/fc-hr).
- **CPU denominator** = 28-rank WRF **V4.7.1**, identical d02 grid (160 × 67 × 45) and dt = 6 s, measured across **two independent finished 72 h L2 runs**: clean compute **≈ 80–86 s/fc-hour** (adopted midpoint **83**), realistic incl. radiation + IO **≈ 85–161 s/fc-hour** (adopted midpoint **123**). Source: `proofs/perf/speedup_denominator.md:49-64`.

### The honest caveats (state these wherever the headline appears)

1. **d02-only / standalone.** The GPU port integrates a single 3 km domain with prescribed lateral boundaries fed from the corpus d01→d02 boundary. It does **not** produce the d01 (9 km) parent or the d03–d05 (1 km) children. The 5.3× is "one GPU replaces one CPU d02," not "one GPU replaces the operational multi-domain nest." The retracted "~50–85×" and "22.26×" headlines came from dividing one GPU d02 against the *whole multi-domain CPU nest wall* — apples-to-oranges. (`proofs/perf/speedup_denominator.md:13,94-103`.)
2. **fp64 on both sides.** The GPU number is fp64; fp32 does **not** help on this workload (§4.1).
3. **Single GPU vs 28 CPU ranks** on the *same* box (cores 4–31). Per-socket-vs-per-GPU, not per-watt or per-dollar.
4. **dt asymmetry credited honestly.** The GPU is stable at dt = 10 s while CPU-WRF runs dt = 6 s; the comparison is per-forecast-hour (same model time). The strict dt-matched floor is ~3.2× (`proofs/perf/speedup_denominator.md:90-92`).

---

## 2. Hardware roofline reference

The roofline is the lens for everything below. A kernel whose **arithmetic intensity** (AI, FLOP per byte of HBM traffic) sits *below* the ridge point is bandwidth- or latency-bound; arithmetic precision then has **no effect** on its runtime.

| Quantity | Value | Source |
|---|---|---|
| FP32 peak (non-tensor) | 104.9 TFLOP/s | `roofline_costonly.json:13` |
| FP64 peak | **1.64 TFLOP/s** (= fp32 / 64 on GeForce Blackwell) | `roofline_costonly.json:14` |
| GDDR7 HBM bandwidth | **1.792 TB/s** | `roofline_costonly.json:15` |
| **FP64 roofline ridge** | **0.915 FLOP/byte** | fp64_peak / BW |
| **FP32 roofline ridge** | **58.5 FLOP/byte** | fp32_peak / BW |

The crucial hardware fact: **GeForce Blackwell fp64 is 1/64 of fp32.** Naively this says "fp64 is slow, so go fp32." The roofline shows why that intuition is wrong for this model: the dycore sits at **AI ≈ 0.40 FLOP/byte** (§3) — below even the fp64 ridge (0.915) and **146× below** the fp32 ridge (58.5). The model is memory- and launch-bound, never compute-bound, so the slow fp64 arithmetic is not on the critical path. Source: `proofs/perf/compute_cycle_analysis.md:21-31`, `proofs/perf/roofline_costonly.json:85-91`.

---

## 3. Where the time goes

### 3.1 Per-step wall-clock decomposition

Warmed per-phase isolated cost on the d02 grid (each phase timed in its own `jax.jit`, with its XLA cost-analysis FLOPs/bytes). Source: `proofs/perf/phase_breakdown.json`.

| Phase | warmed ms (min) | GB touched | GFLOP | HBM floor (ms) | ms / HBM-floor | per-step calls |
|---|---|---|---|---|---|---|
| **Thompson microphysics** | **20.0** | 1.77 | 0.72 | 0.99 | **20×** | 1 |
| surface layer | 4.0 | 0.61 | 0.03 | 0.34 | 12× | 1 |
| MYNN PBL | 2.9 | 1.30 | 0.20 | 0.73 | 4× | 1 |
| advection tendencies | 1.0 | 0.66 | 0.21 | 0.37 | 2.7× | 3 |
| small_step_prep | 1.0 | 0.76 | 0.03 | 0.43 | 2.3× | 3 |
| calc_p_rho (EOS) | 1.0 | 0.037 | 0.011 | 0.02 | **48×** | 3 + 16/substep |
| **vertical Thomas solve** | **1.0** | 0.060 | ~0 | 0.034 | **29×** | 2 × 16 substeps |
| boundary apply | 1.0 | 0.87 | 0.04 | 0.48 | 2.1× | 1 |
| flux-advection augment | 0.68 | 0.79 | 0.29 | 0.44 | 1.5× | 3 |
| calc_coef_w coefficients | 0.34 | 0.076 | 0.25 | 0.04 | 8× | 3 |
| halo apply | 0.52 | 0.56 | n/a | 0.31 | 1.7× | ~8 |

Source: `proofs/perf/phase_breakdown.json:21-87` (min_ms / gbytes / flops per phase), `proofs/perf/compute_cycle_analysis.md:53-65`.

**Reading the table.** Roughly the **coupled step splits as: Thompson microphysics ≈ half (~21 ms of ~44 ms), the dynamical core ≈ one third (launch/bandwidth-bound), and the physics couplers + boundary + radiation cadence the remainder.** The `ms / HBM-floor` column is the smoking gun: phases that move almost no data (EOS 0.037 GB, the tridiagonal solve 0.060 GB, calc_coef_w 0.076 GB) still take ~0.3–1.0 ms — **8×–48× their bandwidth floor** — because their cost is the *kernel launch + dependent vertical sweep*, not the data movement. The acoustic small step calls EOS, calc_coef_w, the momentum updates, and the **two vertical Thomas sweeps 16× per step** (1 + 5 + 10 substeps across RK1/RK2/RK3), so the tiny-kernel launch tax is multiplied 16–48×.

### 3.2 The dycore roofline placement (the crux)

The trustworthy per-step cost is the single-`lax.scan` **body cost-analysis**, which XLA reports once per body and is verified constant across 10/20/40 trip counts (i.e. it is the per-step body, not the whole forecast). Source: `proofs/perf/roofline_costonly.json:57-95`, `proofs/perf/compute_cycle_analysis.md:73`.

| Metric | Value |
|---|---|
| dycore-only per-step wall | 16.9 ms |
| bytes accessed / step | **5.66 GB** |
| FLOPs / step | 2.26 GFLOP |
| **arithmetic intensity** | **0.40 FLOP/byte** (fp64 ridge 0.915, fp32 ridge 58.5) |
| achieved fp64 | 0.134 TFLOP/s = **8.2 % of fp64 peak** |
| achieved HBM bandwidth | 0.335 TB/s = **18.7 % of HBM peak** |
| HBM-bandwidth floor | 5.66 GB / 1.792 TB/s = **3.16 ms** |
| **actual / HBM floor** | **5.3×** (launch/serialization tax) |
| fp64-compute floor | 1.38 ms |

Source: `proofs/perf/roofline_costonly.json:81-91`.

**Verdict: the dycore is memory-bandwidth-bound at ~19 % of HBM peak, with a ~5.3× kernel-launch/serialization tax on top. It is decisively NOT fp64-compute-bound** — fp64 arithmetic is only ~8 % of fp64 peak, and AI 0.40 is 146× below the fp32 ridge. The 5.3× gap between the 16.9 ms wall and the 3.16 ms bandwidth floor is pure launch/latency overhead.

### 3.3 Kernel-level confirmation (nsys)

nsys profiled 36 warmed steps of the coupled forecast. Source: `proofs/perf/nsys_warmed_step_stats_*.csv`, `proofs/perf/compute_cycle_analysis.md:93-104`.

| Quantity | Value | Meaning |
|---|---|---|
| **GPU operations per step** | **~11 160** (7 236 kernels + 3 922 memory ops) | enormous launch count |
| `loop_*` elementwise fusions | **~6 890 / step**, ~7.3 ms, ~1 µs each | the per-substep stencil arithmetic — the dominant genuine cost |
| device memory ops (memcpy/memset) | **~3 922 / step**, ~5.8 ms | scan `concatenate` / `dynamic-update-slice` / halo `pad` traffic |
| **GPU idle fraction** | wall 27–47 ms vs GPU-busy ~15 ms → **~43–68 % of the step idle** | waiting between tiny dependent launches = launch-bound |
| cuSPARSE batched PCR tridiagonal solve (`pcrGtsvBatchSharedMemKernelLoop<double>`) | ~2.0 ms / step | the vertical w/φ implicit solve — **already a parallel solver** |

**Two findings re-shaped the entire optimization picture:**

1. **The vertical implicit w/φ solve is ALREADY a parallel solver.** XLA lowers the `jax.lax.scan` Thomas sweeps to NVIDIA cuSPARSE **batched Parallel-Cyclic-Reduction** (`pcrGtsvBatchSharedMemKernelLoop<double>`). It costs only ~2.0 ms/step — not the bottleneck. The naive "unroll the serial Thomas scan" idea is **moot: the compiler already parallelised it.** (The profiler refuted the initial serial-scan hypothesis.) Source: `proofs/perf/compute_cycle_analysis.md:104-108`, `proofs/perf/nsys_warmed_step_stats_cuda_gpu_kern_sum.csv:4`.
2. **The bottleneck is the ~6 900 tiny elementwise kernels + ~3 900 memory ops** — the per-substep face-average / finite-difference stencils that XLA does not fuse across the substep `lax.scan` boundary, each paying full launch latency with the GPU idle between them. Reducing this kernel COUNT is the only real lever, and it is **precision-invariant**. Source: `proofs/perf/compute_cycle_analysis.md:109-110`.

*Honesty note:* the 36-step warm-up did not fully settle XLA autotuning, so the raw nsys GPU-busy total is inflated by `RedzoneAllocatorKernelImpl` + `DelayKernel` autotuning artifacts (these would vanish with a ≥200-step warm-up). They are *extra* launches on top of an already launch-bound step and do not change the verdict. Source: `proofs/perf/compute_cycle_analysis.md:112`.

---

## 4. Why each lever cannot faithfully improve it further

Four candidate optimizations were each **implemented and measured**, and each was **refuted** by direct measurement. This is the core "optimization homework": the speedup is near-optimal *under the WRF-fidelity constraint*, and the constraint is doing real work in every case.

### 4.1 fp32 dynamics — **measured ~1.00× (0× gain)**

**Hypothesis:** fp64 is 1/64 of fp32 on GeForce Blackwell, so fp32 should roughly double throughput.
**Measured:** ~1.00×. **Mechanism (three-part, from the roofline):**

1. fp32 only accelerates *arithmetic-throughput-bound* kernels, and this model has none — dycore AI 0.40 is 146× below the fp32 ridge, and fp64 arithmetic is ~8 % of the step (§3.2). Speeding up arithmetic that is ~8 % of the cost cannot speed up the step.
2. fp32 *would* halve the bytes — and the step IS bandwidth-bound — but (3) the saving is cancelled at the boundary.
3. **The acoustic core (EOS, calc_coef_w, momentum updates, the implicit w/φ solve, geopotential) is mandatory-fp64 for stability.** An fp32 storage layer for the non-acoustic fields must `convert(f32↔f64)` on entry/exit of the acoustic island every RK stage / substep. These converts are extra elementwise passes over the largest 3-D fields — they *add* HBM traffic (the very thing fp32 was meant to cut) and *add* kernels to an already launch-bound step, while leaving the ~6 900-kernel launch count unchanged.

**Net: ~1.00×.** This refutes the "fp64 is slow so fp32 will be ~2× faster" intuition — that intuition only holds for compute-bound kernels, of which this model has none. Source: `proofs/perf/compute_cycle_analysis.md:116-126`.

### 4.2 CUDA command-buffer graph capture — **regresses the coupled step 15–21%**

**Hypothesis:** the launch tax (~6 900 tiny kernels, 43–68 % GPU idle) is exactly what CUDA graph capture eliminates. Lowering XLA's graph-capture threshold (`--xla_gpu_graph_min_graph_size` from 5 → 1) batches the short dependent chains into CUDA graphs.

**On the dynamics-only dycore it worked: +1.71×** (45.94 → 26.84 ms/step warmed), bit-perturbation only at fp64 round-off, memory-neutral. Source: `proofs/perf/fusion_results.md:64-78`.

**But on the operational COUPLED step it REGRESSES 15–21%** — two independent A/B runs (reversed order, different rep counts) gave **0.826× and 0.866×** (flag = 50.9–53.2 ms/step vs no-flag 43.95–44.09 ms/step), **bitwise-identical** in all 8 fields. Source: `proofs/perf/fusion_confirm_results.md:40-48`.

**Mechanism:** the coupled step is **physics-compute-dominated, not launch-bound** — Thompson microphysics alone is ~21 ms (~half the step) and is already a large fused compute kernel with little launch tax. Forcing the whole program's short chains into CUDA graphs adds graph-capture/submission overhead (the flag's first coupled call is ~30–40 s slower while graphs are captured) that, amortized into steady state, **exceeds** the launch-tax saving on the ~1/3 dynamics fraction. Net 0.83–0.87×. A 24 h coupled run confirmed the flag is bitwise-stable but 18% slower (19.65 vs 16.66 s/fc-hr). Source: `proofs/perf/fusion_confirm_results.md:54-73,99-104`.

**Decision: do NOT bake the flag.** It would slow the operational forecast. The 1.71× holds only for the non-operational dynamics-only configuration. Source: `proofs/perf/fusion_confirm_results.md:130-134`.

### 4.3 fp32 Thompson microphysics — **measured ~1.0× (0× gain)**

**Hypothesis:** Thompson is the one large compute phase (~half the step), so it is the fp32 lever.

**Measured:** ~1.0× (full kernel 11.0 → 11.0 ms; tiled 20748-col grid 42.3 vs 42.9 ms — fp32 *without* unroll is actually a **0.80× regression**). Source: `proofs/thompson_perf/kernel_lever_summary.json:9-24`, `proofs/thompson_perf/THOMPSON_PERF_ANALYSIS.md:65-82`.

**Mechanism — the premise was wrong.** Direct decomposition shows the Thompson kernel is **~85 % sedimentation**, which is a **launch/bandwidth-bound** loop of **64 sequential tiny dependent upwind passes × 4 species** — not fp64-arithmetic-bound:

| Sub-phase | fp64 ms | fp32 ms | share |
|---|---|---|---|
| source/sink process rates | 2.48 | 2.00 | ~15 % |
| **sedimentation (64 substeps × 4 species)** | **9.99** | **9.00** | **~85 %** |

Source: `proofs/thompson_perf/kernel_lever_summary.json:18-24`. This is the **same launch-bound finding as the dycore** — fp32 only helps arithmetic-throughput-bound kernels, and this kernel has none. fp32 is *oracle-faithful* (perturbs moist outputs ≤ ~1 fp32 ULP, rel ≤ 9e-7, at/below the WRF oracle's own fp32 storage granularity — `proofs/thompson_perf/THOMPSON_PERF_ANALYSIS.md:85-101`), so it is kept as a gated opt-in (`GPUWRF_THOMPSON_FP32`), but it is not a default because it gives no speed.

### 4.4 Implicit (backward-Euler) sedimentation — **REJECTED (more diffusive, over-precipitates)**

**Hypothesis:** replace the 64-substep explicit upwind sedimentation with one unconditionally-stable implicit sweep. **Measured kernel speedup: ~2.25–2.44×** (33.8 → 13.9 ms, or 30.5 → 13.5 ms). Source: `proofs/thompson_perf/implicit_sed_timing.json:13-15`, `proofs/thompson_perf/kernel_lever_summary.json:16`.

**But it fails the fidelity gate.** Validated against a **purpose-built precipitating WRF Thompson oracle** (the real Fortran `mp_gt_driver` on a precipitating column — rain/graupel/snow/ice all sedimenting; the pre-existing oracle savepoint was a dry/clear column that could not discriminate a sedimentation change). Source: `proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md:17-38`.

| Scheme | surface precip vs WRF 0.347 mm | qr mean-rel vs WRF | kernel speedup |
|---|---|---|---|
| **faithful explicit (default, shipped)** | 0.393 mm (**+13%**) | 1.53% | 1.00× |
| implicit BE nsub=1 | 0.510 mm (**+47%**) | 5.11% | 2.25× |
| implicit BE nsub=2 | 0.466 mm (+34%) | 3.68% | 2.03× |
| implicit BE nsub=4 | 0.436 mm (+26%) | 2.76% | 1.61× |

Source: `proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md:75-99`.

**The speedup evaporates at the accuracy it needs.** The implicit single-sweep is materially more diffusive — it smears the falling front downward and over-precipitates +47% vs the WRF oracle in one step. Raising nsub recovers accuracy but erodes the win: even nsub=4 is still +26% vs WRF (worse than the faithful default's +13%) while only delivering 1.61×. There is **no nsub that is both ≥2× faster AND as faithful as the explicit default.** GPT-5.5 xhigh independently concurred (REJECT nsub=1 as a default; at most ADR-gated nsub≥4 behind a multi-case precip+skill gate). **Decision: REJECT as a default**; kept as a gated experimental knob (`GPUWRF_THOMPSON_IMPLICIT_SED`, default OFF). Source: `proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md:134-155`.

### 4.5 The one shipped safe win — sedimentation scan-unroll (+~5%, bit-identical)

`GPUWRF_THOMPSON_SED_UNROLL=2` inlines the sedimentation `lax.scan` iterations in order — a **pure launch-count reduction with zero numerical change** (the WRF microphysics oracle is byte-for-byte identical to base; 17/17 Thompson tests pass). Kernel ~1.08× (33.8 → 31.2 ms); coupled-step **~1.05×** (45.53 → 43.56 ms/step), which is what moves the headline from 5.06×/7.5× to **5.29×/7.84×**. Source: `proofs/thompson_perf/kernel_lever_summary.json:11,28`, `proofs/thompson_perf/coupled_timing_base_vs_opt.json:3-23`.

A second safe lever exists in the dynamical core — the **acoustic-substep `lax.scan` unroll** (`GPUWRF_ACOUSTIC_UNROLL`), measured **1.225×** on the dycore at unroll=4 (44.76 → 36.53 ms), passing both idealized close gates (warm bubble + Straka). It is shipped **env-gated, default OFF** because it is not bitwise-identical (cross-substep fp64 reassociation perturbs results at machine-epsilon-per-step, rel ~1e-15) and carries a ~7× cold-compile penalty. Source: `proofs/perf/unroll_ab_verdict.json`, `proofs/perf/compute_cycle_analysis.md:160-191`.

---

## 5. The honest ceiling and residual headroom

**≥10× is not achievable without sacrificing WRF fidelity, and the gap is now well-characterized rather than hand-waved.** The four levers above exhaust the *faithful* per-step optimizations:

- **fp32 (dynamics and Thompson): 0×** — the model is launch/bandwidth-bound, not compute-bound; the mandatory-fp64 acoustic island's boundary converts cancel any bandwidth saving.
- **Command-buffer launch batching: net loss** — the operational coupled step is physics-compute-dominated, so graph-capture overhead exceeds the launch-tax saving.
- **Implicit sedimentation (the only ≥2× kernel lever): fidelity-rejected** — it over-precipitates +47% vs the WRF oracle; the nsub that recovers accuracy erodes the win below 1.6×.
- **Shipped safe wins: ~1.05× (Thompson, bit-identical) + an opt-in 1.225× (dynamics, round-off).**

### What a future restructuring could *theoretically* gain — and why it is high-risk / fidelity-cost

The genuine remaining headroom is the **5.3× gap between the dycore wall (16.9 ms) and its HBM-bandwidth floor (3.16 ms)** — pure launch/serialization tax on the ~6 900 dependent micro-kernels — plus reducing the physics (Thompson) cost. Closing *most* (not all — there are irreducible dependent stencils, and the implicit solve is already cuSPARSE PCR) of the dycore launch tax could push the dycore toward ~5–8 ms and the coupled step toward ~12–22 ms → roughly **~8–11× vs the clean CPU denominator**. Source: `proofs/perf/compute_cycle_analysis.md:148-154`.

But every path to that ceiling is **precision-invariant launch-COUNT reduction** (whole-substep hand-fusion or higher unroll), which:

- perturbs the fp64 acoustic core's bytewise output (cross-substep reassociation — same benign machine-epsilon class as a different XLA/GPU build, but it changes the core's exact result, so it must be re-proven by the full dycore + 24 h coupled gates, not assumed);
- balloons compile time and register pressure (the unroll=4 program already OOMs the coupled path under shared-GPU memory pressure);
- partly lives in physics-owned files (Thompson fusion).

And the one large *algorithmic* lever that would halve Thompson (implicit sedimentation) is fidelity-rejected (§4.4). **So the scientifically-grounded ceiling is ~8–11× under strict WRF fidelity, reachable only by kernel-launch-count reduction (fusion), with precision having no effect — and the current ~5.3×/7.8× is near-optimal for the faithful, default-safe configuration.** ≥10× is conditionally reachable against the *realistic* denominator but is **not** a free win: it trades against compile cost, the fp64 core's bytewise reproducibility, and physics-file ownership, and must be re-certified against the idealized close gates and a free-GPU 24 h coupled stability run before any default flip.

---

## 6. Summary table — the four refuted levers + the shipped win

| Lever | Measured effect | Why it cannot faithfully improve the operational step |
|---|---|---|
| fp32 dynamics | **~1.00×** | launch/bandwidth-bound (AI 0.40, 146× below fp32 ridge); fp64-boundary converts cancel the byte saving |
| CUDA command-buffer graph capture | **0.83–0.87× (15–21% slower)** coupled | coupled step is physics-compute-dominated; graph-capture overhead > launch-tax saving on the 1/3 dynamics |
| fp32 Thompson microphysics | **~1.0×** | Thompson is ~85% sedimentation = launch/bandwidth-bound (64 substeps × 4 species), not compute-bound |
| implicit (backward-Euler) sedimentation | 2.25–2.44× kernel, but **REJECTED** | more diffusive: +47% surface precip vs WRF oracle (nsub=1); accuracy-recovering nsub≥4 erodes win to 1.6× |
| **sedimentation scan-unroll (SHIPPED, default ON)** | **+~5% coupled, bit-identical** | the safe win that moved the headline to 5.3×/7.8× |
| acoustic-substep unroll (shipped, gated OFF) | +22.5% dycore, fp64 round-off | opt-in only: not bitwise-identical + ~7× compile penalty |

---

## 7. Provenance manifest

All artifacts are committed; harnesses are reproducible.

- `proofs/perf/roofline_costonly.json` — authoritative per-step roofline (dycore 2.26 GFLOP / 5.66 GB → AI 0.40, 18.7% HBM, 8.2% fp64, 5.3× over bandwidth floor). Harness `roofline_costonly.py`.
- `proofs/perf/phase_breakdown.json` — per-phase isolated warmed ms + FLOPs/bytes (Thompson 20 ms, EOS 48× over floor, tridiag 29× over floor). Harness `phase_breakdown.py`.
- `proofs/perf/nsys_warmed_step_stats_{cuda_gpu_kern_sum,cuda_gpu_sum,cuda_api_sum}.csv` — kernel-level GPU summary (~7 200 kernels + ~3 900 memory ops/step; cuSPARSE PCR solve). Raw `.nsys-rep` gitignored; regenerate via `run_nsys.sh`.
- `proofs/perf/compute_cycle_analysis.md` — the full roofline / launch-tax / fp32 analysis this document distills.
- `proofs/perf/fusion_results.md` — command-buffer +1.71× on dynamics-only (the now-superseded ≥10× projection).
- `proofs/perf/fusion_confirm_results.md` — the authoritative coupled verdict: command-buffer flag regresses the coupled step 15–21%; honest coupled ~5×/7.4×.
- `proofs/perf/speedup_denominator.md` — the CPU-WRF d02 denominator (two independent L2 72 h runs; 83 clean / 123 realistic s/fc-hr) + provenance + the retraction of "~50–85×".
- `proofs/perf/warmed_timing.json`, `proofs/perf/segscan_24h*.json` — GPU numerator timing (43.556 ms/step → 15.68 s/fc-hr) + 24 h stability gates.
- `proofs/perf/unroll_ab_verdict.json` — acoustic-substep unroll A/B (1.225×, fp64 round-off).
- `proofs/thompson_perf/THOMPSON_PERF_ANALYSIS.md` — fp32 Thompson dead-end + sedimentation decomposition + shipped sed-unroll.
- `proofs/thompson_perf/kernel_lever_summary.json` — all Thompson lever timings + verdicts.
- `proofs/thompson_perf/implicit_sed_timing.json` — implicit-sed kernel timing (2.25–2.44×).
- `proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md` — the precipitating WRF oracle + the implicit-sed ADOPT/REJECT decision (+47% precip → REJECT).
- `proofs/thompson_perf/coupled_timing_base_vs_opt.json` — coupled base-vs-shipped (5.06×/7.5× → 5.29×/7.84×).
