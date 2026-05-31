# v0.1.0 Performance — roofline-grounded current state

Companion: `publish/runtime_optimization_analysis.md`. Target: Canary **d02 (3 km,
159×66×44)** coupled real-case forecast on a single **NVIDIA RTX 5090** (GB202,
cc 12.0), **fp64**, dt = 10 s, 10 acoustic substeps, RRTMG cadence 180. All GPU
timings are warmed (compile excluded). No new runs; synthesizes committed proofs.

## Headline speedup vs 28-rank CPU-WRF v4.7.1 (same workstation, same d02)

| Framing | CPU-WRF d02 (s/fc-hr) | GPU d02 (s/fc-hr) | speedup |
|---|---:|---:|---:|
| Conservative — CPU clean compute | 83 | 15.68 | **5.29×** |
| Realistic — CPU incl. radiation + IO | 123 | 15.68 | **7.84×** |
| dt-matched floor (GPU forced to CPU dt = 6 s) | 83 | ~26.1 | ~3.2× |

GPU numerator = 43.556 ms/step → 15.68 s/fc-hr (with the bit-identical Thompson
sed-unroll). Sources: `proofs/thompson_perf/coupled_timing_base_vs_opt.json`,
`proofs/perf/warmed_timing.json`, `proofs/perf/speedup_denominator.md`.

A separate 24 h pipeline measurement (d03 L3 run, includes IC load + hourly GPU
forecast + wrfout writes + inventory + scoring) recorded a **9.09× speedup** vs the
CPU d02 24 h baseline (pipeline wall 1794 s vs CPU 16305 s).
Source: `proofs/v010_validation/speedup_vs_cpu_24h.json` (status PASS).

## Hardware roofline reference (RTX 5090, GeForce Blackwell)

| Quantity | Value | Source |
|---|---:|---|
| FP32 peak (non-tensor) | 104.88 TFLOP/s | `roofline_costonly.json` peak_specs |
| FP64 peak | **1.639 TFLOP/s** (= fp32 / 64) | `roofline_costonly.json` peak_specs |
| GDDR7 HBM bandwidth | **1.792 TB/s** | `roofline_costonly.json` peak_specs |
| FP64 roofline ridge | **0.914 FLOP/byte** | fp64_peak / BW |
| FP32 roofline ridge | **58.5 FLOP/byte** | fp32_peak / BW |

## Dycore-only per-step roofline placement (the crux)

| Metric | Value | Source |
|---|---:|---|
| dycore-only per-step wall (warmed) | 16.90 ms | `roofline_costonly.json` dycore_only_step |
| bytes accessed / step | 5.66 GB | " |
| FLOPs / step | 2.26 GFLOP | " |
| **arithmetic intensity** | **0.400 FLOP/byte** (fp64 ridge 0.914, fp32 ridge 58.5) | " |
| achieved fp64 | 0.134 TFLOP/s = **8.18 % of fp64 peak** | " |
| achieved HBM bandwidth | 0.335 TB/s = **18.70 % of HBM peak** | " |
| HBM-bandwidth floor | 5.66 GB / 1.792 TB/s = **3.16 ms** | " |
| fp64-compute floor | 1.38 ms | " |
| **actual / HBM floor** | **5.35×** (launch/serialization tax) | " |

**Verdict.** The dycore is **memory-bandwidth-bound at ~19 % of HBM peak**, with a
~5.3× kernel-launch/serialization tax on top. It is **decisively NOT fp64-compute-
bound** — fp64 arithmetic is only ~8 % of fp64 peak, and AI 0.40 is 146× below the
fp32 ridge. The 16.9 ms wall vs 3.16 ms floor gap is pure launch/latency overhead.

## Per-step decomposition (coupled step ≈ 44 ms)

| Phase | warmed ms | ms / HBM-floor | note |
|---|---:|---:|---|
| Thompson microphysics | 20.0 | 20× | ~half the step; ~85 % sedimentation |
| surface layer | 4.0 | 12× | |
| MYNN PBL | 2.9 | 4× | |
| advection tendencies | 1.0 | 2.7× | |
| vertical Thomas solve | 1.0 | 29× | XLA → cuSPARSE batched PCR (already parallel) |
| calc_p_rho (EOS) | 1.0 | 48× | launch-bound tiny kernel |
| boundary apply | 1.0 | 2.1× | |
| (remainder: prep, flux-aug, coef_w, halo) | ~3 | — | |

Source: `proofs/perf/phase_breakdown.json`,
`proofs/perf/compute_cycle_analysis.md`.

## Kernel-level confirmation (nsys, 36 warmed steps)

| Quantity | Value | Meaning |
|---|---:|---|
| GPU operations / step | ~11 160 (7 236 kernels + 3 922 mem ops) | enormous launch count |
| `loop_*` elementwise fusions | ~6 890 / step, ~7.3 ms | dominant genuine cost; ~1 µs each |
| device memory ops | ~3 922 / step, ~5.8 ms | scan concat / dyn-update-slice / halo pad |
| GPU idle fraction | ~43–68 % of the step | launch-bound |
| cuSPARSE PCR tridiagonal solve | ~2.0 ms / step | vertical w/φ solve, already parallel |

Source: `proofs/perf/nsys_warmed_step_stats_*.csv`,
`proofs/perf/compute_cycle_analysis.md`.

## Honest ceiling

≥10× is **not achievable without sacrificing WRF fidelity**. The four candidate
levers are each measured and refuted (see
`publish/tables/optimization_refutations.md`). The genuine remaining headroom is
the 5.3× dycore launch/serialization tax — closable only by precision-invariant
kernel-launch-COUNT reduction (whole-substep hand-fusion / higher unroll), which
perturbs the fp64 core bytewise, balloons compile/register pressure, and partly
lives in physics-owned files. Scientifically-grounded ceiling under strict WRF
fidelity: **~8–11×**; current **~5.3×/7.8×** is near-optimal for the faithful,
default-safe configuration. Source: `publish/runtime_optimization_analysis.md` §5,
`proofs/perf/compute_cycle_analysis.md`.

### CPU/GPU caveats (state with the headline)

1. d02-only / standalone — one GPU d02 vs one CPU d02, not vs the whole
   multi-domain nest (the retracted ~50–85× / 22.26× compared against the full
   nest wall).
2. fp64 on both sides; fp32 does not help this workload.
3. single GPU vs 28 CPU ranks on the same box; per-socket-vs-per-GPU.
4. dt asymmetry credited honestly (GPU dt = 10 s stable, CPU dt = 6 s); per-fc-hour
   comparison, strict dt-matched floor ~3.2×.
