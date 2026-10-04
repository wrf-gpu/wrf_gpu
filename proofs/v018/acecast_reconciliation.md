# AceCAST 10-16x vs wrf_gpu2 1.0-1.3x: quantified reconciliation

Date: 2026-06-16  
Analyst: GPT-5  
Scope: evidence-only reconciliation. No model code changes and no new GPU runs.

## Executive verdict

The AceCAST headline and our v0.17 single-card result are mostly
**apples-to-oranges**. The largest differences are:

1. **AceCAST is node-to-node and multi-GPU.** The cited 16.8x result is one
   Azure ND A100 v4 node with **8 A100 GPUs** versus one HBv3 node with
   **64 CPU cores**, for **inner-domain compute + communication**. Normalized
   per GPU, that is about **2.1x of a 64-core CPU node per A100** before any
   precision/domain/scope caveats. The 10-16x "node-to-node" claim is likewise
   against a GPU node, not one GPU.
2. **Their domains are large contiguous CONUS domains.** Publicly cited cases are
   about **425x300 = 127k horizontal points** and **1501x1201 = 1.8M points**.
   Our v0.17 wall-clock blocker case is the Canary all-7 live nest: about
   **55k total mass columns split across nine domains**, with seven tiny 1 km
   children and about **5,000 advances + 4,400 boundary builds per forecast-hour**.
3. **Their likely precision/hardware class is favorable; ours is deliberately
   unfavorable.** Our shipped number is fp64 on a consumer RTX 5090, whose fp64
   peak is in the same ballpark or worse than the local CPU. Valid fp32/mixed
   experiments in this repo do **not** yield a free 4x: the safe dycore/mixed
   lanes repeatedly measured near **1.0-1.11x**, and storing pressure/geopotential
   totals in fp32 corrupts the PGF/geopotential path by **27x-127x** beyond budget.
4. **We did have real JAX/XLA inefficiencies, but they do not explain a hidden
   10x gap after v0.17.** The big ones were compile churn, high-frequency host
   sync, eager boundary/host dispatch, and small-kernel topology. v0.17 fixed or
   gated those. The measured opt-in fused all-7 path is **1.27x**, or **1.30x**
   with edge-only boundary, at about **96% GPU utilization** and a **~674 s/hr**
   remaining GPU-compute floor. That is not a 10x recoverable hole.

Hypothesis verdict:

| Hypothesis | Verdict |
|---|---|
| (a) Apples-to-oranges | **Yes, dominant.** Multi-GPU node, large domains, likely fp32/mixed/tolerance validation, inner-domain timing, datacenter A100s. |
| (b) Core/dummy-only | **Not the main explanation for AceCAST as cited**, but it applies strongly to other comparisons: Fahrenheit/OpenACC is dycore-only, and our own early 22x/50x/150x-class numbers were on structurally incomplete/core-only or skill-red code. |
| (c) Our extreme inefficiency | **Some real inefficiency existed and was fixed; no evidence of a remaining 10x same-config inefficiency.** Same hardware+precision+tiny-nest headroom is roughly tens of percent to maybe ~1.5-1.6x with optional physics islands, not 10-16x. |

## 1. Multi-GPU node-to-node normalization

External facts checked:

- NVIDIA's AceCAST/Azure writeup states **16.8x faster than WRF for inner-domain
  compute and communication** on **1x NDmA100v4, 8 GPUs**, compared with
  **1x HBv3, 64 CPU cores**.
- The same writeup says the optimal single-job comparison was AceCAST on one
  8-GPU NDmA100 node versus WRF on 16 HBv3 CPU VMs, with AceCAST 7% faster and
  75% lower cost.
- TempoQuest/HPC coverage frames "10-16x" as **node-to-node**, holding resource
  cost against a GPU system, and notes AceCAST is a WRF subset rather than a full
  WRF port.

Arithmetic:

```text
AceCAST Azure headline: 16.8x / 8 A100 = 2.1x of one 64-core CPU node per A100
```

Our v0.17 all-7 proof:

```text
CPU baseline: 7146 s / 8 h = 893 s per forecast-hour, 12-rank CPU WRF
GPU opt-in fused+edge-only: 689 s per forecast-hour
Measured factor: 893 / 689 = 1.30x
```

If we normalize our 1x RTX 5090 result to a hypothetical 64-core CPU node by
assuming ideal CPU scaling from the 12-rank denominator:

```text
1.30x of 12 ranks ~= 0.24x of a 64-core CPU node
AceCAST per A100 ~= 2.1x of a 64-core CPU node
Per-GPU normalized gap ~= 2.1 / 0.24 = 8.6x
```

That 8.6x is **not** a clean software-efficiency gap. It folds in A100 vs RTX
5090 fp64 suitability, 8-GPU node design, large-domain occupancy, precision,
and metric scope. If one instead uses older 24/28-rank denominators, the
normalized gap shrinks, but the qualitative answer does not change: AceCAST's
public number is not "one GPU beats one same-workstation CPU run by 16x."

## 2. Domain size and occupancy

AceCAST:

- CONUS 12 km: **425x300 ~= 127k** horizontal points.
- CONUS 2.5 km: **1501x1201 ~= 1.8M** horizontal points.
- Large single domains amortize launches and expose memory bandwidth/SM
  occupancy.

Our v0.17 all-7:

- About **55k total mass columns**, split across d01, d02, and seven small 1 km
  children.
- The live-nested cadence issues about **5,000 `_advance_chunk` calls** and
  **4,400 boundary builds** per forecast-hour.
- The deepest proof calls out **many ~1.5 us kernels**, no single hotspot, and
  tiny under-filled domains; **82.6% of weighted column-steps** are in the seven
  tiny 1 km leaves.

Measured/projected scaling inside our repo:

- v0.17 all-7, opt-in fused+edge-only: **689 s/hr = 1.30x vs 12-rank CPU**, and
  still not default because fused changes the bit path.
- AC1_FIT standalone d03 1 km, about **145k columns**, measured **557 ms/step**
  warm on RTX 5090 fp64. The 24 h estimate is **~3.8x** vs the correct 12-rank
  CPU at matched dt, or **~5-6x** at the operational 1 km dt, but this is
  honestly labeled standalone/step-rate and omits some live-coupled costs.
- v0.15 large-grid characterization for the fp64 coupled/core step refuted the
  old "6-10x large-grid" story: deployment band **~1.6-2.7x**, centered around
  **~2x**, with an asymptotic large-grid fp64 floor **~1.63x vs CPU-28**.
- Bigswiss **211,600 cols** in fp64 does **not fit** one 32 GB RTX 5090 in the
  production path; no measured GPU wall-clock exists. The core-only projection
  if it fit was **4.35x**, explicitly not a measurement and excluding full
  physics/boundary/I/O.

Conclusion: yes, our **1.3x is strongly a small nested-domain artifact**, but a
large single-domain version on this one fp64 consumer GPU does not become 10-16x.
The evidence supports roughly **2-4x class** on favorable single-card cases if
they fit and are measured honestly; **CONUS-scale** belongs to datacenter
multi-GPU/VRAM, not a single 5090 fp64 run.

## 3. Precision and hardware class

Hardware facts:

- NVIDIA A100 has **9.7 TFLOP/s FP64 CUDA-core** and **19.5 TFLOP/s FP64 Tensor
  Core / FP32**, plus 40-80 GB HBM-class memory per GPU depending SKU.
- RTX 5090 has 32 GB GDDR7 and GeForce-class fp64 throttling. Project proofs
  quantify the local effective truth: fp64 on the 5090 is not a free GPU win,
  and for some WRF fp64 paths the local CPU is competitive.

Repo precision evidence:

- v016/v017 safe mixed-fp32 dycore paths measured about **1.04x**, **0.996x**,
  or **1.105-1.111x**, with essentially **no VRAM relief**.
- Full-working-set fp32 was double-confirmed not viable for valid speed:
  p/ph total storage in fp32 loses bits before any fp64 island can recover them.
  The cited corruption is **~27x geopotential / ~127x PGF** over the accepted
  perturbation-fp32 budget.
- The 4.29x true-fp32 proxy is real as a **cost proxy**, but invalid as a
  forecast because it removes the fp64 cancellation safeguards.
- Optional fp32 physics islands can be material in isolated or gated contexts
  (around **1.5-1.6x** class), but v0.17 all-7 analysis says they still do not
  lift the full precision-safe stack above **2x** on the tiny nested geometry.

Conclusion: precision/hardware explains a large part of the difference. AceCAST
can plausibly use fp32/mixed "within tolerance" on A100-class nodes. Our local
proof standard has repeatedly forced fp64 around the acoustic/PGF/geopotential
cancellation path. That is not a one-line dtype flip.

## 4. JAX/XLA overhead: real, but bounded after v0.17

Real inefficiencies found:

- Cold compile churn: all-7 repeatedly compiled per-domain executables and in a
  cold-cache run produced **0 wrfout after 65 min** before being killed.
- Per-advance host sync: the nested loop drained the device queue about
  **5,000 times per forecast-hour**.
- Eager boundary construction and fine-grained domain recursion: about
  **4,400 boundary builds/hour**.
- Fused cascade was needed to reduce host dispatch from about **47 to 5** per
  root step.

Measured after fixes:

| Mode | Warm s/hr | Factor vs 893 s/hr CPU | GPU util | Status |
|---|---:|---:|---:|---|
| root-sync eager | 1005 | 0.89x | 56% | bitwise, CPU-parity-minus |
| fused cascade | 702 | 1.27x | ~96% | opt-in, tolerance-pass, not bitwise |
| fused + edge-only | 689 | 1.30x | ~96% | opt-in fused; edge-only bit-identical |

The important post-fix number is the **~674 s/hr GPU-compute floor**. Once the
fused path keeps the GPU busy, the remaining all-7 wall is not mostly Python
idle. nsys shows many tiny generic XLA kernels and no single hot spot; the
tiny-nest geometry under-fills the GPU.

What remains recoverable?

- Same hardware, same fp64, same all-7 geometry: no proof of a hidden 2-3x, much
  less 10x. The measured stack ends at **1.30x**; the report's own precision-safe
  optional extensions were **~1.5-1.6x projected**, still below **2x**.
- A hand-written CUDA/OpenACC/Pallas lane may help selected kernels or dycore-only
  cases, but v0.15-v0.17 repeatedly refuted simple command-buffer, launch-packaging,
  Thomas-only, and whole-op fusion as full-step wins. For full physics on this
  tiny-nest workload, the evidence does not support "XLA is hiding AceCAST-class
  speed."
- On different hardware/domain/precision, all bets change: large resident
  datacenter-GPU domains are exactly where a GPU-native code should be measured.

## 5. Metric scope

AceCAST's cited Azure number is **inner-domain compute + communication**. That
can exclude I/O, initialization, some outer-domain/nesting work, and first-run
compile/setup costs.

Our all-7 warm proof includes live-nested orchestration and warm hourly output
work. The host ledger shows output around **34 s/hr** in the warm fused accounting,
about **5%** of the 689 s/hr fused+edge wall. Cold fused compile is a much larger
one-time cost, about **38 minutes**, but it is usually excluded from the warm
speed factor just as AceCAST excludes setup.

Metric scope is therefore a real caveat, but not the main 10x explanation. It is
probably a **single-digit to low-double-digit percent** effect for warm timing,
while multi-GPU/domain/precision/hardware are order-one factors.

## 6. Early wrf_gpu2 speedups and dycore-only comparisons

The early high multipliers are not comparable to the v0.17 result:

- The memory record `project_22x_speedup_honesty_2026_05_28.md` says the
  measured **22.55x** came while the dycore still missed major operators:
  `advance_uv`, full `advance_w`, `calc_p_rho` cadence, `rk_addtend_dry`,
  flux-form advection, physics-as-RK1 bundle, scalar flux accumulation, and
  correct boundary cadence.
- Earlier M7/M11 style **50x/150x** claims were on incomplete or skill-red
  configurations and were later corrected or retired.
- Open-source OpenACC/Fahrenheit-style **~6x** comparisons are explicitly
  dycore-only or otherwise not full WRF physics.

So hypothesis (b) is important historically and for some outside comparisons,
but it should not be used as the sole explanation for AceCAST. AceCAST appears
to be a production-oriented subset/within-tolerance system on large domains,
not merely a dummy dycore benchmark. The gap is mainly apples-to-oranges plus
precision/hardware/domain scale.

## 7. Like-for-like bottom line

For **our current shipped/local configuration**:

- One RTX 5090, fp64, Canary all-7 tiny live nests: measured honest opt-in
  headline **~1.27-1.30x** vs the 12-rank CPU baseline; default-safe path is
  slower/near parity unless the non-bitwise fused mode is enabled.
- Same setup, same validity constraints: **>=2x is not proven reachable**;
  v0.17 concluded it should stop as a single-card tiny-nest goal.

For **favorable single-card but still local RTX 5090 cases**:

- Large standalone 1 km d03-like grid: step-rate estimates can be
  **~3.8-5.7x** vs 12-rank CPU, with clear caveats about live coupling and
  omitted/heavier physics.
- Production fp64 large single domains hit the **32 GB VRAM wall** before
  CONUS-scale cases.

For **AceCAST-like fairer territory**:

- Datacenter A100/H100/H200-class node, large contiguous domains, fp32/mixed or
  native-FP64 hardware, resident state, tolerance validation, inner-domain timing,
  and multi-GPU scaling: **10-16x node-to-node is plausible** and not contradicted
  by our 1.3x local result.
- To make a comparable wrf_gpu2 claim, we would need measured large-domain
  multi-GPU/datacenter runs with explicit precision and metric scope, not
  extrapolation from the all-7 RTX 5090 proof.

## Sources

Local proof/decision sources:

- `proofs/v017/hostgap_fix_opus.md`
- `proofs/v017/hostgap_fix_gpt.md`
- `proofs/v017/gpu_utilization_investigation.md`
- `proofs/v017/hostgap_gpu_util_summary.json`
- `proofs/v017/canary_nested_estimates.md`
- `proofs/v017/canary_grid_selection_INTERIM.txt`
- `proofs/v017/bigswiss_gpu_benchmark.md`
- `proofs/v017/fitting_switzerland_benchmark.md`
- `proofs/perf/v015/kernel_characterization.md`
- `.agent/decisions/KERNEL-OPTIMIZATION-FINDINGS-FINAL.md`
- `.agent/decisions/V017-PERFORMANCE-RELEASE-PLAN.md`
- `proofs/perf/v016/FP32_PERTURBATION_SPEEDUP_RISK_ANALYSIS.md`
- `proofs/perf/v016/FP32_GRADIENT_CHALLENGE_CRITICAL_REVIEW.md`
- `proofs/perf/v017/FP32_MEMORY_COMPUTE_REAUDIT.md`
- `proofs/v018/k1_fp32_stability_scoping.md`
- `<USER_HOME>/.claude/projects/-home-user-src-wrf-gpu2/memory/project_fp32_makeorbreak_overnight_2026_06_14.md`
- `<USER_HOME>/.claude/projects/-home-user-src-wrf-gpu2/memory/project_22x_speedup_honesty_2026_05_28.md`

External web sources checked:

- NVIDIA developer blog, AceCAST on Azure:
  https://developer.nvidia.com/blog/microsoft-and-tempoquest-accelerate-wind-energy-forecasts-with-acecast/
- TempoQuest/HPC coverage of 10-16x node-to-node:
  https://tempoquest.com/blogs/post/meet-tempoquest-a-startup-bringing-weather-code-into-the-accelerated-era
- NVIDIA A100 official specs:
  https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA GeForce RTX 5090 official product page:
  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
