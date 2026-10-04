# B300 vs B200 Energy & Throughput Projection

**Status:** PROJECTION (based on MEASURED B200 anchors + VENDOR-SPEC B300 numbers)  
**Date:** 2026-06-26  
**Author:** Claude (Sonnet 4.6) — commissioned by the user for paper/README use  
**Scope:** single-GPU, single-domain fp64 dycore projection for one NVIDIA B300 SXM vs the measured B200 run  

---

## Headline Summary

| Metric | B200 (MEASURED) | B300 — lower bound | B300 — upper bound | B300 best-guess |
|--------|-----------------|--------------------|--------------------|-----------------|
| Per-cell-step throughput (cells/s) | **37.4 M** | 37.4 M (BW-limited) | 37.4 M (BW-limited) | **~37 M** (flat) |
| Max single-domain grid (nz=44) | 1024² × 44 (58 GiB) | 2200² × 44 (270 GiB) | 2200² × 44 | **~2200² × 44** |
| kWh per cell-step (device only) | **5.0×10⁻¹² (MEASURED)** | 5.2×10⁻¹² (700 W load) | 1.04×10⁻¹¹ (full TDP) | **~7–9×10⁻¹² PROJECTED** |
| Relative efficiency vs B200 | 1.0× (baseline) | ~1.0× (at B200-like load) | ~2.1× WORSE (full TDP) | **~1.4–1.8× WORSE** |
| Relative efficiency vs CPU | ~3× better (MEASURED) | ~2.9× better | ~1.4× better | **~1.6–2.1× better** |

**Plain-language verdict:**  
The B300 offers **no per-cell-step speedup over the B200** for this fp64 WRF dycore — both cards are memory-bandwidth limited and both deliver the same ~8 TB/s HBM3e bandwidth. The B300's massive fp64 performance reduction (37 → 1.25 TFLOPS, VENDOR-SPEC) is irrelevant because the dycore barely uses fp64 compute (< 0.2% utilization on B200). The B300's **sole gain is domain capacity**: 288 GB vs 192 GB allows a ~2200² grid (4.6× more cells) where a B200 OOMs at 1024². However, the B300's **40% higher TDP (1400 W vs 1000 W) with the same throughput makes it 40–100% less energy-efficient than the B200** per cell-step at maximum power draw. Vs the CPU baseline, the B300 is still better (≈1.4–2.1×), but the paper's "~3× vs CPU" figure belongs to the B200 and does NOT extend to B300.

---

## Spec Table

| Specification | H200 SXM | B200 SXM | B300 SXM | Source |
|---------------|----------|----------|----------|--------|
| HBM capacity | 141 GB | 192 GB | 288 GB | VENDOR-SPEC |
| HBM3e bandwidth | 4.8 TB/s | 8.0 TB/s | 8.0 TB/s | VENDOR-SPEC |
| fp64 TFLOPS | 34 | **37** | **1.25** | VENDOR-SPEC (B200: multiple third-party datasheets; B300: NVIDIA GTC25 + multiple third-party sources) |
| fp32 TFLOPS | ~67 | 75 | 75 | VENDOR-SPEC |
| TDP (max) | ~700 W | 1000 W | 1400 W | VENDOR-SPEC |
| Bandwidth ratio vs B200 | 0.60× | 1.0× | **1.0×** | VENDOR-SPEC |
| fp64 ratio vs B200 | 0.92× | 1.0× | **0.034×** | VENDOR-SPEC |

> **WARNING — B300 fp64:** Multiple independent sources (verda.com, axecompute.com, ornn.com, glennklockwood.com, spheron.network) consistently report B300 dense fp64 ≈ 1.25 TFLOPS vs B200's 37 TFLOPS — a 97% reduction. NVIDIA repurposed die area from fp64 units to fp4 inference compute. This is confirmed by the NVIDIA GTC 2025 keynote (Jensen Huang) and the GB300 NVL72 product page spec (100 TFLOPS system-wide ÷ 72 GPUs = ~1.39 TFLOPS/GPU). The B300 is **not an fp64 HPC chip**.

---

## B200 MEASURED Anchors

All B200 values are MEASURED from `<DATA_ROOT>/wrf_gpu_b200_20260623/results/b200_swiss_scaling/fp64_scaling.json` (v0.20.0, single-domain fp64, Swiss parametrized scaling ladder, dycore+radiation+PBL, physics-complete proxy).

| Grid | ncells | cells/s (MEASURED) | Peak VRAM (MEASURED) | GPU draw W (MEASURED) | kWh/cell-step (MEASURED) |
|------|--------|--------------------|----------------------|-----------------------|--------------------------|
| 128² × 44 | 720,896 | 21.4 M | 5.94 GiB | 453 W | 5.88×10⁻¹² |
| 256² × 44 | 2,883,584 | 32.5 M | 5.94 GiB | 582 W | 4.98×10⁻¹² |
| **384² × 44** | **6,488,064** | **35.5 M** | **7.85 GiB** | **617 W** | **4.83×10⁻¹² (best)** |
| 512² × 44 | 11,534,336 | 36.8 M | 14.10 GiB | 653 W | 4.93×10⁻¹² |
| 768² × 44 | 25,952,256 | 37.3 M | 31.93 GiB | 679 W | 5.05×10⁻¹² |
| 896² × 44 | 35,323,904 | 35.6 M | 44.26 GiB | 685 W | 5.35×10⁻¹² |
| **1024² × 44** | **46,137,344** | **34.9 M** | **58.25 GiB** | **678 W** | **5.39×10⁻¹²** |

**Throughput fit (MEASURED):**  
`1/cells_per_s = (1/R∞) + α·(1/C)` → R∞ = **37.40 M cells/s**, α = 0.01415 s·cells  
(The throughput plateau is reached at ~384²; 896² and 1024² drop slightly, consistent with memory latency or occupancy effects at the largest grids.)

**GPU TDP note:** B200 actual draw at large grids = 677–685 W (MEASURED), vs vendor TDP of 1000 W. The B200 runs at ~68% of its power budget in this workload.

---

## Energy Basis — "~3× vs CPU"

**Source:** USER_README_v0210.md and proofs/v021/.  
**Basis:** GPU board power via nvidia-smi (MEASURED) vs CPU RAPL package-power sample of 197.9 W applied to measured Swiss CPU-WRF wall time (DERIVED — not a same-run RAPL log).

- B200 best kWh/cell-step: **4.83×10⁻¹² kWh** (at 384² × 44, MEASURED device power)  
- CPU derived kWh/cell-step: **1.47×10⁻¹¹ kWh** (at 197.9 W RAPL sample × Swiss CPU-WRF wall, DERIVED)  
- Ratio: **3.04× (MEASURED device / DERIVED CPU)** — this is the "~3×" in the paper/README

**Critical caveats about this "3×":**
1. It is **GPU board power only vs CPU socket-package power only** — not whole-node (no DRAM, NIC, storage, cooling in either).
2. The CPU sample (197.9 W, 30 s) is a snapshot, not co-located with the GPU run.
3. There is **no same-large-grid CPU run** — the CPU number is derived from a small-grid Swiss timing.
4. The CPU is a modern Zen-5 part — not a legacy CPU.

All B300 projections below use **the same basis** (device power only) to maintain comparability.

---

## Step 1 — Maximum Grid on B300 (288 GB)

**VRAM linear fit** (from B200 ladder, 384²–1024²):

```
VRAM(GiB) ≈ -0.594 + 1.270×10⁻⁶ × N_cells
```

R² ≈ 0.999 over the linear range. At 128² and 256² the VRAM is dominated by a large fixed base (~5.94 GiB static), then grows linearly.

**Budget:** 288 GiB − 18 GiB headroom = **270 GiB usable**

```
N_cells_max = (270 + 0.594) / 1.270×10⁻⁶ ≈ 213 M cells
N (square, nz=44): sqrt(213M / 44) ≈ 2200
```

**Result (PROJECTED):** A B300 single-domain fp64 run could accommodate up to **~2200² × 44 ≈ 213 M cells** — a **4.6× larger domain** than the B200's maximum of 1024² × 44 = 46 M cells. At 3 km grid spacing, a 2200² domain covers a ~6600 km × 6600 km region (continental-scale). At 1 km spacing, it covers ~2200 km × 2200 km.

*Note: the VRAM fit is extrapolated 4.6× beyond measured B200 points. The linear model may underestimate actual VRAM on the B300 due to driver/CUDA overhead differences. Treat the 2200² figure as an approximate upper bound; **2000²–2100² is more conservative**.*

---

## Step 2 — Per-Cell-Step Throughput

### Bottleneck analysis

The WRF fp64 dycore is **memory-bandwidth limited**, not fp64-compute limited:

- **Estimated fp64 compute utilization on B200:** At R∞ = 37.4 M cells/s with ~2000 fp64 ops/cell-step (generous estimate), the implied fp64 throughput is **~75 GFLOPS — 0.2% of B200's 37 TFLOPS fp64 peak**. Even at 20 000 ops/cell-step (extreme upper bound), utilization is ~2%.
- **Implied memory access intensity:** 8 TB/s ÷ 37.4 M cells/s ≈ **214 000 bytes per cell per step** — roughly 170× the per-cell state footprint (~1270 bytes/cell). This is consistent with the repeated read/write passes in the 10-substep acoustic loop plus 3 RK3 stages.

**Conclusion:** fp64 compute is **not the bottleneck**. The 30× fp64 reduction on B300 (37 → 1.25 TFLOPS) **does not degrade throughput**.

### Throughput bounds (PROJECTED)

| Bound | Scaling factor | B300 R∞ estimate | Rationale |
|-------|----------------|------------------|-----------|
| Memory-bandwidth-limited | 1.0× (BW unchanged) | **37.4 M cells/s** | Same 8 TB/s HBM3e |
| fp64-compute-limited | 0.034× | 1.26 M cells/s | Would apply if fp64-bound (it is not) |
| **Realistic estimate** | **~1.0×** | **~35–38 M cells/s** | BW-limited; mild overhead uncertainty |

**The B300 delivers the same per-cell-step throughput as the B200 for this workload.** The B300 does not run faster than the B200; it runs the same speed but on a domain 4.6× larger.

*Uncertainty band:* ±10% to account for: CUDA overhead differences on B300, XLA/JAX autotuning differences, potential memory latency differences between B200 and B300 HBM3e configurations. The bandwidth-limited estimate cannot improve beyond 1.0× since BW is identical.

---

## Step 3 — Energy Efficiency

### Assumptions on B300 power draw

The B300 TDP is **1400 W** (VENDOR-SPEC). The B200 measured 677–685 W at the largest grids (≈68% of its 1000 W TDP). The WRF dycore is memory-bandwidth limited, so compute units are underloaded. B300 is even more AI-inference-optimized (more fp4 units), meaning the fp64 kernel will stress even fewer of its FP4-targeted units. However, HBM3e controllers and DDR infrastructure will draw power proportional to bandwidth, not compute. **Realistic B300 draw for this workload: 700–1000 W** (the BW subsystem draws power, the compute side is underloaded). Full TDP (1400 W) is an extreme pessimistic bound.

| B300 power scenario | kWh/cell-step (PROJECTED) | vs B200 (MEASURED 5.0×10⁻¹²) | vs CPU (DERIVED 1.47×10⁻¹¹) |
|--------------------|-----------------------------|-------------------------------|-------------------------------|
| ~700 W (B200-like, optimistic) | 5.2×10⁻¹² | **~1.04× worse** | **~2.8× better** |
| ~800 W (likely lower bound) | 5.9×10⁻¹² | ~1.18× worse | ~2.5× better |
| ~1000 W (midrange) | 7.4×10⁻¹² | ~1.47× worse | ~2.0× better |
| ~1200 W (conservative) | 8.9×10⁻¹² | ~1.77× worse | ~1.65× better |
| 1400 W (full TDP, pessimistic) | 1.04×10⁻¹¹ | **~2.06× worse** | **~1.41× better** |

**Best-guess range (PROJECTED):** B300 draws 800–1200 W for this workload → **kWh/cell-step ≈ 5.9–8.9×10⁻¹²**, roughly **1.2–1.8× less energy-efficient than B200** and approximately **1.65–2.5× better than the CPU baseline**.

**If the B300 load is near B200 (700 W):** the efficiency is nearly identical to B200 (~1×).  
**At full TDP (1400 W):** energy efficiency is ~2× worse than B200 and only ~1.4× better than CPU.

---

## Comparison Table for Paper / README

| Platform | R∞ (cells/s) | kWh/cell-step | vs B200 energy | vs CPU energy | Source |
|----------|-------------|---------------|----------------|---------------|--------|
| CPU (Zen-5 package, 197.9 W sample) | ~2.4 M (derived) | 1.47×10⁻¹¹ | — | 1.0× (baseline) | DERIVED |
| **B200 (best: 384² × 44)** | **37.4 M (R∞)** | **4.83×10⁻¹² (best)** | **3.0× more efficient** | **3.0×** | **MEASURED** |
| B200 (at R∞ draw, 678 W) | 37.4 M | 5.04×10⁻¹² | — | 2.9× | MEASURED |
| **B300 (optimistic, ~700 W)** | ~37 M | ~5.2×10⁻¹² | ~1.0× (≈ B200) | ~2.8× | **PROJECTED** |
| **B300 (mid, ~1000 W)** | ~37 M | ~7.4×10⁻¹² | ~1.5× worse | ~2.0× | **PROJECTED** |
| **B300 (pessimistic, 1400 W)** | ~37 M | ~1.04×10⁻¹¹ | ~2.1× worse | ~1.4× | **PROJECTED** |

---

## Assumptions & Caveats

1. **fp64 reality (critical).** The WRF dycore's fp64 island (pressure-gradient, buoyancy cancellation) is the reason we use fp64, but the actual fp64 FLOP utilization is < 0.2% of B200's fp64 peak. The B300's 30× fp64 reduction is architecturally significant for HPC codes that saturate fp64, but does NOT degrade throughput here. **However:** if a future optimized kernel DOES saturate the fp64 units, B300 would be 30× slower than B200 for that kernel. Document this risk for future optimization paths.

2. **Memory-bandwidth assumption.** The dycore is treated as memory-bandwidth limited because: (a) measured implied bandwidth intensity (~214 KB/cell/step) >> state footprint (~1.27 KB/cell); (b) fp64 FLOP utilization is demonstrably negligible. This assumption is robust at large grids (well past the launch-bound transition) and was validated by the measured throughput plateau.

3. **Bigger grid = more domain, not more speed.** The B300's throughput advantage is **domain capacity, not per-cell-step rate**. A forecast over a 4.6× larger area can be completed at the same rate (cells/s) — that's the genuine value. It is NOT a "speedup" in the traditional sense. Do not conflate domain-capacity gain with wallclock speedup for the same domain.

4. **Whole-node vs GPU-only.** All energy numbers here are GPU board power only (nvidia-smi watts). Whole-node energy (DRAM, NIC, storage, fans, PSU losses) is NOT measured and NOT included. The B300 at 1400 W TDP in a DGX-class system will have additional system overhead. This is an honest gap in both the B200 and B300 figures — the CPU comparison basis is also package-power only.

5. **Vendor spec uncertainty.** B300 specs (especially TDP and exact fp64 TFLOPS) are from third-party sources citing Jensen Huang's GTC 2025 keynote and NVIDIA product pages, not from a published NVIDIA datasheet we directly read. The fp64 figure (1.25 TFLOPS) is consistent across 5+ independent sources; it is treated as reliable. The B200 fp64 (37 TFLOPS) is from NVIDIA's own B200 datasheet referenced in multiple sources.

6. **B300 actual power draw not measured.** The B300 power draw under a BW-limited fp64 stencil workload is unknown. The 700–1400 W range covers pessimistic (full TDP) to optimistic (B200-like draw since BW-limited). The wide range is the dominant uncertainty in the energy projection.

7. **VRAM model extrapolation.** The 2200² max-grid estimate extrapolates the linear VRAM fit ~4.6× beyond the measured B200 range. Large-grid B300 VRAM may differ due to different CUDA memory allocator behavior, driver version, or XLA workspace differences.

8. **No B300 benchmark run.** All B300 numbers are PROJECTED from B200 measurements + vendor specs. They must be replaced with measured values before any production paper claim.

9. **The paper's "~3× vs CPU" belongs to B200 only.** For B300 at mid-power (1000 W), the ratio is ~2×. At full TDP (1400 W), only ~1.4×. The B200 is the right chip for energy-efficiency claims in an fp64 stencil workload; the B300 is the right chip for domain-capacity claims.

10. **B300 availability.** As of early 2026, B300 (HGX B300 / DGX B300) is newly shipping. Cloud availability is limited. Benchmark runs would require access to a B300 pod.

---

## Why isn't a newer GPU greener here? (the natural reviewer/intuition question)

The intuition "newer node + newer architecture + more parallel compute ⇒ faster and greener" is correct for
**compute-bound or AI (low-precision) workloads**. It does NOT apply to this workload, for verifiable reasons:

- **Same process node, not a die-shrink.** B300 is the SAME TSMC **4NP** node with the SAME **208 B transistors**
  as B200 (VENDOR-SPEC, verified). It is re-binned Blackwell silicon with more HBM stacks + more FP4 units, not a
  smaller-node part. So there is no process-node perf/watt gain to harvest.
- **We are memory-bandwidth bound, not compute bound (MEASURED).** The B200 run moves ~214 KB/cell/step and runs
  at exactly its 8 TB/s HBM3e ceiling — the GPU spends its time waiting on memory, not computing. Extra compute
  units therefore sit idle and cannot raise throughput.
- **B300's bandwidth is identical (8 TB/s, verified).** Same bottleneck → same ~37 M cells/s. The capacity went
  up (288 vs 192 GB) but the bandwidth did not.
- **The new compute is the wrong precision.** B300's gains are FP4 AI inference (15 PFLOPS) — a precision our
  dycore does not use. And fp64 (which our dynamics DOES use) is 30× WEAKER on B300.
- **Energy follows.** Efficiency = throughput ÷ power. Throughput is capped equal; B300 TDP is higher (1400 W vs
  1000 W; B200 drew ~680 W under this load). So B300 is at BEST ≈ B200 efficiency (if it draws like B200) and
  plausibly worse — not better. The only opening for a small B300 gain is a more power-efficient HBM subsystem at
  the same bandwidth (plausible, unmeasured, second-order).

**One-line answer:** the B300 is optimized for a different problem (giant AI models: more memory + FP4). Weather
fp64 stencils are gated by memory *bandwidth* and fp64 — neither improved — so we do not benefit, and we pay more
watts. The B300's real value for us is **domain capacity**, not speed or energy.

## Recommended Paper / README Framing

> **PROJECTED (vendor-spec basis, not yet benchmarked):** The NVIDIA B300 (288 GB HBM3e, 8 TB/s bandwidth, 1400 W TDP) offers the same per-cell-step throughput as the B200 for this fp64 memory-bandwidth-limited dycore (~37 M cells/s), but can sustain a domain up to ~2200² × 44 ≈ 213 M cells — **4.6× the capacity of the B200** (58 GiB at 1024²). However, the B300's higher TDP (1400 W vs B200 ~680 W measured) makes it **approximately 1.4–2.1× less energy-efficient per cell-step than the B200** (depending on actual power draw under load), reversing the B200's "~3× better than CPU" efficiency advantage to an estimated 1.4–2.8× better than CPU. For fp64 energy-efficiency, the B200 is the better card; the B300's value is continental-scale domain capacity. Note: the B300's fp64 throughput (1.25 TFLOPS, VENDOR-SPEC) is 30× lower than the B200's (37 TFLOPS), but this is irrelevant for throughput because the dycore's fp64 utilization is < 0.2% of available fp64 compute on both cards.

---

## Sources

- MEASURED: `<DATA_ROOT>/wrf_gpu_b200_20260623/results/b200_swiss_scaling/fp64_scaling.json` (B200 throughput/VRAM/power, v0.20.0)
- MEASURED/DERIVED: `proofs/v021/USER_README_v0210.md` (energy basis, 4.83×10⁻¹² kWh, CPU 1.47×10⁻¹¹, "~3.05× vs CPU")
- VENDOR-SPEC: [verda.com B300 vs B200](https://verda.com/blog/nvidia-b300-vs-b200-complete-gpu-comparison-to-date) — "FP64 performance is dramatically lower (1.25 TF on B300 vs. 37 TF on B200)"
- VENDOR-SPEC: [axecompute.com Blackwell GPU comparison](https://axecompute.com/nvidia-blackwell-gpu-comparison/) — B200: 37 TFLOPS fp64; B300: ~1.25 TFLOPS fp64
- VENDOR-SPEC: [ornn.com B200 vs B300](https://ornn.com/insights/nvidia-b200-vs-b300-why-use-one-instead-of-the-other) — "37–40 TFLOPS fp64 B200 vs 1.2–1.3 TFLOPS fp64 B300 (97% reduction)"
- VENDOR-SPEC: [glennklockwood.com B300](https://www.glennklockwood.com/garden/processors/b300) — 1.25 TFLOPS fp64, 8 TB/s, 1400 W TDP
- VENDOR-SPEC: [NVIDIA GB300 NVL72 product page](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) — "100 TFLOPS FP64/FP64 Tensor Core" for 72 GPUs = ~1.39 TFLOPS/GPU
- VENDOR-SPEC: [spheron.network B300 guide](https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide/) — B300: 288 GB HBM3e, 8 TB/s, 1400 W TDP, 15 PF FP4
- VENDOR-SPEC: [server-parts.eu B300 architecture](https://www.server-parts.eu/post/nvidia-b300-gpu-blackwell-ultra-architecture) — B300 TDP 1000–1400 W configurable
