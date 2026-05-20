# Architectural Review: Stage M4 Assessment of the 4x+ Performance Target

**Date:** 2026-05-20  
**Reviewer:** Gemini (Stage M4 Architecture Review)  
**Objective:** Evaluate architectural fitness and project plan for the 4x+ speedup target (RTX 5090 vs. 28-core CPU Ryzen 9 9950X) for the Canary Islands 1 km and 9/3 km nested grids.

---

## 1. Executive Votum

**Votum: CONDITIONAL APPROVAL / REQUIRES URGENT ARCHITECTURAL INTERVENTION**

The JAX-based state-resident model architecture (established in [ADR-001](file:///home/enric/src/wrf_gpu2/.agent/decisions/ADR-001-backend-selection.md) and [ADR-002](file:///home/enric/src/wrf_gpu2/.agent/decisions/ADR-002-state-layout.md)) is **highly optimized** for defeating legacy GPU porting bottlenecks (such as the 200,000+ launch-bound loop structure of legacy OpenACC ports). However, the plan and architecture contain **critical omissions and physical contradictions** that will prevent achieving a 4x+ speedup on a consumer GPU (RTX 5090) compared to a heavily parallelized 28-core CPU baseline. 

The three major showstoppers are:
1. **The FP64 Hardware "Tax" on Consumer GPUs:** The RTX 5090 has a severely throttled double-precision (FP64) compute throughput. If the model is locked to FP64 per [ADR-003](file:///home/enric/src/wrf_gpu2/.agent/decisions/ADR-003-dycore-precision.md), the GPU compute rate is lower than the Ryzen 9 9950X CPU, making it mathematically impossible to achieve a 4x+ speedup.
2. **The Complete Omission of Grid Nesting in the Plan:** The current plan ([PROJECT_PLAN.md](file:///home/enric/src/wrf_gpu2/PROJECT_PLAN.md)) states "no nesting in v0" (`PROJECT_PLAN.md:21`), yet the target requires 9/3/1 km nested grids. Adding nesting post-hoc to JAX's static-shape JIT model is extremely difficult and requires a fundamental redesign of the state container.
3. **Launch-Bound Latency on Outer Nested Domains:** Coarse nested domains (9 km and 3 km) are small grids where actual compute is negligible, meaning their execution time on the GPU will be dominated by host-to-device kernel launch latency.

---

## 2. Hardware Capabilities & The FP64 Throttling Bottleneck

The local Ryzen 9 9950X CPU has 16 physical cores (32 threads). The RTX 5090 is a top-tier Blackwell consumer GPU. However, their raw double-precision (FP64) compute capabilities present a stark contrast due to NVIDIA's product segmentation:

* **GeForce RTX 5090 (Consumer):** Throttled to a **1:64** ratio of FP64 to FP32 throughput. While FP32 performance is a massive ~120 TFLOPS, peak FP64 is limited to **~1.7 - 1.8 TFLOPS**.
* **AMD Ryzen 9 9950X (CPU):** Features dual AVX-512 FMA units per core. At a sustained AVX-512 clock speed of 4.3 GHz, it achieves:
  $$\text{Peak FP64} = 16 \text{ cores} \times 4.3 \text{ GHz} \times 32 \text{ FLOPs/cycle} \approx 2.2 \text{ TFLOPS}$$

### Hardware Comparison Matrix
| Metric | CPU: Ryzen 9 9950X | GPU: GeForce RTX 5090 | GPU-to-CPU Ratio |
| :--- | :--- | :--- | :--- |
| **FP32 Theoretical Peak** | ~3.0 TFLOPS | ~120.0 TFLOPS | **~40.0x** |
| **FP64 Theoretical Peak** | ~2.2 TFLOPS (AVX-512) | ~1.8 TFLOPS (1:64 rate) | **~0.82x (GPU Bottleneck)** |
| **Memory Bandwidth** | ~70 - 80 GB/s (DDR5) | ~1792 GB/s (GDDR7) | **~22.0x - 25.0x** |

**Conclusion:** In pure FP64 compute-intensive scenarios, the GPU has **less** compute power than the Ryzen CPU. While memory-bandwidth-bound stencil dynamics will benefit from the GPU's GDDR7 bandwidth, any compute-bound component (specifically the physics schemes) will run slower on the GPU than on the CPU.

---

## 3. Amdahl's Law Impact on the 4x+ Target

In standard WRF runs, the runtime is split between memory-bandwidth-bound dynamics and compute-bound physics. Let's model a typical baseline scenario where dynamics is 30% of the runtime and physics is 70% of the runtime. 

* **Dynamics speedup on GPU:** Since dynamics is highly memory-bandwidth bound, let's assume it achieves a 15x speedup using the GPU's high bandwidth:
  $$\text{Dynamics GPU Time} = \frac{30\%}{15} = 2\%$$
* **Physics speedup on GPU (FP64 locked):** Since physics is compute-bound and limited by the throttled FP64 ALUs, the GPU is no faster than the CPU:
  $$\text{Physics GPU Time} = \frac{70\%}{1.0} = 70\%$$
* **Overall Speedup:**
  $$\text{Overall Speedup} = \frac{1}{0.02 + 0.70} = 1.39\text{x}$$

Even if the dynamics ran in 0 seconds, the maximum possible speedup is limited by the physics compute speed:
$$\text{Max Speedup} = \frac{1}{0.70} = 1.43\text{x}$$

**Votum:** Under the current [ADR-003](file:///home/enric/src/wrf_gpu2/.agent/decisions/ADR-003-dycore-precision.md) FP64 production lock, the 4x+ speedup target is **physically impossible** to reach.

---

## 4. Omission of Grid Nesting in JAX

The user's operational requirement includes "1km and 9/3km nested grid of the Canary Islands." The current plan and JAX architecture are strictly single-domain. Adding nesting to JAX is a major challenge:

1. **JAX Shape Specialization:** JAX/XLA compiles programs for static array shapes. A nested simulation has three grids of different shapes (e.g., $90 \times 90$, $180 \times 180$, and $500 \times 300$). If JAX JITs separate functions for each grid, compilation times and VRAM usage will double or triple.
2. **Temporal Synchronization and Sub-stepping:** The coarse grid (9 km) may use a timestep of $dt = 30\text{ s}$, while the 3 km grid uses $dt = 10\text{ s}$, and the 1 km grid uses $dt = 3.33\text{ s}$. Orchestrating this temporal coupling inside a compiled `jax.lax.scan` loop is non-trivial and has not been designed.
3. **Boundary Condition Interpolation & Feedback:** Nesting requires interpolating variables spatially and temporally at the boundary of child grids, and feeding back fine-grid variables to parent grids. Doing this in JAX without violating memory residency or triggering host-to-device transfers requires custom, complex JAX stencil indexings.

---

## 5. Launch-Bound Latency on Outer Nested Domains

Although whole-state residency prevents host-to-device transfers, XLA compiles the dynamics and physics into multiple discrete GPU kernels. 

* The M4 reduced dry dycore alone compiles to **24 kernel launches per timestep** ([MILESTONE-M4-CLOSEOUT.md](file:///home/enric/src/wrf_gpu2/.agent/decisions/MILESTONE-M4-CLOSEOUT.md)).
* Host-to-device launch latency on modern Linux systems is 3 - 5 microseconds per kernel. With 30+ kernels per step, launch latency adds **~100 - 150 μs** of overhead.
* For the 9 km outer domain (which might only be $60 \times 60 \times 50$ points), the actual computation time on an RTX 5090 is less than 5 μs.
* The GPU will spend **>95% of its time** waiting for kernel launches rather than doing physics. On nested domains, the latency of outer grids will dominate and degrade the overall speedup.

---

## 6. VRAM Footprint & Table Compilation Storms

The RTX 5090 has 32 GB of VRAM.
1. **Memory Footprint:** A 1 km grid over the Canary Islands ($500 \times 300 \times 50$) contains 7.5 million points. At FP64, storing 100 arrays (prognostics, diagnostics, coordinate parameters, tendencies, history) requires ~6 GB of memory. Nested 9 km and 3 km grids add another ~2 GB. Because JAX uses immutable updates, XLA often allocates temporary buffers during execution. Combined with land-surface (Noah-MP) and radiation (RRTMG) variables, memory usage could easily exceed 32 GB, triggering GPU OOM errors.
2. **Lookup Table Compilation:** Thompson microphysics and RRTMG radiation rely on massive lookup tables. If these are inlined in JAX tracers, the compiler will unroll lookups into nested conditional select statements, leading to compilation OOM on the host or massive instruction bloat in the GPU kernels.

---

## 7. Recommendations and Action Plan

To ensure the Canary Islands performance target is reached, the following architectural adjustments must be made:

### 1. Re-Evaluate and Amend Precision Policy (P0)
* **Transition to Mixed Precision:** Dynamics pressure gradient and mass continuity calculations should remain FP64 to preserve stability. However, dynamics advection and physics tendencies must transition to FP32 or BF16.
* **Unlock Tensor Cores:** Running physics in FP32/BF16 allows the RTX 5090 to utilize its full compute power, providing a 40x compute advantage over the CPU and bypassing the FP64 hardware throttle.

### 2. Design the Nesting Architecture early (P1)
* Do not defer nesting to a post-v0 phase. The `State` and `GridSpec` classes must be redesigned to support multiple grid states.
* Implement a **Static Padding Strategy** or a **Multi-JIT Nesting Coordinator** that maps nested grids into structured JAX pytrees, allowing XLA to compile the coupling operators without shape recompilation storms.

### 3. Handle Lookup Tables via Dynamic Masking / Triton (P1)
* Prevent table unrolling by storing large tables (like the Thompson table and RRTMG tables) as device-resident constant arrays.
* Access tables via dynamic gathering (`jax.lax.gather`) or write custom Triton kernels for microphysics and radiation table lookups, preventing XLA compilation OOM.

### 4. Group Small-Grid Timesteps (P2)
* Use JAX stream scheduling or group operations on the 9 km and 3 km outer domains to execute in parallel, minimizing launch latency overhead.
