# v0.17 README / NCAR Forum: GPU Energy-to-Solution Estimate

Date: 2026-06-15

Purpose: concise, honest framing for the main *system-level* advantage of the WRF-GPU port: on large enough domains, native-FP64 datacenter GPUs can reduce both wall-clock time and energy-to-solution versus conventional CPU clusters.

## Bottom line

For a sufficiently large WRF-GPU case where the domain fills VRAM, the model state stays resident on the GPU, and compile/kernel-launch overhead is amortized:

- **Best near-term FP64 target:** 8x **H200 SXM** or H100/H200-class HGX server.
- **Expected speedup:** about **20-30x versus one high-end dual-socket CPU node**.
- **Expected energy-to-solution:** about **3-5x lower kWh** for the same large forecast.
- **Normalized example:** if a fixed large forecast costs **100 kWh** on a CPU-only allocation, a favorable H200-class GPU run plausibly costs **~20-30 kWh**.
- **GB200 NVL72 rack-scale FP64:** much larger absolute throughput, plausibly **~175-270x versus one CPU node**, but rack power is high, so the energy gain is still likely **~2-3x** unless measured data proves more.
- **GB300 NVL72:** very interesting for memory capacity / global-grid thought experiments (**20 TB GPU memory, 37 TB fast memory**), but NVIDIA's public GB300 page currently lists only **100 TFLOP/s FP64 / FP64 Tensor Core**, so it should not be used as the strongest FP64-performance claim without clarification or measurement.

These are **spec-based estimates, not benchmark claims**.

## Preferred unit

For README / NCAR Forum, use **energy-to-solution**: kWh for the same forecast/domain/resolution/physics configuration. That is clearer and less fragile than “kWh per grid cell” before full measured production benchmarks exist.

After production measurements exist, report additionally:

- kWh per simulated forecast-hour
- kWh per 10^12 3D cell-updates
- wall-clock seconds per simulated forecast-hour

## Hardware assumptions

CPU reference node:

- Dual AMD EPYC 9654-class CPU node
- Approx. **7.4 TFLOP/s FP64 peak** from 192 Zen4 cores at base clock
- Approx. **0.92 TB/s DDR5 memory bandwidth**
- Approx. **1.2 kW full-node power** including CPUs, memory, board, cooling overhead inside the node

GPU assumptions:

- Native-FP64 datacenter GPUs, not consumer RTX.
- Large domain, resident model state, enough work per step to amortize Python/JAX/XLA launch and compile overhead.
- Effective speedup range = min(FP64 ratio, HBM-bandwidth ratio) multiplied by an efficiency factor for stencil/halo/scaling overhead.
- H100/H200 estimates use non-tensor FP64 throughput because WRF stencil/dycore code should not be assumed to map to FP64 Tensor Cores.

## Compact estimate table

| Platform | Public spec basis | Estimated speedup vs 1 dual-EPYC node | Energy for CPU=100 kWh | Interpretation |
|---|---:|---:|---:|---|
| 8x H100 SXM | 272 TFLOP/s FP64, 26.8 TB/s HBM, ~8 kW node | ~16-22x | ~31-42 kWh | Strong FP64; memory smaller than H200. |
| 8x H200 SXM | 272 TFLOP/s FP64, 38.4 TB/s HBM, ~8 kW node | ~22-31x | ~21-30 kWh | Best near-term framing for large WRF-GPU FP64. |
| GB200 NVL72 rack | 2,880 TFLOP/s FP64, 13.4 TB HBM3E, 576 TB/s, assumed ~100 kW rack | ~175-270x | ~31-48 kWh | Huge throughput; energy gain moderated by rack power. |
| GB300 NVL72 rack | 20 TB GPU memory, 37 TB fast memory, 576 TB/s; public page lists 100 TFLOP/s FP64 | memory thought experiment only | do not claim FP64 gain from public spec alone | Great capacity story; FP64 spec needs clarification. |

## Global 1 km thought experiment

Earth surface area is about **510 million km²**, so a 1 km global horizontal grid has roughly **510 million horizontal cells**.

Approximate 3D cell counts:

| Vertical levels | 3D cells |
|---:|---:|
| 60 | ~30.6 billion |
| 80 | ~40.8 billion |
| 100 | ~51.0 billion |

Approximate FP64 state/scratch memory:

| Bytes per 3D cell | 60 levels | 80 levels | 100 levels |
|---:|---:|---:|---:|
| 256 B | ~7.8 TB | ~10.4 TB | ~13.1 TB |
| 512 B | ~15.7 TB | ~20.9 TB | ~26.1 TB |
| 1024 B | ~31.3 TB | ~41.8 TB | ~52.2 TB |

Interpretation:

- A lean global 1 km FP64 dynamical core state could plausibly fit into **GB300 NVL72's 20 TB GPU memory** if the effective working set is near the 256-512 B / 3D-cell range and scratch is tightly controlled.
- A more conservative full-physics, multi-tracer, multi-scratch implementation may need **>20 TB**, but GB300's **37 TB fast memory** still makes it an important architecture target.
- This illustrates the strategic advantage of the port: once the state is resident on large GPU memory, the limiting cost becomes high-bandwidth FP64 throughput and communication, not thousands of CPU-node memory systems.

## Suggested README wording

> On consumer GPUs such as an RTX 5090, FP64 throughput and VRAM capacity make this a deliberately unfavorable test case. The more relevant target for large scientific workloads is native-FP64 datacenter hardware. On large domains where the model state remains GPU-resident and launch/compile overheads are amortized, H100/H200/GB200-class systems should plausibly deliver tens of CPU nodes of throughput per GPU server and several-fold lower energy-to-solution. These estimates are based on public hardware specifications and must be replaced by measured kWh benchmarks as production runs mature.

## One-sentence NCAR Forum version

> On sufficiently large domains where kernel-launch and compilation overheads are amortized, an optimized native-FP64 WRF-GPU port on H100/H200/GB200-class datacenter GPUs should plausibly replace on the order of 20-30 high-end dual-socket CPU nodes per 8-GPU server and reduce energy-to-solution by roughly 3-5x, but these are hardware-spec-based estimates rather than benchmark claims.

## Sources checked

- NVIDIA H100 page: https://www.nvidia.com/en-us/data-center/h100/ — H100 SXM FP64 34 TFLOP/s, FP64 Tensor Core 67 TFLOP/s, 80 GB, 3.35 TB/s, up to 700 W.
- NVIDIA H200 page: https://www.nvidia.com/en-us/data-center/h200/ — H200 SXM FP64 34 TFLOP/s, FP64 Tensor Core 67 TFLOP/s, 141 GB, 4.8 TB/s, up to 700 W.
- NVIDIA GB200 NVL72 page: https://www.nvidia.com/en-us/data-center/gb200-nvl72/ — 72 GPUs, 13.4 TB HBM3E, 576 TB/s, FP64 / FP64 Tensor Core 2,880 TFLOP/s listed.
- NVIDIA GB300 NVL72 page: https://www.nvidia.com/en-us/data-center/gb300-nvl72/ — 72 GPUs, 20 TB GPU memory, 37 TB fast memory, 576 TB/s, FP64 / FP64 Tensor Core 100 TFLOP/s listed.

## Caveats

- These numbers are for *large, optimized, resident-state* cases, not the current RTX 5090 local development setup.
- Real WRF-GPU performance will depend on halo exchange, XLA fusion quality, MPI/NVLink topology, time-step structure, I/O, checkpointing, physics mix, and whether the hottest kernels are bandwidth-bound or compute-bound.
- For publication/release claims, measured wall-clock and measured power are required. Until then, phrase as “plausible hardware-spec-based estimate.”
