# Memory vs Compute Classification

Nsight Compute did not provide hardware counters: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/ncu_hot_kernels.json` records `BLOCKED-PROFILER` for the top three kernels because NVIDIA performance-counter permission failed with `ERR_NVGPUCTRPERM`. The classifications below therefore use Nsight Systems timing, launch counts, kernel names, register/shared-memory metadata from the SQLite export, and source-level HLO/JAX reasoning.

| Kernel | Classification | Basis | Candidate fusion or replacement target |
|---|---|---|---|
| `loop_add_fusion_4` | Launch-bound, with memory-move side effects | 313,200 calls, 0.780 us average, 1 block x 1 thread average. This is too small to be useful device work. | Merge scan-carry field updates in `runtime/operational_mode.py`; inspect `_with_save_family`, `_enforce_operational_precision`, guard/state `replace` paths, and XLA tuple shuffling. |
| `pcrGtsvBatchSharedMemKernelLoop<double>` | Compute/latency-bound solver | 156.667 us average, 10,494 blocks, 64 threads/block, dynamic shared memory. It is real vertical solve work, not a tiny launch. | Replace only with a validated 44-level batched Thomas/PCR variant, or fuse coefficient construction with solve to cut temporaries. |
| `loop_multiply_fusion` | Mixed launch-bound and memory-bound | 158,410 calls, 1.002 us average, mostly elementwise full-array arithmetic. | Fuse with neighboring add/subtract/scaled-tendency and pressure-gradient operations. |
| `loop_subtract_fusion` | Mixed launch-bound and memory-bound | 154,800 calls, 0.929 us average, same pattern as multiply. | Fuse with state-delta, pressure/geopotential perturbation, and guard computations. |
| `pcrGtsvBatchFirstPass<double>` | Compute/latency-bound solver | 28.458 us average with the same batch geometry as the PCR loop. | Same solver-track replacement as the shared-memory loop. |
| `pcrGtsvBatchSharedMemKernelLoop<float>` | Compute/latency-bound solver, lower direct cost | 12.224 us average, 720 calls. | Defer until callsite attribution shows whether it is MYNN or another physics solve. |
| `input_concatenate_fusion` | Memory-bound layout assembly | Concatenate kernels move data between layouts and do little arithmetic. | Fuse pack/unpack into physics/boundary consumers; avoid separate padded-strip construction in the timestep path. |
| `input_transpose_fusion_42` | Memory-bound with register pressure | 92 regs/thread and 42 KB static shared memory; one launch per step. | Rework column adapters so Thompson/MYNN/RRTMG consume a stable vertical-last view or fuse transposes into the column kernels. |
| `loop_reverse_fusion` | Solver-related memory/launch-bound | Reverse-scan naming and 3,600 launches point to back-substitution or reversed loop outputs. | Fold into custom vertical solver or reduce scan materialization. |
| `pcrGtsvBatchFirstPass<float>` | Compute/latency-bound solver, low direct cost | 6.976 us average. | Same as FP32 solver loop. |

Trace-level memory signals:
- GPU memcpy time in the trace is 271.859 ms; 263.636 ms of that is Device-to-Device, moving 39.894 GB across 316,909 D2D copies.
- CUDA API time is dominated by launch overhead: `cuLaunchKernelEx` totals 3,815.142 ms and `cuMemcpyDtoDAsync_v2` API time totals 1,517.955 ms.
- The 2026-05-27 profiler-window fix proves `d2h_inter_kernel_inside_window == 0`; the remaining opportunity is not host transfer but launch count, D2D copies, and transient materialization.

Priority inference:
1. Treat tiny elementwise loops and D2D traffic as the main launch/memory bottleneck.
2. Treat PCR/tridiagonal kernels as real solver work with high correctness risk.
3. Treat transposes/concatenates as memory-layout problems that are worth fixing because they amplify transient pressure even when direct kernel time looks small.
