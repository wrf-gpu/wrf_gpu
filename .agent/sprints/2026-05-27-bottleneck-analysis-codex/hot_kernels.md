# Hot-Kernel Ranking

Source artifacts:
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json`
- `/tmp/m7_profile_artifacts/m7_20260521_warm_360.sqlite`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json`

The cumulative ranking below uses the Nsight Systems SQLite kernel table, because `nsys_summary.json` stores the longest individual launches, not cumulative time by kernel name. The profiled 360-step 3 km d02 warm hour has 645,341 kernel launches and 792.807785 ms of summed GPU kernel time. The same trace records 641,741 `cuLaunchKernelEx` API calls totaling 3,815.142 ms, so launch/API overhead is a first-order bottleneck even though many individual kernels are tiny.

| Rank | Kernel | Calls | Total GPU ms | % GPU kernel time | Why hot | Recommended action |
|---:|---|---:|---:|---:|---|---|
| 1 | `loop_add_fusion_4` | 313,200 | 244.292 | 30.81 | Huge call count; average launch is only 0.780 us and often scalar/small-array XLA loop bookkeeping. | Fuse scan-carry housekeeping and state-update arithmetic; remove repeated per-field add chains where a coarser fused stage can compute the same values. |
| 2 | `pcrGtsvBatchSharedMemKernelLoop<double>` | 1,080 | 169.201 | 21.34 | Batched vertical tridiagonal/PCR solve in FP64; dynamic shared memory 2,048 B and 39 regs/thread. | Treat as fundamental until parity is protected, then evaluate tailored 44-level solver or coefficient+solve fusion. |
| 3 | `loop_multiply_fusion` | 158,410 | 158.696 | 20.02 | Elementwise multiply-heavy scan/body work; average launch is 1.002 us, so launch count and memory traffic both matter. | Fuse with neighboring add/subtract/guard operations in RK/acoustic stages. |
| 4 | `loop_subtract_fusion` | 154,800 | 143.873 | 18.15 | Elementwise subtract-heavy state delta work; same launch pattern as rank 3. | Fuse into pressure-gradient, tendency, and guard kernels rather than materializing standalone deltas. |
| 5 | `pcrGtsvBatchFirstPass<double>` | 1,080 | 30.734 | 3.88 | First pass for FP64 vertical tridiagonal/PCR solve. | Same solver-track action as rank 2; do not optimize before savepoint/Tier-4 guardrails. |
| 6 | `pcrGtsvBatchSharedMemKernelLoop<float>` | 720 | 8.802 | 1.11 | FP32 batched solver path, probably physics/PBL-linked. | Keep as lower priority than FP64 solver; profile callsite split once NCU is available. |
| 7 | `input_concatenate_fusion` | 3,602 | 8.504 | 1.07 | Layout assembly / concatenate kernels from state, boundary, or column adapters. | Eliminate transient concatenates by making adapters consume native layout or by fusing pack/unpack with the consumer. |
| 8 | `input_transpose_fusion_42` | 360 | 8.029 | 1.01 | Column-layout transpose with 92 regs/thread and 42,240 B static shared memory. | Attack physics column layout; avoid repeated `moveaxis` where vertical-last views can be retained inside physics blocks. |
| 9 | `loop_reverse_fusion` | 3,600 | 7.433 | 0.94 | Reverse-scan/back-substitution style work; one long launch reaches 243.904 us. | Link to tridiagonal solver audit; reduce separate reverse kernels if a custom vertical solve is adopted. |
| 10 | `pcrGtsvBatchFirstPass<float>` | 720 | 5.023 | 0.63 | First pass for FP32 batched solver path. | Same as rank 6. |

Largest realistic speedups:
- The top elementwise trio (`loop_add_fusion_4`, `loop_multiply_fusion`, `loop_subtract_fusion`) accounts for 546.861 ms of GPU kernel time, but its larger cost is the hundreds of thousands of launches and associated D2D/API traffic. This is the highest-value optimization area.
- The vertical solver family accounts for 213.759 ms of direct GPU kernel time. It is algorithmically important but not the first wall-clock target unless a solver redesign also removes adjacent temporary buffers.
- Physics/layout kernels are smaller in direct GPU time, but are still good fusion candidates because they create transposes, concatenates, and transient arrays that show up in memory pressure.
