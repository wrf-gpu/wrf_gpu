# BP84: RC3 ordinary-root endpoint evidence for v0.3.2 methods

The measured remaining cost is concentrated in acoustic passes, scalar transport and the movement of stage/carry arrays. The three examined residual families do not provide a supported new ≥0.5 ms/root design in this release window. This statement records their measured budgets and the limits of the bandwidth model. It does **not** certify maximum GPU speed: the assigned acoustic/layout levers remain open, and ncu counters are unavailable (B18).

## Measurement and model

BP82 recorded three ordinary and three night radiation roots on an RTX 5090 with release defaults and real PROD inputs. Each root advances d01 once and d02 three times. The ordinary **kernel + copy mean is 33.108602 ms/root [M]**; radiation is 81.662788 ms/root. The separate W1 spike-robust **kernel-only medians are 30.393184 / 78.092586 ms**. These estimators are not interchangeable.

BP83 combines those timings with actual RC3 optimized-HLO bytes. RC2-to-RC3 correspondence is [I] for this ranking, as requested: hot model sources match and the autotune pin is copied. Traffic estimates count operands/results, allowing slice-aware reads for XLA fusions, at an assumed 1.790 TB/s. Whole custom-call inputs, aliasing, cache reuse, duplicate reads and mixed-source fusions remain uncertain. Percentages are model ratios, not hardware utilization.

The unclipped model totals **36.451622 ms [I]**, exceeding the measured 33.108602 ms; it cannot be a calibrated DRAM lower bound. The six large d02 positive residuals sum 3.517701 ms, giving an optimistic selected-subset ratio of 89.38% (the proposed ‘≈90%’). Including every positive family residual gives 5.281219 ms, plus 0.310154 ms of unmodelled measured work. Clipping each inferred floor to its family measurement yields 27.517228 ms / 33.108602 ms = **83.11% [I]**, also not a hardware-attainment measurement. Negative residuals total -8.934394 ms and remain in BP83; they indicate model/cache uncertainty, not recoverable time.

## Per-family ordinary-root budget

| Family | Device ms/root [M] | Model floor ms [I] | Signed residual ms [I] | Launches/root [M] |
|---|---:|---:|---:|---:|
| dycore: acoustic | 9.835 | 17.373 | -7.539 | 626 |
| MYNN | 4.028 | 4.481 | -0.453 | 325 |
| dycore: advection | 3.304 | 2.161 | 1.143 | 242 |
| runtime/carry glue | 2.499 | 1.545 | 0.954 | 113 |
| dycore: scalar PD transport | 2.076 | 1.274 | 0.802 | 40 |
| lateral boundary | 1.709 | 1.508 | 0.201 | 273 |
| memcpy DtoD | 1.638 | 1.520 | 0.119 | 732 |
| other operators / coupling / unknown source | 1.637 | 1.402 + U | partly U | 185 |
| dycore: RK tendencies/update | 1.274 | 1.529 | -0.255 | 157 |
| Thompson | 1.264 | 0.861 | 0.403 | 8 |
| layout-only (all source families) | 1.238 | 1.265 | -0.027 | 442 |
| dycore: diffusion | 1.118 | 0.781 | 0.336 | 68 |
| Noah-MP | 0.558 | 0.441 | 0.117 | 98 |
| sfclay | 0.375 | 0.138 | 0.237 | 89 |
| outside advance | 0.282 | 0.000 + U | partly U | 99 |
| shared-symbol ambiguous buckets | 0.162 | 0.062 | 0.100 | 127 |
| RRTMG | 0.086 | 0.110 | -0.024 | 17 |
| small HtoD / DtoH transfers | 0.023 | 0.000 + U | partly U | 19 |
| command-buffer control | 0.003 | 0.000 + U | partly U | 2 |

The full d01/d02 and ordinary/radiation split, each exclusive ambiguity bucket, and top native nodes are retained in BP83. Grouping above preserves all measured time; overlapping source candidates are counted once.

## Why the three designs do not qualify

**Advection:** d02 costs 2.782678 ms with a +0.954284 ms model residual. The known u/v momentum fusion is already bitwise, but BD80-L1p measured ordinary device 30.74 → 30.90 ms: −1.20 ms advection was consumed by +0.34 ms native u/v work, +0.99 ms layout movement and +0.17 ms boundary work. d02 transposes changed 248 → 331. Repeating that consumer fusion is not a distinct design; a stable layout boundary is b-core’s assigned P2 prerequisite. The residual alone does not identify removable bytes.

**Carry glue:** d02 costs 2.223472 ms with a +0.864110 ms model residual. Its largest node, loop_add_multiply_fusion_2, already performs the moisture/boundary/stage assembly in **one launch per ownstep** (1.255171 ms across three launches/root). Its static operand/output charges are 203,553,324 + 71,474,832 bytes/launch, including 13 mass-grid outputs. Moving the same pointwise expressions into a separate kernel removes zero intermediate launches inside this fusion and retains the necessary state reads/writes; no ≥0.5 ms saving follows. G2’s qv layout anchor was tried on this carry state: the former hero transpose was already absent, and the anchor added 73 transposes / 265 MB per ownstep.

A distinct producer-to-update design would emit q_new directly from scalar flux/PD tendency kernels: the source builds transport after acoustic integration and then merges frozen boundary tendencies, map scaling and mass normalization (operational_mode.py:5508–5599). For this largest node, assuming eight solely consumed tendency buffers, bypassing them removes 16 dense field transfers, 263,907,072 bytes/root [I], from the intermediate-write/read path. The peak-bandwidth saving is 0.147434 ms; scaling its present 1.255171 ms cost by those bytes / its 825,084,468-byte model gives only 0.401472 ms [I] before new accumulator/boundary operands, packing or layout changes. Five other outputs still require production, so the three launches cannot simply disappear. These are traffic-only scenarios, not an achieved-gain bound; extending fusion across the other scalar stages requires separate producer/carry changes and P2 layout evidence. This design does not yet justify the ≥0.5 ms gate.

**PD:** d02 costs 1.930906 ms with a +0.752225 ms model residual. The two observed scale kernels together cost **0.503148 ms / six launches per root**. For all eight transported species their scale output payload is 131,953,536 bytes/root [I] (44×117×267×8×4×3), a 0.073717 ms write charge at peak; consumer reads/cache traffic are unknown. Eliminating that buffer requires the limiter for the cell and its six neighbours. BD80-L2 expanded seven cells × six faces × species and did not complete compilation within 1500 s. A compact neighbour loop changes compiler structure, but still recomputes those seven limiters, adds their input reads and has no measured occupancy/gain bound. The six saved launches and the scale-node ceiling do not support a net ≥0.5 ms saving after this extra work. A cooperative tile sharing neighbour scales would be a new transport-kernel design with halo synchronization and resource gates, not a short recurrence rewrite.

LW accumulation is parked: its 1.398378 ms/radiation-root node ceiling is only 0.002804 s/fc-h at actual cadence. MYNN coefficient+Thomas fusion is separately blocked by the measured TritonGPUCoalesce hang (BP77/E157); larger radiation tiles were slower on GPU (LP07/E174). Neither dead end is retried.

## What another 2× would require

With the other family costs held fixed, even deleting **all** d02 advection + carry glue + PD work saves only 6.937056 ms/root, limiting device speedup to 1.265× [I]. Deleting these three families in both domains saves 7.878218 ms/root and limits that scoped speedup to 1.312×. A 2× ordinary-device result needs a ≤16.554301 ms/root budget, so it must also reduce other major families, especially the 9.834751 ms acoustic budget, or change the layout/pass structure that feeds several families.

At 1.790 TB/s, that half-time budget permits at most 29.632 GB of actual DRAM traffic before compute, recurrence and synchronization costs [I]. Current actual DRAM bytes are unknown; the 65.248 GB HLO accounting is not that measurement. This condition does not prove 2× impossible or prove that a full rewrite is necessary.

A concrete route to investigate beyond local fusions is a common stable layout plus cooperative spatial/column tiles that retain acoustic coefficients, scalar face fluxes and stage intermediates in registers/shared memory across producer/consumer passes. Such a design must handle h5 stencil halos, seven-cell PD scale dependencies, the vertical acoustic recurrence, nest/specification ownership and the exact RK ordering. It would replace several kernel families together, with new resource/halo and pristine-WRF gates; it is not a change to one remaining physics loop. Whether it delivers 2× is [U]. A3 and P2 are nearer-term assigned steps toward this scope and require their own measurements.

## Whole-run and evidence limits

The independent integrated 24 h PROD warm receipt is **86.507 s = 3.604458 s/fc-h [M]**. CP147 partitions it into startup 17.0655 s, advance host calls 63.130984 s, force-down 0.781062 s, output callbacks 3.8905 s, tail 0.6264 s and residual 1.012554 s. Advance host wall includes device work and synchronization; the async writer overlaps execution. It cannot be added to BP82 device time as an independent cost.

The ordinary profiled root has wall 55.824261 ms and busy union 32.902873 ms [M]. Their difference includes Nsight instrumentation and is not a production host-gap saving. Small HtoD/DtoH rows are recorded and counted; no zero-transfer claim is made. The JSON retains their exact sizes per root.

These are PROD iteration-case costs, not a WN3 release-benchmark or fidelity claim. Radiation samples are night/first-step with KF and extra initial MYNN work, so their full excess is not a pure radiation cadence correction. Daylight, longer-run phase variation and the final composed v0.3.2 source need fresh evidence before transferring this packet to a release maximum-speed claim. nsys gives device duration/counts [M]; semantic grouping, operand traffic and rewrite budgets are [I]; achieved bandwidth, occupancy, stall causes and a global optimality bound remain [U].

## Provenance

- BP83: artifact:b-phys/BP83/ROOFLINE_RC3.md and ROOFLINE_RC3.json.
- BP82 trace: artifact:b-phys/BP82/profile_final/raw/profile.sqlite; W1_MATCHED_ROBUST.json.
- CP147: artifact:compile/v032/whole_run_anatomy.json and its hashed VAL31b extra24h receipt.
- Prior measured decisions: main EXPERIMENTS rows BD80-L1p-momuvn-GPU, G2-moist-anchor, BD80-L2-pd-fused; BP77/E157 and LP07/E174.
- Reproduce this reduction on CPU with endpoint.py; no GPU arm, source edit, snapshot or cache copy.
