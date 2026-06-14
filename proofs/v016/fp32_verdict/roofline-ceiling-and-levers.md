# Performance Ceiling and Architecture Levers

Date: 2026-06-14
Author: GPT-5.5 xhigh
Branch: `worker/gpt/v017-gap-analysis`
Scope: analysis only; no model-code changes.

## Verdict

The honest single-GPU answer for the RTX 5090 is:

- Current fp64 production is already close to the practical JAX/XLA fp64 ceiling for this card: about **1.6-2.7x vs the 28-rank CPU at the 560x280 Canary-1km footprint**, centered near **2x**.
- v016 `mixed_perturb_fp32` is **not** the large fp32 lever: it measures only **1.10-1.11x** and **0x peak-VRAM relief** at 16k/65k columns, and both fp64 and mixed OOM at 147k columns.
- A validated native/operational fp32 rewrite should be planned as **~2-3x single-GPU production speedup plus ~2x VRAM reduction**, not as a proven 6x speedup. The VRAM halving is still strategically important because it unlocks 1km footprints that fp64 OOMs on.
- **A ~6x Canary-1km speedup on this 5090 is not physically supported by the current algorithm and artifacts** unless it also gets a major HBM-traffic/fusion reduction outside the excluded single megakernel path. The HLO roofline proves why: the operational step is low arithmetic-intensity and fp32 becomes memory-bound.

No GPU commands were run for this analysis. No GPU lock was acquired.

## Evidence Base

Primary artifacts read:

- `/home/enric/src/wrf_gpu2/proofs/perf/v016/s2_hlo_stats_2x2_s1.json`
- `/home/enric/src/wrf_gpu2/proofs/perf/v016/s2_hlo_stats.py`
- `/home/enric/src/wrf_gpu2/proofs/perf/v016/fp32_s2_mixed_ladder.json`
- `/home/enric/src/wrf_gpu2/.wt-fp32-opus/proofs/perf/v016/fp32_km_bench.py`
- `/home/enric/src/wrf_gpu2/.wt-fp32-opus/proofs/perf/v016/fp32_km_bench_native.log`
- `proofs/perf/v015/d01_roofline_costonly.py`
- `proofs/perf/compute_cycle_analysis.md`
- `proofs/perf/fp32_downcast_plan.md`
- `proofs/perf/fp32_downcast_spec.md`
- `.agent/decisions/KERNEL-OPTIMIZATION-FINDINGS-FINAL.md`
- `.agent/decisions/REDUCED-PRECISION-EQUIVALENCE-AND-FP32-RIGOR.md`
- `proofs/perf/v015/kernel_characterization.{md,json}`
- `proofs/perf/v015/viability/{true_fp32_cost_proxy,fp32_fp64_ladder,VIABILITY_VERDICT}.json`
- `src/gpuwrf/runtime/operational_mode.py`
- `proofs/multigpu_dgx/*`

Missing requested artifact: `proofs/perf/v016/fp32_km_bench.json` was not present in this worktree, the main `worker/gpt/v016-fp32-s2` worktree, or `.wt-fp32-opus`. The native log exists, but contains only a GPU-lock wait line. The small log failed before execution due a `taskset` invocation error. Native-fp32 remains a placeholder until rerun.

## 1. Roofline

The v016 HLO stats are the decisive quantitative roofline input. They compile one operational-step body at 256x256x44, 65,536 columns, with the v016 S2 precision variants.

| precision | FLOP/step | bytes accessed/step | arithmetic intensity | arg bytes | output bytes | temp bytes | HLO float tokens |
|---|---:|---:|---:|---:|---:|---:|---|
| fp64 | 128.43 GF | 126.00 GB | **1.019 FLOP/B** | 1.120 GB | 1.056 GB | 8.222 GB | f64 76,067 / f32 2,288 |
| mixed_s2 | 125.69 GF | 108.81 GB | **1.155 FLOP/B** | 0.811 GB | 0.747 GB | 8.128 GB | f64 38,248 / f32 38,409 |

RTX 5090 assumptions from the sprint prompt:

- fp32 peak ~= 105 TFLOP/s
- fp64 peak ~= 1.7 TFLOP/s
- HBM ~= 1.8 TB/s

Ridge points:

- fp64 ridge = 1.7 / 1.8 = **0.94 FLOP/B**
- fp32 ridge = 105 / 1.8 = **58.3 FLOP/B**

Interpretation:

- The fp64 graph at AI ~= 1.02 sits **right at the fp64 compute/bandwidth boundary**, slightly on the fp64-compute side. Its ideal floors are about **75.5 ms compute** and **70.0 ms HBM** for the compiled 65k-column step.
- The same algorithm in fp32 is **unambiguously memory-bound**: AI ~= 1 is nowhere near the fp32 ridge of 58 FLOP/B. The 64x fp32 ALU rate is therefore not globally accessible.
- If all downcastable full-grid arrays really halve their bytes, the memory-bound ceiling is roughly **2x** from bytes alone. The fp64-side compute boundary can add a little, giving a hard ideal of about **2.0-2.2x** for the traffic-shaped part.
- Compute-bound subgraphs can see large local fp32 gains. On this card, a pure fp64-compute island can in principle improve by up to **64x** if converted to fp32. That does not imply a 64x, 10x, or 6x full-step speedup because the whole step is low-AI and dominated by repeated array traffic plus residual fp64 islands.

The mixed_s2 HLO is a useful sanity check. It cuts total HLO bytes only **13.6%** (`126.00/108.81 = 1.158x`) and temp arena only **1.1%** (`8.222/8.128 = 1.012x`). The measured warm speedup at 65k columns is **1.113x**, almost exactly the byte-ratio story.

## 2. Reconciling the Data

### Why mixed_s2 is only 1.1x

`fp32_s2_mixed_ladder.json`:

| ncol | fp64 ms/step | mixed_s2 ms/step | speedup | fp64 VRAM | mixed_s2 VRAM |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 70.48 | 63.96 | **1.102x** | 3.767 GiB | 3.767 GiB |
| 65,536 | 255.55 | 229.53 | **1.113x** | 11.609 GiB | 11.609 GiB |
| 147,456 | OOM | OOM | n/a | OOM | OOM |

The HLO explains this:

- fp64 HLO float tokens drop from 76,067 to 38,248, but the mixed graph still has about **half f64 and half f32 float tokens**.
- `convert` ops rise from **132** to **703**, so the graph is still a mixed-precision boundary machine rather than a clean fp32 graph.
- Argument and output sizes drop by about **28-29%**, but the binding temporary arena barely moves.
- HLO bytes fall by only **13.6%**, and measured speedup is **11.3%**.

So mixed_s2 proves the current micro-island precision split did not remove the fp64 common root. It downcasts visible leaves but leaves enough fp64 work and full-grid temporaries that the memory/arena ceiling remains.

### What native fp32 should give

The requested native-fp32 bench JSON is absent. The log at `.wt-fp32-opus/proofs/perf/v016/fp32_km_bench_native.log` contains only:

```text
[with_gpu_lock] opus-native-bench waiting for GPU lock; current holder: holder=v016-sweep ...
```

Method to fill the placeholder:

```bash
bash scripts/with_gpu_lock.sh --label gpt-perf-ceiling -- \
  env GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
      OMP_NUM_THREADS=4 \
      PYTHONPATH=src \
      taskset -c 0-3 \
      python /home/enric/src/wrf_gpu2/.wt-fp32-opus/proofs/perf/v016/fp32_km_bench.py
```

Expected proof fields: warmed ms/step, peak VRAM, compile wall, finite flag, and fp64-vs-native ratios at 2x2, 3x3, 4x4 tile factors.

Before that rerun, the defensible expectation is:

- **Against current mixed_s2:** native fp32 should do much better if it removes the residual fp64 graph and halves the temp arena.
- **Against the physics-valid production step:** native fp32 should be bounded around **~2-3x**, not 6x, unless it also cuts HBM round-trips/fusion count materially.
- **For VRAM:** native fp32 should recover the important win: about **2x smaller working set**, consistent with v015 true-fp32 cost proxy (`147,456` cols fit at 9.65 GiB where fp64 OOMed, and `262,144` cols fit at 16.80 GiB).

### Why the old 4.3x and 3-6x claims must be narrowed

`true_fp32_cost_proxy.json` measured a numerically invalid all-fp32 cost proxy:

- 16k cols: fp64 70.49 ms, true-fp32 16.44 ms = **4.29x**
- 147k cols: fp64 OOM, true-fp32 119.3 ms at 9.65 GiB
- 262k cols: true-fp32 210.6 ms at 16.80 GiB

That proxy is still useful as a proof that fp32 can unlock memory, but it is not a production speed claim:

- It globally toggled x64 off; numerics are explicitly invalid.
- It removed more fp64 work than the validated mixed_s2 architecture currently removes.
- It used a stripped core/cost setup, not the full production path with boundaries, GWD, NoahMP, conservation locks, and long-horizon equivalence.

The v016 HLO data narrows the headline. The validated/near-valid mixed path is 1.1x. A future native path can plausibly recover the VRAM and a meaningful speedup, but a full-step **6x on this 5090 is not supported** by the measured arithmetic intensity and byte traffic. The honest target is **2-3x plus VRAM unlock**; a higher number requires a second lever that reduces HBM traffic or kernel granularity.

## 3. Architecture Levers Beyond Precision

The single fuse-everything megakernel rewrite is explicitly excluded. Intermediate fusion remains in scope.

| lever | estimated gain | mechanism | finite branch-only experiment |
|---|---:|---|---|
| BouLac dense O(nz^2) -> O(nz) | **1.08-1.15x** full step; 7-14.5 ms at 128^2 | Remove the dense `(B,nz,nz)` BouLac reductions, the largest named non-Pallas device item left. | Branch `worker/opus/v017-boulac-onz`: implement WRF incremental form, run L1 oracle + Tier-P field gate, then nsys/NVTX at 128^2 and cost ladder at 65k+. |
| Intermediate RK-stage fusion | **1.15-1.4x** if it cuts 20-30% HLO bytes; higher only with proof | Fuse adjacent pointwise stage prep, tendency augmentation, EOS/calc_p_rho consumers, and small-step finish boundaries so full-grid arrays are not written/read repeatedly. | Branch `worker/gpt/v017-intermediate-fusion-probe`: no physics changes; one staged fused wrapper at a time; acceptance = HLO `bytes accessed` down >=20%, temp down >=10%, Tier-P gate green, no new host transfers. |
| Acoustic carry split / stage-constant closure retry | **0-1.1x**, uncertain | The current acoustic scan carries a broad pytree through 16 substeps/step. Closing over true constants and carrying only mutating leaves can cut scan traffic if the old timing artifact was real. | Branch `worker/gpt/v017-acoustic-carry-split-retry`: redo the reverted carry split with clean cache-hit protocol, HLO byte diff, and 128^2/65k warm timing. Stop if bit/tier deltas exceed gate or bytes do not move. |
| Residual fp64 island elimination after native-fp32 | **1.2-1.8x over mixed_s2**, mostly VRAM | Replace remaining in-loop total/base fp64 materialization with perturbation-authoritative fp32 work arrays and small controlled fp64 coefficient islands. | Branch `worker/opus/v017-native-fp32-hlo`: compile-only first; gate = f64 float tokens <10% of float tokens, converts near named boundaries only, temp bytes <=~4.5 GB at 65k, then native bench JSON. |
| Kernel-launch reduction without changing math | **0-1.05x** based on v015 evidence | v015 command-buffer/capture collapsed launches but was wall-neutral because the step was device-bound. Still useful only if paired with traffic reduction. | Branch-only A/B with current XLA flags on the same 36-step warmed protocol. Acceptance requires wall improvement, not launch count alone. Prior is negative. |
| Data layout/coalescing / transpose cleanup | **0-1.05x** likely | Existing SoA pytree and C-grid layout were not the dominant issue; wrapped transposes were bounded small. | Branch `worker/gpt/v017-layout-audit`: count transpose/copy kernels and HLO `copy`/`transpose`; change one layout hotspot only if a profiler names it. |
| Buffer donation / rematerialization tuning | **0-1.05x** likely | Donation already exists; XLA remat could not get below the intrinsic fp64 arena in v015. | Compile-only XLA flag matrix on HLO temp bytes; reject if field gate or wall regresses. |
| Tensor cores | **~0 for current stencils** | The workload is stencils, scans, reductions, tridiagonal solves, and scalar physics. There is no large dense matmul. TF32/BF16 are not acceptable for the fp64/fp32 physics equivalence gate. | Do not dispatch unless a specific radiation or lookup subkernel is reformulated as a real GEMM with an oracle. Treat as low-priority research, not a speed plan. |

Operational step structure from `operational_mode.py`:

- One public JIT runs a device-resident scan.
- Each step runs physics forcing at step entry, then RK3.
- RK stage cadence for acoustic substeps is **1 + 5 + 10 = 16 acoustic substeps per timestep** when `acoustic_substeps=10`.
- Each RK stage computes advection tendencies, optional flux-form transport, boundary relaxation tendencies, small-step prep, `calc_p_rho`, then an acoustic `lax.scan`.
- Physics slots include microphysics, surface/surface-layer, PBL, optional GWD, cumulus, and conditional radiation refresh.
- v016 HLO has **923 fp64 fusions** for the fp64 one-step compile and **706 fusions** for mixed_s2. v015 nsys clean evidence puts the niter16 no-flags path around **10.5k CUDA kernel instances/step** and **~112-116 ms/step device time**.

This structure is why the most valuable non-megakernel lever is not "one fewer launch"; it is reducing repeated full-grid HBM traffic across the many fusions/scans.

## 4. Multi-GPU / Cluster Scaling

### Current readiness

The code has an opt-in sharding substrate:

- `src/gpuwrf/runtime/sharding.py` defines default-off `ShardingConfig`, x-slab partition/merge, pmap execution, and ppermute halo exchange.
- `contracts/halo.py::apply_halo` is already shaped for optional exchange.
- Fake-device proofs pass for default-off graph invariance, periodic x halos, and representative sharded horizontal operators.
- `proofs/multigpu_dgx/README.md` records a fake operational d02 proof with full physics enabled, radiation held off, and `run_boundary=False`.

But production Canary multi-GPU is not ready:

- Current sharded forecast supports **x-axis only**.
- It rejects `run_boundary=True`; specified/nested boundary decomposition is not implemented.
- It rejects `radiation_static` partitioning.
- Fake CPU meshes do not prove NVLink/NCCL performance or absence of real GPU transfer regressions.
- Full coupled acoustic/RK sharding is only partially de-risked by operator-level proofs.

### Communication scale

Current hot halo field list is:

```text
u, v, w, theta, qv, p, ph, mu
```

For a 560x280x44 1km grid, x-halo width 3, one halo refresh of those fields is approximately:

- fp64: **~4.77 MB per rank per refresh**
- fp32: **~2.39 MB per rank per refresh**

If a full step performs 16-20 effective halo refreshes, that is roughly:

- fp64: **76-95 MB/rank/step**
- fp32: **38-48 MB/rank/step**

On an NVLink/NVL72-class fabric, raw bandwidth is not the first-order limiter for one rack. Latency, number of collective sites, small local domains, and XLA/NCCL overlap are the real risks.

### GB300-NVL72 hardware facts and implication

NVIDIA's current GB300 NVL72 page lists **72 Blackwell Ultra GPUs**, **20 TB GPU memory**, and **up to 576 TB/s GPU memory bandwidth** per rack. That is about **8 TB/s HBM per GPU**. It also lists **130 TB/s NVLink bandwidth** at rack scale and **100 TFLOP/s FP64** at rack scale. Source: NVIDIA GB300 NVL72 product page, https://www.nvidia.com/en-us/data-center/gb300-nvl72/

Implications for this code:

- For fp32/native-fp32, GB300 is bandwidth-rich: a memory-bound stencil/scalar code should benefit strongly per GPU if local domains are large enough.
- For scalar fp64, do not assume the old "datacenter fp64 removes the penalty" story unless the target SKU and compiler path prove it. The official GB300 rack FP64 line is not an H200-style per-GPU scalar-fp64 windfall.
- GB300's rack advantage for this algorithm is therefore mainly **HBM capacity/bandwidth + many GPUs + NVLink**, not tensor cores.

### Scaling envelope

For the single 560x280 Canary-1km domain:

- x-only decomposition is useful to about **8 GPUs**: local x ~= 70 columns, halo overhead from width 3 is about **8.6%** in x.
- At **16 GPUs**, local x ~= 35 and halo overhead rises to about **17%** before latency and under-occupancy.
- At **32 GPUs**, local x ~= 17.5 and halo overhead is about **34%**; efficiency will be poor.
- At **72 GPUs**, x-only decomposition is structurally wrong for this small domain.

With a future 2D decomposition, 72 GPUs could be used, but the 560x280 domain would still have small local tiles (~70x31 for an 8x9 mesh). Halo perimeter overhead is roughly **25-30%** at width 3, and column physics becomes under-occupied. That is not a near-linear 72-GPU strong-scaling target.

Projected strong scaling for one Canary-1km domain, after production sharding is implemented and profiled:

| GPUs | expected efficiency | expected use |
|---:|---:|---|
| 2-4 | **80-95%** | likely good; local domains still large |
| 8 | **65-85%** | best practical single-domain target on x-only or simple 2D |
| 16 | **45-70%** | possible with 2D and overlap; x-only becomes marginal |
| 32 | **25-50%** | useful only with larger domains, batching, or ensembles |
| 72 | **<40% for one 560x280 domain** | use for weak scaling, many domains, nests, or ensemble members |

Weak scaling is much more favorable. If each GPU keeps at least O(100k) columns or a comparable local tile/ensemble bundle, halo overhead falls to a few percent and the expected in-rack efficiency is **75-90%** after real halo overlap and transfer audits pass. Multi-node beyond one NVL72 rack should be treated as **50-80%** until InfiniBand/NCCL traces exist.

## 5. Final Speedup Envelope

Single RTX 5090, current fp64:

- 128^2 production: **119.8 ms/step**, **1.67x vs 24-rank CPU**.
- With fp32-BouLac if compile/stall fixed: **104-108 ms/step**, **1.85-1.93x vs 24-rank CPU**.
- 560x280 Canary-1km: **1.6-2.7x vs 28-rank CPU**, centered near **2x**.

Single RTX 5090, v016 mixed_s2:

- **1.10-1.11x over fp64 cost proxy**.
- **No peak-VRAM relief** at 16k/65k.
- **Does not unlock 147k columns**.

Single RTX 5090, future native-fp32 if numerically validated:

- Defensible production target: **~2-3x vs CPU at Canary-1km**, with a possible upper tail only if residual fp64 islands and mixed-compile pathology disappear cleanly.
- Defensible memory target: **~2x VRAM reduction**, enough to unlock fp64-OOM 1km footprints.
- Not defensible today: **6x single-GPU Canary-1km**. That number requires multiplying current ~2x large-grid fp64 speed by a clean ~3x production fp32 gain. The HLO and mixed_s2 evidence do not support that as an achievable steady-state for the current JAX/XLA algorithm without additional HBM-traffic reduction.

After prioritized levers:

1. Native-fp32 HLO/bench proof and long-horizon equivalence.
2. BouLac O(nz^2)->O(nz).
3. Intermediate fusion targeted by HLO bytes, not launch-count vanity metrics.
4. Production sharding only after specified/nested boundaries and radiation statics are decomposed.

If those land, the project can honestly claim:

> Near the practical optimum for the current JAX/XLA dynamic architecture on a single RTX 5090, excluding the principal-rejected fuse-everything megakernel; remaining large speedups require either a different kernel granularity strategy or multi-GPU weak scaling.

It cannot honestly claim:

> A proven ~6x Canary-1km single-GPU speedup on this RTX 5090.

Co-Authored-By: GPT-5.5 <noreply@openai.com>
