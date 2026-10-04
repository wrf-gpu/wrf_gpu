# BRIEF — Max GPU utilization for a MINI-GRID forecast on one RTX 5090, MINIMAL kernel invasion

**Manager:** 0:1 (gpuwrf). **Dispatched:** 2026-06-30. **Type:** architecture/perf ANALYSIS ONLY — **NO GPU runs** (the GPU is held by the downscale lane / Phase-2; do not take the GPU lock, do not run forecasts). Ground everything in the *real code* + first-principles + the documented perf history.

## Context & goal (why this matters)
- The downscaler (a separate ML model) generalizes poorly across islands → it needs **years of local forecast training data PER island**.
- Immediate plan: generate years of **Tenerife-only** weather as a **2-nest 3 km→1 km** gpuwrf forecast (Canary single-island geometry: d01 ~3 km, d02 ~1 km, each only a few-hundred² cells), over **many independent init-days**.
- PROBLEM: a mini-grid badly **underutilizes the 5090**. Just measured on the 9-nest Canary: GPU util **~1 % (de-fuse) / ~50–96 % (fused)** — even fused, tiny grids are **launch/occupancy-bound** and leave the 5090 SMs idle. A 2-nest 3 km/1 km Tenerife is similarly small.
- GOAL: **maximize GPU throughput (forecast-days per GPU-hour) for this mini-grid with MINIMAL invasion into the dycore/kernel code**, to approach the 5090's *large-grid best-case* throughput → more training data per GPU-hour → a faster feasibility answer for the downscaler.

## Hard constraints
- The **fp64 dycore is near-optimal at fp64 and bit-identical** (validated). **DO NOT rewrite the dycore kernels.** Strongly prefer orchestration / batch-axis / config changes over kernel surgery. **Rank every approach by kernel-invasion level** (none / minimal / moderate / deep).
- The cases are **INDEPENDENT**: same island/geometry, different init-days → **embarrassingly parallel, NO halo/coupling between cases**. This is the key enabler — exploit it.
- **Bit-identity:** each case's output must stay bit-identical (or provably equivalent) to running it standalone — **no cross-case contamination** at seams/borders/reductions.
- **VRAM = 32 GB** (5090). Estimate VRAM cost + how many cases fit.
- Reuse the existing nested machinery (fused cascade default, AOT cheap-key warm-start) where possible.

## Candidate approaches to evaluate (AND propose/score others)
1. **Batched ensemble via `vmap` over independent init-times** (primary candidate): add a leading batch axis B so the state pytree becomes `(B, …)`; advance B Tenerife cases (different ICs/days) in lockstep through ONE compiled program. Fills SMs via B× work per kernel + amortizes per-kernel launch overhead across B.
2. **Spatial stitching**: pack B Tenerife grids into one big grid with buffer borders, run as one domain, discard the borders. (Flagged as complex/wasteful — quantify the waste + seam-contamination risk.)
3. **Concurrent CUDA streams / multiple XLA executables overlapped**, or **multi-process sharing the GPU via CUDA MPS** (N gpuwrf processes).
4. **Others** you identify (persistent kernels, async host-pipelining to hide the ~host-bound nest orchestration, larger-d01-to-amortize, etc.).

## CRITICAL prior-art to RECONCILE (do not naively repeat or dismiss)
- A prior attempt to **`vmap`-batch the heterogeneous LEAVES of ONE nest** (different domain shapes d03–d09) was a **DEAD END: ~2.5× SLOWER** (v0.19).
- BUT that batched HETEROGENEOUS shapes *within one nest*. Batching B **HOMOGENEOUS, INDEPENDENT** cases (identical geometry, different ICs) is a **different problem** — a clean leading batch axis with no ragged shapes and no inter-case coupling. **Determine, with reasoning, whether the prior dead-end applies to the homogeneous-independent-case batching or not.** Find the v0.19 evidence if you can (proofs/, memory) and explain the mechanism of the 2.5× regression so we know whether it recurs here.
- Also reconcile against the established truth: tiny single-domain 129² is ~2.3× slower than CPU (launch/occupancy-bound); fused 9-nest ≈ ~700 s/fc-h, ~1.3× vs CPU; the GPU win "grows with scale". Batching is essentially a way to *manufacture scale* from independent small cases — assess how far that closes the gap to the roofline.

## Code to read (ground the feasibility — be concrete)
- `src/gpuwrf/runtime/domain_tree.py` (the cascade/advance orchestration, `_advance_chunk`, fused vs de-fuse, `block_between`)
- `src/gpuwrf/integration/nested_pipeline.py` (the production nested driver, output boundary)
- the dycore advance + the **state pytree** (shape/structure of the prognostic state — is a leading batch axis a clean `vmap`, or does some op assume rank/shape that breaks under batching?)
- `src/gpuwrf/runtime/operational_mode.py`, `aot_precompile.py` / `aot_cheap_key.py` (AOT/cache implications of a new batch axis)
- the v0.17 host-orchestration evidence: `proofs/v017/gpu_utilization_investigation.md`, `proofs/v017/hostgap_gpu_util_summary.json`

## Required deliverable (write to your assigned path, structured markdown)
1. **Ranked table**: approach | kernel-invasion (none/minimal/moderate/deep) | expected GPU-util gain | VRAM cost & max B @ 32 GB | throughput estimate (cases/GPU-hour or % of large-grid roofline) | bit-identity risk | other risks.
2. **WINNER** + why it beats the rest (esp. on the util-gain ÷ kernel-invasion tradeoff).
3. **Implementation sketch** for the winner: exact files/functions to change, the *minimal-invasion* diff shape (e.g. where the leading batch axis enters, what must be `vmap`'d vs left alone, host-orchestration/output changes), and AOT/cache implications (does B become part of the compile key? recompile cost?).
4. **Complexity estimate**: sprint size S/M/L + the top 3 risks.
5. **"How close to large-grid 5090 best-case"**: a quantitative estimate of the throughput gain and how near it gets to the 5090 roofline/large-grid best-case, with explicit assumptions (occupancy, launch-amortization, memory-bandwidth, the residual host-bound fraction).
6. **Honest failure modes**: where it could disappoint (VRAM ceiling on B, residual host-bound serial orchestration that batching can't hide, compile-time/RAM blowup with the batch axis, bit-identity traps).

Be concrete, read the actual code, and be honest about effort and ceilings. If you hit a blocking question or disagree with this brief, send it to the manager pane `0:1` (delayed-repeated-Enter) and keep working the unblocked parts.
