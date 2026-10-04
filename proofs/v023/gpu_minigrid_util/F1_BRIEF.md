# F1 SPRINT BRIEF — Batched-Ensemble Mini-Grid GPU Saturation (CPU-side impl + CPU bit-gate)

**Frontrunner:** GPT-5.5 (xhigh). **Manager:** 0:1 (tmux pane `0:1`). **Worktree:** `<USER_HOME>/src/wrf_gpu2_wt/v023-f1` (branch `worker/gpt/v023-f1-batched-vmap`, off v0.22.2). **Type:** core-orchestration → frontrunner + critic (Opus critics after you report). **NO GPU** (held by another lane) — implement + run the bit-gate on the JAX **CPU** backend only.

## Read first (the design is DECIDED — implement it, don't re-litigate)
- `proofs/v023/gpu_minigrid_util/SYNTHESIS_AND_RECOMMENDATION.md` — the manager synthesis (WINNER + design)
- `proofs/v023/gpu_minigrid_util/ANALYSIS_opus.md` + `ANALYSIS_gpt.md` — both independent analyses (impl sketches, risks, v0.19 reconciliation)
- `proofs/v023/V0230_ROADMAP.md` — F1 + F1b items + the G1/G2/G3 / H0–H3 endpoints

## Objective (THIS sprint = the CPU-side, GPU-independent part of F1)
Implement **homogeneous independent-case batching via an outer `jax.vmap`**: B independent same-geometry cases through ONE compiled program, batch axis added by `vmap` **AROUND** the existing unbatched orchestration — **fp64 dycore stencils byte-identical, UNTOUCHED.** Deliver the impl + a **PASSING CPU B=2 bit-identity gate** (the G1/H0 correctness gate — runs on the JAX CPU backend, no GPU). GPU gates (VRAM/max-B, throughput) come later; this sprint proves batching is CORRECT + minimal-invasion.

## The design (from the synthesis — do exactly this)
1. **Outer `vmap`, NOT rank+1 in the dycore.** Wrap the existing single-case advance/boundary functions in `jax.vmap` so each mapped call sees NORMAL unbatched ranks. `build_child_boundary_package` / `interp._gather` assume rank-2/3 → do NOT edit them; `vmap` them.
2. **Batch WITHIN a domain, NEVER across domains** d01/d02 (crossing = the v0.19 ragged-shape trap = 2.5× slower). The B axis is over independent CASES, applied to each domain's advance identically.
3. **Batched loader** (`integration/nested_pipeline.py`): load B same-geometry cases; ASSERT identical domain names/dims/nesting-ratios/physics-suite/dt/cadence/static-treedefs; stack per-domain `OperationalCarry` via `tree.map(lambda *xs: jnp.stack(xs,0), ...)`; keep truly-identical static/namelist/geometry UNbatched.
4. **Batched root cascade** for the 2-nest `d01→d02` (the generic fused path skips root parents — add the specific batched root advance: d01 parent step → build d01→d02 boundary packages → d02 child subcycle, all inside the batched wrapper).
5. **De-batched output**: at history cadence, split the batched carry into per-case views, feed the EXISTING writer per case (preserve the NetCDF surface → keeps validation simple).
6. **Fixed-B AOT keys**: B enters the compile key (cheap-key already hashes aval shapes → distinct blob per B). Support a small fixed set (e.g. B∈{1,2,4}); do NOT compile arbitrary B.

## Hard constraints
- **NO dycore kernel edits.** No masking/clamp/`nan_to_num`. Bit-identical per case.
- **NO GPU** (held by another lane; do NOT take `/tmp/wrf_gpu2_gpu.lock`). Run on the JAX CPU backend (`JAX_PLATFORMS=cpu`). The B=2 contamination check is backend-agnostic — a clean `vmap` is byte-identical on CPU.
- Same-geometry, same-W homogeneous batch ONLY (never ragged).
- **Opt-in, off-by-default**: gate the batch path behind an env flag (e.g. `GPUWRF_BATCH_ENSEMBLE=N`) so the DEFAULT single-case path stays byte-identical + inert. (This protects the whole v0.22.2 test suite.)
- Do NOT regress the existing CPU test suite (run the relevant subset).

## THE deliverable gate (must PASS, CPU)
**CPU B=2 bit-identity:** use the smallest available same-geometry 2-domain fixture (or two copies of one case with perturbed ICs so the lanes genuinely differ). Run **B=2 batched** (one program) vs the SAME two cases run **standalone** (single-case path), on the JAX CPU backend, over the smallest meaningful segment (≥ a few steps; ideally ≥1 output interval). **PASS = each batched lane byte-identical (`max_abs_diff = 0.0`) to its standalone reference, ALL fields.** Proves zero cross-lane contamination (no stray batch-axis reduction/broadcast, no `*_save` scratch aliasing). Put the per-field numbers in the report. If you cannot hit exactly 0.0, STOP and report the offending op via the back-channel — a non-zero delta is a real bug, not a tolerance to widen.

## Deliverable
- Implementation on the branch (committed, clear messages).
- The PASSING CPU B=2 gate: a runnable test/script (`tests/` or `proofs/v023/batched_minigrid/`) + its output.
- `proofs/v023/batched_minigrid/F1_IMPL_REPORT.md`: objective · files changed · the vmap design as-built · the CPU B=2 result (`max_abs_diff` per field) · honest kernel-invasion assessment · what remains for the GPU gates (G3 throughput/VRAM) · unresolved risks.
- `touch proofs/v023/batched_minigrid/F1_DONE` as the VERY LAST step.

## Back-channel
Blocked or disagree? Send to manager pane `0:1` via `scripts/tmux_submit.sh 0:1 '<question>'` AND append to `proofs/v023/batched_minigrid/F1_QUESTION.md`, then keep working the unblocked parts. Do NOT stop for a decision you can reasonably default. Begin now.
