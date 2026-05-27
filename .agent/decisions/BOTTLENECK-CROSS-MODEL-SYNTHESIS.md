# Bottleneck Cross-Model Synthesis — Codex vs Gemini agy

**Date**: 2026-05-27
**Inputs**:
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/` (8 deliverables + report)
- `.agent/sprints/2026-05-27-bottleneck-analysis-agy/analysis.md` + summary
**Author**: Claude Opus 4.7 (manager)

## Cross-Model Roadmap Comparison

| Rank | Codex roadmap | Agy roadmap | Synthesis |
|---|---|---|---|
| 1 | Elementwise launch/D2D fusion (~35-70 s/24h, medium risk, medium-high effort) | Cold-JIT cache + AOT (~104 s saving, low risk, 2 sprint-days) | **DISAGREE.** Codex prioritises the warm-path kernel-launch reduction; agy prioritises the cold-start mitigation. They are not in conflict — these are different speedup axes. |
| 2 | Cold-JIT cache + precompile (100 s+ saving, low risk) | XLA tridiagonal_solve primitive replacing Thomas scan (~35 s/24h, medium risk) | **DISAGREE on order; agree on inclusion.** Codex parks the tridiagonal switch as rank-4 (high risk on FP64 acoustic stability); agy ranks it #2. |
| 3 | XLA memory-profile + aliasing (small wall-clock win, medium risk) | Unified surface/PBL adapter (~22 s/24h, low risk) | **DIFFERENT scope.** Codex is memory-headroom-driven; agy is physics-adapter-fusion-driven. |
| 4 | Vertical solver specialization (0.1-0.4 s warm, high risk) | Precision downcast 10 surface/boundary fields (~15 s/24h, low risk) | **DISAGREEMENT.** Codex defers precision; agy ranks it #4 low risk. |
| 5 | Physics column-layout sprint (small warm win, medium-high risk) | Carry shrink + allocator tuning (memory headroom, low risk) | Different sprint, both ranked low-priority. |
| deferred | Precision downcast, multi-GPU | (multi-GPU treated in PART 6, not ranked but planned) | Codex more cautious; agy gives concrete multi-GPU path forward (`shard_map`). |

## Convergent High-Confidence Recommendations

Both models agree on these (different rank but both endorse):

1. **Persistent JAX compile cache + AOT compilation**: ~100 s reduction in cold-start. Both rank it top-3. Lowest risk, immediate win. *Both name `JAX_COMPILATION_CACHE_DIR` and `jax.export` as the mechanism.*

2. **Reduce kernel-launch overhead in the warm timestep loop**: many tiny `loop_*_fusion` kernels add up. Both flag this. The implementation differs (codex says "collapse scan-carry housekeeping + finite guards"; agy says "dycore coupling/decoupling stencil fusion + surface-PBL coupling buffer fusion"), but the diagnosis is the same: too many small kernels.

3. **Multi-GPU work should not start yet**. Both agree the single-GPU warm path needs to be cleaner first. Agy gives a concrete future path (`jax.experimental.shard_map` with halo width 2); codex marks it deferred.

4. **cuSPARSE `pcrGtsvBatchSharedMemKernelLoop` (MYNN tridiagonal) is dominant but already optimal**. Both classify it as fundamental — accept, don't try to replace.

## Divergent Recommendations (Where Models Disagree)

1. **Precision downcast of 10 surface/boundary FP64 → FP32 fields.**
   - Agy: rank #4, low risk, ~15 s/24h saving.
   - Codex: deferred, says "precision downcast: do not prioritize until launch/memory profiling lands. The only large FP64 candidate is `w` (~16.2 MiB saving at 1 km if FP32), and acoustic stability risk is high."
   - **Manager judgment**: agy's specific 10 fields (surface fluxes, accumulation parameters, boundary fields) are physically slow-timescale and bandwidth-bound, not numerically sensitive. Agy's recommendation is defensible. Codex's "wait until profiling" caution applies more to dycore-internal FP64 fields like `w`. Both can be right: do the 10 named non-dycore fields (agy's list), but keep dycore FP64 (codex's caution). Net: **adopt agy's precision sprint, scoped exactly to the 10 surface/boundary fields**.

2. **XLA `tridiagonal_solve` primitive replacing JAX-native Thomas scan.**
   - Agy: rank #2, medium risk, ~35 s/24h saving.
   - Codex: rank #4, high risk.
   - **Manager judgment**: this is a real disagreement about acoustic-solver numerics. The JAX-native Thomas scan currently has B6 0.0 bitwise parity with WRF Fortran. Replacing it with `tridiagonal_solve` could change that. Codex's "high risk" is the correct read for the M6 invariant. **Defer this swap until after the testing-plan execution provides idealized-case + conservation evidence**; if those tests pass with current Thomas scan, the regression risk of switching is unacceptable. If those tests reveal Thomas-scan-specific issues, then the switch becomes justified.

3. **Top-of-roadmap warm-path optimization.**
   - Codex: rank #1, fuse elementwise scan-carry + guards.
   - Agy: rank #3, unified physics adapter.
   - **Manager judgment**: both are real wins. Codex's scope is broader (RK/acoustic scan-carry housekeeping spans more files); agy's scope is tighter (surface-PBL coupling specifically). Doing them sequentially (agy first as a smaller proof of concept, then codex's broader fusion) gives a graded risk profile.

## Recommended Optimization Sprint Sequence

Synthesizing both analyses, this is the manager's recommended forward roadmap:

| # | Sprint | Source | Expected saving | Risk | Effort | When |
|---|---|---|---|---|---|---|
| **O1** | Persistent compile cache + AOT | both (#2/#1) | ~100 s cold start | LOW | 2 days | now if user authorises |
| **O2** | Unified surface_adapter + mynn_adapter | agy (#3) | ~22 s/24h | LOW | 3 days | after O1 |
| **O3** | Precision downcast 10 surface/boundary fields (no dycore!) | agy (#4) | ~15 s/24h + memory | LOW-MED | 3 days | after testing-plan execution validates current state |
| **O4** | Carry shrink + JAX_MEM_FRACTION tuning | agy (#5) | ~2.5 GB headroom | LOW | 3 days | after O3 |
| **O5** | XLA memory profile + aliasing pass | codex (#3) | small wall-clock + headroom | MED | 3 days | after O4 |
| **O6** | Broader elementwise scan-carry fusion | codex (#1) | ~35-70 s/24h | MED | 5-7 days | after O5; needs invariant gates |
| **(deferred)** | XLA tridiagonal_solve swap | agy (#2) / codex (#4) | ~35 s/24h | HIGH | medium | only if testing-plan execution reveals Thomas-scan issues |
| **(deferred)** | Multi-GPU via shard_map | agy (#6) / codex deferred | enables larger domains | HIGH | 2+ weeks | post first paper |

Total achievable saving if O1-O6 land: ~190-230 s on the cold-included 24h, plus ~50 s on warm.

## Items NOT in either roadmap that the manager flags

- **Halo placeholder cleanup**: even before multi-GPU, the halo interface should be tested with a dummy single-rank "halo of zeros" to confirm the interface doesn't break at boundary cells. Low risk, validates the architectural claim.
- **AOT-compiled wheel for distribution**: if O1 lands, the next natural step is an installable wheel with the JAX program already compiled for sm_120 + a target shape. Bigger story for "open-source release plan" section.
- **Optimization-evidence appendix in the paper**: the current paper makes a 22.26x claim. If O1-O3 land before submission, the paper can claim the post-optimization number, with the optimization steps documented. This is a stretch goal.

## Honest note

Neither analysis is wrong; they emphasise different optimization axes. The cross-model exercise is most valuable for naming the **agreements** (compile-cache, kernel-launch reduction, accept cuSPARSE GTSV, defer multi-GPU) and the **disagreements** (precision downcast scope, tridiagonal swap risk). The manager's synthesis above resolves the disagreements with manager judgement and the M6/M7 invariant gates as the deciding criterion.

## Action items this synthesis generates

- This note is **input only** to a future optimization-dispatch sprint. The user said "no rewrite for now, just the full analysis and plan." This synthesis IS the plan.
- The user should pick which (if any) of O1-O6 to dispatch as actual optimization sprints. Manager recommendation: O1 (compile cache) first because it's a clear win with low risk and immediately improves the user's daily-driver experience.
- The optimization-evidence appendix is a paper-rewrite consideration, not a separate sprint.
