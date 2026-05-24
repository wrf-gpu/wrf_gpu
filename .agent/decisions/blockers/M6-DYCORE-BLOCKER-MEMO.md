# M6-DYCORE-BLOCKER-MEMO

**Date**: 2026-05-24
**Author**: Manager (Claude Opus 4.7)
**Status**: BLOCKER ACTIVE — user decision required
**Trigger**: HYBRID exit-rule fired after bug-hunt sprint (`e036ef8`) returned `NO-BUG-LOCALIZED`

## What's blocked

M6 close gate: Tier-3 short-run convergence + initial Tier-4 RMSE vs Gen2 backfill on `U10/V10/T2` at 24h/72h.

## What we now know (concrete evidence)

The current ADR-023 unified path (post-S3-narrow stabilizer cleanup, post-S2.2 hang fix, post-pressure-wiring fix) **cannot run a coupled 1h d02 forecast without catastrophic divergence**, even with the sanitizer + `_mu_continuity_increment` cap active:

| Field | 1h RMSE on real Gen2 d02 | Gen2 24h noise floor | Ratio |
|---|---|---|---|
| T2 | 136.885 K | 0.628 K | 218× |
| U10 | 106.419 m/s | 1.456 m/s | 73× |
| V10 | 102.232 m/s | 1.591 m/s | 64× |

Plus 17.2 billion sanitized nonfinites per 1h run, theta_max = 550 K (sanitizer cap), w_max = ±50 m/s (cap), u/v = ±150 m/s (caps).

**Bug-hunt ruled out 7 single-suspect operator hypotheses.** None of: MPAS recurrence sign flip, `_mu_continuity_increment` removal, `_mpas_w_metric_faces` reference-metric, n_acoustic sweep, physics disable, boundary disable, branch verification moved the first-nonfinite step beyond step 2. Center d02 column coefficients are all finite + positive + weakly diagonally dominant.

**ADR-021 carry-expansion architecture also dead** (proven by clamp-strip test 2026-05-23: theta blowup +/-21,800 K at step 1 without unphysical clamps).

## What this means architecturally

Neither of the two architectures the project has prototyped can deliver an honest 1h coupled forecast on real Gen2 d02:
- ADR-023 (conservative recurrence, small 6-leaf carry): hemorrhages nonfinites at step 2
- ADR-021 (full WRF carry expansion, line-cited port): only "works" with target-shaped clamps

Both share the same underlying deficiency: missing or improperly-coupled **WRF small-step time-averaged scratch state** (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`). ADR-023 doesn't have them at all; ADR-021 has the leaves but their usage isn't correct without further surgery.

## Bounded options for user decision

**Option A: WRF Scratch Hybrid** (ADR-023 + carry expansion for the specific missing scratch only). Add `t_2ave`, `ww`, `muave` to `AcousticScanCarry` (other expansion items deferred). Implement WRF-cited usage. Time estimate: 3-6 sprints, 1-2 weeks calendar. Risk: still may not be enough if the issue is deeper than time-averaging.

**Option B: Full WRF Small-Step Port with Savepoint Harness** (ADR-021 done right). Resurrect the ADR-021 carry-expanded operator, strip the clamps, but build a WRF Fortran-harness savepoint extractor first so every recurrence step has a numerical reference to debug against. Time estimate: 5-9 sprints, 2-4 weeks. Risk: large surface, but most likely to actually work.

**Option C: Third-Path Substrate Scout** (port a working JAX dycore base). Use Dinosaur's IMEX time-integration scaffolding or ICON4Py's vertically-implicit pattern as the substrate, drop the conservative-column-solver attempt entirely. Time estimate: scout sprint first (3-6 hours, just research + recommendation), then 4-8 sprints if scout recommends a workable port. Risk: medium — Dinosaur/ICON4Py are JAX/Python-native but their integration with WRF physics is unproven.

**Option D: Defer M6 close + redirect to M7 work that doesn't need the dycore.** M7-S0a operational/data prologue is done. There are M7 sub-areas (post-processing, packaging, observation verification harness) that can advance without a working dycore. M6 stays open until one of A/B/C produces working baseline. Time estimate: M7 partial work in parallel, M6 close timing unbounded. Risk: shipping with partial-M6 is constitutionally fragile.

## Manager recommendation

**Dispatch Option C scout first (it's small, ~3-6 hours, no architecture commitment).** While the scout works, write a deeper technical comparison of A vs B against the scout's third-path recommendation. After the scout returns, user picks the path with all three options scored on:
- Time-to-1h-coupled
- Confidence that this path actually closes M6
- Reuse of existing investments (M5 physics, S1 sidecars, source-mining table, etc.)
- Carry constraints (which architectural choices survive)

The Option C scout is the cheapest information-gathering step that doesn't pre-commit. Per user standing order #5 ("send out an agent to think about this more top level... try a re-write with a different method that can be tested after one large sprint or so"), dispatching the scout is within manager authority.

## What is being dispatched immediately

Per user's standing order #5 (anti-stuck rule), dispatching ONE third-path scout sprint NOW. This is research, not implementation — manager-authorized.

NOT being dispatched: implementation work on Options A, B, or C. Those require user direction.

## What the user must decide

Once the scout returns (~3-6 hours):
1. Which option (A, B, C, or D) to pursue
2. Whether to keep ADR-023 PROPOSED or formally supersede it
3. Whether ADR-024 (gate policy) can be promoted to ACCEPTED independently
4. Project communication: should the public README/PROJECT_PLAN reflect M6 BLOCKED status, or just keep "in progress"?

## What is NOT broken

- M0-M5 are durable (governance, fixtures, backend, state layout, reduced dycore, physics suite)
- M5 physics (Thompson, MYNN, RRTMG SW+LW with parity) survives any pivot
- M7-S0a operational/data prologue done
- S1 diagnostic infrastructure (12 sidecars, source-mining table) survives any pivot
- ADR-024 warm-bubble gate policy is correct independent of the operator architecture
- Gen2 noise-floor characterization survives any pivot
- The d02 replay infrastructure (post-S2.2 hang fix) survives any pivot
- Stabilizer-provenance scanner (S3-narrow) survives any pivot

## Conclusion

This is not a project failure. It's a correctly-surfaced architectural decision point. The manager spent ~24 hours executing the critic-ratified HYBRID plan exactly as designed, hit the exit rule at exactly the predicted threshold, and now hands the user a bounded set of options backed by concrete code-running evidence.

The biggest single learning from this cycle: the architectural pivot was the right scope, but neither of the two prototyped architectures (small-carry conservative OR full-carry WRF-shape) handles the real coupled forecast. A third path is genuinely needed.

— Manager (Claude Opus 4.7), 2026-05-24
