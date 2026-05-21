# Sprint Contract — M6.x Contingency Option (c) Architecture Scope

**Sprint ID**: `2026-05-22-m6x-contingency-option-c-scope`
**Created**: 2026-05-22 ~23:10 (parallel insurance dispatch)
**Status**: ACTIVE (read-only analysis)
**Trigger**: M6-S5 Opus reviewer §4 flagged Reading B risk (canonical dycore in JAX may be 3-6mo, not 2-4wk). If M6.x fails, manager needs option (c) re-architecture pre-scoped so dispatch can happen in minutes, not hours.
**Parallel with**: M6.x dycore + M6.5-D1 backfill + F-5 baseline.

## Objective

Pre-scope option (c) architectural alternatives for the GPU dycore so that, if M6.x fails Tier-2 + sanitize <5% + 24h forecast finite, the manager can dispatch a ready-built option (c) sprint without 2-4h of re-scoping.

Three candidate architectures to scope:
- **(c1) Klemp-Skamarock vertical-implicit acoustic damping** (WRF NMM-style, splits acoustic into horizontal explicit + vertical implicit; well-conditioned at small dt)
- **(c2) Semi-implicit integration** (treats fast waves implicitly; larger coupled dt; tridiagonal solver each step)
- **(c3) ML-emulator hybrid** (NN learns coupled physics+dycore residual on coupled coarse-grain training set; expensive to train but trivial to run)

## Acceptance

- **AC1 (c1) Klemp-Skamarock scope**: how it differs from M4 reduced + WRF-canonical; what files would change; WRF NMM source pointers if applicable; risk profile; estimated wall budget.
- **AC2 (c2) Semi-implicit scope**: tridiagonal solver requirements in JAX (vmap over columns); precedent in HOMMEXX / SCREAM / Pace if any; risk profile; wall budget.
- **AC3 (c3) ML-emulator scope**: training data source (Gen2 corpus?); architecture (FNO? Transformer? UNet?); precedent (Pangu, GraphCast, NeuralGCM); risk profile; wall budget. Note: this would be a longer-term option, not a 1-2-week sprint.
- **AC4 Decision matrix**: rank c1/c2/c3 on (1) wall-to-PASS, (2) architectural risk, (3) operational viability for Canary 3km daily, (4) end-state code complexity.
- **AC5 Pre-built sprint contract drafts**: 1 sprint contract per option (3 total), ready-to-dispatch templates with file ownership boundaries.
- **AC6 NEW ADR**: `.agent/decisions/ADR-017-m6x-contingency-options-scope.md` — captures the design decisions, the ranking, when to invoke each option.

## Files Worker May Modify

- `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/{design.md, c1-klemp-skamarock-contract.md, c2-semi-implicit-contract.md, c3-ml-emulator-contract.md}` (NEW)
- `.agent/decisions/ADR-017-m6x-contingency-options-scope.md` (NEW)

## Files Worker Must NOT Modify

- ANY src/, tests/, scripts/ — this is pure READ-ONLY design analysis
- ANY existing ADR
- ANY existing sprint contract

## Dispatch

- Worker: **Claude Opus 4.7 xhigh** (architectural design quality; NOT codex)
- Reviewer: not strictly required (this is contingency planning; manager reviews; OR opus self-reviews; OR Gemini second opinion if M6.x fails and we activate)
- Wall-time: **1-3h** (read-only design + 3 contract drafts + 1 ADR)
- Worktree: `/tmp/wrf_gpu2_contingency`
- Branch: `worker/opus/m6x-contingency-option-c-scope`

## HARD RULES

1. READ-ONLY on `src/`, `tests/`, `scripts/`, existing ADRs
2. NO code generation — this is design only
3. Cite published precedent where possible (WRF NMM, HOMMEXX, SCREAM, Pace, NeuralGCM, GraphCast, Pangu)
4. Honest about risk — do not understate option-c complexity
5. `/exit` slash-command

## End-goal context

INSURANCE policy against M6.x failure. If M6.x lands GREEN, this sprint's output becomes archived design rationale. If M6.x lands RED, this saves 2-4h of contingency scoping when project needs to pivot fast.

The fact that this work exists is the value, even if it's never invoked.
