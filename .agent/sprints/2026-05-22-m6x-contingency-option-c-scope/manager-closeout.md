# M6.x Contingency Option (c) Designer — Manager Closeout

**Sprint**: M6.x Contingency Option (c) Architecture Scope
**Status**: **CLOSED — Opus design delivered; c1 contract ready-to-dispatch; c2/c3 contracts deferred (design doc adequate to draft on-demand)**
**Date**: 2026-05-21 ~23:25
**Worker**: Claude Opus 4.7 xhigh
**Wall**: ~10 minutes (much faster than budget 1-3h)

## Headline

Designer delivered 25KB design analysis with three options ranked, decision matrix, invocation rule, and ready-to-dispatch c1 contract (8.8KB). c2 and c3 contracts skipped; ADR-017 skipped. **The design doc is the load-bearing insurance artifact** — c2/c3 contracts and ADR can be drafted in ≤30min if ever needed.

## Decision matrix (Opus §5)

| Option | Wall-to-PASS | Risk | Operational viability | Composite |
|---|---|---|---|---|
| **(c1) Klemp-Skamarock clean-room** | 5-9 days | LOW | HIGH | **#1 (first invocation)** |
| **(c2) Semi-implicit** | 10-20 days | MEDIUM-HIGH | MEDIUM | #2 |
| **(c3) ML-emulator hybrid (c3.C residual)** | 4-8 weeks + validation tail | HIGH | UNKNOWN | #3 (last-resort) |

## Manager binding decision

**Invocation rule** (per design §6, adopted):

If M6.x fails any of:
- Tier-2 lifted-cap invariants FAIL
- sanitize firing rate ≥5%
- 24h forecast NaN-explodes
- speedup <4×

Then dispatch **(c1) Klemp-Skamarock clean-room** as first contingency. Wall-to-decision: <10 min using the pre-drafted contract at `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/c1-klemp-skamarock-contract.md`.

If (c1) also fails within 9 wall-days → escalate to user; either dispatch (c2) or pivot strategy.

(c2) and (c3) contracts to be drafted on-demand at invocation time. Design doc has the AC sketches; codex worker can flesh them out in <1h.

## Notable design surprises

1. **M6.x's 16-32h estimate is optimistic for WRF-canonical fidelity.** Designer judges (c1) clean-room at 5-9 days is realistic faster path than even M6.x's optimistic estimate, because (c1) drops hybrid-coord/off-centering/msf/sumflux baggage.
2. **No published precedent for fully semi-implicit GPU LAM** — c2 is research-grade territory; designer treats this as a yellow flag.
3. **c3 ML-emulator: 1-month Gen2 corpus is per-grid-point comparable to GraphCast's ERA5 training but globally tiny** (1 region vs world). Severe out-of-distribution risk. Only c3.C residual variant is defensible; c3.A/c3.B too data-hungry for our corpus.
4. **(c1) preserves M4 invariants trivially** (SoA pytree, fp64, zero post-init transfer, debug-stripped HLO). (c2) trivially preserves. (c3) requires NN weights as frozen pytree leaf at init.

## Honest limitations (designer §8, valid)

- Tridiag-solve XLA overhead not measured at actual (160×67, nz=45) shape. c1 wall could regress to 7-12 days if XLA hidden cost.
- Gen2 corpus size assumption (~1mo, 10min cadence) needs verification before c3 dispatch.
- (c2) wall estimate speculative — 10-20 day band could be 20-40 days if lateral-BC multigrid pathology.

## Files in repo (after merge to main)

- `/tmp/wrf_gpu2_main_cp/.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/`:
  - `design.md` (25KB; the load-bearing artifact)
  - `c1-klemp-skamarock-contract.md` (8.8KB; ready-to-dispatch)
  - `sprint-contract.md` (3.7KB; this sprint's own contract)
  - `role-prompts/worker.md`
  - `manager-closeout.md` (this file)

## What's next

- ✓ Closeout this sprint (no reviewer required; design doc speaks for itself; manager + Gemini second opinion only invoked if (c1) gets dispatched and goes badly)
- ✓ Continue M6.x worker watch — primary path still preferred
- ✓ Maintain M6.5-D1 worker watch — M7 prereq
- ⏳ If M6.x fails → invoke c1 contract; dispatch in <10min

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 23:25
