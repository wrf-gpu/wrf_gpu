# M6-S7 Manager Closeout — Tier-4 Probtest Scaffold

**Sprint**: M6-S7 Tier-4 probtest prototype
**Status**: **CLOSED — Opus ACCEPT-AS-SCAFFOLD-DEFER-TO-M7**
**Date**: 2026-05-21 ~18:00

## What landed

Worker (codex, ~30min) delivered methodology scaffold + 4 BLOCKED artifacts. Refused silent wrf_l2 substitution / grid-mixing (would violate critic amendment §8 "no tolerance after seeing candidate").

- `src/gpuwrf/validation/tier4_probtest.py` (probtest variance/stratum math, ddof=1 unbiased, k=1.96 ≈ 2σ)
- `proof_schemas.Tier4ProbtestTolerances` registered
- `scripts/m6_run_tier4.py + m6_gate_tier4.py` (gate uses `--allow-heldout-fail`)
- `tests/test_m6_tier4_probtest.py` (8 tests pass)
- 4 artifacts emit `status: BLOCKED` honestly:
  - `ensemble_member_manifest.json` (3/10 pinned-grid d02 available)
  - `probtest_tolerances.json` (sample_size_required: 10)
  - `heldout_candidate_validation.json` (20260519_18z lacks wrfout_d02)
  - `cost_model.json` (29s/member from M6-S2 budget, recommended M7 = 100 members)
- `tolerance_freeze_report.md` (frozen choices BEFORE held-out evaluation, ## Blockers section)

## Opus verdict (binding hybrid decision)

**ACCEPT-AS-SCAFFOLD-DEFER-TO-M7**. Adversarial probes confirmed:
- No other historic period has more members (archive policy strips wrfout_d02_*)
- Math correct (variance/stratum, ddof=1, k=1.96)
- Cost model defensible (29s × 100 ≈ 49min single-GPU; σ-estimator drops from ~24% at n=10 to ~7% at n=100)

## 4 manager follow-ups

- **F-S7-1**: Schedule M6.5-D1 data backfill (CPU WRF re-runs for missing pinned-grid days) only if M7 dispatch demands. Out of M6 close scope.
- **F-S7-2**: M6-S8 binding gate IGNORES Tier-4 BLOCKED status (treat as informational sigma diagnostic; CPU-vs-observation is binding per validation philosophy).
- **F-S7-3**: M7-S2 dispatch must consume `cost_model.json.recommended_m7_ensemble_size = 100` (don't invent new number).
- **F-S7-4**: After M6-S5 verdict, swap provisional 29s/member with S5 number via `--spacetime-budget` argument.

## M6 dispatch impact

- M6-S8 NOT BLOCKED — proceed when M6-S5 and M6-S6 close
- M7 prerequisite: M6.5-D1 data backfill (out of M6 critical path)
- M7-S2 ensemble dispatch consumes cost_model.json directly

## M6 progress now (after this close)

- ✓ S1, S2a, S2, S3, S4, S7 closed (6/8 implementation sprints)
- 🟡 S5 (4× verdict), S6 (Tier-3) in flight (codex)
- ⚪ S8 operational comparison queued (serial final)

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 18:00
