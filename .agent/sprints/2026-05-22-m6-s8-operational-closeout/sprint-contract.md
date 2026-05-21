# Sprint Contract — M6-S8 Operational Closeout

**Sprint ID**: `2026-05-22-m6-s8-operational-closeout`
**Created**: 2026-05-22 ~23:15 (manager-drafted; awaits M6.x close before dispatch)
**Status**: **DRAFT** — dispatches only after M6.x Opus accept

## Trigger

M6-S6 Opus accepted M6-S8 as UNBLOCKED post-M6.x. Three F-min-S8-* follow-ups inherited from M6-S6. M6-S5 Opus mandated disclaimer language. F-5 sprint pins canonical denominator. M6.5-D1 sprint backfills Gen2 corpus for Tier-4 RMSE.

## Objective

Land the M6 operational closeout: GPU-vs-Gen2 d02 Tier-4 RMSE on U10/V10/T2 for ≥1 representative day, plus the two substantive M6-S6 follow-up closures, plus the mandatory disclaimer, plus ADR-007 PASS-vs-PASS-with-disclaimer status.

## Acceptance

- **AC1 Tier-4 operational RMSE**: Use M6.5-D1 RMSE adapter. Run uncapped M6.x dycore on d02 24h forecast (init from Gen2 wrfinput_d02). Compute per-field RMSE U10/V10/T2 against Gen2 wrfout_d02 at +6/+12/+24h. Pass: RMSE within 1.5× of Gen2-vs-AIFS RMSE (since AIFS is the deterministic-truth reference for Gen2). Document threshold rationale.
- **AC2 F-min-S8-1**: Call `water_budget_residual` with Thompson side-channel oracle in coupled operational d02 forecast. Substantively closes M6-S4 R-10. Compute residual per timestep; document distribution + max.
- **AC3 F-min-S8-2**: Add d02 boundary-zone direct GPU-vs-Gen2-wrfout comparison (boundary cells only). Substantively closes M6-S4 R-11 for d02. Per-side per-variable MAE + RMSE.
- **AC4 F-min-S8-3**: When post-M6.x d02-drift retry sprint lands, cross-check operational RMSE against re-measured (uncapped) TSC envelope.
- **AC5 Mandatory disclaimer**: Worker report MUST contain the M6-S5 §5 disclaimer language verbatim:
  > "M6-S5 demonstrates that the GPU pipeline clears the constitutional 4× throughput gate (measured 9.70× end-to-end, or 6.01× at the conservative denominator) under ADR-007 mixed precision and full-domain batching. M6-S5 ALSO demonstrates that the M4 reduced dycore is not stability-grade at WRF-canonical 3km coupled timesteps; a M6.x dycore completion sprint (canonical acoustic with physical sound speed; mu mass continuity) is required before any operational forecast claim or M7 dispatch."
  
  Plus M6.x outcome update: "M6.x [PASS|REVISED]: [evidence summary]"
- **AC6 ADR-007 status**: amend Status to PASS-with-evidence + cite M6.x closure + F-5 denominator + S8 RMSE
- **AC7 Speedup denominator**: use ADR-007-amendment-denominator.md (F-5 sprint output). NO grid-points denominator. NO shopping.
- **AC8 M6 milestone-closeout**: `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` (NEW). Lists all 8 S* + M6.x sprints, status per sprint, evidence per AC, what carries into M7.

## Files Worker May Modify

- `scripts/m6_s8_operational_closeout.py` (NEW)
- `src/gpuwrf/validation/operational_rmse.py` (NEW) — consumes M6.5-D1 adapter
- `tests/test_m6_s8_*.py` (NEW)
- `artifacts/m6/operational/**` (NEW)
- `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` (NEW)
- `.agent/decisions/ADR-007-precision-policy.md` (Status amendment; M6.x will have touched first)

## Dispatch (pending M6.x close)

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY)
- Wall-time: **6-12h**
- Worktree: `/tmp/wrf_gpu2_m6s8` (NEW at dispatch time)
- Branch: `worker/codex/m6-s8-operational-closeout`

## HARD RULES

1. Use M6.x uncapped dycore (not M4 reduced)
2. Use M6.5-D1 RMSE adapter (not custom impl)
3. Use F-5 pinned denominator (not per-sprint shopping)
4. Disclaimer verbatim per AC5
5. M7 dispatch UNBLOCKED only after this closes GREEN

## Pre-dispatch checklist

Before dispatching M6-S8, verify:
- [ ] M6.x Opus accepted PASS
- [ ] M6.5-D1 Opus accepted (RMSE adapter live)
- [ ] F-5 Opus accepted (denominator pinned)

Without all three green, M6-S8 dispatch deferred.

## End-goal context

This closes M6 entirely. After this: M7-S0 dispatch. After M7: Canary 3km daily operational.
