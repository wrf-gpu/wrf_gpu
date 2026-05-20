# Sprint Tracker — Live Dashboard

Manager-maintained. Updated every watchman tick. Source of truth for parallel-management state.

**Per user directive 2026-05-21 ~01:10**: aim to run ≥2 parallel sprints at any time unless project-management reason prevents it. Each module: **coded → tested/corrected → Opus 4.7 reviewer closes** before merge.

## Currently in flight

| Window | Sprint | Role | AI | Status | ETA |
|---|---|---|---|---|---|
| 1 | M6 milestone plan | scout | codex gpt-5.5 xhigh | drafting plan | finishes ~01:50 |
| 2 | M5-S2 MYNN | retroactive reviewer | Claude Opus 4.7 xhigh | reading code + manager closeout | finishes ~02:00 |
| 3 | M5-S3 RRTMG radiation | worker | codex gpt-5.5 xhigh | DISPATCHING NOW | budget 6-12h |

## M5 sprint table (full history)

| Sprint | Worker (codex) | Tester | Reviewer (Opus 4.7) | Merge | Status |
|---|---|---|---|---|---|
| M5-S0 scout (pick first scheme) | codex scout | n/a | codex critical-review | `09a3738` | ✓ CLOSED (ADR-005) |
| M5-S1 Thompson microphysics | codex A1-A6 (6 attempts) | tester A4 (Opus 4.7) ✓ | reviewer A5 (Opus 4.7) ✓ Accept-with-fixes (R-1 caught Gemini hallucination) | `d768194` + `00e7ee8` | ✓ CLOSED |
| M5-S1.x Thompson tables | codex A1 (1 attempt) | n/a (closeout-only) | manager-only (defer remainder to M6) | `fe959d2` + `1868545` | ✓ CLOSED partial; debt → M6 prologue |
| ADR-007 precision policy | codex A1 (1 attempt) | n/a | Gemini side-runner (Accept-with-fixes; pre-quota-revision policy) | `445c49f` + `6c9df22` | ✓ CLOSED |
| M5-S2 MYNN PBL | codex A1 (1 attempt; 55min) | n/a (skipped under bigger-steps) | **RETROACTIVE Opus 4.7 in flight (Window 2)** | `989f143` + `e4abc86` (provisional) | ⚠ PROVISIONAL pending reviewer |
| M5-S3 RRTMG radiation | codex A1 (dispatching now Window 3) | pending | pending Opus 4.7 (mandatory per new sprint-lifecycle hard rule) | pending | OPEN |
| M5 milestone closeout | manager | n/a | manager | `52cacc3` | ⚠ PROVISIONAL pending M5-S2 reviewer + M5-S3 outcome |

## M5 closure dependencies

M5 milestone close was committed (`52cacc3`) but is **provisional** until:
1. M5-S2 retroactive reviewer Accepts (in flight Window 2)
2. M5-S3 RRTMG decision: included in M5 (close + amend M5 closeout) OR moved to M6 (current closeout text fine, but RRTMG sprint goes into M6 scope)

After both above resolve, M5 closeout can be marked final.

## M6 planning (parallel)

M6 plan scout (Window 1) drafts the M6 milestone plan. Per user directive: codex drafts, manager reviews for consensus, then dispatch M6 implementation sprints. M6 plan scout WILL likely recommend a position on RRTMG (M5 vs M6) — even if I dispatch M5-S3 now, scout's recommendation feeds into final M5-vs-M6 assignment.

## Parallelism budget

- ≥2 parallel sprints default per user directive 2026-05-21 ~01:10
- File-ownership orthogonality required (no two parallel sprints touching the same files)
- AI capacity per user 2026-05-20 evening: unlimited Opus + GPT
- Gemini reactive-only; not in normal dispatch rotation

## Update policy

Manager updates this file after every watchman tick. Each row's status moves through: `dispatching → in flight → done → tester → reviewer → CLOSED`. Stuck sprints (>2× budget) get explicit triage notes.

## Recent ticks

- 2026-05-21 00:50 — M5 closeout (provisional), M6 scout dispatched
- 2026-05-21 01:00 — M5-S2 retroactive reviewer dispatched (user-flagged double-AI gap)
- 2026-05-21 01:10 — user invoked parallel-management directive + tracker + RRTMG question
- 2026-05-21 01:15 — tracker created; M5-S3 RRTMG dispatching
