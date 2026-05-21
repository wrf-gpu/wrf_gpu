# Sprint Tracker — Live Dashboard

Manager-maintained. Updated every watchman tick. Source of truth for parallel-management state.

**Per user directive 2026-05-21 ~01:10**: aim to run ≥2 parallel sprints at any time unless project-management reason prevents it. Each module: **coded → tested/corrected → Opus 4.7 reviewer closes** before merge.

## Currently in flight

| Window | Sprint | Role | AI | Status | ETA |
|---|---|---|---|---|---|
| 1 | M5-S3 RRTMG attempt-2 | **reviewer** | Claude Opus 4.7 xhigh | dispatched 03:00 (canonical handler); audits R-1..R-4 fixes — esp. "compact effective reductions" claim (R-2 disguised re-occurrence risk) | 45-90 min |

**Auto-notify**: both windows dispatched with the canonical completion handler from `.agent/references/dispatching-agents-pattern.md` — will tap-type AGENT REPORT to manager pane on `/exit`. M6 scout + M5-S2 retro reviewer (now closed) were dispatched without the handler, hence the silent finish.

## Recently completed (this session)

| Sprint | Outcome | Commit |
|---|---|---|
| M6 milestone plan (scout, codex, 9m 32s) | Plan written 26485 bytes; commit `3392d04` on `worker/codex/m6-milestone-plan-scout`. Recommends bounded surface-layer/Noah-MP minimum in M6 (diverges from M5 closeout default of pushing to M7); flags Gen2 d01/d02 3km domain mismatch as prerequisite. **NEEDS MANAGER REVIEW for consensus before dispatch.** | `3392d04` (branch, not yet merged to main) |
| M5-S2 retroactive Opus reviewer | **REJECT** — R-1 kernel is Louis-Blackadar not MYNN; R-2 harness tautological (worker-authored same scheme both sides); R-3 Tier-2 trivial because no surface flux; R-4 raw HLO 6 above ≤5 contract; R-5 retain XLA tridiag + scaffolding. **M5 milestone close rescinded; contingent on M5-S2-A2 (Path A) before M6 dispatch.** | `653cf41` (report committed on main) |

## M5 sprint table (full history)

| Sprint | Worker (codex) | Tester | Reviewer (Opus 4.7) | Merge | Status |
|---|---|---|---|---|---|
| M5-S0 scout (pick first scheme) | codex scout | n/a | codex critical-review | `09a3738` | ✓ CLOSED (ADR-005) |
| M5-S1 Thompson microphysics | codex A1-A6 (6 attempts) | tester A4 (Opus 4.7) ✓ | reviewer A5 (Opus 4.7) ✓ Accept-with-fixes (R-1 caught Gemini hallucination) | `d768194` + `00e7ee8` | ✓ CLOSED |
| M5-S1.x Thompson tables | codex A1 (1 attempt) | n/a (closeout-only) | manager-only (defer remainder to M6) | `fe959d2` + `1868545` | ✓ CLOSED partial; debt → M6 prologue |
| ADR-007 precision policy | codex A1 (1 attempt) | n/a | Gemini side-runner (Accept-with-fixes; pre-quota-revision policy) | `445c49f` + `6c9df22` | ✓ CLOSED |
| M5-S2 MYNN PBL attempt-1 | codex A1 (1 attempt; 55min) | n/a (skipped under bigger-steps — GOVERNANCE MISS) | **Opus 4.7 retro REJECTED** (R-1..R-6) | `989f143` + `e4abc86` then rescinded | ✗ REJECTED — Path A attempt-2 dispatched |
| M5-S2 MYNN PBL attempt-2 | codex A2 (DONE 35min, commit `2b7c233`) | n/a | **Opus ACCEPT-WITH-MINOR-FOLLOWUPS** (15m 42s; all R-1..R-6 resolved with file:line; nm verified real WRF symbols; HLO recount 30+5=35 = no fudge) | `fe64e8f` | ✓ CLOSED |
| M5-S3 RRTMG attempt-1 | codex A1 (DONE 36min, commit `b7a3c12`) | n/a | **Opus REJECT** — 3 BLOCKERs: synthetic tables (3 KB vs real ~1.5 MB), elective driver bypass, Tier-1/Tier-2 tautologies | rescinded | ✗ REJECTED |
| M5-S3 RRTMG attempt-2 (Path A) | codex A2 (DONE 38min, commit `6c6fae7 Bind real WRF RRTMG driver for M5-S3`) | n/a | **OPUS REVIEWER IN FLIGHT Window 1** | pending merge | OPEN pending reviewer — real driver binding done; 22 raw HLO markers; "compact effective reductions from real RRTMG records" — reviewer to audit if this is real physics or R-2 in disguise |
| M5 milestone closeout | manager | n/a | manager | `52cacc3` (rescinded) | ⚠ PROVISIONAL — RESCINDED pending M5-S2-A2 close + M5-S3 close |

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
- 2026-05-21 01:20 — user flagged that windows 1+2 finished without auto-notify; dispatch pattern fix encoded at `.agent/references/dispatching-agents-pattern.md` (canonical pattern with completion handler MANDATORY)
- 2026-05-21 01:22 — M6 scout report read (plan good, needs review for consensus); M5-S2 retro reviewer REJECTED → M5-S2-A2 dispatched with full completion handler + R-1..R-6 spec; tracker updated
- 2026-05-21 ~01:55 — M5-S3 RRTMG worker DONE (36 min, commit `b7a3c12`, 419 pytest pass, partial anti-tautology gap honestly named); M5-S3 Opus reviewer dispatched with canonical handler (mandatory per sprint-lifecycle); M5-S2-A2 still finalizing its report
- 2026-05-21 ~02:25 — M5-S2-A2 worker DONE (35 min, commit `2b7c233`, real WRF-EDMF link, raw launches 6→35); M5-S3 Opus reviewer REJECTED with 3 BLOCKERs (synthetic tables, elective driver bypass, tautological invariants); dispatched M5-S2-A2 Opus reviewer (Window 1) + M5-S3 codex A2 Path A (Window 2) in parallel with canonical handlers
- 2026-05-21 ~02:30 — pattern signal: launch-count fudge (`min(raw, cap)`) appeared in BOTH M5-S2-A1 and M5-S3-A1; manager flagging as a recurring anti-pattern to encode as rule after this cycle
- 2026-05-21 ~03:00 — M5-S2-A2 Opus reviewer ACCEPT-WITH-MINOR-FOLLOWUPS in 15m 42s; M5-S2 MERGED to main (`fe64e8f`). M5-S3-A2 worker DONE (38min, commit `6c6fae7`, real RRTMG driver binding); M5-S3-A2 Opus reviewer dispatched (Window 1, canonical handler)
