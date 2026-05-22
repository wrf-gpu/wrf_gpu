# Sprint Tracker — Live Dashboard

Manager-maintained. 30-min cadence overnight (per 2026-05-23 standing order).
Manager: Claude Opus 4.7 (1M-context). Replaces previous manager 2026-05-23 ~23:00.

## Currently in flight (3 codex parallel, file-disjoint, all dispatched 2026-05-22 23:48-23:49 UTC)

| Window | Sprint | Role | AI | Wall budget | File ownership |
|---|---|---|---|---|---|
| `2:2` | `2026-05-23-m6x-c2-pivot-critic` | critical-review | codex gpt-5.5 xhigh | 60-120 min | READ-ONLY → `.agent/sprints/.../reviewer-report.md` |
| `2:3` | `2026-05-23-m6x-dycore-alt-methods-scout` | scout (worker) | codex gpt-5.5 xhigh | 60-120 min | READ-ONLY → `.agent/sprints/.../worker-report.md` |
| `2:4` | `2026-05-23-m6x-vertical-acoustic-analytic-oracle` | worker | codex gpt-5.5 xhigh | 4-6 h | `tests/test_m6x_vertical_acoustic_oracle.py`, `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py` |

**Disjointness**: critic + scout are read-only; oracle writes only NEW files under tests/ and a new `validation/analytic_oracles/` subpackage. No file collision possible.

## Recent decisions (manager hand-over 2026-05-23 ~23:00)

| Time | Decision | Commit |
|---|---|---|
| 23:14 | HALT `worker/codex/m6x-c2-A2y-wrf-smallstep-parity` per c2-A2.x reviewer NEEDS-HYBRID-PIVOT verdict | `739d6a9` |
| 23:14 | ADR-022 DRAFT (hybrid JAX IMEX vertical operator, manager's working recommendation) | `739d6a9` |
| 23:14 | ADR-021 DRAFT (full WRF small-step shape vertical port, opposing alternative) | `739d6a9` |
| 23:18 | Three orthogonal pivot sprints dispatched | (sprint contracts) |

## Upstream context the manager inherited

| Reference | Verdict |
|---|---|
| `2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` (Opus 4.7, 9bca47c) | NEEDS-HYBRID-PIVOT (R1, R2, R5, R7 BLOCKING; horizontal half ACCEPT) |
| `2026-05-22-c2-architecture-stepback/worker-report.md` (codex) | Continue c2 with hybrid as fallback (probabilities table §2) |
| `2026-05-22-c2-methodology-stepback/worker-report.md` (gemini) | WRF-port direction is fastest viable path (argues for ADR-021) |

## Critic decision tree (after AGENT REPORTs return)

1. **If critic returns RATIFY-ADR-022** AND scout returns RECOMMEND-PROCEED-WITH-ADR-022 (or NEUTRAL):
   - Ratify ADR-022, drop the DRAFT suffix, status PROPOSED.
   - Dispatch the M6.x-c2-vert-hybrid implementation worker (codex) + Canary 3 km curvilinear smoke worker (codex) in parallel.
2. **If critic returns RATIFY-ADR-021** OR scout returns RECOMMEND-PROCEED-WITH-ADR-021:
   - Ratify ADR-021, status PROPOSED.
   - Dispatch the WRF small-step Fortran harness sprint as the prerequisite, in parallel with carry-expansion contract draft.
3. **If critic and scout disagree** (one RATIFY-022, other RATIFY-021):
   - Dispatch Gemini (reactive-only side-runner) for the tie-break per `dispatching-gemini.md`.
4. **If critic returns RATIFY-NEITHER** or scout returns RECOMMEND-THIRD-OPTION:
   - Read the third-option proposal, write ADR-023 DRAFT, dispatch a second critic round.

In all branches, the analytic-oracle worker (window 2:4) keeps running — its deliverable is needed regardless of which pivot lands.

## Closed this morning (pre-handover) — kept for continuity

The previous manager's tracker said "all-quiet awaiting M6 prologue dispatch." Since then (per `git log --all --since 2026-05-21`):
- M6-S1 through M6-S4 implementation sprints closed with various verdicts (S2 24h forecast PASS d02 FP32; S3 PARTIAL surface+Noah-MP; S4 Tier-2 PASS; S5 FAIL NaN-explode; S6 SCAFFOLD-PARTIAL; S7 SCAFFOLD-DEFER-M7)
- M6.5-D1 Gen2 backfill + ADR-016 closed
- M7-S0a operational/data prologue closed
- M6.x WRF-canonical dycore: c1 path burned A1-A11 + 4 bughunts + Klemp-Skamarock contingency → c2 architecture (ADR-020) → c2-A1, c2-A1', c2-A1'' all ACCEPT → c2-A2 horizontal PGF ACCEPT, c2-A2.x vertical-acoustic REJECT
- All M5 prologue physics work closed (Thompson HLO/tables, MYNN follow-ups, RRTMG s3x→s3y→s3z→s3zz→s3zzz→s3zzzz SW PARITY + s3zzzzz LW PARITY)

## Watchman cadence

30-min ScheduleWakeup loop tick during the user-AFK window:
1. `tmux list-windows -t 2` — confirm ephemeral windows only on `:2`, `:3`, `:4` until they self-destruct
2. Capture each non-protected pane's last 8 lines to verify "Working (...s)" status or report-on-disk
3. Check `.agent/sprints/.../*-done` markers + `*-exit` files for silent completions
4. Update this tracker
5. If any agent has finished without firing AGENT REPORT, manually capture report from disk + kill window

Per user standing order 2026-05-23: windows 0 and 1 of session 2 stay protected (manager + user spare); windows 2+ are ephemeral and torn down on completion.

## Parallelism budget

- ≤3 codex parallel — currently AT cap (3/3 codex).
- Opus available for review/critic/tool-building/test-running roles (separate pool).
- Gemini reactive-only (limited tokens) — engaged on architecture tiebreak only.

## Sprint outcome ledger (since manager handover)

| Sprint | Result | Branch / commit |
|---|---|---|
| 2026-05-22 c2-A2-A2x bundle review (Opus) | NEEDS-HYBRID-PIVOT | `reviewer/opus/c2-A2-A2x-bundle 9bca47c` |
| 2026-05-22 c2-methodology-stepback (Gemini) | "WRF-port is fastest viable" | `reviewer/gemini/c2-methodology-stepback 0b8ae9b` |
| 2026-05-22 c2-architecture-stepback (Codex) | Continue c2; hybrid as fallback | `worker/codex/c2-architecture-stepback 8d97c43` |
| 2026-05-23 m6x-c2-pivot-halt | HALT recorded on main | merge commit (this commit) |

## Recent ticks

- 2026-05-22 ~19:21 — c2-A2 bundle review verdict NEEDS-HYBRID-PIVOT (previous manager)
- 2026-05-22 ~23:00 — manager handover; new manager (Opus 4.7) takes over
- 2026-05-22 ~23:14 — ADR-021 + ADR-022 drafts written; c2-A2.y HALT marker on main
- 2026-05-22 ~23:48 — 3 parallel sprints dispatched (critic + scout + oracle worker)
- 2026-05-22 ~23:54 — manager scheduling 30-min watchman loop; user AFK for the night

— Manager (Claude Opus 4.7 1M-context), 2026-05-23 ~00:00 UTC
