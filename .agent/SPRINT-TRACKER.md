# Sprint Tracker — Live Dashboard

Manager-maintained. 30-min cadence overnight (per 2026-05-23 standing order).
Manager: Claude Opus 4.7 (1M-context). Replaces previous manager 2026-05-23 ~23:00.

## Currently in flight (2 parallel, anti-stuck hedge after ADR-023 fallback trigger fired)

| Window | Sprint | Role | AI | Worktree | Wall budget | Goal |
|---|---|---|---|---|---|---|
| `2:..._failure-diagn-tester` | `2026-05-23-m6x-warm-bubble-failure-diagnostic` | tester (diagnostic) | **Claude Opus 4.7** xhigh (different model — fresh angle on codex-built code) | `/tmp/wrf_gpu2_diag` on `tester/opus/m6x-warm-bubble-failure-diagnostic` | 2-4 h | Diagnose WHY conservative MPAS-recurrence produces w_max=0.04 on warm-bubble. Verdict in §7: WIRING-BUG / SIGN-ERROR / ARCHITECTURAL-GAP-FALLBACK / MAGIC-NUMBER / MIXED. Drives next dispatch. Read-only + one diagnostic script. |
| `2:..._wrf-smallstep-prot-worker` | `2026-05-23-m6x-adr021-wrf-smallstep-prototype` | worker | codex gpt-5.5 xhigh | `/tmp/wrf_gpu2_adr021` on `worker/gpt/m6x-adr021-wrf-smallstep-prototype` | 8-14 h | ADR-021 fallback prototype: expand AcousticScanCarry with WRF small-step scratch (t_2ave, ww, muave, muts, ph_tend); port advance_w + advance_mu_t + calc_coef_w line-for-line from WRF source. Binding gate: warm-bubble w_max ∈ [5,10] at 600s. |

## Recently completed (this watchman session)

| Sprint | Outcome | Branch / commit | Merged on main |
|---|---|---|---|
| `m6x-adr023-public-scan-path-unification` | **PATH UNIFIED but WARM-BUBBLE FAILS** — 4/4 unification gates PASS, 23/23 regression PASS, transfer audit 5/5 PASS, fixture restored via manifest+generator. epssm now plumbed end-to-end (public sweep differs across {0,0.1,0.3}). Honest warm-bubble: w_max=0.041 m/s at 600s (target [5,10]). ADR-023 fallback trigger fires. | `worker/gpt/m6x-adr023-public-scan-path-unification @ e2391d3` | merge `d1f7d0c` |
| `m6x-adr023-d02-boundary-replay-1h` | **HALT-BY-MANAGER-PATH-SPLIT** — d02 worker halted after reviewer found path split; preserved scaffolding (scripts/m6_d02_boundary_replay_1h.py + src/gpuwrf/integration/d02_replay.py + tests/test_m6x_d02_boundary_replay.py). Redispatch after unification. | `worker/gpt/m6x-adr023-d02-boundary-replay-1h @ 47ee1bf` | merge `5260250` |
| `m6x-adr023-production-grade-reviewer` | **REJECT** — 2 binding findings: (1) BLOCKER fixture warm_bubble_2km.npz missing; (2) MAJOR path split — MPAS-recurrence path reached only via pressure_scale=0.0; public scan with non_hydrostatic=True uses _wrf_buoyancy_column_update which ignores epssm + applies prototype stabilization. ADR-023 stays PROPOSED. | `reviewer/opus/m6x-adr023-production-grade-reviewer @ b2f7a05` (16309B report) | merge `5260250` |
| `m6x-adr023-production-grade` | PASS at module-level (production-gate 4/4, MPAS slice RMSE 38.7% → 1.69%) but **path split**: the 1.69% claim does NOT apply to the public coupled-forecast scan path. | `worker/gpt/m6x-adr023-production-grade @ 0a05159` | `f4b04af` |
| `m6x-adr023-mpas-column-slice-oracle` | **PASS** — 4/4 slice tests; MPAS lines 1589-2208 literal port; peak amplitude error 1.92%, trajectory RMSE 38.7% | `worker/gpt/m6x-adr023-mpas-column-slice-oracle @ 4834599` | `0d03bc1` |

## Round 1 outcome (3 sprints dispatched 2026-05-22 23:48-23:49, returned 2026-05-23 00:55-01:10)

| Sprint | Outcome | Branch / commit | Status |
|---|---|---|---|
| `m6x-c2-pivot-critic` | Claimed `RATIFY-EITHER-WITH-CONDITIONS` per pane footer; actual report file LOST to multi-writer race (other agents' `git checkout` discarded uncommitted writes) | — | aligned with scout's third-option finding; promoted to ADR-023 round 2 |
| `m6x-dycore-alt-methods-scout` | `RECOMMEND-THIRD-OPTION` — concrete proposal: ADR-022b conservative column solver (SCREAM/HOMME DIRK Newton + ICON4Py tridiagonal + MPAS Klemp 2007 forward-backward) | `worker/gpt/m6x-dycore-alt-methods-scout @ 67c75ce` → merged main `8582355` | ✓ CLOSED |
| `m6x-vertical-acoustic-analytic-oracle` | 3 RED pytest tests on disk + closed-form analytic module (Skamarock 2008 §3.2 dispersion). Gates the pivot prototype acceptance. | `worker/gpt/m6x-vertical-acoustic-analytic-oracle @ 14caf8f` → merged main `f6965be` | ✓ CLOSED |

## Round 2 outcome — converged on ADR-023

| Sprint | Outcome | Branch / commit |
|---|---|---|
| `m6x-adr023-three-way-critic` | `RATIFY-ADR-023` with 10 required fixes (4 MAJOR, 6 MEDIUM) | `critic/codex/m6x-adr023-three-way-critic @ c1a3ded` → merged main |
| `m6x-adr023-conservative-column-prototype` | **CODE-RUNNING PROOF**: R7 3/3 GREEN, warm-bubble PASS (w_max=8.52 m/s ∈ [5,10]), solver 4/4, c2 horizontal 8/8, transfer audit 5/5, 20 launches, 0 transfers | `worker/gpt/m6x-adr023-conservative-column-prototype @ 1e157f7` → merged main |

**Manager decision (2026-05-23 02:10 UTC)**: **ADR-023 RATIFIED to PROPOSED**. Critic's required-fixes (F1-F10) and prototype caveats (NH tuning heuristics; mu_continuity gated off; no profiler artifact yet) are open work for the next production-grade sprint.

The decisive evidence: prototype demonstrably turned the R7 RED tests GREEN and survived 600s warm-bubble without nonfinite. This proves the conservative column solver direction works in code, not just paper. The prototype's stabilization heuristics are NOT acceptable as production physics — but they are acceptable as proof-of-concept that the architecture supports the conservative-column-solve operator without expanding the carry.

## Next sprint (planned)

**`2026-05-23-m6x-adr023-production-grade`**: replace prototype-grade stabilization with first-principles derivation, add MPAS/WRF column slice oracle (non-tautological, per critic F1), implement coupled `(w, mu, theta, phi)` solve with mu in-scan, run `epssm ∈ {0.0, 0.1, 0.3}` sweep, add `ncu`/`nsys` profiler artifact, document post-solve replacement order. Then the M6 acceptance ladder begins: column slice → warm bubble → 1h d02 replay → 24h/72h Gen2 RMSE.

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
- 2026-05-22 ~23:48 — round 1: 3 parallel sprints dispatched (critic + scout + oracle worker)
- 2026-05-23 ~00:55-01:10 — round 1 agents finished; scout/oracle committed cleanly; critic lost its report to multi-writer race
- 2026-05-23 ~01:24 — watchman tick 1: discovered critic loss + read scout/oracle deliverables
- 2026-05-23 ~01:30 — scout + oracle merged to main; ADR-023-DRAFT written (third option)
- 2026-05-23 ~01:36 — round 2: dispatched critic + prototype in dedicated worktrees (`/tmp/wrf_gpu2_critic_r2`, `/tmp/wrf_gpu2_proto`)
- 2026-05-23 ~02:00 — round 2 agents finished cleanly: critic RATIFY-ADR-023 (committed), prototype passed all acceptance gates
- 2026-05-23 ~02:10 — watchman tick 2: read both reports; merged both branches to main; ADR-023 ratified DRAFT→PROPOSED with critic required-fixes folded; ADR-022-DRAFT superseded
- 2026-05-23 ~02:17 — next-phase sprint dispatched: MPAS column-slice oracle (closes critic F1, F6 acceptance-ladder rung 2)
- 2026-05-23 ~02:33 — slice oracle returned PASS (4/4 tests, 1.92% peak / 38.7% trajectory RMSE) — merged to main
- 2026-05-23 ~02:52 — production-grade sprint dispatched (6-10h, 12 acceptance criteria, target RMSE <15%)
- 2026-05-23 ~03:20 — production-grade returned in 25m: RMSE 38.7%→1.69% (target <15% smashed by 9×). All gates GREEN. Open risk: launch count 20→67.
- 2026-05-23 ~03:30 — 2 parallel sprints dispatched: reviewer (binding lifecycle gate) + d02 boundary replay (F6 rung 4)
- 2026-05-23 ~04:05 — reviewer returned REJECT after 9m: path split + missing fixture; ADR-023 stays PROPOSED
- 2026-05-23 ~04:08 — d02 worker halted (received manager halt message; committed scaffolding + HALT-BY-MANAGER-PATH-SPLIT report); halt + reject merged to main
- 2026-05-23 ~04:08 — unification sprint dispatched (m6x-adr023-public-scan-path-unification, 4-7h)
- 2026-05-23 ~04:24 — unification returned in 16m: path unified + 23/23 regression PASS, but honest warm-bubble fails (w_max=0.04 vs [5,10]). ADR-023 fallback trigger fires.
- 2026-05-23 ~04:47 — anti-stuck hedge dispatched: Opus diagnostic (2-4h, fresh-model angle) + Codex ADR-021 prototype (8-14h, Plan B carry expansion)

— Manager (Claude Opus 4.7 1M-context), 2026-05-23 ~04:50 UTC
