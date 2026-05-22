# Morning Report — 2026-05-23 ~00:00 UTC (manager handover)

**Status**: **M6 in crisis but bounded.** New manager (Opus 4.7) took over from previous manager 2026-05-22 ~23:00. Three parallel sprints in flight to resolve the dycore-vertical-acoustic pivot. User AFK for the night.

## Quick visual

```
M0 ─── M1 ─── M2 ─── M3 ─── M4 ─── M5 ─── M6 ─── M7 ─── M8
 ✓      ✓      ✓      ✓      ✓      ✓     ⚠      ◐      -
                                          pivot
                                          decision
                                          in flight
```

## What changed yesterday (2026-05-22)

After the previous manager's morning-report ("all-quiet awaiting user M6 prologue dispatch"), the project executed:

- M5 prologue closed: Thompson HLO + tables, MYNN follow-ups, RRTMG s3.x through s3.zzzzz reached full SW PARITY + LW PARITY.
- M6 implementation sprints S1-S7 closed with various verdicts (S2 FP32 24h forecast PASS on d02 with zero transfers; S5 NaN-explode; S6/S7 SCAFFOLD).
- M6.5-D1 Gen2 backfill + ADR-016 closed.
- M7-S0a operational/data prologue closed.
- **M6.x WRF-canonical dycore entered a long bug-hunt loop**: c1 path A1-A11 + 4 bughunts + Klemp-Skamarock contingency design. c1 abandoned after warm-bubble retest showed dycore correct at 300 s but blowing up at 350 s.
- c2 architecture (ADR-020) drafted, spike-absorbed, accepted through three review rounds (c2-A1, c2-A1', c2-A1'').
- c2-A2 horizontal PGF + mu continuity + acoustic scan: **ACCEPT** per Opus reviewer.
- c2-A2.x vertical acoustic: **REJECT** per Opus reviewer with verdict **NEEDS-HYBRID-PIVOT** (commit `9bca47c`, 2026-05-22 19:21). Two of three architecture-step-back pivot criteria tripped: no vertical-acoustic oracle; broad unreviewed carry contract change needed.
- Methodology step-back (Gemini): "WRF-port is the fastest viable path".
- Architecture step-back (Codex): "Continue c2 with hybrid as fallback".

## Where we are now

`main` tip: this morning report commit. Latest reviewed state of the dycore:

- c2-A2 horizontal PGF + mu continuity = **ACCEPT** (lives on main at commit `0436279`).
- c2-A2.x vertical acoustic = HALT pending pivot ADR ratification.
- Two pivot ADR drafts on main: `ADR-022-hybrid-vertical-operator-DRAFT.md` (manager's recommendation), `ADR-021-wrf-smallstep-vertical-port-DRAFT.md` (opposing alternative).

## In flight overnight (3 parallel)

1. **Codex critic** (window `2:2`) — argues opposing position on ADR-021 vs ADR-022, returns verdict in §5. Budget 60-120 min.
2. **Codex research scout** (window `2:3`) — comparative table of Pace, Dinosaur, NeuralGCM, ICON4Py, SCREAM, MPAS-A vertical-implicit step choices. Budget 60-120 min.
3. **Codex worker** (window `2:4`) — builds 1-D linear-acoustic analytic oracle test (closes R7, needed regardless of pivot direction). Budget 4-6 h.

## Decision tree at next manager wake

After critic + scout return:

| Scenario | Action |
|---|---|
| Both endorse ADR-022 | Ratify ADR-022, dispatch implementation worker + Canary 3 km curvilinear smoke worker in parallel |
| Both endorse ADR-021 | Ratify ADR-021, dispatch WRF small-step Fortran harness + carry-expansion contract sprints |
| Disagreement | Dispatch Gemini (reactive-only) for tiebreak |
| Third option proposed | Write ADR-023 DRAFT, dispatch second critic round |

The analytic-oracle worker keeps running regardless (deliverable is shared between both pivot directions).

## Risks and how they're being managed

- **Dycore architecture redo cost overruns (HIGH)** — third pivot in same milestone. Mitigation: two parallel research streams (critic + scout) before ratifying, plus the anti-stuck rule mandates a parallel rewrite-with-different-method worker if the chosen ADR fails its first sprint.
- **Tier-4 RMSE vs Gen2 backfill not yet measured (medium)** — first measurement happens at M6-Sprint-3 after the dycore stability proof lands.
- **GPU OOM at d02 / 1 km (medium)** — M6-S6 already OOM'd at d02; deferred risk.
- **Worker spec-gaming pattern (medium)** — caught reliably by Opus reviewer. Skill patches `.proposed.md` still pending merge to `.agent/skills/`.

## Per user standing orders (2026-05-23)

1. Parallel work whenever possible — **honored** (3-way parallel dispatched).
2. Gemini sparingly — **honored** (not engaged in this dispatch; reserved for tiebreak).
3. Status table per dispatch — **honored** (this report + SPRINT-TRACKER.md).
4. 30-min overnight loop — **scheduled** via ScheduleWakeup.
5. Anti-stuck rule on dycore — **honored** (research scout dispatched alongside critic; rewrite worker queued for post-ratification).
6. GPT-5.5 feedback before core decisions — **honored** (critic + scout are both GPT-5.5).

## Recommended next user actions (when you wake up)

1. Read `MILESTONE-M5-CLOSEOUT.md` for the project context if not already read (already merged).
2. Read this report + `.agent/SPRINT-TRACKER.md` for the in-flight state.
3. Read whichever of `reviewer-report.md` (in `2026-05-23-m6x-c2-pivot-critic/`) and `worker-report.md` (in `2026-05-23-m6x-dycore-alt-methods-scout/`) have returned.
4. The manager will either have dispatched the next round automatically (if critic + scout agreed) or will be waiting for your confirmation if they returned a third option / RATIFY-NEITHER. Either way the project is not stuck.

— Manager (Claude Opus 4.7 1M-context), 2026-05-23 ~00:05 UTC
