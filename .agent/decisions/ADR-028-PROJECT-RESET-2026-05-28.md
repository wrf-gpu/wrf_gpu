# ADR-028 — Project Reset for Operational-Skill Closure

**Status**: PROPOSED (awaiting principal sign-off)
**Date**: 2026-05-28
**Decision owner**: Manager (Claude Opus 4.7)
**Supersedes**: nothing immutable; updates the active scope of `.agent/milestones/ROADMAP.md`
**Related**: [`PROJECT-RESET-PLAN-FINAL.md`](PROJECT-RESET-PLAN-FINAL.md), [`MILESTONE-M7-CLOSEOUT-AMENDMENT.md`](MILESTONE-M7-CLOSEOUT-AMENDMENT.md)

## Context

v0.0.1 shipped 2026-05-28 with bitwise dycore savepoint parity at 100 coupled steps vs unmodified WRF v4 and a corrected 22.26× apples-to-apples speedup. The same release documented an operational forecast skill regression (T2 +161-378 %, U10 +214-370 %, V10 +177-353 % RMSE vs CPU WRF on a 5-day Canary case). The principal directive 2026-05-28 declared this an unfinished GPU port: "we are not done … the project scope top level is that it needs to be a fully usable gpu port."

## Decision

Reset the project's active scope to close the operational-skill gap. Roadmap **M8 → M23** over **32-45 honest weeks** (Q1-Q2 2027 target). Adopt the merged final plan at [`PROJECT-RESET-PLAN-FINAL.md`](PROJECT-RESET-PLAN-FINAL.md). Apply the expanded invariant ladder INV-1..11. Replace the inadequate "p > 0.05 paired test" gate with a TOST equivalence test at predeclared margins. Add new milestones the draft missed (static-field parity, lateral boundary completeness, conservation closure, idealized-case suite, validation-corpus build). Freeze v0.0.1 paper and public repo until M23.

## Methodology used for the decision

Three-input merge:
1. Opus draft plan (manager, this conversation).
2. Codex GPT-5.5 xhigh adversarial critic on the draft (`.agent/sprints/2026-05-28-project-reset-critic/critique.md`, `CRITIQUE_COMPLETE`).
3. Codex GPT-5.5 xhigh blinded planner — built independent plan from permitted evidence only, did not read the manager draft (`.agent/sprints/2026-05-28-project-reset-blinded/plan.md`, `PLAN_COMPLETE`).

Both independent reviewers converged on **~33 % position assessment** and **30-45 week honest timeline**. Both flagged the same draft errors:
- "p > 0.05" is not an equivalence test (TOST required).
- Single-case 5-day pinned RMSE is too narrow as an INV-4 ratchet.
- Missing static-field, boundary, conservation, idealized, corpus-availability milestones.
- M11 (Noah-MP) is a full physics port (8-14 weeks), not an extension of hourly replay.
- 22.26× current speedup has only 2.2× margin to the 10× floor; M22 perf re-certification is high risk.

## Consequences

### Adopted
- **Binding goal** at top of every contract: TOST equivalence on ≥15-case Canary L2+L3 seasonal ensemble.
- **Milestone roadmap** M8-M23 across 7 phases (foundation reset → atmospheric correctness → closure/idealized → Noah-MP → skill recovery → statistical equivalence → release).
- **Invariant ladder** INV-1..11 (5 expansions + 5 new invariants from critic). One-way ratchet on D2H, savepoint depth, perf floor, conservation closure, static-field parity, boundary completeness, guard accounting, evaluation sufficiency.
- **Multi-AI cross-check tightening**: at milestone close for M9/M11/M12/M13/M14/M16/M21, a third independent reviewer + blinded proof auditor are mandatory (Gemini agy fills the slot).
- **Statistics reviewer** is mandatory for M20+M21.
- **Auto-notify worker pattern**: every worker tmux dispatch includes `tmux send-keys -t 1 "AGENT REPORT: <name> exit=$?" Enter` so the manager is woken by event, not by polling. Persisted as memory [[feedback-worker-tmux-notify-pattern]].
- **Publish freeze**: `github.com/wrf-gpu/wrf_gpu` v0.0.1 + paper stay frozen until M23 closes with TOST equivalence proven. A one-sentence README disclaimer is pushed early.
- **Speed claim discipline**: only the 22.26× corrected number is in current circulation; M22 re-certifies under final code. 156× claim is dead.

### Rejected from the draft
- Draft's 17-23 week timeline (replaced by 32-45 weeks).
- Draft's "p > 0.05 paired t-test" gate (replaced by TOST with margins).
- Draft's INV-4 "RMSE must decrease or hold equal on the pinned case" (replaced by mini-ensemble median RMSE non-increase + ADR-waivers — too brittle as written would reject legitimate fixes).
- Draft's M10 theta limiter as a standalone milestone (folded into M11 dycore correctness; clip accounting in INV-10).

### Risks
- Timeline pressure: 32-45 weeks pushes v0.1.0 into Q1-Q2 2027 vs the principal's preferred Sept-Oct 2026. Honest > schedule-pressed.
- Noah-MP (M16) is the largest single risk; could blow up to 14+ weeks.
- 22.26× → ≥10× margin is narrow; correctness fixes may force re-tuning before M22 passes.
- Validation corpus (M20) currently fails the five-complete-day gate; data work is non-trivial.

### Reversibility
ADR-028 can be amended by a follow-up ADR if M9 audit reveals scope can be trimmed (e.g. Noah-MP can be deferred to v0.2.0 if its absence does not materially affect TOST equivalence on the 15-case ensemble — which the M9 audit may reveal).

## Acceptance

Principal sign-off required to dispatch M8 (`Evidence freeze + proof registry + savepoint harness`). Once signed, the M8 sprint contract is drafted, dispatched to codex frontrunner + Opus verifier, and the auto-notify pattern is in effect.
