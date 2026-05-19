# M3 Manager Per-Turn Runbook

Self-paced `/loop` mode. One decision-tree step per turn; commit; `ScheduleWakeup`; yield. Active milestone: M3 — GPU State & Grid Skeleton (first milestone where real model-shape code lands).

Read order: `PROJECT_CONSTITUTION.md` → `AGENTS.md` → `PROJECT_PLAN.md` → `.agent/milestones/ROADMAP.md` (M3 section) → `.agent/goals/M3-DONE.md` → this file → auto-memory (especially `feedback_code_quality_bar.md` — M3 IS where that bar starts biting).

## Turn-zero check
If `git log -1 --oneline main` does not contain `Merge ... M2 closeout` or the ADR-001 status is not `ACCEPTED`, halt and tell the user. Otherwise proceed.

## Standard turn
1. Status: `python scripts/check_m3_done.py`.
2. If ok=true → run M3 closeout (§D), stop the loop (no ScheduleWakeup).
3. Otherwise: one decision-tree step, commit, ScheduleWakeup, yield.

## Decision tree (per turn)

In priority order, first match:

### A. Bootstrap repair (same as prior milestones)
- `validate_agentos.py` failing → repair + commit. Yield.
- `./data` symlink broken → recreate. Yield.
- M1 or M2 oracle regresses → STOP, write BLOCKER. Yield.

### B. Active sprint advancement
For each `dir = .agent/sprints/2026-*-m3-*/` in chronological order:

1. Contract stub → manager writes contract with **mandatory `Spacetime Budget` AC table** (per `feedback_code_quality_bar.md`). Commit. Yield.
2. Worker report stub → `dispatch_role.sh worker "$dir" --reasoning high`. Yield.
3. Tester report stub → `dispatch_role.sh tester "$dir" --reasoning xhigh` (Claude Opus 4.7 — **explicitly tasked with aesthetic + efficiency review**, not just correctness). Yield.
4. Reviewer report stub → `dispatch_role.sh reviewer "$dir" --reasoning high`. Yield.
5. Reviewer Decision = Reject → amend contract, redispatch, retry-count++ (cap 5). Yield.
6. All Accept, close_sprint fails → manager writes closeout + memory-patch + commits. Yield.
7. Sprint cleanly closed → merge to main, push, advance to next sprint.

### C. Sprint creation
Per user's "big smart steps" directive, M3 is intended as **one substantial sprint** (M3-S1: core skeleton), optionally followed by an ADR-002 ratification sprint if S1's review uncovers material architecture surprises:

- **M3-S1**: `m3-state-grid-halo-skeleton` — delivers GridSpec + State + halo stub + dummy 1000-step loop + transfer audit + spacetime budget + ADR-002 draft. Single sprint, ~1500–2500 LOC, ~3 days agent time.
- **M3-S2** (optional, only if S1 reviewer flags an ADR-002 dissent): manager writes ADR-002 with Codex critical-review, applies, finalizes.

If S1 delivers cleanly with reviewer Accept on ADR-002 inline, S2 is skipped.

### D. M3 closeout (§D of M3-DONE.md)
1. Write `.agent/decisions/MILESTONE-M3-CLOSEOUT.md`.
2. Flip `.agent/milestones/M3-gpu-state-grid.md` Reviewer Decision → Accepted.
3. Commit `feat(m3): close milestone — device-resident state + zero-transfer dummy loop`.
4. Re-check `check_m3_done.py` → ok.
5. Stop loop. Write user status report.

## Universal per-turn rules
- Never start M4. M3 only.
- Never modify governance / goal / runbook files.
- Always commit before yielding.
- Use `dispatch_role.sh` only; never raw codex/claude from Bash.
- **One step per turn.** Even with "big smart sprints," manager turn granularity is per-decision-tree-step.
- Manager-during-worker hygiene: no commits while a worker is in flight (per `feedback_manager_autonomy.md`).
- **All M3+ sprint contracts MUST include the elegance ACs from `feedback_code_quality_bar.md`** — no exception, no waiver.

## End-of-loop sanity (same as M2)
- Commit pending.
- Write one-page user status report.
- Print to stdout.
- DO NOT call ScheduleWakeup.

## Send-keys mechanism (unchanged)
Agents auto-type report into manager prompt; manager treats as notification, reads disk for truth.
