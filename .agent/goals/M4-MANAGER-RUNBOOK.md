# M4 Manager Runbook — Per-turn Decision Tree

Used by the self-paced `/loop` Claude manager. Mirrors M3 runbook. Loop terminates when `python scripts/check_m4_done.py` returns `{"ok": true}` AND user has explicitly approved ADR-003 (constitutional gate).

## Pre-flight (do once at M4 entry)

- Confirm ADR-002 status = `ACCEPTED 2026-05-19 by user` (state layout locked).
- Confirm `git status` clean on main; last commit is M3 closeout merge.
- Confirm M3 oracle still passes: `python scripts/check_m3_done.py`.
- Confirm `feedback_debuggability_hooks.md` memory loaded.

## Per-turn decision tree

**Step 1 — Read state.**
- `python scripts/check_m4_done.py` → if `{"ok": true}` and ADR-003 user-approved → write `MILESTONE-M4-CLOSEOUT.md`, present status report, stop the loop.
- Else continue.

**Step 2 — Identify active sprint.**
- `ls .agent/sprints/2026-*-m4-*` → newest folder is the active sprint.
- If none: dispatch M4-S1 per the M4-S1 sprint contract.

**Step 3 — Per-sprint lifecycle.**
The lifecycle is: **worker → tester → reviewer → (manager closeout | fix-cycle)**.

- If `.worker-done` missing → worker still in flight; ScheduleWakeup 600s, return.
- If `.worker-done` present + no `tester-report.md` → dispatch tester (Claude Opus 4.7 xhigh).
- If `tester-report.md` present + no `reviewer-report.md` → dispatch reviewer (codex gpt-5.5).
- If `reviewer-report.md` present:
  - Decision = `Accept`: write `manager-closeout.md`; if ADR-003 is the work-product, dispatch codex critical-review next; else merge branch to main + push.
  - Decision = `Accept with required fixes`: amend `sprint-contract.md` with fix-cycle ACs, increment attempt, re-dispatch worker. Mirrors M3 attempt-2 pattern.
  - Decision = `Reject`: amend contract, re-dispatch worker. If reviewer rejects twice with the same root cause, escalate to user.

**Step 4 — ADR-003 critical-review sub-step (after S1 reviewer Accept).**
- Manager finalizes ADR-003 body, writes `.agent/decisions/REVIEW-codex-ADR-003/proposal.md`.
- Dispatch `critical-review` role; apply findings; commit.

**Step 5 — User approval gate (constitutional).**
- After ADR-003 finalized + critical-review applied: present status report to user requesting explicit approval.
- Loop **stops** (no ScheduleWakeup) — user reply unblocks M5.

## Hygiene rules (carried from M3)

- No manager commits while an agent is in flight (shared worktree contamination).
- Sprint folder + branches per role: `worker/gpt/m4-*`, `tester/claude/m4-*`, `reviewer/opus/m4-*`.
- `check_m1_done` + `check_m2_done` + `check_m3_done` regression check before each milestone advance.
- Memory updates only after major architectural inflection (ADR-003 acceptance) using the patch protocol.

## Elegance + debug-hook discipline (carried from M3, raised for M4)

- Every worker-report MUST include the spacetime budget table inline.
- Every worker-report MUST include the **HLO debug-vs-stripped diff hash** (`sha256` of the empty diff = 0 = pass).
- Reviewer MUST attest: "I read every line of `src/gpuwrf/dynamics/step.py` and `src/gpuwrf/debug/asserts.py`; the production HLO is byte-identical to a debug-stripped sibling build."

## Failure modes to watch for (lessons from M3)

- **Hidden H2D transfers via Python scalars in the scanned body.** Make every dt/n_sub/scheme-selection scalar `static_argnums`. The M3 `dt`-static fix is the model.
- **`@jit` cache misses across equivalent-but-distinct grids/states.** `GridSpec.__eq__` already handled; ensure `Tendencies` / debug-flag combinations don't explode the cache.
- **Debug branch leaking into production HLO.** The MUST-empty diff is the only gate; if it's not empty, refactor until it is.
