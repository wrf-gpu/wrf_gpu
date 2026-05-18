# M1 Manager Per-Turn Runbook

The `/goal` loop re-fires the manager every turn until `.agent/goals/M1-DONE.md` is satisfied. This file tells the manager **what to do each turn**. Read it after `M1-DONE.md`.

## Turn-zero (first turn after /goal)

If `git log -1 --oneline` does not contain the literal token `[m1-bootstrap]`, the bootstrap commit is missing — stop and tell the user (this should not happen if the prep pass succeeded). Otherwise, proceed to "Standard turn."

## Standard turn (every turn after bootstrap)

Execute, in order:

1. **Status check.** Run `python scripts/check_m1_done.py`. Print summary.
2. **If `ok == true`** → run "M1 closeout" (below), then stop the loop.
3. **Otherwise**, identify the *next required action* using the decision tree below. Execute exactly one step per turn. Do not chain sprints in one turn — let `/goal` re-fire.

## Decision tree (per-turn next action)

In priority order, take the **first** matching branch:

### A. Bootstrap repair
- If `validate_agentos.py` fails: read the error, repair the missing scaffolding file, commit `chore: repair agentos scaffolding`. Done for this turn.
- If `/mnt/data/wrf_gpu2` or the `./data` symlink is missing: re-run `mkdir -p /mnt/data/wrf_gpu2/{fixtures,runs,profiler_artifacts,cache} && ln -sfn /mnt/data/wrf_gpu2 ./data`. Done for this turn.

### B. Active sprint advancement
For each `dir = .agent/sprints/2026-*-m1-*/` in chronological order:

1. **Sprint contract missing or template-only** (`sprint-contract.md` <600 bytes or contains "## Objective\n\n## Non-Goals"): manager writes the full contract using the patterns from the already-written S1 contract as a model. Commit. Done for this turn.
2. **Worker report still template** (`worker-report.md` <400 bytes or missing `Summary:` token): `bash scripts/dispatch_role.sh worker "$dir"`. Done for this turn.
3. **Tester report still template** (`tester-report.md` <400 bytes or missing `Decision:` token): `bash scripts/dispatch_role.sh tester "$dir"`. Done for this turn.
4. **Reviewer report still template** (`reviewer-report.md` <400 bytes or missing `Decision:` token): `bash scripts/dispatch_role.sh reviewer "$dir"`. Done for this turn.
5. **Reviewer Decision = Reject or Accept with required fixes (fixes not done)**:
   - If `.${role}-retry-count` < retry cap: amend the contract to address the findings (manager edits sprint-contract.md), reset `.worker-done`/`.tester-done`/`.reviewer-done` markers for the affected roles, re-dispatch worker (next turn picks up tester/reviewer). Done for this turn.
   - If retry cap reached on any role: write `.agent/decisions/BLOCKER-${sprint_id}.md` per the M1-DONE.md escalation template, stop the loop.
6. **All three reports present, Decision = Accept, but close_sprint fails**: read the close_sprint error. Usually means manager-closeout.md or memory-patch.md is template. Manager writes them now (no agent needed — manager owns these). Commit. Done for this turn.
7. **Sprint cleanly closed** (close_sprint ok=true): move on; this sprint is done. Loop to next sprint.

### C. Sprint creation
If all currently-created M1 sprints are cleanly closed but `check_m1_done` still fails on proof objects, the manager creates the **next sprint** from the M1 plan's candidate list, in this order:

- S1 `m1-fixture-storage-policy` (already created by bootstrap)
- S2 `m1-analytic-stencil-fixture` — analytic stencil micro fixture (Problem-1 shape: 3D advection-diffusion)
- S3 `m1-analytic-column-fixture` — analytic column micro fixture + WRF variable-map seed
- S4 `m1-canary-wrf-derived-fixture` — single Canary WRF-derived fixture; **MUST prefer slicing an existing run from `/mnt/data/canairy_meteo/runs/` over launching a new CPU WRF job**

Sprint creation steps (per turn):
1. `python scripts/create_sprint.py <sprint-id>`.
2. Write the sprint contract following the structure of `2026-05-18-m1-fixture-storage-policy/sprint-contract.md`: Objective, Non-Goals, File Ownership (narrow), Inputs, Acceptance Criteria (machine-checkable), Validation Commands, Performance Metrics (n/a for fixture sprints), Proof Object, Risks, Handoff Requirements.
3. Commit `chore(sprint): open <sprint-id>`. Done for this turn.

### D. M1 closeout

When all sprints closed and proof objects present:

1. Write `.agent/decisions/MILESTONE-M1-CLOSEOUT.md` with: summary, list of closed sprints (with their reviewer-Decision lines), list of proof objects with paths, residual risks, recommended next-milestone start date, top three things learned.
2. Edit `.agent/milestones/M1-wrf-oracle-fixtures-plan.md`: change `Reviewer Decision: Pending` → `Reviewer Decision: Accepted`. Add a one-line "M1 closed YYYY-MM-DD" entry.
3. Commit `feat(m1): close milestone — fixture oracle established`.
4. Re-run `python scripts/check_m1_done.py`. Should return `ok: true`.
5. Stop the loop. Write a top-level status report to the user using `docs/user-status-report-format.md`.

## Universal per-turn rules

- **Never start M2 work.** This loop is M1-only.
- **Never modify governance files** (constitution, scope, spec, principles, validation strategy, precision policy, performance targets, risk register edits >5 lines, plan §1-§9, this runbook, the goal file).
- **Always commit your turn's work** before yielding. Untracked or staged-only changes between turns cause drift.
- **Use `dispatch_role.sh`** for all agent work. Do not spawn raw codex/claude from `Bash` directly — the dispatcher handles tmux, logs, retries, and the close-on-delivery rule.
- **If a turn fails to produce forward progress** (no commit, no agent dispatch), write a one-line note to `logs/manager-stall.log` and try a different branch of the decision tree on the next turn.
- **If two consecutive turns make no forward progress**, write `BLOCKER-stall-${timestamp}.md` and stop the loop.

## When to ask Codex for a critical review instead of acting

Per the manager-autonomy memory: when the manager has a non-trivial design or architecture choice it would otherwise want to escalate, instead:

1. Create `.agent/decisions/REVIEW-<topic>/proposal.md` with the proposed call + rationale + alternatives considered.
2. Run `bash scripts/dispatch_role.sh critical-review .agent/decisions/REVIEW-<topic>/`.
3. Read the resulting `critical-review.md`. Apply findings or document dissent.
4. Continue.

Do not stop the loop for a routine architectural call — use Codex.

## End-of-loop sanity

When stopping for any reason (M1 closeout, blocker, stall, timeout), the final manager turn must:
- Commit all pending changes.
- Write a one-page user status report.
- Print the report to stdout so the user sees it on attach.
- Echo the stop reason and the current `git log -5 --oneline`.
