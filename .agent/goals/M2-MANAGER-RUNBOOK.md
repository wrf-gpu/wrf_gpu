# M2 Manager Per-Turn Runbook

The manager runs in a **self-paced `/loop`**. Each turn the manager executes one decision-tree step, commits, calls `ScheduleWakeup` with a reasonable next-tick delay (default 1200–1800 s), and yields. Agents finishing out-of-band send their summary back into the manager's tmux window via `tmux send-keys` (handled by `scripts/dispatch_role.sh`).

Read in this order: `PROJECT_CONSTITUTION.md` → `AGENTS.md` → `PROJECT_PLAN.md` (§5 backend bakeoff design) → `.agent/milestones/ROADMAP.md` (M2 section) → `.agent/goals/M2-DONE.md` → this file.

## Turn-zero (first manager turn of the M2 loop)

If `git log -1 --oneline main` does not contain `feat(m1): close milestone`, M1 isn't actually closed — stop and tell the user. Otherwise proceed to "Standard turn."

## Standard turn

1. **Status check.** `python scripts/check_m2_done.py` → print summary.
2. **If `ok == true`** → run "M2 closeout" (§D), stop the loop (do NOT call ScheduleWakeup).
3. **Otherwise** execute exactly one decision-tree step (§B/§C), commit, `ScheduleWakeup`, yield.

## Decision tree (per turn next action)

In priority order, first matching branch:

### A. Bootstrap repair (same as M1)
- `validate_agentos.py` failing → repair + commit + yield.
- `./data` symlink missing → recreate + commit + yield.
- M1 fixtures missing on `main` → halt loop, BLOCKER (this would mean a regression).

### B. Active sprint advancement
For each `dir = .agent/sprints/2026-*-m2-*/` in chronological order:

1. **Sprint contract missing / template-stub** → manager writes contract using the S1/S2/S3 patterns from M1 as templates. Each M2 sprint scoped to one candidate's stencil-OR-column-OR-both work, with explicit File Ownership, AC, and the profiler JSON schema. Commit. Yield.
2. **Worker report still stub** → `bash scripts/dispatch_role.sh worker "$dir" --reasoning high`. Yield.
3. **Tester report still stub** → `bash scripts/dispatch_role.sh tester "$dir" --reasoning xhigh` (Claude Opus 4.7 — cross-AI verification). Yield.
4. **Reviewer report still stub** → `bash scripts/dispatch_role.sh reviewer "$dir" --reasoning high`. Yield.
5. **Reviewer Decision = Reject** → amend contract (or, for hopeless cases, accept candidate failure with reviewer_decision=excluded and write `*-failure.json`). If retry < cap, redispatch worker. If retry cap reached, write `candidate-failure.json` and proceed to next candidate. Yield.
6. **All reports Accept, close_sprint fails** → manager writes `manager-closeout.md` + `memory-patch.md`, commits, yields.
7. **Sprint cleanly closed** → manager merges sprint branch → main, commits, moves to next sprint.

### C. Sprint creation
If all currently-created M2 sprints are cleanly closed but `check_m2_done` still fails on candidate coverage, manager opens the next candidate sprint per the candidate order:

**Recommended order** (least → most ramp-up cost; deviate if scout findings change priorities):

- **S1**: `m2-scout-blackwell-toolchain` — research-only sprint surveying RTX 5090 (cc120) support across all six candidate families. Output: `artifacts/m2/scout/toolchain_support_matrix.json` + per-candidate go/no-go + install/build commands. **Worker may use codex `xhigh` for this — it's research-heavy.** Tester (Claude Opus) verifies citations are current and install commands actually work.
- **S2**: `m2-jax-stencil-column` — JAX implementation of both bakeoff problems.
- **S3**: `m2-cupy-stencil-column` — CuPy raw CUDA (or Numba CUDA) implementation. Often the easiest first non-DSL candidate.
- **S4**: `m2-triton-column` — Triton, column problem primarily (Triton's strength is register-heavy compute kernels).
- **S5**: `m2-cuda-tile-stencil-column` — explicit CUDA C++ tile-resident implementation.
- **S6**: `m2-kokkos-stencil-column` — Kokkos C++ implementation.
- **S7**: `m2-gt4py-stencil` — GT4Py/DaCe, stencil problem primarily.
- **S8**: `m2-adr-001-backend-selection` — manager-owned decision sprint. Manager drafts ADR-001 proposing one backend with cited evidence, opens a Codex critical-review on the proposal, applies findings or records dissent, finalizes ADR.

If a candidate's scout finding (S1) says "no Blackwell support and no remediation path," the manager skips that candidate's per-candidate sprint and writes a one-line candidate-failure artifact directly.

**Sprint structure is not frozen.** If two candidates can share an implementation sprint (e.g. CuPy + Numba both Python low-level), bundle them.

### D. M2 closeout
When `check_m2_done.py` returns `ok: true`:
1. Write `.agent/decisions/MILESTONE-M2-CLOSEOUT.md` per §E in M2-DONE.md.
2. Edit `.agent/milestones/M2-backend-bakeoff.md` → Reviewer Decision: `Accepted (YYYY-MM-DD)` + pointer to closeout.
3. Commit `feat(m2): close milestone — backend bakeoff complete, ADR-001 merged`.
4. Re-run `check_m2_done.py`. Confirm ok.
5. Stop the loop. Write a top-level user status report.

## Universal per-turn rules

- **Never start M3 work.** This loop is M2-only.
- **Never modify governance files** (constitution, scope, spec, principles, validation strategy, precision policy, performance targets, risk register edits >5 lines, plan §1-§10, this runbook, the goal file).
- **Always commit your turn's work** before yielding.
- **Use `dispatch_role.sh`** for all agent work. Tester gets `--reasoning xhigh` (Claude Opus 4.7). Worker and reviewer get `--reasoning high` (codex). Critical-review uses default reasoning.
- **One step per turn.** If you find yourself wanting to do three things, pick the highest-value one and let `ScheduleWakeup` re-fire you.
- **If a turn fails to produce forward progress** (no commit, no agent dispatch), write a one-line note to `logs/manager-stall.log` and try a different decision-tree branch on the next turn.
- **If two consecutive turns make no forward progress**, write `BLOCKER-m2-stall-${timestamp}.md` and stop the loop.

## When to ask Codex for a critical review instead of acting

- **ADR-001** is the required example: manager drafts the decision, runs `dispatch_role.sh critical-review .agent/decisions/REVIEW-codex-ADR-001/` against the proposal, applies findings or records dissent.
- Any "this candidate failed in a surprising way, do I exclude it from ADR-001 or escalate?" decision → spawn a critical-review.
- Do **not** stop the loop for routine candidate-selection sub-decisions; manager owns those.

## End-of-loop sanity

When stopping for any reason:
- Commit all pending changes.
- Write a one-page user status report.
- Print the report so the user sees it on attach.
- **Do NOT call `ScheduleWakeup`.** Omitting the call ends the self-paced /loop.

## Send-keys completion mechanism (unchanged from M1)

Agent's tmux window auto-types a short report line into the manager's tmux window with 5-second visible pause then Enter. The manager treats it as a notification (not source of truth) and reads the role's report file from disk for actual content.
