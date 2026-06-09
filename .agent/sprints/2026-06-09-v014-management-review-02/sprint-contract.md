# Sprint Contract: V0.14 Management Review 02

Date: 2026-06-09 18:26 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Run the periodic Opus 4.8 xhigh management review required after 15 closed
v0.14 sprints. Since Management Review 01, 18 sprint closeouts exist. The
reviewer must challenge whether the current grid-parity-first roadmap,
evidence chain, sprint sizing, parallelization, and debug tooling are still the
fastest rigorous path to v0.14.

This is a drift-control review, not a source/debug sprint.

## Inputs

Read only:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`
- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`
- `.agent/reviews/2026-06-09-v014-management-review-01.md`
- the last 15 closed sprint folders' `sprint-contract.md`,
  `manager-closeout.md`, `memory-patch.md`, and linked proof/review summaries

Relevant latest closed sprint:

- `.agent/sprints/2026-06-09-v014-step1-live-nest-theta-qv-wiring`

Current active sprint:

- `.agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization`

## Write Scope

Allowed:

- `.agent/reviews/2026-06-09-v014-management-review-02.md`

Forbidden:

- No `src/` edits.
- No `proofs/` edits.
- No sprint closeout edits.
- No TOST, Switzerland, FP32, memory source work, GPU jobs, or Hermes.
- No edits to files owned by the active P/PH/MU worker.

## Required Review Questions

1. Is the manager still pursuing the correct v0.14 goal: WRF-faithful-enough,
   GPU-optimized, near compute- and memory-optimal, scalable GPU rewrite?
2. Is grid-cell parity still the correct first gate before FP32, memory, and
   TOST?
3. Is the active `P/PH/MU` boundary-localization sprint the fastest rigorous
   wall-clock next step, or should the method change?
4. Are we underusing or overusing parallel workers?
5. Are proof boundaries too fragmented or too broad?
6. Are any validation, Grid-Delta Atlas, memory, Switzerland, or TOST gates
   missing from the roadmap?
7. Should the v0.14 goal change? A goal change may be recommended only if the
   current goal is technically impossible or clearly no longer the smartest
   useful target under current evidence.

## Output Format

Output exactly:

1. Verdict paragraph, max 120 words.
2. Ranked findings table, max 8 rows: severity, issue, evidence, fix.
3. Next 3 sprints, max 3 bullets, each with objective and proof gate.
4. Goal-change gate: `NO_GOAL_CHANGE` or
   `GOAL_CHANGE_RECOMMENDED: <why>`.
5. Method/tooling verdict: `RIGHT_TOOLS_FASTEST_WALL_CLOCK` or
   `CHANGE_METHOD: <tool/worker/hypothesis path and why>`.
6. Context-sparing handoff: max 10 bullets the manager should remember.

## Validation

Manager validates by checking that:

- the review file exists;
- it follows the required output shape;
- it does not edit forbidden files;
- it is incorporated into the roadmap only after manager review.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'OPUS V014 MANAGEMENT_REVIEW_02 DONE - see .agent/reviews/2026-06-09-v014-management-review-02.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
