You are Opus 4.8 xhigh, independent management reviewer for wrf_gpu2 v0.14.
Goal: prevent roadmap drift. The project goal is a WRF-faithful-enough,
GPU-optimized, near compute- and memory-optimal, scalable GPU rewrite.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`
Sprint: `.agent/sprints/2026-06-09-v014-management-review-02`

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
- current active sprint:
  `.agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization`

Context:

- Since Management Review 01, 18 v0.14 sprint closeouts exist, so this periodic
  review is due.
- Latest closed sprint:
  `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`; production
  live-nest theta/QV initialization is closed.
- Current residual:
  first divergent schema field `T`; largest residual `P` max_abs
  `974.9820434775493`, RMSE `135.98147360593399`, worst Fortran
  `i=1,j=30,k=1`, boundary band true; `PH/MU/W/U` remain material.
- Active sprint:
  Step-1 `P/PH/MU` boundary/operator localization using focused
  boundary/substage proof, optional narrow source fix only if exact bug is
  proven.
- TOST, Switzerland, FP32 source work, and memory follow-ups remain paused until
  grid-cell parity is explained/reduced.

Critique the manager's current 0.14 roadmap, conclusions, proof chain,
parallelization, sprint sizing, next-sprint plan, and debug tooling. Decide
whether the manager is still on the fastest rigorous wall-clock path to the
goal. At top level, answer whether we are using the right tools and methods:
should the next sprint build a focused harness/savepoint/comparator/schema/
visualization, or dispatch a parallel/serial worker to prove/refute a key
hypothesis, instead of chasing another slow runtime reproduction? Evaluate the
method like an expert kernel/runtime debugger: minimize steps to the target,
minimize false-assumption probability, prefer minimal reproducible proof loops,
freeze schemas/boundaries, and avoid expensive full-run iteration unless it is
actually the fastest rigorous path.

Do not propose a goal change unless the current goal is technically impossible
or clearly no longer the smartest useful target under the latest evidence.

Output exactly to:

`.agent/reviews/2026-06-09-v014-management-review-02.md`

Required output shape:

1. Verdict paragraph, max 120 words.
2. Ranked findings table, max 8 rows: severity, issue, evidence, fix.
3. Next 3 sprints, max 3 bullets, each with objective and proof gate.
4. Goal-change gate: `NO_GOAL_CHANGE` or
   `GOAL_CHANGE_RECOMMENDED: <why>`.
5. Method/tooling verdict: `RIGHT_TOOLS_FASTEST_WALL_CLOCK` or
   `CHANGE_METHOD: <tool/worker/hypothesis path and why>`.
6. Context-sparing handoff: max 10 bullets the manager should remember.

Rules:

- Read-only except `.agent/reviews/2026-06-09-v014-management-review-02.md`.
- No `src/` edits.
- No `proofs/` edits.
- No active-worker file edits.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work, no Hermes.

When done, notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'OPUS V014 MANAGEMENT_REVIEW_02 DONE - see .agent/reviews/2026-06-09-v014-management-review-02.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
