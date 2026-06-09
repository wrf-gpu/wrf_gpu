---
name: managing-sprints
description: Guides a manager agent when creating, assigning, gating, and closing evidence-driven sprints.
---

## When to use

Use when planning or running a sprint, assigning agents, enforcing gates, or closing work.

## Inputs required

Project constitution, current milestone, milestone plan, sprint objective, file ownership, validation commands, and required proof object.

## Workflow

1. Open a milestone with a manager-written milestone plan.
2. Get the milestone plan reviewed before implementation sprints start.
3. Create sprint folder from template.
4. Write a narrow sprint contract.
5. Assign owners and reviewers.
6. Confirm validation and performance gates.
7. Collect worker, tester, and reviewer reports.
8. Close with decision and memory-patch proposal.

## Agent dispatch mechanics

Match model+effort to task type (principal effort-tiers): **core/correctness-critical** (dycore, physics, coupling, proof generation) → **Opus 4.8 max** in-process `Agent`; **debugging / writing / review / harness** → **Opus 4.8 xhigh** or **GPT-5.5 xhigh**; go **parallel** for independent, file-disjoint work. ONE GPU job at a time (single GPU); fan out only non-GPU work in parallel.

- **Cross-model debug cadence:** for a complex correctness bug, after two focused GPT/debug sprints on the same problem fail to prove a fix or leave the conclusion methodologically uncertain, dispatch one Opus xhigh critic/debugger to challenge the method, hypotheses, evidence chain, performance implications, and candidate bug itself before the manager commits to the next conclusion. This cadence is principal-confirmed after the 2026-06-09 live-nest/base-source critique: Opus use should be more frequent at these proof gaps, but still targeted escalation rather than routine double-agenting.
- **Debug-tooling and wall-clock check:** at every planning step for a hard
  runtime/kernel-level bug, explicitly ask whether the team is using the right
  method and whether the current plan is the fastest rigorous wall-clock path.
  Runtime bugs can become expensive when each hypothesis needs a slow
  reproduction. It is often faster and cheaper to send one worker in parallel or
  serially to prove/refute a hypothesis, build a focused harness, savepoint
  emitter, comparator, schema freezer, or visualization, than to keep narrowing
  the bug by slow full-runtime steps. Treat one agent sprint spent building a
  valuable debug tool as cheap if it reduces the next 5-10 proof loops, lowers
  false-assumption probability, or makes the result more falsifiable. Prefer
  expert-style debugging methods that minimize number of steps to the target:
  isolate state boundaries, freeze schemas, create minimal reproducer/savepoint
  loops, compare exact oracles, and parallelize independent hypothesis tests
  without colliding on GPU or source ownership.
- **Opus worker/frontrunner:** in-process `Agent` tool (`subagent_type: general-purpose`, `model: opus`). **DISPATCH WITH `run_in_background: true` — STAY RECEPTIVE (principal directive 2026-06-01).** A foreground `Agent` call BLOCKS the entire manager turn until it returns; a 35-min foreground diagnosis agent locked out both the principal's messages and a finished GPT critic. Background agents auto-notify on completion via task-notification, so the manager stays free for the principal + other agents in between. **NEVER sit in a blocking `sleep`/poll loop** to watch an agent — dispatch, yield, react to the notification; do at most a single one-shot status check, never a waiting loop. Manager reviews diff, runs gates, commits/merges.
- **tmux hygiene before dispatch:** before launching new tmux agents, close completed/no-longer-needed worker windows from prior sprints so the shared tmux session remains clean. Do not close active workers, the manager pane, or principal-owned panes.
- **GPT-5.5 critic/debugger (codex):** launch as an **INTERACTIVE codex TUI session in a tmux window** so the principal can attach (`tmux attach`, Ctrl-b <n>) and watch/interject — principal directive 2026-06-01. **`tmux new-window -t <session>:<EXPLICIT-FREE-INDEX>`** (e.g. `-t 0:5`) then `tmux send-keys -t 0:5 'codex -s workspace-write -a never -m gpt-5.5 -c model_reasoning_effort=xhigh "$(cat /tmp/<prompt>.txt)"' Enter`. **GOTCHAS (both bit us 2026-06-01):** (1) `tmux new-window -t 0` means "create AT index 0" → fails "index 0 in use" and the follow-up `send-keys -t 0:` misroutes into the MANAGER's own pane — ALWAYS give an explicit free window index. (2) `--full-auto` is REMOVED in the current codex CLI (`error: unexpected argument '--full-auto'`) — use `-s workspace-write -a never` (sandboxed, auto-progress, no prompts, won't trip the manager's Bash classifier) or `--dangerously-bypass-approvals-and-sandbox` (full access; the principal's own pattern, but the dangerous substring may trip the classifier when sent via Bash). NOT headless `codex exec >log`. The TUI isn't file-logged, so instruct the agent to write its deliverable to an absolute main-repo path + print a unique DONE marker (e.g. `GPT <TOPIC> DONE`); detect via that file + `tmux capture-pane`, then `kill-window`. Completion messages to the manager pane must use delayed repeated Enter presses, for example `tmux send-keys -t 0:2 '<DONE MARKER>' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`, because a single Enter can leave text staged in the Codex TUI. See memory [[Launch all agents in the same tmux session, close their windows when done]].
- **Liveness before re-dispatch:** an in-process Agent's transcript can LAG (look stale 30–90 min) while alive — verify death by PID (`ps -p <pid>`) + no child GPU procs, or you spawn a duplicate that races the branch + GPU.
- **Long GPU runs:** detach (`systemd-run --user --scope` / `nohup setsid`) + commit each proof immediately. The box hibernates; a CUDA context does NOT survive suspend → kill+rerun anything that spanned one. Kill orphan model GPU procs before each launch.
- **Worktree isolation caveat (bit us 2026-06-01):** `isolation: "worktree"` may branch from a STALE commit, not the current HEAD — a nesting agent's worktree came up at an M5-era commit missing recent fixes. Tell worktree agents to verify their base (`git log -1`) and `checkout -b <fresh> <current-tip>` if stale (never hard-reset — auto-denied). Worktree agents commit on their own branch; the manager merges after review + after any GPU sibling frees the branch/index (don't merge into a branch another agent is actively committing to).
- **GPU hand-off between agents:** an agent that arms a "GPU-free monitor" to auto-run its gates will GRAB the GPU the instant it drops free — which can be a GAP BETWEEN a sibling's multi-run sequence, not the sibling's true end. Risk: collision/box-crash. The manager must not dispatch a competing GPU job into that window, and should verify GPU sanity when the holding agent's completion notification arrives.
- **Fable/Mythos heavy-problem lane (principal directive 2026-06-09):** Fable
  (Mythos, tmux `0:1`) is a scarce high-end debug resource. Conserve its tokens:
  do not use it for routine polling, proof grooming, simple instrumentation,
  standard validation triage, or issues likely solvable by one focused GPT 5.5
  sprint. For validation failures, first send GPT 5.5 workers to collect,
  localize, and attempt direct fixes when feasible. Escalate only the unresolved
  hard core to Fable/Mythos, and frame it as one whole endpoint-defined
  assignment, not narrow micro-prompts. The manager remains manager: write the
  contract, freeze file/GPU locks, require proof objects, review the diff, run
  gates, merge or reject, and continue the milestone. Before sending each new
  Fable/Mythos sprint after a completion or context risk, first send `/compact`
  to `tmux 0:1`, wait about two minutes for the TUI to finish compaction and
  return to a prompt, then send the full assignment and press Enter. Use delayed
  repeated Enter presses when needed because the TUI can leave text staged.
- Manager stays manager: re-dispatch dead agents; don't hand-debug.

## Long-roadmap drift prevention

For long, correctness-critical roadmaps such as v0.14, prevent manager drift with
a periodic Opus 4.8 xhigh **management review**:

- Dispatch one Opus 4.8 xhigh management reviewer after every 15 closed sprints
  on the active milestone, and sooner if the roadmap direction changes, the
  proof chain becomes hard to summarize, or the manager considers changing the
  milestone goal.
- The reviewer reads only the compact current handoff/roadmap, the last 15
  sprint closeouts/reviews/proof summaries, and the manager's current
  conclusions. Do not ask it to reread broad source trees unless it identifies a
  specific gap.
- The review goal is drift control, not pair programming: challenge whether the
  manager is still taking the most efficient, highest-leverage, validated path
  to the milestone goal; identify waste, stale assumptions, missing gates,
  under-parallelization, unsafe parallelization, over-narrow or over-broad
  sprints, and whether the current debugging tools are the right ones for the
  problem.
- The reviewer must ask top-level whether more runtime chasing is still the
  cheapest path. If a focused debug tool, savepoint/comparison harness, schema,
  or visualization would make the next proof loop faster and more reliable, the
  reviewer should recommend that tooling sprint explicitly. The reviewer should
  also challenge whether a parallel or serial worker could cheaply prove/refute
  a key hypothesis while the main lane continues, and whether the plan matches
  expert kernel/runtime debugging practice rather than incremental log-chasing.
- Output must be context-sparing: maximum one short verdict paragraph, one
  ranked table of at most eight findings, one "next 3 sprints" recommendation
  list, and one explicit yes/no on whether the current goal should change.
- v0.14 goal changes are not allowed merely because the path is hard or long.
  A v0.14 goal change is allowed only if Opus 4.8 xhigh explicitly agrees in a
  management review that the current goal is technically impossible or no longer
  the smartest useful target under the latest evidence, and the manager records
  the evidence-backed replacement goal in the roadmap.
- Re-anchor major decisions to the project goal: build a WRF-faithful-enough,
  GPU-optimized, near compute- and memory-optimal, scalable GPU rewrite, not a
  station-score workaround or CPU-WRF wrapper.

Reusable Opus management-review prompt:

```text
You are Opus 4.8 xhigh, independent management reviewer for wrf_gpu2 v0.14.
Goal: prevent roadmap drift. The project goal is a WRF-faithful-enough,
GPU-optimized, near compute- and memory-optimal, scalable GPU rewrite.

Read only:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md
- .agent/decisions/V0140-VALIDATION-PLAN.md
- the last 15 sprint folders' sprint-contract.md, manager-closeout.md,
  memory-patch.md, and linked proof/review summaries

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
actually the fastest rigorous path. Do not propose a goal change unless the
current goal is technically impossible or clearly no longer the smartest useful
target under the latest evidence.

Output exactly:
1. Verdict paragraph, max 120 words.
2. Ranked findings table, max 8 rows: severity, issue, evidence, fix.
3. Next 3 sprints, max 3 bullets, each with objective and proof gate.
4. Goal-change gate: "NO_GOAL_CHANGE" or "GOAL_CHANGE_RECOMMENDED: <why>".
5. Method/tooling verdict: "RIGHT_TOOLS_FASTEST_WALL_CLOCK" or
   "CHANGE_METHOD: <tool/worker/hypothesis path and why>".
6. Context-sparing handoff: max 10 bullets the manager should remember.
```

## Hard rules

- No implementation without a sprint contract.
- No first implementation sprint in a milestone without reviewed milestone plan.
- No done claim without proof object.
- No scope expansion without approval.

## Deliverables

Milestone plan, sprint contract, assignments, closeout, merge recommendation, memory patch.

## Validation

Run `python scripts/close_sprint.py <sprint-folder>` at closeout.

## Common failure modes

Overbroad scope, missing file ownership, weak acceptance criteria, and accepting claims without artifacts.
