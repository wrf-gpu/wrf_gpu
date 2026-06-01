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

- **Opus worker/frontrunner:** in-process `Agent` tool (`subagent_type: general-purpose`, `model: opus`). **DISPATCH WITH `run_in_background: true` — STAY RECEPTIVE (principal directive 2026-06-01).** A foreground `Agent` call BLOCKS the entire manager turn until it returns; a 35-min foreground diagnosis agent locked out both the principal's messages and a finished GPT critic. Background agents auto-notify on completion via task-notification, so the manager stays free for the principal + other agents in between. **NEVER sit in a blocking `sleep`/poll loop** to watch an agent — dispatch, yield, react to the notification; do at most a single one-shot status check, never a waiting loop. Manager reviews diff, runs gates, commits/merges.
- **GPT-5.5 critic/debugger (codex):** launch as an **INTERACTIVE codex TUI session in a tmux window** so the principal can attach (`tmux attach`, Ctrl-b <n>) and watch/interject — principal directive 2026-06-01. **`tmux new-window -t <session>:<EXPLICIT-FREE-INDEX>`** (e.g. `-t 0:5`) then `tmux send-keys -t 0:5 'codex -s workspace-write -a never -m gpt-5.5 -c model_reasoning_effort=xhigh "$(cat /tmp/<prompt>.txt)"' Enter`. **GOTCHAS (both bit us 2026-06-01):** (1) `tmux new-window -t 0` means "create AT index 0" → fails "index 0 in use" and the follow-up `send-keys -t 0:` misroutes into the MANAGER's own pane — ALWAYS give an explicit free window index. (2) `--full-auto` is REMOVED in the current codex CLI (`error: unexpected argument '--full-auto'`) — use `-s workspace-write -a never` (sandboxed, auto-progress, no prompts, won't trip the manager's Bash classifier) or `--dangerously-bypass-approvals-and-sandbox` (full access; the principal's own pattern, but the dangerous substring may trip the classifier when sent via Bash). NOT headless `codex exec >log`. The TUI isn't file-logged, so instruct the agent to write its deliverable to an absolute main-repo path + print a unique DONE marker (e.g. `GPT <TOPIC> DONE`); detect via that file + `tmux capture-pane`, then `kill-window`. See memory [[Launch all agents in the same tmux session, close their windows when done]].
- **Liveness before re-dispatch:** an in-process Agent's transcript can LAG (look stale 30–90 min) while alive — verify death by PID (`ps -p <pid>`) + no child GPU procs, or you spawn a duplicate that races the branch + GPU.
- **Long GPU runs:** detach (`systemd-run --user --scope` / `nohup setsid`) + commit each proof immediately. The box hibernates; a CUDA context does NOT survive suspend → kill+rerun anything that spanned one. Kill orphan model GPU procs before each launch.
- **Worktree isolation caveat (bit us 2026-06-01):** `isolation: "worktree"` may branch from a STALE commit, not the current HEAD — a nesting agent's worktree came up at an M5-era commit missing recent fixes. Tell worktree agents to verify their base (`git log -1`) and `checkout -b <fresh> <current-tip>` if stale (never hard-reset — auto-denied). Worktree agents commit on their own branch; the manager merges after review + after any GPU sibling frees the branch/index (don't merge into a branch another agent is actively committing to).
- **GPU hand-off between agents:** an agent that arms a "GPU-free monitor" to auto-run its gates will GRAB the GPU the instant it drops free — which can be a GAP BETWEEN a sibling's multi-run sequence, not the sibling's true end. Risk: collision/box-crash. The manager must not dispatch a competing GPU job into that window, and should verify GPU sanity when the holding agent's completion notification arrives.
- Manager stays manager: re-dispatch dead agents; don't hand-debug.

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
