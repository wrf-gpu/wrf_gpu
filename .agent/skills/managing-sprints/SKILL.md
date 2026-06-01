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

- **Opus worker/frontrunner:** in-process `Agent` tool (`subagent_type: general-purpose`, `model: opus`). Shares the box FS+GPU; manager reviews diff, runs gates, commits/merges; auto-notifies on completion.
- **GPT-5.5 critic/debugger (codex):** launch as an **INTERACTIVE codex TUI session in a tmux window** so the principal can attach (`tmux attach`, Ctrl-b <n>) and watch/interject — principal directive 2026-06-01. `tmux new-window -t <session>:<idx> -n gpt-<role>` then `tmux send-keys '... && codex --full-auto -m gpt-5.5 -c model_reasoning_effort=xhigh "$(cat /tmp/<prompt>.txt)"' Enter`. NOT headless `codex exec >log`. The TUI isn't file-logged, so still instruct the agent to write its deliverable to an absolute main-repo path + print a DONE marker; detect via that file + `tmux capture-pane`. `--dangerously-bypass-...` is classifier-blocked → use `--full-auto`. See memory [[Launch all agents in the same tmux session, close their windows when done]].
- **Liveness before re-dispatch:** an in-process Agent's transcript can LAG (look stale 30–90 min) while alive — verify death by PID (`ps -p <pid>`) + no child GPU procs, or you spawn a duplicate that races the branch + GPU.
- **Long GPU runs:** detach (`systemd-run --user --scope` / `nohup setsid`) + commit each proof immediately. The box hibernates; a CUDA context does NOT survive suspend → kill+rerun anything that spanned one. Kill orphan model GPU procs before each launch.
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
