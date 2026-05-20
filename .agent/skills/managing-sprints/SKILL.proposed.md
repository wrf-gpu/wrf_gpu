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

## Available AI families and their valid roles

| Family | Worker | Tester | Reviewer | Critical-Review | Side-Opinion / Tie-Break / Test-Tool |
|---|---|---|---|---|---|
| Claude (Opus 4.7 / Sonnet 4.6) | yes | yes | yes | yes | yes |
| Codex (gpt-5.5 xhigh) | yes | yes | yes | yes | yes |
| **Gemini 3.5 high-flash (`agy`)** | **no** | **no (sole)** — only as side-runner alongside Claude/codex tester | **no (sole)** — only side-runner | **no (sole)** — only side-runner | **yes — preferred for speed** |

Reason for Gemini constraints: new to this project, no in-repo track record yet. Promote to wider roles after ≥3 successful side-runner deliveries (manager updates the track-record table at `.agent/references/dispatching-gemini.md` after each delivery).

## Hard rules

- No implementation without a sprint contract.
- No first implementation sprint in a milestone without reviewed milestone plan.
- No done claim without proof object.
- No scope expansion without approval.
- **No sole-AI binding decision when Gemini is involved.** Gemini's verdict is always one input among ≥2 other AI opinions. See `.agent/references/dispatching-gemini.md`.
- When dispatching a Gemini side-runner: tee output to a named file alongside other agent reports, save the prompt next to it, name the tmux window `gemini-<role>-<sprint>` so janitor distinguishes AI families.
- Cap on parallelism: at most 2 Gemini side-runners simultaneously for a single decision point (1× Claude opus + 1× codex + 1-2× Gemini is the project ceiling).

## Deliverables

Milestone plan, sprint contract, assignments, closeout, merge recommendation, memory patch.

## Validation

Run `python scripts/close_sprint.py <sprint-folder>` at closeout.

## Common failure modes

Overbroad scope, missing file ownership, weak acceptance criteria, accepting claims without artifacts, and treating a fast-AI side-runner as substitute for a binding deep review.
