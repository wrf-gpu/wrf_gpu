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

| Family | Primary Worker (new sprint impl) | Bug-fix Parallel-Pair | Tester | Reviewer | Critical-Review | Side-Opinion / Tools / Sidecar |
|---|---|---|---|---|---|---|
| **Codex (gpt-5.5 xhigh)** — frontrunner | **yes (default)** | yes | yes | yes | yes (primary for memory/skill patches) | yes |
| **Claude (Opus 4.7 / Sonnet 4.6)** | yes | yes | yes | **yes (primary reviewer)** | yes | yes |
| **Gemini 3.5 high-flash (`agy`)** | no (sole); yes when paired in bug-fix parallel-pair | **yes — mandatory in every parallel-pair** | no (sole); side-runner only | no (sole); side-runner only, and parallel side-runner for large/complex reviews | no (sole); supplementary only | **yes — unconstrained, preferred for speed** |

**Bug-fix parallel-pair rule (mandatory per user directive 2026-05-20)**: every confirmed issue dispatches ≥2 AIs in parallel to identify and propose a fix. One of the two MUST be Gemini (speed advantage). The other is codex or Claude (depth, hallucination check). Manager combines candidates and picks the best (or merges). Without the pair, a single Gemini fix could ship a hallucinated coefficient; with the pair, hallucination risk goes to ~zero.

**Large/complex reviews — Gemini parallel side-runner**: when a review is non-trivial (e.g. milestone closeout, large ADR, contested sprint acceptance), dispatch Gemini in parallel with the primary reviewer (Claude Opus 4.7). Gemini's report is supplementary; the primary reviewer's verdict is binding.

## Hard rules

- No implementation without a sprint contract.
- No first implementation sprint in a milestone without reviewed milestone plan.
- No done claim without proof object.
- No scope expansion without approval.
- **No sole-AI binding decision when Gemini is involved.** Gemini's verdict is always one input among ≥2 other AI opinions. See `.agent/references/dispatching-gemini.md`.
- When dispatching a Gemini side-runner: ALWAYS use the canonical pattern — tmux new-window + `agy --dangerously-skip-permissions -i` (interactive REPL, not `-p` print) + the onboarding prefix from `.agent/references/gemini-onboarding-prompt.md` prepended to the task prompt + pipe-pane logging + completion-handler teardown. Inline `-p` is reserved for throwaway pings only. See `.agent/references/dispatching-gemini.md` Pattern A. Tee output to a named file alongside other agent reports, save the prompt next to it, name the tmux window `gemini-<role>-<sprint>` so janitor distinguishes AI families.
- Cap on parallelism: at most 2 Gemini side-runners simultaneously for a single decision point (1× Claude opus + 1× codex + 1-2× Gemini is the project ceiling).

## Deliverables

Milestone plan, sprint contract, assignments, closeout, merge recommendation, memory patch.

## Validation

Run `python scripts/close_sprint.py <sprint-folder>` at closeout.

## Common failure modes

Overbroad scope, missing file ownership, weak acceptance criteria, accepting claims without artifacts, and treating a fast-AI side-runner as substitute for a binding deep review.
