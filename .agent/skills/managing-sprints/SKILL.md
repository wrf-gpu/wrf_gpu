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
