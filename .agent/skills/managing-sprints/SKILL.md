---
name: managing-sprints
description: Guides a manager agent when creating, assigning, gating, and closing evidence-driven sprints.
---

## When to use

Use when planning or running a sprint, assigning agents, enforcing gates, or closing work.

## Inputs required

Project constitution, current milestone, sprint objective, file ownership, validation commands, and required proof object.

## Workflow

1. Create sprint folder from template.
2. Write a narrow sprint contract.
3. Assign owners and reviewers.
4. Confirm validation and performance gates.
5. Collect worker, tester, and reviewer reports.
6. Close with decision and memory-patch proposal.

## Hard rules

- No implementation without a sprint contract.
- No done claim without proof object.
- No scope expansion without approval.

## Deliverables

Sprint contract, assignments, closeout, merge recommendation, memory patch.

## Validation

Run `python scripts/close_sprint.py <sprint-folder>` at closeout.

## Common failure modes

Overbroad scope, missing file ownership, weak acceptance criteria, and accepting claims without artifacts.
