---
name: writing-execplans
description: Helps agents produce concise execution plans with explicit scope, validation, risks, and rollback.
---

## When to use

Use before difficult implementation, architecture spikes, validation work, or performance experiments.

## Inputs required

Objective, non-goals, prerequisites, allowed files, acceptance criteria, validation commands, metrics, risks, rollback, and review needs.

## Workflow

1. Copy the template.
2. State the smallest useful objective.
3. Define non-goals and file ownership.
4. List pass/fail gates.
5. Add rollback and review requirements.

## Hard rules

- Plans must be testable.
- Do not hide unresolved prerequisites.
- Do not turn a plan into broad project documentation.

## Deliverables

One execution plan in `codex/plans/` or the active sprint folder.

## Validation

Plan is valid when another agent can implement from it without expanding scope.

## Common failure modes

Vague acceptance criteria, missing rollback, missing performance metrics, and unbounded file changes.
