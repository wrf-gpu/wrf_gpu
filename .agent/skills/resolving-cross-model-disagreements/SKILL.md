---
name: resolving-cross-model-disagreements
description: Structures bounded debate between agents or models for architecture, tolerance, and backend decisions.
---

## When to use

Use when credible agents disagree on backend, precision, validation tolerance, or architecture.

## Inputs required

Decision statement, evidence, options, constraints, current ADR if any, and dissenting reports.

## Workflow

1. State the decision in one paragraph.
2. Let each side critique assumptions.
3. Run one response round.
4. Manager writes decision memo with dissent and required proof.

## Hard rules

- Debate is limited to two rounds unless human approves more.
- Evidence outranks model confidence.
- Decision requires proof object or explicit experiment plan.

## Deliverables

Decision dialogue and manager decision memo.

## Validation

Accepted decision has an ADR, experiment, or human escalation.

## Common failure modes

Endless debate, ungrounded preference, and hiding dissent.
