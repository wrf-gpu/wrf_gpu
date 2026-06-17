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
4. **Optional: dispatch a quick third opinion from Gemini 3.5 (`agy -p`) before writing the decision memo.** See `.agent/references/dispatching-gemini.md`. Gemini is fast (~4x Opus) and cheap, so when a 2-AI deadlock is non-obvious to resolve, paying for a third datapoint usually beats running another debate round.
5. Manager writes decision memo with dissent and required proof. If Gemini was consulted, its verdict appears as one of N opinions, never as the deciding vote.

## Hard rules

- Debate is limited to two rounds unless human approves more.
- Evidence outranks model confidence.
- Decision requires proof object or explicit experiment plan.
- **Gemini is never the sole judge or tiebreaker authority — its opinion is always one input among ≥2 other AI opinions.** Reason: model is new to this project, benchmark performance is not a substitute for in-repo track record. Track record column at `.agent/references/dispatching-gemini.md`.

## Deliverables

Decision dialogue and manager decision memo. If Gemini was consulted, also a path to its raw output saved alongside other agent reports.

## Validation

Accepted decision has an ADR, experiment, or human escalation.

## Common failure modes

Endless debate, ungrounded preference, hiding dissent, and treating a fast-AI opinion as evidence-equivalent to a slow-AI deep critique.
