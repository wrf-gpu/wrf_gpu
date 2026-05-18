---
name: maintaining-memory
description: Controls how project memory and skills are updated through evidence, review, validation, and hygiene.
---

## When to use

Use at sprint close, milestone closeout, or when recurring lessons should become durable project memory.

## Inputs required

Manager closeout, tester/reviewer lessons, evidence, proposed destination, and reviewer status.

## Workflow

1. Draft memory patch.
2. Validate required fields.
3. Reviewer checks truth, generality, duplication, and usefulness.
4. Apply only approved minimal patch.
5. Run affected skill evals after skill changes.

## Hard rules

- No self-update of stable memory.
- Do not encode one-off failures as global rules.
- Remove or compress stale memory at milestone hygiene.

## Deliverables

Memory patch, validation result, approved stable-memory or skill edit.

## Validation

Run `python scripts/validate_memory_patch.py <patch>`.

## Common failure modes

Memory bloat, duplicated rules, stale lessons, and review-free updates.
