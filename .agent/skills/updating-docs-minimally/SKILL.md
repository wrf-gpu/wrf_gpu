---
name: updating-docs-minimally
description: Helps agents keep documentation concise, current, and tied to decisions or proof objects.
---

## When to use

Use when updating README files, governance docs, ADRs, sprint reports, or references.

## Inputs required

Document purpose, target reader, changed decision or proof object, and stale content to remove.

## Workflow

1. Identify the default-loaded docs affected.
2. Make the smallest accurate update.
3. Move deep detail into references.
4. Remove duplicate or stale text.
5. Link proof objects where claims are made.

## Hard rules

- Do not add essays to default-loaded docs.
- Do not duplicate rules across many files unless necessary.
- Do not update docs as a substitute for validation.

## Deliverables

Minimal doc patch with clear reason.

## Validation

Docs still point to current contracts, scripts, and proof objects.

## Common failure modes

Token bloat, stale summaries, repeated rules, and burying important constraints.
