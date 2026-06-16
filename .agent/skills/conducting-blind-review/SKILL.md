---
name: conducting-blind-review
description: Guides independent review of sprint work against contracts, tests, artifacts, and project rules.
---

## When to use

Use when reviewing worker output, architecture proposals, memory patches, or performance claims.

## Inputs required

Sprint contract, diff, reports, validation logs, artifacts, and relevant rules.

## Workflow

1. Read contract before worker summary.
2. Check files changed against ownership.
3. Verify proof objects.
4. List findings by severity.
5. Decide accept, request changes, or reject.

## Hard rules

- Findings lead the report.
- Do not accept claims without artifacts.
- Do not fix the implementation while reviewing.

## Deliverables

Reviewer report with findings, evidence, and decision.

## Validation

Review is valid when every acceptance criterion is explicitly pass, fail, or blocked.

## Common failure modes

Trusting worker confidence, missing hidden scope changes, and reviewing style before correctness.
