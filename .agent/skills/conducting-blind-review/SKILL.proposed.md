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
- **Sole-reviewer prohibition: Gemini 3.5 (`agy`) is never the sole reviewer for a sprint, ADR, milestone closeout, or memory/skill patch.** It MAY be dispatched in parallel with a Claude- or codex-class reviewer as a cheap side-runner; its findings then appear as a supplementary section in the manager's decision memo. See `.agent/references/dispatching-gemini.md` for the dispatch pattern.
- A binding review report (the one used to accept or reject) must be authored by Claude (Opus 4.7 / Sonnet 4.6) or codex (gpt-5.5). Gemini's report is supplementary.

## Deliverables

Reviewer report with findings, evidence, and decision. If a Gemini side-runner was used, also its raw output saved next to the binding review report.

## Validation

Review is valid when every acceptance criterion is explicitly pass, fail, or blocked.

## Common failure modes

Trusting worker confidence, missing hidden scope changes, reviewing style before correctness, and treating a fast supplementary opinion as substitute for a deep binding review.
