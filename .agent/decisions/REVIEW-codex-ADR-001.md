# Cross-Model Review Pointer — ADR-001 Backend Selection

Date: 2026-05-19
Subject of review: `.agent/decisions/ADR-001-backend-selection.md`
Reviewer model: Codex `gpt-5.5` `xhigh`
Reviewer process: per `.agent/rules/cross-model-review-policy.md`, manager spawned the critical-review via `bash scripts/dispatch_role.sh critical-review .agent/decisions/REVIEW-codex-ADR-001/ --reasoning xhigh`

## Files

- **Proposal** (input to reviewer): [`REVIEW-codex-ADR-001/proposal.md`](REVIEW-codex-ADR-001/proposal.md) — manager's first-draft ADR-001.
- **Critical review** (output from reviewer): [`REVIEW-codex-ADR-001/critical-review.md`](REVIEW-codex-ADR-001/critical-review.md) — Codex's findings.
- **Revised ADR** (post-application): [`ADR-001-backend-selection.md`](ADR-001-backend-selection.md) — manager's final version after applying findings.

## Codex's Decision

`Accept with required fixes` — JAX as primary v0 backend is endorsed; the proposal as originally written is rejected for merge due to three boundary issues that the revision fixes.

## Findings and manager response

| # | Severity | Codex finding (short) | Manager response | Where applied |
|---|---|---|---|---|
| 1 | blocker | Irreversible approval framed as manager-exercised, not human-approved | Accept | ADR-001 `Status:` line now reads "proposed, pending user acknowledgement at M2 closeout"; user has explicit veto |
| 2 | blocker | GT4Py candidate not in oracle-required form | Accept | Created `artifacts/m2/gt4py/{stencil_failure.json, column_failure.json, maintainability.md, agent_success.json}` |
| 3 | major | Triton fallback "no new ADR required" too broad | Accept | Fallback now per-scheme gated: mini-ADR + cross-model review required before any single scheme moves to Triton |
| 4 | major | Profile fidelity is fallback-derived, not full ncu | Accept | ADR-001 `Evidence summary` explicitly labels metrics as "fallback-profiled and micro-fixture-limited"; M3 follow-up to obtain real ncu when perfmon is unlocked |
| 5 | major | M5 stop/go proof object missing | Accept | ADR-001 § "M5 stop/go gate" added with binding thresholds (local_memory ≤256, regs ≤128, launches ≤10) |
| 6 | minor | Pointer file + review-local sprint contract missing | Accept | This file is the pointer (#6 satisfied) |

## Dissent recorded

Codex explicitly *did not dissent* from JAX as the v0 primary. Codex *did* dissent from merging the proposal as written. The revised ADR addresses every dissent; manager records no counter-dissent.

## Outcome

Revised ADR-001 ready for binding-judgment reviewer (codex `gpt-5.5` `high` via `dispatch_role.sh reviewer`).
