# Cross-Model Review Pointer — ADR-002 State Layout

Date: 2026-05-19
Subject: `.agent/decisions/ADR-002-state-layout.md`
Reviewer: Codex `gpt-5.5` `xhigh` via `dispatch_role.sh critical-review`

## Files

- Proposal: [`REVIEW-codex-ADR-002/proposal.md`](REVIEW-codex-ADR-002/proposal.md) — manager-finalized ADR-002 v1.
- Critical-review: [`REVIEW-codex-ADR-002/critical-review.md`](REVIEW-codex-ADR-002/critical-review.md) — Codex's findings (1 blocker, 3 majors, 2 minors).
- Revised ADR (post-application): [`ADR-002-state-layout.md`](ADR-002-state-layout.md).

## Codex Decision

`Accept with required fixes`. Codex did NOT dissent from the technical direction (SoA, C-grid, fp64, halo as call-shape placeholder); dissent was on rhetorical/audit hygiene.

## Findings and disposition

| # | Severity | Finding (short) | Manager response | Where applied |
|---|---|---|---|---|
| 1 | blocker | Status framed as manager-exercised vs explicit human approval | Accept | ADR-002 `Status:` line: "accepted by manager pending explicit user approval at M3 closeout" |
| 2 | major | `canary_3km_template()` records placeholder Canary terrain provenance | Accept | ADR-002 § Staggering: explicit caveat labeling it as idealized M3 template; real provenance required at M7 |
| 3 | major | Halo future-proofing overclaimed | Accept | ADR-002 § Halo packing: reworded to "call-shape placeholder", explicit "NOT a guarantee" that MPI drops in; dedicated halo ADR required at M3.x or M4 early |
| 4 | major | Review sprint-contract missing | Accept | Role prompt at `REVIEW-codex-ADR-002/role-prompts/critical-review.md` documented as the contract for this review |
| 5 | minor | `agent_success.json` stale (sprint_attempt=1, reviewer_rejections=0) | Accept | Regenerated with sprint_attempt=2, reviewer_rejections_before_handoff=1, fix_cycles=1 |
| 6 | minor | HLO evidence only proves theta carry, not all 8 prognostics | Accept | ADR-002 § Audit trail: HLO line rephrased to "API-level residency + theta hot-path exercise (M4 dycore exercises full-field carry)" |

## Outcome

Revised ADR-002 ready for M3 milestone closeout. Manager applies all 6 findings; no counter-dissent. User explicit approval still required at M3 closeout per the constitution (same pattern as ADR-001).
