# Sprint Contract — M6B0 Plan Critic (post-consultation second opinion)

## Objective

The post-consultation plan rewrite (commit `9527c84`) is a major pivot: B-direct savepoint-first, ADR-023 superseded, ADR-024 accepted, ADR-025 drafted, M6 split into M6a/b/c, A-probe skipped, 10–17 sprint estimate. Per principal's standing order *"get second opinions on core plan re-writes by gpt"*, this critic sprint runs a full-state codex GPT-5.5 critique on the post-consultation plan and the M6B0 sprint contract **before** M6B0's evidence is irreversible.

Read-only critic — no code changes, no sub-sprint dispatch, no ADR promotion. The output is a `reviewer-report.md` that either ratifies, dissents, or proposes specific bounded amendments to the M6B0 sprint contract and the §14 PROJECT_PLAN text.

## Non-Goals

- No code edits anywhere.
- No promotion of any ADR.
- No re-opening of the option A/B/C/D choice (the principal has approved B-direct).
- No remote push.
- No proposing a 6th option.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_plancritic` on branch `critic/codex/m6b0-plan-critic`.

Write-only:
- `.agent/sprints/2026-05-24-m6b0-plan-critic/reviewer-report.md` (deliverable, 2000–4000 words)

Read-only everywhere else.

## Inputs (mandatory)

1. `.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md` — the directive being critiqued
2. `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md` — the ADR
3. `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/sprint-contract.md` — the sprint the M6B0 worker is executing
4. `PROJECT_PLAN.md §14` — the new execution plan section
5. `MILESTONES.md § M6a/b/c` — the new gate split
6. `.agent/decisions/blockers/M6-DYCORE-BLOCKER-MEMO.md` — context
7. `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/findings_real.md` — the catastrophic baseline
8. `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md` — why we are here
9. `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/option_comparison.md` — prior option ranking

## Acceptance Criteria

`reviewer-report.md` with **7 sections**:

### §1 — Diagnosis confirmation
Does the consultant's diagnosis ("project lacks operator-by-operator instrumentation, not WRF un-portability") hold against the S2.1-redo proofs and the S3-hunt A/B toggles? Or is there a counter-reading consistent with the evidence?

### §2 — B-direct vs A-probe trade
Is skipping A-probe (the manager addendum's prior pick) the right call given the consultant's argument that partial-scratch could trap the project in hidden-staging whack-a-mole? Steelman the case for running A-probe as a 1-sprint disposable in parallel with M6B0.

### §3 — M6B0 sprint contract review
Is the M6B0 contract sufficient to demonstrate the savepoint harness works? Specifically:
- Are the 8 stages well-ordered, or do they front-load too much (e.g., should the perturbation negative test land before the bundle extraction)?
- Are the operator boundaries listed in Stage 1 the right minimum set, or are critical ones missing (e.g., divergence damping, lateral boundary application, microphysics coupling at the savepoint level)?
- Is the Tier-1/Tier-2 storage progression sane, or should the worker push directly to a small d02 sub-domain?
- Are the per-field tolerance ladder hooks adequate to expose real errors?

### §4 — ADR-025 draft review
Is the ADR-025 DRAFT shape correct? What's missing? Specifically: should the ADR commit to a file format now (HDF5 vs NetCDF vs Serialbox vs custom) or defer to M6B0's evidence? Should it pre-commit to fp64-only savepoints or allow mixed precision per operator class?

### §5 — Schedule realism
Is the 10–17 sprint estimate to M6 close realistic, or optimistic by ≥2×? What are the top 3 schedule risks that could push this to 25+ sprints? Recommend specific scoping reductions if 25+ is the honest read.

### §6 — Option E (shadow GPU-WRF) priority
Should the AceCAST + FahrenheitResearch/wrf-gpu-port scout sprint run in parallel with M6B0 (as currently dispatched), or be deferred until M6B5 produces evidence? Argue both sides; pick one.

### §7 — Recommendation
ONE of:
- `RATIFY-POST-CONSULTATION-PLAN` — proceed with M6B0 as written.
- `RATIFY-WITH-AMENDMENTS` — list bounded amendments (≤5) to the M6B0 contract or §14 plan. Specify file:line for each.
- `DISSENT` — explain why the plan is wrong; propose the specific alternative; explain why your alternative beats both the consultant's recommendation and the manager addendum.

Plus one paragraph of dissent against your own recommendation (the steelman of the alternative).

## Validation Commands

None — read-only critic.

## Performance Metrics

N/A.

## Proof Object

- `reviewer-report.md` (2000–4000 words, with file:line citations for every claim)
- Branch `critic/codex/m6b0-plan-critic`

Time budget: **60–120 min**.

## Risks

- Spec-gaming: every claim cites file:line OR proof JSON path.
- Re-opening the A/B/C/D vote: hard reject. The principal decided.
- Inventing a 6th option: reject unless it is genuinely orthogonal to A/B/C/D/E.

## Handoff Requirements

Commit + `/exit`. Manager reads `reviewer-report.md` from the file.
