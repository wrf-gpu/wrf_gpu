# Sprint Contract — Speed vs Bitwise Critic (codex GPT-5.5)

## Objective

Per principal directive 2026-05-24 (night): the project's GPU value proposition can be killed by a savepoint-first approach that ships CPU-shape semantics if discipline lapses. Manager has hardened `PROJECT_PLAN.md §14.5.1` with binding validation-mode/operational-mode invariants and pre-drafted M6-perf-design as the bridge sprint. **This critic sprint stress-tests the discipline.**

The question: *Is the two-mode separation (validation-mode bitwise WRF parity + operational-mode GPU-optimized strict-subset) sufficient to preserve max speed on RTX 5090, or does the savepoint-first per-operator-parity framing systemically cap speed regardless of discipline?*

## Non-Goals

- NO code edits.
- NO ADR promotion.
- NO sub-sprint dispatch.
- NO re-opening of the M6 architectural choice (B-direct stays).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_speedcritic` on branch `critic/codex/m6-speed-vs-bitwise-critic`.

Write-only:
- `.agent/sprints/2026-05-24-m6-speed-vs-bitwise-critic/reviewer-report.md`

Read-only everywhere else.

## Inputs

1. `PROJECT_PLAN.md §14.5 + §14.5.1 + §14.5.2` (the hardened invariants — your binding subject)
2. `PROJECT_CONSTITUTION.md` (GPU-resident, no H2D/D2H in timestep loop)
3. `.agent/decisions/ADR-001-backend-selection.md` (JAX-primary; gated Triton)
4. `.agent/decisions/ADR-007-precision-policy.md` (per-field precision authorization matrix)
5. `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-PROPOSED.md`
6. `.agent/sprints/2026-05-25-m6-perf-design/sprint-contract.md` (the bridge sprint)
7. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/worker-report.md`
8. `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/worker-report.md`
9. `feedback_gpu_optimized_core_primacy.md`, `feedback_validation_philosophy.md` (memory)

## Acceptance Criteria

`reviewer-report.md` with **6 labelled sections** (2500-4500 words, file:line citations):

### §1 — Is the two-mode separation sufficient?

For each of these failure modes, judge whether §14.5.1's invariants are tight enough:
- **Carry creep** — operational mode silently inheriting validation-mode scratch fields
- **Operator-boundary fusion lock** — savepoint boundaries becoming runtime sync points
- **Precision lock** — fp64 propagating from validation into operational because no one wrote the per-field downcast justification
- **Solver lock** — serial Thomas becoming "the way" because PCR breaks bitwise parity
- **State layout lock** — AoS / per-field array layout forced by savepoint serialization

For each: is §14.5.1 explicit enough, or does it leave room for the failure to slip through? Recommend specific tightening if any.

### §2 — Are the M6-perf-design acceptance gates the right ones?

The M6-perf-design sprint requires: (a) Tier-4 envelope on T2/U10/V10, (b) ≥1.2× wall-clock vs 28-rank CPU WRF, (c) zero H2D/D2H. Are these the right gates? Specifically:
- Is 1.2× the right speedup threshold? (Original target: 8-12× per deepthink; GPT-5.5 brief cited 5.5× ICON; project pre-commits no number)
- Should KE spectrum slope, conservation residuals, or other Tier-2 invariants also be gated?
- Should the gate include peak-memory ≤ 1km headroom check now (rather than at M7)?

### §3 — Is the perf-design sprint scoped correctly?

The contract names 6+ operators for the per-operator carry/precision/fusion/solver table. Is this the right granularity? Should the table be coarser (per-RK-stage) or finer (per-savepoint-boundary)?

### §4 — Failure mode the project most likely won't catch in time

Of the failure modes in §1, which is the project MOST likely to ship to production unnoticed? Specifically: which one will pass M6c (24h Gen2 consistency) but tank operational speed at scale (M7 daily ops)?

### §5 — Two-mode vs single-mode-with-feature-flag

Steelman the alternative: instead of separate validation/operational entry points, ship ONE entry point with feature flags (`savepoint_emit=False`, `fp64_strict=False`, `kernel_fusion=True`, etc.). Compare:
- Code duplication / bug-surface
- Risk of validation-mode bugs reaching operational
- Engineering velocity

Recommend two-mode-distinct (current plan) or single-mode-flagged. Cite reasoning.

### §6 — Recommendation

ONE of:
- `RATIFY-§14.5-AS-WRITTEN`
- `RATIFY-WITH-AMENDMENTS` (list bounded amendments to §14.5.1, §14.5.2, or M6-perf-design contract; ≤6 amendments)
- `DISSENT` (specific alternative; argue why current §14.5 caps speed regardless of discipline)

Plus one paragraph dissent against your own recommendation.

## Validation Commands

None — read-only critic.

## Performance Metrics

N/A.

## Proof Object

- `reviewer-report.md` (2500-4500 words)
- Branch `critic/codex/m6-speed-vs-bitwise-critic`

Time budget: **60–120 min**.

## Risks

- Spec-gaming: every claim cites file:line OR proof JSON path.
- Inventing speed numbers without backing: reject. Cite actual published numbers (deepthink brief, GPT-5.5 brief, MeteoSwiss ICON, SCREAM, NeuralGCM) or "no public number; project should measure".

## Handoff Requirements

Commit + `/exit`. Manager reads `reviewer-report.md` and folds amendments into §14.5 or M6-perf-design contract before M6B6 closes.
