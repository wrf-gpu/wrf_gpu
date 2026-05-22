# Sprint Contract — M6.x c2 Pivot Critical Review (ADR-021 vs ADR-022)

## Objective

The manager's c2-A2 + c2-A2.x bundle review (commit `9bca47c`, 2026-05-22) returned **NEEDS-HYBRID-PIVOT**. Two ADR drafts now exist on main:

- `.agent/decisions/ADR-022-hybrid-vertical-operator-DRAFT.md` — manager's working recommendation: replace WRF's split-explicit vertical operator with a clean JAX IMEX scheme; keep the c2-A2 horizontal PGF + mu continuity verbatim; relax Tier-1 WRF parity for the vertical operator only; Tier-4 RMSE on U10/V10/T2 stays binding.
- `.agent/decisions/ADR-021-wrf-smallstep-vertical-port-DRAFT.md` — opposing alternative: port `advance_w`/`advance_mu_t`/`calc_coef_w` line-for-line, expand `AcousticScanCarry` with seven new WRF small-step scratch field families, hold Tier-1 WRF-savepoint parity as binding.

This sprint is a **critical-review** (codex / gpt-5.5 / xhigh reasoning). Argue the opposing position to the manager's recommendation, then return a verdict.

## Non-Goals

- No code edits. Read-only critical review.
- No new full architecture proposal beyond ADR-021/022 unless §5 of the report returns `RATIFY-NEITHER`.
- No sub-sprints. Single-shot.

## File Ownership

Write-only to this sprint folder (`.agent/sprints/2026-05-23-m6x-c2-pivot-critic/`). Read-only elsewhere.

## Inputs

Required reading (cite line numbers in your report):

- `.agent/decisions/ADR-022-hybrid-vertical-operator-DRAFT.md`
- `.agent/decisions/ADR-021-wrf-smallstep-vertical-port-DRAFT.md`
- `.agent/sprints/2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` (full pivot review)
- `.agent/sprints/2026-05-22-c2-architecture-stepback/worker-report.md` (codex arch step-back §§3-5)
- `.agent/sprints/2026-05-22-c2-methodology-stepback/worker-report.md` (gemini methodology §§4-6)
- `.agent/decisions/ADR-020-c2-dycore-architecture.md` (current ADR governing c2)
- `src/gpuwrf/dynamics/acoustic_wrf.py` (current code; `acoustic_substep_scan`, `_calc_coef_w`, `vertical_acoustic_update`, `uncouple_horizontal_pgf_tendency`)

Optional context:
- `.agent/references/cpu-wrf-baseline.md` (Gen2 backfill source)
- `PRECISION_POLICY.md` and ADR-007 (precision constraints)
- WRF source `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F`

## Acceptance Criteria

`reviewer-report.md` in this folder containing six labelled sections:

1. **§1 Steelman of ADR-021.** Best possible case for the WRF-port path. Address the manager's three ADR-022-rationale points (pivot criteria, validation philosophy, JAX IMEX precedent) point by point with cited WRF source lines and prior art.

2. **§2 Stress-test of ADR-022.** Independent reading of `acoustic_wrf.py` after the c2-A2 merge plus both step-backs. Identify **at least three** specification weaknesses in ADR-022 that the manager underestimated.

3. **§3 Tier-1 vs Tier-4 tradeoff.** Defensibility of "operational RMSE binds, Tier-1 doesn't" given:
   - Gen2 backfill data quality
   - wrfbdy boundary forcing fidelity (does internal numerics deviation propagate to U10/V10/T2 within 24h?)
   - Gemini methodology claim that "deviation from WRF causes immediate gravity-wave blowup at the boundaries"

4. **§4 Cost re-estimation.** Manager said ADR-022 ≈ 2-4 worker-days, ADR-021 ≈ 5-9. Re-estimate both bottoms-up from the arch step-back row data and the M5-S1/S2/S3 Fortran-harness budgets actually spent.

5. **§5 Verdict** (exactly one):
   - `RATIFY-ADR-022` — manager's recommendation stands; list which §2 weaknesses are non-blocking.
   - `RATIFY-ADR-021` — flip to WRF port; list which ADR-022 specification gaps are blocking.
   - `RATIFY-NEITHER` — both flawed enough to require a third option; specify it.
   - `RATIFY-EITHER-WITH-CONDITIONS` — both workable; specify gate conditions.

6. **§6 Open questions** for the manager before the implementation sprint dispatches.

## Validation Commands

None — read-only review. Cite line numbers and file paths for every claim.

## Performance Metrics

N/A — review sprint.

## Proof Object

`reviewer-report.md` itself. Length budget: 3000–6000 words. Time budget: 60–120 minutes.

## Risks

- Skipping the steelman in §1 (going straight to a verdict) defeats the purpose of cross-model second-opinion. Reject the temptation; spend at least 800 words on the steelman before the stress-test.
- Citing claims without WRF line numbers is the M5 "spec-gaming" pattern — every numerical claim must be anchored.

## Handoff Requirements

When `reviewer-report.md` is complete, type `/exit` as a slash command in the CLI. The wrapper watchdog fires `AGENT REPORT [critical-review / m6x-c2-pivot-critic / codex] exit=<ec> report=...` into the manager pane. Do not write `/exit` as text inside the report.
