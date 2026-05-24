# Sprint Contract — M6.x Doc Refresh

## Objective

Repo-level documentation is stale (most last touched 2026-05-19) and does not reflect the major M6.x architecture pivots: ADR-023 (PROPOSED conservative column solver), ADR-024 (PROPOSED warm-bubble gate policy), the source-mining table from S1, the stabilizer cleanup from S3-narrow (28→20 experiment-backed, 8→37 source-backed). Update them so a fresh reader can understand the project state without spelunking through 393 commits.

## Non-Goals

- No code changes anywhere.
- No new ADRs.
- No promotion of ADR-023 or ADR-024 from PROPOSED to ACCEPTED (that's a reviewer call).
- No changes to MILESTONES.md gates themselves (the gates are governance; the gate-policy ADR-024 documented the redefinition).
- No remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_docrefresh` on branch `worker/gpt/m6x-doc-refresh`.

Write-only:
- `README.md`
- `PROJECT_PLAN.md`
- `RISK_REGISTER.md`
- `MORNING-REPORT.md`
- `.agent/SPRINT-TRACKER.md`
- `.agent/milestones/ROADMAP.md` (if exists and is stale)
- `.agent/sprints/2026-05-24-m6x-doc-refresh/` — proofs + worker-report

**Read-only**: all other governance files, all `src/`, all `tests/`, all ADRs, all sprint folders other than this one.

## Inputs

Required reading (skim the headers for state changes — full content not needed):
- `.agent/decisions/ADR-023-conservative-column-solver.md` (PROPOSED — current architecture)
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` (PROPOSED — gate change)
- `.agent/decisions/source_mining_operator_table.md` (S1 — operator provenance lock)
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-strategy-critic/reviewer-report.md` (CHANGE-THE-GATE verdict)
- `.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md` (HYBRID plan)
- `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md` (Opus MIXED)
- `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/worker-report.md` (S1 PASS)
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/worker-report.md` (S3-narrow PASS)
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/worker-report.md` (intel: ADR-021 strip = catastrophic)
- `.agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/worker-report.md` (Gen2 noise floor)
- `data/fixtures/gen2_baseline/rmse_summary.csv` (numerical anchors)
- Current `README.md`, `PROJECT_PLAN.md`, `MILESTONES.md`, `RISK_REGISTER.md`, `.agent/SPRINT-TRACKER.md`, `MORNING-REPORT.md` (the stale versions you're updating)

## Acceptance Criteria

### 1. README.md update

- Status section reflects: M0-M5 closed; M6 in active dycore stabilization (S2.2 d02 hang debug + S2.1 baseline + S3-real mu replacement + S4 Tier-3 + S5 Tier-4 path); M7 prologue (S0a) done.
- Add a short paragraph on ADR-023 (conservative column solver, PROPOSED) and ADR-024 (warm-bubble gate is operator-sanity, not amplitude).
- Add a "Where the project actually stands" or "Architecture" subsection: ADR-023 unified path on main; ADR-021 carry-expansion proven non-viable without unphysical clamps; source-mining lock at `.agent/decisions/source_mining_operator_table.md`.
- Do NOT rewrite philosophy/scope/constitution; those remain authoritative.

### 2. PROJECT_PLAN.md update

- Add a §13 (or appropriate section) "Recorded operational decisions since manager handover 2026-05-23":
  - ADR-023 PROPOSED (conservative column solver chosen after 3-way critic + prototype + diagnostic)
  - ADR-024 PROPOSED (warm-bubble amplitude → operator-sanity gate)
  - ADR-021 prototype on branch only — not merged (carry-expansion catastrophic without clamps)
  - Critic-ratified HYBRID plan in execution (S1 done, S2/S2.1 partial, S2.2 in flight, S3-narrow done, S4/S5/S6 queued)
- Keep the rest of the PROJECT_PLAN intact (decisions §11, etc. are still valid).

### 3. RISK_REGISTER.md update

Add new rows:
- d02 replay hang (S2/S2.1 timeout; S2.2 working on fix)
- Honest warm-bubble currently FAIL_PHYSICAL_BOUNDS (mu_pert 86 kPa); pending S3-real after S2.1-redo baseline
- Both architectures (ADR-023 and ADR-021) require sourced stabilization to be honest; neither is "finished"
- M6 close gate is Tier-3 + initial Tier-4 RMSE vs Gen2; warm-bubble is diagnostic only (per ADR-024)
- ~20 experiment-backed stabilizers remaining in operator (S3-narrow cleaned 8 of 28; further cleanup deferred to post-S2.1)

### 4. MORNING-REPORT.md fresh writeup

Single-page summary of where the project actually is now. Must include:
- Milestone ledger M0-M8 with current status
- M6 dissection (which sub-sprints done, which in flight, which queued)
- Critic-ratified HYBRID plan with current sprint position
- Two parallel intel results (ADR-021 strip catastrophic, Gen2 baseline characterized)
- Open questions for the user (≤ 3)
- Manager's current best-estimate time-to-M6-close

### 5. .agent/SPRINT-TRACKER.md current state

Bring it fully current. Should accurately reflect:
- Currently in flight (S2.2 + this sprint + Tier3Prep sibling sprint)
- Recently completed in last 24h (S1, S2 partial, S2.1 partial, S3-narrow PASS, intel sprints, strategy critic, gate critic, etc.)
- Queue (S2.1-redo, S3-real, S4, S5, S6)

### 6. No regression

`pytest -q --collect-only` should still find the same tests (validation that no test files were accidentally touched).

`pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m3_transfer_audit.py -v` — all PASS (49+ tests).

### 7. Worker report

`worker-report.md` documenting: files updated + their key delta + before/after summary.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_docrefresh
pytest --collect-only 2>&1 | tee .agent/sprints/2026-05-24-m6x-doc-refresh/proof_test_collection.txt | tail -5
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-doc-refresh/proof_no_regression.txt
```

## Performance Metrics

None.

## Proof Object

- Updated `README.md`, `PROJECT_PLAN.md`, `RISK_REGISTER.md`, `MORNING-REPORT.md`, `.agent/SPRINT-TRACKER.md`
- `proof_test_collection.txt` (verifies no test files touched)
- `proof_no_regression.txt`
- `worker-report.md`

Time budget: **2-4 hours**. Documentation work; should be fast if you read the inputs carefully.

## Risks

- **Doc drift from over-summarization**: the goal is currency, not rewriting. Keep philosophy/scope/constitution sections intact; update only status/decisions sections.
- **Stale numerical claims**: every numerical claim in docs must cite a current source (proof file, ADR section, sprint report).
- **Spec-gaming**: don't write what we WANT to be true. Write what the artifacts on disk SHOW.

## Handoff Requirements

When all proof files on disk + 5 doc files updated + worker-report committed: `/exit`. Wrapper sends AGENT REPORT to manager pane.

## Failure modes the manager will reject

- Touching any file under `src/`, `tests/`, or other ADRs.
- Updating MILESTONES.md gates (they're governance).
- Promoting ADRs from PROPOSED to ACCEPTED.
- Removing the Risk Register's existing rows (only ADD).
