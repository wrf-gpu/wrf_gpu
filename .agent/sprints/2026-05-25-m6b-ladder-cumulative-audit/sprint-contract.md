# Sprint Contract — M6B Ladder Cumulative Audit (opus reviewer, parallel with M6B3)

## Objective

Three consecutive parity sprints (M6B0-R, M6B1, M6B2) have shipped successively. The cumulative artifacts are:
- `external/wrf_savepoint_patch/savepoint_wrapper.F90` (extended 3×)
- `external/wrf_savepoint_patch/solve_em.F.patch` (extended 3×; merge conflict resolved 1×)
- `src/gpuwrf/validation/savepoint_schema.py` (extended 3×)
- `src/gpuwrf/validation/tolerance_ladder.json` (extended 3×)
- `scripts/m6b*_compare.py` (3 new comparator scripts)

This sprint is an **acceptance audit of the cumulative state of the ladder** while M6B3 runs in parallel. It surfaces drift, duplication, schema inconsistencies, latent compatibility issues, or any anti-pattern that would compound through M6B4/B5/B6.

Read-only audit. Output is a memo. No code edits.

## Non-Goals

- NO code edits, schema changes, or tolerance changes.
- NO disagreement with parity verdicts of prior sprints (those are merged).
- NO speculation about future sprints — focus on what is in tree now.
- NO sub-sprint dispatch.
- DO NOT touch the M6B3 worktree at `/tmp/wrf_gpu2_m6b3`.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_ladderaudit` on branch `tester/opus/m6b-ladder-cumulative-audit`.

Write-only:
- `.agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/audit_memo.md`
- `.agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/proof_*.txt`

Read-only everywhere else.

## Inputs

1. M6B0-R, M6B1, M6B2 worker-reports (in `.agent/sprints/`)
2. `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
3. `external/wrf_savepoint_patch/solve_em.F.patch`
4. `src/gpuwrf/validation/savepoint_schema.py`
5. `src/gpuwrf/validation/tolerance_ladder.json`
6. `scripts/m6b*_compare.py` (all 3 lineage scripts)
7. `src/gpuwrf/dynamics/acoustic_wrf.py`, `tridiag_solve.py`, `mu_t_advance.py` (new helpers from the ladder)
8. `tests/test_m6b0r_*.py`, `tests/test_m6b1_*.py`, `tests/test_m6b2_*.py`
9. `PROJECT_PLAN.md §14.5.1 + §14.5.2` (the operational-mode invariants)
10. `feedback_gpu_optimized_core_primacy.md` (memory)

## Acceptance Criteria

### Part 1 — Cumulative schema integrity

Audit `savepoint_schema.py` + `tolerance_ladder.json` for:
- Field-name collisions across operators
- Tolerance entries that contradict each other (e.g., MUTS appears in both M6B1 and a hypothetical other operator with different tolerance)
- Schema versioning consistency
- Are tolerances at-or-tighter-than `1e-11` per Stage-5 declaration; document any laxer entries

### Part 2 — `solve_em.F.patch` quality

Audit the cumulative patch:
- Does it apply cleanly against the canonical WRF source?
- Are all `#ifdef WRF_SAVEPOINT` blocks properly closed?
- Does any hook accept 0 args (latent bug carried from M6B0-R RELINK)?
- Are CPU vs GPU operator path branches consistently instrumented (CPU only per ADR-025)?

### Part 3 — Helper code drift

Audit `acoustic_wrf.py`, `mu_t_advance.py`, `tridiag_solve.py` for:
- Are the WRF-shaped helpers structured similarly (per-WRF-citation pattern)?
- Any shared math that should be extracted to avoid duplication?
- Are any helpers wired into operational runtime by accident (should be validation-only callables)?

### Part 4 — Comparator script drift

Audit `scripts/m6b*_compare.py` for:
- Common pattern (load savepoint → run JAX op → diff with ladder)
- Any divergent tolerance handling
- Any hardcoded paths that should be parameterized

### Part 5 — Operational-compatibility readiness

Per Critic Amendment #1, every parity sprint should classify fields. Verify:
- M6B0-R worker-report has classification (it predates the amendment; document gap)
- M6B1 worker-report has classification
- M6B2 worker-report has classification
- Aggregate "Undecided" fields list — these are M6-perf-design's input scope

### Part 6 — Verdict memo

`audit_memo.md` answers:
1. Cumulative state quality: GOOD / MIXED / DEGRADING
2. Drift / duplication / quality issues found (list)
3. Required cleanup before M6B4 (if any)
4. GO / WAIT / NO-GO for the M6B3 worker's verdict to be merged (if WAIT or NO-GO: specify what to fix first)

### Part 7 — No regression

`pytest --collect-only` — confirm no test files touched.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_ladderaudit
pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/proof_no_touch.txt
```

## Performance Metrics

N/A.

## Risks

- Audit may surface a latent issue that requires a fix sprint before M6B4. That is the audit's value, not its failure.

## Handoff Requirements

When `audit_memo.md` + proofs committed on branch `tester/opus/m6b-ladder-cumulative-audit`: stop. Manager reads memo and decides whether M6B4 dispatches immediately or after cleanup.

Time budget: 60-120 min.
