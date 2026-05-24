# Sprint Contract — M6B2: Tridiagonal Solve Parity (advance_w forward + back sweeps)

## Objective

Third rung of the B-direct ladder. M6B0-R proved `calc_coef_w` parity (FIRST-OPERATOR-PARITY at commit `ac252e8`). M6B1 proved `advance_mu_t` parity (SECOND-OPERATOR-PARITY at `8a0130e`). M6B2 proves the **Thomas vertical-implicit solve** is bitwise-faithful between WRF and JAX `lax.scan` on the same WRF-emitted coefficients.

This isolates the recurrence kernel from coefficient construction (already validated) and prepares M6B3+ for full small-step integration.

## Non-Goals

- NO RMSE tuning. No clamps. No tanh caps. No new stabilizers.
- NO modifications to operational `wrf.exe`. Pre/post sha256 inherited from M6B1.
- NO modifications to other operator code (calc_coef_w/advance_mu_t locked from M6B0-R/M6B1).
- NO multi-operator parity beyond Thomas solve in this sprint.
- NO 1h forecast.
- NO replacement of ADR-023 runtime path (separate sprint).
- NO remote push.
- NO selecting GPU operator path.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b2` on branch `worker/gpt/m6b2-tridiagonal-solve-parity`.

Write-only:
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` — extend with `sp_advance_w_tridiag_fwd_pre/post` + `sp_advance_w_tridiag_back_pre/post`
- `external/wrf_savepoint_patch/solve_em.F.patch` — extend with the 4 new hook call sites around `advance_w` internal Thomas blocks
- `scripts/m6b2_tridiag_solve_compare.py` — JAX-vs-WRF Thomas solve comparator
- `src/gpuwrf/dynamics/tridiag_solve.py` (NEW) — extracted from `acoustic_wrf.py` as a callable; do NOT change runtime semantics — wrap as a callable for the comparator
- `src/gpuwrf/validation/savepoint_schema.py` — add Thomas-stage savepoint kinds; extend tolerance ladder
- `tests/test_m6b2_tridiag_solve_parity.py`
- `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/` — proofs + worker-report

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/worker-report.md`
3. `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/worker-report.md` (the calc_coef_w fix pattern)
4. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/worker-report.md`
5. `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-PROPOSED.md`
6. `PROJECT_PLAN.md §14.5.1` (operational-compatibility invariants — Critic Amendment #1)
7. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:1399-1581` (advance_w internal: rhs_ready :1399; raw W :1477; forward sweep :1533; back sub :1546; Rayleigh :1559; PH final :1581)
8. `src/gpuwrf/validation/tolerance_ladder.json`

## Acceptance Criteria

### Stage 1 — Wrapper extension + rebuild (MANDATORY)

Add `sp_advance_w_tridiag_fwd_pre/post` + `sp_advance_w_tridiag_back_pre/post` hooks (4 new). Capture: tri-diag input (a, alpha, gamma, rhs) at fwd-pre; intermediate state at fwd-post; back-input at back-pre; solved w at back-post.

Rebuild instrumented shim. Pre/post operational sha256 check.

### Stage 2 — Synthetic dry-run extension (MANDATORY)

Extend the synthetic-dryrun script with Thomas-stage fields. Verify fail-closed semantics.

### Stage 3 — Real WRF Thomas-stage extraction (MANDATORY)

Run the rebuilt instrumented shim on:
- Tier-1 column (10 acoustic substeps)
- Tier-2 16×16 patch (10 acoustic substeps)
- Tier-3 golden small-domain (10 acoustic substeps, pinned run-ID inherited from M6B1)

### Stage 4 — First REAL JAX-vs-WRF Thomas solve parity (MANDATORY)

For each tier + each substep:
- Load WRF fwd-pre savepoint (a, alpha, gamma, rhs)
- Run JAX `lax.scan`-based Thomas forward sweep
- Compare against WRF fwd-post savepoint
- Same for back-substitution
- Per-field max-abs delta vs ladder tolerance
- Sanitizer-OFF

If parity NOT achieved: outcome is `PARITY-DEFECT-LOCALIZED-IN-TRIDIAG-{FWD|BACK}` — document; route to M6B2-fix.

### Stage 5 — Kill gate check (Critic Amendment #5)

Count diverging fields across the 3 tiers at substep 1. If >15 → STOP, escalate.

### Stage 6 — Operational-compatibility section (MANDATORY, Critic Amendment #1)

In `worker-report.md`, classify every introduced field / boundary / dtype / solver interface:

| Item | Classification | Evidence |
|---|---|---|
| `sp_advance_w_tridiag_fwd_pre/post` hooks | **Validation-only** | Savepoint emission outside operational timestep loop |
| New `tridiag_solve.py` callable | **Operational-required-with-evidence** if it replaces existing serial Thomas; **Undecided** if PCR/batched-Thomas alternative pending | Cite Tier-4 evidence or mark Undecided |
| Tolerance ladder additions (`tri_a`, `tri_alpha`, etc.) | **Validation-only** | Comparator tolerance, not runtime data |
| `lax.scan` over column for Thomas | **Operational-approved-with-evidence** | Serial recurrence intrinsically scan-shaped; cite WRF source `:1533-1546` |
| (other items) | … | … |

**Undecided items may not enter operational APIs without a follow-up sprint.**

### Stage 7 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py -v
```

### Stage 8 — Worker report

`worker-report.md`: all stages, operational-compatibility classification table, parity result + kill-gate decision, files changed, handoff to M6B3 (scratch-state parity: `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `_save`).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b2
bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_build_rebuild.txt
python scripts/m6b2_tridiag_solve_compare.py --synthetic-dryrun 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_synthetic_dryrun_m6b2.txt
python scripts/m6b2_tridiag_solve_compare.py --tier column --steps 10
python scripts/m6b2_tridiag_solve_compare.py --tier patch16 --steps 10
python scripts/m6b2_tridiag_solve_compare.py --tier golden --steps 10
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_no_regression.txt
```

## Performance Metrics

N/A — correctness sprint.

## Kill Gates

- M6B1 chain not in tree → cannot proceed.
- >15 fields diverge at substep 1 across all tiers → STOP, escalate.
- Operational sha changes → STOP, revert.

## Risks

- Thomas solve internal phases may not have clean entry/exit points; if so, document and either skip Rayleigh/PH-final hooks or split into a finer-grained M6B2.5 sub-sprint.
- The serial nature of Thomas across z is fundamental; do NOT attempt to introduce PCR or batched alternatives in this sprint. Those are M6-perf-design's job.

## Handoff Requirements

When all proofs + worker-report.md committed on branch `worker/gpt/m6b2-tridiagonal-solve-parity`: `/exit`. Manager dispatches M6B3 (scratch state parity).

## Failure modes the manager will reject

- Skipping operational-compatibility section (Critic Amendment #1 violation).
- Tolerance laxer than ladder.
- Multi-operator parity claimed.
- "Pass" with post-sanitize finiteness.
- PCR / batched-Thomas as the parity target (M6-perf-design territory).
- Modifying `acoustic_wrf.py` runtime semantics to chase parity.
