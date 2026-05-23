# Sprint Contract — M6.x S3-narrow: Stabilizer Source Cleanup

## Objective

The S2 stabilizer-provenance scanner found **28 experiment-backed (unsourced) stabilizers** in the operator vs only **8 source-backed** and 0 reject. That's a 3.5× imbalance — the operator depends on many more unsourced numerical aids than sourced ones. This is the architectural root cause that the strategy critic + Opus diagnostic + warm-bubble gate critic all pointed to.

This sprint **does NOT replace the load-bearing mu_continuity_increment** (that needs the real Gen2 baseline from S2.1 to test against). Instead, it's a **bounded source-mining cleanup**: for each of the 28 experiment-backed stabilizers, the worker must:
1. Look it up in the S1-built `.agent/decisions/source_mining_operator_table.md`
2. Either replace with a source-cited equivalent (preferred)
3. Or document why it stays experiment-backed (acceptable if a documented reason exists)
4. Or remove it if not load-bearing (acceptable if no regression)

Net result: experiment-backed count should DROP. Either replaced with source-backed, or formally documented in `acoustic_wrf.py` docstrings.

## Non-Goals

- **No removal of `_mu_continuity_increment` in this sprint.** It's load-bearing; S2.1 baseline is needed first.
- No new physics or new stabilization.
- No carry expansion. `AcousticScanCarry` stays 6-leaf.
- No Newton outer.
- No modification of R7 oracle, MPAS slice oracle, operator-sanity gate, or their tests.
- No d02 replay execution (S2.1 does that).
- No Tier-3 or Tier-4 work.
- No remote push.
- No host/device transfer regression.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s3narrow` on branch `worker/gpt/m6x-s3narrow-stabilizer-source-cleanup`.

Write-only on this branch:
- `src/gpuwrf/dynamics/acoustic_wrf.py` — surgical cleanups. Each change documented inline with a WRF/MPAS/Pace/ICON4Py source citation OR a docstring note explaining why a remaining magic stays.
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — if it has experiment-backed stabilizers, same treatment
- `src/gpuwrf/dynamics/damping.py` — same
- `src/gpuwrf/numerics/tridiagonal_solver.py` (if exists) — same
- `tests/test_m6x_s3narrow_stabilizer_audit.py` (new) — assert the new experiment-backed count is **strictly less** than the S2 baseline of 28
- `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/` — proofs + worker-report

Read-only everywhere else.

## Inputs

Required reading:
- **`.agent/decisions/source_mining_operator_table.md`** (S1) — your canonical reference
- **`.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_stabilizer_provenance_scanner.json`** — the 28-vs-8-vs-0 count + per-stabilizer detail
- **`.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/s3_input_memo.md`** — top-3 recommendations
- `scripts/diagnostic_stabilizer_provenance_scanner.py` — the scanner you must beat
- `src/gpuwrf/dynamics/acoustic_wrf.py` — the file you're cleaning
- WRF source `module_small_step_em.F:619-1597`, MPAS `mpas_atm_time_integration.F:1589-2495` for canonical forms

## Acceptance Criteria

### 1. Stabilizer count reduction

Run `python scripts/diagnostic_stabilizer_provenance_scanner.py --input src/gpuwrf/dynamics/ --output proof_stabilizer_after.json` AFTER your changes. Capture results. Required:
- New experiment-backed count: **strictly less than 28** (the S2 baseline)
- New source-backed count: **strictly greater than 8**
- Reject count: 0
- For each formerly-experiment-backed stabilizer that survives, its docstring or comment must cite an explicit reason ("inherited from slice oracle; remove in S5 if no impact" / "preserved pending S2.1 baseline confirmation" / etc.)

### 2. No regression

All prior tests still PASS (45+ tests). Includes warm-bubble operator-sanity gate — the new verdict must NOT regress from FAIL_PHYSICAL_BOUNDS to something worse. Same-or-better.

### 3. Specific cleanups targeted (highest priority from S2 memo)

From `s3_input_memo.md` top-3:
- **Priority 1 (DEFER)**: `_mu_continuity_increment` — NOT touched in this sprint. Add a docstring note explicitly stating "DEFER to post-S2.1 sprint pending real Gen2 baseline".
- **Priority 2 (DO)**: `MPAS_OMEGA_TO_W_METRIC = 1.35` — replace constant with per-column-per-level metric derived from MPAS `mu_d · ∂z/∂η / g` (or `zz` geometry from `mpas_atm_time_integration.F:2491-2495`). Cite the line.
- **Priority 3 (DO)**: `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38` — demote to slice-only (the `mpas_column_slice.py` oracle uses it; production code shouldn't). Cite the demotion in a comment.

Plus any of the other 25 experiment-backed stabilizers the worker chooses to clean up (judgment call — don't try to do all 28 in one sprint).

### 4. New test asserting the count reduction

`tests/test_m6x_s3narrow_stabilizer_audit.py`:
- `test_experiment_backed_count_below_s2_baseline` — invokes the scanner on `src/gpuwrf/dynamics/` and asserts count < 28
- `test_source_backed_count_above_s2_baseline` — asserts source-backed count > 8
- `test_reject_count_is_zero`
- `test_mu_continuity_increment_remains` — confirms it's still in the code (this sprint does NOT remove it)

### 5. Worker report

`worker-report.md` with:
- Per-stabilizer table: what was cleaned, source citation, before/after status
- The new scanner numbers (e.g., "experiment-backed 28 → 17, source-backed 8 → 19")
- Files changed
- Commands run, exit codes
- Risks
- Handoff to S2.1 / S4

### 6. Branch commits on `worker/gpt/m6x-s3narrow-stabilizer-source-cleanup`

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s3narrow
python scripts/diagnostic_stabilizer_provenance_scanner.py \
  --input src/gpuwrf/dynamics/ \
  --output .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.json \
  | tee .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_stabilizer_after.txt
pytest tests/test_m6x_s3narrow_stabilizer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_audit_test.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/proof_no_regression.txt
```

## Performance Metrics

- Vertical operator launch count: report. Should not regress significantly from current ~67.
- Zero host/device transfers in timestep loop (binding).

## Proof Object

- `proof_stabilizer_after.json` + `.txt`
- `proof_audit_test.txt`
- `proof_no_regression.txt`
- `worker-report.md`
- Branch `worker/gpt/m6x-s3narrow-stabilizer-source-cleanup`

Time budget: **6-10 hours**.

## Risks

- **Cleanup may surface dependencies**: removing one stabilizer may break another. If so, document and back off — don't cascade.
- **Per-level metric for MPAS_OMEGA_TO_W_METRIC**: the slice oracle uses 1.35 as a column-constant; switching to per-level may break the slice test. If so, keep the constant inside the slice oracle module ONLY, and use per-level in the production operator. The contract permits this.
- **The 0.38 buoyancy demotion**: if the warm-bubble harness produces strictly worse output (theta blowup grows from current FAIL state), report immediately — that means production buoyancy was depending on it.
- **Spec-gaming**: don't game the scanner by renaming variables. The scanner uses pattern matching; renamed magic numbers won't pass.

## Handoff Requirements

When all proof files are on disk, audit test passes, no-regression passes, worker-report committed: `/exit`. Wrapper sends AGENT REPORT.

## Failure modes the manager will reject

- Removing `_mu_continuity_increment` (out of scope this sprint).
- Renaming magic numbers to avoid the scanner.
- Skipping source citations for replaced stabilizers.
- Modifying R7 oracle, MPAS slice oracle, or their tests.
- Skipping the scanner-count assertion test.
- Host transfer regression.
