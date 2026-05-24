# Sprint Contract — M6B1: `advance_mu_t` Parity (B-direct rung 2)

## Objective

M6B0-R proved the first real CPU-WRF↔JAX operator parity for `calc_coef_w` on column / 16×16 / golden small-domain tiers. M6B1 extends the same harness to the **mass + thermodynamic update operator** `advance_mu_t` (canonical WRF: `module_small_step_em.F` `:969-1175`). This is the operator where MU, MUDF, MUTS, MUAVE, ww, and θ are updated together — and was a primary suspect in the S3-hunt bug-hunt that returned `NO-BUG-LOCALIZED`. Per-operator parity here will either expose the recurrence/staging fault or move us toward the next operator.

## Non-Goals

- NO new clamps, dampings, tanh caps.
- NO modifications to operational `wrf.exe`. Pre/post sha256 enforced (inherited from M6B0-R).
- NO modifications to `src/gpuwrf/dynamics/acoustic_wrf.py` body without source-cited evidence.
- NO 1h/24h forecast.
- NO multi-operator parity beyond `advance_mu_t` in this sprint.
- NO further WRF Fortran source modifications beyond extending `dyn_em/savepoint_wrapper.F90` (already in tree from M6B0-R).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b1` on branch `worker/gpt/m6b1-advance-mu-t-parity`.

Write-only:
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` — extend with `sp_advance_mu_t_pre`, `sp_advance_mu_t_post`
- `external/wrf_savepoint_patch/solve_em.F.patch` — extend with the two new hook points around `CALL advance_mu_t(...)`
- `external/wrf_savepoint_patch/build.sh` — re-run, verify operational sha unchanged
- `scripts/m6b1_advance_mu_t_compare.py` (NEW) — JAX-vs-WRF for `advance_mu_t`
- `src/gpuwrf/dynamics/mu_t_advance.py` (NEW or extracted from acoustic_wrf.py — exposed only as a callable for the comparator; do NOT change runtime semantics)
- `tests/test_m6b1_advance_mu_t_parity.py` (NEW)
- `src/gpuwrf/validation/savepoint_schema.py` — add `advance_mu_t` enum value if not yet present; extend tolerance ladder for MU/MUTS/MUAVE/ww/θ (Tier-1 ULP-scale; MUTS accumulation exception per ladder rules)
- `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/` — proofs + worker-report

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/worker-report.md` (M6B0-R deliverable + ADR-025 promotion notes)
3. `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-PROPOSED.md` (the architecture)
4. `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md` (`_mu_continuity_increment` was a suspect)
5. `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/findings_real.md` (catastrophic baseline)
6. `.agent/decisions/source_mining_operator_table.md` (operator term provenance)
7. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:969-1175`
8. `solve_em.F` call sites (from env-audit): `3398/3419/3435` for advance_mu_t/_gpu
9. Tolerance ladder from M6B0-R: `src/gpuwrf/validation/savepoint_schema.py`

## Acceptance Criteria

### Stage 1 — Extend wrapper for `advance_mu_t` boundaries (MANDATORY)

Add to `savepoint_wrapper.F90`:
- `sp_advance_mu_t_pre(rkstage, acstep, mu, mut, mudf, muts, muave, ww_in, theta_in)`
- `sp_advance_mu_t_post(rkstage, acstep, mu, mut, mudf, muts, muave, ww_out, theta_out, ph_tend)`

Rebuild the instrumented WRF. Pre/post operational sha256 check from M6B0-R's `build.sh` (already in tree).

Capture proof: `proof_build_rebuild.txt` + `proof_instrumented_sha256_v2.txt`.

### Stage 2 — Synthetic dry-run extension (MANDATORY per Amendment #1 inheritance)

Extend `scripts/m6b0r_synthetic_dryrun.py` (or a thin M6B1 wrapper) with `advance_mu_t` fields. Verify fail-closed semantics for MU and MUTS perturbations.

Capture proof: `proof_synthetic_dryrun_m6b1.txt`.

### Stage 3 — Real WRF `advance_mu_t` extraction (MANDATORY)

Run the rebuilt `wrf.exe.instrumented` on:
- Tier-1 column (10 acoustic substeps)
- Tier-2 16×16 patch (10 acoustic substeps)
- Tier-3 golden small-domain (10 acoustic substeps, pinned run-ID inherited from M6B0-R `proof_golden_slice_runid.txt`)

Capture proofs: `proof_savepoint_advance_mu_t_column.txt`, `proof_savepoint_advance_mu_t_patch16.txt`, `proof_savepoint_advance_mu_t_golden.txt`.

### Stage 4 — First REAL JAX-vs-WRF `advance_mu_t` parity (MANDATORY)

For each tier and each acoustic substep:
- Load WRF pre-state from `sp_advance_mu_t_pre` savepoint
- Run the JAX `advance_mu_t` equivalent (extracted from `acoustic_wrf.py`'s `_mu_continuity_increment` + associated state updates; do NOT change runtime semantics — wrap as a callable for the comparator)
- Compare against WRF post-state from `sp_advance_mu_t_post`
- Per-field max-abs delta vs tolerance ladder. Per-tier pass/fail.
- Sanitizer-OFF.

If parity is NOT achieved: outcome is `PARITY-DEFECT-LOCALIZED-IN-MU-T` — document field, location, magnitude. **Do NOT fix.** Route to a follow-on M6B1-fix sprint with a single named bug + the savepoint evidence.

Capture proof: `proof_advance_mu_t_parity.json` + `proof_advance_mu_t_parity.txt`.

### Stage 5 — Kill-gate check (CRITICAL, Amendment #5 phase)

After Stage 4: count diverging fields across the 3 tiers at the first acoustic substep.
- If >15 fields diverge at substep 1 → **STOP**. Dispatch external WRF-expert human review request OR M6B-rescope sprint (reduce M6c to pre-M7 gate).
- If ≤15 fields diverge → proceed to M6B2 (tridiagonal solve parity).

Capture: `proof_kill_gate_status.txt`.

### Stage 6 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py -v
```
All PASS.

### Stage 7 — Worker report

`worker-report.md`: stages 1-6, parity result, kill-gate status, files changed, risks, handoff to M6B2 (or to fix sprint if `PARITY-DEFECT-LOCALIZED`).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b1
bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/proof_build_rebuild.txt
python scripts/m6b1_advance_mu_t_compare.py --tier column --steps 10
python scripts/m6b1_advance_mu_t_compare.py --tier patch16 --steps 10
python scripts/m6b1_advance_mu_t_compare.py --tier golden --steps 10
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py -v
```

## Performance Metrics

- WRF rebuild wall-time: report informational
- Comparator wall-time per tier: report informational
- Storage: ≤5 GB Tier-3 golden slice (inherited)

## Kill Gates

- M6B0-R did not actually produce `wrf.exe.instrumented` → cannot proceed; depend on M6B0-R re-do.
- >15 fields diverge at substep 1 across all tiers → STOP, escalate.
- Operational sha changes → STOP, revert.

## Risks

- `advance_mu_t` in WRF has internal sub-phases (running averages, MUTS accumulation). The savepoint may need finer-grained internal checkpoints if parity at the pre/post boundary is opaque. Allow a `proof_advance_mu_t_internal_split_recommendation.md` if the worker discovers this.
- The JAX side may not have a clean `advance_mu_t` equivalent extractable from `acoustic_wrf.py`; in that case the worker writes a faithful reimplementation per WRF source citations (NOT a stabilizer-laden version).

## Handoff Requirements

When all proofs + worker-report.md committed on branch `worker/gpt/m6b1-advance-mu-t-parity`: `/exit`. Manager reads `worker-report.md`, dispatches M6B2 (tridiagonal solve parity) OR M6B1-fix (if `PARITY-DEFECT-LOCALIZED`).

## Failure modes the manager will reject

- Skipping golden small-domain tier.
- Tolerance laxer than the ladder.
- Multi-operator parity.
- "Pass" with post-sanitize finiteness.
- Modifying `acoustic_wrf.py` runtime semantics to chase parity.
