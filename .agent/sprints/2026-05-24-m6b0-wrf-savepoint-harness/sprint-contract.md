# Sprint Contract — M6B0: WRF Small-Step Savepoint Harness + JAX Comparator (B-direct, savepoint-first)

## Objective

Build the **CPU WRF `module_small_step_em` savepoint extractor** and the **JAX-side savepoint comparator** that together unlock the B-direct bottom-up port of the WRF small-step. This is the decisive sprint of the post-blocker M6 close per `.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md` and ADR-025-DRAFT.

The goal is **not** to fix the dycore. The goal is to build the *instrumentation* that makes the dycore fixable: an oracle pipeline that turns "first nonfinite at step 2" into "operator X differs from CPU WRF at savepoint Y in field Z by N units."

## Non-Goals

- **NO RMSE tuning.** This sprint does not touch ADR-023 stabilizers.
- **NO new clamps, dampings, tanh caps, or warm-bubble-specific guards.**
- **NO post-sanitize acceptance.** Sanitizer-bypass is the only valid evidence.
- **NO public performance claim.**
- **NO Option C substrate work.**
- **NO RK3 outer loop port.** Only `module_small_step_em` boundaries.
- **NO modifications to `src/gpuwrf/dynamics/acoustic_wrf.py`** beyond imports needed for the comparator. The acoustic operator itself stays untouched in this sprint.
- **NO remote push.**
- **NO regression of the operational Gen2 CPU WRF build.** The instrumented build MUST live under a separate path and not overwrite `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe`.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b0` on branch `worker/gpt/m6b0-wrf-savepoint-harness`.

Write-only:
- `external/wrf_savepoint_patch/` (NEW dir) — Fortran patches or wrapper layer for CPU WRF instrumentation; build scripts; isolated instrumented `wrf.exe` path
- `scripts/m6b0_wrf_savepoint_extract.py` (NEW) — orchestrator that runs the instrumented CPU WRF on one Canary d02 column / 16×16 patch / golden slice and produces savepoint files
- `scripts/m6b0_jax_savepoint_compare.py` (NEW) — JAX-side comparator: load savepoint, run JAX operator on the same inputs, emit per-operator delta
- `scripts/m6b0_perturbation_negative_test.py` (NEW) — deliberate-perturbation negative test: inject a known error and verify the comparator fails loudly
- `src/gpuwrf/validation/savepoint_io.py` (NEW) — savepoint reader/writer library (HDF5 or NetCDF or NPY bundle; decision recorded in ADR-025 update)
- `src/gpuwrf/validation/savepoint_schema.py` (NEW) — schema definitions: run-ID, WRF version/commit, namelist, domain, map factors, vertical grid, timestep, RK stage, acoustic substep, stagger, units, variable provenance
- `tests/test_m6b0_savepoint_schema.py` (NEW)
- `tests/test_m6b0_perturbation_negative.py` (NEW)
- `tests/test_m6b0_coefficient_parity.py` (NEW) — the first operator parity test
- `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md` — fill in the open questions during the sprint (decided format, decided schema field list, decided tolerance ladder)
- `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/` — proofs + worker-report

Read-only everywhere else.

## Inputs (required reading)

1. **`.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md`** — the authoritative directive for this sprint
2. **`.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md`** — the ADR you are filling in
3. **`.agent/decisions/blockers/M6-DYCORE-BLOCKER-MEMO.md`** — context for why we're here
4. `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md` + `worker-report.md` — why single-operator-bug-hunt failed
5. `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/findings_real.md` — the catastrophic baseline this is replacing
6. `.agent/decisions/source_mining_operator_table.md` — operator term provenance (9 rows, file:line)
7. **CPU WRF source**: `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/` + `module_small_step_em.F` (especially `:619-651`, `:828-868`, `:902-942`, `:1094-1175`, `:1340-1597`). Build env: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`. Compiler: NVIDIA HPC SDK `nvfortran` 26.3.
8. `MILESTONES.md § M6` (split into M6a/M6b/M6c)
9. `PROJECT_PLAN.md § 14` (post-blocker execution plan)
10. `PROJECT_CONSTITUTION.md` (invariants: JAX primary, GPU-resident, no host/device transfer in timestep loop)
11. Gen2 baseline: `/mnt/data/canairy_meteo/runs/wrf_l3/` (d02 3km Canary)

## Acceptance Criteria

### Stage 1: Instrumented CPU WRF build (MANDATORY)

- A separate instrumented `wrf.exe` exists at a non-production path (suggested `/home/enric/src/wrf_gpu2/external/wrf_savepoint_patch/build/wrf.exe.instrumented`). The operational Gen2 `wrf.exe` MUST remain byte-identical (`sha256` proof captured).
- The instrumentation hooks fire at a minimum these operator boundaries:
  1. coefficient construction for the vertical implicit solve (`calc_coef_w` or equivalent)
  2. `MU`, `MUTS`, `MUAVE`, `ww` state at the start and end of each acoustic substep
  3. `t_2ave` running-average update points
  4. `ph_tend` accumulation
  5. `advance_w` entry/exit
  6. pressure/geopotential restoration
  7. one full acoustic substep boundary
  8. one full RK stage boundary
- Each savepoint includes: variable name, dtype, shape, stagger (mass/u/v/w/eta-half/eta-full), units, RK stage index, acoustic substep index, run-ID, WRF commit hash, namelist hash, dt, domain index, map factors, vertical-grid parameters.

Capture proof: `proof_instrumented_build.txt` (build log + sha256 of both wrf.exe files + path).

### Stage 2: Savepoint schema + I/O library (MANDATORY)

- `src/gpuwrf/validation/savepoint_schema.py` defines a `SavepointMetadata` dataclass and a `Savepoint` container with all fields above.
- `src/gpuwrf/validation/savepoint_io.py` provides `write_savepoint(path, savepoint)` and `read_savepoint(path) -> Savepoint`.
- File format: HDF5 (default) or NetCDF or NPY bundle — decided during the sprint and recorded in ADR-025.
- `tests/test_m6b0_savepoint_schema.py` passes: round-trip write→read preserves all metadata; tampered files raise a clear error.

Capture proof: `proof_schema_roundtrip.txt`.

### Stage 3: First savepoint bundle on real Canary d02 (MANDATORY)

Run the instrumented CPU WRF on:
- **Tier 1**: one Canary d02 column (1×1×nz). Capture savepoints across all instrumented operator boundaries for 1, 2, 5, 10 acoustic steps.
- **Tier 2**: one 16×16 Canary d02 patch (extracted from `/mnt/data/canairy_meteo/runs/wrf_l3/`). Same step coverage.
- **Tier 3 (stretch)**: one full Canary d02 domain over 10 acoustic substeps. If storage budget exceeds 20 GB, defer the Tier-3 bundle and document the storage estimate.

Capture proofs:
- `proof_savepoint_bundle_column.txt` (file listing + sizes + metadata summary)
- `proof_savepoint_bundle_patch.txt`
- `proof_savepoint_storage_estimate.txt` (extrapolation to full d02 + recommendation)

### Stage 4: JAX-side comparator with deliberate-perturbation negative test (MANDATORY)

- `scripts/m6b0_jax_savepoint_compare.py` loads a savepoint, runs the JAX operator on the same inputs (with the JAX implementation of the operator under test), and emits per-field delta.
- The comparator must support **at least one operator class** in this sprint: vertical-solve coefficient construction (`calc_coef_w`-equivalent). Other operators are M6B1+ scope.
- **Mandatory deliberate-perturbation negative test** (`scripts/m6b0_perturbation_negative_test.py`): inject a known +1e-6 perturbation into one input field, run the comparator, verify it reports the delta correctly and refuses to declare parity. If the comparator silently passes a perturbed test, the sprint FAILS.

Capture proofs:
- `proof_comparator_parity_clean.txt` — clean run shows parity
- `proof_comparator_perturbation_caught.txt` — perturbation detected; sprint succeeds only if this file exists with delta correctly reported
- `proof_comparator_tolerance_ladder.txt` — per-field tolerance values chosen and rationale

### Stage 5: First operator parity demonstration (MANDATORY)

- Run the comparator on the coefficient-construction savepoint over the Tier-1 column AND the Tier-2 16×16 patch.
- Sanitizer-off. No clamps. No caps.
- Report per-field max-abs delta and document whether parity is achieved within the chosen tolerance ladder.
- If parity is NOT achieved, that is a valid sprint outcome — document the discrepancy and route to M6B1 as a known-defect target. Do NOT attempt to fix the JAX operator in this sprint. Do NOT add stabilizers.

Capture proof: `proof_first_operator_parity.json` (per-field delta, tolerance, pass/fail per field).

### Stage 6: ADR-025 fills in open questions (MANDATORY)

Edit `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md` to resolve at least:
- Savepoint file format (decided)
- Schema field list (frozen)
- WRF Fortran instrumentation strategy (in-tree patch / wrapper / preprocessor — decided)
- Tolerance ladder per operator class (proposed; reviewable)
- Whether a golden-domain precedes the full d02 (decided)

The ADR moves from DRAFT to PROPOSED at the M6B0 reviewer-report stage (out of scope for the worker; reviewer's call).

### Stage 7: No regression (MANDATORY)

```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m6x_tier3_convergence_infra.py tests/test_m3_transfer_audit.py tests/test_m6b0_savepoint_schema.py tests/test_m6b0_perturbation_negative.py tests/test_m6b0_coefficient_parity.py -v | tee .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_no_regression.txt
```

All PASS. New tests added; existing tests unchanged.

### Stage 8: Worker report

`worker-report.md` with: stage-by-stage status, file format chosen, schema list, storage estimate, comparator design notes, first-operator parity result (pass/fail + delta), ADR-025 changes summary, files changed, commands run, transfer audit (must be 0 H2D/D2H inside any timestep loop the JAX side touches), risks, handoff to M6B1.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b0
# Stage 1: build instrumented WRF (use the env script + nvfortran)
bash external/wrf_savepoint_patch/build.sh | tee .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_instrumented_build.txt
sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe external/wrf_savepoint_patch/build/wrf.exe.instrumented >> .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_instrumented_build.txt

# Stage 3: extract savepoints
python scripts/m6b0_wrf_savepoint_extract.py --tier column --steps 10 \
  --output .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/savepoints/column/ \
  | tee .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_savepoint_bundle_column.txt
python scripts/m6b0_wrf_savepoint_extract.py --tier patch16 --steps 10 \
  --output .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/savepoints/patch16/ \
  | tee .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_savepoint_bundle_patch.txt

# Stage 4: comparator + perturbation negative
python scripts/m6b0_perturbation_negative_test.py \
  --savepoint .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/savepoints/column/ \
  | tee .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_comparator_perturbation_caught.txt

# Stage 5: first operator parity
python scripts/m6b0_jax_savepoint_compare.py --operator coefficient_construction \
  --savepoint .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/savepoints/column/ \
  --output .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_first_operator_parity.json \
  | tee .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_comparator_parity_clean.txt

# Stage 7: no regression (full test pass)
pytest <test list above> -v | tee .agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_no_regression.txt
```

## Performance Metrics

- N/A for this sprint. Performance is gated by post-M6c per PROJECT_PLAN §14.5.
- Transfer audit: 0 H2D/D2H bytes inside the JAX timestep loop (the comparator's I/O is OUTSIDE the JAX-compiled region — this must be demonstrated explicitly).

## Proof Object

- `proof_instrumented_build.txt`
- `proof_schema_roundtrip.txt`
- `proof_savepoint_bundle_column.txt`, `proof_savepoint_bundle_patch.txt`, `proof_savepoint_storage_estimate.txt`
- `proof_comparator_parity_clean.txt`, `proof_comparator_perturbation_caught.txt`, `proof_comparator_tolerance_ladder.txt`
- `proof_first_operator_parity.json`
- `proof_no_regression.txt`
- `worker-report.md`
- ADR-025 updated (DRAFT → ready-for-PROPOSED)
- Branch `worker/gpt/m6b0-wrf-savepoint-harness`

Time budget: **8–16 hours of focused work**, fewer if Serialbox is already available in the Canairy WRF build.

## Risks

- **Authorization for WRF Fortran patching**: instrumenting the operational build risks Gen2 disruption. Hard requirement: separate build path, byte-identical operational wrf.exe, sha256 proof.
- **Storage explosion**: full-domain savepoints across 10 steps × 12 operators × all 3D fields could be tens of GB. Mitigation: Tier-1 column + Tier-2 16×16 patch first; document Tier-3 full-domain storage estimate, don't run it blind.
- **Comparator false-positive**: tolerances too loose hide real differences. Mitigation: mandatory deliberate-perturbation negative test.
- **Confabulation**: every claim cites file:line + proof JSON path.
- **CPU budget**: bound to cores 0-3 via `taskset -c 0-3 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`. 28 cores reserved for overnight CPU WRF.
- **Multi-writer worktree race**: this sprint runs in its own worktree `/tmp/wrf_gpu2_m6b0`, no shared file ownership with other sprints.

## Handoff Requirements

When all proof files on disk + ADR-025 updated + worker-report.md committed on branch `worker/gpt/m6b0-wrf-savepoint-harness`: `/exit`. Wrapper sends AGENT REPORT to manager pane (session 2 via patched dispatch wrapper).

## Failure modes the manager will reject

- Modifying `src/gpuwrf/dynamics/acoustic_wrf.py` body (only imports allowed).
- Adding new stabilizers, clamps, dampings.
- Acceptance based on post-sanitize finiteness.
- Skipping the deliberate-perturbation negative test.
- Operational Gen2 wrf.exe modified or path overwritten.
- "Parity demonstrated" without per-field max-abs delta vs tolerance evidence.
- Claiming success on stretch Tier-3 full d02 without column + 16×16 tiers complete.
- Renaming a stabilizer to evade the anti-clamp scanner.
- Multi-suspect changes claimed as "the savepoint harness."
- Skipping no-regression run.
