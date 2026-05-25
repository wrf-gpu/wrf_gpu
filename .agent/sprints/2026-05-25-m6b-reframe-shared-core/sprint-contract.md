# Sprint Contract — M6b Reframe: Shared `dynamics/core/` + Amendment #1 Supersession

## Objective

Per reframe-critic verdict `REFRAME-TO-SHARED-CORE` (commit `critic/codex/m6b-operational-discipline-reframe` `b9ea231`): the "operational composes own variants" rule (Critic Amendment #1) is replaced with a narrower rule. Validation and operational both import a shared **pure numerical core**; validation wrappers (savepoint emission, HDF5 layout, fp64 strictness) stay validation-only; operational wrappers (carry pruning, fusion, precision, kernel selection, segmentation) stay operational-only.

This collapses the 7 interface mismatches the critic enumerated (carry, RK schedule, coefficient cadence, boundary lead time, physics sequence, thermo offsets, precision) into ONE composition contract.

## Non-Goals

- NO PCR / batched-Thomas / cuSPARSE solver changes.
- NO precision downcast expansion beyond what's already in ADR-007.
- NO 1h forecast (step-1 + 10s + warmed D2H probe only).
- NO M6c, M6 close claim.
- NO modifications to operational `wrf.exe`. Pre/post sha256 (`1ec3815...`).
- NO new physics or new operators.
- NO sanitizer / clamps.
- NO remote push.
- NO removal of validation-only diagnostic instrumentation (savepoint emitter, HDF5 schema, comparator infrastructure).

## File Ownership

Work in worktree `/tmp/wrf_gpu2_reframe` on branch `worker/gpt/m6b-reframe-shared-core`.

Write-only:
- `src/gpuwrf/dynamics/core/` (NEW directory)
  - `__init__.py`
  - `acoustic.py` — `AcousticCoreState`, `advance_mu_t_core`, `w_solve_core`, `acoustic_substep_core`, `acoustic_scan_core` (move math from validation; do NOT rewrite)
  - `dycore.py` — `rk_stage_core`, `dycore_timestep_core` (RK schedule as data, not code)
  - `coupled.py` — `coupled_timestep_core` (dycore + physics + boundary composition)
- `src/gpuwrf/dynamics/validation_wrappers.py` (NEW or rename `acoustic_loop.py`+`dycore_step.py`+`coupled_step.py`)
  - Wraps core; emits savepoint dicts; fp64 strict; sanitizer-friendly debug modes
  - Validation-only
- `src/gpuwrf/runtime/operational_mode.py` — refactor to call core directly
- `src/gpuwrf/runtime/operational_state.py` — adjust if core needs different state shape (minimize change)
- `PROJECT_PLAN.md §14.5.1` — supersession note for Amendment #1
- `.agent/decisions/ADR-028-shared-dynamics-core-DRAFT.md` (NEW) — the architectural reframe ADR
- `tests/test_m6b_shared_core_*.py` (NEW) — new core tests + parity tests
- `.agent/sprints/2026-05-25-m6b-reframe-shared-core/` — proofs + worker-report

Read-only:
- `src/gpuwrf/dynamics/coupled_step.py` + `acoustic_loop.py` + `dycore_step.py` (the validation reference; MOVE math out, don't rewrite)
- All validation-mode tests must continue to pass without modification

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-operational-discipline-reframe-critic/reviewer-report.md` (the verdict + §4 restructure plan + §5 7-defect list)
3. `src/gpuwrf/dynamics/coupled_step.py` (the bitwise-validated composition — source-of-truth for math)
4. `src/gpuwrf/dynamics/acoustic_loop.py` + `dycore_step.py` (their math also goes into core)
5. `src/gpuwrf/runtime/operational_mode.py` (the current operational composer; 5 defects in)
6. `PROJECT_PLAN.md §14.5.1` (the Amendment #1 that's being superseded)
7. `feedback_gpu_optimized_core_primacy.md` (the principal directive — must still be honored)

## Acceptance Criteria

### Stage 1 — Create `src/gpuwrf/dynamics/core/` (MANDATORY)

Move (don't rewrite) the math from validation's `coupled_step.py` + `dycore_step.py` + `acoustic_loop.py` into core/. Core must be:
- Pure functions (no IO, no Python branching on traced values)
- No savepoint emission, no HDF5 calls
- No sanitizer, no clamps
- Accepts inputs as typed pytrees, returns outputs as typed pytrees
- Same math as validation — verified via Stage 4

### Stage 2 — Convert validation wrappers to call core (MANDATORY)

`validation_wrappers.py` becomes thin: imports core; adds savepoint emission, HDF5 logging, debug instrumentation, fp64 enforcement. Existing test_m6b6_*.py must still PASS with the same 0.0 bitwise verdict.

### Stage 3 — Convert operational mode to call core (MANDATORY)

`operational_mode.py` becomes thin: imports core; sets operational precision (ADR-007), carry pruning hooks, segmentation. NO custom `_wrf_small_step_acoustic` (the 5-defect victim — delete entirely; core's acoustic_substep_core takes over).

### Stage 4 — B6 validation parity (BINDING REGRESSION GATE)

`python scripts/m6b6_coupled_step_compare.py --tier golden` MUST still report `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED` with `max_abs_delta: 0.0`. If any field's delta deviates from 0.0 by even 1 ULP → REJECT.

Capture: `proof_b6_unchanged.txt`.

### Stage 5 — Real-IC step-1 parity (MANDATORY — the empirical test of the reframe)

`scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 1`. Acceptance: max-abs delta on every field < 1e-10 (ideally bitwise — since both operational and validation now import the same core).

The 5 previous defects (advance_mu_t commit, W cadence, dt_sub, ph_tend, ru_m/rv_m/ww_m mass coupling) should be ZERO BY CONSTRUCTION because there is no longer separate composition code to drift.

Capture: `proof_step1_parity_reframed.json`.

### Stage 6 — 10s bounded probe (MANDATORY)

`scripts/m6b_carry_expansion_probe.py --runs 1 --duration-s 10` on real Gen2 IC. Acceptance: bounded theta, no nonfinite.

Capture: `proof_10s_bounded.txt`.

### Stage 7 — Warmed inter-kernel D2H = 0 (MANDATORY per ADR-027)

`scripts/m6b_d2h_warmed_recapture.py`. Acceptance: inter-kernel D2H == 0 (the lift work from `268e38d` must survive the refactor).

### Stage 8 — ADR-028 + Amendment #1 supersession (MANDATORY)

`.agent/decisions/ADR-028-shared-dynamics-core-DRAFT.md`:
- Decision: "Validation and operational both import `dynamics.core`. Validation wrappers may not be imported by operational runtime. The principal directive (pilots wrong if incompatible with GPU-optimized core) still binds via the wrapper boundary: savepoint emission, HDF5 layout, fp64 strictness, snapshot dicts remain validation-only and absent from operational graphs."
- Cite reframe critic §4 verbatim where relevant
- Resolve at PROPOSED: the architectural shape is implementation-validated by this sprint

PROJECT_PLAN.md §14.5.1: amend Amendment #1 with a supersession block: "REVISED 2026-05-25: validation and operational both import `dynamics.core`. Validation wrappers (savepoint emission, HDF5, fp64 strict, snapshots) may not enter operational. Carry pruning, fusion, precision downcast, kernel selection are still operational concerns."

### Stage 9 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_*.py -v
```

All previously-passing tests must continue to pass. New shared-core tests added but no existing test modified semantically.

### Stage 10 — Worker report

`worker-report.md`: per-stage status, module restructure diff summary, the 7 interface mismatches → 0 (per Stage 5 evidence), B6 unchanged proof, files changed, **M6b honest 1h V3 dispatch recommendation** (`READY-FOR-M6b-HONEST-1H-V3` if all gates pass).

## Validation Commands

Standard pattern. The discriminator gates are Stage 4 (B6 0.0 unchanged) and Stage 5 (real-IC step-1 parity); if both pass, the reframe is real.

## Kill Gates

- B6 regression (any non-zero delta) → REJECT, revert.
- Step-1 real-IC parity > 1e-10 → reframe incomplete; named remaining defect; route to follow-up.
- 10s probe nonfinite → ditto.
- D2H regression (inter-kernel D2H > 0) → ditto.
- Operational sha256 changes → STOP.

## Risks

- **Move-don't-rewrite**: the math must be character-for-character identical to what validation already proved. A subtle rewrite bug would break B6.
- The restructure touches many files; merge conflict risk with any future parallel sprint is high. Hold other dispatches until this lands.
- Validation tests may have implicit assumptions about module import paths; some test-file updates may be needed (path-only, not semantic).

## Handoff Requirements

When all 10 stages PASS + ADR-028 DRAFT + Amendment #1 supersession + worker-report committed: `/exit`. Manager dispatches **M6b honest 1h V3** with the reframed operational mode.

Time budget: **60-120 min**. Move-don't-rewrite keeps the surface small.

## Failure modes the manager will reject

- B6 regression.
- Imports of validation wrappers into operational runtime.
- Sanitizer / clamps added.
- PCR / precision changes outside ADR-007 authorization.
- Wholesale rewrite of validated math.
- Skipping ADR-028 / Amendment #1 supersession.
