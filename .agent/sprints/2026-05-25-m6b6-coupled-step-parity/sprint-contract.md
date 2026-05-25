# Sprint Contract — M6B6: Coupled Step Parity (physics on, boundary on, sanitizer off, 10 steps)

## Objective

**Final parity rung of B-direct ladder.** M6B5 proved full dycore step parity with physics+boundary OFF (worst delta 1.11e-16 ≈ FP64 ULP across 10 timesteps × 3 RK × 10 acoustic substeps). M6B6 turns physics and lateral boundary application ON and demonstrates the same coupled-step parity. After M6B6, the B-direct ladder is complete and M6-perf-design becomes the gate to M6b honest 1h Canary forecast.

## Non-Goals

- NO RMSE tuning. No new stabilizers.
- NO modifications to operational `wrf.exe`. Pre/post sha256 (1ec3815...).
- NO 1h forecast (M6b's job).
- NO operational-mode wire-in (M6-perf-design's job).
- NO modifications to operator-level helpers or composition helpers (acoustic_loop.py / dycore_step.py / mu_t_advance.py / tridiag_solve.py / small_step_scratch.py / acoustic_wrf.py — all locked).
- NO PCR / alternative solver (deferred).
- NO new physics scheme port — must use already-implemented M5 physics (Thompson, MYNN, RRTMG-LW, RRTMG-SW).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b6` on branch `worker/gpt/m6b6-coupled-step-parity`.

Write-only:
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` — add `sp_coupled_step_complete` (full prognostic + scratch + physics tendency state at end of one timestep with physics+boundary on)
- `external/wrf_savepoint_patch/solve_em.F.patch` — extend (patches must remain RC=0 dry-run)
- `scripts/m6b6_coupled_step_compare.py` (uses `comparator_common.py`)
- `src/gpuwrf/dynamics/coupled_step.py` (NEW, **VALIDATION-ONLY**) — wraps dycore_step + M5 physics calls + lateral boundary application per WRF source ordering; DO NOT wire into operational runtime
- `src/gpuwrf/validation/savepoint_schema.py` — extend with coupled-step-boundary kind; bump SCHEMA_VERSION to `m6b6-savepoint-v7` (add to SUPPORTED tuple)
- `src/gpuwrf/validation/tolerance_ladder.json` — add per-coupled-step tolerances (laxer than dycore-only due to physics tendency variability; document)
- `tests/test_m6b6_coupled_step_parity.py`
- `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/` — proofs + worker-report

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/worker-report.md` (composition pattern; FP64 ULP delta evidence)
3. `external/wrf_savepoint_patch/HOOK_INVENTORY.md`
4. `external/wrf_savepoint_patch/solve_em.F.patch` (RC=0 baseline)
5. `src/gpuwrf/validation/comparator_common.py`
6. `src/gpuwrf/dynamics/dycore_step.py` (M6B5 deliverable, locked)
7. M5 physics callable entry points: `src/gpuwrf/physics/` (Thompson, MYNN, RRTMG-LW, RRTMG-SW)
8. WRF source: `solve_em.F` (physics interleaving with RK3 + boundary application)
9. `PROJECT_PLAN.md §14.5.1` (Amendment #1 mandatory)

## Acceptance Criteria

### Stage 1 — Wrapper extension + rebuild (MANDATORY)

Add `sp_coupled_step_complete(istep, full state + physics tendencies + boundary state)`. Pre/post operational sha256. Patches RC=0.

### Stage 2 — Synthetic dry-run extension (MANDATORY)

### Stage 3 — Real WRF coupled-step extraction (MANDATORY)

Run rebuilt shim on Tier-1/2/3 with **physics ON** (matching M5 ported set: Thompson `mp_physics=8`, MYNN `bl_pbl_physics=5`, RRTMG `ra_lw_physics=4,ra_sw_physics=4`) and **lateral boundary ON** (`specified=.true.` with Gen2 wrfbdy file inputs). 10 full timesteps each tier.

### Stage 4 — First REAL JAX-vs-WRF coupled-step parity (MANDATORY)

`coupled_step.py` composes `dycore_step` + M5 physics + boundary application per WRF source ordering (cite line ranges).

For each tier + each timestep: per-field max-abs delta vs ladder tolerance. Sanitizer-OFF.

Outcome: `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED` or `PARITY-DEFECT-LOCALIZED-AT-{PHYSICS|BOUNDARY}`.

### Stage 5 — Kill gate (Amendment #5)

>15 fields diverge at step 1 → STOP, escalate.

### Stage 6 — Operational-compatibility section (MANDATORY, Amendment #1)

Same classification table as prior sprints. Default Validation-only for new hooks/callables/ladder entries.

### Stage 7 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py -v
```

### Stage 8 — Worker report

`worker-report.md` + handoff to **M6-perf-design** (`.agent/sprints/2026-05-25-m6-perf-design/sprint-contract.md`, already pre-drafted with Stage 1.5 solver bakeoff per PCR scout).

## Validation Commands

Standard pattern (build → patch dryrun → synthetic dryrun → three-tier compare → regression). See M6B5 for exact form.

## Performance Metrics

N/A — correctness sprint.

## Kill Gates

- >15 fields diverge at step 1 → STOP.
- Patch RC ≠ 0 → STOP.
- Operational sha changes → STOP.
- Operational-approved without Tier-4 → REJECT.
- M5 physics callable not available → escalate; M5 closure is a hard prerequisite.

## Risks

- Physics tendency variance is larger than dycore numerics — per-coupled-step tolerance must be laxer (document growth ratio).
- Lateral boundary inflow may carry Gen2 wrfbdy interpolation error; document expected sensitivity at boundary rows.
- M5 physics may have different floating-point order than WRF Fortran physics — accept up to 1e-10 abs delta on physics-tendency fields with rationale (per ADR-007 fp64-strict rows that override defaults).

## Handoff Requirements

When all proofs + worker-report committed: `/exit`. Manager dispatches **M6-perf-design**.

## Failure modes the manager will reject

- Patch breakage.
- Skipping Amendment #1.
- Modifying locked helpers.
- Post-sanitize finiteness as acceptance.
- "Pass" with physics disabled (M6B5 territory).
