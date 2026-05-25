# Sprint Contract — M6B5: Full Dycore Step Parity (physics off, boundary off, 10 steps, sanitizer off)

## Objective

Sixth rung of B-direct ladder. M6B4 proved the acoustic substep + loop composition. M6B5 wraps that loop in WRF's RK3 outer (`solve_em.F` outer loop), runs **10 full timesteps**, and demonstrates full-dycore parity against WRF with physics and lateral boundary application **disabled**. This isolates the pure dycore (advection + acoustic + scratch update) from the rest of the model.

Sanitizer-OFF acceptance. This is the first sprint where multi-RK-stage + multi-timestep composition is tested — if M6B5 passes, M6B6 (physics on, boundary on) is the only remaining parity rung before M6b honest 1h.

## Non-Goals

- NO RMSE tuning.
- NO modifications to operational `wrf.exe`. Pre/post sha256 enforced (1ec3815...).
- NO physics or boundary application (those are M6B6).
- NO 1h forecast.
- NO modifications to operator-level helpers (locked from M6B0-R/B1/B2/B3).
- NO modifications to `acoustic_loop.py` (locked from M6B4).
- NO operational-mode wire-in.
- NO PCR / alternative solver (M6-perf-design territory).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b5` on branch `worker/gpt/m6b5-full-dycore-step-parity`.

Write-only:
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` — add `sp_dycore_step_complete` (full prognostic + scratch state at end of one timestep, all RK stages done)
- `external/wrf_savepoint_patch/solve_em.F.patch` — extend with hook at end of timestep loop iteration (patches must remain RC=0 dry-run)
- `scripts/m6b5_dycore_step_compare.py` (uses `comparator_common.py`)
- `src/gpuwrf/dynamics/dycore_step.py` (NEW, **VALIDATION-ONLY**) — composes acoustic_loop + RK3 outer using WRF-shaped semantics; DO NOT wire into operational runtime
- `src/gpuwrf/validation/savepoint_schema.py` — extend with timestep-boundary kind; bump SCHEMA_VERSION to `m6b5-savepoint-v6` (add to SUPPORTED tuple)
- `src/gpuwrf/validation/tolerance_ladder.json` — add per-timestep tolerances (slightly laxer than per-substep due to compounding; document geometric-growth bound across 10 steps)
- `tests/test_m6b5_dycore_step_parity.py`
- `.agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/` — proofs + worker-report

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/worker-report.md` (the composition pattern)
3. `.agent/sprints/2026-05-25-m6b-ladder-hygiene-cleanup/worker-report.md` (patches RC=0, comparator_common.py)
4. `external/wrf_savepoint_patch/HOOK_INVENTORY.md` (current 28 hooks)
5. `external/wrf_savepoint_patch/solve_em.F.patch` (RC=0 baseline)
6. `src/gpuwrf/validation/comparator_common.py`
7. `src/gpuwrf/dynamics/acoustic_loop.py` (M6B4 deliverable)
8. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F` (the RK3 outer + timestep main loop)
9. `PROJECT_PLAN.md §14.5.1` (Amendment #1 classification mandatory)

## Acceptance Criteria

### Stage 1 — Wrapper extension + rebuild (MANDATORY)

Add `sp_dycore_step_complete(istep, mu, mut, mudf, muts, muave, ww, theta, ph_tend, u, v, w, ph, p, t_2ave)`. Pre/post operational sha256 check. Patches RC=0.

### Stage 2 — Synthetic dry-run extension (MANDATORY)

### Stage 3 — Real WRF dycore-step extraction (MANDATORY)

Run rebuilt shim on:
- Tier-1 column / Tier-2 16×16 / Tier-3 golden small-domain
- 10 full timesteps (each with 3 RK stages × N acoustic substeps)
- Physics and boundary application **disabled** in the WRF namelist (`mp_physics=0`, `bl_pbl_physics=0`, `ra_lw_physics=0`, `ra_sw_physics=0`, `cu_physics=0`, `sf_sfclay_physics=0`, `sf_surface_physics=0`, `specified=.false.`)

### Stage 4 — First REAL JAX-vs-WRF dycore-step parity (MANDATORY)

`src/gpuwrf/dynamics/dycore_step.py` composes `acoustic_loop` + RK3 outer (WRF source cited).

For each tier + each timestep:
- Per-field max-abs delta vs ladder tolerance
- Sanitizer-OFF
- Document accumulated error growth over 10 steps

Outcome: `SIXTH-DYCORE-STEP-PARITY-ACHIEVED` or `PARITY-DEFECT-LOCALIZED-AT-STEP-N`.

### Stage 5 — Kill gate (Amendment #5)

>15 fields diverge at step 1 → STOP, escalate.

### Stage 6 — Operational-compatibility section (MANDATORY, Amendment #1)

| Item | Classification | Evidence |
|---|---|---|
| `sp_dycore_step_complete` hook | Validation-only | savepoint emission |
| `dycore_step.py` callable | Validation-only | NOT wired into operational runtime |
| New ladder entries (per-timestep tolerances) | Validation-only | tolerance values |
| Schema v6 extension | Validation-only | additive |

### Stage 7 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py -v
```

### Stage 8 — Worker report

`worker-report.md`: stages, per-tier per-timestep parity, error-growth bound, operational-compat table, kill-gate, files, handoff to M6B6 (physics on, boundary on).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b5
bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/proof_build_rebuild.txt
patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/solve_em.F.patch 2>&1 | tee .agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/proof_patch_dryrun.txt
patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/module_small_step_em.F.patch 2>&1 | tee -a .agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/proof_patch_dryrun.txt
python scripts/m6b5_dycore_step_compare.py --synthetic-dryrun
python scripts/m6b5_dycore_step_compare.py --tier column --steps 10
python scripts/m6b5_dycore_step_compare.py --tier patch16 --steps 10
python scripts/m6b5_dycore_step_compare.py --tier golden --steps 10
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/proof_no_regression.txt
```

## Performance Metrics

N/A — correctness sprint.

## Kill Gates

- >15 fields diverge at step 1 → STOP, escalate.
- Patch RC ≠ 0 → STOP, repair.
- Operational sha changes → STOP, revert.
- Operational-approved classification without Tier-4 citation → REJECT.

## Risks

- 10-step error growth may exceed tolerance even with per-substep parity. Pick tolerance from the geometric-growth bound (~M6B4-substep-tol × 10 substeps × 3 RK × 10 steps = 300×); document.
- Physics+boundary disabled means lateral-boundary mass leakage is a concern; document expected behavior (small drift in domain-mean MU).
- M6B3 scratch CALL-site insertions still queued (hook-ABI sprint); use the Python-extracted reference path.

## Handoff Requirements

When all proofs + worker-report committed: `/exit`. Manager dispatches M6B6 (physics on, boundary on, sanitizer off, 10 steps).

## Failure modes the manager will reject

- Patch breakage.
- Skipping Amendment #1 classification.
- Modifying operator-level helpers or acoustic_loop.py.
- Multi-timestep with physics or boundary on (that's M6B6).
- Post-sanitize finiteness as acceptance.
