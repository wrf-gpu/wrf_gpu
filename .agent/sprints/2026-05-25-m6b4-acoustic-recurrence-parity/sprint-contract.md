# Sprint Contract — M6B4: Acoustic Recurrence Parity (one acoustic substep → all substeps in one RK stage)

## Objective

Fifth rung of B-direct ladder. M6B0-R/B1/B2/B3 validated per-operator parity (calc_coef_w, advance_mu_t, Thomas solve, scratch state). M6B4 composes those operators into the **full acoustic substep** (one inner loop iteration) and then into the **full acoustic loop** (all substeps within one RK stage). This is where per-operator parity must compose without drift.

The acoustic recurrence is where ADR-023 originally failed (T2 → 136 K). With each piece now validated, the composition is the next decisive test.

## Non-Goals

- NO RMSE tuning. No new stabilizers.
- NO modifications to operational `wrf.exe`. Pre/post sha256 (1ec3815...).
- NO multi-RK-stage coupling — M6B5 (full dycore step) is the next rung.
- NO 1h forecast.
- NO operational-mode wire-in.
- NO solver alternatives (Thomas only; PCR is M6-perf-design).
- NO remote push.
- NO modification of operator-level helpers (acoustic_wrf.py / mu_t_advance.py / tridiag_solve.py / small_step_scratch.py) — they are M6B0-R/B1/B2/B3 deliverables.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b4` on branch `worker/gpt/m6b4-acoustic-recurrence-parity`.

Write-only:
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` — add `sp_acoustic_substep_complete` (per-substep snapshot of full prognostic state) and `sp_acoustic_loop_complete` (end of all substeps in one RK stage)
- `external/wrf_savepoint_patch/solve_em.F.patch` — extend with the 2 new hook call sites (carefully — patch is now valid RC=0; do not break it)
- `external/wrf_savepoint_patch/module_small_step_em.F.patch` — (optional) extend if recurrence boundary is inside `advance_w`/`small_step_finish`
- `scripts/m6b4_acoustic_recurrence_compare.py` (uses `src/gpuwrf/validation/comparator_common.py`)
- `src/gpuwrf/dynamics/acoustic_loop.py` (NEW, **VALIDATION-ONLY**) — composes the validated operators into the acoustic substep + loop using WRF-shaped semantics; DO NOT wire into operational runtime
- `src/gpuwrf/validation/savepoint_schema.py` — extend with new substep/loop boundary kinds; bump SCHEMA_VERSION to `m6b4-savepoint-v5` (add to `SUPPORTED_SCHEMA_VERSIONS` tuple)
- `src/gpuwrf/validation/tolerance_ladder.json` — add per-substep tolerances (slightly laxer than per-operator due to composition error; document rationale)
- `tests/test_m6b4_acoustic_recurrence_parity.py`
- `.agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/` — proofs + worker-report

Read-only everywhere else.

## Inputs (mandatory)

1. `.agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/worker-report.md` (pattern + Amendment #1 classification)
3. `.agent/sprints/2026-05-25-m6b-ladder-hygiene-cleanup/worker-report.md` (HOOK_INVENTORY + patch is now RC=0; honor that)
4. `external/wrf_savepoint_patch/HOOK_INVENTORY.md` (28 hooks current, 0 with non-empty body; this sprint adds 2 more)
5. `external/wrf_savepoint_patch/solve_em.F.patch` (now RC=0; extend carefully)
6. `src/gpuwrf/validation/comparator_common.py` (use the shared utilities)
7. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F` (full small-step), `solve_em.F` (the acoustic substep loop `DO iteration = 1, number_of_small_steps`)
8. `PROJECT_PLAN.md §14.5.1` (Amendment #1: operational-compatibility section MANDATORY)

## Acceptance Criteria

### Stage 1 — Wrapper extension + rebuild (MANDATORY)

Add 2 new hooks:
- `sp_acoustic_substep_complete(rkstage, substep, mu, mut, mudf, muts, muave, ww, theta, ph_tend, u, v, w, ph, p, t_2ave)` — full prognostic + scratch state at end of one acoustic substep
- `sp_acoustic_loop_complete(rkstage, mu, mut, mudf, muts, muave, ww, theta, ph_tend, u, v, w, ph, p, t_2ave)` — same state at end of all substeps in one RK stage

Pre/post operational sha256 check. Patches must remain RC=0 dry-run.

### Stage 2 — Synthetic dry-run extension (MANDATORY)

Verify fail-closed semantics for the new boundary fields.

### Stage 3 — Real WRF acoustic recurrence extraction (MANDATORY)

Run rebuilt shim on:
- Tier-1 column / Tier-2 16×16 / Tier-3 golden small-domain
- 10 acoustic substeps in one RK stage (so 10 `sp_acoustic_substep_complete` + 1 `sp_acoustic_loop_complete` savepoints per RK call)

### Stage 4 — First REAL JAX-vs-WRF acoustic composition parity (MANDATORY)

Build `src/gpuwrf/dynamics/acoustic_loop.py` (validation-only) that:
- Calls the validated `calc_coef_w_wrf_coefficients` once per RK stage
- Loops over acoustic substeps; in each: calls `advance_mu_t_wrf`, `thomas_*_scan`, scratch updates per `small_step_scratch.py`
- Composes them WRF-shaped (cite WRF source for the ordering)

For each tier + each substep:
- Compare JAX `sp_acoustic_substep_complete`-equivalent against WRF savepoint
- Compare JAX `sp_acoustic_loop_complete`-equivalent against WRF savepoint after all substeps
- Per-field max-abs delta vs ladder tolerance (per-substep tolerance allowed slightly looser due to composition error accumulation — document the geometric-growth bound per substep)
- Sanitizer-OFF

Outcome: `FIFTH-OPERATOR-COMPOSITION-PARITY-ACHIEVED` or `PARITY-DEFECT-LOCALIZED-AT-SUBSTEP-N`.

### Stage 5 — Kill gate (Amendment #5)

>15 fields diverge at substep 1 across all tiers → STOP, escalate (this is the COMPOSITION test; if it fails, the per-operator parities don't actually compose cleanly — that's a major architectural finding).

### Stage 6 — Operational-compatibility section (MANDATORY, Amendment #1)

| Item | Classification | Evidence |
|---|---|---|
| `sp_acoustic_substep_complete/loop_complete` hooks | Validation-only | savepoint emission |
| `acoustic_loop.py` callable | Validation-only OR Operational-approved-with-evidence | Cite Tier-4 evidence if Operational; default to Validation-only |
| New ladder entries (per-substep tolerances) | Validation-only | tolerance values are validation tolerances |
| Schema v5 extension | Validation-only | NO operational state API change |

### Stage 7 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py -v
```

### Stage 8 — Worker report

`worker-report.md`: all stages, composition-parity result per tier per substep, operational-compatibility table, kill-gate decision, files changed, handoff to M6B5 (full dycore step).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b4
bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_build_rebuild.txt
patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/solve_em.F.patch 2>&1 | tee .agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_patch_dryrun.txt
patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/module_small_step_em.F.patch 2>&1 | tee -a .agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_patch_dryrun.txt
python scripts/m6b4_acoustic_recurrence_compare.py --synthetic-dryrun
python scripts/m6b4_acoustic_recurrence_compare.py --tier column --substeps 10
python scripts/m6b4_acoustic_recurrence_compare.py --tier patch16 --substeps 10
python scripts/m6b4_acoustic_recurrence_compare.py --tier golden --substeps 10
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_no_regression.txt
```

## Performance Metrics

N/A — correctness sprint.

## Kill Gates

- >15 fields diverge at substep 1 → STOP (composition is broken; per-operator parities don't compose).
- Operational sha changes → STOP, revert.
- `patch -p1 --dry-run` RC ≠ 0 on either patch → STOP, repair.
- Any field classified `operational-approved-with-evidence` without Tier-4 citation → REJECT.

## Risks

- Composition error: even with per-operator parity, the recurrence's accumulated round-off may exceed per-substep tolerance. Document the geometric-growth bound and pick per-substep tolerance accordingly.
- M6B3's CALL-site insertions were deferred to hook-ABI sprint — `acoustic_loop.py` must not depend on them; can use Python-extracted reference instead.
- Comparator infrastructure now uses `comparator_common.py` — import correctly, don't reintroduce dedup.

## Handoff Requirements

When all proofs + worker-report committed on branch `worker/gpt/m6b4-acoustic-recurrence-parity`: `/exit`. Manager dispatches M6B5 (full dycore step parity: physics off, boundary off, sanitizer off, 10 steps).

## Failure modes the manager will reject

- Patch breakage (RC ≠ 0).
- Skipping operational-compatibility section.
- Modifying operator-level helpers (those are M6B0-R/B1/B2/B3 deliverables; locked).
- Multi-RK-stage coupling.
- Post-sanitize finiteness as acceptance.
- "Composition parity" without per-substep delta evidence.
