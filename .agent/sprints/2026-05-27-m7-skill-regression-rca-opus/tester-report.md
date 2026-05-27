# Tester Report — M7 Skill Regression RCA (Opus angle)

**Sprint**: `2026-05-27-m7-skill-regression-rca-opus`
**Role**: tester (claude opus 4.7) on branch
`tester/opus/m7-skill-regression-rca-opus`
**Mode**: read-only diagnostic, per sprint contract Hard Rule #1
("No code changes"). Zero files under `src/`, `scripts/`, or governance
modified. Tests directory not modified (the sprint contract restricts
me to `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/**`; the
generic role-prompt template's "you may edit tests/" line is
superseded by the sprint contract per AGENTS.md read-order rule 7
"current sprint contract" over rule 8 "role template").

## Why this report is opus-only

The sprint dispatch spawned a parallel opus angle (architecture audit)
and a codex angle (empirical bisection). At report time:

- No `worker/opus/m7-skill-regression-rca-opus` branch exists.
- The opus sprint directory contains only `sprint-contract.md`,
  `role-prompts/tester.md`, and the tester-completion bookkeeping
  files — no `worker-report.md` was written.
- The codex sibling
  (`.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/`) also
  has no proof objects committed yet (only its `sprint-contract.md`
  and `role-prompts/`).

Per the sprint role assignment, this report is therefore the entire
opus-angle deliverable. AC1-AC7 architectural audit work is reported
here; AC8 (this report) renders the verdict.

## Acceptance criteria coverage

| AC  | Topic                                  | Where addressed                              | Result                       |
|-----|----------------------------------------|----------------------------------------------|------------------------------|
| AC1 | Boundary-forcing path audit            | `architecture_audit.md` §AC1                 | DEFECT NAMED (width-1 strip) |
| AC2 | Physics coupling order audit           | `architecture_audit.md` §AC2                 | DEFECT NAMED (theta/mu reset)|
| AC3 | Surface / SST / land state audit       | `architecture_audit.md` §AC3                 | DEFECT NAMED (frozen IC)     |
| AC4 | Radiation cadence audit                | `architecture_audit.md` §AC4                 | CONFIRMED (cadence=999999)   |
| AC5 | Wind damping / diffusion audit         | `architecture_audit.md` §AC5                 | MINOR (skeleton unwired)     |
| AC6 | Top 3 suspects + plausibility ranking  | `top_3_suspects.md`                          | THREE PRIMARY + ONE HONORABLE|
| AC7 | Cross-check codex sibling              | `architecture_audit.md` final section        | DEFERRED (no codex artifacts)|
| AC8 | Tester report with verdict             | this file                                    | DELIVERED                    |

## Re-runs and verifications performed (read-only)

Per the contract's tester instructions ("re-run every validation
command in the contract from a clean shell"), this sprint is
diagnostic with **no validation commands** defined — the contract
specifies file-reading and report-writing as the only deliverables.
Re-runs I did perform from a clean shell:

1. Confirmed git state on the assigned branch is clean and matches
   the dispatch commit `e1fc63a`.
2. Re-listed the predecessor sprint artifacts and confirmed the
   verdict file `gpu_vs_cpu_skill_diff.json` reports T2 RMSE 7.86 vs
   CPU 2.15, U10 RMSE 11.31 vs CPU 2.31, V10 RMSE 9.44 vs CPU 2.75.
3. Re-read the operational namelist baked into
   `pipeline_run_20260521.json` and confirmed
   `radiation_cadence_steps: 999999`, `dt_s: 10`, `acoustic_substeps:
   10`, `run_physics: true`, `run_boundary: true`,
   `use_vertical_solver: true`.
4. Re-read `bounds_check_l2_d02.json` and verified the
   `theta_lower_30_max_k = 355.36 K`, `theta_max_k = 492.67 K`
   constants are identical across all 24 hourly snapshots — direct
   on-disk evidence consistent with Suspect #1's "theta is reset
   every step" mechanism.
5. Walked the integration entry point chain
   `daily_pipeline._default_forecast_fn` →
   `operational_mode.run_forecast_operational` →
   `_scan_forecast_segment` → `_physics_boundary_step` →
   `_rk_scan_step` + the guard branch. Confirmed line numbers cited
   in `architecture_audit.md` match the in-tree code.
6. Confirmed `disable_guards=False` is the only allowed production
   setting via `tests/test_m6_guard_disabled_debug.py:393-405`
   (Stage-2 safe-default proof rejects `disable_guards=True`). The
   guard branch is therefore the **only** operational code path.
7. Confirmed `boundary_replay.decode_wrfbdy` is implemented and
   knows about WRF's `bxs/bys/btxs/btys` and `bdy_width` dimension,
   but the integration path uses `_field_sides_3d` (single outermost
   row) instead. The decoded WRF boundary tendencies are read for
   *comparison* but never *consumed*.

## Tests I considered adding but did not

- `tests/test_m7_rca_theta_reset.py` — pin the guard branch by
  asserting that `_physics_boundary_step(state)` with a non-trivial
  theta tendency produces `state.theta == physical_origin.theta` when
  `disable_guards=False`. Decision: defer to the fix sprint; pinning
  a defect that the fix sprint will intentionally rewrite produces
  test churn and a misleading regression signal.
- `tests/test_m7_rca_boundary_width.py` — pin that
  `state.mu_bdy.shape[2] == 1`. Same defer reasoning.
- `tests/test_m7_rca_mynn_zero_flux.py` — pin that `mynn_adapter`
  passes literal zeros as surface-flux inputs. Same defer reasoning.

If the fix sprint diverges from the recommended ordering in
`top_3_suspects.md` (Sprint A → B → C → D → E), the corresponding
pin tests should be added by the *fix-sprint* tester so they fail at
exactly the right intermediate state and pass once the fix lands.

## Gaps and risks

- **Codex sibling not yet reporting** (AC7 deferred). When codex'
  empirical bisection lands, re-rank as follows:
  - If codex AC3 first-hour T2 max diff is < 1 K and grows linearly
    with hour, that confirms Suspect #1 (interior advection broken,
    accumulating).
  - If codex AC4 boundary-vs-interior shows the U10 deviation
    concentrated in the boundary zone, escalate Suspect #4 ahead of
    Suspect #3.
  - If codex AC5 physics-on/off bracket shows
    `physics=False` and `physics=True, radiation=off` are within
    0.1 K T2, that confirms Suspect #2 (physics path is contributing
    little) and demotes Suspect #3 by comparison.
- **WRF Fortran reference not directly readable** from this audit
  shell — the contract pointed at
  `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/` but that path
  was not exposed. WRF-side line citations in `architecture_audit.md`
  rely on the inline WRF source-anchor comments already in the GPU
  code. If a Fortran cross-check is required for the fix sprint's
  ADR, request a separate scout to mount that tree read-only.
- **Honesty caveat on Suspect ranking**: the three primary suspects
  interact multiplicatively. Fixing any one alone is unlikely to
  recover full skill; the recommendation in `top_3_suspects.md` is
  to fix in the order A (theta/mu reset) → B (MYNN/surface coupling)
  → C (radiation) before re-measuring.

## Files written

- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/architecture_audit.md`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/tester-report.md`
  (this file)

## Files NOT modified

- All of `src/gpuwrf/**` (Hard Rule #1).
- All of `scripts/**`.
- All of `tests/**` (see "Tests I considered" — deferred to fix
  sprint per defect-pinning convention).
- Governance files (`AGENTS.md`, `CLAUDE.md`, `PROJECT_*.md`,
  `.agent/rules/`, `.agent/goals/`).
- Memory under `/home/enric/.claude/projects/`.

## Decision: STRONG_SUSPECTS_NAMED

Three high-confidence architectural defects in the M7 operational GPU
forecast path explain the AEMET-verified skill regression
(GPU T2 +266 % RMSE, U10 +390 % RMSE, V10 +243 % RMSE vs CPU WRF):

1. **Dycore theta/mu state is reset every timestep** in
   `runtime/operational_mode.py:_physics_boundary_step` (guard branch
   lines 548-563). The RK3 + acoustic advance of `theta` and `mu`
   is discarded every step, so heat is not advected by winds and the
   pressure split is stale. This is the dominant suspect for all
   three failing fields. Independently corroborated on disk by
   identical `theta_max_k` and `theta_lower_30_max_k` values across
   24 consecutive hourly bounds-check snapshots.
2. **Surface fluxes are not coupled into the atmosphere**. The MYNN
   PBL kernel is hard-fed three `zeros` arrays for its
   surface-heat/moisture/momentum bottom-BC inputs
   (`coupling/physics_couplers.py:266-269`), and
   `surface_adapter` (lines 284-312) writes the fluxes into State but
   never applies them as tendencies to `theta, qv, u, v`. Ordering is
   inverse to WRF (`mynn` is called before `surface_adapter` in
   `_physics_boundary_step:564-568`), so MYNN never sees same-step
   fluxes even if it were rewired to consume them.
3. **Surface state frozen at IC + RRTMG never invoked**.
   `t_skin, SST, SMOIS, SH2O, TSLB` are loaded once from
   `wrfinput_d02` at index 0 (`io/land_state.py:64 del time`) and
   never updated. `DailyPipelineConfig.radiation_cadence_steps =
   999999` (`integration/daily_pipeline.py:76`) makes
   `run_forecast_operational` never call `rrtmg_adapter` across an
   8640-step (24 h × dt=10 s) integration. The surface energy
   balance therefore has no degrees of freedom — neither radiative
   forcing nor skin response carries diurnal information. Dominant
   first-order contributor to T2 bias, secondary to wind bias via
   the stability-dependent diagnostic-wind reduction.

An additional defect (Suspect #4: lateral boundary forcing uses only
the outermost parent row instead of WRF's `spec_bdy_width=5` strip;
`coupling/boundary_apply.py` + `io/boundary_replay._sides:133-139` +
`integration/d02_replay._field_sides_3d:218-228`) is named in
`top_3_suspects.md` but its skill cost is currently masked by Suspect
#1 — re-rank after the fix.

Recommended fix-sprint sequence (see `top_3_suspects.md` for
acceptance criteria per sprint):

  **Sprint A → B together** (theta/mu reset removal + MYNN/surface
  coupling reorder) → **Sprint C** (RRTMG cadence) → **Sprint D**
  (prognostic or hourly-refresh surface state) → **Sprint E**
  (lateral `bdy_width`-strip fix) → **Sprint F** (re-measure AEMET).

The diagnosis is publishable as a closeout of the RCA sprint. The
M7 closeout itself remains NOT-PUBLISHABLE until Sprints A-F land
and the +20 % skill tolerance is met on a re-run of the
`m7-honest-speedup-skill-diff` proof harness.
