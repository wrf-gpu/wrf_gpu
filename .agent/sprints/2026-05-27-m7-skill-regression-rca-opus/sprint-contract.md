# Sprint Contract — M7 Skill Regression Root-Cause Analysis (Opus, architecture angle)

**Sprint ID**: `2026-05-27-m7-skill-regression-rca-opus`
**Created**: 2026-05-27 (user direction: publish-ready validation; skill regression discovered)
**Status**: READY — DIAGNOSTIC ONLY
**Predecessor**: `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md` (GPU +243-440% RMSE vs CPU on T2/U10/V10); `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` (L2 d02 also FAIL)

## Objective

The GPU forecast wall-clock is fast (~50× CPU) but the forecast is materially **less skillful** than CPU WRF on AEMET station scoring. T2 RMSE is 266% worse, U10 is 390% worse, V10 is 243% worse. The bounds are physically plausible (no NaN, no overflow), so it's not an obvious instability — it's a systematic operational bias.

This sprint is the **opus angle** root-cause investigation, paired with a parallel codex sprint at the **kernel/data-flow level**. You read the boundary forcing path, the physics-coupling architecture, and the integration order; identify where the GPU pipeline could systematically deviate from CPU WRF semantics. Architecture diagnosis only — no code changes.

## Acceptance

- **AC1 — Boundary-forcing path audit**: read `src/gpuwrf/integration/d02_replay.py`, `src/gpuwrf/coupling/boundary_apply.py`, `src/gpuwrf/io/boundary_replay.py`. Trace how Gen2 d01 hourly wrfouts become d02 boundary tendencies. Verify against the WRF Fortran reference `share/module_bc.F` (or equivalent). Identify any place the GPU port differs from WRF Fortran semantics (e.g., interpolation order, lateral relax zone profile shape, time-interpolation cadence, tendency vs absolute-value handling, mass-coupled vs uncoupled boundary fields, qx_bdy hydrometeor handling).

- **AC2 — Physics coupling order audit**: read `src/gpuwrf/coupling/physics_couplers.py`, `src/gpuwrf/runtime/operational_mode.py` (the RK step composition), and the WRF Fortran `dyn_em/solve_em.F` for comparison. WRF integrates physics + dynamics in a specific sequence (radiation → microphysics → PBL → surface → cumulus etc., interleaved with the dycore RK3). Verify the GPU pipeline matches the WRF sequence. Common failure modes: physics applied after wrong dycore stage, fluxes double-counted, microphysics qv mass-coupling order, PBL stress applied with wrong sign/scaling.

- **AC3 — Surface layer / SST / land state audit**: T2 is a 2m-diagnostic computed from surface temperature + 10m wind + surface fluxes. The 266% T2 regression suggests the surface energy balance is off. Trace `src/gpuwrf/coupling/physics_couplers.py:surface_adapter` and `src/gpuwrf/io/land_state.py`. Verify SST handling (is SST held constant from IC? Reapplied each step? Drifting?), soil moisture initialization, surface roughness map.

- **AC4 — Microphysics + radiation cadence**: the operational namelist runs RRTMG at `radiation_cadence_steps=999999` (i.e. **effectively disabled** for the 24h pipeline run, per `pipeline_run_20260521.json`). That alone could explain a large T2 bias — no shortwave/longwave forcing means surface energy balance is unrealistic. Confirm and document.

- **AC5 — Wind field damping / diffusion**: U10/V10 RMSE 390-243% worse suggests wind fields are either over-damped or under-damped. Audit hyperdiffusion coefficients, MYNN PBL output coupling, and any sponge-layer / horizontal-diffusion settings vs WRF operational defaults.

- **AC6 — Top 3 suspects + plausibility ranking**: produce `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` listing the highest-confidence root causes from AC1-AC5, with: hypothesis, evidence in code (file:line), expected size of skill impact, suggested fix sprint scope, dependence on the codex sibling probe.

- **AC7 — Cross-check with codex**: read `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/` artifacts when present. Mark which architectural suspects the codex probe's empirical findings reinforce or falsify.

- **AC8 — Tester report**: verdict `STRONG_SUSPECTS_NAMED` / `INCONCLUSIVE`. Include explicit prioritization for the fix sprint.

## Files Tester May Read

- All of `src/gpuwrf/**`
- Reference WRF Fortran source if accessible: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/` (per `cpu-wrf-baseline.md`)
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/` artifacts
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/` artifacts
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` (the namelist used)
- M6 + M7 closeouts

## Files Tester May Modify

- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/**` only

## Hard Rules

1. **No code changes.** Diagnostic only.
2. **CPU pinning**: `taskset -c 0-3`.
3. **Do not interfere with tmux `0:1`** (nightly WRF).
4. **No memory updates** without manager approval.
5. **No remote push.**
6. **Honest INCONCLUSIVE**: if no architectural suspect rises above medium confidence, say so plainly. We'd rather have an honest "we don't know yet" than a wrong fix.

## Dependencies

- Honest speedup + skill diff sprint merged (commit `2dfc73b`)
- L2 d02 replay merged (commit `833c61c`)
- WRF Fortran source readable

## Proof Objects

- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` (AC6)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/architecture_audit.md` (AC1-AC5 consolidated)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/tester-report.md` (AC8)

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h
- Branch: `tester/opus/m7-skill-regression-rca-opus`
- Worktree: `/tmp/wrf_gpu2_rcaopus`
- GPU usage: NONE

## Companion sprint

`2026-05-27-m7-skill-regression-rca-codex` — parallel codex sprint at the kernel/empirical-bisection level. Both reports feed the fix sprint.
