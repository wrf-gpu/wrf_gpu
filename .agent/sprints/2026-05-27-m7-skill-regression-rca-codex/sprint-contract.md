# Sprint Contract — M7 Skill Regression RCA (Codex, empirical/bisection angle)

**Sprint ID**: `2026-05-27-m7-skill-regression-rca-codex`
**Created**: 2026-05-27 (user direction: publish-ready validation; skill regression discovered)
**Status**: READY — DIAGNOSTIC + MINIMAL INSTRUMENTATION
**Predecessor**: same as opus sibling (`2026-05-27-m7-skill-regression-rca-opus`)

## Objective

Pair to the opus RCA sprint. While opus reads the architecture for systematic deviations from WRF semantics, this sprint **empirically bisects** the skill regression: when does the GPU forecast first deviate from CPU WRF, and which field/region/operator carries the divergence?

You run instrumented forecasts at low cost (short horizons) and use diff measurements to localize. Minimal code is allowed — diagnostic-only instrumentation scripts under `scripts/`, but no changes to `src/gpuwrf/**` (the production code is frozen during diagnosis).

## Acceptance

- **AC1 — Hour-by-hour deviation curve**: produce a per-hour, per-field deviation table comparing the GPU 24h forecast at `/tmp/m7_pipeline_runs/20260521/wrfout_d02_*` against the Gen2 CPU reference `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfout_d02_*`. For each hour 0..23, for each of T2, U10, V10, T (3D), U (3D), V (3D), QVAPOR, P, PSFC: compute mean(GPU - CPU), max(|GPU - CPU|), correlation. Emit `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/hour_by_hour_deviation.json`.

- **AC2 — Spatial deviation maps**: for hour 1, hour 6, hour 12, hour 24, dump (GPU - CPU) on T2/U10/V10/PSFC as a small NetCDF artifact under `/tmp/m7_rca_artifacts/`. Find: is the deviation spatially uniform (suggests bias) or localized (suggests boundary or terrain artifact)? Emit `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/spatial_deviation_summary.json` with: pattern classification, max-error location, percentile statistics.

- **AC3 — First-hour deviation**: at the **first output hour (lead = 1h)**, what's the largest field-wise (GPU - CPU) per State field? Is the GPU already diverged at lead=1, or does the divergence accumulate? This is the critical bisection question. Emit `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/first_hour_diff.json`.

- **AC4 — Boundary vs interior diagnosis**: split each (GPU - CPU) field into interior (≥ 5 cells from any lateral boundary) vs boundary (< 5 cells from lateral boundary). If the deviation is concentrated in the boundary zone, this implicates the BC application path. If interior-dominated, implicates physics. Emit `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/boundary_vs_interior.json`.

- **AC5 — Sanity probe — physics ON vs OFF**: run a 1-hour GPU forecast with `radiation_cadence_steps=999999` (current default, radiation effectively off) and another with `run_physics=False` entirely (dynamics-only). Compare both against CPU. This bracket isolates "is it physics-driven divergence?" vs "is it dycore-only?". Emit `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/physics_on_off_bracket.json`.

- **AC6 — Pinpoint suspect operator**: based on AC1-AC5, name the **single most likely point of divergence**: (a) boundary application, (b) microphysics coupling, (c) PBL coupling, (d) surface/SST drift, (e) radiation absence, (f) advection scheme, (g) pressure-gradient force scheme, (h) other. Provide quantitative evidence (e.g., "first-hour T2 max diff = 8K, concentrated in the boundary zone → BC application").

- **AC7 — Worker report**: verdict `ROOT_CAUSE_LOCALIZED` / `MULTIPLE_CONTRIBUTORS` / `INCONCLUSIVE`. Include the fix-sprint recommendation.

## Files Worker May Modify

- `scripts/m7_rca_hour_by_hour.py` (NEW — analysis only, reads existing wrfouts)
- `scripts/m7_rca_spatial_maps.py` (NEW)
- `scripts/m7_rca_physics_bracket.py` (NEW)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/**`
- `tests/test_m7_rca_helpers.py` (NEW — optional, pin the diff math)

## Files Worker Must Not Modify

- `src/gpuwrf/**` — production code is frozen during diagnosis (no fixes in this sprint)
- governance files
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **No `src/gpuwrf/**` modifications.** Diagnosis sprint only.
2. **CPU pinning**: `taskset -c 0-3`.
3. **GPU**: yes — AC5 runs 1-hour forecasts. Short windows (1h each, ~6s warm). 3 forecasts total, ~30s GPU work.
4. **Do not interfere with tmux `0:1`** (nightly WRF).
5. **No remote push.** Local commit on `worker/gpt/m7-skill-regression-rca-codex` only.
6. **Honest INCONCLUSIVE**: same as opus sibling — better an honest "not yet localized" than a wrong fix.

## Dependencies

- Honest speedup + skill diff merged (commit `2dfc73b`)
- GPU 24h wrfouts present at `/tmp/m7_pipeline_runs/20260521/`
- CPU reference at `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/`
- RTX 5090 available

## Proof Objects

- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/hour_by_hour_deviation.json` (AC1)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/spatial_deviation_summary.json` (AC2)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/first_hour_diff.json` (AC3 — critical bisection)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/boundary_vs_interior.json` (AC4)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/physics_on_off_bracket.json` (AC5)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/worker-report.md` (AC7)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/m7-skill-regression-rca-codex`
- Worktree: `/tmp/wrf_gpu2_rcacodex`
- GPU usage: minimal (3 × 1-hour forecasts for AC5)

## Companion sprint

`2026-05-27-m7-skill-regression-rca-opus` — opus architecture audit. Both reports feed the fix sprint.
