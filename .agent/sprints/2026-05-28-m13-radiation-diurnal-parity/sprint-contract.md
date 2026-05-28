# Sprint Contract — M13: Radiation + diurnal physics parity

**Sprint ID**: `2026-05-28-m13-radiation-diurnal-parity`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m13-radiation-diurnal-parity`
**Worktree**: `/tmp/wrf_gpu2_m13`
**Wall-time**: 6-18 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

M9.C confirmed SWDOWN max 1122 W/m² mean RMSE 335 W/m² is REAL_BUG, and GLW max 414 W/m² mean RMSE 343 W/m². The current RRTMG adapter in `src/gpuwrf/coupling/physics_couplers.py:379-381` uses **constant** albedo=0.15, emissivity=0.98, coszen=0.50 (per iter2 evidence). These constants must become **time-varying** with solar position and surface-type-dependent.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/m9/divergence_map_v2.json` — SWDOWN/GLW defect evidence
2. `src/gpuwrf/coupling/physics_couplers.py` — RRTMG adapter (current constants)
3. WRF `phys/module_radiation_driver.F` for cosine-zenith computation pattern (read-only)
4. `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json` — iter2 baseline

## Acceptance

### AC1 — Cosine-zenith time-varying

Replace `coszen=0.50` constant with `coszen` computed from solar position (latitude, longitude, time-of-day). Use a deterministic ephemeris formula consistent with WRF `phys/module_radiation_driver.F`. Function `_compute_coszen(lat, lon, time_utc)` in physics_couplers.py.

### AC2 — Albedo + emissivity surface-type-dependent

Replace `albedo=0.15, emissivity=0.98` constants with per-land-use-type lookups from `lu_index` (which M10 now provides via `state.lu_index`). Use WRF's MODIS-IGBP or USGS land-use-to-albedo table — whichever matches the inputs.

### AC3 — Hour-1 SWDOWN parity

For Canary 20260521 hour 1, GPU `SWDOWN` reproduces WRF wrfout `SWDOWN` to within **10 % per-cell** on cells where WRF SWDOWN > 50 W/m² (daytime cells). Emit `proofs/m13/radiation_parity_hour_1.json`.

### AC4 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC5 — 24h skill non-regression

Re-run Canary 20260521 24h with radiation fix. Re-run skill diff. Emit `proofs/m13/post_m13_skill_diff.json`. Acceptance:
- SWDOWN mean RMSE drops ≥ 50 % vs `divergence_map_v2.json` value.
- T2 RMSE does not worsen vs `proofs/m10/post_m10_skill_diff.json`.
- T2 RMSE may improve since better radiation drives better surface diurnal cycle.

### AC6 — Worker report

Standard format. Verdict `M13_COMPLETE` if AC1-AC5 all pass; `M13_PARTIAL` otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`.
3. **Files writable**: `src/gpuwrf/coupling/physics_couplers.py` (RRTMG section + new `_compute_coszen` + new albedo/emissivity LUT), `proofs/m13/**`, `.agent/sprints/2026-05-28-m13-radiation-diurnal-parity/**`.
4. **Files NOT writable**: surface flux, MYNN, dycore, BC, state contracts (read-only), governance.
5. **Coordination with M11+M12**: avoid touching anything else; if AC requires data from M12 (e.g. updated surface flux) note as a known dependency and stop with PARTIAL.
6. **No remote push.**
7. **Manager repo ONLY**.
8. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m13 DONE exit=$?" Enter`.
9. **End with verdict**: `M13_COMPLETE` / `M13_PARTIAL` + headline SWDOWN RMSE reduction percentage.
