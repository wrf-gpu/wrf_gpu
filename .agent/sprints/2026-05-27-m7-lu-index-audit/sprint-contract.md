# Sprint Contract — M7 LU_INDEX Audit (Sprint D)

**Sprint ID**: `2026-05-27-m7-lu-index-audit`
**Created**: 2026-05-27 (parallel to algorithmic fix sprint)
**Status**: READY — research + small surgical fix
**Predecessor**: `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/first_hour_diff.json` (LU_INDEX max abs diff = 14 categories at lead=1h)

## Objective

The RCA codex empirical bisection found that the GPU forecast's `LU_INDEX` (land-use category) differs from the Gen2 reference by **max 14 categories** at the first output hour. `LANDMASK` and `HGT` match, so this is not a fundamental terrain or land/sea mask issue — it's the **land-use category labeling** of the land cells.

LU_INDEX feeds the surface scheme's roughness lookup, albedo, emissivity, soil-type indexing, vegetation parameters — every land-physics quantity. A 14-category mismatch means we're treating large land areas as the wrong biome (e.g., desert vs forest vs cropland). This alone could cause substantial surface flux errors that compound with the algorithmic defects already named.

This sprint audits + fixes the LU_INDEX ingestion. Static-field fix only — NO dycore/physics code touched.

## Acceptance

- **AC1 — Source-of-truth audit**: read `src/gpuwrf/io/land_state.py` and identify where `LU_INDEX` is loaded from `wrfinput_d02`. Compare to how Gen2 CPU WRF reads `LU_INDEX`. Is it a different file, a different time index, a re-projection step, or a cast/round defect? Emit `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_source_audit.md`.

- **AC2 — Spatial mismatch map**: compute (GPU LU_INDEX) vs (Gen2 reference LU_INDEX) on the same grid for the 20260521 case. Emit `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_map.nc` (small NetCDF). Identify: is the mismatch uniform spatial-shift? Random? Concentrated over specific biomes? Categorical-mapping mistake (e.g., MODIS vs USGS classification ID offset)?

- **AC3 — Fix proposal**: based on AC1+AC2, propose the minimal fix. Possibilities:
  - `LU_INDEX` cast from FP to int with wrong rounding mode
  - Off-by-one in a categorical lookup
  - Wrong source file (e.g., reading from L3 instead of L2 path; reading the wrong domain's geo_em)
  - Missing categorical remapping (MODIS_IGBP_NOAH ↔ USGS scheme)

- **AC4 — Fix applied**: implement the named fix in `src/gpuwrf/io/land_state.py` (or whichever io module owns LU_INDEX ingestion). Add a unit test that pins the post-fix `LU_INDEX` distribution against the Gen2 reference's distribution within ±1 category for ≥99% of cells.

- **AC5 — Sanity probe**: re-run a 1-hour forecast on 20260521 with the LU_INDEX fix; compare HFX/LH/PBLH/T2 at lead=1 against Gen2. Expect noticeable improvement on land cells (the most-mismatched ones). Document the improvement; emit `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_fix_lead1_verification.json`.

- **AC6 — Invariant preservation**: 20260521 multi-step parity step 2 still 0.0 bitwise (the static-field fix should not affect the dycore at all if it's truly static).

- **AC7 — Worker report**: verdict `LU_FIXED` / `LU_PARTIAL` / `BLOCKED`.

## Files Worker May Modify

- `src/gpuwrf/io/land_state.py` (AC4 fix — the static-field ingestion)
- `tests/test_m7_lu_index_audit.py` (NEW)
- `.agent/sprints/2026-05-27-m7-lu-index-audit/**`

## Files Worker Must Not Modify

- `src/gpuwrf/runtime/operational_mode.py` — Sprint A+B+C handle this
- `src/gpuwrf/coupling/physics_couplers.py` — Sprint A+B+C handle this
- `src/gpuwrf/dynamics/**`, `src/gpuwrf/contracts/**`, `src/gpuwrf/validation/**`
- governance files, `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **Static-field fix only.** No dycore, no physics adapters, no validation code touched.
2. **CPU pinning**: `taskset -c 0-3`.
3. **GPU**: minimal — AC5 runs a 1-hour forecast.
4. **No remote push.** Local commit on `worker/gpt/m7-lu-index-audit` only.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/m7-lu-index-audit`
- Worktree: `/tmp/wrf_gpu2_lufix`
- GPU usage: minimal
- Parallel-safe with algorithmic fix sprint (different file ownership) + publication draft (no code overlap)
