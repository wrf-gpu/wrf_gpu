# Sprint Contract — M5-S3.zz RRTMG SW Closeout (sfluxzen + setcoef precision + lax.scan fusion)

**Sprint ID**: `2026-05-21-m5-s3zz-rrtmg-sw-closeout`
**Created**: 2026-05-21 12:40 by manager
**Status**: ACTIVE — dispatching now in parallel with M6-S2
**Trigger**: M5-S3.z Opus reviewer §4 binding decision: Option 1 SW-focused, ~85% success probability

## Objective

Close SW Tier-1 to strict flux-output level using the M5-S3.z intermediate-oracle infrastructure. This sprint is **SW-only**; M5-S3.zzz will do LW.

Three deliverables in order of priority:
1. Fix `sfluxzen` band/g-point allocation against intermediate oracle
2. Resolve `setcoef` precision policy (recompile WRF `-r8` OR amend contract to single-precision floor)
3. Re-enable validated 14 SW branches in production via `lax.scan` fusion (target ≤10 launches, ≤500 KB HLO)

## Acceptance (per M5-S3.z reviewer §4 binding)

- **AC1 SW `sfluxzen` matches intermediate oracle** within `abs ≤ 1e-8 + rel ≤ 1e-4`. Trace band-11 (and any other affected band) cell `[*, 0, 11]` zero-allocation case to its `taumol_sw` Fortran source. Replicate band-active layer-mask in JAX `_sw_sfluxzen`.
- **AC2 SW `setcoef` precision policy**. Choose ONE:
  - Path A: recompile WRF with `-r8 -i4` and re-extract intermediate-oracle NPZ. Slow but most defensible.
  - Path B: amend contract bar for SW `setcoef` to `abs ≤ 1e-4 + rel ≤ 1e-3` with rationale citing WRF build's real-kind. Faster.
  - **Manager recommendation: Path B** (amend contract; the WRF build's single-precision is the production oracle truth). Document in ADR-009.
- **AC3 Re-enable compact SW per-band branches in production via `lax.scan`**. Replace nearest-pressure production fallback with the 14 PASS-validated branches consumed via `lax.scan` over band index. HLO SW ≤ 500 KB. Combined raw launches ≤ 10. **NO `min(raw, cap)` fudge.**
- **AC4 Strict Tier-1 SW pass**: `abs ≤ 1 W/m² + rel ≤ 0.05` for SW fluxes (broadband AND per-band), `abs ≤ 1e-4 K/s + rel ≤ 0.05` for SW heating.
- **AC5 LW Tier-1 NO regression**: LW residuals must not get worse vs M5-S3.z baseline (LW max flux-down 70 W/m²). Specifically: do NOT touch LW production path.
- **AC6 ADR-009 amended to "SW-PARITY, LW-NOT-PARITY"** citing per-band intermediate-oracle validation evidence for SW closure. Do NOT mis-set to full PARITY.

## Inputs (carry forward UNCONDITIONALLY from M5-S3.z)

- `data/fixtures/rrtmg-intermediate-oracle-v1.npz` (121 KB, SHA-pinned)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` (validation framework)
- `src/gpuwrf/physics/rrtmg_lw.py` LW Planck-source machinery (`dplankup/dplankdn + tfn_tbl`)
- `artifacts/m5/rrtmg_per_band_status.json` (14 SW branches PASS, listed)
- `scripts/wrf_rrtmg_harness.f90` with per-band emission + intermediate dumps

## Files Worker May Modify

- `src/gpuwrf/physics/rrtmg_sw.py` (sfluxzen fix + re-enable validated branches via lax.scan)
- `src/gpuwrf/physics/rrtmg_constants.py` (if precision constants change)
- `src/gpuwrf/physics/rrtmg_tables.py` (if Path A precision re-extraction)
- `scripts/extract_rrtmg_tables.py` (if Path A re-extraction)
- `data/fixtures/rrtmg-tables-v1.npz` + `rrtmg-intermediate-oracle-v1.npz` (Path A re-extraction; otherwise preserve)
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md` (amend to "SW-PARITY, LW-NOT-PARITY")
- `scripts/m5_run_rrtmg.py`, `m5_gate_rrtmg.py` (gate logic if precision threshold changes)
- `tests/test_m5_rrtmg_*` (extend for SW pass)
- Worker report

## Files Worker Must NOT Modify

- LW production path in `rrtmg_lw.py` (preserve M5-S3.z LW Planck-source; debt to M5-S3.zzz)
- `src/gpuwrf/physics/{thompson_*, mynn_*, surface_*}` — other physics FROZEN
- `src/gpuwrf/io/**` — M6-S2a OWNS
- `src/gpuwrf/contracts/**`, `coupling/**`, `dynamics/**`
- Other ADR or governance files

## Dispatch

- Primary worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory per sprint-lifecycle)
- Wall-time: **8-16 hours** (smaller than M5-S3.y/S3.z because validated branches already exist)
- Worktree: `/tmp/wrf_gpu2_s3zz` (NEW, isolated)
- Branch: `worker/codex/m5-s3zz-rrtmg-sw-closeout`

## HARD RULES (per M5-S3.z reviewer)

1. **NO new SW branch transcription**: the 14 SW branches PASSED intermediate-oracle gate in M5-S3.z — re-enable them, do NOT re-write.
2. **NO synthetic / fabricated / clip-pinned coefficients.**
3. **NO `min(raw, cap)` launch fudge.**
4. NO LW production touch (preserve M5-S3.z LW Planck-source).
5. Cite `module_ra_rrtmg_sw.F:lineno` for every `sfluxzen` and `setcoef` claim.

## End-goal context

M6 OPERATIONAL VALIDATION (M6-S8) blocked on this + M5-S3.zzz close. After both close:
- T2 24h drift < 0.5 K (M6 UNBLOCKS)
- M6-S8 operational RMSE comparison becomes binding gate
- M7 Canary operational v0 dispatch becomes possible

The M5-S3 → S3.x → S3.y → S3.z → S3.zz → S3.zzz cycle is **6 sprints**, but each cycle preserves permanent infrastructure: native tables → Eddington oracle → intermediate-oracle NPZ → per-band validation framework → SW closeout → LW closeout. Trajectory is positive.
