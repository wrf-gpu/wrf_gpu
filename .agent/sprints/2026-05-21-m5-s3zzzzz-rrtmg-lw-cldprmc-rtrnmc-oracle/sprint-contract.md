# Sprint Contract — M5-S3.zzzzz RRTMG LW cldprmc + rtrnmc Intermediate-Oracle + LW Broadband Closeout

**Sprint ID**: `2026-05-21-m5-s3zzzzz-rrtmg-lw-cldprmc-rtrnmc-oracle`
**Created**: 2026-05-21 15:15 — dispatching parallel with M5-S3.zzzz
**Status**: ACTIVE
**Trigger**: M5-S3.zzz Opus §3 binding decision — analog to SW S3.zzzz; same intermediate-oracle methodology

## Objective

Close LW Tier-1 broadband by extracting WRF intermediate state at `cldprmc_lw → rtrnmc` boundary, validating JAX LW cloud-optics + transfer-solver per-quantity per-band per-layer, fixing branches that fail.

**MANAGER INTERFACE-FREEZE** (binding for both M5-S3.zzzz SW and M5-S3.zzzzz LW dispatches):
- Harness record name prefixes: SW = `cldprmc_sw_*` / `spcvmc_*`; LW = `cldprmc_lw_*` / `rtrnmc_*`
- Validator naming: SW = `validate_sw_cldprmc_*` / `validate_sw_spcvmc_*`; LW = `validate_lw_cldprmc_*` / `validate_lw_rtrnmc_*`
- JSON schema keys: `sw_cldprmc_bands` vs `lw_cldprmc_bands` (disjoint top-level)
- Production code file-disjoint: SW worker owns `rrtmg_sw.py`; LW worker (this sprint) owns `rrtmg_lw.py`

## Acceptance

- **AC1 — WRF harness `cldprmc_lw` intermediate dumps** per-band per-layer per-g-point:
  - `cldfmc(lay, ig)` — MCICA cloud mask per Monte Carlo subcolumn (LW)
  - `taucmc(lay, ig)` — cloud optical depth (LW absorption-only convention)
  - Cloud effective optical-depth scaling factors per WRF `cldprmc_lw`
- **AC2 — WRF harness `rtrnmc` intermediate dumps** per-band per-layer per-g-point:
  - Per-g-point source recurrence: `pfracs(lay, igc)`, `plansum`
  - Surface emission + downward stream contributions
  - `tfn_tbl` source-correction lookup outputs
  - Per-g-point `zfd, zfu` BEFORE broadband accumulation
- **AC3 — Intermediate-oracle NPZ extension** to `rrtmg-intermediate-oracle-v1.npz` (extend, do NOT replace existing arrays)
- **AC4 — JAX validation framework extension** in `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`:
  - `validate_lw_cldprmc_taucmc(jax, wrf)` — abs ≤ 1e-4 + rel ≤ 1e-3 (single-precision floor per WRF -r4)
  - `validate_lw_cldprmc_cldfmc(jax, wrf)` — same tolerance
  - `validate_lw_rtrnmc_per_gpoint_flux(jax_zfd, jax_zfu, wrf_zfd, wrf_zfu, band)` — per-band per-g-point
  - `validate_lw_rtrnmc_source_recurrence(jax, wrf, band)` — per-band
- **AC5 — Branch-by-branch fix** of any failing LW cloud-optics or transfer-solver quantity:
  - Per-band debt list extended in `rrtmg_per_band_status.json.lw_cldprmc_bands`
  - Revert any branch that re-fails Tier-1 LW to nearest-pressure approximation (sister-sprint methodology)
- **AC6 — Strict Tier-1 LW PASS at flux-output level**: `abs ≤ 1 W/m² + rel ≤ 0.05` for LW fluxes (broadband + per-band)
- **AC7 — SW NO regression**: `git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py` MUST be empty (SW is M5-S3.zzzz territory)
- **AC8 — ADR-009 status update**: 
  - If LW Tier-1 PASS AND SW Tier-1 PASS (S3.zzzz also closed): amend to "SW-PARITY, LW-PARITY"
  - If LW PASS only: amend to "SW-PARTIAL, LW-PARITY"
  - If LW FAIL: hold NOT-PARITY + recommend next sprint

## Inputs (carry forward UNCONDITIONALLY)

- All M5-S3.zzz outputs: 16-band LW taumol+fracs branches; intermediate-oracle NPZ; ADR-009 NOT-PARITY hold
- `data/fixtures/rrtmg-intermediate-oracle-v1.npz` (gas-optical-depth + Planck-source oracles already in)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` — extend
- `scripts/wrf_rrtmg_harness.f90` — extend with `cldprmc_lw_*` + `rtrnmc_*` records

## Files Worker May Modify

- `src/gpuwrf/physics/rrtmg_lw.py` (cloud-optics + transfer fixes; lax.scan barrier preserved)
- `src/gpuwrf/physics/rrtmg_tables.py` (if new LW table dimensions needed)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` (extend with `lw_cldprmc_*` + `lw_rtrnmc_*` validators)
- `scripts/wrf_rrtmg_harness.f90` (extend with `cldprmc_lw_*` + `rtrnmc_*` record dumps — interface-freeze names)
- `scripts/m5_generate_rrtmg_fixture.py` (parse new LW records)
- `data/fixtures/rrtmg-intermediate-oracle-v1.npz` (extend, do NOT touch existing arrays)
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md` (amend per outcome)
- `tests/test_m5_rrtmg_*` (extend with LW cldprmc + rtrnmc tests)
- Worker report

## Files Worker Must NOT Modify

- **`src/gpuwrf/physics/rrtmg_sw.py`** — M5-S3.zzzz territory (file-disjoint)
- Other physics modules (frozen)
- io/, dynamics/, contracts/, coupling/

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory)
- Wall-time: **16-32h**
- Worktree: `/tmp/wrf_gpu2_s3zzzzz` (NEW)
- Branch: `worker/codex/m5-s3zzzzz-rrtmg-lw-cldprmc-rtrnmc-oracle`

## HARD RULES

1. File-disjoint: NO `rrtmg_sw.py` touches.
2. Interface-freeze names MUST be followed (cldprmc_lw_* / rtrnmc_*; validate_lw_*; lw_cldprmc_bands JSON key).
3. NO synthetic / clip-pinned coefficients.
4. NO `min(raw, cap)` launch fudge.
5. Cite `module_ra_rrtmg_lw.F:lineno` (`cldprmc_lw` at ~`:1200-1400`; `rtrnmc` at `:3270-3340`).
6. WATCHDOG + multi-Enter pattern in launcher.

## End-goal context

M5 RRTMG PARITY = S3.zzzz (SW broadband) + S3.zzzzz (LW broadband) BOTH close. After both: ADR-009 → SW-PARITY + LW-PARITY → M6-S8 operational T2 binding gate meaningful → Canary daily forecast unblocks.
