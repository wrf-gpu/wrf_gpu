# Worker Report — v0.14 Noah-MP d01 LU16 Nonfinite Closure

Worker: Fable high · Branch: `worker/fable/v014-noahmp-d01-lu16` · Base: `f1ecc035`
Fix commit: `80a693e2`
Proof: `proofs/v014/noahmp_d01_lu16_nonfinite_closure.md`

## Objective

Close the v0.14 release blocker: exact-branch Canary L2 nested 1h preflight
exits rc=1 with 51 nonfinite d01 land cells (LU_INDEX=16) in
T2/UST/HFX/LH/TSK/TH2/LWUPB/LWUPT/OLR.

## Root cause (proven)

**Noah-MP WATER soil/veg category gather was off by one.**
`water_hydro._category_index` used `category - 1` against the frozen 1-based
parameter tables (`(ncat+1,)` rows, all-zero dummy row 0 — WRF
`TRANSFER_MP_PARAMETERS` indexes `BEXP_TABLE(SOILTYPE)` directly).

- ISLTYP=1 (sand) read the all-zero row → SMCMAX=0 → `smc/smcmax=inf` →
  `0·inf=NaN` in `_wdfcnd1` → NaN SMOIS/SH2O at the **first** WATER call →
  NaN ground energy balance → the observed output set. The 51 bad cells are
  **exactly** the domain's 51 ISLTYP=1 land cells (1:1; LU16 merely coincident).
- All other soil categories silently ran WATER with the previous category's
  hydraulic parameters since S4 (finite → never tripped the conservation-bound
  savepoint gate).
- `noahmp_driver._gather_vec` + phenology already index 1-based; WATER was the
  only outlier (d02 was finite by luck: no sand cells there).

## Fix

One line, WRF-faithful, in `src/gpuwrf/physics/noahmp/water_hydro.py`:
`_category_index` now indexes the 1-based table directly
(`clip(category, 0, size-1)`), matching the driver. No masks, no clamps.

## Files changed

- `src/gpuwrf/physics/noahmp/water_hydro.py` — the fix (+ docstring contract)
- `tests/test_v014_noahmp_water_soil_category.py` — NEW: 3 regression tests
  (gather-row identity vs driver; one-step WATER on the exact failing dry-sand
  config; all-category finite sweep). All 3 FAIL pre-fix, PASS post-fix.
- `proofs/v014/noahmp_d01_lu16_nonfinite_repro.py` — CPU repro (real
  wrfinput_d01 warm-start, 200 production noah_mp_step steps)
- `proofs/v014/noahmp_d01_lu16_bisect.py` — stage-level NaN bisection
- `proofs/v014/noahmp_d01_lu16_nonfinite_closure.md` — proof artifact
- `proofs/noahmp/water_savepoint_parity.json` — regenerated post-fix

## Commands run / proof objects

- CPU repro pre-fix: smois NaN at step 1, 204 entries = 51 sand cells × 4
  layers. Post-fix: 200 steps finite, physical evening cooling.
- Stage bisection: phenology/precip/energy/phasechange finite; first NaN =
  `noahmp_water_hydro` (stage 5).
- S4 water savepoint gate (real-WRF oracle, pristine tables): conservation
  11/11 PASS, finite — unchanged binding verdict; parity numbers comparable
  (savepoint columns are too parameter-insensitive to discriminate — that is
  how the bug survived S4 review).
- Noah-MP pytest set: 47+8 pass (energy savepoint test needs
  `WRF_PRISTINE_ROOT=/home/enric/src/wrf_pristine/WRF` in worktrees; the
  default path is worktree-relative — pre-existing harness issue, not physics).
- GPU bounded confirmation (same exact-branch 1h preflight, lock acquired via
  `scripts/run_gpu_lowprio.sh`):
  `/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z`
  → **GREEN: rc=0, PASS_SHORT_GPU_PREFLIGHT / PIPELINE_GREEN,
  all_domains_finite=true (d01+d02), 0 nonfinite cells in the 9 previously-bad
  output fields, T2 over all 475 LU16 land cells finite (285.4–301.0 K),
  peak total VRAM 9783 MiB (no memory regression vs the failing run's ~10042).**

## Unresolved risks

1. **Soil-moisture history bias**: every Noah-MP run since S4 used one-off
   hydraulic parameters per category in WATER (energy path was correct).
   Multi-hour land gates (H4) and TOST should be re-scored after merge.
2. `_soil_param` treats any length-4 1-D array as a per-layer field (NSOIL
   collision) — inert for real 20-row tables; documented in the new test.
3. Pre-existing: proof-script default `WRF_PRISTINE_ROOT` resolves
   worktree-relative and breaks inside `.claude/worktrees/*` (export the env
   var, or fix centrally later).

## Next decision needed

Manager: merge `worker/fable/v014-noahmp-d01-lu16` (fix `80a693e2` + proof
commits; GPU preflight already GREEN on this branch), then re-run the H4
Noah-MP land gate (soil-water evolution now correctly parameterized for ALL
categories, not only sand).
