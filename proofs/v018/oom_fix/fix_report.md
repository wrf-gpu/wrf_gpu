# v0.18 OOM Fix Report

Status: PASS.

## Fix

- Lowered production RRTMG column tile defaults from 16384 to 2048 in `rrtmg_lw.py` and `rrtmg_sw.py`.
- Added a production-width tiled MYNN cold-start QKE initializer in `physics_couplers.py`.
- Avoided large optional host metadata reductions in `d02_replay.py` for d03-scale arrays.

## Root Cause

- Radiation tiling was not the residual 2.09 GiB failure: d03 standalone LW/SW at tile 2048 peaked at 7.90 GiB / 5.90 GiB and tile 2048 vs 4096 barely changed the whole-run peak.
- The residual 2.10 GiB allocation was localized to d03 MYNN cold-start dense BouLac initialization:
  `prewarm_d03_run.stderr` shows `RESOURCE_EXHAUSTED` on `f64[144801,44]` reduce inside `_boulac_length_dense`.
- Tiling that cold-start initializer over the existing production MYNN column tile width removes the alloc failure while preserving the dense algorithm.

## Hard Gates

### Proof 1: Bit Identity

- Current-source Switzerland recheck:
  - old caps: `switzerland_old_tile16384_recheck_run.json`
  - new defaults: `switzerland_new_default2048_recheck_run.json`
  - exact compare: `bit_identity_recheck_compare.json`
- Result: PASS, 26/26 wrfout fields exact, `max_abs_diff_overall=0.0`.
- The new MYNN cold-start tile wrapper also passed GPU production-width identity:
  `mynn_coldstart_tile_identity.json`, `qke_max_abs_diff=0.0`, `pblh_max_abs_diff=0.0`.

### Proof 2: d03-Scale Peak VRAM

- Tile defaults: `tile_defaults.json`
  - Switzerland d01 remains one tile: 1764 columns.
  - AC1_FIT d03 uses 71 tiles at cap 2048 for 144801 columns.
- Standalone d03 radiation:
  - LW 2048: SMI peak 7.90 GiB.
  - SW 2048: SMI peak 5.90 GiB.
- d03 prewarm after MYNN cold-start tiling:
  - `prewarm_d03_tiled_run.json`: PASS, peak 13.53 GiB.
  - `prewarm_d03_tiled_marked_run.json`: PASS, peak 17.47 GiB.
- Old failure baseline: previous real path OOMed near 31.8/32 GiB and failed a 2.09-2.10 GiB allocation.

### Proof 3: AC1_FIT Short No-OOM Smoke

- Cold cache run: `real_ac1fit_halfhour_tiled_smoke.json`, PASS.
- Warm cache decisive run: `real_ac1fit_halfhour_tiled_warm.json`, PASS.
- Warm run:
  - d03 reached and ended on radiation gate: 900 own steps, `radt=30`, 1800 forecast seconds.
  - all domains finite.
  - peak VRAM: 18.34 GiB.
  - median VRAM: 18.12 GiB.
  - forecast wall: 867.34 s for 0.5 forecast hours = 1734.69 s/forecast-hour.
  - overall util: mean 69.57%, median 96%, p95 100%, idle<10 23.0%.
  - forecast-only util: mean 75.60%, median 97%, p95 100%, idle<10 16.7%.
  - steady after first 120 forecast seconds: mean 85.05%, median 97%, p95 100%, busy>=80 82.1%, idle<10 7.0%.
- Decision: no fused-mode escalation required; the bit-identical default warm path reaches high steady utilization once the one-time cache/load section is excluded.

### Proof 4: Small-Case Non-Regression

- Switzerland d01 recheck old/new both `PIPELINE_GREEN`.
- Forecast-only wall:
  - old cap 16384: 24.15 s.
  - new default 2048: 23.94 s.
- Small case remains one tile, so no launch-storm regression is introduced for this gate.

## Verification Commands

- `python3 -m py_compile src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/integration/d02_replay.py proofs/v018/oom_fix/prewarm_ac1fit_advance.py proofs/v018/oom_fix/real_ac1fit_halfhour_smoke.py proofs/v018/oom_fix/mynn_coldstart_tile_identity.py proofs/v018/oom_fix/compare_wrfout_exact.py proofs/v018/oom_fix/tile_default_probe.py`
- `git diff --check`
- `JAX_PLATFORM_NAME=cpu PYTHONPATH=src JAX_ENABLE_X64=true pytest -q tests/test_v014_mynn_coldstart_init.py`

## Notes

- `GPUWRF_MYNN_BOULAC_ONZ=1` was tested as a diagnostic and is not used: it avoided the dense cold-start OOM but reproduced the known XLA compile/autotune pathology.
- The current checked-out source has no active `GPUWRF_NESTED_FUSE` implementation; no fused measurement was needed because the default warm run met the utilization gate.
