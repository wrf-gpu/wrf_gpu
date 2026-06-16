# v0.17 Release Candidate Manifest

Branch: `worker/perf/v017-rc`
Base: `v0.16.0` (`322f0c69`)
Scope: performance release candidate assembled from validated v0.17 content only.

## IN

- `bl_pbl_physics=3` GFS PBL is operational: accepted, scan-wired, catalog-implemented, and GPU-smoked through the operational physics block.
- `mp_physics=24` WSM7 and `mp_physics=26` WDM7 are operational: accepted, scan-wired, catalog-implemented, and GPU-smoked through the operational physics block.
- v0.17 hail substrate needed by WSM7/WDM7 is present: `qh`, `Nh`, `qvolg`, `qvolh`, and `hail_acc` append-only State/registry support.
- Default-off performance levers are present:
  - `GPUWRF_FP32_PHYSICS=0` by default.
  - `GPUWRF_MYNN_BOULAC_ONZ=0` by default.
- Performance conclusion deliverable is present:
  - `proofs/perf/v017/V017_PERFORMANCE_CONCLUSION.md`
  - `proofs/perf/v017/plots/fig1_device_busy_vs_gap.png`
  - `proofs/perf/v017/plots/fig2_lever_bars.png`
  - `proofs/perf/v017/plots/fig3_roofline.png`
  - `proofs/perf/v017/plots/fig4_r2_launchcount.png`

## OUT

These are v0.18-scoped and are not claimed operational in this RC:

- Cumulus SAS/KF/Grell scaffolds: `cu_physics=4/93/94/95/96/99` are not accepted, scan-wired, or catalog-implemented. Pre-existing v0.16 reference-only `cu_physics=5/14/16` remain fail-closed outside the operational scan.
- Ferrier: `mp_physics=5/95` are not accepted, scan-wired, or catalog-implemented.
- Urban and lake: UCM/BEP/BEM and lake are not operational.
- Goddard/GFDL radiation: `ra_lw_physics=5` remains the pre-existing reference-only fail-closed path, not scan-wired or implemented; GFDL `ra_*_physics=99` is not operational.
- RUC/SSiB/ShinHong/GBM reference-only work is not operational. `sf_surface_physics=3` and the ShinHong/GBM PBL options are not in the RC accept, scan, or implemented sets.

Final operational deltas:

- `ACCEPTED_MP_PHYSICS = (0, 1, 2, 3, 4, 6, 8, 10, 14, 16, 24, 26, 28)`
- `ACCEPTED_BL_PBL_PHYSICS = (0, 1, 2, 3, 5, 7, 8, 99)`
- `SCHEME_STEP_SPECS = 41`

## Gates

Consistency gates:

- `assert_interfaces_consistent`: PASS
- `assert_registry_consistent`: PASS
- `assert_catalog_consistent`: PASS
- Final `SCHEME_STEP_SPECS` count: `41`

Focused CPU suite:

- Command: `PYTHONPATH=src:. JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3 pytest -q tests/contracts/test_v060_physics_interfaces.py tests/test_namelist_check.py tests/test_scheme_catalog_fail_closed.py tests/test_v017_qh_hail_state.py tests/test_v016_thompson_aero_threading.py tests/test_wsm7_savepoint_parity.py tests/test_wdm7_savepoint_parity.py tests/test_m7_restart_checkpoint_roundtrip.py tests/test_v017_fp32_physics.py --tb=short`
- Result: `201 passed, 2 skipped`

Full CPU no-new-failures comparison:

- Base `v0.16.0`: `1174 passed, 380 skipped, 271 failed, 50 errors, 2 xfailed`
- RC after stale-test fix: `1283 passed, 382 skipped, 271 failed, 50 errors, 2 xfailed`
- Filtered `lastfailed` comparison against collected RC node ids:
  - `new_in_rc_live_count = 0`
  - `gone_in_rc_live_count = 0`

Hard Gate #37, no-default-slowdown:

- Benchmark: short GPU default-Thompson operational physics step, `mp_physics=8`, other physics disabled, `16x16x24`, all v0.17 perf flags forced off.
- Lock wrapper: `bash scripts/with_gpu_lock.sh --label gpt-rc -- ...`; lock released after each run.
- Base `v0.16.0` (`322f0c69`):
  - State slots: `62`
  - Common output digest: `de8b21905d5f7125f7dad1b6e526047a4b2422cdf070c8274afdab5c02c97d1a`
  - Warm median: `3.677903994685039 ms/step`
  - Warm mean: `3.619831231965994 ms/step`
- RC (`f9545a1b`):
  - State slots: `67`
  - Common output digest: `de8b21905d5f7125f7dad1b6e526047a4b2422cdf070c8274afdab5c02c97d1a`
  - Warm median: `3.344821510836482 ms/step`
  - Warm mean: `3.4134216296176114 ms/step`
- Verdict: PASS. Common default-Thompson fields are byte-identical. RC median is `-9.056%` versus base in this short run; no default slowdown detected.

GPU operational smokes:

- WSM7 `mp_physics=24`: PASS, GPU backend, finite output, `qv` mutated, `qh` finite, `hail_acc` finite; lock released.
- WDM7 `mp_physics=26`: PASS, GPU backend, finite output, `qv` mutated, `qh` finite, `hail_acc` finite; lock released.
- GFS PBL `bl_pbl_physics=3` with `sf_sfclay_physics=1`: PASS, GPU backend, finite theta/u, theta and u mutated; lock released.
