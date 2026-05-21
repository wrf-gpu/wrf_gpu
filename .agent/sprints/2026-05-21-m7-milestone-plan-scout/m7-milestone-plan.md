# M7 Milestone Plan Scout - Canary Operational v0

Author: Codex gpt-5.5 xhigh scout
Date: 2026-05-21
Status: Draft for critic review and manager integration

Input caveat: `.agent/sprints/2026-05-21-m6-milestone-plan-scout/m6-milestone-plan.md` is not present in this worktree. This draft follows the available M6 critic and manager-amendment pattern instead: explicit sequencing, disjoint file ownership, concrete proof objects, and critic-facing schemas.

## Section 1 - Goal Restatement

M7 delivers the first daily operational GPU forecast pipeline for the Canary target, not another offline validation case.

The v0 must-do target is:

- Run a repeatable 18Z Canary 3 km daily forecast cycle on the RTX 5090 from live AIFS IC/BC.
- Cover the Gen2 L2 domain shape: d01 9 km outer plus d02 3 km regional, one-way nested, matching Gen2 geometry.
- Produce operational output, post-processed surface products, verification metrics, restart/recovery evidence, and an operational proof index.
- Compare against the Gen2 CPU WRF baseline for the same AIFS cycle, per `.agent/references/cpu-wrf-baseline.md`.

The conditional target is:

- Add 1 km nested output only if M7-S2 proves RTX 5090 32 GB can carry the required state, compile the nested loop, and run without timestep-loop host/device transfers.
- If 1 km fails the memory/compile gate, M7 closes only as "3 km operational v0" with a 1 km deviation/blocker document. No public 1 km claim may be made.

Daily cadence for v0:

- Target init: 18Z only.
- AIFS readiness: reuse the Gen2 live-18Z pattern, which polls from 01:25 UTC to 05:25 UTC for yesterday's 18Z AIFS cycle (`Gen2/scripts/poll_live_18z_cycle_v1.py`).
- Forecast: minimum 24 h for M7 v0; optional 48 h diagnostic if M6/M7 performance budget allows.
- Post-processing: hourly NetCDF-like output -> operational Zarr plus small public JSON/PNG/CSV products.
- Verification: compare GPU vs Gen2 and GPU/Gen2 vs station observations at +6, +12, +24 h; +48 h is reported only when a 48 h diagnostic run exists.

Do not claim a "00Z in, 06Z publish" SLA. Gen2 explicitly records that source-arrival/runtime proof is missing for that claim.

## Section 2 - Sprint Sequence

### M7-S0 - Operational Readiness Prologue

Objective: freeze the M7 operational contract before code sprints. This is the gate that verifies M6 actually delivered the required interfaces and proof schemas.

Non-goals: no forecast-driver rewrite, no new physics, no public dashboard.

Wall-time estimate: 12-18 h.

File ownership:

- May modify: `src/gpuwrf/ops/__init__.py`, `src/gpuwrf/ops/config.py`, `src/gpuwrf/io/aifs_catalog.py`, `src/gpuwrf/io/proof_schemas.py` append-only, `scripts/m7_preflight_operational.py`, `tests/test_m7_preflight_operational.py`, `artifacts/m7/prologue/**`.
- Must not modify: `src/gpuwrf/physics/**`, `src/gpuwrf/dynamics/**`, `src/gpuwrf/coupling/driver.py`, M6 ADRs except by manager-approved patch.
- Read-only: `/mnt/data/canairy_meteo/**`.

Acceptance:

- `artifacts/m7/prologue/m6_inheritance_gate.json`:
  - `m6_closeout_status`, `m6_s2a_closed`, `m6_s2_closed`, `m6_s3_closed`, `adr_010_boundary_amended`, `adr_011_present`, `proof_schema_registry_present`, `blocking_items`.
- `artifacts/m7/prologue/gen2_baseline_inventory.json`:
  - pins at least one complete L3 24 h run and one complete L2 72 h run by path, mtime, file count, domains, namelist checksum, and AIFS month file.
  - example L3 inventory to pin: `/mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z/`, containing `wrfinput_d01..d05`, `wrfbdy_d01`, 25 hourly `wrfout_d0{1..5}` files, `namelist.input`, `namelist.output`, and `rsl.out/error.0000..0027`.
  - example L2 inventory to pin: `/mnt/data/canairy_meteo/runs/wrf_l2/20260519_18z_l2_72h_20260520T025228Z/`, containing `wrfinput_d01`, `wrfinput_d02`, `wrfbdy_d01`, 73 hourly `wrfout_d01` and 73 hourly `wrfout_d02` files, plus `thin_gridded_d01_v1.nc` and `thin_gridded_d02_v1.nc`.
- `artifacts/m7/prologue/operational_contract.json`:
  - `cycle_hour_utc = 18`, `forecast_hours_minimum = 24`, `aifs_poll_start_utc = "01:25"`, `aifs_poll_timeout_utc = "05:25"`, output roots, retention policy, station-source priority, failure statuses.
- `artifacts/m7/prologue/aifs_ingest_contract.json`:
  - source paths, upstream source (`dynamical-ecmwf-aifs-single` icechunk and ECMWF open-data GRIB), variables, pressure levels, steps, interpolation policy, fallback policy.

Dependencies and parallelism:

- Blocks all M7 implementation sprints.
- Can run while final M6 closeout is being assembled, but must emit `BLOCKED` if M6-S2a/S2/S3 evidence is missing.
- M7-S0 must explicitly handle the current fact that ADR-011 is not present in this worktree; if still absent at M7 dispatch, S0 blocks S1.

### M7-S1 - 3 km Daily Pipeline

Objective: implement the real 18Z daily d01->d02 GPU run path from live AIFS, replacing M6's Gen2 backfill/boundary-replay input with operational IC/BC ingest.

Non-goals: 1 km nests, output styling, METplus integration, multi-GPU.

Wall-time estimate: 36-60 h.

File ownership:

- May modify: `src/gpuwrf/io/aifs_ingest.py`, `src/gpuwrf/io/wps_met_em.py`, `src/gpuwrf/ops/daily_cycle.py`, `src/gpuwrf/ops/run_manifest.py`, `src/gpuwrf/coupling/operational_driver.py`, `scripts/m7_run_daily_3km.py`, `tests/test_m7_aifs_ingest.py`, `tests/test_m7_daily_3km.py`, `artifacts/m7/3km/**`, `data/ops/m7/cycles/**`.
- May use but not rewrite: `src/gpuwrf/io/gen2_accessor.py`, `src/gpuwrf/io/validation.py`, `src/gpuwrf/coupling/driver.py`, `src/gpuwrf/coupling/boundary_apply.py`.
- Must not modify: `src/gpuwrf/physics/**` and `src/gpuwrf/dynamics/**` except via separate reviewed sprint.

Acceptance:

- `artifacts/m7/3km/aifs_to_state_manifest.json`:
  - `cycle_id`, `aifs_source`, `raw_grib_manifest`, `met_em_or_native_regrid_manifest`, `wrfinput_compatibility`, `grid_spec_sha`, `state_fields_loaded`, `boundary_fields_loaded`, `boundary_cadence_hours = 6`, `time_interpolation = "linear"`.
- `artifacts/m7/3km/daily_3km_run_summary.json`:
  - `cycle_id`, `domains = ["d01","d02"]`, `grid_shape`, `forecast_hours`, `dt_seconds`, `steps`, `output_times`, `status`, `wall_time_s`, `compile_time_s`, `warm_run_wall_time_s`, `device`, `jax_version`, `commit`, `input_manifest`.
- `artifacts/m7/3km/transfer_audit.json`:
  - `host_to_device_bytes_init`, `host_to_device_bytes_timestep_loop`, `device_to_host_bytes_timestep_loop`, `device_to_host_bytes_output`, `audit_method`, `pass`.
  - Hard gate: `*_timestep_loop` must be zero.
- `artifacts/m7/3km/gen2_same_cycle_comparison.json`:
  - `gen2_run_path`, `gpu_output_path`, per variable `U10/V10/T2/Q2/RAINNC` at +6/+12/+24, norms, masks, regridding details, per-variable status.
- Output directory: `/mnt/data/wrf_gpu2/operational/m7/cycles/<YYYYMMDD>_18z/` with `run_manifest.json`, hourly model output, and `latest` symlink updated atomically.

Dependencies and parallelism:

- Depends on S0 and on M6-S2 closing the real d02 forecast-driver prerequisites from M6-S1: FP32 ratification, real GridSpec metrics, real temporary-byte accounting, robust radiation cadence, and boundary-forcing State leaves.
- S2 memory audit may start after S1 compiles the d01/d02 state factory, but S3 cannot start until S1 has a successful 3 km run.
- S4 post-processing can start on S1 output after a 6 h smoke artifact exists, but final S4 waits for S1 24 h.

### M7-S2 - 1 km Memory, Compile, and Residency Audit

Objective: decide whether 1 km can be part of M7 v0 on the RTX 5090 32 GB target.

Non-goals: no operational 1 km claim, no tiling implementation unless the audit fails and manager approves a scoped follow-up.

Wall-time estimate: 12-20 h.

File ownership:

- May modify: `src/gpuwrf/profiling/memory.py`, `src/gpuwrf/ops/nesting_plan.py`, `scripts/m7_audit_1km_memory.py`, `tests/test_m7_1km_memory_audit.py`, `artifacts/m7/memory/**`.
- Must not modify: `src/gpuwrf/coupling/operational_driver.py` except by S1/S3 owner coordination.

Acceptance:

- `artifacts/m7/memory/rtx5090_1km_memory_audit.json`:
  - `hardware = "RTX 5090 32GB"`, `driver`, `cuda`, `jaxlib`, `domains`, `domain_shapes`, `state_leaf_count`, `persistent_state_bytes`, `physics_table_bytes`, `boundary_buffer_bytes`, `output_buffer_bytes`, `peak_hbm_bytes`, `compile_time_s`, `hlo_bytes`, `max_temp_bytes`, `host_device_transfer_bytes_timestep_loop`, `verdict`.
- `artifacts/m7/memory/nesting_compile_smoke.json`:
  - compiles static one-way d01/d02/d03 scan shape for at least 12 model hours or emits a concrete compiler/memory failure artifact.
- PASS means: peak HBM <= 26 GB, compile <= 45 min, no timestep-loop transfers, and no XLA OOM/retry instability.
- FAIL classes: `FAIL_HBM`, `FAIL_COMPILE_TIME`, `FAIL_XLA_TEMPORARIES`, `FAIL_TRANSFER`, `FAIL_SCHEMA`.

Dependencies and parallelism:

- Depends on S0 and enough S1 state/ingest factory to allocate real grids.
- If PASS, S3 can dispatch.
- If FAIL, manager chooses between a tiling/streaming design sprint and a 3 km-only M7 close. The scout recommendation is not to implement tiling inside M7-S2.

### M7-S3 - Conditional 1 km Pipeline

Objective: if S2 passes, run the Gen2 L3 one-way nesting chain on GPU: d01 9 km, d02 3 km, and at least d03 1 km Tenerife. If memory headroom permits, include d04 Gran Canaria and d05 La Palma sibling 1 km nests.

Non-goals: two-way nesting, multi-GPU halo exchange, sub-km L4.

Wall-time estimate: 36-72 h if S2 passes; 12-18 h for a deviation/tiling design if S2 fails.

File ownership:

- May modify: `src/gpuwrf/ops/nested_cycle.py`, `src/gpuwrf/coupling/nested_driver.py`, `src/gpuwrf/coupling/nest_boundary.py`, `scripts/m7_run_daily_1km.py`, `tests/test_m7_nested_driver.py`, `tests/test_m7_nest_boundary.py`, `artifacts/m7/1km/**`, `data/ops/m7/cycles/**/d03*/**`.
- Must not modify: S4 post-processing modules, S5 verification modules, physics kernels.

Acceptance:

- `artifacts/m7/1km/nested_run_summary.json`:
  - `cycle_id`, `domains`, `grid_shapes`, `parent_child_map`, `parent_time_step_ratio`, `feedback = 0`, `forecast_hours`, `wall_time_s`, `compile_time_s`, `output_paths`, `status`.
- `artifacts/m7/1km/nest_boundary_consistency.json`:
  - per child domain, per side, per boundary variable, interpolation policy, parent sample path, child boundary path, continuity residual norms, pass/fail.
- `artifacts/m7/1km/gen2_l3_comparison.json`:
  - GPU vs Gen2 L3 for d03 and any included sibling nests at +6/+12/+24, with masks and regridding policy.
- If S2 failed, replacement artifact `artifacts/m7/1km/one_km_deviation_or_tiling_plan.md`:
  - exact failure class, evidence path, proposed tiling/streaming architecture, host/device-transfer risk, and whether an ADR is required.

Dependencies and parallelism:

- Depends on S1 successful 24 h 3 km and S2 PASS.
- Can run partly in parallel with S4/S5 tooling after S1, but final S3 comparison depends on S4 conversion if S4 owns common output format.

### M7-S4 - Post-processing and Operational Products

Objective: produce operational outputs that are useful outside the model loop: hourly NetCDF-compatible files, Zarr stores, station-point extracts, and public-facing JSON summaries.

Non-goals: public website design, ML correction, GRIB publication unless cheap.

Wall-time estimate: 18-30 h.

File ownership:

- May modify: `src/gpuwrf/postprocess/diagnostics.py`, `src/gpuwrf/postprocess/output_writers.py`, `src/gpuwrf/postprocess/station_extract.py`, `scripts/m7_postprocess_cycle.py`, `tests/test_m7_postprocess.py`, `artifacts/m7/postprocess/**`, `data/ops/m7/products/**`.
- Must not modify: forecast drivers except through stable output interfaces.

Acceptance:

- `artifacts/m7/postprocess/product_manifest.json`:
  - `cycle_id`, `input_output_paths`, `products`, `variables`, `units`, `grid`, `chunks`, `compression`, `checksums`, `retention_class`.
- `artifacts/m7/postprocess/diagnostics_inventory.json`:
  - must include U10, V10, 10 m wind speed/direction, T2, Q2/RH2 if available, accumulated precip, surface pressure/MSLP if available, cloud/radiation diagnostics if M6/M5 evidence supports them.
- `artifacts/m7/postprocess/station_extract_manifest.json`:
  - station registry path, station count, variable count, interpolation method, output parquet/CSV path.
- Output format:
  - v0 must write NetCDF-like hourly domain files and cycle-level Zarr.
  - GRIB2 is post-v0 unless a sprint proves a robust writer and variable table.

Dependencies and parallelism:

- Can start after S1 6 h smoke output.
- Final acceptance depends on S1 24 h and S3 only for 1 km products.
- Feeds S5 verification and S7 monitoring.

### M7-S5 - Operational Verification Against Stations and Gen2

Objective: make the operational claim falsifiable. This sprint owns station verification and the METplus-equivalent decision.

Non-goals: training ML correction, changing station QC rules, widening observation networks.

Wall-time estimate: 24-36 h.

File ownership:

- May modify: `src/gpuwrf/validation/ops_verification.py`, `src/gpuwrf/validation/station_obs.py`, `scripts/m7_verify_operational.py`, `tests/test_m7_ops_verification.py`, `artifacts/m7/verification/**`.
- May read but not modify: Gen2 station artifacts and manifests.
- Must not modify: forecast driver, post-processing writer internals except via documented product manifest.

Acceptance:

- `artifacts/m7/verification/observation_source_manifest.json`:
  - candidate sources, selected sources, station counts, variables, freshness, license/access notes, trust masks, exclusions.
  - minimum sources: `/home/enric/src/canairy_meteo/Gen2/manifests/high_quality_station_registry_v1.json` (246 stations in this worktree), AEMET 106-station manifest, GRAFCAN 57-station manifest, and Gen2 station cube paths.
- `artifacts/m7/verification/metplus_decision.md`:
  - decision: custom METplus-equivalent first, METplus adapter post-v0 unless the manager requires METplus proper.
  - rationale must cite install/runtime burden and prove score equivalence for RMSE/bias/count.
- `artifacts/m7/verification/ops_scores.json`:
  - schema: `cycle_id`, `domain`, `station_sample`, `lead_hours`, per variable `gpu_vs_gen2_rmse`, `gen2_vs_obs_rmse`, `gpu_vs_obs_rmse`, `bias`, `n`, `status`.
  - binding v0 gate for U10, V10, T2 when observations are available: `gpu_vs_gen2_rmse <= gen2_vs_obs_rmse` per variable and lead (+6/+12/+24). Q2/RH2 is reported as partial until humidity observation coverage is proven.
- `artifacts/m7/verification/scorecard_summary.md`:
  - short status: `GREEN`, `PARTIAL`, `BLOCKED`, or `FAIL`.

Dependencies and parallelism:

- Depends on S4 product manifest and S1/S3 outputs.
- Can develop observation loaders after S0.
- Blocks S8 closeout. If observations are unavailable, M7 may close only as `PROVISIONAL_3KM_PIPELINE`, not operationally validated.

### M7-S6 - Restart and Crash Recovery

Objective: prove that daily operations are idempotent and recoverable, including a minimal `wrfrst` compatibility matrix or explicit deviation.

Non-goals: full WRF restart bitwise parity, multi-cycle data assimilation.

Wall-time estimate: 18-30 h.

File ownership:

- May modify: `src/gpuwrf/io/restart.py`, `src/gpuwrf/ops/recovery.py`, `scripts/m7_restart_smoke.py`, `scripts/m7_recover_cycle.py`, `tests/test_m7_restart_recovery.py`, `artifacts/m7/restart/**`.
- Must not modify: state layout without ADR. If new State leaves are required, stop and request ADR/contract amendment.

Acceptance:

- `artifacts/m7/restart/wrfrst_compatibility_matrix.json`:
  - rows for `read_gen2_wrfrst`, `write_project_restart`, `write_wrf_compatible_wrfrst`, `continue_project_restart`, `restart_vs_continuous_delta`, each with `status`, `variables`, `missing_fields`, `deviation`, `artifact_paths`.
  - Note: Gen2 namelists currently use `restart_interval = 100000`, so M7 must intentionally generate restart files for this proof.
- `artifacts/m7/restart/restart_continuity.json`:
  - 12 h continuous vs 6 h + restart + 6 h comparison for core fields and surface variables; thresholds and pass/fail.
- `artifacts/m7/restart/idempotency.json`:
  - cycle lock behavior, rerun behavior, atomic output publication, partial-output cleanup, crash injection points, recovery status.
- `artifacts/m7/restart/cold_start_compile_cache.json`:
  - cold compile time, warm cache time, cache path, hash keys, failure fallback.

Dependencies and parallelism:

- Can start after S1 produces a 12 h run path.
- Does not block S4/S5 development, but blocks S8 closeout.

### M7-S7 - Monitoring, Alerting, and Ops Dashboard Hooks

Objective: make the single-machine v0 operable without watching logs manually.

Non-goals: public polished dashboard, multi-user service hardening, cloud deployment.

Wall-time estimate: 12-24 h.

File ownership:

- May modify: `src/gpuwrf/ops/monitoring.py`, `src/gpuwrf/ops/status_api.py`, `scripts/m7_monitor_cycle.py`, `scripts/m7_emit_status.py`, `tests/test_m7_monitoring.py`, `artifacts/m7/monitoring/**`, `data/ops/m7/status/**`.
- Must not modify Gen2 FastAPI code in `/home/enric/src/canairy_meteo/Gen2/web/**` during M7 unless a separate cross-repo contract is approved.

Acceptance:

- `artifacts/m7/monitoring/ops_status_schema.json`:
  - states: `WAITING_AIFS`, `AIFS_LATE`, `RUNNING`, `POSTPROCESSING`, `VERIFYING`, `PUBLISHED`, `FAILED`, `STALE_PUBLISHED`.
  - includes timestamps, cycle_id, current step, last good cycle, wall-time budget, error class, artifact index path.
- `artifacts/m7/monitoring/alert_policy.json`:
  - alert classes for AIFS late, GPU OOM, compile timeout, transfer violation, verification fail, stale station observations, disk low.
- `artifacts/m7/monitoring/status_snapshot_examples.json`:
  - at least one healthy cycle and one injected-failure cycle.
- A local dashboard/data hook is enough for v0: `/mnt/data/wrf_gpu2/operational/m7/status/latest.json`.

Dependencies and parallelism:

- Can start after S0 contract.
- Final live examples depend on S1/S4/S5/S6.

### M7-S8 - Milestone Soak and Closeout

Objective: close M7 only with an evidence pack that a reviewer can audit without rerunning the whole system.

Non-goals: new features, post-close fixes, M8 packaging.

Wall-time estimate: 12-24 h after all required sprint artifacts exist; longer if a multi-day live soak is required.

File ownership:

- May modify: `.agent/decisions/MILESTONE-M7-CLOSEOUT.md`, `artifacts/m7/closeout/**`, `scripts/m7_build_closeout_index.py`, `tests/test_m7_closeout_index.py`.
- Must not modify: implementation code except closeout-index plumbing.

Acceptance:

- `artifacts/m7/closeout/proof_index.json`:
  - indexes every required artifact, checksum, producing command, status, reviewer report, and unresolved risk.
- `artifacts/m7/closeout/operational_soak.json`:
  - at least three successful cycles from pinned AIFS inputs, one of which should be live/latest if available at closeout time. If not possible, the manager must record why.
- `artifacts/m7/closeout/m7_exit_status.json`:
  - `three_km_operational_status`, `one_km_status`, `verification_status`, `restart_status`, `performance_status`, `m8_dispatch_recommendation`.
- `.agent/decisions/MILESTONE-M7-CLOSEOUT.md`:
  - concise manager closeout, proof links, risks, and M8 readiness decision.

Dependencies and parallelism:

- Serial after S1, S4, S5, S6, S7; includes S3 only if 1 km passed S2.
- Requires independent review per sprint lifecycle because milestone closeout is non-exempt.

## Section 3 - IC/BC Source

AIFS is the M7 source. This is already decided in `PROJECT_PLAN.md` item 11.6 and Gen2 uses it operationally.

How AIFS arrives:

- Historical/monthly surface archive: `/mnt/data/canairy_meteo/data/aifs_single/aifs_single_YYYYMM.nc`.
- Upstream archive for AIFS Single: `s3://dynamical-ecmwf-aifs-single/ecmwf-aifs-single-forecast/v0.1.0.icechunk`, opened anonymously in `us-west-2` by `/home/enric/src/canairy_meteo/scripts/data_acquisition/download_aifs_canary.py`.
- Live WRF forcing path: ECMWF open-data GRIB via `Gen2/scripts/prepare_aifs_pure_forcing.py` with `AIFS_OPENDATA_SOURCE=azure` by default in the Gen2 wrapper.
- Gen2 AIFS live readiness path: `Gen2/scripts/poll_live_18z_cycle_v1.py`, state at `/mnt/data/canairy_meteo/data/state/live_18z_state.json`, log at `/mnt/data/canairy_meteo/data/logs/live_18z_polling.log`.

Source variables for WRF-style forcing:

- Pressure levels: `u`, `v`, `t`, `q`, `z` at 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa.
- Surface: `10u`, `10v`, `2t`, `2d`, `sp`, `msl`, `skt`, `lsm`, `z`.
- Soil: `vsw`, `sot`, translated from AIFS 0-7 cm / 7-28 cm to WRF-facing 0-10 cm / 10-40 cm.
- For a 24 h v0 cycle, required forcing steps are 0, 6, 12, 18, 24 h. Gen2 WPS uses `interval_seconds = 21600`.

Ingest into model coordinates:

- v0 should reuse the Gen2 WPS/WPS-like transformation as an ingest oracle, not invent a new AIFS regridder in the first operational sprint.
- Gen2 WPS paths already produce `geo_em` and `met_em` files from the AIFS forcing bundle using Lambert projection metadata:
  - `map_proj = lambert`, `ref_lat = 28.3`, `ref_lon = -16.4`, `truelat1 = 25.0`, `truelat2 = 30.0`, `stand_lon = -16.4`.
  - geog root: `/mnt/data/canairy_meteo/artifacts/wps_geog/WPS_GEOG_LOW_RES`.
- M7-S1 loads those products or a native equivalent into `GridSpec` and `State`, with a proof object that checks field names, units, grid shape, terrain/static provenance, and checksums.

Initial vs boundary cadence:

- Initial state at t=0 comes from AIFS-derived metgrid/real-equivalent fields.
- Outer d01 lateral boundaries come from AIFS every 6 h and are linearly interpolated inside the timestep loop.
- d02 boundaries come from one-way d01 parent output/state, not from a synthetic Gen2 `wrfbdy_d02`.
- d03/d04/d05 boundaries, if enabled, come from one-way d02 parent state with `parent_time_step_ratio = 3`.

Failure mode:

- If AIFS is absent by 05:25 UTC, write `AIFS_LATE` and do not silently publish a fresh forecast.
- If the previous successful cycle is less than the configured freshness threshold, the system may publish it as `STALE_PUBLISHED` with explicit age; otherwise no public output.
- If AIFS GRIB is partially missing or fails the raw validation key count, block the run before GPU allocation and write `artifacts/m7/<cycle>/aifs_failure.json`.

## Section 4 - Operational Schedule

M7 v0 schedule:

- One daily 18Z cycle.
- Poll for AIFS readiness starting 01:25 UTC next day.
- Hard readiness timeout: 05:25 UTC.
- Launch GPU cycle immediately after readiness, subject to single-machine lock.
- Target publish window for 3 km v0: by 08:00 UTC when AIFS arrives before 05:25 UTC. This target is provisional until S1 wall-time evidence lands.

Wall-clock budget:

- 3 km 24 h cold-start target: <= 90 min including compile and initialization.
- 3 km 24 h warm-cache target: <= 45 min excluding AIFS download/WPS prep.
- Post-processing target: <= 15 min.
- Verification target: <= 15 min for station and Gen2 comparison.
- 1 km target, if enabled: <= 4 h total forecast wall time for d01/d02/d03; d04/d05 only if S2/S3 evidence says the total cycle still fits the publish window.

HBM/run residency:

- Do not rely on HBM persistence across daily runs. Each cycle cold-starts from disk/AIFS and allocates persistent GPU state.
- The high-frequency model state must remain device-resident during the forecast loop.
- JAX compile cache on disk is allowed and must be measured. Device HBM may be freed after each cycle.

Output path and retention:

- Operational root: `/mnt/data/wrf_gpu2/operational/m7/`.
- Cycle root: `/mnt/data/wrf_gpu2/operational/m7/cycles/<YYYYMMDD>_18z/`.
- Current pointer: `/mnt/data/wrf_gpu2/operational/m7/latest`.
- Status pointer: `/mnt/data/wrf_gpu2/operational/m7/status/latest.json`.
- Retention proposal:
  - raw hourly model files: 14 days,
  - Zarr/product outputs: 90 days,
  - proof JSON, scorecards, manifests: retained through M8,
  - large profiler dumps: retained until M7 closeout unless referenced by M8 docs.

## Section 5 - Nesting and Regridding

Gen2 domain geometry to match:

- L2 3 km chain: d01 9 km, d02 3 km.
- L3 1 km chain: d01 9 km, d02 3 km, d03 1 km Tenerife, d04 1 km Gran Canaria, d05 1 km La Palma.
- Gen2 `defaults.yaml` and recent namelists use:
  - `time_step = 18`,
  - `e_vert = 45`,
  - `e_we = 94, 160, 94, 70, 70`,
  - `e_sn = 60, 67, 76, 61, 58`,
  - `dx = 9000, 3000, 1000, 1000, 1000`,
  - `dy = 9000, 3000, 1000, 1000, 1000`,
  - `parent_id = 1, 1, 2, 2, 2`,
  - `i_parent_start = 1, 24, 52, 84, 9`,
  - `j_parent_start = 1, 20, 20, 10, 36`,
  - `parent_grid_ratio = 1, 3, 3, 3, 3`,
  - `parent_time_step_ratio = 1, 3, 3, 3, 3`,
  - `feedback = 0`,
  - `spec_bdy_width = 5`.

Scope decision:

- M7 v0 must implement one-way nesting only. This matches Gen2's `feedback = 0` and avoids a two-way JAX/XLA state update problem inside the operational milestone.
- M7 v0 must not implement dynamic nest creation. Domain shapes and parent-child maps are static compile-time structures.
- 3 km v0 is d01->d02.
- 1 km v0 is conditional and should prefer d03 Tenerife first, then d04/d05 siblings if memory and runtime permit. If the manager insists on "all Canary 1 km" before M8, S2 must prove all d03/d04/d05 fit in the same operational cycle.

JAX-XLA nesting issues:

- Parent/child time-step ratio 3 should be expressed as static nested scans, not runtime conditionals. M6-S1 already found that dynamic `lax.cond` radiation cadence caused a device-to-host predicate transfer; M7 must avoid repeating that pattern.
- Parent-to-child boundary interpolation should happen on device from parent state/output buffers.
- Child-to-parent feedback is out of scope for v0.
- Regridding proof must separate AIFS->d01, d01->d02, and d02->d03/d04/d05 errors.

## Section 6 - Verification Framework

Recommendation: custom METplus-equivalent first, METplus adapter later.

Rationale:

- M7's binding need is a small set of deterministic scores over known station sources, not the whole METplus ecosystem.
- Gen2 already has station truth and WRF point-shadow artifacts in local formats.
- A proper METplus install/format sprint is still useful, but it should not block the first v0 if the custom verifier writes transparent schemas and reproduces RMSE/bias/count.

Observation sources:

- Primary registry: `/home/enric/src/canairy_meteo/Gen2/manifests/high_quality_station_registry_v1.json` (246 high-quality stations in this worktree; sources include AEMET, Cabildo, La Palma Smart Island, GRAFCAN, WU, Puertos, IAC, Windguru, Cabezo, and others).
- AEMET manifest: `/home/enric/src/canairy_meteo/Gen2/manifests/aemet_stations_canary_v1.yaml` (106 stations; live endpoint `/api/observacion/convencional/todas`, station endpoint `/api/observacion/convencional/datos/estacion/{idema}`).
- AEMET data: `/home/enric/src/canairy_meteo/Gen2/artifacts/datasets/aemet_stations/*_daily_v1.parquet`.
- GRAFCAN manifest: `/home/enric/src/canairy_meteo/Gen2/manifests/grafcan_sitcan_stations_v1.yaml` (57 stations; API requires `GRAFCAN_API_KEY`; variables include temp, RH, pressure, wind, precip, and solar on 23/57).
- Existing station cube: `/home/enric/src/canairy_meteo/Gen2/artifacts/datasets/station_benchmark_cube_v3_candidate.parquet` (6,935,614 rows, 193 stations, 2016-04-21 to 2026-05-07 in this worktree).
- Existing WRF matched skill products: `wrf_case_bank_skill_0_24h_matched_v1.parquet` and `wrf_case_bank_skill_25_72h_matched_v1.parquet`.

KPIs:

- Required at +6, +12, +24 h for 24 h v0: U10 RMSE/bias, V10 RMSE/bias, wind-speed RMSE, wind-direction MAE for non-calm cases, T2 RMSE/bias, station count, and matched-case count.
- Q2/RH2: report where station humidity coverage and model diagnostic mapping are proven; otherwise `PARTIAL_Q2`.
- Precipitation: diagnostic unless event sample is adequate.
- +48 h: schema must support it; closeout requires it only if M7 extends 3 km horizon beyond 24 h.

Binding gate:

- For observed U10, V10, and T2: GPU-vs-Gen2 RMSE must be <= Gen2-vs-observation RMSE at each required lead and variable over the selected station sample.
- S7/Tier-4 style tolerances are reported as sanity checks, not as a loosening factor.
- If observations are unavailable or stale, M7 cannot claim operational validation.

Public dashboard:

- M7 v0 produces local status JSON and scorecard artifacts.
- Public dashboard integration belongs to M8 unless the manager opens a separate UI sprint.

## Section 7 - Restart and Crash Recovery

M7 must produce a restart compatibility matrix, not hand-wave restarts.

Minimum matrix rows:

- Read Gen2 `wrfrst`: likely `NOT_AVAILABLE` for current backfills because Gen2 namelists set `restart_interval = 100000`; if no files exist, prove with inventory.
- Write project-native restart: required.
- Continue from project-native restart: required.
- Write WRF-compatible `wrfrst`: preferred, but can be `DEVIATION` if schema gaps are explicit.
- Restart continuity: required 12 h continuous vs 6 h + restart + 6 h.

Daily idempotency:

- Cycle IDs are immutable: `<YYYYMMDD>_18z`.
- A cycle has a lock file, a `run_manifest.json`, a status JSON, and an atomic publish step.
- Rerunning an already complete cycle should either no-op or write a new attempt directory without corrupting `latest`.
- Partial cycle cleanup must preserve failure proof objects.

Cold-start and compile:

- JAX compile time is an operational cost and must be in every run summary.
- Compile cache is allowed, but the closeout must show both cold and warm behavior.
- HBM persistence across cycles is not a recovery strategy.

## Section 8 - M7-Specific Risks

| Risk | Impact | Mitigation |
|---|---|---|
| AIFS late or missing | No operational cycle | S0/S1 readiness gate, explicit `AIFS_LATE`, stale-last-good only with age label, no silent fallback |
| AIFS/WPS dependency hides native ingest gaps | Operational path depends on WRF tooling | Accept for v0 if provenance is exact; open post-v0 native regrid ADR only after M7 |
| RTX 5090 32 GB fails 1 km compile or HBM | 1 km cannot be claimed | S2 memory/compile audit before S3; deviation doc or tiling sprint |
| JAX nested scans introduce hidden D2H transfers | Violates constitution/performance targets | Static scans; transfer audit required for S1/S2/S3 |
| HBM persistence across daily runs is assumed | Crash/reboot loses state | Cold-start each cycle; persist restarts/output to disk |
| Single-machine SRE burden | Daily forecast becomes manual babysitting | S7 statuses, alert policy, idempotent recovery |
| Observation source gaps | Verification not falsifiable | S5 observation manifest; no operational validation claim if stale/unavailable |
| Gen2 CPU denominator unfairness | Speedup claim invalid | Reuse M6-S2a CPU denominator extractor; domain-scoped comparison only |
| Surface/Noah/RRTMG debt from M6 | T2/Q2/precip misleading | S0 blocks or marks variables provisional based on M6-S3/M5 radiation evidence |
| Output format creep | Post-processing consumes milestone | NetCDF-like + Zarr required; GRIB public feed post-v0 unless cheap |

## Section 9 - Estimated M7 Wall-Time

| Sprint | Estimate | Critical path role |
|---|---:|---|
| M7-S0 prologue | 12-18 h | serial gate |
| M7-S1 3 km daily pipeline | 36-60 h | critical |
| M7-S2 1 km memory audit | 12-20 h | parallel after S1 state factory; gates S3 |
| M7-S3 1 km pipeline or deviation | 36-72 h if PASS; 12-18 h if FAIL | conditional critical path |
| M7-S4 post-processing | 18-30 h | parallel after S1 smoke; final after S1 |
| M7-S5 verification | 24-36 h | critical for operational claim |
| M7-S6 restart/recovery | 18-30 h | parallel after S1 12 h |
| M7-S7 monitoring/alerting | 12-24 h | parallel after S0 |
| M7-S8 closeout | 12-24 h plus any soak wait | serial close |

3 km-only critical path:

- S0 -> S1 -> max(S4, S5, S6, S7 finalization) -> S8.
- Expected wall: 5-8 working days with one or two implementation workers plus required reviews, assuming M6 closes cleanly.

3 km + 1 km critical path:

- S0 -> S1 -> S2 -> S3, with S4/S5/S6/S7 overlapping after S1.
- Expected wall: 8-12 working days if 1 km compiles and no tiling is needed.

Hard blockers:

- M6-S2a/ADR-011 absent or failed.
- M6-S2 fails zero-transfer/24 h d02.
- M6-S3 surface layer is absent and the manager requires T2/Q2 as binding.
- S1 cannot ingest live AIFS into real terrain/static GridSpec.
- S5 cannot obtain station observations.

## Section 10 - Exit Criteria for M7 Close

Before M8 dispatch, the following must exist:

- `artifacts/m7/prologue/m6_inheritance_gate.json` with no blocking M6 debts.
- `artifacts/m7/prologue/operational_contract.json`.
- At least three pinned 18Z cycle roots under `/mnt/data/wrf_gpu2/operational/m7/cycles/`, each with `run_manifest.json`, model outputs, status, and proof objects.
- `artifacts/m7/3km/daily_3km_run_summary.json` for at least one successful full 24 h 3 km cycle.
- `artifacts/m7/3km/transfer_audit.json` proving zero timestep-loop transfers.
- `artifacts/m7/3km/gen2_same_cycle_comparison.json`.
- `artifacts/m7/memory/rtx5090_1km_memory_audit.json`.
- If 1 km passed: `artifacts/m7/1km/nested_run_summary.json` and `gen2_l3_comparison.json`.
- If 1 km failed: `artifacts/m7/1km/one_km_deviation_or_tiling_plan.md`, with no public 1 km claim.
- `artifacts/m7/postprocess/product_manifest.json`.
- `artifacts/m7/verification/observation_source_manifest.json`.
- `artifacts/m7/verification/ops_scores.json` with GREEN/PARTIAL/BLOCKED/FAIL statuses for U10, V10, T2, Q2/RH2, and precip.
- `artifacts/m7/restart/wrfrst_compatibility_matrix.json`.
- `artifacts/m7/restart/restart_continuity.json`.
- `artifacts/m7/monitoring/ops_status_schema.json` and `status/latest.json` examples.
- `artifacts/m7/closeout/proof_index.json`.
- `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` with independent review status and M8 recommendation.

M7 cannot close as "Canary operational v0" if:

- the 3 km run is not repeatable,
- verification lacks observations for U10/V10/T2 and no explicit human-approved blocker is recorded,
- any timestep-loop host/device transfer remains,
- AIFS ingest provenance is not machine-readable,
- restart/recovery is untested,
- or the closeout proof index cannot be audited without reading raw logs.
