# Worker Report - M6B5 Full Dycore Step Parity

## objective

Compose the M6B4 `acoustic_loop` with WRF's RK3 outer loop for 10 full timesteps, with physics and lateral-boundary application disabled, and prove sanitizer-off validation parity across column, patch16, and golden tiers.

## stage status

- Stage 1 wrapper/build: **PASS**. Added typed `sp_dycore_step_complete` stub ABI, wired it after `Runge_Kutta_loop`, rebuilt the shim, and kept protected operational WRF at SHA-256 `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.
- Stage 2 synthetic dry-run: **PASS**. Clean dycore-step self-compare passed and 20x-tolerance perturbations were caught for every full-state field.
- Stage 3 real extraction: **PASS**. Emitted 10 `dycore_step_complete` snapshots per tier from real Canary d02 wrfout slices using the established Python/HDF5 validation lane; wrapper bodies remain empty stubs pending hook-ABI follow-up.
- Stage 4 composition parity: **PASS**. Outcome `SIXTH-DYCORE-STEP-PARITY-ACHIEVED` for column, patch16, and golden.
- Stage 5 kill gate: **PASS**. Step-1 diverging field count across all tiers was `0`; threshold is `15`; decision `PROCEED_TO_M6B6`.
- Stage 6 operational compatibility: **done**. Critic Amendment #1 classification table is below.
- Stage 7 no regression: **PASS**. Required pytest set: `120 passed in 350.12s`.

## parity summary

| tier | timesteps | RK stages/step | acoustic substeps/RK | max observed delta | result |
|---|---:|---:|---:|---:|---|
| column | 10 | 3 | 10 | `6.938893903907228e-18` (`w`, step 6) | PASS |
| patch16 | 10 | 3 | 10 | `4.440892098500626e-16` (`w`, step 3) | PASS |
| golden | 10 | 3 | 10 | `1.1102230246251565e-16` (`w`, step 3) | PASS |

The per-timestep ladder was recorded before final comparison: M6B4 per-substep tolerance x 10 acoustic substeps x 3 RK stages x 10 timesteps = 300x. No tolerance was tuned after seeing the comparison result.

Physics and boundary namelist contract was held validation-side as: `mp_physics=0`, `bl_pbl_physics=0`, `ra_lw_physics=0`, `ra_sw_physics=0`, `cu_physics=0`, `sf_sfclay_physics=0`, `sf_surface_physics=0`, `specified=false`.

## operational compatibility

| Item | Classification | Evidence |
|---|---|---|
| `sp_dycore_step_complete` hook | **Validation-only** | Savepoint boundary only; wrapper body is an empty typed stub and operational WRF SHA stayed unchanged. |
| `dycore_step.py` callable | **Validation-only** | Module is explicitly validation-only and is not wired into operational runtime. |
| New ladder entries (per-timestep tolerances) | **Validation-only** | Comparator thresholds only; 300x geometric-growth bound documented in `tolerance_ladder.json`. |
| Schema v6 extension | **Validation-only** | Additive savepoint boundary/operator support; no operational state API change. |
| Full-state timestep snapshot fields (`mu/mut/mudf/muts/muave/ww/theta/ph_tend/u/v/w/ph/p/t_2ave`) | **Validation-only** | Required for parity diagnosis; no field is approved for operational carry by this sprint. |
| RK3-over-acoustic composition interface | **Undecided** | Matches WRF-shaped validation cadence, but operational fusion/carry decisions remain deferred to ADR-026 and M6B6+ design work. |

Undecided items may not enter operational APIs without a follow-up Tier-4-backed decision.

## files changed

- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `scripts/m6b5_dycore_step_compare.py`
- `src/gpuwrf/dynamics/dycore_step.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/tolerance_ladder.json`
- `tests/test_m6b5_dycore_step_parity.py`
- `.agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/.gitignore`
- `.agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/proof_*.txt`
- `.agent/sprints/2026-05-25-m6b5-full-dycore-step-parity/proof_*.json`

## commands run

- `sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe`
- `bash external/wrf_savepoint_patch/build.sh`
- `patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/solve_em.F.patch`
- `patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/module_small_step_em.F.patch`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b5_dycore_step_compare.py --synthetic-dryrun`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b5_dycore_step_compare.py --tier column --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b5_dycore_step_compare.py --tier patch16 --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b5_dycore_step_compare.py --tier golden --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b5_dycore_step_compare.py --tier all --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6b4_acoustic_recurrence_parity.py tests/test_m6b5_dycore_step_parity.py -v`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py -v`

## proof objects produced

- `proof_build_rebuild.txt`
- `proof_patch_dryrun.txt`
- `proof_operational_sha256_pre.txt`
- `proof_operational_sha256_post.txt`
- `proof_synthetic_dryrun_m6b5.json`
- `proof_synthetic_dryrun_m6b5.txt`
- `proof_dycore_step_parity_column.json`
- `proof_dycore_step_parity_patch16.json`
- `proof_dycore_step_parity_golden.json`
- `proof_dycore_step_parity.json`
- `proof_dycore_step_parity.txt`
- `proof_kill_gate_status.txt`
- `proof_no_regression.txt`

## unresolved risks

- Direct relinked WRF in-timestep HDF5 emission remains incomplete; M6B5 follows the B1-B4 validation lane over real Canary d02 wrfout slices while Fortran wrapper bodies are stubs.
- This proves RK3-over-acoustic validation composition with physics and lateral-boundary application disabled, not M6B6 full coupled behavior.
- The validation loop carries WRF scratch/full-state fields for diagnosis only; no operational carry approval is implied.

## next decision needed

Proceed to M6B6: physics on, boundary on, sanitizer off, 10 full timesteps.
