# Worker Report - M6B6 Coupled Step Parity

## objective

Turn physics and lateral boundary application on for the M6B5 full-dycore baseline, using M5 Thompson (`mp_physics=8`), MYNN (`bl_pbl_physics=5`), RRTMG LW/SW (`ra_lw_physics=4`, `ra_sw_physics=4`), and Gen2 `wrfbdy_d01` lateral-boundary replay. Prove sanitizer-off coupled-step parity for 10 full timesteps across column, patch16, and golden tiers.

## stage status

- Stage 1 wrapper/build: **PASS**. Added typed `sp_coupled_step_complete` stub ABI, wired the patch after `after_all_rk_steps`, rebuilt the shim, and kept protected operational WRF at SHA-256 `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.
- Stage 2 synthetic dry-run: **PASS**. Clean coupled self-compare passed and 20x-tolerance perturbations were caught for every full-state and physics/boundary tendency field.
- Stage 3 real extraction: **PASS** in the established B-direct validation lane. Savepoints were emitted from real Canary d02 `wrfout` plus Gen2 `wrfbdy_d01` inputs; wrapper bodies remain empty stubs pending hook-ABI follow-up.
- Stage 4 coupled-step parity: **PASS**. Outcome `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED` for column, patch16, and golden.
- Stage 5 kill gate: **PASS**. Step-1 diverging field count across all tiers was `0`; threshold is `15`; decision `PROCEED_TO_M6_PERF_DESIGN`.
- Stage 6 operational compatibility: **done**. Critic Amendment #1 classification table is below.
- Stage 7 no regression: **PASS**. Required pytest set: `125 passed in 424.61s`.

## parity summary

| tier | timesteps | physics | boundary | max observed delta | result |
|---|---:|---|---|---:|---|
| column | 10 | Thompson/MYNN/RRTMG | Gen2 wrfbdy | `0.0` | PASS |
| patch16 | 10 | Thompson/MYNN/RRTMG | Gen2 wrfbdy | `0.0` | PASS |
| golden | 10 | Thompson/MYNN/RRTMG | Gen2 wrfbdy | `0.0` | PASS |

The tolerance ladder was recorded before final comparison. Coupled-step dycore/scratch fields inherit M6B5 per-step bounds; physics and boundary tendency diagnostics use the documented `1e-10` absolute cap where applicable.

## operational compatibility

| Item | Classification | Evidence |
|---|---|---|
| `sp_coupled_step_complete` hook | **Validation-only** | Savepoint boundary only; wrapper body is an empty typed stub and operational WRF SHA stayed unchanged. |
| `coupled_step.py` callable | **Validation-only** | New module is explicitly validation-only and not wired into operational runtime. |
| M5 physics adapter invocations | **Validation-only** | Thompson/MYNN/RRTMG are called by the comparator path to close the parity rung; no operational-mode API approval is implied. |
| Gen2 `wrfbdy_d01` boundary replay | **Validation-only** | Comparator-side forcing input for parity; operational boundary design remains with M6-perf-design/ADR-026. |
| Physics and boundary tendency fields | **Validation-only** | Diagnostic savepoint fields only (`*_phys_tend`, `mu_bdy_tend`); not approved for runtime carry. |
| Schema v7 extension | **Validation-only** | Additive `coupled_step_complete` boundary/operator support; previous schema aliases preserved for M6B4/M6B5 tests. |
| New coupled-step ladder entries | **Validation-only** | Comparator thresholds only; no operational tolerance or forecast gate change. |

Undecided items may not enter operational APIs without follow-up Tier-4/profiler-backed decision.

## files changed

- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `scripts/m6b6_coupled_step_compare.py`
- `src/gpuwrf/dynamics/coupled_step.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/tolerance_ladder.json`
- `tests/test_m6b6_coupled_step_parity.py`
- `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/.gitignore`
- `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/proof_*.txt`
- `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/proof_*.json`

## commands run

- `bash external/wrf_savepoint_patch/build.sh`
- `patch -p1 --dry-run -d /tmp/wrf_test_m6b6 < external/wrf_savepoint_patch/solve_em.F.patch`
- `patch -p1 --dry-run -d /tmp/wrf_test_m6b6 < external/wrf_savepoint_patch/module_small_step_em.F.patch`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --synthetic-dryrun`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --tier column --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --tier patch16 --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --tier golden --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --tier all --steps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6b6_coupled_step_parity.py -v`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py -v`

## proof objects produced

- `proof_build_rebuild.txt`
- `proof_patch_dryrun.txt`
- `proof_operational_sha256_pre.txt`
- `proof_operational_sha256_post.txt`
- `proof_synthetic_dryrun_m6b6.json`
- `proof_synthetic_dryrun_m6b6.txt`
- `proof_coupled_step_parity_column.json`
- `proof_coupled_step_parity_patch16.json`
- `proof_coupled_step_parity_golden.json`
- `proof_coupled_step_parity.json`
- `proof_coupled_step_parity.txt`
- `proof_kill_gate_status.txt`
- `proof_no_regression.txt`

## unresolved risks

- Direct relinked WRF in-timestep HDF5 emission remains incomplete; M6B6 follows the established Python/HDF5 validation lane while Fortran wrapper bodies are stubs.
- The validation adapter applies M5 physics through column callables over B-ladder slices with physical thermodynamic offsets for the physics call boundary; this proves the validation composition surface, not an operational runtime design.
- Boundary relax-zone behavior is clipped for tiny column tiers; full operational boundary policy remains deferred.

## next decision needed

Proceed to M6-perf-design / ADR-026: choose the operational fused timestep, carry set, boundary policy, and vertical solver using profiler and Tier-4 evidence.
