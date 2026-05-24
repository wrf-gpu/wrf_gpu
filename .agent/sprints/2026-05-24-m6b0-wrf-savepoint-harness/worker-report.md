# Worker Report — M6B0 WRF Savepoint Harness

## objective

Build the first WRF small-step savepoint schema, extractor, JAX comparator, perturbation negative test, proof bundle, and ADR-025 decision update for the B-direct savepoint-first path.

## stage status

- Stage 1 instrumented WRF build: **partial / isolated**. `external/wrf_savepoint_patch/build.sh` creates `external/wrf_savepoint_patch/build/wrf.exe.instrumented`, hook registry, and Fortran hook-anchor patch artifact. The protected operational WRF binary remained byte-identical. Important caveat: this is an isolated wrapper/extractor harness, not a relinked Fortran `wrf.exe` that directly emits savepoints.
- Stage 2 schema + I/O: **done**. Format chosen: `npz-bundle-v1`. `SavepointMetadata`, `VariableMetadata`, and `Savepoint` validate schema, dtype, shape, stagger, units, provenance, run IDs, WRF commit/hash, namelist hash, dt, domain, map factors, vertical grid, RK stage, and acoustic substep.
- Stage 3 Canary d02 savepoint bundles: **done for column and 16x16 patch**. Source was real `wrfinput_d02` under `/mnt/data/canairy_meteo/runs/wrf_l3/20260429_18z_l3_24h_20260524T204451Z/`. Bundles cover steps 1, 2, 5, and 10 across 11 required boundaries. Tier-3 full d02 deferred with storage estimate.
- Stage 4 comparator + perturbation negative: **done**. `+1e-6` theta perturbation is detected and parity is refused.
- Stage 5 first operator parity: **done** for coefficient construction on column and patch16. Clean comparator reports zero max-abs deltas for all coefficient fields because M6B0 savepoints use the same JAX coefficient formula as the current harness oracle; M6B1 must replace this with true Fortran `calc_coef_w` dumps.
- Stage 6 ADR-025: **done**. ADR remains DRAFT, but M6B0 decisions are filled in.
- Stage 7 no regression: **done**. Contract test list passed.
- Stage 8 report: **done**.

## files changed

- `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md`
- `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_*.txt`
- `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/proof_first_operator_parity.json`
- `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/savepoints/**`
- `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/worker-report.md`
- `external/wrf_savepoint_patch/**`
- `scripts/m6b0_wrf_savepoint_extract.py`
- `scripts/m6b0_jax_savepoint_compare.py`
- `scripts/m6b0_perturbation_negative_test.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/savepoint_io.py`
- `tests/test_m6b0_savepoint_schema.py`
- `tests/test_m6b0_perturbation_negative.py`
- `tests/test_m6b0_coefficient_parity.py`

No edits were made to `src/gpuwrf/dynamics/acoustic_wrf.py`.

## commands run

- `bash external/wrf_savepoint_patch/build.sh | tee .../proof_instrumented_build.txt`
- `sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe external/wrf_savepoint_patch/build/wrf.exe.instrumented >> .../proof_instrumented_build.txt`
- `pytest tests/test_m6b0_savepoint_schema.py -v | tee .../proof_schema_roundtrip.txt`
- `python scripts/m6b0_wrf_savepoint_extract.py --tier column --steps 10 --output .../savepoints/column/ | tee .../proof_savepoint_bundle_column.txt`
- `python scripts/m6b0_wrf_savepoint_extract.py --tier patch16 --steps 10 --output .../savepoints/patch16/ | tee .../proof_savepoint_bundle_patch.txt`
- `python scripts/m6b0_perturbation_negative_test.py --savepoint .../savepoints/column/ | tee .../proof_comparator_perturbation_caught.txt`
- `python scripts/m6b0_jax_savepoint_compare.py` via inline combined column+patch proof writer for `proof_comparator_parity_clean.txt` and `proof_first_operator_parity.json`
- Contract no-regression pytest list with `taskset -c 0-3 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ... | tee .../proof_no_regression.txt`

## proof objects produced

- `proof_instrumented_build.txt`: stable WRF SHA-256 unchanged before/after; isolated instrumented wrapper path and hash recorded.
- `proof_schema_roundtrip.txt`: 2 passed.
- `proof_savepoint_bundle_column.txt`: 44 savepoints, 133545 bytes.
- `proof_savepoint_bundle_patch.txt`: 44 savepoints, 7634917 bytes.
- `proof_savepoint_storage_estimate.txt`: full d02 linear estimate 0.291 GiB for current boundary set; full Tier-3 deferred.
- `proof_comparator_parity_clean.txt`: coefficient construction parity passes on column and patch16.
- `proof_comparator_perturbation_caught.txt`: perturbation caught; comparator reports failed fields and refuses parity.
- `proof_comparator_tolerance_ladder.txt`: predeclared per-field tolerances and rationale.
- `proof_first_operator_parity.json`: per-field max-abs deltas, tolerances, and pass/fail for column and patch16.
- `proof_no_regression.txt`: 58 passed, 1 warning in 282.85s.

## transfer audit

Comparator I/O happens before the JAX coefficient call and no timestep loop is executed by the comparator. Proof JSON records `h2d_d2h_inside_timestep_loop_bytes=0`. No GPU performance claim is made.

## unresolved risks

- The WRF-side artifact is not yet a true relinked Fortran `wrf.exe` with direct savepoint writes. M6B1 should either implement the real Fortran hook build or have the reviewer explicitly accept the wrapper/extractor as sufficient for M6B0.
- First-operator clean parity is self-consistency against the current JAX coefficient builder, not independent Fortran `calc_coef_w` parity. This is useful for the comparator and negative-test machinery, but not enough to prove WRF coefficient equivalence.
- The `npz-bundle-v1` format is fine for M6B0 column/patch fixtures. HDF5 may become preferable when full scratch-state fields and full-domain tiers arrive.
- Full d02 storage estimate is linear from patch16 and may understate true size once all WRF scratch arrays are emitted without high compression.

## next decision needed

Reviewer/manager should decide whether M6B1 starts by converting the isolated hook registry into a true Fortran-emitting instrumented `wrf.exe`, or whether the current wrapper/extractor is accepted as the M6B0 bootstrap and M6B1 proceeds directly to true `calc_coef_w` Fortran dump parity.
