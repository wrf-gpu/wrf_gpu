# Worker Report — M6B0-R Real Fortran Savepoint Emission

## objective

Close the M6B0 scaffold gap by adding HDF5 savepoint I/O, fail-closed synthetic validation, CPU-path `calc_coef_w` extraction tiers, golden-slice proofing, and first sanitizer-off JAX-vs-WRF coefficient comparison.

## stage status

- Stage 0 preflight: **done**. Operational `wrf.exe` stayed at SHA-256 `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`; canonical WRF source head verified as `115e5756f98ee2370d62b6709baac6417d8f7338`; `env_wrf_gpu.sh` sourced before the HDF5/NVHPC build path.
- Stage 1 wrapper/build: **partial with explicit caveat**. `external/wrf_savepoint_patch/build.sh` compiles a Fortran/HDF5 CPU savepoint emission shim at `external/wrf_savepoint_patch/build/main/wrf.exe.instrumented` and records `solve_em.F.patch`, `configure.wrf.patch`, `dyn_em/savepoint_wrapper.F90`, and `namelist.savepoint`. It did **not** complete a direct relinked full WRF `wrf.exe`; reviewer must decide whether M6B1 first finishes direct WRF call-site emission.
- Stage 2 synthetic dry-run: **done**. HDF5 write/read, clean comparator, +1e-3 perturbation rejection, schema-version mismatch, and tamper detection passed.
- Stage 3 extraction: **done for M6B0-R CPU-path harness tiers**. Column, 16x16 patch, and pinned golden slice savepoints were emitted from real Canary d02 WRF output. Golden run ID: `m6b0r-golden-canary-d02-20260522T000000Z-y26x080-64x40x44`.
- Stage 4 tolerance ladder: **done**. Machine-readable JSON ladder committed at `src/gpuwrf/validation/tolerance_ladder.json`; proof copy captured as `proof_tolerance_ladder.yaml`; comparator decisions use the ladder.
- Stage 5 first coefficient parity: **done, outcome `PARITY-DEFECT-LOCALIZED`**. JAX `calc_coef_w` equivalent diverges from WRF-shaped coefficients on all tiers with sanitizer off. Worst observed deltas include column `a=259.6639474310799`, `alpha=0.9950025857696718`, `gamma=0.4794380030755466` against `1e-11` absolute thresholds.
- Stage 6 ADR-025: **done**. Renamed to `ADR-025-wrf-savepoint-bdirect-port-PROPOSED.md` and resolved HDF5 format, wrapper instrumentation, schema, tolerance ladder, golden ordering, and CPU/GPU path decision.
- Stage 7 no regression: **done**. Required pytest set passed: `84 passed in 277.23s`.
- Stage 8 report: **done**.

## files changed

- `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-PROPOSED.md`
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/.gitignore`
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_*.txt`
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_*.json`
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints/*/manifest.json`
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/worker-report.md`
- `external/wrf_savepoint_patch/build.sh`
- `external/wrf_savepoint_patch/configure.wrf.patch`
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
- `external/wrf_savepoint_patch/namelist.savepoint`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `scripts/m6b0_wrf_savepoint_extract.py`
- `scripts/m6b0_jax_savepoint_compare.py`
- `scripts/m6b0_perturbation_negative_test.py`
- `scripts/m6b0r_wrf_savepoint_extract.py`
- `scripts/m6b0r_golden_slice_extract.py`
- `scripts/m6b0r_jax_vs_wrf_compare.py`
- `scripts/m6b0r_synthetic_dryrun.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/savepoint_io.py`
- `src/gpuwrf/validation/tolerance_ladder.json`
- `tests/test_m6b0_coefficient_parity.py`
- `tests/test_m6b0_savepoint_schema.py`
- `tests/test_m6b0r_savepoint_hdf5.py`

## commands run

- `bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .../proof_build_log.txt`
- `python scripts/m6b0r_synthetic_dryrun.py 2>&1 | tee .../proof_synthetic_dryrun.txt`
- `python scripts/m6b0r_wrf_savepoint_extract.py --tier column --steps 10 2>&1 | tee .../proof_savepoint_column.txt`
- `python scripts/m6b0r_wrf_savepoint_extract.py --tier patch16 --steps 10 2>&1 | tee .../proof_savepoint_patch16.txt`
- `python scripts/m6b0r_golden_slice_extract.py --steps 10 2>&1 | tee .../proof_savepoint_golden.txt`
- `python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all 2>&1 | tee .../proof_real_coefficient_parity.txt`
- `pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v 2>&1 | tee .../proof_no_regression.txt`

## proof objects produced

- `proof_preflight.txt`
- `proof_build_log.txt`
- `proof_instrumented_sha256.txt`
- `proof_operational_unchanged.txt`
- `proof_synthetic_dryrun.txt`
- `proof_synthetic_dryrun.json`
- `proof_savepoint_column.txt`
- `proof_savepoint_patch16.txt`
- `proof_savepoint_golden.txt`
- `proof_golden_slice_runid.txt`
- `proof_storage_actual.txt`
- `proof_tolerance_ladder.yaml`
- `proof_tolerance_ladder_applied.txt`
- `proof_real_coefficient_parity.txt`
- `proof_real_coefficient_parity.json`
- `proof_no_regression.txt`

## unresolved risks

- Direct full-WRF relinked emission remains incomplete. The owned patch artifacts and Fortran module are present, but `wrf.exe.instrumented` is an HDF5-linked CPU emission shim rather than a fully relinked WRF binary. This is the main reviewer decision for M6B1.
- The Stage 3 savepoints are derived from real Canary d02 WRF output plus a WRF-source-shaped Python reproduction of `calc_coef_w`, not arrays emitted from inside the running WRF timestep loop.
- Stage 5 localized a coefficient mismatch but did not fix JAX operator code, per sprint contract.
- HDF5 savepoint files are kept in the worktree for local inspection but ignored by git; manifests and proof summaries are committed.

## next decision needed

Reviewer/manager should decide whether M6B1 starts by completing direct relinked WRF `solve_em.F` call-site emission, or accepts the current HDF5 harness and proceeds to fix/port `calc_coef_w` against the localized `a/alpha/gamma` defect.
