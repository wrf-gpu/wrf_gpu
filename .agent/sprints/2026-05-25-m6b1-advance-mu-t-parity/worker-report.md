# Worker Report — M6B1 `advance_mu_t` Parity

## objective

Extend the M6B0-R savepoint harness to `advance_mu_t`, emit column / 16x16 / golden savepoints, and demonstrate first sanitizer-off JAX-vs-WRF-shaped parity for MU/MUDF/MUTS/MUAVE/ww/theta/ph_tend.

## stage status

- Stage 1 wrapper/build: **done**. `savepoint_wrapper.F90` now exposes typed `sp_advance_mu_t_pre/post` hooks and `solve_em.F.patch` records call-site hooks around CPU `advance_mu_t`. Instrumented shim rebuilt; protected operational WRF stayed at SHA-256 `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.
- Stage 2 synthetic dry-run: **done**. MU and MUTS clean self-compare passed; 20x-tolerance perturbations were caught.
- Stage 3 extraction: **done** for column, 16x16, and pinned golden tiers, 10 acoustic substeps each. Golden run ID: `m6b1-golden-canary-d02-20260522T000000Z-y26x080-64x40x44`.
- Stage 4 parity: **PASS**. `proof_advance_mu_t_parity.json` reports all three tiers passed for `mu`, `mudf`, `muts`, `muave`, `ww`, `theta`, and `ph_tend`.
- Stage 5 kill gate: **PASS**. Substep-1 diverging field count is `0`, threshold `15`; decision `PROCEED_TO_M6B2`.
- Stage 6 no regression: **PASS**. Required pytest set: `90 passed in 294.29s`.

## files changed

- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `scripts/m6b1_advance_mu_t_compare.py`
- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/tolerance_ladder.json`
- `tests/test_m6b1_advance_mu_t_parity.py`
- `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/.gitignore`
- `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/proof_*.txt`
- `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/proof_*.json`
- `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/savepoints/*/manifest.json`
- `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/worker-report.md`

## commands run

- `bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/proof_build_rebuild.txt`
- `sha256sum external/wrf_savepoint_patch/build/main/wrf.exe.instrumented | tee .../proof_instrumented_sha256_v2.txt`
- `sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe | tee .../proof_operational_sha256_post.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b1_advance_mu_t_compare.py --synthetic-dryrun 2>&1 | tee .../proof_synthetic_dryrun_m6b1.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b1_advance_mu_t_compare.py --tier column --steps 10 --output .../proof_advance_mu_t_parity_column.json 2>&1 | tee .../proof_savepoint_advance_mu_t_column.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b1_advance_mu_t_compare.py --tier patch16 --steps 10 --output .../proof_advance_mu_t_parity_patch16.json 2>&1 | tee .../proof_savepoint_advance_mu_t_patch16.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b1_advance_mu_t_compare.py --tier golden --steps 10 --output .../proof_advance_mu_t_parity_golden.json 2>&1 | tee .../proof_savepoint_advance_mu_t_golden.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b1_advance_mu_t_compare.py --tier all --steps 10 --output .../proof_advance_mu_t_parity.json 2>&1 | tee .../proof_advance_mu_t_parity.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py -v 2>&1 | tee .../proof_no_regression.txt`

## proof objects produced

- `proof_build_rebuild.txt`
- `proof_instrumented_sha256_v2.txt`
- `proof_operational_sha256_post.txt`
- `proof_synthetic_dryrun_m6b1.txt`
- `proof_synthetic_dryrun_m6b1.json`
- `proof_savepoint_advance_mu_t_column.txt`
- `proof_savepoint_advance_mu_t_patch16.txt`
- `proof_savepoint_advance_mu_t_golden.txt`
- `proof_advance_mu_t_parity_column.json`
- `proof_advance_mu_t_parity_patch16.json`
- `proof_advance_mu_t_parity_golden.json`
- `proof_advance_mu_t_parity.txt`
- `proof_advance_mu_t_parity.json`
- `proof_kill_gate_status.txt`
- `proof_no_regression.txt`
- `savepoints/column/manifest.json`
- `savepoints/patch16/manifest.json`
- `savepoints/golden/manifest.json`

## unresolved risks

- Direct relinked WRF call-site emission is still not completed. As in M6B0-R, the local `wrf.exe.instrumented` is an HDF5-linked CPU shim, and extraction uses a WRF-source-shaped Python/JAX transcription over real Canary d02 wrfout slices. This proves the comparator/helper path, not an in-timestep dump from operational WRF.
- `ww`, `theta_tend`, `mu_tend`, and `ph_tend` are not available in wrfout history at the `advance_mu_t` boundary. The fixture initializes unavailable boundary-local scratch/tendency fields consistently and records this through the manifest/proof notes.
- The new helper intentionally does not alter production `acoustic_wrf.py` runtime semantics; M6B2 should continue the B-direct ladder rather than treating this as a runtime dycore fix.

## next decision needed

Proceed to M6B2 tridiagonal solve parity, or require a separate infrastructure sprint to replace the shim/transcription extraction lane with direct relinked WRF call-site HDF5 emission before further B-direct operator proofs.
