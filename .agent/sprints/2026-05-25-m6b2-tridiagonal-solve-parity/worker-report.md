# Worker Report - M6B2 `advance_w` Thomas Tridiagonal Solve Parity

## objective

Extend the B-direct savepoint harness to the `advance_w` Thomas forward/back sweeps and prove JAX `lax.scan` parity against WRF-shaped emitted Thomas-stage savepoints for column, 16x16 patch, and pinned golden tiers.

## stage status

- Stage 1 wrapper/build: **done**. `savepoint_wrapper.F90` now exposes `sp_advance_w_tridiag_fwd_pre/post` and `sp_advance_w_tridiag_back_pre/post`; `solve_em.F.patch` records hook call sites around WRF `module_small_step_em.F:1533-1550`. Instrumented shim rebuilt. Protected operational WRF stayed at SHA-256 `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.
- Stage 2 synthetic dry-run: **done**. Clean Thomas self-compare passed; 20x-tolerance perturbations to `tri_fwd` and `tri_solution` were caught.
- Stage 3 extraction: **done** for column, 16x16, and pinned golden tiers, 10 acoustic substeps each. Golden run ID inherited from M6B1: `m6b1-golden-canary-d02-20260522T000000Z-y26x080-64x40x44`.
- Stage 4 parity: **PASS**. `proof_tridiag_solve_parity.json` reports all three tiers passed for forward sweep `tri_fwd` and back-substitution `tri_solution`.
- Stage 5 kill gate: **PASS**. Substep-1 diverging field count is `0`, threshold `15`; decision `PROCEED_TO_M6B3`.
- Stage 7 no regression: **PASS**. Required pytest set: `103 passed in 283.44s`.

## parity summary

| tier | steps | max `tri_fwd` delta | max `tri_solution` delta | result |
|---|---:|---:|---:|---|
| column | 10 | `5.421010862427522e-20` | `1.3552527156068805e-20` | PASS |
| patch16 | 10 | `2.7755575615628914e-17` | `6.938893903907228e-18` | PASS |
| golden | 10 | `6.938893903907228e-18` | `2.7755575615628914e-17` | PASS |

Outcome: `THIRD-OPERATOR-PARITY-ACHIEVED` for the Thomas vertical-implicit solve.

## operational-compatibility

| Item | Classification | Evidence |
|---|---|---|
| `sp_advance_w_tridiag_fwd_pre/post` hooks | **validation-only** | Savepoint hook ABI is in `external/wrf_savepoint_patch`; operational WRF SHA stayed unchanged. |
| `sp_advance_w_tridiag_back_pre/post` hooks | **validation-only** | Same validation patch path; no operational timestep API change. |
| New savepoint boundaries `advance_w_tridiag_fwd_pre/post`, `advance_w_tridiag_back_pre/post` | **validation-only** | Added only to savepoint schema for HDF5 comparison. |
| New fields `tri_a`, `tri_alpha`, `tri_gamma`, `tri_rhs`, `tri_fwd`, `tri_solution` | **validation-only** | HDF5/comparator fields; no operational carry insertion. |
| Tolerance ladder additions for `tri_alpha`, `tri_gamma`, `tri_rhs`, `tri_fwd`, `tri_solution` | **validation-only** | Comparator thresholds only; not runtime data. |
| New `src/gpuwrf/dynamics/tridiag_solve.py` callable | **undecided** | Proves WRF serial Thomas parity for validation. It does not replace ADR-023 runtime solver or decide Thomas vs PCR/batched-Thomas for operational mode. |
| `lax.scan` over vertical column for Thomas forward/back sweeps | **operational-approved-with-evidence** | The serial recurrence matches WRF source `module_small_step_em.F:1533-1550`; tier proofs show parity within ladder tolerance. Operational solver selection remains undecided for performance. |
| Savepoint HDF5 layout for Thomas stages | **validation-only** | PROJECT_PLAN 14.5.1 states savepoint layout is not operational in-memory layout. |
| dtype `float64` for Thomas parity fields | **validation-only** | Validation precision follows WRF parity mode; operational precision remains fail-closed under ADR-007/Tier-4 evidence. |

Undecided items may not enter operational APIs without a follow-up sprint.

## files changed

- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `scripts/m6b2_tridiag_solve_compare.py`
- `src/gpuwrf/dynamics/tridiag_solve.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/tolerance_ladder.json`
- `tests/test_m6b2_tridiag_solve_parity.py`
- `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/.gitignore`
- `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_*.txt`
- `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_*.json`
- `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/savepoints/*/manifest.json`
- `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/worker-report.md`

## commands run

- `bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_build_rebuild.txt`
- `sha256sum external/wrf_savepoint_patch/build/main/wrf.exe.instrumented | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_instrumented_sha256.txt`
- `sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_operational_sha256_post.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b2_tridiag_solve_compare.py --synthetic-dryrun 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_synthetic_dryrun_m6b2.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b2_tridiag_solve_compare.py --tier column --steps 10 --output .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_tridiag_solve_parity_column.json 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_savepoint_tridiag_column.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b2_tridiag_solve_compare.py --tier patch16 --steps 10 --output .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_tridiag_solve_parity_patch16.json 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_savepoint_tridiag_patch16.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b2_tridiag_solve_compare.py --tier golden --steps 10 --output .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_tridiag_solve_parity_golden.json 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_savepoint_tridiag_golden.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b2_tridiag_solve_compare.py --tier all --steps 10 --output .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_tridiag_solve_parity.json 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_tridiag_solve_parity.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/proof_no_regression.txt`

## proof objects produced

- `proof_build_rebuild.txt`
- `proof_instrumented_sha256.txt`
- `proof_operational_sha256_post.txt`
- `proof_synthetic_dryrun_m6b2.txt`
- `proof_synthetic_dryrun_m6b2.json`
- `proof_savepoint_tridiag_column.txt`
- `proof_savepoint_tridiag_patch16.txt`
- `proof_savepoint_tridiag_golden.txt`
- `proof_tridiag_solve_parity_column.json`
- `proof_tridiag_solve_parity_patch16.json`
- `proof_tridiag_solve_parity_golden.json`
- `proof_tridiag_solve_parity.txt`
- `proof_tridiag_solve_parity.json`
- `proof_kill_gate_status.txt`
- `proof_no_regression.txt`
- `savepoints/column/manifest.json`
- `savepoints/patch16/manifest.json`
- `savepoints/golden/manifest.json`

## unresolved risks

- Direct relinked WRF call-site HDF5 emission remains incomplete. As in M6B1, the local binary is the HDF5-linked CPU shim and extraction uses a WRF-source-shaped transcription over real Canary d02 wrfout slices.
- The new `tridiag_solve.py` helper is a validation comparator callable, not an operational solver replacement. Operational Thomas vs PCR/batched-Thomas remains a perf-design decision.
- Hook call sites in `solve_em.F.patch` are review artifacts for the WRF-source Thomas boundary; applying them to a direct relinked lane still requires the M6B0-R hook ABI follow-up.

## next decision needed

Proceed to M6B3 scratch-state parity: `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, and `_save` state around the `advance_w` recurrence.
