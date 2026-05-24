# Worker Report - M6B0-R Relink Completion

## objective

Complete the WRF relink lane by copying canonical WRF into `external/wrf_savepoint_patch/source_copy/`, applying the savepoint patches to the copy, building a `-DWRF_SAVEPOINT` `wrf.exe`, protecting the operational binary SHA, and comparing column-tier calc_coef_w artifacts.

## outcome

**PARTIAL: relink build completed; real timestep-loop HDF5 savepoint emission is not completed.**

The relinked WRF binary was built at `external/wrf_savepoint_patch/source_copy/main/wrf.exe` with SHA-256 `dbfd8e3d4acf5c0660f81a6ed6fd76e2eb8a345f5f05ad5d11d30a7c100cf05b`. The operational WRF binary remained unchanged at `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.

The inherited M6B0-R `solve_em.F.patch` exposes zero-argument hooks and `savepoint_wrapper.F90` has empty hook bodies. Therefore the relinked binary cannot emit field-bearing `calc_coef_w` HDF5 savepoints from inside the timestep loop without a follow-up patch-interface change that passes arrays and metadata into the Fortran wrapper. The relinked-lane HDF5 files produced in this sprint are generated from the existing M6B0-R WRF-source-shaped Python extraction and are explicitly marked as partial proof, not real WRF timestep-loop emission.

## stages

- Stage 1 source copy + patch: **done with artifact repair**. The original patches were malformed unified diffs; they were repaired as valid patch artifacts, then applied to `source_copy/`. `source_copy/` is ignored.
- Stage 2 compile: **done**. `./compile em_real -j 4` completed after sourcing `env_wrf_gpu.sh`. Because `/bin/csh` is absent, copied csh shebangs were rewritten to the available `tcsh` inside `source_copy/` only. The canonical `small_step_gpu2.o` was preserved to avoid an unrelated NVHPC 26.3 relink failure on `__pgi_ieee_is_finite_dev_r4`.
- Stage 3 relinked run/savepoints: **blocked for real emission**. `mpirun -np 1 ./wrf.exe` starts the relinked binary and reaches WRF namelist open, but the Fortran hooks cannot write arrays. Column HDF5 files were generated only through the existing Python extraction path.
- Stage 4 shim-vs-relinked: **pass for generated column files**. `calc_coef_w_pre_step001` field deltas are zero for `theta`, `dz_m`, and `mut`.
- Stage 5 JAX-vs-relinked-lane: **PARITY-DEFECT-LOCALIZED**. Column deltas match M6B0-R: `a=259.6639474310799`, `alpha=0.9950025857696718`, `gamma=0.4794380030755466`.
- Stage 6 no regression: **pass**. `84 passed in 284.61s`.

## files changed

- `external/wrf_savepoint_patch/.gitignore`
- `external/wrf_savepoint_patch/build_relinked.sh`
- `external/wrf_savepoint_patch/configure.wrf.patch`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `scripts/m6b0r_relinked_extract.py`
- `scripts/m6b0r_relinked_vs_shim_compare.py`
- `.agent/sprints/2026-05-24-m6b0r-relink-completion/proof_*.txt`
- `.agent/sprints/2026-05-24-m6b0r-relink-completion/proof_*.json`
- `.agent/sprints/2026-05-24-m6b0r-relink-completion/worker-report.md`

## commands run

- `bash external/wrf_savepoint_patch/build_relinked.sh 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_build_relink.txt`
- `python scripts/m6b0r_relinked_extract.py --tier column --steps 10 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_relinked_run.txt`
- `python scripts/m6b0r_relinked_vs_shim_compare.py --tier column 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_shim_vs_relinked_delta.txt`
- `python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier column --savepoint-root external/wrf_savepoint_patch/savepoints/relinked --output .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_jax_vs_relinked_calc_coef_w.json 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_jax_vs_relinked_calc_coef_w.txt`
- `pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_no_regression.txt`

## proof objects produced

- `proof_source_copy_sha256.txt`
- `proof_patch_apply.txt`
- `proof_build_relink.txt`
- `proof_compile.txt`
- `proof_relinked_sha256.txt`
- `proof_operational_unchanged_post_relink.txt`
- `proof_relinked_run.txt`
- `proof_relinked_run_binary_probe.txt`
- `proof_relinked_savepoints_listing.txt`
- `proof_shim_vs_relinked_delta.txt`
- `proof_shim_vs_relinked_delta.json`
- `proof_jax_vs_relinked_calc_coef_w.txt`
- `proof_jax_vs_relinked_calc_coef_w.json`
- `proof_no_regression.txt`

## unresolved risks

- Real WRF timestep-loop HDF5 savepoint emission remains blocked until `solve_em.F.patch` passes field arrays and indices into wrapper routines and `savepoint_wrapper.F90` writes HDF5 from those arguments.
- The column "relinked" savepoints are not emitted by relinked WRF; they are relinked-lane generated files from the prior Python extraction/oracle path.
- The WRF binary probe starts under MPI but was not converted into a full golden-slice WRF integration run in this sprint because the current Fortran hooks could not emit the required proof files even if the model ran.

## next decision needed

Manager should decide whether to open a follow-up sprint to change the savepoint wrapper ABI and solve_em call sites so real WRF timestep-loop HDF5 emission is possible, or to accept the relink build proof as sufficient and continue defect analysis against the existing Python-shaped oracle.
