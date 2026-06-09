# V0.14 Source/Save Boundary Handoff

## objective

Close the V0.14 instrumentation gap by locating and emitting the first d02 step-6000 WRF boundary where current-step source/save-family leaves required by JAX `DryPhysicsTendencies` exist. Preserve a consistent native-state/source boundary, then either run a strict same-input JAX proof or emit a precise blocked verdict.

Verified starting point: branch `worker/gpt/v013-close-manager`, HEAD `b41c840e841cb9efcf9bd14767e3e93b13762583`; `b41c840e` is the current HEAD and therefore an ancestor of the worktree state.

## files changed

- `proofs/v014/source_save_boundary_hook.py`
- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/source_save_boundary_hook.md`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_sources.py`
- `proofs/v014/same_input_single_rk_parity_sources.json`
- `proofs/v014/same_input_single_rk_parity_sources.md`
- `.agent/reviews/2026-06-09-v014-source-save-boundary.md`

Disposable scratch only:

- `/mnt/data/wrf_gpu2/v014_source_save_boundary/WRF/dyn_em/solve_em.F`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/source_save_output/source_save_after_rk_tendency_d2_step_6000_is_1_ie_23_js_1_je_17.txt`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/source_save_output/source_save_after_rk_tendency_d2_step_6000_is_1_ie_23_js_18_je_33.txt`

No production `src/gpuwrf/**` edits; `git diff -- src/gpuwrf` is empty.

## commands run

- `git rev-parse --abbrev-ref HEAD`
- `git rev-parse HEAD`
- `git merge-base --is-ancestor b41c840e HEAD`
- `rsync -a /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/WRF /mnt/data/wrf_gpu2/v014_source_save_boundary/`
- `rsync -a /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/run_case3 /mnt/data/wrf_gpu2/v014_source_save_boundary/`
- `diff -u --label a/dyn_em/solve_em.F --label b/dyn_em/solve_em.F /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/WRF/dyn_em/solve_em.F /mnt/data/wrf_gpu2/v014_source_save_boundary/WRF/dyn_em/solve_em.F > proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu tcsh ./compile em_real`
- `timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 WRFGPU2_SOURCE_SAVE_BOUNDARY=1 WRFGPU2_SOURCE_SAVE_BOUNDARY_ROOT=/mnt/data/wrf_gpu2/v014_source_save_boundary/source_save_output WRFGPU2_SOURCE_SAVE_BOUNDARY_GRID=2 WRFGPU2_SOURCE_SAVE_BOUNDARY_START_STEP=6000 WRFGPU2_SOURCE_SAVE_BOUNDARY_END_STEP=6000 mpirun --oversubscribe -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/source_save_boundary_hook.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/source_save_boundary_hook.py`
- `python -m json.tool proofs/v014/source_save_boundary_hook.json >/tmp/source_save_boundary_hook.validated.json`
- `python -m py_compile proofs/v014/same_input_single_rk_parity_sources.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_sources.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_sources.json >/tmp/same_input_single_rk_parity_sources.validated.json`
- `git diff -- src/gpuwrf`

## proof objects produced

- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/source_save_boundary_hook.md`
- `proofs/v014/same_input_single_rk_parity_sources.json`
- `proofs/v014/same_input_single_rk_parity_sources.md`

Hook verdict: `SOURCE_SAVE_BOUNDARY_HOOK_READY`.

Boundary emitted: after `first_rk_step_part1`, `first_rk_step_part2`, and `rk_tendency`; before `relax_bdy_dry`, `rk_addtend_dry`, `spec_bdy_dry`, `small_step_prep`, and `advance_uv`.

Dry source/save leaves present: `ru_tendf`, `rv_tendf`, `rw_tendf`, `ph_tendf`, `t_tendf`, `mu_tendf`, `h_diabatic`, `u_save`, `v_save`, `w_save`, `ph_save`, `t_save`.

Native dry state preservation: exact on overlap against the full pre-RK step-entry reference, with `220609` compared values and worst max abs `0.0`.

Patch width: 17x17 source patch with one conservative 8-cell-halo-valid mass cell.

Strict same-input verdict: `SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.

## unresolved risks

- The strict JAX comparison did not run. The current repo has no proof-only loader/wrapper that builds a full same-boundary `State`, `OperationalCarry`, `OperationalNamelist`, `GridSpec`, `DycoreMetrics`, and `DryPhysicsTendencies` from WRF-emitted fields only.
- Full-domain same-boundary promoted carry leaves are still not emitted: `t_2ave`, `ww`, `mudf`, `muave`, `muts`, `ph_tend`, `mu_save`, `ww_save`, `rthraten`, active physics carry, and boundary leaves.
- Existing post-RK/pre-halo truth is not full-domain/full-vertical enough for a strict single-RK comparison: it is K1 mass/U/V and kstag 0/1 W/PH, while the source/save hook is a 17x17 patch.
- `moist_adv_opt=1` and `scalar_adv_opt=1`; only P_QV `moist_old` is initialized and emitted at this boundary, and `scalar_old` is not valid here. A final-stage scalar-limiter proof needs a consistent old-field strategy or a narrower dry-only wrapper.
- Tile-overlap duplicate records exist on the staggered V interface: `V_SOURCE.RV_TEND`, `V_SOURCE.RV_TENDF`, and `V_SOURCE.V_SAVE` differ across the two overlapping records while `V_NEW` and `V_OLD` match. The current proof does not use this as a full-domain truth surface; any future wrapper must define tile ownership or halo-exchange semantics explicitly.

## next decision needed

Choose whether V0.14 should build the missing proof-only full-domain source/save loader/wrapper, or first expand WRF instrumentation to emit the additional same-boundary carry/boundary fields and a full-domain/full-vertical truth surface.
