# Review: V0.14 Step-1 RK1 Source Boundary

Verdict: `STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.

objective: split WRF first_rk_step_part1/part2, rk_tendency, and rk_addtend_dry/spec_bdy_dry against JAX `_physics_step_forcing` and dry tendency construction before `small_step_prep`.

files changed:
- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary.md`
- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md`

commands run:
- `python -m py_compile proofs/v014/step1_rk1_source_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_source_boundary.py`
- `python -m json.tool proofs/v014/step1_rk1_source_boundary.json >/tmp/step1_rk1_source_boundary.validated.json`
- `git diff -- src/gpuwrf`
- `cp -a --reflink=auto prior Step-1 WRF scratch into /mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary`
- `tcsh ./compile em_real (scratch WRF, conda wrf-build toolchain)`
- `WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY=1 WRFGPU2_STEP1_TP_LOCALIZATION=1 WRFGPU2_SOURCE_SAVE_BOUNDARY=1 mpirun --oversubscribe -np 28 ./wrf.exe`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_rk1_source_boundary.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_rk1_source_boundary.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/wrf_truth`

unresolved risks:
- The after-rk_tendency hook reused the prior patch-window source-save emitter; because the first material source-boundary classification occurs earlier at first_rk_step_part1 T_STATE, this did not block the verdict.
- No production source fix was applied; the next decision must split WRF first_rk_step_part1 internals against the JAX physics adapter output before choosing a source representation.

next decision: Split WRF first_rk_step_part1 internals against the JAX physics adapter output for the named state field.
