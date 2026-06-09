# Worker Report

## Summary:

Built a focused Step-1 RK1 source-boundary comparator and localized the first
material mismatch before acoustic/small-step work.

Final verdict:
`STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.

The proof uses disposable, env-gated WRF instrumentation under
`/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/**`; production
`src/gpuwrf/**` was not changed.

## Files Changed

- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary.md`
- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md`

## Commands Run

- `python -m py_compile proofs/v014/step1_rk1_source_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_source_boundary.py`
- `python -m json.tool proofs/v014/step1_rk1_source_boundary.json >/tmp/step1_rk1_source_boundary.validated.json`
- `git diff -- src/gpuwrf`
- `cp -a --reflink=auto` from the prior Step-1 WRF scratch tree into `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary`
- `tcsh ./compile em_real` in the scratch WRF tree with the `wrf-build` toolchain
- `WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY=1 WRFGPU2_STEP1_TP_LOCALIZATION=1 WRFGPU2_SOURCE_SAVE_BOUNDARY=1 mpirun --oversubscribe -np 28 ./wrf.exe`

## Proof Objects Produced

- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary.md`
- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md`
- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/wrf_truth`

## Result

The first localized material source-boundary mismatch is:

- surface: `after_first_rk_step_part1`
- field: `T_STATE`
- WRF vs JAX operational carry max_abs `5.490173101425171`, RMSE `1.9175184863907806`
- WRF vs `_physics_step_forcing.state` max_abs `5.490142455570492`, RMSE `1.9174736017582765`

The prior RK1 `small_step_prep` continuity check remains exact for `T_WORK` and
`P_WORK`, both max_abs `0.0`. Therefore the next debug target is inside WRF
`first_rk_step_part1` and the corresponding JAX physics adapter output, not
`small_step_prep`, acoustic, FP32, memory, Switzerland, or TOST.

## Unresolved Risks

The sprint localizes the boundary but does not yet split the internal calls
inside `first_rk_step_part1`. No production source fix was applied.

## Next Decision

Open a focused `first_rk_step_part1` internals sprint. The proof must identify
the exact WRF internal mutation or forcing leaf that changes `T_STATE`, and then
compare the corresponding JAX physics adapter state/tendency path before any
source representation or production fix is chosen.
