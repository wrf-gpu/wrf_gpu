# Worker Report

## Summary:

Built a focused WRF solve_em pre-`first_rk_step_part1` handoff comparator and
localized the current Step-1 `T_STATE` residual to the JAX live-nest Step-1
loader/carry boundary.

Final verdict:
`STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.

The proof uses disposable, env-gated WRF instrumentation under
`/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/**`; production
`src/gpuwrf/**` was not changed.

## Files Changed

- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_pre_part1_handoff.json`
- `proofs/v014/step1_pre_part1_handoff.md`
- `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`

## Commands Run

- `git log -1 --oneline --decorate`
- `git merge-base --is-ancestor 588686d6 HEAD`
- `cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/WRF /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF`
- `cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/run /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/run`
- `tcsh ./compile em_real` in the scratch WRF tree with the `wrf-build` toolchain
- `WRFGPU2_STEP1_PRE_PART1_HANDOFF=1 WRFGPU2_STEP1_PRE_PART1_HANDOFF_ROOT=/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth mpirun --oversubscribe -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/step1_pre_part1_handoff.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_pre_part1_handoff.py`
- `python -m json.tool proofs/v014/step1_pre_part1_handoff.json >/tmp/step1_pre_part1_handoff.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects Produced

- `proofs/v014/step1_pre_part1_handoff.json`
- `proofs/v014/step1_pre_part1_handoff.md`
- `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`

## Result

- WRF `T_STATE` delta from `after_step_increment` to
  `before_first_rk_step_part1_call`: max_abs `0.0`, RMSE `0.0`.
- WRF solve_em pre-call vs prior part1-entry `T_STATE` continuity: max_abs
  `0.0`.
- WRF pre-call `T_STATE` vs raw JAX live-nest input state
  (`State.theta - 300 K`): max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- Full-vs-perturbation theta was explicitly checked:
  `WRF_T_STATE_IS_PERTURBATION_THETA`.

Therefore the current `T_STATE` residual is already present in the JAX
live-nest Step-1 loader/carry state before `_physics_step_forcing`.

## Unresolved Risks

The JAX loader/carry internals are not yet split. No production source fix was
applied.

## Next Decision

Open a JAX loader/carry split sprint for `T_STATE`: compare raw wrfinput/native
state, live-nest child-state construction, carry construction, and haloed
step-entry state against the WRF solve_em pre-call truth.
