# Worker Report

## Summary:

Built a focused internal WRF `first_rk_step_part1` truth/comparator and
localized the first material Step-1 `T_STATE` residual upstream of
`first_rk_step_part1`.

Final verdict:
`STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`.

The proof uses disposable, env-gated WRF instrumentation under
`/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/**`; production
`src/gpuwrf/**` was not changed.

## Files Changed

- `proofs/v014/step1_part1_physics_state_mutation.py`
- `proofs/v014/step1_part1_physics_state_mutation.json`
- `proofs/v014/step1_part1_physics_state_mutation.md`
- `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`

## Commands Run

- `git log -1 --oneline --decorate`
- `git merge-base --is-ancestor c18795af HEAD`
- `cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/WRF /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/WRF`
- `tcsh ./compile em_real` in the scratch WRF tree with the `wrf-build` toolchain
- `WRFGPU2_STEP1_PART1_PHYSICS_STATE_MUTATION=1 WRFGPU2_STEP1_PART1_PHYSICS_STATE_MUTATION_ROOT=/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth mpirun --oversubscribe -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/step1_part1_physics_state_mutation.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part1_physics_state_mutation.py`
- `python -m json.tool proofs/v014/step1_part1_physics_state_mutation.json >/tmp/step1_part1_physics_state_mutation.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects Produced

- `proofs/v014/step1_part1_physics_state_mutation.json`
- `proofs/v014/step1_part1_physics_state_mutation.md`
- `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`
- `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth`

## Result

- `part1_entry_before_init_zero_tendency` `T_STATE` vs JAX live-nest step-entry
  state: max_abs `5.490173101425171`, RMSE `1.9175184863907806`.
- Largest WRF internal `T_STATE` delta from part1 entry across all instrumented
  `first_rk_step_part1` boundaries: max_abs `0.0`.
- `part1_exit` retains the same residual against JAX carry/state surfaces.

Therefore WRF `first_rk_step_part1` is not the source of this `T_STATE`
divergence. The first material mismatch is already present at the call entry.

## Unresolved Risks

The upstream source before `first_rk_step_part1` entry is still not localized.
No production source fix was applied.

## Next Decision

Move upstream to the live-nest/WRF handoff immediately before
`first_rk_step_part1` entry. The next proof should compare the WRF call-site
entry state against JAX carry/state construction and the accepted Step-1 loader
state before selecting any fix.
