# Worker Report

## Summary:

Built a focused Step-1 substage WRF-vs-JAX comparator and localized the
remaining T/P-family divergence after live-nest base initialization closure.

Final verdict:
`STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.

The proof uses a disposable, env-gated WRF instrumentation under
`/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/**`; production
`src/gpuwrf/**` was not changed.

## Files Changed

- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization.md`
- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`

## Commands Run

- `python -m py_compile proofs/v014/step1_t_p_operator_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_t_p_operator_localization.py`
- `python -m json.tool proofs/v014/step1_t_p_operator_localization.json >/tmp/step1_t_p_operator_localization.validated.json`
- `git diff -- src/gpuwrf`
- `tcsh ./compile em_real` in `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/WRF`
- `WRFGPU2_STEP1_TP_LOCALIZATION=1 mpirun --oversubscribe -np 28 run/wrf.exe`

## Proof Objects Produced

- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization.md`
- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`
- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`

## Result

The first strict and first material T/P-family mismatch is:

- surface: `after_rk_addtend_before_small_step_prep`
- RK stage: `1`
- field: `T_STATE`

At that same boundary, the largest material residuals are tendency-family fields
such as `PH_TEND`, `RW_TEND`, `PH_TENDF`, `T_TEND`, and `T_TENDF`. The next
emitted RK1 `small_step_prep` work arrays then match for `T_WORK` and `P_WORK`
with max_abs `0.0`, so the current proof does not justify continuing acoustic
or pressure-refresh debugging.

The final accepted strict comparison still diverges, with first field `T` and
largest field `P` max_abs `1561.2503728885986`.

## Unresolved Risks

Only two early WRF substage boundaries were emitted. If the RK1 stage-entry
state/tendency boundary is fixed and residuals remain, the next truth surface
should extend into acoustic/pre-finish substeps.

The proof-local CPU loader still mirrors production live-nest init because
`build_replay_case` is GPU-only at `State.zeros`.

## Next Decision

Open a focused boundary sprint between WRF `first_rk_step_part1/part2` and JAX
`_physics_step_forcing`. Start with `T_STATE`, `T_TENDF/T_TEND`, `PH_TENDF`,
`PH_TEND`, `RW_TEND`, and dry state/carry handoff. Do not continue acoustic
debugging yet.
