# Manager Closeout

## Outcome

The sprint is closed as a validated JAX-loader localization proof.

Final verdict:
`STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.

The v0.14 Step-1 `T_STATE` divergence is not produced by WRF solve_em pre-call
state mutation, WRF `first_rk_step_part1`, or a full-vs-perturbation theta
mapping error. It is already present in the raw JAX live-nest Step-1
state/carry before `_physics_step_forcing`.

## Proof Objects

- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_pre_part1_handoff.json`
- `proofs/v014/step1_pre_part1_handoff.md`
- `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and pending-memory artifacts only.
No production model source changed in this sprint.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_pre_part1_handoff.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_pre_part1_handoff.py`
- `python -m json.tool proofs/v014/step1_pre_part1_handoff.json >/tmp/step1_pre_part1_handoff.manager.validated.json`
- `git diff -- src/gpuwrf`

The rerun reproduced the verdict and left `src/gpuwrf` unchanged.

## Key Numbers

- WRF `T_STATE` delta from `after_step_increment` to
  `before_first_rk_step_part1_call`: max_abs `0.0`, RMSE `0.0`.
- WRF solve_em pre-call vs prior part1-entry `T_STATE` continuity: max_abs
  `0.0`.
- WRF pre-call `T_STATE` vs raw JAX live-nest input state
  (`State.theta - 300 K`): max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- WRF pre-call `T_STATE` vs JAX haloed step-entry state
  (`State.theta - 300 K`): max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- Theta semantic conclusion: `WRF_T_STATE_IS_PERTURBATION_THETA`.

## Scope Changes

None. WRF instrumentation was disposable scratch only, CPU-only, and env-gated.
No TOST, Switzerland, FP32, memory source work, GPU, or production source edit
was performed.

## Lessons

Do not continue upstream WRF solve_em instrumentation for this `T_STATE`
residual. The fastest rigorous path is now a JAX loader/carry split.

## Next Sprint

Open `v014-step1-jax-loader-tstate`: split raw wrfinput/native state,
live-nest child-state construction, carry construction, and haloed step-entry
state for `T_STATE` against WRF solve_em pre-call truth. The proof gate is an
exact loader stage or a narrow performance-compatible fix with before/after
Step-1 proof.
