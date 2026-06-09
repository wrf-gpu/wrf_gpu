# Manager Closeout

## Outcome

The sprint is closed as a validated upstream-localization proof.

Final verdict:
`STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`.

The v0.14 Step-1 `T_STATE` divergence is not produced by WRF
`first_rk_step_part1`. The full residual is already present at that routine's
entry, and WRF does not materially change `T_STATE` through the instrumented
internal part1 boundaries.

## Proof Objects

- `proofs/v014/step1_part1_physics_state_mutation.py`
- `proofs/v014/step1_part1_physics_state_mutation.json`
- `proofs/v014/step1_part1_physics_state_mutation.md`
- `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`
- `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and pending-memory artifacts only.
No production model source changed in this sprint.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_part1_physics_state_mutation.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part1_physics_state_mutation.py`
- `python -m json.tool proofs/v014/step1_part1_physics_state_mutation.json >/tmp/step1_part1_physics_state_mutation.manager.validated.json`
- `git diff -- src/gpuwrf`

The rerun reproduced the verdict and left `src/gpuwrf` unchanged.

## Key Numbers

- `part1_entry_before_init_zero_tendency` `T_STATE` vs JAX live-nest step-entry
  state: max_abs `5.490173101425171`, RMSE `1.9175184863907806`.
- Largest WRF internal `T_STATE` delta from part1 entry:
  `after_init_zero_tendency`, max_abs `0.0`.
- `part1_exit` vs JAX `_physics_step_forcing.carry.state`: max_abs
  `5.490173101425171`, RMSE `1.9175184863907806`.
- `part1_exit` vs JAX `_physics_step_forcing.state`: max_abs
  `5.490142455570492`, RMSE `1.9174736017582765`.

## Scope Changes

None. WRF instrumentation was disposable scratch only, CPU-only, and env-gated.
No TOST, Switzerland, FP32, memory source work, GPU, or production source edit
was performed.

## Lessons

Do not continue into radiation/surface/PBL/cumulus leaves for this `T_STATE`
residual. The fastest rigorous path remains an upstream call-site/carry
comparator immediately before `first_rk_step_part1`.

## Next Sprint

Open `v014-step1-pre-part1-handoff`: compare WRF call-site state immediately
before `first_rk_step_part1` to JAX live-nest Step-1 loader/carry/state
surfaces. The proof gate is a precise upstream state-construction or mapping
source, or a narrow performance-compatible fix with before/after Step-1 proof.
