# Manager Closeout

## Outcome

The sprint is closed as a validated loader-stage localization proof.

Final verdict:
`STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.

`T_STATE` is already wrong in raw d02 wrfinput theta and remains bit-identical
through the JAX live-nest base-init, boundary-package, initial-carry, and
haloed-step-entry stages. The live-nest stage nevertheless closes the large
`PB/PHB/MUB` base residuals, so the remaining bug is that the JAX live-nest
base initialization updates base fields without applying WRF's matching
`t_2`/theta initialization semantics.

## Proof Objects

- `proofs/v014/step1_jax_loader_tstate.py`
- `proofs/v014/step1_jax_loader_tstate.json`
- `proofs/v014/step1_jax_loader_tstate.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`
- reused WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and pending-memory artifacts
only. No production model source changed in this sprint.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_jax_loader_tstate.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_loader_tstate.py`
- `python -m json.tool proofs/v014/step1_jax_loader_tstate.json >/tmp/step1_jax_loader_tstate.manager.validated.json`
- `git diff -- src/gpuwrf`
- `git diff --check -- proofs/v014/step1_jax_loader_tstate.py proofs/v014/step1_jax_loader_tstate.json proofs/v014/step1_jax_loader_tstate.md .agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`

The rerun reproduced the verdict and left `src/gpuwrf` unchanged.

## Key Numbers

- Raw child `T_STATE` vs WRF pre-call: max_abs
  `5.490173101425171`, RMSE `1.9175184863907806`.
- Live child `T_STATE` vs WRF pre-call: max_abs
  `5.490173101425171`, RMSE `1.9175184863907806`.
- Haloed step-entry `T_STATE` vs WRF pre-call: max_abs
  `5.490173101425171`, RMSE `1.9175184863907806`.
- `T_STATE` transition max_abs raw->live, live->boundary, boundary->carry,
  carry->halo: all `0.0`.
- `PB` raw max_abs `2627.3828125` improves to live max_abs
  `0.05357326504599769`.
- Haloed step-entry interior max_abs `5.490173101425171`; boundary-band max_abs
  `5.284271240234375`.

## Scope Changes

None. No TOST, Switzerland, FP32, memory source work, GPU, Hermes, or production
source edit was performed.

## Lessons

Do not keep debugging boundary package, carry construction, halo, or physics for
this `T_STATE` residual. The next proof/fix target is WRF live-nest
initialization semantics around `med_nest_initial` and `start_domain_em`, where
WRF passes `nest%t_2`, `nest%p`, and moisture through the base-init path after
terrain/base blending.

## Next Sprint

Open `v014-step1-live-nest-theta-semantics`: prove the exact WRF
`T_STATE`/theta reconstruction after live-nest base initialization, compare
candidate GPU/JAX formulas against the WRF pre-call truth, then apply the
smallest initialization-only production fix if the candidate closes the residual.
