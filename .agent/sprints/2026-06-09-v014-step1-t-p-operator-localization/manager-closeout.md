# Manager Closeout

## Outcome

The sprint is closed as a validated localization proof.

Final verdict:
`STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.

The v0.14 grid-parity bug is now earlier than acoustic/small-step pressure
refresh. The first strict and first material T/P-family mismatch is `T_STATE`
at the RK1 stage-entry boundary after WRF `first_rk_step_part1/part2` and
`rk_addtend_dry/spec_bdy_dry`, before `small_step_prep`.

## Proof Objects

- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization.md`
- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`
- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and pending-memory artifacts only.
No production model source changed in this sprint.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_t_p_operator_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_t_p_operator_localization.py`
- `python -m json.tool proofs/v014/step1_t_p_operator_localization.json >/tmp/step1_t_p_operator_localization.manager.validated.json`
- `git diff -- src/gpuwrf`

The rerun reproduced the verdict and left `src/gpuwrf` unchanged.

## Key Numbers

- First strict/material T/P-family mismatch:
  `after_rk_addtend_before_small_step_prep`, RK1, `T_STATE`.
- Largest material residual at that boundary:
  `PH_TEND` max_abs `794096.1875`, RMSE `26657.670278603327`.
- RK1 `after_small_step_prep_calc_p_rho` work arrays:
  `T_WORK` max_abs `0.0`, `P_WORK` max_abs `0.0`.
- Final accepted strict comparison still diverges:
  first field `T`, top residual `P` max_abs `1561.2503728885986`.

## Scope Changes

None. WRF instrumentation was disposable scratch only, CPU-only, and env-gated.
No TOST, Switzerland, FP32, memory source work, GPU, or production source edit
was performed.

## Lessons

The fastest rigorous path was the correct one: build a substage truth/comparator
instead of spending more GPU or full-runtime cycles. The next sprint should not
debug acoustic internals; it should split WRF `first_rk_step_part1/part2`,
`rk_tendency`, `relax_bdy_dry`, `rk_addtend_dry`, and `spec_bdy_dry` against
JAX `_physics_step_forcing` and RK tendency construction.

## Next Sprint

Open `v014-step1-rk1-source-boundary`: directly compare WRF first-RK
physics/source/tendency outputs with JAX `_physics_step_forcing` and
pre-`small_step_prep` state/carry. The proof gate is a narrower source boundary
or a small performance-compatible fix with before/after Step-1 proof.
