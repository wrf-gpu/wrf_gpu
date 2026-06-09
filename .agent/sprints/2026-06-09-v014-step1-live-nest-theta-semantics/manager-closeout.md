# Manager Closeout

## Outcome

The sprint is closed as a validated partial localization proof.

Final verdict:
`STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.

The proof confirms that the earlier hypothesis was incomplete. WRF does not
match by running `adjust_tempqv` directly on raw dry NetCDF `T`; with
`USE_THETA_M=1`, WRF solve-time `grid%t_2` uses moist-theta semantics. The
dominant semantic sequence is dry-to-moist theta conversion followed by
`adjust_tempqv`.

## Proof Objects

- `proofs/v014/step1_live_nest_theta_semantics.py`
- `proofs/v014/step1_live_nest_theta_semantics.json`
- `proofs/v014/step1_live_nest_theta_semantics.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`

## Merge Decision:

Merge proof, review, sprint closeout, roadmap, and memory updates only. Do not
merge a production model source patch from this sprint, because the candidate
does not fully close the gate and accepted same-boundary `QVAPOR` truth is
missing.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_live_nest_theta_semantics.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_semantics.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_semantics.json >/tmp/step1_live_nest_theta_semantics.manager.validated.json`
- `git diff -- src/gpuwrf`

The rerun reproduced the verdict, validated JSON, recorded `gpu_used=false`,
and left `src/gpuwrf` unchanged.

## Key Numbers

- Raw/current live dry `T_STATE` max_abs versus WRF pre-call:
  `5.490173101425171`.
- Direct `adjust_tempqv` on raw dry `T`: max_abs `5.490177290476879`.
- WRF dry-to-moist theta conversion only: max_abs `0.753296811070129`.
- WRF moist-theta conversion plus `adjust_tempqv`: max_abs
  `0.00541785382188209`, RMSE `5.068868142015466e-05`, p99
  `4.546931764011239e-05`.
- Same candidate with fp32 arithmetic: max_abs `0.00543212890625`.

## Scope Changes

None. No TOST, Switzerland validation, FP32, memory source work, GPU, Hermes, or
production source edit was performed.

## Next Sprint

Use the QVAPOR schema sprint result to run a minimal WRF savepoint extension at
`before_first_rk_step_part1_call`, emitting `QVAPOR` from `moist(i,k,j,P_QV)`.
Then rerun the theta proof against same-boundary `T_STATE` and `QVAPOR` before
any production patch.
