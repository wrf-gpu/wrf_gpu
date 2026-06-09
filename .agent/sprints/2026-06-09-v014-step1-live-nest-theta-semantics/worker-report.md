# Worker Report

Summary: The live-nest theta semantics proof is complete as a partial
localization, with no production source edits.

## Objective

Prove whether WRF live-nest `adjust_tempqv` semantics after terrain/base
blending close the Step-1 pre-call `T_STATE` residual.

## Files Changed

- `proofs/v014/step1_live_nest_theta_semantics.py`
- `proofs/v014/step1_live_nest_theta_semantics.json`
- `proofs/v014/step1_live_nest_theta_semantics.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`

No `src/gpuwrf/**` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/step1_live_nest_theta_semantics.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_semantics.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_semantics.json >/tmp/step1_live_nest_theta_semantics.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects

- `proofs/v014/step1_live_nest_theta_semantics.json`
- `proofs/v014/step1_live_nest_theta_semantics.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`

## Result

Final verdict:
`STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.

The WRF semantic path is not direct `adjust_tempqv` on raw dry NetCDF `T`.
For `USE_THETA_M=1`, WRF first uses moist-theta in-memory semantics and then
applies `adjust_tempqv`. That sequence reduces `T_STATE` max_abs from
`5.490173101425171` to `0.00541785382188209`, with RMSE
`5.068868142015466e-05`, but does not close the prior `1e-3 K` material gate.

## Unresolved Risks

- Accepted same-boundary WRF pre-call `QVAPOR` truth is missing.
- The remaining millikelvin `T_STATE` residual is not yet explained.
- A production patch would be premature until the same-boundary moisture truth
  and residual cause are resolved.

## Next Decision

Run a minimal WRF savepoint extension that emits `QVAPOR` at the existing
`before_first_rk_step_part1_call` boundary, then rerun the theta proof against
same-boundary `T_STATE` and `QVAPOR`.
