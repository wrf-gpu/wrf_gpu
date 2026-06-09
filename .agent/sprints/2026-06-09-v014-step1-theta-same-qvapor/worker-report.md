# Worker Report

Summary: Same-boundary QVAPOR rerun completed and proved the remaining theta
tail is an interior residual, not a boundary-only tail.

## Objective

Rerun the Step-1 live-nest theta semantics proof using the validated
same-boundary WRF pre-call `QVAPOR` root, then classify the final `T_STATE`
residual as boundary-band or interior.

## Files Changed

- `proofs/v014/step1_theta_same_qvapor.py`
- `proofs/v014/step1_theta_same_qvapor.json`
- `proofs/v014/step1_theta_same_qvapor.md`
- `.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`

No `src/gpuwrf/**` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/step1_theta_same_qvapor.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_theta_same_qvapor.py`
- `python -m json.tool proofs/v014/step1_theta_same_qvapor.json >/tmp/step1_theta_same_qvapor.validated.json`
- `git diff -- src/gpuwrf`

The manager reran the same validation commands and reproduced the verdict.

## Proof Objects

- `proofs/v014/step1_theta_same_qvapor.json`
- `proofs/v014/step1_theta_same_qvapor.md`
- `.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`
- same-boundary QVAPOR root:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`

## Result

Final verdict:
`STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.

Key metrics:

- Final `theta_m_then_adjust_tempqv` max_abs:
  `0.00541785382188209 K`.
- p99: `4.546931764011239e-05 K`; p99.9:
  `0.0004691662256855125 K`.
- Boundary band (`distance_to_edge <= 5`) max_abs:
  `0.0005722015491755883 K`.
- Interior (`distance_to_edge > 5`) max_abs:
  `0.00541785382188209 K`.
- Worst cell: zero index `{k:1,y:9,x:17}`, Fortran
  `{i:18,j:10,k:2}`, horizontal boundary distance `9`.
- Candidate QVAPOR after `adjust_tempqv` versus same-boundary WRF pre-call
  QVAPOR: max_abs `3.838436518426372e-06`, RMSE
  `2.852916741433691e-08`.

## Method Correction

During the sprint the worker corrected an important methodology issue: WRF's
theta/`adjust_tempqv` formula transcription must use raw child input `QVAPOR`
as the pre-adjust input, while the filtered same-boundary QVAPOR root is the
accepted pre-call truth comparator. The final proof uses that corrected method.

## Unresolved Risks

The remaining theta residual is small but interior and above the prior `1e-3 K`
material max_abs gate. A source patch is not authorized without WRF
intermediate theta/`adjust_tempqv` pressure inputs, or an equivalent proof, for
the residual cell.

## Next Decision

Open a WRF-intermediate savepoint sprint that emits WRF's exact pre/post
`adjust_tempqv` `t_2`, QVAPOR, and pressure inputs for the worst cell or compact
full field. Use that to distinguish formula/transcription error, pressure-input
residual, and acceptable fp/rounding tail before any production patch.
