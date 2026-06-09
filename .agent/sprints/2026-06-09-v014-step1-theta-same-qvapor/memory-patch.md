# Memory Patch: V0.14 Step-1 Theta Same-Boundary QVAPOR

Date: 2026-06-09

## Memory Update

The same-boundary QVAPOR rerun is closed.

Record:

- Verdict: `STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.
- Proof objects: `proofs/v014/step1_theta_same_qvapor.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`.
- Same-boundary QVAPOR root:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- Final candidate `theta_m_then_adjust_tempqv` max_abs:
  `0.00541785382188209 K`; p99 `4.546931764011239e-05 K`; p99.9
  `0.0004691662256855125 K`.
- Boundary band (`distance_to_edge <= 5`) max_abs:
  `0.0005722015491755883 K`.
- Interior (`distance_to_edge > 5`) max_abs:
  `0.00541785382188209 K`.
- Worst cell: zero `{k:1,y:9,x:17}`, Fortran `{i:18,j:10,k:2}`, horizontal
  boundary distance `9`.
- Candidate QVAPOR after `adjust_tempqv` versus WRF pre-call QVAPOR:
  max_abs `3.838436518426372e-06`, RMSE `2.852916741433691e-08`.
- No production `src/gpuwrf/**` source changed.

Next memory:

- Same-boundary QVAPOR is no longer the blocker.
- Do not apply a production theta/`adjust_tempqv` patch yet.
- Next sprint must emit or recover WRF exact theta/`adjust_tempqv`
  intermediate values for the residual cell/path, especially pre/post `t_2`,
  QVAPOR, `p_old`, `p_new`, `mub`, `mub_save`, `c3h`, `c4h`, `p_top`, and
  pressure/base inputs.

## Reviewer Status:

Accepted after manager validation and review.
