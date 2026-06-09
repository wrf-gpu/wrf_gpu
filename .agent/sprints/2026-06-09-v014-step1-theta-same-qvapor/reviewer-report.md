# Reviewer Report

Decision: ACCEPT_INTERIOR_RESIDUAL_VERDICT.

## Review

The proof is acceptable and prevents a premature source patch. It uses the
validated same-boundary QVAPOR root as the WRF pre-call truth comparator and
keeps the WRF formula transcription's pre-adjust QVAPOR input consistent with
the prior proof. This distinction matters because same-boundary QVAPOR is
post-`adjust_tempqv`, not the formula's initial QVAPOR input.

## Evidence

- Verdict:
  `STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.
- Accepted QVAPOR truth source:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- Candidate QVAPOR after `adjust_tempqv` matches WRF pre-call QVAPOR closely:
  max_abs `3.838436518426372e-06`.
- Final `T_STATE` max_abs remains `0.00541785382188209 K`.
- Boundary-band max_abs is only `0.0005722015491755883 K`, but interior max_abs
  is the full `0.00541785382188209 K`.
- Worst cell is interior by the sprint definition:
  Fortran `{i:18,j:10,k:2}`, boundary distance `9`.

## Required Follow-Up

Emit WRF's exact `adjust_tempqv` intermediate values for the worst cell or a
compact field: pre/post `t_2`, QVAPOR, `p_old`, `p_new`, `mub`,
`mub_save`, `c3h`, `c4h`, `p_top`, and any pressure/rounding inputs WRF uses.
Do not patch production theta semantics until that evidence explains or bounds
the interior residual.
