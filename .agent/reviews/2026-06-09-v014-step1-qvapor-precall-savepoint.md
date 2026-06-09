# Review: V0.14 Step-1 QVAPOR Pre-call Savepoint

Decision: ACCEPT_SAVEPOINT_READY.

## Verdict

The savepoint is valid for the next theta proof. The new disposable WRF output
adds same-boundary `QVAPOR` at `before_first_rk_step_part1_call` while proving
that all previously accepted fields remain unchanged.

## Evidence

- Verdict: `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`.
- New filtered root:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- 28 old and 28 new pre-call tile files match by filename.
- Old fields are text-identical with max_abs `0.0`.
- QVAPOR count `461736`, shape `[44,66,159]`, all finite.
- `git diff -- src/gpuwrf` is empty.

## Risk

This closes the missing truth artifact only. It does not close the live-nest
theta residual or the larger base-state split. A production patch before the
next theta rerun would still be premature.

## Follow-up

Run the theta semantics proof against the filtered QVAPOR root and classify the
worst residual cell. If the remaining max_abs is a boundary outlier while p99
is already small, document the gate semantics explicitly instead of chasing a
single edge cell with global state changes.
