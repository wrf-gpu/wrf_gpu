# Memory Patch: V0.14 Step-1 QVAPOR Pre-Call Truth Schema

Date: 2026-06-09

Closed 2026-06-09.

Reason:

- The live-nest theta proof found a likely truth-schema gap: accepted WRF
  pre-call `T_STATE` truth does not include same-boundary `QVAPOR`.
- Existing `QVAPOR` truth artifacts may be post-RK or different-boundary and
  must not be silently substituted.

Expected memory after close:

- Record whether same-boundary WRF pre-call `QVAPOR` truth exists.
- If missing, record the exact minimal WRF savepoint needed to close the next
  theta/`adjust_tempqv` proof.

## Reviewer Status:

Accepted as sprint-local memory.

Evidence:

- `proofs/v014/step1_qvapor_precall_truth_schema.json` records verdict
  `STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`.
- Accepted pre-call `before_first_rk_step_part1_call` truth lacks `QVAPOR`.
- Existing QVAPOR truth is post-RK/pre-halo and must not be reused for the
  pre-call theta proof.
- Proposed savepoint:
  `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`.
