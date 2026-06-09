# V0.14 Step-1 QVAPOR Pre-Call Truth Schema

Opened 2026-06-09.

Purpose: establish whether authoritative same-boundary WRF pre-call `QVAPOR`
truth exists for the Step-1 live-nest theta proof.

Closed evidence:

- `proofs/v014/step1_qvapor_precall_truth_schema.json` verdict:
  `STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`.
- Same-boundary WRF pre-call `QVAPOR` truth does not exist in the accepted
  `before_first_rk_step_part1_call` truth set.
- Existing `QVAPOR` truth is post-RK/pre-halo (`rk_step 4`) or otherwise
  different-boundary and must not be used for the pre-call theta proof.
- The exact minimal savepoint spec is recorded in
  `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`.

Next step: run a CPU-only WRF savepoint around the existing
`before_first_rk_step_part1_call` hook to emit `moist(i,k,j,P_QV)` as `QVAPOR`,
then rerun the live-nest theta semantics proof with same-boundary `T_STATE` and
`QVAPOR`.
