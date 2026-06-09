# Memory Patch: V0.14 Step-1 QVAPOR Pre-Call Savepoint

Date: 2026-06-09

## Memory Update

The Step-1 same-boundary WRF pre-call `QVAPOR` truth now exists.

Record:

- Sprint verdict: `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`.
- Proof objects:
  `proofs/v014/step1_qvapor_precall_savepoint.{py,json,md}` and
  `proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`.
- Filtered truth root for the next theta rerun:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- Raw WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/wrf_truth`.
- QVAPOR shape/count: `[44,66,159]`, `461736`, all finite.
- Existing accepted pre-call fields
  `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT/W_STATE/PH_STATE/PHB` are
  text-identical to the prior accepted pre-call dump, max_abs `0.0`.
- No production `src/gpuwrf/**` source changed.

Next memory:

- Rerun `proofs/v014/step1_live_nest_theta_semantics.py` with the filtered
  root and classify the remaining worst residual cell before any production
  theta/`adjust_tempqv` patch.

## Reviewer Status:

Accepted after manager validation and review. This replaces the prior pending
memory note for this sprint.
