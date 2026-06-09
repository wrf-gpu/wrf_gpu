# V0.14 Step-1 QVAPOR Pre-Call Savepoint

Opened and closed 2026-06-09.

Result: `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`.

The disposable CPU-WRF Step-1 pre-call hook now emits same-boundary `QVAPOR` at
`before_first_rk_step_part1_call`. Production `src/gpuwrf/**` was not edited.

Proof:

- `proofs/v014/step1_qvapor_precall_savepoint.{py,json,md}`
- `proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-savepoint.md`

Truth roots:

- raw: `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/wrf_truth`
- filtered pre-call-only:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`

Key facts:

- QVAPOR shape `[44,66,159]`, count `461736`, all finite.
- Prior accepted fields
  `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT/W_STATE/PH_STATE/PHB` are
  text-identical to the accepted pre-call dump, max_abs `0.0`.

Next: rerun live-nest theta semantics against this filtered root and classify
the worst residual cell before any production theta or `adjust_tempqv` patch.
