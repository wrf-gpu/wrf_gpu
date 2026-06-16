# V0.14 Step-1 QVAPOR Pre-call Savepoint

Verdict: `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`.

## Result

- CPU-only proof; GPU used: `False`.
- Production `src/gpuwrf` source changed: `False`.
- New raw WRF truth root: `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/wrf_truth`.
- Filtered pre-call-only truth root for the next proof: `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- WRF patch diff artifact: `/home/user/src/wrf_gpu2/proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`.
- File counts: `{'accepted_precall_files': 28, 'new_precall_files': 28, 'new_raw_all_hook_files': 168, 'filtered_precall_files': 28}`.
- Mass shape z/y/x: `[44, 66, 159]`.
- WPH shape z/y/x: `[45, 66, 159]`.
- QVAPOR count `461736`, all finite `True`, min `2.687635060283355e-06`, max `0.01170683559030294`, mean `0.0021903549088611598`.

## Old-field identity check

| Field | Count | Max abs | RMSE | Text identical |
|---|---:|---:|---:|---|
| `T_STATE` | 461736 | 0.0 | 0.0 | True |
| `P_STATE` | 461736 | 0.0 | 0.0 | True |
| `PB` | 461736 | 0.0 | 0.0 | True |
| `MU_STATE` | 461736 | 0.0 | 0.0 | True |
| `MUB` | 461736 | 0.0 | 0.0 | True |
| `MUT` | 461736 | 0.0 | 0.0 | True |
| `W_STATE` | 472230 | 0.0 | 0.0 | True |
| `PH_STATE` | 472230 | 0.0 | 0.0 | True |
| `PHB` | 472230 | 0.0 | 0.0 | True |

## Interpretation

The new disposable WRF hook appended `QVAPOR` to the accepted `before_first_rk_step_part1_call` mass record without changing the previous mass or W/PH fields. All previous fields are text-identical to the accepted pre-call dump, so the new QVAPOR field is a valid same-boundary truth input for the next theta-semantics rerun.

This proof does not authorize a production theta or `adjust_tempqv` patch by itself. The next sprint must rerun the theta proof using the filtered same-boundary QVAPOR root and classify the remaining worst cell before a source patch decision.
