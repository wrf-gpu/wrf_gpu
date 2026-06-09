# V0.14 Step-1 Theta Same-Boundary QVAPOR

Verdict: `STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `912b7371` present: `True`.
- Same-boundary QVAPOR root: `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- Real run `USE_THETA_M`: `{'wrfinput_attr': 1, 'wrfout_attr': 1, 'namelist_output': 'USE_THETA_M=1          ,'}`.
- Raw/current live dry `T_STATE` vs WRF pre-call: max_abs `5.490173101425171` / `5.490173101425171`.
- `adjust_tempqv` directly on raw dry `T` with `use_theta_m=1`: max_abs `5.490177290476879`.
- WRF dry-to-moist theta conversion only: max_abs `0.753296811070129`, rmse `0.015916793982291767`.
- WRF `theta_m` conversion plus `adjust_tempqv`: max_abs `0.00541785382188209`, rmse `5.068868142015466e-05`, p99 `4.546931764011239e-05`, p99.9 `0.0004691662256855125`.
- Same candidate with fp32 arithmetic: max_abs `0.00543212890625`, rmse `5.171032344972347e-05`.
- Final candidate boundary band (`distance_to_edge <= 5`): max_abs `0.0005722015491755883`, rmse `2.6153316080892344e-05`, p99.9 `0.0003082591272069537`.
- Final candidate interior (`distance_to_edge > 5`): max_abs `0.00541785382188209`, rmse `5.635969695643974e-05`, p99.9 `0.0006121642677166802`.
- Candidate QVAPOR after `adjust_tempqv` vs same-boundary WRF pre-call QVAPOR: max_abs `3.838436518426372e-06`, rmse `2.852916741433691e-08`.

## Worst Cell

- Zero index (`k,y,x`): `{'k': 1, 'y': 9, 'x': 17}`; Fortran index: `{'i': 18, 'j': 10, 'k': 2}`.
- Boundary distance: `9`; boundary band: `False`.
- WRF value `-0.24365234375`, candidate `-0.2382344899281179`, delta `0.00541785382188209`.
- QVAPOR `0.007874930277466774`; candidate QVAPOR after adjust `0.007871171017318369`.
- Pressure/base inputs for the worst cell are recorded under `comparisons.final_candidate_residual.worst_cell.available_pressure_base_inputs` in the JSON.

## Interpretation

- The proof loaded accepted WRF `QVAPOR` truth only from the filtered same-boundary pre-call root, not from `wrfout_d02`.
- The WRF formula transcription keeps the prior proof's raw child input `QVAPOR`; the same-boundary root is the accepted pre-call truth comparator.
- The final candidate residual is classified by the contract's all-cell and boundary/interior max_abs gates.
- Next decision: Emit or recover WRF theta_m/adjust_tempqv intermediate inputs for the residual cell before patching production.

Detailed tables are in `proofs/v014/step1_theta_same_qvapor.json`.
