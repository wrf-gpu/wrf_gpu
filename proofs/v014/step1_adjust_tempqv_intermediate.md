# V0.14 Step-1 Adjust-TempQV Intermediate

Verdict: `STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

## Result

- CPU-only proof; GPU used: `False`.
- Required ancestor `c3620d09` present: `True`.
- Disposable WRF hook patch: `/home/enric/src/wrf_gpu2/proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`.
- Rebuilt executable: `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF/main/wrf.exe`; hook string present: `True`.
- WRF run return code: `0`.
- WRF run log: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/logs/wrf_run_mpirun_np28_manager.log`.
- WRF hook file: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`; status `READY`.

## Interpretation

WRF in-routine pressure/base inputs differ materially from the JAX reconstruction.

## Target Comparison

| Field | WRF | JAX | WRF-JAX |
|---|---:|---:|---:|
| `t_2_post` | -2.4365234375000000e-01 | -2.3823448992811791e-01 | -5.4178538218820904e-03 |
| `qvapor_post` | 7.8749302774667740e-03 | 7.8711710173183687e-03 | 3.7592601484053023e-06 |
| `p` | 1.7501562500000000e+03 | 1.7501562500000000e+03 | 0.0000000000000000e+00 |
| `p_old` | 9.1811789062500000e+04 | 9.1811786848075688e+04 | 2.2144243121147156e-03 |
| `p_new` | 9.2686187500000000e+04 | 9.2668693492976337e+04 | 1.7494007023662562e+01 |
| `pb_old_equiv` | 9.0061632812500000e+04 | 9.0061630598075688e+04 | 2.2144243121147156e-03 |
| `pb_new_equiv` | 9.0936031250000000e+04 | 9.0918537242976337e+04 | 1.7494007023662562e+01 |
| `mub` | 8.6812250000000000e+04 | 8.6794574960128695e+04 | 1.7675039871304762e+01 |
| `mub_save` | 8.5928687500000000e+04 | 8.5928687500000000e+04 | 0.0000000000000000e+00 |
| `c3h` | 9.8962819576263428e-01 | 9.8962819576263428e-01 | 0.0000000000000000e+00 |
| `c4h` | 2.4178623199462891e+01 | 2.4178623199462891e+01 | 0.0000000000000000e+00 |
| `p_top` | 5.0000000000000000e+03 | 5.0000000000000000e+03 | 0.0000000000000000e+00 |

## Handoff

objective: emit exact CPU-WRF `adjust_tempqv` intermediates for d02 i=18,j=10,k=2 and compare to the JAX proof.

files changed:
- `proofs/v014/step1_adjust_tempqv_intermediate.py`
- `proofs/v014/step1_adjust_tempqv_intermediate.json`
- `proofs/v014/step1_adjust_tempqv_intermediate.md`
- `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_adjust_tempqv_intermediate.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_adjust_tempqv_intermediate.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md`

unresolved risks:
- Only one target cell was emitted; broader savepoint may still be needed for neighborhood effects.

next decision needed: Use the classified mismatch to decide whether to patch pressure/base inputs or formula transcription.
