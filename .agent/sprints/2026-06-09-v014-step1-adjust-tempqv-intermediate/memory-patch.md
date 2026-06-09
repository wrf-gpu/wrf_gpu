# Memory Patch: V0.14 Step-1 Adjust-TempQV Intermediate Truth

Date: 2026-06-09

Reviewer Status: APPROVED_FOR_PENDING_MEMORY

Record this sprint as closed with verdict
`STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

Proof objects:

- `proofs/v014/step1_adjust_tempqv_intermediate.*`
- WRF hook output:
  `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- manager run log:
  `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/logs/wrf_run_mpirun_np28_manager.log`

Important facts:

- QVAPOR and theta transcription are not the next proven source-fix surface.
- WRF and JAX match for `p`, `mub_save`, `c3h`, `c4h`, and `p_top`.
- WRF current `mub` is higher than the JAX candidate by
  `17.67503987130476 Pa`.
- WRF `pb_new_equiv` and `p_new` are higher by
  `17.49400702366256 Pa`.
- The post-adjust theta residual remains
  `0.00541785382188209 K`.

Manager conclusion:

The next sprint should split current live-nest `MUB/PB` construction around
WRF `blend_terrain` / base recomputation and the JAX live-nest base-init
candidate. Keep TOST, Switzerland, FP32 source landing, and memory source work
paused until this grid-parity branch is fixed or explicitly bounded.
