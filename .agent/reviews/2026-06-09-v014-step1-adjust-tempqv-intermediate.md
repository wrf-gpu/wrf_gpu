# Review: V0.14 Step-1 Adjust-TempQV Intermediate

Verdict: `STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

Findings:
- HIGH: WRF in-routine pressure/base inputs differ materially from the JAX reconstruction.

Evidence:
- Patch diff: `/home/enric/src/wrf_gpu2/proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`
- Compile log: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/logs/compile_em_real_tcsh.log`
- Run log: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/logs/wrf_run_mpirun_np28_manager.log`
- Hook output: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`

Next decision: Use the classified mismatch to decide whether to patch pressure/base inputs or formula transcription.
