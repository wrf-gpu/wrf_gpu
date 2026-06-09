# Worker Report: V0.14 Step-1 Adjust-TempQV Intermediate Truth

Date: 2026-06-09

Summary: The sprint produced an env-gated disposable CPU-WRF hook around
`nest_init_utils.F::adjust_tempqv`, rebuilt the scratch WRF executable, and
captured exact WRF intermediates for d02 Fortran cell `{i:18,j:10,k:2}` after
the manager reran MPI outside the Codex PMIx sandbox blocker. The final proof
classifies the residual as
`STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

## Files Changed

- `proofs/v014/step1_adjust_tempqv_intermediate.py`
- `proofs/v014/step1_adjust_tempqv_intermediate.json`
- `proofs/v014/step1_adjust_tempqv_intermediate.md`
- `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md`

No production `src/gpuwrf/**` file was edited.

## Commands Run

- `/home/enric/miniconda3/bin/tcsh ./compile em_real` in the disposable WRF tree
- manager unsandboxed MPI rerun with
  `WRFGPU2_STEP1_ADJUST_TEMPQV_INTERMEDIATE=1` and `mpirun --oversubscribe -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/step1_adjust_tempqv_intermediate.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_adjust_tempqv_intermediate.py`
- `python -m json.tool proofs/v014/step1_adjust_tempqv_intermediate.json >/tmp/step1_adjust_tempqv_intermediate.manager.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects

- `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/logs/wrf_run_mpirun_np28_manager.log`
- `proofs/v014/step1_adjust_tempqv_intermediate.json`
- `proofs/v014/step1_adjust_tempqv_intermediate.md`
- `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`

## Result

WRF and JAX agree for `p`, `mub_save`, `c3h`, `c4h`, and `p_top`, but differ
materially for current `mub`, `pb_new_equiv`, and `p_new` by about `17.5 Pa`.
The `t_2_post` residual remains exactly the previously observed
`0.00541785382188209 K`.

The next step is a targeted current-`MUB`/base-input split around WRF
`blend_terrain` / live-nest base recomputation and the JAX reconstruction used
by the theta proof.
