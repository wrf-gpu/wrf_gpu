# Worker Report

Summary: The QVAPOR pre-call savepoint is complete, but the final artifacts
were manager-finished after the GPT tmux worker stalled.

## Objective

Create same-boundary CPU-WRF `QVAPOR` truth at
`before_first_rk_step_part1_call` for d02 Step 1 RK1, without touching
production `src/gpuwrf/**`.

## Files Changed

- `proofs/v014/step1_qvapor_precall_savepoint.py`
- `proofs/v014/step1_qvapor_precall_savepoint.json`
- `proofs/v014/step1_qvapor_precall_savepoint.md`
- `proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-savepoint.md`

Scratch-only WRF source was edited under
`/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF/dyn_em/solve_em.F`.
No production model source was changed.

## Commands Run

- Disposable WRF rebuild under the wrf-build environment; log:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/logs/compile_qvapor_precall_savepoint_env.log`.
- Manager 28-rank CPU-WRF truth capture:
  `mpirun --oversubscribe -np 28 ./wrf.exe` with
  `WRFGPU2_STEP1_PRE_PART1_HANDOFF_ROOT=/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/wrf_truth`.
- `python -m py_compile proofs/v014/step1_qvapor_precall_savepoint.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_savepoint.py`
- `python -m json.tool proofs/v014/step1_qvapor_precall_savepoint.json >/tmp/step1_qvapor_precall_savepoint.manager.validated.json`
- `git diff -- src/gpuwrf`

The GPT worker also attempted WRF launches inside the Codex sandbox; those hit
OpenMPI/PMIx socket initialization errors before launch. The manager reran the
same WRF command outside that sandbox successfully.

## Proof Objects

- `proofs/v014/step1_qvapor_precall_savepoint.json`
- `proofs/v014/step1_qvapor_precall_savepoint.md`
- `proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`

## Result

Final verdict:
`STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`.

The new pre-call group contains 28 files. `QVAPOR` has full d02 mass shape
`[44,66,159]`, count `461736`, and all values are finite. Existing
`T_STATE/P_STATE/PB/MU_STATE/MUB/MUT/W_STATE/PH_STATE/PHB` records are
text-identical to the accepted pre-call dump with max_abs `0.0`.

## Unresolved Risks

This proof only creates the missing QVAPOR truth. It does not yet authorize a
production theta or `adjust_tempqv` patch, and it does not address the larger
base-state split residual.

## Next Decision

Rerun the live-nest theta semantics proof using the filtered same-boundary
QVAPOR root, classify the worst residual cell as boundary/interior, then decide
whether an init-only patch is justified or whether to prioritize the base-state
split fix first.
