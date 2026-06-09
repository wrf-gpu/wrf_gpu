# Review: V0.14 Step-1 T/P Operator Localization

Verdict: `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_P_STATE`.

objective: localize the remaining Step-1 strict same-input T/P divergence after live-nest base initialization closure.

files changed:
- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization.md`
- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`

commands run:
- `python -m py_compile proofs/v014/step1_t_p_operator_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_t_p_operator_localization.py`
- `python -m json.tool proofs/v014/step1_t_p_operator_localization.json >/tmp/step1_t_p_operator_localization.validated.json`
- `git diff -- src/gpuwrf`
- `tcsh ./compile em_real (scratch WRF under /mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/WRF)`
- `WRFGPU2_STEP1_TP_LOCALIZATION=1 mpirun --oversubscribe -np 28 run/wrf.exe`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_t_p_operator_localization.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_t_p_operator_localization.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`

unresolved risks:
- Only two early WRF substage boundaries were emitted; if this boundary is fixed and residuals remain, acoustic/pre-finish substep truth is the next surface.
- The proof-local CPU loader still mirrors production live-nest init because build_replay_case is GPU-only at State.zeros.

next decision: Resolve the WRF/JAX RK1 stage-entry state mismatch after WRF first_rk_step_part1/part2 and before JAX small_step_prep, starting with field P_STATE; do not continue acoustic debugging yet.
