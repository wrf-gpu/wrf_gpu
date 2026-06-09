# Review: V0.14 Step-1 Pre-Part1 Handoff

Verdict: `STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.

objective: move one boundary upstream from WRF `first_rk_step_part1` and classify why `T_STATE` already diverges at part1 entry.

files changed:
- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_pre_part1_handoff.json`
- `proofs/v014/step1_pre_part1_handoff.md`
- `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`

commands run:
- `cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/WRF /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF`
- `cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/run /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/run`
- `tcsh ./compile em_real (scratch WRF, conda wrf-build toolchain)`
- `WRFGPU2_STEP1_PRE_PART1_HANDOFF=1 WRFGPU2_STEP1_PRE_PART1_HANDOFF_ROOT=/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth mpirun --oversubscribe -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/step1_pre_part1_handoff.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_pre_part1_handoff.py`
- `python -m json.tool proofs/v014/step1_pre_part1_handoff.json >/tmp/step1_pre_part1_handoff.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_pre_part1_handoff.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_pre_part1_handoff.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`

unresolved risks:
- This proof localizes the divergence to the JAX live-nest Step-1 loader/carry boundary, but does not yet split the loader internals.
- No production source fix was made or gated.

next decision needed: Localize the JAX live-nest Step-1 loader/carry construction for `T_STATE` before `_physics_step_forcing`.
