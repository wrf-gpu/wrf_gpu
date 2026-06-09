# Review: V0.14 Step-1 Part1 Physics-State Mutation

Verdict: `STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`.

objective: split the first material Step-1 `T_STATE` mismatch inside or at entry to WRF `first_rk_step_part1`.

files changed:
- `proofs/v014/step1_part1_physics_state_mutation.py`
- `proofs/v014/step1_part1_physics_state_mutation.json`
- `proofs/v014/step1_part1_physics_state_mutation.md`
- `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`

commands run:
- `cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/WRF /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/WRF`
- `tcsh ./compile em_real (scratch WRF, conda wrf-build toolchain)`
- `WRFGPU2_STEP1_PART1_PHYSICS_STATE_MUTATION=1 WRFGPU2_STEP1_PART1_PHYSICS_STATE_MUTATION_ROOT=/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth mpirun --oversubscribe -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/step1_part1_physics_state_mutation.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part1_physics_state_mutation.py`
- `python -m json.tool proofs/v014/step1_part1_physics_state_mutation.json >/tmp/step1_part1_physics_state_mutation.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_part1_physics_state_mutation.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_part1_physics_state_mutation.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth`

unresolved risks:
- This proof localizes the first material T_STATE residual before first_rk_step_part1 executes.
- WRF T_STATE does not need a production physics-state mutation fix inside first_rk_step_part1 unless a later proof finds the upstream handoff source.

next decision needed: Move upstream to the live-nest/WRF handoff immediately before first_rk_step_part1 entry.
