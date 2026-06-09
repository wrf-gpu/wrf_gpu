# Review: V0.14 Step-1 First-RK Part1 P-State Split

Verdict: `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.

objective: split the Step-1 `P/MU/W` residual around WRF `first_rk_step_part1`, especially `phy_prep`, and name the exact upstream surface/contract if WRF part1 is clean.

files changed:
- `proofs/v014/step1_first_rk_part1_p_state_split.py`
- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

commands run:
- `python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_first_rk_part1_p_state_split.py`
- `python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json >/tmp/step1_first_rk_part1_p_state_split.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_first_rk_part1_p_state_split.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_first_rk_part1_p_state_split.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`
- reused WRF pre-call truth root `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`
- reused WRF part1 truth root `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth`
- reused WRF source-boundary truth root `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/wrf_truth`

unresolved risks:
- This proof names the missing live-nest perturbation-state contract but does not implement the formula for WRF's P/MU/W initialization.
- No Step-1 post-acoustic/pre-refresh run was made because the first material P/MU/W residual is already upstream of first_rk_step_part1.
- The next patch must preserve GPU residency and avoid any CPU-WRF runtime dependency.

next decision: Open a narrow live-nest perturbation-state sprint: transcribe/prove WRF raw-child -> pre-first_rk_step_part1 initialization for P_STATE, MU_STATE, and W_STATE, then apply only a GPU-native initialization fix if that closes this boundary.
