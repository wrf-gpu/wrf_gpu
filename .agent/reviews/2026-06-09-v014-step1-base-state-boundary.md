# Review: V0.14 Step-1 Base-State Boundary

Verdict: `STEP1_BASE_STATE_BOUNDARY_LOCALIZED_P_SURF_MUB_FP32_SOURCE_ARITHMETIC`.

objective: close or precisely localize the WRF `start_domain_em` base-state boundary before the Step-1 `AL/ALT` pass.

files changed:
- `proofs/v014/step1_base_state_boundary.py`
- `proofs/v014/step1_base_state_boundary.json`
- `proofs/v014/step1_base_state_boundary.md`
- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`

commands run:
- `python -m py_compile proofs/v014/step1_base_state_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_base_state_boundary.py`
- `python -m json.tool proofs/v014/step1_base_state_boundary.json >/tmp/step1_base_state_boundary.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_base_state_boundary.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_base_state_boundary.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`
- `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

ranked hypotheses/exclusions:
- rank 1: SUPPORTED_LOCALIZED - The remaining Step-1 base gap is the exact WRF p_surf/MUB source arithmetic feeding AL/ALT.
- rank 2: SUPPORTED_BY_SOURCE_AND_FLAGS - The WRF branch is multi-domain real start_domain with rebalance disabled: PB/T_INIT/ALB are reconstituted from MUB, PHB is not re-integrated in that later block.
- rank 3: REFUTED - Terrain/blend input is the dominant residual.
- rank 4: REFUTED_AS_DOMINANT - Constants or cp=1004.0 vs WRF cp=1004.5 are the dominant residual.
- rank 5: REFUTED_BY_PREDECESSOR_AND_CURRENT_ABLATION - Coefficient indexing or PH/MU time-level selection is the cause.
- excluded: No production src/gpuwrf edit was made.
- excluded: No GPU, TOST, Switzerland, FP32 production source, memory production source, or Hermes path was used.
- excluded: Terrain was falsified as dominant by substituting WRF HT into the proof-local fp32 formula with no P improvement.
- excluded: cp/constants were falsified as dominant: cp=1004.0 vs 1004.5 does not move MUB/PB and leaves the same downstream P gap.
- excluded: Coefficient indexing is unlikely: exact WRF MUB with current metrics closes downstream P/MU gates.
- excluded: PHB integration order remains a small base residual, but not the dominant downstream P/MU blocker after WRF MUB substitution.

unresolved risks:
- No production source patch was applied.
- This proof did not emit a fresh WRF p_surf_before_mub scalar; it recovers p_surf from WRF MUB + P_TOP.
- The exact production-compatible p_surf arithmetic helper is still missing; local NumPy/JAX-style fp32 formula remains above P/MU gates.

next decision: Do not patch d02_replay from the current p_surf formula yet. The next source contract should either instrument one disposable WRF boundary immediately around the p_surf expression/MUB assignment to capture p_surf_before_mub and MUB exactly, or implement a narrowly gated WRF-compatible fp32/libm p_surf helper and require P_STATE <= 1 Pa and MU_STATE <= 0.01 Pa in this same proof before production patching.
