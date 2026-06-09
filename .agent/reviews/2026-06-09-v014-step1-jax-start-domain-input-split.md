# Review: V0.14 Step-1 JAX Start-Domain Input Split

Verdict: `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.

objective: close or precisely localize the current JAX live-nest `start_domain` input gap for Step-1 P/MU/W initialization.

files changed:
- `proofs/v014/step1_jax_start_domain_input_split.py`
- `proofs/v014/step1_jax_start_domain_input_split.json`
- `proofs/v014/step1_jax_start_domain_input_split.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`

commands run:
- `python -m py_compile proofs/v014/step1_jax_start_domain_input_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_start_domain_input_split.py`
- `python -m json.tool proofs/v014/step1_jax_start_domain_input_split.json >/tmp/step1_jax_start_domain_input_split.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_jax_start_domain_input_split.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_jax_start_domain_input_split.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`
- `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

ranked hypotheses/exclusions:
- rank 1: SUPPORTED_LOCALIZED_NOT_PATCH_READY - Dominant current JAX formula residual is diagnosed AL/ALT, fed by base-state reconstruction.
- rank 2: SUPPORTED_BY_FALSIFIER - The missing production contract is WRF start_domain base reconstruction precision/source order.
- rank 3: REFUTED - Final blended terrain is the dominant source.
- rank 4: REFUTED - Time-level selection, PH_STATE, or pre-press MU is the source.
- rank 5: REFUTED_AS_DOMINANT - PB or theta alone is the dominant pressure source.
- excluded: WRF start_domain P/press_adj/W source ordering remains accepted from the predecessor proof.
- excluded: Time-level selection is not the P/ALT cause: T1/T2, MU1/MU2, and PH1/PH2 match exactly at the hypsometric surface.
- excluded: PH_STATE and pre-press MU are not the input gap: current JAX PH and MU match WRF PH1/MU1 to round-off/zero.
- excluded: Terrain blend is not dominant: HT and HT_FINE residuals are tiny, and terrain substitution does not improve press_adj MU.
- excluded: A narrow production patch is not safe yet: proof-local WRF-like fp32 base recompute does not close P/MU gates.

unresolved risks:
- No production source patch was applied.
- The exact WRF base-state reconstruction/source-order contract is still missing; the best proof-local fp32/cp=1004.5 candidate remains above P/MU gates.
- Direct WRF AL/ALT substitution proves the perturbation formula path, but production cannot use WRF truth arrays at runtime.

next decision: Emit or reproduce the exact WRF start_domain base-state source boundary before the hypsometric AL/ALT pass: p_surf, MUB immediately after assignment, PB/T_INIT/ALB after the multi-domain reconstitution block, PHB after base integration, C3F/C4F/C3H/C4H as used in memory, imask/rebalance/hybrid flags, and scalar constants. The next worker should close that base reconstruction to WRF PHB+MUB, then apply the P/MU/W perturbation init patch already proven by direct AL/ALT substitution.
