# Worker Report: V0.14 Step-1 JAX Start-Domain Input Split

Summary: produced a CPU-only proof that split the current JAX live-nest
`start_domain` input residual and localized the remaining `P/MU` gap to
base-state reconstruction feeding fp32 `AL/ALT` diagnosis. No production source
edit was made.

objective: close or precisely localize the current JAX live-nest `start_domain`
input gap identified by the predecessor sprint.

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

- `proofs/v014/step1_jax_start_domain_input_split.json`
- `proofs/v014/step1_jax_start_domain_input_split.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`
- predecessor WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

result:

- Verdict:
  `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.
- Current pressure formula versus WRF P remains too large:
  max_abs `3.9458582235092763`, RMSE `0.3832298992869327`.
- Replacing diagnosed ALT with WRF ALT reduces pressure max_abs to
  `0.07605321895971429`.
- FP32 ALT diagnosis with WRF `PHB+MUB` reduces pressure max_abs to `0.0859375`.
- Best local WRF-order fp32/cp=1004.5 base candidate still leaves
  `P_STATE` max_abs `2.828125` and `MU_STATE` max_abs `0.011962890625`.

ranked hypotheses/exclusions:

- Supported: dominant residual is diagnosed `AL/ALT`, fed by base-state
  reconstruction.
- Supported: the missing production contract is exact WRF `start_domain`
  base-state precision/source order before the hypsometric `AL/ALT` pass.
- Refuted as dominant: terrain/final blend, time-level selection, `PH_STATE`,
  pre-press `MU`, `PB` alone, and theta alone.

unresolved risks:

- No production source patch was applied.
- Direct WRF `AL/ALT` substitution proves the formula path, but production
  cannot use WRF truth arrays at runtime.
- The exact WRF base-state reconstruction boundary still needs to be emitted or
  reproduced before patching `d02_replay.py`.

next decision needed:

Open the next sprint to emit or reproduce the exact WRF base-state boundary
before hypsometric `AL/ALT`: `p_surf`, post-assignment `MUB`,
`PB/T_INIT/ALB`, `PHB`, active hybrid coefficients, flags, and scalar
constants.
