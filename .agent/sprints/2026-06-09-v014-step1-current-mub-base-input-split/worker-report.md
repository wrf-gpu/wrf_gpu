# Worker Report: V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09

Summary: The sprint produced a CPU-only proof that explains the `17.5 Pa`
current-MUB/base-input mismatch. The residual is caused by comparing two
different WRF call boundaries: WRF `adjust_tempqv` uses transient
post-`blend_terrain`/pre-`start_domain` current `MUB`, while the prior JAX
theta proof used final post-`start_domain` base `MUB`.

## Files Changed

- `proofs/v014/step1_current_mub_base_input_split.py`
- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_current_mub_base_input_split.md`
- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

No production `src/gpuwrf/**` file was edited.

## Commands Run

- `git rev-parse HEAD`
- `git merge-base --is-ancestor 9a7016d9 HEAD`
- recovered prior accepted WRF adjust hook from
  `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- CPU-only proof-side JAX live-nest target recompute
- `python -m py_compile proofs/v014/step1_current_mub_base_input_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_current_mub_base_input_split.py`
- `python -m json.tool proofs/v014/step1_current_mub_base_input_split.json >/tmp/step1_current_mub_base_input_split.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects

- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_current_mub_base_input_split.md`
- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

## Result

Final verdict:
`STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

The proof-side WRF MUB blend gives `86812.250452109511`, matching the WRF
`adjust_tempqv` hook `86812.25`. The prior theta proof used final base MUB
`86794.574960128695`, which also matches WRF pre-part1 final MUB
`86794.5703125`. Therefore the source-changing next sprint should add a
transient live-nest adjust-base path for theta/QV adjustment while preserving
the final post-`start_domain` BaseState.
