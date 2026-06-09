# Review: V0.14 Step-1 Current-MUB/Base-Input Split

Verdict: `STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

Findings:
- HIGH: The residual is caused by using the final post-`start_domain` base MUB as the current `adjust_tempqv` MUB. WRF uses the transient post-`blend_terrain`/pre-`start_domain` MUB.
- MEDIUM: The requested grouped pressure formula is not what WRF `adjust_tempqv` executes; the verified source formula is `p + c4h + c3h*mub + p_top`.
- LOW: This sandbox could not write the new `/mnt/data` scratch root, so the new WRF hook is delivered as a proposed disposable patch while scalar WRF truth is recovered from the prior accepted hook.

Evidence:
- WRF adjust hook: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- JAX recompute source: `proofs/v014/step1_live_nest_init_rerun.py::build_live_nest_step1_inputs`
- Patch diff artifact: `/home/enric/src/wrf_gpu2/proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`

Handoff:
objective: explain the current-MUB/base-input mismatch driving the Step-1 live-nest theta residual.

files changed:
- `proofs/v014/step1_current_mub_base_input_split.py`
- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_current_mub_base_input_split.md`
- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

commands run:
- `python -m py_compile proofs/v014/step1_current_mub_base_input_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_current_mub_base_input_split.py`
- `python -m json.tool proofs/v014/step1_current_mub_base_input_split.json >/tmp/step1_current_mub_base_input_split.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_current_mub_base_input_split.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_current_mub_base_input_split.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

unresolved risks:
- Fresh WRF terrain/PHB target emission could not be run because /mnt/data scratch writes are read-only in this sandbox.
- The source-changing sprint should validate the transient MUB blend over the full domain before patching production initialization.

next decision needed: Open the smallest source-changing sprint to add a transient live-nest adjust base path: compute WRF post-blend/pre-start_domain MUB for adjust_tempqv, use it only for theta/QV adjustment, keep final BaseState from start_domain, and rerun the Step-1 theta proof.
