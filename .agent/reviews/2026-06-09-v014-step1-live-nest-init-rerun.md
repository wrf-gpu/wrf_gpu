# Review: V0.14 Step-1 Live-Nest Init Rerun

Verdict: `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`.

objective: rerun the strict d02 step-1 same-input comparison using the production native live-nest child base initialization semantics wired into a CPU-only proof loader.

files changed:
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md`

commands run:
- `python -m py_compile proofs/v014/step1_live_nest_init_rerun.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py`
- `python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_live_nest_init_rerun.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_live_nest_init_rerun.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz` reused, not rebuilt

unresolved risks:
- The proof-local CPU loader mirrors production live-nest init semantics but bypasses build_replay_case because State.zeros is GPU-only.
- Residuals after base closure identify a field-level symptom, not yet the exact dycore or physics operator.

next decision: Run the next operator-localization sprint at field T; largest max_abs after base closure is P.
