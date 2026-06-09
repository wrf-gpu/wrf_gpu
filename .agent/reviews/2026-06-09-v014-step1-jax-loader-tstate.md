# Review: V0.14 Step-1 JAX Loader T_STATE

Verdict: `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.

objective: split the JAX live-nest Step-1 loader/carry construction for `T_STATE` against accepted WRF solve_em pre-`first_rk_step_part1` truth.

files changed:
- `proofs/v014/step1_jax_loader_tstate.py`
- `proofs/v014/step1_jax_loader_tstate.json`
- `proofs/v014/step1_jax_loader_tstate.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`

commands run:
- `python -m py_compile proofs/v014/step1_jax_loader_tstate.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_loader_tstate.py`
- `python -m json.tool proofs/v014/step1_jax_loader_tstate.json >/tmp/step1_jax_loader_tstate.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_jax_loader_tstate.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_jax_loader_tstate.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth` reused, not rebuilt

unresolved risks:
- This proof localizes the T_STATE loader/carry split but does not implement WRF live-nest T_STATE semantics.
- No production source fix was made, so the extra source-edit proof chain was not run.

next decision needed: Localize WRF live-nest initialization T_STATE/theta semantics; JAX updates PB/PHB/MUB but carries raw wrfinput theta unchanged.
