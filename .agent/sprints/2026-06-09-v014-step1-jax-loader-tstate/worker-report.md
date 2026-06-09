# Worker Report

## Summary:

Built a CPU-only staged JAX loader/carry comparator for Step-1 `T_STATE` and
localized the residual to the live-nest state/base semantic split.

Final verdict:
`STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.

The proof shows `T_STATE` is already different in raw d02 wrfinput theta and is
carried unchanged through live-nest base init, boundary packaging, initial carry,
and haloed step-entry. In the same live-nest stage, `PB/PHB/MUB` improve to
small residuals, so the remaining issue is not boundary package, carry, or halo.

## Files Changed

- `proofs/v014/step1_jax_loader_tstate.py`
- `proofs/v014/step1_jax_loader_tstate.json`
- `proofs/v014/step1_jax_loader_tstate.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`

## Commands Run

- `git log -1 --oneline --decorate`
- `git merge-base --is-ancestor 99df65e0 HEAD`
- `python -m py_compile proofs/v014/step1_jax_loader_tstate.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_loader_tstate.py`
- `python -m json.tool proofs/v014/step1_jax_loader_tstate.json >/tmp/step1_jax_loader_tstate.validated.json`
- `git diff -- src/gpuwrf`
- `git diff --check -- proofs/v014/step1_jax_loader_tstate.py proofs/v014/step1_jax_loader_tstate.json proofs/v014/step1_jax_loader_tstate.md .agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`

## Proof Objects

- `proofs/v014/step1_jax_loader_tstate.py`
- `proofs/v014/step1_jax_loader_tstate.json`
- `proofs/v014/step1_jax_loader_tstate.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`
- reused WRF truth: `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`

## Key Results

- `T_STATE` max_abs versus WRF pre-call stays
  `5.490173101425171` for raw, live, boundary-packaged, carry, and haloed
  step-entry states.
- `T_STATE` transition max_abs is `0.0` for raw->live, live->boundary,
  boundary->carry, and carry->halo.
- `PB` improves from raw max_abs `2627.3828125` to live max_abs
  `0.05357326504599769`.
- Haloed step-entry `T_STATE` residual is not boundary-only: interior max_abs
  `5.490173101425171`, boundary-band max_abs `5.284271240234375`.
- CPU backend only; `gpu_used=false`; `src/gpuwrf` diff was empty.

## Risks

No production source fix was made. The next sprint must localize WRF
live-nest initialization `T_STATE`/theta semantics and then decide the smallest
GPU-native source patch.

## Handoff

Next target: `med_nest_initial` / `start_domain_em` live-nest `t_2` semantics.
Do not resume TOST, Switzerland, FP32, memory source work, or GPU validation
until this state/base split is fixed or proven acceptable.
