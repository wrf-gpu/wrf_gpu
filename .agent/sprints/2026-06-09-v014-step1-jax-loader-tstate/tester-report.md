# Tester Report

## Tests Added Or Run

Manager reran:

- `python -m py_compile proofs/v014/step1_jax_loader_tstate.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_loader_tstate.py`
- `python -m json.tool proofs/v014/step1_jax_loader_tstate.json >/tmp/step1_jax_loader_tstate.manager.validated.json`
- `git diff -- src/gpuwrf`
- `git diff --check -- proofs/v014/step1_jax_loader_tstate.py proofs/v014/step1_jax_loader_tstate.json proofs/v014/step1_jax_loader_tstate.md .agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`

## Results

- Python compilation passed.
- CPU proof rerun reproduced verdict
  `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.
- JSON validation passed.
- Whitespace check passed.
- `git diff -- src/gpuwrf` was empty.
- JSON records CPU backend, `CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`,
  `gpu_device_count=0`, `gpu_used=false`, and `production_src_edits=false`.

## Fixtures Used

- WRF pre-call truth reused from
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`.
- No WRF rebuild, no GPU validation, no TOST, no Switzerland validation.

## Gaps

The sprint is a localization proof only. It does not apply the WRF
live-nest `T_STATE` initialization fix.

## Decision:

Pass.
