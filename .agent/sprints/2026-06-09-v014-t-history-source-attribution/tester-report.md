# Tester Report

Decision: pass.

Manager reran the worker's required gates in the main process after the worker
finished.

Commands rerun by manager:

- `python -m py_compile proofs/v014/jax_t_history_source_attribution.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_t_history_source_attribution.py >/tmp/jax_t_history_source_attribution.manager.stdout 2>/tmp/jax_t_history_source_attribution.manager.stderr`
- `python -m json.tool proofs/v014/jax_t_history_source_attribution.json >/tmp/jax_t_history_source_attribution.manager.validated.json`
- `git diff --check -- proofs/v014/jax_t_history_source_attribution.py proofs/v014/jax_t_history_source_attribution.json proofs/v014/jax_t_history_source_attribution.md .agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

Observed result:

- Proof script compiles.
- CPU-only proof rerun exits zero.
- Terminal output is compact:
  `T_EVOLUTION_MISMATCH_CONFIRMED best_T_HIST_SRC=captured_pre_halo_state.theta_minus_300:3.3545763228707983 best_T_THM=captured_final_carry.t_2ave_minus_300:3.677881697025043`.
- Manager-captured stderr is empty.
- JSON validates.
- `git diff --check` passes for the sprint artifacts.

Coverage limits:

- No GPU was used.
- No TOST or Switzerland validation was run.
- No production `src/` or WRF source was edited.
